"""The checkout fence around a persona dispatch round."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.daily_session as ds
from engine.dispatch_guard import (
    CONCERNS_FILE,
    DispatchWroteDataError,
    assert_data_tree_unchanged,
    guard_concerns,
    snapshot_data_tree,
)
from scripts.session_guard import SessionAnchor, _anchor_path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    # A subdirectory: the suite's isolated session-state dir is tmp_path itself.
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "data").mkdir()
    (root / "data" / "tracked.json").write_text("{}\n")
    (root / "engine").mkdir()
    (root / "engine" / "x.py").write_text("X = 1\n")
    (root / ".gitignore").write_text(
        "data/cache/\ndata/session_state/\ndata/market/today.json\n"
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


@pytest.fixture(autouse=True)
def site_packages(tmp_path_factory, monkeypatch) -> Path:
    """A private site-packages, so the real venv never enters a test's snapshot."""
    site = tmp_path_factory.mktemp("site-packages")
    (site / "existing.pth").write_text("/some/path\n")
    monkeypatch.setattr("engine.dispatch_guard._site_packages_dirs", lambda: [site])
    return site


def _concerns(repo: Path) -> list[dict]:
    path = repo / CONCERNS_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_anchor(started: datetime, base_sha: str = "a" * 40) -> None:
    path = _anchor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    anchor = SessionAnchor(date(2026, 10, 9), base_sha, started)
    path.write_text(json.dumps(anchor.to_dict()))


# --- what passes -----------------------------------------------------------


def test_a_round_that_writes_nothing_passes(repo, capsys) -> None:
    snapshot_data_tree("r", repo)
    assert_data_tree_unchanged("r", repo)
    assert "no abort signal changed" in capsys.readouterr().out
    assert _concerns(repo) == []


def test_gitignored_paths_are_outside_the_fence(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "cache").mkdir()
    (repo / "data" / "cache" / "x").write_text("ignored")
    assert_data_tree_unchanged("r", repo)
    assert _concerns(repo) == []


def test_a_fetch_moving_a_remote_ref_passes(repo) -> None:
    """A fetch is harmless: refs/remotes/ and FETCH_HEAD are outside."""
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    snapshot_data_tree("r", repo)
    _git(repo, "update-ref", "refs/remotes/origin/main", head)
    (repo / ".git" / "FETCH_HEAD").write_text(f"{head}\t\tbranch 'main'\n")
    assert_data_tree_unchanged("r", repo)


def test_end_called_twice_passes_both_times(repo) -> None:
    """A re-run end after a downstream error must evaluate, not refuse."""
    snapshot_data_tree("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    assert_data_tree_unchanged("r", repo, anchor="s")
    assert snap.is_file()
    assert_data_tree_unchanged("r", repo, anchor="s")


def test_end_after_a_pass_ignores_the_sessions_own_later_writes(repo, capsys) -> None:
    """step_author_all writes the outbox after the round; a re-run end must pass."""
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")
    (repo / "data" / "outbox.jsonl").write_text("{}\n")
    (repo / "data" / "tracked.json").write_text('{"later": 1}\n')
    capsys.readouterr()
    assert_data_tree_unchanged("r", repo, anchor="s")
    assert "[r]: already verified this session" in capsys.readouterr().out
    assert_data_tree_unchanged("r", repo, anchor="s")


def test_the_pass_state_is_a_field_of_the_snapshot_not_a_sibling_file(repo) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    guard_dir = repo / ".git" / "midas-dispatch-guard"
    snap = guard_dir / "r@s.json"
    assert "passed" not in json.loads(snap.read_text())
    assert_data_tree_unchanged("r", repo, anchor="s")
    assert json.loads(snap.read_text())["passed"] is True
    assert [p.name for p in guard_dir.iterdir()] == ["r@s.json"]


def test_a_failed_end_leaves_no_marker_so_it_keeps_failing(repo) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    (repo / "data" / "planted.json").write_text("evil")
    for _ in range(2):
        with pytest.raises(DispatchWroteDataError, match=r"data/planted\.json"):
            assert_data_tree_unchanged("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    assert "passed" not in json.loads(snap.read_text())


def test_a_second_begin_takes_a_fresh_baseline(repo) -> None:
    """Begin is once per dispatch: what is on disk at begin is the baseline."""
    snapshot_data_tree("r", repo, anchor="s")
    snapshot_data_tree("r", repo, anchor="s")
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match=r"data/planted\.json"):
        assert_data_tree_unchanged("r", repo, anchor="s")


def test_a_redispatch_after_a_pass_is_a_new_bracket(repo) -> None:
    """The begin clears the pass state, so a write in the re-dispatch is caught."""
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")
    snapshot_data_tree("r", repo, anchor="s")
    (repo / "data" / "tracked.json").write_text('{"subagent": 1}\n')
    with pytest.raises(DispatchWroteDataError, match=r"data/tracked\.json"):
        assert_data_tree_unchanged("r", repo, anchor="s")


def test_an_anchorless_pass_leaves_nothing_behind(repo) -> None:
    """Fail closed: an anchorless snapshot is keyed by round name alone.

    Left behind it would persist across ad-hoc runs and answer "already
    verified" to a later end that had no begin. Anchorless runs are local or
    manual, so an end re-run after a pass raising "did not run" is deliberate.
    """
    snapshot_data_tree("r", repo)
    assert_data_tree_unchanged("r", repo)
    assert list((repo / ".git" / "midas-dispatch-guard").iterdir()) == []
    with pytest.raises(DispatchWroteDataError, match="did not run"):
        assert_data_tree_unchanged("r", repo)


def test_begin_on_an_interrupted_round_that_wrote_raises(repo) -> None:
    """A round interrupted mid-dispatch, then resumed: its write is not baseline."""
    snapshot_data_tree("r", repo, anchor="s")
    (repo / "data" / "tracked.json").write_text('{"subagent": 1}\n')
    with pytest.raises(DispatchWroteDataError, match=r"data/tracked\.json") as exc:
        snapshot_data_tree("r", repo, anchor="s")
    assert "found at begin" in str(exc.value)
    # The old baseline is kept, so end still names the write.
    with pytest.raises(DispatchWroteDataError, match=r"data/tracked\.json"):
        assert_data_tree_unchanged("r", repo, anchor="s")


def test_begin_on_an_interrupted_round_records_report_changes(repo) -> None:
    """A report-class write by the interrupted dispatch is a concern, not baseline."""
    snapshot_data_tree("r", repo, anchor="s")
    state = repo / "data" / "session_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "stray.json").write_text("{}\n")
    snapshot_data_tree("r", repo, anchor="s")
    assert [(c["round"], c["anchor"], c["path"], c["kind"]) for c in _concerns(repo)] == [
        ("r", "s", "data/session_state/stray.json", "appeared")
    ]
    # The new baseline includes it: end does not report it a second time.
    assert_data_tree_unchanged("r", repo, anchor="s")
    assert len(_concerns(repo)) == 1


def test_begin_does_not_record_report_changes_after_a_pass(repo) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")
    state = repo / "data" / "session_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "results.json").write_text("{}\n")
    snapshot_data_tree("r", repo, anchor="s")
    assert _concerns(repo) == []


@pytest.mark.parametrize("field", ["abort", "report"])
def test_a_snapshot_whose_field_is_not_a_mapping_is_rebaselined(repo, field) -> None:
    """Unreadable at begin means a fresh baseline, never an AttributeError."""
    snapshot_data_tree("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    doc = json.loads(snap.read_text())
    doc[field] = ["not", "a", "mapping"]
    snap.write_text(json.dumps(doc))
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")


@pytest.mark.parametrize(
    "damage", [{"report": []}, {"report": "x"}, {"prefix": None}, {"prefix": "/elsewhere", "report": []}]
)
def test_a_malformed_report_never_skips_the_abort_comparison(repo, damage) -> None:
    """The abort check reads `abort` alone; the report diff is its own step."""
    snapshot_data_tree("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    doc = json.loads(snap.read_text())
    doc.update(damage)
    if damage.get("prefix", 1) is None:
        del doc["prefix"]
    snap.write_text(json.dumps(doc))
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match="found at begin"):
        snapshot_data_tree("r", repo, anchor="s")


def test_a_malformed_report_at_begin_is_recorded_not_absorbed(repo) -> None:
    """Regression (review round 9): a report section begin cannot diff was
    re-baselined away with no trace; it must leave a concern behind."""
    snapshot_data_tree("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    doc = json.loads(snap.read_text())
    doc["report"] = []
    snap.write_text(json.dumps(doc))
    snapshot_data_tree("r", repo, anchor="s")
    assert [(c["path"], c["kind"]) for c in _concerns(repo)] == [
        ("<report class>", "could not be evaluated")
    ]


def test_begin_on_a_clean_interrupted_round_then_end_passes(repo) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")


def test_begin_after_a_pass_does_not_compare(repo) -> None:
    """A passed snapshot is a finished round: the session's own writes are fine."""
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")
    (repo / "data" / "outbox.jsonl").write_text("{}\n")
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")


def test_an_anchorless_begin_never_compares(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "tracked.json").write_text('{"subagent": 1}\n')
    snapshot_data_tree("r", repo)
    assert_data_tree_unchanged("r", repo)


def test_a_truncated_snapshot_fails_end_and_begin_overwrites_it(repo) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    snap = repo / ".git" / "midas-dispatch-guard" / "r@s.json"
    snap.write_text('{"anchor": "s", "abo')
    with pytest.raises(DispatchWroteDataError, match="could not be evaluated"):
        assert_data_tree_unchanged("r", repo, anchor="s")
    snapshot_data_tree("r", repo, anchor="s")
    assert_data_tree_unchanged("r", repo, anchor="s")


def test_begin_writes_atomically_and_leaves_no_temp_file(repo, monkeypatch) -> None:
    snapshot_data_tree("r", repo, anchor="s")
    snapshot_data_tree("r", repo, anchor="s")
    guard_dir = repo / ".git" / "midas-dispatch-guard"
    assert [p.name for p in guard_dir.iterdir()] == ["r@s.json"]

    def refuse(src, dst):
        raise OSError("replace refused")

    monkeypatch.setattr("engine.dispatch_guard.os.replace", refuse)
    snap = guard_dir / "r@s.json"
    before = snap.read_text()
    with pytest.raises(OSError):
        snapshot_data_tree("r", repo, anchor="s")
    assert snap.read_text() == before  # the old snapshot is intact, not half-written
    assert [p.name for p in guard_dir.iterdir()] == ["r@s.json"]


def test_the_snapshot_lives_outside_data(repo) -> None:
    snapshot_data_tree("r", repo)
    assert not any(p.name == "r.json" for p in (repo / "data").rglob("*"))
    assert (repo / ".git" / "midas-dispatch-guard" / "r.json").exists()


# --- what aborts -----------------------------------------------------------


def test_a_new_file_under_data_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match=r"data/planted\.json"):
        assert_data_tree_unchanged("r", repo)


def test_a_modified_tracked_file_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "tracked.json").write_text('{"a": 1}\n')
    with pytest.raises(DispatchWroteDataError, match=r"data/tracked\.json"):
        assert_data_tree_unchanged("r", repo)


def test_a_second_edit_of_an_already_dirty_tracked_file_is_named(repo) -> None:
    (repo / "data" / "tracked.json").write_text('{"a": 1}\n')
    snapshot_data_tree("r", repo)
    (repo / "data" / "tracked.json").write_text('{"a": 2}\n')
    with pytest.raises(DispatchWroteDataError, match=r"data/tracked\.json"):
        assert_data_tree_unchanged("r", repo)


def test_an_edited_untracked_file_and_a_deleted_one_are_named(repo) -> None:
    (repo / "data" / "pre.json").write_text("a")
    snapshot_data_tree("r", repo)
    (repo / "data" / "pre.json").write_text("b")
    (repo / "data" / "tracked.json").unlink()
    with pytest.raises(DispatchWroteDataError) as err:
        assert_data_tree_unchanged("r", repo)
    assert "data/pre.json" in str(err.value)
    assert "data/tracked.json" in str(err.value)


def test_an_edit_of_a_tracked_file_outside_data_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "engine" / "x.py").write_text("X = 2\n")
    with pytest.raises(DispatchWroteDataError, match=r"engine/x\.py"):
        assert_data_tree_unchanged("r", repo)


def test_a_new_untracked_file_outside_data_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "roster.yaml").write_text("agents: []\n")
    with pytest.raises(DispatchWroteDataError, match=r"roster\.yaml"):
        assert_data_tree_unchanged("r", repo)


def test_a_commit_made_during_the_round_is_named(repo) -> None:
    """A subagent commit leaves status clean; step_git_commit_push would push it."""
    snapshot_data_tree("r", repo)
    (repo / "engine" / "x.py").write_text("X = 2\n")
    _git(repo, "commit", "-q", "-am", "planted")
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True,
    ).stdout == b""
    with pytest.raises(DispatchWroteDataError, match=r"HEAD \(changed\)"):
        assert_data_tree_unchanged("r", repo)


def test_a_new_branch_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    _git(repo, "branch", "planted")
    with pytest.raises(
        DispatchWroteDataError, match=r"ref:refs/heads/planted \(appeared\)"
    ):
        assert_data_tree_unchanged("r", repo)


def test_a_new_tag_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    _git(repo, "tag", "planted")
    with pytest.raises(
        DispatchWroteDataError, match=r"ref:refs/tags/planted \(appeared\)"
    ):
        assert_data_tree_unchanged("r", repo)


def test_a_written_file_hidden_by_skip_worktree_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "tracked.json").write_text('{"evil": 1}\n')
    _git(repo, "update-index", "--skip-worktree", "data/tracked.json")
    with pytest.raises(
        DispatchWroteDataError, match=r"index-flag:data/tracked\.json \(appeared\)"
    ):
        assert_data_tree_unchanged("r", repo)


def test_a_new_file_hidden_by_info_exclude_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(exist_ok=True)
    with exclude.open("a") as fh:
        fh.write("data/planted.json\n")
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match=r"git:info/exclude"):
        assert_data_tree_unchanged("r", repo)


def test_a_new_git_hook_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\ncurl evil\n")
    with pytest.raises(
        DispatchWroteDataError, match=r"git:hooks/pre-commit \(appeared\)"
    ):
        assert_data_tree_unchanged("r", repo)


def test_a_git_config_change_is_named(repo) -> None:
    snapshot_data_tree("r", repo)
    _git(repo, "config", "core.hooksPath", "/tmp/elsewhere")
    with pytest.raises(DispatchWroteDataError, match=r"git:config \(changed\)"):
        assert_data_tree_unchanged("r", repo)


def test_a_staged_rename_names_both_paths(repo) -> None:
    snapshot_data_tree("r", repo)
    _git(repo, "mv", "data/tracked.json", "data/renamed.json")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert status.startswith("R ")  # the R-entry branch is what this exercises
    with pytest.raises(DispatchWroteDataError) as err:
        assert_data_tree_unchanged("r", repo)
    assert "data/renamed.json (appeared)" in str(err.value)
    assert "data/tracked.json (appeared)" in str(err.value)


def test_a_staged_change_to_an_already_dirty_file_is_named(repo) -> None:
    (repo / "engine" / "x.py").write_text("X = 2\n")
    snapshot_data_tree("r", repo)
    _git(repo, "add", "engine/x.py")
    with pytest.raises(
        DispatchWroteDataError, match=r"index-status:engine/x\.py \(changed\)"
    ):
        assert_data_tree_unchanged("r", repo)


def test_an_abort_records_no_concern(repo) -> None:
    snapshot_data_tree("r", repo)
    (repo / "data" / "planted.json").write_text("evil")
    (repo / "data" / "market").mkdir()
    (repo / "data" / "market" / "today.json").write_text("{}")
    with pytest.raises(DispatchWroteDataError):
        assert_data_tree_unchanged("r", repo)
    assert _concerns(repo) == []


# --- what is reported ------------------------------------------------------


def test_a_result_under_session_state_results_is_reported_not_exempt(repo) -> None:
    """Results are persisted after the round's end, so inside a round they report."""
    snapshot_data_tree("r", repo)
    results = repo / "data" / "session_state" / "results"
    results.mkdir(parents=True)
    (results / "x.json").write_text("{}")
    assert_data_tree_unchanged("r", repo)
    assert [c["path"] for c in _concerns(repo)] == [
        "data/session_state/results/x.json"
    ]


def test_another_file_under_session_state_passes_and_is_reported(
    repo, capsys
) -> None:
    snapshot_data_tree("r", repo)
    state = repo / "data" / "session_state"
    state.mkdir(parents=True)
    (state / "state.json").write_text("{}")
    assert_data_tree_unchanged("r", repo)
    assert _concerns(repo) == [
        {
            "anchor": None,
            "round": "r",
            "path": "data/session_state/state.json",
            "kind": "appeared",
        }
    ]
    assert "reported data/session_state/state.json" in capsys.readouterr().out


def test_orchestrator_writes_under_session_state_are_not_reported(repo) -> None:
    state = repo / "data" / "session_state"
    (state / "prompts").mkdir(parents=True)
    (state / "prompts" / "sibling.txt").write_text("honest prompt")
    (state / "dispatch_ledger.jsonl").write_text('{"n": 1}\n')
    snapshot_data_tree("r", repo)
    (state / "prompts" / "sibling.txt").write_text("next prompt")
    (state / "prompts" / "new.txt").write_text("another")
    with (state / "dispatch_ledger.jsonl").open("a") as fh:
        fh.write('{"n": 2}\n')
    assert_data_tree_unchanged("r", repo)
    assert _concerns(repo) == []


def test_an_edit_of_ignored_today_json_passes_and_is_reported(repo) -> None:
    today = repo / "data" / "market" / "today.json"
    today.parent.mkdir()
    today.write_text('{"SPY": 500}\n')
    snapshot_data_tree("r", repo)
    today.write_text('{"SPY": 5}\n')
    assert_data_tree_unchanged("r", repo)
    assert [(c["path"], c["kind"]) for c in _concerns(repo)] == [
        ("data/market/today.json", "changed")
    ]


def test_a_new_pth_file_in_site_packages_is_reported(repo, site_packages) -> None:
    snapshot_data_tree("r", repo)
    (site_packages / "zz-evil.pth").write_text("import os; os.system('x')\n")
    assert_data_tree_unchanged("r", repo)
    assert [(c["path"], c["kind"]) for c in _concerns(repo)] == [
        ("site-packages:zz-evil.pth", "appeared")
    ]


def test_another_sys_prefix_at_end_skips_site_packages(
    repo, site_packages, monkeypatch, capsys
) -> None:
    snapshot_data_tree("r", repo)
    monkeypatch.setattr("engine.dispatch_guard.sys.prefix", "/elsewhere")
    (site_packages / "zz-evil.pth").write_text("x\n")
    assert_data_tree_unchanged("r", repo)
    assert _concerns(repo) == []
    out = capsys.readouterr().out
    assert "snapshot taken under" in out and "now under /elsewhere" in out
    assert "site-packages not compared" in out


def test_concerns_summarise_one_line_per_round_for_this_anchor(repo) -> None:
    snapshot_data_tree("r1", repo, anchor="s1")
    snapshot_data_tree("r2", repo, anchor="s1")
    state = repo / "data" / "session_state"
    state.mkdir(parents=True)
    (state / "a.json").write_text("a")
    assert_data_tree_unchanged("r1", repo, anchor="s1")
    assert_data_tree_unchanged("r1", repo, anchor="s1")  # a re-run end
    (state / "b.json").write_text("b")
    assert_data_tree_unchanged("r2", repo, anchor="s1")
    assert guard_concerns("s1", repo) == [
        "dispatch guard [r1] reported changes to ignored inputs: "
        "data/session_state/a.json (appeared)",
        "dispatch guard [r2] reported changes to ignored inputs: "
        "data/session_state/a.json (appeared), data/session_state/b.json (appeared)",
    ]
    assert guard_concerns("another-session", repo) == []


def test_a_long_concern_is_bounded(repo) -> None:
    snapshot_data_tree("r", repo)
    state = repo / "data" / "session_state"
    state.mkdir(parents=True)
    for i in range(40):
        (state / f"result-{i:02}.json").write_text("x")
    assert_data_tree_unchanged("r", repo)
    (concern,) = guard_concerns(None, repo)
    assert len(concern) < 400
    assert concern.endswith(" more")


# --- a check that cannot run is not a pass ---------------------------------


def test_end_with_no_begin_raises(repo) -> None:
    with pytest.raises(DispatchWroteDataError, match="did not run"):
        assert_data_tree_unchanged("never-taken", repo)


def test_a_snapshot_from_another_anchor_is_missing(repo) -> None:
    snapshot_data_tree("r", repo, anchor="earlier-fire")
    with pytest.raises(DispatchWroteDataError, match="did not run"):
        assert_data_tree_unchanged("r", repo, anchor="this-fire")


def test_an_error_while_checking_is_a_dispatch_error_not_a_pass(
    repo, monkeypatch
) -> None:
    snapshot_data_tree("r", repo)

    def broken(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "status"])

    monkeypatch.setattr("engine.dispatch_guard._git", broken)
    with pytest.raises(DispatchWroteDataError, match="could not be evaluated"):
        assert_data_tree_unchanged("r", repo)


# --- the session wrappers ---------------------------------------------------


def test_session_step_wrappers_delegate(repo, monkeypatch) -> None:
    monkeypatch.setattr("engine.dispatch_guard._REPO_ROOT", repo)
    assert ds.step_guard_dispatch_begin("step2-trading") is None
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match=r"data/planted\.json"):
        ds.step_guard_dispatch_end("step2-trading")


def test_session_wrappers_key_the_snapshot_by_anchor(repo, monkeypatch) -> None:
    """A snapshot an earlier fire left behind is not this session's."""
    monkeypatch.setattr("engine.dispatch_guard._REPO_ROOT", repo)
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc))
    ds.step_guard_dispatch_begin("step6-posts")
    ds.step_guard_dispatch_end("step6-posts")
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc), "b" * 40)
    with pytest.raises(DispatchWroteDataError, match="did not run"):
        ds.step_guard_dispatch_end("step6-posts")


def test_every_run_is_a_new_guard_key(repo) -> None:
    """Step 0c anchors each run: a new start time is a new key, so an earlier
    run's snapshot (its writes reverted by Step 0's reset) is never compared."""
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc))
    first = ds._dispatch_guard_anchor()
    assert first == ds._dispatch_guard_anchor()
    assert first.startswith("2026-10-09-") and len(first) == len("2026-10-09-") + 12
    _write_anchor(datetime(2026, 10, 9, 23, 30, tzinfo=timezone.utc))
    assert ds._dispatch_guard_anchor() != first
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc), "b" * 40)
    assert ds._dispatch_guard_anchor() != first


def test_a_round_rebegun_in_the_same_run_compares_at_begin(repo, monkeypatch) -> None:
    monkeypatch.setattr("engine.dispatch_guard._REPO_ROOT", repo)
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc))
    ds.step_guard_dispatch_begin("step6-posts")
    (repo / "data" / "planted.json").write_text("evil")
    with pytest.raises(DispatchWroteDataError, match="found at begin"):
        ds.step_guard_dispatch_begin("step6-posts")


def test_a_new_run_never_compares_against_an_earlier_runs_snapshot(
    repo, monkeypatch
) -> None:
    """Step 0 reverted the earlier run's writes: comparing would false-abort."""
    monkeypatch.setattr("engine.dispatch_guard._REPO_ROOT", repo)
    _write_anchor(datetime(2026, 10, 9, 22, 0, tzinfo=timezone.utc))
    ds.step_guard_dispatch_begin("step6-posts")
    (repo / "data" / "planted.json").write_text("evil")
    _write_anchor(datetime(2026, 10, 9, 22, 40, tzinfo=timezone.utc))
    ds.step_guard_dispatch_begin("step6-posts")


# --- file names git cannot decode as UTF-8 ----------------------------------


_NON_UTF8 = b"data/caf\xe9.json"


def _make_non_utf8_file(repo: Path) -> None:
    try:
        with open(os.fsencode(repo) + b"/" + _NON_UTF8, "wb") as fh:
            fh.write(b"x")
    except OSError as exc:
        pytest.skip(f"this filesystem refuses non-UTF-8 file names ({exc})")


def test_a_non_utf8_name_before_begin_does_not_crash_begin(repo) -> None:
    _make_non_utf8_file(repo)
    snapshot_data_tree("r", repo)
    assert_data_tree_unchanged("r", repo)


def test_a_non_utf8_name_created_during_the_round_is_a_dispatch_error(repo) -> None:
    snapshot_data_tree("r", repo)
    _make_non_utf8_file(repo)
    with pytest.raises(DispatchWroteDataError, match=r"caf\\xe9\.json"):
        assert_data_tree_unchanged("r", repo)


def _status_also_lists(monkeypatch, entry: bytes) -> None:
    """Make ``git status`` report a non-UTF-8 path, on any filesystem."""
    import engine.dispatch_guard as guard

    real = guard._git

    def fake(root, *args):
        out = real(root, *args)
        return out + entry if args[:1] == ("status",) else out

    monkeypatch.setattr(guard, "_git", fake)


def test_a_reported_non_utf8_path_before_begin_does_not_crash_begin(
    repo, monkeypatch
) -> None:
    _status_also_lists(monkeypatch, b"?? " + _NON_UTF8 + b"\0")
    snapshot_data_tree("r", repo)
    assert_data_tree_unchanged("r", repo)


def test_a_reported_non_utf8_path_during_the_round_is_a_dispatch_error(
    repo, monkeypatch
) -> None:
    snapshot_data_tree("r", repo)
    _status_also_lists(monkeypatch, b"?? " + _NON_UTF8 + b"\0")
    with pytest.raises(DispatchWroteDataError, match=r"caf\\xe9\.json \(appeared\)"):
        assert_data_tree_unchanged("r", repo)
