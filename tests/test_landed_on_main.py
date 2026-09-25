"""scripts.landed_on_main — the record dispatch-session-integrity dispatches from.

J6 money review round 5, M-a: the action used to infer the landed commit from
history, and after a refused rebase-and-retry it inferred another writer's.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.landed_on_main import LANDED_FILENAME, landed_by_push, record_landed_on_main

MOVED = " \tHEAD:refs/heads/main\t1111111..2222222\nDone\n"


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
           "HOME": str(tmp_path), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "c"], cwd=repo, check=True, env=env)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    return repo, head


def test_head_is_recorded_under_runner_temp(tmp_path, monkeypatch):
    repo, head = _repo(tmp_path)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))

    assert record_landed_on_main(repo, MOVED) == head
    assert (tmp_path / LANDED_FILENAME).read_text() == head + "\n"


def test_a_later_landing_overwrites_the_earlier(tmp_path, monkeypatch):
    # A run that landed several commits dispatches its last one.
    repo, head = _repo(tmp_path)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    (tmp_path / LANDED_FILENAME).write_text("0" * 40 + "\n")

    record_landed_on_main(repo, MOVED)
    assert (tmp_path / LANDED_FILENAME).read_text() == head + "\n"


def test_nothing_is_written_outside_actions(tmp_path, monkeypatch):
    repo, _head = _repo(tmp_path)
    monkeypatch.delenv("RUNNER_TEMP", raising=False)

    assert record_landed_on_main(repo, MOVED) is None
    assert not (repo / LANDED_FILENAME).exists()


def test_an_unreadable_head_records_nothing(tmp_path, monkeypatch):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))

    assert record_landed_on_main(not_a_repo, MOVED) is None
    assert not (tmp_path / LANDED_FILENAME).exists()


def test_only_a_push_that_moved_main_counts_as_landed() -> None:
    """Follow-up money review r3, M2: a retry push that exits 0 with
    "[up to date]" moved nothing — HEAD is another writer's commit."""
    assert landed_by_push(MOVED)
    assert landed_by_push("*\tHEAD:refs/heads/main\t[new branch]\n")
    assert landed_by_push("+\tHEAD:refs/heads/main\t1..2 (forced update)\n")
    assert not landed_by_push("=\tHEAD:refs/heads/main\t[up to date]\nDone\n")
    assert not landed_by_push("!\tHEAD:refs/heads/main\t[rejected] (fetch first)\n")
    assert not landed_by_push(" \tHEAD:refs/heads/triggers/x\t1..2\n")  # not main
    assert not landed_by_push("")
    assert not landed_by_push(None)


def test_an_up_to_date_push_records_nothing(tmp_path, monkeypatch):
    repo, _head = _repo(tmp_path)
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))

    assert record_landed_on_main(repo, "=\tHEAD:refs/heads/main\t[up to date]\n") is None
    assert not (tmp_path / LANDED_FILENAME).exists()
