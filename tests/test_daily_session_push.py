"""Lock in `step_git_commit_push` push-to-main contract.

RemoteTrigger sandbox sessions check out a throwaway branch
(`claude/<slug>`); a bare `git push` publishes that branch instead of
advancing main, so the daily snapshot never reaches the public Vercel
deploy. The helper must always push HEAD to `origin main`. This test
guards against a future refactor silently dropping that refspec — see
the Apr 30 incident.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts import daily_session


def _fake_run_factory(call_log: list[list[str]]) -> object:
    """Return a stub `subprocess.run` that records args and fakes outputs."""

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        call_log.append(list(args))
        cmd = args[1] if len(args) > 1 else ""
        # Default: nothing staged, HEAD ahead of origin/main by 1 commit.
        if args[:3] == ["git", "diff", "--cached"]:
            return subprocess.CompletedProcess(args, returncode=0)  # nothing to commit
        if args[:3] == ["git", "rev-list", "--count"]:
            return subprocess.CompletedProcess(args, returncode=0, stdout="1\n")
        return subprocess.CompletedProcess(args, returncode=0)

    return fake_run


class TestStepGitCommitPushTargetsMain:
    def test_push_uses_explicit_head_to_main_refspec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(daily_session.subprocess, "run", _fake_run_factory(calls))

        daily_session.step_git_commit_push(dry_run=False)

        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        assert push_calls, "step_git_commit_push must issue a git push"
        # Exactly one push, and it must specify HEAD:main — not a bare push.
        assert len(push_calls) == 1
        assert push_calls[0] == ["git", "push", "origin", "HEAD:main"], (
            "RemoteTrigger sessions run on a throwaway branch; a bare "
            "`git push` publishes that branch instead of advancing main. "
            "The push must always target origin/main explicitly."
        )

    def test_skips_push_when_head_already_at_origin_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(list(args))
            if args[:3] == ["git", "diff", "--cached"]:
                return subprocess.CompletedProcess(args, returncode=0)
            if args[:3] == ["git", "rev-list", "--count"]:
                return subprocess.CompletedProcess(args, returncode=0, stdout="0\n")
            return subprocess.CompletedProcess(args, returncode=0)

        monkeypatch.setattr(daily_session.subprocess, "run", fake_run)

        daily_session.step_git_commit_push(dry_run=False)

        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        assert push_calls == [], "Should not push when HEAD is already at origin/main."

    def test_dry_run_skips_all_git_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(daily_session.subprocess, "run", _fake_run_factory(calls))

        daily_session.step_git_commit_push(dry_run=True)

        assert calls == [], "Dry run must not invoke git."

    def test_falls_back_to_sandbox_branch_when_main_push_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the harness 403s `HEAD:main`, push the sandbox branch instead.

        The auto-merge-session.yml workflow then verifies session-integrity
        rules and merges to main. Lock the fallback path in — silent
        regression here would re-strand daily snapshots on a sandbox branch
        as on 2026-05-08.
        """

        calls: list[list[str]] = []

        def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(list(args))
            if args[:3] == ["git", "diff", "--cached"]:
                return subprocess.CompletedProcess(args, returncode=0)
            if args[:3] == ["git", "rev-list", "--count"]:
                return subprocess.CompletedProcess(args, returncode=0, stdout="1\n")
            if args[:4] == ["git", "push", "origin", "HEAD:main"]:
                return subprocess.CompletedProcess(
                    args,
                    returncode=1,
                    stdout="",
                    stderr="remote: HTTP 403 from harness proxy\n",
                )
            if args[:2] == ["git", "rev-parse"] and "--abbrev-ref" in args:
                return subprocess.CompletedProcess(
                    args, returncode=0, stdout="claude/happy-goldberg-VlIfz\n"
                )
            return subprocess.CompletedProcess(args, returncode=0)

        monkeypatch.setattr(daily_session.subprocess, "run", fake_run)

        daily_session.step_git_commit_push(dry_run=False)

        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        assert push_calls == [
            ["git", "push", "origin", "HEAD:main"],
            ["git", "push", "origin", "HEAD"],
        ], (
            "On main-push failure, helper must fall back to pushing the "
            "current sandbox branch so auto-merge-session.yml can take it "
            "the rest of the way to main."
        )


def _sandbox_fake(calls: list[list[str]], branch: str, branch_push_rc: int = 0):
    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(list(args))
        if args[:3] == ["git", "diff", "--cached"]:
            return subprocess.CompletedProcess(args, returncode=0)
        if args[:3] == ["git", "rev-list", "--count"]:
            return subprocess.CompletedProcess(args, returncode=0, stdout="1\n")
        if args[:2] == ["git", "rev-parse"] and "--abbrev-ref" in args:
            return subprocess.CompletedProcess(args, returncode=0, stdout=f"{branch}\n")
        if args[:4] == ["git", "push", "-u", "origin"]:
            return subprocess.CompletedProcess(
                args, returncode=branch_push_rc, stdout="", stderr="rejected\n"
            )
        return subprocess.CompletedProcess(args, returncode=0)

    return fake_run


class TestSandboxBranchIsPublishedAfterMain:
    """Regression: on 2026-09-23 the cloud stop hook reported the sandbox
    branch as unpushed after the session had landed on main, and the
    orchestrator spent a turn pushing it by hand."""

    def test_claude_branch_is_pushed_after_main(self, monkeypatch, capsys) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            daily_session.subprocess, "run", _sandbox_fake(calls, "claude/dreamy-x")
        )
        daily_session.step_git_commit_push(dry_run=False)
        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        # Main first, always: the sandbox branch is published only after it.
        assert push_calls == [
            ["git", "push", "origin", "HEAD:main"],
            ["git", "push", "-u", "origin", "HEAD"],
        ]
        assert "Also pushed sandbox branch 'claude/dreamy-x'" in capsys.readouterr().out

    def test_other_branch_is_left_alone(self, monkeypatch) -> None:
        # Control: a local run on a feature branch pushes main and nothing else.
        calls: list[list[str]] = []
        monkeypatch.setattr(
            daily_session.subprocess, "run", _sandbox_fake(calls, "feature/x")
        )
        daily_session.step_git_commit_push(dry_run=False)
        push_calls = [c for c in calls if c[:2] == ["git", "push"]]
        assert push_calls == [["git", "push", "origin", "HEAD:main"]]

    def test_failed_branch_push_is_a_warning(self, monkeypatch, capsys) -> None:
        calls: list[list[str]] = []
        monkeypatch.setattr(
            daily_session.subprocess,
            "run",
            _sandbox_fake(calls, "claude/dreamy-x", branch_push_rc=1),
        )
        daily_session.step_git_commit_push(dry_run=False)  # must not raise
        out = capsys.readouterr().out
        assert "[WARN] Sandbox branch 'claude/dreamy-x' not pushed" in out


class TestSessionCommitSanitisesConcerns:
    """Money review round 2, N1: the model typed the session commit itself,
    and a newline inside a Concerns: value put `[restate]` alone on a line.
    The commit is now made by a helper that flattens each concern to one
    line before it becomes a trailer."""

    def _repo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        repo = tmp_path / "repo"
        (repo / "data").mkdir(parents=True)
        for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", *args], cwd=repo, check=True)
        (repo / "data" / "x.json").write_text("{}")
        monkeypatch.setattr(daily_session, "_PROJECT_ROOT", repo)
        return repo

    def _message(self, repo: Path) -> str:
        return subprocess.run(
            ["git", "log", "-1", "--format=%B"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout

    def test_a_multi_line_concern_becomes_one_trailer_line(self, tmp_path, monkeypatch):
        from datetime import date

        repo = self._repo(tmp_path, monkeypatch)
        daily_session.step_commit_session(
            date(2026, 9, 25),
            concerns=["step X moved a row; needs\n[restate]", "  ", "catalysts\ttruncated"],
        )
        msg = self._message(repo)
        assert msg.splitlines()[0] == "chore: weekday session 2026-09-25"
        assert "[restate]" not in [line.strip() for line in msg.splitlines()]
        trailers = subprocess.run(
            ["git", "log", "-1", "--format=%(trailers:key=Concerns,valueonly)"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        assert [t for t in trailers if t] == [
            "step X moved a row; needs [restate]",
            "catalysts truncated",
        ]

    def test_no_concern_means_no_trailer(self, tmp_path, monkeypatch):
        from datetime import date

        repo = self._repo(tmp_path, monkeypatch)
        daily_session.step_commit_session(date(2026, 9, 25))
        assert self._message(repo).strip() == "chore: weekday session 2026-09-25"

    def test_a_bare_string_is_one_concern_not_one_per_letter(self, tmp_path, monkeypatch):
        """Money review round 3, M-B: `concerns="one sentence"` iterated the
        string and committed one `Concerns:` trailer per character."""
        from datetime import date

        repo = self._repo(tmp_path, monkeypatch)
        daily_session.step_commit_session(date(2026, 9, 25), concerns="one sentence")
        trailers = subprocess.run(
            ["git", "log", "-1", "--format=%(trailers:key=Concerns,valueonly)"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split("\n")
        assert [t for t in trailers if t] == ["one sentence"]
