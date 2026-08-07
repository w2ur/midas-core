"""Tests for the per-order processing logic in scripts.check_triggers.

Verifies:
- process_fired_order with a real Fill: inbox written, pending deleted,
  committer called once with correct paths.
- process_fired_order with None (idempotency skip): no inbox write, pending
  deleted (zombie cleanup), committer called.
- committer raising: processing of a subsequent order continues (push failure
  tolerance).
- Ordering invariant: commit happens after inbox + pending changes for each
  individual order (not batched).

The `committer` parameter is injectable so tests can verify ordering and
None-handling without git or network.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pytest

from engine.config import get_config

from engine.orders import Fill, Order, read_inbox
from engine.triggers import list_pending, save_pending


# ---------------------------------------------------------------------------
# Minimal broker_env fixture (mirrors pattern from test_check_triggers.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def broker_env(midas_data_root, monkeypatch):
    cfg = get_config()
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.agent_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    ticker_ccy_path = cfg.ticker_currencies_path
    outbox = cfg.orders_dir / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    inbox = cfg.orders_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    pm_base = midas_data_root / "portfolios"
    pm_base.mkdir()
    pending_dir = cfg.orders_dir / "pending"
    cancels_dir = cfg.orders_dir / "cancels"
    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)
    return {
        "ohlcv": ohlcv,
        "config_dir": config_dir,
        "ticker_ccy": ticker_ccy_path,
        "outbox": outbox,
        "inbox": inbox,
        "pm_base": pm_base,
        "pending": pending_dir,
    }


def _make_order(order_id: str = "ord_2026-05-10_satoshi_001") -> Order:
    return Order(
        order_id=order_id,
        ts=datetime(2026, 5, 10, 20, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker="BTC-EUR",
        shares=0.01,
        reasoning="test",
        currency="EUR",
        trigger={"op": ">=", "level": 85000.0},
        expires="2026-06-10",
    )


def _make_fill(order_id: str = "ord_2026-05-10_satoshi_001") -> Fill:
    return Fill(
        order_id=order_id,
        ts_filled=datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc),
        status="filled",
        fill_price=85123.45,
        fill_currency="EUR",
        notional_base=851.23,
        fees=0.0,
        reason=None,
        trigger_fired=True,
    )


TODAY = date(2026, 5, 17)


# ---------------------------------------------------------------------------
# process_fired_order — fired order (real Fill)
# ---------------------------------------------------------------------------


class TestProcessFiredOrderWithFill:
    def test_inbox_written(self, broker_env) -> None:
        """When called with a real Fill, the fill must appear in the inbox."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        calls: list[dict] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            calls.append({"order_id": order_id, "today": today, "paths": paths})

        process_fired_order(order, fill, TODAY, fake_committer)

        fills = read_inbox(TODAY)
        matching = [f for f in fills if f.order_id == order.order_id]
        assert len(matching) == 1
        assert matching[0].status == "filled"

    def test_pending_deleted(self, broker_env) -> None:
        """When called with a real Fill, the pending file must be removed."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        process_fired_order(order, fill, TODAY, lambda *a, **k: None)

        assert list_pending() == []

    def test_committer_called_once(self, broker_env) -> None:
        """Committer must be called exactly once per fired order."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        calls: list[dict] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            calls.append({"order_id": order_id, "today": today, "paths": paths})

        process_fired_order(order, fill, TODAY, fake_committer)

        assert len(calls) == 1

    def test_committer_receives_inbox_path(self, broker_env) -> None:
        """Committer paths must include the inbox JSONL file for today."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        captured_paths: list[list[str]] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            captured_paths.append(paths)

        process_fired_order(order, fill, TODAY, fake_committer)

        assert captured_paths, "Committer was not called"
        paths = captured_paths[0]
        # At least one path should reference today's inbox JSONL
        inbox_file = f"{TODAY.isoformat()}.jsonl"
        assert any(inbox_file in p for p in paths), (
            f"Expected inbox path containing '{inbox_file}' in {paths}"
        )

    def test_committer_receives_pending_path(self, broker_env) -> None:
        """Committer paths must include the pending JSON file for the order."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        captured_paths: list[list[str]] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            captured_paths.append(paths)

        process_fired_order(order, fill, TODAY, fake_committer)

        assert captured_paths, "Committer was not called"
        paths = captured_paths[0]
        pending_file = f"{order.order_id}.json"
        assert any(pending_file in p for p in paths), (
            f"Expected pending path containing '{pending_file}' in {paths}"
        )


# ---------------------------------------------------------------------------
# process_fired_order — None (idempotency skip / zombie pending file)
# ---------------------------------------------------------------------------


class TestProcessFiredOrderWithNone:
    def test_no_inbox_write_on_none(self, broker_env) -> None:
        """When fill_or_none is None, no fill line must be written to the inbox."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)

        process_fired_order(order, None, TODAY, lambda *a, **k: None)

        fills = read_inbox(TODAY)
        assert fills == [], f"Expected empty inbox, got {fills}"

    def test_pending_deleted_on_none(self, broker_env) -> None:
        """When fill_or_none is None, the zombie pending file must be cleaned up."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)

        process_fired_order(order, None, TODAY, lambda *a, **k: None)

        assert list_pending() == []

    def test_committer_called_on_none(self, broker_env) -> None:
        """Committer must still be called (to commit the zombie-file cleanup)."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)

        calls: list[dict] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            calls.append({"order_id": order_id, "today": today, "paths": paths})

        process_fired_order(order, None, TODAY, fake_committer)

        assert len(calls) == 1

    def test_committer_not_given_inbox_path_on_none(self, broker_env) -> None:
        """On None, committer should NOT receive the inbox file (no write happened)."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)

        captured_paths: list[list[str]] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            captured_paths.append(paths)

        process_fired_order(order, None, TODAY, fake_committer)

        if captured_paths:
            inbox_file = f"{TODAY.isoformat()}.jsonl"
            paths = captured_paths[0]
            assert not any(inbox_file in p for p in paths), (
                f"Inbox path must not appear in committer call when fill is None: {paths}"
            )


# ---------------------------------------------------------------------------
# Committer failure tolerance
# ---------------------------------------------------------------------------


class TestCommitterFailureTolerance:
    def test_second_order_processed_after_committer_raises(self, broker_env) -> None:
        """If the committer raises for order 1, order 2 must still be fully processed."""
        from scripts.check_triggers import process_fired_order

        order1 = _make_order("ord_2026-05-10_satoshi_001")
        order2 = _make_order("ord_2026-05-10_satoshi_002")
        save_pending(order1)
        save_pending(order2)

        fill1 = _make_fill("ord_2026-05-10_satoshi_001")
        fill2 = _make_fill("ord_2026-05-10_satoshi_002")

        call_count = 0

        def failing_then_ok_committer(
            order_id: str, today: date, paths: list[str]
        ) -> None:
            nonlocal call_count
            call_count += 1
            if order_id == order1.order_id:
                raise RuntimeError("Simulated push failure")

        # Processing order1 should not raise even if committer raises.
        try:
            process_fired_order(order1, fill1, TODAY, failing_then_ok_committer)
        except Exception:
            pytest.fail("process_fired_order must not propagate committer exceptions")

        # Processing order2 must succeed even after order1's committer raised.
        try:
            process_fired_order(order2, fill2, TODAY, failing_then_ok_committer)
        except Exception:
            pytest.fail(
                "process_fired_order must continue after prior committer failure"
            )

        # Both fills must be in the inbox.
        fills = read_inbox(TODAY)
        ids = [f.order_id for f in fills]
        assert order1.order_id in ids
        assert order2.order_id in ids

    def test_committer_raise_does_not_block_pending_cleanup(self, broker_env) -> None:
        """Even if committer raises, the pending file must already be deleted
        (mutation happened before commit attempt)."""
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        def always_fails(order_id: str, today: date, paths: list[str]) -> None:
            raise RuntimeError("Simulated push failure")

        process_fired_order(order, fill, TODAY, always_fails)

        # Pending must be gone even though commit failed.
        assert list_pending() == []


# ---------------------------------------------------------------------------
# Ordering invariant: commit message references order_id
# ---------------------------------------------------------------------------


class TestCommitMessageContent:
    def test_committer_receives_order_id(self, broker_env) -> None:
        """Committer's first positional arg must be the order_id."""
        from scripts.check_triggers import process_fired_order

        oid = "ord_2026-05-17_satoshi_042"
        order = _make_order(oid)
        save_pending(order)
        fill = _make_fill(oid)

        received_ids: list[str] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            received_ids.append(order_id)

        process_fired_order(order, fill, TODAY, fake_committer)

        assert received_ids == [oid]


# ---------------------------------------------------------------------------
# Portfolio directory included in per-fire commit pathspec
# ---------------------------------------------------------------------------


class TestCommitPathspecIncludesPortfolio:
    def test_committer_receives_portfolio_dir(self, broker_env) -> None:
        """Real-fill path: committer paths must include the agent's portfolio directory.

        execute_triggered_order mutates portfolio.json and trades.json via
        apply_trade. The per-fire commit must capture these mutations atomically,
        so the agent's portfolio directory must appear in the pathspec.
        """
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)
        fill = _make_fill()

        captured_paths: list[list[str]] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            captured_paths.append(list(paths))

        process_fired_order(order, fill, TODAY, fake_committer)

        assert captured_paths, "Committer was not called"
        paths = captured_paths[0]
        expected_fragment = f"portfolios/{order.agent_id}"
        assert any(expected_fragment in p for p in paths), (
            f"Expected a path containing '{expected_fragment}' in committer args {paths}. "
            "Portfolio mutations from apply_trade must be included in the per-fire commit."
        )

    def test_zombie_cleanup_commit_excludes_portfolio(self, broker_env) -> None:
        """None (zombie) path: committer paths must NOT include the portfolio directory.

        When fill_or_none is None the order was already filled in a prior run —
        no portfolio mutation occurs. Including the portfolio directory in the
        pathspec would stage unrelated changes and violate the atomicity contract.
        """
        from scripts.check_triggers import process_fired_order

        order = _make_order()
        save_pending(order)

        captured_paths: list[list[str]] = []

        def fake_committer(order_id: str, today: date, paths: list[str]) -> None:
            captured_paths.append(list(paths))

        process_fired_order(order, None, TODAY, fake_committer)

        assert captured_paths, "Committer was not called"
        paths = captured_paths[0]
        portfolio_fragment = f"portfolios/{order.agent_id}"
        assert not any(portfolio_fragment in p for p in paths), (
            f"Portfolio path must not appear in committer args on zombie cleanup: {paths}"
        )


# ---------------------------------------------------------------------------
# Integration tests: _git_add_commit against a real tmp git repo
# ---------------------------------------------------------------------------


def _init_git_repo(repo: Path, bare: Path) -> None:
    """Initialise a working repo + a bare remote, configure local identity."""
    import subprocess as sp

    bare.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)

    repo.mkdir(parents=True, exist_ok=True)
    sp.run(["git", "init", str(repo)], check=True, capture_output=True)
    sp.run(
        ["git", "remote", "add", "origin", str(bare)],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    sp.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )

    # Create an initial commit so `main` branch exists in both repo and bare.
    readme = repo / "README.md"
    readme.write_text("init\n")
    sp.run(["git", "add", "README.md"], cwd=str(repo), check=True, capture_output=True)
    sp.run(
        ["git", "commit", "-m", "init"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )
    sp.run(
        ["git", "push", "--set-upstream", "origin", "HEAD:main"],
        cwd=str(repo),
        check=True,
        capture_output=True,
    )


def _commit_count(repo: Path) -> int:
    """Return the number of commits on HEAD in the working repo."""
    import subprocess as sp

    result = sp.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def _bare_commit_count(bare: Path) -> int:
    """Return the number of commits on main in the bare remote."""
    import subprocess as sp

    result = sp.run(
        ["git", "rev-list", "--count", "main"],
        cwd=str(bare),
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


class TestGitAddCommitIntegration:
    """Drive the real _git_add_commit against a tmp git repo with a local bare remote."""

    def test_staged_change_commits_and_pushes(self, tmp_path, monkeypatch) -> None:
        """(a) A staged change: commit lands in the working repo and the push
        reaches the bare remote."""
        import scripts.check_triggers as ct

        repo = tmp_path / "repo"
        bare = tmp_path / "bare.git"
        _init_git_repo(repo, bare)

        initial_bare_count = _bare_commit_count(bare)

        monkeypatch.setattr(ct, "_PROJECT_ROOT", repo)

        # Write a file that will be staged.
        data_file = repo / "data.txt"
        data_file.write_text("trigger fired\n")

        ct._git_add_commit("ord_test_001", date(2026, 5, 17), [str(data_file)])

        # The working repo should have one more commit than before.
        assert _commit_count(repo) == initial_bare_count + 1
        # The bare remote should also have one more commit (push succeeded).
        assert _bare_commit_count(bare) == initial_bare_count + 1

    def test_nothing_staged_skips_commit(self, tmp_path, monkeypatch) -> None:
        """(b) When nothing is staged, no commit is created."""
        import scripts.check_triggers as ct

        repo = tmp_path / "repo"
        bare = tmp_path / "bare.git"
        _init_git_repo(repo, bare)

        initial_count = _commit_count(repo)

        monkeypatch.setattr(ct, "_PROJECT_ROOT", repo)

        # Pass the already-tracked, unmodified README — git add succeeds but
        # stages nothing, so diff --cached exits 0 and we skip the commit.
        tracked_unmodified = str(repo / "README.md")
        ct._git_add_commit("ord_test_002", date(2026, 5, 17), [tracked_unmodified])

        assert _commit_count(repo) == initial_count, (
            "No new commit should be created when nothing is staged."
        )

    def test_push_race_rebase_and_retry_succeeds(self, tmp_path, monkeypatch) -> None:
        """(c) Push rejected because the bare remote has a competing commit:
        rebase + retry succeeds, leaving both repos in sync."""
        import subprocess as sp
        import scripts.check_triggers as ct

        repo = tmp_path / "repo"
        bare = tmp_path / "bare.git"
        _init_git_repo(repo, bare)

        monkeypatch.setattr(ct, "_PROJECT_ROOT", repo)

        # Simulate a competing commit arriving on the bare remote via a second
        # clone, bypassing the working repo.
        clone = tmp_path / "clone"
        sp.run(["git", "clone", str(bare), str(clone)], check=True, capture_output=True)
        sp.run(
            ["git", "config", "user.email", "other@example.com"],
            cwd=str(clone),
            check=True,
            capture_output=True,
        )
        sp.run(
            ["git", "config", "user.name", "Other"],
            cwd=str(clone),
            check=True,
            capture_output=True,
        )
        competing = clone / "competing.txt"
        competing.write_text("competing change\n")
        sp.run(
            ["git", "add", "competing.txt"],
            cwd=str(clone),
            check=True,
            capture_output=True,
        )
        sp.run(
            ["git", "commit", "-m", "competing"],
            cwd=str(clone),
            check=True,
            capture_output=True,
        )
        sp.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=str(clone),
            check=True,
            capture_output=True,
        )

        # Now write a file in the working repo and commit it before pushing.
        # The push will fail because the bare has moved ahead.
        data_file = repo / "local_change.txt"
        data_file.write_text("local trigger fire\n")

        bare_count_before = _bare_commit_count(bare)

        ct._git_add_commit("ord_test_003", date(2026, 5, 17), [str(data_file)])

        # After rebase + retry, the bare remote should now have:
        # initial (1) + competing (1) + our change (1) = 3 commits.
        assert _bare_commit_count(bare) == bare_count_before + 1, (
            "After rebase-and-retry the working commit should reach the bare remote."
        )

    def test_rebase_abort_on_failed_rebase(self, tmp_path, monkeypatch) -> None:
        """Abort guard: if git pull --rebase returns non-zero, git rebase --abort
        is called and _git_add_commit returns without raising."""
        import subprocess as _real_subprocess
        import scripts.check_triggers as ct

        repo = tmp_path / "repo"
        bare = tmp_path / "bare.git"
        _init_git_repo(repo, bare)

        monkeypatch.setattr(ct, "_PROJECT_ROOT", repo)

        abort_calls: list[list[str]] = []
        original_run = _real_subprocess.run

        def patched_run(cmd, *args, **kwargs):
            if cmd[:3] == ["git", "pull", "--rebase"]:
                # Simulate rebase failure.
                class _FakeResult:
                    returncode = 1

                return _FakeResult()
            if cmd[:3] == ["git", "rebase", "--abort"]:
                abort_calls.append(cmd)
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(
            ct, "subprocess", type("_M", (), {"run": staticmethod(patched_run)})()
        )

        # Stage a real change so we get past the empty-staging check and commit,
        # but then hit the simulated push failure → rebase path.
        # We also need to make the first push fail.
        push_attempt = [0]
        original_run2 = _real_subprocess.run

        def patched_run2(cmd, *args, **kwargs):
            if cmd == ["git", "push", "origin", "HEAD:main"] and push_attempt[0] == 0:
                push_attempt[0] += 1

                class _FakeResult:
                    returncode = 1

                return _FakeResult()
            if cmd[:3] == ["git", "pull", "--rebase"]:

                class _FakeResult:
                    returncode = 1

                return _FakeResult()
            if cmd[:3] == ["git", "rebase", "--abort"]:
                abort_calls.append(cmd)
                return original_run2(cmd, *args, **kwargs)
            return original_run2(cmd, *args, **kwargs)

        # Reset and use the combined patcher.
        monkeypatch.setattr(
            ct,
            "subprocess",
            type("_M", (), {"run": staticmethod(patched_run2)})(),
        )

        data_file = repo / "abort_test.txt"
        data_file.write_text("abort guard test\n")

        # Must not raise even though both push and rebase fail.
        ct._git_add_commit("ord_test_004", date(2026, 5, 17), [str(data_file)])

        assert abort_calls, (
            "git rebase --abort should have been called on rebase failure"
        )
