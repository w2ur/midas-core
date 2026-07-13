"""Tests for scripts.session_state — resumable step tracking.

Coverage:
- mark / is_done / clear roundtrip
- malformed state file → is_done returns False
- atomicity: os.replace failing after tmp write → original intact
- second call to a wrapped step is skipped (via idempotent_step decorator)
- journal write atomicity (engine.agent_memory.save_journal)
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

import scripts.session_state as ss
from engine import agent_memory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_state_dir() -> Path:
    """Return the session-state directory already redirected by conftest.

    The conftest autouse fixture patches ``ss._STATE_DIR`` for every test;
    this non-autouse fixture simply exposes that path to tests that need to
    inspect files written there.
    """
    return ss._STATE_DIR


@pytest.fixture()
def today() -> date:
    return date(2026, 6, 12)


# ---------------------------------------------------------------------------
# mark_done / is_done / clear roundtrip
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_is_done_false_before_marking(self, today: date) -> None:
        assert not ss.is_done("step_fill_orders", day=today)

    def test_is_done_true_after_marking(self, today: date) -> None:
        ss.mark_done("step_fill_orders", day=today)
        assert ss.is_done("step_fill_orders", day=today)

    def test_mark_multiple_steps(self, today: date) -> None:
        ss.mark_done("step_fetch_market_data", day=today)
        ss.mark_done("step_fill_orders", day=today)
        assert ss.is_done("step_fetch_market_data", day=today)
        assert ss.is_done("step_fill_orders", day=today)
        assert not ss.is_done("step_save_memories", day=today)

    def test_clear_removes_state(self, today: date) -> None:
        ss.mark_done("step_fill_orders", day=today)
        ss.clear(day=today)
        assert not ss.is_done("step_fill_orders", day=today)

    def test_clear_missing_file_is_noop(self, today: date) -> None:
        # Should not raise even if no file exists.
        ss.clear(day=today)

    def test_mark_is_idempotent(self, today: date) -> None:
        ss.mark_done("step_fill_orders", day=today)
        ss.mark_done("step_fill_orders", day=today)
        assert ss.is_done("step_fill_orders", day=today)

    def test_day_isolation(self) -> None:
        day_a = date(2026, 6, 10)
        day_b = date(2026, 6, 11)
        ss.mark_done("step_fill_orders", day=day_a)
        assert ss.is_done("step_fill_orders", day=day_a)
        assert not ss.is_done("step_fill_orders", day=day_b)

    def test_state_file_is_json(self, today: date, isolated_state_dir: Path) -> None:
        ss.mark_done("step_fill_orders", day=today)
        path = isolated_state_dir / f"{today.isoformat()}.json"
        data = json.loads(path.read_text())
        assert "step_fill_orders" in data
        # Value is an ISO timestamp string.
        assert isinstance(data["step_fill_orders"], str)
        assert "T" in data["step_fill_orders"]


# ---------------------------------------------------------------------------
# Malformed state file
# ---------------------------------------------------------------------------


class TestMalformedFile:
    def test_invalid_json_returns_not_done(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        path = isolated_state_dir / f"{today.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("NOT VALID JSON", encoding="utf-8")
        assert not ss.is_done("step_fill_orders", day=today)

    def test_json_non_object_returns_not_done(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        path = isolated_state_dir / f"{today.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('["step_fill_orders"]', encoding="utf-8")
        assert not ss.is_done("step_fill_orders", day=today)

    def test_empty_file_returns_not_done(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        path = isolated_state_dir / f"{today.isoformat()}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        assert not ss.is_done("step_fill_orders", day=today)


# ---------------------------------------------------------------------------
# Atomicity: os.replace failing → original file must be intact
# ---------------------------------------------------------------------------


class TestAtomicity:
    def test_original_intact_when_replace_fails(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        """A crash inside os.replace must not corrupt an already-written state."""
        # Write an initial good state.
        ss.mark_done("step_fetch_market_data", day=today)

        # Verify the file exists and is valid.
        path = isolated_state_dir / f"{today.isoformat()}.json"
        original_content = path.read_text(encoding="utf-8")
        assert "step_fetch_market_data" in original_content

        # Monkeypatch os.replace to raise after the tmp file is written.
        real_replace = os.replace

        def failing_replace(src: str, dst: str) -> None:
            # The tmp file has been written at this point — raise to simulate a crash.
            raise OSError("simulated replace failure")

        with patch("os.replace", side_effect=failing_replace):
            with pytest.raises(OSError, match="simulated replace failure"):
                ss.mark_done("step_fill_orders", day=today)

        # Original state file must still be intact.
        current_content = path.read_text(encoding="utf-8")
        assert current_content == original_content
        # The failed step must not appear in the state.
        data = json.loads(current_content)
        assert "step_fill_orders" not in data

    def test_no_orphan_tmp_after_replace_fails(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        """The tmp file must be cleaned up even if os.replace raises."""
        isolated_state_dir.mkdir(parents=True, exist_ok=True)

        with patch("os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                ss.mark_done("step_fill_orders", day=today)

        # No .state_tmp_* files should remain.
        leftover = list(isolated_state_dir.glob(".state_tmp_*"))
        assert leftover == [], f"Orphan tmp files: {leftover}"


# ---------------------------------------------------------------------------
# Wrapped step skips on second call (idempotent_step decorator)
# ---------------------------------------------------------------------------


class TestIdempotentStepDecorator:
    def test_wrapped_step_skips_on_second_call(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        """A step decorated with @idempotent_step should only run its body once."""
        from scripts.daily_session import idempotent_step

        call_count = 0

        @idempotent_step(skip_return=0)
        def step_fake_counter() -> int:
            nonlocal call_count
            call_count += 1
            return call_count

        # The autouse fixture already redirects _STATE_DIR so state is isolated.
        result1 = step_fake_counter()
        assert call_count == 1
        assert result1 == 1

        # Second call: step is marked done → body must be skipped.
        result2 = step_fake_counter()
        assert call_count == 1, "Body must not run on second call"
        assert result2 == 0, "Skip return must be returned"

    def test_wrapped_step_marks_done_on_success(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        from scripts.daily_session import idempotent_step

        @idempotent_step(skip_return=None)
        def step_simple_noop() -> None:
            pass

        assert not ss.is_done("step_simple_noop")
        step_simple_noop()
        assert ss.is_done("step_simple_noop")

    def test_wrapped_step_not_marked_done_on_exception(
        self, today: date, isolated_state_dir: Path
    ) -> None:
        from scripts.daily_session import idempotent_step

        @idempotent_step(skip_return=None)
        def step_that_fails() -> None:
            raise RuntimeError("deliberate failure")

        with pytest.raises(RuntimeError):
            step_that_fails()

        assert not ss.is_done("step_that_fails")


# ---------------------------------------------------------------------------
# Journal write atomicity (engine.agent_memory.save_journal)
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_journal_dir(midas_data_root: Path) -> Path:
    """Redirect journal writes to an isolated tmp path via config env redirect."""
    from engine.config import get_config

    return get_config().journal_dir


class TestJournalAtomicity:
    def test_existing_journal_intact_when_replace_fails(
        self, temp_journal_dir: Path
    ) -> None:
        """save_journal must leave the original file intact if os.replace raises."""
        # Write an initial journal.
        original_content = "Day 1. All is well.\n"
        agent_memory.save_journal("satoshi", original_content)
        path = temp_journal_dir / "satoshi.md"
        assert path.read_text(encoding="utf-8") == original_content

        # Simulate a crash in os.replace.
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                agent_memory.save_journal("satoshi", "CORRUPTED CONTENT\n")

        # Original must be intact.
        assert path.read_text(encoding="utf-8") == original_content

    def test_no_orphan_tmp_journal_after_replace_fails(
        self, temp_journal_dir: Path
    ) -> None:
        """No .journal_tmp_* files should remain after a failed write."""
        with patch("os.replace", side_effect=OSError("boom")):
            with pytest.raises(OSError):
                agent_memory.save_journal("satoshi", "content\n")

        leftover = list(temp_journal_dir.glob(".journal_tmp_*"))
        assert leftover == [], f"Orphan tmp files: {leftover}"

    def test_successful_write_roundtrip(self, temp_journal_dir: Path) -> None:
        """Normal write still works correctly after atomicity refactor."""
        content = "Day 7. Feeling bullish on BTC.\n"
        agent_memory.save_journal("satoshi", content)
        assert agent_memory.load_journal("satoshi") == content

    def test_write_creates_directory_atomically(self, midas_data_root: Path) -> None:
        """save_journal creates the journal directory if it does not exist."""
        from engine.config import get_config

        agent_memory.save_journal("satoshi", "hello\n")
        assert (get_config().journal_dir / "satoshi.md").exists()
