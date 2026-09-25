"""Tests for engine.token_cost — the len/4 prompt-size proxy and session ledger."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine import token_cost
from engine.token_cost import (
    PROXY_LABEL,
    SessionCostLedger,
    estimate_tokens,
    record_dispatch,
    reset_session_costs,
    session_cost_totals,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_estimate_tokens_is_len_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


@given(text=st.text())
def test_estimate_tokens_matches_floor_div_property(text: str) -> None:
    assert estimate_tokens(text) == len(text) // 4


class TestSessionCostLedger:
    def test_empty_ledger_totals(self) -> None:
        ledger = SessionCostLedger()
        assert ledger.is_empty
        totals = ledger.totals()
        assert totals["proxy"] == PROXY_LABEL
        assert totals["total_dispatches"] == 0
        assert totals["total_est_tokens"] == 0
        assert totals["by_agent"] == {}

    def test_record_accumulates_per_agent(self) -> None:
        ledger = SessionCostLedger()
        assert ledger.record("satoshi", "a" * 400) == 100
        ledger.record("satoshi", "b" * 40)  # +10
        ledger.record("world", "c" * 80)  # +20
        totals = ledger.totals()
        assert totals["total_dispatches"] == 3
        assert totals["total_prompt_chars"] == 520
        assert totals["total_est_tokens"] == 130
        assert totals["by_agent"]["satoshi"] == {
            "dispatches": 2,
            "prompt_chars": 440,
            "est_tokens": 110,
        }
        assert totals["by_agent"]["world"]["est_tokens"] == 20

    def test_totals_is_a_snapshot_copy(self) -> None:
        # Mutating a returned totals block must not corrupt the ledger.
        ledger = SessionCostLedger()
        ledger.record("satoshi", "a" * 40)
        totals = ledger.totals()
        totals["by_agent"]["satoshi"]["est_tokens"] = 999
        assert ledger.totals()["by_agent"]["satoshi"]["est_tokens"] == 10

    def test_reset_clears(self) -> None:
        ledger = SessionCostLedger()
        ledger.record("satoshi", "hello world")
        assert not ledger.is_empty
        ledger.reset()
        assert ledger.is_empty


def test_module_ledger_record_and_totals() -> None:
    reset_session_costs()
    record_dispatch("goldfinger", "x" * 200)
    totals = session_cost_totals()
    assert totals["by_agent"]["goldfinger"]["est_tokens"] == 50
    assert totals["total_dispatches"] == 1
    reset_session_costs()
    assert session_cost_totals()["total_dispatches"] == 0


# ---------------------------------------------------------------------------
# The ledger survives the process boundary.
#
# Regression: d7b276fe2 — the 2026-09-23 bundle recorded 0 dispatches against the 33 the
# session really made. The orchestrator runs every step as its own
# `python -c` process, so an in-memory ledger fed by one process was empty in
# the process that assembled the bundle.
# ---------------------------------------------------------------------------


def _record_in_child(agent_id: str, chars: int, model: str | None) -> None:
    """Record one dispatch from a separate interpreter — the orchestrator's shape."""
    code = (
        "from engine.token_cost import record_dispatch; "
        f"record_dispatch({agent_id!r}, 'x' * {chars}, model={model!r})"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        check=True,
        env={**os.environ},
    )


class TestLedgerCrossesProcesses:
    @pytest.fixture
    def shared_ledger(self, midas_data_root, monkeypatch):
        # Drop the autouse per-test override so parent and child resolve the
        # same config-backed path under MIDAS_DATA_DIR.
        monkeypatch.setattr(token_cost, "_LEDGER_PATH", None)
        return midas_data_root

    def test_dispatches_recorded_by_other_processes_reach_the_totals(
        self, shared_ledger
    ) -> None:
        _record_in_child("satoshi", 400, "opus")
        _record_in_child("world", 80, "opus")
        _record_in_child("the-oracle", 40, "sonnet")
        totals = session_cost_totals()
        assert totals["total_dispatches"] == 3
        assert totals["total_est_tokens"] == 100 + 20 + 10
        assert totals["by_agent"]["satoshi"]["dispatches"] == 1

    def test_reset_clears_what_other_processes_recorded(self, shared_ledger) -> None:
        _record_in_child("satoshi", 400, "opus")
        reset_session_costs()
        assert session_cost_totals()["total_dispatches"] == 0

    def test_persisted_ledger_lives_under_session_state(self, shared_ledger) -> None:
        from engine.config import get_config

        record_dispatch("satoshi", "x" * 40, model="opus")
        assert (get_config().session_state_dir / "dispatch_ledger.jsonl").exists()


def test_unreadable_ledger_line_is_skipped_not_fatal(capsys) -> None:
    # A counter must never take a session down: a torn line is dropped with a
    # warning, and every readable row still counts.
    reset_session_costs()
    record_dispatch("satoshi", "x" * 40, model="opus")
    path = token_cost._ledger_path()
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")
    record_dispatch("world", "x" * 80, model="opus")
    totals = session_cost_totals()
    assert totals["total_dispatches"] == 2
    assert "dispatch ledger" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The model is recorded per dispatch.
# ---------------------------------------------------------------------------


def test_each_dispatch_records_its_model_alias_and_resolved_id(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-opus-test-id")
    monkeypatch.delenv("ANTHROPIC_DEFAULT_SONNET_MODEL", raising=False)
    reset_session_costs()
    record_dispatch("satoshi", "x" * 40, model="opus")
    record_dispatch("the-oracle", "x" * 40, model="sonnet")
    record_dispatch("legacy", "x" * 40)
    rows = session_cost_totals()["dispatches"]
    assert rows == [
        {"agent_id": "satoshi", "model": "opus", "model_id": "claude-opus-test-id"},
        # An alias the environment does not pin resolves to no id: unknown is
        # recorded as unknown, never guessed.
        {"agent_id": "the-oracle", "model": "sonnet", "model_id": None},
        {"agent_id": "legacy", "model": None, "model_id": None},
    ]


def test_resolve_model_id_reads_the_harness_alias_override(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "claude-sonnet-test-id")
    assert token_cost.resolve_model_id("sonnet") == "claude-sonnet-test-id"
    # A full id is its own resolution; nothing to look up.
    assert token_cost.resolve_model_id("claude-opus-5") == "claude-opus-5"
    assert token_cost.resolve_model_id(None) is None
    monkeypatch.delenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", raising=False)
    assert token_cost.resolve_model_id("haiku") is None
