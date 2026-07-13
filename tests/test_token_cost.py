"""Tests for engine.token_cost — the len/4 prompt-size proxy and session ledger."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from engine.token_cost import (
    PROXY_LABEL,
    SessionCostLedger,
    estimate_tokens,
    record_dispatch,
    reset_session_costs,
    session_cost_totals,
)


def test_estimate_tokens_is_len_over_four() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


@settings(max_examples=300)
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
