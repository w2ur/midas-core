"""Tests for scripts.restate_bundles — restating per-day archive bundles.

Regression: the site publishes two different figures for the same agent on
the same day (data/output/*.json's own leaderboard vs the restated
data/portfolios/*/snapshots.json). These tests cover the merge/re-rank logic
in isolation — the underlying valuation math (build_leaderboard_rows,
replay_holdings, _resolve_session_date) already has its own test coverage.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts import restate_bundles as rb
from scripts.restate_bundles import AgentState, restate_bundle_leaderboard


def _state(currency: str = "EUR") -> AgentState:
    return AgentState(
        trades=[],
        currency=currency,
        initial_capital=10_000.0,
        by_date={},
        unresolved=set(),
    )


def test_touched_row_is_recomputed_via_build_leaderboard_rows(monkeypatch):
    # "a" resolves to a summary (touched); its return_pct comes straight
    # from build_leaderboard_rows, not from any arithmetic in this module.
    monkeypatch.setattr(
        rb,
        "_agent_summary_for_date",
        lambda state, bundle_date: {"cash": 0.0, "positions": [], "currency": "EUR"},
    )
    monkeypatch.setattr(
        rb,
        "build_leaderboard_rows",
        lambda summaries, on: [{"agent": "a", "return_pct": 42.0}],
    )
    bundle = {"leaderboard": [{"agent": "a", "return_pct": 1.23, "rank": 1}]}
    result = restate_bundle_leaderboard(bundle, "2026-08-04", {"a": _state()})

    assert bundle["leaderboard"] == [{"agent": "a", "return_pct": 42.0, "rank": 1}]
    assert result.touched_agents == ["a"]
    assert result.frozen_agents == []
    assert len(result.changes) == 1
    assert result.changes[0].old_return_pct == 1.23
    assert result.changes[0].new_return_pct == 42.0


def test_untouched_row_return_pct_is_byte_identical(monkeypatch):
    # "b" has no eligible summary (either no snapshot row for this date, or
    # the row is one of the 176 left unresolved) -> return_pct must survive
    # unchanged, verbatim.
    monkeypatch.setattr(rb, "_agent_summary_for_date", lambda state, bundle_date: None)
    original_row = {"agent": "b", "return_pct": -3.4567, "rank": 1}
    bundle = {"leaderboard": [dict(original_row)]}

    result = restate_bundle_leaderboard(bundle, "2026-05-20", {"b": _state()})

    assert bundle["leaderboard"][0]["return_pct"] == original_row["return_pct"]
    assert result.frozen_agents == ["b"]
    assert result.changes == []


def test_frozen_agent_missing_from_agent_states_is_also_frozen():
    # An agent with no AgentState at all (e.g. never restated) must be
    # treated exactly like "no eligible summary" -- frozen, not an error.
    original_row = {"agent": "ghost", "return_pct": 5.0, "rank": 1}
    bundle = {"leaderboard": [dict(original_row)]}

    result = restate_bundle_leaderboard(bundle, "2026-05-20", {})

    assert bundle["leaderboard"] == [original_row]
    assert result.frozen_agents == ["ghost"]


def test_rank_is_recomputed_across_touched_and_frozen_rows(monkeypatch):
    # "a" is touched and moves from last to first place; "b" is frozen and
    # its own return_pct never changes, but its rank must still shift to
    # reflect the new sort order -- rank is a property of the whole array.
    monkeypatch.setattr(
        rb,
        "_agent_summary_for_date",
        lambda state, bundle_date: (
            {"cash": 0.0, "positions": [], "currency": "EUR"}
            if state.currency == "touched"
            else None
        ),
    )
    monkeypatch.setattr(
        rb,
        "build_leaderboard_rows",
        lambda summaries, on: [{"agent": "a", "return_pct": 99.0}],
    )
    bundle = {
        "leaderboard": [
            {"agent": "b", "return_pct": 10.0, "rank": 1},
            {"agent": "a", "return_pct": -5.0, "rank": 2},
        ]
    }
    agent_states = {"a": _state("touched"), "b": _state("frozen")}

    restate_bundle_leaderboard(bundle, "2026-08-04", agent_states)

    assert [r["agent"] for r in bundle["leaderboard"]] == ["a", "b"]
    assert bundle["leaderboard"][0]["rank"] == 1
    assert bundle["leaderboard"][1] == {"agent": "b", "return_pct": 10.0, "rank": 2}


@pytest.mark.parametrize("legacy_key", ["eur_mtm", "mtm_eur", "value_eur"])
def test_legacy_eur_value_key_stays_consistent_with_new_return_pct(
    monkeypatch, legacy_key
):
    monkeypatch.setattr(
        rb,
        "_agent_summary_for_date",
        lambda state, bundle_date: {"cash": 0.0, "positions": [], "currency": "EUR"},
    )
    monkeypatch.setattr(
        rb,
        "build_leaderboard_rows",
        lambda summaries, on: [{"agent": "a", "return_pct": 25.0}],
    )
    bundle = {
        "leaderboard": [
            {"agent": "a", "return_pct": 12.0, legacy_key: 11200.0, "rank": 1}
        ]
    }

    restate_bundle_leaderboard(bundle, "2026-05-01", {"a": _state()})

    row = bundle["leaderboard"][0]
    assert row["return_pct"] == 25.0
    # 25% return on the 10,000 EUR inception baseline == 12,500.
    assert row[legacy_key] == pytest.approx(12_500.0)
    assert set(row.keys()) == {"agent", "return_pct", legacy_key, "rank"}


def test_summary_eligible_agent_dropped_by_build_leaderboard_rows_falls_back_to_frozen(
    monkeypatch,
):
    # Regression: "a" has an eligible summary (touched), but
    # build_leaderboard_rows itself drops it — the real path is a held
    # position needing FX conversion with no rate available on this exact
    # bundle_date (engine.valuation.portfolio_mtm returns None; Task 12).
    # Before the fix, a summary-eligible agent that build_leaderboard_rows
    # then dropped ended up in neither computed_rows nor frozen_rows and
    # vanished from bundle["leaderboard"] entirely — violating this
    # function's own "never adds/drops a row" invariant.
    monkeypatch.setattr(
        rb,
        "_agent_summary_for_date",
        lambda state, bundle_date: {"cash": 0.0, "positions": [], "currency": "EUR"},
    )
    monkeypatch.setattr(rb, "build_leaderboard_rows", lambda summaries, on: [])
    original_row = {"agent": "a", "return_pct": 1.23, "rank": 1}
    bundle = {"leaderboard": [dict(original_row)]}

    result = restate_bundle_leaderboard(bundle, "2026-08-04", {"a": _state()})

    assert bundle["leaderboard"] == [original_row]
    assert result.frozen_agents == ["a"]
    assert result.touched_agents == []
    assert result.changes == []


def test_agent_summary_for_date_returns_none_without_a_snapshot_row():
    state = _state()
    assert rb._agent_summary_for_date(state, "2026-08-04") is None


def test_agent_summary_for_date_returns_none_when_row_is_unresolved():
    state = AgentState(
        trades=[],
        currency="EUR",
        initial_capital=10_000.0,
        by_date={"2026-08-04": {"date": "2026-08-04", "cash": 100.0}},
        unresolved={"2026-08-04"},
    )
    assert rb._agent_summary_for_date(state, "2026-08-04") is None


def test_agent_summary_for_date_replays_holdings_to_the_resolved_session_date(
    monkeypatch,
):
    row = {"date": "2026-08-04", "cash": 100.0}
    state = AgentState(
        trades=["trade-sentinel"],
        currency="USD",
        initial_capital=10_000.0,
        by_date={"2026-08-04": row},
        unresolved=set(),
    )
    monkeypatch.setattr(
        rb, "_resolve_session_date", lambda trades, r, ic: date(2026, 8, 5)
    )
    monkeypatch.setattr(
        rb, "replay_holdings", lambda trades, on: ({"AAPL": 2.0}, -500.0)
    )

    summary = rb._agent_summary_for_date(state, "2026-08-04")

    assert summary == {
        "cash": 100.0,
        "positions": [{"ticker": "AAPL", "shares": 2.0}],
        "currency": "USD",
    }
