"""Tests for scripts/resolve_manager_outcomes.py (Task C5b) — TDD.

Covers:
- Matured BUY that rose → positive realized_return_pct.
- Matured BUY that fell → negative realized_return_pct.
- Matured SELL that fell → positive realized_return_pct (sign inversion).
- Alpha = realized - MSCI over the same window (flat MSCI → alpha == realized).
- Not-yet-matured decision (insufficient forward rows) → left pending, not resolved.
- Idempotency: running twice → no duplicate (date,ticker,action) entries.
- Cap at 90: feed 95 matured → 90 most-recent kept.
- HOLD positions never produce a resolved entry.
- Output entries contain ONLY the 5 whitelisted fields (no reasoning/render leakage).

All filesystem state uses tmp fixtures — no real data dirs written.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_review(
    review_dir: Path, decision_date: str, positions: list[dict], **kwargs
) -> None:
    """Write a manager-review/{date}.json fixture."""
    review_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "date": decision_date,
        "conviction": kwargs.get("conviction", 8),
        "positions": positions,
        "hold_reasoning": kwargs.get("hold_reasoning", ""),
        "render": kwargs.get("render", "(render)"),
    }
    (review_dir / f"{decision_date}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def _write_ohlcv(store: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    """Write a minimal OHLCV JSONL file for a ticker."""
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{ticker}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row_date, close in rows:
            f.write(
                json.dumps({"date": row_date, "close": close, "adj_close": close})
                + "\n"
            )


def _make_msci_series(entries: list[tuple[str, float]]) -> list[dict]:
    """Build a minimal msci_world series from (date, portfolio_value) tuples."""
    return [{"date": d, "portfolio_value": v, "currency": "EUR"} for d, v in entries]


# ---------------------------------------------------------------------------
# Import helper — avoids module-level import before files exist.
# ---------------------------------------------------------------------------


def _import():
    from scripts.resolve_manager_outcomes import resolve_outcomes

    return resolve_outcomes


# ---------------------------------------------------------------------------
# Core resolution tests
# ---------------------------------------------------------------------------


def test_matured_buy_that_rose_is_positive(tmp_path):
    """BUY + ticker rose → positive realized_return_pct."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "bullish",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    # 5 trading days: entry on/before 2026-05-01, exit 5 days forward
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),  # entry
            ("2026-05-02", 105.0),
            ("2026-05-05", 108.0),
            ("2026-05-06", 110.0),
            ("2026-05-07", 112.0),
            ("2026-05-08", 120.0),  # resolution (5th trading day after entry)
        ],
    )
    msci = _make_msci_series(
        [
            ("2026-05-01", 10000.0),
            ("2026-05-08", 10000.0),  # flat MSCI
        ]
    )

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    entry = result[0]
    assert entry["ticker"] == "AAPL"
    assert entry["action"] == "BUY"
    assert entry["realized_return_pct"] > 0


def test_matured_buy_that_fell_is_negative(tmp_path):
    """BUY + ticker fell → negative realized_return_pct."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "bullish",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 95.0),
            ("2026-05-05", 92.0),
            ("2026-05-06", 90.0),
            ("2026-05-07", 88.0),
            ("2026-05-08", 80.0),  # fell
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    assert result[0]["realized_return_pct"] < 0


def test_matured_sell_that_fell_is_positive(tmp_path):
    """SELL + ticker fell → positive realized_return_pct (sign inverted).

    realized_return_pct for SELL = -(fwd_return), so a declining ticker
    after a SELL call is a correct directional call → positive return.
    """
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "SELL",
                "size_eur": 200,
                "reasoning": "bearish",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 95.0),
            ("2026-05-05", 90.0),
            ("2026-05-06", 85.0),
            ("2026-05-07", 82.0),
            ("2026-05-08", 75.0),  # fell 25%
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    assert result[0]["action"] == "SELL"
    # Ticker fell → SELL was right → positive
    assert result[0]["realized_return_pct"] > 0


def test_alpha_flat_msci_equals_realized(tmp_path):
    """When MSCI is flat over the window, alpha should equal realized_return_pct."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),  # +10%
        ],
    )
    # Flat MSCI: alpha == realized
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    entry = result[0]
    assert abs(entry["alpha_vs_msci_pct"] - entry["realized_return_pct"]) < 0.01


def test_alpha_computation_with_moving_msci(tmp_path):
    """Alpha = realized - MSCI return over same window."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),  # +10%
        ],
    )
    # MSCI rose 5% over the window
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10500.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    entry = result[0]
    # realized = +10%, msci = +5%, alpha = +5%
    assert abs(entry["realized_return_pct"] - 10.0) < 0.1
    assert abs(entry["alpha_vs_msci_pct"] - 5.0) < 0.1


def test_not_yet_matured_left_pending(tmp_path):
    """Decision with fewer than horizon forward rows remains pending (not resolved)."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    # Only 3 forward rows available (horizon = 5) — not mature
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            # No 4th or 5th forward row
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 5, 10),  # "today" close to entry — not enough forward data
        horizon_trading_days=5,
    )

    # Should have 0 resolved (not mature)
    assert result == []


def test_idempotency_no_duplicate_entries(tmp_path):
    """Running resolve_outcomes twice never produces duplicate (date,ticker,action)."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result1 = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )
    # Second run passes result1 as existing_resolved
    result2 = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
        existing_resolved=result1,
    )

    keys = [(e["date"], e["ticker"], e["action"]) for e in result2]
    assert len(keys) == len(set(keys)), "Duplicate (date,ticker,action) entries found"
    assert len(result2) == 1


def test_cap_at_90_keeps_most_recent(tmp_path):
    """Feeding 95 matured decisions → resolved list capped at 90 (most-recent kept)."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"
    msci_entries: list[tuple[str, float]] = []

    # Write 95 decisions with different dates, each matured with 1 trading day horizon.
    from datetime import timedelta

    base_entry = date(2025, 1, 1)
    for i in range(95):
        decision_date = (base_entry + timedelta(days=i * 14)).isoformat()
        exit_date = (base_entry + timedelta(days=i * 14 + 7)).isoformat()
        _write_review(
            review_dir,
            decision_date,
            [
                {
                    "ticker": "AAPL",
                    "action": "BUY",
                    "size_eur": 400,
                    "reasoning": f"r{i}",
                    "entry_guidance": "",
                    "stop_loss": None,
                }
            ],
        )
        path = store / "AAPL.jsonl"
        store.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"date": decision_date, "close": 100.0, "adj_close": 100.0})
                + "\n"
            )
            f.write(
                json.dumps({"date": exit_date, "close": 110.0, "adj_close": 110.0})
                + "\n"
            )
        msci_entries.append((decision_date, 10000.0))
        msci_entries.append((exit_date, 10000.0))

    msci = _make_msci_series(list(dict(msci_entries).items()))

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 12, 31),
        horizon_trading_days=1,
    )

    assert len(result) == 90
    # Most-recent 90 by date should be kept
    dates = sorted(set(e["date"] for e in result))
    oldest_in_result = min(dates)
    # The 5 oldest decision dates (i=0..4) should be dropped
    oldest_possible = (base_entry + timedelta(days=5 * 14)).isoformat()
    assert oldest_in_result >= oldest_possible


def test_hold_positions_never_resolved(tmp_path):
    """HOLD positions must not produce any resolved entry."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "HOLD",
                "size_eur": 0,
                "reasoning": "hold",
                "entry_guidance": "",
                "stop_loss": None,
            },
            {
                "ticker": "MSFT",
                "action": "HOLD",
                "size_eur": 0,
                "reasoning": "hold",
                "entry_guidance": "",
                "stop_loss": None,
            },
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 110.0),
            ("2026-05-05", 115.0),
            ("2026-05-06", 118.0),
            ("2026-05-07", 120.0),
            ("2026-05-08", 125.0),
        ],
    )
    _write_ohlcv(
        store,
        "MSFT",
        [
            ("2026-05-01", 200.0),
            ("2026-05-02", 205.0),
            ("2026-05-05", 210.0),
            ("2026-05-06", 215.0),
            ("2026-05-07", 218.0),
            ("2026-05-08", 220.0),
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert result == []


def test_output_entries_whitelist_only(tmp_path):
    """Resolved entries contain ONLY date, ticker, action, realized_return_pct,
    alpha_vs_msci_pct — no reasoning, render, or other fields leak through."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "SECRET REASONING DO NOT LEAK",
                "entry_guidance": "entry hint",
                "stop_loss": 95.0,
            }
        ],
        render="RENDER DO NOT LEAK",
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 1
    entry = result[0]
    allowed_keys = {
        "date",
        "ticker",
        "action",
        "realized_return_pct",
        "alpha_vs_msci_pct",
    }
    assert set(entry.keys()) == allowed_keys, (
        f"Unexpected keys in resolved entry: {set(entry.keys()) - allowed_keys}"
    )
    assert "reasoning" not in entry
    assert "render" not in entry
    assert "entry_guidance" not in entry
    assert "stop_loss" not in entry


def test_resolved_json_skipped(tmp_path):
    """The resolved.json file itself is not processed as a decision."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    store = tmp_path / "ohlcv"
    review_dir.mkdir(parents=True, exist_ok=True)

    # Write a valid resolved.json — it must be skipped as a decision source
    resolved_payload = [
        {
            "date": "2026-04-01",
            "ticker": "FAKE",
            "action": "BUY",
            "realized_return_pct": 99.0,
            "alpha_vs_msci_pct": 99.0,
        }
    ]
    (review_dir / "resolved.json").write_text(json.dumps(resolved_payload))

    _write_review(
        review_dir,
        "2026-05-01",
        [
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            }
        ],
    )
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
        existing_resolved=resolved_payload,
    )

    # Should have 1 (AAPL) + 1 (FAKE from existing) = 2, but FAKE had no ticker in store
    # meaning only AAPL was resolved from the review dir; FAKE carries from existing
    assert any(e["ticker"] == "AAPL" for e in result)
    # No double-FAKE from re-resolving resolved.json
    fake_entries = [e for e in result if e["ticker"] == "FAKE"]
    assert len(fake_entries) == 1


def test_result_sorted_deterministically(tmp_path):
    """Output is sorted (date, ticker, action) — deterministic order."""
    resolve_outcomes = _import()

    # Both tickers in the same review file in reverse-alpha order (MSFT before AAPL)
    # to verify the sort, not the insertion order.
    review_dir2 = tmp_path / "manager-review"
    _write_review(
        review_dir2,
        "2026-05-01",
        [
            {
                "ticker": "MSFT",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            },
            {
                "ticker": "AAPL",
                "action": "BUY",
                "size_eur": 400,
                "reasoning": "test",
                "entry_guidance": "",
                "stop_loss": None,
            },
        ],
    )
    store = tmp_path / "ohlcv"
    _write_ohlcv(
        store,
        "AAPL",
        [
            ("2026-05-01", 100.0),
            ("2026-05-02", 102.0),
            ("2026-05-05", 104.0),
            ("2026-05-06", 106.0),
            ("2026-05-07", 108.0),
            ("2026-05-08", 110.0),
        ],
    )
    _write_ohlcv(
        store,
        "MSFT",
        [
            ("2026-05-01", 200.0),
            ("2026-05-02", 205.0),
            ("2026-05-05", 210.0),
            ("2026-05-06", 215.0),
            ("2026-05-07", 218.0),
            ("2026-05-08", 220.0),
        ],
    )
    msci = _make_msci_series([("2026-05-01", 10000.0), ("2026-05-08", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir2,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert len(result) == 2
    # Should be sorted (date, ticker, action) — AAPL before MSFT
    assert result[0]["ticker"] == "AAPL"
    assert result[1]["ticker"] == "MSFT"


def test_round_trip_via_write_resolved(tmp_path):
    """write_resolved → read back → same content (atomic write integration test)."""
    from scripts.resolve_manager_outcomes import write_resolved

    review_dir = tmp_path / "manager-review"
    review_dir.mkdir(parents=True)
    resolved_path = review_dir / "resolved.json"

    entries = [
        {
            "date": "2026-05-01",
            "ticker": "AAPL",
            "action": "BUY",
            "realized_return_pct": 10.0,
            "alpha_vs_msci_pct": 5.0,
        },
    ]
    write_resolved(entries, resolved_path)

    read_back = json.loads(resolved_path.read_text(encoding="utf-8"))
    assert read_back == entries


def test_malformed_and_empty_review_files_do_not_crash(tmp_path):
    """Empty-positions and malformed-JSON review files are skipped, no crash."""
    resolve_outcomes = _import()

    review_dir = tmp_path / "manager-review"
    review_dir.mkdir(parents=True)
    store = tmp_path / "ohlcv"

    # Malformed JSON — should be silently skipped.
    (review_dir / "2026-05-01.json").write_text("not valid json", encoding="utf-8")

    # Empty positions list — no entries to resolve.
    _write_review(review_dir, "2026-05-02", [])

    msci = _make_msci_series([("2026-05-01", 10000.0)])

    result = resolve_outcomes(
        review_dir=review_dir,
        store=store,
        msci_series=msci,
        today=date(2026, 6, 1),
        horizon_trading_days=5,
    )

    assert result == []
