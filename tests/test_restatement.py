"""Tests for engine.restatement — trade replay and snapshot re-valuation.

Regression: 4b6b8556 — the committed OHLCV store had 29,348 rows revised
(partial bars + 11 stock splits whose pre-split history was un-adjusted).
Fills were correct; every published portfolio valuation was priced off
wrong closes. These primitives re-derive holdings from the trade ledger and
re-price them from the (now-corrected) store, without ever touching
recorded cash — see the module docstring for why.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from engine.config import get_config
from engine.restatement import MissingPriceError, replay_holdings, revalue_snapshot

TRADES = [
    {
        "timestamp": "2026-04-17T20:00:00+00:00",
        "action": "BUY",
        "ticker": "BTC-EUR",
        "shares": 0.1,
        "price": 50000.0,
        "total": 5000.0,
        "fees": 5.0,
    },
    {
        "timestamp": "2026-05-02T20:00:00+00:00",
        "action": "BUY",
        "ticker": "ETH-EUR",
        "shares": 2.0,
        "price": 2000.0,
        "total": 4000.0,
        "fees": 4.0,
    },
    {
        "timestamp": "2026-06-01T20:00:00+00:00",
        "action": "SELL",
        "ticker": "BTC-EUR",
        "shares": 0.04,
        "price": 60000.0,
        "total": 2400.0,
        "fees": 2.4,
    },
]


# ---------------------------------------------------------------------------
# replay_holdings
# ---------------------------------------------------------------------------


def test_replay_holdings_before_any_trade_is_empty():
    assert replay_holdings(TRADES, date(2026, 4, 16)) == ({}, 0.0)


def test_replay_holdings_accumulates_buys():
    pos, cash = replay_holdings(TRADES, date(2026, 5, 2))
    assert pos == {"BTC-EUR": 0.1, "ETH-EUR": 2.0}
    assert cash == pytest.approx(-9009.0)  # -(5000+5) - (4000+4)


def test_replay_holdings_applies_sells_and_credits_cash_net_of_fees():
    pos, cash = replay_holdings(TRADES, date(2026, 6, 1))
    assert pos["BTC-EUR"] == pytest.approx(0.06)
    assert cash == pytest.approx(-9009.0 + 2400.0 - 2.4)


def test_replay_holdings_is_inclusive_of_the_as_of_date():
    # A trade timestamped on as_of IS included — sessions run at 20:00 UTC and
    # the snapshot they produce reflects that day's fills.
    pos, _ = replay_holdings(TRADES, date(2026, 4, 17))
    assert pos == {"BTC-EUR": 0.1}


def test_replay_holdings_drops_a_position_closed_to_zero():
    trades = TRADES + [
        {
            "timestamp": "2026-06-02T20:00:00+00:00",
            "action": "SELL",
            "ticker": "BTC-EUR",
            "shares": 0.06,
            "price": 61000.0,
            "total": 3660.0,
            "fees": 3.7,
        }
    ]
    pos, _ = replay_holdings(trades, date(2026, 6, 2))
    assert "BTC-EUR" not in pos
    assert set(pos) == {"ETH-EUR"}


def test_replay_holdings_ignores_trades_out_of_timestamp_order_in_the_input():
    # The trade log is append-only but the replayed order must be chronological
    # regardless of how the caller happens to hand the list in.
    shuffled = [TRADES[2], TRADES[0], TRADES[1]]
    pos, cash = replay_holdings(shuffled, date(2026, 6, 1))
    assert pos["BTC-EUR"] == pytest.approx(0.06)
    assert cash == pytest.approx(-9009.0 + 2400.0 - 2.4)


def test_replay_holdings_rejects_an_unknown_action():
    bad = [{**TRADES[0], "action": "SHORT"}]
    with pytest.raises(ValueError, match="SHORT"):
        replay_holdings(bad, date(2026, 4, 17))


# ---------------------------------------------------------------------------
# revalue_snapshot
# ---------------------------------------------------------------------------


def _seed_ohlcv(ticker: str, close: float, on: str = "2026-06-01") -> None:
    """Write a single-row OHLCV JSONL for `ticker` into the config store."""
    store: Path = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{ticker}.jsonl").write_text(
        json.dumps({"date": on, "close": close}) + "\n", encoding="utf-8"
    )


def test_revalue_snapshot_single_currency_book(midas_data_root):
    _seed_ohlcv("MSFT", 300.0)
    _seed_ohlcv("AAPL", 150.0)
    positions = {"MSFT": 10.0, "AAPL": 4.0}
    portfolio_value, positions_value = revalue_snapshot(
        positions, cash=1000.0, market_date=date(2026, 6, 1), currency="USD"
    )
    # positions_value = 10*300 + 4*150 = 3600; portfolio_value = cash + positions_value.
    assert positions_value == pytest.approx(3600.0)
    assert portfolio_value == pytest.approx(4600.0)


def test_revalue_snapshot_cash_only_book_needs_no_prices(midas_data_root):
    portfolio_value, positions_value = revalue_snapshot(
        {}, cash=750.0, market_date=date(2026, 6, 1), currency="EUR"
    )
    assert positions_value == 0.0
    assert portfolio_value == 750.0


def test_revalue_snapshot_missing_price_raises_named_exception(midas_data_root):
    # MSFT has no seeded price at all — must raise, never silently price at 0.
    with pytest.raises(MissingPriceError, match="MSFT"):
        revalue_snapshot(
            {"MSFT": 5.0}, cash=0.0, market_date=date(2026, 6, 1), currency="USD"
        )


def test_revalue_snapshot_converts_a_foreign_currency_holding(midas_data_root):
    # RTX is a plain-suffix ticker → _ticker_currency resolves it to USD; the
    # book is EUR. Stored EURUSD=X close is EUR→USD, so USD→EUR = 1/rate.
    _seed_ohlcv("RTX", 100.0)
    _seed_ohlcv("EURUSD=X", 1.25)
    positions_value_expected = 10 * 100.0 * (1 / 1.25)  # 800 EUR
    portfolio_value, positions_value = revalue_snapshot(
        {"RTX": 10.0}, cash=200.0, market_date=date(2026, 6, 1), currency="EUR"
    )
    assert positions_value == pytest.approx(positions_value_expected)
    assert portfolio_value == pytest.approx(200.0 + positions_value_expected)


def test_revalue_snapshot_missing_fx_rate_raises_named_exception(midas_data_root):
    # RTX priced fine, but no EURUSD=X fixture → FX rate unavailable → must raise.
    _seed_ohlcv("RTX", 100.0)
    with pytest.raises(MissingPriceError):
        revalue_snapshot(
            {"RTX": 10.0}, cash=0.0, market_date=date(2026, 6, 1), currency="EUR"
        )


def test_revalue_snapshot_same_currency_ticker_needs_no_fx(midas_data_root):
    # BTC-EUR resolves to EUR via the -EUR suffix heuristic; book is EUR too.
    _seed_ohlcv("BTC-EUR", 60000.0)
    portfolio_value, positions_value = revalue_snapshot(
        {"BTC-EUR": 0.1}, cash=50.0, market_date=date(2026, 6, 1), currency="EUR"
    )
    assert positions_value == pytest.approx(6000.0)
    assert portfolio_value == pytest.approx(6050.0)


def test_revalue_snapshot_ignores_a_dust_position_below_epsilon(midas_data_root):
    # A position at ~0 shares (float drift from replay) must not demand a price.
    portfolio_value, positions_value = revalue_snapshot(
        {"GHOST": 1e-12}, cash=100.0, market_date=date(2026, 6, 1), currency="USD"
    )
    assert positions_value == 0.0
    assert portfolio_value == 100.0
