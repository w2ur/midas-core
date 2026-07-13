"""Tests for engine.valuation helpers.

portfolio_mtm / portfolio_mtm_eur feed the drawdown halt (paper_broker) and the
EUR leaderboard (leaderboard), so they carry real weight. Tests seed an
isolated OHLCV store via the midas_data_root fixture (MIDAS_DATA_DIR redirect).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engine.config import get_config
from engine.valuation import mtm_base_currency, portfolio_mtm, portfolio_mtm_eur


def _seed_ohlcv(ticker: str, close: float, on: str = "2026-06-01") -> None:
    """Write a single-row OHLCV JSONL for `ticker` into the config store."""
    store: Path = get_config().ohlcv_dir
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{ticker}.jsonl").write_text(
        json.dumps({"date": on, "close": close}) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Existing behaviour
# ---------------------------------------------------------------------------


def test_mtm_base_currency_cash_only() -> None:
    summary = {"cash": 1000.0, "currency": "EUR", "positions": []}
    assert mtm_base_currency(summary) == 1000.0


# ---------------------------------------------------------------------------
# Multi-position pricing
# ---------------------------------------------------------------------------


def test_mtm_multi_position(midas_data_root) -> None:
    _seed_ohlcv("MSFT", 300.0)
    _seed_ohlcv("AAPL", 150.0)
    summary = {
        "cash": 1000.0,
        "currency": "USD",
        "positions": [
            {"ticker": "MSFT", "shares": 10},  # 3000
            {"ticker": "AAPL", "shares": 4},  # 600
        ],
    }
    # 1000 cash + 3000 + 600 = 4600
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(4600.0)


# ---------------------------------------------------------------------------
# Missing-ticker price
# ---------------------------------------------------------------------------


def test_mtm_missing_ticker_price_is_skipped(midas_data_root) -> None:
    _seed_ohlcv("MSFT", 300.0)
    summary = {
        "cash": 500.0,
        "currency": "USD",
        "positions": [
            {"ticker": "MSFT", "shares": 2},  # 600
            {"ticker": "GHOST", "shares": 99},  # no price row → contributes 0
        ],
    }
    # A missing price never crashes and never guesses: 500 + 600 = 1100.
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(1100.0)


# ---------------------------------------------------------------------------
# String-position (pre-trade) zero-value
# ---------------------------------------------------------------------------


def test_mtm_string_positions_are_zero_valued(midas_data_root) -> None:
    _seed_ohlcv("MSFT", 300.0)
    # Legacy shape: positions as bare ticker strings (assumed zero shares).
    summary = {"cash": 2000.0, "currency": "USD", "positions": ["MSFT", "AAPL"]}
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# EUR conversion with a real FX fixture
# ---------------------------------------------------------------------------


def test_mtm_eur_with_fx_fixture(midas_data_root) -> None:
    _seed_ohlcv("MSFT", 400.0)
    # Stored EURUSD=X close is EUR→USD; USD→EUR = 1/close. 1.25 → 0.8 EUR/USD.
    _seed_ohlcv("EURUSD=X", 1.25)
    summary = {
        "cash": 100.0,
        "currency": "USD",
        "positions": [{"ticker": "MSFT", "shares": 1}],  # 400 USD
    }
    # Native = 100 + 400 = 500 USD; EUR = 500 * (1/1.25) = 400 EUR.
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(500.0)
    assert portfolio_mtm_eur(summary, date(2026, 6, 1)) == pytest.approx(400.0)


def test_mtm_eur_native_eur_needs_no_fx(midas_data_root) -> None:
    _seed_ohlcv("SGLN.L", 50.0)
    summary = {
        "cash": 250.0,
        "currency": "EUR",
        "positions": [{"ticker": "SGLN.L", "shares": 2}],  # 100
    }
    # EUR portfolio: no FX conversion, EUR mtm == native mtm.
    assert portfolio_mtm_eur(summary, date(2026, 6, 1)) == pytest.approx(350.0)


def test_mtm_eur_returns_none_without_fx_rate(midas_data_root) -> None:
    _seed_ohlcv("MSFT", 400.0)
    # No EURUSD=X seeded → USD→EUR rate unavailable → None (not a silent 0).
    summary = {
        "cash": 100.0,
        "currency": "USD",
        "positions": [{"ticker": "MSFT", "shares": 1}],
    }
    assert portfolio_mtm_eur(summary, date(2026, 6, 1)) is None


# ---------------------------------------------------------------------------
# Property: mtm(cash, positions) == cash + Σ shares × price  (non-negative inputs)
# ---------------------------------------------------------------------------

_PRICES = {"AAA": 10.0, "BBB": 20.0, "CCC": 5.0}
_amount = st.floats(
    min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
)


@settings(max_examples=400, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(cash=_amount, a=_amount, b=_amount, c=_amount)
def test_mtm_is_cash_plus_position_value(midas_data_root, cash, a, b, c) -> None:
    for ticker, price in _PRICES.items():
        _seed_ohlcv(ticker, price)
    summary = {
        "cash": cash,
        "currency": "USD",
        "positions": [
            {"ticker": "AAA", "shares": a},
            {"ticker": "BBB", "shares": b},
            {"ticker": "CCC", "shares": c},
        ],
    }
    expected = cash + a * _PRICES["AAA"] + b * _PRICES["BBB"] + c * _PRICES["CCC"]
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(
        expected, rel=1e-9, abs=1e-6
    )
