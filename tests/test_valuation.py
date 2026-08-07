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


def test_mtm_refuses_the_book_when_a_position_has_no_price(midas_data_root) -> None:
    """Changed 2026-08-07 (review W4.5): a missing price used to be skipped.

    Skipping values the position at zero, and the total still looks plausible
    — 1,100 here, for a book holding 99 shares of something. That was the
    weakest of the three answers this codebase gave to the same question; a
    refusal is at least distinguishable from a right answer downstream, and
    `build_leaderboard_rows` already drops a book it cannot value.
    """
    _seed_ohlcv("MSFT", 300.0)
    summary = {
        "cash": 500.0,
        "currency": "USD",
        "positions": [
            {"ticker": "MSFT", "shares": 2},
            {"ticker": "GHOST", "shares": 99},  # no price row anywhere
        ],
    }
    assert portfolio_mtm(summary, date(2026, 6, 1)) is None


def test_mtm_still_values_a_book_whose_positions_all_price(midas_data_root) -> None:
    """The control: refusing must not become refusing everything."""
    _seed_ohlcv("MSFT", 300.0)
    summary = {
        "cash": 500.0,
        "currency": "USD",
        "positions": [{"ticker": "MSFT", "shares": 2}],
    }
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
    # Regression fix: this test previously used "SGLN.L" — but
    # engine.paper_broker._ticker_currency maps the ".L" suffix to GBP, not
    # EUR, so the ticker actually contradicted the test's own stated intent
    # ("no FX conversion needed"). It only passed pre-fix because
    # portfolio_mtm did no per-position currency resolution at all — exactly
    # the bug this module now fixes. Swapped for "AIR.PA" (".PA" → EUR),
    # which genuinely needs no conversion in a EUR book.
    _seed_ohlcv("AIR.PA", 50.0)
    summary = {
        "cash": 250.0,
        "currency": "EUR",
        "positions": [{"ticker": "AIR.PA", "shares": 2}],  # 100
    }
    # EUR portfolio, EUR-native position: no FX conversion, EUR mtm == native mtm.
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
# Per-position FX conversion (a cross-currency position inside the book)
#
# Regression: portfolio_mtm summed `shares * price` with no per-position FX
# conversion, so a EUR book holding a GBP-listed ticker was mispriced as if
# the GBP close were a EUR close. engine.restatement.revalue_snapshot and
# scripts/daily_session._compute_positions_value already convert correctly
# via the same helpers (engine.quotes.latest_price + engine.fx.convert);
# this was the third, un-fixed occurrence.
# ---------------------------------------------------------------------------


def test_mtm_converts_non_native_currency_position(midas_data_root) -> None:
    # EUR book holding a sterling-listed (".L" suffix) ticker.
    #
    # The store price is 5.00 — five POUNDS. Since 2026-08-07 the store is
    # ISO-denominated: the LSE's pence are divided by 100 at ingest, in
    # scripts.fetch_ohlcv._normalise_vendor_units, so no read path scales.
    # This fixture has now held all three values in turn — 5.0 encoding the
    # original defect (a `.L` quote read as pounds when the store held pence),
    # 500.0 under the read-side normalisation that replaced it, and 5.0 again
    # now that the division moved to ingest. The assertion never changed,
    # because the real-world position never did. Dedicated unit coverage for
    # where the division lives is in tests/test_quotes.py.
    _seed_ohlcv("TSCO.L", 5.0)
    # Stored EURGBP=X close is EUR→GBP; GBP→EUR = 1/close.
    _seed_ohlcv("EURGBP=X", 0.85)
    summary = {
        "cash": 1000.0,
        "currency": "EUR",
        "positions": [{"ticker": "TSCO.L", "shares": 10}],  # 5000p = 50 GBP
    }
    # Native GBP value = 50; converted to EUR at 1/0.85 = 58.823529...
    expected = 1000.0 + 50.0 / 0.85
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(expected)


def test_mtm_returns_none_when_position_fx_rate_unavailable(midas_data_root) -> None:
    # EUR book holding a sterling-listed ticker, but no EURGBP=X rate seeded —
    # the book cannot be accurately valued, so the whole result is None
    # (never a partial total that silently drops the unconvertible position).
    _seed_ohlcv("TSCO.L", 500.0)  # pence; see the test above
    summary = {
        "cash": 1000.0,
        "currency": "EUR",
        "positions": [{"ticker": "TSCO.L", "shares": 10}],
    }
    assert portfolio_mtm(summary, date(2026, 6, 1)) is None


def test_mtm_single_currency_book_unaffected_by_fx_fix(midas_data_root) -> None:
    # Control: a book whose every position is already in its own currency
    # must price identically to the pre-fix behaviour — no FX lookup should
    # even be attempted.
    _seed_ohlcv("MSFT", 300.0)
    _seed_ohlcv("AAPL", 150.0)
    summary = {
        "cash": 1000.0,
        "currency": "USD",
        "positions": [
            {"ticker": "MSFT", "shares": 10},
            {"ticker": "AAPL", "shares": 4},
        ],
    }
    assert portfolio_mtm(summary, date(2026, 6, 1)) == pytest.approx(4600.0)


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


# ---------------------------------------------------------------------------
# W4.5 — one missing-price policy, not three
#
# The review's finding: "snapshots fall back to avg_cost, leaderboard/drawdown
# value at zero, restatement raises — same book, three published answers."
# Two of those answers were numbers, which is what made the divergence
# survivable: a wrong number is indistinguishable from a right one downstream.
# ---------------------------------------------------------------------------


def _unpriceable_book():
    from engine.portfolio import Portfolio, Position

    return Portfolio(
        cash=500.0,
        currency="USD",
        last_updated=date(2026, 6, 1),
        positions=[
            Position(
                ticker="GHOST",
                shares=99.0,
                avg_cost=50.0,
                date_opened=date(2026, 4, 17),
                grid_level=0,
            )
        ],
    )


@pytest.mark.parametrize(
    "reason_case, seed, expected_reason",
    [
        ("no price row at all", None, "NO_PRICE_DATA"),
        ("no resolvable currency", "FOO.ZZ", "CURRENCY_UNRESOLVED"),
    ],
)
def test_value_position_names_the_condition(
    midas_data_root, reason_case, seed, expected_reason
):
    """The two ways a price can be missing are not the same problem.

    A registry gap is fixed in `data/ticker_currencies.json`; a data gap is
    fixed by the fetch job. Collapsing them into one message sends whoever
    reads it to the wrong place.
    """
    from engine.valuation import value_position

    ticker = seed or "GHOST"
    if seed:
        _seed_ohlcv(seed, 42.0)
    result = value_position(ticker, 99.0, "USD", date(2026, 6, 1))
    assert result.value is None
    assert result.reason == expected_reason, reason_case


def test_all_three_valuation_paths_refuse_the_same_fixture(midas_data_root):
    """The done-when for W4.5, stated as an assertion.

    Same position, same date, three call paths that used to answer
    `avg_cost`, `0`, and `raise` respectively.
    """
    from scripts.daily_session import _compute_positions_value
    from engine.restatement import MissingPriceError, revalue_snapshot
    from engine.valuation import portfolio_mtm, value_position

    on = date(2026, 6, 1)
    book = _unpriceable_book()

    # 1. the shared decision
    assert value_position("GHOST", 99.0, "USD", on).reason == "NO_PRICE_DATA"

    # 2. leaderboard / drawdown rail — was: silently zero
    assert portfolio_mtm(book.to_dict(), on) is None

    # 3. snapshot writer — was: avg_cost
    with pytest.raises(MissingPriceError) as snapshot_exc:
        _compute_positions_value(book, on)

    # 4. restatement — already raised; must still name the same reason
    with pytest.raises(MissingPriceError) as restate_exc:
        revalue_snapshot({"GHOST": 99.0}, 500.0, on, "USD")

    assert snapshot_exc.value.reason == "NO_PRICE_DATA"
    assert restate_exc.value.reason == "NO_PRICE_DATA"
