"""Tests for engine.tax_shadow — after-tax shadow ledger (reporting only).

TDD: these tests were written before the implementation.

French tax basis:
- Securities and crypto are siloed regimes; losses in one cannot offset gains in the other.
- PFU = 30% flat on net annual REALIZED gains per regime.
- Securities: per-ticker weighted-average cost basis (PRU).
- Crypto: global-portfolio weighted-average (PVCT method, approximated).

Spec:
    BUY 10@100 total=1000 fee=1.25
        → basis per share = (1000 + 1.25) / 10 = 100.125
    SELL 10@120 total=1200 fee=1.25
        → proceeds_net = 1200 - 1.25 = 1198.75
        → cost_of_sold  = 100.125 * 10 = 1001.25
        → gain = 1198.75 - 1001.25 = 197.50
        → PFU = 0.30 * 197.50 = 59.25
"""

from __future__ import annotations

import pytest

from engine.tax_shadow import compute_tax_shadow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buy(year: str, ticker: str, total: float, fees: float) -> dict:
    return {
        "id": f"ord_{year}_test_buy_{ticker}",
        "timestamp": f"{year}-01-15T10:00:00+00:00",
        "action": "BUY",
        "ticker": ticker,
        "shares": 10.0,
        "price": total / 10.0,
        "total": total,
        "fees": fees,
        "reasoning": "test buy",
    }


def _make_sell(
    year: str, ticker: str, total: float, fees: float, shares: float = 10.0
) -> dict:
    return {
        "id": f"ord_{year}_test_sell_{ticker}",
        "timestamp": f"{year}-06-15T10:00:00+00:00",
        "action": "SELL",
        "ticker": ticker,
        "shares": shares,
        "price": total / shares,
        "total": total,
        "fees": fees,
        "reasoning": "test sell",
    }


# ---------------------------------------------------------------------------
# Securities round-trip
# ---------------------------------------------------------------------------


@pytest.mark.live_cast
def test_securities_full_sell_gain_and_pfu():
    """BUY 10@total=1000 fee=1.25, SELL 10@total=1200 fee=1.25 → gain=197.50, PFU=59.25."""
    trades = [
        _make_buy("2026", "AAPL", total=1000.0, fees=1.25),
        _make_sell("2026", "AAPL", total=1200.0, fees=1.25),
    ]
    result = compute_tax_shadow(trades)

    sec = result["securities"]
    assert sec["realized_gain_by_year"]["2026"] == pytest.approx(197.50, abs=0.01)
    assert sec["pfu_due_by_year"]["2026"] == pytest.approx(59.25, abs=0.01)
    assert sec["lifetime_realized"] == pytest.approx(197.50, abs=0.01)
    assert sec["lifetime_pfu"] == pytest.approx(59.25, abs=0.01)
    # Crypto regime must be zero
    assert result["crypto"]["lifetime_realized"] == pytest.approx(0.0, abs=0.001)
    assert result["crypto"]["lifetime_pfu"] == pytest.approx(0.0, abs=0.001)


@pytest.mark.live_cast
def test_securities_partial_sell_uses_weighted_average_remaining_basis():
    """BUY 10@1000 fee=1.25, SELL 5@total=650 fee=1.25.

    avg_cost_per_share = (1000 + 1.25) / 10 = 100.125
    cost_of_sold = 100.125 * 5 = 500.625
    proceeds_net = 650 - 1.25 = 648.75
    gain = 648.75 - 500.625 = 148.125 → rounds to 148.12 (banker's rounding at 2dp)

    Remaining basis: (1001.25 - 500.625) = 500.625 for 5 shares → 100.125/share (unchanged)

    Note: output values are rounded to 2dp; tolerance is abs=0.01 to accommodate rounding.
    """
    trades = [
        _make_buy("2026", "AAPL", total=1000.0, fees=1.25),
        _make_sell("2026", "AAPL", total=650.0, fees=1.25, shares=5.0),
    ]
    result = compute_tax_shadow(trades)

    sec = result["securities"]
    # Output is 2dp-rounded; exact pre-rounding gain = 148.125, rounds to 148.12
    assert sec["realized_gain_by_year"]["2026"] == pytest.approx(148.125, abs=0.01)
    assert sec["pfu_due_by_year"]["2026"] == pytest.approx(148.125 * 0.30, abs=0.01)


def test_securities_loss_year_zero_pfu_loss_recorded():
    """Loss year → PFU = 0, loss is recorded in realized_loss_by_year."""
    trades = [
        _make_buy("2026", "AAPL", total=1000.0, fees=1.25),
        _make_sell("2026", "AAPL", total=800.0, fees=1.25),
    ]
    result = compute_tax_shadow(trades)

    sec = result["securities"]
    # gain is negative — no PFU
    assert (
        "2026" not in sec["pfu_due_by_year"]
        or sec["pfu_due_by_year"].get("2026", 0.0) == 0.0
    )
    # loss_by_year should record the loss (positive magnitude)
    assert sec["realized_loss_by_year"]["2026"] > 0
    assert sec["lifetime_pfu"] == pytest.approx(0.0, abs=0.001)


@pytest.mark.live_cast
def test_regimes_never_net_crypto_loss_does_not_offset_securities_gain():
    """Crypto loss in 2026 must NOT reduce PFU on securities gain."""
    # Securities: gain
    sec_buy = _make_buy("2026", "AAPL", total=1000.0, fees=1.25)
    sec_sell = _make_sell("2026", "AAPL", total=1200.0, fees=1.25)
    # Crypto: loss (BUY high, SELL low)
    crypto_buy = _make_buy("2026", "BTC-EUR", total=5000.0, fees=20.0)
    crypto_sell = _make_sell("2026", "BTC-EUR", total=3000.0, fees=12.0)

    trades = [sec_buy, sec_sell, crypto_buy, crypto_sell]
    result = compute_tax_shadow(trades)

    sec = result["securities"]
    crypto = result["crypto"]

    # Securities PFU on gain ≈ 59.25, unaffected by crypto loss
    assert sec["pfu_due_by_year"].get("2026", 0.0) == pytest.approx(59.25, abs=0.01)
    # Crypto net is negative → 0 PFU, loss recorded
    assert crypto["pfu_due_by_year"].get("2026", 0.0) == pytest.approx(0.0, abs=0.001)
    assert crypto["realized_loss_by_year"]["2026"] > 0


@pytest.mark.live_cast
def test_crypto_disposal_computes_gain_in_crypto_regime_only():
    """Crypto gain stays in crypto regime; securities regime stays zero."""
    trades = [
        _make_buy("2026", "BTC-EUR", total=5000.0, fees=20.0),
        _make_sell("2026", "BTC-EUR", total=6000.0, fees=24.0),
    ]
    result = compute_tax_shadow(trades)

    crypto = result["crypto"]
    sec = result["securities"]

    # Crypto has a gain
    assert crypto["lifetime_realized"] > 0
    assert crypto["lifetime_pfu"] > 0

    # Securities has no activity
    assert sec["lifetime_realized"] == pytest.approx(0.0, abs=0.001)
    assert sec["lifetime_pfu"] == pytest.approx(0.0, abs=0.001)


def test_crypto_gain_uses_pvct_method_label():
    """Crypto output must carry the per-coin-PRU-approx method label for transparency."""
    trades = [
        _make_buy("2026", "BTC-EUR", total=5000.0, fees=20.0),
        _make_sell("2026", "BTC-EUR", total=6000.0, fees=24.0),
    ]
    result = compute_tax_shadow(trades)
    assert result["crypto"]["method"] == "per-coin-PRU-approx"


def test_empty_trades_zero_everything_no_crash():
    """Empty trade list → all zeros, no exception."""
    result = compute_tax_shadow([])

    assert result["securities"]["lifetime_realized"] == 0.0
    assert result["securities"]["lifetime_pfu"] == 0.0
    assert result["crypto"]["lifetime_realized"] == 0.0
    assert result["crypto"]["lifetime_pfu"] == 0.0
    assert result["securities"]["realized_gain_by_year"] == {}
    assert result["securities"]["realized_loss_by_year"] == {}
    assert result["crypto"]["realized_gain_by_year"] == {}
    assert result["crypto"]["realized_loss_by_year"] == {}


def test_output_json_shape_complete():
    """Output dict must contain all required top-level and regime keys."""
    result = compute_tax_shadow([])

    # Top-level
    assert "agent" in result
    assert "generated_at" in result
    assert "securities" in result
    assert "crypto" in result
    assert "notes" in result

    # Per-regime keys
    for regime in ("securities", "crypto"):
        r = result[regime]
        assert "realized_gain_by_year" in r
        assert "realized_loss_by_year" in r
        assert "pfu_due_by_year" in r
        assert "lifetime_realized" in r
        assert "lifetime_pfu" in r

    # Crypto-specific
    assert "method" in result["crypto"]


def test_output_monetary_values_rounded_to_2dp():
    """Monetary values in output must be rounded to 2 decimal places."""
    trades = [
        _make_buy("2026", "AAPL", total=1000.0, fees=1.25),
        _make_sell("2026", "AAPL", total=1200.0, fees=1.25),
    ]
    result = compute_tax_shadow(trades)
    gain = result["securities"]["realized_gain_by_year"]["2026"]
    # Check it's rounded to 2dp (round to 2dp should equal itself)
    assert round(gain, 2) == gain


def test_multi_year_aggregates():
    """Gains across two years aggregate correctly in lifetime totals."""
    buy_2025 = {
        "id": "ord_buy_2025",
        "timestamp": "2025-03-01T10:00:00+00:00",
        "action": "BUY",
        "ticker": "AAPL",
        "shares": 10.0,
        "price": 100.0,
        "total": 1000.0,
        "fees": 1.25,
        "reasoning": "test",
    }
    sell_2025 = {
        "id": "ord_sell_2025",
        "timestamp": "2025-09-01T10:00:00+00:00",
        "action": "SELL",
        "ticker": "AAPL",
        "shares": 10.0,
        "price": 120.0,
        "total": 1200.0,
        "fees": 1.25,
        "reasoning": "test",
    }
    buy_2026 = {
        "id": "ord_buy_2026",
        "timestamp": "2026-01-01T10:00:00+00:00",
        "action": "BUY",
        "ticker": "MSFT",
        "shares": 5.0,
        "price": 200.0,
        "total": 1000.0,
        "fees": 1.25,
        "reasoning": "test",
    }
    sell_2026 = {
        "id": "ord_sell_2026",
        "timestamp": "2026-07-01T10:00:00+00:00",
        "action": "SELL",
        "ticker": "MSFT",
        "shares": 5.0,
        "price": 240.0,
        "total": 1200.0,
        "fees": 1.25,
        "reasoning": "test",
    }
    result = compute_tax_shadow([buy_2025, sell_2025, buy_2026, sell_2026])
    sec = result["securities"]
    assert "2025" in sec["realized_gain_by_year"]
    assert "2026" in sec["realized_gain_by_year"]
    # Lifetime should be sum of both years
    total_gain = (
        sec["realized_gain_by_year"]["2025"] + sec["realized_gain_by_year"]["2026"]
    )
    assert sec["lifetime_realized"] == pytest.approx(total_gain, abs=0.01)


def test_fx_ticker_in_securities_regime():
    """FX tickers (e.g. EURUSD=X) are classified as securities (non-crypto) by fees.classify_ticker."""
    trades = [
        {
            "id": "ord_fx_buy",
            "timestamp": "2026-01-01T10:00:00+00:00",
            "action": "BUY",
            "ticker": "EURUSD=X",
            "shares": 100.0,
            "price": 1.10,
            "total": 110.0,
            "fees": 0.002,
            "reasoning": "fx test",
        },
        {
            "id": "ord_fx_sell",
            "timestamp": "2026-06-01T10:00:00+00:00",
            "action": "SELL",
            "ticker": "EURUSD=X",
            "shares": 100.0,
            "price": 1.15,
            "total": 115.0,
            "fees": 0.0023,
            "reasoning": "fx sell test",
        },
    ]
    result = compute_tax_shadow(trades)
    # FX classified as equity/fx → goes into securities regime
    assert result["securities"]["lifetime_realized"] != 0
    assert result["crypto"]["lifetime_realized"] == pytest.approx(0.0, abs=0.001)


def test_buy_only_no_realized_gain():
    """BUY with no SELL → no realized gain, no PFU."""
    trades = [_make_buy("2026", "AAPL", total=5000.0, fees=2.50)]
    result = compute_tax_shadow(trades)
    assert result["securities"]["lifetime_realized"] == pytest.approx(0.0, abs=0.001)
    assert result["securities"]["lifetime_pfu"] == pytest.approx(0.0, abs=0.001)


@pytest.mark.live_cast
def test_oversell_does_not_corrupt_pool():
    """BUY 5 then SELL 10 — oversell must floor pool at 0, not go negative.

    A subsequent BUY+SELL in the same year must compute a clean gain on the
    fresh position without being tainted by the clamped oversell state.
    """
    # Step 1: buy 5 shares, then attempt to sell 10 (oversell by 5).
    buy5 = {
        "id": "ord_buy5",
        "timestamp": "2026-01-10T10:00:00+00:00",
        "action": "BUY",
        "ticker": "AAPL",
        "shares": 5.0,
        "price": 100.0,
        "total": 500.0,
        "fees": 0.0,
        "reasoning": "oversell test buy",
    }
    sell10 = {
        "id": "ord_sell10",
        "timestamp": "2026-02-10T10:00:00+00:00",
        "action": "SELL",
        "ticker": "AAPL",
        "shares": 10.0,
        "price": 110.0,
        "total": 1100.0,
        "fees": 0.0,
        "reasoning": "oversell test sell",
    }
    # Step 2: fresh BUY+SELL at known prices so we can verify the gain exactly.
    buy_fresh = {
        "id": "ord_buy_fresh",
        "timestamp": "2026-03-10T10:00:00+00:00",
        "action": "BUY",
        "ticker": "AAPL",
        "shares": 10.0,
        "price": 200.0,
        "total": 2000.0,
        "fees": 0.0,
        "reasoning": "clean buy",
    }
    sell_fresh = {
        "id": "ord_sell_fresh",
        "timestamp": "2026-04-10T10:00:00+00:00",
        "action": "SELL",
        "ticker": "AAPL",
        "shares": 10.0,
        "price": 250.0,
        "total": 2500.0,
        "fees": 0.0,
        "reasoning": "clean sell",
    }

    result = compute_tax_shadow([buy5, sell10, buy_fresh, sell_fresh])
    sec = result["securities"]

    # The pool must never go negative — the gain from the oversell SELL uses
    # the capped fraction (5/5 = 1.0) so allocated_cost = full basis = 500.
    # gain_oversell = 1100 - 500 = 600.
    # The clean subsequent trade: avg_cost = 200, gain = 2500 - 2000 = 500.
    # Total 2026 gain = 600 + 500 = 1100 (no negative basis corruption).
    assert sec["lifetime_realized"] == pytest.approx(1100.0, abs=0.01)
    assert sec["lifetime_pfu"] == pytest.approx(1100.0 * 0.30, abs=0.01)
    # Verify no negative state leaked — lifetime_realized must be positive (not NaN/negative).
    assert sec["lifetime_realized"] > 0
