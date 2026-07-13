"""Portfolio valuation — mark-to-market in native currency and EUR equivalent.

Prices positions from the committed OHLCV store (data/market/ohlcv/) and
converts to EUR via engine/fx.py. Used by daily_log for leaderboard ranking
and by the orchestrator for budget verification.
"""

from __future__ import annotations

from datetime import date

from engine.fx import to_eur
from engine.ohlcv_store import latest_close_on_or_before as _latest_close


def portfolio_mtm(portfolio_summary: dict, on: date | None = None) -> float:
    """Return the mark-to-market value of a portfolio in its native currency.

    Parameters
    ----------
    portfolio_summary:
        Dict with keys `cash`, `positions` (list of {ticker, shares} or just tickers)
        and `currency`. For list-of-tickers format, positions are assumed to be zero
        at time of valuation (used before any trades land).
    on:
        Valuation date, defaults to today.
    """
    cash = portfolio_summary.get("cash", 0.0)
    positions = portfolio_summary.get("positions", [])
    # Compatibility: positions might be a list of ticker strings or a list of dicts.
    total = cash
    for p in positions:
        if isinstance(p, dict):
            ticker = p.get("ticker")
            shares = p.get("shares", 0)
        else:
            ticker = p
            shares = 0
        if not ticker or shares == 0:
            continue
        price = _latest_close(ticker, on)
        if price is not None:
            total += shares * price
    return total


def portfolio_mtm_eur(portfolio_summary: dict, on: date | None = None) -> float | None:
    """Mark-to-market in EUR. Returns None if FX rate unavailable."""
    native = portfolio_mtm(portfolio_summary, on)
    currency = portfolio_summary.get("currency", "USD")
    if currency == "EUR":
        return native
    return to_eur(native, currency, on)


def mtm_base_currency(portfolio_summary: dict, on: date | None = None) -> float:
    """Mark-to-market in the portfolio's own base currency (no FX conversion).

    Wrapper around portfolio_mtm that preserves the native currency — intended
    for per-agent drawdown checks (same currency throughout, no FX noise).
    """
    return portfolio_mtm(portfolio_summary, on)
