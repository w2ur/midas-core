"""Portfolio valuation — mark-to-market in native currency and EUR equivalent.

Prices positions from the committed OHLCV store (data/market/ohlcv/) and
converts to EUR via engine/fx.py. Used by daily_log for leaderboard ranking
and by the orchestrator for budget verification.
"""

from __future__ import annotations

from datetime import date

from engine.fx import convert as _fx_convert
from engine.fx import to_eur
from engine.quotes import latest_price as _latest_price


def portfolio_mtm(portfolio_summary: dict, on: date | None = None) -> float | None:
    """Return the mark-to-market value of a portfolio in its native currency.

    A position priced in a currency other than the book's own `currency` is
    converted before being summed, via the same pair of helpers the fill path
    (`engine.paper_broker`) and the restatement engine
    (`engine.restatement.revalue_snapshot`, `scripts/daily_session._compute_positions_value`)
    use: `engine.quotes.latest_price` to read the close already denominated in
    the ticker's ISO currency (pence → pounds for an LSE listing) and
    `engine.fx.convert` to convert that value into the book's currency.
    Reusing these, rather than reimplementing the store read and FX lookup
    here, is what keeps all pricing paths in agreement (this was the third,
    previously un-fixed occurrence of the same conversion gap).

    Parameters
    ----------
    portfolio_summary:
        Dict with keys `cash`, `positions` (list of {ticker, shares} or just tickers)
        and `currency`. For list-of-tickers format, positions are assumed to be zero
        at time of valuation (used before any trades land).
    on:
        Valuation date, defaults to today.

    Returns
    -------
    float | None
        The mark-to-market value in `portfolio_summary["currency"]`, or
        `None` if a held position's currency needs converting and the
        required FX rate is unavailable. Returning `None` for the whole
        book — rather than silently summing everything else and dropping
        just the unconvertible position — avoids understating the book by a
        precise-looking but wrong number; a missing *price* (ticker has no
        OHLCV row at all) is a separate, pre-existing case and is still
        skipped as zero, since there we truly have no information to guess
        from either way. `portfolio_mtm_eur` and `mtm_base_currency`
        propagate this `None` rather than raising, matching the leaderboard
        call path's existing "skip this one book" contract
        (`engine.leaderboard.build_leaderboard_rows` already drops any agent
        whose EUR-MTM is `None`).
    """
    cash = portfolio_summary.get("cash", 0.0)
    positions = portfolio_summary.get("positions", [])
    currency = portfolio_summary.get("currency", "USD")

    # The layering debt this docstring used to record — a pure
    # currency-resolution helper (`_ticker_currency`) living inside the
    # execution/broker layer while three pricing modules imported it, forcing
    # a lazy import here to break a valuation -> paper_broker -> valuation
    # cycle — is resolved: `engine.quotes` is that lower-level module, and
    # this is now a plain top-level import.
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
        quote = _latest_price(ticker, on)
        if quote is None:
            continue
        price, ticker_currency = quote
        native_value = shares * price
        if ticker_currency == currency:
            total += native_value
        else:
            converted = _fx_convert(native_value, ticker_currency, currency, on)
            if converted is None:
                return None
            total += converted
    return total


def portfolio_mtm_eur(portfolio_summary: dict, on: date | None = None) -> float | None:
    """Mark-to-market in EUR. Returns None if FX rate unavailable."""
    native = portfolio_mtm(portfolio_summary, on)
    if native is None:
        return None
    currency = portfolio_summary.get("currency", "USD")
    if currency == "EUR":
        return native
    return to_eur(native, currency, on)


def mtm_base_currency(portfolio_summary: dict, on: date | None = None) -> float | None:
    """Mark-to-market in the portfolio's own base currency (no top-level FX
    conversion to EUR — a held position in a different currency than the
    book's own is still converted into the book's currency, see
    `portfolio_mtm`).

    Wrapper around portfolio_mtm — intended for per-agent drawdown checks.
    Returns `None` under the same condition `portfolio_mtm` does: a held
    position's currency differs from the book's and the FX rate needed to
    convert it is unavailable. Callers must handle `None` explicitly (see
    `engine.paper_broker._drawdown_pct`).
    """
    return portfolio_mtm(portfolio_summary, on)
