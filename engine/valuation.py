"""Portfolio valuation — mark-to-market in native currency and EUR equivalent.

Prices positions from the committed OHLCV store (data/market/ohlcv/) and
converts to EUR via engine/fx.py. Used by daily_log for leaderboard ranking
and by the orchestrator for budget verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from engine.fx import convert as _fx_convert
from engine.fx import to_eur
from engine.quotes import latest_price as _latest_price
from engine.quotes import ticker_currency as _ticker_currency


@dataclass(frozen=True)
class PositionValuation:
    """One position's value in a book's currency, or why it has none.

    `reason` uses the broker's own vocabulary (`NO_PRICE_DATA`,
    `NO_FX_RATE`, `CURRENCY_UNRESOLVED`) so a valuation failure and a fill
    rejection describe the same condition with the same word.
    """

    value: float | None
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


def value_position(
    ticker: str, shares: float, book_currency: str, on: date | None = None
) -> PositionValuation:
    """Value one position in `book_currency` — the single pricing decision.

    This exists because the same question was being answered three different
    ways. Asked to price a position it could not price, the snapshot writer
    fell back to `avg_cost`, the leaderboard valued it at **zero**, and the
    restatement engine raised. Same book, same missing row, three published
    answers — and two of them were numbers, which is the problem: a wrong
    number is indistinguishable from a right one downstream, while a refusal
    is not.

    All three now refuse. `portfolio_mtm` returns `None` for the whole book
    (its callers already drop a book they cannot value), and the snapshot
    writer raises so the session skips that book's row for the day rather
    than publishing a valuation it had to invent — snapshots are immutable,
    so an invented one is permanent.
    """
    quote = _latest_price(ticker, on)
    if quote is None:
        # `latest_price` returns None for two distinct conditions; separate
        # them so the diagnostic points at the right thing. A registry gap is
        # not a data gap and is fixed somewhere else entirely.
        if _ticker_currency(ticker) is None:
            return PositionValuation(None, "CURRENCY_UNRESOLVED")
        return PositionValuation(None, "NO_PRICE_DATA")

    native_value = shares * quote.price
    if quote.currency == book_currency:
        return PositionValuation(native_value)

    converted = _fx_convert(native_value, quote.currency, book_currency, on)
    if converted is None:
        return PositionValuation(None, "NO_FX_RATE")
    return PositionValuation(converted)


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
        `None` if **any** held position cannot be valued — no price, no
        resolvable quote currency, or no FX rate. Returning `None` for the
        whole book, rather than summing everything else, avoids understating
        it by a precise-looking but wrong number.

        Missing *price* used to be treated differently here: the position was
        skipped, which values it at zero. That was the weakest of the three
        answers the codebase gave to the same question (2026-08-07 review,
        W4.5) — a book quietly missing a position is exactly as wrong as one
        summing an unconverted foreign close, and harder to notice because
        the total still looks plausible. All three paths now refuse; see
        `value_position`.

        `portfolio_mtm_eur` and `mtm_base_currency` propagate this `None`
        rather than raising, matching the leaderboard call path's existing
        "skip this one book" contract
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
        valuation = value_position(ticker, shares, currency, on)
        if not valuation.ok:
            return None
        total += valuation.value
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
