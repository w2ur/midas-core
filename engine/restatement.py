"""Restatement engine — pure trade replay and snapshot re-valuation primitives.

Built after commit 4b6b8556 corrected 29,348 rows in the committed OHLCV
store (partial bars + 11 stock splits whose pre-split history was
un-adjusted). Every published portfolio valuation was priced off wrong
closes — the fills themselves were correct, only the valuations were wrong.
This module provides the pure building blocks a later runner uses to
re-derive corrected valuations; it performs no I/O of its own.

Two primitives:

- ``replay_holdings`` derives {ticker: shares} as of any market date purely
  from the trade ledger (``trades.json``), by replaying BUY/SELL in
  chronological order from a zero baseline. It also returns the net cash
  effect of that replay (``cash_delta``) — but this is for the caller to
  **cross-check against the recorded cash figure and refuse on divergence**,
  never to overwrite it. Recorded cash is authoritative: fills executed at
  the prices they executed at, and that ledger stands. A book whose
  replayed holdings/cash diverge from its recorded state — e.g.
  sharp-shooter-eur, whose 2026-05-21 lost-fill ledger artifact
  (METHODOLOGY anchor ``lost-fill-2026-05-21``) is a confirmed broker fill
  that never landed in ``portfolio.json`` — must be surfaced by the caller,
  not silently absorbed here.

- ``revalue_snapshot`` re-prices a {ticker: shares} book from the committed
  OHLCV store as of a market date, converting any ticker not already in the
  book's base currency. It reuses the exact helpers the live desk already
  uses for this: ``engine.quotes.latest_price`` (the same reader
  ``engine.paper_broker`` uses for fills and ``engine.valuation.portfolio_mtm``
  uses for point valuation — the row's raw ``close``, never ``adj_close``,
  normalised out of any vendor sub-unit such as LSE pence and paired with
  the ticker's ISO currency) and ``engine.fx.convert`` (the same FX helper the broker
  calls to convert a fill's notional). Reusing these, unmodified, is
  deliberate: if this module disagreed with the live desk on how a position
  is priced or which currency it trades in, the restatement would be wrong by
  definition — that is the whole point of the exercise.
"""

from __future__ import annotations

from datetime import date, datetime

from engine.fx import convert as fx_convert
from engine.valuation import value_position
from engine.quotes import latest_price

# A held position never reaches exactly zero via floating-point trade
# replay; treat anything under this magnitude as closed.
_EPSILON = 1e-9


class MissingPriceError(Exception):
    """Raised when a price or FX rate needed to revalue a position is unavailable.

    Carries ``symbol`` (a ticker, or a ``"FROM->TO"`` currency-pair
    description for an FX-rate miss) and ``market_date`` so callers can log
    or surface the gap precisely. Raised instead of silently pricing a
    position at zero, which would understate the book and read as a loss
    that never happened.
    """

    #: Reason codes from `engine.valuation.value_position` → readable phrase.
    #: The code is kept in the message alongside the phrase: the phrase is for
    #: the human reading a session log, the code is what makes a valuation
    #: failure greppable against the broker rejection describing the same
    #: condition.
    _PHRASES = {
        "NO_PRICE_DATA": "price",
        "NO_FX_RATE": "FX rate",
        "CURRENCY_UNRESOLVED": "resolvable quote currency",
    }

    def __init__(self, symbol: str, market_date: date, *, what: str = "price") -> None:
        self.symbol = symbol
        self.market_date = market_date
        self.reason = what if what in self._PHRASES else None
        if self.reason:
            what = f"{self._PHRASES[self.reason]} ({self.reason})"
        super().__init__(
            f"No {what} for {symbol!r} on or before {market_date.isoformat()} in the "
            "committed OHLCV store — refusing to price at zero"
        )


def replay_holdings(trades: list[dict], as_of: date) -> tuple[dict[str, float], float]:
    """Derive {ticker: shares} and net cash effect from a trade ledger.

    Applies every trade timestamped on or before ``as_of`` to a zero
    baseline, in chronological order (the input list is not assumed to
    already be sorted). BUY adds shares and deducts ``total + fees`` from
    the running cash delta; SELL subtracts shares and credits
    ``total - fees``. A ticker whose share count falls within ``_EPSILON``
    of zero is dropped from the result rather than left as a near-zero dust
    entry.

    ``as_of`` is inclusive: a trade timestamped exactly on ``as_of`` is
    included, since sessions run at a fixed time of day and the snapshot
    they produce reflects that day's fills.

    Parameters
    ----------
    trades:
        Trade records as read from ``trades.json`` — each a dict with
        ``timestamp`` (ISO 8601), ``action`` ("BUY"/"SELL"), ``ticker``,
        ``shares``, ``total``, and ``fees``.
    as_of:
        Replay cutoff (inclusive).

    Returns
    -------
    tuple[dict[str, float], float]
        ``(holdings, cash_delta)``. ``cash_delta`` is the net cash effect of
        the replayed trades relative to an unspecified starting cash
        balance — it is for cross-checking against the recorded cash
        figure, never for overwriting it. Recorded cash is authoritative.

    Raises
    ------
    ValueError
        If a trade's ``action`` is neither "BUY" nor "SELL".
    """
    ordered = sorted(trades, key=lambda t: datetime.fromisoformat(t["timestamp"]))

    holdings: dict[str, float] = {}
    cash_delta = 0.0

    for trade in ordered:
        trade_date = datetime.fromisoformat(trade["timestamp"]).date()
        if trade_date > as_of:
            continue

        action = trade["action"].upper()
        ticker = trade["ticker"]
        shares = trade["shares"]
        total = trade["total"]
        fees = trade.get("fees", 0.0)

        if action == "BUY":
            holdings[ticker] = holdings.get(ticker, 0.0) + shares
            cash_delta -= total + fees
        elif action == "SELL":
            holdings[ticker] = holdings.get(ticker, 0.0) - shares
            cash_delta += total - fees
        else:
            raise ValueError(
                f"Invalid trade action: {trade['action']!r}. Expected 'BUY' or 'SELL'."
            )

        if abs(holdings.get(ticker, 0.0)) < _EPSILON:
            holdings.pop(ticker, None)

    return holdings, cash_delta


def revalue_snapshot(
    positions: dict[str, float],
    cash: float,
    market_date: date,
    currency: str,
) -> tuple[float, float]:
    """Re-price a {ticker: shares} book from the OHLCV store as of a market date.

    Recorded ``cash`` is passed straight through and never recomputed — only
    the valuation of held positions changes. Each position is priced via
    ``engine.quotes.latest_price`` (the same reader the live desk uses for
    fills and point valuation, which returns the close already denominated
    in the ticker's ISO currency) and, when that currency differs from
    ``currency``, converted via ``engine.fx.convert`` — the same FX path the
    broker uses to convert a fill's notional. This is what makes
    the primitive correct for a book whose positions are not all in its own
    base currency (e.g. an EUR book holding a USD-listed ticker).

    A position whose share count is within ``_EPSILON`` of zero is skipped
    entirely — it needs no price.

    Parameters
    ----------
    positions:
        {ticker: shares}, e.g. as returned by ``replay_holdings``.
    cash:
        Recorded cash, in ``currency``. Passed through unchanged.
    market_date:
        Valuation date. Prices use the latest close on or before this date.
    currency:
        The book's base currency (the currency ``cash`` and the returned
        values are denominated in).

    Returns
    -------
    tuple[float, float]
        ``(portfolio_value, positions_value)``, both in ``currency``.
        ``portfolio_value == cash + positions_value``.

    Raises
    ------
    MissingPriceError
        If a position's ticker has no price in the committed OHLCV store on
        or before ``market_date``, or if converting its native currency to
        ``currency`` requires an FX rate that isn't available.
    """
    positions_value = 0.0

    for ticker, shares in positions.items():
        if abs(shares) < _EPSILON:
            continue

        # engine.valuation.value_position is the single pricing decision all
        # three valuation paths now share (snapshots, leaderboard, this one).
        # It answers in the broker's own vocabulary, so a valuation failure
        # and a fill rejection describe the same condition with the same word.
        valuation = value_position(ticker, shares, currency, market_date)
        if not valuation.ok:
            raise MissingPriceError(ticker, market_date, what=valuation.reason)
        positions_value += valuation.value

    return cash + positions_value, positions_value
