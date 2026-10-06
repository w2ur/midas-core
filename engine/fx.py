"""Foreign exchange conversion helpers.

Reads daily FX rates from the committed OHLCV store at data/market/ohlcv/
(pairs fetched via the forex-majors universe). Primary use case: converting
non-EUR portfolio values to EUR for cross-agent comparison and real-money
reporting.

Rates are daily closes — intraday precision is neither available in the
store nor needed for portfolio-level reporting.
"""

from __future__ import annotations

import bisect
import json
import math
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

from engine.config import get_config

#: Every currency pair the store holds a rate for, as its vendor ticker.
#: ``ABCDEF=X`` stores how many ``DEF`` one ``ABC`` buys. **The one table**:
#: both directions of every pair are routed from it (``_ROUTES``), and
#: ``tests/test_fx.py`` fails when the store holds a ``*=X`` file this table
#: does not list. It used to be two hand-written maps, one keyed on EUR and
#: one on USD, and the USD one omitted ``GBPUSD=X`` and ``USDJPY=X``: the store
#: held both rates while GBP->USD and JPY->USD answered None (round-3 review,
#: 2026-10-06).
STORE_PAIRS: tuple[str, ...] = (
    "AUDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "EURUSD=X",
    "GBPJPY=X",
    "GBPUSD=X",
    "NZDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "USDJPY=X",
)


def _routes(pairs: Iterable[str]) -> dict[tuple[str, str], tuple[str, bool]]:
    """``(from, to) -> (ticker, inverted)`` for both directions of each pair.

    ``inverted`` is True when the stored rate is to-per-from the wrong way
    round, i.e. ``1 / close`` is the rate asked for.
    """
    routes: dict[tuple[str, str], tuple[str, bool]] = {}
    for ticker in pairs:
        base, quote = ticker[:3], ticker[3:6]
        routes[(base, quote)] = (ticker, False)
        routes[(quote, base)] = (ticker, True)
    return routes


_ROUTES = _routes(STORE_PAIRS)

#: The currency every pair not stored directly is composed through. Every
#: currency in ``STORE_PAIRS`` has a stored pair against it.
_PIVOT = "USD"


def _load_store_series(ticker: str) -> dict[str, float]:
    """Return {date_iso: raw close} for a ticker, or empty dict if missing.

    Raw `close`, never `adj_close` — same basis as every other read path
    (`engine.ohlcv_store` module docstring). For an FX pair the two fields
    are equal anyway (a currency pair pays no dividend); reading `close`
    keeps the rule uniform rather than resting on that.
    """
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    if not path.exists():
        return {}
    series: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = row.get("date")
            close = row.get("close")
            if d and close is not None:
                series[d] = float(close)
    return series


#: ``{file path: (sorted dates, values)}`` while a ``store_cache()`` block is
#: open, else None: every lookup then reads the file afresh, as it always did.
_SERIES_CACHE: dict[Path, tuple[list[str], list[float]]] | None = None


@contextmanager
def store_cache() -> Iterator[None]:
    """Read each pair's file at most once inside the block.

    **Opt-in and scoped** (round-4 review, 2026-10-06). The coin flip asks a
    rate for every currency and every date of a run, and each ask re-read and
    re-parsed the whole pair file, then scanned every date. A build or a
    replay opens this block around a run in which the store is not rewritten;
    nothing outside a block is cached, so the broker, which reads a rate a
    few times per session and may run after a store write in the same
    process, behaves exactly as before. Re-entrant: an inner block reuses the
    outer one's cache, and only the outermost exit drops it.
    """
    global _SERIES_CACHE
    if _SERIES_CACHE is not None:
        yield
        return
    _SERIES_CACHE = {}
    try:
        yield
    finally:
        _SERIES_CACHE = None


def _sorted(series: dict[str, float]) -> tuple[list[str], list[float]]:
    dates = sorted(series)
    return dates, [series[d] for d in dates]


def _latest_in(dates: list[str], values: list[float], target: date) -> float | None:
    """The value of the newest date ≤ ``target`` in sorted ``dates``, by
    bisection: the cached path, where one sort serves every ask of a run."""
    i = bisect.bisect_right(dates, target.isoformat())
    return values[i - 1] if i else None


def _latest_on_or_before(series: dict[str, float], target: date) -> float | None:
    """The latest value with date ≤ ``target``, or None: one O(n) pass, no
    sort. The uncached path (outside ``store_cache()``), where the series is
    read afresh for one ask and sorting it would cost more than the scan."""
    target_iso = target.isoformat()
    best: str | None = None
    for d in series:
        if d <= target_iso and (best is None or d > best):
            best = d
    return None if best is None else series[best]


def _store_value(ticker: str, on: date) -> float | None:
    """The stored close of ``ticker`` on or before ``on``: memoised and
    bisected inside a ``store_cache()`` block, a linear scan of a fresh read
    outside one."""
    if _SERIES_CACHE is None:
        return _latest_on_or_before(_load_store_series(ticker), on)
    # Keyed on the resolved file, not the ticker: a block that outlives a
    # ``MIDAS_DATA_DIR`` switch reads the new store's file, never the old one's.
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    if path not in _SERIES_CACHE:
        _SERIES_CACHE[path] = _sorted(_load_store_series(ticker))
    return _latest_in(*_SERIES_CACHE[path], on)


def get_rate(
    from_currency: str, to_currency: str, on: date | None = None
) -> float | None:
    """Return the exchange rate: how many `to_currency` per 1 `from_currency` on `on`.

    Returns None if the rate cannot be computed from the available data.
    Uses the most recent available close on or before `on` (defaults to today).

    A pair in ``STORE_PAIRS`` is read directly, in either direction; any
    other pair is composed through USD, which every stored currency has a
    pair against. A currency with no stored pair (SEK, DKK, NOK, PLN today)
    has no rate to or from anything.
    """
    if from_currency == to_currency:
        return 1.0
    if on is None:
        on = date.today()

    direct = _ROUTES.get((from_currency, to_currency))
    if direct is not None:
        return _stored_rate(*direct, on)

    # Not stored directly: compose through the pivot, e.g. CHF->EUR =
    # (USD per CHF) x (EUR per USD). Both legs must be stored pairs.
    if _PIVOT in (from_currency, to_currency):
        return None
    leg_in = _ROUTES.get((from_currency, _PIVOT))
    leg_out = _ROUTES.get((_PIVOT, to_currency))
    if leg_in is None or leg_out is None:
        return None
    to_pivot = _stored_rate(*leg_in, on)
    from_pivot = _stored_rate(*leg_out, on)
    if to_pivot is None or from_pivot is None:
        return None
    return to_pivot * from_pivot


def rate_tickers(from_currency: str, to_currency: str) -> tuple[str, ...]:
    """The stored pair(s) ``get_rate`` reads for this conversion: one ticker
    for a stored pair, the two legs for one composed through USD, and ``()``
    when no route exists or the currencies are equal. Reads no file; it names
    the store row a missing rate is waiting for."""
    if from_currency == to_currency:
        return ()
    direct = _ROUTES.get((from_currency, to_currency))
    if direct is not None:
        return (direct[0],)
    if _PIVOT in (from_currency, to_currency):
        return ()
    leg_in = _ROUTES.get((from_currency, _PIVOT))
    leg_out = _ROUTES.get((_PIVOT, to_currency))
    if leg_in is None or leg_out is None:
        return ()
    return (leg_in[0], leg_out[0])


def _stored_rate(ticker: str, inverted: bool, on: date) -> float | None:
    """One stored pair's rate on or before ``on``, or None.

    A close that is not a positive finite number is no rate: zero used to be
    the only one refused, while a negative or NaN close would have priced.
    """
    val = _store_value(ticker, on)
    if val is None or not math.isfinite(val) or val <= 0:
        return None
    return 1.0 / val if inverted else val


def convert(
    amount: float, from_currency: str, to_currency: str, on: date | None = None
) -> float | None:
    """Convert `amount` from one currency to another using the rate on `on`.

    Returns None if the rate is unavailable. Useful for portfolio valuation.
    """
    rate = get_rate(from_currency, to_currency, on)
    if rate is None:
        return None
    return amount * rate


def to_eur(amount: float, from_currency: str, on: date | None = None) -> float | None:
    """Convenience: convert `amount` from `from_currency` to EUR on `on`."""
    return convert(amount, from_currency, "EUR", on)
