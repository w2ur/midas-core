"""Which positions in a valuation were marked at a close their exchange has passed.

A snapshot row is dated on one market date, but each position in it is priced
at the newest close the store holds for that ticker on or before that date
(`engine.quotes.latest_price`). The two differ legitimately (a weekend, a
holiday: the exchange did not trade) and illegitimately (the exchange traded
and the store does not hold this ticker's close yet). Until 2026-10-03 nothing
recorded which: the 2026-09-30 goldfinger row valued 4GLD.DE and PPFB.DE at
their 09-29 closes while the rest of `.DE` held 09-30, and published a value
byte-identical to the day before with no trace of why.

A mark is **stale** when its ticker's bucket (`engine.market_calendar`) holds a
trading date (one at least half its live members hold: a tie counts) after the
mark's `price_date`, on or before the row's date: the exchange demonstrably traded and this close is not in the store. That
is the broker's stale-price rule with a lag of one rather than two, because a
disclosure has no tolerance to grant: a one-day-late fund still trades (the
rail allows it), and its mark is still a day old (this records it).

Three edges, stated rather than hidden:

- **A bucket too small to judge** (fewer than ``MIN_BUCKET_POPULATION`` live
  members: `.F`, `.NYB`) falls back to a weekday comparison against the close
  the session can hold. For a bucket a same-evening close run collects
  (`engine.market_calendar.close_run_bucket`) that is the last weekday on or
  before the row's date; for any other (`.F` and `.NYB` both: only the
  morning run's previous-day rule fetches them) it is the weekday before the
  row's date, the futures rule below, because a weekday row holds them at
  D-1 by design. Over-disclosing a holiday is the safe side of a disclosure;
  a weekend is not a holiday, and the weekend refresh writes a Saturday- and a
  Sunday-dated row for every book, so a Friday close in them is not stale.
- **Futures** (`=F`) share the no-suffix bucket with US listings, but their
  daily bar completes at 00:00 UTC and the same-evening close runs never fetch
  them: inside a row dated on a US trading day they mark at the previous
  completed bar by design (CLAUDE.md, Session Cadence). They are judged against
  that instead: stale when ``price_date`` is before the weekday preceding the
  row's date.
- **A whole bucket behind** (the 2026-09-22 class: SPY and most of the US
  served with no close) is invisible here, because the store cannot tell it
  from a holiday: the bucket holds no newer majority date, so no member reads
  behind it. That night is named per bucket by the bundle's ``exchange_dates``
  note (`scripts.fetch_market_data`), not by this list.

The list is written on every new snapshot row (`PortfolioManager.add_snapshot`),
empty when nothing was stale, so an empty list means "checked, none" and an
absent key means "a row from before the check existed". Old rows are never
touched: snapshots are immutable.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

from engine.market_calendar import bucket_lag, store_close_run_bucket


def _previous_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _weekday_on_or_before(d: date) -> date:
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def is_stale_mark(
    ticker: str, price_date: date, on: date, store: Path | None = None
) -> bool:
    """True when ``ticker``'s bucket traded after ``price_date``, by ``on``."""
    if price_date >= on:
        return False
    if ticker.endswith("=F"):
        return price_date < _previous_weekday(on)
    lag = bucket_lag(ticker, price_date, on, store=store).lag
    if lag is None:
        if store_close_run_bucket(ticker) is None:
            return price_date < _previous_weekday(on)
        return price_date < _weekday_on_or_before(on)
    return lag >= 1


def find_stale_marks(
    marks: Iterable[tuple[str, date]], on: date, store: Path | None = None
) -> list[dict]:
    """``[{ticker, price_date}]`` for every stale mark, sorted by ticker.

    ``marks`` is one ``(ticker, price_date)`` per priced position.
    """
    out = [
        {"ticker": ticker, "price_date": price_date.isoformat()}
        for ticker, price_date in marks
        if is_stale_mark(ticker, price_date, on, store=store)
    ]
    return sorted(out, key=lambda m: m["ticker"])


__all__ = ["find_stale_marks", "is_stale_mark"]
