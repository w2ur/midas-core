"""Which population a symbol trades with, and how far behind it a price is.

Two things live here, both read from the committed OHLCV store and nothing else
(no calendar library, no network: the session is sandboxed).

**Buckets.** ``bucket_of`` is the population a symbol is judged within: its
Yahoo exchange suffix (``".PA"``, the text after the LAST dot) when it has one,
otherwise its instrument class (``"crypto"``, ``"fx"``), otherwise ``""`` (US
listings, `=F` futures, `^` indices). It is the function `scripts.fetch_ohlcv`
rates vendor holes with (``hole_bucket`` there is this function), moved into
the engine on 2026-10-03 so the broker can share it (plan 2026-10-03, 1.3).

**Bucket lag.** A stored close carries the date of its row (``Quote.as_of``,
1.1). ``bucket_lag`` says how many of the bucket's own trading days that date
trails the bucket by, as of a trade date. A bucket trades on a date when AT
LEAST half its live members hold a row for it (plan 2026-10-03, 1.3); its
reference is the newest such date on or before the trade date. This is looser
than `engine.store_gaps`' strict majority on purpose: there a tie must not
flag a member's gap, here a tie must not understate a member's lag, so each
leans the way its own caller can afford (the rail toward refusing). The lag is the number of bucket trading days in
``(as_of, reference]``. So:

- a bucket closed wholesale (a holiday) holds no new majority date, its
  reference does not move, and every member reads lag 0: it stays green;
- a chronic day-late fund (4GLD.DE) reads lag 1 on the night `.DE` lands D and
  it does not;
- a frozen series (CTVA at its 09-30 close on 10-02, MNST at its 08-10 close
  through its unrestated split) reads lag 2 and more.

A **live** member is a store file with at least one row inside the lookback
window: a symbol that has left every universe and stopped advancing, or one
the vendor stopped serving (MATIC-USD), must not dilute the majority. Below
``MIN_BUCKET_POPULATION`` live members the bucket says nothing about its
exchange (`.F` and `.NYB` hold one file each), and ``bucket_lag`` returns
``lag=None``: the caller abstains, and says so, rather than reading a
one-member bucket as agreeing with itself.
"""

from __future__ import annotations

import json
import os
from array import array
from bisect import bisect_right
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from engine.config import get_config
from engine.fees import classify_ticker

#: Bucket trading days a price may trail its bucket and still fill. One, so
#: the chronic day-late UCITS funds (4GLD.DE, PPFB.DE: about a day late on most
#: nights, measured 2026-10-03) keep trading, and a two-session freeze does not.
MAX_BUCKET_LAG_DAYS = 1

#: Calendar days of history a bucket's trading days are read over. Wide enough
#: to hold more than two weeks of sessions across any holiday cluster, so a
#: price far older than the window still reads as many days behind.
LOOKBACK_DAYS = 31

#: Fewest live members a bucket needs before its majority means anything. The
#: same floor `scripts.fetch_market_data.exchange_dates` reports buckets at.
MIN_BUCKET_POPULATION = 5


def bucket_of(symbol: str, crypto: frozenset[str] = frozenset()) -> str:
    """The population ``symbol``'s missing closes are rated within.

    A Yahoo exchange suffix (``".PA"``: what follows the LAST dot, so
    ``BT.A.L`` is ``.L``) when there is one. Otherwise the instrument class, by
    the repo's own classifier (`engine.fees.classify_ticker`): ``"crypto"``,
    also for any pair in ``crypto`` (the set `scripts.fetch_ohlcv` fetches in
    its crypto-only mode, which carries pairs such as HBAR-USD that the fee
    allowlist does not), ``"fx"``, and ``""`` for the rest — US listings and
    the handful of `=F` futures. Money review r1 (J6 follow-ups), M1: crypto
    and FX folded into the US bucket, where a hole across all 34 crypto pairs
    read 5.3% and passed.
    """
    head, dot, tail = symbol.rpartition(".")
    if dot and head and tail:
        return f".{tail}"
    if symbol in crypto:
        return "crypto"
    asset_class = classify_ticker(symbol)
    return "" if asset_class == "equity" else asset_class


def _looks_like_crypto_pair(symbol: str) -> bool:
    """A dotless ``BASE-USD`` / ``BASE-EUR`` store file.

    The broker has no crypto universe to pass ``bucket_of`` and must not
    resolve one at fill time, so it reads the pair shape instead. Measured on
    the 2026-10-03 store: it catches the four pairs the fee allowlist misses
    (BNB-USD, SHIB-USD, ICP-USD, HBAR-USD) and no share class (BRK-B, BF-B).
    """
    return "." not in symbol and symbol.endswith(("-USD", "-EUR"))


def store_bucket(symbol: str) -> str:
    """``bucket_of`` for a store symbol, with crypto read from its shape."""
    crypto = frozenset({symbol}) if _looks_like_crypto_pair(symbol) else frozenset()
    return bucket_of(symbol, crypto)


#: The exchange suffixes the `--close-run eu` evening pass collects. Every
#: venue here has closed by 16:30 UTC on a winter day (Euronext, Xetra, SIX,
#: the LSE, the Nordics, Madrid, Milan, Vienna, Warsaw, Athens, Dublin,
#: Lisbon). NOT `.F`: the Frankfurt floor trades until 20:00 local, so its bar
#: is still forming when this pass runs. A suffix absent here stays on the
#: morning run, whose previous-day rule is right for any close hour.
EU_CLOSE_SUFFIXES = frozenset(
    {
        ".AS", ".AT", ".BR", ".CO", ".DE", ".HE", ".IR", ".L", ".LS", ".MC",
        ".MI", ".OL", ".PA", ".ST", ".SW", ".VI", ".WA",
    }
)


def close_run_bucket(symbol: str, crypto: frozenset[str] = frozenset()) -> str | None:
    """Which same-evening pass collects ``symbol``: ``"eu"``, ``"us"`` or None.

    Why there are evening passes at all (measured by `eu-close-probe.yml`,
    2026-08-14..18): the vendor publishes a cash-equity day's close the same
    evening — populated from about 1.5 h after the bell, still there at 20:18
    UTC for Europe — then WITHDRAWS it overnight (a null row by 22:23, still
    null at 07:22, the US included) and restores it the next afternoon. The
    06:00 morning run sits inside that withdrawal, and its real start (4-7 h
    late, GitHub's scheduler) lands on the restoration edge, which is where
    the store's random one-day holes came from. Collecting in the evening
    asks for the bar while it exists.

    ``"us"`` is a US cash listing: no exchange suffix, and not a 24/7 or
    settlement-priced instrument (crypto, `=X` FX, `=F` futures — their daily
    bar completes at 00:00 UTC and stays on the morning run's previous-day
    rule). Indices (`^VIX`) count as US: they print with the cash close.

    Moved here from `scripts.fetch_ohlcv` on 2026-10-04: `engine.stale_marks`
    reads it to know whether a session holds a symbol at its row's own date
    (a close-run bucket) or at the previous one (everything else).
    """
    bucket = bucket_of(symbol, crypto)
    if bucket in EU_CLOSE_SUFFIXES:
        return "eu"
    if bucket == "" and not symbol.endswith("=F"):
        return "us"
    return None


def store_close_run_bucket(symbol: str) -> str | None:
    """``close_run_bucket`` for a store symbol, with crypto read from its shape."""
    crypto = frozenset({symbol}) if _looks_like_crypto_pair(symbol) else frozenset()
    return close_run_bucket(symbol, crypto)


# ---------------------------------------------------------------------------
# Per-file row dates, cached on (mtime, size)
# ---------------------------------------------------------------------------

_DATE_PREFIX = '{"date": "'

# path -> ((mtime_ns, size), sorted date ordinals). Ordinals in an int array
# rather than strings: the whole store is ~3M rows.
_DATES: dict[str, tuple[tuple[int, int], array]] = {}


def _row_dates(path: Path, stat: os.stat_result) -> array:
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _DATES.get(str(path))
    if cached is not None and cached[0] == key:
        return cached[1]
    ordinals: list[int] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            # Every row the ingest writes opens with its date; anything else
            # (a hand-edited line, a future field order) is parsed properly.
            if line.startswith(_DATE_PREFIX):
                raw = line[len(_DATE_PREFIX) : len(_DATE_PREFIX) + 10]
            else:
                try:
                    raw = json.loads(line).get("date")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if not isinstance(raw, str):
                    continue
            try:
                ordinals.append(date.fromisoformat(raw).toordinal())
            except ValueError:
                continue
    out = array("i", sorted(set(ordinals)))
    _DATES[str(path)] = (key, out)
    return out


class BucketLag(NamedTuple):
    """How far ``as_of`` trails ``bucket`` on ``on``.

    ``lag`` is ``None`` when the bucket is too small to judge (fewer than
    ``MIN_BUCKET_POPULATION`` live members) or holds no majority date in the
    window; ``reference`` is the bucket's newest majority date, if any.
    """

    bucket: str
    population: int
    reference: date | None
    lag: int | None


def bucket_lag(
    symbol: str, as_of: date, on: date, store: Path | None = None
) -> BucketLag:
    """Bucket trading days between ``as_of`` and ``symbol``'s bucket, on ``on``.

    Only rows dated on or before ``on`` count, so a replay of a past trade date
    reads the bucket as it stood then (as far as the store still holds it).
    """
    store = store if store is not None else get_config().ohlcv_dir
    bucket = store_bucket(symbol)
    lo = (on - timedelta(days=LOOKBACK_DAYS)).toordinal()
    hi = on.toordinal()
    held: Counter[int] = Counter()
    population = 0
    try:
        entries = list(os.scandir(store))
    except OSError:
        return BucketLag(bucket, 0, None, None)
    for entry in entries:
        name = entry.name
        if not name.endswith(".jsonl"):
            continue
        if store_bucket(name[: -len(".jsonl")]) != bucket:
            continue
        try:
            dates = _row_dates(Path(entry.path), entry.stat())
        except OSError:
            continue
        window = dates[bisect_right(dates, lo) : bisect_right(dates, hi)]
        if not window:
            continue
        population += 1
        held.update(window)
    if population < MIN_BUCKET_POPULATION:
        return BucketLag(bucket, population, None, None)
    # At least half (a tie counts): see the module docstring.
    trading = sorted(d for d, n in held.items() if 2 * n >= population)
    if not trading:
        return BucketLag(bucket, population, None, None)
    reference = trading[-1]
    # Bucket trading days in (as_of, reference]. A price at or past the
    # reference (a live crypto quote, a fund that landed before its exchange's
    # majority) counts none, so it is not behind.
    lag = len(trading) - bisect_right(trading, as_of.toordinal())
    return BucketLag(bucket, population, date.fromordinal(reference), lag)


def is_stale(
    symbol: str, as_of: date, on: date, store: Path | None = None
) -> bool:
    """True when ``as_of`` trails its bucket by more than ``MAX_BUCKET_LAG_DAYS``.

    False when the bucket cannot be judged (see ``bucket_lag``): an abstention,
    not a pass, and the broker's suspension rail is what covers a frozen symbol
    in a bucket too small to see it.
    """
    result = bucket_lag(symbol, as_of, on, store=store)
    return result.lag is not None and result.lag > MAX_BUCKET_LAG_DAYS


__all__ = [
    "EU_CLOSE_SUFFIXES",
    "LOOKBACK_DAYS",
    "MAX_BUCKET_LAG_DAYS",
    "MIN_BUCKET_POPULATION",
    "BucketLag",
    "bucket_lag",
    "bucket_of",
    "close_run_bucket",
    "is_stale",
    "store_bucket",
    "store_close_run_bucket",
]
