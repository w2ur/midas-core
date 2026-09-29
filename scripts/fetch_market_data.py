"""Build today's market benchmark snapshot from the committed OHLCV store.

Writes `data/market/today.json` — a single 4-benchmark snapshot consumed by
agent prompts for daily commentary. NOT the source of truth for prices
(that's the OHLCV store, populated by the fetch-ohlcv GitHub Action).

Default behavior reads from the OHLCV store only — no network. The trading
session sandbox is HTTP-blocked, and the cron has already written every
ticker we need. yfinance is offered as an opt-in fallback for local dev.

Each benchmark resolves through a list of fallbacks (primary ticker first,
proxy tickers after). The first hit wins, and the resulting `notes` field
records exactly which source produced each value.

Freshness is asserted here, at the front of the session (2026-08-07 review,
W3.1). Everything downstream — snapshots, the leaderboard, the drawdown rail —
prices off this same store, and a store that stopped advancing produces a
plausible-looking valuation at a stale close. Snapshots are immutable, so that
valuation is permanent. See `EQUITY_BENCHMARKS` / `MAX_EQUITY_STALENESS_DAYS`.

Usage:
    python scripts/fetch_market_data.py
    python scripts/fetch_market_data.py --allow-network   # local dev only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

# Add project root to sys.path so engine imports work when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.market_data import latest_close_and_date_from_store


# Each benchmark maps to an ordered list of (ticker, multiplier, label) sources.
# The first source whose ticker is in the OHLCV store wins. Multipliers convert
# proxy tickers to the benchmark scale (e.g. SPY × 10 ≈ S&P 500 index level).
#
# Order matters — primary ticker first, proxies after.
_BENCHMARK_SOURCES: dict[str, list[tuple[str, float, str]]] = {
    "sp500": [
        ("^GSPC", 1.0, "^GSPC"),
        ("SPY", 10.0, "SPY*10 proxy"),
    ],
    "msci_world": [
        ("URTH", 1.0, "URTH"),
    ],
    "gold": [
        ("GC=F", 1.0, "GC=F"),
        ("GLD", 10.0, "GLD*10 proxy"),
    ],
    "btc": [
        ("BTC-USD", 1.0, "BTC-USD"),
    ],
}

# The benchmarks that only advance on a cash-equity session. They are the
# staleness probe: crypto trades every day, so BTC alone cannot tell a healthy
# store from one whose equity feed died three weeks ago.
EQUITY_BENCHMARKS = ("sp500", "msci_world")

#: Fewest store files an exchange bucket needs before its newest date is
#: reported by `exchange_dates`: below this a bucket is a handful of names whose
#: staleness says nothing about the exchange (`.F` and `.NYB` hold one each).
MIN_EXCHANGE_POPULATION = 5


def _newest_stored_date(path: Path) -> str | None:
    """The `date` of a store file's last non-empty line, read from the tail."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        if line.strip():
            try:
                d = json.loads(line).get("date")
            except (json.JSONDecodeError, AttributeError):
                return None
            return d if isinstance(d, str) else None
    return None


def _exchange_bucket(symbol: str) -> str | None:
    """The cash-equity exchange a store symbol trades on: its Yahoo suffix, or
    ``"US"`` for a suffix-less listing. None for anything whose daily bar is
    not a cash close — crypto pairs, `=X` FX, `=F` futures and `^` indices."""
    if "=" in symbol or symbol.startswith("^"):
        return None
    head, dot, tail = symbol.rpartition(".")
    if dot and head and tail:
        return f".{tail}"
    if symbol.endswith(("-USD", "-EUR")):
        return None
    return "US"


def exchange_dates(store: Path | None = None) -> dict[str, str]:
    """Per cash-equity exchange, the newest close AT LEAST HALF its store files hold.

    Reads the tail of every `*.jsonl` in the store (cheap: one seek per file).
    Half rather than the maximum, because a single first-ingest file or one
    day-late fund must not speak for an exchange; half rather than all, because
    the chronic day-late UCITS funds would otherwise hold `.DE` a day behind
    forever. Buckets under `MIN_EXCHANGE_POPULATION` are left out.

    This is what tells the bundle — and a reader of it — which exchanges the
    row's positions were actually marked at. Since the same-evening close runs
    (2026-09-28) the European and US closes arrive in two separate passes, so
    "the equity date" is no longer one date by construction: a European bucket
    the vendor published late is a day behind the row, and a European bucket
    the session read before the US pass landed is a day AHEAD of it.
    """
    store = store if store is not None else get_config().ohlcv_dir
    if not store.exists():
        return {}
    newest: dict[str, list[str]] = {}
    for path in sorted(store.glob("*.jsonl")):
        bucket = _exchange_bucket(path.stem)
        if bucket is None:
            continue
        d = _newest_stored_date(path)
        if d is not None:
            newest.setdefault(bucket, []).append(d)
    out: dict[str, str] = {}
    for bucket, dates in newest.items():
        if len(dates) < MIN_EXCHANGE_POPULATION:
            continue
        ranked = sorted(dates, reverse=True)
        out[bucket] = ranked[len(ranked) // 2]
    return out


# Calendar days, not trading days. The bound is set by which market DATES can
# legitimately be missing between the newest stored close and the session
# reading it — NOT by the clock order of the collector and the session, which
# an earlier wording ("the OHLCV cron runs after the session") was read both
# ways and settled neither.
#
# The longest ordinary gap is a Good-Friday or Christmas weekend: the cash
# market's last close is Thursday's, Friday and the weekend produce no bar at
# all, and the session that follows runs before its own day's close exists —
# so the newest close it can possibly see is 4 calendar days old, with nothing
# wrong anywhere. 5 is not reachable: no market this store follows closes for
# five consecutive calendar days.
#
# That reasoning is independent of the collection hour. The fetch moved from
# 22:30 UTC to 06:00 UTC on 2026-08-12 and the reachable lag is identical
# either side of the move, so this threshold did not change with it.
MAX_EQUITY_STALENESS_DAYS = 4


class StaleMarketDataError(RuntimeError):
    """Raised when the equity side of the OHLCV store has stopped advancing.

    Fatal by design, and deliberately raised *before* the session authors
    anything. The failure it guards against is silent: `fetch-ohlcv` exits 0
    on a total vendor outage, the store keeps its last-good rows, and every
    downstream consumer prices happily against them. Nothing in a snapshot,
    a leaderboard row or a fill says "this number is three weeks old".
    """


def _resolve_benchmark(name: str) -> tuple[float, str, str]:
    """Return (value, source_label, source_date) for a benchmark, store-only.

    Raises RuntimeError if no source is available — should never happen in
    production once the OHLCV cron has run at least once.
    """
    for ticker, multiplier, label in _BENCHMARK_SOURCES[name]:
        result = latest_close_and_date_from_store(ticker)
        if result is None:
            continue
        close, src_date = result
        return (close * multiplier, label, src_date)
    raise RuntimeError(
        f"No OHLCV source available for benchmark '{name}'. "
        f"Tried: {[t for t, _, _ in _BENCHMARK_SOURCES[name]]}"
    )


def _assert_equity_freshness(
    source_dates: dict[str, str], today: date, max_days: int
) -> str:
    """Abort unless an equity benchmark closed within *max_days* of *today*.

    Returns the newest equity source date (ISO). Raises `StaleMarketDataError`
    otherwise — the session must not price a book against a store that stopped
    advancing.

    Only the equity benchmarks are probed. Gold and BTC keep advancing through
    a weekend and through an equity-vendor outage, so a max over all four
    always looks fresh; that is precisely how a dead equity feed stays
    invisible.
    """
    equity_dates = [source_dates[b] for b in EQUITY_BENCHMARKS if b in source_dates]
    if not equity_dates:  # pragma: no cover — _resolve_benchmark raises first
        raise StaleMarketDataError(
            "No equity benchmark resolved, so store freshness cannot be "
            "established. Refusing to publish a valuation."
        )
    newest = max(equity_dates)
    age = (today - date.fromisoformat(newest)).days
    if age > max_days:
        raise StaleMarketDataError(
            f"The equity side of the OHLCV store is {age} calendar days stale: "
            f"newest close among {list(EQUITY_BENCHMARKS)} is {newest}, today is "
            f"{today.isoformat()} (limit {max_days} days). The fetch-ohlcv cron "
            "has most likely been failing. Abort the session — do not publish a "
            "snapshot at these prices; snapshots are immutable."
        )
    return newest


def fetch_and_save(
    output_path: Path | None = None,
    allow_network: bool = False,
    *,
    today: date | None = None,
    max_equity_staleness_days: int = MAX_EQUITY_STALENESS_DAYS,
) -> dict:
    """Build today's snapshot and persist to disk.

    Parameters
    ----------
    output_path:
        Destination file. Defaults to data/market/today.json.
    allow_network:
        If True, prefer `MarketDataFetcher.fetch_benchmarks` over the store.
        Off by default — the OHLCV store is authoritative and the trading
        sandbox can't make outbound HTTP anyway.
    today:
        Reference date for the freshness gate. Defaults to the system date;
        injectable so a fixture can be dated without freezing the clock.
    max_equity_staleness_days:
        Gate threshold — see `MAX_EQUITY_STALENESS_DAYS`.

    Returns
    -------
    dict
        Saved payload: {"date", "equity_date", "benchmarks", "notes"}.

        `date` is the newest close across *all* benchmarks; `equity_date` is
        the newest close among the equity ones. They differ on any weekend or
        market holiday, when crypto has advanced and equities have not — and
        the snapshot written at `date` then values equity positions at
        `equity_date`'s close. That is a correct mark (`latest_price` reads
        the last close on-or-before the date), but it used to go unrecorded;
        `equity_date` plus the `mixed_dates` note make it legible. Since the
        same-evening close runs (2026-09-28) the note also covers the other
        direction — equities at today's close, crypto/gold at yesterday's
        completed bar — and `notes["exchange_dates"]` records, per exchange,
        the close at least half its store files hold, with `exchange_behind`
        / `exchange_ahead` naming any exchange marked at a different close
        than the row's.

    Raises
    ------
    StaleMarketDataError
        If no equity benchmark has closed within `max_equity_staleness_days`.
    """
    if output_path is None:
        output_path = get_config().data_dir / "data" / "market" / "today.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if allow_network:
        try:
            return _fetch_with_network(output_path)
        except Exception as exc:  # noqa: BLE001 — explicit fallback path
            print(f"[WARN] Network fetch failed ({exc}); falling back to OHLCV store.")

    benchmarks: dict[str, float] = {}
    notes: dict[str, str] = {"note": "built from committed OHLCV store"}
    source_dates: dict[str, str] = {}
    latest_date: str | None = None

    for name in _BENCHMARK_SOURCES:
        value, label, src_date = _resolve_benchmark(name)
        benchmarks[name] = round(value, 4 if name == "msci_world" else 2)
        notes[f"{name}_source"] = f"{label} (OHLCV store, {src_date})"
        source_dates[name] = src_date
        if latest_date is None or src_date > latest_date:
            latest_date = src_date

    reference_day = today if today is not None else date.today()
    equity_date = _assert_equity_freshness(
        source_dates, reference_day, max_equity_staleness_days
    )

    snapshot_date = latest_date or reference_day.isoformat()
    if equity_date < snapshot_date:
        notes["mixed_dates"] = (
            f"snapshot dated {snapshot_date} (crypto/gold); equity positions "
            f"are marked at the {equity_date} close — no equity session since."
        )
    else:
        # The other direction, since the same-evening close runs (2026-09-28):
        # the equity benchmarks carry today's close while crypto, FX and
        # futures — whose bar completes at 00:00 UTC and lands with the morning
        # run — are still at the previous completed bar. A correct mark,
        # recorded for the same reason the first direction is.
        behind = {
            name: src_date
            for name, src_date in source_dates.items()
            if name not in EQUITY_BENCHMARKS and src_date < snapshot_date
        }
        if behind:
            listed = ", ".join(f"{name} at {d}" for name, d in sorted(behind.items()))
            notes["mixed_dates"] = (
                f"snapshot dated {snapshot_date} (equity close); {listed} — "
                "crypto, FX and futures positions are marked at the previous "
                "completed UTC bar, which the morning collector lands."
            )

    # Which exchanges the row's cash positions were actually marked at. The
    # benchmarks above are all US-listed, so `equity_date` alone cannot see a
    # European bucket the vendor published late (a day BEHIND the row) or one
    # the session read before the US close pass landed (a day AHEAD of it,
    # which is a mislabelled row and is said out loud).
    exchanges = exchange_dates()
    if exchanges:
        notes["exchange_dates"] = exchanges
        behind_ex = {ex: d for ex, d in exchanges.items() if d < equity_date}
        ahead_ex = {ex: d for ex, d in exchanges.items() if d > snapshot_date}
        if behind_ex:
            notes["exchange_behind"] = (
                "marked at an older close than the row's equity date: "
                + ", ".join(f"{ex} at {d}" for ex, d in sorted(behind_ex.items()))
            )
        if ahead_ex:
            notes["exchange_ahead"] = (
                f"marked at a NEWER close than the row date {snapshot_date}: "
                + ", ".join(f"{ex} at {d}" for ex, d in sorted(ahead_ex.items()))
                + " — the row is dated on the benchmarks, which had not advanced"
            )

    payload = {
        "date": snapshot_date,
        "equity_date": equity_date,
        "benchmarks": benchmarks,
        "notes": notes,
    }

    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Market data saved to {output_path}")
    print(f"  Date:       {payload['date']}")
    print(
        f"  Equity:     {equity_date} "
        f"({(reference_day - date.fromisoformat(equity_date)).days}d old, "
        f"limit {max_equity_staleness_days}d)"
    )
    if "mixed_dates" in notes:
        print(f"  [note] {notes['mixed_dates']}")
    if "exchange_behind" in notes:
        print(f"  [WARN] {notes['exchange_behind']}")
    if "exchange_ahead" in notes:
        print(f"  [WARN] {notes['exchange_ahead']}")
    for name in ("sp500", "msci_world", "gold", "btc"):
        print(f"  {name:11s}: {benchmarks[name]}  ({notes[f'{name}_source']})")
    return payload


def _fetch_with_network(output_path: Path) -> dict:
    """Legacy network path. Kept for local dev — production runs the
    store-only path."""
    from datetime import timedelta

    from engine.market_data import MarketDataFetcher

    today = date.today()
    start = today - timedelta(days=7)
    cache_dir = _PROJECT_ROOT / "data" / "cache"
    fetcher = MarketDataFetcher(cache_dir=cache_dir)
    df = fetcher.fetch_benchmarks(start=start, end=today)
    if df.empty:
        raise RuntimeError("No benchmark data returned via network.")

    latest_row = df.iloc[-1]
    latest_date = df.index[-1].date()
    payload = {
        "date": latest_date.isoformat(),
        # The network frame is a single aligned index, so there is no
        # equity-vs-crypto date split to record here.
        "equity_date": latest_date.isoformat(),
        "benchmarks": {
            "sp500": round(float(latest_row["sp500"]), 2),
            "msci_world": round(float(latest_row["msci_world"]), 4),
            "gold": round(float(latest_row["gold"]), 2),
            "btc": round(float(latest_row["btc"]), 2),
        },
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Market data saved to {output_path} (via network)")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Prefer yfinance over the OHLCV store. Local dev only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fetch_and_save(allow_network=args.allow_network)
