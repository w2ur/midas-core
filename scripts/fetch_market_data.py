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

# Calendar days, not trading days. The longest ordinary equity gap is a
# Thu-close → Tue-session Easter/Christmas weekend: the session on Tuesday
# reads Thursday's close because the OHLCV cron runs *after* the session, so
# 4 days of legitimate lag is reachable without anything being wrong. 5 is not:
# no market this store follows closes for five consecutive calendar days.
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
        `equity_date` plus the `mixed_dates` note make it legible.

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
