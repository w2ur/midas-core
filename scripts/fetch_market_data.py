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


def fetch_and_save(
    output_path: Path | None = None, allow_network: bool = False
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

    Returns
    -------
    dict
        Saved payload: {"date", "benchmarks": {...}, "notes": {...}}
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
    latest_date: str | None = None

    for name in _BENCHMARK_SOURCES:
        value, label, src_date = _resolve_benchmark(name)
        benchmarks[name] = round(value, 4 if name == "msci_world" else 2)
        notes[f"{name}_source"] = f"{label} (OHLCV store, {src_date})"
        if latest_date is None or src_date > latest_date:
            latest_date = src_date

    payload = {
        "date": latest_date or date.today().isoformat(),
        "benchmarks": benchmarks,
        "notes": notes,
    }

    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Market data saved to {output_path}")
    print(f"  Date:       {payload['date']}")
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
