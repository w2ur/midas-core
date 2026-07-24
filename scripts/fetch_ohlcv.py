"""Fetch historical OHLCV for every symbol any strategy or portfolio might reference.

Runs in a trusted environment (local dev or GitHub Actions) where yfinance
works reliably. Output lives at data/market/ohlcv/{SYMBOL}.jsonl — one row per
trading day, append-only, committed to git so sandboxed agents can read it.

Not to be confused with scripts/fetch_market_data.py, which writes a single
benchmark snapshot for the daily session dashboard.

Usage:
    python scripts/fetch_ohlcv.py
    python scripts/fetch_ohlcv.py --history-days 60     # short refresh
    python scripts/fetch_ohlcv.py --symbols AAPL,MSFT   # targeted
    python scripts/fetch_ohlcv.py --dry-run             # list resolved symbols
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import yfinance as yf

from engine.config import get_config
from engine.ohlcv_ingest import (
    append_new_rows,
    existing_dates as _existing_dates,
    flatten_columns,
)
from engine.tickers import (
    load_registry,
    merge as _merge_registry,
    resolve_name,
    save_registry,
)
from engine.universes.index import (
    get_sp500_tickers,
    get_dow30_tickers,
    get_nasdaq100_tickers,
    get_cac40_tickers,
    get_dax_tickers,
    get_ftse100_tickers,
    get_stoxx600_tickers,
)
from engine.universes.alternative import (
    get_congressional_tickers,
    get_insider_tickers,
    get_high_short_tickers,
)
from engine.universes.assets import (
    get_crypto_tickers,
    get_crypto_eur_tickers,
    get_forex_tickers,
    get_metals_tickers,
    get_voo_only,
    get_classic_60_40,
    get_bearish_etf_tickers,
    get_bearish_etf_ucits_tickers,
    get_commodities_eur_tickers,
)


def _fetch_ticker_info(symbol: str) -> dict | None:
    """Fetch yfinance .info for a symbol. Returns None on any failure.

    Names are best-effort — a yfinance hiccup must never fail the OHLCV run.
    """
    try:
        return yf.Ticker(symbol).info  # type: ignore[no-any-return]
    except Exception as exc:
        print(f"  ! {symbol}: info fetch error — {exc}", file=sys.stderr)
        return None


# Reference symbols always fetched — used for market commentary and regime detection
# even when no strategy directly references them.
_MARKET_CONTEXT = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",  # Broad indices (ETFs)
    "^VIX",  # Volatility
    "GLD",
    "SLV",
    "TLT",  # Risk-off / safe-haven
    "BTC-USD",
    "ETH-USD",  # Crypto reference
    "DX-Y.NYB",  # US Dollar Index
]

# Crypto reference subset — used for weekend fetches (crypto trades 24/7).
_MARKET_CONTEXT_CRYPTO = ["BTC-USD", "ETH-USD"]

# Static universes not covered by their own resolver.
_ETF_SECTORS = [
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLC",
    "XLY",
    "XLP",
    "XLU",
    "XLRE",
    "XLB",
]
_ETF_BROAD = [
    "VOO",
    "QQQ",
    "VEA",
    "VWO",
    "GLD",
    "BND",
    "TLT",
    "IWM",
    "DIA",
    "HYG",
    "URTH",
    "VGK",
]


def _collect_holdings() -> set[str]:
    """Return every ticker currently held across all portfolios."""
    holdings: set[str] = set()
    portfolios_dir = get_config().portfolios_dir
    if not portfolios_dir.exists():
        return holdings
    for portfolio_dir in portfolios_dir.iterdir():
        portfolio_file = portfolio_dir / "portfolio.json"
        if not portfolio_file.exists():
            continue
        with portfolio_file.open() as f:
            data = json.load(f)
        for position in data.get("positions", []):
            ticker = position.get("ticker")
            if ticker:
                holdings.add(ticker)
    return holdings


def _collect_universe_symbols() -> set[str]:
    """Union of every ticker across every declared universe resolver."""
    symbols: set[str] = set()
    resolvers = [
        get_sp500_tickers,
        get_dow30_tickers,
        get_nasdaq100_tickers,
        get_cac40_tickers,
        get_dax_tickers,
        get_ftse100_tickers,
        get_stoxx600_tickers,
        get_crypto_tickers,
        get_crypto_eur_tickers,
        get_forex_tickers,
        get_metals_tickers,
        get_voo_only,
        get_classic_60_40,
        get_bearish_etf_tickers,
        get_bearish_etf_ucits_tickers,
        get_commodities_eur_tickers,
        get_congressional_tickers,
        get_insider_tickers,
        get_high_short_tickers,
    ]
    for resolver in resolvers:
        try:
            symbols.update(resolver())
        except Exception as exc:
            print(f"  ! {resolver.__name__} failed: {exc}", file=sys.stderr)
    symbols.update(_ETF_SECTORS)
    symbols.update(_ETF_BROAD)
    return symbols


def _all_symbols() -> list[str]:
    universe = _collect_universe_symbols()
    holdings = _collect_holdings()
    context = set(_MARKET_CONTEXT)
    return sorted(universe | holdings | context)


def _crypto_symbols() -> list[str]:
    """Weekend fetch subset — crypto pairs only (24/7 markets).

    Union of `crypto-top20` (USD) + `crypto-top20-eur` + crypto context +
    any currently held ticker ending in `-EUR` or `-USD` that looks like
    a crypto pair (upper-case ticker, not a stock).
    """
    symbols: set[str] = set()
    for resolver in (get_crypto_tickers, get_crypto_eur_tickers):
        try:
            symbols.update(resolver())
        except Exception as exc:
            print(f"  ! {resolver.__name__} failed: {exc}", file=sys.stderr)
    symbols.update(_MARKET_CONTEXT_CRYPTO)
    for held in _collect_holdings():
        if held.endswith("-EUR") or held.endswith("-USD"):
            symbols.add(held)
    return sorted(symbols)


def _fetch_symbol(symbol: str, start: date, end: date) -> pd.DataFrame | None:
    """Fetch OHLCV for a single symbol. Returns None on failure."""
    try:
        df = yf.download(
            symbol,
            start=str(start),
            end=str(end + timedelta(days=1)),  # yfinance end is exclusive
            auto_adjust=False,  # keep raw Close + Adj Close separately
            progress=False,
            threads=False,
        )
    except Exception as exc:
        print(f"  ! {symbol}: download error — {exc}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    return flatten_columns(df)


def _write_rows(symbol: str, df: pd.DataFrame) -> int:
    """Append new daily rows to data/market/ohlcv/{SYMBOL}.jsonl.

    Thin wrapper over engine.ohlcv_ingest.append_new_rows — resolves the
    config-backed store path, then delegates the normalize/merge/idempotent-append
    logic to the tested engine module.
    """
    path = get_config().ohlcv_dir / f"{symbol}.jsonl"
    return append_new_rows(path, df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-days",
        type=int,
        default=730,
        help="Days of history to fetch on first run (default 730 ≈ 2 years)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated override list (skip universe resolution)",
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help="Restrict to crypto pairs (weekend fetch — crypto trades 24/7)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List symbols without fetching"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help=(
            "Force a full re-fetch of the --history-days window for every "
            "symbol, ignoring existing rows. Used once to deepen the OHLCV "
            "store; new rows are deduped against existing dates so this is "
            "idempotent."
        ),
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help=(
            "Skip OHLCV download — only refresh the data/tickers.json "
            "registry. Used for the one-time bootstrap and for cheap "
            "re-runs after a universe change."
        ),
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    elif args.crypto_only:
        symbols = _crypto_symbols()
    else:
        symbols = _all_symbols()

    print(f"Resolved {len(symbols)} symbols to fetch.")
    if args.dry_run:
        for s in symbols:
            print(f"  {s}")
        return 0

    end = date.today()
    default_start = end - timedelta(days=args.history_days)

    registry_updates: dict[str, dict] = {}

    total_new = 0
    failures = 0
    for i, symbol in enumerate(symbols, start=1):
        if not args.names_only:
            path = get_config().ohlcv_dir / f"{symbol}.jsonl"
            if args.backfill:
                start = default_start
            elif path.exists():
                existing = _existing_dates(path)
                if existing:
                    last = max(datetime.fromisoformat(d).date() for d in existing)
                    if last >= end - timedelta(days=1):
                        registry_updates[symbol] = resolve_name(
                            symbol, _fetch_ticker_info(symbol)
                        )
                        continue  # OHLCV already up to date; still refresh name
                    start = last + timedelta(days=1)
                else:
                    start = default_start
            else:
                start = default_start

            df = _fetch_symbol(symbol, start, end)
            if df is None:
                failures += 1
            else:
                n = _write_rows(symbol, df)
                total_new += n
                if i % 25 == 0 or n > 0:
                    print(f"  [{i}/{len(symbols)}] {symbol}: +{n} rows")

        registry_updates[symbol] = resolve_name(symbol, _fetch_ticker_info(symbol))

    if registry_updates:
        existing_reg = load_registry()
        merged = _merge_registry(existing_reg, registry_updates)
        save_registry(merged)
        non_null = sum(1 for v in registry_updates.values() if v.get("name"))
        print(
            f"Refreshed tickers registry: {non_null}/{len(registry_updates)} "
            f"symbols resolved to a name."
        )

    if args.names_only:
        print(f"\nDone (names-only).")
    else:
        print(
            f"\nDone. Wrote {total_new} new rows across {len(symbols)} "
            f"symbols. {failures} failures."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
