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
from engine.corporate_actions import detect_split
from engine.quotes import vendor_unit_scale
from engine.ohlcv_ingest import (
    MergeResult,
    existing_dates as _existing_dates,
    fetch_window_start,
    flatten_columns,
    merge_rows,
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


#: Fraction of symbols that may return nothing before the run is a failure
#: rather than a set of delistings. See the exit path in `main` for why this is
#: a rate and not a count.
MAX_FAILURE_RATE = 0.10


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
        # Silent until 2026-08-07 (W2.5). An empty frame is how a delisting, a
        # renamed symbol and a vendor-side outage all present, and they are
        # indistinguishable here — but a run where hundreds of them happen at
        # once is very distinguishable from one where two do, and that is the
        # signal the exit code below is built on.
        print(f"  ! {symbol}: vendor returned no rows", file=sys.stderr)
        return None
    return _normalise_vendor_units(symbol, flatten_columns(df))


#: Price columns yfinance serves in the vendor's quote unit. `Volume` is a
#: share count, not a price, and must never be scaled.
_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close")


def _normalise_vendor_units(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Scale a fetched frame from the vendor's quote unit into ISO currency.

    The LSE quotes in pence, so yfinance serves `LLOY.L` at 116.60 meaning
    GBP 1.166. **The store is ISO-denominated**, so that division happens
    here, once, on the way in — rather than in every one of the many places
    that later read a price back out. Three read paths forgot it before it
    was centralised, and centralising it on the read side still left agents
    reading raw pence out of the store while their books were in pounds.

    Applies to the freshly fetched frame only, which is also what
    `engine.corporate_actions.detect_split` compares against the store — so
    the two stay in the same units and a normalised store does not read as a
    clean 100:1 "split" on every London ticker.
    """
    scale = vendor_unit_scale(symbol)
    if scale == 1.0:
        return df
    out = df.copy()
    for column in _PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column] * scale
    return out


def _write_rows(
    symbol: str,
    df: pd.DataFrame,
    revise_from: str | None = None,
    *,
    guard_anomalies: bool = False,
) -> MergeResult:
    """Merge new daily rows into data/market/ohlcv/{SYMBOL}.jsonl.

    Thin wrapper over engine.ohlcv_ingest.merge_rows — resolves the config-backed
    store path, then delegates the normalize/merge/idempotent-write logic to the
    tested engine module.

    ``guard_anomalies`` turns on the ingest tripwire, quarantining a revision or
    new row that moves implausibly far. It is on for the nightly one-day-window
    run, where rewriting history wholesale is never legitimate, and OFF for a
    resweep, where a real corporate action legitimately rewrites hundreds of
    rows and ``engine.corporate_actions.detect_split`` is what adjudicates them.
    """
    path = get_config().ohlcv_dir / f"{symbol}.jsonl"
    quarantine = None
    if guard_anomalies:
        quarantine = get_config().data_dir / "data" / "market" / "quarantine" / f"{symbol}.jsonl"
    return merge_rows(path, df, revise_from, quarantine=quarantine)


def _read_store_rows(path: Path) -> list[dict]:
    """Read every row of a {SYMBOL}.jsonl store file as dicts, pre-write.

    Missing file, blank lines, and unparseable lines are skipped rather than
    raising — mirrors engine.ohlcv_ingest.existing_dates' defensive parsing,
    so one corrupt line never aborts the run.
    """
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _apply_split_to_holders(symbol: str, ratio: float) -> list[str]:
    """Adjust every portfolio currently holding `symbol` for a detected split.

    Prints one loud, clearly-marked line per adjusted agent — ticker, ratio,
    agent, and before/after share count — so a detected split is visible in
    the session/workflow log, not just a silent JSON mutation. No
    confirmation step: safety here is the detector's own conservative
    thresholds (engine.corporate_actions.detect_split), matching this
    engine's existing "rails in code, not prompts" design — see the
    Brain/Hands principle in CLAUDE.md.

    Delayed import — engine.portfolio is not otherwise needed on this
    script's hot path, and importing it at call time keeps test
    monkeypatching of get_config().portfolios_dir respected.

    Returns the agent_ids whose position was actually adjusted (a no-op for
    any agent not holding the ticker).
    """
    from engine.portfolio import PortfolioManager

    portfolios_dir = get_config().portfolios_dir
    if not portfolios_dir.exists():
        return []
    manager = PortfolioManager(base_dir=portfolios_dir)
    adjusted: list[str] = []
    for portfolio_dir in sorted(portfolios_dir.iterdir()):
        if not (portfolio_dir / "portfolio.json").exists():
            continue
        agent_id = portfolio_dir.name
        before = next(
            (p for p in manager.load(agent_id).positions if p.ticker == symbol), None
        )
        if before is None:
            continue
        if not manager.apply_split(agent_id, symbol, ratio):
            continue
        after = next(p for p in manager.load(agent_id).positions if p.ticker == symbol)
        print(
            f"    [SPLIT ADJUSTED] agent={agent_id} ticker={symbol} "
            f"ratio={ratio:.4f} shares {before.shares:.4f} -> {after.shares:.4f} "
            f"avg_cost {before.avg_cost:.4f} -> {after.avg_cost:.4f}",
            file=sys.stderr,
        )
        adjusted.append(agent_id)
    return adjusted


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
        "--resweep",
        action="store_true",
        help=(
            "Re-fetch the full --history-days window AND allow every row in it to "
            "be revised. Corrects partial bars frozen anywhere in history, unlike "
            "--backfill which re-fetches but writes append-only. Requires an "
            "explicit --symbols list: it rewrites committed history. Also the "
            "path that runs corporate-action (stock split) detection — see "
            "engine.corporate_actions — since that needs a genuine historical "
            "overlap the 1-day incremental revision window can't provide."
        ),
    )
    parser.add_argument(
        "--resweep-held",
        action="store_true",
        help=(
            "Convenience form of --resweep, scoped automatically to every "
            "ticker currently held across all portfolios (via "
            "_collect_holdings()) instead of a hand-maintained --symbols "
            "list. This is the scheduled trigger for split detection: a real "
            "split only matters at all for a ticker some agent actually "
            "holds, and that set is small (on the order of 30 symbols) — "
            "cheap enough to resweep on a schedule. Mutually exclusive with "
            "--symbols (it resolves its own) and --backfill."
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

    if args.resweep and not args.symbols and not args.resweep_held:
        parser.error("--resweep requires an explicit --symbols list")
    if args.resweep and args.backfill:
        parser.error("--resweep and --backfill are mutually exclusive")
    if args.resweep_held and args.symbols:
        parser.error(
            "--resweep-held resolves its own --symbols list — do not pass both"
        )
    if args.resweep_held and args.backfill:
        parser.error("--resweep-held and --backfill are mutually exclusive")

    if args.symbols:
        symbols = sorted({s.strip() for s in args.symbols.split(",") if s.strip()})
    elif args.resweep_held:
        symbols = sorted(_collect_holdings())
        # Reuses the exact --resweep code path below (full-window revise_from
        # + split detection) — the only difference is how `symbols` got
        # resolved.
        args.resweep = True
        if not symbols:
            print("No open positions across any portfolio — nothing to resweep.")
            return 0
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

    # Universal 1-day revision window: re-request the trailing stored day and
    # let its final value replace it. Crypto trades 24/7; commodity futures
    # (`=F`) have already opened the next Globex session by 22:30 UTC; FX (`=X`)
    # rolls at 17:00 ET and drifts mildly, worst on Fridays. All three are
    # still forming when first written. Cash equities and ETFs ARE final at
    # fetch time — for them the re-fetched bar is identical, `merge_rows` finds
    # nothing to replace, and no already-stored row changes value or position.
    # (The file is still rewritten to append the day's new bar, and a genuine
    # revision does rewrite it — what is invariant is the stored rows, not the
    # file's mtime.) So a blanket window adds no churn and needs no instrument
    # taxonomy to maintain as universes grow.
    revise_days = 1

    registry_updates: dict[str, dict] = {}

    total_new = 0
    total_revised = 0
    total_quarantined = 0
    failures = 0
    splits_detected = 0
    for i, symbol in enumerate(symbols, start=1):
        if not args.names_only:
            path = get_config().ohlcv_dir / f"{symbol}.jsonl"
            revise_from: str | None = None

            if args.resweep:
                start = end - timedelta(days=args.history_days)
                revise_from = start.isoformat()
            elif args.backfill:
                start = end - timedelta(days=args.history_days)
            else:
                existing = _existing_dates(path) if path.exists() else set()
                last = (
                    max(datetime.fromisoformat(d).date() for d in existing)
                    if existing
                    else None
                )
                window_start = fetch_window_start(
                    last, end, args.history_days, revise_days=revise_days
                )
                if window_start is None:
                    registry_updates[symbol] = resolve_name(
                        symbol, _fetch_ticker_info(symbol)
                    )
                    continue  # OHLCV already up to date; still refresh name
                start = window_start
                if revise_days and last is not None:
                    revise_from = start.isoformat()

            df = _fetch_symbol(symbol, start, end)
            if df is None:
                failures += 1
            else:
                # Split detection needs a genuinely historical overlap between
                # what's already stored and what was just re-fetched — only
                # --resweep re-requests the full history window against
                # already-committed rows still due to be revised. The
                # universal 1-day revision window (revise_from set below,
                # every run) overlaps on exactly one trailing date, which
                # detect_split's own "at least a handful of rows" gate
                # already refuses on its own; gating on --resweep here avoids
                # the wasted store re-read on every symbol, every night, for
                # a comparison that structurally cannot fire.
                if args.resweep:
                    stored_rows = _read_store_rows(path)
                    ratio = detect_split(stored_rows, df)
                    if ratio is not None:
                        splits_detected += 1
                        print(
                            f"  ! {symbol}: SPLIT DETECTED, ratio={ratio:.4f}",
                            file=sys.stderr,
                        )
                        adjusted = _apply_split_to_holders(symbol, ratio)
                        if not adjusted:
                            print(
                                f"    (no agent currently holds {symbol})",
                                file=sys.stderr,
                            )
                # The tripwire is off on a resweep: a real corporate action
                # legitimately rewrites hundreds of rows there, and detect_split
                # above is what adjudicates them. On the nightly path a
                # wholesale rewrite of history is never legitimate.
                n, r, q = _write_rows(
                    symbol, df, revise_from, guard_anomalies=not args.resweep
                )
                total_new += n
                total_revised += r
                total_quarantined += q
                if i % 25 == 0 or n > 0 or r > 0 or q > 0:
                    suffix = f", !{q} quarantined" if q else ""
                    print(
                        f"  [{i}/{len(symbols)}] {symbol}: +{n} rows, "
                        f"~{r} revised{suffix}"
                    )

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
            f"\nDone. Wrote {total_new} new rows ({total_revised} revised) "
            f"across {len(symbols)} symbols. {failures} failures."
            + (f" {splits_detected} split(s) detected." if splits_detected else "")
            + (
                f" {total_quarantined} row(s) QUARANTINED — see "
                "data/market/quarantine/."
                if total_quarantined
                else ""
            )
        )
    # A quarantined row is a refusal to ingest something that looked like a
    # unit flip or a bad tick. It is not a crash, but it must not scroll past
    # in a green run either: exiting non-zero routes it to the workflow's
    # failure-issue action, which files a persistent issue.
    if total_quarantined:
        return 1

    # Neither must a run in which the vendor answered for almost nothing
    # (2026-08-07 review, W2.5). This script exited 0 regardless of the
    # failure count, so a total Yahoo outage produced a green run, "No OHLCV
    # changes to commit", and a session pricing a stale store the next evening
    # with nothing anywhere saying the data had not arrived.
    #
    # The threshold is a rate, not a count: individually dead symbols are
    # normal and permanent here (MATIC-USD and UNI-USD have served nothing
    # since March and April 2025), so any absolute floor would either fire
    # every night or never. 10% of ~1,150 symbols is ~115 — two orders of
    # magnitude above the handful of known-dead names, and far below what any
    # real outage looks like.
    if not args.names_only and symbols:
        failure_rate = failures / len(symbols)
        if failure_rate > MAX_FAILURE_RATE:
            print(
                f"\nFAILED: {failures} of {len(symbols)} symbols returned no "
                f"data ({failure_rate:.0%}, limit {MAX_FAILURE_RATE:.0%}). "
                "This is a vendor-side or network failure, not a set of "
                "delistings. The store has NOT been refreshed; a session "
                "running against it tonight would price at stale closes.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
