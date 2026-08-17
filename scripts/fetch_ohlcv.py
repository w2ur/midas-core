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
from engine.corporate_actions import (
    CorporateAction,
    detect_split,
    explain_quarantine,
    ratios_agree,
)
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


#: Fraction of *already-covered* symbols that may return nothing before the run
#: is a failure rather than a set of delistings. See the exit path in `main` for
#: why this is a rate, and why the denominator is store coverage rather than the
#: whole symbol list.
MAX_FAILURE_RATE = 0.10

#: Deliberate non-zero exits. Distinct from 1 on purpose: 1 is what an
#: unhandled traceback exits with, and the workflow must be able to tell "I
#: refused this data, commit the rest and go red" from "I crashed at symbol 500
#: of 1,150". Committing the latter publishes a store where a third of the
#: symbols carry today's bar and the rest stop at yesterday's — and the session
#: that prices it writes ONE immutable snapshot row per book, so the books mark
#: at different dates with no writer responsible and no way to correct it
#: (`add_snapshot` refuses a later session's rewrite of that market date).
EXIT_QUARANTINED = 2
EXIT_VENDOR_OUTAGE = 3

#: The exits on which the workflow should still commit what arrived.
COMMITTABLE_EXITS = (0, EXIT_QUARANTINED, EXIT_VENDOR_OUTAGE)


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
        quarantine = (
            get_config().data_dir / "data" / "market" / "quarantine" / f"{symbol}.jsonl"
        )
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


def _ledger_path() -> Path:
    """Where adjudicated corporate actions are recorded.

    Beside the store it describes, so it travels under the same push gate:
    evidence that only exists on an ephemeral runner is no evidence.
    """
    return get_config().data_dir / "data" / "market" / "corporate_actions.jsonl"


def _adjudicated_keys() -> set[tuple[str, str]] | None:
    """Every (symbol, effective) already applied. ``None`` means UNREADABLE.

    The distinction is load-bearing and the caller must not collapse it.
    `apply_split` mutates real share counts, and both the nightly fetch and
    the weekly resweep can see the same action; applying one twice halves a
    book twice, silently, with the cost basis invariant hiding it. So a
    ledger we cannot read has to stop adjudication altogether rather than
    degrade to "assume nothing was applied" — the money path fails closed.
    """
    path = _ledger_path()
    if not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                keys.add((record["symbol"], record["effective"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(
            f"ERROR: corporate-action ledger at {path} is unreadable ({exc}); "
            "refusing to adjudicate anything this run.",
            file=sys.stderr,
        )
        return None
    return keys


def _record_adjudication(
    action: CorporateAction, holders: list[str], refused_count: int
) -> None:
    """Append one row. The ledger is the disclosure artifact for this path."""
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "symbol": action.symbol,
        "effective": action.effective,
        "shares_ratio": action.shares_ratio,
        "price_ratio": action.price_ratio,
        "source": "vendor-calendar",
        "adjudicated_at": date.today().isoformat(),
        "rows_explained": refused_count,
        "holders_adjusted": holders,
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _fetch_actions(symbol: str) -> list[CorporateAction]:
    """The vendor's split calendar for one symbol.

    Fetched ONLY for symbols the tripwire actually refused — 0 on a healthy
    night, 3 on the worst night this month. `yf.download` does not carry
    corporate actions, so this is one extra request per symbol; doing it up
    front for the whole ~1 150-symbol universe would not be affordable, and
    that cost is the reason adjudication sits downstream of quarantine rather
    than inside the merge.
    """
    try:
        splits = yf.Ticker(symbol).splits
    except Exception as exc:  # noqa: BLE001 - vendor errors are not our bug
        print(f"WARN: could not read {symbol}'s action calendar ({exc})", file=sys.stderr)
        return []
    if splits is None or len(splits) == 0:
        return []
    actions: list[CorporateAction] = []
    for ts, value in splits.items():
        try:
            ratio = float(value)
        except (TypeError, ValueError):
            continue
        if ratio <= 0:
            continue
        effective = ts.date().isoformat() if hasattr(ts, "date") else str(ts)[:10]
        actions.append(CorporateAction(symbol, effective, ratio))
    return actions


def _adjudicate(
    refused_rows: dict[str, tuple], *, start: date, end: date
) -> tuple[int, int]:
    """Try to explain each symbol's refused rows by a real corporate action.

    Returns ``(symbols_adjudicated, rows_explained)`` — explained, never
    "still refused". The caller subtracts from the tripwire's own count, so
    anything this function does not positively account for stays red. A
    refused row whose detail never reached us (a MergeResult reporting a
    count with no rows) must not be able to turn the run green by being
    invisible: that is the "check that cannot fail" shape this repo keeps
    paying for.

    An explained symbol gets a scoped full-history re-merge with the tripwire
    OFF, which accepts the new rows AND whatever back-history the vendor
    restated as ONE decision. That is deliberate: the two limits are
    asymmetric (a new row may move 40%, a revision only 20%), so a single
    action could otherwise be half-ingested — JMAT.L's store took 08-13 at
    +32% through the new-row door while its matching restatement of 08-12 was
    refused, leaving the series straddling two bases. The asymmetry itself is
    correct and is not being changed; what was missing is knowing the two
    sides are one event.

    Anything unexplained stays quarantined and the run still goes red. This
    path can only ever turn a refusal into an ingest when the vendor's own
    calendar says so.
    """
    if not refused_rows:
        return 0, 0

    already = _adjudicated_keys()
    if already is None:
        return 0, 0

    adjudicated = 0
    rows_explained = 0
    for symbol in sorted(refused_rows):
        rows = refused_rows[symbol]
        calendar = _fetch_actions(symbol)
        action = explain_quarantine(rows, calendar)
        if action is None:
            # Say WHICH of the two conditions failed. They call for different
            # responses and the run's only other signal is a red X. The ratio
            # case also degrades with time: a new row's ratio is measured
            # against `latest_stored_close`, which is FROZEN for the length of
            # the freeze, so the longer a symbol stays stuck the further the
            # live price drifts from the implied ratio until no calendar entry
            # can explain it any more. Silence there would read as "no
            # corporate action" when it means "you left this too long".
            if not calendar:
                print(
                    f"  ? {symbol}: {len(rows)} refused row(s), no corporate "
                    "action on the vendor's calendar — left quarantined.",
                    file=sys.stderr,
                )
            else:
                observed = sorted({round(r.ratio, 4) for r in rows})
                print(
                    f"  ? {symbol}: {len(rows)} refused row(s) at ratio(s) "
                    f"{observed} match no calendar action "
                    f"{[(a.effective, round(a.price_ratio, 4)) for a in calendar]} "
                    "— left quarantined.",
                    file=sys.stderr,
                )
            continue
        if (action.symbol, action.effective) in already:
            # Already applied on an earlier run; the rows were refused again
            # only because the store still carries the old basis. Re-merging
            # is safe and idempotent, but the holdings must NOT move twice.
            print(
                f"  = {symbol}: {action.effective} action already adjudicated; "
                "re-merging the store only.",
                file=sys.stderr,
            )
            holders: list[str] = []
        else:
            holders = _apply_split_to_holders(symbol, action.shares_ratio)
            _record_adjudication(action, holders, len(rows))
            already.add((action.symbol, action.effective))

        # The store side: re-fetch the full window and merge it with the
        # tripwire OFF, revising from the window start so both the new rows
        # and any restated history land together. Exactly what
        # `--resweep --symbols <sym>` does — reused rather than reimplemented,
        # so there is one rewrite path, not two.
        df = _fetch_symbol(symbol, start, end)
        if df is None:
            print(
                f"    (could not re-fetch {symbol}; store unchanged, "
                "the action is recorded and will apply on the next run)",
                file=sys.stderr,
            )
            continue
        remerged = _write_rows(
            symbol, df, revise_from=start.isoformat(), guard_anomalies=False
        )
        if not (remerged.appended or remerged.revised):
            # The calendar explained the refusal but the re-merge landed
            # nothing, so the store is STILL frozen on the pre-action close.
            # Counting these rows as explained would drop `unadjudicated` to
            # zero and exit 0 — and every later night would refuse the same
            # rows, hit the `already` branch, and exit 0 again, making the
            # freeze permanent AND invisible. That is exactly what the
            # quarantine exit code exists to surface. Reachable: Yahoo serves
            # a date over one window and not another, in both directions.
            print(
                f"    ({symbol}: re-merge wrote nothing — the store is still "
                "frozen; leaving the rows refused so the run stays red)",
                file=sys.stderr,
            )
            continue

        print(
            f"  ! {symbol}: ADJUDICATED — {action.effective} corporate action, "
            f"shares x{action.shares_ratio:.6g} (price x{action.price_ratio:.6g}), "
            f"{len(rows)} refused row(s) accepted; store re-merged "
            f"(+{remerged.appended} rows, ~{remerged.revised} revised)"
            + (f"; adjusted {', '.join(holders)}" if holders else ""),
            file=sys.stderr,
        )
        adjudicated += 1
        rows_explained += len(rows)
    return adjudicated, rows_explained


def _warn_if_unheld(symbol: str, adjusted: list[str]) -> None:
    """Say so when a real split touched nobody's book — only where one was applied.

    Printed on the branches that actually attempted an adjustment. On the
    ledger-skip branches nothing was attempted, and "no agent currently holds
    X" there would read as a fact about the roster rather than as the reason
    the holdings were left alone.
    """
    if not adjusted:
        print(f"    (no agent currently holds {symbol})", file=sys.stderr)


def _match_detected_split(symbol: str, ratio: float) -> CorporateAction | None:
    """Find the calendar action a drift-inferred ratio corresponds to.

    `detect_split` returns a bare ratio with no effective date, and the ledger
    is keyed on (symbol, effective) — so the calendar is what lets a
    drift-inferred detection share an identity with a calendar-adjudicated
    one. Compared on `shares_ratio`, which is the convention `detect_split`
    already returns (0.75 for JMAT.L's 3:4, not its 1.3333 price ratio).
    """
    for action in _fetch_actions(symbol):
        if ratios_agree(ratio, action.shares_ratio):
            return action
    return None


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

    # NEVER today (2026-08-12). `end` is the newest day this run will ask the
    # vendor for, and asking for today is asking for a bar that is still
    # forming. On a cash market that costs nothing — the day has not opened, so
    # the vendor returns nothing — but a 24/7 instrument (crypto, FX, and
    # futures on Globex) is served a partial bar the moment the UTC day opens.
    # Measured on 2026-08-12: Yahoo returns a same-day BTC-USD close at 07:49
    # UTC. Under the 06:00 cron that row is roughly six hours old, the 20:00
    # session publishes it, and `PortfolioManager.add_snapshot` then freezes it
    # — the next morning's revision arrives too late to move the published
    # mark. Under the old 22:30 cron the same row was ~22.5 h formed, which was
    # wrong more cheaply rather than right.
    #
    # Ending at yesterday means every row this run stores is a COMPLETE daily
    # bar, on every instrument. The published mark becomes a final close.
    #
    # The snapshot dating is unchanged by this, which is the point: at 06:00 on
    # day D the store advances to D-1 and the 20:00 session on D publishes a
    # D-1-dated row — exactly what the 22:30 cron on D-1 produced. Verified
    # against the live record (`data/portfolios/*/snapshots.json`): the 08-04
    # session published 08-03, the 08-05 session published 08-04.
    end = date.today() - timedelta(days=1)

    # Universal 1-day revision window: re-request the trailing stored day and
    # let its final value replace it.
    #
    # It no longer catches PARTIAL bars — with `end` at yesterday nothing
    # partial is ever stored. What it catches now is a genuine vendor
    # revision of an already-complete bar, which is real and measurable:
    # `GC=F`/`PL=F`/`CL=F` each moved on 13 of 22 shared days (up to +3.4%)
    # and FX on 5 of 23 (worst -1.56%) when that was measured under the 22:30
    # schedule. Yahoo also restates raw `close` outright for a corporate
    # action, which is what `detect_split` adjudicates on the weekly resweep.
    #
    # For a bar the vendor does not revise, the re-fetch is identical,
    # `merge_rows` finds nothing to replace, and no already-stored row changes
    # value or position. (The file is still rewritten to append the day's new
    # bar, and a genuine revision does rewrite it — what is invariant is the
    # stored rows, not the file's mtime.) So a blanket window adds no churn
    # and needs no instrument taxonomy to maintain as universes grow.
    revise_days = 1

    registry_updates: dict[str, dict] = {}

    total_new = 0
    total_revised = 0
    total_quarantined = 0
    #: symbol -> the rows the tripwire refused this run, for adjudication.
    refused_rows: dict[str, tuple] = {}
    # Split by whether the store already covers the symbol — see the exit path.
    # `considered_covered` counts every covered symbol the run formed a view on,
    # including those skipped as already-current: they are evidence the store is
    # healthy, and dropping them collapses the denominator.
    considered_covered = 0
    covered_failures = 0
    unresolved: list[str] = []
    served = 0
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
                    # A symbol the store already carries through `end` is
                    # evidence the store is healthy, so it belongs in the
                    # gate's denominator even though nothing was requested for
                    # it. Skipping it here collapses that denominator onto the
                    # handful of permanently-dead names that ARE still
                    # attempted (their store is stale, so they never take this
                    # branch) — MATIC-USD and UNI-USD alone read 2/2 = 100%
                    # and fail a fully current store. Reachable whenever most
                    # symbols are already up to date: a second run in one UTC
                    # day, or the Tue-Sat full run (`0 6 * * 2-6`) delayed past
                    # midnight into Sunday — its `end` is then Saturday, so it
                    # writes the Saturday rows for the 24/7 names before the
                    # Sun-Mon crypto-only run (`0 6 * * 0,1`), whose `end` is
                    # also Saturday, asks for them.
                    if path.exists():
                        considered_covered += 1
                    continue  # OHLCV already up to date; still refresh name
                start = window_start
                if revise_days and last is not None:
                    revise_from = start.isoformat()

            # Whether the store already covers this symbol decides which
            # population its failure belongs to, and it has to be read before
            # the write below creates the file.
            covered = path.exists()
            if covered:
                considered_covered += 1

            df = _fetch_symbol(symbol, start, end)
            if df is None:
                if covered:
                    covered_failures += 1
                else:
                    unresolved.append(symbol)
            else:
                served += 1
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
                        # Route through the SAME ledger the nightly
                        # adjudication uses, or one action gets applied twice.
                        # Reachable and not hypothetical: an unrestated split
                        # (MNST's shape) is adjudicated from the calendar on
                        # the nightly path, which doubles the shares; days
                        # later the vendor restates history (JMAT.L proves it
                        # happens late), this resweep sees the drift, and
                        # `detect_split` returns the same ratio again — shares
                        # x4, with the cost-basis invariant hiding it on every
                        # application. `detect_split` reports no effective
                        # date, so the calendar supplies one; a drift-only
                        # detection with no matching calendar entry keeps the
                        # old unledgered behaviour and says so out loud.
                        action = _match_detected_split(symbol, ratio)
                        already = _adjudicated_keys()
                        if action is None:
                            print(
                                "    (drift-inferred, no matching calendar "
                                "entry — applying unledgered)",
                                file=sys.stderr,
                            )
                            adjusted = _apply_split_to_holders(symbol, ratio)
                            _warn_if_unheld(symbol, adjusted)
                        elif already is None:
                            print(
                                "    (ledger unreadable — refusing to adjust "
                                "holdings this run)",
                                file=sys.stderr,
                            )
                            adjusted = []
                        elif (action.symbol, action.effective) in already:
                            print(
                                f"    ({action.effective} action already "
                                "adjudicated; holdings left alone)",
                                file=sys.stderr,
                            )
                            adjusted = []
                        else:
                            adjusted = _apply_split_to_holders(
                                symbol, action.shares_ratio
                            )
                            _record_adjudication(action, adjusted, 0)
                            _warn_if_unheld(symbol, adjusted)
                # The tripwire is off on a resweep: a real corporate action
                # legitimately rewrites hundreds of rows there, and detect_split
                # above is what adjudicates them. On the nightly path a
                # wholesale rewrite of history is never legitimate.
                merged = _write_rows(
                    symbol, df, revise_from, guard_anomalies=not args.resweep
                )
                n, r, q = merged.appended, merged.revised, merged.quarantined
                total_new += n
                total_revised += r
                total_quarantined += q
                if merged.refused:
                    # Kept for the adjudication pass below, which needs the
                    # dates and ratios rather than the count.
                    refused_rows[symbol] = merged.refused
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

    # A refused row is not necessarily a defect: a real corporate action has
    # the same shape as the units flip the tripwire hunts. Ask the vendor's own
    # calendar before going red, because nothing else in the system can turn a
    # refusal back into an ingest — `resweep-held-tickers` sweeps HELD tickers
    # only, so a universe ticker nobody holds froze indefinitely while staying
    # tradable at its stale price. Unexplained rows still go red.
    adjudicated_symbols, rows_explained = _adjudicate(
        refused_rows, start=end - timedelta(days=args.history_days), end=end
    )
    # Subtracted from the tripwire's own count, not reported independently:
    # only rows positively explained by the vendor's calendar come off the
    # tally, so any refusal this pass did not account for still goes red.
    unadjudicated = total_quarantined - rows_explained

    # Derived, not tracked in parallel: the two populations partition every
    # failure, so a separate counter could only ever disagree with them.
    failures = covered_failures + len(unresolved)

    if args.names_only:
        print(f"\nDone (names-only).")
    else:
        print(
            f"\nDone. Wrote {total_new} new rows ({total_revised} revised) "
            f"across {len(symbols)} symbols. {failures} failures."
            + (f" {splits_detected} split(s) detected." if splits_detected else "")
            + (
                f" {adjudicated_symbols} symbol(s) ADJUDICATED against the "
                "vendor's action calendar."
                if adjudicated_symbols
                else ""
            )
            + (
                f" {unadjudicated} row(s) QUARANTINED — see "
                "data/market/quarantine/."
                if unadjudicated
                else ""
            )
        )
    # A symbol the store has never covered is a different fault from one that
    # stopped answering, and folding the two together is what made the rate
    # below unusable. 118 of the 120 failures on 2026-08-07 were Refinitiv-style
    # codes in data/universes/stoxx600.json that Yahoo has no route for at all
    # (AIRP.PA, BNPP.PA, CAGR.PA, ATCOa.ST — Yahoo wants AI.PA, BNP.PA, ACA.PA,
    # ATCO-A.ST). They will never resolve, so failing the run on them would
    # freeze the store permanently. They are reported instead, because a
    # universe carrying ~120 unfetchable tickers is a real defect — just not
    # this script's, and not one a red run can fix.
    if unresolved:
        print(
            f"\nWARN: {len(unresolved)} symbol(s) have never served a row and "
            "have no store file. These are ticker-resolution failures, not a "
            "vendor outage — most likely non-Yahoo symbol formats in "
            f"data/universes/. First 10: {', '.join(sorted(unresolved)[:10])}",
            file=sys.stderr,
        )

    # Neither must a run in which the vendor answered for almost nothing
    # (2026-08-07 review, W2.5). This script exited 0 regardless of the
    # failure count, so a total Yahoo outage produced a green run, "No OHLCV
    # changes to commit", and a session pricing a stale store the next evening
    # with nothing anywhere saying the data had not arrived.
    #
    # The threshold is a rate, not a count: individually dead symbols are
    # normal and permanent here, so any absolute floor would either fire every
    # night or never. The denominator is the symbols the store ALREADY COVERS,
    # not the whole list. Against the whole list the guard was mis-calibrated
    # from the day it shipped: the steady-state baseline was 121 failures of
    # 1,150 (10.5%), already over the 10% limit, so every full-universe run
    # from 2026-08-07 onward exited 1 and the store stopped advancing — the
    # exact outage it was written to detect, caused by the detector. Against
    # store coverage the same night reads 2 of ~1,030 (0.2%), and a genuine
    # vendor outage still reads ~100%.
    if not args.names_only and considered_covered:
        covered_rate = covered_failures / considered_covered
        if covered_rate > MAX_FAILURE_RATE:
            print(
                f"\nFAILED: {covered_failures} of {considered_covered} symbols "
                "the store already covers returned no data "
                f"({covered_rate:.0%}, limit {MAX_FAILURE_RATE:.0%}). A symbol "
                "that served yesterday and serves nothing today is a "
                "vendor-side or network failure, not a delisting. Whatever did "
                "arrive is still committed, but the store is incomplete; a "
                "session running against it tonight would price some books at "
                "stale closes.",
                file=sys.stderr,
            )
            return EXIT_VENDOR_OUTAGE

    # Nothing is covered yet — a fork's first bootstrap, or a run after the
    # store directory moved. The rate above cannot measure anything there, and
    # returning 0 unconditionally would reopen W2.5's exact hole on the one run
    # with nothing to fall back on. What IS decidable without coverage is
    # whether the vendor answered at all: a bootstrap where every single symbol
    # came back empty is an outage, while one where 1,030 of 1,150 served is a
    # healthy bootstrap with a bad universe.
    if not args.names_only and not considered_covered and symbols and not served:
        print(
            f"\nFAILED: none of the {len(symbols)} symbols returned any data, "
            "and the store has no prior coverage to compare against. On an "
            "empty store this is the only decidable outage signal — a "
            "bootstrap in which the vendor answered for nothing at all.",
            file=sys.stderr,
        )
        return EXIT_VENDOR_OUTAGE

    # A quarantined row is a refusal to ingest something that looked like a
    # unit flip or a bad tick. It is not a crash, but it must not scroll past
    # in a green run either: exiting non-zero routes it to the workflow's
    # failure-issue action, which files a persistent issue. Checked last so the
    # reports above are never cut off by an early return.
    if unadjudicated:
        return EXIT_QUARANTINED
    return 0


if __name__ == "__main__":
    sys.exit(main())
