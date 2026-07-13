"""Factor research: generate and backtest all valid selector × manager combinations.

Usage examples:
    python scripts/run_all_combos.py
    python scripts/run_all_combos.py --universes sp500,etf-broad --from 2024-01-01
    python scripts/run_all_combos.py --selectors golden-cross,rsi-oversold --managers equal-weight,trailing-stop
    python scripts/run_all_combos.py --output data/factor-research.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

# Add project root to sys.path so engine imports work when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.backtest import GROSS_OF_COSTS_WARNING, run_backtest
from engine.market_data import MarketDataFetcher
from engine.survivorship import survivorship_warning
from engine.universes import resolve_universe as _resolve_universe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "market"

# Selectors that work in backtesting (excludes claude-analysis and data-follow)
_BACKTESTABLE_SELECTORS = [
    "golden-cross",
    "rsi-oversold",
    "dip-entry",
    "earnings-beat",
    "sector-cycle",
    "fear-greed",
    "random",
]

# Only managers with a DISTINCT implemented behavior. grid-conservative,
# trailing-stop, and rebalance-monthly are aliases for equal-weight (see
# engine/adapter.py) — advertising them in the default grid produced identical
# columns dressed up as different strategies.
_DEFAULT_MANAGERS = [
    "equal-weight",
    "volatility-sized",
]

# Default to dow30 (survivorship-stable) rather than sp500: backtesting
# against sp500's *current* membership over a historical window inflated an
# early Midas run ~194% (see engine.survivorship / METHODOLOGY.md).
_DEFAULT_UNIVERSES = ["dow30", "etf-broad"]

# Universe resolution is delegated to the single engine registry via the
# `_resolve_universe` import alias above (engine.universes.resolve_universe).


def _git_sha() -> str | None:
    """Return the current git HEAD SHA, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Combo runner
# ---------------------------------------------------------------------------


def _run_combo(
    universe_id: str,
    selector: str,
    manager: str,
    price_data,
    start: date,
    end: date,
) -> dict | None:
    """Run a single universe × selector × manager combination."""
    combo_id = f"{universe_id}__{selector}__{manager}"
    spec_dict = {
        "id": combo_id,
        "name": f"{universe_id} | {selector} | {manager}",
        "universe": universe_id,
        "selector": selector,
        "manager": manager,
        "funding": {"initial": 10000, "monthly_addition": 0},
        "dividends": "reinvest",
        "rules": {
            "max_positions": 10,
            "max_position_pct": 20.0,
            "min_hold_days": 3,
        },
    }

    try:
        result = run_backtest(spec_dict, price_data)
        return {
            "universe": universe_id,
            "selector": selector,
            "manager": manager,
            "combo_id": combo_id,
            "total_return": round(result.total_return, 6),
            "cagr": round(result.cagr, 6),
            "sharpe": round(result.sharpe, 6),
            "max_drawdown": round(result.max_drawdown, 6),
        }
    except Exception as exc:
        print(f"  [FAIL] {combo_id}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Heatmap printer
# ---------------------------------------------------------------------------


def _print_heatmap(
    results: list[dict],
    selectors: list[str],
    managers: list[str],
    universe_id: str,
) -> None:
    """Print a heatmap with selectors as rows and managers as columns."""
    # Build lookup: (universe, selector, manager) -> total_return
    lookup: dict[tuple[str, str, str], float] = {}
    for r in results:
        if r["universe"] == universe_id:
            key = (r["universe"], r["selector"], r["manager"])
            lookup[key] = r["total_return"]

    if not lookup:
        return

    col_width = 12
    sel_width = max(len(s) for s in selectors)

    # Header
    print(f"\n  Universe: {universe_id}")
    header = f"  {'Selector':<{sel_width}}"
    for mgr in managers:
        header += f"  {mgr[:col_width]:>{col_width}}"
    print(header)
    print("  " + "-" * (sel_width + (col_width + 2) * len(managers)))

    for sel in selectors:
        row = f"  {sel:<{sel_width}}"
        for mgr in managers:
            val = lookup.get((universe_id, sel, mgr))
            if val is None:
                row += f"  {'N/A':>{col_width}}"
            else:
                row += f"  {val:>+{col_width - 1}.2%} "
        print(row)


def _print_full_heatmap(
    results: list[dict],
    selectors: list[str],
    managers: list[str],
    universes: list[str],
) -> None:
    """Print heatmaps for all universes."""
    print()
    print("=" * 72)
    print("  FACTOR RESEARCH HEATMAP — Total Return % (selectors × managers)")
    print("=" * 72)

    for universe_id in universes:
        _print_heatmap(results, selectors, managers, universe_id)

    print()
    print("  Legend: N/A = combination failed or was not run")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and backtest all selector × manager combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--universes",
        default=",".join(_DEFAULT_UNIVERSES),
        help=f"Comma-separated universe IDs. Default: {','.join(_DEFAULT_UNIVERSES)}",
    )
    parser.add_argument(
        "--selectors",
        default=",".join(_BACKTESTABLE_SELECTORS),
        help=f"Comma-separated selector IDs. Default: all backtestable selectors.",
    )
    parser.add_argument(
        "--managers",
        default=",".join(_DEFAULT_MANAGERS),
        help=f"Comma-separated manager IDs. Default: {','.join(_DEFAULT_MANAGERS)}",
    )
    parser.add_argument(
        "--from",
        dest="start",
        metavar="START_DATE",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD). Default: 2024-01-01.",
    )
    parser.add_argument(
        "--to",
        dest="end",
        metavar="END_DATE",
        default=date.today().isoformat(),
        help="End date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default="data/factor-research.json",
        help="JSON output file. Default: data/factor-research.json.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    universes = [u.strip() for u in args.universes.split(",") if u.strip()]
    selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    managers = [m.strip() for m in args.managers.split(",") if m.strip()]

    total_combos = len(universes) * len(selectors) * len(managers)
    print(
        f"Running {total_combos} combinations "
        f"({len(universes)} universes × {len(selectors)} selectors × {len(managers)} managers)"
    )
    print(f"Period: {start} → {end}\n")

    print(f"[WARN] {GROSS_OF_COSTS_WARNING}", file=sys.stderr)
    for universe_id in universes:
        warning = survivorship_warning(universe_id, start)
        if warning is not None:
            print(f"[WARN] {warning}", file=sys.stderr)

    fetcher = MarketDataFetcher(cache_dir=_CACHE_DIR)

    # Pre-fetch price data per universe to avoid redundant downloads.
    print("Pre-fetching price data…")
    universe_prices: dict[str, object] = {}
    for universe_id in universes:
        print(f"  → {universe_id} …", end=" ", flush=True)
        try:
            tickers = _resolve_universe(universe_id)
            prices = fetcher.fetch_prices(tickers, start, end)
            universe_prices[universe_id] = prices
            print(f"{len(tickers)} tickers")
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)

    # Generate and run all combos.
    print("\nRunning combinations…")
    results: list[dict] = []
    combo_num = 0

    for universe_id, selector, manager in itertools.product(
        universes, selectors, managers
    ):
        combo_num += 1
        if universe_id not in universe_prices:
            print(f"  [{combo_num:>4}/{total_combos}] SKIP (no data for {universe_id})")
            continue

        combo_id = f"{universe_id}__{selector}__{manager}"
        print(f"  [{combo_num:>4}/{total_combos}] {combo_id} …", end=" ", flush=True)

        result = _run_combo(
            universe_id,
            selector,
            manager,
            universe_prices[universe_id],
            start,
            end,
        )
        if result is not None:
            results.append(result)
            print(f"{result['total_return']:+.2%}")
        else:
            print("FAILED")

    # Print heatmap.
    if results:
        _print_full_heatmap(results, selectors, managers, universes)

    # Save output.
    output_path = (
        _PROJECT_ROOT / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Stamp provenance so a results file is reproducible: when it was generated,
    # the exact commit, and the arguments that produced it. `results` stays a
    # top-level key so consumers can read both the metadata and the rows.
    payload = {
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "git_sha": _git_sha(),
        "args": {
            "universes": universes,
            "selectors": selectors,
            "managers": managers,
            "from": start.isoformat(),
            "to": end.isoformat(),
        },
        "warnings": [GROSS_OF_COSTS_WARNING],
        "results": results,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to: {output_path}")
    print(f"Total: {len(results)}/{total_combos} combinations succeeded.")


if __name__ == "__main__":
    main()
