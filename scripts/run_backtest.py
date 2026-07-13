"""CLI script for running backtests on Midas strategies.

Usage examples:
    python scripts/run_backtest.py --strategy golden-cross-sp500
    python scripts/run_backtest.py --all
    python scripts/run_backtest.py --strategy coin-flip-baseline --from 2023-01-01 --to 2024-12-31
    python scripts/run_backtest.py --all --output results.json
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

from engine.backtest import GROSS_OF_COSTS_WARNING, run_backtest
from engine.market_data import MarketDataFetcher
from engine.survivorship import survivorship_warning
from engine.types import StrategySpec
from engine.universes import resolve_universe as _resolve_universe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STRATEGIES_DIR = _PROJECT_ROOT / "data" / "strategies"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "market"

_COIN_FLIP_ID = "coin-flip-baseline"


# Universe resolution is delegated to the single engine registry via the
# `_resolve_universe` import alias above (engine.universes.resolve_universe).


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------


def _run_single(
    spec_path: Path,
    start: date,
    end: date,
    fetcher: MarketDataFetcher,
) -> dict | None:
    """Load a strategy spec, fetch data, and run a backtest.

    Returns a result dict on success, None on failure (error is printed).
    """
    try:
        spec_dict = json.loads(spec_path.read_text())
        warning = survivorship_warning(spec_dict["universe"], start)
        if warning is not None:
            print(f"  [WARN] {warning}", file=sys.stderr)
        tickers = _resolve_universe(spec_dict["universe"])
        price_data = fetcher.fetch_prices(tickers, start, end)

        result = run_backtest(spec_dict, price_data)

        return {
            "strategy_id": result.strategy_id,
            "strategy_name": result.strategy_name,
            "total_return": round(result.total_return, 6),
            "cagr": round(result.cagr, 6),
            "sharpe": round(result.sharpe, 6),
            "max_drawdown": round(result.max_drawdown, 6),
            "warnings": [warning] if warning is not None else [],
        }
    except Exception as exc:
        strategy_id = spec_path.stem
        print(f"[FAIL] {strategy_id}: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Leaderboard printer
# ---------------------------------------------------------------------------


def _print_leaderboard(results: list[dict]) -> None:
    """Print a leaderboard table sorted by total return descending."""
    sorted_results = sorted(results, key=lambda r: r["total_return"], reverse=True)

    col_id = max(len(r["strategy_id"]) for r in sorted_results)
    col_id = max(col_id, len("Strategy ID"))

    header = (
        f"{'#':>3}  "
        f"{'Strategy ID':<{col_id}}  "
        f"{'Total Return':>12}  "
        f"{'CAGR':>8}  "
        f"{'Sharpe':>8}  "
        f"{'Max DD':>8}"
    )
    separator = "-" * len(header)

    print()
    print("=" * len(header))
    print("  BACKTEST LEADERBOARD")
    print("=" * len(header))
    print(header)
    print(separator)

    for rank, r in enumerate(sorted_results, start=1):
        is_baseline = r["strategy_id"] == _COIN_FLIP_ID
        marker = " *" if is_baseline else "  "
        strategy_id_display = r["strategy_id"] + ("*" if is_baseline else "")
        print(
            f"{rank:>3}. "
            f"{strategy_id_display:<{col_id}}  "
            f"{r['total_return']:>+11.2%}  "
            f"{r['cagr']:>+7.2%}  "
            f"{r['sharpe']:>+7.3f}  "
            f"{r['max_drawdown']:>+7.2%}"
        )

    print(separator)
    print("  * = coin-flip baseline (random selector, equal-weight manager)")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Midas strategy backtests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--strategy",
        metavar="STRATEGY_ID",
        help="Run a single strategy by ID (filename without .json).",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run all strategies in data/strategies/ and print a leaderboard.",
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
        help="Save results as JSON to this file path.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"[WARN] {GROSS_OF_COSTS_WARNING}", file=sys.stderr)

    fetcher = MarketDataFetcher(cache_dir=_CACHE_DIR)

    if args.strategy:
        spec_path = _STRATEGIES_DIR / f"{args.strategy}.json"
        if not spec_path.exists():
            print(f"Error: strategy file not found: {spec_path}", file=sys.stderr)
            sys.exit(1)

        result = _run_single(spec_path, start, end, fetcher)
        if result is None:
            sys.exit(1)

        print(f"\nStrategy : {result['strategy_name']}")
        print(f"ID       : {result['strategy_id']}")
        print(f"Period   : {start} → {end}")
        print(f"Return   : {result['total_return']:+.2%}")
        print(f"CAGR     : {result['cagr']:+.2%}")
        print(f"Sharpe   : {result['sharpe']:+.3f}")
        print(f"Max DD   : {result['max_drawdown']:+.2%}")

        output_data = result

    else:  # --all
        spec_paths = sorted(_STRATEGIES_DIR.glob("*.json"))
        if not spec_paths:
            print(f"No strategy files found in {_STRATEGIES_DIR}", file=sys.stderr)
            sys.exit(1)

        print(f"Running {len(spec_paths)} strategies from {start} to {end}…\n")

        results: list[dict] = []
        for path in spec_paths:
            print(f"  → {path.stem} …", end=" ", flush=True)
            result = _run_single(path, start, end, fetcher)
            if result is not None:
                results.append(result)
                print(f"{result['total_return']:+.2%}")
            else:
                print("FAILED")

        if results:
            _print_leaderboard(results)

        output_data = results

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
