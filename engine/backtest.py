"""Backtest runner — wraps bt's execution with StrategySpec.

Provides:
- BacktestResult: dataclass holding all performance metrics and raw series
- run_backtest: run a single strategy spec against price data
- run_batch: run multiple specs and collect results, tolerating per-strategy errors
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bt
import pandas as pd

from engine.adapter import build_bt_strategy
from engine.types import StrategySpec

# bt backtests run GROSS of costs. The paper broker's per-asset-class fee model
# (engine.fees.fee_for) is keyed on the ticker to pick the asset class, but
# bt's ``commissions(quantity, price)`` hook does not pass the ticker — so a
# faithful mapping of fee_for into bt requires a non-trivial signature
# adaptation (a per-ticker commission wrapper bound at Backtest construction).
# That full wiring is DEFERRED; until then every bt result is gross of fees and
# callers surface this warning. See METHODOLOGY.md (Costs).
GROSS_OF_COSTS_WARNING = (
    "GROSS_OF_COSTS: backtest returns do not model brokerage fees. The paper "
    "broker applies a per-asset-class fee model on live paper fills, but bt "
    "backtests here are gross of costs — read returns accordingly."
)


# ---------------------------------------------------------------------------
# Key normalisation helpers
# ---------------------------------------------------------------------------

_RULES_CAMEL_TO_SNAKE: dict[str, str] = {
    "maxPositions": "max_positions",
    "maxPositionPct": "max_position_pct",
    "minHoldDays": "min_hold_days",
}


def _normalise_spec_dict(spec_dict: dict) -> dict:
    """Return a copy of *spec_dict* with camelCase rules keys converted to snake_case.

    StrategySpec.from_dict expects snake_case keys in the ``rules`` sub-dict, but
    callers (e.g. the frontend) may send camelCase.  Both forms are accepted; any
    unknown keys are passed through unchanged so future additions don't silently break.
    """
    result = dict(spec_dict)
    if "rules" in result:
        raw_rules = dict(result["rules"])
        normalised_rules = {
            _RULES_CAMEL_TO_SNAKE.get(k, k): v for k, v in raw_rules.items()
        }
        result["rules"] = normalised_rules
    return result


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """All outputs from a single backtest run."""

    strategy_id: str
    strategy_name: str
    total_return: float  # e.g. 0.15 for 15 %
    cagr: float  # Compound annual growth rate
    sharpe: float  # Daily Sharpe ratio
    max_drawdown: float  # e.g. -0.12 for -12 %
    daily_values: pd.Series  # Daily portfolio values (dates as index)
    transactions: pd.DataFrame | None  # Trade log from bt (may be None)

    def to_snapshots(self) -> list[dict]:
        """Convert daily portfolio values to a list of snapshot dicts.

        Each dict has:
            ``date``           — ISO-8601 date string (YYYY-MM-DD)
            ``portfolioValue`` — portfolio value as a float
        """
        return [
            {"date": idx.date().isoformat(), "portfolioValue": float(val)}
            for idx, val in self.daily_values.items()
        ]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_backtest(
    spec_dict: dict,
    price_data: pd.DataFrame,
    initial_capital: float | None = None,
) -> BacktestResult:
    """Run a single backtest from a strategy spec dictionary.

    Parameters
    ----------
    spec_dict:
        Strategy specification as a plain dict.  Both snake_case and camelCase
        keys are accepted in the ``rules`` sub-dict.
    price_data:
        DataFrame of daily close prices (columns = tickers, index = dates).
    initial_capital:
        Starting capital.  Falls back to ``spec.funding.initial`` when omitted.

    Returns
    -------
    BacktestResult
        Populated with performance metrics and the raw daily value series.
    """
    normalised = _normalise_spec_dict(spec_dict)
    spec = StrategySpec.from_dict(normalised)

    capital = initial_capital if initial_capital is not None else spec.funding.initial

    strategy = build_bt_strategy(spec, price_data)
    backtest = bt.Backtest(strategy, price_data, initial_capital=capital)
    result = bt.run(backtest)

    stats = result.stats
    strategy_name = spec.id  # bt uses the strategy id as the column name

    total_return = float(stats.loc["total_return", strategy_name])
    cagr = float(stats.loc["cagr", strategy_name])
    sharpe = float(stats.loc["daily_sharpe", strategy_name])
    max_drawdown = float(stats.loc["max_drawdown", strategy_name])

    # result[strategy_name].prices is an index normalised to 100; use
    # strategy.values for absolute portfolio values in the reporting currency.
    daily_values: pd.Series = result.backtests[strategy_name].strategy.values

    try:
        transactions: pd.DataFrame | None = result.get_transactions(strategy_name)
    except Exception:
        transactions = None

    return BacktestResult(
        strategy_id=spec.id,
        strategy_name=spec.name,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        daily_values=daily_values,
        transactions=transactions,
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_batch(
    spec_dicts: list[dict],
    price_data: pd.DataFrame,
) -> list[BacktestResult]:
    """Run multiple strategy specs against the same price data.

    Failures are caught per-strategy: a strategy that raises will be skipped
    with an error message printed to stdout, and the rest continue.

    Parameters
    ----------
    spec_dicts:
        List of strategy spec dicts.
    price_data:
        Shared price DataFrame for all strategies.

    Returns
    -------
    list[BacktestResult]
        One entry per successfully completed strategy (order preserved).
    """
    results: list[BacktestResult] = []
    for spec_dict in spec_dicts:
        strategy_id = spec_dict.get("id", "<unknown>")
        try:
            results.append(run_backtest(spec_dict, price_data))
        except Exception as exc:
            print(f"[run_batch] Strategy {strategy_id!r} failed: {exc}")
    return results
