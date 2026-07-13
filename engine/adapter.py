"""Strategy adapter — converts StrategySpec into bt.Strategy pipeline.

This is the bridge between JSON strategy specs and bt's backtesting engine.
It uses a registry pattern: string IDs map to factory functions that produce
lists of bt.Algo objects for the selector and manager stages of the pipeline.

Pipeline structure:
    RunDaily → [selector algos] → SelectN → [manager algos] → LimitWeights → Rebalance
"""

from __future__ import annotations

from typing import Callable

import bt
import pandas as pd
import pandas_ta as ta

from engine.types import StrategySpec

# ---------------------------------------------------------------------------
# Type aliases for factory functions
# ---------------------------------------------------------------------------

SelectorFactory = Callable[[StrategySpec, pd.DataFrame], list[bt.Algo]]
ManagerFactory = Callable[[StrategySpec, pd.DataFrame], list[bt.Algo]]

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

SELECTOR_REGISTRY: dict[str, SelectorFactory] = {}
MANAGER_REGISTRY: dict[str, ManagerFactory] = {}


def register_selector(name: str) -> Callable[[SelectorFactory], SelectorFactory]:
    """Decorator to register a selector factory under *name*."""

    def decorator(fn: SelectorFactory) -> SelectorFactory:
        SELECTOR_REGISTRY[name] = fn
        return fn

    return decorator


def register_manager(name: str) -> Callable[[ManagerFactory], ManagerFactory]:
    """Decorator to register a manager factory under *name*."""

    def decorator(fn: ManagerFactory) -> ManagerFactory:
        MANAGER_REGISTRY[name] = fn
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_bt_strategy(spec: StrategySpec, price_data: pd.DataFrame) -> bt.Strategy:
    """Build a bt.Strategy from a StrategySpec and price data.

    Raises ValueError if the selector or manager is not registered.
    """
    if spec.selector not in SELECTOR_REGISTRY:
        raise ValueError(
            f"No selector registered for {spec.selector!r}. "
            f"Available: {sorted(SELECTOR_REGISTRY)}"
        )
    if spec.manager not in MANAGER_REGISTRY:
        raise ValueError(
            f"No manager registered for {spec.manager!r}. "
            f"Available: {sorted(MANAGER_REGISTRY)}"
        )

    selector_algos = SELECTOR_REGISTRY[spec.selector](spec, price_data)
    manager_algos = MANAGER_REGISTRY[spec.manager](spec, price_data)

    max_weight = spec.rules.max_position_pct / 100.0

    # If the selector doesn't set temp['stat'] (most signal-based ones don't),
    # add StatTotalReturn so SelectN can rank and cap positions.
    needs_stat = not any(
        isinstance(a, bt.algos.StatTotalReturn) for a in selector_algos
    )

    pipeline: list[bt.Algo] = [
        bt.algos.RunDaily(),
        *selector_algos,
    ]
    if needs_stat:
        pipeline.append(bt.algos.StatTotalReturn(lookback=pd.DateOffset(months=1)))
    pipeline.extend(
        [
            bt.algos.SelectN(spec.rules.max_positions),
            *manager_algos,
            bt.algos.LimitWeights(max_weight),
            bt.algos.Rebalance(),
        ]
    )

    return bt.Strategy(spec.id, pipeline)


# =========================================================================
# Built-in selectors
# =========================================================================


@register_selector("random")
def _selector_random(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """Reproducible random selection.

    bt.algos.SelectRandomly reads numpy's global RNG, so factor-research runs
    were not reproducible run-to-run. Seed deterministically from
    (strategy id, window start) via the same SelectRandomlySeeded the coin-flip
    baselines already use.
    """
    from engine.selectors.random_seeded import SelectRandomlySeeded, make_seed

    if len(price_data.index) > 0:
        start_iso = price_data.index.min().date().isoformat()
    else:
        start_iso = "1970-01-01"
    seed = make_seed(spec.id, start_iso)
    return [SelectRandomlySeeded(n=spec.rules.max_positions, seed=seed)]


@register_selector("golden-cross")
def _selector_golden_cross(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """50-day MA crosses above 200-day MA."""
    ma_fast = price_data.rolling(50).mean()
    ma_slow = price_data.rolling(200).mean()
    signal = ma_fast > ma_slow
    return [bt.algos.SelectWhere(signal)]


@register_selector("rsi-oversold")
def _selector_rsi_oversold(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """RSI < 30 signals oversold entry."""
    rsi_frames = {}
    for col in price_data.columns:
        rsi_series = ta.rsi(price_data[col], length=14)
        rsi_frames[col] = rsi_series
    rsi_df = pd.DataFrame(rsi_frames, index=price_data.index)
    signal = rsi_df < 30
    return [bt.algos.SelectWhere(signal)]


@register_selector("dip-entry")
def _selector_dip_entry(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """Price is 5%+ below 20-day rolling high."""
    rolling_high = price_data.rolling(20).max()
    signal = price_data <= rolling_high * 0.95
    return [bt.algos.SelectWhere(signal)]


@register_selector("fear-greed")
def _selector_fear_greed(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """Rolling 20-day volatility > 0.25 (annualized) as fear proxy."""
    daily_returns = price_data.pct_change()
    rolling_vol = daily_returns.rolling(20).std() * (252**0.5)
    signal = rolling_vol > 0.25
    return [bt.algos.SelectWhere(signal)]


@register_selector("data-follow")
def _selector_data_follow(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Placeholder — selects all available securities."""
    return [bt.algos.SelectAll()]


@register_selector("buy-and-hold")
def _selector_buy_and_hold(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Select all tickers — buy and hold everything."""
    return [bt.algos.SelectAll()]


@register_selector("earnings-beat")
def _selector_earnings_beat(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Placeholder — momentum proxy using total return ranking."""
    return [
        bt.algos.SelectAll(),
        bt.algos.StatTotalReturn(lookback=pd.DateOffset(months=3)),
    ]


@register_selector("sector-cycle")
def _selector_sector_cycle(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Sector rotation using 3-month total return ranking."""
    return [
        bt.algos.SelectAll(),
        bt.algos.StatTotalReturn(lookback=pd.DateOffset(months=3)),
    ]


# =========================================================================
# Built-in managers
# =========================================================================


@register_manager("equal-weight")
def _manager_equal_weight(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    return [bt.algos.WeighEqually()]


# NOTE: the five managers below (grid-conservative, trailing-stop, scaled-exit,
# time-boxed, rebalance-monthly) are ALIASES for equal-weight — their distinct
# position-management behavior is NOT yet implemented. They are kept registered
# (rather than raising) only because committed strategy specs and the backtester
# API still reference them; removing them requires rewriting those specs. The
# default factor-research grid no longer advertises them (see run_all_combos
# _DEFAULT_MANAGERS), and the README axis list marks them as aspirational.
# Making them raise instead of silently equal-weighting is DEFERRED.
@register_manager("grid-conservative")
def _manager_grid_conservative(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """ALIAS for equal-weight — grid scaling is not implemented."""
    return [bt.algos.WeighEqually()]


@register_manager("grid-aggressive")
def _manager_grid_aggressive(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Inverse volatility weighting with 1-month lookback."""
    return [bt.algos.WeighInvVol(lookback=pd.DateOffset(months=1))]


@register_manager("trailing-stop")
def _manager_trailing_stop(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """ALIAS for equal-weight — trailing-stop exits are not implemented."""
    return [bt.algos.WeighEqually()]


@register_manager("scaled-exit")
def _manager_scaled_exit(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """ALIAS for equal-weight — scaled exits are not implemented."""
    return [bt.algos.WeighEqually()]


@register_manager("time-boxed")
def _manager_time_boxed(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """ALIAS for equal-weight — time-boxed holding is not implemented."""
    return [bt.algos.WeighEqually()]


@register_manager("rebalance-monthly")
def _manager_rebalance_monthly(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """ALIAS for equal-weight — monthly-only rebalance is not implemented."""
    return [bt.algos.WeighEqually()]


@register_manager("volatility-sized")
def _manager_volatility_sized(
    spec: StrategySpec, price_data: pd.DataFrame
) -> list[bt.Algo]:
    """Inverse volatility weighting with 3-month lookback."""
    return [bt.algos.WeighInvVol(lookback=pd.DateOffset(months=3))]


@register_manager("fixed-60-40")
def _manager_fixed_60_40(spec: StrategySpec, price_data: pd.DataFrame) -> list[bt.Algo]:
    """Fixed 60/40 allocation: 60% VOO, 40% BND."""
    return [bt.algos.WeighSpecified(VOO=0.6, BND=0.4)]
