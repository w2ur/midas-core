"""Deterministic baseline-manager — Gate C benchmark portfolio.

Internal comparison portfolio only. NOT a public trading agent.
Excluded from roster.ts, the leaderboard, posts, journals, and the public
output bundle by the ``trading_roster`` role filter (role != trader).

Every parameter below is resolved per-allocator by
``MidasConfig.baseline_params`` and passed in; the module-level constants are
the LIVE DESK's pinned values, kept as defaults so existing callers and tests
are unchanged. They are not the module's opinion. A desk running two
allocators gets two control books, and a non-EUR desk gets its own currency —
without that, the book carries an uncontrolled FX leg in the exact comparison
it exists to control for.

Rules (live-desk values shown; all are configurable)
-----
- Capital: EUR 2,000 virtual, initialized on first run.
- Signal: collect tickers whose research note has action_bias in
  {"strong_buy", "buy"}. Eligible if ≥2 distinct agents mark it that day.
- Ranking: count desc → strong_buy-weight desc (strong_buy=2, buy=1) →
  ticker alphabetical (deterministic tie-break).
- Target: top 6 from the ranked eligible set.
- Sizing: equal-weight EUR 300/position (6 × 300 = 1800, leaving ~200 buffer).
  Held target positions are not trimmed or topped-up: equal-weight applies at
  entry only; subsequent per-position drift is left intentionally to minimize
  churn and fees.
- Cadence: rebalance ONLY on the first weekday (Mon-Fri) of each calendar month,
  OR on the very first run (portfolio has never been rebalanced / does not exist).
- Rebalance: SELL positions not in target; BUY each target up to ~EUR 300 at
  the day's close price (fractional shares allowed). Fees applied via fee_for.
- Skip tickers with no price in the store — log and exclude.

The public roster (get_config().trading_roster) is unchanged — this book is
never added to it. The baseline-manager directory is iterated by
step_update_snapshots (intentional — daily valuation snapshots are desired).
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Callable

from engine.config import get_config
from engine.fees import fee_for
from engine.research_note import ResearchNote
from engine.types import Trade

logger = logging.getLogger(__name__)

STRATEGY_ID = "baseline-manager"
INITIAL_CAPITAL_EUR = 2000.0
POSITION_SIZE_EUR = 300.0
MAX_POSITIONS = 6
_BUY_SIGNALS = frozenset({"strong_buy", "buy"})
_STRONG_BUY_WEIGHT = 2
_BUY_WEIGHT = 1
_REASONING = "baseline-manager monthly rebalance: ≥2-agent buy consensus"


# ---------------------------------------------------------------------------
# Pure functions (I/O-free, fully unit-testable)
# ---------------------------------------------------------------------------


def eligible_tickers(
    notes: list[tuple[str, ResearchNote]],
    max_positions: int = MAX_POSITIONS,
) -> list[str]:
    """Compute ranked eligible tickers from the day's research notes.

    Eligibility: action_bias in {"strong_buy", "buy"} AND ≥2 distinct agents
    mark the ticker.

    Ranking (desc priority):
      1. count of distinct agents marking the ticker.
      2. strong_buy-weight sum (strong_buy=2, buy=1) across distinct agents.
      3. ticker alphabetical (ascending) — deterministic tie-break.

    Returns at most ``max_positions`` tickers (live desk: 6).
    """
    # Per-ticker per-agent tracking.
    # ticker → {agent_id → best_weight_for_that_agent}
    ticker_agent_weights: dict[str, dict[str, int]] = {}

    for agent_id, note in notes:
        if note.action_bias not in _BUY_SIGNALS:
            continue
        weight = _STRONG_BUY_WEIGHT if note.action_bias == "strong_buy" else _BUY_WEIGHT
        for ticker in note.tickers:
            if ticker not in ticker_agent_weights:
                ticker_agent_weights[ticker] = {}
            # Take the max weight if the same agent emits multiple notes for the ticker.
            existing = ticker_agent_weights[ticker].get(agent_id, 0)
            ticker_agent_weights[ticker][agent_id] = max(existing, weight)

    # Filter to tickers with ≥2 distinct agents.
    eligible: list[tuple[str, int, int]] = []  # (ticker, count, total_weight)
    for ticker, agent_weights in ticker_agent_weights.items():
        count = len(agent_weights)
        if count < 2:
            continue
        total_weight = sum(agent_weights.values())
        eligible.append((ticker, count, total_weight))

    # Sort: count desc, weight desc, ticker asc.
    eligible.sort(key=lambda x: (-x[1], -x[2], x[0]))

    return [ticker for ticker, _, _ in eligible[:max_positions]]


def is_rebalance_day(d: date) -> bool:
    """Return True if d is the first weekday (Mon-Fri) of its calendar month."""
    # Find the first weekday of the month.
    first = date(d.year, d.month, 1)
    # weekday(): Mon=0, Tue=1, ..., Fri=4, Sat=5, Sun=6
    offset = (7 - first.weekday()) % 7 if first.weekday() >= 5 else 0
    first_weekday = date(d.year, d.month, 1 + offset)
    return d == first_weekday


def rebalance(
    portfolio: dict,
    target_tickers: list[str],
    price_lookup: Callable[[str, date], float | None],
    on: date,
    position_size: float = POSITION_SIZE_EUR,
    reasoning: str = _REASONING,
    max_positions: int = MAX_POSITIONS,
) -> list[Trade]:
    """Compute the list of Trades needed to reach the target allocation.

    Parameters
    ----------
    portfolio:
        Dict with "cash" (float) and "positions" (list of dicts with
        "ticker", "shares", "avg_cost").
    target_tickers:
        Ordered list of eligible tickers (already capped at ``max_positions``).
    price_lookup:
        Callable(ticker, date) → float | None. Returns None when no price.
    on:
        The rebalance date (used for price lookup and trade timestamps).
    position_size:
        Target notional per position, in the control book's own currency
        (which is the allocator's ``home_currency``, not necessarily EUR).
    max_positions:
        Cap on concurrent positions; defaults to the live desk's 6.
    reasoning:
        Reasoning string applied to every trade.

    Returns
    -------
    list[Trade]
        Trades in execution order: SELLs first (to free cash), then BUYs.
        Held target positions are not trimmed or topped-up: equal-weight applies
        at entry only; subsequent per-position drift is left intentionally to
        minimize churn and fees. No I/O — caller is responsible for applying
        trades via PortfolioManager.
    """
    now = datetime.now(timezone.utc)
    existing_positions: dict[str, float] = {
        p["ticker"]: p["shares"] for p in portfolio.get("positions", [])
    }
    target_set = set(target_tickers)

    # Tickers in target with a valid price.
    priced_targets: list[str] = []
    for ticker in target_tickers:
        price = price_lookup(ticker, on)
        if price is None:
            logger.warning(
                "baseline-manager: no price for %s on %s — excluding from target",
                ticker,
                on,
            )
        else:
            priced_targets.append(ticker)

    priced_target_set = set(priced_targets)

    trades: list[Trade] = []

    # --- SELLs: positions not in the priced target set ---
    for ticker, shares in existing_positions.items():
        if ticker in priced_target_set:
            continue
        price = price_lookup(ticker, on)
        if price is None:
            logger.warning(
                "baseline-manager: no price for held position %s on %s — cannot sell, skipping",
                ticker,
                on,
            )
            continue
        notional = shares * price
        fee = fee_for(ticker, notional)
        trades.append(
            Trade(
                id=f"bm_{on.isoformat()}_{uuid.uuid4().hex[:8]}",
                timestamp=now,
                action="SELL",
                ticker=ticker,
                shares=shares,
                price=price,
                total=notional,
                fees=fee,
                reasoning=reasoning,
            )
        )

    # --- BUYs: priced targets not already held ---
    for ticker in priced_targets:
        if ticker in existing_positions:
            # Already holds a position — do not re-buy.
            continue
        price = price_lookup(ticker, on)
        if price is None:
            continue
        shares = position_size / price
        notional = shares * price  # == position_size (floating-point safe)
        fee = fee_for(ticker, notional)
        trades.append(
            Trade(
                id=f"bm_{on.isoformat()}_{uuid.uuid4().hex[:8]}",
                timestamp=now,
                action="BUY",
                ticker=ticker,
                shares=shares,
                price=price,
                total=notional,
                fees=fee,
                reasoning=reasoning,
            )
        )

    return trades


def __getattr__(name: str) -> object:
    """Lazily expose ``_OHLCV_STORE`` as the current config's OHLCV dir (PEP 562).

    ``scripts.daily_session.step_build_baseline_manager`` reads
    ``engine.baseline_manager._OHLCV_STORE`` as the default store for its phantom
    fills. Resolving it through ``get_config()`` at access time keeps the
    Hands-side baseline manager honouring MIDAS_DATA_DIR redirection — nothing is
    frozen at import.
    """
    if name == "_OHLCV_STORE":
        return get_config().ohlcv_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
