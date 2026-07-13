"""Manager decision → orders conversion (Task C5).

Bridges the LLM Manager's structured ManagerDecision (C4) to the broker's Order
primitive. Pure and deterministic: no I/O, no LLM, no network. The session step
(scripts.daily_session.step_apply_manager_decision) is responsible for writing
the resulting orders to the SEPARATE manager-outbox and filling them into the
the-manager book.

Sizing mirrors engine.baseline_manager: shares = size_eur / close_price (fractional
shares allowed). Positions with action HOLD, or with no store price, are skipped.

Conditional (trigger) orders are supported: when a ManagerPosition carries a
trigger+expires, the resulting Order carries them too and the paper broker routes
it to the pending channel instead of filling immediately (same watcher path as
agent trigger orders, Task 5).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Callable

from engine.manager_decision import ManagerDecision
from engine.orders import Order, make_order_id

logger = logging.getLogger(__name__)

# the-manager runs a EUR book (Task C5 / real-money mandate: French tax resident).
MANAGER_AGENT_ID = "the-manager"
MANAGER_CURRENCY = "EUR"
# Intentionally matches the baseline-manager book size for a fair Gate C comparison.
# NOTE: MANAGER_INITIAL_CAPITAL_EUR is no longer used by the session — Task 5
# sources initial capital from AllocatorSpec.initial_capital (roster.yaml) via
# the step helpers.  This constant is retained as a legacy default only; removing
# it would be a breaking change for any callers that import it directly.
MANAGER_INITIAL_CAPITAL_EUR = 2000.0


def manager_decision_to_orders(
    decision: ManagerDecision,
    trade_date: date,
    price_lookup: Callable[[str], float | None],
    *,
    agent_id: str = MANAGER_AGENT_ID,
    currency: str = MANAGER_CURRENCY,
) -> list[Order]:
    """Convert a ManagerDecision into a list of broker Orders.

    Parameters
    ----------
    decision:
        The parsed (and conviction-gated) ManagerDecision. HOLD positions are
        skipped; only BUY/SELL produce orders.
    trade_date:
        The session's trading date. Used for deterministic order_ids and the
        order timestamp's date.
    price_lookup:
        Callable(ticker) -> close_price | None. Positions whose ticker has no
        store price are skipped (logged) — the same de-risking discipline as
        engine.baseline_manager.rebalance.
    agent_id:
        The allocator agent id used for order_id generation and Order.agent_id.
        Defaults to MANAGER_AGENT_ID ("the-manager").
    currency:
        The booking currency for the orders. Defaults to MANAGER_CURRENCY ("EUR").

    Returns
    -------
    list[Order]
        One Order per BUY/SELL position with a known price, in input order.
        Order ids are deterministic (ord_{date}_{agent_id}_{seq:03d}) so
        re-running the conversion yields identical ids — the fill path is
        idempotent on them.
    """
    orders: list[Order] = []
    seq = 0
    ts = datetime.now(timezone.utc)
    for pos in decision.positions:
        if pos.action == "HOLD":
            continue

        price = price_lookup(pos.ticker)
        if price is None:
            logger.warning(
                "manager_decision_to_orders: no store price for %s — skipping %s order",
                pos.ticker,
                pos.action,
            )
            continue
        if price <= 0:
            logger.warning(
                "manager_decision_to_orders: non-positive price %s for %s — skipping",
                price,
                pos.ticker,
            )
            continue

        shares = pos.size_eur / price
        if not (shares > 0):
            # size_eur == 0 cannot produce a tradable order (Order requires shares > 0).
            logger.warning(
                "manager_decision_to_orders: size_eur %d for %s yields no shares — skipping",
                pos.size_eur,
                pos.ticker,
            )
            continue

        seq += 1
        order = Order(
            order_id=make_order_id(trade_date, agent_id, seq),
            ts=ts,
            agent_id=agent_id,
            action=pos.action,
            ticker=pos.ticker,
            shares=shares,
            reasoning=pos.reasoning,
            currency=currency,
            trigger=pos.trigger,
            expires=pos.expires,
        )
        orders.append(order)

    return orders
