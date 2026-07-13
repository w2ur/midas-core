"""Tests for engine/manager_orders.py — manager_decision_to_orders conversion.

TDD: tests written before implementation to drive the design.
"""

from __future__ import annotations

from datetime import date

from engine.manager_decision import ManagerDecision, ManagerPosition
from engine.manager_orders import manager_decision_to_orders


def _make_position(
    ticker: str = "BTC-EUR",
    action: str = "BUY",
    size_eur: int = 300,
    reasoning: str = "Strong momentum.",
    trigger: dict | None = None,
    expires: str | None = None,
) -> ManagerPosition:
    return ManagerPosition(
        ticker=ticker,
        action=action,
        size_eur=size_eur,
        entry_guidance="",
        stop_loss=None,
        reasoning=reasoning,
        trigger=trigger,
        expires=expires,
    )


def _make_decision(
    positions: list[ManagerPosition], conviction: int = 8
) -> ManagerDecision:
    return ManagerDecision(
        positions=positions,
        conviction=conviction,
        hold_reasoning="",
    )


def _price_lookup(ticker: str) -> float | None:
    prices = {
        "BTC-EUR": 30000.0,
        "PHAG.L": 60.0,
        "ETH-EUR": 1500.0,
    }
    return prices.get(ticker)


# ---------------------------------------------------------------------------
# Basic market order (existing behavior, must not regress)
# ---------------------------------------------------------------------------


class TestBasicMarketOrders:
    def test_buy_produces_single_order(self) -> None:
        decision = _make_decision([_make_position("BTC-EUR", "BUY", 300)])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert len(orders) == 1
        order = orders[0]
        assert order.action == "BUY"
        assert order.ticker == "BTC-EUR"
        assert order.trigger is None
        assert order.expires is None

    def test_hold_position_is_skipped(self) -> None:
        decision = _make_decision([_make_position("BTC-EUR", "HOLD", 0, "Waiting.")])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert orders == []

    def test_no_price_position_is_skipped(self) -> None:
        decision = _make_decision([_make_position("UNKNOWN", "BUY", 300)])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert orders == []

    def test_shares_calculated_from_size_eur(self) -> None:
        decision = _make_decision([_make_position("PHAG.L", "BUY", 300)])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert len(orders) == 1
        assert abs(orders[0].shares - (300 / 60.0)) < 1e-9


# ---------------------------------------------------------------------------
# Trigger orders — Task 5
# ---------------------------------------------------------------------------


class TestTriggerOrders:
    def test_decision_to_orders_emits_trigger(self) -> None:
        """ManagerPosition with trigger/expires → Order.trigger/Order.expires populated."""
        pos = _make_position(
            "PHAG.L",
            "BUY",
            300,
            trigger={"op": ">=", "level": 65.0},
            expires="2026-07-15",
        )
        decision = _make_decision([pos])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert len(orders) == 1
        order = orders[0]
        assert order.trigger == {"op": ">=", "level": 65.0}
        assert order.expires == "2026-07-15"

    def test_lte_trigger_emitted(self) -> None:
        """op '<=' trigger is also passed through."""
        pos = _make_position(
            "BTC-EUR",
            "BUY",
            300,
            trigger={"op": "<=", "level": 25000.0},
            expires="2026-07-10",
        )
        decision = _make_decision([pos])
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert len(orders) == 1
        assert orders[0].trigger == {"op": "<=", "level": 25000.0}
        assert orders[0].expires == "2026-07-10"

    def test_market_and_trigger_orders_coexist(self) -> None:
        """Two positions — one market, one trigger — both produce orders."""
        positions = [
            _make_position("BTC-EUR", "BUY", 300, "Market entry."),
            _make_position(
                "PHAG.L",
                "BUY",
                200,
                "Conditional entry.",
                trigger={"op": ">=", "level": 65.0},
                expires="2026-07-15",
            ),
        ]
        decision = _make_decision(positions)
        orders = manager_decision_to_orders(decision, date(2026, 6, 28), _price_lookup)
        assert len(orders) == 2
        market_orders = [o for o in orders if o.trigger is None]
        trigger_orders = [o for o in orders if o.trigger is not None]
        assert len(market_orders) == 1
        assert len(trigger_orders) == 1
        assert market_orders[0].ticker == "BTC-EUR"
        assert trigger_orders[0].ticker == "PHAG.L"
