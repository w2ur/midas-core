"""Tests for the Task 8 additions to scripts.daily_session:
- step_author_orders forwards trigger/expires
- step_author_cancels writes CancelRequest records
- render_active_triggers_for_agent renders pending orders for the prompt
- CONDITIONAL_ORDER_INSTRUCTIONS is non-empty and documents the schema
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.orders import Order, read_outbox
from engine.triggers import list_pending, read_cancels, save_pending
from engine.portfolio import PortfolioManager
from scripts.daily_session import (
    CONDITIONAL_ORDER_INSTRUCTIONS,
    render_active_triggers_for_agent,
    step_author_all,
    step_author_cancels,
    step_author_orders,
)


class TestAuthorConditionalOrders:
    def test_market_order_unchanged(self, midas_data_root) -> None:
        d = date(2026, 5, 17)
        n = step_author_orders(
            "satoshi",
            trades=[
                {
                    "action": "BUY",
                    "ticker": "BTC-EUR",
                    "shares": 0.01,
                    "reasoning": "dip",
                }
            ],
            trade_date=d,
            currency="EUR",
        )
        assert n == 1
        out = read_outbox(d)
        assert out[0].trigger is None
        assert out[0].expires is None

    def test_conditional_order_persists_trigger_and_expires(
        self, midas_data_root
    ) -> None:
        d = date(2026, 5, 17)
        step_author_orders(
            "satoshi",
            trades=[
                {
                    "action": "SELL",
                    "ticker": "BTC-EUR",
                    "shares": 0.01,
                    "reasoning": "trim at 85k",
                    "trigger": {"op": ">=", "level": 85000.0},
                    "expires": "2026-06-17",
                }
            ],
            trade_date=d,
            currency="EUR",
        )
        out = read_outbox(d)
        assert out[0].trigger == {"op": ">=", "level": 85000.0}
        assert out[0].expires == "2026-06-17"


class TestAuthorCancels:
    def test_cancel_written_to_cancels_dir(self, midas_data_root) -> None:
        d = date(2026, 5, 17)
        n = step_author_cancels(
            "satoshi",
            cancels=[
                {
                    "target_order_id": "ord_2026-05-10_satoshi_003",
                    "reasoning": "thesis changed",
                }
            ],
            trade_date=d,
        )
        assert n == 1
        cancels = read_cancels(d)
        assert len(cancels) == 1
        assert cancels[0].target_order_id == "ord_2026-05-10_satoshi_003"
        assert cancels[0].agent_id == "satoshi"

    def test_empty_cancels_is_no_op(self, midas_data_root) -> None:
        d = date(2026, 5, 17)
        assert step_author_cancels("satoshi", cancels=[], trade_date=d) == 0
        assert read_cancels(d) == []


class TestRenderActiveTriggers:
    def test_no_triggers_returns_friendly_empty_string(self, midas_data_root) -> None:
        out = render_active_triggers_for_agent("satoshi")
        assert "no active triggers" in out.lower()

    def test_renders_each_pending_order_for_agent(self, midas_data_root) -> None:
        save_pending(
            Order(
                order_id="ord_2026-05-10_satoshi_003",
                ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="SELL",
                ticker="BTC-EUR",
                shares=0.01,
                reasoning="trim",
                currency="EUR",
                trigger={"op": ">=", "level": 85000.0},
                expires="2026-06-17",
            )
        )
        save_pending(
            Order(
                order_id="ord_2026-05-12_satoshi_001",
                ts=datetime(2026, 5, 12, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="BUY",
                ticker="ETH-EUR",
                shares=0.5,
                reasoning="buy dip",
                currency="EUR",
                trigger={"op": "<=", "level": 2800.0},
                expires="2026-06-01",
            )
        )
        out = render_active_triggers_for_agent("satoshi")
        assert "ord_2026-05-10_satoshi_003" in out
        assert "BTC-EUR" in out
        assert "85000" in out
        assert "2026-06-17" in out
        assert "ord_2026-05-12_satoshi_001" in out
        assert "ETH-EUR" in out

    def test_renders_only_orders_for_requested_agent(self, midas_data_root) -> None:
        save_pending(
            Order(
                order_id="ord_satoshi_x",
                ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="SELL",
                ticker="BTC-EUR",
                shares=0.01,
                reasoning="trim",
                currency="EUR",
                trigger={"op": ">=", "level": 85000.0},
                expires="2026-06-17",
            )
        )
        save_pending(
            Order(
                order_id="ord_world_y",
                ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
                agent_id="world",
                action="BUY",
                ticker="MSFT",
                shares=1,
                reasoning="test",
                currency="EUR",
                trigger={"op": "<=", "level": 400.0},
                expires="2026-06-17",
            )
        )
        satoshi_view = render_active_triggers_for_agent("satoshi")
        assert "BTC-EUR" in satoshi_view
        assert "MSFT" not in satoshi_view


class TestConditionalOrderInstructions:
    def test_instructions_constant_is_nonempty_and_documents_schema(self) -> None:
        assert "trigger" in CONDITIONAL_ORDER_INSTRUCTIONS
        assert "expires" in CONDITIONAL_ORDER_INSTRUCTIONS
        assert (
            '">="' in CONDITIONAL_ORDER_INSTRUCTIONS
            or "'>='" in CONDITIONAL_ORDER_INSTRUCTIONS
            or ">=" in CONDITIONAL_ORDER_INSTRUCTIONS
        )
        assert "cancels" in CONDITIONAL_ORDER_INSTRUCTIONS.lower()


class TestAuthorAll:
    def test_authors_orders_and_cancels_for_multiple_agents(
        self, tmp_path, midas_data_root
    ) -> None:
        pm_base = tmp_path / "portfolios"
        pm = PortfolioManager(pm_base)
        pm.initialize("satoshi", initial_capital=10_000.0, currency="EUR")
        pm.initialize("world", initial_capital=10_000.0, currency="EUR")

        d = date(2026, 5, 20)
        agent_results = {
            "satoshi": {
                "commentary": "...",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "BTC-EUR",
                        "shares": 0.01,
                        "reasoning": "dip",
                    },
                    {
                        "action": "SELL",
                        "ticker": "ETH-EUR",
                        "shares": 0.5,
                        "reasoning": "trim at 4k",
                        "trigger": {"op": ">=", "level": 4000.0},
                        "expires": "2026-06-20",
                    },
                ],
                "cancels": [
                    {"target_order_id": "ord_old_001", "reasoning": "stale thesis"},
                ],
            },
            "world": {
                "commentary": "...",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "MSFT",
                        "shares": 2,
                        "reasoning": "earnings",
                    },
                ],
            },
        }
        summary = step_author_all(agent_results, d, portfolio_manager=pm)

        assert summary == {
            "satoshi": {"orders": 2, "cancels": 1},
            "world": {"orders": 1, "cancels": 0},
        }
        out = read_outbox(d)
        assert len(out) == 3
        cancels = read_cancels(d)
        assert len(cancels) == 1
        assert cancels[0].target_order_id == "ord_old_001"

    def test_missing_trades_and_cancels_keys_treated_as_empty(
        self, tmp_path, midas_data_root
    ) -> None:
        pm_base = tmp_path / "portfolios"
        pm = PortfolioManager(pm_base)
        pm.initialize("satoshi", initial_capital=10_000.0, currency="EUR")

        d = date(2026, 5, 20)
        summary = step_author_all(
            {"satoshi": {"commentary": "..."}}, d, portfolio_manager=pm
        )

        assert summary == {"satoshi": {"orders": 0, "cancels": 0}}
        assert read_outbox(d) == []
        assert read_cancels(d) == []
