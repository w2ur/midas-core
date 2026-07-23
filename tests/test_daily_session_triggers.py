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

from engine.orders import Order, read_dropped, read_outbox
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
        assert len(n) == 1
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


class TestAuthorOrderNormalization:
    """Agents emit loose output (lowercase actions, HOLD/invalid pseudo-trades);
    an unattended session must not crash at Order construction (2026-07-17), must
    audit every drop, and must return only the authored trades so narration never
    references a phantom fill."""

    def test_lowercase_action_is_normalized(self, midas_data_root) -> None:
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                {
                    "action": "buy",
                    "ticker": "BTC-EUR",
                    "shares": 0.01,
                    "reasoning": "dip",
                },
                {
                    "action": "Sell",
                    "ticker": "ETH-EUR",
                    "shares": 0.5,
                    "reasoning": "trim",
                },
            ],
            trade_date=d,
            currency="EUR",
        )
        assert len(authored) == 2
        assert {o.action for o in read_outbox(d)} == {"BUY", "SELL"}
        assert read_dropped(d) == []

    def test_hold_and_non_tradeable_actions_are_dropped_and_audited(
        self, midas_data_root
    ) -> None:
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "monsieur-forex",
            trades=[
                {
                    "action": "HOLD",
                    "ticker": "EURUSD=X",
                    "shares": 1,
                    "reasoning": "wait",
                },
                {
                    "action": "buy",
                    "ticker": "EURUSD=X",
                    "shares": 1000,
                    "reasoning": "long",
                },
                {
                    "action": "",
                    "ticker": "GBPUSD=X",
                    "shares": 500,
                    "reasoning": "blank",
                },
            ],
            trade_date=d,
            currency="EUR",
        )
        # Only the buy is authored and returned; HOLD and blank are dropped.
        assert [t["reasoning"] for t in authored] == ["long"]
        assert [o.action for o in read_outbox(d)] == ["BUY"]
        dropped = read_dropped(d)
        assert [r.reason for r in dropped] == [
            "NON_TRADEABLE_ACTION",
            "NON_TRADEABLE_ACTION",
        ]
        assert dropped[0].raw["action"] == "HOLD"  # raw preserved verbatim

    def test_nonpositive_and_nonfinite_shares_dropped_not_crashed(
        self, midas_data_root
    ) -> None:
        # shares<=0/NaN/inf would raise in Order.__post_init__; must be audited.
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                {
                    "action": "BUY",
                    "ticker": "BTC-EUR",
                    "shares": 0,
                    "reasoning": "zero",
                },
                {
                    "action": "SELL",
                    "ticker": "ETH-EUR",
                    "shares": -3,
                    "reasoning": "neg",
                },
                {
                    "action": "BUY",
                    "ticker": "SOL-EUR",
                    "shares": "nan",
                    "reasoning": "nan",
                },
            ],
            trade_date=d,
            currency="EUR",
        )
        assert authored == []
        assert read_outbox(d) == []
        assert [r.reason for r in read_dropped(d)] == ["INVALID_SHARES"] * 3

    def test_missing_ticker_or_shares_is_dropped_and_audited(
        self, midas_data_root
    ) -> None:
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                {"action": "BUY", "shares": 1, "reasoning": "no ticker"},
                {"action": "BUY", "ticker": "BTC-EUR", "reasoning": "no shares"},
                {
                    "action": "BUY",
                    "ticker": "ETH-EUR",
                    "shares": "lots",
                    "reasoning": "bad",
                },
                {"action": "BUY", "ticker": "SOL-EUR", "shares": 2, "reasoning": "ok"},
            ],
            trade_date=d,
            currency="EUR",
        )
        assert len(authored) == 1
        assert [o.ticker for o in read_outbox(d)] == ["SOL-EUR"]
        assert [r.reason for r in read_dropped(d)] == [
            "MISSING_TICKER",
            "INVALID_SHARES",
            "INVALID_SHARES",
        ]

    def test_malformed_conditional_is_dropped_as_invalid_order(
        self, midas_data_root
    ) -> None:
        # expires without trigger raises in Order.__post_init__; the try/except
        # net must record it as INVALID_ORDER, not crash the session.
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                {
                    "action": "BUY",
                    "ticker": "BTC-EUR",
                    "shares": 1,
                    "reasoning": "bad conditional",
                    "expires": "2026-08-01",
                },
            ],
            trade_date=d,
            currency="EUR",
        )
        assert authored == []
        assert read_outbox(d) == []
        assert [r.reason for r in read_dropped(d)] == ["INVALID_ORDER"]

    def test_step_author_all_filters_dropped_from_narration(
        self, tmp_path, midas_data_root
    ) -> None:
        # The phantom-trade fix: step_author_all replaces result["trades"] with
        # only the authored trades so posts/journal/bundle never see a drop.
        pm = PortfolioManager(tmp_path / "portfolios")
        pm.initialize("satoshi", initial_capital=10_000.0, currency="EUR")
        d = date(2026, 5, 17)
        agent_results = {
            "satoshi": {
                "commentary": "...",
                "trades": [
                    {
                        "action": "HOLD",
                        "ticker": "BTC-EUR",
                        "shares": 1,
                        "reasoning": "x",
                    },
                    {
                        "action": "buy",
                        "ticker": "BTC-EUR",
                        "shares": 0.01,
                        "reasoning": "y",
                    },
                ],
            }
        }
        summary = step_author_all(agent_results, d, portfolio_manager=pm)
        assert summary["satoshi"]["orders"] == 1
        # result["trades"] now holds only the authored buy — the HOLD is gone,
        # and its action is normalized to match the outbox Order.
        kept = agent_results["satoshi"]["trades"]
        assert [t["reasoning"] for t in kept] == ["y"]
        assert kept[0]["action"] == "BUY"  # narration matches the ledger, not "buy"

    def test_boolean_shares_dropped_not_authored_as_one(self, midas_data_root) -> None:
        # float(True) == 1.0 would sneak past isfinite/>0 and author a phantom
        # 1-share order; a boolean must be dropped as INVALID_SHARES instead.
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                {"action": "BUY", "ticker": "BTC-EUR", "shares": True, "reasoning": "b"}
            ],
            trade_date=d,
            currency="EUR",
        )
        assert authored == []
        assert read_outbox(d) == []
        assert [r.reason for r in read_dropped(d)] == ["INVALID_SHARES"]

    def test_null_ticker_dropped_not_authored_as_string_none(
        self, midas_data_root
    ) -> None:
        # null ticker stringifies to "None"; it must be MISSING_TICKER, not a
        # phantom order for ticker "None".
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[{"action": "BUY", "ticker": None, "shares": 1, "reasoning": "x"}],
            trade_date=d,
            currency="EUR",
        )
        assert authored == []
        assert read_outbox(d) == []
        assert [r.reason for r in read_dropped(d)] == ["MISSING_TICKER"]

    def test_non_dict_trade_element_dropped_not_crashed(self, midas_data_root) -> None:
        d = date(2026, 5, 17)
        authored = step_author_orders(
            "satoshi",
            trades=[
                "BUY BTC-EUR",
                None,
                {
                    "action": "buy",
                    "ticker": "BTC-EUR",
                    "shares": 0.01,
                    "reasoning": "ok",
                },
            ],
            trade_date=d,
            currency="EUR",
        )
        assert len(authored) == 1
        assert [r.reason for r in read_dropped(d)] == [
            "MALFORMED_TRADE",
            "MALFORMED_TRADE",
        ]

    def test_null_trades_does_not_crash(self, tmp_path, midas_data_root) -> None:
        # "trades": null must not crash len()/enumerate at the boundary.
        pm = PortfolioManager(tmp_path / "portfolios")
        pm.initialize("satoshi", initial_capital=10_000.0, currency="EUR")
        d = date(2026, 5, 17)
        agent_results = {"satoshi": {"commentary": "flat", "trades": None}}
        summary = step_author_all(agent_results, d, portfolio_manager=pm)
        assert summary["satoshi"]["orders"] == 0
        assert agent_results["satoshi"]["trades"] == []

    def test_narration_filter_survives_idempotent_resume(
        self, tmp_path, midas_data_root
    ) -> None:
        # Finding: the filter must NOT live only inside the idempotent authoring
        # step, or a resumed fire (authoring skipped) re-exposes phantom trades.
        pm = PortfolioManager(tmp_path / "portfolios")
        pm.initialize("satoshi", initial_capital=10_000.0, currency="EUR")
        d = date(2026, 5, 17)
        raw = [
            {"action": "HOLD", "ticker": "BTC-EUR", "shares": 1, "reasoning": "hold"},
            {"action": "buy", "ticker": "BTC-EUR", "shares": 0.01, "reasoning": "buy"},
        ]
        first = {"satoshi": {"trades": [dict(t) for t in raw]}}
        step_author_all(first, d, portfolio_manager=pm)
        assert [t["reasoning"] for t in first["satoshi"]["trades"]] == ["buy"]
        # Simulate resume: agent_results rebuilt from raw output; _author_all_orders
        # is now skipped by idempotency, but the filter must still trim the HOLD.
        resumed = {"satoshi": {"trades": [dict(t) for t in raw]}}
        step_author_all(resumed, d, portfolio_manager=pm)
        assert [t["reasoning"] for t in resumed["satoshi"]["trades"]] == ["buy"]


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
