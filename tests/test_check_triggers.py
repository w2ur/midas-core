"""Tests for scripts.check_triggers — the conditional-order watcher."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config

from engine.orders import Order, read_inbox
from engine.portfolio import PortfolioManager
from engine.triggers import list_pending, save_pending


# ---------------------------------------------------------------------------
# Fixtures (replicated from test_paper_broker — cross-file fixture sharing
# via conftest is the alternative; inline keeps this test file self-contained).
# ---------------------------------------------------------------------------


@pytest.fixture
def broker_env(midas_data_root, monkeypatch):
    cfg = get_config()
    ohlcv = cfg.ohlcv_dir
    ohlcv.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.agent_config_dir
    config_dir.mkdir(parents=True, exist_ok=True)
    ticker_ccy_path = cfg.ticker_currencies_path
    outbox = cfg.orders_dir / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    inbox = cfg.orders_dir / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    pm_base = midas_data_root / "portfolios"
    pm_base.mkdir()
    pending_dir = cfg.orders_dir / "pending"
    cancels_dir = cfg.orders_dir / "cancels"
    manager_inbox = cfg.orders_dir / "manager-inbox"
    manager_pending = cfg.orders_dir / "manager-pending"
    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)
    return {
        "ohlcv": ohlcv,
        "config_dir": config_dir,
        "ticker_ccy": ticker_ccy_path,
        "outbox": outbox,
        "inbox": inbox,
        "pm_base": pm_base,
        "pending": pending_dir,
        "cancels": cancels_dir,
        "manager_inbox": manager_inbox,
        "manager_pending": manager_pending,
    }


def _write_config(config_dir: Path, agent_id: str, **overrides) -> None:
    """Seed per-agent safety rails via roster.yaml in the tmp MIDAS_DATA_DIR.

    As of Task 4, AgentConfig.load() reads from get_config().roster.
    ``config_dir`` is kept in the signature for call-site compatibility.
    """
    import yaml
    from engine.config import get_config, reset_config_cache

    safety = {
        "max_order_notional": 10_000.0,
        "max_orders_per_day": 10,
        "daily_drawdown_halt_pct": -50.0,
        "allowed_universe": [],
        "dry_run": False,
    }
    safety.update(overrides)
    cfg = get_config()
    roster_path = cfg.data_dir / "roster.yaml"
    data = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
    if agent_id not in data["agents"]:
        data["agents"][agent_id] = {"display_name": agent_id, "role": "trader"}
    data["agents"][agent_id]["safety"] = safety
    roster_path.write_text(yaml.dump(data), encoding="utf-8")
    reset_config_cache()


def _init_portfolio(
    pm_base: Path, agent_id: str, cash: float = 10_000.0, currency: str = "EUR"
) -> PortfolioManager:
    pm = PortfolioManager(pm_base)
    pm.initialize(agent_id, initial_capital=cash, currency=currency)
    return pm


def _seed_pending(broker_env, **overrides) -> Order:
    defaults = dict(
        order_id="ord_2026-05-10_satoshi_003",
        ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="SELL",
        ticker="BTC-EUR",
        shares=0.01,
        reasoning="trim at 85k",
        currency="EUR",
        trigger={"op": ">=", "level": 85000.0},
        expires="2026-06-10",
    )
    defaults.update(overrides)
    o = Order(**defaults)
    save_pending(o)
    return o


# ---------------------------------------------------------------------------
# Blackout window
# ---------------------------------------------------------------------------


class TestBlackoutWindow:
    @pytest.mark.parametrize("hh,mm", [(19, 55), (20, 0), (20, 15), (20, 30)])
    def test_blackout_skips_processing(self, broker_env, hh, mm) -> None:
        from scripts import check_triggers

        _seed_pending(broker_env)
        fake_now = datetime(2026, 5, 17, hh, mm, tzinfo=timezone.utc)
        result = check_triggers.run(now=fake_now, portfolio_manager=None)
        assert result["blacked_out"] is True
        assert len(list_pending()) == 1  # untouched

    @pytest.mark.parametrize("hh,mm", [(19, 54), (20, 31), (3, 0), (14, 30)])
    def test_normal_hours_do_run(self, broker_env, monkeypatch, hh, mm) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        # Force trigger NOT to fire so the run is a no-op besides the blackout check.
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 70000.0)
        fake_now = datetime(2026, 5, 17, hh, mm, tzinfo=timezone.utc)
        result = check_triggers.run(now=fake_now, portfolio_manager=pm)
        assert result["blacked_out"] is False


# ---------------------------------------------------------------------------
# Trigger fire / no-fire / unavailable price
# ---------------------------------------------------------------------------


class TestTriggerFire:
    def test_fire_when_price_meets_trigger(self, broker_env, monkeypatch) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        o = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        # Seed a position so the SELL can be filled (rails check NO_POSITION_TO_SELL).
        # Cash must cover the seed BUY (0.1 BTC × 70000 = 7000 EUR).
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        from engine.types import Trade

        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []
        fills = read_inbox(fake_now.date())
        triggered = [f for f in fills if f.order_id == o.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].trigger_fired is True
        assert triggered[0].fill_price == 85123.45

    def test_no_fire_when_price_doesnt_meet_trigger(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert len(list_pending()) == 1
        assert read_inbox(fake_now.date()) == []

    def test_price_unavailable_carries_forward(self, broker_env, monkeypatch) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: None)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert len(list_pending()) == 1
        assert read_inbox(fake_now.date()) == []


# ---------------------------------------------------------------------------
# Expiry takes precedence over firing
# ---------------------------------------------------------------------------


class TestExpiry:
    def test_expired_order_removed_and_rejection_logged(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        o = _seed_pending(broker_env, expires="2026-04-01")  # already expired
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )  # would fire if not expired
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []
        fills = read_inbox(fake_now.date())
        expired = [f for f in fills if f.order_id == o.order_id]
        assert len(expired) == 1
        assert expired[0].status == "rejected"
        assert expired[0].reason == "TRIGGER_EXPIRED"
        assert expired[0].trigger_fired is True


# ---------------------------------------------------------------------------
# Safety rails apply at fire time, not declaration time
# ---------------------------------------------------------------------------


class TestBrokerRailsApplyOnFire:
    def test_insufficient_cash_at_fire_time_logged_as_rejection(
        self, broker_env, monkeypatch
    ) -> None:
        """The agent's portfolio has 0 cash when the trigger fires — rejection, not fill."""
        from scripts import check_triggers
        from engine import triggers as triggers_mod

        _seed_pending(broker_env, action="BUY", trigger={"op": "<=", "level": 90000.0})
        _write_config(broker_env["config_dir"], "satoshi")
        # Initialize with 0 cash so a BUY will fail INSUFFICIENT_CASH.
        pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=0.0, currency="EUR")
        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        assert list_pending() == []  # pending removed regardless of outcome
        fills = read_inbox(fake_now.date())
        assert any(
            f.status == "rejected" and f.reason == "INSUFFICIENT_CASH" for f in fills
        )


# ---------------------------------------------------------------------------
# Manager channel isolation: the watcher fires Manager pending orders into the
# Manager's OWN inbox, never the public one the site joins by order_id.
# ---------------------------------------------------------------------------


class TestManagerChannelIsolation:
    @pytest.mark.live_cast
    def test_manager_pending_fires_into_manager_inbox(
        self, broker_env, monkeypatch
    ) -> None:
        from scripts import check_triggers
        from engine import triggers as triggers_mod
        from engine.triggers import MANAGER_PENDING_DIR, save_pending

        # A Manager conditional BUY whose trigger is already satisfied at fire price.
        order = Order(
            order_id="ord_2026-05-10_the-manager_001",
            ts=datetime(2026, 5, 10, 20, 2, tzinfo=timezone.utc),
            agent_id="the-manager",
            action="BUY",
            ticker="BTC-EUR",
            shares=0.01,
            reasoning="accumulate on dip",
            currency="EUR",
            trigger={"op": "<=", "level": 90000.0},
            expires="2026-06-10",
        )
        save_pending(order, pending_dir=MANAGER_PENDING_DIR)

        _write_config(broker_env["config_dir"], "the-manager")
        pm = _init_portfolio(
            broker_env["pm_base"], "the-manager", cash=10_000.0, currency="EUR"
        )
        cash_before = pm.load("the-manager").cash

        monkeypatch.setattr(triggers_mod, "get_current_price", lambda t, today: 80000.0)
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        # (a) Filled fill lands in the MANAGER inbox.
        manager_fills = read_inbox(
            fake_now.date(), inbox_dir=broker_env["manager_inbox"]
        )
        triggered = [f for f in manager_fills if f.order_id == order.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].trigger_fired is True
        assert triggered[0].fill_price == 80000.0

        # (b) NOTHING leaks into the public inbox.
        assert read_inbox(fake_now.date()) == []

        # (c) the-manager book mutated: cash down, position opened.
        book = pm.load("the-manager")
        assert book.cash < cash_before
        assert any(p.ticker == "BTC-EUR" and p.shares > 0 for p in book.positions)

        # (d) Manager pending file deleted; public pending untouched (empty).
        assert list_pending(pending_dir=MANAGER_PENDING_DIR) == []
        assert list_pending() == []

    def test_public_channel_unchanged(self, broker_env, monkeypatch) -> None:
        """Regression guard: a public pending order still fires into the public inbox."""
        from scripts import check_triggers
        from engine import triggers as triggers_mod
        from engine.triggers import MANAGER_PENDING_DIR

        o = _seed_pending(broker_env)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=8000.0, currency="EUR"
        )
        from engine.types import Trade

        pm.apply_trade(
            "satoshi",
            Trade(
                id="seed_001",
                timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                action="BUY",
                ticker="BTC-EUR",
                shares=0.1,
                price=70000.0,
                total=7000.0,
                fees=0.0,
                reasoning="seed",
            ),
        )
        monkeypatch.setattr(
            triggers_mod, "get_current_price", lambda t, today: 85123.45
        )
        fake_now = datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc)
        check_triggers.run(now=fake_now, portfolio_manager=pm)

        # Public fill lands in the public inbox exactly as before.
        fills = read_inbox(fake_now.date())
        triggered = [f for f in fills if f.order_id == o.order_id]
        assert len(triggered) == 1
        assert triggered[0].status == "filled"
        assert triggered[0].fill_price == 85123.45
        assert list_pending() == []
        # Manager inbox stays empty for a public fire.
        assert read_inbox(fake_now.date(), inbox_dir=broker_env["manager_inbox"]) == []
        assert list_pending(pending_dir=MANAGER_PENDING_DIR) == []
