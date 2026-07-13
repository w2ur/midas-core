"""Tests for idempotent fill path keyed on order_id.

Re-running fill_day twice on the same day must produce:
- identical portfolio state
- each order_id exactly once in the inbox

Same guarantee for execute_triggered_order: if the order_id is already
anywhere in the inbox, the call is a no-op (returns None).
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config

from engine.orders import Fill, Order, append_order, read_inbox
from engine.portfolio import PortfolioManager


# ---------------------------------------------------------------------------
# Fixtures (mirror broker_env pattern from test_paper_broker.py)
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
    monkeypatch.setattr("engine.paper_broker._TICKER_CURRENCY_OVERRIDES", None)
    return {
        "ohlcv": ohlcv,
        "config_dir": config_dir,
        "ticker_ccy": ticker_ccy_path,
        "outbox": outbox,
        "inbox": inbox,
        "pm_base": pm_base,
    }


def _seed_ohlcv(ohlcv_dir: Path, ticker: str, rows: list[tuple[str, float]]) -> None:
    path = ohlcv_dir / f"{ticker}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for d, close in rows:
            f.write(json.dumps({"date": d, "close": close, "adj_close": close}) + "\n")


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
    pm_base: Path, agent_id: str, cash: float = 10_000.0, currency: str = "USD"
) -> PortfolioManager:
    pm = PortfolioManager(pm_base)
    pm.initialize(agent_id, initial_capital=cash, currency=currency)
    return pm


def _make_order(
    order_id: str,
    agent_id: str,
    action: str,
    ticker: str,
    shares: float,
    currency: str = "USD",
    trade_date: date = date(2026, 4, 17),
) -> Order:
    return Order(
        order_id=order_id,
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id=agent_id,
        action=action,
        ticker=ticker,
        shares=shares,
        reasoning="test",
        currency=currency,
    )


TRADE_DATE = date(2026, 4, 17)


# ---------------------------------------------------------------------------
# A. fill_day twice on a fixture day with one BUY order
#    → portfolio identical to a single run; inbox has exactly 1 line
# ---------------------------------------------------------------------------


def test_fill_day_idempotent_single_buy(broker_env):
    """Running fill_day twice must not double-fill a BUY order."""
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=5_000.0)

    order = _make_order("ord_2026-04-17_agent1_001", "agent1", "BUY", "VOO", 5)
    append_order(TRADE_DATE, order)

    # First run — fills normally
    fills1 = fill_day(TRADE_DATE, pm)
    p_after_first = pm.load("agent1")
    cash_after_first = p_after_first.cash
    shares_after_first = (
        p_after_first.positions[0].shares if p_after_first.positions else 0
    )

    # Second run — must skip the already-processed order
    fills2 = fill_day(TRADE_DATE, pm)
    p_after_second = pm.load("agent1")

    # Portfolio must be identical
    assert p_after_second.cash == cash_after_first
    assert p_after_second.positions == p_after_first.positions

    # Inbox must have exactly one line for this order_id
    inbox_fills = read_inbox(TRADE_DATE)
    matching = [f for f in inbox_fills if f.order_id == "ord_2026-04-17_agent1_001"]
    assert len(matching) == 1, f"Expected 1 inbox entry, got {len(matching)}"


# ---------------------------------------------------------------------------
# B. Mixed day (one fillable + one rejectable), run twice
#    → inbox has exactly 2 lines total (one per order_id)
# ---------------------------------------------------------------------------


def test_fill_day_idempotent_mixed_day(broker_env):
    """Mixed day: one fillable order + one rejectable (insufficient cash).
    Running twice must produce exactly 2 inbox lines total."""
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=600.0)

    # First order fills (600 cash, 1 share at 500 → 500 notional)
    order_ok = _make_order("ord_2026-04-17_agent1_001", "agent1", "BUY", "VOO", 1)
    # Second order gets rejected (only ~100 cash left after first fill)
    order_bad = _make_order("ord_2026-04-17_agent1_002", "agent1", "BUY", "VOO", 1)
    append_order(TRADE_DATE, order_ok)
    append_order(TRADE_DATE, order_bad)

    # First run
    fill_day(TRADE_DATE, pm)
    inbox_after_first = read_inbox(TRADE_DATE)
    assert len(inbox_after_first) == 2

    # Second run
    fill_day(TRADE_DATE, pm)
    inbox_after_second = read_inbox(TRADE_DATE)

    # Still exactly 2 lines — no duplicates
    assert len(inbox_after_second) == 2

    # Each order_id appears exactly once
    ids = [f.order_id for f in inbox_after_second]
    assert ids.count("ord_2026-04-17_agent1_001") == 1
    assert ids.count("ord_2026-04-17_agent1_002") == 1

    # Statuses are preserved as-is from the first run
    by_id = {f.order_id: f for f in inbox_after_second}
    assert by_id["ord_2026-04-17_agent1_001"].status == "filled"
    assert by_id["ord_2026-04-17_agent1_002"].status == "rejected"
    assert by_id["ord_2026-04-17_agent1_002"].reason == "INSUFFICIENT_CASH"


# ---------------------------------------------------------------------------
# C. execute_triggered_order on an order already in ANY day's inbox
#    → no new fill, no portfolio mutation, returns None
# ---------------------------------------------------------------------------


def test_execute_triggered_order_skips_already_filled(broker_env):
    """execute_triggered_order must return None without mutating portfolio
    when the order_id already appears in the inbox (any day)."""
    from engine.paper_broker import execute_triggered_order
    from engine.orders import append_fill, INBOX_DIR

    _seed_ohlcv(broker_env["ohlcv"], "BTC-EUR", [("2026-04-17", 80_000.0)])
    _write_config(broker_env["config_dir"], "satoshi")
    pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=5_000.0, currency="EUR")

    order = Order(
        order_id="ord_2026-04-17_satoshi_001",
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker="BTC-EUR",
        shares=0.05,
        reasoning="dip buy",
        currency="EUR",
        trigger={"op": "<=", "level": 80_000.0},
        expires="2026-05-17",
    )

    # Simulate a prior fill already in the inbox (could be from any date)
    prior_fill = Fill(
        order_id="ord_2026-04-17_satoshi_001",
        ts_filled=datetime(2026, 4, 17, 21, 0, 0, tzinfo=timezone.utc),
        status="filled",
        fill_price=80_000.0,
        fill_currency="EUR",
        notional_base=4_000.0,
        fees=0.0,
        reason=None,
        trigger_fired=True,
    )
    append_fill(TRADE_DATE, prior_fill)

    portfolio_before = pm.load("satoshi")
    cash_before = portfolio_before.cash
    positions_before = list(portfolio_before.positions)

    # Attempt to execute again
    result = execute_triggered_order(order, TRADE_DATE, pm, fire_price=80_000.0)

    # Must return None (skip sentinel)
    assert result is None, f"Expected None, got {result}"

    # Portfolio must be unchanged
    portfolio_after = pm.load("satoshi")
    assert portfolio_after.cash == cash_before
    assert portfolio_after.positions == positions_before


def test_execute_triggered_order_skips_when_in_different_day_inbox(broker_env):
    """Inbox scan covers ALL inbox files, not just today's."""
    from engine.paper_broker import execute_triggered_order
    from engine.orders import append_fill

    _write_config(broker_env["config_dir"], "satoshi")
    pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=5_000.0, currency="EUR")

    order = Order(
        order_id="ord_2026-04-15_satoshi_001",
        ts=datetime(2026, 4, 15, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker="BTC-EUR",
        shares=0.05,
        reasoning="dip",
        currency="EUR",
        trigger={"op": "<=", "level": 78_000.0},
        expires="2026-05-15",
    )

    # Write the fill to a DIFFERENT day's inbox (April 15 authoring, filled April 16)
    prior_fill = Fill(
        order_id="ord_2026-04-15_satoshi_001",
        ts_filled=datetime(2026, 4, 16, 10, 0, 0, tzinfo=timezone.utc),
        status="filled",
        fill_price=78_000.0,
        fill_currency="EUR",
        notional_base=3_900.0,
        fees=0.0,
        reason=None,
        trigger_fired=True,
    )
    append_fill(date(2026, 4, 16), prior_fill)

    portfolio_before = pm.load("satoshi")
    cash_before = portfolio_before.cash

    # Watcher fires today (April 17) but order already filled on April 16
    result = execute_triggered_order(order, TRADE_DATE, pm, fire_price=78_000.0)

    assert result is None
    portfolio_after = pm.load("satoshi")
    assert portfolio_after.cash == cash_before


# ---------------------------------------------------------------------------
# C2. execute_triggered_order idempotency is scoped to the given inbox_dir
#     A fill in the Manager inbox must suppress a re-fire when inbox_dir=manager_inbox.
#     The same fill must NOT suppress execution when the default (public) inbox is used.
# ---------------------------------------------------------------------------


def test_execute_triggered_order_idempotency_respects_inbox_dir(broker_env, tmp_path):
    """inbox_dir kwarg scopes the idempotency scan.

    - Fill in manager_inbox + inbox_dir=manager_inbox → returns None (skip).
    - Same fill in manager_inbox, but inbox_dir omitted (public default) → does NOT
      skip (returns a Fill, since the order is not in the public inbox).
    """
    from engine.paper_broker import execute_triggered_order
    from engine.orders import append_fill, MANAGER_INBOX_DIR

    manager_inbox = MANAGER_INBOX_DIR
    manager_inbox.mkdir(parents=True, exist_ok=True)

    _seed_ohlcv(broker_env["ohlcv"], "BTC-EUR", [("2026-04-17", 80_000.0)])
    _write_config(broker_env["config_dir"], "satoshi")
    _init_portfolio(broker_env["pm_base"], "satoshi", cash=5_000.0, currency="EUR")

    order = Order(
        order_id="ord_2026-04-17_satoshi_mgr001",
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker="BTC-EUR",
        shares=0.05,
        reasoning="manager conditional dip",
        currency="EUR",
        trigger={"op": "<=", "level": 80_000.0},
        expires="2026-05-17",
    )

    # Simulate a prior fill already in the MANAGER inbox (written there by the watcher).
    prior_fill = Fill(
        order_id="ord_2026-04-17_satoshi_mgr001",
        ts_filled=datetime(2026, 4, 17, 21, 0, 0, tzinfo=timezone.utc),
        status="filled",
        fill_price=80_000.0,
        fill_currency="EUR",
        notional_base=4_000.0,
        fees=0.0,
        reason=None,
        trigger_fired=True,
    )
    append_fill(TRADE_DATE, prior_fill, inbox_dir=manager_inbox)

    # --- Case 1: with inbox_dir=manager_inbox → scan finds the fill → skip (None) ---
    pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=5_000.0, currency="EUR")
    result_manager = execute_triggered_order(
        order, TRADE_DATE, pm, fire_price=80_000.0, inbox_dir=manager_inbox
    )
    assert result_manager is None, (
        "Expected idempotency skip (None) when fill is in the manager inbox "
        f"and inbox_dir=manager_inbox; got {result_manager}"
    )

    # --- Case 2: inbox_dir omitted (public default) → fill NOT found → executes ---
    # Re-init portfolio so we have enough cash for the BUY to succeed.
    pm2 = _init_portfolio(
        broker_env["pm_base"], "satoshi", cash=5_000.0, currency="EUR"
    )
    result_public = execute_triggered_order(order, TRADE_DATE, pm2, fire_price=80_000.0)
    assert result_public is not None, (
        "Expected a real Fill when using the public inbox (fill lives only in manager "
        "inbox); got None — guard over-scanned and blocked a valid second channel"
    )


# ---------------------------------------------------------------------------
# D. Seed-based randomized test: random sequence of valid orders,
#    fill_day once vs twice → identical end state
# ---------------------------------------------------------------------------


def test_fill_day_idempotent_randomized(broker_env):
    """Generate a random sequence of valid BUY orders, run fill_day once,
    then run it again. Portfolio and inbox must be identical."""
    from engine.paper_broker import fill_day

    rng = random.Random(0)
    trade_date = date(2026, 4, 20)

    # Seed OHLCV with a few tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices = {"AAPL": 200.0, "MSFT": 420.0, "GOOGL": 175.0}
    for ticker, price in prices.items():
        _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-20", price)])

    _write_config(
        broker_env["config_dir"],
        "agent_rand",
        max_order_notional=10_000.0,
        max_orders_per_day=20,
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent_rand", cash=50_000.0)

    # Generate 6 random BUY orders with deterministic IDs
    n_orders = 6
    for seq in range(n_orders):
        ticker = rng.choice(tickers)
        shares = rng.randint(1, 3)
        order = Order(
            order_id=f"ord_2026-04-20_agent_rand_{seq:03d}",
            ts=datetime(2026, 4, 20, 20, 0, 0, tzinfo=timezone.utc),
            agent_id="agent_rand",
            action="BUY",
            ticker=ticker,
            shares=float(shares),
            reasoning="random test",
            currency="USD",
        )
        append_order(trade_date, order)

    # First run
    fill_day(trade_date, pm)
    p_after_first = pm.load("agent_rand")
    inbox_after_first = read_inbox(trade_date)

    # Second run — must be a no-op
    fill_day(trade_date, pm)
    p_after_second = pm.load("agent_rand")
    inbox_after_second = read_inbox(trade_date)

    # Portfolio must be identical
    assert p_after_second.cash == p_after_first.cash
    assert len(p_after_second.positions) == len(p_after_first.positions)
    for pos_a, pos_b in zip(
        sorted(p_after_first.positions, key=lambda p: p.ticker),
        sorted(p_after_second.positions, key=lambda p: p.ticker),
    ):
        assert pos_a.ticker == pos_b.ticker
        assert pos_a.shares == pos_b.shares

    # Inbox must have no duplicates — each order_id appears exactly once
    assert len(inbox_after_second) == len(inbox_after_first)
    second_ids = [f.order_id for f in inbox_after_second]
    assert len(second_ids) == len(set(second_ids)), "Duplicate order_ids in inbox"


# ---------------------------------------------------------------------------
# E. Cancel-channel cross-run idempotency
#    Pending conditional order + a cancel for it; run fill_day twice.
#    → Run 2 adds NO new inbox lines; exactly one CANCELLED_BY_AGENT line
#      total; pending file stays deleted after both runs.
# ---------------------------------------------------------------------------


def test_fill_day_cancel_channel_idempotent(broker_env):
    """Running fill_day twice when a cancel targets a pending conditional order
    must not produce duplicate inbox lines on the second run."""
    from engine.paper_broker import fill_day
    from engine.triggers import (
        CancelRequest,
        append_cancel,
        delete_pending,
        list_pending,
        save_pending,
    )

    _write_config(broker_env["config_dir"], "satoshi")
    pm = _init_portfolio(
        broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
    )

    target_order_id = "ord_2026-04-10_satoshi_007"
    cancel_date = date(2026, 4, 17)

    # Seed a pending conditional order (authored in a prior session).
    save_pending(
        Order(
            order_id=target_order_id,
            ts=datetime(2026, 4, 10, 20, 0, 0, tzinfo=timezone.utc),
            agent_id="satoshi",
            action="BUY",
            ticker="BTC-EUR",
            shares=0.05,
            reasoning="dip buy",
            currency="EUR",
            trigger={"op": "<=", "level": 75_000.0},
            expires="2026-05-10",
        )
    )

    # Today's session authors a cancel for that pending order.
    append_cancel(
        cancel_date,
        CancelRequest(
            request_id="cnl_2026-04-17_satoshi_001",
            ts=datetime(2026, 4, 17, 20, 5, 0, tzinfo=timezone.utc),
            agent_id="satoshi",
            target_order_id=target_order_id,
            reasoning="thesis changed",
        ),
    )

    # First run — cancel fires, pending file is removed, inbox gets one entry.
    fill_day(cancel_date, pm)
    inbox_after_first = read_inbox(cancel_date)
    cancelled_first = [f for f in inbox_after_first if f.order_id == target_order_id]
    assert len(cancelled_first) == 1
    assert cancelled_first[0].status == "rejected"
    assert cancelled_first[0].reason == "CANCELLED_BY_AGENT"
    assert list_pending() == [], "Pending file must be deleted after first run"

    # Second run — must be a no-op on the cancel channel.
    fill_day(cancel_date, pm)
    inbox_after_second = read_inbox(cancel_date)

    # No new inbox lines were added.
    assert len(inbox_after_second) == len(inbox_after_first), (
        f"Run 2 added {len(inbox_after_second) - len(inbox_after_first)} extra inbox line(s)"
    )

    # Still exactly one CANCELLED_BY_AGENT entry for this order_id.
    cancelled_second = [f for f in inbox_after_second if f.order_id == target_order_id]
    assert len(cancelled_second) == 1
    assert cancelled_second[0].reason == "CANCELLED_BY_AGENT"

    # Pending file must remain absent.
    assert list_pending() == [], "Pending file must stay deleted after second run"
