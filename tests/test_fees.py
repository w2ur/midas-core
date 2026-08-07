"""Tests for engine.fees — per-class fee model.

TDD: these tests were written before the implementation.
Tests cover:
- classify_ticker: crypto / fx / equity
- fee_for: floor binding, rates for all classes
- Cash impact in paper broker: BUY deducts notional + fee, SELL adds notional - fee
- INSUFFICIENT_CASH fires when cash covers notional but not notional + fee
- Triggered-fill path carries fees
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config

from engine.orders import Fill, Order, append_order
from engine.portfolio import PortfolioManager
from engine.types import Trade


# ---------------------------------------------------------------------------
# Fixtures (mirror broker_env from test_paper_broker.py)
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
    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)
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
        "max_order_notional": 100_000.0,
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
# 1. classify_ticker
# ---------------------------------------------------------------------------


class TestClassifyTicker:
    def test_btc_eur_is_crypto(self):
        from engine.fees import classify_ticker

        assert classify_ticker("BTC-EUR") == "crypto"

    def test_eth_usd_is_crypto(self):
        from engine.fees import classify_ticker

        assert classify_ticker("ETH-USD") == "crypto"

    def test_eurusd_fx_is_fx(self):
        from engine.fees import classify_ticker

        assert classify_ticker("EURUSD=X") == "fx"

    def test_gbpusd_fx_is_fx(self):
        from engine.fees import classify_ticker

        assert classify_ticker("GBPUSD=X") == "fx"

    def test_aapl_is_equity(self):
        from engine.fees import classify_ticker

        assert classify_ticker("AAPL") == "equity"

    def test_mc_pa_is_equity(self):
        from engine.fees import classify_ticker

        assert classify_ticker("MC.PA") == "equity"

    def test_voo_is_equity(self):
        from engine.fees import classify_ticker

        assert classify_ticker("VOO") == "equity"

    def test_unknown_ticker_no_suffix_is_equity(self):
        from engine.fees import classify_ticker

        assert classify_ticker("UNKNOWN") == "equity"


# ---------------------------------------------------------------------------
# 2. fee_for — floor and rate correctness
# ---------------------------------------------------------------------------


class TestFeeFor:
    def test_equity_small_order_hits_floor(self):
        """€100 equity notional → fee is floor 1.25 (0.05% of 100 = 0.05 < 1.25)."""
        from engine.fees import fee_for

        fee = fee_for("AAPL", 100.0)
        assert fee == pytest.approx(1.25)

    def test_equity_large_order_rate_applies(self):
        """€10,000 equity notional → 0.05% = 5.0 > 1.25 floor."""
        from engine.fees import fee_for

        fee = fee_for("AAPL", 10_000.0)
        assert fee == pytest.approx(5.0)

    def test_equity_floor_boundary(self):
        """Exactly at floor boundary: 0.0005 * 2500 = 1.25 — exactly the floor."""
        from engine.fees import fee_for

        fee = fee_for("VOO", 2500.0)
        assert fee == pytest.approx(1.25)

    def test_crypto_rate(self):
        """BTC-EUR, €4000 notional → 0.4% taker = 16.0."""
        from engine.fees import fee_for

        fee = fee_for("BTC-EUR", 4_000.0)
        assert fee == pytest.approx(16.0)

    def test_fx_rate(self):
        """EURUSD=X, €10,000 notional → 0.002% = 0.20."""
        from engine.fees import fee_for

        fee = fee_for("EURUSD=X", 10_000.0)
        assert fee == pytest.approx(0.20)

    def test_fee_non_negative_on_zero_notional(self):
        """Edge case: zero notional → fee is 0 for FX/crypto (no floor on those)."""
        from engine.fees import fee_for

        assert fee_for("EURUSD=X", 0.0) == pytest.approx(0.0)
        assert fee_for("BTC-EUR", 0.0) == pytest.approx(0.0)

    def test_equity_zero_notional_returns_floor(self):
        """Equity has a floor: even zero notional returns the floor."""
        from engine.fees import fee_for

        assert fee_for("AAPL", 0.0) == pytest.approx(1.25)


# ---------------------------------------------------------------------------
# 3. BUY cash debit includes fee
# ---------------------------------------------------------------------------


def test_buy_cash_debit_includes_fee(broker_env):
    """After a BUY, cash is reduced by notional + fee, not just notional."""
    from engine.paper_broker import fill_day

    ticker = "VOO"
    price = 500.0
    shares = 10.0
    notional = price * shares  # 5000.0

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", price)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    cash_initial = 10_000.0
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=cash_initial)

    append_order(TRADE_DATE, _make_order("ord_001", "agent1", "BUY", ticker, shares))

    fills = fill_day(TRADE_DATE, pm)

    assert fills[0].status == "filled"

    from engine.fees import fee_for

    expected_fee = fee_for(ticker, notional)
    assert fills[0].fees == pytest.approx(expected_fee)

    p = pm.load("agent1")
    assert p.cash == pytest.approx(cash_initial - notional - expected_fee)


# ---------------------------------------------------------------------------
# 4. SELL cash credit nets fee
# ---------------------------------------------------------------------------


def test_sell_cash_credit_nets_fee(broker_env):
    """After a SELL, cash is increased by notional - fee."""
    from engine.paper_broker import fill_day
    from engine.fees import fee_for

    ticker = "VOO"
    price = 500.0
    shares = 5.0
    notional = price * shares  # 2500.0

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", price)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    cash_initial = 1_000.0
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=cash_initial)

    # Seed a position via apply_trade with fees=0 to keep cash accounting simple
    seed = Trade(
        id="seed",
        timestamp=datetime(2026, 4, 16, 20, 0, 0, tzinfo=timezone.utc),
        action="BUY",
        ticker=ticker,
        shares=10,
        price=400.0,
        total=4_000.0,
        fees=0.0,
        reasoning="seed",
    )
    # Manually set cash to cover seed without involving broker
    (broker_env["pm_base"] / "agent1" / "portfolio.json").write_text(
        json.dumps(
            {
                "cash": 5_000.0,
                "currency": "USD",
                "last_updated": "2026-04-16",
                "positions": [],
            }
        ),
        encoding="utf-8",
    )
    pm.apply_trade("agent1", seed)

    cash_before_sell = pm.load("agent1").cash

    append_order(TRADE_DATE, _make_order("ord_sell", "agent1", "SELL", ticker, shares))

    fills = fill_day(TRADE_DATE, pm)

    assert fills[0].status == "filled"

    expected_fee = fee_for(ticker, notional)
    assert fills[0].fees == pytest.approx(expected_fee)

    p = pm.load("agent1")
    assert p.cash == pytest.approx(cash_before_sell + notional - expected_fee)


# ---------------------------------------------------------------------------
# 5. INSUFFICIENT_CASH when cash covers notional but not notional + fee
# ---------------------------------------------------------------------------


def test_insufficient_cash_triggers_when_notional_plus_fee_exceeds_cash(broker_env):
    """If cash == notional exactly (no room for fee), BUY must be rejected INSUFFICIENT_CASH."""
    from engine.paper_broker import fill_day
    from engine.fees import fee_for

    ticker = "AAPL"
    price = 200.0
    shares = 5.0
    notional = price * shares  # 1000.0
    fee = fee_for(ticker, notional)  # should be max(1.25, 0.0005 * 1000) = 1.25

    # Cash is exactly enough for notional but not notional + fee
    cash = notional  # 1000.0 — not enough with fee

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", price)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=cash)

    append_order(TRADE_DATE, _make_order("ord_tight", "agent1", "BUY", ticker, shares))

    fills = fill_day(TRADE_DATE, pm)

    assert fills[0].status == "rejected"
    assert fills[0].reason == "INSUFFICIENT_CASH"


def test_buy_succeeds_when_cash_covers_notional_plus_fee(broker_env):
    """If cash >= notional + fee exactly, BUY must fill."""
    from engine.paper_broker import fill_day
    from engine.fees import fee_for

    ticker = "AAPL"
    price = 200.0
    shares = 5.0
    notional = price * shares  # 1000.0
    fee = fee_for(ticker, notional)

    # Cash is exactly notional + fee
    cash = notional + fee

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", price)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=cash)

    append_order(TRADE_DATE, _make_order("ord_exact", "agent1", "BUY", ticker, shares))

    fills = fill_day(TRADE_DATE, pm)

    assert fills[0].status == "filled"
    p = pm.load("agent1")
    # Cash should be zero after exact-cover fill
    assert p.cash == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 6. Triggered-fill path carries fees
# ---------------------------------------------------------------------------


def test_triggered_fill_carries_fees(broker_env):
    """execute_triggered_order must compute and apply fees, not use 0.0."""
    from engine.paper_broker import execute_triggered_order
    from engine.fees import fee_for

    ticker = "BTC-EUR"
    fire_price = 80_000.0
    shares = 0.05
    notional = fire_price * shares  # 4000.0

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", fire_price)])
    _write_config(broker_env["config_dir"], "satoshi", max_order_notional=100_000.0)
    cash_initial = 10_000.0
    pm = _init_portfolio(
        broker_env["pm_base"], "satoshi", cash=cash_initial, currency="EUR"
    )

    order = Order(
        order_id="ord_2026-04-17_satoshi_001",
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker=ticker,
        shares=shares,
        reasoning="dip buy",
        currency="EUR",
        trigger={"op": "<=", "level": fire_price},
        expires="2026-05-17",
    )

    fill = execute_triggered_order(order, TRADE_DATE, pm, fire_price=fire_price)

    assert fill is not None
    assert fill.status == "filled"

    expected_fee = fee_for(ticker, notional)
    assert fill.fees == pytest.approx(expected_fee)

    p = pm.load("satoshi")
    assert p.cash == pytest.approx(cash_initial - notional - expected_fee)


def test_triggered_fill_insufficient_cash_includes_fee(broker_env):
    """execute_triggered_order must reject INSUFFICIENT_CASH when cash covers notional but not notional+fee."""
    from engine.paper_broker import execute_triggered_order
    from engine.fees import fee_for

    ticker = "BTC-EUR"
    fire_price = 80_000.0
    shares = 0.05
    notional = fire_price * shares  # 4000.0
    fee = fee_for(ticker, notional)

    cash = notional  # exactly notional, not enough with fee

    _seed_ohlcv(broker_env["ohlcv"], ticker, [("2026-04-17", fire_price)])
    _write_config(broker_env["config_dir"], "satoshi", max_order_notional=100_000.0)
    pm = _init_portfolio(broker_env["pm_base"], "satoshi", cash=cash, currency="EUR")

    order = Order(
        order_id="ord_2026-04-17_satoshi_002",
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="satoshi",
        action="BUY",
        ticker=ticker,
        shares=shares,
        reasoning="dip buy",
        currency="EUR",
        trigger={"op": "<=", "level": fire_price},
        expires="2026-05-17",
    )

    fill = execute_triggered_order(order, TRADE_DATE, pm, fire_price=fire_price)

    assert fill is not None
    assert fill.status == "rejected"
    assert fill.reason == "INSUFFICIENT_CASH"
