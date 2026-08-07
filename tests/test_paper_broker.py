"""Tests for engine.paper_broker — 9 safety rails + fill logic.

Each test uses the broker_env fixture to isolate filesystem state:
- OHLCV store, agent config dir, ticker currencies override, outbox, inbox,
  and PortfolioManager base dir all live under tmp_path.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.fees import fee_for
from engine.config import get_config
from engine.orders import Fill, Order, append_order
from engine.portfolio import PortfolioManager


# ---------------------------------------------------------------------------
# Fixtures
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
    # The OHLCV store, agent-config, ticker-currency, outbox, and inbox dirs all
    # resolve via get_config() under the redirected root — nothing to patch.
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
    """rows: list of (iso_date, close)."""
    path = ohlcv_dir / f"{ticker}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for d, close in rows:
            f.write(json.dumps({"date": d, "close": close, "adj_close": close}) + "\n")


def _write_config(config_dir: Path, agent_id: str, **overrides) -> None:
    """Seed per-agent safety rails via roster.yaml in the tmp MIDAS_DATA_DIR.

    As of Task 4, AgentConfig.load() reads from get_config().roster, not from
    data/agent_config/{id}.json. This helper modifies the roster.yaml that was
    seeded into the tmp root by the midas_data_root fixture so that subsequent
    AgentConfig.load(agent_id) calls return the expected values.
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
# 1. Fills a valid BUY
# ---------------------------------------------------------------------------


def test_fills_valid_buy_and_updates_portfolio(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["single-voo"],
        max_order_notional=10_000.0,
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=5000.0, currency="USD")

    order = _make_order("ord_001", "agent1", "BUY", "VOO", 5)
    append_order(TRADE_DATE, order)

    fills = fill_day(TRADE_DATE, pm)

    assert len(fills) == 1
    assert fills[0].status == "filled"
    assert fills[0].fill_price == 500.0
    assert fills[0].notional_base == 2500.0
    p = pm.load("agent1")
    # BUY debits notional + fee. fee_for(VOO, 2500) = max(1.25, 0.0005*2500) = 1.25 (floor binds).
    assert fills[0].fees == fee_for("VOO", 2500.0)
    assert p.cash == 5000.0 - 2500.0 - fee_for("VOO", 2500.0)
    assert len(p.positions) == 1
    assert p.positions[0].ticker == "VOO"
    assert p.positions[0].shares == 5


# ---------------------------------------------------------------------------
# 2. Fills a valid SELL
# ---------------------------------------------------------------------------


def test_fills_valid_sell_and_updates_portfolio(broker_env):
    from engine.paper_broker import fill_day
    from engine.types import Trade

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=1000.0)

    # Seed a position by applying a BUY trade directly through the PortfolioManager.
    buy = Trade(
        id="seed_buy",
        timestamp=datetime(2026, 4, 16, 20, 0, 0, tzinfo=timezone.utc),
        action="BUY",
        ticker="VOO",
        shares=5,
        price=400.0,
        total=2000.0,
        fees=0.0,
        reasoning="seed",
    )
    pm.initialize(
        "agent1", initial_capital=3000.0
    )  # already there; noop but ensures state
    # Reset portfolio cash to cover the seed buy.
    p = pm.load("agent1")
    p.cash = 3000.0
    (broker_env["pm_base"] / "agent1" / "portfolio.json").write_text(
        json.dumps(p.to_dict()), encoding="utf-8"
    )
    pm.apply_trade("agent1", buy)

    p_before = pm.load("agent1")
    cash_before = p_before.cash

    sell = _make_order("ord_sell", "agent1", "SELL", "VOO", 2)
    append_order(TRADE_DATE, sell)

    fills = fill_day(TRADE_DATE, pm)

    assert len(fills) == 1
    assert fills[0].status == "filled"
    p_after = pm.load("agent1")
    # SELL credits notional - fee. fee_for(VOO, 1000) = max(1.25, 0.0005*1000) = 1.25 (floor binds).
    assert fills[0].fees == fee_for("VOO", 1000.0)
    assert p_after.cash == cash_before + 2 * 500.0 - fee_for("VOO", 1000.0)
    # After selling 2 of 5, 3 remain.
    assert p_after.positions[0].shares == 3


def test_rejects_sell_when_fee_exceeds_proceeds(broker_env):
    """A SELL whose fee is >= the gross proceeds must be rejected, not booked.

    The €1.25 equity fee floor exceeds the proceeds of a tiny sale, so the fill
    would net <= 0 cash (portfolio.cash += total - fees). Reject it.
    """
    from engine.paper_broker import fill_day
    from engine.types import Trade

    _seed_ohlcv(broker_env["ohlcv"], "PENNY", [("2026-04-17", 1.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=100.0)

    # Seed a position to sell.
    pm.apply_trade(
        "agent1",
        Trade(
            id="seed_penny",
            timestamp=datetime(2026, 4, 16, 20, 0, 0, tzinfo=timezone.utc),
            action="BUY",
            ticker="PENNY",
            shares=10,
            price=1.0,
            total=10.0,
            fees=0.0,
            reasoning="seed",
        ),
    )

    # Sell 1 share: proceeds = 1.0, equity fee floor = 1.25 >= 1.0 → reject.
    assert fee_for("PENNY", 1.0) >= 1.0
    append_order(TRADE_DATE, _make_order("ord_tiny_sell", "agent1", "SELL", "PENNY", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "FEE_EXCEEDS_PROCEEDS"
    # Position untouched — nothing was booked.
    assert pm.load("agent1").positions[0].shares == 10


# ---------------------------------------------------------------------------
# 3. Rejects malformed outbox (shares <= 0)
# ---------------------------------------------------------------------------


def test_rejects_invalid_shares_from_malformed_outbox(broker_env):
    from engine.paper_broker import fill_day

    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1")

    # Hand-write a bad line (shares=0) — bypasses Order validation.
    bad = {
        "order_id": "ord_bad",
        "ts": "2026-04-17T20:00:00Z",
        "agent_id": "agent1",
        "action": "BUY",
        "ticker": "VOO",
        "shares": 0,
        "reasoning": "r",
        "currency": "USD",
    }
    outbox_path = broker_env["outbox"] / f"{TRADE_DATE.isoformat()}.jsonl"
    outbox_path.write_text(json.dumps(bad) + "\n", encoding="utf-8")

    fills = fill_day(TRADE_DATE, pm)
    assert len(fills) == 1
    assert fills[0].status == "rejected"
    assert fills[0].reason == "INVALID_SHARES"


# ---------------------------------------------------------------------------
# 4. Rejects corrupted JSON line
# ---------------------------------------------------------------------------


def test_rejects_corrupted_json_outbox_line(broker_env):
    from engine.paper_broker import fill_day

    _write_config(broker_env["config_dir"], "agent1")
    pm = _init_portfolio(broker_env["pm_base"], "agent1")

    outbox_path = broker_env["outbox"] / f"{TRADE_DATE.isoformat()}.jsonl"
    outbox_path.write_text("this is not json at all\n", encoding="utf-8")

    fills = fill_day(TRADE_DATE, pm)
    assert len(fills) == 1
    assert fills[0].status == "rejected"
    assert fills[0].reason == "INVALID_SHARES"


# ---------------------------------------------------------------------------
# 5. Rejects when over max_orders_per_day
# ---------------------------------------------------------------------------


def test_rejects_when_over_max_orders_per_day(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["single-voo"],
        max_orders_per_day=1,
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=20_000.0)

    append_order(TRADE_DATE, _make_order("ord_1", "agent1", "BUY", "VOO", 1))
    append_order(TRADE_DATE, _make_order("ord_2", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert len(fills) == 2
    statuses = {f.order_id: (f.status, f.reason) for f in fills}
    assert statuses["ord_1"][0] == "filled"
    assert statuses["ord_2"] == ("rejected", "MAX_ORDERS_PER_DAY")


# ---------------------------------------------------------------------------
# 6. Rejects when notional exceeds cap
# ---------------------------------------------------------------------------


def test_rejects_when_notional_exceeds_cap(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["single-voo"],
        max_order_notional=100.0,
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    # 1 share × 500 = 500 > 100 cap
    append_order(TRADE_DATE, _make_order("ord_big", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "MAX_ORDER_NOTIONAL"


# ---------------------------------------------------------------------------
# 7. Rejects when cash insufficient
# ---------------------------------------------------------------------------


def test_rejects_when_cash_insufficient(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=100.0)  # too little

    append_order(TRADE_DATE, _make_order("ord_pricy", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "INSUFFICIENT_CASH"


# ---------------------------------------------------------------------------
# 8. Rejects ticker outside universe
# ---------------------------------------------------------------------------


def test_rejects_when_ticker_outside_universe(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "MSFT", [("2026-04-17", 400.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    append_order(TRADE_DATE, _make_order("ord_off", "agent1", "BUY", "MSFT", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "TICKER_NOT_IN_UNIVERSE"


def test_non_empty_universe_resolving_empty_is_loud_not_fail_open(broker_env):
    """Regression: a non-empty allowed_universe that resolves to an empty
    allowlist must be a loud config error, NOT allow-all.

    Before the fix, a placeholder universe resolving to [] left allowed_tickers
    empty, and the broker treats an empty allowed set as allow-everything —
    silently disabling the TICKER_NOT_IN_UNIVERSE rail (fail open). The guard
    now raises instead of letting the off-universe order through.
    """
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "MSFT", [("2026-04-17", 400.0)])
    # "dividend-aristocrats" is a declared-but-unimplemented placeholder that
    # resolves to []; the restriction is non-empty, so this is a config error.
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["dividend-aristocrats"],
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)
    append_order(TRADE_DATE, _make_order("ord_open", "agent1", "BUY", "MSFT", 1))

    with pytest.raises(ValueError, match="refusing to fail open"):
        fill_day(TRADE_DATE, pm)


# ---------------------------------------------------------------------------
# 9. Rejects when no price data
# ---------------------------------------------------------------------------


def test_rejects_when_no_price_data(broker_env):
    from engine.paper_broker import fill_day

    # Note: allow_universe empty → no universe check. No OHLCV seeded.
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=[])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    append_order(TRADE_DATE, _make_order("ord_noprice", "agent1", "BUY", "UNKNOWN", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "NO_PRICE_DATA"


def test_rejects_with_no_fx_rate_when_ticker_currency_has_no_rate(
    broker_env, monkeypatch
):
    """Ticker has a price but cross-currency conversion fails → NO_FX_RATE.

    Distinguishes "missing OHLCV" from "missing FX pair" in the inbox audit.
    """
    from engine.paper_broker import fill_day

    # EUR-base agent buying a USD-denominated ticker; force FX lookup to fail.
    _seed_ohlcv(broker_env["ohlcv"], "FOREIGN", [("2026-04-17", 100.0)])
    _write_config(broker_env["config_dir"], "agent_eur", allowed_universe=[])
    pm = _init_portfolio(
        broker_env["pm_base"], "agent_eur", cash=10_000.0, currency="EUR"
    )

    # Force the ticker to resolve to USD so a cross-currency hop is required,
    # then make fx_convert return None.
    broker_env["ticker_ccy"].write_text('{"FOREIGN": "USD"}', encoding="utf-8")
    monkeypatch.setattr("engine.quotes._TICKER_CURRENCY_OVERRIDES", None)
    monkeypatch.setattr("engine.paper_broker.fx_convert", lambda *a, **kw: None)

    append_order(
        TRADE_DATE,
        _make_order("ord_nofx", "agent_eur", "BUY", "FOREIGN", 1, currency="EUR"),
    )

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "NO_FX_RATE"


# ---------------------------------------------------------------------------
# 10. Uses latest close on-or-before trade date
# ---------------------------------------------------------------------------


def test_uses_latest_close_on_or_before_when_today_missing(broker_env):
    from engine.paper_broker import fill_day

    # Store has 2026-04-16 but not 2026-04-17.
    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-16", 499.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    append_order(TRADE_DATE, _make_order("ord_staleprice", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "filled"
    assert fills[0].fill_price == 499.0


# ---------------------------------------------------------------------------
# 11. Rejects SELL with no position
# ---------------------------------------------------------------------------


def test_rejects_sell_when_no_position(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    append_order(TRADE_DATE, _make_order("ord_sellnothing", "agent1", "SELL", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "NO_POSITION_TO_SELL"


# ---------------------------------------------------------------------------
# 12. Rejects SELL with insufficient shares
# ---------------------------------------------------------------------------


def test_rejects_sell_when_insufficient_shares(broker_env):
    from engine.paper_broker import fill_day
    from engine.types import Trade

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    # Seed holding 5 shares via apply_trade.
    seed = Trade(
        id="seed",
        timestamp=datetime(2026, 4, 16, tzinfo=timezone.utc),
        action="BUY",
        ticker="VOO",
        shares=5,
        price=400.0,
        total=2000.0,
        fees=0.0,
        reasoning="seed",
    )
    pm.apply_trade("agent1", seed)

    append_order(TRADE_DATE, _make_order("ord_over", "agent1", "SELL", "VOO", 10))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "rejected"
    assert fills[0].reason == "INSUFFICIENT_SHARES"


# ---------------------------------------------------------------------------
# 13. Drawdown halt rejects all orders
# ---------------------------------------------------------------------------


def test_rejects_all_orders_when_drawdown_halt_triggered(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["single-voo"],
        daily_drawdown_halt_pct=-5.0,
    )
    # Portfolio is tiny (cash=100) but previous snapshot claimed 10_000 → big drawdown.
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=100.0)
    pm.add_snapshot(
        "agent1",
        date(2026, 4, 16),
        portfolio_value=10_000.0,
        cash=10_000.0,
        positions_value=0.0,
        benchmarks={},
    )

    append_order(TRADE_DATE, _make_order("ord_a", "agent1", "BUY", "VOO", 1))
    append_order(TRADE_DATE, _make_order("ord_b", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert len(fills) == 2
    assert all(f.status == "rejected" for f in fills)
    assert all(f.reason == "DAILY_DRAWDOWN_HALT" for f in fills)


# ---------------------------------------------------------------------------
# 14. Ticker currency override takes precedence
# ---------------------------------------------------------------------------


def test_ticker_currency_override_takes_precedence_over_heuristic(broker_env):
    from engine.paper_broker import fill_day

    # MSFT default would be USD. Override says EUR.
    broker_env["ticker_ccy"].write_text(json.dumps({"MSFT": "EUR"}), encoding="utf-8")
    _seed_ohlcv(broker_env["ohlcv"], "MSFT", [("2026-04-17", 100.0)])
    _write_config(
        broker_env["config_dir"], "agent1", allowed_universe=[]
    )  # no allowlist
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0, currency="EUR")

    append_order(TRADE_DATE, _make_order("ord_eur", "agent1", "BUY", "MSFT", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "filled"
    assert fills[0].fill_currency == "EUR"
    # Because ticker_ccy == base_ccy, notional_base == notional_native, no FX conversion.
    assert fills[0].notional_base == 100.0


# ---------------------------------------------------------------------------
# 15. apply_trade failure → APPLY_TRADE_FAILED, loop continues
# ---------------------------------------------------------------------------


def test_apply_trade_failure_rejects_cleanly_and_continues_loop(
    broker_env, monkeypatch
):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)

    original_apply = PortfolioManager.apply_trade
    call_state = {"n": 0}

    def flaky_apply(self, sid, trade):
        call_state["n"] += 1
        if call_state["n"] == 1:
            raise ValueError("simulated failure")
        return original_apply(self, sid, trade)

    monkeypatch.setattr(PortfolioManager, "apply_trade", flaky_apply)

    append_order(TRADE_DATE, _make_order("ord_boom", "agent1", "BUY", "VOO", 1))
    append_order(TRADE_DATE, _make_order("ord_ok", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert len(fills) == 2
    by_id = {f.order_id: f for f in fills}
    assert by_id["ord_boom"].status == "rejected"
    assert by_id["ord_boom"].reason == "APPLY_TRADE_FAILED"
    assert by_id["ord_ok"].status == "filled"


# ---------------------------------------------------------------------------
# 16. Agent not in roster falls back to safe defaults
# ---------------------------------------------------------------------------


def test_unlisted_agent_falls_back_to_defaults(broker_env):
    """An agent not present in roster.yaml must return safe defaults from AgentConfig.load.

    Previously this test wrote a malformed JSON file and verified the fallback.
    As of Task 4, AgentConfig.load reads from get_config().roster; any agent_id
    not found there gets the safe defaults regardless of the data/agent_config/ dir.
    """
    from engine.paper_broker import AgentConfig

    cfg = AgentConfig.load("ghostagent")  # not in roster → defaults
    assert cfg.max_order_notional == 500.0
    assert (
        cfg.max_orders_per_day == 5
    )  # original missing-file default (commit 320e0d53)
    assert cfg.daily_drawdown_halt_pct == -5.0
    assert cfg.allowed_universe == []
    assert cfg.dry_run is False


# ---------------------------------------------------------------------------
# 17. dry_run mode
# ---------------------------------------------------------------------------


def test_dry_run_fills_inbox_but_does_not_mutate_portfolio(broker_env):
    from engine.paper_broker import fill_day

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(
        broker_env["config_dir"],
        "agent1",
        allowed_universe=["single-voo"],
        dry_run=True,
    )
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10_000.0)
    cash_before = pm.load("agent1").cash

    append_order(TRADE_DATE, _make_order("ord_dry", "agent1", "BUY", "VOO", 1))

    fills = fill_day(TRADE_DATE, pm)
    assert fills[0].status == "filled"
    p_after = pm.load("agent1")
    assert p_after.cash == cash_before
    assert len(p_after.positions) == 0


# ---------------------------------------------------------------------------
# Conditional order routing
# ---------------------------------------------------------------------------


class TestConditionalOrderRouting:
    def test_market_order_fills_as_before(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        """Sanity: a no-trigger order still goes through the existing fill path."""
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day

        d = date(2026, 5, 17)
        _seed_ohlcv(broker_env["ohlcv"], "BTC-EUR", [("2026-05-17", 80000.0)])
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

        append_order(
            d,
            Order(
                order_id="ord_market_001",
                ts=datetime(2026, 5, 17, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="BUY",
                ticker="BTC-EUR",
                shares=0.001,
                reasoning="dip",
                currency="EUR",
            ),
        )
        fill_day(d, pm)
        fills = read_inbox(d)
        assert len(fills) == 1
        assert fills[0].order_id == "ord_market_001"
        assert fills[0].status == "filled"

    def test_conditional_order_routed_to_pending_no_inbox_record(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        """A conditional order does NOT produce an inbox fill; it goes to pending."""
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day
        from engine.triggers import list_pending

        d = date(2026, 5, 17)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

        append_order(
            d,
            Order(
                order_id="ord_conditional_001",
                ts=datetime(2026, 5, 17, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="SELL",
                ticker="BTC-EUR",
                shares=0.01,
                reasoning="trim at 85k",
                currency="EUR",
                trigger={"op": ">=", "level": 85000.0},
                expires="2026-06-17",
            ),
        )
        fill_day(d, pm)

        fills = read_inbox(d)
        assert not any(f.order_id == "ord_conditional_001" for f in fills)
        pending = list_pending()
        assert len(pending) == 1
        assert pending[0].order_id == "ord_conditional_001"

    def test_conditional_without_expires_rejected_at_broker(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        """A conditional missing expires gets a TRIGGER_NO_EXPIRY rejection (not silently pending forever).

        Note: Order.__post_init__ rejects expires-without-trigger but ALLOWS
        trigger-without-expires (the agent could forget to set expires). The
        broker enforces the requirement here.
        """
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day
        from engine.triggers import list_pending

        d = date(2026, 5, 17)
        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

        append_order(
            d,
            Order(
                order_id="ord_no_expiry_001",
                ts=datetime(2026, 5, 17, 20, 2, tzinfo=timezone.utc),
                agent_id="satoshi",
                action="SELL",
                ticker="BTC-EUR",
                shares=0.01,
                reasoning="trim",
                currency="EUR",
                trigger={"op": ">=", "level": 85000.0},
                expires=None,
            ),
        )
        fill_day(d, pm)

        fills = read_inbox(d)
        rejections = [f for f in fills if f.order_id == "ord_no_expiry_001"]
        assert len(rejections) == 1
        assert rejections[0].status == "rejected"
        assert rejections[0].reason == "TRIGGER_NO_EXPIRY"
        assert list_pending() == []


# ---------------------------------------------------------------------------
# Cancel request processing
# ---------------------------------------------------------------------------


class TestCancelRequestProcessing:
    def test_cancel_removes_pending_and_writes_rejection(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day
        from engine.triggers import (
            CancelRequest,
            append_cancel,
            list_pending,
            save_pending,
        )

        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

        # Seed pending order from a previous session.
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
                expires="2026-06-10",
            )
        )

        # Today's session writes a cancel for that order.
        d = date(2026, 5, 17)
        append_cancel(
            d,
            CancelRequest(
                request_id="cnl_2026-05-17_satoshi_001",
                ts=datetime(2026, 5, 17, 20, 5, tzinfo=timezone.utc),
                agent_id="satoshi",
                target_order_id="ord_2026-05-10_satoshi_003",
                reasoning="thesis changed",
            ),
        )

        fill_day(d, pm)

        assert list_pending() == []
        fills = read_inbox(d)
        cancelled = [f for f in fills if f.order_id == "ord_2026-05-10_satoshi_003"]
        assert len(cancelled) == 1
        assert cancelled[0].status == "rejected"
        assert cancelled[0].reason == "CANCELLED_BY_AGENT"

    def test_cancel_targeting_nonexistent_pending_writes_rejection(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day
        from engine.triggers import CancelRequest, append_cancel

        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

        d = date(2026, 5, 17)
        append_cancel(
            d,
            CancelRequest(
                request_id="cnl_x",
                ts=datetime(2026, 5, 17, 20, 5, tzinfo=timezone.utc),
                agent_id="satoshi",
                target_order_id="ord_does_not_exist",
                reasoning="oops",
            ),
        )
        fill_day(d, pm)

        fills = read_inbox(d)
        no_op = [f for f in fills if f.order_id == "ord_does_not_exist"]
        assert len(no_op) == 1
        assert no_op[0].status == "rejected"
        assert no_op[0].reason == "CANCEL_TARGET_NOT_FOUND"

    def test_duplicate_cancel_in_same_session_idempotent(
        self, broker_env, monkeypatch, tmp_path
    ) -> None:
        """Two cancels for the same target_order_id in one session: first wins as
        CANCELLED_BY_AGENT, second sees the pending already gone and becomes
        CANCEL_TARGET_NOT_FOUND. Same target_order_id, two distinct rejection records."""
        pending_dir = get_config().orders_dir / "pending"
        cancels_dir = get_config().orders_dir / "cancels"

        from engine.orders import read_inbox
        from engine.paper_broker import fill_day
        from engine.triggers import (
            CancelRequest,
            append_cancel,
            list_pending,
            save_pending,
        )

        _write_config(broker_env["config_dir"], "satoshi")
        pm = _init_portfolio(
            broker_env["pm_base"], "satoshi", cash=10_000.0, currency="EUR"
        )

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
                expires="2026-06-10",
            )
        )

        d = date(2026, 5, 17)
        for seq in (1, 2):
            append_cancel(
                d,
                CancelRequest(
                    request_id=f"cnl_2026-05-17_satoshi_{seq:03d}",
                    ts=datetime(2026, 5, 17, 20, 5, tzinfo=timezone.utc),
                    agent_id="satoshi",
                    target_order_id="ord_2026-05-10_satoshi_003",
                    reasoning=f"cancel attempt #{seq}",
                ),
            )

        fill_day(d, pm)

        assert list_pending() == []
        fills = read_inbox(d)
        targeting = [f for f in fills if f.order_id == "ord_2026-05-10_satoshi_003"]
        reasons = sorted(f.reason for f in targeting)
        assert reasons == ["CANCELLED_BY_AGENT", "CANCEL_TARGET_NOT_FOUND"]


# ---------------------------------------------------------------------------
# executed_sha provenance — stamp the HEAD commit the broker ran against
# ---------------------------------------------------------------------------


def test_fill_day_stamps_executed_sha(broker_env, monkeypatch):
    from engine import paper_broker
    from engine.paper_broker import fill_day

    monkeypatch.setattr(paper_broker, "_current_commit_sha", lambda: "f" * 40)

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=5000.0, currency="USD")

    append_order(TRADE_DATE, _make_order("ord_001", "agent1", "BUY", "VOO", 5))

    fills = fill_day(TRADE_DATE, pm)

    assert len(fills) == 1
    assert fills[0].executed_sha == "f" * 40


def test_fill_day_stamps_executed_sha_on_rejections(broker_env, monkeypatch):
    """Provenance must cover rejections too, not just filled orders."""
    from engine import paper_broker
    from engine.paper_broker import fill_day

    monkeypatch.setattr(paper_broker, "_current_commit_sha", lambda: "a" * 40)

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=10.0, currency="USD")

    # Cash far below notional -> INSUFFICIENT_CASH rejection.
    append_order(TRADE_DATE, _make_order("ord_001", "agent1", "BUY", "VOO", 5))

    fills = fill_day(TRADE_DATE, pm)

    assert len(fills) == 1
    assert fills[0].status == "rejected"
    assert fills[0].executed_sha == "a" * 40


def test_fill_day_executed_sha_none_when_not_a_git_repo(broker_env, monkeypatch):
    """Graceful degradation: a None SHA leaves the field unset, never crashes."""
    from engine import paper_broker
    from engine.paper_broker import fill_day

    monkeypatch.setattr(paper_broker, "_current_commit_sha", lambda: None)

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=5000.0, currency="USD")

    append_order(TRADE_DATE, _make_order("ord_001", "agent1", "BUY", "VOO", 5))

    fills = fill_day(TRADE_DATE, pm)

    assert fills[0].executed_sha is None
    assert "executed_sha" not in fills[0].to_dict()


def test_execute_triggered_order_stamps_executed_sha(broker_env, monkeypatch):
    from engine import paper_broker
    from engine.paper_broker import execute_triggered_order

    monkeypatch.setattr(paper_broker, "_current_commit_sha", lambda: "c" * 40)

    _seed_ohlcv(broker_env["ohlcv"], "VOO", [("2026-04-17", 500.0)])
    _write_config(broker_env["config_dir"], "agent1", allowed_universe=["single-voo"])
    pm = _init_portfolio(broker_env["pm_base"], "agent1", cash=5000.0, currency="USD")

    order = Order(
        order_id="ord_2026-04-17_agent1_001",
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="agent1",
        action="BUY",
        ticker="VOO",
        shares=2,
        reasoning="trigger buy",
        currency="USD",
        trigger={"op": "<=", "level": 500.0},
        expires="2026-05-17",
    )

    fill = execute_triggered_order(order, TRADE_DATE, pm, fire_price=500.0)

    assert fill is not None
    assert fill.executed_sha == "c" * 40


# ---------------------------------------------------------------------------
# TestSafetyFold — verify rails are read from roster config, not JSON files
# ---------------------------------------------------------------------------


class TestSafetyFold:
    """Safety rails now come from roster.yaml; these tests prove the fold is correct."""

    def test_roster_agent_reads_rails_from_config(self, midas_data_root):
        """A real roster trader's AgentConfig.load() returns the config-folded values."""
        from engine.config import get_config
        from engine.paper_broker import AgentConfig

        cfg = get_config()
        aid = cfg.trading_roster[0]  # first trader in roster
        rails = AgentConfig.load(aid)
        spec_safety = cfg.roster[aid].safety
        assert rails.max_order_notional == spec_safety.max_order_notional
        assert rails.daily_drawdown_halt_pct == spec_safety.daily_drawdown_halt_pct
        assert rails.max_orders_per_day == spec_safety.max_orders_per_day
        assert rails.dry_run == spec_safety.dry_run

    @pytest.mark.live_cast
    def test_satoshi_has_folded_permissive_values(self, midas_data_root):
        """satoshi's safety fold preserved the permissive paper-only rails."""
        from engine.paper_broker import AgentConfig

        rails = AgentConfig.load("satoshi")
        assert rails.max_order_notional == 1_000_000.0
        assert rails.daily_drawdown_halt_pct == -95.0
        assert rails.max_orders_per_day == 100
        assert rails.dry_run is False

    @pytest.mark.live_cast
    def test_yolo_sapiens_eur_has_folded_permissive_values(self, midas_data_root):
        """Spot-check a second trader to confirm the fold is roster-wide."""
        from engine.paper_broker import AgentConfig

        rails = AgentConfig.load("yolo-sapiens-eur")
        assert rails.max_order_notional == 1_000_000.0
        assert rails.daily_drawdown_halt_pct == -95.0

    def test_not_in_roster_agent_gets_safe_defaults(self, midas_data_root):
        """the-manager is not in roster.yaml; it must receive safe defaults.

        These must match the ORIGINAL broker's missing-file defaults exactly
        (commit 320e0d53): notional 500, max_orders_per_day 5, drawdown -5%.
        """
        from engine.paper_broker import AgentConfig

        rails = AgentConfig.load("the-manager")
        assert rails.max_order_notional == 500.0
        assert rails.daily_drawdown_halt_pct == -5.0
        assert rails.max_orders_per_day == 5  # original default, NOT 100
        assert rails.dry_run is False
