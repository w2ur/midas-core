"""Hermetic end-to-end proof of the public demo-desk Brain/Hands loop.

Mirrors examples/demo-desk/README.md without any network: a tmp copy of the
demo desk becomes the active roster, synthetic OHLCV rows stand in for the
yfinance bootstrap, one outbox order fills and mutates the portfolio, and one
deliberately oversized order trips a broker safety rail. No ``live_cast``
marker — this must run on the demo desk itself (it seeds its own demo cast), so
it ships to midas-core via the manifest glob and guards the walkthrough there.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.config import get_config, reset_config_cache
from engine.fees import fee_for
from engine.orders import Order, append_order, read_inbox
from engine.portfolio import PortfolioManager

_DEMO_DESK = Path(__file__).resolve().parents[1] / "examples" / "demo-desk"
TRADE_DATE = date(2026, 4, 17)


@pytest.fixture
def demo_desk_root(midas_data_root: Path) -> Path:
    """Make the demo desk the active roster under the redirected MIDAS_DATA_DIR.

    ``midas_data_root`` seeds the LIVE roster.yaml; overwrite it with the demo
    desk's cast so ``get_config().roster`` resolves demo-momentum & friends,
    then re-clear the config cache so the swap takes effect.
    """
    shutil.copy(_DEMO_DESK / "roster.yaml", midas_data_root / "roster.yaml")
    shutil.copytree(
        _DEMO_DESK / ".claude", midas_data_root / ".claude", dirs_exist_ok=True
    )
    reset_config_cache()
    return midas_data_root


def _seed_ohlcv(ticker: str, iso_date: str, close: float) -> None:
    path = get_config().ohlcv_dir / f"{ticker}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"date": iso_date, "close": close, "adj_close": close}) + "\n"
        )


def _order(order_id: str, action: str, ticker: str, shares: float) -> Order:
    return Order(
        order_id=order_id,
        ts=datetime(2026, 4, 17, 20, 0, 0, tzinfo=timezone.utc),
        agent_id="demo-momentum",
        action=action,
        ticker=ticker,
        shares=shares,
        reasoning="walkthrough fixture",
        currency="USD",
    )


def test_demo_desk_fill_and_rejection(demo_desk_root: Path) -> None:
    from engine.paper_broker import fill_day

    # The demo roster is live: demo-momentum trades USD with a 5000 notional cap.
    cfg = get_config()
    assert "demo-momentum" in cfg.roster
    assert cfg.roster["demo-momentum"].safety.max_order_notional == 5000.0

    _seed_ohlcv("SPY", "2026-04-17", 700.0)
    pm = PortfolioManager(base_dir=cfg.portfolios_dir)
    pm.initialize("demo-momentum", initial_capital=10_000.0, currency="USD")

    # One order that fills (3 * 700 = 2100 < 5000) and one that trips the cap
    # (10 * 700 = 7000 > 5000).
    append_order(TRADE_DATE, _order("ord_fill", "BUY", "SPY", 3))
    append_order(TRADE_DATE, _order("ord_reject", "BUY", "SPY", 10))

    fills = fill_day(TRADE_DATE, pm)

    assert len(fills) == 2
    by_id = {f.order_id: f for f in fills}

    filled = by_id["ord_fill"]
    assert filled.status == "filled"
    assert filled.fill_price == 700.0
    assert filled.notional_base == 2100.0

    rejected = by_id["ord_reject"]
    assert rejected.status == "rejected"
    assert rejected.reason == "MAX_ORDER_NOTIONAL"

    # The fill mutated the portfolio: cash debited by notional + fee, position opened.
    fee = fee_for("SPY", 2100.0)
    assert filled.fees == fee
    portfolio = pm.load("demo-momentum")
    assert portfolio.cash == pytest.approx(10_000.0 - 2100.0 - fee)
    assert len(portfolio.positions) == 1
    assert portfolio.positions[0].ticker == "SPY"
    assert portfolio.positions[0].shares == 3

    # The inbox is the durable Hands-side record both fills landed in.
    inbox = {f.order_id: f for f in read_inbox(TRADE_DATE)}
    assert inbox["ord_fill"].status == "filled"
    assert inbox["ord_reject"].reason == "MAX_ORDER_NOTIONAL"
