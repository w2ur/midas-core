"""Tests for PortfolioManager."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.portfolio import PortfolioManager
from engine.types import Portfolio, Position, Trade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trade(
    action: str,
    ticker: str,
    shares: float,
    price: float,
    fees: float = 0.0,
    reasoning: str = "test trade",
    trade_id: str = "t1",
    ts: datetime | None = None,
) -> Trade:
    if ts is None:
        ts = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)
    total = shares * price
    return Trade(
        id=trade_id,
        timestamp=ts,
        action=action,
        ticker=ticker,
        shares=shares,
        price=price,
        total=total,
        fees=fees,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path: Path) -> PortfolioManager:
    return PortfolioManager(base_dir=tmp_path)


@pytest.fixture
def initialized_manager(manager: PortfolioManager) -> PortfolioManager:
    manager.initialize("test-strategy", initial_capital=10_000.0)
    return manager


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_creates_portfolio_dir(
        self, manager: PortfolioManager, tmp_path: Path
    ) -> None:
        manager.initialize("my-strategy", 5_000.0)
        assert (tmp_path / "my-strategy").is_dir()

    def test_creates_portfolio_file(
        self, manager: PortfolioManager, tmp_path: Path
    ) -> None:
        manager.initialize("my-strategy", 5_000.0)
        assert (tmp_path / "my-strategy" / "portfolio.json").exists()

    def test_creates_trades_file(
        self, manager: PortfolioManager, tmp_path: Path
    ) -> None:
        manager.initialize("my-strategy", 5_000.0)
        assert (tmp_path / "my-strategy" / "trades.json").exists()

    def test_creates_snapshots_file(
        self, manager: PortfolioManager, tmp_path: Path
    ) -> None:
        manager.initialize("my-strategy", 5_000.0)
        assert (tmp_path / "my-strategy" / "snapshots.json").exists()

    def test_idempotent(self, manager: PortfolioManager) -> None:
        """Calling initialize twice does not overwrite the existing portfolio."""
        manager.initialize("my-strategy", 10_000.0)
        manager.apply_trade("my-strategy", _make_trade("BUY", "AAPL", 10, 100.0))
        manager.initialize("my-strategy", 10_000.0)  # second call
        portfolio = manager.load("my-strategy")
        # Should still have the position from the first session, not a fresh state.
        assert any(p.ticker == "AAPL" for p in portfolio.positions)


class TestLoadEmpty:
    def test_load_empty_portfolio(self, initialized_manager: PortfolioManager) -> None:
        portfolio = initialized_manager.load("test-strategy")
        assert portfolio.cash == 10_000.0
        assert portfolio.positions == []

    def test_empty_trade_log(self, initialized_manager: PortfolioManager) -> None:
        assert initialized_manager.load_trades("test-strategy") == []

    def test_empty_snapshots(self, initialized_manager: PortfolioManager) -> None:
        assert initialized_manager.load_snapshots("test-strategy") == []


# ---------------------------------------------------------------------------
# BUY trade tests
# ---------------------------------------------------------------------------


class TestBuyTrade:
    def test_cash_decreases_by_total_plus_fees(
        self, initialized_manager: PortfolioManager
    ) -> None:
        trade = _make_trade("BUY", "MSFT", shares=10, price=200.0, fees=1.0)
        initialized_manager.apply_trade("test-strategy", trade)
        portfolio = initialized_manager.load("test-strategy")
        # 10000 - (10 * 200 + 1) = 10000 - 2001 = 7999
        assert portfolio.cash == pytest.approx(7_999.0)

    def test_position_added(self, initialized_manager: PortfolioManager) -> None:
        trade = _make_trade("BUY", "MSFT", shares=10, price=200.0)
        initialized_manager.apply_trade("test-strategy", trade)
        portfolio = initialized_manager.load("test-strategy")
        assert len(portfolio.positions) == 1
        pos = portfolio.positions[0]
        assert pos.ticker == "MSFT"
        assert pos.shares == 10
        assert pos.avg_cost == pytest.approx(200.0)

    def test_buy_same_ticker_averages_cost(
        self, initialized_manager: PortfolioManager
    ) -> None:
        t1 = _make_trade("BUY", "TSLA", shares=10, price=100.0, trade_id="t1")
        t2 = _make_trade("BUY", "TSLA", shares=10, price=200.0, trade_id="t2")
        initialized_manager.apply_trade("test-strategy", t1)
        initialized_manager.apply_trade("test-strategy", t2)
        portfolio = initialized_manager.load("test-strategy")
        assert len(portfolio.positions) == 1
        pos = portfolio.positions[0]
        assert pos.shares == 20
        assert pos.avg_cost == pytest.approx(150.0)

    def test_buy_multiple_tickers(self, initialized_manager: PortfolioManager) -> None:
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("BUY", "AAPL", 5, 100.0, trade_id="t1")
        )
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("BUY", "GOOG", 2, 200.0, trade_id="t2")
        )
        portfolio = initialized_manager.load("test-strategy")
        assert len(portfolio.positions) == 2
        tickers = {p.ticker for p in portfolio.positions}
        assert tickers == {"AAPL", "GOOG"}


# ---------------------------------------------------------------------------
# SELL trade tests
# ---------------------------------------------------------------------------


class TestSellTrade:
    def _buy_shares(
        self, manager: PortfolioManager, ticker: str, shares: float, price: float
    ) -> None:
        manager.apply_trade(
            "test-strategy", _make_trade("BUY", ticker, shares, price, trade_id="buy")
        )

    def test_cash_increases_by_total_minus_fees(
        self, initialized_manager: PortfolioManager
    ) -> None:
        self._buy_shares(initialized_manager, "AAPL", 10, 150.0)
        cash_after_buy = initialized_manager.load("test-strategy").cash

        trade = _make_trade(
            "SELL", "AAPL", shares=5, price=160.0, fees=0.5, trade_id="sell"
        )
        initialized_manager.apply_trade("test-strategy", trade)

        portfolio = initialized_manager.load("test-strategy")
        # cash_after_buy + (5 * 160 - 0.5) = cash_after_buy + 799.5
        assert portfolio.cash == pytest.approx(cash_after_buy + 799.5)

    def test_shares_decrease(self, initialized_manager: PortfolioManager) -> None:
        self._buy_shares(initialized_manager, "AAPL", 10, 150.0)
        initialized_manager.apply_trade(
            "test-strategy",
            _make_trade("SELL", "AAPL", shares=4, price=160.0, trade_id="sell"),
        )
        portfolio = initialized_manager.load("test-strategy")
        pos = next(p for p in portfolio.positions if p.ticker == "AAPL")
        assert pos.shares == 6

    def test_full_sell_removes_position(
        self, initialized_manager: PortfolioManager
    ) -> None:
        self._buy_shares(initialized_manager, "NVDA", 5, 400.0)
        initialized_manager.apply_trade(
            "test-strategy",
            _make_trade("SELL", "NVDA", shares=5, price=450.0, trade_id="sell"),
        )
        portfolio = initialized_manager.load("test-strategy")
        assert not any(p.ticker == "NVDA" for p in portfolio.positions)

    def test_sell_more_than_held_raises(
        self, initialized_manager: PortfolioManager
    ) -> None:
        self._buy_shares(initialized_manager, "META", 5, 300.0)
        with pytest.raises(ValueError, match="only 5"):
            initialized_manager.apply_trade(
                "test-strategy",
                _make_trade("SELL", "META", shares=10, price=310.0, trade_id="sell"),
            )

    def test_sell_nonexistent_position_raises(
        self, initialized_manager: PortfolioManager
    ) -> None:
        with pytest.raises(ValueError, match="no open position"):
            initialized_manager.apply_trade(
                "test-strategy",
                _make_trade("SELL", "AMD", shares=5, price=150.0, trade_id="sell"),
            )

    def test_invalid_action_raises(self, initialized_manager: PortfolioManager) -> None:
        with pytest.raises(ValueError, match="Invalid trade action"):
            initialized_manager.apply_trade(
                "test-strategy",
                _make_trade("HOLD", "AAPL", shares=5, price=150.0, trade_id="hold"),
            )


# ---------------------------------------------------------------------------
# Trade log tests
# ---------------------------------------------------------------------------


class TestTradeLog:
    def test_trade_appended_to_log(self, initialized_manager: PortfolioManager) -> None:
        trade = _make_trade("BUY", "AAPL", shares=10, price=150.0, trade_id="t1")
        initialized_manager.apply_trade("test-strategy", trade)
        trades = initialized_manager.load_trades("test-strategy")
        assert len(trades) == 1
        assert trades[0]["id"] == "t1"
        assert trades[0]["ticker"] == "AAPL"
        assert trades[0]["action"] == "BUY"

    def test_multiple_trades_accumulated(
        self, initialized_manager: PortfolioManager
    ) -> None:
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("BUY", "AAPL", 10, 150.0, trade_id="t1")
        )
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("BUY", "MSFT", 5, 200.0, trade_id="t2")
        )
        trades = initialized_manager.load_trades("test-strategy")
        assert len(trades) == 2
        assert {t["id"] for t in trades} == {"t1", "t2"}

    def test_sell_also_appended(self, initialized_manager: PortfolioManager) -> None:
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("BUY", "AAPL", 10, 150.0, trade_id="buy1")
        )
        initialized_manager.apply_trade(
            "test-strategy", _make_trade("SELL", "AAPL", 5, 160.0, trade_id="sell1")
        )
        trades = initialized_manager.load_trades("test-strategy")
        assert len(trades) == 2
        actions = [t["action"] for t in trades]
        assert "BUY" in actions
        assert "SELL" in actions


# ---------------------------------------------------------------------------
# Snapshot tests
# ---------------------------------------------------------------------------


class TestSnapshots:
    def test_snapshot_appended(self, initialized_manager: PortfolioManager) -> None:
        initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=10_500.0,
            cash=3_000.0,
            positions_value=7_500.0,
            benchmarks={"sp500": 5200.0, "btc": 65000.0},
        )
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap["date"] == "2024-06-01"
        assert snap["portfolio_value"] == pytest.approx(10_500.0)
        assert snap["cash"] == pytest.approx(3_000.0)
        assert snap["positions_value"] == pytest.approx(7_500.0)
        assert snap["benchmarks"]["sp500"] == pytest.approx(5200.0)

    def test_multiple_snapshots_accumulated(
        self, initialized_manager: PortfolioManager
    ) -> None:
        for day in range(1, 4):
            initialized_manager.add_snapshot(
                "test-strategy",
                snapshot_date=date(2024, 6, day),
                portfolio_value=10_000.0 + day * 100,
                cash=5_000.0,
                positions_value=5_000.0 + day * 100,
                benchmarks={"sp500": 5200.0},
            )
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert len(snapshots) == 3
        assert snapshots[0]["date"] == "2024-06-01"
        assert snapshots[2]["date"] == "2024-06-03"

    def test_snapshot_replaces_existing_for_same_date(
        self, initialized_manager: PortfolioManager
    ) -> None:
        """A re-run on the same date should overwrite, not append a duplicate."""
        initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=float("nan"),
            cash=3_000.0,
            positions_value=float("nan"),
            benchmarks={"sp500": 5200.0},
        )
        initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=10_500.0,
            cash=3_000.0,
            positions_value=7_500.0,
            benchmarks={"sp500": 5200.0},
        )
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert len(snapshots) == 1
        assert snapshots[0]["portfolio_value"] == pytest.approx(10_500.0)

    def test_same_session_rerun_corrects_its_own_row(
        self, initialized_manager: PortfolioManager
    ) -> None:
        """Idempotency: a resumed session may overwrite the row it just wrote."""
        for value in (float("nan"), 10_500.0):
            written = initialized_manager.add_snapshot(
                "test-strategy",
                snapshot_date=date(2024, 6, 1),
                portfolio_value=value,
                cash=3_000.0,
                positions_value=7_500.0,
                benchmarks={"sp500": 5200.0},
                session_date=date(2024, 6, 3),
            )
            assert written is True

        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert len(snapshots) == 1
        assert snapshots[0]["portfolio_value"] == pytest.approx(10_500.0)
        assert snapshots[0]["date"] == "2024-06-01"
        assert snapshots[0]["session_date"] == "2024-06-03"

    def test_later_session_refuses_to_rewrite_earlier_row(
        self, initialized_manager: PortfolioManager
    ) -> None:
        """Regression: 0da774525 — the 2026-08-03 session rewrote the weekend
        refresh's 2026-08-02 row because the OHLCV store had not advanced, so a
        published equity-curve point silently gained trades made a day later."""
        initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=10_500.0,
            cash=3_000.0,
            positions_value=7_500.0,
            benchmarks={"sp500": 5200.0},
            session_date=date(2024, 6, 1),
        )
        written = initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=99_999.0,
            cash=1.0,
            positions_value=99_998.0,
            benchmarks={"sp500": 5200.0},
            session_date=date(2024, 6, 2),
        )

        assert written is False
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert len(snapshots) == 1
        assert snapshots[0]["portfolio_value"] == pytest.approx(10_500.0)
        assert snapshots[0]["session_date"] == "2024-06-01"

    def test_legacy_row_without_session_date_is_not_overwritten(
        self, initialized_manager: PortfolioManager
    ) -> None:
        """Rows written before session_date existed fail closed, not open."""
        initialized_manager._write_json(
            initialized_manager._snapshots_path("test-strategy"),
            [
                {
                    "date": "2024-06-01",
                    "portfolio_value": 10_500.0,
                    "cash": 3_000.0,
                    "positions_value": 7_500.0,
                    "benchmarks": {"sp500": 5200.0},
                }
            ],
        )
        written = initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=99_999.0,
            cash=1.0,
            positions_value=99_998.0,
            benchmarks={"sp500": 5200.0},
            session_date=date(2024, 6, 2),
        )

        assert written is False
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert snapshots[0]["portfolio_value"] == pytest.approx(10_500.0)

    def test_new_market_date_still_appends(
        self, initialized_manager: PortfolioManager
    ) -> None:
        """The refusal is scoped to a contested date — the curve keeps growing."""
        initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 1),
            portfolio_value=10_500.0,
            cash=3_000.0,
            positions_value=7_500.0,
            benchmarks={"sp500": 5200.0},
            session_date=date(2024, 6, 1),
        )
        written = initialized_manager.add_snapshot(
            "test-strategy",
            snapshot_date=date(2024, 6, 2),
            portfolio_value=10_600.0,
            cash=3_000.0,
            positions_value=7_600.0,
            benchmarks={"sp500": 5250.0},
            session_date=date(2024, 6, 3),
        )

        assert written is True
        snapshots = initialized_manager.load_snapshots("test-strategy")
        assert [s["date"] for s in snapshots] == ["2024-06-01", "2024-06-02"]


class TestBudgetGuard:
    """Verify that apply_trade rejects BUY trades exceeding available cash."""

    def test_buy_exceeding_cash_is_rejected(self, tmp_path: Path) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=1000.0)

        expensive_trade = Trade(
            id="t999",
            timestamp=datetime(2026, 4, 14, 22, 0),
            action="BUY",
            ticker="NVDA",
            shares=100,
            price=200.0,
            total=20000.0,
            fees=0.0,
            reasoning="Too expensive",
        )
        with pytest.raises(ValueError, match="Insufficient cash"):
            pm.apply_trade("test", expensive_trade)

        # Portfolio should be unchanged
        portfolio = pm.load("test")
        assert portfolio.cash == 1000.0
        assert len(portfolio.positions) == 0

    def test_buy_within_budget_succeeds(self, tmp_path: Path) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=1000.0)

        trade = Trade(
            id="t001",
            timestamp=datetime(2026, 4, 14, 22, 0),
            action="BUY",
            ticker="AAPL",
            shares=5,
            price=100.0,
            total=500.0,
            fees=0.0,
            reasoning="Affordable",
        )
        pm.apply_trade("test", trade)
        portfolio = pm.load("test")
        assert portfolio.cash == 500.0
        assert len(portfolio.positions) == 1

    def test_sequential_buys_respect_remaining_cash(self, tmp_path: Path) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=1000.0)

        # First buy: $600
        pm.apply_trade(
            "test",
            Trade(
                id="t001",
                timestamp=datetime(2026, 4, 14, 22, 0),
                action="BUY",
                ticker="AAPL",
                shares=6,
                price=100.0,
                total=600.0,
                fees=0.0,
                reasoning="First",
            ),
        )

        # Second buy: $500 — should fail, only $400 left
        with pytest.raises(ValueError, match="Insufficient cash"):
            pm.apply_trade(
                "test",
                Trade(
                    id="t002",
                    timestamp=datetime(2026, 4, 14, 22, 0),
                    action="BUY",
                    ticker="MSFT",
                    shares=5,
                    price=100.0,
                    total=500.0,
                    fees=0.0,
                    reasoning="Over budget",
                ),
            )

        portfolio = pm.load("test")
        assert portfolio.cash == 400.0
        assert len(portfolio.positions) == 1


class TestApplySplit:
    """PortfolioManager.apply_split — corporate-action share adjustment.

    Same read-mutate-write shape as apply_trade, but scoped to a single
    position's shares/avg_cost — cash and every other position are left
    untouched, and (unlike a trade) nothing is appended to trades.json.
    """

    def test_holder_gets_shares_scaled_and_cost_basis_preserved(
        self, tmp_path: Path
    ) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=2000.0)
        pm.apply_trade(
            "test",
            Trade(
                id="t001",
                timestamp=datetime(2026, 4, 14, 22, 0),
                action="BUY",
                ticker="CRWD",
                shares=3,
                price=400.0,
                total=1200.0,
                fees=0.0,
                reasoning="Pre-split buy",
            ),
        )

        assert pm.apply_split("test", "CRWD", 4.0) is True

        portfolio = pm.load("test")
        position = next(p for p in portfolio.positions if p.ticker == "CRWD")
        assert position.shares == pytest.approx(12.0)
        assert position.avg_cost == pytest.approx(100.0)
        assert position.shares * position.avg_cost == pytest.approx(1200.0)
        assert portfolio.cash == pytest.approx(2000.0 - 1200.0)  # cash untouched

    def test_non_holder_is_untouched_and_returns_false(self, tmp_path: Path) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=1000.0)
        pm.apply_trade(
            "test",
            Trade(
                id="t001",
                timestamp=datetime(2026, 4, 14, 22, 0),
                action="BUY",
                ticker="AAPL",
                shares=5,
                price=100.0,
                total=500.0,
                fees=0.0,
                reasoning="Unrelated position",
            ),
        )

        assert pm.apply_split("test", "CRWD", 4.0) is False

        portfolio = pm.load("test")
        position = next(p for p in portfolio.positions if p.ticker == "AAPL")
        assert position.shares == 5.0
        assert position.avg_cost == 100.0

    def test_split_does_not_touch_the_trade_log(self, tmp_path: Path) -> None:
        pm = PortfolioManager(base_dir=tmp_path)
        pm.initialize("test", initial_capital=2000.0)
        pm.apply_trade(
            "test",
            Trade(
                id="t001",
                timestamp=datetime(2026, 4, 14, 22, 0),
                action="BUY",
                ticker="CRWD",
                shares=3,
                price=400.0,
                total=1200.0,
                fees=0.0,
                reasoning="Pre-split buy",
            ),
        )
        before = pm.load_trades("test")

        pm.apply_split("test", "CRWD", 4.0)

        assert pm.load_trades("test") == before
