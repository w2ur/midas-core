"""End-to-end integration tests for the Midas backtest pipeline.

These tests hit Yahoo Finance (real network calls) and exercise the full
pipeline: data fetch → strategy spec → backtest → metrics.

Marks:
    integration — all tests in this module are slow/network-bound.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from engine.backtest import BacktestResult, run_backtest
from engine.market_data import MarketDataFetcher
from engine.portfolio import PortfolioManager
from engine.types import Trade


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STRATEGIES_DIR = _REPO_ROOT / "data" / "strategies"

_LARGE_CAP_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "JPM", "V", "UNH"]
_GOLDEN_CROSS_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]

# Date ranges
_SIX_MONTHS_START = date(2024, 1, 2)
_SIX_MONTHS_END = date(2024, 7, 1)

_EIGHTEEN_MONTHS_START = date(2023, 1, 2)
_EIGHTEEN_MONTHS_END = date(2024, 7, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trade(
    action: str,
    ticker: str,
    shares: float,
    price: float,
    trade_id: str = "t1",
    reasoning: str = "integration test trade",
) -> Trade:
    return Trade(
        id=trade_id,
        timestamp=datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        action=action,
        ticker=ticker,
        shares=shares,
        price=price,
        total=shares * price,
        fees=0.0,
        reasoning=reasoning,
    )


def _assert_valid_backtest_result(result: BacktestResult) -> None:
    """Assert that a BacktestResult has well-formed metrics."""
    assert isinstance(result, BacktestResult)
    assert isinstance(result.total_return, float)
    assert isinstance(result.cagr, float)
    assert isinstance(result.sharpe, float)
    assert isinstance(result.max_drawdown, float)
    # max_drawdown must be <= 0 (a drawdown is a loss)
    assert result.max_drawdown <= 0.0
    assert len(result.daily_values) > 0
    # Snapshots must be well-formed
    snapshots = result.to_snapshots()
    assert len(snapshots) > 0
    assert "date" in snapshots[0]
    assert "portfolioValue" in snapshots[0]
    assert isinstance(snapshots[0]["portfolioValue"], float)


# ---------------------------------------------------------------------------
# Test 1 — coin-flip-baseline end-to-end
# ---------------------------------------------------------------------------

class TestCoinFlipBacktestEndToEnd:
    """Load coin-flip-baseline spec, fetch 6 months of real prices, run backtest."""

    def test_coin_flip_backtest_end_to_end(self, tmp_path: Path) -> None:
        # Load spec from disk
        spec_path = _STRATEGIES_DIR / "coin-flip-baseline.json"
        with spec_path.open() as f:
            spec_dict = json.load(f)

        # Fetch real prices for a fixed window (avoids flakiness from moving dates)
        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(_LARGE_CAP_TICKERS, _SIX_MONTHS_START, _SIX_MONTHS_END)

        assert len(prices) >= 100, f"Expected >=100 trading days, got {len(prices)}"
        assert set(_LARGE_CAP_TICKERS).issubset(set(prices.columns))

        # Run backtest
        result = run_backtest(spec_dict, prices)

        assert result.strategy_id == "coin-flip-baseline"
        _assert_valid_backtest_result(result)

    def test_coin_flip_daily_values_start_near_initial_capital(self, tmp_path: Path) -> None:
        spec_path = _STRATEGIES_DIR / "coin-flip-baseline.json"
        with spec_path.open() as f:
            spec_dict = json.load(f)

        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(_LARGE_CAP_TICKERS, _SIX_MONTHS_START, _SIX_MONTHS_END)
        result = run_backtest(spec_dict, prices)

        # First portfolio value should be within 5% of initial capital (10 000)
        first_value = float(result.daily_values.iloc[0])
        assert 9000 <= first_value <= 11000, f"First portfolio value {first_value} is outside expected range"


# ---------------------------------------------------------------------------
# Test 2 — golden-cross end-to-end (needs 200+ days for MA calculation)
# ---------------------------------------------------------------------------

class TestGoldenCrossBacktestEndToEnd:
    """Load golden-cross spec, fetch 18 months of real prices, run backtest."""

    def test_golden_cross_backtest_end_to_end(self, tmp_path: Path) -> None:
        spec_path = _STRATEGIES_DIR / "golden-cross-dca.json"
        with spec_path.open() as f:
            spec_dict = json.load(f)

        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(_GOLDEN_CROSS_TICKERS, _EIGHTEEN_MONTHS_START, _EIGHTEEN_MONTHS_END)

        assert len(prices) >= 200, (
            f"Golden cross needs 200+ days for MA calculation, got {len(prices)}"
        )

        result = run_backtest(spec_dict, prices)

        assert result.strategy_id == "golden-cross-dca"
        _assert_valid_backtest_result(result)

    def test_golden_cross_result_has_transactions_or_none(self, tmp_path: Path) -> None:
        """Transactions field is either a DataFrame or None — never raises."""
        spec_path = _STRATEGIES_DIR / "golden-cross-dca.json"
        with spec_path.open() as f:
            spec_dict = json.load(f)

        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(_GOLDEN_CROSS_TICKERS, _EIGHTEEN_MONTHS_START, _EIGHTEEN_MONTHS_END)
        result = run_backtest(spec_dict, prices)

        # transactions is Optional[DataFrame] — just assert it doesn't blow up
        import pandas as pd
        assert result.transactions is None or isinstance(result.transactions, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 3 — PortfolioManager round-trip
# ---------------------------------------------------------------------------

class TestPortfolioManagerRoundtrip:
    """Initialize portfolio, apply trades, add snapshot, verify persistence."""

    def test_portfolio_manager_buy_sell_roundtrip(self, tmp_path: Path) -> None:
        manager = PortfolioManager(base_dir=tmp_path)
        strategy_id = "integration-test"

        # Initialize
        manager.initialize(strategy_id, initial_capital=20_000.0)
        portfolio = manager.load(strategy_id)
        assert portfolio.cash == pytest.approx(20_000.0)
        assert portfolio.positions == []

        # Apply BUY trade
        buy = _make_trade("BUY", "AAPL", shares=10, price=180.0, trade_id="buy-1")
        manager.apply_trade(strategy_id, buy)

        portfolio = manager.load(strategy_id)
        assert portfolio.cash == pytest.approx(20_000.0 - 10 * 180.0)
        assert len(portfolio.positions) == 1
        pos = portfolio.positions[0]
        assert pos.ticker == "AAPL"
        assert pos.shares == pytest.approx(10.0)
        assert pos.avg_cost == pytest.approx(180.0)

        # Apply SELL trade (partial)
        sell = _make_trade("SELL", "AAPL", shares=5, price=200.0, trade_id="sell-1")
        manager.apply_trade(strategy_id, sell)

        portfolio = manager.load(strategy_id)
        expected_cash = 20_000.0 - 10 * 180.0 + 5 * 200.0
        assert portfolio.cash == pytest.approx(expected_cash)
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].shares == pytest.approx(5.0)

        # Trade log must have both trades
        trades = manager.load_trades(strategy_id)
        assert len(trades) == 2
        actions = {t["action"] for t in trades}
        assert actions == {"BUY", "SELL"}

    def test_portfolio_manager_snapshot_persists(self, tmp_path: Path) -> None:
        manager = PortfolioManager(base_dir=tmp_path)
        strategy_id = "snapshot-test"
        manager.initialize(strategy_id, initial_capital=10_000.0)

        # Add a snapshot
        manager.add_snapshot(
            strategy_id,
            snapshot_date=date(2024, 6, 1),
            portfolio_value=10_500.0,
            cash=5_000.0,
            positions_value=5_500.0,
            benchmarks={"sp500": 5300.0, "btc": 67000.0},
        )

        snapshots = manager.load_snapshots(strategy_id)
        assert len(snapshots) == 1
        snap = snapshots[0]
        assert snap["date"] == "2024-06-01"
        assert snap["portfolio_value"] == pytest.approx(10_500.0)
        assert snap["cash"] == pytest.approx(5_000.0)
        assert snap["positions_value"] == pytest.approx(5_500.0)
        assert snap["benchmarks"]["sp500"] == pytest.approx(5300.0)

    def test_portfolio_manager_full_sell_clears_position(self, tmp_path: Path) -> None:
        manager = PortfolioManager(base_dir=tmp_path)
        strategy_id = "full-sell-test"
        manager.initialize(strategy_id, initial_capital=10_000.0)

        manager.apply_trade(strategy_id, _make_trade("BUY", "MSFT", 5, 300.0, "buy-1"))
        manager.apply_trade(strategy_id, _make_trade("SELL", "MSFT", 5, 320.0, "sell-1"))

        portfolio = manager.load(strategy_id)
        assert portfolio.positions == []
        assert portfolio.cash == pytest.approx(10_000.0 + 5 * (320.0 - 300.0))


# ---------------------------------------------------------------------------
# Test 4 — Strategy spec → backtest full pipeline
# ---------------------------------------------------------------------------

class TestStrategySpecToBacktestPipeline:
    """Load an actual JSON spec from data/strategies/, resolve universe subset,
    fetch prices, and run the full backtest — verifying the complete pipeline."""

    @pytest.mark.parametrize("spec_filename,tickers,start,end", [
        (
            "coin-flip-baseline.json",
            ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "JPM", "V", "UNH"],
            date(2024, 1, 2),
            date(2024, 7, 1),
        ),
        (
            "golden-cross-sp500.json",
            ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"],
            date(2023, 1, 2),
            date(2024, 7, 1),
        ),
    ])
    def test_spec_to_backtest_pipeline(
        self,
        tmp_path: Path,
        spec_filename: str,
        tickers: list[str],
        start: date,
        end: date,
    ) -> None:
        # Step 1: Load spec from disk
        spec_path = _STRATEGIES_DIR / spec_filename
        with spec_path.open() as f:
            spec_dict = json.load(f)

        # Step 2: Fetch prices (cache in tmp_path to avoid repeated downloads)
        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(tickers, start, end)

        assert not prices.empty, f"No price data fetched for {tickers}"
        assert len(prices) >= 50, f"Too few rows ({len(prices)}) to run a meaningful backtest"

        # Step 3: Run backtest
        result = run_backtest(spec_dict, prices)

        # Step 4: Verify output
        assert result.strategy_id == spec_dict["id"]
        _assert_valid_backtest_result(result)

    def test_spec_metadata_flows_through_to_result(self, tmp_path: Path) -> None:
        """strategy_id and strategy_name on the result match what's in the spec."""
        spec_path = _STRATEGIES_DIR / "coin-flip-baseline.json"
        with spec_path.open() as f:
            spec_dict = json.load(f)

        fetcher = MarketDataFetcher(cache_dir=tmp_path)
        prices = fetcher.fetch_prices(_LARGE_CAP_TICKERS, _SIX_MONTHS_START, _SIX_MONTHS_END)
        result = run_backtest(spec_dict, prices)

        assert result.strategy_id == spec_dict["id"]
        assert result.strategy_name == spec_dict["name"]
