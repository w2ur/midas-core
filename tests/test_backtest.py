"""Tests for the backtest runner — run_backtest and run_batch."""

import pytest
import pandas as pd
import numpy as np

from engine.backtest import run_backtest, run_batch, BacktestResult


class TestBacktestRunner:
    @pytest.fixture
    def sample_prices(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        np.random.seed(42)
        tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]
        data = {}
        for t in tickers:
            base = np.random.uniform(100, 500)
            returns = np.random.normal(0.001, 0.02, 300)
            prices = base * np.cumprod(1 + returns)
            data[t] = prices
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def spec_dict(self):
        return {
            "id": "test-golden-equal",
            "name": "Test Golden Cross Equal Weight",
            "universe": "sp500",
            "selector": "golden-cross",
            "manager": "equal-weight",
            "funding": {"initial": 10000},
            "dividends": "cash",
            "rules": {"maxPositions": 3, "maxPositionPct": 40, "minHoldDays": 3},
        }

    def test_run_single_backtest(self, sample_prices, spec_dict):
        result = run_backtest(spec_dict, sample_prices)
        assert isinstance(result, BacktestResult)
        assert result.strategy_id == "test-golden-equal"
        assert result.total_return is not None
        assert result.sharpe is not None
        assert result.max_drawdown is not None
        assert len(result.daily_values) > 0

    def test_run_batch(self, sample_prices, spec_dict):
        specs = [
            spec_dict,
            {**spec_dict, "id": "test-random", "name": "Random", "selector": "random"},
        ]
        results = run_batch(specs, sample_prices)
        assert len(results) == 2

    def test_to_snapshots(self, sample_prices, spec_dict):
        result = run_backtest(spec_dict, sample_prices)
        snapshots = result.to_snapshots()
        assert len(snapshots) > 0
        assert "date" in snapshots[0]
        assert "portfolioValue" in snapshots[0]

    def test_custom_initial_capital(self, sample_prices, spec_dict):
        result = run_backtest(spec_dict, sample_prices, initial_capital=50000)
        # First daily value should be close to 50000
        assert abs(result.daily_values.iloc[0] - 50000) < 1000
