"""Tests for the strategy adapter — StrategySpec to bt.Strategy pipeline."""

import pytest
import pandas as pd
import numpy as np
import bt

from engine.types import StrategySpec, FundingConfig, StrategyRules
from engine.adapter import build_bt_strategy, SELECTOR_REGISTRY, MANAGER_REGISTRY


class TestAdapter:
    @pytest.fixture
    def sample_prices(self):
        """200 days of fake price data for 5 tickers."""
        dates = pd.bdate_range("2025-01-01", periods=200)
        np.random.seed(42)
        tickers = ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]
        data = {}
        for t in tickers:
            base = np.random.uniform(100, 500)
            returns = np.random.normal(0.001, 0.02, 200)
            prices = base * np.cumprod(1 + returns)
            data[t] = prices
        return pd.DataFrame(data, index=dates)

    def _make_spec(self, selector: str, manager: str, **overrides) -> StrategySpec:
        """Helper to build a StrategySpec with defaults."""
        defaults = {
            "id": f"test-{selector}-{manager}",
            "name": f"Test {selector} {manager}",
            "universe": "sp500",
            "selector": selector,
            "manager": manager,
            "funding": FundingConfig(initial=10000),
            "dividends": "cash",
            "rules": StrategyRules(
                max_positions=5, max_position_pct=30, min_hold_days=3
            ),
        }
        defaults.update(overrides)
        return StrategySpec(**defaults)

    # ----- Registry tests -----

    def test_registries_populated(self):
        assert "golden-cross" in SELECTOR_REGISTRY
        assert "rsi-oversold" in SELECTOR_REGISTRY
        assert "dip-entry" in SELECTOR_REGISTRY
        assert "fear-greed" in SELECTOR_REGISTRY
        assert "data-follow" in SELECTOR_REGISTRY
        assert "earnings-beat" in SELECTOR_REGISTRY
        assert "sector-cycle" in SELECTOR_REGISTRY
        assert "random" in SELECTOR_REGISTRY
        assert "buy-and-hold" in SELECTOR_REGISTRY

    def test_manager_registry_populated(self):
        assert "equal-weight" in MANAGER_REGISTRY
        assert "grid-conservative" in MANAGER_REGISTRY
        assert "grid-aggressive" in MANAGER_REGISTRY
        assert "trailing-stop" in MANAGER_REGISTRY
        assert "scaled-exit" in MANAGER_REGISTRY
        assert "time-boxed" in MANAGER_REGISTRY
        assert "rebalance-monthly" in MANAGER_REGISTRY
        assert "volatility-sized" in MANAGER_REGISTRY
        assert "fixed-60-40" in MANAGER_REGISTRY

    # ----- Error handling -----

    def test_unknown_selector_raises(self, sample_prices):
        spec = StrategySpec(
            id="bad",
            name="Bad",
            universe="sp500",
            selector="nonexistent",
            manager="equal-weight",
            funding=FundingConfig(),
            dividends="cash",
            rules=StrategyRules(),
        )
        with pytest.raises(ValueError, match="No selector registered"):
            build_bt_strategy(spec, sample_prices)

    def test_unknown_manager_raises(self, sample_prices):
        spec = StrategySpec(
            id="bad",
            name="Bad",
            universe="sp500",
            selector="random",
            manager="nonexistent",
            funding=FundingConfig(),
            dividends="cash",
            rules=StrategyRules(),
        )
        with pytest.raises(ValueError, match="No manager registered"):
            build_bt_strategy(spec, sample_prices)

    # ----- Build + run tests for each selector -----

    def test_build_simple_strategy(self, sample_prices):
        spec = self._make_spec("random", "equal-weight")
        strategy = build_bt_strategy(spec, sample_prices)
        assert strategy.name == spec.id

    def test_build_and_run_backtest(self, sample_prices):
        spec = self._make_spec("random", "equal-weight")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(
            strategy, sample_prices, initial_capital=spec.funding.initial
        )
        result = bt.run(test)
        assert result.stats is not None

    def test_random_selector_is_reproducible(self, sample_prices):
        """Seeded random selection: two runs of the same spec+window match.

        Regression for the switch from bt.algos.SelectRandomly (numpy global
        RNG, non-reproducible) to SelectRandomlySeeded.
        """

        def _run():
            spec = self._make_spec("random", "equal-weight", id="repro-check")
            strategy = build_bt_strategy(spec, sample_prices)
            test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
            return float(bt.run(test).stats.loc["total_return", "repro-check"])

        assert _run() == _run()

    def test_golden_cross_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "golden-cross", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_rsi_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "rsi-oversold", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_dip_entry_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "dip-entry", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_fear_greed_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "fear-greed", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_data_follow_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "data-follow", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_earnings_beat_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "earnings-beat", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_sector_cycle_selector_runs(self, sample_prices):
        spec = self._make_spec(
            "sector-cycle", "equal-weight", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    # ----- Manager variations -----

    def test_grid_conservative_manager(self, sample_prices):
        spec = self._make_spec("random", "grid-conservative")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_grid_aggressive_manager(self, sample_prices):
        spec = self._make_spec("random", "grid-aggressive")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_trailing_stop_manager(self, sample_prices):
        spec = self._make_spec("random", "trailing-stop")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_scaled_exit_manager(self, sample_prices):
        spec = self._make_spec("random", "scaled-exit")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_time_boxed_manager(self, sample_prices):
        spec = self._make_spec("random", "time-boxed")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_rebalance_monthly_manager(self, sample_prices):
        spec = self._make_spec("random", "rebalance-monthly")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_volatility_sized_manager(self, sample_prices):
        spec = self._make_spec("random", "volatility-sized")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    # ----- Baseline selector/manager tests -----

    def test_buy_and_hold_selector_runs(self, sample_prices):
        spec = self._make_spec("buy-and-hold", "equal-weight")
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    def test_fixed_60_40_manager_runs(self):
        """fixed-60-40 manager requires VOO and BND columns."""
        dates = pd.bdate_range("2025-01-01", periods=200)
        np.random.seed(0)
        data = {}
        for t in ["VOO", "BND"]:
            base = np.random.uniform(100, 300)
            returns = np.random.normal(0.0005, 0.01, 200)
            data[t] = base * np.cumprod(1 + returns)
        prices = pd.DataFrame(data, index=dates)

        spec = self._make_spec(
            "buy-and-hold",
            "fixed-60-40",
            rules=StrategyRules(max_positions=2, max_position_pct=100, min_hold_days=1),
        )
        strategy = build_bt_strategy(spec, prices)
        test = bt.Backtest(strategy, prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    # ----- Cross-combination test -----

    def test_golden_cross_with_volatility_sizing(self, sample_prices):
        spec = self._make_spec(
            "golden-cross", "volatility-sized", rules=StrategyRules(max_positions=3)
        )
        strategy = build_bt_strategy(spec, sample_prices)
        test = bt.Backtest(strategy, sample_prices, initial_capital=10000)
        result = bt.run(test)
        assert result.stats is not None

    # ----- Coverage: all VALID_SELECTORS and VALID_MANAGERS have registrations -----

    def test_all_valid_selectors_registered(self):
        from engine.types import VALID_SELECTORS

        # claude-analysis is handled by a separate agent, not bt
        bt_selectors = VALID_SELECTORS - {"claude-analysis"}
        for sel in bt_selectors:
            assert sel in SELECTOR_REGISTRY, f"Selector {sel!r} not registered"

    def test_all_valid_managers_registered(self):
        from engine.types import VALID_MANAGERS

        for mgr in VALID_MANAGERS:
            assert mgr in MANAGER_REGISTRY, f"Manager {mgr!r} not registered"

    # ----- Strategy structure -----

    def test_strategy_has_correct_name(self, sample_prices):
        spec = self._make_spec("random", "equal-weight", id="my-custom-id")
        strategy = build_bt_strategy(spec, sample_prices)
        assert strategy.name == "my-custom-id"

    def test_limit_weights_uses_spec_value(self, sample_prices):
        """Verify the LimitWeights algo uses the spec's max_position_pct."""
        spec = self._make_spec(
            "random",
            "equal-weight",
            rules=StrategyRules(max_positions=5, max_position_pct=25),
        )
        strategy = build_bt_strategy(spec, sample_prices)
        # Find the LimitWeights algo in the pipeline
        limit_algos = [
            a for a in strategy.stack.algos if isinstance(a, bt.algos.LimitWeights)
        ]
        assert len(limit_algos) == 1
        assert limit_algos[0].limit == 0.25
