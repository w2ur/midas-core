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
        # Exact, not "within 1000". The series opens before any trade, so the
        # first value IS the capital; a 2% tolerance would pass a run that
        # started 900 short. See TestBacktestValues for the general case.
        assert result.daily_values.iloc[0] == pytest.approx(50000)


class TestBacktestValues:
    """Value assertions, not shape assertions.

    Everything above checks that fields are populated and non-None, which a
    runner returning the wrong number passes. These cross-check the reported
    metrics against the one series they must both derive from — `daily_values`
    comes from `strategy.values` while `total_return` and `max_drawdown` come
    from bt's `stats` table, so agreement between them is a real check on two
    independently-sourced numbers rather than a restatement of one.
    """

    @pytest.fixture
    def sample_prices(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        np.random.seed(42)
        data = {}
        for t in ["AAPL", "MSFT", "NVDA", "GOOG", "AMZN"]:
            base = np.random.uniform(100, 500)
            returns = np.random.normal(0.001, 0.02, 300)
            data[t] = base * np.cumprod(1 + returns)
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def spec_dict(self):
        return {
            "id": "test-values",
            "name": "Test Values",
            "universe": "sp500",
            "selector": "golden-cross",
            "manager": "equal-weight",
            "funding": {"initial": 10000},
            "dividends": "cash",
            "rules": {"maxPositions": 3, "maxPositionPct": 40, "minHoldDays": 3},
        }

    def test_total_return_reproduces_from_the_daily_series(
        self, sample_prices, spec_dict
    ):
        result = run_backtest(spec_dict, sample_prices)
        values = result.daily_values
        derived = values.iloc[-1] / values.iloc[0] - 1
        assert result.total_return == pytest.approx(derived, rel=1e-9)

    def test_max_drawdown_reproduces_from_the_daily_series(
        self, sample_prices, spec_dict
    ):
        result = run_backtest(spec_dict, sample_prices)
        values = result.daily_values
        derived = (values / values.cummax() - 1).min()
        assert result.max_drawdown == pytest.approx(derived, rel=1e-9)

    def test_max_drawdown_is_a_loss_and_total_return_is_signed_correctly(
        self, sample_prices, spec_dict
    ):
        """Sign errors survive every `is not None` assertion."""
        result = run_backtest(spec_dict, sample_prices)
        assert result.max_drawdown <= 0.0
        values = result.daily_values
        assert (result.total_return > 0) == (values.iloc[-1] > values.iloc[0])

    def test_the_series_starts_at_exactly_the_initial_capital(
        self, sample_prices, spec_dict
    ):
        """Was asserted as `abs(first - 50000) < 1000` — a 2% tolerance."""
        for capital in (10_000, 50_000, 250_000):
            result = run_backtest(spec_dict, sample_prices, initial_capital=capital)
            assert result.daily_values.iloc[0] == pytest.approx(capital)

    def test_initial_capital_falls_back_to_the_spec_when_omitted(
        self, sample_prices, spec_dict
    ):
        result = run_backtest(spec_dict, sample_prices)
        assert result.daily_values.iloc[0] == pytest.approx(
            spec_dict["funding"]["initial"]
        )

    def test_return_is_not_invariant_to_initial_capital(self, sample_prices, spec_dict):
        """`bt.Backtest` defaults to integer_positions=True.

        Share counts round down, so the rounding residue — and therefore the
        return — depends on how much capital there is to round. Documented
        here because it is surprising, it is the same mechanism that makes the
        coin-flip control scale-sensitive, and a future switch to
        `integer_positions=False` would silently change published behaviour.
        """
        small = run_backtest(spec_dict, sample_prices, initial_capital=10_000)
        large = run_backtest(spec_dict, sample_prices, initial_capital=250_000)
        assert small.total_return != pytest.approx(large.total_return, rel=1e-6)

    def test_snapshots_carry_the_daily_series_verbatim(self, sample_prices, spec_dict):
        result = run_backtest(spec_dict, sample_prices)
        snapshots = result.to_snapshots()
        assert len(snapshots) == len(result.daily_values)
        for snapshot, (index, value) in zip(snapshots, result.daily_values.items()):
            assert snapshot["date"] == index.date().isoformat()
            assert snapshot["portfolioValue"] == pytest.approx(float(value))

    def test_snapshot_dates_are_iso_and_strictly_increasing(
        self, sample_prices, spec_dict
    ):
        dates = [
            s["date"] for s in run_backtest(spec_dict, sample_prices).to_snapshots()
        ]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)
        assert all(len(d) == 10 and d[4] == "-" and d[7] == "-" for d in dates)

    def test_camelcase_and_snake_case_rules_produce_identical_results(
        self, sample_prices, spec_dict
    ):
        """`_normalise_spec_dict` accepts both; nothing checked they agree."""
        snake = {
            **spec_dict,
            "rules": {"max_positions": 3, "max_position_pct": 40, "min_hold_days": 3},
        }
        camel = run_backtest(spec_dict, sample_prices)
        converted = run_backtest(snake, sample_prices)
        assert camel.total_return == pytest.approx(converted.total_return)
        assert list(camel.daily_values) == pytest.approx(list(converted.daily_values))

    def test_run_batch_preserves_order_and_ids(self, sample_prices, spec_dict):
        specs = [
            {**spec_dict, "id": "first"},
            {**spec_dict, "id": "second", "selector": "random"},
            {**spec_dict, "id": "third"},
        ]
        results = run_batch(specs, sample_prices)
        assert [r.strategy_id for r in results] == ["first", "second", "third"]

    def test_run_batch_skips_a_failing_strategy_and_keeps_the_rest(
        self, sample_prices, spec_dict, capsys
    ):
        """Documented tolerance, never exercised."""
        specs = [
            {**spec_dict, "id": "good-one"},
            {**spec_dict, "id": "broken", "selector": "no-such-selector"},
            {**spec_dict, "id": "good-two"},
        ]
        results = run_batch(specs, sample_prices)
        assert [r.strategy_id for r in results] == ["good-one", "good-two"]
        assert "broken" in capsys.readouterr().out

    def test_a_batch_result_equals_the_same_spec_run_alone(
        self, sample_prices, spec_dict
    ):
        """Batching must not perturb a strategy — the specs share price data."""
        alone = run_backtest(spec_dict, sample_prices)
        (batched,) = run_batch([spec_dict], sample_prices)
        assert batched.total_return == pytest.approx(alone.total_return)
        assert batched.max_drawdown == pytest.approx(alone.max_drawdown)


class TestBacktestIsGrossOfCosts:
    """The warning is a published claim; pin that it still describes the code."""

    def test_the_warning_constant_says_what_the_runner_does(self):
        from engine.backtest import GROSS_OF_COSTS_WARNING

        assert "GROSS_OF_COSTS" in GROSS_OF_COSTS_WARNING
        assert "fee" in GROSS_OF_COSTS_WARNING.lower()

    def test_the_runner_passes_no_commission_model_to_bt(self):
        """If a fee model is ever wired in, this test must be updated with it.

        A backtest that silently started modelling costs while METHODOLOGY
        still said it did not would be a published-claim divergence, which is
        the class this repo treats most seriously.
        """
        import inspect

        from engine import backtest as module

        source = inspect.getsource(module.run_backtest)
        assert "commissions" not in source
