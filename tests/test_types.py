"""Tests for engine/types.py — core data types for the Midas trading system."""

import json
import pytest
from datetime import date, datetime
from pathlib import Path

from engine.types import (
    Trade,
    Position,
    BenchmarkValues,
    DailySnapshot,
    FundingConfig,
    StrategyRules,
    Portfolio,
    StrategySpec,
)


# ---------------------------------------------------------------------------
# Trade
# ---------------------------------------------------------------------------

class TestTrade:
    def test_buy_trade_creation(self):
        t = Trade(
            id="t001",
            timestamp=datetime(2024, 1, 15, 10, 30),
            action="BUY",
            ticker="AAPL",
            shares=10.0,
            price=185.0,
            total=1850.0,
            fees=1.0,
            reasoning="Golden cross signal",
        )
        assert t.id == "t001"
        assert t.action == "BUY"
        assert t.ticker == "AAPL"
        assert t.shares == 10.0
        assert t.price == 185.0
        assert t.total == 1850.0
        assert t.fees == 1.0
        assert t.reasoning == "Golden cross signal"

    def test_sell_trade_creation(self):
        t = Trade(
            id="t002",
            timestamp=datetime(2024, 2, 1, 14, 0),
            action="SELL",
            ticker="MSFT",
            shares=5.0,
            price=400.0,
            total=2000.0,
            fees=1.5,
            reasoning="Trailing stop triggered",
        )
        assert t.action == "SELL"
        assert t.ticker == "MSFT"
        assert t.total == 2000.0


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_position_creation(self):
        p = Position(
            ticker="AAPL",
            shares=10.0,
            avg_cost=185.0,
            date_opened=date(2024, 1, 15),
            grid_level=0,
        )
        assert p.ticker == "AAPL"
        assert p.shares == 10.0
        assert p.avg_cost == 185.0
        assert p.grid_level == 0

    def test_cost_basis_property(self):
        p = Position(
            ticker="AAPL",
            shares=10.0,
            avg_cost=185.0,
            date_opened=date(2024, 1, 15),
            grid_level=0,
        )
        assert p.cost_basis == pytest.approx(1850.0)

    def test_cost_basis_fractional_shares(self):
        p = Position(
            ticker="BTC-USD",
            shares=0.5,
            avg_cost=60000.0,
            date_opened=date(2024, 3, 1),
            grid_level=1,
        )
        assert p.cost_basis == pytest.approx(30000.0)

    def test_zero_shares_rejected(self):
        with pytest.raises(ValueError, match="shares"):
            Position(
                ticker="AAPL",
                shares=0.0,
                avg_cost=185.0,
                date_opened=date(2024, 1, 15),
                grid_level=0,
            )

    def test_negative_shares_rejected(self):
        with pytest.raises(ValueError, match="shares"):
            Position(
                ticker="AAPL",
                shares=-5.0,
                avg_cost=185.0,
                date_opened=date(2024, 1, 15),
                grid_level=0,
            )


# ---------------------------------------------------------------------------
# BenchmarkValues
# ---------------------------------------------------------------------------

class TestBenchmarkValues:
    def test_benchmark_creation(self):
        b = BenchmarkValues(sp500=5000.0, msci_world=3200.0, gold=2100.0, btc=65000.0)
        assert b.sp500 == 5000.0
        assert b.msci_world == 3200.0
        assert b.gold == 2100.0
        assert b.btc == 65000.0


# ---------------------------------------------------------------------------
# DailySnapshot
# ---------------------------------------------------------------------------

class TestDailySnapshot:
    def test_daily_snapshot_creation(self):
        benchmarks = BenchmarkValues(sp500=5000.0, msci_world=3200.0, gold=2100.0, btc=65000.0)
        snap = DailySnapshot(
            date=date(2024, 1, 15),
            portfolio_value=12000.0,
            cash=2000.0,
            positions_value=10000.0,
            benchmarks=benchmarks,
        )
        assert snap.portfolio_value == 12000.0
        assert snap.cash == 2000.0
        assert snap.benchmarks.sp500 == 5000.0


# ---------------------------------------------------------------------------
# FundingConfig
# ---------------------------------------------------------------------------

class TestFundingConfig:
    def test_defaults(self):
        fc = FundingConfig()
        assert fc.initial == 10000.0
        assert fc.monthly_addition == 0.0
        assert fc.weekly_addition == 0.0

    def test_custom_values(self):
        fc = FundingConfig(initial=25000.0, monthly_addition=500.0)
        assert fc.initial == 25000.0
        assert fc.monthly_addition == 500.0
        assert fc.weekly_addition == 0.0


# ---------------------------------------------------------------------------
# StrategyRules
# ---------------------------------------------------------------------------

class TestStrategyRules:
    def test_defaults(self):
        sr = StrategyRules()
        assert sr.max_positions == 10
        assert sr.max_position_pct == 20.0
        assert sr.min_hold_days == 3

    def test_custom_values(self):
        sr = StrategyRules(max_positions=5, max_position_pct=15.0, min_hold_days=7)
        assert sr.max_positions == 5


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def _make_positions(self):
        return [
            Position(ticker="AAPL", shares=10.0, avg_cost=185.0,
                     date_opened=date(2024, 1, 15), grid_level=0),
            Position(ticker="MSFT", shares=5.0, avg_cost=400.0,
                     date_opened=date(2024, 1, 20), grid_level=0),
        ]

    def test_portfolio_with_positions(self):
        positions = self._make_positions()
        p = Portfolio(cash=3000.0, positions=positions, last_updated=date(2024, 2, 1))
        assert p.cash == 3000.0
        assert len(p.positions) == 2

    def test_cost_basis_with_positions(self):
        positions = self._make_positions()
        p = Portfolio(cash=3000.0, positions=positions, last_updated=date(2024, 2, 1))
        # AAPL: 10 * 185 = 1850, MSFT: 5 * 400 = 2000 → total = 3850
        assert p.cost_basis == pytest.approx(3850.0)

    def test_cost_basis_empty_portfolio(self):
        p = Portfolio(cash=10000.0, positions=[], last_updated=date(2024, 1, 1))
        assert p.cost_basis == 0.0

    def test_to_dict_roundtrip(self):
        positions = self._make_positions()
        original = Portfolio(cash=3000.0, positions=positions, last_updated=date(2024, 2, 1))
        d = original.to_dict()
        restored = Portfolio.from_dict(d)

        assert restored.cash == original.cash
        assert restored.last_updated == original.last_updated
        assert len(restored.positions) == len(original.positions)
        assert restored.positions[0].ticker == "AAPL"
        assert restored.positions[0].shares == 10.0
        assert restored.positions[0].avg_cost == 185.0
        assert restored.positions[1].ticker == "MSFT"

    def test_to_dict_is_json_serializable(self):
        positions = self._make_positions()
        p = Portfolio(cash=3000.0, positions=positions, last_updated=date(2024, 2, 1))
        d = p.to_dict()
        # Must not raise
        json_str = json.dumps(d)
        assert "AAPL" in json_str

    def test_from_dict_empty_positions(self):
        p = Portfolio(cash=10000.0, positions=[], last_updated=date(2024, 1, 1))
        restored = Portfolio.from_dict(p.to_dict())
        assert restored.cash == 10000.0
        assert restored.positions == []


# ---------------------------------------------------------------------------
# StrategySpec
# ---------------------------------------------------------------------------

VALID_SPEC_DICT = {
    "id": "golden-dow30",
    "name": "Golden Cross on Dow 30",
    "universe": "dow30",
    "selector": "golden-cross",
    "manager": "equal-weight",
    "funding": {
        "initial": 10000.0,
        "monthly_addition": 500.0,
        "weekly_addition": 0.0,
    },
    "dividends": "reinvest",
    "rules": {
        "max_positions": 10,
        "max_position_pct": 20.0,
        "min_hold_days": 3,
    },
}


class TestStrategySpec:
    def test_from_dict_valid(self):
        spec = StrategySpec.from_dict(VALID_SPEC_DICT)
        assert spec.id == "golden-dow30"
        assert spec.name == "Golden Cross on Dow 30"
        assert spec.universe == "dow30"
        assert spec.selector == "golden-cross"
        assert spec.manager == "equal-weight"
        assert spec.dividends == "reinvest"
        assert spec.funding.initial == 10000.0
        assert spec.funding.monthly_addition == 500.0
        assert spec.rules.max_positions == 10

    def test_from_dict_funding_defaults(self):
        d = {**VALID_SPEC_DICT, "funding": {"initial": 5000.0}}
        spec = StrategySpec.from_dict(d)
        assert spec.funding.initial == 5000.0
        assert spec.funding.monthly_addition == 0.0
        assert spec.funding.weekly_addition == 0.0

    def test_from_dict_rules_defaults(self):
        d = {**VALID_SPEC_DICT}
        del d["rules"]
        spec = StrategySpec.from_dict(d)
        assert spec.rules.max_positions == 10
        assert spec.rules.max_position_pct == 20.0
        assert spec.rules.min_hold_days == 3

    def test_invalid_universe_rejected(self):
        d = {**VALID_SPEC_DICT, "universe": "not-a-real-universe"}
        with pytest.raises(ValueError, match="universe"):
            StrategySpec.from_dict(d)

    def test_invalid_selector_rejected(self):
        d = {**VALID_SPEC_DICT, "selector": "magic-selector"}
        with pytest.raises(ValueError, match="selector"):
            StrategySpec.from_dict(d)

    def test_invalid_manager_rejected(self):
        d = {**VALID_SPEC_DICT, "manager": "yolo-manager"}
        with pytest.raises(ValueError, match="manager"):
            StrategySpec.from_dict(d)

    def test_from_json(self, tmp_path):
        spec_file = tmp_path / "golden-dow30.json"
        spec_file.write_text(json.dumps(VALID_SPEC_DICT))
        spec = StrategySpec.from_json(spec_file)
        assert spec.id == "golden-dow30"
        assert spec.universe == "dow30"

    def test_all_valid_universes_accepted(self):
        from engine.types import VALID_UNIVERSES
        for universe in VALID_UNIVERSES:
            d = {**VALID_SPEC_DICT, "universe": universe}
            spec = StrategySpec.from_dict(d)
            assert spec.universe == universe

    def test_all_valid_selectors_accepted(self):
        from engine.types import VALID_SELECTORS
        for selector in VALID_SELECTORS:
            d = {**VALID_SPEC_DICT, "selector": selector}
            spec = StrategySpec.from_dict(d)
            assert spec.selector == selector

    def test_all_valid_managers_accepted(self):
        from engine.types import VALID_MANAGERS
        for manager in VALID_MANAGERS:
            d = {**VALID_SPEC_DICT, "manager": manager}
            spec = StrategySpec.from_dict(d)
            assert spec.manager == manager
