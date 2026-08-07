"""Tests for strategy spec JSON files in data/strategies/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.types import StrategySpec

STRATEGIES_DIR = Path(__file__).resolve().parents[1] / "data" / "strategies"


# ---------------------------------------------------------------------------
# Bulk validity tests
# ---------------------------------------------------------------------------


def test_all_specs_are_valid():
    """Every JSON file in data/strategies/ must parse without error and have
    an id matching the filename stem."""
    spec_files = list(STRATEGIES_DIR.glob("*.json"))
    assert len(spec_files) >= 15, (
        f"Expected at least 15 strategy specs, found {len(spec_files)}"
    )
    for path in spec_files:
        data = json.loads(path.read_text(encoding="utf-8"))
        spec = StrategySpec.from_dict(data)  # Should not raise
        assert spec.id == path.stem, (
            f"{path.name}: id field {spec.id!r} does not match filename stem {path.stem!r}"
        )


def test_coin_flip_baseline_exists():
    """The null-hypothesis baseline strategy must exist."""
    path = STRATEGIES_DIR / "coin-flip-baseline.json"
    assert path.exists(), "coin-flip-baseline.json is missing from data/strategies/"


# ---------------------------------------------------------------------------
# Individual spec property tests
# ---------------------------------------------------------------------------


class TestGoldenCrossSP500:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "golden-cross-sp500.json")

    def test_universe(self):
        assert self._spec().universe == "sp500"

    def test_selector(self):
        assert self._spec().selector == "golden-cross"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_funding(self):
        spec = self._spec()
        assert spec.funding.initial == 10000
        assert spec.funding.monthly_addition == 500

    def test_dividends(self):
        assert self._spec().dividends == "reinvest"


class TestRSIContrarianSP500:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "rsi-contrarian-sp500.json")

    def test_universe(self):
        assert self._spec().universe == "sp500"

    def test_selector(self):
        assert self._spec().selector == "rsi-oversold"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_dividends(self):
        assert self._spec().dividends == "reinvest"


class TestBuyTheDipConservative:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "buy-the-dip-conservative.json")

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_selector(self):
        assert self._spec().selector == "dip-entry"


class TestBuyTheDipAggressive:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "buy-the-dip-aggressive.json")

    def test_manager(self):
        assert self._spec().manager == "grid-aggressive"

    def test_selector(self):
        assert self._spec().selector == "dip-entry"


class TestPelosiTracker:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "pelosi-tracker.json")

    def test_universe(self):
        assert self._spec().universe == "congress"

    def test_selector(self):
        assert self._spec().selector == "data-follow"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_dividends(self):
        assert self._spec().dividends == "cash"

    def test_max_positions(self):
        assert self._spec().rules.max_positions == 15


class TestInsiderShadow:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "insider-shadow.json")

    def test_universe(self):
        assert self._spec().universe == "insiders"

    def test_selector(self):
        assert self._spec().selector == "data-follow"

    def test_dividends(self):
        assert self._spec().dividends == "cash"


class TestDogsOfTheDow:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "dogs-of-the-dow.json")

    def test_universe(self):
        assert self._spec().universe == "dow30"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_dividends(self):
        assert self._spec().dividends == "reinvest"


class TestFearGreed:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "fear-greed.json")

    def test_universe(self):
        assert self._spec().universe == "etf-broad"

    def test_selector(self):
        assert self._spec().selector == "fear-greed"

    def test_manager(self):
        assert self._spec().manager == "volatility-sized"

    def test_dividends(self):
        assert self._spec().dividends == "cash"


class TestDividendAristocratsDRIP:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "dividend-aristocrats-drip.json")

    def test_universe(self):
        assert self._spec().universe == "dividend-aristocrats"

    def test_dividends(self):
        assert self._spec().dividends == "reinvest"


class TestCoinFlipBaseline:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "coin-flip-baseline.json")

    def test_selector_is_random(self):
        assert self._spec().selector == "random"

    def test_no_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 0

    def test_dividends_cash(self):
        assert self._spec().dividends == "cash"


class TestGoldenCrossDCA:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "golden-cross-dca.json")

    def test_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 500

    def test_same_core_as_sp500(self):
        spec = self._spec()
        assert spec.universe == "sp500"
        assert spec.selector == "golden-cross"
        assert spec.manager == "equal-weight"


class TestGoldenCrossLump:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "golden-cross-lump.json")

    def test_no_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 0

    def test_same_core_as_dca(self):
        spec = self._spec()
        assert spec.universe == "sp500"
        assert spec.selector == "golden-cross"
        assert spec.manager == "equal-weight"


class TestBaselineVOOHold:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "baseline-voo-hold.json")

    def test_universe(self):
        assert self._spec().universe == "single-voo"

    def test_selector(self):
        assert self._spec().selector == "buy-and-hold"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_no_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 0

    def test_dividends_reinvest(self):
        assert self._spec().dividends == "reinvest"

    def test_single_position(self):
        assert self._spec().rules.max_positions == 1


class TestBaselineEqualWeight:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "baseline-equal-weight.json")

    def test_universe(self):
        assert self._spec().universe == "etf-broad"

    def test_selector(self):
        assert self._spec().selector == "buy-and-hold"

    def test_manager(self):
        assert self._spec().manager == "equal-weight"

    def test_no_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 0

    def test_dividends_reinvest(self):
        assert self._spec().dividends == "reinvest"

    def test_max_positions(self):
        assert self._spec().rules.max_positions == 10


class TestBaseline6040:
    def _spec(self) -> StrategySpec:
        return StrategySpec.from_json(STRATEGIES_DIR / "baseline-60-40.json")

    def test_universe(self):
        assert self._spec().universe == "classic-60-40"

    def test_selector(self):
        assert self._spec().selector == "buy-and-hold"

    def test_manager(self):
        assert self._spec().manager == "fixed-60-40"

    def test_no_monthly_addition(self):
        assert self._spec().funding.monthly_addition == 0

    def test_dividends_reinvest(self):
        assert self._spec().dividends == "reinvest"

    def test_two_positions(self):
        assert self._spec().rules.max_positions == 2
