"""Tests for scripts.daily_session.build_portfolio_summaries.

The orchestrator must use this helper to populate the bundle's `agents` map
for ALL 10 trading agents — running ones get fresh data, non-runners get
carry-forward portfolio state. Without it, weekend bundles ship with only
the running agents and the site dossier build fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.config import get_config
from engine.portfolio import PortfolioManager

pytestmark = pytest.mark.live_cast


@pytest.fixture
def tmp_portfolios(midas_data_root: Path) -> Path:
    # build_portfolio_summaries reads get_config().portfolios_dir; redirect the
    # data root (MIDAS_DATA_DIR) and seed/assert against that resolved dir.
    portfolios_dir = get_config().portfolios_dir
    portfolios_dir.mkdir(parents=True, exist_ok=True)
    return portfolios_dir


class TestBuildPortfolioSummaries:
    def test_returns_summary_for_every_existing_portfolio(
        self, tmp_portfolios: Path
    ) -> None:
        from scripts.daily_session import build_portfolio_summaries

        pm = PortfolioManager(base_dir=tmp_portfolios)
        # Seed 3 of the 10 agents (e.g. weekend roster).
        pm.initialize("satoshi", 10000.0, currency="EUR")
        pm.initialize("yolo-sapiens-eur", 10000.0, currency="EUR")
        pm.initialize("yolo-sapiens-usd", 10000.0, currency="USD")

        summaries = build_portfolio_summaries()

        assert set(summaries.keys()) == {
            "satoshi",
            "yolo-sapiens-eur",
            "yolo-sapiens-usd",
        }
        for aid, s in summaries.items():
            assert set(s.keys()) == {"cash", "deployed", "positions", "currency"}
            assert s["cash"] == 10000.0
            assert s["deployed"] == 0.0
            assert s["positions"] == []

    def test_skips_agents_with_no_portfolio_json(self, tmp_portfolios: Path) -> None:
        """Defensive — production has all 10 portfolios on disk, but the helper
        must not crash if one is missing."""
        from scripts.daily_session import build_portfolio_summaries

        pm = PortfolioManager(base_dir=tmp_portfolios)
        pm.initialize("satoshi", 10000.0, currency="EUR")

        summaries = build_portfolio_summaries()
        assert "satoshi" in summaries
        # The other 9 are absent on disk → not in the dict.
        for aid in get_config().trading_roster:
            if aid != "satoshi":
                assert aid not in summaries

    def test_only_iterates_canonical_roster(self, tmp_portfolios: Path) -> None:
        """If a stray directory exists in data/portfolios that's not in the
        canonical roster (e.g. a deprecated agent), it must NOT appear in the
        summaries — bundle keys are bounded by ROSTER, not the filesystem."""
        from scripts.daily_session import build_portfolio_summaries

        pm = PortfolioManager(base_dir=tmp_portfolios)
        pm.initialize("satoshi", 10000.0, currency="EUR")
        pm.initialize("retired-agent-from-2025", 5000.0, currency="EUR")

        summaries = build_portfolio_summaries()
        assert "satoshi" in summaries
        assert "retired-agent-from-2025" not in summaries
