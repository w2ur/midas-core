"""Tests for scripts.build_tax_shadow — role-based exclusion of non-traders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from engine.config import get_config, reset_config_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaxShadowSkipsNonTradersByRole:
    """build_tax_shadow_all must process only agents in trading_roster.

    This means the-manager (role=allocator) and baseline-manager (not in
    roster at all) are excluded. The 10 traders ARE included.
    """

    def test_skips_allocator_by_role(self, midas_data_root, monkeypatch) -> None:
        """the-manager (role=allocator) is excluded even when trades.json exists."""
        from scripts.build_tax_shadow import build_tax_shadow_all

        portfolios_dir = midas_data_root / "data" / "portfolios"
        output_dir = midas_data_root / "data" / "tax_shadow"

        # Create the-manager dir with a valid (but empty) trades.json.
        manager_dir = portfolios_dir / "the-manager"
        manager_dir.mkdir(parents=True, exist_ok=True)
        (manager_dir / "trades.json").write_text(json.dumps([]))

        written = build_tax_shadow_all(
            portfolios_dir=portfolios_dir, output_dir=output_dir
        )
        assert "the-manager" not in written

    def test_skips_baseline_manager_by_role(self, midas_data_root, monkeypatch) -> None:
        """baseline-manager (not in roster) is excluded even when trades.json exists."""
        from scripts.build_tax_shadow import build_tax_shadow_all

        portfolios_dir = midas_data_root / "data" / "portfolios"
        output_dir = midas_data_root / "data" / "tax_shadow"

        baseline_dir = portfolios_dir / "baseline-manager"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "trades.json").write_text(json.dumps([]))

        written = build_tax_shadow_all(
            portfolios_dir=portfolios_dir, output_dir=output_dir
        )
        assert "baseline-manager" not in written

    def test_includes_trader_by_role(self, midas_data_root, monkeypatch) -> None:
        """A genuine trader (in trading_roster) with trades.json IS processed."""
        from scripts.build_tax_shadow import build_tax_shadow_all

        cfg = get_config()
        # Pick the first trader from the real roster.
        trader_id = cfg.trading_roster[0]

        portfolios_dir = midas_data_root / "data" / "portfolios"
        output_dir = midas_data_root / "data" / "tax_shadow"

        trader_dir = portfolios_dir / trader_id
        trader_dir.mkdir(parents=True, exist_ok=True)
        # Empty trades list — compute_tax_shadow returns zeroed ledger.
        (trader_dir / "trades.json").write_text(json.dumps([]))

        written = build_tax_shadow_all(
            portfolios_dir=portfolios_dir, output_dir=output_dir
        )
        assert trader_id in written

    def test_combined_dirs_only_trader_written(
        self, midas_data_root, monkeypatch
    ) -> None:
        """When all three dirs exist, only the trader appears in written list."""
        from scripts.build_tax_shadow import build_tax_shadow_all

        cfg = get_config()
        trader_id = cfg.trading_roster[0]

        portfolios_dir = midas_data_root / "data" / "portfolios"
        output_dir = midas_data_root / "data" / "tax_shadow"

        for agent_id in ("the-manager", "baseline-manager", trader_id):
            agent_dir = portfolios_dir / agent_id
            agent_dir.mkdir(parents=True, exist_ok=True)
            (agent_dir / "trades.json").write_text(json.dumps([]))

        written = build_tax_shadow_all(
            portfolios_dir=portfolios_dir, output_dir=output_dir
        )
        assert "the-manager" not in written
        assert "baseline-manager" not in written
        assert trader_id in written

    def test_novel_allocator_excluded_by_role(self, midas_data_root) -> None:
        """A NEW allocator not in _NON_AGENT_DIRS is excluded by role check.

        This is the distinguishing test between name-based and role-based
        exclusion. A name-based approach (_NON_AGENT_DIRS) cannot exclude
        "the-strategist" (not in the set); a role-based approach excludes it
        because its role == "allocator" and it is not in trading_roster.
        """
        from scripts.build_tax_shadow import build_tax_shadow_all

        # Inject a second allocator into the tmp roster.
        roster_path = midas_data_root / "roster.yaml"
        data = yaml.safe_load(roster_path.read_text(encoding="utf-8"))
        data["agents"]["the-strategist"] = {
            "display_name": "The Strategist",
            "role": "allocator",
            "allocator": {"channels_prefix": "strategist"},
        }
        roster_path.write_text(yaml.dump(data), encoding="utf-8")
        reset_config_cache()

        portfolios_dir = midas_data_root / "data" / "portfolios"
        output_dir = midas_data_root / "data" / "tax_shadow"

        strategist_dir = portfolios_dir / "the-strategist"
        strategist_dir.mkdir(parents=True, exist_ok=True)
        (strategist_dir / "trades.json").write_text(json.dumps([]))

        written = build_tax_shadow_all(
            portfolios_dir=portfolios_dir, output_dir=output_dir
        )
        assert "the-strategist" not in written
