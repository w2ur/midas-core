"""Integration test: demo-allocator proves the opt-in allocator role is forkable.

Exercises the FORKER paths that William's live config does not:
- USD home_currency  → render_risk_budget_prose non-EUR fallback branch
- No policy.prose_override → generic render_policy_prose template
- baseline.enabled: false → D5 gate skips baseline-manager creation
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from engine.config import get_config, reset_config_cache

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo-desk"


@pytest.fixture(autouse=True)
def _demo_env(monkeypatch):
    monkeypatch.setenv("MIDAS_DATA_DIR", str(DEMO))
    reset_config_cache()
    yield
    reset_config_cache()


def test_demo_has_one_allocator_not_the_manager():
    cfg = get_config()
    assert cfg.allocators == ("demo-allocator",)
    assert "the-manager" not in cfg.roster
    assert "demo-allocator" not in cfg.trading_roster


def test_demo_allocator_spec_defaults():
    alloc = get_config().allocator_spec("demo-allocator")
    assert alloc.channels_prefix == "demo-allocator"
    assert alloc.risk_budget.min_conviction >= 0


def test_demo_jurisdiction_is_neutral():
    assert (
        get_config().jurisdiction.tax_rate_pct == 0.0
    )  # no FR specifics leak into core demo


def test_non_eur_render_risk_budget_fallback():
    """USD home_currency exercises the generic else branch in render_risk_budget_prose."""
    from engine.manager_context import render_risk_budget_prose

    rb = get_config().allocator_spec("demo-allocator").risk_budget
    prose = render_risk_budget_prose(rb, "USD", 2000.0)
    assert "USD" in prose
    assert "≈25% of a ~EUR 2,000 book" not in prose


def test_generic_policy_prose_no_fr_tax():
    """No prose_override and 0% tax → generic paper-trading template, not French tax prose."""
    from engine.manager_context import render_policy_prose

    jur = get_config().jurisdiction
    prose = render_policy_prose(jur, (), None)
    assert "Paper trading" in prose
    assert "30%" not in prose


def test_d5_baseline_disabled(tmp_path):
    """baseline.enabled: false → step_build_baseline_manager creates no baseline-manager book."""
    from scripts.daily_session import step_build_baseline_manager

    portfolios_dir = tmp_path / "portfolios"
    step_build_baseline_manager(
        {},
        datetime.date(2026, 7, 1),
        portfolios_dir=portfolios_dir,
    )
    assert not (portfolios_dir / "baseline-manager").exists()
