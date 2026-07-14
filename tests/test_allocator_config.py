"""SP2 — allocator role config: the-manager as role: allocator + jurisdiction."""

from __future__ import annotations

import pytest

from engine.config import get_config, reset_config_cache

pytestmark = pytest.mark.live_cast


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


class TestAllocatorRoster:
    def test_allocators_tuple(self):
        assert get_config().allocators == ("the-manager",)

    def test_manager_excluded_from_trading_roster(self):
        assert "the-manager" not in get_config().trading_roster

    def test_manager_identity(self):
        spec = get_config().roster["the-manager"]
        assert spec.role == "allocator"
        assert spec.home_currency == "EUR"
        assert spec.initial_capital == 2000.0
        assert spec.persona == "the-manager.md"

    def test_manager_safety_matches_legacy_defaults(self):
        # Parity: the-manager previously fell through AgentConfig.load to
        # 500 / 5 / -5.0 (commit 320e0d53). The roster block must reproduce it.
        s = get_config().roster["the-manager"].safety
        assert (
            s.max_order_notional,
            s.max_orders_per_day,
            s.daily_drawdown_halt_pct,
        ) == (
            500.0,
            5,
            -5.0,
        )

    def test_allocator_risk_budget(self):
        rb = get_config().allocator_spec("the-manager").risk_budget
        assert rb.max_positions == 6
        assert rb.per_position_cap == 400.0
        assert rb.cash_floor == 150.0
        assert rb.max_trades_per_week == 2
        assert rb.min_conviction == 6

    def test_allocator_blocklist(self):
        alloc = get_config().allocator_spec("the-manager")
        assert set(alloc.blocklist) == {"SQQQ", "SPXS", "SPXU", "TQQQ", "UPRO", "SOXL"}

    def test_allocator_channels_and_memory(self):
        alloc = get_config().allocator_spec("the-manager")
        assert alloc.channels_prefix == "manager"
        assert alloc.outcome_resolution_days == 10
        assert alloc.outcome_memory_same_max == 5
        assert alloc.outcome_memory_other_max == 3
        assert alloc.baseline_enabled is True

    def test_jurisdiction_fr(self):
        j = get_config().jurisdiction
        assert j.tax_rate_pct == 30.0
        assert j.fees["equity"]["rate_pct"] == 0.05
        assert j.fees["equity"]["floor"] == 1.25
        assert j.fees["crypto"]["taker_pct"] == 0.40
        assert j.fees["fx"]["spread_pct"] == 0.002


class TestNoAllocator:
    def test_absent_allocator_block(self, tmp_path, monkeypatch):
        # A roster with no allocator yields an empty tuple.
        roster = tmp_path / "roster.yaml"
        roster.write_text(
            "globals:\n"
            "  day_one: '2026-04-17'\n"
            "  currencies: [EUR]\n"
            "  initial_capital: 10000.0\n"
            "  global_reference: {label: MSCI World, ticker: URTH, currency: EUR}\n"
            "  agents_dir: .claude/agents\n"
            "agents:\n"
            "  solo:\n"
            "    display_name: Solo\n"
            "    role: trader\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
        reset_config_cache()
        assert get_config().allocators == ()
        assert get_config().jurisdiction.tax_rate_pct == 0.0  # core no-op default
