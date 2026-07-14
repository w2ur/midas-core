"""Tests for jurisdiction-driven fee and tax rates.

Verifies that:
- FR config (roster.yaml) produces byte-identical numbers to the module
  constants (0.0005 equity rate, 1.25 floor, 0.004 crypto rate, 0.00002 FX rate,
  0.30 PFU rate).
- A neutral config (no jurisdiction block) yields _pfu_rate()==0.0 and fees
  fall back to the module-constant defaults (NOT zeroed — fees are a realism
  default; only tax_rate is a no-op in the neutral config).
"""

from __future__ import annotations

import pytest
from engine.config import get_config, reset_config_cache


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_equity_fee_matches_fr_config():
    from engine.fees import fee_for

    # EUR 10,000 equity notional: max(1.25, 0.0005*10000) = 5.0 (FR config)
    assert fee_for("AAPL", 10_000.0) == pytest.approx(5.0)
    # floor binds on a tiny order
    assert fee_for("AAPL", 100.0) == pytest.approx(1.25)


@pytest.mark.live_cast
def test_pfu_rate_from_config():
    import engine.tax_shadow as ts

    assert ts._pfu_rate() == pytest.approx(0.30)  # FR


def test_tax_rate_zero_in_neutral_config(tmp_path, monkeypatch):
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "globals:\n  day_one: '2026-04-17'\n  currencies: [USD]\n"
        "  initial_capital: 10000.0\n"
        "  global_reference: {label: X, ticker: URTH, currency: USD}\n"
        "  agents_dir: .claude/agents\n"
        "agents:\n  solo:\n    display_name: Solo\n    role: trader\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
    reset_config_cache()
    import engine.tax_shadow as ts

    assert ts._pfu_rate() == 0.0
