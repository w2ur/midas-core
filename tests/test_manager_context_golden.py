"""Golden characterization test — the rendered POLICY + RISK BUDGET blocks for
William's config must be byte-identical across the SP2 refactor.

The golden strings are captured from the pre-refactor render (module constants)
via:
    python -c "from datetime import date; from engine.manager_context import *; \
      print(render_manager_context(build_manager_context([],None,[],{},{},date(2026,7,1), \
      {'initial_capital':2000.0,'currency':'EUR'})))"

The golden is the arbiter of byte-identical parity — never edit it to match new
output. If a refactor changes the render, fix the code, not the golden.

Note on the RISK BUDGET extraction: the block starts with a header line followed
by a blank line, so a naive ``.split("\\n\\n", 1)[0]`` would capture only the
header. We split on the next section boundary (``\\n\\n===``) — with an empty
outcome memory + no triggers, RISK BUDGET is the final section, so this captures
the whole block including every bullet.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.config import get_config, reset_config_cache
from engine.manager_context import build_manager_context, render_manager_context

pytestmark = pytest.mark.live_cast


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def _ctx():
    from engine.manager_context import render_policy_prose, render_risk_budget_prose

    cfg = get_config()
    alloc = cfg.allocator_spec("the-manager")
    return build_manager_context(
        notes=[],
        portfolio=None,
        resolved_decisions=[],
        price_lookup={},
        ticker_registry={},
        as_of=date(2026, 7, 1),
        config={
            "initial_capital": 2000.0,
            "currency": "EUR",
            "policy_prose": render_policy_prose(
                cfg.jurisdiction, alloc.blocklist, alloc.policy_prose_override
            ),
            "risk_budget_prose": render_risk_budget_prose(
                alloc.risk_budget, "EUR", 2000.0
            ),
            "outcome_memory_same_max": alloc.outcome_memory_same_max,
            "outcome_memory_other_max": alloc.outcome_memory_other_max,
        },
    )


def test_policy_block_byte_identical():
    rendered = render_manager_context(_ctx())
    policy = rendered.split("=== POLICY ===\n", 1)[1].split("\n\n=== RISK BUDGET", 1)[0]
    assert policy == EXPECTED_POLICY


def test_risk_budget_block_byte_identical():
    rendered = render_manager_context(_ctx())
    rb = rendered.split("=== RISK BUDGET ===\n", 1)[1].split("\n\n===", 1)[0]
    assert rb == EXPECTED_RISK_BUDGET


# --- Goldens: the EXACT pre-refactor output (module constants), captured
# 2026-07-01 with the generator command in the module docstring.
EXPECTED_POLICY = """FEE AND TAX POLICY (French tax resident — these are GIVEN facts, not suggestions)

TAX
- PFU 30% flat on all realised gains (securities + crypto).
- Securities and crypto are SILOED regimes: losses never cross-subsidise gains in
  the other regime.
- No wash-sale rule in France: you may sell at a loss and immediately re-enter the
  same position.
- Securities losses must be declared the year they occur; 10-year carry-forward.
- Crypto: prefer crypto-to-crypto rebalancing (tax-free sursis under Art. 150 VH bis);
  when de-risking, park in stablecoins (USDC/USDT) — NOT EUR — to defer taxable events.
  Batch any EUR cash-out: each EUR realisation triggers a full-portfolio PVCT snapshot.
- The EUR 305 annual disposal threshold is irrelevant at this scale (blown on first sell).
- PEA ineligible for this universe — all accounts are CTO under full PFU.

FEES (round-trip cost the trade must exceed by ~2x to be worth doing)
- Equity/ETF (IBIE): ~0.05% + EUR 1.25/order floor.
- Crypto (Kraken): 0.40% taker / 0.25% maker.
- FX (OANDA Europe): ~0.002%.

PRIIPS BLOCKLIST — NOT buyable by EU retail via IBKR
These US-domiciled leveraged/inverse ETFs are blocked: SOXL, SPXS, SPXU, SQQQ, TQQQ, UPRO. Use UCITS substitutes (3USS.L, QQQS.L) or 1x inverse (SH, PSQ) only."""

EXPECTED_RISK_BUDGET = """RISK BUDGET (hard constraints — DEFAULT ACTION IS HOLD)

- Maximum open positions: 6
- Per-position cap: ~EUR 400 (≈25% of a ~EUR 2,000 book)
- Cash floor: EUR 150 must remain uninvested at all times
- Turnover limit: ≤2 trades per week
- Conviction threshold: conviction < 6 → do NOT trade; hold cash instead
- When in doubt, HOLD. A missed opportunity costs nothing; a bad trade costs capital."""
