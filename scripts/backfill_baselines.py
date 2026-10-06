"""Refresh data/baselines/ for Day 1 → today.

Passive benchmarks and the global reference are append-or-keep
(engine.baselines.merge_baseline_series): an already-published date is kept
as-is and a mismatch is classified, new dates are appended. Pass
``restate_series={...}`` (with a ``changelog_entry``) to build_all_baselines
directly for a deliberate, publicly logged restatement, which rewrites only
the scoped published rows and appends nothing — this script does not
expose that on the CLI, it is a Python-level escape hatch, not a routine one. Each coin flip is
advanced from its persisted state (``data/baselines/<agent>/state/
coinflip.json``) over new dates only and can never be restated: a scope
naming it is refused (plan 1.6). Universe ticker lists and max_positions are
derived from the roster config at call time via resolve_agent_universe —
no hardcoded dicts; they are also what ``scripts/init_coinflip_state.py``
repicks with.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.baselines import build_all_baselines
from engine.config import get_config, resolve_agent_universe


def _universes_by_agent() -> dict[str, list[str]]:
    """Build {agent_id: [ticker, ...]} for every trading agent from config."""
    cfg = get_config()
    return {aid: resolve_agent_universe(cfg.roster[aid]) for aid in cfg.trading_roster}


def _max_positions_by_agent() -> dict[str, int]:
    """Build {agent_id: max_positions} for every trading agent from config."""
    cfg = get_config()
    return {aid: cfg.roster[aid].max_positions for aid in cfg.trading_roster}


def main() -> None:
    today = date.today()
    cfg = get_config()
    build_all_baselines(
        universes_by_agent=_universes_by_agent(),
        from_date=cfg.day_one,
        to_date=today,
        max_positions_by_agent=_max_positions_by_agent(),
    )
    print(f"Baselines written to data/baselines/ for {cfg.day_one} → {today}")


if __name__ == "__main__":
    main()
