"""Rebuild data/baselines/ for Day 1 → today.

Idempotent: always overwrites. Universe ticker lists and max_positions are
derived from the roster config at call time via resolve_agent_universe —
no hardcoded dicts.
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
