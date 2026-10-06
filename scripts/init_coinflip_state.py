#!/usr/bin/env python3
"""One-off migration (plan 1.6, 2026-10-05): give every coin flip a state.

Until 1.6 each coin flip was a ``bt`` backtest recomputed from day one on every
session, and the append-only merge kept each published row as its own session
computed it. From 1.6 on, ``engine.baselines.advance_coin_flip`` advances a
persisted state over new dates only, and refuses a published series that has
no state. This script writes that first state.

For each trading agent with a benchmark, from that agent's **last published
coin-flip row**:

- the state date is that row's date;
- the book is that row's ``portfolio_value``, in cash, and the first repick
  runs at that date's close in the *current* store, with the agent's
  *current* universe and ``max_positions`` (the same resolvers the session
  uses), sizing whole shares in the series' currency (each close converted
  at the rate of the state date, the valuation date, as the books are; a
  ticker with no resolvable currency or no rate is not drawn). There is no flat cash day: the next session's
  first new row is already invested.

That repick is the one seam the migration introduces, at the state date,
disclosed at METHODOLOGY ``#stateful-coinflip-2026-10-05``; after it the
series is path-continuous. No published row is touched.

Refuses to overwrite an existing state file unless given ``--force``.

Exit codes: 0 every state written (or, with ``--dry-run``, computed);
2 could not run (an existing state without ``--force``, an unreadable series,
an agent with no published row, or nothing to migrate) — never a partial
"healthy".
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.baselines import (  # noqa: E402
    coin_flip_state_path,
    init_coin_flip_state,
    write_coin_flip_state,
)
from engine.config import get_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--force", action="store_true", help="overwrite existing state files")
    parser.add_argument("--dry-run", action="store_true", help="compute and print, write nothing")
    args = parser.parse_args(argv)

    from scripts.backfill_baselines import _max_positions_by_agent, _universes_by_agent

    cfg = get_config()
    agents = [a for a in cfg.trading_roster if cfg.roster[a].benchmark is not None]
    if not agents:
        print("ERROR: no trading agent with a benchmark; nothing to migrate.", file=sys.stderr)
        return 2

    plans = []
    for agent_id in agents:
        series_path = cfg.baselines_dir / agent_id / "coinflip.json"
        state_path = coin_flip_state_path(series_path)
        if state_path.exists() and not args.force:
            print(
                f"ERROR: {agent_id}: {state_path} already exists; refusing to "
                f"overwrite it without --force.",
                file=sys.stderr,
            )
            return 2
        try:
            rows = json.loads(series_path.read_text())
        except (OSError, ValueError) as exc:
            print(f"ERROR: {agent_id}: cannot read {series_path} ({exc}).", file=sys.stderr)
            return 2
        if not rows:
            print(f"ERROR: {agent_id}: {series_path} has no published row.", file=sys.stderr)
            return 2
        plans.append((agent_id, state_path, rows[-1]))

    universes = _universes_by_agent()
    max_positions = _max_positions_by_agent()
    for agent_id, state_path, last in plans:
        state = init_coin_flip_state(
            agent_id,
            universes.get(agent_id, []),
            max_positions.get(agent_id, 5),
            date.fromisoformat(last["date"]),
            float(last["portfolio_value"]),
            cfg.roster[agent_id].benchmark.currency,
        )
        held = ", ".join(
            f"{t} x{h.shares} @ {h.mark_close:g} {h.currency} x{h.mark_rate:g} ({h.mark_date})"
            for t, h in sorted(state.holdings.items())
        )
        print(
            f"{agent_id}: state {state.date}, value {state.portfolio_value:.2f}, "
            f"cash {state.cash:.2f}, holdings [{held}]"
        )
        if not args.dry_run:
            write_coin_flip_state(state_path, state, agent_id)
    print("dry run: nothing written." if args.dry_run else f"{len(plans)} state file(s) written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
