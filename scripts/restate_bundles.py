"""Restate the per-day archive bundles' leaderboard from restated snapshots.

Built after ``ae6718f6a`` restated ``data/portfolios/*/snapshots.json`` from
the corrected OHLCV store and FX path (see ``restate_valuations.py``). The
per-day archive bundles at ``data/output/YYYY-MM-DD.json`` carry their own
``leaderboard`` array — computed at session time from the *old* store — so
``site/src/pages/archive/[date].astro`` currently publishes a stale figure
for every date, alongside the now-correct ``data/portfolios/*/snapshots.json``
record of the same underlying fact.

**Scope.** Only ``bundle["leaderboard"]`` is restated. ``portfolio.deployed``
(``bundle["agents"][id]["portfolio"]["deployed"]``), despite being named in
the originating brief, is deliberately left untouched: it is
``Portfolio.cost_basis`` (``shares × avg_cost``, see ``engine/types.py``), a
pure trade-ledger figure with no price-store or FX dependency at all, and it
has no counterpart in ``snapshots.json`` to restate it *from*. Neither
correction on this branch (the OHLCV sweep, the FX-conversion fix) touches
it. Restating it would mean inventing a value this script has no basis for.
See ``task-11-report.md`` for the full reasoning. Every other bundle field
(commentary, trades, posts, reasoning, ``portfolio.cash``, ``portfolio.positions``)
is narrative or ledger-recorded, never touched.

**Eligibility mirrors the 176-row exclusion exactly.** For a given bundle
date and agent, this script looks up that agent's ``snapshots.json`` row
keyed on the *same* date string. If no such row exists, or the row is one of
the ones ``restate_valuations.restate_agent`` left unresolved (its effective
session date could not be reconstructed from the ledger), that agent's
leaderboard entry is carried over byte-for-byte. It reuses
``restate_valuations._resolve_session_date`` directly (not a re-derivation)
so eligibility can never disagree with what is already on disk.

For every other (touched) agent, holdings are replayed to the row's
resolved effective session date (``engine.restatement.replay_holdings``) and
priced via ``engine.leaderboard.build_leaderboard_rows`` — the same pure
ranking/return primitive that powers ``data/leaderboard/current.json`` and
every other leaderboard the site renders. This script never recomputes a
return percentage by hand.

A small number of early bundles (8/87) carry a legacy fourth leaderboard key
(``eur_mtm`` / ``mtm_eur`` / ``value_eur`` — the schema was renamed twice
before settling on the modern 3-key shape) holding the raw EUR mark-to-market
value the percentage was derived from. For a touched row carrying one of
these, the value is re-derived from the same freshly-computed ``return_pct``
via the identical relationship ``build_leaderboard_rows`` itself is built on
(``return_pct = (eur_mtm / 10_000 - 1) * 100``), so the two figures in a row
never disagree with each other. This key is not read by any current site
code (``site/src/lib/output.ts``'s ``LeaderboardRow`` type has only
``agent``/``return_pct``/``rank``) but leaving it stale while the sibling
percentage changes would make the row internally self-contradictory.

Rank is a derived property of the *whole* array's sort order, not a
per-agent fact — freezing an untouched agent's ``return_pct`` cannot mean
freezing its rank too, since other entries around it may have moved. Every
bundle's leaderboard is re-sorted and re-ranked after merging touched and
frozen rows, via ``rank_leaderboard_rows`` — the same sort the live builder
uses, so a restated bundle cannot drift onto a different ranking metric.

Dry-run is the default. Pass ``--apply`` to write.

Usage:
    python scripts/restate_bundles.py                 # dry run
    python scripts/restate_bundles.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.disclosure import (
    UndisclosedRestatementError,
    require_changelog_entry,
)
from engine.config import get_config
from engine.leaderboard import build_leaderboard_rows, rank_leaderboard_rows
from engine.portfolio import PortfolioManager
from engine.restatement import replay_holdings
from scripts.restate_valuations import (
    _initial_capital,
    _resolve_session_date,
    restate_agent,
)

# See module docstring — the legacy raw-EUR-value companion key, renamed
# twice across this project's history before the schema settled on the
# modern 3-key row.
_LEGACY_EUR_VALUE_KEYS = ("eur_mtm", "mtm_eur", "value_eur")

# Every row field the 2026-08-14 benchmark-relative metric introduced, and
# therefore every field a restatement of a post-2026-08-14 bundle must refresh
# from the freshly-computed rows. Leaving one out is not cosmetic: the fields
# are legs of one arithmetic — `return_pct`, `return_local_pct` and
# `fx_translation_pp` reconcile as
# `(1 + return_pct) = (1 + return_local_pct) x (1 + fx_translation_pp)`, and
# both `vs_*` fields are subtractions on `return_local_pct` — so a refreshed
# subset published beside a stale one is a bundle whose own numbers contradict
# each other. `currency` is here because it labels `return_local_pct`.
_METRIC_ERA_FIELDS = (
    "currency",
    "return_local_pct",
    "vs_benchmark_pp",
    "vs_coinflip_pp",
    "fx_translation_pp",
)


@dataclass
class AgentState:
    trades: list[dict]
    currency: str
    initial_capital: float
    by_date: dict[str, dict]
    unresolved: set[str]


@dataclass
class RowChange:
    bundle_date: str
    agent_id: str
    old_return_pct: float
    new_return_pct: float

    @property
    def delta_pp(self) -> float:
        return self.new_return_pct - self.old_return_pct


@dataclass
class BundleResult:
    bundle_date: str
    touched_agents: list[str] = field(default_factory=list)
    frozen_agents: list[str] = field(default_factory=list)
    changes: list[RowChange] = field(default_factory=list)


def _load_agent_states(
    manager: PortfolioManager, agent_ids: list[str]
) -> dict[str, AgentState]:
    """Precompute per-agent ledger + eligibility state, once, for every agent
    that ever appears in a bundle leaderboard. Reuses
    ``restate_valuations.restate_agent`` so the touched/untouched split can
    never diverge from what ``ae6718f6a`` already wrote to
    ``snapshots.json``.
    """
    states: dict[str, AgentState] = {}
    for agent_id in agent_ids:
        trades = manager.load_trades(agent_id)
        snapshots = manager.load_snapshots(agent_id)
        currency = manager.load(agent_id).currency
        initial_capital = _initial_capital(agent_id, trades, snapshots)
        result = restate_agent(agent_id, manager)
        states[agent_id] = AgentState(
            trades=trades,
            currency=currency,
            initial_capital=initial_capital,
            by_date={row["date"]: row for row in snapshots},
            unresolved=set(result.unresolved_dates),
        )
    return states


def _agent_summary_for_date(state: AgentState, bundle_date: str) -> dict | None:
    """Return a ``build_leaderboard_rows``-shaped portfolio summary for this
    agent as of ``bundle_date``, or ``None`` if the date is not eligible
    (no snapshot row for it, or the row is one of the 176 left untouched).
    """
    row = state.by_date.get(bundle_date)
    if row is None:
        return None
    if bundle_date in state.unresolved:
        return None

    session_date = _resolve_session_date(state.trades, row, state.initial_capital)
    if session_date is None:
        # Shouldn't happen — restate_agent already resolved this row and it
        # is therefore not in `state.unresolved` — but a row that fails to
        # resolve here is exactly as unrestatable as one that failed in
        # restate_valuations.py. Leave it frozen rather than raise.
        return None

    positions, _cash_delta = replay_holdings(state.trades, session_date)
    return {
        "cash": row["cash"],
        "positions": [{"ticker": t, "shares": s} for t, s in positions.items()],
        "currency": state.currency,
    }


def restate_bundle_leaderboard(
    bundle: dict, bundle_date: str, agent_states: dict[str, AgentState]
) -> BundleResult:
    """Mutate ``bundle["leaderboard"]`` in place; return a change report.

    Preserves the original set of agents in the array (never adds/drops a
    row) and the row's own key set (a legacy EUR-value key stays a legacy
    EUR-value key). Only ``return_pct``, the legacy value key if present,
    and every row's ``rank`` (a derived property of the whole array's sort
    order) can change.

    Having an eligible summary is necessary but not sufficient for a row to
    actually be recomputed: ``build_leaderboard_rows`` itself drops any
    agent whose EUR-MTM comes back ``None`` — e.g. a held position needed
    FX-converting into the book's currency and the rate is unavailable on
    this exact ``bundle_date`` (``engine.valuation.portfolio_mtm`` returns
    ``None`` in that case; see Task 12). A summary-eligible agent that
    ``build_leaderboard_rows`` drops falls back to frozen here too — at its
    originally published value — rather than silently vanishing from the
    array, which would violate the "never adds/drops a row" invariant above.
    """
    old_leaderboard = bundle["leaderboard"]
    result = BundleResult(bundle_date=bundle_date)

    touched_summaries: dict[str, dict] = {}
    legacy_key_by_agent: dict[str, str] = {}
    frozen_rows: list[dict] = []
    old_by_agent = {row["agent"]: row for row in old_leaderboard}

    for row in old_leaderboard:
        agent_id = row["agent"]
        state = agent_states.get(agent_id)
        summary = _agent_summary_for_date(state, bundle_date) if state else None
        if summary is None:
            frozen_rows.append(dict(row))
            result.frozen_agents.append(agent_id)
            continue
        touched_summaries[agent_id] = summary
        for key in _LEGACY_EUR_VALUE_KEYS:
            if key in row:
                legacy_key_by_agent[agent_id] = key
                break

    computed_rows: list[dict] = []
    if touched_summaries:
        fresh_rows = build_leaderboard_rows(
            touched_summaries, on=date.fromisoformat(bundle_date)
        )
        fresh_by_agent = {fresh_row["agent"]: fresh_row for fresh_row in fresh_rows}
        for agent_id in touched_summaries:
            old_row = old_by_agent[agent_id]
            fresh_row = fresh_by_agent.get(agent_id)
            if fresh_row is None:
                # Summary-eligible but build_leaderboard_rows still dropped
                # it (unpriceable on this date) — freeze rather than drop
                # the row. See docstring above.
                frozen_rows.append(dict(old_row))
                result.frozen_agents.append(agent_id)
                continue

            result.touched_agents.append(agent_id)
            new_return_pct = fresh_row["return_pct"]
            # Start from a copy of the row as originally published, not from
            # build_leaderboard_rows's own dict shape: preserves this row's
            # exact key order (early bundles had `rank` before `agent`,
            # unlike the modern schema) and any key this module doesn't know
            # about. Only return_pct (and, below, a legacy value key) are
            # overwritten in place; rank is reset for the whole array after
            # the merge/sort below.
            new_row = dict(old_row)
            new_row["return_pct"] = new_return_pct
            # Metric era: a bundle whose rows carry vs_benchmark_pp was
            # published under the 2026-08-14 benchmark-relative ranking, so
            # its restatement refreshes those fields from the same fresh rows
            # (they were computed as of this bundle's date). A pre-re-rank
            # bundle keeps its raw-return era untouched — adding the fields
            # would re-rank history under a metric that did not exist when
            # the bundle was published.
            if any("vs_benchmark_pp" in r for r in old_leaderboard):
                for key in _METRIC_ERA_FIELDS:
                    if key in fresh_row:
                        new_row[key] = fresh_row[key]
                    else:
                        new_row.pop(key, None)
            legacy_key = legacy_key_by_agent.get(agent_id)
            if legacy_key is not None:
                # Re-derive the legacy EUR-value companion from the fresh
                # return_pct via the exact inverse of the relationship
                # build_leaderboard_rows itself computes — never an
                # independent calculation. See module docstring.
                new_row[legacy_key] = (1 + new_return_pct / 100) * 10_000
            computed_rows.append(new_row)

            old_pct = old_row["return_pct"]
            if abs(new_return_pct - old_pct) > 1e-9:
                result.changes.append(
                    RowChange(
                        bundle_date=bundle_date,
                        agent_id=agent_id,
                        old_return_pct=old_pct,
                        new_return_pct=new_return_pct,
                    )
                )

    # The one definition of the board's order — a row without vs_benchmark_pp
    # (every pre-2026-08-14 bundle) ranks on raw return exactly as it always
    # did, so old-era bundles restate identically to before.
    merged = rank_leaderboard_rows(computed_rows + frozen_rows)

    bundle["leaderboard"] = merged
    return result


def _bundle_leaderboard_agent_ids(output_dir: Path) -> list[str]:
    """Every agent id that appears in any bundle's leaderboard array."""
    ids: set[str] = set()
    for path in sorted(output_dir.glob("*.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for row in bundle.get("leaderboard", []):
            ids.add(row["agent"])
    return sorted(ids)


def run(apply: bool) -> list[BundleResult]:
    cfg = get_config()
    manager = PortfolioManager(base_dir=cfg.portfolios_dir)
    output_dir = cfg.output_dir

    agent_ids = _bundle_leaderboard_agent_ids(output_dir)
    agent_states = _load_agent_states(manager, agent_ids)

    results: list[BundleResult] = []
    largest: RowChange | None = None

    for path in sorted(output_dir.glob("*.json")):
        bundle_date = path.stem
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if "leaderboard" not in bundle:
            continue

        result = restate_bundle_leaderboard(bundle, bundle_date, agent_states)
        results.append(result)

        for change in result.changes:
            if largest is None or abs(change.delta_pp) > abs(largest.delta_pp):
                largest = change

        if result.changes:
            print(
                f"{bundle_date}: {len(result.changes)} row(s) changed, "
                f"{len(result.frozen_agents)} frozen"
            )
            for change in result.changes:
                print(
                    f"    {change.agent_id}: {change.old_return_pct:+.4f}% -> "
                    f"{change.new_return_pct:+.4f}% ({change.delta_pp:+.4f}pp)"
                )

        if apply:
            path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    total_changed_rows = sum(len(r.changes) for r in results)
    total_frozen = sum(len(r.frozen_agents) for r in results)
    print("\n=== Summary ===")
    print(f"  bundles processed: {len(results)}")
    print(f"  leaderboard rows changed: {total_changed_rows}")
    print(f"  leaderboard rows left frozen (no data / unresolved): {total_frozen}")
    if largest is not None:
        print(
            f"  largest single move: {largest.bundle_date} {largest.agent_id}  "
            f"{largest.old_return_pct:+.4f}% -> {largest.new_return_pct:+.4f}% "
            f"({largest.delta_pp:+.4f}pp)"
        )
    else:
        print("  no rows changed")
    print(f"  mode: {'APPLY' if apply else 'DRY RUN'}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restate the per-day archive bundles' leaderboard from "
        "restated snapshots.json."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the restated bundle files. Default is dry-run (no writes).",
    )
    parser.add_argument(
        "--changelog-entry",
        metavar="ANCHOR",
        help="Anchor id of the METHODOLOGY.md changelog entry disclosing this "
        "restatement. Required with --apply (see engine.disclosure).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print without writing (default behavior — this flag is "
        "a no-op provided for explicitness).",
    )
    args = parser.parse_args()

    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")

    if args.apply:
        try:
            require_changelog_entry(
                args.changelog_entry, what="Restating published archive bundles"
            )
        except UndisclosedRestatementError as exc:
            parser.exit(2, f"{exc}\n")

    run(apply=args.apply)


if __name__ == "__main__":
    main()
