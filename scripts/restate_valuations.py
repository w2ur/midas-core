"""Restate published portfolio valuations from the corrected OHLCV store.

Built after commit 4b6b8556 corrected 29,348 rows in the committed OHLCV
store (partial bars + 11 un-adjusted stock splits) and 3b26822f3 fixed the
live valuation path's foreign-currency conversion. Every row in every
``data/portfolios/*/snapshots.json`` was priced off the old, wrong store —
this script re-derives each row's valuation from the corrected one, using
the pure primitives in ``engine.restatement`` (``replay_holdings`` +
``revalue_snapshot``), the same code path the live desk now uses.

**Recorded cash is authoritative and is never recomputed.** Fills executed
at the prices they executed at; only *valuations* (positions_value,
portfolio_value, benchmarks) change. ``data/portfolios/*/trades.json`` is
never touched.

**Ledger cross-check design (revised twice after investigation — see
task-5-report.md "Round 2" and "Round 3"):**

- *Book-level gate* (whether to write a book at all): a **final
  reconciliation** —
  ``initial_capital + replay_holdings(trades, today())[1] == manager.load(agent_id).cash``
  (today's live cash). Precise, unambiguous, passes for all 12 books today.
  A book whose *final* cash isn't explained by its own trade ledger (the
  one class of defect this actually needs to catch, e.g. a confirmed fill
  whose portfolio write never landed) is refused unless named via
  ``--allow-ledger-divergence``.
- *Per-row holdings clock* (what composition to price, per row): **not**
  ``replay_holdings(trades, row["date"])``. That mixes two different
  clocks — ``row["cash"]`` is a snapshot of whatever the *session* that
  wrote the row had already done (which routinely differs from the row's
  own market-date label — see below), while a naive date-bounded replay
  uses the label. Combining them silently substitutes a different set of
  holdings than what was actually published for that date, which is a
  composition change, not a valuation-only one. The coherent fix: solve
  for the row's true **effective session date** — the date whose
  ledger-replay cash exactly matches the row's own recorded ``cash``
  (which is authoritative and never modified) — and replay holdings to
  *that* date. Pricing still happens at the row's own ``date`` (market
  date), unchanged. See ``_resolve_session_date``.
- Verified mechanically on real rows, not asserted: the OHLCV cron landing
  after the 20:00 UTC session so a session's own snapshot is dated the
  *previous* market day while its cash already includes that day's fills
  (forward lag — the common case, resolved by the forward scan below);
  the reverse — a row dated D whose cash was captured *before* D's own
  same-day trades were applied (backward lag — deliberately **not**
  resolved by this script's forward-only scan, since guessing a
  composition backward from partial information is exactly the invented
  history this script must not produce); and same-session trades
  microseconds apart whose recorded timestamps don't perfectly reflect
  application order.
- A row whose effective session date **can't be found** (no date within
  the forward search window reproduces its recorded cash) is left
  **completely untouched** — same ``portfolio_value``, ``positions_value``,
  and ``benchmarks`` as published. Inventing a composition for it would be
  worse than leaving a stale (but honestly-labeled-as-unrestated) row.
  Counts are reported per agent for disclosure.

Dry-run is the default — a tool that rewrites the published track record
does not do so by accident. Pass ``--apply`` to write.

Usage:
    python scripts/restate_valuations.py                                  # dry run
    python scripts/restate_valuations.py --apply
    python scripts/restate_valuations.py --apply --allow-ledger-divergence sharp-shooter-eur
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.ohlcv_store import latest_close_on_or_before
from engine.portfolio import PortfolioManager
from engine.disclosure import (
    UndisclosedRestatementError,
    require_changelog_entry,
)
from engine.restatement import MissingPriceError, replay_holdings, revalue_snapshot
from scripts.fetch_market_data import _BENCHMARK_SOURCES

# A replayed cash figure within this many currency units of the recorded
# figure is treated as agreeing — sub-cent float noise, not a real
# divergence.
_CASH_TOLERANCE = 0.01

# Forward-only search window (days) for a legacy row's effective session
# date. 60 days comfortably covers every real staleness stretch observed in
# this ledger (the longest confirmed one is ~30 days, satoshi's
# 2026-04-30..2026-05-30 block) without being so wide it risks a spurious
# coincidental cash match far from the row's own date.
_SESSION_DATE_SEARCH_WINDOW_DAYS = 60

# Stop-condition thresholds (see task brief) — printed as loud warnings so a
# human reviewing the dry run can't miss them. Not enforced in code: the
# only hard, code-enforced gate is the ledger cross-check below.
_ROW_MOVE_WARN_PCT = 15.0
_HEADLINE_MOVE_WARN_PCT = 3.0


class RestatementError(RuntimeError):
    """Raised when a row can't be restated (missing price/FX rate, bad ledger)."""


@dataclass
class RowChange:
    row_date: str
    old_value: float
    new_value: float

    @property
    def pct(self) -> float:
        if not self.old_value:
            return 0.0
        return (self.new_value - self.old_value) / self.old_value * 100.0


@dataclass
class AgentResult:
    agent_id: str
    currency: str
    new_rows: list[dict]
    changes: list[RowChange] = field(default_factory=list)
    # Rows whose effective session date couldn't be resolved (no date within
    # the search window reproduces the row's recorded cash) — left
    # completely untouched, reported here for disclosure.
    unresolved_dates: list[str] = field(default_factory=list)
    row_errors: list[tuple[str, str]] = field(default_factory=list)
    old_headline_pct: float = 0.0
    new_headline_pct: float = 0.0
    # Book-level gate: initial_capital + full-ledger replay vs today's live
    # cash (manager.load(agent_id).cash). This is the one check that
    # actually distinguishes "the ledger explains this book's cash" from a
    # real unreconciled gap (e.g. a confirmed fill that never landed).
    final_reconciliation_diff: float = 0.0

    @property
    def diverges(self) -> bool:
        return abs(self.final_reconciliation_diff) > _CASH_TOLERANCE

    @property
    def largest_change(self) -> RowChange | None:
        if not self.changes:
            return None
        return max(self.changes, key=lambda c: abs(c.new_value - c.old_value))

    @property
    def headline_delta_pp(self) -> float:
        return self.new_headline_pct - self.old_headline_pct


def _initial_capital(agent_id: str, trades: list[dict], snapshots: list[dict]) -> float:
    """Derive an agent's inception cash from its own ledger + first snapshot.

    ``roster.yaml``'s ``initial_capital`` (10,000 for every trader,
    2,000 for the-manager) is a EUR-equivalent *target*, not the actual
    home-currency starting cash: commit f8bf8f038 ("10-agent EUR-aware
    roster") wiped and reinitialized all 10 books "at €10,000 equivalent in
    their base currency" — EUR agents got EUR 10,000 cash, but USD agents
    (sharp-shooter-usd, steady-eddie-usd, yolo-sapiens-usd) got USD 11,784.11
    (or a very slightly different figure per agent, from the EUR/USD rate at
    each agent's own initialization moment), never a flat $10,000.
    ``baseline-manager`` predates roster.yaml entirely and has its own
    constant (``engine.baseline_manager.INITIAL_CAPITAL_EUR``) which is
    accurate, but deriving empirically here avoids a second special case.

    Instead, back-solve the true inception cash from the ledger itself:
    ``snapshots[0]["cash"] - cash_delta(trades, as_of=snapshots[0]["date"])``.
    This is exact for every agent regardless of currency or config drift,
    and self-consistent with the same ``replay_holdings`` primitive used
    for every other row.
    """
    if not snapshots:
        raise RestatementError(
            f"{agent_id!r} has no snapshots — cannot derive inception cash."
        )
    first = snapshots[0]
    _positions, cash_delta = replay_holdings(trades, date.fromisoformat(first["date"]))
    return first["cash"] - cash_delta


def _portfolio_agent_ids() -> list[str]:
    """Every portfolio directory with both trades.json and snapshots.json.

    Deliberately wider than ``get_config().trading_roster`` (the 10 public
    traders): the private the-manager and baseline-manager books are
    snapshotted by the same daily-session step (see
    ``step_update_snapshots``) and were priced off the same wrong store, so
    they need restating too.
    """
    portfolios_dir = get_config().portfolios_dir
    ids = []
    for entry in sorted(portfolios_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "trades.json").exists() and (entry / "snapshots.json").exists():
            ids.append(entry.name)
    return ids


def _benchmarks_as_of(market_date: date) -> dict[str, float]:
    """Recompute the shared 4-benchmark dict as of a historical market date.

    Mirrors ``scripts.fetch_market_data._resolve_benchmark`` exactly (same
    source-priority list, same multipliers, same rounding) but reads
    ``latest_close_on_or_before(ticker, market_date)`` instead of "latest in
    the store" — the historical equivalent of what that function computes
    when run live on a given day.
    """
    benchmarks: dict[str, float] = {}
    for name, sources in _BENCHMARK_SOURCES.items():
        for ticker, multiplier, _label in sources:
            price = latest_close_on_or_before(ticker, market_date)
            if price is None:
                continue
            value = price * multiplier
            benchmarks[name] = round(value, 4 if name == "msci_world" else 2)
            break
        else:
            raise RestatementError(
                f"No OHLCV source available for benchmark {name!r} as of "
                f"{market_date.isoformat()}. Tried: {[t for t, _, _ in sources]}"
            )
    return benchmarks


def _cash_matches(
    trades: list[dict], as_of: date, initial_capital: float, recorded_cash: float
) -> bool:
    _positions, cash_delta = replay_holdings(trades, as_of)
    return abs((initial_capital + cash_delta) - recorded_cash) < _CASH_TOLERANCE


def _resolve_session_date(
    trades: list[dict], row: dict, initial_capital: float
) -> date | None:
    """Return the market date whose ledger-replay cash matches this row's
    recorded cash — the row's true holdings clock — or None if unresolvable.

    ``session_date`` is tried first when the row carries it, but is **not**
    trusted blindly — it is verified against recorded cash exactly like a
    search candidate. This matters: on 3 of the 12 books' most recent rows
    (e.g. sharp-shooter-eur's 2026-08-03/08-04 rows, `session_date` one day
    ahead of `date`), the recorded field does not actually reconcile —
    replaying to it produces holdings that include a trade the row's own
    cash does *not* yet reflect, which would silently invent a wrong
    composition exactly the way this design exists to avoid. If
    ``session_date`` doesn't verify, fall through to the forward search
    below rather than trusting an unverified field.

    Otherwise (the common case — this field didn't exist for most of this
    project's history, or failed verification) forward-scan from the row's
    own ``date`` up to ``_SESSION_DATE_SEARCH_WINDOW_DAYS`` days ahead for
    the first date whose replayed cumulative cash matches ``row["cash"]``
    within ``_CASH_TOLERANCE``. Forward-only and bounded deliberately: an
    unbounded or backward search risks landing on a coincidental match that
    invents a composition rather than reconstructing the real one — a row
    whose true holdings clock lies *before* its own market date (verified to
    happen too, e.g. goldfinger's 2026-06-09 row) is therefore left
    unresolved rather than guessed at.
    """
    recorded_cash = row["cash"]

    session_date_field = row.get("session_date")
    if session_date_field:
        candidate = date.fromisoformat(session_date_field)
        if _cash_matches(trades, candidate, initial_capital, recorded_cash):
            return candidate

    row_date = date.fromisoformat(row["date"])
    for offset in range(0, _SESSION_DATE_SEARCH_WINDOW_DAYS + 1):
        candidate = row_date + timedelta(days=offset)
        if _cash_matches(trades, candidate, initial_capital, recorded_cash):
            return candidate
    return None


def _final_reconciliation_diff(
    agent_id: str, trades: list[dict], initial_capital: float, manager: PortfolioManager
) -> float:
    """Book-level ledger gate: initial_capital + full replay vs today's live cash.

    Replays the *entire* trade ledger (as_of=today, so every trade is
    included regardless of timestamp) and compares against
    ``manager.load(agent_id).cash`` — the book's current, live cash on disk.
    This is precise and unambiguous, unlike a per-row date-bounded compare
    (see the module docstring): it catches the one defect class that
    matters (a trade whose cash effect never landed anywhere in the
    ledger), without the false positives a per-row check produces from
    session-timing artifacts in legacy rows.
    """
    _positions, full_delta = replay_holdings(trades, date.today())
    live_cash = manager.load(agent_id).cash
    return (initial_capital + full_delta) - live_cash


def restate_agent(agent_id: str, manager: PortfolioManager) -> AgentResult:
    trades = manager.load_trades(agent_id)
    snapshots = manager.load_snapshots(agent_id)
    currency = manager.load(agent_id).currency
    initial_capital = _initial_capital(agent_id, trades, snapshots)
    final_reconciliation_diff = _final_reconciliation_diff(
        agent_id, trades, initial_capital, manager
    )

    new_rows: list[dict] = []
    changes: list[RowChange] = []
    unresolved_dates: list[str] = []
    row_errors: list[tuple[str, str]] = []

    for row in snapshots:
        row_date = date.fromisoformat(row["date"])
        recorded_cash = row["cash"]

        session_date = _resolve_session_date(trades, row, initial_capital)
        if session_date is None:
            # Can't establish which holdings this row's cash actually
            # reflects — leave the row completely untouched rather than
            # invent a composition. See module docstring.
            unresolved_dates.append(row["date"])
            new_rows.append(row)
            continue

        try:
            positions, _cash_delta = replay_holdings(trades, session_date)
            # Pricing stays anchored to the row's own market date — only the
            # holdings clock changed, not the pricing clock.
            new_pv, new_positions_value = revalue_snapshot(
                positions, recorded_cash, row_date, currency
            )
            new_benchmarks = _benchmarks_as_of(row_date)
        except (MissingPriceError, RestatementError, ValueError) as exc:
            row_errors.append((row["date"], str(exc)))
            new_rows.append(row)
            continue

        new_row = dict(
            row
        )  # preserves key presence/order (e.g. legacy rows without session_date)
        new_row["portfolio_value"] = new_pv
        new_row["positions_value"] = new_positions_value
        new_row["benchmarks"] = new_benchmarks
        new_rows.append(new_row)

        old_pv = row["portfolio_value"]
        if abs(new_pv - old_pv) > 1e-6:
            changes.append(
                RowChange(row_date=row["date"], old_value=old_pv, new_value=new_pv)
            )

    result = AgentResult(
        agent_id=agent_id,
        currency=currency,
        new_rows=new_rows,
        changes=changes,
        unresolved_dates=unresolved_dates,
        row_errors=row_errors,
        final_reconciliation_diff=final_reconciliation_diff,
    )
    if snapshots:
        result.old_headline_pct = (
            snapshots[-1]["portfolio_value"] / initial_capital - 1
        ) * 100
        result.new_headline_pct = (
            new_rows[-1]["portfolio_value"] / initial_capital - 1
        ) * 100
    return result


def _print_agent_table(result: AgentResult, will_write: bool) -> None:
    print(f"\n=== {result.agent_id} ({result.currency}) ===")
    print(
        f"  rows: {len(result.new_rows)}  changed: {len(result.changes)}  "
        f"unresolved (left untouched): {len(result.unresolved_dates)}"
    )
    if result.unresolved_dates:
        preview = ", ".join(result.unresolved_dates[:6])
        more = (
            f" (+{len(result.unresolved_dates) - 6} more)"
            if len(result.unresolved_dates) > 6
            else ""
        )
        print(f"    unresolved dates: {preview}{more}")

    if result.row_errors:
        print(f"  [ERROR] {len(result.row_errors)} row(s) could not be restated:")
        for row_date, msg in result.row_errors:
            print(f"    {row_date}: {msg}")

    largest = result.largest_change
    if largest is not None:
        flag = (
            " ⚠ EXCEEDS 15% STOP THRESHOLD"
            if abs(largest.pct) > _ROW_MOVE_WARN_PCT
            else ""
        )
        print(
            f"  largest single move: {largest.row_date}  "
            f"{largest.old_value:.2f} -> {largest.new_value:.2f}  ({largest.pct:+.2f}%){flag}"
        )
        first = result.changes[0]
        last = result.changes[-1]
        print(f"  first affected date: {first.row_date}")
        print(f"  last affected date:  {last.row_date}")
    else:
        print("  no rows changed")

    delta = result.headline_delta_pp
    flag = (
        " ⚠ EXCEEDS few-pp STOP THRESHOLD"
        if abs(delta) > _HEADLINE_MOVE_WARN_PCT
        else ""
    )
    print(
        f"  headline return: {result.old_headline_pct:+.2f}% -> "
        f"{result.new_headline_pct:+.2f}%  (Δ {delta:+.2f}pp){flag}"
    )

    if result.diverges:
        print(
            f"  [LEDGER DIVERGENCE — GATING] final reconciliation "
            f"(initial_capital + full replay vs today's live cash) off by "
            f"{result.final_reconciliation_diff:+.4f} {result.currency}"
        )
    else:
        print(
            f"  ledger cross-check (final reconciliation): OK "
            f"(diff={result.final_reconciliation_diff:+.6f})"
        )

    print(f"  -> {'WRITE' if will_write else 'SKIPPED (not written)'}")


def run(apply: bool, allow_ledger_divergence: list[str]) -> dict[str, AgentResult]:
    manager = PortfolioManager(base_dir=get_config().portfolios_dir)
    allow_set = set(allow_ledger_divergence)
    results: dict[str, AgentResult] = {}

    total_rows_changed = 0
    total_rows_unresolved = 0
    diverging_agents: list[str] = []
    unexpected_divergences: list[str] = []

    for agent_id in _portfolio_agent_ids():
        result = restate_agent(agent_id, manager)
        results[agent_id] = result

        will_write = (not result.diverges) or (agent_id in allow_set)
        _print_agent_table(result, will_write)

        total_rows_changed += len(result.changes)
        total_rows_unresolved += len(result.unresolved_dates)
        if result.diverges:
            diverging_agents.append(agent_id)
            if agent_id != "sharp-shooter-eur":
                unexpected_divergences.append(agent_id)

        if apply and will_write:
            path = manager._snapshots_path(agent_id)  # noqa: SLF001 — same private writer PortfolioManager itself uses
            with path.open("w") as f:
                json.dump(result.new_rows, f, indent=2)

    print("\n=== Summary ===")
    print(f"  agents processed: {len(results)}")
    print(f"  total rows changed: {total_rows_changed}")
    print(f"  total rows left unresolved (untouched): {total_rows_unresolved}")
    print(f"  ledger divergence: {diverging_agents or 'none'}")
    if unexpected_divergences:
        print(
            f"  [STOP CONDITION] ledger divergence on agent(s) other than "
            f"sharp-shooter-eur: {unexpected_divergences}"
        )
    print(f"  mode: {'APPLY' if apply else 'DRY RUN'}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restate published portfolio valuations from the corrected OHLCV store."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write the restated snapshots.json files. Default is dry-run (no writes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print without writing (default behavior — this flag is a no-op "
        "provided for explicitness).",
    )
    parser.add_argument(
        "--changelog-entry",
        metavar="ANCHOR",
        help="Anchor id of the METHODOLOGY.md changelog entry disclosing this "
        "restatement. Required with --apply: a published number does not move "
        "undisclosed (see engine.disclosure).",
    )
    parser.add_argument(
        "--allow-ledger-divergence",
        nargs="+",
        default=[],
        metavar="AGENT_ID",
        help="Agent IDs whose ledger cross-check divergence is known and accepted "
        "(e.g. sharp-shooter-eur's documented 2026-05-21 lost fill). Any other "
        "agent that diverges is refused, never written.",
    )
    args = parser.parse_args()

    if args.apply and args.dry_run:
        parser.error("--apply and --dry-run are mutually exclusive.")

    # Only --apply is gated. A dry run publishes nothing, and forcing the
    # changelog entry to exist before you can even see what would move would
    # mean writing the disclosure before knowing the numbers.
    if args.apply:
        try:
            require_changelog_entry(
                args.changelog_entry, what="Restating published valuations"
            )
        except UndisclosedRestatementError as exc:
            parser.exit(2, f"{exc}\n")

    run(apply=args.apply, allow_ledger_divergence=args.allow_ledger_divergence)


if __name__ == "__main__":
    main()
