"""Ledger-integrity invariant: every filled inbox row has a matching trade.

## The bug this guards against

Between commit 418e763bf (2026-05-18, watcher script introduced) and
8ff48861e (2026-05-23, the fix), `scripts/check_triggers.py`'s
`commit_and_push()` staged only `data/orders/pending` and `data/orders/inbox`
— `data/portfolios` was NOT in the git-add pathspec. Conditional-order fills
ran `apply_trade()` correctly and wrote `portfolio.json` + `trades.json` to
the GitHub Actions runner's local disk, but those paths were never
`git add`ed, so the mutation was discarded when the ephemeral runner was torn
down. The inbox row survived (it WAS staged) and was pushed; the portfolio
mutation vanished. Silent, one-directional data loss: the ledger of record
(inbox) said a trade happened; the ledger of state (trades.json) disagreed.

Exactly one conditional order fired in that window: `ord_2026-05-21_sharp-
shooter-eur_001` (SELL 1 ASML.AS @ EUR1249, fired 2026-05-21T21:47:43Z, see
commit 6609f1c58 which touches only the inbox append and pending-file
deletion — no `data/portfolios/` diff). Its `status:"filled"` inbox row
exists; no matching trade exists in
`data/portfolios/sharp-shooter-eur/trades.json`.

The code path is fixed today (`commit_and_push` includes `data/portfolios`;
see `tests/test_watcher_ordering.py::test_committer_receives_portfolio_dir`).
What did not exist before this file is any check that a fill confirmation
actually landed in a portfolio — the divergence went unnoticed for two
months and was only caught because a public fill count looked wrong.

## The join key

`data/orders/inbox/*.jsonl` (and the per-allocator `{prefix}-inbox/*.jsonl`
channels) rows carry `order_id`; `data/portfolios/<agent>/trades.json` rows
carry `id`. Verified directly (see `_load_trade_ids_by_agent` /
`_load_filled_order_ids`): every trade `id` for an agent is drawn 1:1 from
that agent's own filled inbox rows — e.g.
`data/portfolios/sharp-shooter-eur/trades.json` contains a trade with
`id == "ord_2026-04-17_sharp-shooter-eur_001"`, which is exactly the
`order_id` of the matching inbox row. So **the join key is `order_id == id`,
verbatim, no transformation** (confirmed by
`engine.paper_broker.fill_day`, which stamps `order.order_id` onto both the
`Fill` it appends to the inbox and the `Trade` it hands to
`PortfolioManager.apply_trade`).

The agent that owns a given order/trade is derived from the order_id itself
(`ord_{date}_{agent_id}_{seq}`, underscore-delimited into exactly 4 parts —
verified against all 172 real filled rows in the committed ledger, zero
exceptions; see `_parse_agent_id`). This is a fallback for locating *which
portfolio directory* to open, not a substitute for the id join, which is
exact-string.

## Which portfolio directories are in scope

`data/portfolios/` holds 11 directories, but only 10 traders
(`cfg.trading_roster`, role="trader") + 1 allocator (`cfg.allocators`,
role="allocator", i.e. `the-manager`) are backed by a real order/fill flow.
`baseline-manager` is a synthetic passive-benchmark book (its `trades.json`
is `[]` — it is written by the baselines step, never by `apply_trade`) and
is not declared in `roster.yaml` at all, so it falls out of
`trading_roster | allocators` for free — the same role-based exclusion
`tests/test_build_tax_shadow.py` uses (rather than a directory-name
blocklist). `the-manager` IS included here (unlike in
`build_tax_shadow`, which excludes it for a tax-specific reason — the
Manager's allocator fills aren't PFU-taxable trading activity) because its
`manager-inbox` fills are real portfolio mutations that must reconcile the
same way a trader's do; excluding it would make the allocator-channel
coverage this file exists for (see the module docstring's second bug-fix
requirement) pointless.

## Removing the known exception

`_KNOWN_FORWARD_DIVERGENCES` below is a live TODO, not a permanent
allowlist. The two candidate reconciliation models (replay the SELL against
current portfolio state, or backfill the trade as of its original fill
timestamp) move `sharp-shooter-eur`'s reported return in opposite
directions — that decision belongs to the repo owner, not to this test. Once
the owner reconciles `data/portfolios/sharp-shooter-eur/trades.json`, the
order_id in that set will start matching a real trade, and
`test_known_exceptions_are_still_actually_diverged` will fail — that failure
is the signal to delete the entry (and the reverse: if it fails, the
underlying divergence is gone; remove it here, not by widening the
exception, since data/ is explicitly out of scope for this task).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config import get_config

# ---------------------------------------------------------------------------
# The one documented, greppable exception.
# ---------------------------------------------------------------------------

# order_id -> human-readable reason. Forward-direction only (filled inbox row,
# no matching trade) — see module docstring "Removing the known exception".
_KNOWN_FORWARD_DIVERGENCES: dict[str, str] = {
    # Empty since 2026-08-02. The one entry that lived here —
    # ord_2026-05-21_sharp-shooter-eur_001, SELL 1 ASML.AS @ EUR1249, lost to the
    # commit_and_push() pathspec bug — was reconciled into trades.json, and the
    # 2026-06-24 sale it had made possible (ord_2026-06-24_sharp-shooter-eur_001,
    # which sold a share the book no longer owned) was voided in the inbox with
    # reason NO_POSITION_TO_SELL. Kept as an explicit empty dict, not deleted, so a
    # future divergence has an obvious documented home rather than an inline exemption.
}


# No known reverse-direction divergences (trade with no matching filled inbox
# row) exist in the committed ledger as of this writing. Kept as an explicit,
# named, empty set — not omitted — so a future reverse divergence has an
# obvious place to be documented rather than silently exempted inline.
_KNOWN_REVERSE_DIVERGENCES: frozenset[tuple[str, str]] = frozenset()


# ---------------------------------------------------------------------------
# Join-key + agent-derivation helpers
# ---------------------------------------------------------------------------


def _parse_agent_id(order_id: str) -> str:
    """Derive the owning agent id from an order_id: ord_{date}_{agent_id}_{seq}.

    Agent ids never contain underscores (they use hyphens: "sharp-shooter-eur",
    "monsieur-forex", ...) and dates never contain underscores either, so
    splitting on "_" always yields exactly 4 parts. Raises loudly on anything
    else rather than silently guessing — a malformed order_id is itself worth
    surfacing, not swallowing.
    """
    parts = order_id.split("_")
    if len(parts) != 4 or parts[0] != "ord":
        raise ValueError(
            f"order_id {order_id!r} does not match the ord_{{date}}_{{agent}}_"
            f"{{seq}} convention (got {len(parts)} '_'-delimited parts)"
        )
    return parts[2]


def _load_filled_order_ids(inbox_dirs: list[Path]) -> dict[str, str]:
    """Scan inbox-shaped JSONL dirs for status:"filled" rows.

    Returns {order_id: agent_id}. Malformed JSON lines are skipped (matches
    the best-effort convention of engine.orders.inbox_order_ids); a row with
    an order_id that fails _parse_agent_id is NOT skipped — it propagates,
    since that indicates the join key itself is broken, which this check
    exists to catch.
    """
    filled: dict[str, str] = {}
    for inbox_dir in inbox_dirs:
        if not inbox_dir.exists():
            continue
        for path in sorted(inbox_dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("status") != "filled":
                    continue
                order_id = row["order_id"]
                filled[order_id] = _parse_agent_id(order_id)
    return filled


def _load_trade_ids_by_agent(
    portfolios_dir: Path, valid_agents: set[str]
) -> dict[str, set[str]]:
    """Read {agent}/trades.json for every agent in valid_agents that exists
    on disk; returns {agent_id: {trade id, ...}}."""
    trade_ids: dict[str, set[str]] = {}
    for agent_id in valid_agents:
        trades_path = portfolios_dir / agent_id / "trades.json"
        if not trades_path.exists():
            continue
        trades = json.loads(trades_path.read_text(encoding="utf-8"))
        trade_ids[agent_id] = {t["id"] for t in trades}
    return trade_ids


# ---------------------------------------------------------------------------
# Pure reconciliation logic (no disk I/O) — shared by the real-ledger test
# and the mutation-check / unit tests below.
# ---------------------------------------------------------------------------


def find_forward_divergences(
    filled_order_ids: dict[str, str],
    trade_ids_by_agent: dict[str, set[str]],
    known_exceptions: dict[str, str] = _KNOWN_FORWARD_DIVERGENCES,
) -> list[str]:
    """order_ids with a filled inbox row but no matching trade, minus known
    exceptions. Returns diagnostic strings naming the order_id and agent."""
    diagnostics = []
    for order_id, agent_id in filled_order_ids.items():
        if order_id in known_exceptions:
            continue
        if order_id not in trade_ids_by_agent.get(agent_id, set()):
            diagnostics.append(
                f"{order_id}: status=filled in inbox but no matching trade "
                f"in data/portfolios/{agent_id}/trades.json"
            )
    return diagnostics


def find_reverse_divergences(
    filled_order_ids: dict[str, str],
    trade_ids_by_agent: dict[str, set[str]],
    known_exceptions: frozenset[tuple[str, str]] = _KNOWN_REVERSE_DIVERGENCES,
) -> list[str]:
    """(agent, trade_id) pairs with a trade but no matching filled inbox row,
    minus known exceptions. Returns diagnostic strings."""
    diagnostics = []
    for agent_id, trade_ids in trade_ids_by_agent.items():
        for trade_id in trade_ids:
            if (agent_id, trade_id) in known_exceptions:
                continue
            if trade_id not in filled_order_ids:
                diagnostics.append(
                    f"{agent_id}/{trade_id}: trade in trades.json but no "
                    f"status=filled row in the inbox for order_id={trade_id}"
                )
    return diagnostics


# ---------------------------------------------------------------------------
# Real-ledger tests — read the committed data/, not a tmp fixture. These
# deliberately do NOT use the midas_data_root fixture: the point is to
# reconcile the actual repo state, not a synthetic one.
# ---------------------------------------------------------------------------


def _real_inbox_dirs() -> list[Path]:
    cfg = get_config()
    dirs = [cfg.orders_dir / "inbox"]
    for agent_id in cfg.allocators:
        prefix = cfg.allocator_spec(agent_id).channels_prefix
        dirs.append(cfg.orders_dir / f"{prefix}-inbox")
    return dirs


def _real_valid_agents() -> set[str]:
    cfg = get_config()
    return set(cfg.trading_roster) | set(cfg.allocators)


@pytest.mark.live_cast
class TestRealLedgerIntegrity:
    """Reconcile the actual committed ledger under data/.

    Cast-coupled: reads the live desk's committed ledger. midas-core ships only
    generic data (strategies, universes, ticker maps) — never the order ledger —
    so on the demo desk these have nothing to reconcile and
    ``test_allocator_channel_is_included`` correctly fails rather than passing
    vacuously. The other two classes in this file are hermetic and still run
    everywhere, which is why the marker is on the class and not the module.
    """

    def test_forward_integrity(self) -> None:
        """Every real filled inbox row (public + allocator channels) has a
        matching trade, except the one documented, dated exception."""
        filled = _load_filled_order_ids(_real_inbox_dirs())
        trade_ids = _load_trade_ids_by_agent(
            get_config().portfolios_dir, _real_valid_agents()
        )
        diagnostics = find_forward_divergences(filled, trade_ids)
        assert not diagnostics, (
            "Ledger divergence: filled inbox row(s) with no matching trade:\n"
            + "\n".join(diagnostics)
        )

    def test_reverse_integrity(self) -> None:
        """Every real trade has a matching filled inbox row — no phantom
        trades that never went through the fill flow."""
        filled = _load_filled_order_ids(_real_inbox_dirs())
        trade_ids = _load_trade_ids_by_agent(
            get_config().portfolios_dir, _real_valid_agents()
        )
        diagnostics = find_reverse_divergences(filled, trade_ids)
        assert not diagnostics, (
            "Ledger divergence: trade(s) with no matching filled inbox row:\n"
            + "\n".join(diagnostics)
        )

    def test_known_exceptions_are_still_actually_diverged(self) -> None:
        """Guard against exception rot: if the owner reconciles the missing
        trade, the exception becomes stale and must be deleted from
        _KNOWN_FORWARD_DIVERGENCES (see module docstring), not left dangling
        as a silent mask over a bug that no longer exists."""
        filled = _load_filled_order_ids(_real_inbox_dirs())
        trade_ids = _load_trade_ids_by_agent(
            get_config().portfolios_dir, _real_valid_agents()
        )
        for order_id in _KNOWN_FORWARD_DIVERGENCES:
            agent_id = filled.get(order_id) or _parse_agent_id(order_id)
            still_diverged = order_id not in trade_ids.get(agent_id, set())
            assert still_diverged, (
                f"{order_id} now has a matching trade — remove it from "
                "_KNOWN_FORWARD_DIVERGENCES in tests/test_ledger_integrity.py."
            )

    def test_allocator_channel_is_included(self) -> None:
        """Sanity check that the allocator channel is actually being read —
        a check that silently globbed zero files would pass vacuously."""
        cfg = get_config()
        assert cfg.allocators, "expected at least one allocator (the-manager)"
        allocator_dirs = _real_inbox_dirs()[1:]
        assert allocator_dirs, "expected at least one allocator inbox dir"
        assert any(d.exists() and list(d.glob("*.jsonl")) for d in allocator_dirs), (
            "no allocator inbox JSONL files found — the allocator-channel "
            "coverage this file exists for would be untested"
        )


# ---------------------------------------------------------------------------
# Mutation / unit tests on the pure reconciliation functions — no disk I/O,
# no fixtures, no risk to data/. These are what the "done when" mutation
# checks exercise directly (see PR description): rather than editing files
# under data/, we build synthetic filled/trade maps in memory.
# ---------------------------------------------------------------------------


class TestReconciliationLogic:
    def test_forward_divergence_is_detected_and_diagnostic(self) -> None:
        filled = {"ord_2026-01-01_satoshi_001": "satoshi"}
        trade_ids: dict[str, set[str]] = {"satoshi": set()}  # trade missing

        diagnostics = find_forward_divergences(filled, trade_ids, known_exceptions={})

        assert len(diagnostics) == 1
        assert "ord_2026-01-01_satoshi_001" in diagnostics[0]
        assert "satoshi" in diagnostics[0]

    def test_forward_divergence_absent_when_trade_matches(self) -> None:
        filled = {"ord_2026-01-01_satoshi_001": "satoshi"}
        trade_ids = {"satoshi": {"ord_2026-01-01_satoshi_001"}}

        diagnostics = find_forward_divergences(filled, trade_ids, known_exceptions={})

        assert diagnostics == []

    def test_forward_divergence_suppressed_by_known_exception(self) -> None:
        filled = {"ord_2026-01-01_satoshi_001": "satoshi"}
        trade_ids: dict[str, set[str]] = {"satoshi": set()}

        diagnostics = find_forward_divergences(
            filled,
            trade_ids,
            known_exceptions={"ord_2026-01-01_satoshi_001": "test exception"},
        )

        assert diagnostics == []

    def test_reverse_divergence_is_detected_and_diagnostic(self) -> None:
        """Second mutation-check technique: simulate deleting one real trade
        row by omitting it from an in-memory trade_ids_by_agent map — no file
        under data/ is touched. Equivalent to monkeypatching the loader to
        return trades.json minus one row."""
        filled: dict[str, str] = {}  # no filled inbox row at all
        trade_ids = {"satoshi": {"ord_2026-01-01_satoshi_001"}}

        diagnostics = find_reverse_divergences(
            filled, trade_ids, known_exceptions=frozenset()
        )

        assert len(diagnostics) == 1
        assert "ord_2026-01-01_satoshi_001" in diagnostics[0]
        assert "satoshi" in diagnostics[0]

    def test_reverse_divergence_absent_when_fill_matches(self) -> None:
        filled = {"ord_2026-01-01_satoshi_001": "satoshi"}
        trade_ids = {"satoshi": {"ord_2026-01-01_satoshi_001"}}

        diagnostics = find_reverse_divergences(
            filled, trade_ids, known_exceptions=frozenset()
        )

        assert diagnostics == []

    def test_reverse_divergence_suppressed_by_known_exception(self) -> None:
        filled: dict[str, str] = {}
        trade_ids = {"satoshi": {"ord_2026-01-01_satoshi_001"}}

        diagnostics = find_reverse_divergences(
            filled,
            trade_ids,
            known_exceptions=frozenset({("satoshi", "ord_2026-01-01_satoshi_001")}),
        )

        assert diagnostics == []


class TestParseAgentId:
    def test_parses_hyphenated_agent_id(self) -> None:
        assert _parse_agent_id("ord_2026-04-17_sharp-shooter-eur_001") == (
            "sharp-shooter-eur"
        )

    def test_parses_single_word_agent_id(self) -> None:
        assert _parse_agent_id("ord_2026-04-17_satoshi_001") == "satoshi"

    def test_rejects_malformed_order_id(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            _parse_agent_id("not-an-order-id")
