"""Orchestrator for the daily Midas trading session.

Two modes:

1. **Snapshot-only CLI** (`run_daily_session()`): fetch market data, snapshot
   portfolios, commit & push. Intended for manual EOD runs and CI health
   checks — does NOT dispatch Claude agents. Every step is idempotent.

2. **Full Ring 1 + Ring 2 pipeline** (step_* helpers, called from an
   orchestrating Claude Code session that parallelises agent dispatch):
     - step_author_orders()                → data/orders/outbox/
     - step_fill_orders()                  → data/orders/inbox/ + portfolio mutation
     - step_build_baseline_manager()       → data/portfolios/baseline-manager/ (Gate C)
     - step_resolve_manager_outcomes()     → data/orders/manager-review/resolved.json (C5b)
     - step_build_manager_prompt()         → LLM Manager prompt (C3 context + persona)
     - step_apply_manager_decision()       → data/orders/manager-{outbox,inbox,review}/
                                             + data/portfolios/the-manager/ (PAPER, private)
     - step_build_post_prompts()           → prompts for the orchestrator
     - step_load_memories()                → dict[agent_id, str]
     - step_build_leaderboard()            → ranked rows (EUR mtm / €10k inception)
     - step_build_oracle_prompt()          → Oracle prompt (optionally with journals)
     - build_portfolio_summaries()         → dict for ALL 10 agents (carry-forward)
     - step_save_content()                 → data/posts/, data/blog/, data/output/
     - step_build_memory_update_prompts()  → Ring 2 session-end rewrite prompts
     - step_save_memories()                → data/agent_memory/
     - step_build_baselines()              → data/baselines/ (idempotent recompute)
     - step_build_tax_shadow()            → data/tax_shadow/ (reporting only, after baselines)

Usage (snapshot-only):
    python scripts/daily_session.py
    python scripts/daily_session.py --dry-run   # skip git commit/push
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

# Add project root to sys.path so engine imports work when run directly.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.session_state import clear as _clear_state
from scripts.session_state import is_done as _is_done
from scripts.session_state import mark_done as _mark_done

from engine.agent_memory import (
    build_memory_update_prompt,
    load_journal,
    save_journal,
)
from engine.baseline_manager import (
    INITIAL_CAPITAL_EUR,
    POSITION_SIZE_EUR,
    STRATEGY_ID,
    eligible_tickers,
    is_rebalance_day,
    rebalance,
)
from engine.blog import build_oracle_prompt, save_daily_blog_draft
from engine.ohlcv_store import latest_close_on_or_before
from engine.orders import (
    DroppedTrade,
    Order,
    append_dropped,
    append_order,
    make_order_id,
)
from engine.research_note import parse_research_note
from engine.triggers import (
    CancelRequest,
    append_cancel,
    list_pending,
)
from engine.types import Portfolio
from engine.output_bundle import (
    assemble_output_bundle,
    get_day_number,
    save_output_bundle,
)
from engine.paper_broker import fill_day
from engine.portfolio import PortfolioManager
from engine.config import AgentSpec, AllocatorSpec, get_config
from engine.posts import (
    PostPayload,
    build_post_prompt,
    save_daily_posts,
)
from scripts.fetch_market_data import fetch_and_save as fetch_market_data


# ---------------------------------------------------------------------------
# Idempotency decorator
# ---------------------------------------------------------------------------


def idempotent_step(skip_return: Any) -> Callable[[_F], _F]:
    """Decorator that makes a ``step_*`` function resumable.

    On entry, if the step's name is already recorded as done in the session
    state for today, the body is skipped and ``skip_return`` is returned
    immediately (log a message so the orchestrator sees the skip).

    On successful completion, the step name is recorded via ``mark_done``.
    On exception, nothing is recorded — the next run will retry.

    ``skip_return`` should be the neutral/empty value appropriate for the
    function's return type (e.g., ``{}`` for dict, ``[]`` for list, ``0``
    for int, ``None`` for None).  Callers must tolerate receiving this value.
    """

    def decorator(fn: _F) -> _F:
        step_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if _is_done(step_name):
                print(f"\n[SKIP] {step_name} already completed this session.")
                return skip_return
            result = fn(*args, **kwargs)
            _mark_done(step_name)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------


def step_fetch_market_data() -> dict:
    """Step 1 — Fetch today's benchmark values.

    Not wrapped with @idempotent_step: no side effects; fresh data wanted on
    resume; return feeds snapshots.
    """
    print("\n=== Step 1: Fetch market data ===")
    return fetch_market_data()


# ---------------------------------------------------------------------------
# Ring 1 content pipeline — step functions for the orchestrator to call
# after Claude agents have produced their {commentary, trades} output.
# Not wired into run_daily_session(); call them from the orchestrating session.
# ---------------------------------------------------------------------------


def _classify_trade(
    t: object, *, order_id: str, agent_id: str, currency: str
) -> tuple[str | None, str, "Order | None"]:
    """Pure classifier shared by the authoring loop and the narration filter so
    the two never diverge (and the filter stays correct on a resumed session
    where authoring was skipped).

    Returns ``(reason, detail, order)``. ``reason`` is None and ``order`` is set
    when the trade is a valid, authorable order; otherwise ``reason`` is the drop
    code (``order`` is None). ``detail`` is extra context for the stdout log only.
    Handles every loose shape without raising: a non-dict element, a boolean
    ``shares`` (``float(True) == 1.0`` would otherwise author a phantom 1 share),
    non-finite/non-positive shares, and any Order-validator rejection.
    """
    if not isinstance(t, dict):
        return "MALFORMED_TRADE", f" (not a dict: {type(t).__name__})", None
    # Only accept string action/ticker: a JSON null ticker would otherwise
    # stringify to "None" and slip past the missing-ticker check as a phantom.
    raw_action = t.get("action")
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""
    raw_ticker = t.get("ticker")
    ticker = raw_ticker.strip() if isinstance(raw_ticker, str) else ""
    raw_shares = t.get("shares")
    if isinstance(raw_shares, bool):  # bool is an int subclass; reject it
        shares: float | None = None
    else:
        try:
            shares = float(raw_shares)
        except (TypeError, ValueError):
            shares = None
    if action not in ("BUY", "SELL"):
        return "NON_TRADEABLE_ACTION", "", None
    if not ticker:
        return "MISSING_TICKER", "", None
    if shares is None or not math.isfinite(shares) or shares <= 0:
        return "INVALID_SHARES", "", None
    try:
        order = Order(
            order_id=order_id,
            ts=datetime.now(timezone.utc),
            agent_id=agent_id,
            action=action,
            ticker=ticker,
            shares=shares,
            reasoning=t.get("reasoning", ""),
            currency=currency,
            trigger=t.get("trigger"),
            expires=t.get("expires"),
        )
    except ValueError as exc:  # e.g. malformed trigger/expires
        return "INVALID_ORDER", f" ({exc})", None
    return None, "", order


def _normalized_trade(t: dict, order: "Order") -> dict:
    """The trade dict for narration — raw plus the normalized (uppercased action,
    stripped ticker, float shares) fields the outbox Order actually carries, so
    the story never shows ``buy`` where the trade card shows ``BUY``."""
    return {**t, "action": order.action, "ticker": order.ticker, "shares": order.shares}


def step_author_orders(
    agent_id: str, trades: list[dict], trade_date: date, currency: str
) -> list[dict]:
    """Step 3a — convert an agent's trades[] into outbox orders.

    Not wrapped with @idempotent_step: per-agent inner helper invoked via
    step_author_all; step-name would collide across agents.

    Parameters
    ----------
    agent_id:
        Agent authoring the orders (e.g. "satoshi").
    trades:
        List of trade dicts with keys: action ("BUY"|"SELL"), ticker, shares, reasoning.
    trade_date:
        The session's trading date.
    currency:
        Agent's portfolio base currency (e.g. "EUR"). Matches portfolio.currency.

    Returns
    -------
    The list of trades actually authored to the outbox (a subset of ``trades``).
    Any trade that is not a valid order — non-BUY/SELL action, missing ticker,
    non-finite/non-positive shares, or a shape the Order validator rejects (e.g.
    a malformed trigger/expires) — is dropped rather than raising, so an
    unattended session never crashes on loose agent output (2026-07-17 incident).
    Every drop is recorded to the committed dropped-trade ledger
    (``data/orders/dropped/``) with a reason code (a tamper-evident audit trail,
    not a silent skip). Callers narrate the returned list — never the raw input —
    so a dropped trade is never posted, journaled, or bundled as a phantom fill.

    Each trade may optionally include ``trigger`` and ``expires`` for conditional
    orders. See CONDITIONAL_ORDER_INSTRUCTIONS for the schema.
    """
    trades = trades or []
    print(f"\n=== Step 3a: Author orders for {agent_id} ({len(trades)} trades) ===")
    authored: list[dict] = []
    for seq, t in enumerate(trades, start=1):
        order_id = make_order_id(trade_date, agent_id, seq)
        reason, detail, order = _classify_trade(
            t, order_id=order_id, agent_id=agent_id, currency=currency
        )
        if reason is not None:
            print(f"  [SKIP] {agent_id} trade {seq}: {reason}{detail} {t!r}")
            append_dropped(
                trade_date,
                DroppedTrade(
                    ts=datetime.now(timezone.utc),
                    agent_id=agent_id,
                    reason=reason,
                    raw=t if isinstance(t, dict) else {"malformed": repr(t)},
                ),
            )
            continue
        append_order(trade_date, order)
        authored.append(_normalized_trade(t, order))
    return authored


CONDITIONAL_ORDER_INSTRUCTIONS = """\
Conditional orders (optional):
You can defer a trade until a price condition is hit. Add `trigger` and `expires`
fields to any trade in your `trades` array:

    {
      "action": "SELL", "ticker": "BTC-EUR", "shares": 0.01,
      "reasoning": "trim at resistance",
      "trigger": {"op": ">=", "level": 85000.0},
      "expires": "2026-06-17"
    }

Supported `op` values (v1): ">=" and "<=". Comparisons are inclusive at the level.
`expires` must be an ISO date (YYYY-MM-DD); on or after that date the order is
cancelled with reason TRIGGER_EXPIRED. A watcher cron evaluates triggers every
15 minutes against live prices (live for crypto via ccxt; daily-close for everything
else via the committed OHLCV store).

Safety rails (notional cap, cash check, position check, FX availability) are
evaluated AT FIRE TIME, not declaration time — so a trigger that fires after
your cash is depleted will reject with INSUFFICIENT_CASH and you'll see it in
your inbox next session. Conditional orders do not consume your daily order
cap; only the fills do (at fire time).

Cancellations: emit a `cancels` array alongside `trades` to remove pending
orders authored on previous days:

    "cancels": [
      {"target_order_id": "ord_2026-05-10_satoshi_003", "reasoning": "thesis changed"}
    ]

Your currently-active triggers are shown in the section below — review them
each session and cancel or stack as your thesis evolves.
"""


def render_active_triggers_for_agent(agent_id: str) -> str:
    """Render this agent's pending conditional orders as a human-readable list.

    Used in the daily-session prompt so the agent sees what's queued before
    authoring new trades. Returns a single string ready to drop into the prompt.
    """
    mine = [o for o in list_pending() if o.agent_id == agent_id]
    if not mine:
        return "Active triggers: (no active triggers)"
    lines = ["Active triggers:"]
    for o in mine:
        op = o.trigger["op"]
        level = o.trigger["level"]
        lines.append(
            f"  - {o.order_id}: {o.action} {o.shares} {o.ticker} "
            f"if price {op} {level:g}  (expires {o.expires})  "
            f"— reasoning: {o.reasoning}"
        )
    return "\n".join(lines)


def step_author_cancels(
    agent_id: str,
    cancels: list[dict],
    trade_date: date,
) -> int:
    """Step 3a-bis — convert an agent's cancels[] into cancel requests.

    Not wrapped with @idempotent_step: per-agent inner helper invoked via
    step_author_all; step-name would collide across agents.

    Each cancel dict requires `target_order_id` and optionally `reasoning`.
    Returns the number of cancels appended to data/orders/cancels/.
    """
    if not cancels:
        return 0
    print(
        f"\n=== Step 3a-bis: Author cancels for {agent_id} ({len(cancels)} cancels) ==="
    )
    for seq, c in enumerate(cancels, start=1):
        request_id = f"cnl_{trade_date.isoformat()}_{agent_id}_{seq:03d}"
        append_cancel(
            trade_date,
            CancelRequest(
                request_id=request_id,
                ts=datetime.now(timezone.utc),
                agent_id=agent_id,
                target_order_id=c["target_order_id"],
                reasoning=c.get("reasoning", ""),
            ),
        )
    return len(cancels)


def step_author_all(
    agent_results: dict[str, dict],
    trade_date: date,
    portfolio_manager: PortfolioManager | None = None,
) -> dict[str, dict[str, int]]:
    """Step 3 — author orders + cancels, then trim narration trades.

    Single entry point the trigger prose calls instead of looping in prose
    over `step_author_orders` (which was the 2026-05-15 leaderboard-bug pattern).

    Idempotency is hand-inlined (rather than via ``@idempotent_step``) because
    the two halves have different resume semantics under the SAME ``step_author_all``
    marker (a rename would orphan any pre-existing marker and re-author on a
    cross-deploy resume):

    - **Authoring** (outbox + cancels writes) runs once — guarded by the
      done-marker so a resumed session never double-writes orders.
    - **The narration filter** runs on BOTH paths — including the skip path — so
      ``result["trades"]`` is always trimmed to the authorable trades. Folding it
      into the guarded body would lose it on a resumed fire (``data/session_state``
      survives the sandbox ``git reset``), letting a dropped trade resurface as a
      phantom fill. The filter recomputes purely (no portfolio load — currency is
      irrelevant to classification), so it is safe even when authoring is skipped.

    Returns {agent_id: {"orders": N, "cancels": M}} for caller logging ({} on the
    skip path).
    """
    if portfolio_manager is None:
        portfolio_manager = PortfolioManager(base_dir=get_config().portfolios_dir)

    if _is_done("step_author_all"):
        print("\n[SKIP] step_author_all already completed this session.")
        _filter_narration_trades(agent_results, trade_date)
        return {}

    print("\n=== Step 3: Author orders + cancels (all agents) ===")
    summary: dict[str, dict[str, int]] = {}
    for agent_id, result in agent_results.items():
        portfolio = portfolio_manager.load(agent_id)
        authored = step_author_orders(
            agent_id,
            result.get("trades") or [],
            trade_date,
            portfolio.currency,
        )
        # Trim in-line on the normal path (already classified — no rework).
        result["trades"] = authored
        n_cancels = step_author_cancels(
            agent_id,
            result.get("cancels") or [],
            trade_date,
        )
        summary[agent_id] = {"orders": len(authored), "cancels": n_cancels}
    _mark_done("step_author_all")
    return summary


def _filter_narration_trades(agent_results: dict[str, dict], trade_date: date) -> None:
    """Replace each agent's ``result["trades"]`` with only the authorable trades
    (normalized), so the Oracle, posts, journal, and output bundle never narrate a
    dropped/invalid trade as a phantom fill. Pure and re-runnable — recomputes from
    the raw trades via the shared classifier, loading no portfolios (currency does
    not affect classification) — so it is correct and crash-free on a resumed
    session where authoring was skipped.
    """
    for agent_id, result in agent_results.items():
        kept: list[dict] = []
        for seq, t in enumerate(result.get("trades") or [], start=1):
            reason, _detail, order = _classify_trade(
                t,
                order_id=make_order_id(trade_date, agent_id, seq),
                agent_id=agent_id,
                currency="",  # unused by classification; avoids a portfolio load
            )
            if reason is None:
                kept.append(_normalized_trade(t, order))
        result["trades"] = kept


@idempotent_step(skip_return=[])
def step_fill_orders(trade_date: date, portfolio_manager: PortfolioManager) -> list:
    """Step 3b — invoke the paper broker on the day's outbox.

    The broker reads data/orders/outbox/YYYY-MM-DD.jsonl, applies safety rails,
    writes data/orders/inbox/YYYY-MM-DD.jsonl, and (for successful fills)
    mutates portfolios via PortfolioManager.apply_trade.
    """
    print("\n=== Step 3b: Fill orders (paper broker) ===")
    fills = fill_day(trade_date, portfolio_manager)
    filled = sum(1 for f in fills if f.status == "filled")
    rejected = sum(1 for f in fills if f.status == "rejected")
    print(f"  {filled} filled, {rejected} rejected out of {len(fills)}")
    return fills


# ---------------------------------------------------------------------------
# Allocator resolution (SP2 Task 5)
#
# Every manager session step is opt-in: a roster that omits the allocator block
# (role='allocator') runs each step as a clean skip. The four steps below source
# ALL of their channel dirs, book identity, prose, and gates from the resolved
# allocator spec — never from module constants — so a forker can rename channels,
# retune the risk budget, or drop the allocator entirely by editing roster.yaml
# alone. William's sole allocator (`the-manager`) resolves as the default and
# reproduces the legacy paths / prose / conviction gate byte-for-byte.
# ---------------------------------------------------------------------------


def _resolve_allocator(
    allocator_id: str | None = None,
) -> tuple[str, AllocatorSpec, AgentSpec] | tuple[None, None, None]:
    """Resolve the target allocator to ``(id, AllocatorSpec, AgentSpec)``.

    Returns ``(None, None, None)`` when the deployment configures no allocator
    (no roster entry with ``role='allocator'``) — the opt-out path every manager
    step treats as a clean skip. Ship-one default: with a single allocator and no
    explicit ``allocator_id``, resolve to that sole allocator.
    """
    cfg = get_config()
    allocs = cfg.allocators
    if not allocs:
        return None, None, None
    aid = allocator_id or allocs[0]
    return aid, cfg.allocator_spec(aid), cfg.roster[aid]


@idempotent_step(skip_return=None)
def step_build_baseline_manager(
    agent_results: dict[str, dict],
    trade_date: date | None = None,
    portfolios_dir: Path | None = None,
    ohlcv_store: Path | None = None,
    allocator_id: str | None = None,
) -> None:
    """Step 3c — run the deterministic baseline-manager rebalance.

    Internal Gate C benchmark portfolio. NOT a public trading agent.
    Excluded from the leaderboard and the public output bundle by the
    ``trading_roster`` role filter (role != trader).

    Parameters
    ----------
    agent_results:
        {agent_id: result_dict} from the session's agent round. Each result
        may contain a "research_note" key parsed by parse_research_note.
    trade_date:
        The session's trading date. Defaults to today.
    portfolios_dir:
        Override for data/portfolios/ (test monkeypatching). Derived from
        get_config().portfolios_dir when None.
    ohlcv_store:
        Override for the OHLCV store path (test monkeypatching).

    Rebalances only on the first weekday of each month, or on the very first
    run (portfolio does not exist yet). On all other days, does nothing.
    step_update_snapshots iterates portfolio dirs and will still snapshot
    baseline-manager on non-rebalance days — which is intentional.
    """
    import engine.baseline_manager as bm_module

    print("\n=== Step 3c: Baseline-manager rebalance ===")

    aid, alloc, _spec = _resolve_allocator(allocator_id)
    if aid is None or not alloc.baseline_enabled:
        print("  No allocator with baseline enabled — skipping.")
        return

    if trade_date is None:
        trade_date = date.today()

    resolved_portfolios_dir = portfolios_dir or (get_config().portfolios_dir)
    resolved_ohlcv_store = ohlcv_store or bm_module._OHLCV_STORE

    manager = PortfolioManager(base_dir=resolved_portfolios_dir)
    portfolio_exists = (
        resolved_portfolios_dir / STRATEGY_ID / "portfolio.json"
    ).exists()

    is_first_run = not portfolio_exists
    if is_first_run:
        manager.initialize(
            STRATEGY_ID, initial_capital=INITIAL_CAPITAL_EUR, currency="EUR"
        )
        print(f"  Initialized {STRATEGY_ID} portfolio (EUR {INITIAL_CAPITAL_EUR:.0f})")

    if not is_first_run and not is_rebalance_day(trade_date):
        print(
            f"  {trade_date} is not a rebalance day — skipping (snapshots will still update)."
        )
        return

    # Collect research notes from all agents.
    notes: list[tuple[str, object]] = []
    for agent_id, result in agent_results.items():
        raw_note = result.get("research_note")
        note = parse_research_note(raw_note)
        if note is not None:
            notes.append((agent_id, note))

    target = eligible_tickers(notes)
    print(f"  Eligible tickers ({len(target)}): {target if target else '(none)'}")

    portfolio_dict = manager.load(STRATEGY_ID).to_dict()

    def _price_lookup(ticker: str, on: date) -> float | None:
        from engine.ohlcv_store import latest_close_on_or_before as _lcob

        return _lcob(ticker, on, store=resolved_ohlcv_store)

    trades = rebalance(
        portfolio=portfolio_dict,
        target_tickers=target,
        price_lookup=_price_lookup,
        on=trade_date,
        position_size_eur=POSITION_SIZE_EUR,
    )

    for trade in trades:
        try:
            manager.apply_trade(STRATEGY_ID, trade)
        except ValueError as exc:
            print(
                f"  [WARN] baseline-manager trade failed ({trade.ticker} {trade.action}): {exc}"
            )

    sells = sum(1 for t in trades if t.action == "SELL")
    buys = sum(1 for t in trades if t.action == "BUY")
    print(f"  Applied {sells} sell(s) + {buys} buy(s) to {STRATEGY_ID}")


# ---------------------------------------------------------------------------
# LLM Manager (Task C5) — PAPER, fully off every public surface.
#
# Authors to a SEPARATE manager-outbox, fills into a SEPARATE the-manager book,
# and writes a manager-review audit artifact. the-manager is NOT in
# get_config().trading_roster / roster.ts / the output bundle, and its
# fills go to manager-inbox (NOT the public inbox the site joins by order_id), so
# it never leaks into the narrative. The outcome-resolution loop (Task C5b) is
# step_resolve_manager_outcomes, which runs at Step 3c-bis BEFORE this block.
# ---------------------------------------------------------------------------


@idempotent_step(skip_return=None)
def step_resolve_manager_outcomes(
    today: date,
    review_dir: Path | None = None,
    ohlcv_store: Path | None = None,
    msci_path: Path | None = None,
    resolved_path: Path | None = None,
    allocator_id: str | None = None,
) -> None:
    """Step 3c-bis — resolve matured Manager decisions into numeric outcome memory.

    Must run BEFORE step_build_manager_prompt (Step 3d) so the Manager sees
    freshly-matured outcomes in the same session that produced the underlying
    decisions.

    Reads every manager-review/{date}.json, computes forward return for each
    non-HOLD position that has reached the horizon (10 trading days by default),
    and writes the result atomically to manager-review/resolved.json.

    Parameters
    ----------
    today:
        The session's reference date (used to name the idempotency key and
        passed through to resolve_outcomes as an informational bound).
    review_dir:
        Override for data/orders/manager-review/. Derived from
        get_config().orders_dir when None (monkeypatch-friendly for tests).
    ohlcv_store:
        Override for the OHLCV store path. Derived from get_config().ohlcv_dir
        when None.
    msci_path:
        Override for the MSCI World series JSON file path. Derived from
        get_config().baselines_dir when None.
    resolved_path:
        Override for the output resolved.json path. Defaults to
        review_dir/resolved.json when None.
    """
    from engine.ohlcv_store import OHLCV_STORE
    from engine.orders import allocator_channel_dir as _order_channel_dir
    from scripts.resolve_manager_outcomes import (
        load_existing_resolved,
        resolve_outcomes,
        write_resolved,
    )

    print("\n=== Step 3c-bis: Resolve manager outcomes ===")

    aid, alloc, _spec = _resolve_allocator(allocator_id)
    if aid is None:
        print("  No allocator configured — skipping.")
        return

    resolved_review_dir = review_dir or _order_channel_dir(
        alloc.channels_prefix, "review"
    )
    resolved_store = ohlcv_store or OHLCV_STORE
    resolved_msci_path = msci_path or (
        get_config().baselines_dir / "global" / "msci_world.json"
    )
    resolved_resolved_path = resolved_path or (resolved_review_dir / "resolved.json")

    # Load MSCI series (graceful on missing/malformed).
    try:
        msci_series: list[dict] = json.loads(
            resolved_msci_path.read_text(encoding="utf-8")
        )
        if not isinstance(msci_series, list):
            msci_series = []
    except (json.JSONDecodeError, OSError):
        msci_series = []

    existing = load_existing_resolved(resolved_resolved_path)
    updated = resolve_outcomes(
        review_dir=resolved_review_dir,
        store=resolved_store,
        msci_series=msci_series,
        today=today,
        horizon_trading_days=alloc.outcome_resolution_days,
        existing_resolved=existing,
    )
    write_resolved(updated, resolved_resolved_path)
    new_count = len(updated) - len(existing)
    print(f"  Manager outcomes resolved: {new_count} new, {len(updated)} total.")


def step_build_manager_prompt(
    agent_results: dict[str, dict],
    trade_date: date,
    ohlcv_store: Path | None = None,
    allocator_id: str | None = None,
) -> str:
    """Step 3d — build the LLM allocator's decision prompt (mirror of the Oracle).

    Not wrapped with @idempotent_step: pure prompt builder with no side effects —
    on resume it must rebuild the real prompt so the orchestrator's untracked LLM
    dispatch can re-run; idempotency lives on step_apply_manager_decision.

    Assembles the C3 manager context from each agent's research note, the
    allocator's portfolio (empty book if absent), and resolved decisions. Wraps the
    rendered context with the allocator persona via wrap_persona_prompt and returns
    the prompt string. Does NOT call Claude. Returns "" when the deployment has no
    allocator (opt-out).

    Every channel dir, book identity, prose block, and memory cap is sourced from
    the resolved allocator spec — William's sole allocator reproduces the legacy
    the-manager output byte-for-byte.
    """
    from engine.manager_context import (
        build_manager_context,
        load_ticker_registry,
        render_manager_context,
        render_policy_prose,
        render_risk_budget_prose,
    )
    from engine.ohlcv_store import latest_close_on_or_before as _lcob
    from engine.orders import allocator_channel_dir as _order_channel_dir
    from engine.persona_dispatch import wrap_persona_prompt
    from engine.triggers import allocator_channel_dir as _trigger_channel_dir
    from engine.triggers import list_pending

    print("\n=== Step 3d: Build Manager prompt ===")

    aid, alloc, spec = _resolve_allocator(allocator_id)
    if aid is None:
        print("  No allocator configured — skipping.")
        return ""

    cfg = get_config()
    portfolios_dir = cfg.portfolios_dir
    resolved_store = ohlcv_store

    # Parse each agent's research note (drop None — same tolerance as the baseline).
    notes: list[tuple[str, object]] = []
    for agent_id, result in agent_results.items():
        note = parse_research_note(result.get("research_note"))
        notes.append((agent_id, note))

    # Load the allocator's portfolio if it exists, else None (→ empty book in C3).
    portfolio: dict | None = None
    manager_path = portfolios_dir / aid / "portfolio.json"
    if manager_path.exists():
        manager = PortfolioManager(base_dir=portfolios_dir)
        portfolio = manager.load(aid).to_dict()

    # Resolved decisions: written by step_resolve_manager_outcomes (Step 3c-bis, C5b).
    # That step runs BEFORE this one, so resolved.json is already up-to-date.
    resolved_decisions: list[dict] = []
    resolved_path = (
        _order_channel_dir(alloc.channels_prefix, "review") / "resolved.json"
    )
    if resolved_path.exists():
        try:
            resolved_decisions = json.loads(resolved_path.read_text(encoding="utf-8"))
            if not isinstance(resolved_decisions, list):
                resolved_decisions = []
        except (json.JSONDecodeError, OSError):
            resolved_decisions = []

    # Build a price_lookup over every ticker in scope (notes + held positions).
    scope: set[str] = set()
    for _, note in notes:
        if note is not None:
            scope.update(note.tickers)
    if portfolio is not None:
        for pos in portfolio.get("positions", []):
            scope.add(pos["ticker"])

    price_lookup: dict[str, tuple[float, str]] = {}
    for ticker in scope:
        if resolved_store is not None:
            close = _lcob(ticker, trade_date, store=resolved_store)
        else:
            close = _lcob(ticker, trade_date)
        if close is not None:
            price_lookup[ticker] = (close, trade_date.isoformat())

    active_triggers = list_pending(
        pending_dir=_trigger_channel_dir(alloc.channels_prefix, "pending")
    )

    ctx = build_manager_context(
        notes=notes,
        portfolio=portfolio,
        resolved_decisions=resolved_decisions,
        price_lookup=price_lookup,
        ticker_registry=load_ticker_registry(),
        as_of=trade_date,
        config={
            "initial_capital": spec.initial_capital,
            "currency": spec.home_currency,
            "policy_prose": render_policy_prose(
                cfg.jurisdiction,
                alloc.blocklist,
                alloc.policy_prose_override,
            ),
            "risk_budget_prose": render_risk_budget_prose(
                alloc.risk_budget, spec.home_currency, spec.initial_capital
            ),
            "outcome_memory_same_max": alloc.outcome_memory_same_max,
            "outcome_memory_other_max": alloc.outcome_memory_other_max,
        },
        active_triggers=active_triggers,
    )
    rendered = render_manager_context(ctx)
    wrapped, _model = wrap_persona_prompt(aid, rendered)
    print(
        f"  Built Manager prompt ({len(notes)} notes, {len(price_lookup)} priced,"
        f" {len(active_triggers)} active trigger(s))"
    )
    return wrapped


@idempotent_step(skip_return=None)
def step_apply_manager_decision(
    raw_decision: dict | None,
    trade_date: date,
    ohlcv_store: Path | None = None,
    allocator_id: str | None = None,
) -> None:
    """Step 3e — apply the LLM allocator's decision to its PAPER book.

    1. parse_manager_decision(raw_decision) — conviction gate applied in code
       (threshold sourced from the allocator's risk budget).
    2. Write the allocator's review audit artifact (rendered decision + full
       positions/reasoning/conviction). Written EVERY day, even on a hold — it is
       the load-bearing record of what the allocator decided and why.
    3. Convert non-HOLD positions to Orders, append to the allocator's outbox.
    4. fill_day against the allocator channel (separate outbox/inbox) and a
       PortfolioManager rooted at data/portfolios (the allocator's book, init from
       its spec if absent). All 15 rails + fees + idempotency apply identically.

    Returns cleanly (no artifacts) when the deployment has no allocator (opt-out).
    Public surfaces are untouched: orders never enter the public outbox, fills
    never enter the public inbox.
    """
    from engine.manager_decision import parse_manager_decision, render_manager_decision
    from engine.manager_orders import manager_decision_to_orders
    from engine.ohlcv_store import latest_close_on_or_before as _lcob
    from engine.orders import allocator_channel_dir as _order_channel_dir
    from engine.triggers import allocator_channel_dir as _trigger_channel_dir

    print("\n=== Step 3e: Apply Manager decision ===")

    aid, alloc, spec = _resolve_allocator(allocator_id)
    if aid is None:
        print("  No allocator configured — skipping.")
        return

    prefix = alloc.channels_prefix

    portfolios_dir = get_config().portfolios_dir
    manager = PortfolioManager(base_dir=portfolios_dir)
    if not (portfolios_dir / aid / "portfolio.json").exists():
        manager.initialize(
            aid,
            initial_capital=spec.initial_capital,
            currency=spec.home_currency,
        )
        print(
            f"  Initialized {aid} book ({spec.home_currency} {spec.initial_capital:.0f})"
        )

    decision = parse_manager_decision(
        raw_decision, min_conviction=alloc.risk_budget.min_conviction
    )

    # --- Audit artifact: written every day, hold or trade. ---
    review_dir = _order_channel_dir(prefix, "review")
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir / f"{trade_date.isoformat()}.json"
    if decision is None:
        review_payload: dict = {
            "date": trade_date.isoformat(),
            "conviction": None,
            "positions": [],
            "hold_reasoning": "",
            "render": "[Manager Decision] (no parseable decision)",
        }
    else:
        review_payload = {
            "date": trade_date.isoformat(),
            "conviction": decision.conviction,
            "positions": [p.to_dict() for p in decision.positions],
            "hold_reasoning": decision.hold_reasoning,
            "render": render_manager_decision(decision),
        }
    review_path.write_text(
        json.dumps(review_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  Wrote manager review {review_path.name}")

    if decision is None:
        print("  No parseable decision — review written, no orders.")
        return

    # --- Convert non-HOLD positions to orders (skip unpriceable). ---
    def _price(ticker: str) -> float | None:
        if ohlcv_store is not None:
            return _lcob(ticker, trade_date, store=ohlcv_store)
        return _lcob(ticker, trade_date)

    orders = manager_decision_to_orders(
        decision, trade_date, _price, agent_id=aid, currency=spec.home_currency
    )
    if not orders:
        print("  Hold (or no priceable positions) — no orders authored.")
        return

    outbox_dir = _order_channel_dir(prefix, "outbox")
    for order in orders:
        append_order(trade_date, order, outbox_dir=outbox_dir)
    print(f"  Authored {len(orders)} manager order(s) to {prefix}-outbox")

    # --- Fill on the SEPARATE allocator channel. ---
    fills = fill_day(
        trade_date,
        manager,
        outbox_dir=outbox_dir,
        inbox_dir=_order_channel_dir(prefix, "inbox"),
        pending_dir=_trigger_channel_dir(prefix, "pending"),
        cancels_dir=_trigger_channel_dir(prefix, "cancels"),
    )
    filled = sum(1 for f in fills if f.status == "filled")
    rejected = sum(1 for f in fills if f.status == "rejected")
    print(f"  Manager fills: {filled} filled, {rejected} rejected")


def step_build_post_prompts(
    agent_results: dict[str, dict],
    oracle_blog: str | None = None,
) -> dict[str, str]:
    """Step 5a — build post-generation prompts for each trading agent.

    Not wrapped with @idempotent_step: pure prompt builder with no side effects
    — on resume it must rebuild real prompts so the orchestrator's untracked LLM
    dispatch can re-run; idempotency lives on the persisting steps
    (step_save_content / step_save_memories).

    Does NOT call Claude. Returns a dict of {agent_id: prompt_str} the orchestrator
    dispatches to each agent. The Oracle is excluded — it gets a different prompt
    via step_build_oracle_prompt.

    When the post round runs AFTER the Oracle (current pipeline ordering),
    pass `oracle_blog=blog_draft.body_md` so each agent can react to the
    Oracle's framing as well as to other agents' raw moves.
    """
    print("\n=== Step 5a: Build post prompts ===")
    prompts: dict[str, str] = {}
    for agent_id in agent_results:
        if agent_id in get_config().trading_roster:  # trading agents only
            prompts[agent_id] = build_post_prompt(
                agent_id, agent_results, oracle_blog=oracle_blog
            )
    print(f"  Built {len(prompts)} post prompts")
    return prompts


def step_build_leaderboard(
    portfolio_summaries: dict[str, dict],
    on: date | None = None,
) -> list[dict]:
    """Step 5a-bis — canonical leaderboard for the day.

    Not wrapped with @idempotent_step: pure derivation from portfolios, cheap
    to recompute, and downstream consumers need real rows on resume.

    Thin wrapper around engine.leaderboard.build_leaderboard_rows so the
    same logic powers the weekend refresh script and the watcher.
    """
    from engine.leaderboard import build_leaderboard_rows

    print("\n=== Step 5a-bis: Build leaderboard ===")
    rows = build_leaderboard_rows(portfolio_summaries, on=on)
    if rows:
        print(
            f"  Ranked {len(rows)} agents (top: {rows[0]['agent']} {rows[0]['return_pct']:+.2f}%)"
        )
    else:
        print("  No agents had computable EUR-MTM — leaderboard empty.")
    return rows


@idempotent_step(skip_return=None)
def step_write_current_leaderboard(
    rows: list[dict],
    trigger: str,
) -> Path | None:
    """Step 9b — Write data/leaderboard/current.json.

    Live leaderboard artifact consumed by the site's homepage widget.
    Separate from the per-day output bundle (which stays narrative-bound).
    Idempotent: full-overwrites the file each call.
    """
    print("\n=== Step 9b: Write current leaderboard ===")
    leaderboard_dir = get_config().leaderboard_dir
    leaderboard_dir.mkdir(parents=True, exist_ok=True)
    path = leaderboard_dir / "current.json"

    now_iso = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    payload = {
        "updated_at": now_iso,
        "trigger": trigger,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"  Wrote {path} (trigger={trigger}, rows={len(rows)})")
    return path


def step_build_oracle_prompt(
    market_data: dict,
    agent_results: dict[str, dict],
    agent_posts: dict[str, list[dict]] | None = None,
    leaderboard: list[dict] | None = None,
    agent_memories: dict[str, str] | None = None,
) -> str:
    """Step 5b — build The Oracle's daily narration prompt.

    Not wrapped with @idempotent_step: pure prompt builder with no side effects
    — on resume it must rebuild real prompts so the orchestrator's untracked LLM
    dispatch can re-run; idempotency lives on the persisting steps
    (step_save_content / step_save_memories).

    Does NOT call Claude. Returns the prompt string the orchestrator dispatches
    to the-oracle agent. When `agent_memories` is provided, each agent's latest
    journal is digested into the prompt so The Oracle can quote specific entries.
    """
    print("\n=== Step 5b: Build Oracle prompt ===")
    day_number = get_day_number()
    prompt = build_oracle_prompt(
        day_number=day_number,
        market_data=market_data,
        agent_results=agent_results,
        agent_posts=agent_posts,
        leaderboard=leaderboard,
        agent_memories=agent_memories,
    )
    print(f"  Built Oracle prompt (day {day_number})")
    return prompt


def step_load_memories(agent_ids: list[str]) -> dict[str, str]:
    """Step 5c — load each agent's journal from disk for Oracle prompt assembly.

    Not wrapped with @idempotent_step: pure prompt builder with no side effects
    — on resume it must rebuild real prompts so the orchestrator's untracked LLM
    dispatch can re-run; idempotency lives on the persisting steps
    (step_save_content / step_save_memories).

    Returns a dict keyed by agent_id. Missing journals become empty strings so
    the Oracle prompt can still render a "first session" marker.
    """
    print("\n=== Step 5c: Load agent memories ===")
    memories = {aid: load_journal(aid) for aid in agent_ids}
    non_empty = sum(1 for v in memories.values() if v.strip())
    print(f"  Loaded {non_empty}/{len(agent_ids)} non-empty journals")
    return memories


def step_build_memory_update_prompts(
    agent_results: dict[str, dict],
    agent_posts: dict[str, list[dict]],
    portfolio_summaries: dict[str, dict],
    day_number: int | None = None,
) -> dict[str, str]:
    """Step 7a — build session-end journal-rewrite prompts for every agent.

    Not wrapped with @idempotent_step: pure prompt builder with no side effects
    — on resume it must rebuild real prompts so the orchestrator's untracked LLM
    dispatch can re-run; idempotency lives on the persisting steps
    (step_save_content / step_save_memories).

    Does NOT call Claude. Returns {agent_id: prompt} the orchestrator dispatches.
    Covers all 11 agents (the 10 traders plus the-oracle). Each agent reads its
    current journal from disk in-prompt; we embed it here so the dispatched
    prompt is fully self-contained.

    ``day_number`` defaults to None and is computed internally via
    ``get_day_number()`` when not supplied. The trigger-doc call omits this
    argument (the production contract); callers that already hold the day number
    may still pass it explicitly to avoid the extra I/O.
    """
    if day_number is None:
        day_number = get_day_number()
    print("\n=== Step 7a: Build memory-update prompts ===")
    prompts: dict[str, str] = {}
    # Traders
    for agent_id, result in agent_results.items():
        prompts[agent_id] = build_memory_update_prompt(
            agent_id=agent_id,
            day_number=day_number,
            current_journal=load_journal(agent_id),
            trades_today=result.get("trades", []),
            posts_today=agent_posts.get(agent_id, []),
            portfolio_summary=portfolio_summaries.get(agent_id, {}),
        )
    # The Oracle doesn't trade; its journal update prompt has no trades.
    prompts["the-oracle"] = build_memory_update_prompt(
        agent_id="the-oracle",
        day_number=day_number,
        current_journal=load_journal("the-oracle"),
        trades_today=[],
        posts_today=agent_posts.get("the-oracle", []),
        portfolio_summary={"currency": "EUR"},
    )
    print(f"  Built {len(prompts)} memory-update prompts")
    return prompts


@idempotent_step(skip_return=0)
def step_save_memories(new_journals: dict[str, str]) -> int:
    """Step 7b — persist rewritten journals back to data/agent_memory/.

    Parameters
    ----------
    new_journals:
        {agent_id: new_journal_content} returned by orchestrator after dispatch.
        Empty/blank values are skipped so a partial round doesn't wipe a journal.

    Returns the number of journals actually written.
    """
    print("\n=== Step 7b: Save updated memories ===")
    written = 0
    for agent_id, content in new_journals.items():
        if not content or not content.strip():
            print(f"  [SKIP] {agent_id}: empty response")
            continue
        save_journal(agent_id, content)
        written += 1
    print(f"  Saved {written}/{len(new_journals)} journals")
    return written


def build_portfolio_summaries() -> dict[str, dict]:
    """Build the canonical per-agent portfolio summary dict for ALL 10 trading
    agents. Use this output as the `portfolio_summaries` argument to
    `step_save_content` so the bundle's agents map carries forward last-known
    portfolio state for non-running agents (weekend cadence, etc.).

    Reads `data/portfolios/{agent_id}/portfolio.json` via PortfolioManager.
    Agents with no portfolio.json on disk are skipped (defensive — should not
    happen in production after Day 1).

    Summary shape: {cash, deployed, positions, currency}
    where `positions` is the Portfolio.to_dict() position list and
    `deployed` is `portfolio.cost_basis`.
    """
    portfolios_dir = get_config().portfolios_dir
    manager = PortfolioManager(base_dir=portfolios_dir)

    summaries: dict[str, dict] = {}
    for agent_id in get_config().trading_roster:
        if not (portfolios_dir / agent_id / "portfolio.json").exists():
            continue
        portfolio = manager.load(agent_id)
        d = portfolio.to_dict()
        summaries[agent_id] = {
            "cash": d["cash"],
            "deployed": portfolio.cost_basis,
            "positions": d["positions"],
            "currency": d["currency"],
        }
    return summaries


@idempotent_step(skip_return={})
def step_save_content(
    bundle_date: date,
    market_data: dict,
    agent_results: dict[str, dict],
    agent_posts: dict[str, list[PostPayload]],
    portfolio_summaries: dict[str, dict],
    leaderboard: list[dict],
    blog_draft,
    oracle_posts: list[PostPayload],
) -> dict:
    """Step 6 — persist posts, blog draft, and output bundle.

    Returns the assembled bundle dict so callers can log its shape or pass it
    onwards (e.g. to a future publisher).
    """
    print("\n=== Step 6: Save content ===")
    posts_path = save_daily_posts(bundle_date, agent_posts)
    print(f"  Saved posts → {posts_path.name}")
    blog_path = save_daily_blog_draft(bundle_date, blog_draft)
    print(f"  Saved blog draft → {blog_path.name}")
    bundle = assemble_output_bundle(
        bundle_date=bundle_date,
        market_data=market_data,
        agent_results=agent_results,
        agent_posts=agent_posts,
        portfolio_summaries=portfolio_summaries,
        leaderboard=leaderboard,
        blog_draft=blog_draft,
        oracle_posts=oracle_posts,
    )
    bundle_path = save_output_bundle(bundle_date, bundle)
    print(f"  Saved output bundle → {bundle_path.name}")
    return bundle


def _compute_positions_value(
    portfolio: Portfolio, on: date, store: Path | None = None
) -> float:
    """Mark a portfolio's open positions to market using per-ticker latest closes.

    For each position, walks the OHLCV store for the most recent close at or
    before `on`. Falls back to avg_cost when no row exists. This carries
    European tickers forward when their same-day close hasn't landed yet —
    avoids the NaN portfolio_value bug where pandas left-joined a pricing
    DataFrame whose `iloc[-1]` row contained NaN for lagging markets.
    """
    total = 0.0
    for p in portfolio.positions:
        price = latest_close_on_or_before(p.ticker, on, store=store)
        if price is None:
            price = p.avg_cost
        total += p.shares * price
    return total


@idempotent_step(skip_return=[])
def step_update_snapshots(market_payload: dict) -> list[str]:
    """Step 4 — Append daily snapshots for all active portfolios.

    A portfolio is "active" if it has a portfolio.json on disk.

    Note: this iterates portfolio dirs, so the internal `the-manager` and
    `baseline-manager` books accrue committed snapshots here. That is intentional
    private valuation tracking — both are excluded from every public surface by
    the ``trading_roster`` role filter (role != trader), so this is not a leak.

    Parameters
    ----------
    market_payload:
        The dict returned by fetch_market_data, containing "date" and
        "benchmarks".

    Returns
    -------
    list[str]
        Strategy IDs that were snapshotted.
    """
    print("\n=== Step 4: Update daily snapshots ===")

    portfolios_dir = get_config().portfolios_dir
    if not portfolios_dir.exists():
        print("  No portfolios directory found — skipping.")
        return []

    manager = PortfolioManager(base_dir=portfolios_dir)
    snapshot_date = date.fromisoformat(market_payload["date"])
    benchmarks = market_payload["benchmarks"]

    snapshotted: list[str] = []

    for portfolio_dir in sorted(portfolios_dir.iterdir()):
        if not portfolio_dir.is_dir():
            continue
        portfolio_json = portfolio_dir / "portfolio.json"
        if not portfolio_json.exists():
            continue

        strategy_id = portfolio_dir.name

        try:
            portfolio = manager.load(strategy_id)
        except Exception as exc:
            print(f"  [WARN] Could not load {strategy_id}: {exc}")
            continue

        positions_value = _compute_positions_value(portfolio, snapshot_date)
        portfolio_value = portfolio.cash + positions_value

        manager.add_snapshot(
            strategy_id=strategy_id,
            snapshot_date=snapshot_date,
            portfolio_value=portfolio_value,
            cash=portfolio.cash,
            positions_value=positions_value,
            benchmarks=benchmarks,
        )

        print(
            f"  Snapshotted {strategy_id}: value={portfolio_value:.2f}, cash={portfolio.cash:.2f}"
        )
        snapshotted.append(strategy_id)

    if not snapshotted:
        print("  No active portfolios found.")

    return snapshotted


@idempotent_step(skip_return=None)
def step_build_baselines() -> None:
    """Step 9a — Baselines.

    Recomputes data/baselines/ for Day 1 → today, full-rewrite and idempotent.
    Runs AFTER portfolio mutations so the benchmark window matches the
    freshly-appended agent snapshots. Uses backfill_baselines constants as
    the single source of truth for universes + max_positions.
    """
    print("\n=== Step 9a: Build baselines ===")
    from datetime import date as _date

    from engine.baselines import build_all_baselines
    from scripts.backfill_baselines import _max_positions_by_agent, _universes_by_agent

    cfg = get_config()
    build_all_baselines(
        universes_by_agent=_universes_by_agent(),
        from_date=cfg.day_one,
        to_date=_date.today(),
        max_positions_by_agent=_max_positions_by_agent(),
    )


@idempotent_step(skip_return=None)
def step_build_tax_shadow() -> None:
    """Step 9c — After-tax shadow ledger (reporting only).

    Reads each agent's data/portfolios/{agent}/trades.json and writes
    data/tax_shadow/{agent}.json with realized PFU estimates per French
    tax law.  Runs AFTER step_build_baselines.  Pure computation — never
    mutates portfolio state.
    """
    print("\n=== Step 9c: Build tax shadow ledgers ===")
    from scripts.build_tax_shadow import build_tax_shadow_all

    written = build_tax_shadow_all(
        portfolios_dir=get_config().portfolios_dir,
        output_dir=get_config().tax_shadow_dir,
    )
    print(f"  Wrote {len(written)} tax shadow ledger(s).")


def step_git_commit_push(dry_run: bool = False) -> None:
    """Step 5 — Git commit and push data changes.

    Not wrapped with @idempotent_step because it calls ``_clear_state()`` on
    success, which wipes the whole state file — so a post-clear ``_mark_done``
    would be immediately lost.  Re-entry protection is still wired manually via
    ``_is_done`` at the top; ``_mark_done`` is intentionally absent (the cleared
    state file is the finished-session signal).
    """
    _step_name = "step_git_commit_push"
    if _is_done(_step_name):
        print(f"\n[SKIP] {_step_name} already completed this session.")
        return

    print("\n=== Step 5: Git commit and push ===")

    if dry_run:
        print("  [DRY RUN] Skipping git operations.")
        # no _mark_done: _clear_state() below removes the whole state file — a finished session leaves no state
        _clear_state()
        return

    data_dir = str(_PROJECT_ROOT / "data")

    try:
        # Stage data/ changes.
        subprocess.run(["git", "add", data_dir], cwd=_PROJECT_ROOT, check=True)

        # Commit any staged data changes the orchestrator hasn't already
        # committed. (Orchestrators that commit themselves with a richer
        # message — "chore: weekday session …" — will land here with nothing
        # left staged, which is fine.)
        diff_result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=_PROJECT_ROOT,
        )
        if diff_result.returncode != 0:
            today_str = date.today().isoformat()
            commit_msg = f"chore: daily snapshot {today_str}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=_PROJECT_ROOT,
                check=True,
            )
            print(f"  Committed: {commit_msg}")

        # Always push HEAD to origin/main. Cloud sandbox sessions
        # (RemoteTrigger) check out a throwaway branch like `claude/<slug>`;
        # without an explicit refspec, `git push` would publish that branch
        # instead of advancing main, leaving the daily snapshot off the
        # public deploy. Fast-forward only — anything else is a real
        # conflict that should fail loudly.
        ahead = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            cwd=_PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if int(ahead.stdout.strip() or "0") == 0:
            print("  Nothing to push — HEAD is at origin/main.")
        else:
            # Primary path: push directly to origin/main. Fallback path
            # (added 2026-05-08 after the harness started 403'ing main pushes
            # from cloud sandboxes): push the sandbox branch instead, and let
            # .github/workflows/auto-merge-session.yml take the merge to main.
            push_main = subprocess.run(
                ["git", "push", "origin", "HEAD:main"],
                cwd=_PROJECT_ROOT,
                capture_output=True,
                text=True,
            )
            if push_main.returncode == 0:
                print("  Pushed to origin/main.")
            else:
                stderr = (push_main.stderr or "").strip()
                stdout = (push_main.stdout or "").strip()
                print(f"  [WARN] Push to origin/main failed: {stderr or stdout}")
                print(
                    "  Falling back to push current branch — auto-merge-session.yml will take it to main."
                )

                subprocess.run(
                    ["git", "push", "origin", "HEAD"],
                    cwd=_PROJECT_ROOT,
                    check=True,
                )
                branch_name = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=_PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                print(
                    f"  Pushed to sandbox branch '{branch_name}'. Watch for auto-merge-session workflow on GitHub."
                )

    except subprocess.CalledProcessError as exc:
        print(f"  [ERROR] Git operation failed: {exc}")
        raise

    # no _mark_done: _clear_state() below removes the whole state file — a finished session leaves no state
    _clear_state()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_daily_session(dry_run: bool = False) -> None:
    """Snapshot-only EOD run.

    Dispatches no Claude agents — the full trading round is driven by an
    orchestrating Claude Code session that calls the step_* helpers directly.
    Use this CLI for manual snapshot refreshes or CI health checks.
    """
    print(f"Midas daily snapshot — {date.today().isoformat()}")
    print("=" * 50)

    market_payload = step_fetch_market_data()
    step_update_snapshots(market_payload)
    step_build_baselines()
    step_build_tax_shadow()
    step_git_commit_push(dry_run=dry_run)

    print("\n=== Snapshot complete ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Midas daily trading session orchestrator."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps but skip the git commit and push.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_daily_session(dry_run=args.dry_run)
