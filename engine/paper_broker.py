"""Paper broker — Hands side of the Brain/Hands split.

Reads orders from data/orders/outbox/, applies the safety rails, fills at end-of-day
close from the committed OHLCV store (latest-on-or-before the trade date — critical
because the daily session fires at 20:00 UTC but fetch-ohlcv.yml runs at 22:30 UTC),
writes data/orders/inbox/, mutates portfolios via PortfolioManager.apply_trade.

Rejection reason codes:
- INVALID_SHARES: malformed outbox line or shares <= 0
- MAX_ORDERS_PER_DAY: per-agent daily order cap exceeded
- MAX_ORDER_NOTIONAL: order notional (base currency) > per-agent cap
- TICKER_NOT_IN_UNIVERSE: allowed_universe is non-empty and ticker not in union
- NO_PRICE_DATA: no row in OHLCV store for ticker <= trade_date
- CURRENCY_UNRESOLVED: ticker's quote currency is in neither map and its suffix is unknown
- PRICE_IMPLAUSIBLE: fill price outside [1/5, 5]x its reference (prior close on BUY, avg_cost on SELL)
- TRIGGER_LEVEL_IMPLAUSIBLE: conditional order's level outside [0.2, 5.0]x the latest close
- VALUATION_UNAVAILABLE: the book cannot be valued, so its relative rails cannot be evaluated
- NO_FX_RATE: ticker currency ≠ base and no FX rate available to convert notional
- INSUFFICIENT_CASH: BUY cost > portfolio cash (post earlier fills)
- NO_POSITION_TO_SELL: SELL on a ticker not held
- INSUFFICIENT_SHARES: SELL shares > held shares
- FEE_EXCEEDS_PROCEEDS: SELL whose fee is >= the gross proceeds (nets <= 0 cash)
- DAILY_DRAWDOWN_HALT: agent's drawdown <= cap; ALL their orders rejected
- APPLY_TRADE_FAILED: PortfolioManager.apply_trade raised; broker continues with next order
- TRIGGER_NO_EXPIRY: conditional order without an expires date (agent error)
- CANCELLED_BY_AGENT: cancel request matched a pending order; pending file removed
- CANCEL_TARGET_NOT_FOUND: cancel request targeted an order_id not in pending
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from engine.config import get_config
from engine.fees import fee_for
from engine.fx import convert as fx_convert
from engine.orders import Fill, Order, append_fill, inbox_order_ids
from engine.quotes import (
    _load_ticker_currency_overrides,  # noqa: F401 — re-exported, see below
    latest_price,
    store_quote,
    ticker_currency,
)
from engine.triggers import (
    delete_pending,
    read_cancels,
    save_pending,
)
from engine.portfolio import PortfolioManager
from engine.types import Trade
from engine.universes import resolve_universe
from engine.valuation import mtm_base_currency

# The OHLCV store, agent-config dir, and ticker-currency override path are all
# resolved lazily through get_config() inside the functions that use them, so a
# forker who sets MIDAS_DATA_DIR (the Hands reading prices from the redirected
# store) is honoured — nothing is frozen at import. Price reads default to
# latest_close_on_or_before's own lazy store default (get_config().ohlcv_dir).

logger = logging.getLogger(__name__)

#: The complete set of rejection/cancel reason codes this broker can emit.
#: The module docstring above documents what each one means; `tests/test_reason_codes.py`
#: asserts the three views (this set, the docstring, the emitted literals) agree.
#: The watcher in `scripts/check_triggers.py` owns a sixteenth code, TRIGGER_EXPIRED,
#: which is deliberately NOT in this set — it is a different enforcement point.
REJECTION_REASON_CODES = frozenset(
    {
        "INVALID_SHARES",
        "MAX_ORDERS_PER_DAY",
        "MAX_ORDER_NOTIONAL",
        "TICKER_NOT_IN_UNIVERSE",
        "NO_PRICE_DATA",
        "CURRENCY_UNRESOLVED",
        "PRICE_IMPLAUSIBLE",
        "TRIGGER_LEVEL_IMPLAUSIBLE",
        "VALUATION_UNAVAILABLE",
        "NO_FX_RATE",
        "INSUFFICIENT_CASH",
        "NO_POSITION_TO_SELL",
        "INSUFFICIENT_SHARES",
        "FEE_EXCEEDS_PROCEEDS",
        "DAILY_DRAWDOWN_HALT",
        "APPLY_TRADE_FAILED",
        "TRIGGER_NO_EXPIRY",
        "CANCELLED_BY_AGENT",
        "CANCEL_TARGET_NOT_FOUND",
    }
)


@dataclass
class AgentConfig:
    """Per-agent safety rails sourced from roster.yaml via get_config().roster[agent_id].safety.

    daily_drawdown_halt_pct uses NEGATIVE values. The broker halts all of the
    agent's orders when the computed drawdown % is strictly less than this value
    (i.e. -7.0 < -5.0 → halt). A value of 0.0 disables the halt for that agent.

    Agents not in the roster (e.g. baseline-manager) receive safe defaults:
    max_order_notional=500, max_orders_per_day=5, daily_drawdown_halt_pct=-5.0,
    allowed_universe=[], dry_run=False — identical to the original broker's
    missing-file defaults (commit 320e0d53).

    Note: the-manager IS now in the roster (role: allocator) with an explicit
    safety block reproducing those same defaults.  It is excluded from the
    public trading surfaces by the ``trading_roster`` role filter (role !=
    trader).
    """

    max_order_notional: float
    max_orders_per_day: int
    daily_drawdown_halt_pct: float
    allowed_universe: list[str]
    dry_run: bool
    max_order_notional_pct: float | None = None

    @classmethod
    def load(cls, agent_id: str) -> "AgentConfig":
        spec = get_config().roster.get(agent_id)
        if spec is None:
            return cls(
                max_order_notional=500.0,
                max_orders_per_day=5,
                daily_drawdown_halt_pct=-5.0,
                allowed_universe=[],
                dry_run=False,
            )
        s = spec.safety
        return cls(
            max_order_notional=s.max_order_notional,
            max_order_notional_pct=s.max_order_notional_pct,
            max_orders_per_day=s.max_orders_per_day,
            daily_drawdown_halt_pct=s.daily_drawdown_halt_pct,
            allowed_universe=list(s.allowed_universe),
            dry_run=s.dry_run,
        )


#: A fill price more than this multiple of its reference — or less than its
#: reciprocal — is refused. The reference is the prior stored close on a BUY
#: and the position's own `avg_cost` on a SELL, both already in the ticker's
#: ISO currency, so the comparison is unit-clean.
#:
#: 5x is deliberately loose. It is not a market-move detector: a stock can
#: legitimately halve, and a crypto pair can legitimately double. What it
#: catches is the class that has actually happened here — a unit or basis
#: error, which arrives as a factor of 100 (pence quoted as pounds) or of the
#: split ratio. Two orders of magnitude of headroom over a 100x error, and
#: nothing in the committed ledger sits anywhere near the band.
PRICE_BAND = 5.0

#: A conditional order's trigger level outside this multiple of the latest
#: stored close is refused at intake, before it can ever fire. All 68 live
#: pending orders sit within [0.77, 2.06]; the 2026-08-07 pence-stop (level
#: 111.00 against a 1.16 price) sits at 95.7.
TRIGGER_LEVEL_MIN = 0.2
TRIGGER_LEVEL_MAX = 5.0


def _price_out_of_band(price: float, reference: float | None) -> bool:
    """True when `price` is implausibly far from `reference`.

    A missing or non-positive reference means "no opinion" — the check
    abstains rather than guessing, because the alternative is rejecting every
    first-ever BUY in a ticker.
    """
    if reference is None or reference <= 0 or price <= 0:
        return False
    ratio = price / reference
    return ratio > PRICE_BAND or ratio < 1.0 / PRICE_BAND


def _reference_price(order: Order, portfolio, trade_date: date) -> float | None:
    """What today's fill price for `order` should look roughly like.

    SELL compares against the position's own `avg_cost` — the price the book
    actually paid, in the ticker's currency, which is the tightest reference
    available and the one that catches a basis change under an existing
    holding.

    BUY has no position to lean on, so it compares against the **prior**
    stored close rather than today's: today's close is the very number under
    suspicion, and comparing it to itself is the classic check that cannot
    fail. `None` (a first-ever BUY, or a ticker with a single stored row)
    means the check abstains.
    """
    if order.action == "SELL":
        position = next(
            (p for p in portfolio.positions if p.ticker == order.ticker), None
        )
        return position.avg_cost if position is not None else None
    previous = latest_price(order.ticker, trade_date - timedelta(days=1))
    return previous.price if previous is not None else None


def _trigger_level_out_of_band(order: Order, trade_date: date) -> bool:
    """True when a conditional order's level is nowhere near the live price.

    Caught at **intake**, not at fire time: a pending order sits on disk for
    days, and the point is that it never becomes an armed instruction in the
    first place. This is the rail for incident #9 — a stop authored in pence
    (level 111.00) against a ticker trading at GBP 1.16, which no other check
    in the system could see.

    Abstains when the ticker has no price and when the level is non-positive
    (the Order validator already refuses that shape).
    """
    if order.trigger is None:
        return False
    level = order.trigger.get("level")
    if not isinstance(level, (int, float)) or level <= 0:
        return False
    quote = latest_price(order.ticker, trade_date)
    if quote is None or quote.price <= 0:
        return False
    ratio = level / quote.price
    return ratio < TRIGGER_LEVEL_MIN or ratio > TRIGGER_LEVEL_MAX


def _book_value(
    agent_id: str, portfolio_manager: PortfolioManager, on: date
) -> float | None:
    """The book's current value in its own currency, for the relative rails.

    Falls back through: today's mark-to-market → the last published snapshot
    → the roster's initial capital. The fallbacks matter because the cap has
    to exist on day one (no snapshot yet) and must not evaporate the moment
    an FX rate is missing. `None` only when the agent is off-roster AND has
    no snapshot, which the caller treats as "cannot evaluate the rail".
    """
    portfolio = portfolio_manager.load(agent_id)
    value = mtm_base_currency(portfolio.to_dict(), on)
    if value is not None:
        return value
    snaps = portfolio_manager.load_snapshots(agent_id)
    if snaps:
        return snaps[-1].get("portfolio_value")
    spec = get_config().roster.get(agent_id)
    return spec.initial_capital if spec is not None else None


def _notional_cap(config: AgentConfig, book_value: float | None) -> float | None:
    """The per-order notional ceiling, in the book's base currency.

    `None` means the rail cannot be evaluated — a percentage cap against a
    book that will not value. The caller refuses the order rather than
    treating an unknown ceiling as no ceiling.
    """
    if config.max_order_notional_pct is None:
        return config.max_order_notional
    if book_value is None:
        return None
    return book_value * config.max_order_notional_pct / 100.0


# Currency resolution moved to engine.quotes (2026-08-07). It was a pure
# ticker→currency helper living inside the execution layer while three other
# pricing modules imported it, and its suffix heuristic mapped every `.L`
# ticker to GBP — but the LSE quotes in pence, so `.L` positions were valued
# 100x high, and `PHAG.L` is not even sterling. These names stay re-exported
# because engine.restatement, engine.valuation, scripts/daily_session and the
# test suite all import them from here.
_ticker_currency = ticker_currency


def _drawdown_pct(
    agent_id: str, portfolio_manager: PortfolioManager, today: date
) -> float | None:
    """Drawdown % from the most recent snapshot, in the portfolio's base currency.

    Returns 0.0 — a known "no drawdown" — when no snapshot exists yet (first
    day of the experiment) or when the previous value was zero.

    Returns **None** when today's value cannot be computed at all, because a
    held position's currency differs from the book's own and the FX rate
    needed to convert it is unavailable (`mtm_base_currency` returns `None`
    there — see `engine.valuation.portfolio_mtm`). This used to be 0.0, on a
    "can't determine it, don't halt on unknown data" reading. That reading is
    backwards for a rail: it makes the halt *disappear* precisely when the
    book has become unvaluable, which is the state most likely to accompany a
    real problem. The caller now refuses the batch with
    VALUATION_UNAVAILABLE instead — a rail that cannot be evaluated is not a
    rail that passed.
    """
    snaps = portfolio_manager.load_snapshots(agent_id)
    if not snaps:
        return 0.0
    portfolio = portfolio_manager.load(agent_id)
    summary = portfolio.to_dict()
    today_value = mtm_base_currency(summary, today)
    if today_value is None:
        return None
    prev_value = snaps[-1]["portfolio_value"]
    if prev_value == 0:
        return 0.0
    return (today_value - prev_value) / prev_value * 100.0


def _current_commit_sha() -> str | None:
    """HEAD commit SHA the broker is executing against, or None outside a git repo.

    Stamped onto every Fill (engine.orders.Fill.executed_sha) for tamper-evident
    provenance: anyone can ``git checkout <sha>`` and re-derive the exact outbox
    order and price store the broker saw. Returns None — never raises — when git
    is unavailable or the working tree is not a repo, so the audit stamp can
    degrade silently without ever blocking a fill.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_config().data_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    sha = result.stdout.strip()
    return sha or None


def _reject(order_id: str, reason: str) -> Fill:
    return Fill(
        order_id=order_id,
        ts_filled=datetime.now(timezone.utc),
        status="rejected",
        fill_price=None,
        fill_currency=None,
        notional_base=None,
        fees=None,
        reason=reason,
    )


def _resolve_allowed_tickers(allowed_universe: list[str], agent_id: str) -> set[str]:
    """Resolve an agent's allowed_universe list to a set of tickers.

    An empty ``allowed_universe`` means "no restriction" (allow-all) and yields
    an empty set — the callers treat an empty set as allow-everything.

    Fail-open guard: if ``allowed_universe`` is NON-empty but every entry fails
    to resolve (unknown or unimplemented-placeholder universe), that is a loud
    config error — raising, not silently returning an empty set. An empty set
    from a non-empty config would otherwise disable the TICKER_NOT_IN_UNIVERSE
    rail and let the agent trade anything.
    """
    if not allowed_universe:
        return set()

    allowed_tickers: set[str] = set()
    for u in allowed_universe:
        try:
            allowed_tickers.update(resolve_universe(u))
        except KeyError:
            logger.warning(
                "Unknown/unimplemented universe %s in %s config", u, agent_id
            )

    if not allowed_tickers:
        raise ValueError(
            f"allowed_universe {allowed_universe!r} for {agent_id} resolved to an "
            f"empty allowlist — refusing to fail open (allow-all). Fix the roster "
            f"universe id(s) or remove the restriction (allowed_universe: [])."
        )
    return allowed_tickers


def _read_outbox_lines(
    trade_date: date, outbox_dir: Path | None = None
) -> tuple[list[Order], list[str]]:
    """Read outbox JSONL with defensive parsing.

    Returns (orders, invalid_order_ids). Malformed lines (bad JSON or shares<=0
    or other Order validation failure) produce synthesized IDs in invalid_order_ids
    so they can be reported as INVALID_SHARES rejections instead of crashing.

    ``outbox_dir`` defaults to the public OUTBOX_DIR (resolved at call time so test
    monkeypatching is respected). Pass MANAGER_OUTBOX_DIR for the Manager channel.
    """
    # Delayed import — respects test monkeypatching of OUTBOX_DIR.
    from engine import orders as orders_module

    orders: list[Order] = []
    invalid_ids: list[str] = []
    base = outbox_dir if outbox_dir is not None else orders_module.OUTBOX_DIR
    path = base / f"{trade_date.isoformat()}.jsonl"
    if not path.exists():
        return orders, invalid_ids

    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Try to parse JSON. On failure, synthesize an ID.
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid outbox JSON on line %d: %s", idx, exc)
                invalid_ids.append(f"malformed_{trade_date.isoformat()}_{idx:03d}")
                continue
            # Try to build an Order. On failure (shares<=0, missing keys, bad action),
            # fall back to the raw order_id if available.
            try:
                orders.append(Order.from_dict(raw))
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning("Invalid outbox order on line %d: %s", idx, exc)
                oid = raw.get("order_id") if isinstance(raw, dict) else None
                invalid_ids.append(
                    oid or f"malformed_{trade_date.isoformat()}_{idx:03d}"
                )

    return orders, invalid_ids


def _process_one(
    order: Order,
    config: AgentConfig,
    portfolio_manager: PortfolioManager,
    trade_date: date,
    filled_count: int,
    allowed_tickers: set[str],
) -> Fill:
    """Validate and either fill or reject a single order.

    Does NOT mutate filled_count; the caller tracks that. Does mutate the
    portfolio via apply_trade when the order fills (unless dry_run).
    """
    if filled_count >= config.max_orders_per_day:
        return _reject(order.order_id, "MAX_ORDERS_PER_DAY")

    if allowed_tickers and order.ticker not in allowed_tickers:
        return _reject(order.order_id, "TICKER_NOT_IN_UNIVERSE")

    # Asked before the price read, so the two failures stay distinguishable:
    # latest_price returns None both for "no row in the store" and for "no
    # resolvable currency", and an operator reading the inbox needs to know
    # which. An unresolvable currency is a registry gap, not a data gap.
    if ticker_currency(order.ticker) is None:
        return _reject(order.order_id, "CURRENCY_UNRESOLVED")

    # store defaults to get_config().ohlcv_dir inside latest_price — resolved at
    # call time so MIDAS_DATA_DIR redirection reaches the broker's fills.
    # latest_price returns the close ALREADY in the ticker's ISO currency: a
    # pence-quoted LSE close arrives here as pounds, so `price` (and therefore
    # Trade.price, Fill.fill_price and the stored avg_cost) is always in
    # `ticker_ccy` — the unit Fill.fill_currency claims it is in.
    quote = latest_price(order.ticker, trade_date)
    if quote is None:
        return _reject(order.order_id, "NO_PRICE_DATA")
    price, ticker_ccy = quote

    portfolio = portfolio_manager.load(order.agent_id)
    base_ccy = portfolio.currency

    if _price_out_of_band(price, _reference_price(order, portfolio, trade_date)):
        return _reject(order.order_id, "PRICE_IMPLAUSIBLE")

    notional_native = order.shares * price

    if ticker_ccy == base_ccy:
        notional_base = notional_native
    else:
        converted = fx_convert(notional_native, ticker_ccy, base_ccy, trade_date)
        if converted is None:
            return _reject(order.order_id, "NO_FX_RATE")
        notional_base = converted

    cap = _notional_cap(
        config, _book_value(order.agent_id, portfolio_manager, trade_date)
    )
    if cap is None:
        return _reject(order.order_id, "VALUATION_UNAVAILABLE")
    if notional_base > cap:
        return _reject(order.order_id, "MAX_ORDER_NOTIONAL")

    fee = fee_for(order.ticker, notional_base, base_ccy, trade_date)

    if order.action == "BUY" and notional_base + fee > portfolio.cash:
        return _reject(order.order_id, "INSUFFICIENT_CASH")

    if order.action == "SELL":
        position = next(
            (p for p in portfolio.positions if p.ticker == order.ticker), None
        )
        if position is None:
            return _reject(order.order_id, "NO_POSITION_TO_SELL")
        if order.shares > position.shares:
            return _reject(order.order_id, "INSUFFICIENT_SHARES")
        if fee >= notional_base:
            # Fee floor (e.g. equity €1.25) at least eats the proceeds: the
            # sale would net <= 0 cash. Reject rather than book a value-losing
            # fill (portfolio.cash += total - fees).
            return _reject(order.order_id, "FEE_EXCEEDS_PROCEEDS")

    trade = Trade(
        id=order.order_id,
        timestamp=order.ts,
        action=order.action,
        ticker=order.ticker,
        shares=order.shares,
        price=price,
        total=notional_base,
        fees=fee,
        reasoning=order.reasoning,
    )

    if not config.dry_run:
        try:
            portfolio_manager.apply_trade(order.agent_id, trade)
        except ValueError as exc:
            logger.warning("apply_trade failed for %s: %s", order.order_id, exc)
            return _reject(order.order_id, "APPLY_TRADE_FAILED")

    return Fill(
        order_id=order.order_id,
        ts_filled=datetime.now(timezone.utc),
        status="filled",
        fill_price=price,
        fill_currency=ticker_ccy,
        notional_base=notional_base,
        fees=fee,
        reason=None,
    )


def fill_day(
    trade_date: date,
    portfolio_manager: PortfolioManager,
    outbox_dir: Path | None = None,
    inbox_dir: Path | None = None,
    pending_dir: Path | None = None,
    cancels_dir: Path | None = None,
) -> list[Fill]:
    """Fill all outbox orders for a trade date.

    Order of operations:
      1. Process cancel requests — remove targeted pending orders, write
         CANCELLED_BY_AGENT (or CANCEL_TARGET_NOT_FOUND) rejections to inbox.
      2. Read outbox, split into conditional (trigger set) and market (trigger None).
      3. Conditional orders → save_pending() with TRIGGER_NO_EXPIRY rejection
         for any missing the expires field. No inbox fill on successful registration.
      4. Market orders → existing per-agent processing (drawdown halt, rails, fill).

    Malformed outbox lines (bad JSON or shares<=0) produce INVALID_SHARES rejections
    rather than crashing the pass.

    ``outbox_dir`` / ``inbox_dir`` default to the public OUTBOX_DIR / INBOX_DIR.
    Pass the Manager channel dirs (MANAGER_OUTBOX_DIR / MANAGER_INBOX_DIR, Task C5)
    to fill a SEPARATE outbox into a SEPARATE inbox without touching the public
    flow.

    ``pending_dir`` / ``cancels_dir`` default to the public PENDING_DIR / CANCELS_DIR.
    Pass MANAGER_PENDING_DIR / MANAGER_CANCELS_DIR (from engine.triggers) to route
    conditional orders and cancel requests on the Manager channel to their own dirs,
    keeping them isolated from the public trigger watcher's files.

    The portfolio is still selected per order via the order's agent_id, so a
    the-manager order routes to data/portfolios/the-manager/ via the passed
    portfolio_manager. All 15 rails, fees, and idempotency apply identically on
    either channel. With all four channel args None the behaviour is byte-for-byte
    identical to the legacy single-channel path.
    """
    fills: list[Fill] = []

    # Resolve the executing commit once per batch — every fill in this pass was
    # produced against the same HEAD. _emit stamps it before the disk write so the
    # provenance lands in the committed JSONL, not just the in-memory list.
    executed_sha = _current_commit_sha()

    def _emit(f: Fill) -> None:
        f.executed_sha = executed_sha
        fills.append(f)
        append_fill(trade_date, f, inbox_dir=inbox_dir)

    # Load order_ids already in today's inbox before processing anything.
    # Any order_id found here is skipped silently — it was processed in a
    # prior run of fill_day for the same date (e.g. a session restart after
    # a push failure). This makes fill_day structurally idempotent.
    # inbox_order_ids reads engine.orders.INBOX_DIR at call time when inbox_dir
    # is None, so test monkeypatching of that attribute is respected automatically.
    already_processed: set[str] = inbox_order_ids(trade_date, inbox_dir=inbox_dir)

    # --- Pass 1: process cancel requests ---
    # Note: within a single run, duplicate cancels targeting the same order_id
    # are allowed to produce multiple inbox lines (the first removes/rejects,
    # subsequent ones see the pending file already gone → CANCEL_TARGET_NOT_FOUND).
    # The cross-run idempotency guard (checking already_processed) handles re-runs:
    # on the second run, the target_order_id will already be in the inbox and the
    # cancel entries are skipped.
    for cancel in read_cancels(trade_date, cancels_dir=cancels_dir):
        if cancel.target_order_id in already_processed:
            continue
        removed = delete_pending(cancel.target_order_id, pending_dir=pending_dir)
        reason = "CANCELLED_BY_AGENT" if removed else "CANCEL_TARGET_NOT_FOUND"
        _emit(_reject(cancel.target_order_id, reason))

    # --- Pass 2: read outbox and split conditional vs market ---
    orders, invalid_ids = _read_outbox_lines(trade_date, outbox_dir=outbox_dir)
    for oid in invalid_ids:
        if oid in already_processed:
            continue
        _emit(_reject(oid, "INVALID_SHARES"))
        already_processed.add(oid)

    market_orders: list[Order] = []
    for o in orders:
        if o.order_id in already_processed:
            continue
        if o.trigger is None:
            market_orders.append(o)
            continue
        if o.expires is None:
            _emit(_reject(o.order_id, "TRIGGER_NO_EXPIRY"))
            already_processed.add(o.order_id)
            continue
        if _trigger_level_out_of_band(o, trade_date):
            _emit(_reject(o.order_id, "TRIGGER_LEVEL_IMPLAUSIBLE"))
            already_processed.add(o.order_id)
            continue
        save_pending(o, pending_dir=pending_dir)
        # No inbox record on successful registration — the agent sees it
        # next session in their "Active triggers" prompt section.

    # --- Pass 3: existing market-order fill loop ---
    by_agent: dict[str, list[Order]] = {}
    for o in market_orders:
        by_agent.setdefault(o.agent_id, []).append(o)

    for agent_id, agent_orders in by_agent.items():
        config = AgentConfig.load(agent_id)

        # A book that will not value fails the batch closed. The drawdown rail
        # is the one rail that can stop a whole day's trading, and letting it
        # silently pass on an unvaluable book (the previous 0.0 fallback) meant
        # it switched itself off exactly when the book was in an unusual state.
        drawdown = _drawdown_pct(agent_id, portfolio_manager, trade_date)
        if drawdown is None:
            for o in agent_orders:
                _emit(_reject(o.order_id, "VALUATION_UNAVAILABLE"))
                already_processed.add(o.order_id)
            continue

        if drawdown < config.daily_drawdown_halt_pct:
            for o in agent_orders:
                _emit(_reject(o.order_id, "DAILY_DRAWDOWN_HALT"))
                already_processed.add(o.order_id)
            continue

        allowed_tickers = _resolve_allowed_tickers(config.allowed_universe, agent_id)

        filled = 0
        for o in agent_orders:
            f = _process_one(
                o, config, portfolio_manager, trade_date, filled, allowed_tickers
            )
            _emit(f)
            already_processed.add(o.order_id)
            if f.status == "filled":
                filled += 1

    return fills


def execute_triggered_order(
    order: Order,
    trade_date: date,
    portfolio_manager: PortfolioManager,
    fire_price: float,
    *,
    inbox_dir: Path | None = None,
) -> Fill | None:
    """Run the rails, then stamp the executing commit SHA on the resulting Fill.

    Thin public wrapper around _execute_triggered_order so the watcher's fired
    fills carry the same git provenance (Fill.executed_sha) as same-session market
    fills. A None result (idempotent no-op — already in inbox) is passed through
    unstamped: there is no fill to attribute.
    """
    fill = _execute_triggered_order(
        order, trade_date, portfolio_manager, fire_price, inbox_dir=inbox_dir
    )
    if fill is not None:
        fill.executed_sha = _current_commit_sha()
    return fill


def _execute_triggered_order(
    order: Order,
    trade_date: date,
    portfolio_manager: PortfolioManager,
    fire_price: float,
    *,
    inbox_dir: Path | None = None,
) -> Fill | None:
    """Execute a fired conditional order through the same safety rails as market orders.

    Differences from market-order processing:
      - `fire_price` is the live price observed by the watcher, used as fill_price
        instead of a store read. It is ALREADY ISO-denominated (ccxt, or the
        store, which has been normalised at ingest since 2026-08-07), so
        `engine.quotes.store_quote` only labels it — it applies no scaling, and
        must not: scaling here would divide every LSE fire price by 100 a second
        time. The rails (notional cap, cash check, position check) are therefore
        evaluated in the ticker's ISO currency, exactly as on the market path.
      - The returned Fill always has trigger_fired=True so the agent and the site
        can distinguish scheduled fills from market fills.
      - Does NOT consult MAX_ORDERS_PER_DAY (a triggered fire is not a same-day order).
      - Does NOT consult DAILY_DRAWDOWN_HALT — that rail lives at the fill_day batch
        level, not inside _process_one. A triggered fire that should be halted by
        drawdown will still fire here; the agent sees the fill in their inbox and
        can re-author cautiously next session. Revisit if this becomes a problem.
        Does still respect MAX_ORDER_NOTIONAL, TICKER_NOT_IN_UNIVERSE, INSUFFICIENT_CASH,
        NO_POSITION_TO_SELL, INSUFFICIENT_SHARES, NO_FX_RATE, APPLY_TRADE_FAILED.

    ``inbox_dir`` scopes the idempotency scan. Defaults to the public INBOX_DIR;
    pass MANAGER_INBOX_DIR when firing a Manager-channel conditional order so the
    guard scans the correct channel. Mirror of the same kwarg on fill_day.

    Caller is responsible for appending the returned Fill to the inbox and removing
    the pending file (so the watcher can decide policy if it wants).

    Returns None if the order_id already appears in ANY file under ``inbox_dir``
    (any date), meaning this order was already filled or rejected in a prior watcher
    run. The caller must treat None as a no-op: do not write a second inbox line,
    do not mutate the portfolio, do not remove the pending file again.
    """
    # Idempotency check: scan all inbox files for this order_id before executing.
    # Triggered orders may fire days after authoring, so the existing fill can
    # live in any date's inbox file — not just today's.
    # inbox_order_ids resolves inbox_dir at call time (defaulting to INBOX_DIR),
    # so test monkeypatching of that attribute is respected automatically.
    if order.order_id in inbox_order_ids(None, inbox_dir=inbox_dir):
        logger.info(
            "execute_triggered_order: %s already in inbox — skipping", order.order_id
        )
        return None

    config = AgentConfig.load(order.agent_id)
    portfolio = portfolio_manager.load(order.agent_id)
    base_ccy = portfolio.currency
    # `fire_price` comes from the watcher, which reads it from ccxt or from the
    # OHLCV store — both ISO-denominated since the store was normalised at
    # ingest (2026-08-07). This is the one pricing path that cannot go through
    # latest_price (the price is handed in, not read here), so it attaches the
    # currency explicitly rather than pairing the raw value with
    # `ticker_currency` by hand. It must NOT scale: doing so would divide every
    # LSE fire price by 100 a second time.
    denominated = store_quote(order.ticker, fire_price)
    if denominated is None:
        f = _reject(order.order_id, "CURRENCY_UNRESOLVED")
        f.trigger_fired = True
        return f
    fire_price, ticker_ccy = denominated

    # The fire price came from ccxt or from the store; the store is the
    # reference either way. A SELL leans on the position's own avg_cost, as
    # on the market path. This is the check that would have refused a pence
    # level firing against a pounds book.
    if order.action == "SELL":
        reference = _reference_price(order, portfolio, trade_date)
    else:
        stored = latest_price(order.ticker, trade_date)
        reference = stored.price if stored is not None else None
    if _price_out_of_band(fire_price, reference):
        f = _reject(order.order_id, "PRICE_IMPLAUSIBLE")
        f.trigger_fired = True
        return f

    notional_native = order.shares * fire_price

    if ticker_ccy == base_ccy:
        notional_base = notional_native
    else:
        converted = fx_convert(notional_native, ticker_ccy, base_ccy, trade_date)
        if converted is None:
            f = _reject(order.order_id, "NO_FX_RATE")
            f.trigger_fired = True
            return f
        notional_base = converted

    allowed_tickers = _resolve_allowed_tickers(config.allowed_universe, order.agent_id)

    if allowed_tickers and order.ticker not in allowed_tickers:
        f = _reject(order.order_id, "TICKER_NOT_IN_UNIVERSE")
        f.trigger_fired = True
        return f

    cap = _notional_cap(
        config, _book_value(order.agent_id, portfolio_manager, trade_date)
    )
    if cap is None:
        f = _reject(order.order_id, "VALUATION_UNAVAILABLE")
        f.trigger_fired = True
        return f
    if notional_base > cap:
        f = _reject(order.order_id, "MAX_ORDER_NOTIONAL")
        f.trigger_fired = True
        return f

    fee = fee_for(order.ticker, notional_base, base_ccy, trade_date)

    if order.action == "BUY" and notional_base + fee > portfolio.cash:
        f = _reject(order.order_id, "INSUFFICIENT_CASH")
        f.trigger_fired = True
        return f

    if order.action == "SELL":
        position = next(
            (p for p in portfolio.positions if p.ticker == order.ticker), None
        )
        if position is None:
            f = _reject(order.order_id, "NO_POSITION_TO_SELL")
            f.trigger_fired = True
            return f
        if order.shares > position.shares:
            f = _reject(order.order_id, "INSUFFICIENT_SHARES")
            f.trigger_fired = True
            return f
        if fee >= notional_base:
            f = _reject(order.order_id, "FEE_EXCEEDS_PROCEEDS")
            f.trigger_fired = True
            return f

    trade = Trade(
        id=order.order_id,
        timestamp=order.ts,
        action=order.action,
        ticker=order.ticker,
        shares=order.shares,
        price=fire_price,
        total=notional_base,
        fees=fee,
        reasoning=order.reasoning,
    )

    if not config.dry_run:
        try:
            portfolio_manager.apply_trade(order.agent_id, trade)
        except ValueError as exc:
            logger.warning(
                "apply_trade failed for triggered order %s: %s", order.order_id, exc
            )
            f = _reject(order.order_id, "APPLY_TRADE_FAILED")
            f.trigger_fired = True
            return f

    return Fill(
        order_id=order.order_id,
        ts_filled=datetime.now(timezone.utc),
        status="filled",
        fill_price=fire_price,
        fill_currency=ticker_ccy,
        notional_base=notional_base,
        fees=fee,
        reason=None,
        trigger_fired=True,
    )


if __name__ == "__main__":
    from datetime import date as _date

    from engine.config import get_config as _gc
    from engine.portfolio import PortfolioManager as _PM

    _cfg = _gc()
    _pm = _PM(base_dir=_cfg.data_dir / "data" / "portfolios")
    _fills = fill_day(_date.today(), _pm)
    _filled = sum(1 for f in _fills if f.status == "filled")
    _rejected = sum(1 for f in _fills if f.status == "rejected")
    print(f"fill-day: {_filled} filled, {_rejected} rejected out of {len(_fills)}")
