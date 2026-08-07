"""Orders — outbox/inbox serde for the Brain/Hands split.

Order: agent-authored trade request (in data/orders/outbox/YYYY-MM-DD.jsonl).
Fill:  paper broker confirmation (in data/orders/inbox/YYYY-MM-DD.jsonl).

Both are append-only JSONL, one record per line. UTC timestamps serialized with
Z suffix for readability; deserialization round-trips cleanly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from engine.config import get_config

# Order-channel directories are resolved lazily through ``get_config()`` (never
# frozen at import) so MIDAS_DATA_DIR redirection takes effect. The public names
# below remain readable as module attributes via ``__getattr__`` (PEP 562):
#
#   OUTBOX_DIR / INBOX_DIR            — public Brain/Hands trade flow.
#   MANAGER_OUTBOX_DIR / _INBOX_DIR  — the LLM Manager's isolated channel (Task
#     C5). Its orders, fills, and decision-review audit artifact live here —
#     NEVER in the public outbox/inbox the site joins by order_id. the-manager is
#     not in get_config().trading_roster, so it is auto-excluded from the output
#     bundle, leaderboard, and journals; routing its fills here keeps it off the
#     trade-card join as well.
#   MANAGER_REVIEW_DIR               — Manager decision-review artifacts.

TRIGGER_OPS: tuple[str, ...] = ("<=", ">=")


@dataclass
class Order:
    """Agent-authored trade request. Validated at construction.

    Long-only invariant: shares must be strictly positive — any attempt at
    short-selling (negative shares) or no-op orders (zero) is rejected.

    Optional fields for conditional orders:
      - trigger: {"op": ">="|"<=", "level": float} — fires when current price
        crosses level in the given direction. None → fills immediately end-of-day.
      - expires: ISO date string (YYYY-MM-DD). On or after this date the
        watcher cancels the pending order with reason TRIGGER_EXPIRED.
    """

    order_id: str
    ts: datetime
    agent_id: str
    action: str  # "BUY" | "SELL"
    ticker: str
    shares: float
    reasoning: str
    currency: str
    trigger: dict | None = None
    expires: str | None = None

    def __post_init__(self) -> None:
        if not (self.shares > 0):
            raise ValueError(f"Order.shares must be > 0, got {self.shares}")
        if self.action not in ("BUY", "SELL"):
            raise ValueError(
                f"Order.action must be 'BUY' or 'SELL', got {self.action!r}"
            )
        if self.trigger is not None:
            if not isinstance(self.trigger, dict):
                raise ValueError(
                    f"Order.trigger must be a dict, got {type(self.trigger).__name__}"
                )
            op = self.trigger.get("op")
            if op not in TRIGGER_OPS:
                raise ValueError(
                    f"Order.trigger.op must be one of {TRIGGER_OPS}, got {op!r}"
                )
            level = self.trigger.get("level")
            if not isinstance(level, (int, float)) or isinstance(level, bool):
                raise ValueError("Order.trigger.level must be a number")
        if self.expires is not None and self.trigger is None:
            raise ValueError("Order.expires requires trigger to be set")
        if self.expires is not None:
            try:
                date.fromisoformat(self.expires)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Order.expires must be ISO date YYYY-MM-DD, got {self.expires!r}"
                ) from exc

    def to_dict(self) -> dict:
        d = {
            "order_id": self.order_id,
            "ts": self.ts.isoformat().replace("+00:00", "Z"),
            "agent_id": self.agent_id,
            "action": self.action,
            "ticker": self.ticker,
            "shares": self.shares,
            "reasoning": self.reasoning,
            "currency": self.currency,
        }
        if self.trigger is not None:
            d["trigger"] = self.trigger
        if self.expires is not None:
            d["expires"] = self.expires
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Order":
        return cls(
            order_id=d["order_id"],
            ts=datetime.fromisoformat(d["ts"].replace("Z", "+00:00")),
            agent_id=d["agent_id"],
            action=d["action"],
            ticker=d["ticker"],
            shares=float(d["shares"]),
            reasoning=d.get("reasoning", ""),
            currency=d["currency"],
            trigger=d.get("trigger"),
            expires=d.get("expires"),
        )


@dataclass
class Fill:
    """Paper broker confirmation.

    Status is "filled" or "rejected"; reason set only on rejections.
    trigger_fired: True when the fill came from a conditional order whose
      trigger condition was hit by the watcher (not a same-session market fill).

    Currency convention (filled orders):
      - fill_price, fill_currency — the ticker's NATIVE currency (e.g., MSFT → USD)
      - notional_base             — the agent's BASE currency (post-FX conversion)
    This asymmetry means a USD ticker bought by an EUR agent produces:
        fill_price=400.0, fill_currency="USD", notional_base=360.0  (EUR-equivalent).
    The `_base` suffix is explicit so downstream consumers never confuse the
    two — critical for audit trails and tax reporting later.
    """

    order_id: str
    ts_filled: datetime
    status: str  # "filled" | "rejected"
    fill_price: float | None
    fill_currency: str | None
    notional_base: float | None
    fees: float | None
    reason: str | None
    trigger_fired: bool = False
    # Provenance: the git HEAD commit the broker was executing against when this
    # fill was produced. Tamper-evident audit trail — `git checkout <executed_sha>`
    # re-derives the exact outbox order and price store the broker saw. Stamped by
    # the broker (engine.paper_broker); None when run outside a git repo.
    executed_sha: str | None = None

    def __post_init__(self) -> None:
        if self.status not in ("filled", "rejected"):
            raise ValueError(
                f"Fill.status must be 'filled' or 'rejected', got {self.status!r}"
            )

    def to_dict(self) -> dict:
        d = {
            "order_id": self.order_id,
            "ts_filled": self.ts_filled.isoformat().replace("+00:00", "Z"),
            "status": self.status,
            "fill_price": self.fill_price,
            "fill_currency": self.fill_currency,
            "notional_base": self.notional_base,
            "fees": self.fees,
            "reason": self.reason,
        }
        if self.trigger_fired:
            d["trigger_fired"] = True
        if self.executed_sha is not None:
            d["executed_sha"] = self.executed_sha
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Fill":
        return cls(
            order_id=d["order_id"],
            ts_filled=datetime.fromisoformat(d["ts_filled"].replace("Z", "+00:00")),
            status=d["status"],
            fill_price=d.get("fill_price"),
            fill_currency=d.get("fill_currency"),
            notional_base=d.get("notional_base"),
            fees=d.get("fees"),
            reason=d.get("reason"),
            trigger_fired=bool(d.get("trigger_fired", False)),
            executed_sha=d.get("executed_sha"),
        )


@dataclass
class DroppedTrade:
    """A trade the Brain dropped before it reached the broker.

    Some agent output is not a valid order — a lowercase/HOLD action, a missing
    ticker, non-finite/non-positive shares, or a shape the Order validator
    rejects (e.g. a malformed trigger/expires). Rather than crash the unattended
    session at Order construction (2026-07-17 incident) or drop it with no trace,
    the authoring step records it here (data/orders/dropped/YYYY-MM-DD.jsonl,
    committed) so the git ledger keeps a tamper-evident audit trail — the Brain-
    side analogue of the broker's inbox rejection codes. ``raw`` preserves the
    trade dict exactly as the agent emitted it.
    """

    ts: datetime
    agent_id: str
    reason: (
        str  # NON_TRADEABLE_ACTION | MISSING_TICKER | INVALID_SHARES | INVALID_ORDER
    )
    raw: dict

    def to_dict(self) -> dict:
        return {
            "ts": self.ts.isoformat().replace("+00:00", "Z"),
            "agent_id": self.agent_id,
            "reason": self.reason,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DroppedTrade":
        return cls(
            ts=datetime.fromisoformat(d["ts"].replace("Z", "+00:00")),
            agent_id=d["agent_id"],
            reason=d["reason"],
            raw=d.get("raw", {}),
        )


def make_order_id(d: date, agent_id: str, seq: int) -> str:
    """Deterministic order ID: ord_{iso_date}_{agent_id}_{seq:03d}."""
    return f"ord_{d.isoformat()}_{agent_id}_{seq:03d}"


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        try:
            return [json.loads(line) for line in f if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def append_order(d: date, order: Order, outbox_dir: Path | None = None) -> None:
    """Append an order to the outbox JSONL for date ``d``.

    ``outbox_dir`` defaults to the public OUTBOX_DIR. Pass MANAGER_OUTBOX_DIR to
    write to the separate Manager channel (Task C5). The default is resolved at
    call time through get_config() so MIDAS_DATA_DIR redirection is respected.
    """
    base = outbox_dir if outbox_dir is not None else get_config().orders_dir / "outbox"
    _append_jsonl(base / f"{d.isoformat()}.jsonl", order.to_dict())


def read_outbox(d: date, outbox_dir: Path | None = None) -> list[Order]:
    base = outbox_dir if outbox_dir is not None else get_config().orders_dir / "outbox"
    return [Order.from_dict(r) for r in _read_jsonl(base / f"{d.isoformat()}.jsonl")]


def append_fill(d: date, fill: Fill, inbox_dir: Path | None = None) -> None:
    """Append a fill to the inbox JSONL for date ``d``.

    ``inbox_dir`` defaults to the public INBOX_DIR. Pass MANAGER_INBOX_DIR to
    write to the separate Manager channel (Task C5). The default is resolved at
    call time through get_config() so MIDAS_DATA_DIR redirection is respected.
    """
    base = inbox_dir if inbox_dir is not None else get_config().orders_dir / "inbox"
    _append_jsonl(base / f"{d.isoformat()}.jsonl", fill.to_dict())


def read_inbox(d: date, inbox_dir: Path | None = None) -> list[Fill]:
    base = inbox_dir if inbox_dir is not None else get_config().orders_dir / "inbox"
    return [Fill.from_dict(r) for r in _read_jsonl(base / f"{d.isoformat()}.jsonl")]


def append_dropped(
    d: date, record: DroppedTrade, dropped_dir: Path | None = None
) -> None:
    """Append a dropped-trade audit record for date ``d`` (committed ledger)."""
    base = (
        dropped_dir if dropped_dir is not None else get_config().orders_dir / "dropped"
    )
    _append_jsonl(base / f"{d.isoformat()}.jsonl", record.to_dict())


def read_dropped(d: date, dropped_dir: Path | None = None) -> list[DroppedTrade]:
    base = (
        dropped_dir if dropped_dir is not None else get_config().orders_dir / "dropped"
    )
    return [
        DroppedTrade.from_dict(r) for r in _read_jsonl(base / f"{d.isoformat()}.jsonl")
    ]


def inbox_order_ids(d: date | None = None, inbox_dir: Path | None = None) -> set[str]:
    """Collect all order_ids already present in inbox JSONL files.

    If `d` is given, scan only that day's inbox file.
    If `d` is None, scan ALL inbox files (for execute_triggered_order which
    must check the full history — a triggered order may have been authored days
    earlier and fired on a later date, landing in a different inbox file).

    ``inbox_dir`` defaults to the public INBOX_DIR. Pass MANAGER_INBOX_DIR so the
    idempotency check operates on the Manager channel (Task C5). The default is
    resolved at call time through get_config() so MIDAS_DATA_DIR redirection is
    respected.

    Reads raw JSONL lines and extracts the ``order_id`` field. **A malformed
    line raises** (2026-08-07 review, W7.4).

    This used to skip silently, on the reasoning that "a corrupt line cannot
    retroactively cause a double-fill if the original write succeeded". That
    reasoning is wrong, and the error is instructive: it conflates the write
    having succeeded with the record being *readable*. If the unparseable line
    is the fill record for order X, then X is missing from this set, and the
    only thing standing between X and a second fill is exactly this set. A
    hole here is a hole in the money path.

    So it fails loud, matching `_read_jsonl`'s contract for the same data. The
    cost of raising is bounded and visible: the watcher goes red and files an
    issue. The cost of continuing is paying twice.

    Raises
    ------
    ValueError
        If any line in any inbox file is not valid JSON.
    """
    base = inbox_dir if inbox_dir is not None else get_config().orders_dir / "inbox"
    if d is not None:
        paths = [base / f"{d.isoformat()}.jsonl"]
    else:
        if not base.exists():
            return set()
        paths = list(base.glob("*.jsonl"))

    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for idx, raw_line in enumerate(f, 1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {idx} of {path} — refusing to "
                        f"build the already-filled set from an unreadable "
                        f"ledger, because a missing order_id here permits a "
                        f"double-fill: {exc}"
                    ) from exc
                oid = record.get("order_id")
                if oid:
                    ids.add(oid)
    return ids


def allocator_channel_dir(prefix: str, kind: str) -> "Path":
    """Resolve an allocator channel dir: data/orders/{prefix}-{kind}.
    kind ∈ {outbox, inbox, review}. Default prefix 'manager' reproduces the
    legacy MANAGER_*_DIR paths byte-for-byte."""
    return get_config().orders_dir / f"{prefix}-{kind}"


_LAZY_DIRS = {
    "OUTBOX_DIR": "outbox",
    "INBOX_DIR": "inbox",
    "MANAGER_OUTBOX_DIR": "manager-outbox",
    "MANAGER_INBOX_DIR": "manager-inbox",
    "MANAGER_REVIEW_DIR": "manager-review",
}


def __getattr__(name: str) -> object:
    """Resolve the order-channel dir constants lazily (PEP 562).

    Readers (``engine.orders.OUTBOX_DIR``, ``from engine.orders import INBOX_DIR``,
    the broker's ``orders_module.OUTBOX_DIR``, scripts) get the CURRENT config's
    path at access time, so MIDAS_DATA_DIR redirection is honoured and nothing is
    frozen at import. Tests that need a different location pass the explicit
    ``*_dir`` argument or redirect via MIDAS_DATA_DIR.
    """
    sub = _LAZY_DIRS.get(name)
    if sub is not None:
        return get_config().orders_dir / sub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
