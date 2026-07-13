"""Conditional triggers — pending order storage, cancel I/O, evaluation.

Pending orders live as one JSON file per order at
data/orders/pending/{order_id}.json (easy to add/remove individually, no
concurrency issue at typical volumes — agents author maybe a dozen each).

Cancellations live as append-only JSONL at
data/orders/cancels/YYYY-MM-DD.jsonl (mirrors outbox/inbox shape).

Evaluation and price-fetch dispatch are added in subsequent tasks.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from engine.config import get_config
from engine.orders import Order

logger = logging.getLogger(__name__)

# Pending/cancel channel dirs are resolved lazily through get_config() (never
# frozen at import) so MIDAS_DATA_DIR redirection takes effect. The public names
# PENDING_DIR / CANCELS_DIR / MANAGER_PENDING_DIR / MANAGER_CANCELS_DIR remain
# readable as module attributes via __getattr__ (PEP 562).


@dataclass
class CancelRequest:
    """Agent-authored cancellation of a pending conditional order.

    Recorded in data/orders/cancels/YYYY-MM-DD.jsonl. The broker processes
    these during fill_day: each cancel removes the target from PENDING_DIR
    and writes a rejection-style Fill (reason="CANCELLED_BY_AGENT") to the
    inbox so the next session sees what was cancelled.
    """

    request_id: str
    ts: datetime
    agent_id: str
    target_order_id: str
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "ts": self.ts.isoformat().replace("+00:00", "Z"),
            "agent_id": self.agent_id,
            "target_order_id": self.target_order_id,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CancelRequest":
        return cls(
            request_id=d["request_id"],
            ts=datetime.fromisoformat(d["ts"].replace("Z", "+00:00")),
            agent_id=d["agent_id"],
            target_order_id=d["target_order_id"],
            reasoning=d.get("reasoning", ""),
        )


# ---------- Pending order storage ----------


def save_pending(order: Order, pending_dir: Path | None = None) -> None:
    """Persist a conditional order to its per-order JSON file. Overwrites if same order_id."""
    if order.trigger is None:
        raise ValueError(
            f"save_pending requires order.trigger, got None for {order.order_id}"
        )
    base = (
        pending_dir if pending_dir is not None else get_config().orders_dir / "pending"
    )
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{order.order_id}.json"
    path.write_text(json.dumps(order.to_dict(), indent=2), encoding="utf-8")


def list_pending(pending_dir: Path | None = None) -> list[Order]:
    """Return all pending conditional orders, deterministic by order_id."""
    base = (
        pending_dir if pending_dir is not None else get_config().orders_dir / "pending"
    )
    if not base.exists():
        return []
    out: list[Order] = []
    for path in sorted(base.glob("*.json")):
        try:
            out.append(Order.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed pending file %s: %s", path.name, exc)
            continue
    return out


def delete_pending(order_id: str, pending_dir: Path | None = None) -> bool:
    """Remove a pending order's file. Returns True if removed, False if absent."""
    base = (
        pending_dir if pending_dir is not None else get_config().orders_dir / "pending"
    )
    path = base / f"{order_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


# ---------- Cancel request I/O ----------


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def append_cancel(
    d: date, cancel: CancelRequest, cancels_dir: Path | None = None
) -> None:
    base = (
        cancels_dir if cancels_dir is not None else get_config().orders_dir / "cancels"
    )
    _append_jsonl(base / f"{d.isoformat()}.jsonl", cancel.to_dict())


def read_cancels(d: date, cancels_dir: Path | None = None) -> list[CancelRequest]:
    base = (
        cancels_dir if cancels_dir is not None else get_config().orders_dir / "cancels"
    )
    path = base / f"{d.isoformat()}.jsonl"
    if not path.exists():
        return []
    out: list[CancelRequest] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(CancelRequest.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed cancel line %d in %s: %s", idx, path.name, exc
                )
    return out


# ---------- Evaluation ----------


def evaluate_trigger(price: float, trigger: dict) -> bool:
    """Return True if the price satisfies the trigger condition.

    Supported ops (v1): ">=" and "<=". Comparisons are inclusive at the level.
    """
    op = trigger["op"]
    level = float(trigger["level"])
    if op == ">=":
        return price >= level
    if op == "<=":
        return price <= level
    raise ValueError(f"unknown trigger op: {op!r}")


def is_expired(order: Order, today: date) -> bool:
    """Return True if today is on or after the order's expiry date.

    Expiry is inclusive: an order with expires=2026-05-17 is expired on 2026-05-17.
    Orders with no expires field never expire (defensive — the authoring step
    enforces expiry, but this keeps the watcher safe if a manually-edited file
    is missing the field).
    """
    if order.expires is None:
        return False
    return today >= date.fromisoformat(order.expires)


# ---------- Price fetch dispatch ----------

from engine.ohlcv_store import (
    latest_close_on_or_before,
)  # placed here to keep module-level imports tidy

_CRYPTO_BASES = frozenset(
    {
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "ADA",
        "DOGE",
        "DOT",
        "LINK",
        "LTC",
        "BCH",
        "AVAX",
        "ATOM",
        "XLM",
        "FIL",
        "MATIC",
        "UNI",
    }
)
_CRYPTO_QUOTES = frozenset({"EUR", "USD", "GBP", "JPY", "CHF"})


def is_crypto_ticker(ticker: str) -> bool:
    """True for tickers like BTC-EUR/ETH-USD where the base is in the crypto allowlist."""
    if "-" not in ticker:
        return False
    base, _, quote = ticker.partition("-")
    return base in _CRYPTO_BASES and quote in _CRYPTO_QUOTES


_crypto_exchange = None


def _get_crypto_exchange():
    """Lazy-init a single ccxt exchange. Coinbase is the primary, no auth needed.

    Module-level singleton so we don't pay init cost per price fetch.
    """
    global _crypto_exchange
    if _crypto_exchange is None:
        import ccxt  # local import — keep top of module clean

        _crypto_exchange = ccxt.coinbase()
    return _crypto_exchange


def get_current_price(ticker: str, today: date) -> float | None:
    """Return latest price for trigger evaluation.

    - Crypto (BTC-EUR etc.): live fetch via ccxt (Coinbase). Returns None on any error.
    - Everything else: latest close from the committed OHLCV store on-or-before `today`.

    The crypto path is intraday and 24/7; equity/FX triggers effectively re-evaluate
    once per day, after fetch-ohlcv.yml updates the store post-close.
    """
    if is_crypto_ticker(ticker):
        try:
            exchange = _get_crypto_exchange()
            base, _, quote = ticker.partition("-")
            symbol = f"{base}/{quote}"
            tick = exchange.fetch_ticker(symbol)
            last = tick.get("last")
            return float(last) if last is not None else None
        except Exception:
            # ccxt raises a wide variety of exception classes; treat all as "price unavailable
            # right now" and carry the pending order forward.
            return None
    return latest_close_on_or_before(ticker, today)


def allocator_channel_dir(prefix: str, kind: str) -> "Path":
    """Resolve an allocator pending/cancels dir: data/orders/{prefix}-{kind}.
    kind ∈ {pending, cancels}. Default prefix 'manager' == legacy paths."""
    return get_config().orders_dir / f"{prefix}-{kind}"


_LAZY_DIRS = {
    "PENDING_DIR": "pending",
    "CANCELS_DIR": "cancels",
    "MANAGER_PENDING_DIR": "manager-pending",
    "MANAGER_CANCELS_DIR": "manager-cancels",
}


def __getattr__(name: str) -> object:
    """Resolve the pending/cancel channel dir constants lazily (PEP 562).

    Readers (``engine.triggers.PENDING_DIR``, ``from engine.triggers import
    MANAGER_PENDING_DIR``, scripts) get the CURRENT config's path at access time,
    so MIDAS_DATA_DIR redirection is honoured and nothing is frozen at import.
    Tests that need a different location pass the explicit ``*_dir`` argument or
    redirect via MIDAS_DATA_DIR.
    """
    sub = _LAZY_DIRS.get(name)
    if sub is not None:
        return get_config().orders_dir / sub
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
