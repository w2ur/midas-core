"""Pure read/compute helpers for the private Streamlit Manager page.

The Manager (`the-manager`) is the real-money *decision* path: it consumes
structured research notes, authors orders on a separate channel, fills into a
private paper book, and is graded against a deterministic twin
(`baseline-manager`) toward a Gate C go/no-go decision. None of this is part of
the public narrative — it is rendered only on the local Streamlit dashboard
(`app/pages/06_manager.py`), which is not deployed.

This module isolates the filesystem reads and the small NAV-comparison maths so
the page stays a thin view and the logic is unit-testable without a UI. It
deliberately depends only on the stdlib (no pandas / plotly) — those live in the
page.

Data sources (all under the repo's ``data/`` tree):
  - ``portfolios/the-manager/{portfolio,snapshots}.json``
  - ``portfolios/baseline-manager/{portfolio,snapshots}.json``
  - ``orders/manager-review/{date}.json``   — one daily decision artifact
  - ``orders/manager-review/resolved.json`` — matured 10-day outcomes

Every reader degrades gracefully: a missing or malformed file yields an empty /
``None`` result rather than raising, because the page must render cleanly before
the Manager has ever run.
"""

from __future__ import annotations

import json
from pathlib import Path

from engine.config import get_config
from engine.orders import allocator_channel_dir as _order_dir

# Snapshot record: {"date": "YYYY-MM-DD", "portfolio_value": float, ...}
Snapshot = dict
Decision = dict
ResolvedOutcome = dict


def book_paths(allocator_id: str = "the-manager") -> dict[str, Path]:
    """Return the canonical filesystem paths for an allocator's book.

    Keys: ``portfolio``, ``snapshots``, ``review_dir``, ``resolved``.
    ``allocator_id`` must be a registered allocator (role='allocator'); the
    channel prefix is read from its config, NOT derived from the id string.
    Default ``allocator_id="the-manager"`` reproduces the legacy paths.
    Not valid for ``baseline-manager`` (a deterministic twin, not an allocator —
    it has no review channel and ``allocator_spec`` would raise).
    """
    cfg = get_config()
    prefix = cfg.allocator_spec(allocator_id).channels_prefix
    review_dir = _order_dir(prefix, "review")
    portfolio_dir = cfg.portfolios_dir / allocator_id
    return {
        "portfolio": portfolio_dir / "portfolio.json",
        "snapshots": portfolio_dir / "snapshots.json",
        "review_dir": review_dir,
        "resolved": review_dir / "resolved.json",
    }


# ---------------------------------------------------------------------------
# Snapshots + NAV maths
# ---------------------------------------------------------------------------


def read_snapshots(path: Path) -> list[Snapshot]:
    """Load a portfolio's ``snapshots.json`` as a date-sorted list.

    Rows missing ``date`` or ``portfolio_value`` are dropped. Returns ``[]`` on a
    missing / malformed / non-list file.
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []

    rows = [
        r for r in raw if isinstance(r, dict) and "date" in r and "portfolio_value" in r
    ]
    rows.sort(key=lambda r: r["date"])
    return rows


def return_pct(snapshots: list[Snapshot]) -> float | None:
    """Percent return from the first to the last snapshot.

    ``None`` if there is nothing to measure (empty) or the base value is zero;
    ``0.0`` for a single point.
    """
    if not snapshots:
        return None
    base = snapshots[0]["portfolio_value"]
    if not base:
        return None
    last = snapshots[-1]["portfolio_value"]
    return (last / base - 1) * 100


def _last_nav(snapshots: list[Snapshot]) -> float | None:
    return snapshots[-1]["portfolio_value"] if snapshots else None


def build_manager_summary(
    manager_snapshots: list[Snapshot],
    baseline_snapshots: list[Snapshot],
    initial: float,
) -> dict:
    """Status-strip figures comparing the Manager to its baseline twin.

    ``has_run`` is False until the Manager has at least one snapshot. ``gap_pct``
    (Manager return minus baseline return — the Gate C signal) is ``None`` unless
    both series have a measurable return.
    """
    manager_return = return_pct(manager_snapshots)
    baseline_return = return_pct(baseline_snapshots)

    gap = (
        manager_return - baseline_return
        if manager_return is not None and baseline_return is not None
        else None
    )

    return {
        "has_run": bool(manager_snapshots),
        "initial": initial,
        "manager_nav": _last_nav(manager_snapshots),
        "manager_return_pct": manager_return,
        "baseline_nav": _last_nav(baseline_snapshots),
        "baseline_return_pct": baseline_return,
        "gap_pct": gap,
    }


# ---------------------------------------------------------------------------
# Portfolio (point-in-time cash + positions)
# ---------------------------------------------------------------------------


def read_portfolio(path: Path) -> dict | None:
    """Load a ``portfolio.json`` ({cash, positions, ...}). ``None`` if absent/bad."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


# ---------------------------------------------------------------------------
# Decisions + resolved outcomes
# ---------------------------------------------------------------------------


def load_decisions(review_dir: Path) -> list[Decision]:
    """Load every ``manager-review/{date}.json`` audit artifact, newest first.

    ``resolved.json`` (the outcomes file, not a per-day decision) is excluded.
    Unreadable files are skipped. Returns ``[]`` if the directory is absent.
    """
    review_dir = Path(review_dir)
    if not review_dir.is_dir():
        return []

    decisions: list[Decision] = []
    for f in review_dir.glob("*.json"):
        if f.name == "resolved.json":
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            decisions.append(payload)

    decisions.sort(key=lambda d: d.get("date", ""), reverse=True)
    return decisions


def load_resolved(resolved_path: Path) -> list[ResolvedOutcome]:
    """Load ``manager-review/resolved.json`` (matured outcomes). ``[]`` if absent/bad."""
    try:
        raw = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return raw if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# Authored-order terminal status
#
# The decision log records what the Manager *authored*; a conditional order's
# outcome (fired / expired / cancelled) or open state (armed) lives in the
# separate broker channels. These helpers join the two so the log can mark an
# armed-but-unfired conditional distinctly from an executed buy — the
# 2026-07-18 confusion where "BUY QQQ3.L €250" (a trigger that expired unfired)
# read identically to a real fill.
# ---------------------------------------------------------------------------


def index_manager_inbox(inbox_dir: Path) -> dict[str, dict]:
    """Map ``order_id -> its terminal inbox record`` across every ``{date}.jsonl``.

    A conditional authored on day D fills (or expires) on a *later* day, so its
    record lands in the inbox file dated by the processing day — not D. The join
    therefore has to scan the whole directory rather than a single date's file.
    Files are read in sorted (chronological) order so a later record wins on a
    duplicated id. Blank/malformed lines and unreadable files are skipped.
    Returns ``{}`` if the directory is absent.
    """
    index: dict[str, dict] = {}
    inbox_dir = Path(inbox_dir)
    if not inbox_dir.is_dir():
        return index
    for f in sorted(inbox_dir.glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            oid = rec.get("order_id") if isinstance(rec, dict) else None
            if oid:
                index[oid] = rec
    return index


def authored_status(
    decision: Decision,
    *,
    inbox_index: dict[str, dict],
    pending_dir: Path,
    outbox_dir: Path,
) -> list[str]:
    """Per-position terminal status label for an authored daily decision.

    Each authored position is matched to the order the broker actually emitted
    by reading that day's outbox (``{outbox_dir}/{date}.jsonl``) and joining on
    ``ticker`` — deliberately NOT on the position's index. The broker emits an
    order only for *tradable* positions: ``manager_decision_to_orders`` skips
    HOLD actions, tickers with no store price, and zero-size positions. So a
    position's index in the persisted decision does not track the order's
    ``seq``, and inferring ``order_id`` from the index would shift every label
    after a skipped position (the exact mislabeling this whole feature fixes).
    Reading the ``order_id`` straight from the outbox is immune to that drift.

    The resolved ``order_id`` is looked up against the inbox (``inbox_index``
    from :func:`index_manager_inbox`) then the pending channel:

      - ``"filled"``            — a market or fired-conditional fill
      - ``"expired MM-DD"``     — a conditional that expired unfired
      - ``"cancelled MM-DD"``   — cancelled by the agent
      - ``"rejected: REASON"``  — any other broker rejection
      - ``"armed"``             — still-open conditional in ``pending_dir``
      - ``""``                  — no emitted order (a position the broker
                                  skipped) or no terminal record yet (in flight)

    Returns one label per position, in order; ``[]`` for a HOLD day. Never
    raises — a missing date/outbox degrades every label to ``""``.
    """
    positions = decision.get("positions", []) or []
    ids_by_ticker = _outbox_order_ids_by_ticker(outbox_dir, decision.get("date", ""))
    pending_dir = Path(pending_dir)

    labels: list[str] = []
    for pos in positions:
        queue = ids_by_ticker.get(pos.get("ticker"))
        if not queue:
            labels.append("")  # broker emitted no order for this position
            continue
        oid = queue.pop(0)  # consume in emit order if a ticker repeats in a day
        rec = inbox_index.get(oid)
        if rec is not None:
            labels.append(_inbox_status_label(rec))
        elif (pending_dir / f"{oid}.json").is_file():
            labels.append("armed")
        else:
            labels.append("")
    return labels


def _outbox_order_ids_by_ticker(
    outbox_dir: Path, date_str: str
) -> dict[str, list[str]]:
    """``ticker -> [order_id, ...]`` from a day's outbox, preserving emit order.

    The outbox (``{outbox_dir}/{date}.jsonl``) is the broker's authoritative
    record of the orders it actually created for that session — one line per
    emitted order, each carrying both ``order_id`` and ``ticker``. Repeated
    tickers keep their emit order so a caller can match repeated authored
    positions positionally within a ticker. A missing/malformed file or line
    contributes nothing; an absent date yields ``{}``.
    """
    out: dict[str, list[str]] = {}
    if not date_str:
        return out
    path = Path(outbox_dir) / f"{date_str}.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        ticker = rec.get("ticker")
        oid = rec.get("order_id")
        if ticker and oid:
            out.setdefault(ticker, []).append(oid)
    return out


def _inbox_status_label(rec: dict) -> str:
    """Terminal label for a single inbox fill record."""
    if rec.get("status") == "filled":
        return "filled"
    reason = rec.get("reason")
    if reason == "TRIGGER_EXPIRED":
        return f"expired {_short_date(rec)}".rstrip()
    if reason == "CANCELLED_BY_AGENT":
        return f"cancelled {_short_date(rec)}".rstrip()
    if reason:
        return f"rejected: {reason}"
    return rec.get("status") or ""


def _short_date(rec: dict) -> str:
    """``MM-DD`` from a record's ``ts_filled`` ISO timestamp, or ``""``."""
    ts = rec.get("ts_filled")
    if isinstance(ts, str) and len(ts) >= 10:
        return ts[5:10]  # "YYYY-MM-DDT..." -> "MM-DD"
    return ""
