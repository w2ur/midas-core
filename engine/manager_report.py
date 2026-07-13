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
