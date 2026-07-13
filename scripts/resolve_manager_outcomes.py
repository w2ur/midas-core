"""Resolve matured Manager decisions into numeric outcome entries.

For each non-HOLD position in ``data/orders/manager-review/{date}.json``, this
module computes the forward return over a ``horizon_trading_days`` window and
stores the result in ``data/orders/manager-review/resolved.json``.

Maturation rule
---------------
Given a decision on ``decision_date`` for ticker T with ``horizon_trading_days=N``:
  1. Collect all OHLCV rows for T dated AFTER ``decision_date`` in chronological
     order.  These are the "forward rows" (ticker's own trading calendar).
  2. If fewer than N forward rows are available in the store, the decision is NOT
     yet mature — skip it (leave pending).
  3. ``entry_price``  = latest close on or before ``decision_date`` (same function
     used everywhere in the codebase).
  4. ``exit_price``   = close of the N-th forward row.
  5. ``fwd_return``   = (exit_price / entry_price - 1) * 100   [%]
  6. ``realized_return_pct`` = ``fwd_return``  for BUY
                              = ``-fwd_return`` for SELL
     Positive always means "the directional call was correct".
  7. MSCI return over the same window = (msci_value_at_exit / msci_value_at_entry
     - 1) * 100, where msci_value_at_{entry,exit} is the closest-on-or-before
     portfolio_value in the msci_series list.
  8. ``alpha_vs_msci_pct`` = ``realized_return_pct`` - ``msci_return``.

Whitelist contract (Oracle-Fallacy guard)
-----------------------------------------
Output entries contain EXACTLY five fields:
    date, ticker, action, realized_return_pct, alpha_vs_msci_pct
No reasoning, render, size_eur, entry_guidance, stop_loss, or conviction fields
are ever written.  This is defence-in-depth alongside the C3 sanitise guard.

Idempotency
-----------
Decisions are keyed by ``(date, ticker, action)``.  A key already present in
``existing_resolved`` is never re-resolved.

Ordering and cap
----------------
The resolved list is sorted by (date, ticker, action) ascending and capped at
90 entries (oldest dropped when over the limit).

Usage (CLI)
-----------
    python scripts/resolve_manager_outcomes.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.ohlcv_store import latest_close_on_or_before

# Whitelist of fields allowed in the resolved output (Oracle-Fallacy guard).
_ALLOWED_FIELDS = frozenset(
    {"date", "ticker", "action", "realized_return_pct", "alpha_vs_msci_pct"}
)

_CAP = 90


# ---------------------------------------------------------------------------
# MSCI value lookup
# ---------------------------------------------------------------------------


def _msci_value_on_or_before(target_date: str, msci_series: list[dict]) -> float | None:
    """Return the MSCI World portfolio_value on or just before target_date.

    msci_series is a list of {date, portfolio_value, ...} dicts sorted in any
    order.  Uses the same "latest on-or-before" logic as the OHLCV store.
    """
    best_date: str | None = None
    best_value: float | None = None
    for entry in msci_series:
        entry_date = entry.get("date")
        if entry_date is None or entry_date > target_date:
            continue
        if best_date is None or entry_date > best_date:
            best_date = entry_date
            val = entry.get("portfolio_value")
            best_value = float(val) if val is not None else None
    return best_value


# ---------------------------------------------------------------------------
# Core pure function
# ---------------------------------------------------------------------------


def resolve_outcomes(
    review_dir: Path,
    store: Path,
    msci_series: list[dict],
    today: date,
    horizon_trading_days: int = 10,
    existing_resolved: list[dict] | None = None,
) -> list[dict]:
    """Compute newly-matured Manager decision outcomes and merge with existing.

    Parameters
    ----------
    review_dir:
        Directory containing per-day ``{YYYY-MM-DD}.json`` audit artifacts.
        ``resolved.json`` in this directory is ignored (not a decision file).
    store:
        Path to the OHLCV store (directory of ``{TICKER}.jsonl`` files).
        Passed through to ``latest_close_on_or_before`` for entry prices.
    msci_series:
        List of MSCI World snapshot dicts, each with keys ``date`` (ISO str)
        and ``portfolio_value`` (float).  Used to compute the benchmark return
        over each decision window.
    today:
        Reference date.  Decisions whose N-th forward row is still in the
        future (i.e. the store has fewer than N rows after the decision date)
        remain pending regardless of ``today``.
    horizon_trading_days:
        Number of trading days (per the ticker's own store) to advance from
        the decision date to the resolution date.  Default 10.
    existing_resolved:
        Previously-resolved entries to merge with (passed explicitly for
        idempotency).  Defaults to ``[]`` when None.

    Returns
    -------
    list[dict]
        Merged, sorted, capped resolved list (at most ``_CAP`` entries).
        Each entry contains exactly the five whitelisted fields.
    """
    if existing_resolved is None:
        existing_resolved = []

    # Build the set of already-resolved keys to avoid re-resolution.
    resolved_keys: set[tuple[str, str, str]] = {
        (e["date"], e["ticker"], e["action"]) for e in existing_resolved
    }

    new_entries: list[dict] = []

    # Iterate every review file (skip resolved.json itself).
    if not review_dir.exists():
        combined = list(existing_resolved)
        return _sort_and_cap(combined)

    for path in sorted(review_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "resolved.json":
            continue
        if not path.name.endswith(".json"):
            continue

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        decision_date: str = payload.get("date", path.stem)
        positions: list[dict] = payload.get("positions", [])

        for pos in positions:
            action = pos.get("action", "")
            if action == "HOLD":
                continue

            ticker = pos.get("ticker", "")
            if not ticker:
                continue

            key = (decision_date, ticker, action)
            if key in resolved_keys:
                continue

            entry = _resolve_position(
                decision_date=decision_date,
                ticker=ticker,
                action=action,
                store=store,
                msci_series=msci_series,
                horizon_trading_days=horizon_trading_days,
            )
            if entry is not None:
                new_entries.append(entry)
                resolved_keys.add(key)

    combined = list(existing_resolved) + new_entries
    return _sort_and_cap(combined)


def _resolve_position(
    decision_date: str,
    ticker: str,
    action: str,
    store: Path,
    msci_series: list[dict],
    horizon_trading_days: int,
) -> dict | None:
    """Resolve a single non-HOLD position, or return None if not yet mature.

    Maturation: advance ``horizon_trading_days`` trading days from
    ``decision_date`` using the ticker's own available dated rows AFTER the
    decision date.  If the store has fewer than ``horizon_trading_days`` rows
    with date > ``decision_date``, the decision is not yet mature.
    """
    # Collect all OHLCV rows for this ticker with date > decision_date,
    # sorted chronologically.
    forward_rows = _forward_rows(ticker, decision_date, store)
    if len(forward_rows) < horizon_trading_days:
        return None

    exit_row = forward_rows[horizon_trading_days - 1]
    exit_date = exit_row[0]
    exit_price = exit_row[1]

    # Entry price: latest close on or before the decision date.
    entry_price = latest_close_on_or_before(
        ticker, date.fromisoformat(decision_date), store=store
    )
    if entry_price is None or entry_price == 0.0:
        return None

    fwd_return = (exit_price / entry_price - 1.0) * 100.0

    if action == "BUY":
        realized_return_pct = fwd_return
    else:
        # SELL: positive when the ticker fell (correct directional call).
        realized_return_pct = -fwd_return

    # MSCI benchmark return over the same window.
    msci_entry = _msci_value_on_or_before(decision_date, msci_series)
    msci_exit = _msci_value_on_or_before(exit_date, msci_series)
    if msci_entry is not None and msci_exit is not None and msci_entry != 0.0:
        msci_return = (msci_exit / msci_entry - 1.0) * 100.0
    else:
        # Deliberate: if MSCI reference data is missing for the window, alpha
        # degrades to == realized_return rather than dropping the resolved
        # outcome.  A resolved realized return with degraded alpha is more
        # useful than no outcome at all.  Ticker price gaps DO skip (return
        # None / leave pending above); only the alpha reference degrades here.
        msci_return = 0.0

    alpha_vs_msci_pct = realized_return_pct - msci_return

    return {
        "date": decision_date,
        "ticker": ticker,
        "action": action,
        "realized_return_pct": round(realized_return_pct, 2),
        "alpha_vs_msci_pct": round(alpha_vs_msci_pct, 2),
    }


def _forward_rows(
    ticker: str, decision_date: str, store: Path
) -> list[tuple[str, float]]:
    """Return OHLCV rows for ``ticker`` with date strictly after ``decision_date``.

    Returns a list of (date_str, price) tuples sorted chronologically.
    Uses adj_close when available, falls back to close.
    """
    path = store / f"{ticker}.jsonl"
    if not path.exists():
        return []

    rows: list[tuple[str, float]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_date = row.get("date")
            if row_date is None or row_date <= decision_date:
                continue
            val = (
                row.get("adj_close")
                if row.get("adj_close") is not None
                else row.get("close")
            )
            if val is None:
                continue
            rows.append((row_date, float(val)))

    rows.sort(key=lambda t: t[0])
    return rows


def _sort_and_cap(entries: list[dict]) -> list[dict]:
    """Sort by (date, ticker, action) ascending and cap at _CAP entries (newest kept)."""
    sorted_entries = sorted(
        entries,
        key=lambda e: (e.get("date", ""), e.get("ticker", ""), e.get("action", "")),
    )
    if len(sorted_entries) > _CAP:
        # Drop the oldest (lowest date) entries.
        sorted_entries = sorted_entries[len(sorted_entries) - _CAP :]
    return sorted_entries


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_existing_resolved(resolved_path: Path) -> list[dict]:
    """Read the existing resolved.json, returning [] on missing or malformed file."""
    if not resolved_path.exists():
        return []
    try:
        data = json.loads(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []
        return data
    except (json.JSONDecodeError, OSError):
        return []


def write_resolved(entries: list[dict], resolved_path: Path) -> None:
    """Atomically write the resolved list to ``resolved_path``.

    Uses tmp + os.replace (same pattern as session_state.py and agent_memory.py)
    to guarantee a crash mid-write never leaves a corrupt file.
    """
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=resolved_path.parent, prefix=".resolved_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, resolved_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Load, resolve, and write manager outcome memory.

    Reads from paths derived from engine.config.get_config() and the sole
    allocator's AllocatorSpec (channels_prefix, outcome_resolution_days).
    Intended for manual CLI runs.  The production path is
    scripts.daily_session.step_resolve_manager_outcomes.

    When the deployment has no allocator (empty ``cfg.allocators``), the
    function exits cleanly with a status message and writes nothing.
    """
    cfg = get_config()
    allocs = cfg.allocators
    if not allocs:
        print("  No allocator configured — nothing to resolve.")
        return

    from engine.orders import allocator_channel_dir

    alloc = cfg.allocator_spec(allocs[0])
    review_dir = allocator_channel_dir(alloc.channels_prefix, "review")
    resolved_path = review_dir / "resolved.json"
    msci_path = cfg.baselines_dir / "global" / "msci_world.json"
    try:
        msci_series = json.loads(msci_path.read_text(encoding="utf-8"))
        if not isinstance(msci_series, list):
            msci_series = []
    except (json.JSONDecodeError, OSError):
        msci_series = []

    existing = load_existing_resolved(resolved_path)
    from engine.ohlcv_store import OHLCV_STORE

    updated = resolve_outcomes(
        review_dir=review_dir,
        store=OHLCV_STORE,
        msci_series=msci_series,
        today=date.today(),
        horizon_trading_days=alloc.outcome_resolution_days,
        existing_resolved=existing,
    )
    write_resolved(updated, resolved_path)
    new_count = len(updated) - len(existing)
    print(f"  Resolved manager outcomes: {new_count} new, {len(updated)} total.")


if __name__ == "__main__":
    main()
