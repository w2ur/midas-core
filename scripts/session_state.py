"""Session step-state for resumable daily sessions.

Tracks which ``step_*`` functions have completed for a given day.  State is
persisted to ``data/session_state/YYYY-MM-DD.json`` as a JSON object mapping
step name → ISO-8601 completion timestamp.

All writes are atomic (write to a tmp file, then ``os.replace``), so a crash
mid-write never leaves a corrupt state file.

Markers are scoped to a session, not to a day (2026-08-07 review, W3.3)
---------------------------------------------------------------------
A marker used to be keyed on nothing but ``(function name, UTC date)``. Under
that key an ad-hoc invocation — an operator calling ``step_build_baselines()``
by hand at noon to check something — writes a marker that the evening's real
session reads as "already done", and that step silently does not run. The
guard against a skipped step is a `[SKIP]` line on stdout in a cloud sandbox,
which is the same place the 2026-07-31 stall went unnoticed.

Every marker now also records the ledger base it was written against: the
``base_sha`` of the session anchor (``scripts.session_guard``), or ``""`` when
there is no anchor at all — which is exactly what an ad-hoc invocation looks
like. ``is_done`` answers True only when the stored base matches the caller's
current one, so:

- a hand-run step (no anchor) cannot satisfy an anchored session, and
- an anchored session's markers cannot satisfy a later hand-run.

A file written before this change carries no base and is treated as ``""``
— fail-closed, the same convention as a legacy snapshot row without a
``session_date``.

Re-anchoring onto a moved base *does* discard the day's markers, and that is
deliberate: the steps already run were computed against a ledger that no
longer exists. ``assert_session_fresh`` aborts outright when the movement
touched the ledger; when it did not, re-running is the conservative answer.

Timezone note: State files are keyed on UTC dates (sessions fire 20:00 UTC);
``engine.output_bundle.get_day_number`` and commit messages use local dates —
safe in UTC-pinned CI/sandbox runners, would drift for a local operator near
midnight.

Typical usage
-------------
::

    from scripts.session_state import mark_done, is_done, clear

    if is_done("step_fill_orders"):
        print("already filled, skipping")
    else:
        step_fill_orders(...)
        mark_done("step_fill_orders")

    # At the very end of a fully-completed session:
    clear()
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from engine.config import get_config


def _state_dir() -> Path:
    """Resolve the session-state dir.

    Override-aware: the test suite's autouse fixture monkeypatches ``_STATE_DIR``
    to isolate per-test markers; when no override is set, resolve from config so a
    forker's ``MIDAS_DATA_DIR`` redirection reaches session markers too (never
    frozen at import).
    """
    override = globals().get("_STATE_DIR")
    return override if override is not None else get_config().session_state_dir


def __getattr__(name: str) -> object:
    """Expose ``_STATE_DIR`` as a lazy config-backed path (PEP 562).

    Lets ``monkeypatch.setattr(session_state, "_STATE_DIR", tmp)`` keep working
    (the existence check resolves here) while production stays config-driven.
    """
    if name == "_STATE_DIR":
        return get_config().session_state_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _state_path(day: date | None = None) -> Path:
    d = day if day is not None else datetime.now(timezone.utc).date()
    return _state_dir() / f"{d.isoformat()}.json"


# Reserved key holding the ledger base the markers in this file were written
# against. Underscore-free names are step names; this one cannot collide with
# a ``step_*`` function.
_ANCHOR_KEY = "__base_sha__"


def current_base_sha() -> str:
    """The base SHA of the live session anchor, or ``""`` when unanchored.

    Imported lazily: ``session_guard`` shells out to git on import-adjacent
    paths and is irrelevant to a caller that only wants to read markers.
    Any failure to resolve an anchor is reported as "no anchor" rather than
    raising — an unreadable anchor must not take the session down here; the
    session's own ``assert_session_fresh`` is where a missing anchor is fatal.
    """
    try:
        from scripts.session_guard import load_anchor

        anchor = load_anchor()
    except Exception:  # noqa: BLE001 — unreadable anchor == no anchor
        return ""
    return anchor.base_sha if anchor is not None else ""


def _load_state(day: date | None = None) -> dict[str, str]:
    """Read the state file for *day*. Returns {} on missing or malformed file."""
    path = _state_path(day)
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        return data
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(state: dict[str, str], day: date | None = None) -> None:
    """Atomically persist *state* for *day* using tmp + os.replace."""
    path = _state_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a sibling tmp file in the same directory so os.replace is
    # guaranteed to be on the same filesystem (required for atomicity).
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".state_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        # Clean up the orphaned tmp on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def mark_done(step: str, day: date | None = None) -> None:
    """Record *step* as completed for *day* (default: today UTC).

    Idempotent: calling twice just updates the timestamp.

    Stamps the file with the current session anchor's ``base_sha``. If the
    file was written under a *different* base, its markers belong to another
    run and are dropped rather than merged — keeping a mixed-provenance file
    would let one run's marker satisfy another's ``is_done``, which is the
    bug this scoping exists to close.
    """
    base = current_base_sha()
    state = _load_state(day)
    if state.get(_ANCHOR_KEY, "") != base:
        state = {}
    state[_ANCHOR_KEY] = base
    state[step] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_state(state, day)


def is_done(step: str, day: date | None = None) -> bool:
    """Return True if *step* was completed for *day* by *this* session.

    "This session" means: the marker was written against the same ledger base
    the caller is anchored to. A marker from a hand-run step (no anchor) or
    from a run anchored elsewhere answers False — see the module docstring.
    """
    state = _load_state(day)
    if state.get(_ANCHOR_KEY, "") != current_base_sha():
        return False
    return step in state


def clear(day: date | None = None) -> None:
    """Delete the state file for *day* (default: today UTC).

    Called at the end of a fully-completed session so the state directory
    stays clean and the next session starts fresh.
    """
    path = _state_path(day)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
