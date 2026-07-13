"""Session step-state for resumable daily sessions.

Tracks which ``step_*`` functions have completed for a given day.  State is
persisted to ``data/session_state/YYYY-MM-DD.json`` as a JSON object mapping
step name → ISO-8601 completion timestamp.

All writes are atomic (write to a tmp file, then ``os.replace``), so a crash
mid-write never leaves a corrupt state file.

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
    """
    state = _load_state(day)
    state[step] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _write_state(state, day)


def is_done(step: str, day: date | None = None) -> bool:
    """Return True if *step* was previously marked done for *day*."""
    return step in _load_state(day)


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
