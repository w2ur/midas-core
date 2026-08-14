"""Freshness guard for long-running sessions.

Origin — the 2026-07-31 weekday session
---------------------------------------
The RemoteTrigger sandbox fired on time at 20:00 UTC on Friday 2026-07-31 and
stalled roughly five minutes in, while rebuilding the Python 3.12 venv. It did
not resume until Sunday 2026-08-02 at ~12:00 UTC — about 63 hours later.

Every guard in the pipeline passed, because every guard was correct *at the
moment it ran*: Step 0 realigned to origin/main when main really was at
02949e3, and the session date really was 2026-07-31. Nothing between Step 0 and
the final push ever re-asked whether the world had moved on.

It had. By the time the sandbox woke up, main carried the 2026-07-31 OHLCV
commit, five 2026-08-01 trigger fires, the 2026-08-01 weekend refresh and a
2026-08-02 ledger reconciliation. The session authored a complete set of
2026-07-31 artifacts from the pre-stall snapshot, committed them, and could not
fast-forward. That failure was the good outcome: merging would have rewritten
portfolios, snapshots and baselines from a base that predates the 08-01 fills,
silently reverting them — the 2026-05-05 corruption class.

What saved main was luck, not design: the merge happened to conflict. A stale
session touching a narrower set of files would have merged cleanly.

This module is the missing check. It pins the session's clock and its ledger
base at Step 0, then re-validates both before anything irreversible.

Why "ledger moved", not "main moved"
------------------------------------
main advances during a healthy session all the time — the sentiment-digest
commit lands at ~20:26, 26 minutes into a 20:00 session. Aborting on any
movement would fail good sessions.

What actually invalidates a session is movement in the *ledger*: portfolios,
orders, baselines, the leaderboard. Those are the files a session recomputes
from the base snapshot it read at Step 0, so a concurrent mutation there means
our recomputation is built on state that no longer exists. Data-only commits
(prices, sentiment, site copy) are harmless and only warn.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Typical weekday session runs 30-45 minutes. Three hours is generous headroom
# for a slow agent round while still catching a suspend/resume (the 2026-07-31
# stall was 63 hours).
MAX_SESSION_WALL_CLOCK = timedelta(hours=3)

# Paths whose movement on main invalidates an in-flight session, because the
# session recomputes them from the snapshot it read at Step 0.
LEDGER_PATHS = (
    "data/portfolios/",
    "data/orders/",
    "data/baselines/",
    "data/leaderboard/",
    "data/agent_memory/",
    "data/output/",
)

# Another session or refresh landing on main means ours is superseded outright.
_SESSION_COMMIT_RE = re.compile(
    r"^chore: (?:(?:weekday|weekend|daily)(?: crypto)? session|weekend refresh) "
)


class StaleSessionError(RuntimeError):
    """Raised when the session's view of the world is no longer current.

    Always fatal: the correct response is to abandon the run and let the next
    scheduled session start clean. A stale session must never be reconciled by
    hand — its artifacts were derived from a base snapshot that no longer
    describes the ledger.
    """


@dataclass(frozen=True)
class SessionAnchor:
    session_date: date
    base_sha: str
    started_at: datetime

    def to_dict(self) -> dict:
        return {
            "session_date": self.session_date.isoformat(),
            "base_sha": self.base_sha,
            "started_at": self.started_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionAnchor":
        return cls(
            session_date=date.fromisoformat(d["session_date"]),
            base_sha=d["base_sha"],
            started_at=datetime.fromisoformat(d["started_at"]),
        )


def _anchor_path() -> Path:
    # Resolved through session_state rather than get_config() directly so the
    # anchor and the step markers it now scopes (W3.3) can never end up in two
    # different directories — including under the test suite's single
    # session-state override.
    from scripts.session_state import _state_dir

    return Path(_state_dir()) / "anchor.json"


def _git(*args: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def anchor_session(session_date: date) -> SessionAnchor:
    """Step 0c — pin the session's clock and ledger base.

    Call once, immediately after the Step 0 realignment to origin/main and
    before any agent is dispatched. Overwrites any previous anchor, so a
    re-run from the top re-anchors cleanly.
    """
    a = SessionAnchor(
        session_date=session_date,
        base_sha=_git("rev-parse", "origin/main"),
        started_at=datetime.now(timezone.utc),
    )
    path = _anchor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(a.to_dict(), indent=1), encoding="utf-8")
    print(
        f"  [anchor] session={a.session_date} base={a.base_sha[:8]} "
        f"started={a.started_at.isoformat()}"
    )
    return a


def load_anchor() -> SessionAnchor | None:
    path = _anchor_path()
    if not path.exists():
        return None
    return SessionAnchor.from_dict(json.loads(path.read_text(encoding="utf-8")))


def clear_anchor() -> None:
    _anchor_path().unlink(missing_ok=True)


def assert_session_fresh(
    stage: str,
    *,
    max_wall_clock: timedelta = MAX_SESSION_WALL_CLOCK,
    fetch: bool = True,
) -> None:
    """Abort the session if its view of the world has gone stale.

    Three independent checks, any of which is fatal:

    1. **Wall clock** — more than ``max_wall_clock`` since the anchor. This is
       the direct suspend/resume detector; the 2026-07-31 stall was 63 hours.
    2. **Calendar** — today is no longer the session date. A session must never
       publish artifacts dated for a day that has already passed.
    3. **Ledger** — main has gained a session/refresh commit, or any commit
       touching ``LEDGER_PATHS``, since the anchor.

    No anchor on disk is itself an error: it means Step 0c was skipped, and an
    unguarded session is what produced this module.
    """
    a = load_anchor()
    if a is None:
        raise StaleSessionError(
            f"[{stage}] no session anchor on disk — anchor_session() was never "
            "called. Run the Step 0 realignment and anchor before authoring."
        )

    now = datetime.now(timezone.utc)
    elapsed = now - a.started_at
    if elapsed > max_wall_clock:
        raise StaleSessionError(
            f"[{stage}] session has been running {elapsed} (limit "
            f"{max_wall_clock}). The sandbox almost certainly stalled and "
            f"resumed; its view of the repo is {elapsed} out of date. "
            "Abandon this run — do not commit. The next scheduled session "
            "will start clean."
        )

    if now.date() != a.session_date:
        raise StaleSessionError(
            f"[{stage}] session is dated {a.session_date} but today is "
            f"{now.date()}. The clock moved under the run; its artifacts would "
            "be published for a day that has already closed. Abandon this run."
        )

    if fetch:
        subprocess.run(
            ["git", "fetch", "origin", "main"],
            cwd=Path(__file__).resolve().parent.parent,
            check=True,
            capture_output=True,
        )

    head_main = _git("rev-parse", "origin/main")
    if head_main == a.base_sha:
        return

    subjects = _git("log", "--format=%s", f"{a.base_sha}..origin/main").splitlines()
    superseding = [s for s in subjects if _SESSION_COMMIT_RE.match(s)]
    if superseding:
        raise StaleSessionError(
            f"[{stage}] another session landed on main while this one ran: "
            f"{superseding!r}. This session is superseded. Abandon this run."
        )

    changed = _git("diff", "--name-only", f"{a.base_sha}..origin/main").splitlines()
    touched_ledger = sorted({p for p in changed if p.startswith(LEDGER_PATHS)})
    if touched_ledger:
        raise StaleSessionError(
            f"[{stage}] the ledger moved on main while this session ran "
            f"({len(touched_ledger)} file(s), e.g. {touched_ledger[:3]}). "
            "This session recomputed portfolios from a snapshot that is no "
            "longer current; committing would revert those changes. "
            "Abandon this run."
        )

    print(
        f"  [fresh] main advanced {len(subjects)} commit(s) since anchor, "
        "none touching the ledger — continuing."
    )
