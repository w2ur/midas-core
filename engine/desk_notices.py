"""Desk notices — time-boxed operational facts injected into persona prompts.

An agent only knows what its prompt and its journal tell it. When the desk's
*infrastructure* misbehaves — a watcher that could not publish its fills, a
collector that stopped advancing — the agents infer a cause from the evidence
they can see, write that inference into their journals, and then reason from it
for as long as the journal survives. That is what happened over
2026-08-24..09-04: a branch-protection rule rejected every push the conditional-
order watcher made, so ten hit triggers never reached the ledger, and nine of
ten trader journals plus the Oracle's now say conditional orders are "advisory".

There is no other channel to correct that. The live weekday prompt lives in a
RemoteTrigger config on claude.ai; its source (`docs/triggers/weekday-session.md`)
is hashed by `scripts/prompt_hash.py`, so editing it obliges the owner to
re-paste the live prompt by hand — the wrong place for a fact that expires in
two weeks. Every persona-authored output, by contrast, goes through
`engine.persona_dispatch.wrap_persona_prompt`, which runs inside the session
from the checkout. A committed, dated notice file read at that point needs no
re-paste and expires on its own.

Design constraints, in order of priority:

1. **Never fatal.** A missing, unreadable or malformed notice file yields no
   block and a logged warning. Losing a session over a piece of desk prose
   would be far worse than the agents missing the notice.
2. **Dated, not permanent.** Each notice carries an inclusive ``[from, until]``
   window. A stale fact left in an agent's prompt forever is the failure this
   file would otherwise become; the window is what makes it self-retiring.
3. **Audience-scoped.** ``traders`` reaches the ``role: trader`` books only —
   the Oracle narrates and holds no book, and the allocator's channel is
   isolated by design. ``all`` reaches every dispatched persona.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

from engine.config import get_config

logger = logging.getLogger(__name__)

#: Audience values a notice may declare. Anything else is malformed.
AUDIENCES = ("traders", "all")

_HEADER = "--- DESK NOTICE ---"
_FOOTER = "--- END DESK NOTICE ---"
_PREAMBLE = (
    "Operational facts about the desk itself, current as of {today}. They are "
    "not trading advice and not a view on any market; they describe how the "
    "machinery around you is behaving. Where a notice contradicts something "
    "your own journal says about the desk, the notice is the correct account."
)


@dataclass(frozen=True)
class DeskNotice:
    """One dated notice. ``start``/``end`` are the JSON ``from``/``until``."""

    id: str
    start: date
    end: date
    audience: str
    text: str

    def covers(self, today: date) -> bool:
        """Inclusive on both ends — a one-day notice sets from == until."""
        return self.start <= today <= self.end


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _parse_notice(raw: object, index: int) -> DeskNotice | None:
    """Validate one entry. Returns None (and warns) on anything malformed.

    Per-entry rather than whole-file rejection: one bad row should not silence
    the others, and silence is precisely the failure mode this module exists to
    end.
    """
    if not isinstance(raw, dict):
        logger.warning("desk notice #%d is not an object; skipped", index)
        return None
    notice_id = raw.get("id")
    text = raw.get("text")
    audience = raw.get("audience")
    if not isinstance(notice_id, str) or not notice_id:
        logger.warning("desk notice #%d has no usable id; skipped", index)
        return None
    if not isinstance(text, str) or not text.strip():
        logger.warning("desk notice %r has no text; skipped", notice_id)
        return None
    if audience not in AUDIENCES:
        logger.warning(
            "desk notice %r declares audience %r, not one of %s; skipped",
            notice_id,
            audience,
            ", ".join(AUDIENCES),
        )
        return None
    try:
        start = date.fromisoformat(str(raw["from"]))
        end = date.fromisoformat(str(raw["until"]))
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "desk notice %r has a missing or unparseable from/until window; skipped",
            notice_id,
        )
        return None
    if end < start:
        logger.warning(
            "desk notice %r ends (%s) before it starts (%s); skipped",
            notice_id,
            end,
            start,
        )
        return None
    return DeskNotice(
        id=notice_id, start=start, end=end, audience=audience, text=text.strip()
    )


def load_notices() -> list[DeskNotice]:
    """Read every well-formed notice from ``data/desk_notices.json``.

    An absent file is the normal state of a desk with nothing to announce, and
    returns ``[]`` without a warning. A file that exists but cannot be read or
    parsed as a list returns ``[]`` *with* one — that is a mistake someone made,
    not a default.
    """
    path = get_config().desk_notices_path
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        logger.warning("desk notices unreadable at %s (%s); no notice block", path, exc)
        return []
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "desk notices at %s are not valid JSON (%s); no notice block", path, exc
        )
        return []
    if not isinstance(payload, list):
        logger.warning(
            "desk notices at %s are a %s, expected a list; no notice block",
            path,
            type(payload).__name__,
        )
        return []
    parsed = [_parse_notice(entry, i) for i, entry in enumerate(payload)]
    return [n for n in parsed if n is not None]


def _audience_matches(notice: DeskNotice, agent_id: str) -> bool:
    if notice.audience == "all":
        return True
    # "traders": role-scoped, so the narrator and the allocator are excluded.
    # An agent absent from the roster is not a trader — fail closed rather than
    # widening the audience on an unknown id.
    try:
        return agent_id in get_config().trading_roster
    except Exception:  # pragma: no cover - config failure is handled upstream
        return False


def active_notices(agent_id: str, today: date | None = None) -> list[DeskNotice]:
    """Notices in window for ``today`` and addressed to ``agent_id``."""
    when = today or _today()
    return [
        n for n in load_notices() if n.covers(when) and _audience_matches(n, agent_id)
    ]


def render_notice_block(agent_id: str, today: date | None = None) -> str:
    """The DESK NOTICE block for this persona, or "" when there is nothing.

    Returns a block that already carries its own leading and trailing newline,
    so the caller can drop it into a template slot and get byte-identical output
    to the no-notice case when it is empty.
    """
    when = today or _today()
    try:
        notices = active_notices(agent_id, when)
    except Exception as exc:  # never fatal — a lost session is worse
        logger.warning("desk notices could not be resolved (%s); no notice block", exc)
        return ""
    if not notices:
        return ""
    lines = [_HEADER, _PREAMBLE.format(today=when.isoformat()), ""]
    for notice in notices:
        lines.append(f"[{notice.id}] {notice.text}")
    lines.append(_FOOTER)
    return "\n" + "\n".join(lines) + "\n"
