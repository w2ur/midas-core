"""Restatement disclosure: a published number does not move undisclosed.

`METHODOLOGY.md` carries a changelog whose stated purpose is that "methodology
changes are logged here rather than silently applied". It has held up well —
except once, and the exception is the whole reason this module exists. On
2026-08-02 a commit rewrote a book's trade ledger and restated 41 of its
snapshots, moving that agent's published return from +3.30% to +0.19%. No
changelog entry was written. The omission was found five days later "by a
cross-check on a different task, not by any process", and the entry it should
have had was left asserting *"The ledger is not being rewritten"* for those
five days.

So: the scripts that move published numbers now refuse to run without naming
the changelog anchor that discloses them, and verify the anchor actually
exists. The disclosure lands in the same commit as the change, or the change
does not happen.

This is deliberately a *pre*condition, not a post-hoc audit. An audit tells
you afterwards that you forgot; a precondition means you cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

from engine.config import get_config

#: `<a id="..."></a>` — the anchor form the changelog already uses, and the
#: one `site/tests/ledger-notes.test.ts` already resolves links against.
_ANCHOR = re.compile(r'<a\s+id="([^"]+)"')


class UndisclosedRestatementError(RuntimeError):
    """Raised when a restatement would run without a changelog anchor."""


def methodology_path() -> Path:
    return get_config().data_dir / "METHODOLOGY.md"


def known_anchors() -> set[str]:
    """Every anchor id defined in METHODOLOGY.md, or empty if it is absent."""
    path = methodology_path()
    if not path.exists():
        return set()
    return set(_ANCHOR.findall(path.read_text(encoding="utf-8")))


def require_changelog_entry(anchor: str | None, *, what: str) -> str:
    """Verify `anchor` names a real METHODOLOGY changelog entry.

    Returns the anchor on success. Raises `UndisclosedRestatementError` when it
    is missing or does not resolve — never merely warns. A warning printed into
    a script's output is read by whoever is already paying attention, which is
    not the failure mode being guarded against.

    A fork with no METHODOLOGY.md is not held to this project's disclosure
    convention: there is nothing to write into, and refusing would make the
    restatement tooling unusable downstream for no honesty gain.
    """
    anchors = known_anchors()
    if not anchors and not methodology_path().exists():
        return anchor or ""

    if not anchor:
        raise UndisclosedRestatementError(
            f"{what} moves published numbers and requires disclosure.\n"
            "Pass --changelog-entry <anchor>, where <anchor> is the id of a "
            'METHODOLOGY.md changelog entry (`<a id="..."></a>`) describing '
            "what moved and why.\n"
            "Write the entry first — the disclosure ships in the same commit "
            "as the change, which is what makes it a record rather than a "
            "reconstruction."
        )

    if anchor not in anchors:
        raise UndisclosedRestatementError(
            f"No METHODOLOGY.md changelog entry with anchor {anchor!r}.\n"
            f"Found {len(anchors)} anchor(s); the most recent are: "
            f"{sorted(anchors)[:5]}\n"
            "An anchor that does not resolve is a dead link in the public "
            "record, which is worse than no link at all."
        )
    return anchor
