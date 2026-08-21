"""The generic session protocol must describe every step core ships.

Why this exists
---------------
`docs/session-protocol.md` is the harness-agnostic contract a fork implements to
run an autonomous session. It is prose, and prose drifts from code silently: a
step added to `scripts/daily_session.py` that the protocol never mentions is a
step no fork will ever call, and the only symptom is that their desk quietly
does less than ours.

Both files ship to midas-core (`scripts/sync_core.CORE_DOCS`), so this runs in
the public framework repo too and is what keeps the published contract honest.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = _ROOT / "docs" / "session-protocol.md"
ORCHESTRATOR = _ROOT / "scripts" / "daily_session.py"

#: The protocol names each step as inline code — `step_fetch_market_data`.
_DOC_STEP = re.compile(r"`(step_[a-z0-9_]+)`")
_DEF_STEP = re.compile(r"^def (step_[a-z0-9_]+)", re.MULTILINE)


def documented_steps(path: Path = PROTOCOL) -> set[str]:
    return set(_DOC_STEP.findall(path.read_text(encoding="utf-8")))


def implemented_steps(path: Path = ORCHESTRATOR) -> set[str]:
    return set(_DEF_STEP.findall(path.read_text(encoding="utf-8")))


def test_the_protocol_documents_every_shipped_step():
    implemented = implemented_steps()
    assert implemented, "no step_* definitions found — the regex or the file moved"
    documented = documented_steps()
    assert documented == implemented, (
        "session protocol and orchestrator disagree.\n"
        f"  in the code, absent from the doc: {sorted(implemented - documented)}\n"
        f"  in the doc, absent from the code: {sorted(documented - implemented)}"
    )


def test_the_check_can_fail(tmp_path):
    """Falsifiable control: a doc that drops one step must be rejected.

    Without this, a regex that silently matched nothing would make the test
    above pass by vacuity in exactly the case it exists to catch.
    """
    victim = sorted(implemented_steps())[0]
    doctored = tmp_path / "session-protocol.md"
    doctored.write_text(
        PROTOCOL.read_text(encoding="utf-8").replace(
            f"`{victim}`", "`step_deleted_by_control`"
        ),
        encoding="utf-8",
    )
    assert documented_steps(doctored) != implemented_steps()
    assert victim not in documented_steps(doctored)
