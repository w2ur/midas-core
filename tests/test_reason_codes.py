"""The reason-code count is documented in five places. This binds it to the code.

Origin: on 2026-07-24 every document said "14 distinct rejection/cancel reason
codes" while the broker emitted 15 and the watcher added a 16th. The module
docstring was right and everything downstream was stale.
"""

import re
from pathlib import Path

import pytest

from engine.paper_broker import REJECTION_REASON_CODES

REPO_ROOT = Path(__file__).resolve().parents[1]
BROKER_SRC = REPO_ROOT / "engine" / "paper_broker.py"
WATCHER_SRC = REPO_ROOT / "scripts" / "check_triggers.py"
SITE_RAILS = REPO_ROOT / "site" / "src" / "lib" / "rails.ts"

WATCHER_CODES = {"TRIGGER_EXPIRED"}


def test_declared_set_has_nineteen_broker_codes():
    """15 → 19 on 2026-08-07: CURRENCY_UNRESOLVED, PRICE_IMPLAUSIBLE,
    TRIGGER_LEVEL_IMPLAUSIBLE and VALUATION_UNAVAILABLE (reliability review
    W1). The literal count is the point — it is what forces every mirror of
    this list to be looked at."""
    assert len(REJECTION_REASON_CODES) == 19


def test_every_code_emitted_in_the_broker_is_declared():
    """Quoted SCREAMING_CASE literals in the broker are exactly the declared codes."""
    source = BROKER_SRC.read_text(encoding="utf-8")
    emitted = set(re.findall(r'"([A-Z][A-Z_]{5,})"', source))
    assert emitted == set(REJECTION_REASON_CODES)


def test_module_docstring_enumerates_the_same_codes():
    from engine import paper_broker

    documented = set(re.findall(r"^- ([A-Z_]+):", paper_broker.__doc__ or "", re.M))
    assert documented == set(REJECTION_REASON_CODES)


def test_watcher_owns_the_sixteenth_code():
    """TRIGGER_EXPIRED lives in the watcher, not the broker — keep them distinct."""
    source = WATCHER_SRC.read_text(encoding="utf-8")
    assert 'reason="TRIGGER_EXPIRED"' in source
    assert not WATCHER_CODES & set(REJECTION_REASON_CODES)


@pytest.mark.skipif(
    not SITE_RAILS.exists(), reason="site/ is live-only, absent in midas-core"
)
def test_site_rail_registry_matches_the_engine():
    source = SITE_RAILS.read_text(encoding="utf-8")
    listed = set(re.findall(r'code:\s*"([A-Z_]+)"', source))
    assert listed == set(REJECTION_REASON_CODES) | WATCHER_CODES
