"""Guard: the live trading roster is the expected set, and every agent's
universe resolves to a non-empty ticker list via valid registry names.

Deliberately does NOT pin resolved-ticker membership. ``data/universes/*.json``
refreshes on a weekly cron, so a membership fingerprint (count + sha256) would
false-alarm on every legitimate refresh — exactly the recurring-false-alarm
trap to avoid. The stable invariants are: the roster identity, that each live
agent's universe is composed of *valid* registry names (a typo would raise),
and that resolution yields a non-empty set. The resolve_agent_universe
composition logic itself is unit-tested in tests/test_config.py, and the
per-agent universe NAMES are pinned in the roster snapshot test.
"""

from __future__ import annotations

import pytest

from engine.config import get_config, reset_config_cache, resolve_agent_universe
from engine.universes import resolve_universe

pytestmark = pytest.mark.live_cast

EXPECTED_TRADING_ROSTER = frozenset(
    {
        "monsieur-forex",
        "steady-eddie-eur",
        "steady-eddie-usd",
        "sharp-shooter-eur",
        "sharp-shooter-usd",
        "world",
        "goldfinger",
        "yolo-sapiens-eur",
        "yolo-sapiens-usd",
        "satoshi",
    }
)


def test_trading_roster_matches_expected_set() -> None:
    """A new or removed live trading agent must be a deliberate change."""
    reset_config_cache()
    assert set(get_config().trading_roster) == EXPECTED_TRADING_ROSTER


def test_each_agent_universe_names_resolve_nonempty() -> None:
    """Every live agent's universe is valid registry names and resolves non-empty.

    Catches a typo'd universe name (``resolve_universe`` raises ``KeyError``)
    and a totally broken resolution, without coupling to mutable membership.
    """
    reset_config_cache()
    cfg = get_config()
    for agent_id in cfg.trading_roster:
        spec = cfg.roster[agent_id]
        names = (
            [spec.universe] if isinstance(spec.universe, str) else list(spec.universe)
        )
        for name in names:
            resolve_universe(name)  # raises KeyError on an invalid universe name
        assert resolve_agent_universe(spec), (
            f"{agent_id}: resolved to an empty universe"
        )
