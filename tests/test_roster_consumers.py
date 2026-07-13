"""Roster consumers must read the cast from config, not module dicts."""

from datetime import date

import pytest

from engine.config import get_config, reset_config_cache


@pytest.fixture(autouse=True)
def _default_env(monkeypatch):
    monkeypatch.delenv("MIDAS_DATA_DIR", raising=False)
    reset_config_cache()
    yield
    reset_config_cache()


def test_build_post_prompt_uses_config_display_name():
    from engine.posts import build_post_prompt

    cfg = get_config()
    aid = cfg.trading_roster[0]
    prompt = build_post_prompt(aid, {aid: {"commentary": "hi", "trades": []}})
    assert cfg.roster[aid].display_name in prompt


def test_output_bundle_roster_is_trading_roster():
    from engine.output_bundle import ROSTER

    assert ROSTER == get_config().trading_roster


def test_module_dicts_removed():
    import engine.posts as posts
    import engine.baselines as baselines

    assert not hasattr(posts, "AGENT_DISPLAY_NAMES")
    assert not hasattr(baselines, "AGENT_BENCHMARKS")
