"""Tests for engine.persona_dispatch.

These tests use the real .claude/agents/*.md files committed to the repo —
the helper exists precisely to read them, so substituting fixtures would
hide the failure mode this module was created to fix.
"""

from __future__ import annotations

import pytest

from engine.persona_dispatch import load_persona, wrap_persona_prompt

pytestmark = pytest.mark.live_cast

ROSTER = [
    "steady-eddie-eur",
    "steady-eddie-usd",
    "sharp-shooter-eur",
    "sharp-shooter-usd",
    "yolo-sapiens-eur",
    "yolo-sapiens-usd",
    "satoshi",
    "monsieur-forex",
    "goldfinger",
    "world",
    "the-oracle",
]


def test_load_persona_satoshi_strips_frontmatter():
    body, model = load_persona("satoshi")
    assert model == "opus"
    assert not body.startswith("---")
    assert "name: satoshi" not in body
    assert body.startswith("You are **Satoshi**")


def test_load_persona_the_oracle_present_and_has_model():
    body, model = load_persona("the-oracle")
    # Oracle is deliberately downgraded to sonnet: Opus first-token latency on
    # the narrative+10-agent-context prompt routinely exceeded the cloud
    # streaming idle timeout (~60s). Sonnet starts streaming in 2-10s.
    assert model == "sonnet"
    assert "Oracle" in body


def test_load_persona_traders_use_opus():
    # Trade-round dispatches don't hit the streaming timeout — keep Opus.
    for agent_id in [
        "steady-eddie-eur",
        "sharp-shooter-eur",
        "yolo-sapiens-eur",
        "satoshi",
        "monsieur-forex",
        "goldfinger",
        "world",
    ]:
        _, model = load_persona(agent_id)
        assert model == "opus", f"{agent_id}: expected opus, got {model}"


@pytest.mark.parametrize("agent_id", ROSTER)
def test_load_persona_roster_all_have_non_empty_body(agent_id):
    body, model = load_persona(agent_id)
    assert body, f"{agent_id}: empty body"
    assert model, f"{agent_id}: no model frontmatter"


def test_load_persona_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_persona("does-not-exist")


def test_wrap_persona_prompt_includes_body_and_task():
    task = "Output the literal string OK and nothing else."
    wrapped, model = wrap_persona_prompt("satoshi", task)
    assert model == "opus"
    assert "Satoshi" in wrapped
    assert task in wrapped
    assert "--- PERSONA (satoshi) ---" in wrapped
    assert "--- END PERSONA ---" in wrapped
    assert "--- TASK ---" in wrapped


def test_wrap_persona_prompt_persona_precedes_task():
    wrapped, _ = wrap_persona_prompt("the-oracle", "TASK_SENTINEL")
    assert wrapped.index("--- PERSONA") < wrapped.index("--- TASK ---")
    assert wrapped.index("--- END PERSONA") < wrapped.index("TASK_SENTINEL")
