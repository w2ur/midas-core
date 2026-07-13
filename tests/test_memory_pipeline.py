"""Tests for the Ring 2 memory round in scripts.daily_session.

Covers:
- step_load_memories loads the right files in the right order
- step_build_memory_update_prompts covers all 10 traders + the-oracle
- step_save_memories skips blank responses (no-wipe safety)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine import agent_memory
from scripts.daily_session import (
    step_build_memory_update_prompts,
    step_load_memories,
    step_save_memories,
)


@pytest.fixture
def tmp_journals(midas_data_root: Path) -> Path:
    from engine.config import get_config

    return get_config().journal_dir


class TestStepLoadMemories:
    def test_missing_journals_returned_as_empty(self, tmp_journals: Path) -> None:
        memories = step_load_memories(["satoshi", "goldfinger"])
        assert memories == {"satoshi": "", "goldfinger": ""}

    def test_loads_existing_journals(self, tmp_journals: Path) -> None:
        agent_memory.save_journal("satoshi", "BTC is king.")
        memories = step_load_memories(["satoshi", "goldfinger"])
        assert "BTC is king" in memories["satoshi"]
        assert memories["goldfinger"] == ""


class TestStepBuildMemoryUpdatePrompts:
    def test_includes_oracle_even_without_agent_result(
        self, tmp_journals: Path
    ) -> None:
        prompts = step_build_memory_update_prompts(
            agent_results={
                "satoshi": {"trades": [], "commentary": "Steady."},
            },
            agent_posts={"satoshi": [{"text": "Holding."}]},
            portfolio_summaries={
                "satoshi": {"portfolio_value_base": 10_000.0, "currency": "EUR"}
            },
            day_number=1,
        )
        assert "satoshi" in prompts
        assert "the-oracle" in prompts
        # The Oracle's prompt mentions its own id and the Day number.
        assert "the-oracle" in prompts["the-oracle"]
        assert "Day 1" in prompts["the-oracle"]

    def test_trader_prompt_embeds_current_journal(self, tmp_journals: Path) -> None:
        agent_memory.save_journal("satoshi", "Day 0: BTC cycle theory.")
        prompts = step_build_memory_update_prompts(
            agent_results={"satoshi": {"trades": [], "commentary": "Steady."}},
            agent_posts={},
            portfolio_summaries={"satoshi": {"currency": "EUR"}},
            day_number=2,
        )
        assert "BTC cycle theory" in prompts["satoshi"]


class TestStepSaveMemories:
    def test_saves_non_empty_journals(self, tmp_journals: Path) -> None:
        written = step_save_memories(
            {
                "satoshi": "Day 1. Bought more BTC.",
                "goldfinger": "Day 1. Gold held.",
            }
        )
        assert written == 2
        assert "Bought more BTC" in agent_memory.load_journal("satoshi")
        assert "Gold held" in agent_memory.load_journal("goldfinger")

    def test_skips_blank_responses_without_overwriting(
        self, tmp_journals: Path
    ) -> None:
        agent_memory.save_journal("satoshi", "Day 0 seed.")
        written = step_save_memories({"satoshi": "   \n  "})
        assert written == 0
        # Seed must still be there — a blank response from Claude must not wipe state.
        assert "Day 0 seed" in agent_memory.load_journal("satoshi")

    def test_partial_response_preserves_other_journals(
        self, tmp_journals: Path
    ) -> None:
        agent_memory.save_journal("satoshi", "Day 0 satoshi.")
        agent_memory.save_journal("goldfinger", "Day 0 goldfinger.")
        step_save_memories({"satoshi": "Day 1 satoshi.", "goldfinger": ""})
        assert "Day 1 satoshi" in agent_memory.load_journal("satoshi")
        assert "Day 0 goldfinger" in agent_memory.load_journal("goldfinger")
