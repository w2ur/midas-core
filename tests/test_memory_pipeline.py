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


def narrator_id() -> str:
    """The desk's narrator, resolved rather than named.

    These tests hardcoded `the-oracle` and passed on the demo desk only
    because `step_build_memory_update_prompts` hardcoded it too — the test and
    the code shared one wrong assumption, so neither could catch it. Core's
    demo roster declares its own narrator under a different id; resolving here
    means these run on both desks and actually exercise the resolution.
    """
    from engine.config import get_config

    narrators = get_config().narrators
    assert narrators, "this desk declares no role: narrator"
    return narrators[0]


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
        oracle_id = narrator_id()
        assert "satoshi" in prompts
        assert oracle_id in prompts
        # The narrator's prompt mentions its own id and the Day number.
        assert oracle_id in prompts[oracle_id]
        assert "Day 1" in prompts[oracle_id]

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


class TestNarratorPrompt:
    """Regression: 0da774525 — the Oracle was fed the trader journal template,
    whose three fact slots (trades, posts, portfolio value) are all structurally
    empty for a narrator. Its own posts are never in `agent_posts` either. The
    prompt therefore carried zero session facts and the Oracle narrated a dark
    desk across sessions that placed 1-27 orders, compounding daily off its own
    prior journal (a fabricated "blank streak", Day 79 to Day 85).
    """

    def _prompts(self, **overrides):
        kwargs = dict(
            agent_results={
                "satoshi": {
                    "trades": [
                        {
                            "action": "BUY",
                            "shares": 3,
                            "ticker": "BTC-EUR",
                            "reasoning": "Box broke the wrong way; adding.",
                        }
                    ],
                    "commentary": "Re-armed the peel rungs.",
                },
                "goldfinger": {"trades": [], "commentary": "Waiting on yields."},
            },
            agent_posts={"satoshi": [{"text": "See you at 68k."}]},
            portfolio_summaries={"satoshi": {"portfolio_value_base": 10_000.0}},
            day_number=85,
            leaderboard=[
                {"rank": 1, "agent": "satoshi", "return_pct": -9.73},
                {"rank": 2, "agent": "goldfinger", "return_pct": -18.30},
            ],
            oracle_posts=[{"text": "Scoreboard: the twins are 15 points apart."}],
        )
        kwargs.update(overrides)
        return step_build_memory_update_prompts(**kwargs)

    def test_oracle_prompt_carries_the_desks_activity(self, tmp_journals: Path) -> None:
        from engine.posts import display_name

        oracle = self._prompts()[narrator_id()]

        assert "BTC-EUR" in oracle
        assert "Re-armed the peel rungs" in oracle
        # Resolved at runtime, not hardcoded: the demo desk in midas-core has a
        # different cast, so a literal "Satoshi" would fail there.
        assert display_name("satoshi") in oracle

    def test_oracle_prompt_carries_leaderboard_and_own_posts(
        self, tmp_journals: Path
    ) -> None:
        oracle = self._prompts()[narrator_id()]

        assert "-9.7%" in oracle
        assert "-18.3%" in oracle
        assert "the twins are 15 points apart" in oracle

    def test_oracle_prompt_never_claims_an_empty_desk(self, tmp_journals: Path) -> None:
        """The exact strings that produced the fabricated blank streak."""
        oracle = self._prompts()[narrator_id()]

        assert "(no trades today)" not in oracle
        assert "(no posts today)" not in oracle
        # The narrator holds no book; a 0.00 PV line invited "no PV today".
        assert "PORTFOLIO VALUE TODAY" not in oracle

    def test_oracle_prompt_accepts_postpayload_objects(
        self, tmp_journals: Path
    ) -> None:
        """`step_save_content` types posts as PostPayload; the same variable
        must be able to feed this step without an AttributeError."""
        from engine.posts import PostPayload

        payload = PostPayload(
            agent_id=narrator_id(),
            text="Scoreboard: the twins are 15 points apart.",
            mentions=[],
            kind="scoreboard",
            parent_id=None,
            refs={},
            post_at="2026-08-03T20:00:00Z",
        )
        oracle = self._prompts(oracle_posts=[payload])[narrator_id()]

        assert "the twins are 15 points apart" in oracle
