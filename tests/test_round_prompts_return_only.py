"""The post and journal rounds return text; they never write files or search.

Both rounds run inside a dispatch-guard fence (engine.dispatch_guard). A
journal the agent saved to disk itself, as "Rewrite your journal" invited,
would abort the session, so the prompts must say plainly that the text is
returned and the session writes it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.daily_session import (
    step_build_memory_update_prompts,
    step_build_post_prompts,
)

NO_FILES = (
    "Do not create, edit or delete any file — return the text in your response; "
    "the session writes it."
)
NO_SEARCH = "Do not search the web in this task."


def _trader() -> str:
    from engine.config import get_config

    return get_config().trading_roster[0]


def test_every_post_prompt_says_return_only_and_no_search(
    midas_data_root: Path,
) -> None:
    agent = _trader()
    prompts = step_build_post_prompts(
        {agent: {"commentary": "Holding.", "trades": []}}, oracle_blog="A calm day."
    )
    assert prompts
    for text in prompts.values():
        assert NO_FILES in text
        assert NO_SEARCH in text


@pytest.mark.parametrize("role", ["trader", "narrator"])
def test_every_journal_prompt_says_return_only_and_no_search(
    midas_data_root: Path, role: str
) -> None:
    from engine.config import get_config

    agent = _trader()
    prompts = step_build_memory_update_prompts(
        agent_results={agent: {"trades": [], "commentary": "Steady."}},
        agent_posts={},
        portfolio_summaries={agent: {"currency": "EUR"}},
        day_number=3,
        leaderboard=[],
    )
    target = agent if role == "trader" else get_config().narrators[0]
    text = prompts[target]
    assert NO_FILES in text
    assert NO_SEARCH in text
    assert "you do not save it" in text
