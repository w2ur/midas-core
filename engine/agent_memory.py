"""Per-agent persistent journals (Ring 2 memory).

Each agent owns a first-person, biased markdown file at
`data/agent_memory/{agent_id}.md`. Loaded into the daily prompt, rewritten by
the agent at session end. The file is committed to git so the sandboxed remote
run can read prior entries without out-of-band state.

The Oracle reads a digest of all 11 journals to cite specific entries in its
narration (e.g. "Satoshi swore off ETH on Day 12 — today he bought it back").

Size budget: ~1000 tokens per journal, enforced via `journal_excerpt`. Agents
are responsible for pruning their own history at rewrite time; this module
provides a safety net that trims the prompt-side injection to the tail.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from engine.config import get_config

# Roughly 1000 tokens of English at 4 chars/token.
DEFAULT_MAX_CHARS = 4000


def _path_for(agent_id: str) -> Path:
    return get_config().journal_dir / f"{agent_id}.md"


def load_journal(agent_id: str) -> str:
    """Read an agent's journal. Returns empty string when the file is absent."""
    path = _path_for(agent_id)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def save_journal(agent_id: str, content: str) -> Path:
    """Write an agent's journal atomically (tmp + os.replace). Creates the directory."""
    journal_dir = get_config().journal_dir
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = _path_for(agent_id)
    if not content.endswith("\n"):
        content = content + "\n"
    # Write to a sibling tmp file so os.replace is on the same filesystem.
    fd, tmp_name = tempfile.mkstemp(dir=journal_dir, prefix=".journal_tmp_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path


def journal_excerpt(content: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Return the tail of the journal, cut at a line boundary when possible.

    The tail matters more than the head: newer entries describe the agent's
    current stance and recent grudges. When the content exceeds the budget,
    drop leading whole lines until it fits.
    """
    if len(content) <= max_chars:
        return content
    # Try to cut at a line boundary so we don't leave a half-sentence at the top.
    overflow = len(content) - max_chars
    boundary = content.find("\n", overflow)
    if boundary == -1 or boundary >= len(content) - 1:
        return content[-max_chars:]
    return content[boundary + 1 :]


def format_memory_section(agent_id: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Render an agent's journal as a labeled markdown block for prompt injection.

    Returns a block that starts with a heading so the agent can locate it in
    their prompt. Empty journal returns an explicit "first session" notice.
    """
    content = load_journal(agent_id)
    if not content.strip():
        return (
            "## Your journal\n\n"
            "(No prior entries. This is your first session — write your journal from scratch at day-end.)\n"
        )
    excerpt = journal_excerpt(content, max_chars)
    return "## Your journal (latest entries)\n\n" + excerpt.rstrip() + "\n"


def format_oracle_digest(
    agent_memories: dict[str, str], per_agent_chars: int = 250
) -> str:
    """Render a digest of all agents' journals for The Oracle's prompt.

    One short excerpt per agent — enough to cite a recent stance or prediction
    without blowing the Oracle's context. Ordering matches the input dict.
    """
    if not agent_memories:
        return "(No agent journals available.)"
    lines = []
    for agent_id, content in agent_memories.items():
        excerpt = journal_excerpt(content, per_agent_chars).strip()
        if not excerpt:
            excerpt = "(empty journal)"
        lines.append(f"### {agent_id}\n{excerpt}")
    return "\n\n".join(lines)


def build_memory_update_prompt(
    agent_id: str,
    day_number: int,
    current_journal: str,
    trades_today: list[dict],
    posts_today: list[dict],
    portfolio_summary: dict,
) -> str:
    """Session-end prompt asking the agent to rewrite its journal.

    The prompt is deliberately permissive: first person, biased, in character.
    The agent decides what to keep and what to drop. We do NOT instruct a
    specific schema — this is narrative, not structured data.
    """
    trades_s = (
        "\n".join(
            f"- {t.get('action', '?')} {t.get('shares', '?')} {t.get('ticker', '?')}: "
            f"{t.get('reasoning', '')}"
            for t in trades_today
        )
        or "(no trades today)"
    )
    posts_s = (
        "\n".join(f'- "{p.get("text", "")}"' for p in posts_today) or "(no posts today)"
    )
    pv = portfolio_summary.get("portfolio_value_base") or portfolio_summary.get(
        "cash", 0.0
    )
    currency = portfolio_summary.get("currency", "EUR")

    journal_section = (
        current_journal.rstrip()
        if current_journal.strip()
        else "(empty — write it fresh)"
    )

    return f"""You are {agent_id}. This is your private journal on Day {day_number}.

YOUR CURRENT JOURNAL:
{journal_section}

TODAY'S TRADES:
{trades_s}

TODAY'S POSTS:
{posts_s}

PORTFOLIO VALUE TODAY: {pv:,.2f} {currency}

INSTRUCTIONS: Rewrite your journal in first person, in character, biased.
Hard ceiling: 250 tokens. Aim shorter. Prune ruthlessly — drop anything
older than ~3 sessions unless it still drives a decision. Suggested shape:
1-2 short opening lines on stance, then 4-6 terse bullets (rules of thumb,
open positions worth noting, gripes about other agents). No headings, no
ceremony, no padding. If it doesn't change tomorrow's trade or your mood,
cut it.

Respond with the full rewritten journal as plain markdown. No JSON, no code
fences, no preamble. Just the journal body.
"""
