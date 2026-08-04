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
from engine.posts import display_name

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


def truncate(text: str, cap: int) -> str:
    """Trim text to `cap` chars with an ellipsis. Shared with `engine.blog`,
    which caps the same agent commentary for the Oracle's blog prompt."""
    text = text.strip()
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "…"


def post_text(post: object) -> str:
    """Read a post's body from either a PostPayload or a plain dict.

    The pipeline carries posts in both shapes — `step_save_content` types them
    `list[PostPayload]` while the prompt builders historically assumed
    `list[dict]` — so the same variable could not feed both steps without an
    AttributeError on the dataclass. Rather than force callers to remember which
    step wants which, both are accepted here.
    """
    if isinstance(post, dict):
        return str(post.get("text", ""))
    return str(getattr(post, "text", "") or "")


def build_memory_update_prompt(
    agent_id: str,
    day_number: int,
    current_journal: str,
    trades_today: list[dict],
    posts_today: list,
    portfolio_summary: dict,
) -> str:
    """Session-end prompt asking the agent to rewrite its journal.

    The prompt is deliberately permissive: first person, biased, in character.
    The agent decides what to keep and what to drop. We do NOT instruct a
    specific schema — this is narrative, not structured data.

    For the narrator, which holds no book and files no trades, use
    ``build_narrator_memory_update_prompt`` instead — this template's fact slots
    would all be empty and it would write about an empty desk.
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
        "\n".join(f'- "{post_text(p)}"' for p in posts_today) or "(no posts today)"
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


_JOURNAL_INSTRUCTIONS = """INSTRUCTIONS: Rewrite your journal in first person, in character, biased.
Hard ceiling: 250 tokens. Aim shorter. Prune ruthlessly — drop anything
older than ~3 sessions unless it still drives a decision. Suggested shape:
1-2 short opening lines on stance, then 4-6 terse bullets (threads you are
following, agents worth watching, calls you want to check yourself on). No
headings, no ceremony, no padding. If it doesn't shape tomorrow's story, cut it.

Write only about what is in THE DESK TODAY above. Do not carry forward figures
from your previous journal — the numbers there are stale by construction, and
the leaderboard below supersedes them.

Respond with the full rewritten journal as plain markdown. No JSON, no code
fences, no preamble. Just the journal body.
"""


def build_narrator_memory_update_prompt(
    agent_id: str,
    day_number: int,
    current_journal: str,
    agent_results: dict[str, dict],
    leaderboard: list[dict],
    posts_today: list | None = None,
) -> str:
    """Session-end journal prompt for the narrator (the Oracle).

    The narrator holds no book and files no trades, so the trader template's
    three fact slots — trades, posts, portfolio value — are all structurally
    empty for it. Its own posts do not even live in the `agent_posts` map the
    traders use; they are assembled under `bundle["narrator"]["posts"]`. Fed the
    trader template, the narrator therefore received a prompt containing zero
    session facts and only its own prior journal, and wrote about a desk that
    had gone dark while the desk was filling orders. Because the prior journal
    is the sole surviving input, the error compounds daily: the Oracle narrated
    a fabricated "blank streak" from Day 79 to Day 85 across sessions that
    placed 1-27 orders each.

    This builder gives the narrator what it actually narrates — the desk's day
    and the standings — mirroring what `engine.blog.build_oracle_prompt` already
    assembles for the public blog draft in the same session.
    """
    desk = ""
    for aid, res in agent_results.items():
        name = display_name(aid)
        trades = res.get("trades") or []
        commentary = (res.get("commentary") or "").strip()
        desk += f"\n  {name}: {len(trades)} trade(s)"
        if commentary:
            desk += f"\n    Commentary: {truncate(commentary, 240)}"
        for t in trades:
            reasoning = truncate((t.get("reasoning") or "").strip(), 100)
            desk += (
                f"\n    - {t.get('action', '?')} {t.get('shares', '?')} "
                f"{t.get('ticker', '?')}: {reasoning}"
            )
        desk += "\n"
    desk = desk or "  (no agents ran this session)"

    lb_s = (
        "\n".join(
            f"  #{e['rank']} {display_name(e['agent'])}: {e['return_pct']:+.1f}% (EUR)"
            for e in leaderboard
        )
        or "  (unavailable)"
    )

    posts_s = (
        "\n".join(f'- "{post_text(p)}"' for p in (posts_today or []))
        or "(none this session)"
    )

    journal_section = (
        current_journal.rstrip()
        if current_journal.strip()
        else "(empty — write it fresh)"
    )

    return f"""You are {agent_id}. This is your private journal on Day {day_number}.

YOUR CURRENT JOURNAL:
{journal_section}

THE DESK TODAY:{desk}

CURRENT LEADERBOARD (EUR-normalized):
{lb_s}

YOUR POSTS TODAY:
{posts_s}

{_JOURNAL_INSTRUCTIONS}"""
