"""Blog draft generation — The Oracle's prompt builder, parser, and saver."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from engine.agent_memory import format_oracle_digest, truncate as _truncate
from engine.config import get_config
from engine.research import ORACLE_MAX_SEARCHES, render_research_instructions
from engine.posts import display_name as _display_name, PostPayload

logger = logging.getLogger(__name__)

# Trim caps applied to the Oracle prompt so first-token latency stays under
# the cloud streaming idle threshold. Verbatim agent commentary is not what
# the Oracle needs — the trades show actions, the leaderboard shows outcomes.
_ORACLE_COMMENTARY_CAP = 240
_ORACLE_TRADE_REASONING_CAP = 100


def _slugify(text: str) -> str:
    """Lower-case ASCII slug (letters/digits joined by single hyphens)."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "untitled"


@dataclass
class BlogDraft:
    """Daily blog post draft produced by The Oracle."""

    title: str
    body_md: str
    slug: str

    def to_dict(self) -> dict:
        return {"title": self.title, "body_md": self.body_md, "slug": self.slug}

    @classmethod
    def from_dict(cls, d: dict) -> "BlogDraft":
        # The Oracle sometimes omits or blanks required keys; degrade to sane
        # defaults so the blog step never crashes the unattended session on
        # loose output (2026-07-17 incident). A "Day N: …" title slugifies to
        # the "day-n-…" convention.
        title = d.get("title") or "Midas Daily"
        body_md = d.get("body_md") or ""
        slug = d.get("slug") or _slugify(title)
        return cls(title=title, body_md=body_md, slug=slug)


def _order_label(trade: dict) -> str:
    """One authored order as the narrator sees it.

    A conditional order (one carrying ``trigger``) is labelled ARMED with its
    condition (#73): rendered like a market order, a protective stop read as
    an executed sale, and the Oracle narrated stops and rails that had not
    fired as exits, which six agents then corrected in their own posts.
    """
    head = f"{trade.get('action', '?')} {trade.get('shares', '')} {trade.get('ticker', '?')}"
    trigger = trade.get("trigger")
    if not isinstance(trigger, dict):
        return head
    condition = f"if price {trigger.get('op', '?')} {trigger.get('level', '?')}"
    expires = trade.get("expires")
    until = f", until {expires}" if expires else ""
    return f"ARMED (not executed) {head} {condition}{until}"


def build_oracle_prompt(
    day_number: int,
    market_data: dict,
    agent_results: dict[str, dict],
    agent_posts: dict[str, list[dict]] | None = None,
    leaderboard: list[dict] | None = None,
    agent_memories: dict[str, str] | None = None,
    session_date: date | None = None,
) -> str:
    """Build The Oracle's daily prompt — blog draft + narrator posts.

    When `agent_memories` is provided (Ring 2 onwards), a journal digest is
    appended so The Oracle can cite specific prior entries in its narration.

    `agent_posts` is optional: when the Oracle runs BEFORE the post round
    (current pipeline ordering), pass `None` or an empty dict and the
    "AGENT POSTS TODAY" section is suppressed.

    `session_date` is the as-of date of the research block (defaults to today in UTC).
    """
    research = render_research_instructions(
        ORACLE_MAX_SEARCHES, session_date or datetime.now(timezone.utc).date(), places_orders=False
    )
    agent_posts = agent_posts or {}
    leaderboard = leaderboard or []
    market = "\n".join(
        f"  {k}: {v:,.2f}"
        for k, v in market_data.items()
        if isinstance(v, (int, float))
    )

    agents_s = ""
    for aid, res in agent_results.items():
        name = _display_name(aid)
        commentary = _truncate(res.get("commentary", ""), _ORACLE_COMMENTARY_CAP)
        agents_s += f"\n  {name}:\n    Commentary: {commentary}\n"
        for t in res.get("trades", []):
            reasoning = _truncate(t.get("reasoning", ""), _ORACLE_TRADE_REASONING_CAP)
            agents_s += f"    - {_order_label(t)}: {reasoning}\n"

    posts_s = ""
    for aid, posts in agent_posts.items():
        name = _display_name(aid)
        posts_s += f"\n  {name}:\n"
        for p in posts:
            text = p.get("text", "") if isinstance(p, dict) else str(p)
            posts_s += f'    - "{text}"\n'
    posts_block = f"\n\nAGENT POSTS TODAY:{posts_s}" if posts_s else ""

    def _lb_line(e: dict) -> str:
        # Rank orders on vs_benchmark_pp since 2026-08-14. The narrator must
        # see the ranked quantity, or a -9.4% book at #1 reads as an error it
        # will "correct" or explain away — the Day 79-85 fabrication class.
        vs = e.get("vs_benchmark_pp")
        if vs is not None:
            return (
                f"  #{e['rank']} {_display_name(e['agent'])}: "
                f"{vs:+.1f}pp vs benchmark (EUR return {e['return_pct']:+.1f}%)"
            )
        return (
            f"  #{e['rank']} {_display_name(e['agent'])}: {e['return_pct']:+.1f}% (EUR)"
        )

    lb_s = "\n".join(_lb_line(e) for e in leaderboard)

    journal_section = ""
    if agent_memories:
        journal_section = (
            "\n\nAGENT JOURNAL DIGEST (latest in-character entries — cite them when relevant):\n"
            + format_oracle_digest(agent_memories)
        )

    return f"""You are The Oracle, narrator of the Midas experiment. Day {day_number}.

MARKET DATA TODAY:
{market}

AGENT ACTIVITY TODAY:{agents_s}{posts_block}

Orders marked ARMED are conditional: they execute only if the price condition
is hit later, and none has executed today. Never describe an ARMED order as a
sale, exit, purchase or fill that happened.

CURRENT LEADERBOARD (EUR-normalized):
{lb_s}{journal_section}

INSTRUCTIONS: produce a daily blog draft and 1-3 narrator posts following your agent definition.

{research}

OUTPUT FORMAT — JSON object, no other text:
{{
  "blog_draft": {{"title": "Day {day_number}: ...", "body_md": "...", "slug": "day-{day_number}-..."}},
  "posts": [{{"text": "...", "mentions": ["agent-id"], "kind": "scoreboard|recap|highlight"}}],
  "sources": [{{"query": "...", "url": "...", "used_for": "..."}}]
}}
("sources" is OPTIONAL: one entry per WebSearch you ran; omit it if you did not search.)
"""


def _load_response_json(response: str) -> dict:
    """Strip a code fence and parse the narrator's JSON; ``{}`` for any loose shape.

    Degrades every loose Oracle shape rather than crash the unattended session
    (2026-07-17): truncated/non-JSON output (the Oracle repeatedly trips the
    cloud streaming idle timeout) and a non-dict payload both resolve to ``{}``.
    """
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[start:end]).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def oracle_sources(response: str) -> object:
    """The raw ``sources`` value of a narrator response, or ``None``. Never raises."""
    return _load_response_json(response).get("sources")


def parse_oracle_response(
    response: str, agent_id: str | None = None
) -> tuple[BlogDraft, list[PostPayload]]:
    """Parse a narrator's JSON response (handles code-fenced input).

    ``agent_id`` is the narrator the posts are attributed to. When omitted it
    resolves from `role: narrator` in the roster rather than being hardcoded
    to `the-oracle`: on a fork whose narrator is named anything else, every
    post carried an author id no agent on that desk has, so display names and
    crests resolved to nothing. Falls back to `the-oracle` only when the desk
    declares no narrator at all, which is a configuration this function should
    never be reached on — a warning, not a raise, because the module's whole
    policy is to degrade rather than kill an unattended session.
    """
    data = _load_response_json(response)
    blog_draft = data.get("blog_draft")
    draft = BlogDraft.from_dict(blog_draft if isinstance(blog_draft, dict) else {})
    raw_posts = data.get("posts")
    if not isinstance(raw_posts, list):
        raw_posts = []
    narrator_id = agent_id
    if narrator_id is None:
        narrators = get_config().narrators
        if narrators:
            narrator_id = narrators[0]
        else:
            narrator_id = "the-oracle"
            logger.warning(
                "no agent declares role: narrator; attributing posts to "
                "%r, which is almost certainly not an agent on this desk",
                narrator_id,
            )
    posts = []
    for p in raw_posts:
        if not isinstance(p, dict):
            continue
        try:
            posts.append(PostPayload.from_agent_output(narrator_id, p))
        except (KeyError, TypeError, ValueError):
            continue
    return draft, posts


def save_daily_blog_draft(d: date, draft: BlogDraft) -> Path:
    """Save a blog draft as markdown with YAML frontmatter. Title is always quoted."""
    blog_dir = get_config().blog_dir
    blog_dir.mkdir(parents=True, exist_ok=True)
    path = blog_dir / f"{d.isoformat()}.md"
    frontmatter = (
        "---\n"
        f'title: "{draft.title}"\n'
        f"slug: {draft.slug}\n"
        f"date: {d.isoformat()}\n"
        "---\n\n"
    )
    path.write_text(frontmatter + draft.body_md, encoding="utf-8")
    return path
