"""Blog draft generation — The Oracle's prompt builder, parser, and saver."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from engine.agent_memory import format_oracle_digest, truncate as _truncate
from engine.config import get_config
from engine.posts import display_name as _display_name, PostPayload

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


def build_oracle_prompt(
    day_number: int,
    market_data: dict,
    agent_results: dict[str, dict],
    agent_posts: dict[str, list[dict]] | None = None,
    leaderboard: list[dict] | None = None,
    agent_memories: dict[str, str] | None = None,
) -> str:
    """Build The Oracle's daily prompt — blog draft + narrator posts.

    When `agent_memories` is provided (Ring 2 onwards), a journal digest is
    appended so The Oracle can cite specific prior entries in its narration.

    `agent_posts` is optional: when the Oracle runs BEFORE the post round
    (current pipeline ordering), pass `None` or an empty dict and the
    "AGENT POSTS TODAY" section is suppressed.
    """
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
            agents_s += f"    - {t['action']} {t.get('shares', '')} {t['ticker']}: {reasoning}\n"

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

CURRENT LEADERBOARD (EUR-normalized):
{lb_s}{journal_section}

INSTRUCTIONS: produce a daily blog draft and 1-3 narrator posts following your agent definition.

OUTPUT FORMAT — JSON object, no other text:
{{
  "blog_draft": {{"title": "Day {day_number}: ...", "body_md": "...", "slug": "day-{day_number}-..."}},
  "posts": [{{"text": "...", "mentions": ["agent-id"], "kind": "scoreboard|recap|highlight"}}]
}}
"""


def parse_oracle_response(response: str) -> tuple[BlogDraft, list[PostPayload]]:
    """Parse The Oracle's JSON response (handles code-fenced input)."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[start:end]).strip()
    # Degrade every loose Oracle shape rather than crash the unattended session
    # (2026-07-17): truncated/non-JSON output (the Oracle repeatedly trips the
    # cloud streaming idle timeout), a non-dict payload or blog_draft, and a
    # null/absent/non-list posts — plus any malformed post element — all resolve
    # to safe empties instead of raising.
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    blog_draft = data.get("blog_draft")
    draft = BlogDraft.from_dict(blog_draft if isinstance(blog_draft, dict) else {})
    raw_posts = data.get("posts")
    if not isinstance(raw_posts, list):
        raw_posts = []
    posts = []
    for p in raw_posts:
        if not isinstance(p, dict):
            continue
        try:
            posts.append(PostPayload.from_agent_output("the-oracle", p))
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
