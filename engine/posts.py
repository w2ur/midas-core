"""Post generation — agent display names and schedule backed by roster config.

Imported by engine/daily_log.py, engine/blog.py, and engine/output_bundle.py.
All cast data is read from get_config().roster at call time — no module-level
frozen dicts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from engine.config import get_config


def display_name(agent_id: str) -> str:
    """Return the human-readable display name for an agent, falling back to agent_id."""
    spec = get_config().roster.get(agent_id)
    return spec.display_name if spec else agent_id


def _post_time(agent_id: str) -> str:
    """Return the post_time for an agent, defaulting to '12:00'."""
    spec = get_config().roster.get(agent_id)
    return spec.post_time if spec and spec.post_time else "12:00"


def _voice(agent_id: str) -> str:
    """Return the voice description for an agent."""
    spec = get_config().roster.get(agent_id)
    return spec.voice if spec else ""


def resolved_post_time(agent_id: str, post_date: date) -> str:
    """Resolve the agent's post time — fixed returns verbatim, 'random' returns deterministic HH:MM.

    'random' is seeded by MD5(date_iso::agent_id), truncated to a minute-offset
    within the 09:00-22:00 Paris window. Same (date, agent) always returns the
    same time — needed for SSG-friendly feed rendering.
    """
    raw = _post_time(agent_id)
    if raw != "random":
        return raw
    seed = f"{post_date.isoformat()}::{agent_id}"
    digest = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    start_h, end_h = 9, 22
    span_min = (end_h - start_h) * 60
    offset = digest % span_min
    h, m = divmod(offset, 60)
    return f"{start_h + h:02d}:{m:02d}"


@dataclass
class PostPayload:
    agent_id: str
    text: str
    mentions: list[str]
    kind: str
    parent_id: str | None
    refs: dict[str, Any]
    post_at: str

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "text": self.text,
            "mentions": self.mentions,
            "kind": self.kind,
            "parent_id": self.parent_id,
            "refs": self.refs,
            "post_at": self.post_at,
        }

    @classmethod
    def from_agent_output(cls, agent_id: str, raw: dict) -> "PostPayload":
        return cls(
            agent_id=agent_id,
            text=raw["text"],
            mentions=raw.get("mentions", []),
            kind=raw.get("kind", "trade"),
            parent_id=raw.get("parent_id"),
            refs=raw.get("refs", {}),
            post_at=raw.get("post_at", _post_time(agent_id)),
        )


def build_post_prompt(
    agent_id: str,
    all_results: dict[str, dict],
    oracle_blog: str | None = None,
) -> str:
    """Build the post-generation prompt for a single trading agent.

    When `oracle_blog` (the Oracle's body_md for today) is provided, it is
    injected as a context block so the agent can react to the Oracle's
    framing of the day, not just to other agents' raw moves.
    """
    display = display_name(agent_id)
    voice = _voice(agent_id)
    schedule = _post_time(agent_id)

    own = all_results.get(agent_id, {})
    own_section = f"YOUR COMMENTARY: {own.get('commentary', 'No commentary.')}\n"
    own_trades = own.get("trades", [])
    if own_trades:
        own_section += "YOUR TRADES:\n"
        for t in own_trades:
            own_section += f"  - {t['action']} {t.get('shares', '')} {t['ticker']}: {t.get('reasoning', '')}\n"
    else:
        own_section += "YOUR TRADES: None today.\n"

    others_section = "OTHER AGENTS TODAY:\n"
    for other_id, res in all_results.items():
        if other_id == agent_id:
            continue
        name = display_name(other_id)
        commentary = res.get("commentary", "")
        trades = res.get("trades", [])
        others_section += f"\n  {name}:\n    Commentary: {commentary}\n"
        if trades:
            for t in trades:
                others_section += f"    - {t['action']} {t.get('shares', '')} {t['ticker']}: {t.get('reasoning', '')}\n"

    oracle_section = (
        f"\nORACLE'S NARRATIVE TODAY:\n{oracle_blog}\n" if oracle_blog else ""
    )

    return f"""You are {display} writing short posts for the Midas Feed.

VOICE: {voice}

{own_section}
{others_section}{oracle_section}

INSTRUCTIONS:
- Write 1-3 posts. Soft 280-char guideline per post (readability, not a hard limit).
- At least one post about your own moves today. Prefix every ticker with $ ($BTC-EUR, $MSFT, $GLD) — the feed linkifies them to /ticker/SLUG.
- If another agent did something worth reacting to, write a post about it. Mention them by display name.
- Stay in character: {voice}
- Be specific — real numbers, real tickers, real reasoning. No vague platitudes.
- Your posts appear in the feed around {schedule}.

OUTPUT — JSON array, no other text:
[{{"text": "...", "mentions": ["agent-id-if-mentioned"], "kind": "trade|roast|market-take"}}]
"""


def parse_post_response(agent_id: str, response_text: str) -> list[PostPayload]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip().startswith("```") else len(lines)
        text = "\n".join(lines[start:end]).strip()
    raw = json.loads(text)
    return [PostPayload.from_agent_output(agent_id, r) for r in raw]


def save_daily_posts(post_date: date, all_posts: dict[str, list[PostPayload]]) -> Path:
    posts_dir = get_config().posts_dir
    posts_dir.mkdir(parents=True, exist_ok=True)
    path = posts_dir / f"{post_date.isoformat()}.json"
    out = {aid: [p.to_dict() for p in posts] for aid, posts in all_posts.items()}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path
