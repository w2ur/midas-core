"""Tests for engine.blog."""

import json
from datetime import date
from pathlib import Path

import pytest

from engine.blog import (
    BlogDraft,
    build_oracle_prompt,
    parse_oracle_response,
    save_daily_blog_draft,
)


class TestBlogDraft:
    def test_roundtrip(self) -> None:
        d = BlogDraft(title="Day 2: YOLO Leads", body_md="# Body", slug="day-2-yolo")
        out = d.to_dict()
        assert out == {
            "title": "Day 2: YOLO Leads",
            "body_md": "# Body",
            "slug": "day-2-yolo",
        }
        reconstructed = BlogDraft.from_dict(out)
        assert reconstructed == d


class TestBuildOraclePrompt:
    @pytest.mark.live_cast
    def test_includes_all_agent_data(self) -> None:
        agent_results = {
            "steady-eddie-eur": {"commentary": "Holding.", "trades": []},
            "yolo-sapiens-eur": {
                "commentary": "Going all in.",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "SX5E.PA",
                        "shares": 3,
                        "reasoning": "3x EU.",
                    }
                ],
            },
        }
        agent_posts = {
            "steady-eddie-eur": [{"text": "Patience.", "kind": "trade"}],
        }
        leaderboard = [
            {"agent": "yolo-sapiens-eur", "return_pct": 14.2, "rank": 1},
            {"agent": "steady-eddie-eur", "return_pct": 2.1, "rank": 2},
        ]
        market_data = {
            "sp500": 7000.0,
            "gold": 4900.0,
            "btc": 75000.0,
            "eur_usd": 1.1784,
        }
        prompt = build_oracle_prompt(
            day_number=2,
            market_data=market_data,
            agent_results=agent_results,
            agent_posts=agent_posts,
            leaderboard=leaderboard,
        )
        assert "Day 2" in prompt
        assert "Steady Eddie EUR" in prompt
        assert "YOLO Sapiens EUR" in prompt
        assert "SX5E.PA" in prompt
        assert "14.2" in prompt
        assert "Patience." in prompt

    def test_prompt_mentions_market_data(self) -> None:
        prompt = build_oracle_prompt(
            day_number=1,
            market_data={"sp500": 7000.0, "eur_usd": 1.18},
            agent_results={},
            agent_posts={},
            leaderboard=[],
        )
        assert "7,000" in prompt or "7000" in prompt

    def test_journal_digest_omitted_when_memories_none(self) -> None:
        prompt = build_oracle_prompt(
            day_number=1,
            market_data={},
            agent_results={},
            agent_posts={},
            leaderboard=[],
        )
        assert "JOURNAL DIGEST" not in prompt

    def test_long_commentary_is_truncated(self) -> None:
        long = "X" * 2000
        prompt = build_oracle_prompt(
            day_number=1,
            market_data={},
            agent_results={"satoshi": {"commentary": long, "trades": []}},
            agent_posts={},
            leaderboard=[],
        )
        # The full 2000-X commentary must NOT appear — would push the prompt
        # past the cloud streaming first-token threshold for the Oracle.
        assert long not in prompt
        assert "…" in prompt
        # Truncated form is bounded — first-token latency depends on it.
        assert prompt.count("X") < 300

    def test_long_trade_reasoning_is_truncated(self) -> None:
        long_reasoning = "Y" * 500
        prompt = build_oracle_prompt(
            day_number=1,
            market_data={},
            agent_results={
                "satoshi": {
                    "commentary": "ok",
                    "trades": [
                        {
                            "action": "BUY",
                            "ticker": "BTC-EUR",
                            "shares": 1,
                            "reasoning": long_reasoning,
                        }
                    ],
                }
            },
            agent_posts={},
            leaderboard=[],
        )
        assert long_reasoning not in prompt
        assert prompt.count("Y") < 200

    def test_journal_digest_included_when_memories_provided(self) -> None:
        prompt = build_oracle_prompt(
            day_number=1,
            market_data={},
            agent_results={},
            agent_posts={},
            leaderboard=[],
            agent_memories={
                "satoshi": "BTC cycle thesis: mid-markup.",
                "goldfinger": "Gold is the only honest asset.",
            },
        )
        assert "JOURNAL DIGEST" in prompt
        assert "### satoshi" in prompt
        assert "BTC cycle thesis" in prompt
        assert "honest asset" in prompt


class TestParseOracleResponse:
    def test_clean_json(self) -> None:
        resp = json.dumps(
            {
                "blog_draft": {"title": "Day 2", "body_md": "Body", "slug": "day-2"},
                "posts": [
                    {"text": "Scoreboard.", "mentions": [], "kind": "scoreboard"},
                ],
            }
        )
        draft, posts = parse_oracle_response(resp)
        assert draft.title == "Day 2"
        assert draft.slug == "day-2"
        assert len(posts) == 1
        assert posts[0].kind == "scoreboard"

    def test_with_code_fences(self) -> None:
        inner = json.dumps(
            {
                "blog_draft": {"title": "T", "body_md": "B", "slug": "s"},
                "posts": [],
            }
        )
        resp = f"```json\n{inner}\n```"
        draft, posts = parse_oracle_response(resp)
        assert draft.title == "T"
        assert posts == []


class TestSaveDailyBlogDraft:
    def test_writes_frontmatter_and_body(self, midas_data_root: Path) -> None:
        draft = BlogDraft(
            title="Day 1: Opening",
            body_md="# Day 1\n\nThe agents made their first trades.",
            slug="day-1",
        )
        path = save_daily_blog_draft(date(2026, 4, 17), draft)
        assert path.name == "2026-04-17.md"
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert 'title: "Day 1: Opening"' in content
        assert "slug: day-1" in content
        assert "date: 2026-04-17" in content
        assert "The agents made their first trades." in content

    def test_quotes_title_even_when_it_contains_colon(
        self, midas_data_root: Path
    ) -> None:
        draft = BlogDraft(title="Day 2: The Split", body_md="Content.", slug="day-2")
        path = save_daily_blog_draft(date(2026, 4, 17), draft)
        content = path.read_text(encoding="utf-8")
        # Title containing colon MUST be quoted to stay valid YAML
        assert 'title: "Day 2: The Split"' in content
