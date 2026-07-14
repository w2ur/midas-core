"""Tests for the posts module."""

import json
from datetime import date
from pathlib import Path

import pytest

from engine.config import get_config
from engine.posts import (
    PostPayload,
    build_post_prompt,
    display_name,
    parse_post_response,
    resolved_post_time,
    save_daily_posts,
)


class TestAgentRoster:
    """Roster-shape assertions backed by config (no frozen module dicts)."""

    @pytest.mark.live_cast
    def test_12_agents_in_roster(self) -> None:
        cfg = get_config()
        expected = {
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
            "the-manager",
        }
        assert set(cfg.roster.keys()) == expected

    @pytest.mark.live_cast
    def test_10_trading_agents(self) -> None:
        cfg = get_config()
        trading = {
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
        }
        assert trading == set(cfg.trading_roster)

    def test_oracle_not_in_trading_roster(self) -> None:
        assert "the-oracle" not in get_config().trading_roster

    def test_every_agent_has_a_voice(self) -> None:
        cfg = get_config()
        for aid, spec in cfg.roster.items():
            if spec.role in ("trader", "narrator"):
                assert spec.voice, f"{aid} has empty voice"


class TestPostPayload:
    def test_create_and_roundtrip(self) -> None:
        post = PostPayload(
            agent_id="satoshi",
            text="Loaded BTC.",
            mentions=["goldfinger"],
            kind="trade",
            parent_id=None,
            refs={"order_id": "ord_2026-04-17_satoshi_001", "tags": ["BTC-EUR"]},
            post_at="23:00",
        )
        d = post.to_dict()
        assert d["kind"] == "trade"
        assert d["refs"]["order_id"] == "ord_2026-04-17_satoshi_001"
        reconstructed = PostPayload.from_agent_output("satoshi", d)
        assert reconstructed == post

    @pytest.mark.live_cast
    def test_default_post_at_uses_schedule(self) -> None:
        raw = {"text": "x", "kind": "trade"}
        p = PostPayload.from_agent_output("goldfinger", raw)
        assert p.post_at == "11:00"

    def test_oracle_default_post_at(self) -> None:
        raw = {"text": "x", "kind": "scoreboard"}
        p = PostPayload.from_agent_output("the-oracle", raw)
        assert p.post_at == "12:00"


class TestResolvedPostTime:
    @pytest.mark.live_cast
    def test_fixed_time_returned_verbatim(self) -> None:
        assert resolved_post_time("goldfinger", date(2026, 4, 17)) == "11:00"

    def test_random_is_deterministic_per_date_agent(self) -> None:
        d = date(2026, 4, 17)
        assert resolved_post_time("yolo-sapiens-eur", d) == resolved_post_time(
            "yolo-sapiens-eur", d
        )

    @pytest.mark.live_cast
    def test_random_differs_by_date(self) -> None:
        a = resolved_post_time("yolo-sapiens-eur", date(2026, 4, 17))
        b = resolved_post_time("yolo-sapiens-eur", date(2026, 4, 18))
        assert a != b

    @pytest.mark.live_cast
    def test_random_differs_by_agent(self) -> None:
        d = date(2026, 4, 17)
        a = resolved_post_time("yolo-sapiens-eur", d)
        b = resolved_post_time("yolo-sapiens-usd", d)
        assert a != b

    def test_random_in_window(self) -> None:
        result = resolved_post_time("yolo-sapiens-eur", date(2026, 4, 17))
        hh, mm = map(int, result.split(":"))
        total_min = hh * 60 + mm
        assert 9 * 60 <= total_min < 22 * 60


class TestBuildPostPrompt:
    pytestmark = pytest.mark.live_cast

    def test_includes_own_and_others(self) -> None:
        results = {
            "satoshi": {
                "commentary": "Loading the dip.",
                "trades": [
                    {
                        "action": "BUY",
                        "ticker": "BTC-EUR",
                        "shares": 0.01,
                        "reasoning": "F&G 12",
                    }
                ],
            },
            "goldfinger": {"commentary": "Gold consolidating.", "trades": []},
        }
        prompt = build_post_prompt("satoshi", results)
        assert "F&G 12" in prompt
        assert "Goldfinger" in prompt
        assert "Gold consolidating" in prompt
        assert "280" in prompt  # soft char guideline mentioned

    def test_prompt_contains_schedule(self) -> None:
        prompt = build_post_prompt(
            "satoshi", {"satoshi": {"commentary": "", "trades": []}}
        )
        assert "23:00" in prompt


class TestParsePostResponse:
    def test_clean_json(self) -> None:
        resp = json.dumps([{"text": "Loaded BTC.", "mentions": [], "kind": "trade"}])
        posts = parse_post_response("satoshi", resp)
        assert len(posts) == 1
        assert posts[0].kind == "trade"

    def test_with_code_fences(self) -> None:
        resp = '```json\n[{"text":"x","mentions":[],"kind":"trade"}]\n```'
        posts = parse_post_response("satoshi", resp)
        assert posts[0].text == "x"

    def test_reply_with_parent_id(self) -> None:
        resp = json.dumps(
            [
                {
                    "text": "No way that hedges.",
                    "mentions": ["yolo-sapiens-usd"],
                    "kind": "reply",
                    "parent_id": "post_xyz",
                },
            ]
        )
        posts = parse_post_response("steady-eddie-usd", resp)
        assert posts[0].parent_id == "post_xyz"


class TestSaveDailyPosts:
    def test_saves_grouped_by_agent(self, midas_data_root: Path) -> None:
        posts = {
            "satoshi": [PostPayload("satoshi", "x", [], "trade", None, {}, "23:00")],
            "the-oracle": [
                PostPayload("the-oracle", "y", [], "scoreboard", None, {}, "12:00")
            ],
        }
        path = save_daily_posts(date(2026, 4, 17), posts)
        data = json.loads(path.read_text())
        assert list(data.keys()) == ["satoshi", "the-oracle"]
        assert data["satoshi"][0]["kind"] == "trade"
