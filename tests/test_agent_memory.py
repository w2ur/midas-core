"""Tests for engine.agent_memory — per-agent persistent journals."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine import agent_memory
from engine.agent_memory import (
    DEFAULT_MAX_CHARS,
    build_memory_update_prompt,
    format_memory_section,
    format_oracle_digest,
    journal_excerpt,
    load_journal,
    save_journal,
)


@pytest.fixture
def temp_journal_dir(midas_data_root: Path) -> Path:
    """Redirect journal writes to an isolated tmp path via config env redirect."""
    from engine.config import get_config

    return get_config().journal_dir


class TestLoadSave:
    def test_load_missing_returns_empty_string(self, temp_journal_dir: Path) -> None:
        assert load_journal("nobody") == ""

    def test_roundtrip(self, temp_journal_dir: Path) -> None:
        save_journal("satoshi", "Day 1. BTC at 75k. I'm early.")
        assert load_journal("satoshi") == "Day 1. BTC at 75k. I'm early.\n"

    def test_save_adds_trailing_newline(self, temp_journal_dir: Path) -> None:
        path = save_journal("x", "no newline here")
        assert path.read_text(encoding="utf-8").endswith("\n")

    def test_save_preserves_existing_trailing_newline(
        self, temp_journal_dir: Path
    ) -> None:
        path = save_journal("x", "already has one\n")
        assert path.read_text(encoding="utf-8") == "already has one\n"

    def test_save_creates_directory(self, midas_data_root: Path) -> None:
        from engine.config import get_config

        save_journal("x", "hello")
        assert (get_config().journal_dir / "x.md").exists()

    def test_utf8_roundtrip(self, temp_journal_dir: Path) -> None:
        content = "Today I felt — actually, pretty good. Café, €, 🎯."
        save_journal("agent", content)
        assert load_journal("agent") == content + "\n"


class TestJournalExcerpt:
    def test_short_content_unchanged(self) -> None:
        content = "short entry\n"
        assert journal_excerpt(content, max_chars=100) == content

    def test_long_content_trimmed_to_budget(self) -> None:
        content = "\n".join(f"line {i}" for i in range(1000))
        excerpt = journal_excerpt(content, max_chars=100)
        assert len(excerpt) <= 100
        # Tail-preserving: last line must still be present.
        assert "line 999" in excerpt

    def test_excerpt_cuts_at_line_boundary_when_possible(self) -> None:
        content = "aaa\nbbbbbbb\nccc\n"
        excerpt = journal_excerpt(content, max_chars=10)
        # Must not start mid-line ("bbbbb"); cut drops the overflow line.
        assert not excerpt.startswith("bbbb")

    def test_excerpt_never_exceeds_budget_on_random_inputs(self) -> None:
        # Lightweight property-style check over a range of sizes.
        for n_lines in (1, 5, 50, 500):
            content = "\n".join("x" * 40 for _ in range(n_lines)) + "\n"
            assert len(journal_excerpt(content, max_chars=200)) <= max(200, 40) + 1


class TestFormatMemorySection:
    def test_empty_journal_returns_first_session_notice(
        self, temp_journal_dir: Path
    ) -> None:
        section = format_memory_section("brand-new")
        assert "first session" in section.lower()
        assert section.startswith("## Your journal")

    def test_non_empty_journal_injected(self, temp_journal_dir: Path) -> None:
        save_journal("satoshi", "Day 1. I bought BTC.")
        section = format_memory_section("satoshi")
        assert "Day 1. I bought BTC." in section
        assert "latest entries" in section

    def test_respects_max_chars(self, temp_journal_dir: Path) -> None:
        save_journal("verbose", "x" * 20_000)
        section = format_memory_section("verbose", max_chars=500)
        # Heading + trimmed body — total should be within budget + heading overhead.
        assert len(section) < 700


class TestOracleDigest:
    def test_empty_memories(self) -> None:
        assert format_oracle_digest({}) == "(No agent journals available.)"

    def test_digest_lists_each_agent(self) -> None:
        digest = format_oracle_digest(
            {
                "satoshi": "BTC cycle theory: we're mid-markup.",
                "goldfinger": "Gold is the only honest asset.",
            },
            per_agent_chars=200,
        )
        assert "### satoshi" in digest
        assert "### goldfinger" in digest
        assert "BTC cycle" in digest
        assert "honest asset" in digest

    def test_empty_individual_journal_labeled(self) -> None:
        digest = format_oracle_digest({"brand-new": ""}, per_agent_chars=100)
        assert "(empty journal)" in digest


class TestMemoryUpdatePrompt:
    def test_includes_required_sections(self) -> None:
        prompt = build_memory_update_prompt(
            agent_id="satoshi",
            day_number=42,
            current_journal="Day 41. Held ETH.",
            trades_today=[
                {
                    "action": "SELL",
                    "ticker": "ETH-EUR",
                    "shares": 2,
                    "reasoning": "Broke support.",
                }
            ],
            posts_today=[{"text": "Bye ETH."}],
            portfolio_summary={"portfolio_value_base": 9876.54, "currency": "EUR"},
        )
        assert "Day 42" in prompt
        assert "satoshi" in prompt
        assert "Held ETH." in prompt
        assert "SELL 2 ETH-EUR" in prompt
        assert "Bye ETH." in prompt
        assert "9,876.54 EUR" in prompt
        assert "first person" in prompt

    def test_handles_empty_trades_and_posts(self) -> None:
        prompt = build_memory_update_prompt(
            agent_id="x",
            day_number=1,
            current_journal="",
            trades_today=[],
            posts_today=[],
            portfolio_summary={"cash": 10000.0, "currency": "EUR"},
        )
        assert "(no trades today)" in prompt
        assert "(no posts today)" in prompt
        assert "(empty — write it fresh)" in prompt


def test_default_budget_matches_documented_token_size() -> None:
    # Documented: ~1000 tokens at 4 chars/token => ~4000 chars.
    assert DEFAULT_MAX_CHARS == 4000
