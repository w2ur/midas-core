"""Roles are resolved from the roster, never hardcoded to the live cast.

Two defects of the same shape, both invisible on the live desk because the
live desk's names happen to match the constants:

- `step_build_memory_update_prompts` named `the-oracle` directly, so a fork
  whose narrator is called anything else fell through to the *trader* journal
  template. That template's three fact slots are all structurally empty for an
  agent that holds no book, which is exactly the input that produced the
  fabricated blank streak recorded at METHODOLOGY `#oracle-fabrication`.
- `engine/baseline_manager` hardcoded the control book's id, capital, position
  size and the literal currency "EUR". Two allocators on one desk would share
  a single control book, and a non-EUR desk would carry an uncontrolled FX leg
  in the very comparison the book exists to remove.

The live-desk assertions here are also a regression guard in the other
direction: the control book's published history must not move, so a roster
edit that changed its id or size would fail this file rather than silently
restating a book.
"""

from __future__ import annotations

import textwrap

import pytest

from engine.config import MidasConfig, _load, get_config


LIVE_ROSTER = """
globals:
  day_one: '2026-04-17'
  currencies: [EUR, USD]
  initial_capital: 10000.0
  global_reference:
    label: MSCI World
    ticker: URTH
    currency: EUR
agents:
  trader-one:
    display_name: Trader One
    voice: flat
    post_time: "20:00"
    home_currency: EUR
    initial_capital: 10000.0
    max_positions: 5
    universe: sp500
    persona: trader-one.md
  scribe:
    display_name: The Scribe
    voice: wry
    post_time: ''
    home_currency: EUR
    initial_capital: 0.0
    max_positions: 0
    universe: null
    persona: scribe.md
    role: narrator
  chief:
    display_name: The Chief
    voice: dry
    post_time: ''
    home_currency: USD
    initial_capital: 5000.0
    max_positions: 4
    universe: null
    persona: chief.md
    role: allocator
    allocator:
      channels_prefix: chief
      risk_budget:
        max_positions: 4
      baseline:
        enabled: true
"""


def _config_from(yaml_text: str, tmp_path) -> MidasConfig:
    """Build a config from a throwaway roster, without touching the live desk."""
    root = tmp_path / yaml_text.__hash__().__abs__().__str__()[:8]
    root.mkdir(parents=True, exist_ok=True)
    (root / "roster.yaml").write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return _load(root)


class TestNarratorResolution:
    @pytest.mark.live_cast
    def test_the_live_desk_narrator_is_the_oracle(self):
        """The regression this replaces: the id used to be a literal."""
        assert get_config().narrators == ("the-oracle",)

    def test_a_fork_narrator_resolves_under_its_own_name(self, tmp_path):
        cfg = _config_from(LIVE_ROSTER, tmp_path)
        assert cfg.narrators == ("scribe",)
        # The control that makes the assertion above mean something: the
        # hardcoded id is NOT on this desk, so the old code would have built a
        # prompt for an agent that does not exist.
        assert "the-oracle" not in cfg.roster

    def test_a_desk_with_no_narrator_resolves_empty_rather_than_raising(self, tmp_path):
        """The demo desk declares none; iterating must be a no-op, not a crash."""
        cfg = _config_from(LIVE_ROSTER.replace("    role: narrator\n", ""), tmp_path)
        assert cfg.narrators == ()

    def test_narrators_are_not_traders(self, tmp_path):
        cfg = _config_from(LIVE_ROSTER, tmp_path)
        assert "scribe" not in cfg.trading_roster
        assert cfg.trading_roster == ("trader-one",)


class TestBaselineResolution:
    @pytest.mark.live_cast
    def test_the_live_control_book_is_unchanged(self):
        """Pinned values, because this book's published history must not move.

        `strategy_id` and `position_size` are set explicitly in roster.yaml.
        Left to derivation they would be `the-manager-baseline` (orphaning
        data/portfolios/baseline-manager/) and 2000/6 = 333.33.
        """
        b = get_config().baseline_params("the-manager")
        assert b.strategy_id == "baseline-manager"
        assert b.initial_capital == 2000.0
        assert b.position_size == 300.0
        assert b.max_positions == 6
        assert b.currency == "EUR"

    def test_capital_currency_and_positions_are_derived_not_pinned(self, tmp_path):
        """A fork setting nothing gets its OWN allocator's numbers.

        `chief` is a USD book of 5,000 with a 4-position budget, and declares
        `baseline: {enabled: true}` and nothing else.
        """
        b = _config_from(LIVE_ROSTER, tmp_path).baseline_params("chief")
        assert b.currency == "USD", "the FX leg this fix exists to remove"
        assert b.initial_capital == 5000.0
        assert b.max_positions == 4
        assert b.position_size == pytest.approx(1250.0)  # 5000 / 4

    def test_each_allocator_gets_its_own_book(self, tmp_path):
        """Two allocators must not share one control book."""
        second = LIVE_ROSTER + (
            "  deputy:\n"
            "    display_name: The Deputy\n"
            "    voice: terse\n"
            "    post_time: ''\n"
            "    home_currency: EUR\n"
            "    initial_capital: 3000.0\n"
            "    max_positions: 3\n"
            "    universe: null\n"
            "    persona: deputy.md\n"
            "    role: allocator\n"
            "    allocator:\n"
            "      channels_prefix: deputy\n"
            "      risk_budget:\n"
            "        max_positions: 3\n"
            "      baseline:\n"
            "        enabled: true\n"
        )
        cfg = _config_from(second, tmp_path)
        assert set(cfg.allocators) == {"chief", "deputy"}
        ids = {cfg.baseline_params(a).strategy_id for a in cfg.allocators}
        assert ids == {"chief-baseline", "deputy-baseline"}, (
            "each allocator needs its own control book; a shared id would let "
            "one allocator's rebalance overwrite the other's holdings"
        )

    def test_an_explicit_pin_wins_over_derivation(self, tmp_path):
        pinned = LIVE_ROSTER.replace(
            "      baseline:\n        enabled: true\n",
            "      baseline:\n"
            "        enabled: true\n"
            "        strategy_id: legacy-book\n"
            "        position_size: 42.0\n"
            "        max_positions: 2\n"
            "        initial_capital: 99.0\n",
        )
        b = _config_from(pinned, tmp_path).baseline_params("chief")
        assert (b.strategy_id, b.position_size, b.max_positions) == (
            "legacy-book",
            42.0,
            2,
        )
        assert b.initial_capital == 99.0
        # Currency is never pinnable: it is the allocator's own, so the two
        # cannot drift apart.
        assert b.currency == "USD"

    def test_baseline_enabled_still_reads_through_the_alias(self, tmp_path):
        cfg = _config_from(LIVE_ROSTER, tmp_path)
        assert cfg.allocator_spec("chief").baseline_enabled is True
        off = _config_from(
            LIVE_ROSTER.replace("enabled: true", "enabled: false"), tmp_path
        )
        assert off.allocator_spec("chief").baseline_enabled is False


class TestTheNarratorGetsTheNarratorTemplate:
    """The defect itself, not just the config that feeds it.

    `build_memory_update_prompt` (trader) and `build_narrator_memory_update_prompt`
    produce visibly different documents. The trader template renders a
    PORTFOLIO VALUE line, which is the tell: an agent that holds no book gets
    `0.00`, and that is what the Oracle was reading when it narrated a dark
    desk across sessions that placed 1-27 orders each.
    """

    def _build(self, root, monkeypatch):
        from engine.config import reset_config_cache
        import scripts.daily_session as ds

        monkeypatch.setenv("MIDAS_DATA_DIR", str(root))
        reset_config_cache()
        try:
            return ds.step_build_memory_update_prompts(
                agent_results={"trader-one": {"trades": [], "commentary": "x"}},
                agent_posts={},
                portfolio_summaries={"trader-one": {}},
                day_number=42,
                leaderboard=[
                    {"rank": 1, "agent": "trader-one", "return_pct": 1.0}
                ],
                oracle_posts=[{"text": "the desk traded today", "tag": "note"}],
            )
        finally:
            reset_config_cache()

    def test_a_fork_narrator_gets_the_narrator_prompt(self, tmp_path, monkeypatch):
        root = tmp_path / "fork"
        root.mkdir()
        (root / "roster.yaml").write_text(textwrap.dedent(LIVE_ROSTER), encoding="utf-8")
        prompts = self._build(root, monkeypatch)

        assert "scribe" in prompts, (
            "the narrator got no prompt at all; before this fix the loop wrote "
            "prompts['the-oracle'] on a desk with no such agent"
        )
        assert "the-oracle" not in prompts
        # The narrator template carries the session's facts and no book.
        assert "PORTFOLIO VALUE" not in prompts["scribe"]
        # The control: the trader on the same desk DOES get that template, so
        # the assertion above is discriminating rather than vacuous.
        assert "PORTFOLIO VALUE" in prompts["trader-one"]

    def test_a_desk_with_no_narrator_builds_only_trader_prompts(
        self, tmp_path, monkeypatch
    ):
        root = tmp_path / "silent"
        root.mkdir()
        (root / "roster.yaml").write_text(
            textwrap.dedent(LIVE_ROSTER).replace("    role: narrator\n", ""),
            encoding="utf-8",
        )
        prompts = self._build(root, monkeypatch)
        assert set(prompts) == {"trader-one"}
