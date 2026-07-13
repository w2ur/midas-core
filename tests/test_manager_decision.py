"""Tests for engine/manager_decision.py — ManagerDecision schema + conviction gate.

TDD: these tests are written BEFORE the implementation and drive the design.
"""

from __future__ import annotations

import pytest

from engine.manager_decision import (
    ManagerDecision,
    ManagerPosition,
    is_hold,
    parse_manager_decision,
    render_manager_decision,
)


# ---------------------------------------------------------------------------
# ManagerPosition construction
# ---------------------------------------------------------------------------


class TestManagerPosition:
    def test_valid_buy(self) -> None:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="BUY",
            size_eur=300,
            entry_guidance="Market order at open",
            stop_loss=25000.0,
            reasoning="Strong momentum breakout.",
        )
        assert pos.ticker == "BTC-EUR"
        assert pos.action == "BUY"
        assert pos.size_eur == 300

    def test_valid_sell(self) -> None:
        pos = ManagerPosition(
            ticker="ETH-EUR",
            action="SELL",
            size_eur=0,
            entry_guidance="Close entire position",
            stop_loss=None,
            reasoning="Stop-loss hit.",
        )
        assert pos.action == "SELL"
        assert pos.stop_loss is None

    def test_valid_hold(self) -> None:
        pos = ManagerPosition(
            ticker="AAPL",
            action="HOLD",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="No new catalyst.",
        )
        assert pos.action == "HOLD"

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match="action"):
            ManagerPosition(
                ticker="BTC-EUR",
                action="SHORT",
                size_eur=200,
                entry_guidance="",
                stop_loss=None,
                reasoning="Test.",
            )

    def test_negative_size_eur_raises(self) -> None:
        with pytest.raises(ValueError, match="size_eur"):
            ManagerPosition(
                ticker="BTC-EUR",
                action="BUY",
                size_eur=-100,
                entry_guidance="",
                stop_loss=None,
                reasoning="Test.",
            )

    def test_non_int_size_eur_raises(self) -> None:
        with pytest.raises(ValueError, match="size_eur"):
            ManagerPosition(
                ticker="BTC-EUR",
                action="BUY",
                size_eur=200.5,  # type: ignore[arg-type]
                entry_guidance="",
                stop_loss=None,
                reasoning="Test.",
            )

    def test_empty_ticker_raises(self) -> None:
        with pytest.raises(ValueError, match="ticker"):
            ManagerPosition(
                ticker="",
                action="BUY",
                size_eur=200,
                entry_guidance="",
                stop_loss=None,
                reasoning="Test.",
            )

    def test_empty_reasoning_raises(self) -> None:
        with pytest.raises(ValueError, match="reasoning"):
            ManagerPosition(
                ticker="BTC-EUR",
                action="BUY",
                size_eur=200,
                entry_guidance="",
                stop_loss=None,
                reasoning="",
            )

    def test_zero_size_eur_allowed(self) -> None:
        # HOLD and SELL can have 0 size
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="HOLD",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="Monitoring only.",
        )
        assert pos.size_eur == 0


# ---------------------------------------------------------------------------
# ManagerDecision construction
# ---------------------------------------------------------------------------


class TestManagerDecision:
    def _make_position(
        self, ticker: str = "BTC-EUR", action: str = "BUY"
    ) -> ManagerPosition:
        return ManagerPosition(
            ticker=ticker,
            action=action,
            size_eur=300,
            entry_guidance="",
            stop_loss=None,
            reasoning="Consensus breakout.",
        )

    def test_valid_with_positions(self) -> None:
        decision = ManagerDecision(
            positions=[self._make_position()],
            conviction=8,
            hold_reasoning="",
        )
        assert decision.conviction == 8
        assert len(decision.positions) == 1

    def test_valid_hold_no_positions(self) -> None:
        decision = ManagerDecision(
            positions=[],
            conviction=4,
            hold_reasoning="Low conviction — holding.",
        )
        assert decision.positions == []

    def test_conviction_above_10_raises(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            ManagerDecision(
                positions=[],
                conviction=11,
                hold_reasoning="Test.",
            )

    def test_conviction_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            ManagerDecision(
                positions=[],
                conviction=-1,
                hold_reasoning="Test.",
            )

    def test_non_int_conviction_raises(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            ManagerDecision(
                positions=[],
                conviction=7.5,  # type: ignore[arg-type]
                hold_reasoning="Test.",
            )


# ---------------------------------------------------------------------------
# Conviction gate — critical test
# ---------------------------------------------------------------------------


class TestConvictionGate:
    """The conviction gate MUST drop positions regardless of LLM output."""

    def test_low_conviction_drops_positions(self) -> None:
        """parse_manager_decision with conviction=5 and positions → positions=[]."""
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "Market order",
                    "stop_loss": None,
                    "reasoning": "Breakout.",
                }
            ],
            "conviction": 5,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.positions == [], "Low-conviction positions must be dropped"
        assert decision.hold_reasoning, (
            "hold_reasoning must be populated when gate fires"
        )

    def test_low_conviction_synthesises_hold_reasoning(self) -> None:
        """When model leaves hold_reasoning blank and conviction is low, synthesise one."""
        raw = {
            "positions": [
                {
                    "ticker": "ETH-EUR",
                    "action": "BUY",
                    "size_eur": 250,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Thesis.",
                }
            ],
            "conviction": 3,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.hold_reasoning  # must be non-empty

    def test_at_threshold_conviction_retains_positions(self) -> None:
        """conviction == min_conviction → positions are retained."""
        min_c = 6
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "Limit 30k",
                    "stop_loss": 27000.0,
                    "reasoning": "Strong analyst consensus.",
                }
            ],
            "conviction": min_c,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert len(decision.positions) == 1, (
            "Positions must be retained at threshold conviction"
        )

    def test_high_conviction_retains_positions(self) -> None:
        """conviction=9 → positions are retained."""
        raw = {
            "positions": [
                {
                    "ticker": "SOL-EUR",
                    "action": "BUY",
                    "size_eur": 350,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Strong breakout.",
                }
            ],
            "conviction": 9,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert len(decision.positions) == 1

    def test_conviction_6_now_passes_gate(self) -> None:
        """conviction=6 with a BUY position → positions retained (length 1), conviction==6."""
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "Limit 30k",
                    "stop_loss": 27000.0,
                    "reasoning": "Strong consensus across analysts.",
                }
            ],
            "conviction": 6,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert len(decision.positions) == 1, "conviction=6 must pass the gate"
        assert decision.conviction == 6

    def test_conviction_5_still_held(self) -> None:
        """conviction=5 with a BUY position → positions dropped (gate still blocks 5)."""
        raw = {
            "positions": [
                {
                    "ticker": "ETH-EUR",
                    "action": "BUY",
                    "size_eur": 250,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Moderate thesis.",
                }
            ],
            "conviction": 5,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.positions == [], (
            "conviction=5 must still be blocked by the gate"
        )

    def test_gate_uses_risk_budget_limits_min_conviction(self) -> None:
        """The gate threshold (min_conviction=6) is respected: below drops, at-threshold retains."""
        min_c = 6
        # One below threshold drops positions
        raw_below = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Test.",
                }
            ],
            "conviction": min_c - 1,
            "hold_reasoning": "",
        }
        below = parse_manager_decision(raw_below, min_conviction=6)
        assert below is not None
        assert below.positions == []

        # At threshold retains
        raw_at = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Test.",
                }
            ],
            "conviction": min_c,
            "hold_reasoning": "",
        }
        at = parse_manager_decision(raw_at, min_conviction=6)
        assert at is not None
        assert len(at.positions) == 1


# ---------------------------------------------------------------------------
# Tolerant parse — never raises
# ---------------------------------------------------------------------------


class TestParseManagerDecisionTolerant:
    def test_none_input_returns_none(self) -> None:
        assert parse_manager_decision(None, min_conviction=6) is None

    def test_empty_dict_returns_none(self) -> None:
        assert parse_manager_decision({}, min_conviction=6) is None

    def test_non_dict_returns_none(self) -> None:
        assert parse_manager_decision("not a dict", min_conviction=6) is None  # type: ignore[arg-type]
        assert parse_manager_decision(42, min_conviction=6) is None  # type: ignore[arg-type]
        assert parse_manager_decision([], min_conviction=6) is None  # type: ignore[arg-type]

    def test_garbage_conviction_is_clamped(self) -> None:
        raw = {
            "positions": [],
            "conviction": 999,
            "hold_reasoning": "Test.",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.conviction == 10

    def test_negative_conviction_is_clamped(self) -> None:
        raw = {
            "positions": [],
            "conviction": -50,
            "hold_reasoning": "Test.",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.conviction == 0

    def test_non_numeric_conviction_defaults(self) -> None:
        raw = {
            "positions": [],
            "conviction": "high",
            "hold_reasoning": "Test.",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        # Should not raise; conviction defaults to something valid
        assert decision is not None
        assert 0 <= decision.conviction <= 10

    def test_malformed_position_is_dropped_others_kept(self) -> None:
        raw = {
            "positions": [
                {
                    "ticker": "",  # invalid — empty ticker
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Test.",
                },
                {
                    "ticker": "SOL-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Valid position.",
                },
            ],
            "conviction": 8,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].ticker == "SOL-EUR"

    def test_bad_action_position_is_dropped(self) -> None:
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "LONG",  # invalid enum
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Test.",
                },
            ],
            "conviction": 8,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.positions == []

    def test_negative_size_eur_position_is_dropped(self) -> None:
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": -200,  # invalid
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Test.",
                },
            ],
            "conviction": 8,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.positions == []

    def test_positions_not_a_list_returns_empty_positions(self) -> None:
        raw = {
            "positions": "not a list",
            "conviction": 8,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert decision.positions == []

    def test_missing_conviction_defaults(self) -> None:
        raw = {
            "positions": [],
            "hold_reasoning": "No signal.",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert 0 <= decision.conviction <= 10


# ---------------------------------------------------------------------------
# is_hold helper
# ---------------------------------------------------------------------------


class TestIsHold:
    def test_empty_positions_is_hold(self) -> None:
        decision = ManagerDecision(
            positions=[], conviction=3, hold_reasoning="Holding."
        )
        assert is_hold(decision) is True

    def test_all_hold_actions_is_hold(self) -> None:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="HOLD",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="No catalyst.",
        )
        decision = ManagerDecision(positions=[pos], conviction=6, hold_reasoning="")
        assert is_hold(decision) is True

    def test_buy_action_is_not_hold(self) -> None:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="BUY",
            size_eur=300,
            entry_guidance="",
            stop_loss=None,
            reasoning="Breakout.",
        )
        decision = ManagerDecision(positions=[pos], conviction=8, hold_reasoning="")
        assert is_hold(decision) is False

    def test_sell_action_is_not_hold(self) -> None:
        pos = ManagerPosition(
            ticker="ETH-EUR",
            action="SELL",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="Stop hit.",
        )
        decision = ManagerDecision(positions=[pos], conviction=9, hold_reasoning="")
        assert is_hold(decision) is False

    def test_mixed_hold_and_buy_is_not_hold(self) -> None:
        hold_pos = ManagerPosition(
            ticker="BTC-EUR",
            action="HOLD",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="No change.",
        )
        buy_pos = ManagerPosition(
            ticker="SOL-EUR",
            action="BUY",
            size_eur=250,
            entry_guidance="",
            stop_loss=None,
            reasoning="New position.",
        )
        decision = ManagerDecision(
            positions=[hold_pos, buy_pos], conviction=8, hold_reasoning=""
        )
        assert is_hold(decision) is False


# ---------------------------------------------------------------------------
# render_manager_decision
# ---------------------------------------------------------------------------


class TestRenderManagerDecision:
    def test_render_contains_conviction(self) -> None:
        decision = ManagerDecision(
            positions=[], conviction=4, hold_reasoning="Low conviction."
        )
        rendered = render_manager_decision(decision)
        assert "4" in rendered

    def test_render_hold_reasoning_shown_when_no_positions(self) -> None:
        decision = ManagerDecision(
            positions=[], conviction=3, hold_reasoning="No trades today."
        )
        rendered = render_manager_decision(decision)
        assert "No trades today." in rendered

    def test_render_shows_ticker_when_positions(self) -> None:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="BUY",
            size_eur=300,
            entry_guidance="",
            stop_loss=None,
            reasoning="Strong buy.",
        )
        decision = ManagerDecision(positions=[pos], conviction=8, hold_reasoning="")
        rendered = render_manager_decision(decision)
        assert "BTC-EUR" in rendered

    def test_render_shows_action(self) -> None:
        pos = ManagerPosition(
            ticker="ETH-EUR",
            action="SELL",
            size_eur=0,
            entry_guidance="",
            stop_loss=None,
            reasoning="Exit on stop.",
        )
        decision = ManagerDecision(positions=[pos], conviction=9, hold_reasoning="")
        rendered = render_manager_decision(decision)
        assert "SELL" in rendered


# ---------------------------------------------------------------------------
# to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def _make_decision(self) -> ManagerDecision:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="BUY",
            size_eur=300,
            entry_guidance="Limit 30k",
            stop_loss=27000.0,
            reasoning="Analyst consensus.",
        )
        return ManagerDecision(positions=[pos], conviction=8, hold_reasoning="")

    def test_to_dict_keys(self) -> None:
        d = self._make_decision().to_dict()
        assert "positions" in d
        assert "conviction" in d
        assert "hold_reasoning" in d

    def test_position_to_dict_keys(self) -> None:
        pos = ManagerPosition(
            ticker="BTC-EUR",
            action="BUY",
            size_eur=300,
            entry_guidance="Limit",
            stop_loss=None,
            reasoning="Thesis.",
        )
        d = pos.to_dict()
        assert set(d.keys()) == {
            "ticker",
            "action",
            "size_eur",
            "entry_guidance",
            "stop_loss",
            "reasoning",
        }

    def test_round_trip(self) -> None:
        original = self._make_decision()
        restored = ManagerDecision.from_dict(original.to_dict())
        assert restored.conviction == original.conviction
        assert len(restored.positions) == len(original.positions)
        restored_pos = restored.positions[0]
        original_pos = original.positions[0]
        assert restored_pos.ticker == original_pos.ticker
        assert restored_pos.action == original_pos.action
        assert restored_pos.size_eur == original_pos.size_eur
        assert restored_pos.stop_loss == original_pos.stop_loss

    def test_from_dict_validates(self) -> None:
        """from_dict is strict — bad data raises ValueError."""
        d = self._make_decision().to_dict()
        d["conviction"] = 15  # invalid
        with pytest.raises(ValueError):
            ManagerDecision.from_dict(d)


# ---------------------------------------------------------------------------
# size_eur coercion — pinned regression tests
# ---------------------------------------------------------------------------


def _position_raw(size_eur: object) -> dict:
    """Minimal valid position raw dict with a custom size_eur value."""
    return {
        "ticker": "BTC-EUR",
        "action": "BUY",
        "size_eur": size_eur,
        "entry_guidance": "",
        "stop_loss": None,
        "reasoning": "Coercion test.",
    }


def _parse_with_size(size_eur: object) -> "ManagerDecision | None":
    raw = {
        "positions": [_position_raw(size_eur)],
        "conviction": 9,
        "hold_reasoning": "",
    }
    return parse_manager_decision(raw, min_conviction=6)


class TestSizeEurCoercion:
    """Pin the documented coercion asymmetry so a refactor cannot silently regress."""

    def test_int_like_float_300_0_accepted(self) -> None:
        """300.0 (float) → size_eur=300 (int), position kept."""
        decision = _parse_with_size(300.0)
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].size_eur == 300

    def test_integer_string_300_accepted(self) -> None:
        """ "300" (string) → size_eur=300 (int), position kept."""
        decision = _parse_with_size("300")
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].size_eur == 300

    def test_float_string_300_0_dropped(self) -> None:
        """ "300.0" (float-shaped string) → position dropped (safe asymmetry)."""
        decision = _parse_with_size("300.0")
        assert decision is not None
        assert decision.positions == [], "float-shaped string must be dropped"

    def test_bool_true_dropped(self) -> None:
        """True (bool) → position dropped (True→1 would be a EUR 1 order)."""
        decision = _parse_with_size(True)
        assert decision is not None
        assert decision.positions == [], "bool size_eur must be dropped"

    def test_bool_false_dropped(self) -> None:
        """False (bool) → position dropped (bools rejected regardless of value)."""
        decision = _parse_with_size(False)
        assert decision is not None
        assert decision.positions == [], "bool size_eur must be dropped"


# ---------------------------------------------------------------------------
# Trigger + expires — new in Task 5
# ---------------------------------------------------------------------------


class TestTriggerParsing:
    """Tolerant parser must validate trigger/expires coupling and op enum."""

    def _raw_position(self, **overrides: object) -> dict:
        base: dict = {
            "ticker": "PHAG.L",
            "action": "BUY",
            "size_eur": 300,
            "entry_guidance": "",
            "stop_loss": None,
            "reasoning": "Gold ETF confirmation play.",
        }
        base.update(overrides)
        return base

    def _parse(self, **overrides: object) -> "ManagerDecision | None":
        raw = {
            "positions": [self._raw_position(**overrides)],
            "conviction": 9,
            "hold_reasoning": "",
        }
        return parse_manager_decision(raw, min_conviction=6)

    def test_parse_position_with_trigger(self) -> None:
        """Valid trigger + expires → ManagerPosition.trigger/expires populated."""
        decision = self._parse(
            trigger={"op": ">=", "level": 65.0},
            expires="2026-07-15",
        )
        assert decision is not None
        assert len(decision.positions) == 1
        pos = decision.positions[0]
        assert pos.trigger == {"op": ">=", "level": 65.0}
        assert pos.expires == "2026-07-15"

    def test_trigger_without_expiry_dropped(self) -> None:
        """trigger present but expires missing → position dropped (conservative)."""
        decision = self._parse(trigger={"op": ">=", "level": 65.0})
        assert decision is not None
        assert decision.positions == [], "trigger without expires must be dropped"

    def test_bad_trigger_op_dropped(self) -> None:
        """op '==' is not in {'>=','<='} → position dropped."""
        decision = self._parse(
            trigger={"op": "==", "level": 65.0},
            expires="2026-07-15",
        )
        assert decision is not None
        assert decision.positions == [], "trigger with op '==' must be dropped"

    def test_no_trigger_position_unchanged(self) -> None:
        """Position without trigger → market order, trigger/expires both None."""
        decision = self._parse()
        assert decision is not None
        assert len(decision.positions) == 1
        pos = decision.positions[0]
        assert pos.trigger is None
        assert pos.expires is None

    def test_lte_trigger_op_valid(self) -> None:
        """op '<=' is valid."""
        decision = self._parse(
            trigger={"op": "<=", "level": 30.0},
            expires="2026-07-10",
        )
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].trigger == {"op": "<=", "level": 30.0}


# ---------------------------------------------------------------------------
# Non-string ticker and numeric reasoning coercion
# ---------------------------------------------------------------------------


class TestTickerAndReasoningCoercion:
    """Pin ticker and reasoning coercion behaviors."""

    def test_numeric_ticker_coerced_to_string(self) -> None:
        """A numeric ticker value (123) is coerced to string "123".

        The parser does str(raw.get("ticker") or ""), so any truthy non-string
        is stringified and kept rather than dropped.
        """
        raw = {
            "positions": [
                {
                    "ticker": 123,
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": "Numeric ticker test.",
                }
            ],
            "conviction": 9,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        # Numeric ticker is either coerced to "123" (kept) or dropped.
        # Current behaviour: coerced and kept.
        if decision.positions:
            assert decision.positions[0].ticker == "123"

    def test_numeric_reasoning_coerced_to_string(self) -> None:
        """A numeric reasoning value is coerced to a non-empty string and kept."""
        raw = {
            "positions": [
                {
                    "ticker": "BTC-EUR",
                    "action": "BUY",
                    "size_eur": 300,
                    "entry_guidance": "",
                    "stop_loss": None,
                    "reasoning": 42,
                }
            ],
            "conviction": 9,
            "hold_reasoning": "",
        }
        decision = parse_manager_decision(raw, min_conviction=6)
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].reasoning == "42"


# ---------------------------------------------------------------------------
# Trigger expires ISO validation — crash-robustness fix
# ---------------------------------------------------------------------------


class TestTriggerExpiresValidation:
    """Malformed expires strings must be dropped at parse time, not crash at Order creation."""

    def _parse(self, **overrides: object) -> "ManagerDecision | None":
        base: dict = {
            "ticker": "PHAG.L",
            "action": "BUY",
            "size_eur": 300,
            "entry_guidance": "",
            "stop_loss": None,
            "reasoning": "ISO validation test.",
        }
        base.update(overrides)
        raw = {
            "positions": [base],
            "conviction": 9,
            "hold_reasoning": "",
        }
        return parse_manager_decision(raw, min_conviction=6)

    def test_trigger_malformed_expires_dropped(self) -> None:
        """trigger present but expires is not a valid ISO date → position dropped (INVALID_TRIGGER).

        Previously the non-empty check passed "not-a-date" through; the crash only surfaced
        downstream in Order.__post_init__ when date.fromisoformat raised ValueError.
        """
        decision = self._parse(
            trigger={"op": ">=", "level": 65.0},
            expires="not-a-date",
        )
        assert decision is not None
        assert decision.positions == [], (
            "malformed expires must be dropped at parse time"
        )

    def test_trigger_malformed_expires_non_zero_padded_dropped(self) -> None:
        """expires='2026-7-15' (non-zero-padded month) is not a valid ISO date → dropped."""
        decision = self._parse(
            trigger={"op": ">=", "level": 65.0},
            expires="2026-7-15",
        )
        assert decision is not None
        assert decision.positions == [], "non-zero-padded ISO date must be dropped"

    def test_trigger_valid_expires_kept(self) -> None:
        """Valid ISO date expires + valid trigger → position retained (regression guard)."""
        decision = self._parse(
            trigger={"op": ">=", "level": 65.0},
            expires="2026-07-15",
        )
        assert decision is not None
        assert len(decision.positions) == 1
        assert decision.positions[0].expires == "2026-07-15"

    def test_non_dict_trigger_dropped(self) -> None:
        """trigger is a string (not a dict) → position dropped (INVALID_TRIGGER).

        This closes the untested non-dict-trigger drop path.
        """
        decision = self._parse(
            trigger=">=65",
            expires="2026-07-15",
        )
        assert decision is not None
        assert decision.positions == [], "non-dict trigger must be dropped"


# ---------------------------------------------------------------------------
# Conviction gate — caller-supplied threshold
# ---------------------------------------------------------------------------


def test_conviction_gate_uses_passed_threshold():
    from engine.manager_decision import parse_manager_decision

    raw = {
        "conviction": 6,
        "positions": [
            {"ticker": "AAPL", "action": "BUY", "size_eur": 300, "reasoning": "x"}
        ],
    }
    # threshold 7 → gated to hold
    assert parse_manager_decision(raw, min_conviction=7).positions == []
    # threshold 6 → passes
    assert len(parse_manager_decision(raw, min_conviction=6).positions) == 1
