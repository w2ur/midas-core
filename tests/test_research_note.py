"""Tests for engine.research_note."""

import pytest

from engine.research_note import (
    ACTION_BIAS_VALUES,
    HORIZON_VALUES,
    ResearchNote,
    parse_research_note,
    render_research_note,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_note(**overrides) -> ResearchNote:
    """Return a valid ResearchNote, with optional field overrides."""
    defaults = dict(
        thesis="BTC is entering the markup phase of the halving cycle.",
        conviction=7,
        tickers=["BTC-EUR", "ETH-EUR"],
        action_bias="buy",
        horizon="weeks",
        catalysts="Spot ETF inflows accelerating; on-chain LTH accumulation.",
        currency="EUR",
    )
    defaults.update(overrides)
    return ResearchNote(**defaults)


def _valid_dict(**overrides) -> dict:
    """Return a valid raw dict suitable for parse_research_note."""
    defaults = dict(
        thesis="BTC is entering the markup phase of the halving cycle.",
        conviction=7,
        tickers=["BTC-EUR", "ETH-EUR"],
        action_bias="buy",
        horizon="weeks",
        catalysts="Spot ETF inflows accelerating; on-chain LTH accumulation.",
        currency="EUR",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_action_bias_values_is_frozenset(self) -> None:
        assert isinstance(ACTION_BIAS_VALUES, frozenset)

    def test_horizon_values_is_frozenset(self) -> None:
        assert isinstance(HORIZON_VALUES, frozenset)

    def test_action_bias_contains_expected_values(self) -> None:
        assert ACTION_BIAS_VALUES == frozenset(
            {"strong_buy", "buy", "hold", "reduce", "exit"}
        )

    def test_horizon_contains_expected_values(self) -> None:
        assert HORIZON_VALUES == frozenset({"days", "weeks", "months"})


# ---------------------------------------------------------------------------
# ResearchNote construction — valid
# ---------------------------------------------------------------------------


class TestResearchNoteValidConstruction:
    def test_basic_construction(self) -> None:
        note = _valid_note()
        assert note.thesis == "BTC is entering the markup phase of the halving cycle."
        assert note.conviction == 7
        assert note.tickers == ["BTC-EUR", "ETH-EUR"]
        assert note.action_bias == "buy"
        assert note.horizon == "weeks"
        assert note.currency == "EUR"

    def test_all_action_bias_values_accepted(self) -> None:
        for bias in ACTION_BIAS_VALUES:
            note = _valid_note(action_bias=bias)
            assert note.action_bias == bias

    def test_all_horizon_values_accepted(self) -> None:
        for h in HORIZON_VALUES:
            note = _valid_note(horizon=h)
            assert note.horizon == h

    def test_conviction_boundary_zero(self) -> None:
        note = _valid_note(conviction=0)
        assert note.conviction == 0

    def test_conviction_boundary_ten(self) -> None:
        note = _valid_note(conviction=10)
        assert note.conviction == 10

    def test_usd_currency_accepted(self) -> None:
        note = _valid_note(currency="USD")
        assert note.currency == "USD"

    def test_thesis_at_280_chars_accepted(self) -> None:
        thesis = "x" * 280
        note = _valid_note(thesis=thesis)
        assert len(note.thesis) == 280

    def test_catalysts_at_200_chars_accepted(self) -> None:
        catalysts = "y" * 200
        note = _valid_note(catalysts=catalysts)
        assert len(note.catalysts) == 200


# ---------------------------------------------------------------------------
# ResearchNote construction — validation failures
# ---------------------------------------------------------------------------


class TestResearchNoteValidationFailures:
    def test_invalid_action_bias_raises(self) -> None:
        with pytest.raises(ValueError, match="action_bias"):
            _valid_note(action_bias="panic_sell")

    def test_conviction_above_10_raises(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            _valid_note(conviction=11)

    def test_conviction_below_0_raises(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            _valid_note(conviction=-1)

    def test_thesis_over_280_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="thesis"):
            _valid_note(thesis="x" * 281)

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            _valid_note(horizon="years")

    def test_invalid_currency_raises(self) -> None:
        with pytest.raises(ValueError, match="currency"):
            _valid_note(currency="GBP")

    def test_catalysts_over_200_chars_raises(self) -> None:
        with pytest.raises(ValueError, match="catalysts"):
            _valid_note(catalysts="z" * 201)

    def test_empty_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="tickers"):
            _valid_note(tickers=[])

    def test_empty_thesis_raises(self) -> None:
        with pytest.raises(ValueError, match="thesis"):
            _valid_note(thesis="")


# ---------------------------------------------------------------------------
# parse_research_note — tolerant constructor
# ---------------------------------------------------------------------------


class TestParseResearchNote:
    def test_none_returns_none(self) -> None:
        assert parse_research_note(None) is None

    def test_empty_dict_returns_none(self) -> None:
        assert parse_research_note({}) is None

    def test_valid_dict_returns_note(self) -> None:
        note = parse_research_note(_valid_dict())
        assert note is not None
        assert note.conviction == 7
        assert note.action_bias == "buy"

    def test_bad_action_bias_returns_none_no_raise(self) -> None:
        raw = _valid_dict(action_bias="aggressive_long")
        result = parse_research_note(raw)
        assert result is None

    def test_bad_horizon_returns_none_no_raise(self) -> None:
        raw = _valid_dict(horizon="century")
        result = parse_research_note(raw)
        assert result is None

    def test_bad_currency_returns_none_no_raise(self) -> None:
        raw = _valid_dict(currency="JPY")
        result = parse_research_note(raw)
        assert result is None

    def test_missing_thesis_returns_none(self) -> None:
        raw = _valid_dict()
        del raw["thesis"]
        result = parse_research_note(raw)
        assert result is None

    def test_missing_tickers_returns_none(self) -> None:
        raw = _valid_dict()
        del raw["tickers"]
        result = parse_research_note(raw)
        assert result is None

    def test_over_length_thesis_is_truncated(self) -> None:
        raw = _valid_dict(thesis="x" * 300)
        note = parse_research_note(raw)
        assert note is not None
        assert len(note.thesis) == 280

    def test_over_length_catalysts_is_truncated(self) -> None:
        raw = _valid_dict(catalysts="y" * 250)
        note = parse_research_note(raw)
        assert note is not None
        assert len(note.catalysts) == 200

    def test_conviction_above_10_is_clamped(self) -> None:
        raw = _valid_dict(conviction=15)
        note = parse_research_note(raw)
        assert note is not None
        assert note.conviction == 10

    def test_conviction_below_0_is_clamped(self) -> None:
        raw = _valid_dict(conviction=-3)
        note = parse_research_note(raw)
        assert note is not None
        assert note.conviction == 0

    def test_no_raise_on_malformed_note(self) -> None:
        raw = _valid_dict(action_bias="nonsense", horizon="nonsense")
        result = parse_research_note(raw)
        assert result is None

    def test_empty_tickers_returns_none(self) -> None:
        raw = _valid_dict(tickers=[])
        result = parse_research_note(raw)
        assert result is None

    def test_empty_thesis_returns_none(self) -> None:
        raw = _valid_dict(thesis="")
        result = parse_research_note(raw)
        assert result is None


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundtrip:
    def test_to_dict_from_dict_roundtrip(self) -> None:
        note = _valid_note()
        d = note.to_dict()
        reconstructed = ResearchNote.from_dict(d)
        assert reconstructed == note

    def test_to_dict_contains_expected_keys(self) -> None:
        note = _valid_note()
        d = note.to_dict()
        assert set(d.keys()) == {
            "thesis",
            "conviction",
            "tickers",
            "action_bias",
            "horizon",
            "catalysts",
            "currency",
        }

    def test_from_dict_parses_correctly(self) -> None:
        d = {
            "thesis": "Gold is a safe haven.",
            "conviction": 5,
            "tickers": ["GLD"],
            "action_bias": "hold",
            "horizon": "months",
            "catalysts": "Inflation rising.",
            "currency": "USD",
        }
        note = ResearchNote.from_dict(d)
        assert note.thesis == "Gold is a safe haven."
        assert note.conviction == 5
        assert note.currency == "USD"


# ---------------------------------------------------------------------------
# render_research_note
# ---------------------------------------------------------------------------


class TestRenderResearchNote:
    def test_render_contains_thesis(self) -> None:
        note = _valid_note()
        rendered = render_research_note(note)
        assert "markup phase" in rendered

    def test_render_contains_conviction(self) -> None:
        note = _valid_note(conviction=9)
        rendered = render_research_note(note)
        assert "9" in rendered

    def test_render_contains_action_bias(self) -> None:
        note = _valid_note(action_bias="strong_buy")
        rendered = render_research_note(note)
        assert "strong_buy" in rendered

    def test_render_contains_tickers(self) -> None:
        note = _valid_note(tickers=["BTC-EUR", "ETH-EUR"])
        rendered = render_research_note(note)
        assert "BTC-EUR" in rendered

    def test_render_contains_horizon(self) -> None:
        note = _valid_note(horizon="months")
        rendered = render_research_note(note)
        assert "months" in rendered

    def test_render_is_string(self) -> None:
        note = _valid_note()
        assert isinstance(render_research_note(note), str)


# ---------------------------------------------------------------------------
# Regression tests — contract violations caught by code review
# ---------------------------------------------------------------------------


class TestParseResearchNoteContractViolations:
    """parse_research_note must NEVER raise on any JSON-decodable input."""

    # --- Non-dict raw input ---

    def test_string_input_returns_none_no_raise(self) -> None:
        # Regression: non-dict truthy input hit .get() and raised AttributeError.
        result = parse_research_note("skipped")  # type: ignore[arg-type]
        assert result is None

    def test_list_input_returns_none_no_raise(self) -> None:
        result = parse_research_note([1, 2, 3])  # type: ignore[arg-type]
        assert result is None

    def test_int_input_returns_none_no_raise(self) -> None:
        result = parse_research_note(42)  # type: ignore[arg-type]
        assert result is None

    def test_bool_input_returns_none_no_raise(self) -> None:
        # True is truthy and an int subclass — must not reach .get().
        result = parse_research_note(True)  # type: ignore[arg-type]
        assert result is None

    # --- Non-string thesis / catalysts ---

    def test_int_thesis_does_not_raise(self) -> None:
        # Regression: int thesis hit len() before the try block → TypeError.
        raw = _valid_dict(thesis=999)
        result = parse_research_note(raw)
        # Acceptable outcomes: None (meaningless numeric thesis) or a coerced valid note.
        # Must not raise.
        assert result is None or isinstance(result, ResearchNote)  # noqa: E501

    def test_int_catalysts_does_not_raise(self) -> None:
        # Regression: int catalysts hit len() before the try block → TypeError.
        raw = _valid_dict(catalysts=42)
        result = parse_research_note(raw)
        assert result is None or isinstance(result, ResearchNote)

    # --- Bare-string tickers ---

    def test_bare_string_ticker_is_not_exploded(self) -> None:
        # Regression: list("AAPL") → ["A","A","P","L"] — silent corruption.
        raw = _valid_dict(tickers="AAPL")
        result = parse_research_note(raw)
        # Must be either ["AAPL"] (single-element) or None — NOT a 4-element char list.
        if result is not None:
            assert result.tickers != list("AAPL"), (
                "bare string ticker was character-exploded into a list"
            )
            assert result.tickers == ["AAPL"]

    # --- bool conviction in __post_init__ ---

    def test_conviction_true_raises_value_error(self) -> None:
        # Regression: isinstance(True, int) → True, so True was accepted as conviction=1.
        with pytest.raises(ValueError, match="conviction"):
            _valid_note(conviction=True)  # type: ignore[arg-type]

    def test_conviction_false_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="conviction"):
            _valid_note(conviction=False)  # type: ignore[arg-type]
