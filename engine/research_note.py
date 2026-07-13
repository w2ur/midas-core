"""Research note — structured analyst view emitted alongside paper trades.

Each trading agent emits one ResearchNote per session alongside their
unchanged trades/commentary/cancels output. The note captures the agent's
VIEW on the market (thesis, conviction, direction, horizon) but explicitly
NOT position sizing — sizing is the future Manager's responsibility.

Persistence contract
--------------------
parse_research_note is the tolerant entry point for agent output: it clamps
or truncates safe violations (over-length text, out-of-range conviction) and
returns None for unrecoverable issues (missing required fields, bad enums).
A bad note degrades to "no signal from this agent" — it never causes a
session failure.

The Manager and any future consumer should import ACTION_BIAS_VALUES and
HORIZON_VALUES as canonical sets (single source of truth).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical value sets (single source of truth)
# ---------------------------------------------------------------------------

ACTION_BIAS_VALUES: frozenset[str] = frozenset(
    {"strong_buy", "buy", "hold", "reduce", "exit"}
)

HORIZON_VALUES: frozenset[str] = frozenset({"days", "weeks", "months"})

_VALID_CURRENCIES: frozenset[str] = frozenset({"EUR", "USD"})

_THESIS_MAX_LEN: int = 280
_CATALYSTS_MAX_LEN: int = 200


# ---------------------------------------------------------------------------
# ResearchNote dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResearchNote:
    """Structured analyst view emitted by a trading agent.

    Fields
    ------
    thesis:
        The agent's core market view in ≤280 chars.
    conviction:
        Conviction score 0 (no conviction) to 10 (maximum).
    tickers:
        Tickers most relevant to this note (non-empty list).
    action_bias:
        Directional signal — one of ACTION_BIAS_VALUES.
    horizon:
        Expected time horizon — one of HORIZON_VALUES.
    catalysts:
        Key catalysts or risks in ≤200 chars.
    currency:
        Agent's base currency — "EUR" or "USD".

    Note: this captures VIEW only. Position sizing is the Manager's job.
    """

    thesis: str
    conviction: int
    tickers: list[str]
    action_bias: str
    horizon: str
    catalysts: str
    currency: str

    def __post_init__(self) -> None:
        if not self.thesis:
            raise ValueError("ResearchNote.thesis must be a non-empty string")
        if len(self.thesis) > _THESIS_MAX_LEN:
            raise ValueError(
                f"ResearchNote.thesis must be ≤{_THESIS_MAX_LEN} chars, "
                f"got {len(self.thesis)}"
            )
        if type(self.conviction) is not int or not (0 <= self.conviction <= 10):
            raise ValueError(
                f"ResearchNote.conviction must be an int 0-10, got {self.conviction!r}"
            )
        if not self.tickers:
            raise ValueError("ResearchNote.tickers must be a non-empty list")
        if self.action_bias not in ACTION_BIAS_VALUES:
            raise ValueError(
                f"ResearchNote.action_bias must be one of {sorted(ACTION_BIAS_VALUES)}, "
                f"got {self.action_bias!r}"
            )
        if self.horizon not in HORIZON_VALUES:
            raise ValueError(
                f"ResearchNote.horizon must be one of {sorted(HORIZON_VALUES)}, "
                f"got {self.horizon!r}"
            )
        if len(self.catalysts) > _CATALYSTS_MAX_LEN:
            raise ValueError(
                f"ResearchNote.catalysts must be ≤{_CATALYSTS_MAX_LEN} chars, "
                f"got {len(self.catalysts)}"
            )
        if self.currency not in _VALID_CURRENCIES:
            raise ValueError(
                f"ResearchNote.currency must be one of {sorted(_VALID_CURRENCIES)}, "
                f"got {self.currency!r}"
            )

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "thesis": self.thesis,
            "conviction": self.conviction,
            "tickers": list(self.tickers),
            "action_bias": self.action_bias,
            "horizon": self.horizon,
            "catalysts": self.catalysts,
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchNote":
        """Deserialize from a dictionary (inverse of to_dict). Validates strictly."""
        return cls(
            thesis=d["thesis"],
            conviction=int(d["conviction"]),
            tickers=list(d["tickers"]),
            action_bias=d["action_bias"],
            horizon=d["horizon"],
            catalysts=d["catalysts"],
            currency=d["currency"],
        )


# ---------------------------------------------------------------------------
# Tolerant constructor for agent output
# ---------------------------------------------------------------------------


def parse_research_note(raw: dict | None) -> ResearchNote | None:
    """Tolerant constructor for agent-emitted research note dicts.

    Contract
    --------
    - None or empty dict → None (agent emitted no note).
    - Missing required fields (thesis, tickers) → None + warning logged.
    - Bad enum (action_bias, horizon, currency) → None + warning logged.
    - Over-length thesis/catalysts → truncated to max length, note returned.
    - Out-of-range conviction → clamped to [0, 10], note returned.

    This method NEVER raises. A bad note degrades to "no signal from this
    agent" and never causes a session failure.
    """
    if not raw:
        return None

    if not isinstance(raw, dict):
        logger.warning(
            "parse_research_note: expected dict, got %s — skipping note",
            type(raw).__name__,
        )
        return None

    # --- Required field checks (unrecoverable) ---
    # Coerce to str so that numeric values don't raise on len() below.
    thesis = str(raw.get("thesis") or "")
    if not thesis:
        logger.warning("parse_research_note: missing or empty 'thesis' — skipping note")
        return None

    tickers = raw.get("tickers")
    if not tickers:
        logger.warning(
            "parse_research_note: missing or empty 'tickers' — skipping note"
        )
        return None

    # Guard bare-string tickers (single ticker passed as "AAPL" instead of ["AAPL"]).
    if isinstance(tickers, str):
        tickers = [tickers]
    elif not isinstance(tickers, list):
        logger.warning(
            "parse_research_note: 'tickers' must be a list or string, got %s — skipping note",
            type(tickers).__name__,
        )
        return None
    # Ensure all elements are strings; skip non-string elements with a warning.
    tickers = [t for t in tickers if isinstance(t, str)]
    if not tickers:
        logger.warning(
            "parse_research_note: 'tickers' list contained no valid string elements — skipping note"
        )
        return None

    action_bias = raw.get("action_bias")
    if action_bias not in ACTION_BIAS_VALUES:
        logger.warning(
            "parse_research_note: invalid action_bias %r (expected one of %s) — skipping note",
            action_bias,
            sorted(ACTION_BIAS_VALUES),
        )
        return None

    horizon = raw.get("horizon")
    if horizon not in HORIZON_VALUES:
        logger.warning(
            "parse_research_note: invalid horizon %r (expected one of %s) — skipping note",
            horizon,
            sorted(HORIZON_VALUES),
        )
        return None

    currency = raw.get("currency")
    if currency not in _VALID_CURRENCIES:
        logger.warning(
            "parse_research_note: invalid currency %r (expected one of %s) — skipping note",
            currency,
            sorted(_VALID_CURRENCIES),
        )
        return None

    # --- Safe coercions ---
    if len(thesis) > _THESIS_MAX_LEN:
        logger.warning(
            "parse_research_note: thesis length %d > %d — truncating",
            len(thesis),
            _THESIS_MAX_LEN,
        )
        thesis = thesis[:_THESIS_MAX_LEN]

    # Coerce to str so that numeric catalysts don't raise on len() below.
    catalysts = str(raw.get("catalysts") or "")
    if len(catalysts) > _CATALYSTS_MAX_LEN:
        logger.warning(
            "parse_research_note: catalysts length %d > %d — truncating",
            len(catalysts),
            _CATALYSTS_MAX_LEN,
        )
        catalysts = catalysts[:_CATALYSTS_MAX_LEN]

    try:
        # int() truncates floats intentionally (e.g. 7.9 → 7).
        conviction = int(raw.get("conviction", 5))
    except (TypeError, ValueError):
        conviction = 5
    conviction = max(0, min(10, conviction))

    try:
        return ResearchNote(
            thesis=thesis,
            conviction=conviction,
            tickers=list(tickers),
            action_bias=action_bias,
            horizon=horizon,
            catalysts=catalysts,
            currency=currency,
        )
    except (ValueError, TypeError) as exc:
        logger.warning(
            "parse_research_note: construction failed (%s) — skipping note", exc
        )
        return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_research_note(note: ResearchNote) -> str:
    """Compact human-readable block rendering of a ResearchNote.

    Intended for Oracle/site/manager-context consumption. Single-block output,
    no trailing newline.
    """
    tickers_str = ", ".join(note.tickers)
    return (
        f"[Research Note]\n"
        f"  thesis:     {note.thesis}\n"
        f"  conviction: {note.conviction}/10\n"
        f"  bias:       {note.action_bias}\n"
        f"  horizon:    {note.horizon}\n"
        f"  tickers:    {tickers_str}\n"
        f"  catalysts:  {note.catalysts}\n"
        f"  currency:   {note.currency}"
    )
