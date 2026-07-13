"""Lightweight token/cost visibility for persona dispatch.

We do not have real token accounting from the orchestrator's untracked LLM
dispatches, so this module provides a cheap, deterministic *proxy*: characters
divided by four, the widely-used rough tokens-per-char heuristic for English.
It is a visibility signal (how heavy was today's prompt load, and which agent
dominated it), not a billing figure.

The persona dispatch path (``engine.persona_dispatch.wrap_persona_prompt``) feeds
every wrapped prompt into the process-level session ledger. The daily output
bundle reads the accumulated totals into a ``session_costs`` block.
"""

from __future__ import annotations

_CHARS_PER_TOKEN = 4
PROXY_LABEL = "len/4"


def estimate_tokens(text: str | None) -> int:
    """Return the character-count token proxy: ``len(text) // 4``.

    Deterministic and network-free. ``None``/empty → 0.
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


class SessionCostLedger:
    """Accumulates per-dispatch prompt-size proxies across a session.

    Keyed by agent id. Each entry tracks the dispatch count, total prompt
    characters, and total estimated tokens (the ``len/4`` proxy). ``totals``
    returns a JSON-serializable block suitable for the output bundle.
    """

    def __init__(self) -> None:
        self._by_agent: dict[str, dict[str, int]] = {}

    def record(self, agent_id: str, prompt: str) -> int:
        """Record one dispatch for ``agent_id``. Returns the dispatch's token proxy."""
        chars = len(prompt) if prompt else 0
        est = estimate_tokens(prompt)
        entry = self._by_agent.setdefault(
            agent_id, {"dispatches": 0, "prompt_chars": 0, "est_tokens": 0}
        )
        entry["dispatches"] += 1
        entry["prompt_chars"] += chars
        entry["est_tokens"] += est
        return est

    def reset(self) -> None:
        self._by_agent.clear()

    @property
    def is_empty(self) -> bool:
        return not self._by_agent

    def totals(self) -> dict:
        """Return the session totals block.

        Shape::

            {"proxy": "len/4", "total_dispatches": int, "total_prompt_chars": int,
             "total_est_tokens": int, "by_agent": {id: {dispatches, prompt_chars,
             est_tokens}}}
        """
        by_agent = {aid: dict(entry) for aid, entry in self._by_agent.items()}
        return {
            "proxy": PROXY_LABEL,
            "total_dispatches": sum(e["dispatches"] for e in by_agent.values()),
            "total_prompt_chars": sum(e["prompt_chars"] for e in by_agent.values()),
            "total_est_tokens": sum(e["est_tokens"] for e in by_agent.values()),
            "by_agent": by_agent,
        }


# Process-level default ledger. The persona dispatch path feeds it; the output
# bundle reads it. A module singleton because dispatch and bundle assembly are
# decoupled call sites within a session process.
_SESSION_LEDGER = SessionCostLedger()


def record_dispatch(agent_id: str, prompt: str) -> int:
    """Record a dispatch on the process-level session ledger. Returns the token proxy."""
    return _SESSION_LEDGER.record(agent_id, prompt)


def session_cost_totals() -> dict:
    """Return the session totals block from the process-level ledger."""
    return _SESSION_LEDGER.totals()


def reset_session_costs() -> None:
    """Clear the process-level ledger (used at session start and in tests)."""
    _SESSION_LEDGER.reset()
