"""Lightweight token/cost visibility for persona dispatch.

We do not have real token accounting from the orchestrator's untracked LLM
dispatches, so this module provides a cheap, deterministic *proxy*: characters
divided by four, the widely-used rough tokens-per-char heuristic for English.
It is a visibility signal (how heavy was today's prompt load, and which agent
dominated it), not a billing figure.

The persona dispatch path (``engine.persona_dispatch.wrap_persona_prompt``) feeds
every wrapped prompt into the session ledger. The daily output bundle reads the
accumulated totals into a ``session_costs`` block.

The ledger is persisted, one JSON row per dispatch
-------------------------------------------------
The orchestrator runs each step as its own ``python -c`` process, so a ledger
held only in memory was empty in the process that assembled the bundle: the
2026-09-23 bundle recorded 0 dispatches against the 33 the session made, and
every bundle before it read the same way. Each dispatch is therefore appended
to ``<session_state_dir>/dispatch_ledger.jsonl`` (gitignored), and the totals
are rebuilt from that file. ``scripts.session_guard.anchor_session`` resets it
at Step 0c, so a session counts its own dispatches only.

A torn or unreadable line is skipped with a warning, never raised: losing a
session over a visibility counter would cost far more than a miscount.

The model is recorded per dispatch
----------------------------------
Each row carries the ``model`` the dispatch was made with (the persona's
frontmatter alias, e.g. ``"opus"``) and ``model_id``. The personas follow
floating aliases on purpose (decision D4, 2026-09-24), so the alias alone does
not say which release wrote a record.

``model_id`` has two sources, and the observation wins:

- **Observed** (``transcript_model_ids``): Claude Code writes every subagent's
  transcript to ``<config>/projects/<project>/<session id>/subagents/agent-*.jsonl``,
  each assistant line naming the release that answered, beside an
  ``agent-*.meta.json`` holding the ``model`` the Task call passed. When this
  session's transcripts show exactly ONE release for an alias, every dispatch
  on that alias records it. The totals also carry ``observed_model_ids`` —
  alias to every release seen — so an alias that answered as two releases in
  one session (the alias moved mid-session) is visible, and no row claims
  either.
- **Predicted** (``resolve_model_id``): what the harness's alias override
  (``ANTHROPIC_DEFAULT_<ALIAS>_MODEL``) pins in this process's environment,
  read before the dispatch. ``None`` when nothing pins it — the expected value
  in the cloud, where the routine pins nothing.

When neither answers, ``model_id`` is ``None``: unknown is recorded as unknown,
never guessed. The transcript path was built and tested locally; whether the
cloud sandbox writes those transcripts and exposes ``CLAUDE_CODE_SESSION_ID``
to the orchestrator's Python is proven only by the first cloud session's
bundle — a ``null`` there means it does not.

``total_dispatches`` counts ``wrap_persona_prompt`` calls — prompt wraps, not
API dispatches. They are equal when the orchestrator follows the prompt; a
wrap that is never dispatched, or a retry that re-wraps, counts once more.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from engine.config import get_config

_CHARS_PER_TOKEN = 4
PROXY_LABEL = "len/4"
LEDGER_FILENAME = "dispatch_ledger.jsonl"

# Aliases the harness resolves itself. Anything else passed as a model is
# already a release id and resolves to itself.
_ALIASES = ("opus", "sonnet", "haiku")


def estimate_tokens(text: str | None) -> int:
    """Return the character-count token proxy: ``len(text) // 4``.

    Deterministic and network-free. ``None``/empty → 0.
    """
    if not text:
        return 0
    return len(text) // _CHARS_PER_TOKEN


def resolve_model_id(model: str | None) -> str | None:
    """Return the release id ``model`` resolves to, or ``None`` when unknown.

    An alias (``opus``/``sonnet``/``haiku``) resolves through the harness
    override ``ANTHROPIC_DEFAULT_<ALIAS>_MODEL`` when it is set; a value that is
    not an alias is already an id. ``None`` in, ``None`` out.
    """
    if not model:
        return None
    if model not in _ALIASES:
        return model
    return os.environ.get(f"ANTHROPIC_DEFAULT_{model.upper()}_MODEL") or None


class SessionCostLedger:
    """Accumulates per-dispatch prompt-size proxies across a session.

    Keyed by agent id. Each entry tracks the dispatch count, total prompt
    characters, and total estimated tokens (the ``len/4`` proxy). ``totals``
    returns a JSON-serializable block suitable for the output bundle, including
    one row per dispatch, in dispatch order, naming the model it ran on.
    """

    def __init__(self) -> None:
        self._by_agent: dict[str, dict[str, int]] = {}
        self._dispatches: list[dict] = []

    def record(
        self,
        agent_id: str,
        prompt: str,
        model: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Record one dispatch for ``agent_id``. Returns the dispatch's token proxy."""
        return self._add(
            agent_id,
            len(prompt) if prompt else 0,
            estimate_tokens(prompt),
            model,
            model_id,
        )

    def _add(
        self,
        agent_id: str,
        chars: int,
        est: int,
        model: str | None,
        model_id: str | None,
    ) -> int:
        entry = self._by_agent.setdefault(
            agent_id,
            {"dispatches": 0, "prompt_chars": 0, "est_tokens": 0},
        )
        entry["dispatches"] += 1
        entry["prompt_chars"] += chars
        entry["est_tokens"] += est
        self._dispatches.append(
            {"agent_id": agent_id, "model": model, "model_id": model_id}
        )
        return est

    def reset(self) -> None:
        self._by_agent.clear()
        self._dispatches.clear()

    @property
    def is_empty(self) -> bool:
        return not self._by_agent

    def totals(self) -> dict:
        """Return the session totals block.

        Shape::

            {"proxy": "len/4", "total_dispatches": int, "total_prompt_chars": int,
             "total_est_tokens": int,
             "by_agent": {id: {dispatches, prompt_chars, est_tokens}},
             "dispatches": [{agent_id, model, model_id}, ...]}
        """
        by_agent = {aid: dict(entry) for aid, entry in self._by_agent.items()}
        return {
            "proxy": PROXY_LABEL,
            "total_dispatches": sum(e["dispatches"] for e in by_agent.values()),
            "total_prompt_chars": sum(e["prompt_chars"] for e in by_agent.values()),
            "total_est_tokens": sum(e["est_tokens"] for e in by_agent.values()),
            "by_agent": by_agent,
            "dispatches": [dict(row) for row in self._dispatches],
        }


#: Test override for the ledger file. ``None`` resolves it from config at call
#: time, so a fork's ``MIDAS_DATA_DIR`` reaches it; the suite sets a per-test path.
_LEDGER_PATH: Path | None = None


def _ledger_path() -> Path:
    if _LEDGER_PATH is not None:
        return Path(_LEDGER_PATH)
    return get_config().session_state_dir / LEDGER_FILENAME


def record_dispatch(agent_id: str, prompt: str, model: str | None = None) -> int:
    """Append one dispatch to the session ledger. Returns the token proxy.

    ``model`` is the value the dispatch is made with (the persona's frontmatter
    alias); its resolved release id is recorded beside it.
    """
    chars = len(prompt) if prompt else 0
    est = estimate_tokens(prompt)
    row = {
        "agent_id": agent_id,
        "prompt_chars": chars,
        "est_tokens": est,
        "model": model,
        "model_id": resolve_model_id(model),
    }
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One short line per write, opened in append mode: the parallel dispatch
    # rounds are prepared from separate processes, and appends of this size
    # do not interleave.
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return est


#: Test override for Claude Code's config directory (the transcripts' root).
#: ``None`` resolves it at call time: ``$CLAUDE_CONFIG_DIR``, else ``~/.claude``.
#: The suite points it at an empty tmp directory, so a test run from inside a
#: Claude Code session never reads that session's real transcripts.
_TRANSCRIPT_ROOT: Path | None = None

#: Claude Code's placeholder for a message no model wrote.
_SYNTHETIC_MODEL = "<synthetic>"


def _transcript_root() -> Path:
    if _TRANSCRIPT_ROOT is not None:
        return Path(_TRANSCRIPT_ROOT)
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(configured) if configured else Path.home() / ".claude"


def transcript_model_ids() -> dict[str, list[str]]:
    """Map each alias this session dispatched on to the releases that answered.

    Reads this session's subagent transcripts only (``CLAUDE_CODE_SESSION_ID``;
    none set, nothing read). A transcript whose Task call named no model
    inherited one and is not evidence about any alias. Unreadable files and
    lines are skipped: a visibility record must never cost the session.
    """
    session = os.environ.get("CLAUDE_CODE_SESSION_ID")
    if not session:
        return {}
    observed: dict[str, set[str]] = {}
    for transcript in sorted(
        _transcript_root().glob(f"projects/*/{session}/subagents/agent-*.jsonl")
    ):
        try:
            meta = json.loads(
                transcript.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            lines = transcript.read_text(encoding="utf-8").splitlines()
        except (OSError, json.JSONDecodeError):
            continue
        alias = meta.get("model") if isinstance(meta, dict) else None
        if not isinstance(alias, str) or not alias:
            continue
        for line in lines:
            try:
                entry = json.loads(line)
                model = entry["message"]["model"] if entry.get("type") == "assistant" else None
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                continue
            if isinstance(model, str) and model and model != _SYNTHETIC_MODEL:
                observed.setdefault(alias, set()).add(model)
    return {alias: sorted(ids) for alias, ids in sorted(observed.items())}


def _load_ledger() -> SessionCostLedger:
    ledger = SessionCostLedger()
    path = _ledger_path()
    if not path.exists():
        return ledger
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            ledger._add(
                str(row["agent_id"]),
                int(row["prompt_chars"]),
                int(row["est_tokens"]),
                row.get("model"),
                row.get("model_id"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            print(
                f"[token_cost] skipping unreadable dispatch ledger line {lineno} "
                f"in {path}",
                file=sys.stderr,
            )
    return ledger


def session_cost_totals() -> dict:
    """Return the session totals block, rebuilt from the persisted ledger.

    Each dispatch's ``model_id`` is the release its transcripts show when they
    show exactly one for its alias (see the module docstring); the block also
    carries ``observed_model_ids``, everything the transcripts showed.
    """
    totals = _load_ledger().totals()
    observed = transcript_model_ids()
    for row in totals["dispatches"]:
        seen = observed.get(row.get("model") or "")
        if seen and len(seen) == 1:
            row["model_id"] = seen[0]
    totals["observed_model_ids"] = observed
    return totals


def reset_session_costs() -> None:
    """Clear the session ledger (called by ``anchor_session`` and in tests)."""
    _ledger_path().unlink(missing_ok=True)
