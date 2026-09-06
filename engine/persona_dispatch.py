"""Persona-injection helpers for Task tool dispatch.

Project-level subagent files in `.claude/agents/*.md` are NOT auto-registered
as dispatchable `subagent_type` values in the Claude Code Task tool — neither
in local sessions nor in cloud RemoteTrigger sessions. Only user-level agents
(`~/.claude/agents/`) and harness-provided agents (e.g. `general-purpose`)
appear in the dispatchable registry.

To preserve per-agent context isolation (each subagent gets a fresh window —
critical for the journal-rewrite loop, which is the long-term memory store),
we dispatch every persona-authored task through `subagent_type="general-purpose"`
with the persona's body injected into the prompt as a system-style preamble.

The orchestrator session must NEVER author persona content directly. Wrapping
via this module is the substitute for the auto-registration we don't have.
"""

from __future__ import annotations

import re
import sys
from datetime import date

from engine.config import get_config
from engine.desk_notices import render_notice_block
from engine.token_cost import record_dispatch

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

_WRAPPER_TEMPLATE = """\
You are operating as the persona defined below. Stay strictly in this persona \
for the entire task — voice, mandate, universe, base currency, position limits, \
all of it. Do NOT break character or speak as a generic assistant. Output exactly \
what the persona's task instructions ask for; do not add meta-commentary about \
being an AI or about the dispatch mechanism.

--- PERSONA ({agent_id}) ---
{persona_body}
--- END PERSONA ---
{notice_block}
--- TASK ---
{task_prompt}
"""


def load_persona(agent_id: str) -> tuple[str, str | None]:
    """Read .claude/agents/{agent_id}.md and split frontmatter from body.

    Returns (body, model) where:
      - body: the persona instructions, with frontmatter stripped, trailing
        whitespace trimmed.
      - model: the `model:` frontmatter value (e.g. "opus", "sonnet"), or None
        if not specified.

    Raises FileNotFoundError if the persona file does not exist.
    """
    path = get_config().agents_dir / f"{agent_id}.md"
    raw = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return raw.strip(), None
    frontmatter = match.group(1)
    body = raw[match.end() :].strip()
    model: str | None = None
    for line in frontmatter.splitlines():
        if line.startswith("model:"):
            value = line.split(":", 1)[1].strip()
            model = value or None
            break
    return body, model


def wrap_persona_prompt(
    agent_id: str, task_prompt: str, today: date | None = None
) -> tuple[str, str | None]:
    """Wrap a task prompt with the persona's body for general-purpose dispatch.

    Returns (wrapped_prompt, model). Callers dispatch via:
        Task(subagent_type="general-purpose", model=model, prompt=wrapped_prompt)

    `model` may be None when the persona has no `model:` frontmatter — in that
    case omit the `model` parameter and let the harness pick the default.

    Any in-window desk notice addressed to this persona (see
    `engine.desk_notices`) is injected between the persona body and the task, so
    a time-boxed operational fact reaches every agent without the owner
    re-pasting the live RemoteTrigger prompt. `today` overrides the UTC date the
    notice window is evaluated against; production passes nothing. With no
    notice in window the wrapped prompt is byte-identical to what this function
    produced before notices existed.
    """
    body, model = load_persona(agent_id)
    wrapped = _WRAPPER_TEMPLATE.format(
        agent_id=agent_id,
        persona_body=body,
        notice_block=render_notice_block(agent_id, today),
        task_prompt=task_prompt,
    )
    # Token/cost visibility: record the character-count proxy (len/4) for this
    # dispatch into the session ledger, and log it. Proxy only — we have no real
    # token accounting from the orchestrator's untracked dispatch.
    est = record_dispatch(agent_id, wrapped)
    print(
        f"[dispatch] {agent_id}: ~{est} tokens (len/4 proxy, {len(wrapped)} chars)",
        file=sys.stderr,
    )
    return wrapped, model
