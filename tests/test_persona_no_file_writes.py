"""No persona tells its agent to write a file under data/.

Every dispatch round runs inside the dispatch guard (engine.dispatch_guard):
a file an agent writes in the checkout aborts the session, and a journal is
rewritten by the session from the text the agent returns. A persona that says
"maintain your journal at data/..." invites exactly the write that aborts.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from engine.config import get_config, reset_config_cache

# A write verb that governs a data/ path later in the same sentence. Reading a
# path and then writing prose ("read data/x before writing the blog") is fine.
_WRITE_TO_DATA = re.compile(
    r"\b(?:write|edit|maintain|update|save|append|create|modify|overwrite|record)"
    r"\w*\b[^.\n]{0,120}?`?data/",
    re.IGNORECASE,
)


def _personas() -> list[Path]:
    # Resolved as engine.persona_dispatch resolves it, so a desk configured
    # with its own personas (midas-core's examples/demo-desk) is read from
    # there. Resolved inside the test, never at import: collection must not
    # populate the config cache (tests/conftest.py documents that guarantee).
    return sorted(get_config().agents_dir.glob("*.md"))


def test_the_persona_directory_is_read() -> None:
    assert len(_personas()) > 1


def test_no_persona_tells_its_agent_to_write_under_data() -> None:
    offenders = {
        path.name: hits
        for path in _personas()
        if (hits := [m.group(0) for m in _WRITE_TO_DATA.finditer(path.read_text())])
    }
    assert offenders == {}, f"personas that tell their agent to write a file: {offenders}"


def test_importing_this_module_does_not_load_the_config() -> None:
    reset_config_cache()
    importlib.reload(importlib.import_module(__name__))
    assert get_config.cache_info().currsize == 0


def test_the_pattern_catches_the_line_it_was_written_for() -> None:
    old = "You also maintain your own journal at `data/agent_memory/the-oracle.md`."
    assert _WRITE_TO_DATA.search(old)
    assert not _WRITE_TO_DATA.search(
        "Read your journal from data/agent_memory/x.md before writing the blog."
    )
