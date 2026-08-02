"""Shared pytest fixtures for the Midas test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def midas_data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect MIDAS_DATA_DIR to an isolated tmp project root.

    Replaces the old practice of monkeypatching individual module-level path
    constants (``engine.orders.OUTBOX_DIR``, ``engine.posts.POSTS_DIR``, …),
    which were removed when the engine moved to config-driven, lazily-resolved
    paths. After this fixture runs, every ``get_config().*`` path property (and
    the lazy module attributes backed by them) resolves under ``root/data`` /
    ``root/.claude``. A real ``roster.yaml`` is seeded because ``get_config()``
    requires one at the data root.

    Returns the resolved project root. Code that writes via the engine's lazy
    paths lands under this root; tests seed and assert against
    ``get_config().<prop>``.
    """
    from engine.config import _LEGACY_ROOT, reset_config_cache

    root = tmp_path.resolve()
    shutil.copy(_LEGACY_ROOT / "roster.yaml", root / "roster.yaml")
    monkeypatch.setenv("MIDAS_DATA_DIR", str(root))
    reset_config_cache()
    yield root
    reset_config_cache()


@pytest.fixture(autouse=True)
def _isolated_session_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the session-state directory to a tmp path for every test.

    The ``idempotent_step`` decorator in ``scripts.daily_session`` calls
    ``scripts.session_state.is_done`` / ``mark_done``, which resolves
    ``_STATE_DIR`` at call time from the ``session_state`` module namespace.
    Monkeypatching ``_STATE_DIR`` here prevents step-completion markers from
    leaking between tests and eliminates side-effects from the real
    ``data/session_state/`` directory.
    """
    import scripts.session_state as ss

    monkeypatch.setattr(ss, "_STATE_DIR", tmp_path / "session_state")


def _running_live_cast() -> bool:
    """True on the live Midas desk, False only on the confirmed demo desk.

    Cast-coupled tests (``@pytest.mark.live_cast``) hardcode the live roster and
    are skipped ONLY when the suite affirmatively resolves the demo desk (whose
    agent ids are all ``demo-*``). Any other outcome — a live roster, an empty
    roster, or an unreadable/malformed one — returns True so the tests RUN. This
    fails closed: the whole point of guards like ``test_live_switch`` is to catch
    a bad live config on CI, so an unreadable roster must never silently skip
    them. Reads the roster directly from the resolved data dir — it does NOT call
    ``get_config()``, so the lru-cache is never populated at collection time.
    """
    from engine.config import _resolve_data_dir

    try:
        raw = yaml.safe_load((_resolve_data_dir() / "roster.yaml").read_text("utf-8"))
        agent_ids = list((raw or {}).get("agents", {}))
    except (OSError, yaml.YAMLError, AttributeError):
        return True  # roster unreadable → assume live and RUN (fail closed)
    # Skip only when we can confirm the demo desk (non-empty, all demo-* ids).
    return not (bool(agent_ids) and all(aid.startswith("demo-") for aid in agent_ids))


def pytest_collection_modifyitems(config, items):
    """Skip ``live_cast``-marked tests when the demo desk is the active roster."""
    if _running_live_cast():
        return
    skip = pytest.mark.skip(reason="cast-coupled; live desk only (demo cast active)")
    for item in items:
        if "live_cast" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _neutralize_session_guard(request, monkeypatch):
    """Let step_* unit tests run without a live session anchor.

    ``scripts.session_guard.assert_session_fresh`` is deliberately fatal when no
    anchor exists — an unguarded session is what produced the 2026-07-31 stall.
    That contract is right in production and wrong for unit tests, which call
    ``step_author_all`` / ``step_git_commit_push`` directly and have no session.

    Patched here rather than behind an env-var bypass so the escape hatch cannot
    leak into a real run. ``tests/test_session_guard.py`` opts out — it is the
    suite that exercises the guard itself.
    """
    if request.node.fspath.basename == "test_session_guard.py":
        return
    monkeypatch.setattr(
        "scripts.daily_session.assert_session_fresh", lambda *a, **k: None
    )
