"""Shared pytest fixtures for the Midas test suite."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


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
