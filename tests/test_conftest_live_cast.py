"""Guards for the ``live_cast`` skip predicate in ``tests/conftest.py``.

The predicate decides whether ``@pytest.mark.live_cast`` tests run (live desk)
or skip (demo desk). It must fail *closed* — an unreadable roster runs them,
so CI guards like ``test_live_switch`` can never be silently disabled.
"""

from __future__ import annotations

from tests.conftest import _running_live_cast


def test_live_cast_fails_closed_on_unreadable_roster(tmp_path, monkeypatch):
    # No roster.yaml at this data dir → unreadable → assume live and RUN.
    monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
    assert _running_live_cast() is True


def test_live_cast_skips_on_confirmed_demo_desk(tmp_path, monkeypatch):
    (tmp_path / "roster.yaml").write_text(
        "globals: {}\nagents:\n  demo-momentum: {}\n  demo-value: {}\n"
    )
    monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
    assert _running_live_cast() is False  # all demo-* ids → skip


def test_live_cast_runs_on_live_desk(tmp_path, monkeypatch):
    (tmp_path / "roster.yaml").write_text(
        "globals: {}\nagents:\n  satoshi: {}\n  the-manager: {}\n"
    )
    monkeypatch.setenv("MIDAS_DATA_DIR", str(tmp_path))
    assert _running_live_cast() is True  # non-demo ids → run
