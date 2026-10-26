"""Guards for the ``live_cast`` skip predicate in ``tests/conftest.py``.

The predicate decides whether ``@pytest.mark.live_cast`` tests run (live desk)
or skip (demo desk). It must fail *closed* — an unreadable roster runs them,
so CI guards like ``test_live_switch`` can never be silently disabled.
"""

from __future__ import annotations

from tests.conftest import _default_to_demo_desk, _running_live_cast


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


def _fake_checkout(root, with_agents=False):
    demo = root / "examples" / "demo-desk"
    demo.mkdir(parents=True)
    (demo / "roster.yaml").write_text("agents: {}\n")
    if with_agents:
        (root / ".claude" / "agents").mkdir(parents=True)


# Regression: this commit — bare `pytest` in midas-core failed
# test_the_persona_directory_is_read because MIDAS_DATA_DIR was unset.
def test_unset_data_dir_defaults_to_the_demo_desk(tmp_path):
    _fake_checkout(tmp_path)
    env: dict[str, str] = {}
    assert _default_to_demo_desk(tmp_path, env) is True
    assert env["MIDAS_DATA_DIR"] == str(tmp_path / "examples" / "demo-desk")


def test_explicit_data_dir_is_never_overridden(tmp_path):
    _fake_checkout(tmp_path)
    env = {"MIDAS_DATA_DIR": "/somewhere/else"}
    assert _default_to_demo_desk(tmp_path, env) is False
    assert env["MIDAS_DATA_DIR"] == "/somewhere/else"


def test_a_checkout_with_its_own_desk_is_left_alone(tmp_path):
    _fake_checkout(tmp_path, with_agents=True)
    env: dict[str, str] = {}
    assert _default_to_demo_desk(tmp_path, env) is False
    assert env == {}
