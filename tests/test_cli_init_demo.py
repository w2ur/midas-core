"""`midas init-demo` materialises the packaged demo desk onto disk."""

from __future__ import annotations

import importlib
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from engine.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]

# The name the fixture desk is installed under. Deliberately NOT
# ``midas_demo_desk``: an editable install of midas-core registers a meta-path
# finder for that name, which wins over anything this test puts on sys.path, so
# a same-named fixture would be silently bypassed in a CI environment (which
# does run `pip install -e .`) and quietly test the checkout instead.
FIXTURE_PACKAGE = "midas_demo_desk_fixture"

DEMO_AGENTS = {
    "demo-allocator.md",
    "demo-momentum.md",
    "demo-oracle.md",
    "demo-value.md",
}

# roster.yaml + README.md + 4 personas + the committed universes. __init__.py is
# packaging scaffolding and must not reach the user's desk.
EXPECTED_FILE_COUNT = (
    2 + len(DEMO_AGENTS) + len(list((REPO_ROOT / "data" / "universes").glob("*.json")))
)


@pytest.fixture
def packaged_demo_desk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Install a copy of the demo desk the way the wheel ships it.

    The wheel maps ``examples/demo-desk`` to a package name and ships the
    ``data/universes`` fixture beside it. Rebuilding that layout on a throwaway
    ``site-packages`` (rather than importing whatever the dev happened to have
    installed) keeps the test identical in the live repo and its public mirror,
    and still exercises the real ``importlib.resources`` resolution a
    pip-installed user gets.
    """
    site_packages = tmp_path / "site-packages"
    package = site_packages / FIXTURE_PACKAGE
    shutil.copytree(REPO_ROOT / "examples" / "demo-desk", package)
    # Rebuilt from the committed universes either way: the mirror's checkout
    # carries this fixture (its resolvers regenerate it), the live one does not.
    shutil.rmtree(package / "data", ignore_errors=True)
    universes = package / "data" / "universes"
    universes.mkdir(parents=True)
    for source in sorted((REPO_ROOT / "data" / "universes").glob("*.json")):
        shutil.copy2(source, universes / source.name)

    monkeypatch.syspath_prepend(str(site_packages))
    monkeypatch.setattr("engine.cli._DEMO_PACKAGE", FIXTURE_PACKAGE)
    sys.modules.pop(FIXTURE_PACKAGE, None)
    importlib.invalidate_caches()
    yield package
    sys.modules.pop(FIXTURE_PACKAGE, None)


def test_init_demo_writes_the_whole_packaged_tree(
    tmp_path: Path, packaged_demo_desk: Path, capsys: pytest.CaptureFixture[str]
):
    """Hidden directories included — `.claude/agents` is the desk's whole cast."""
    target = tmp_path / "my-desk"

    assert main(["init-demo", str(target)]) == 0

    assert (target / "roster.yaml").read_bytes() == (
        packaged_demo_desk / "roster.yaml"
    ).read_bytes()
    assert (target / "README.md").is_file()
    assert {p.name for p in (target / ".claude" / "agents").glob("*.md")} == DEMO_AGENTS

    expected_universes = {
        p.name for p in (packaged_demo_desk / "data" / "universes").glob("*.json")
    }
    assert expected_universes, "no universe fixture to copy — the assertion is vacuous"
    assert {
        p.name for p in (target / "data" / "universes").glob("*.json")
    } == expected_universes

    out = capsys.readouterr().out
    assert f"export MIDAS_DATA_DIR={target}" in out


def test_init_demo_leaves_the_packaging_scaffolding_behind(
    tmp_path: Path, packaged_demo_desk: Path
):
    """A user's desk is desk content only — no __init__.py, no __pycache__."""
    target = tmp_path / "my-desk"
    # Present for real once anything imports the package; seeded here so the
    # assertion below does not depend on whether it happened to be imported.
    (packaged_demo_desk / "__pycache__").mkdir(exist_ok=True)
    (packaged_demo_desk / "__pycache__" / "__init__.cpython-312.pyc").touch()
    assert (packaged_demo_desk / "__init__.py").is_file(), (
        "the desk is not a regular package — resolution would go back to a "
        "namespace path and break under an editable install"
    )

    assert main(["init-demo", str(target)]) == 0

    written = sorted(p for p in target.rglob("*") if p.is_file())
    assert len(written) == EXPECTED_FILE_COUNT
    assert not (target / "__init__.py").exists()
    assert not (target / "__pycache__").exists()


def test_init_demo_accepts_an_existing_empty_directory(
    tmp_path: Path, packaged_demo_desk: Path
):
    target = tmp_path / "my-desk"
    target.mkdir()

    assert main(["init-demo", str(target)]) == 0
    assert (target / "roster.yaml").is_file()


def test_init_demo_refuses_a_non_empty_target(
    tmp_path: Path, packaged_demo_desk: Path, capsys: pytest.CaptureFixture[str]
):
    target = tmp_path / "my-desk"
    assert main(["init-demo", str(target)]) == 0
    (target / "roster.yaml").write_text("mine\n", encoding="utf-8")
    capsys.readouterr()

    assert main(["init-demo", str(target)]) == 1

    assert (target / "roster.yaml").read_text(encoding="utf-8") == "mine\n"
    assert "not empty" in capsys.readouterr().err


def test_init_demo_force_writes_into_a_non_empty_target(
    tmp_path: Path, packaged_demo_desk: Path
):
    target = tmp_path / "my-desk"
    assert main(["init-demo", str(target)]) == 0
    (target / "roster.yaml").write_text("mine\n", encoding="utf-8")

    assert main(["init-demo", str(target), "--force"]) == 0

    assert (target / "roster.yaml").read_bytes() == (
        packaged_demo_desk / "roster.yaml"
    ).read_bytes()


def test_init_demo_reports_a_missing_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Run from a checkout that never installed the package: fail, don't crash."""
    monkeypatch.setattr("engine.cli._DEMO_PACKAGE", "midas_demo_desk_not_installed")

    assert main(["init-demo", str(tmp_path / "my-desk")]) == 1
    assert "not installed" in capsys.readouterr().err


def test_installed_demo_desk_resolves_to_a_single_directory():
    """The installed package must resolve to one directory, not a namespace path.

    Regression: examples/demo-desk shipped without __init__.py, so the wheel's
    ``midas_demo_desk`` was a namespace package. An editable install then adds a
    second, non-directory portion (``__editable__…finder.__path_hook__``) and
    ``files()`` raises NotADirectoryError — `midas init-demo` crashed for every
    `pip install -e .` user. Skipped where the package is not installed; CI
    installs it, so this runs there.
    """
    from importlib.resources import files

    from engine.cli import _DEMO_PACKAGE

    pytest.importorskip(_DEMO_PACKAGE)

    source = files(_DEMO_PACKAGE)
    assert (source / "roster.yaml").is_file()
    assert (source / ".claude" / "agents" / "demo-momentum.md").is_file()
