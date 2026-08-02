"""The hand-written packaging manifest must not drift from the source tree.

``pyproject.toml`` lists ``[tool.setuptools] packages`` explicitly instead of
using auto-discovery, because the demo desk has to reach the wheel under an
importable name (``midas_demo_desk``) that discovery would never find. The cost
of that choice is a list that goes stale in silence: add ``engine/whatever/``
with an ``__init__.py`` and the wheel simply ships without it. This is the
guard for that.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from setuptools import find_packages

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACKAGE = "midas_demo_desk"


def _setuptools_config() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["tool"]["setuptools"]


def test_declared_packages_match_discovery():
    """Every importable engine/scripts package is declared, and vice versa."""
    declared = _setuptools_config()["packages"]
    code_packages = sorted(p for p in declared if not p.startswith(DEMO_PACKAGE))
    discovered = sorted(
        find_packages(where=str(REPO_ROOT), include=["engine*", "scripts*"])
    )
    assert discovered, "package discovery found nothing — the comparison is vacuous"
    assert code_packages == discovered


def test_demo_desk_ships_under_an_importable_name():
    """`midas init-demo` resolves this package name; the mapping must hold."""
    config = _setuptools_config()
    assert DEMO_PACKAGE in config["packages"]
    source = REPO_ROOT / config["package-dir"][DEMO_PACKAGE]
    assert (source / "roster.yaml").is_file()
    assert sorted(p.name for p in (source / ".claude" / "agents").glob("*.md"))
