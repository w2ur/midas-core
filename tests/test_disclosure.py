"""Restatement disclosure is a precondition, not an audit.

The failure this guards is documented in METHODOLOGY.md itself: on 2026-08-02
a commit rewrote a book's ledger and restated 41 snapshots, moving its
published return from +3.30% to +0.19%, with no changelog entry. It surfaced
five days later "by a cross-check on a different task, not by any process".

The distinction the tests below insist on is precondition vs audit. An audit
tells you afterwards that you forgot. A precondition means the restatement
does not run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from engine.disclosure import (
    UndisclosedRestatementError,
    known_anchors,
    require_changelog_entry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_methodology(root: Path, body: str) -> None:
    (root / "METHODOLOGY.md").write_text(body, encoding="utf-8")


class TestRequireChangelogEntry:
    def test_a_resolving_anchor_is_accepted(self, midas_data_root):
        _write_methodology(
            midas_data_root, '- <a id="units-2026-08-07"></a>**Units changed.**\n'
        )
        assert (
            require_changelog_entry("units-2026-08-07", what="Restating")
            == "units-2026-08-07"
        )

    def test_a_missing_anchor_is_refused(self, midas_data_root):
        _write_methodology(midas_data_root, '- <a id="units-2026-08-07"></a>**x**\n')
        with pytest.raises(UndisclosedRestatementError) as exc:
            require_changelog_entry(None, what="Restating valuations")
        assert "Restating valuations" in str(exc.value)

    def test_an_anchor_that_does_not_resolve_is_refused(self, midas_data_root):
        """A dead link in the public record is worse than no link."""
        _write_methodology(midas_data_root, '- <a id="real"></a>**x**\n')
        with pytest.raises(UndisclosedRestatementError):
            require_changelog_entry("typo-in-the-anchor", what="Restating")

    def test_a_fork_without_a_methodology_is_not_held_to_it(self, midas_data_root):
        """Refusing here would break the restatement tooling downstream for no
        honesty gain — there is nothing to write the disclosure into."""
        assert not (midas_data_root / "METHODOLOGY.md").exists()
        assert require_changelog_entry(None, what="Restating") == ""

    @pytest.mark.skipif(
        not (REPO_ROOT / "METHODOLOGY.md").exists(),
        reason="METHODOLOGY.md is a live-desk document; forks need not carry one",
    )
    def test_anchors_are_read_from_the_real_document(self):
        """The parser must find this repo's actual anchors, not just fixtures.

        A regex that silently matched nothing would make every anchor look
        invalid — and the "no METHODOLOGY" escape hatch would then make every
        anchor look valid instead, depending on which branch it hit. Pin it
        against the real file.
        """
        anchors = known_anchors()
        assert "lost-fill-2026-05-21" in anchors
        assert len(anchors) >= 5


class TestScriptsRefuseToApplyUndisclosed:
    """End-to-end: the CLI itself must refuse, not just the helper.

    A precondition that exists only in a library function is one import away
    from being bypassed by the next caller.

    Run against an isolated MIDAS_DATA_DIR carrying its own METHODOLOGY.md,
    so these exercise the gate rather than the live desk's document — and so
    they run identically on a fork, where the real file does not exist.
    """

    @staticmethod
    def _env(root: Path) -> dict:
        (root / "METHODOLOGY.md").write_text(
            '## Methodology changelog\n\n- <a id="a-real-entry"></a>**Something moved.**\n',
            encoding="utf-8",
        )
        return dict(os.environ, MIDAS_DATA_DIR=str(root))

    @pytest.mark.parametrize("script", ["restate_valuations.py", "restate_bundles.py"])
    def test_apply_without_a_changelog_entry_exits_nonzero(self, script, midas_data_root):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script), "--apply"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=self._env(midas_data_root),
        )
        assert result.returncode != 0
        assert "requires disclosure" in (result.stdout + result.stderr)

    @pytest.mark.parametrize("script", ["restate_valuations.py", "restate_bundles.py"])
    def test_apply_with_an_unresolvable_anchor_exits_nonzero(self, script, midas_data_root):
        result = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / script),
                "--apply",
                "--changelog-entry",
                "no-such-anchor-exists",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=self._env(midas_data_root),
        )
        assert result.returncode != 0
        assert "no-such-anchor-exists" in (result.stdout + result.stderr)

    @pytest.mark.parametrize("script", ["restate_valuations.py", "restate_bundles.py"])
    def test_a_dry_run_is_not_gated(self, script, midas_data_root):
        """You cannot write the disclosure before you know what would move.

        Run against an empty MIDAS_DATA_DIR: there is nothing to restate, so
        this measures the gate rather than the restatement (pointing it at the
        real desk made this test take 40 seconds per script).
        """
        env = dict(os.environ, MIDAS_DATA_DIR=str(midas_data_root))
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script), "--dry-run"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        assert "requires disclosure" not in (result.stdout + result.stderr)
