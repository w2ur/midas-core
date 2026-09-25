"""Record the commit a workflow run landed on `main`, for the session-integrity dispatch.

Every workflow that pushes `main` with GITHUB_TOKEN ends with
`.github/actions/dispatch-session-integrity`, because GitHub starts no
`on: push` run for that token's pushes. The action has to know WHICH commit
this run landed, and it used to infer it: "the newest commit on main that is
ahead of `before`". That inference is wrong after a rebase (J6 money review
round 5, M-a): a push refused, a `pull --rebase` onto another writer's commit
M', then a retry refused too — HEAD's parent is now M', M' is on main and ahead
of `before`, and the action dispatched M'. A commit this run never touched was
re-checked, its concerns re-filed, and the writer's own "not dispatched" issue
closed as recovered by a run that landed nothing.

So the run now SAYS what it landed. Every push path writes the sha it just
pushed to `$RUNNER_TEMP/landed-on-main.sha` the moment `git push … HEAD:main`
succeeds, and the action dispatches that sha or nothing. The shell twin of
`record_landed_on_main` is in `.github/actions/push-with-retry`; the Python
push paths (`scripts/check_triggers._push_head`,
`scripts/refresh_leaderboard._push_with_rebase_retry`) call this one.

The file is overwritten, not appended: a run that landed several commits
(a watcher with several fires) dispatches its last one, which is the contract
the action documents. `$RUNNER_TEMP` is per job and absent outside Actions; with
it unset nothing is written, the same rule `check_triggers.write_run_report`
follows.

A failure to write is logged and swallowed: the push itself succeeded, and the
action reads a missing record as "landed nothing" — at worst one commit goes
unchecked, which is what every commit was before 2026-09-25, never a false pass.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

RUNNER_TEMP_ENV = "RUNNER_TEMP"

#: The file name under `$RUNNER_TEMP`. The action names it by this literal;
#: tests/test_ci_guards.py pins the two together.
LANDED_FILENAME = "landed-on-main.sha"


def landed_record_path() -> Path | None:
    """Where the landed sha is recorded, or None outside an Actions job."""
    runner_temp = os.environ.get(RUNNER_TEMP_ENV)
    if not runner_temp:
        return None
    return Path(runner_temp) / LANDED_FILENAME


def record_landed_on_main(cwd: Path) -> str | None:
    """Record HEAD of the repository at ``cwd`` as the commit this run landed.

    Call it only right after `git push origin HEAD:main` succeeded — HEAD is
    then exactly the commit main holds. Returns the sha recorded, or None when
    nothing was written.
    """
    path = landed_record_path()
    if path is None:
        return None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True
    )
    sha = head.stdout.strip()
    if head.returncode != 0 or not sha:
        logger.warning("Could not read HEAD to record the landed commit.")
        return None
    try:
        path.write_text(sha + "\n", encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not record the landed commit in %s: %s", path, exc)
        return None
    return sha
