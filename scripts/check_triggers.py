"""Conditional-order watcher.

Runs on two cadences (split 2026-08-17, see the workflow headers):
check-triggers.yml daily over ALL pending orders, and
check-triggers-crypto.yml hourly with --crypto-only. Crypto is the only
class whose price moves intraday (live ccxt); everything else reads the
once-daily OHLCV store, so an hourly full sweep re-read identical data.
Walks pending orders, fetches current prices, fires when triggers are hit,
expires old ones (expiry is date-based, so the daily sweep owns it). Blackout window 19:55-21:00 UTC to avoid commit-races with the
20:00 UTC daily session (see BLACKOUT_END for why it tracks the session).

Usage:
    python scripts/check_triggers.py            # normal run
    python scripts/check_triggers.py --dry-run  # evaluate ONLY: report what
                                                # would fire/expire, mutate
                                                # nothing, commit nothing,
                                                # push nothing

--dry-run is a read-only rehearsal of the whole sweep. It evaluates every
pending order in both channels — expiry and trigger alike — and logs what would
happen, but performs no book mutation, no inbox append, no pending-file delete,
no git add/commit/push and no leaderboard refresh. The summary dict still counts
fired/expired (so the rehearsal is informative) and carries `dry_run: True`.

It was NOT dry until 2026-09-05. `main()` read `args.dry_run` only after `run()`
had returned, and only to skip the tail `commit_and_push()`; `run()` had no such
parameter at all, so `_process_channel` still called `execute_triggered_order`
(book mutation + inbox append + pending delete) and the per-order `_commit`
closure, which commits AND pushes to main. A local `--dry-run` on 2026-09-05
07:07 UTC pushed three real fills to origin/main. The flag now stops the money
path at its source, in `run()`, rather than at the end of it.

Push path (2026-09-05): every commit tries `git push origin HEAD:main`, then a
rebase-and-retry. If main still refuses — as a branch-protection rule did for
every bot push from 2026-08-24 to 09-04, losing ten fired orders with their
runners — the commit is pushed to a per-run `triggers/<date>-<run id>` branch
instead, and .github/workflows/auto-merge-session.yml merges it into main. A
run whose commits reached that branch exits 0 with a ::warning:: (the fill is
durable); only a commit that reached neither main nor the branch exits 1. A run
that finds an unmerged `triggers/*` branch on origin refuses to evaluate at
all, because main is then a stale ledger and evaluating against it re-fires
those orders at a new price.

Run report (2026-09-05): a failure-issue on this path used to tell the reader
to "check whether any pending order's trigger was hit" — the run knows exactly
which orders fired or expired, at what price, and where each commit ended up.
`write_run_report()` writes that twice under `$RUNNER_TEMP` (set in every
Actions job, absent locally — with neither it nor `$WATCHER_REPORT_PATH` set,
nothing is written): `watcher-report.json`, which both workflows upload as the
`watcher-report-<run id>` artifact, and `watcher-report.md`, which the "Read
watcher run report" step in both workflows reads back and hands to
`.github/actions/failure-issue` as `details`. The table is ALSO appended to
`$GITHUB_STEP_SUMMARY` for the run's summary tab — but that file is
PER-STEP (GitHub: "unique to the current step and changes for each step in a
job"), so the reader step must not read it: until 2026-09-05 it did, read its
own empty file, and every issue went out without the table. A dry run writes
none of these — it writes nothing at all, and a report of what a rehearsal
*would* have done is not the same fact.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import subprocess
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from engine.config import get_config
from engine.orders import Fill, Order, append_fill
from engine.paper_broker import execute_triggered_order
from engine.portfolio import PortfolioManager
from engine.triggers import (
    delete_pending,
    evaluate_trigger,
    get_current_price,
    is_expired,
    list_pending,
)
from engine.leaderboard import build_leaderboard_rows as _build_leaderboard_rows
from scripts.daily_session import (
    build_portfolio_summaries as _build_portfolio_summaries,
)

logger = logging.getLogger(__name__)

# The weekday session's start. It is not scheduled from this repo — the cron
# lives in the RemoteTrigger config on claude.ai — but three things here are
# functions of it (BLACKOUT_START, BLACKOUT_END and the auto-merge deferral
# window), so it is written down ONCE and they are stated against it.
# `tests/test_check_triggers.py` pins the two derived offsets.
SESSION_START = time(20, 0)

# SESSION_START minus five minutes: a fire started just before the session
# anchors can still be committing and pushing when it does, so the watcher
# stops EVALUATING early. That lead is about the watcher's own multi-minute
# fire path and about nothing else — in particular it is NOT the window
# auto-merge-session.yml defers its merge in, which is a single fast push and
# so starts at SESSION_START (round-2 review, 2026-09-05: deferring from 19:55
# was the slice that trapped a branch — the merge waited, the 20:00 session
# then wrote the same dated inbox file, and the stale check refused the branch
# forever while every watcher run refused to evaluate behind it).
BLACKOUT_START = time(19, 55)
# THE BLACKOUT END IS A FUNCTION OF THE SESSION START — move one and move the
# other, in the same change. That coupling is the whole content of this
# constant, and it has now been got wrong in both directions.
#
# The window has to outlast the *merge* to main, not just the sandbox commit,
# because the merge is when main actually moves. Measured over the 15 weekday
# sessions before 2026-08-07, against a 20:00 UTC session: commits landed
# 20:09-20:39, auto-merges 20:12-20:45 (worst: 2026-08-05, merged 20:45). So
# the tail runs to session start + ~45 min, and the end wants ~15 min on top.
#
# A fire inside that tail is not a lost fire but a lost SESSION: the fire
# commits to main, the session's own `assert_session_fresh` sees the ledger
# move, raises StaleSessionError, and discards a completed run.
#
# History, because the shape repeats:
#   20:30 → 21:00  2026-08-07  original end predated the merge step entirely;
#                              two of the last five sessions landed past it.
#   21:00 → 21:30  2026-08-10  session moved to 20:30 UTC (22:30 Paris), which
#                              slid the whole distribution to 20:42-21:15.
#   21:30 → 21:00  2026-08-11  session moved back to 20:00 UTC.
#
# It does not clear everything — 2026-07-29 committed at 21:46. A blackout is
# a race-narrower, not a lock; `session_guard` is the correctness mechanism.
#
# NB the cron is UTC and ignores DST, so a session anchored to a Paris wall
# time silently shifts an hour in late October and this must be revisited.
BLACKOUT_END = time(21, 0)

# Type alias for the injectable committer used in process_fired_order.
# Signature: (order_id, today, paths) -> None
# paths is a list of absolute path strings to git-add before committing.
Committer = Callable[[str, date, list[str]], None]

# Outcomes of a commit-and-push attempt. A COMMIT failure and a PUSH failure are
# different facts and must not be collapsed: a stranded *commit* is recoverable
# (the next successful push carries it), whereas a failed commit means the
# mutation is on disk and in NOTHING — it dies with the runner. Only the former
# may ever be cleared by the published-state check.
#
# PUSHED_FALLBACK is a third fact, between the two: origin/main refused the
# push (and the rebase retry) but the commit reached the run's `triggers/*`
# branch, where .github/workflows/auto-merge-session.yml takes it to main. The
# fill is DURABLE at that point — it is on origin — so it is a warning, not a
# failure. PUSH_FAILED now means the commit reached neither main nor the
# branch, which is the only case where it still dies with the runner.
COMMIT_OK = "ok"
COMMIT_FAILED = "commit-failed"
PUSHED_FALLBACK = "pushed-fallback"
PUSH_FAILED = "push-failed"

# Every fallback branch the watcher creates lives under this prefix, and
# auto-merge-session.yml triggers on exactly this pattern (`triggers/**`).
# `tests/test_ci_guards.py` pins the two together.
FALLBACK_BRANCH_PREFIX = "triggers/"

# The paths a watcher commit may touch. auto-merge-session.yml carries the same
# list and refuses a fallback branch whose diff steps outside it; the watcher's
# own `commit_and_push` stages only subsets of these. Kept here so the Python
# side and the workflow can be pinned against each other.
WATCHER_PATHS = (
    "data/orders/",
    "data/portfolios/",
    "data/leaderboard/",
    "data/tax_shadow/",
)

# Env vars the run report reads. Named rather than hard-coded inline so a test
# can assert against the same constant the code uses.
REPORT_PATH_ENV = "WATCHER_REPORT_PATH"
RUNNER_TEMP_ENV = "RUNNER_TEMP"
STEP_SUMMARY_ENV = "GITHUB_STEP_SUMMARY"
# The two files under $RUNNER_TEMP. Both watcher workflows name them by these
# exact basenames (the upload-artifact glob and the "Read watcher run report"
# step); tests/test_ci_guards.py pins the workflow text to these constants.
REPORT_JSON_FILENAME = "watcher-report.json"
REPORT_MD_FILENAME = "watcher-report.md"

# The destinations a report entry's "commit" field can name. One of them is a
# literal branch name (`triggers/<date>-<run id>`), not a constant.
REPORT_COMMIT_MAIN = "main"
REPORT_COMMIT_STRANDED = "stranded"
# An `errors` entry never had a commit to strand: the broker raised, so no fill
# was written and nothing was staged. Distinct from "stranded", which means a
# mutation exists on disk and reached nothing.
REPORT_COMMIT_NONE = "not executed"


class _FallbackBranch:
    """The one branch a watcher run may push to when origin/main refuses it.

    ``pushed`` flips to True on the first successful push and is what
    `_nothing_is_stranded` reads: a commit that is on this branch is on
    origin, so it is not stranded.

    ``fell_back`` flips to True the moment the branch is ATTEMPTED, whether
    that push succeeds or not, and is what `_push_head`'s step 0 reads. The
    two are deliberately different bits: `pushed` answers "is this commit on
    origin", `fell_back` answers "has this run stopped being allowed to
    rewrite its own history". A run whose first fallback push ALSO failed
    still holds local commits whose shas a later fill's `executed_sha` names
    (`engine.paper_broker` stamps it from `git rev-parse HEAD` at execution
    time), so keying step 0 on `pushed` let the next order's `pull --rebase`
    rewrite them: the retry then pushed the rewritten twins to main, the
    published row pointed at a sha reachable from nothing on origin, and
    `_nothing_is_stranded` cleared the failure so the run exited 0 (round-5
    review, 2026-09-05).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.pushed = False
        self.fell_back = False


# One branch per PROCESS, created lazily on the first refused push and reset at
# the top of `run()`. Module state rather than a parameter because the two
# push sites (`_git_add_commit` per fire, `commit_and_push` for the tail) are
# reached through different call chains and must land on the same branch.
_fallback: _FallbackBranch | None = None


def fallback_branch_name(today: date) -> str:
    """`triggers/<UTC date>-<GitHub run id>`, unique per run BY CONSTRUCTION.

    Uniqueness over reuse-and-force, deliberately. The alternative — one
    `triggers/<date>` branch per day, re-pushed with `--force-with-lease` —
    would let a run overwrite the branch of an EARLIER run whose merge had
    failed loudly and was waiting for a human, and a force-push of a fill
    branch is how a fill gets lost a second time. A unique name means no run
    can ever touch another run's commits; the merge workflow deletes the
    branch on success, so nothing accumulates on the happy path, and on the
    unhappy path the branch stays as the evidence. `--force-with-lease` is
    still passed on every push (see `_push_to_fallback_branch`), but as a
    tripwire on this assumption rather than as the idempotency mechanism.

    Outside Actions (no GITHUB_RUN_ID) a random token stands in for the run
    id, because a local run that hits this path must not collide either.
    GITHUB_RUN_ATTEMPT is folded in when a workflow run is re-run: same run
    id, different attempt, and the first attempt's branch may still exist.
    """
    run_id = os.environ.get("GITHUB_RUN_ID") or f"local-{secrets.token_hex(4)}"
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    if attempt and attempt != "1":
        run_id = f"{run_id}-r{attempt}"
    return f"{FALLBACK_BRANCH_PREFIX}{today.isoformat()}-{run_id}"


def _fallback_branch() -> _FallbackBranch:
    global _fallback
    if _fallback is None:
        today = datetime.now(timezone.utc).date()
        _fallback = _FallbackBranch(fallback_branch_name(today))
    return _fallback


def _reset_fallback() -> None:
    global _fallback
    _fallback = None


def in_blackout(now: datetime) -> bool:
    """True if `now` (UTC) is inside the daily-session blackout window."""
    t = now.time()
    return BLACKOUT_START <= t <= BLACKOUT_END


def merge_deferred(now: datetime) -> bool:
    """True if auto-merge-session.yml must NOT push a watcher branch to main.

    Narrower than `in_blackout` at the front edge, on purpose. The watcher's
    blackout starts five minutes before the session so a fire it has already
    STARTED cannot still be pushing when the session anchors; the merge is one
    fetch-merge-push and needs no such lead. Those five minutes were not free:
    a fire at 19:54 pushed its fallback branch, the merge dispatched at ~19:55
    deferred, and the 20:00 session then appended its own fills to the SAME
    `data/orders/inbox/<date>.jsonl` — which the stale check reads as overlap,
    so that branch could never merge again, and `unmerged_fallback_branches()`
    then stopped fires AND expiries desk-wide until a human deleted it. Found
    by the round-2 review, 2026-09-05.

    The deadline it does not remove: a branch that survives PAST the session
    (a lost dispatch, a merge that failed) is genuinely stale and stays a
    human decision. That is stated in auto-merge-session.yml's header and in
    the issue it files, rather than being called self-healing.
    """
    return SESSION_START <= now.time() <= BLACKOUT_END


def process_fired_order(
    order: Order,
    fill_or_none: Fill | None,
    today: date,
    committer: Committer,
    *,
    inbox_dir: Path | None = None,
    pending_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Process a single fired order: write fill, clean up pending, commit.

    This is the per-fire durability unit. Each fired order is committed
    (and pushed) immediately after its full mutation set, so the orphan-state
    window between a fill and its git record is closed.

    fill_or_none semantics:
    - Real Fill: append to inbox, delete pending, call committer with both paths.
    - None (idempotency skip from execute_triggered_order): the order was already
      filled in a prior watcher run. The pending file is a zombie — delete it
      and commit that cleanup (no inbox write, no portfolio mutation).

    committer is called AFTER mutations so the commit captures real disk state.
    If committer raises (e.g. push failure), the mutation is already on disk and
    in a local git commit; the exception is caught and logged here — processing
    of subsequent orders must continue.

    ``inbox_dir`` / ``pending_dir`` select the channel. Defaults of None preserve
    byte-identical public behavior (public INBOX_DIR / PENDING_DIR). The Manager
    channel passes MANAGER_INBOX_DIR / MANAGER_PENDING_DIR so its fills are
    isolated from the public inbox the site joins by order_id.

    ``dry_run`` is the SECOND lock on the same door, not the first: in a dry run
    `_process_channel` never reaches this function, because it stops before
    `execute_triggered_order`, which is what mutates the book. This guard is
    here so no future caller can reach the inbox append / pending delete /
    commit through a dry run by wiring a new path into this helper — the same
    two-locks idiom `_nothing_is_stranded` already uses. Its consumer is
    `tests/test_check_triggers.TestDryRunIsActuallyDry`.
    """
    from engine import orders as _orders
    from engine import triggers as _triggers

    if dry_run:
        logger.info(
            "DRY-RUN: %s would be processed (inbox append, pending delete, "
            "commit) — nothing written.",
            order.order_id,
        )
        return

    paths: list[str] = []

    if fill_or_none is not None:
        # Real fill: write to inbox and include its path in the commit.
        append_fill(today, fill_or_none, inbox_dir=inbox_dir)
        base_inbox = inbox_dir or _orders.INBOX_DIR
        inbox_path = str(base_inbox / f"{today.isoformat()}.jsonl")
        paths.append(inbox_path)
        # execute_triggered_order mutates portfolio.json and trades.json via
        # PortfolioManager.apply_trade. Include the agent's portfolio directory
        # so the per-fire commit captures the full mutation set atomically.
        # Directory-level add covers portfolio.json + trades.json (and any other
        # files the fill may have touched in that directory).
        portfolio_dir = str(get_config().portfolios_dir / order.agent_id)
        paths.append(portfolio_dir)

    # Remove the pending file regardless (fill or zombie cleanup).
    delete_pending(order.order_id, pending_dir=pending_dir)
    base_pending = pending_dir or _triggers.PENDING_DIR
    pending_path = str(base_pending / f"{order.order_id}.json")
    paths.append(pending_path)

    if fill_or_none is None:
        logger.info(
            "process_fired_order: %s already in inbox — zombie pending file cleaned up",
            order.order_id,
        )

    try:
        committer(order.order_id, today, paths)
    except Exception as exc:
        # Push (or commit) failure must not stop processing of remaining orders.
        # The mutation is already on disk; the local commit exists if git-commit
        # succeeded. A trailing push attempt at the end of the watcher run will
        # carry any un-pushed commits. See _git_add_commit for retry logic.
        logger.warning(
            "committer raised for %s (non-fatal, remaining orders will be processed): %s",
            order.order_id,
            exc,
        )


def _remote_branch_sha(branch: str) -> str | None:
    """The sha origin currently holds for ``branch``, or None when absent.

    None is also the answer when `ls-remote` itself fails: the lease built
    from it then says "must not exist", and if the branch does exist the push
    is refused — loud, never a silent overwrite.
    """
    out = subprocess.run(
        ["git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    first = (out.stdout or "").strip().split("\n")[0].split()
    return first[0] if first else None


def _push_to_fallback_branch(label: str) -> str:
    """Push HEAD to this run's `triggers/*` branch after origin/main refused it.

    The branch name is unique per run (see `fallback_branch_name`), so nobody
    else writes to it and every push here is one of our own commits on top of
    our own previous push. `--force-with-lease=<branch>:<sha>` is still given,
    for two reasons that are not idempotency:

    - the FIRST push carries an empty expectation, which git reads as "the ref
      must not exist" — so if the uniqueness assumption is ever wrong (a run
      id reused, a stale branch nobody deleted), the push is refused rather
      than silently replacing someone else's fill;
    - later pushes in the same run are fast-forwards by construction —
      once a run has fallen back, `_push_head` never rebases again (see its
      step 0) — so the lease on them is a second tripwire, not a mechanism:
      it names the sha we last saw on origin, and a push that finds anything
      else there is refused rather than layered on someone else's commit.

    The expected sha is read from origin right before the push rather than
    remembered, because auto-merge-session.yml may have merged and DELETED the
    branch between two fires of the same run — a remembered sha would then
    fail the lease on a branch that no longer exists, and the fill would be
    reported stranded when it could have gone out.
    """
    fb = _fallback_branch()
    # Set BEFORE the push, not on success: from here on this run may not
    # rebase, whatever origin answers. See `_FallbackBranch.fell_back`.
    fb.fell_back = True
    current = _remote_branch_sha(fb.name)
    lease = f"--force-with-lease=refs/heads/{fb.name}:{current or ''}"
    result = subprocess.run(
        ["git", "push", lease, "origin", f"HEAD:refs/heads/{fb.name}"],
        cwd=_PROJECT_ROOT,
    )
    if result.returncode == 0:
        fb.pushed = True
        logger.warning(
            "origin/main refused %s; pushed to fallback branch %s — "
            "auto-merge-session.yml takes it to main. The fill is durable.",
            label,
            fb.name,
        )
        return PUSHED_FALLBACK
    logger.error(
        "%s reached NEITHER origin/main nor the fallback branch %s; the commit "
        "exists only on this runner and dies with it.",
        label,
        fb.name,
    )
    return PUSH_FAILED


def _push_head(label: str) -> str:
    """Push HEAD to origin/main; on refusal, to the run's fallback branch.

    Returns COMMIT_OK when main took it, PUSHED_FALLBACK when only the
    `triggers/*` branch did, PUSH_FAILED when neither did. Shared by the
    per-fire commit and the tail commit so the two cannot drift apart again
    (they were two copies of the same rebase-and-retry until 2026-09-05).

    Strategy:
    0. If an earlier push THIS run already fell back — ATTEMPTED the branch,
       successfully or not — go straight to the branch: no attempt on main,
       and above all no `pull --rebase`. Once a commit is on origin its sha is
       published — a later fill's `executed_sha` (stamped from
       `git rev-parse HEAD` at execution time) names it, and
       `git checkout <executed_sha>` is the provenance METHODOLOGY promises. A
       rebase after the first fallback rewrote that sha: the fill's row then
       pointed at a commit reachable from nothing on origin. It also let a run
       end with the branch holding pre-rebase commits while main held the
       rewritten twins — not ancestors of each other, so auto-merge-session's
       `merge-base --is-ancestor` test said "not merged", a real merge of
       duplicate history conflicted, and the watcher refused to evaluate
       behind a branch whose fills were already on main. One fallback per run
       commits the run to the branch.

       The trigger is `fell_back`, not `pushed` (round-5 review, 2026-09-05).
       A run whose fallback push ALSO failed is the worst case for a rebase,
       not the exempt one: the next order's `executed_sha` still names the
       previous order's local commit, and the old guard let the very next
       push rebase it away and then succeed on main — a published fill row
       naming a sha origin has never held, on a run that exited 0 because
       `_nothing_is_stranded` found HEAD on main. Once main has refused, the
       branch is this run's only destination.
    1. `git push origin HEAD:main`.
    2. On failure, `git pull --rebase origin main` and retry once — this is
       the collision case (another writer landed first), which a rebase fixes.
    3. On failure again, push to the fallback branch. This is the REFUSAL
       case: from 2026-08-24 to 09-04 a branch-protection rule rejected every
       bot push to main with GH006, the rebase retry could not help, and ten
       fired orders died with their runners. A refused main is a recurring
       hazard on this account (2026-05-08 sandbox 403, then GH006), not a
       one-off, so the commit goes somewhere durable instead of nowhere.
    """
    if _fallback is not None and _fallback.fell_back:
        logger.info(
            "An earlier push this run fell back to %s; pushing %s straight "
            "to it (no main attempt, no rebase — this run's shas are "
            "published provenance now).",
            _fallback.name,
            label,
        )
        return _push_to_fallback_branch(label)

    result = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        cwd=_PROJECT_ROOT,
    )
    if result.returncode == 0:
        logger.info("Committed + pushed %s.", label)
        return COMMIT_OK

    logger.warning("Push failed for %s; retrying after git pull --rebase.", label)
    rebase = subprocess.run(
        ["git", "pull", "--rebase", "origin", "main"],
        cwd=_PROJECT_ROOT,
    )
    if rebase.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], cwd=_PROJECT_ROOT)
        logger.warning("Rebase failed for %s; aborted.", label)
    else:
        retry = subprocess.run(
            ["git", "push", "origin", "HEAD:main"],
            cwd=_PROJECT_ROOT,
        )
        if retry.returncode == 0:
            logger.info("Committed + pushed %s after rebase.", label)
            return COMMIT_OK
        logger.warning("Retry push also failed for %s.", label)

    return _push_to_fallback_branch(label)


def _git_add_commit(order_id: str, today: date, paths: list[str]) -> str:
    """git-add the given paths, then commit with a per-order message.

    Returns COMMIT_OK when there was nothing to push or main took the push,
    PUSHED_FALLBACK when the commit reached only the run's `triggers/*` branch
    (durable — a warning), PUSH_FAILED when it reached neither (stranded — the
    caller turns that into a non-zero exit, see main()). Processing never
    stops here: the remaining orders in the batch must still be evaluated.

    Skips the commit if there are no staged changes (defensive — the pending
    file may already be absent if a prior run cleaned it up).
    """
    subprocess.run(["git", "add", *paths], cwd=_PROJECT_ROOT, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=_PROJECT_ROOT,
    )
    if diff.returncode == 0:
        logger.info("Nothing staged for %s — skipped commit.", order_id)
        return COMMIT_OK

    msg = f"chore(triggers): execute {order_id} {today.isoformat()}"
    subprocess.run(["git", "commit", "-m", msg], cwd=_PROJECT_ROOT, check=True)
    return _push_head(order_id)


def _describe(order: Order, price: float | None) -> str:
    """One-line dry-run report for a pending order.

    ``price`` is None on the expiry branch, and deliberately so: expiry is
    date-based and is decided before any price is read, so quoting one would
    mean issuing a live ccxt call the real run never makes — a dry run must not
    do more work than the run it rehearses.
    """
    trigger = order.trigger or {}
    parts = [
        f"{order.order_id} [{order.agent_id}]",
        f"{order.action} {order.shares} {order.ticker}",
        f"trigger {trigger.get('op')} {trigger.get('level')}",
        f"expires {order.expires}",
    ]
    if price is None:
        parts.append("observed price not read (expiry is date-based)")
    else:
        parts.append(f"observed price {price}")
        # Quote-currency notional, i.e. what the broker's rails would see
        # before FX conversion into the book's currency. Not converted here:
        # a dry run must not depend on an FX lookup that could fail.
        parts.append(f"would-be notional {order.shares * price:.2f} (quote ccy)")
    return " | ".join(parts)


def _report_entry(
    order: Order,
    kind: str,
    observed_price: float | None,
    fill: Fill | None,
    *,
    error: str | None = None,
) -> dict:
    """One row of the run report — the REAL counterpart to `_describe`.

    `fill` is the Fill actually written this run: a rejection (expiry) has
    `fill_price=None`; the idempotency-skip case (`f is None` in
    `_process_channel` — the order was already filled in a prior run) is
    passed through as `fill=None` too, so `fill_price`/`notional` are simply
    absent rather than guessed from a record this run did not produce.

    `kind` is "fired", "expired" or "error". The third was missing until the
    round-2 review, 2026-09-05, and its absence inverted the one alert this
    report exists to serve: `execute_triggered_order` raising on a FIRED
    trigger appended no entry at all, so a run whose only event was that
    failure shipped `failure-issue` a table reading "No orders fired or
    expired this run." — under a body naming the broker crash. The reader of
    the single alert channel was told the run was a no-op.

    `error` is the exception's own text, carried so the table can name what
    the broker refused rather than only that it refused.

    `commit` starts unset — a fired order's own per-fire commit resolves it
    immediately (see `_process_channel`); an expired order's resolves only
    after the batched tail commit runs (see `main()`), because expiries share
    ONE commit across the whole run. An "error" entry resolves at creation to
    `REPORT_COMMIT_NONE`: there is nothing to commit and nothing to strand.
    """
    trigger = order.trigger or {}
    return {
        "order_id": order.order_id,
        "agent_id": order.agent_id,
        "ticker": order.ticker,
        "action": order.action,
        "shares": order.shares,
        "op": trigger.get("op"),
        "level": trigger.get("level"),
        "observed_price": observed_price,
        "fill_price": fill.fill_price if fill is not None else None,
        "notional": fill.notional_base if fill is not None else None,
        "kind": kind,  # "fired" | "expired" | "error"
        "error": error,
        "commit": REPORT_COMMIT_NONE if kind == "error" else None,
    }


def _report_commit_label(outcome: str) -> str:
    """Map a push outcome to the report's three-way vocabulary.

    COMMIT_FAILED collapses into REPORT_COMMIT_STRANDED deliberately: from the
    report's point of view "no commit was ever created" and "a commit exists
    but reached nowhere" are the same fact — the mutation is on disk and in
    nothing durable.

    No caller retroactively relabels a STRANDED entry after a later self-heal
    (`_nothing_is_stranded()` clearing `push_failed`) in the common case: a
    push-failed order whose commit is later carried to origin by a later
    order's successful push clears `push_failed` for the WHOLE run before
    `write_run_report` is called, and a run with nothing else wrong then exits
    0 — so a reader only ever sees STRANDED inside a report attached to a
    genuinely failed run. The one gap: if a DIFFERENT order's commit fails
    outright in the same run (COMMIT_FAILED, which self-heal cannot clear),
    the run still exits 1 for that reason, and a self-healed PUSH_FAILED entry
    from earlier in the same batch would still read STRANDED even though it
    reached origin. Accepted rather than tracked per-entry: it requires both
    failure modes in one run, which has not been observed.
    """
    if outcome == COMMIT_OK:
        return REPORT_COMMIT_MAIN
    if outcome == PUSHED_FALLBACK:
        return _fallback.name if _fallback is not None else "unknown-fallback-branch"
    return REPORT_COMMIT_STRANDED  # PUSH_FAILED or COMMIT_FAILED


def _fmt_report_value(value) -> str:
    return "—" if value is None else str(value)


def _report_markdown_table(entries: list[dict]) -> str:
    """Human-readable counterpart to the JSON report, for $GITHUB_STEP_SUMMARY."""
    lines = ["### Watcher run report", ""]
    if not entries:
        lines.append("No orders fired, expired or failed this run.")
        lines.append("")
        return "\n".join(lines)
    lines.append(
        "| order | agent | kind | action | ticker | trigger | observed price | fill price | notional | commit |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for e in entries:
        trigger = f"{e['op']} {e['level']}" if e.get("op") is not None else "—"
        lines.append(
            "| {order_id} | {agent_id} | {kind} | {action} {shares} | {ticker} | "
            "{trigger} | {observed_price} | {fill_price} | {notional} | {commit} |".format(
                order_id=e["order_id"],
                agent_id=e["agent_id"],
                kind=e["kind"],
                action=e["action"],
                shares=e["shares"],
                ticker=e["ticker"],
                trigger=trigger,
                observed_price=_fmt_report_value(e.get("observed_price")),
                fill_price=_fmt_report_value(e.get("fill_price")),
                notional=_fmt_report_value(e.get("notional")),
                commit=e.get("commit") or "?",
            )
        )
    lines.append("")
    # The table says an order errored; only this says what the broker raised,
    # and that is the whole content of the alert on this path.
    failed = [e for e in entries if e.get("kind") == "error"]
    if failed:
        lines.append(
            "**`execute_triggered_order` raised on a FIRED trigger — the "
            "pending file survives and the order retries:**"
        )
        lines.append("")
        for e in failed:
            lines.append(f"- `{e['order_id']}` [{e['agent_id']}]: {e.get('error')}")
        lines.append("")
    return "\n".join(lines)


def write_run_report(entries: list[dict], *, now: datetime) -> None:
    """Write the run's fired/expired/failed orders as JSON and as a markdown table.

    Three writes, two of them with a named consumer:

    - JSON to `$WATCHER_REPORT_PATH` when set, else to
      `$RUNNER_TEMP/watcher-report.json`. Consumer: the `actions/upload-artifact`
      step in both watcher workflows (artifact `watcher-report-<run id>`),
      which is what outlives the runner.
    - The markdown table to `$RUNNER_TEMP/watcher-report.md`. Consumer: the
      "Read watcher run report" step in both watcher workflows, which hands
      it to `.github/actions/failure-issue` as `details`. It is a separate
      file, not `$GITHUB_STEP_SUMMARY`, because THAT path is per-step: a later
      step reading it gets a fresh empty file. The reader did exactly that
      until 2026-09-05 and every issue went out without the table.
    - The same table appended to `$GITHUB_STEP_SUMMARY` when set, for this
      step's own summary tab. No consumer reads it back; it is display.

    `$RUNNER_TEMP` is present in every Actions job and absent locally; with
    neither it nor `$WATCHER_REPORT_PATH` set the file writes are skipped
    rather than landing in whatever directory happens to be current.

    Called once, near the end of `main()`, after every order has been
    evaluated and every push attempted — the report describes the FINAL
    state, not a running total. Never called on a dry run: nothing was
    written, so there is nothing to report `write_run_report` did not
    already skip by construction (dry runs return out of `main()` before
    this is ever reached).

    Failures here are logged and swallowed: a reporting side-channel must
    never be the reason the run's own exit code changes.
    """
    runner_temp = os.environ.get(RUNNER_TEMP_ENV)
    report_path = os.environ.get(REPORT_PATH_ENV)
    if not report_path:
        report_path = (
            str(Path(runner_temp) / REPORT_JSON_FILENAME) if runner_temp else None
        )
    table = _report_markdown_table(entries)

    if report_path:
        payload = {
            "generated_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
            "orders": entries,
        }
        try:
            Path(report_path).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
            )
        except OSError as exc:
            logger.warning("Could not write watcher report to %s: %s", report_path, exc)

    if runner_temp:
        md_path = Path(runner_temp) / REPORT_MD_FILENAME
        try:
            md_path.write_text(table, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write watcher report to %s: %s", md_path, exc)

    summary_path = os.environ.get(STEP_SUMMARY_ENV)
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(table)
        except OSError as exc:
            logger.warning("Could not append to %s: %s", STEP_SUMMARY_ENV, exc)


def _process_channel(
    pending: list[Order],
    now: datetime,
    today: date,
    portfolio_manager: PortfolioManager | None,
    summary: dict,
    *,
    inbox_dir: Path | None = None,
    pending_dir: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Process one channel's pending orders into the shared summary.

    ``inbox_dir`` / ``pending_dir`` select the channel: None → public channel
    (byte-identical legacy behavior); MANAGER_INBOX_DIR / MANAGER_PENDING_DIR →
    the isolated Manager channel. The same ``portfolio_manager`` serves both — it
    is keyed by ``order.agent_id`` (the Manager book lives at
    data/portfolios/the-manager).

    ``dry_run`` keeps the EVALUATION identical — every order is still tested for
    expiry, still priced, still run through `evaluate_trigger`, and still
    counted into the summary — and stops immediately before the first mutation
    on each branch. Nothing is appended, deleted, mutated or committed, and
    `execute_triggered_order` is never called: it is the broker, and it moves
    the book. The report is written to the log instead.
    """
    # Late binding: tests monkeypatch `engine.triggers.get_current_price` so we
    # must call it through the module attribute, not the imported name.
    from engine import triggers as _triggers

    for order in pending:
        if is_expired(order, today):
            if dry_run:
                summary["expired"] += 1
                logger.info("DRY-RUN would EXPIRE %s", _describe(order, None))
                continue
            f = Fill(
                order_id=order.order_id,
                ts_filled=now,
                status="rejected",
                fill_price=None,
                fill_currency=None,
                notional_base=None,
                fees=None,
                reason="TRIGGER_EXPIRED",
                trigger_fired=True,
            )
            append_fill(today, f, inbox_dir=inbox_dir)
            delete_pending(order.order_id, pending_dir=pending_dir)
            summary["expired"] += 1
            # `commit` is resolved once, in main(), after the batched tail
            # commit that ALL of a run's expiries share — see `_report_entry`.
            summary["report"].append(_report_entry(order, "expired", None, f))
            continue

        price = _triggers.get_current_price(order.ticker, today=today)
        if price is None:
            summary["carried"] += 1
            continue
        if not evaluate_trigger(price, order.trigger):
            summary["carried"] += 1
            continue

        if dry_run:
            # The trigger IS hit. Stop here, before the broker:
            # execute_triggered_order runs apply_trade and moves the book.
            summary["fired"] += 1
            logger.info("DRY-RUN would FIRE %s", _describe(order, price))
            continue

        # Trigger hit — execute through the broker safety rails. Pass inbox_dir so
        # the idempotency scan is scoped to the correct channel.
        try:
            f = execute_triggered_order(
                order, today, portfolio_manager, fire_price=price, inbox_dir=inbox_dir
            )
        except Exception as exc:
            logger.exception(
                "execute_triggered_order failed for %s: %s", order.order_id, exc
            )
            summary["errors"] += 1
            # The report has to carry this or it contradicts the issue that
            # cites it — see `_report_entry`. The price IS known (the trigger
            # was evaluated against it), so it is reported; there is no fill.
            summary["report"].append(
                _report_entry(
                    order,
                    "error",
                    price,
                    None,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        # Report entry now, so the caller of `_commit` (below) has something
        # to attach an outcome to; a fired order's commit resolves within THIS
        # order's own push, unlike an expired order's shared tail commit.
        entry = _report_entry(order, "fired", price, f)
        summary["report"].append(entry)

        # Per-fire durability: commit immediately after this order's mutation set.
        # If f is None the order was already in the inbox (idempotency skip);
        # process_fired_order cleans up the zombie pending file and commits that.
        def _commit(order_id: str, on: date, paths: list[str]) -> None:
            # The RAISE path counts as well. `_git_add_commit` runs git-add and
            # git-commit under check=True, and process_fired_order swallows any
            # exception so the batch continues — so a failed commit used to be
            # invisible here while a failed push was not. That case is strictly
            # worse: the fill is on disk with no commit at all.
            try:
                outcome = _git_add_commit(order_id, on, paths)
            except Exception:
                # git-add / git-commit run under check=True, so this is a
                # COMMIT failure: nothing was created to be carried later.
                summary["commit_failures"] += 1
                entry["commit"] = REPORT_COMMIT_STRANDED
                raise
            entry["commit"] = _report_commit_label(outcome)
            if outcome == PUSHED_FALLBACK:
                summary["fallback_pushes"] += 1
            elif outcome == PUSH_FAILED:
                summary["push_failures"] += 1

        process_fired_order(
            order,
            f,
            today,
            _commit,
            inbox_dir=inbox_dir,
            pending_dir=pending_dir,
        )
        summary["fired"] += 1


def run(
    now: datetime,
    portfolio_manager: PortfolioManager | None,
    *,
    crypto_only: bool = False,
    dry_run: bool = False,
) -> dict:
    """Process all pending orders across all channels. Returns a summary dict.

    All channels are processed into ONE summary: the public channel (default
    dirs) first, then one isolated channel per allocator (pending_dir →
    inbox_dir derived from channels_prefix). Allocator fills never reach the
    public inbox the site joins by order_id. If there are no allocators the
    loop runs zero times — correct opt-out for forks without an allocator role.

    portfolio_manager may be None ONLY during blackout (we short-circuit before use).

    ``crypto_only`` restricts the sweep to tickers `is_crypto_ticker` accepts.
    This exists for cadence, not correctness: crypto prices are a live 24/7 ccxt
    fetch, so those orders are the only ones whose evaluation changes intraday.
    Everything else reads `latest_close_on_or_before` from the OHLCV store that
    fetch-ohlcv.yml refreshes once a day, so re-checking it hourly re-reads
    byte-identical data. The two cadences are split across check-triggers.yml
    (daily, full) and check-triggers-crypto.yml (hourly, crypto).

    Filtering happens here rather than in `_process_channel` so a skipped order
    is never touched at all — in particular it is NOT expired. Expiry is
    date-based (`is_expired`), so the daily full sweep remains its sole owner
    and an hourly crypto pass cannot retire a non-crypto order early.

    ``dry_run`` makes the whole sweep read-only: every order in every channel is
    still evaluated and counted, and nothing is written. `main()` used to hold
    this flag alone, AFTER this function had returned, which is how a
    `--dry-run` pushed three real fills to main on 2026-09-05. The money path
    starts here, so the flag has to start here too. The summary carries
    ``dry_run`` so a reader of the log can tell a rehearsal from a real run.
    """
    from engine import orders as _orders
    from engine import triggers as _triggers

    # One fallback branch per run, named lazily on the first refused push.
    _reset_fallback()

    summary = {
        "blacked_out": False,
        # Present on every summary, blackout included, so a log line can never
        # be read as a real run when it was a rehearsal (or the reverse).
        "dry_run": dry_run,
        "checked": 0,
        "fired": 0,
        "expired": 0,
        "carried": 0,
        "errors": 0,
        # Per-fire commits that origin/main refused and that reached the run's
        # `triggers/*` branch instead. The fill is DURABLE — it is on origin,
        # and auto-merge-session.yml takes it to main — so this is a warning,
        # not a failure: main() exits 0 and prints a ::warning:: line naming
        # the branch. See tests/test_check_triggers.TestRefusedPushFallsBack.
        "fallback_pushes": 0,
        # Per-fire commits whose push reached NEITHER main nor the fallback
        # branch (after the rebase-retry). The fill is on disk and in a local
        # commit that dies with the runner; the pending file still exists on
        # origin, so the next evaluation legitimately re-fires it and the
        # system self-heals. What does NOT self-heal is nobody knowing — hence
        # main()'s non-zero exit. See the class docstring of
        # tests/test_check_triggers.TestFailedPushExitsNonZero.
        "push_failures": 0,
        # Distinct from push_failures on purpose: a commit that could not be
        # created leaves the mutation on disk and in nothing at all, so it can
        # never be cleared by the published-state check below.
        "commit_failures": 0,
        # Per-order entries for the run report (`write_run_report`): one per
        # fired or expired order, in the vocabulary `_report_entry` builds. A
        # fired order's `commit` is resolved as soon as its own push is; an
        # expired order's is resolved once, in main(), against the batched
        # tail commit all of a run's expiries share.
        "report": [],
    }
    if in_blackout(now):
        summary["blacked_out"] = True
        logger.info("In blackout window %s — skipping.", now.time().isoformat())
        return summary

    today = now.date()

    cfg = get_config()

    def _select(orders: list[Order]) -> list[Order]:
        if not crypto_only:
            return orders
        return [o for o in orders if _triggers.is_crypto_ticker(o.ticker)]

    # Public channel — default dirs, byte-identical legacy behavior.
    public_pending = _select(list_pending())
    # Allocator channels — one isolated pending/inbox per allocator; fills
    # never reach the public inbox the site joins by order_id.
    # For William (sole allocator the-manager, prefix "manager") this produces
    # the same manager-pending / manager-inbox paths as before.
    allocator_channels = [
        (
            cfg.allocator_spec(aid).channels_prefix,
            _select(
                list_pending(
                    pending_dir=_triggers.allocator_channel_dir(
                        cfg.allocator_spec(aid).channels_prefix, "pending"
                    )
                )
            ),
        )
        for aid in cfg.allocators
    ]
    summary["checked"] = len(public_pending) + sum(
        len(pl) for _, pl in allocator_channels
    )

    _process_channel(
        public_pending, now, today, portfolio_manager, summary, dry_run=dry_run
    )
    for prefix, pending in allocator_channels:
        _process_channel(
            pending,
            now,
            today,
            portfolio_manager,
            summary,
            inbox_dir=_orders.allocator_channel_dir(prefix, "inbox"),
            pending_dir=_triggers.allocator_channel_dir(prefix, "pending"),
            dry_run=dry_run,
        )

    return summary


def refresh_leaderboard_artifact(trigger: str, on: date) -> None:
    """Best-effort refresh of data/leaderboard/current.json after a fire.

    Wrapped: any failure here is logged but never raised. The fill is the
    critical bit; the leaderboard is derived state and resyncs on the next
    fire / weekend refresh / weekday session.
    """
    try:
        summaries = _build_portfolio_summaries()
        rows = _build_leaderboard_rows(summaries, on=on)
        leaderboard_dir = get_config().leaderboard_dir
        leaderboard_dir.mkdir(parents=True, exist_ok=True)
        path = leaderboard_dir / "current.json"
        now_iso = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        artifact = {"updated_at": now_iso, "trigger": trigger, "rows": rows}
        path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")
        logger.info("Refreshed %s after fire (rows=%d)", path, len(rows))
    except Exception as exc:
        logger.warning("Leaderboard refresh failed (non-fatal): %s", exc)


def commit_and_push() -> str:
    """Commit any remaining data/orders/ and data/leaderboard/ changes and push.

    Returns COMMIT_OK when there was nothing to do or main took the push,
    COMMIT_FAILED when no commit could be created at all, PUSHED_FALLBACK when
    the commit reached only the run's `triggers/*` branch, PUSH_FAILED when it
    reached neither main nor that branch.

    Called at end of run for: expired-order sweeps (batched, recoverable) and
    the leaderboard artifact. Per-fire commits for triggered orders are handled
    individually inside process_fired_order / _git_add_commit.
    """
    cfg = get_config()
    data_dirs = [
        str(_PROJECT_ROOT / "data" / "orders" / "pending"),
        str(_PROJECT_ROOT / "data" / "orders" / "inbox"),
    ]
    # Append one pending+inbox pair per allocator. For William (sole allocator
    # the-manager, prefix "manager") this produces the same manager-pending /
    # manager-inbox entries as before — byte-identical.
    for aid in cfg.allocators:
        prefix = cfg.allocator_spec(aid).channels_prefix
        data_dirs.append(str(_PROJECT_ROOT / "data" / "orders" / f"{prefix}-pending"))
        data_dirs.append(str(_PROJECT_ROOT / "data" / "orders" / f"{prefix}-inbox"))
    data_dirs.extend(
        [
            str(_PROJECT_ROOT / "data" / "portfolios"),
            str(_PROJECT_ROOT / "data" / "leaderboard"),
        ]
    )
    # A failing git-add or git-commit is reported, not raised: main() decides
    # the run's verdict last, and a traceback here would kill the process
    # before it got there — losing the distinction between "nothing landed"
    # and "the batch finished with one stranded commit".
    try:
        subprocess.run(["git", "add", *data_dirs], cwd=_PROJECT_ROOT, check=True)
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=_PROJECT_ROOT,
        )
        if diff.returncode == 0:
            logger.info("No trigger changes to commit.")
            return COMMIT_OK
        msg = (
            f"chore(triggers): execute fired/expired conditions "
            f"{date.today().isoformat()}"
        )
        subprocess.run(["git", "commit", "-m", msg], cwd=_PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Tail commit failed (%s); the mutation is on disk and in no commit.",
            exc,
        )
        return COMMIT_FAILED
    return _push_head("tail (leaderboard/expired)")


def _nothing_is_stranded() -> bool:
    """True when every local commit is already on origin/main.

    A push pushes the whole branch, so a later order's successful push carries
    an earlier order's rejected commit with it. Trusting the failure counter
    alone therefore files "commits could not be pushed" when nothing is
    stranded — noise on the one alert channel that has to stay trustworthy.

    So the verdict is taken from the published state rather than from a proxy
    for it, the same reasoning as the baseline-freshness guard: a count of past
    failures cannot answer "is anything stranded now", and the remote can.

    Two conditions, because "unpushed" is only half of "stranded":

    1. nothing the watcher writes is still uncommitted in the worktree, and
    2. every local commit is already on origin/main — OR on this run's
       `triggers/*` fallback branch, which is on origin too and which
       auto-merge-session.yml takes to main. A commit that reached the
       branch is durable; it is a warning (`fallback_pushes`), not stranded.

    Condition 1 is what stops this from MASKING the failure it sits next to. If
    git-commit is broken outright, nothing gets committed, HEAD is trivially an
    ancestor of origin/main, and an unqualified ancestor test would report all
    clear on a run whose fill exists only on the runner's disk. The caller also
    keeps commit failures in their own counter, which this cannot clear; this
    is the second lock on the same door.

    Fails CLOSED — any error here reports the failure rather than clearing it.
    """
    try:
        dirty = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                str(_PROJECT_ROOT / "data" / "orders"),
                str(_PROJECT_ROOT / "data" / "portfolios"),
                str(_PROJECT_ROOT / "data" / "leaderboard"),
            ],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if dirty.returncode != 0 or (dirty.stdout or "").strip():
            logger.warning(
                "Watcher-written paths are still uncommitted: %s",
                (dirty.stdout or "").strip()[:300] or "could not read git status",
            )
            return False
        fetched = subprocess.run(["git", "fetch", "origin", "main"], cwd=_PROJECT_ROOT)
        if fetched.returncode != 0:
            return False
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"],
            cwd=_PROJECT_ROOT,
        )
        if ancestor.returncode == 0:
            return True
        # Not on main. On the fallback branch counts too — but only a branch
        # THIS run actually pushed to, re-fetched, never assumed: a branch name
        # that was generated and never reached origin proves nothing.
        fb = _fallback
        if fb is None or not fb.pushed:
            return False
        fetched = subprocess.run(["git", "fetch", "origin", fb.name], cwd=_PROJECT_ROOT)
        if fetched.returncode != 0:
            return False
        on_branch = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD"],
            cwd=_PROJECT_ROOT,
        )
        return on_branch.returncode == 0
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Could not check whether commits are stranded: %s", exc)
        return False


def unmerged_fallback_branches() -> list[str] | None:
    """`triggers/*` branches origin still holds, or None when it cannot be read.

    A fallback branch that still exists is a previous run's fills that have
    NOT reached main — its merge is queued, or it failed loudly and a human
    has the issue. A run that evaluates against main in that state sees the
    pending file back and the inbox row absent, and RE-FIRES the same order
    at a new price: exactly the double-execution the fallback exists to
    prevent, one merge-failure away. So main() refuses to evaluate while any
    such branch exists. Fails closed: an unreadable remote answers None and
    the caller treats it as a refusal too.
    """
    out = subprocess.run(
        [
            "git",
            "ls-remote",
            "--heads",
            "origin",
            f"refs/heads/{FALLBACK_BRANCH_PREFIX}*",
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return None
    names: list[str] = []
    for line in (out.stdout or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].startswith("refs/heads/"):
            names.append(parts[1][len("refs/heads/") :])
    return names


def main_tip_ahead_of_checkout() -> str | None:
    """origin/main's sha when it differs from HEAD, "" when it matches, None on error.

    `unmerged_fallback_branches()` asks origin a question about BRANCHES; this
    asks it about the tip the run is about to evaluate. Both watcher workflows
    check out with `fetch-depth: 1` at job start, so what `run()` reads is
    whatever main was at that moment — and nothing serialises that against
    auto-merge-session, whose concurrency group is per branch while the
    watchers share `check-triggers`. Round-2 review, 2026-09-05: a crypto run
    queued behind the daily sweep checks out a main that still lacks a fill;
    the merge lands and DELETES the fallback branch a few seconds later, so
    the branch guard above sees a clean origin and passes; the run then fires
    the already-filled order a second time at that moment's price. The rebase
    conflict on `portfolio.json` was the only thing that kept the duplicate
    off main — luck, not a guard.

    ONE `ls-remote`, no fetch: a `git fetch origin main` into a depth-1
    checkout deepens it, and this repo's history is measured in gigabytes. The
    workflows re-anchor the checkout (fetch --depth=1 + reset --hard) right
    after they merge any waiting fallback branch, so on a healthy run this
    compares two identical shas and costs one remote round-trip.

    Any difference refuses, without asking which paths moved. The window this
    covers is checkout → guard, i.e. seconds, so an unrelated writer landing
    inside it is rare enough that refusing is cheaper than being clever about
    it — and "which paths moved" cannot be answered without the fetch this
    exists to avoid.

    Fails closed, like its sibling: an unreadable origin answers None and the
    caller refuses. A run that refuses loses one cycle; a run that fires twice
    is a restatement.
    """
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "refs/heads/main"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if remote.returncode != 0:
        return None
    parts = (remote.stdout or "").split()
    if len(parts) < 2 or parts[1] != "refs/heads/main":
        return None
    remote_sha = parts[0]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        return None
    return "" if (head.stdout or "").strip() == remote_sha else remote_sha


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Conditional-order watcher.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Evaluate every pending order and report what would fire or "
            "expire, writing nothing: no fill, no inbox row, no pending "
            "delete, no commit, no push, no leaderboard refresh."
        ),
    )
    parser.add_argument(
        "--crypto-only",
        action="store_true",
        help=(
            "Only evaluate crypto tickers (live 24/7 ccxt price). Non-crypto "
            "orders are left untouched, including expiry — the daily full "
            "sweep owns those. See run() for why the cadences are split."
        ),
    )
    args = parser.parse_args()

    portfolio_manager = PortfolioManager(base_dir=get_config().portfolios_dir)
    now = datetime.now(timezone.utc)

    # A previous run's fills that have not reached main yet make main a stale
    # view of the ledger: evaluating against it would re-fire those orders at
    # a new price. Refuse, loudly (exit 1 → failure-issue), until the branch is
    # merged or a human has dealt with it. A dry run only reports it — it
    # writes nothing, so it cannot re-fire anything.
    stale = unmerged_fallback_branches()
    if stale is None or stale:
        detail = (
            "could not read origin's branch list"
            if stale is None
            else ", ".join(sorted(stale))
        )
        if args.dry_run:
            logger.warning(
                "Unmerged fallback branch(es) on origin (%s) — a real run "
                "would refuse to evaluate until they reach main.",
                detail,
            )
        else:
            logger.error(
                "Refusing to evaluate: unmerged fallback branch(es) on origin "
                "(%s). Their fills are not on main, so evaluating now would "
                "re-fire those orders at a new price. Wait for "
                "auto-merge-session.yml, or merge/inspect the branch by hand.",
                detail,
            )
            sys.exit(1)

    # ...and the same question about the tip itself. The branch list above was
    # read from origin a moment ago; the CHECKOUT is older and may never have
    # been refreshed, so a merge that landed in between leaves a clean origin
    # and a stale worktree — the fill absent, the pending file present, and
    # this run about to re-fire it. See `main_tip_ahead_of_checkout`.
    ahead = main_tip_ahead_of_checkout()
    if ahead is None or ahead:
        detail = "could not read origin/main" if ahead is None else f"now at {ahead}"
        if args.dry_run:
            logger.warning(
                "origin/main has moved since this checkout (%s) — a real run "
                "would refuse to evaluate against it.",
                detail,
            )
        else:
            logger.error(
                "Refusing to evaluate: origin/main moved since this checkout "
                "(%s). This working tree no longer shows what main holds, so "
                "an order already filled there would be fired again at a new "
                "price. The next run checks out the new tip.",
                detail,
            )
            sys.exit(1)

    summary = run(
        now=now,
        portfolio_manager=portfolio_manager,
        crypto_only=args.crypto_only,
        dry_run=args.dry_run,
    )
    logger.info("Watcher summary: %s", summary)

    if args.dry_run:
        # run() already refused every mutation; this return additionally keeps
        # the tail (leaderboard refresh + commit_and_push) out of the run. The
        # exit-code logic below is deliberately skipped too: there is nothing
        # to strand when nothing was written, so a dry run is always green.
        logger.info(
            "Dry-run — nothing was written: would fire %d, would expire %d, "
            "carried %d, checked %d.",
            summary["fired"],
            summary["expired"],
            summary["carried"],
            summary["checked"],
        )
        return
    if summary["blacked_out"]:
        return

    # Every order has already been evaluated and every push already attempted;
    # the exit code is decided last, on purpose. Aborting mid-batch on the first
    # failed push would strand the orders behind it unevaluated, which is a
    # worse failure than the one being reported.
    push_failed = summary["push_failures"] > 0
    commit_failed = summary["commit_failures"] > 0
    if summary["fired"] > 0 or summary["expired"] > 0:
        if summary["fired"] > 0:
            refresh_leaderboard_artifact(trigger="trigger-fire", on=now.date())
        outcome = commit_and_push()
        # All of a run's expiries share this ONE commit — resolve every
        # still-unresolved expired entry against it now that it is known.
        tail_label = _report_commit_label(outcome)
        for entry in summary["report"]:
            if entry["kind"] == "expired" and entry["commit"] is None:
                entry["commit"] = tail_label
        if outcome == COMMIT_FAILED:
            commit_failed = True
        elif outcome == PUSHED_FALLBACK:
            summary["fallback_pushes"] += 1
        elif outcome == PUSH_FAILED:
            push_failed = True

    # ONLY a push failure may be cleared, and only by the published state. A
    # commit failure means there is no commit for a later push to carry.
    if push_failed and not commit_failed and _nothing_is_stranded():
        # A later order's push carried the earlier failure's commit to main
        # or to the fallback branch.
        logger.info(
            "A push failed earlier, but every local commit is now on origin "
            "(main or the fallback branch) — nothing is stranded."
        )
        push_failed = False

    # Written now, after every push has been attempted and self-heal has had
    # its say, so the report describes the FINAL state — see
    # `_report_commit_label` for the one gap in that guarantee.
    write_run_report(summary["report"], now=now)

    # `errors` is execute_triggered_order raising on a FIRED trigger: the worst
    # failure on this path, and it exited 0. The pending file survives and the
    # order is retried, so nothing is lost — but a broker that refuses an order
    # by crashing, hourly and forever, is exactly what has to reach a human.
    if push_failed or commit_failed or summary["errors"] > 0:
        # A warning inside a green run reached nobody. Both watchers route
        # failure through .github/actions/failure-issue, and a non-zero exit is
        # what that consumer reads.
        logger.error(
            "Watcher run did not fully succeed (uncommitted: %s, stranded "
            "commits: %s, order errors: %d). Exiting 1 so the failure is "
            "reported.",
            commit_failed,
            push_failed,
            summary["errors"],
        )
        sys.exit(1)

    if summary["fallback_pushes"] > 0:
        # Durable but not yet on main: a WARNING inside a green run, on
        # purpose. Exit 1 here would file the "did not complete" issue for a
        # run that completed and lost nothing, and would teach the reader that
        # the issue means nothing. The ::warning:: form annotates the Actions
        # run; auto-merge-session.yml files its own issue if the merge fails.
        branch = _fallback.name if _fallback is not None else "?"
        line = (
            f"origin/main refused {summary['fallback_pushes']} push(es); the "
            f"commits are on fallback branch {branch} and "
            "auto-merge-session.yml takes them to main. Nothing is lost."
        )
        logger.warning(line)
        print(f"::warning::{line}", flush=True)


if __name__ == "__main__":
    main()
