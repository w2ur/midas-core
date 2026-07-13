"""Conditional-order watcher.

Runs every 15 min via .github/workflows/check-triggers.yml. Walks pending
orders, fetches current prices, fires when triggers are hit, expires old
ones. Blackout window 19:55-20:30 UTC to avoid commit-races with the
20:00 UTC daily session.

Usage:
    python scripts/check_triggers.py            # normal run
    python scripts/check_triggers.py --dry-run  # evaluate but don't commit
"""

from __future__ import annotations

import argparse
import json
import logging
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

BLACKOUT_START = time(19, 55)
BLACKOUT_END = time(20, 30)

# Type alias for the injectable committer used in process_fired_order.
# Signature: (order_id, today, paths) -> None
# paths is a list of absolute path strings to git-add before committing.
Committer = Callable[[str, date, list[str]], None]


def in_blackout(now: datetime) -> bool:
    """True if `now` (UTC) is inside the daily-session blackout window."""
    t = now.time()
    return BLACKOUT_START <= t <= BLACKOUT_END


def process_fired_order(
    order: Order,
    fill_or_none: Fill | None,
    today: date,
    committer: Committer,
    *,
    inbox_dir: Path | None = None,
    pending_dir: Path | None = None,
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
    """
    from engine import orders as _orders
    from engine import triggers as _triggers

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


def _git_add_commit(order_id: str, today: date, paths: list[str]) -> None:
    """git-add the given paths, then commit with a per-order message.

    Skips the commit if there are no staged changes (defensive — the pending
    file may already be absent if a prior run cleaned it up).

    Push strategy (failure-tolerant):
    1. Try `git push origin HEAD:main`.
    2. On failure, `git pull --rebase` then retry the push once.
    3. If the retry still fails, log a warning and continue. The commit exists
       locally; the next watcher fire or end-of-run push attempt will carry it.
    """
    subprocess.run(["git", "add", *paths], cwd=_PROJECT_ROOT, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=_PROJECT_ROOT,
    )
    if diff.returncode == 0:
        logger.info("Nothing staged for %s — skipped commit.", order_id)
        return

    msg = f"chore(triggers): execute {order_id} {today.isoformat()}"
    subprocess.run(["git", "commit", "-m", msg], cwd=_PROJECT_ROOT, check=True)

    result = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        cwd=_PROJECT_ROOT,
    )
    if result.returncode == 0:
        logger.info("Committed + pushed %s.", order_id)
        return

    # Push failed — attempt a rebase-based retry once.
    logger.warning("Push failed for %s; retrying after git pull --rebase.", order_id)
    rebase = subprocess.run(
        ["git", "pull", "--rebase", "origin", "main"],
        cwd=_PROJECT_ROOT,
    )
    if rebase.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], cwd=_PROJECT_ROOT)
        logger.warning("Rebase failed for %s; aborted, commit stays local.", order_id)
        return
    retry = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        cwd=_PROJECT_ROOT,
    )
    if retry.returncode != 0:
        logger.warning(
            "Retry push also failed for %s; commit exists locally and will be "
            "carried by the next successful push attempt.",
            order_id,
        )


def _process_channel(
    pending: list[Order],
    now: datetime,
    today: date,
    portfolio_manager: PortfolioManager | None,
    summary: dict,
    *,
    inbox_dir: Path | None = None,
    pending_dir: Path | None = None,
) -> None:
    """Process one channel's pending orders into the shared summary.

    ``inbox_dir`` / ``pending_dir`` select the channel: None → public channel
    (byte-identical legacy behavior); MANAGER_INBOX_DIR / MANAGER_PENDING_DIR →
    the isolated Manager channel. The same ``portfolio_manager`` serves both — it
    is keyed by ``order.agent_id`` (the Manager book lives at
    data/portfolios/the-manager).
    """
    # Late binding: tests monkeypatch `engine.triggers.get_current_price` so we
    # must call it through the module attribute, not the imported name.
    from engine import triggers as _triggers

    for order in pending:
        if is_expired(order, today):
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
            continue

        price = _triggers.get_current_price(order.ticker, today=today)
        if price is None:
            summary["carried"] += 1
            continue
        if not evaluate_trigger(price, order.trigger):
            summary["carried"] += 1
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
            continue

        # Per-fire durability: commit immediately after this order's mutation set.
        # If f is None the order was already in the inbox (idempotency skip);
        # process_fired_order cleans up the zombie pending file and commits that.
        process_fired_order(
            order,
            f,
            today,
            _git_add_commit,
            inbox_dir=inbox_dir,
            pending_dir=pending_dir,
        )
        summary["fired"] += 1


def run(now: datetime, portfolio_manager: PortfolioManager | None) -> dict:
    """Process all pending orders across all channels. Returns a summary dict.

    All channels are processed into ONE summary: the public channel (default
    dirs) first, then one isolated channel per allocator (pending_dir →
    inbox_dir derived from channels_prefix). Allocator fills never reach the
    public inbox the site joins by order_id. If there are no allocators the
    loop runs zero times — correct opt-out for forks without an allocator role.

    portfolio_manager may be None ONLY during blackout (we short-circuit before use).
    """
    from engine import orders as _orders
    from engine import triggers as _triggers

    summary = {
        "blacked_out": False,
        "checked": 0,
        "fired": 0,
        "expired": 0,
        "carried": 0,
        "errors": 0,
    }
    if in_blackout(now):
        summary["blacked_out"] = True
        logger.info("In blackout window %s — skipping.", now.time().isoformat())
        return summary

    today = now.date()

    cfg = get_config()

    # Public channel — default dirs, byte-identical legacy behavior.
    public_pending = list_pending()
    # Allocator channels — one isolated pending/inbox per allocator; fills
    # never reach the public inbox the site joins by order_id.
    # For William (sole allocator the-manager, prefix "manager") this produces
    # the same manager-pending / manager-inbox paths as before.
    allocator_channels = [
        (
            cfg.allocator_spec(aid).channels_prefix,
            list_pending(
                pending_dir=_triggers.allocator_channel_dir(
                    cfg.allocator_spec(aid).channels_prefix, "pending"
                )
            ),
        )
        for aid in cfg.allocators
    ]
    summary["checked"] = len(public_pending) + sum(
        len(pl) for _, pl in allocator_channels
    )

    _process_channel(public_pending, now, today, portfolio_manager, summary)
    for prefix, pending in allocator_channels:
        _process_channel(
            pending,
            now,
            today,
            portfolio_manager,
            summary,
            inbox_dir=_orders.allocator_channel_dir(prefix, "inbox"),
            pending_dir=_triggers.allocator_channel_dir(prefix, "pending"),
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


def commit_and_push() -> None:
    """Commit any remaining data/orders/ and data/leaderboard/ changes and push.

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
    subprocess.run(["git", "add", *data_dirs], cwd=_PROJECT_ROOT, check=True)
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=_PROJECT_ROOT,
    )
    if diff.returncode == 0:
        logger.info("No trigger changes to commit.")
        return
    msg = (
        f"chore(triggers): execute fired/expired conditions {date.today().isoformat()}"
    )
    subprocess.run(["git", "commit", "-m", msg], cwd=_PROJECT_ROOT, check=True)
    result = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        cwd=_PROJECT_ROOT,
    )
    if result.returncode == 0:
        logger.info("Committed + pushed tail (leaderboard/expired).")
        return

    logger.warning("Tail push failed; retrying after git pull --rebase.")
    rebase = subprocess.run(
        ["git", "pull", "--rebase", "origin", "main"],
        cwd=_PROJECT_ROOT,
    )
    if rebase.returncode != 0:
        subprocess.run(["git", "rebase", "--abort"], cwd=_PROJECT_ROOT)
        logger.warning("Tail rebase failed; aborted, commit stays local.")
        return
    retry = subprocess.run(
        ["git", "push", "origin", "HEAD:main"],
        cwd=_PROJECT_ROOT,
    )
    if retry.returncode != 0:
        logger.warning(
            "Tail retry push also failed; commit exists locally and will be "
            "carried by the next successful push attempt."
        )
    else:
        logger.info("Committed + pushed tail after rebase (leaderboard/expired).")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Conditional-order watcher.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate but don't commit."
    )
    args = parser.parse_args()

    portfolio_manager = PortfolioManager(base_dir=get_config().portfolios_dir)
    now = datetime.now(timezone.utc)
    summary = run(now=now, portfolio_manager=portfolio_manager)
    logger.info("Watcher summary: %s", summary)

    if args.dry_run:
        logger.info("Dry-run — skipping commit.")
        return
    if summary["blacked_out"]:
        return
    if summary["fired"] == 0 and summary["expired"] == 0:
        return
    if summary["fired"] > 0:
        refresh_leaderboard_artifact(trigger="trigger-fire", on=now.date())
    commit_and_push()


if __name__ == "__main__":
    main()
