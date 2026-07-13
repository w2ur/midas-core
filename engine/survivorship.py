"""Survivorship-bias detection for index-membership universes.

The committed universe files (``data/universes/*.json``) store *today's* index
constituents. A backtest whose start date predates the file's last refresh is
implicitly trading the tickers that are in the index **now** — silently
excluding names that were dropped (or did not yet exist) at the start date.
That is textbook survivorship bias: it biases returns upward because the
long-dead losers were never in the sample.

This is not hypothetical for Midas. An early factor-research run backtested
against the S&P 500's *current* membership from 2024 and reported returns
inflated by roughly **194%** versus the same run on a survivorship-free
universe. See ``METHODOLOGY.md`` (Known distortions) for the incident.

The mitigation here is a *warning*, not a correction: we cannot reconstruct
point-in-time index membership without a historical constituents feed we do
not have. The honest move is to flag the run loudly and steer callers toward
stable universes (``dow30``, ``etf-broad``) for historical backtests.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from engine.config import get_config

# Index universes whose membership drifts enough that a start date before the
# file's refresh introduces material survivorship bias. ``dow30`` is
# deliberately excluded: its 30-name membership is stable and slow-moving, and
# it is the recommended default for historical runs.
SURVIVORSHIP_PRONE_UNIVERSES: frozenset[str] = frozenset(
    {"sp500", "nasdaq100", "stoxx-600", "cac40", "dax", "ftse100"}
)

# Universe id → committed data-file stem, where they differ.
_FILE_STEM_OVERRIDES: dict[str, str] = {"stoxx-600": "stoxx600"}


def universe_last_refresh(universe_id: str) -> date | None:
    """Return the last-refresh date of a universe's committed constituents file.

    Uses the file's modification time as the refresh proxy. Returns ``None``
    when the file is absent. The universe files are refreshed out-of-band
    (``scripts/refresh_universes.py`` / the weekly cron), so the mtime is the
    best available "constituents as of" signal.
    """
    stem = _FILE_STEM_OVERRIDES.get(universe_id, universe_id)
    path = get_config().universes_dir / f"{stem}.json"
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()


def survivorship_warning(universe_id: str, start: date) -> str | None:
    """Return a SURVIVORSHIP_BIAS warning string, or ``None`` if the run is clean.

    Fires only for the survivorship-prone index universes, and only when the
    backtest ``start`` predates the constituents file's last refresh — i.e.
    when the run is trading today's membership over a historical window.
    """
    if universe_id not in SURVIVORSHIP_PRONE_UNIVERSES:
        return None
    refreshed = universe_last_refresh(universe_id)
    if refreshed is None:
        return None
    if start >= refreshed:
        return None
    return (
        f"SURVIVORSHIP_BIAS: {universe_id} constituents are as of "
        f"{refreshed.isoformat()} but the backtest starts {start.isoformat()}. "
        f"Names that left the index (or did not yet exist) before the refresh "
        f"are silently excluded, biasing returns upward — an early Midas run was "
        f"inflated ~194% this way. Prefer a stable universe (dow30, etf-broad) "
        f"for historical backtests."
    )
