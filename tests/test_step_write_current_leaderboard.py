import json

from engine.config import get_config


def test_step_write_current_leaderboard_writes_file(midas_data_root):
    from scripts import daily_session

    leaderboard_dir = get_config().leaderboard_dir
    leaderboard_dir.mkdir(parents=True)

    rows = [{"rank": 1, "agent": "a", "return_pct": 5.0}]
    daily_session.step_write_current_leaderboard(
        rows=rows,
        trigger="session-2026-05-22",
    )

    path = leaderboard_dir / "current.json"
    payload = json.loads(path.read_text())
    assert payload["trigger"] == "session-2026-05-22"
    assert payload["rows"] == rows
    assert payload["updated_at"].endswith("Z")


def test_step_write_current_leaderboard_creates_dir_if_missing(midas_data_root):
    """Idempotency: the step works even if data/leaderboard/ doesn't pre-exist."""
    from scripts import daily_session

    daily_session.step_write_current_leaderboard(
        rows=[],
        trigger="session-2026-05-22",
    )
    assert (get_config().leaderboard_dir / "current.json").exists()
