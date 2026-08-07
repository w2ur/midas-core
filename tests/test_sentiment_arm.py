"""Tests for the sentiment A/B arm check (2026-08-07 review, W3.5).

The pre-registered A/B (METHODOLOGY.md) gives two agents a committed news
feed and eight agents nothing. The feed is collected by a scheduled job and
read from disk by the agents, so "which arm ran" is a property of *timing*,
not of code — and until this check existed, a session that read yesterday's
headlines was indistinguishable from one that read today's.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from engine.config import get_config
from scripts.daily_session import step_check_sentiment_freshness

_TODAY = date(2026, 8, 7)


@pytest.fixture
def treatment_desk(midas_data_root: Path) -> Path:
    """A desk with exactly one agent in the treatment arm.

    The arm-mechanics tests must not depend on the live cast declaring the
    experiment: on the demo desk (and in the core mirror) no agent does, and
    every one of them would otherwise assert `not-running`. Seeding the arm
    here keeps the mechanics hermetic and gives forks the same coverage.
    """
    import yaml

    from engine.config import reset_config_cache

    roster = midas_data_root / "roster.yaml"
    raw = yaml.safe_load(roster.read_text(encoding="utf-8"))
    for spec in raw["agents"].values():
        spec.pop("sentiment_arm", None)
    first = next(iter(raw["agents"]))
    raw["agents"][first]["sentiment_arm"] = "treatment"
    roster.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    reset_config_cache()
    return midas_data_root


def _digest(news_dir: Path, symbol: str, dates: list[str]) -> None:
    news_dir.mkdir(parents=True, exist_ok=True)
    (news_dir / f"{symbol}.jsonl").write_text(
        "".join(
            json.dumps({"date": d, "items": [{"title": f"headline {d}"}]}) + "\n"
            for d in dates
        ),
        encoding="utf-8",
    )


class TestArmDetection:
    def test_same_day_digest_is_the_treatment_arm(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        news = tmp_path / "news"
        _digest(news, "BTC-EUR", ["2026-08-06", "2026-08-07"])

        record = step_check_sentiment_freshness(
            _TODAY, news_dir=news, log_path=tmp_path / "arm.jsonl"
        )
        assert record["arm"] == "treatment"
        assert record["latest_digest"] == "2026-08-07"
        assert record["age_days"] == 0

    def test_previous_day_digest_is_a_confounded_arm(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        """The condition that was live at every session before 2026-08-07:
        the collector committed after the session realigned."""
        news = tmp_path / "news"
        _digest(news, "BTC-EUR", ["2026-08-05", "2026-08-06"])

        record = step_check_sentiment_freshness(
            _TODAY, news_dir=news, log_path=tmp_path / "arm.jsonl"
        )
        assert record["arm"] == "degraded-to-control"
        assert record["age_days"] == 1

    def test_missing_news_directory_is_reported_not_raised(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        record = step_check_sentiment_freshness(
            _TODAY, news_dir=tmp_path / "absent", log_path=tmp_path / "arm.jsonl"
        )
        assert record["arm"] == "degraded-to-control"
        assert record["latest_digest"] is None
        assert record["age_days"] is None

    def test_unreadable_digest_does_not_take_down_the_check(
        self, treatment_desk: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        news = tmp_path / "news"
        _digest(news, "BTC-EUR", ["2026-08-07"])
        (news / "BROKEN.jsonl").write_text("{not json\n", encoding="utf-8")

        record = step_check_sentiment_freshness(
            _TODAY, news_dir=news, log_path=tmp_path / "arm.jsonl"
        )
        assert record["arm"] == "treatment"
        assert "BROKEN.jsonl" in capsys.readouterr().out

    def test_never_aborts_the_session(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        """Explicit: losing a session over a news feed is worse than running
        a confounded arm and saying so."""
        step_check_sentiment_freshness(
            _TODAY, news_dir=tmp_path / "absent", log_path=tmp_path / "arm.jsonl"
        )  # must not raise


class TestArmLog:
    def test_appends_one_row_per_session_date(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        news = tmp_path / "news"
        log = tmp_path / "arm.jsonl"
        _digest(news, "BTC-EUR", ["2026-08-06"])
        step_check_sentiment_freshness(date(2026, 8, 6), news_dir=news, log_path=log)
        _digest(news, "BTC-EUR", ["2026-08-06", "2026-08-07"])
        step_check_sentiment_freshness(_TODAY, news_dir=news, log_path=log)

        rows = [json.loads(x) for x in log.read_text().splitlines()]
        assert [r["session_date"] for r in rows] == ["2026-08-06", "2026-08-07"]
        assert [r["arm"] for r in rows] == ["treatment", "treatment"]

    def test_rerunning_a_session_replaces_its_own_row(
        self, treatment_desk: Path, tmp_path: Path
    ) -> None:
        news = tmp_path / "news"
        log = tmp_path / "arm.jsonl"
        step_check_sentiment_freshness(_TODAY, news_dir=news, log_path=log)
        _digest(news, "BTC-EUR", ["2026-08-07"])
        step_check_sentiment_freshness(_TODAY, news_dir=news, log_path=log)

        rows = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["arm"] == "treatment"


@pytest.mark.live_cast
class TestTreatmentRosterParity:
    """`roster.yaml` declares the arm; the persona files are what the agents
    actually read. Two places, so they can drift — and a drift here is not a
    bug that shows up as a crash, it is an experiment quietly measuring a
    different treatment group than the one it reports."""

    @staticmethod
    def _agents_dir() -> Path:
        return Path(get_config().data_dir) / ".claude" / "agents"

    def test_roster_arm_matches_the_personas_that_read_the_feed(self) -> None:
        agents_dir = self._agents_dir()
        if not agents_dir.exists():  # demo desk / core mirror
            pytest.skip("no live persona directory at this data root")

        reads_feed = {
            path.stem
            for path in sorted(agents_dir.glob("*.md"))
            if "data/market/news/" in path.read_text(encoding="utf-8")
        }
        assert reads_feed == set(get_config().sentiment_treatment)

    def test_the_probe_can_fail(self) -> None:
        """Control for the test above: the grep it relies on must actually
        distinguish personas, not match every file in the directory."""
        agents_dir = self._agents_dir()
        if not agents_dir.exists():
            pytest.skip("no live persona directory at this data root")
        all_personas = {p.stem for p in agents_dir.glob("*.md")}
        assert len(all_personas) > len(get_config().sentiment_treatment) > 0


class TestNoTreatmentDesk:
    def test_a_desk_with_no_treatment_arm_is_a_no_op(
        self, midas_data_root: Path, tmp_path: Path
    ) -> None:
        """Core ships this helper to forks whose roster declares no arm. It
        must not report a confounded experiment nobody is running."""
        from engine.config import reset_config_cache

        roster = midas_data_root / "roster.yaml"
        roster.write_text(
            "\n".join(
                ln
                for ln in roster.read_text(encoding="utf-8").splitlines()
                if "sentiment_arm:" not in ln
            )
            + "\n",
            encoding="utf-8",
        )
        reset_config_cache()
        assert get_config().sentiment_treatment == ()

        record = step_check_sentiment_freshness(
            _TODAY,
            news_dir=tmp_path / "absent",
            log_path=tmp_path / "arm.jsonl",
        )
        assert record["arm"] == "not-running"
        assert record["agents"] == []
        # And it must not create an arm log on a desk with no experiment.
        assert not (tmp_path / "arm.jsonl").exists()


@pytest.mark.live_cast
class TestCollectorSchedule:
    def test_cron_runs_before_the_session(self) -> None:
        """A 19:00 nominal cron committed at 20:08-20:39 — after the 20:00
        session realigned. The hour is load-bearing, so it is pinned."""
        wf = (
            Path(get_config().data_dir)
            / ".github"
            / "workflows"
            / "fetch-sentiment.yml"
        )
        if not wf.exists():
            pytest.skip("workflow not present at this data root")
        crons = re.findall(r'cron:\s*"(\d+)\s+(\d+)\s', wf.read_text(encoding="utf-8"))
        assert crons, "fetch-sentiment has no schedule"
        for _minute, hour in crons:
            assert int(hour) <= 18, (
                "the sentiment collector must be scheduled early enough that "
                "its commit lands before the 20:00 session; measured drift "
                "from nominal to committed is ~1h10m-1h40m"
            )
