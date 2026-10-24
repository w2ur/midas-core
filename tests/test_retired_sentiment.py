"""The sentiment A/B is gone; what it left behind must stay harmless.

The collector, the arm check and the config's `sentiment_arm` field were deleted
(plan task A4) after the A/B ended unconcluded on 2026-10-09. Its data
(`data/market/sentiment_arm.jsonl`, `data/market/news/`) stays committed as the
record, and a fork's roster may still declare the old key.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from engine.config import get_config, reset_config_cache


def test_a_roster_that_still_declares_sentiment_arm_loads(
    midas_data_root: Path,
) -> None:
    """An older fork's roster carries the key; the loader must ignore it, not
    refuse the whole desk."""
    roster = midas_data_root / "roster.yaml"
    raw = yaml.safe_load(roster.read_text(encoding="utf-8"))
    first = next(iter(raw["agents"]))
    raw["agents"][first]["sentiment_arm"] = "treatment"
    roster.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    reset_config_cache()

    spec = get_config().roster[first]

    assert spec.id == first
    assert not hasattr(spec, "sentiment_arm")


def _personas_reading_the_feed(agents_dir: Path) -> set[str]:
    return {
        path.stem
        for path in sorted(agents_dir.glob("*.md"))
        if "data/market/news/" in path.read_text(encoding="utf-8")
    }


@pytest.mark.live_cast
def test_no_persona_reads_the_retired_news_feed() -> None:
    """Nothing writes `data/market/news/` any more, so a persona that still
    tells its agent to read it sends the agent to a frozen directory."""
    agents_dir = Path(get_config().data_dir) / ".claude" / "agents"
    if not agents_dir.exists():  # demo desk / core mirror
        pytest.skip("no live persona directory at this data root")

    assert _personas_reading_the_feed(agents_dir) == set()


def test_the_persona_probe_can_fail(tmp_path: Path) -> None:
    """Control for the test above: the probe must tell a reader from a
    non-reader, not match every file or none."""
    (tmp_path / "reader.md").write_text(
        "read data/market/news/{TICKER}.jsonl", encoding="utf-8"
    )
    (tmp_path / "other.md").write_text("read your journal", encoding="utf-8")
    assert _personas_reading_the_feed(tmp_path) == {"reader"}
