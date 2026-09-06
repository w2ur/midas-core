"""Tests for engine.desk_notices and its injection into persona dispatch.

Hermetic by construction: every case builds its own notice file and its own
persona under ``midas_data_root``, and the roster ids come from the resolved
config rather than being hardcoded, so the file runs unchanged on the demo desk
in midas-core. The one test that reads this desk's committed notice file is
marked ``live_cast`` — ``data/desk_notices.json`` is live-desk state and is not
in the mirror's manifest.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from engine.config import get_config
from engine.desk_notices import active_notices, load_notices, render_notice_block
from engine.persona_dispatch import wrap_persona_prompt

WINDOW = {"from": "2026-09-05", "until": "2026-09-19"}


def _notice(**overrides) -> dict:
    base = {
        "id": "test-notice",
        "audience": "all",
        "text": "The watcher published nothing for eleven days.",
        **WINDOW,
    }
    base.update(overrides)
    return base


def _write_notices(root: Path, payload) -> Path:
    path = get_config().desk_notices_path
    # Guards the one hazard in this file: these helpers resolve paths through
    # get_config(), so a test that forgot midas_data_root (or listed it after
    # another config-reading fixture) would write into the real repo.
    assert path.is_relative_to(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_persona(root: Path, agent_id: str) -> None:
    agents = get_config().agents_dir
    assert agents.is_relative_to(root)
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{agent_id}.md").write_text(
        f"---\nname: {agent_id}\nmodel: opus\n---\nPERSONA_SENTINEL for {agent_id}.\n",
        encoding="utf-8",
    )


@pytest.fixture
def trader_id() -> str:
    return get_config().trading_roster[0]


@pytest.fixture
def narrator_id() -> str:
    return get_config().narrators[0]


# --- window -----------------------------------------------------------------


@pytest.mark.parametrize(
    "today",
    ["2026-09-05", "2026-09-12", "2026-09-19"],
    ids=["first-day", "mid-window", "last-day"],
)
def test_window_is_inclusive_on_both_ends(midas_data_root, trader_id, today):
    _write_notices(midas_data_root, [_notice()])
    assert [n.id for n in active_notices(trader_id, date.fromisoformat(today))] == [
        "test-notice"
    ]


@pytest.mark.parametrize(
    "today",
    ["2026-09-04", "2026-09-20"],
    ids=["day-before", "day-after"],
)
def test_window_excludes_outside_dates(midas_data_root, trader_id, today):
    _write_notices(midas_data_root, [_notice()])
    assert active_notices(trader_id, date.fromisoformat(today)) == []
    assert render_notice_block(trader_id, date.fromisoformat(today)) == ""


def test_only_in_window_notices_are_rendered(midas_data_root, trader_id):
    _write_notices(
        midas_data_root,
        [
            _notice(id="current"),
            _notice(id="expired", **{"from": "2026-08-01", "until": "2026-08-31"}),
        ],
    )
    block = render_notice_block(trader_id, date(2026, 9, 10))
    assert "[current]" in block
    assert "[expired]" not in block


# --- audience ---------------------------------------------------------------


def test_traders_audience_reaches_a_trader(midas_data_root, trader_id):
    _write_notices(midas_data_root, [_notice(audience="traders")])
    assert [n.id for n in active_notices(trader_id, date(2026, 9, 10))] == [
        "test-notice"
    ]


def test_traders_audience_skips_the_narrator(midas_data_root, narrator_id):
    # The Oracle narrates and holds no book, so a notice aimed at the books is
    # not aimed at it.
    _write_notices(midas_data_root, [_notice(audience="traders")])
    assert active_notices(narrator_id, date(2026, 9, 10)) == []


def test_all_audience_reaches_trader_and_narrator(
    midas_data_root, trader_id, narrator_id
):
    _write_notices(midas_data_root, [_notice(audience="all")])
    for agent_id in (trader_id, narrator_id):
        assert [n.id for n in active_notices(agent_id, date(2026, 9, 10))] == [
            "test-notice"
        ]


def test_unknown_audience_is_skipped_with_a_warning(midas_data_root, trader_id, caplog):
    _write_notices(midas_data_root, [_notice(audience="everyone")])
    with caplog.at_level(logging.WARNING, logger="engine.desk_notices"):
        assert active_notices(trader_id, date(2026, 9, 10)) == []
    assert "audience" in caplog.text


# --- missing / malformed ----------------------------------------------------


def test_missing_file_yields_no_block_and_no_warning(
    midas_data_root, trader_id, caplog
):
    # An absent file is the normal state of a desk with nothing to announce.
    assert not get_config().desk_notices_path.exists()
    with caplog.at_level(logging.WARNING, logger="engine.desk_notices"):
        assert load_notices() == []
        assert render_notice_block(trader_id, date(2026, 9, 10)) == ""
    assert caplog.text == ""


@pytest.mark.parametrize(
    "payload",
    [
        "{not json at all",
        '{"id": "solo"}',
        "[[1, 2, 3]]",
        '[{"id": "no-text", "audience": "all", "from": "2026-09-05", "until": "2026-09-19"}]',
        '[{"id": "bad-date", "audience": "all", "text": "x", "from": "yesterday", "until": "2026-09-19"}]',
        '[{"id": "backwards", "audience": "all", "text": "x", "from": "2026-09-19", "until": "2026-09-05"}]',
    ],
    ids=[
        "invalid-json",
        "object-not-list",
        "entry-not-object",
        "no-text",
        "unparseable-date",
        "end-before-start",
    ],
)
def test_malformed_file_yields_no_block_and_warns(
    midas_data_root, trader_id, caplog, payload
):
    _write_notices(midas_data_root, payload)
    with caplog.at_level(logging.WARNING, logger="engine.desk_notices"):
        assert render_notice_block(trader_id, date(2026, 9, 10)) == ""
    assert caplog.records, "a malformed notice file must log a warning"


def test_one_malformed_entry_does_not_silence_the_others(
    midas_data_root, trader_id, caplog
):
    _write_notices(midas_data_root, [{"id": 42}, _notice(id="good")])
    with caplog.at_level(logging.WARNING, logger="engine.desk_notices"):
        block = render_notice_block(trader_id, date(2026, 9, 10))
    assert "[good]" in block
    assert caplog.records


# --- injection into the wrapped prompt --------------------------------------


def test_block_sits_after_the_persona_and_before_the_task(midas_data_root, trader_id):
    _write_persona(midas_data_root, trader_id)
    _write_notices(midas_data_root, [_notice()])
    wrapped, model = wrap_persona_prompt(
        trader_id, "TASK_SENTINEL", today=date(2026, 9, 10)
    )
    assert model == "opus"
    assert (
        wrapped.index("PERSONA_SENTINEL")
        < wrapped.index("--- END PERSONA ---")
        < wrapped.index("--- DESK NOTICE ---")
        < wrapped.index("--- END DESK NOTICE ---")
        < wrapped.index("--- TASK ---")
        < wrapped.index("TASK_SENTINEL")
    )
    assert "The watcher published nothing for eleven days." in wrapped


def test_no_notice_leaves_the_wrapped_prompt_unchanged(midas_data_root, trader_id):
    # The empty block must be byte-identical to the pre-notice wrapper, so a
    # desk with nothing to announce pays nothing for the mechanism.
    _write_persona(midas_data_root, trader_id)
    without, _ = wrap_persona_prompt(trader_id, "TASK", today=date(2026, 9, 10))
    _write_notices(
        midas_data_root, [_notice(**{"from": "2026-01-01", "until": "2026-01-02"})]
    )
    still_without, _ = wrap_persona_prompt(trader_id, "TASK", today=date(2026, 9, 10))
    assert without == still_without
    assert "DESK NOTICE" not in without
    assert "--- END PERSONA ---\n\n--- TASK ---" in without


def test_a_malformed_file_never_breaks_dispatch(midas_data_root, trader_id, caplog):
    # Losing a session over desk prose would be far worse than a missed notice.
    _write_persona(midas_data_root, trader_id)
    _write_notices(midas_data_root, "{not json at all")
    with caplog.at_level(logging.WARNING, logger="engine.desk_notices"):
        wrapped, _ = wrap_persona_prompt(
            trader_id, "TASK_SENTINEL", today=date(2026, 9, 10)
        )
    assert "TASK_SENTINEL" in wrapped
    assert "DESK NOTICE" not in wrapped


# --- the committed notice file ----------------------------------------------


@pytest.mark.live_cast
def test_committed_notices_are_well_formed_and_carry_the_watcher_outage():
    path = get_config().desk_notices_path
    assert path.exists(), "this desk ships data/desk_notices.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list)
    # Every committed entry must survive the parser — a notice silently dropped
    # for a typo is a notice nobody is told about.
    assert len(load_notices()) == len(raw)
    outage = next(n for n in load_notices() if n.id == "watcher-outage-2026-08-24")
    assert outage.audience == "all"
    assert (outage.start, outage.end) == (date(2026, 9, 5), date(2026, 9, 19))
    assert "2026-08-24" in outage.text and "2026-09-04" in outage.text
    assert "advisory" in outage.text
