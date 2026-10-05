"""Instrument status registry (plan 2026-10-03, Stage 1.2).

`engine.instrument_status` records which symbols the store must not be
trusted for. These tests pin the four things the plan asks of it:

1. a tripwire refusal writes `suspended` (through the real `merge_rows`);
2. adjudication clears it: the calendar path once its re-merge lands, and a
   human `clear`; a calendar that cannot explain the rows, or a re-merge that
   lands nothing, leaves it suspended;
3. an unreadable registry fails closed and says so;
4. the seed reads only unadjudicated quarantine rows: MNST, JMAT.L and BYND
   (ledgered) and MRNA (re-merged by hand, no ledger row) are not suspended,
   CTVA (neither) is.

Fixtures are embedded literals copied from the committed quarantine, ledger and
store on 2026-10-03, never read from them: this file ships to midas-core, which
has neither. `tests/test_instrument_status_live.py` holds the committed files.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from engine import instrument_status as status
from engine.config import get_config
from engine.corporate_actions import CorporateAction
from engine.ohlcv_ingest import MergeResult, QuarantinedRow
from scripts import fetch_ohlcv as fo

_FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _frame(rows: dict[str, float]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in rows], name="Date")
    data = [[c, c, c, c, c, 100] for c in rows.values()]
    return pd.DataFrame(data, index=idx, columns=pd.Index(_FIELDS))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. A tripwire refusal writes `suspended`
# ---------------------------------------------------------------------------


class TestTripwireSuspends:
    def test_a_refused_row_marks_the_symbol_suspended(self, midas_data_root):
        """The real merge path: CTVA's 2026-10-01 print, x0.16 its stored close."""
        store = get_config().ohlcv_dir / "CTVA.jsonl"
        _write_jsonl(store, [{"date": "2026-09-30", "close": 77.65}])
        merged = fo._write_rows(
            "CTVA", _frame({"2026-10-01": 12.57}), "2026-09-30", guard_anomalies=True
        )
        assert merged.quarantined == 1, "fixture no longer trips the tripwire"
        entry = status.load()["CTVA"]
        assert entry.status == status.SUSPENDED
        assert entry.since == "2026-10-01"
        assert entry.source == "tripwire"
        assert status.status_of("CTVA") == status.SUSPENDED

    def test_a_clean_row_writes_nothing(self, midas_data_root):
        """Control: the registry is not written on a healthy merge."""
        store = get_config().ohlcv_dir / "AAPL.jsonl"
        _write_jsonl(store, [{"date": "2026-09-30", "close": 250.0}])
        merged = fo._write_rows(
            "AAPL", _frame({"2026-10-01": 251.0}), "2026-09-30", guard_anomalies=True
        )
        assert merged.appended == 1 and merged.quarantined == 0
        assert not status.registry_path().exists()
        assert status.status_of("AAPL") is None

    def test_a_second_refusal_keeps_the_first_date(self, midas_data_root):
        """A symbol refused night after night still says when it froze."""
        status.mark_suspended(
            "CTVA", since="2026-10-01", source="tripwire", reason="first night"
        )
        status.mark_suspended(
            "CTVA", since="2026-10-02", source="tripwire", reason="second night"
        )
        entry = status.load()["CTVA"]
        assert entry.since == "2026-10-01"
        assert entry.reason == "first night"


# ---------------------------------------------------------------------------
# 2. Adjudication clears it
# ---------------------------------------------------------------------------


class TestAdjudicationClears:
    ACTION = CorporateAction("MNST", "2026-08-11", shares_ratio=2.0)
    REFUSED = (QuarantinedRow("MNST", "2026-08-11", "new-row", 91.43, 45.53, 0.497976),)

    def _arrange(self, monkeypatch, *, actions, lands: bool):
        status.mark_suspended(
            "MNST", since="2026-08-11", source="tripwire", reason="refused"
        )
        monkeypatch.setattr(fo, "_fetch_actions", lambda symbol: list(actions))
        monkeypatch.setattr(
            fo, "_fetch_symbol", lambda *a, **k: _frame({"2026-08-11": 45.53})
        )
        monkeypatch.setattr(
            fo,
            "_write_rows",
            lambda *a, **k: MergeResult(1, 0) if lands else MergeResult(0, 0),
        )
        monkeypatch.setattr(fo, "_apply_split_to_holders", lambda symbol, ratio: [])

    def _adjudicate(self):
        return fo._adjudicate(
            {"MNST": self.REFUSED}, start=date(2026, 5, 1), end=date(2026, 8, 12)
        )

    def test_a_calendar_split_that_lands_clears_the_status(
        self, midas_data_root, monkeypatch
    ):
        self._arrange(monkeypatch, actions=[self.ACTION], lands=True)
        assert self._adjudicate() == (1, 1)
        assert status.status_of("MNST") is None

    def test_no_calendar_action_leaves_it_suspended(self, midas_data_root, monkeypatch):
        self._arrange(monkeypatch, actions=[], lands=True)
        assert self._adjudicate() == (0, 0)
        assert status.status_of("MNST") == status.SUSPENDED

    def test_a_remerge_that_lands_nothing_leaves_it_suspended(
        self, midas_data_root, monkeypatch
    ):
        """The calendar explaining a refusal is not proof the store moved."""
        self._arrange(monkeypatch, actions=[self.ACTION], lands=False)
        assert self._adjudicate() == (0, 0)
        assert status.status_of("MNST") == status.SUSPENDED

    def test_a_human_clear_removes_it_and_needs_a_reason(self, midas_data_root):
        status.mark_suspended(
            "CTVA", since="2026-10-01", source="tripwire", reason="refused"
        )
        with pytest.raises(SystemExit):
            status.main(["clear", "CTVA"])  # argparse: --reason is required
        assert status.status_of("CTVA") == status.SUSPENDED
        with pytest.raises(ValueError):
            status.clear("CTVA", reason="   ")
        assert status.main(["clear", "CTVA", "--reason", "spin-off, VYLR"]) == 0
        assert status.status_of("CTVA") is None
        assert status.main(["clear", "CTVA", "--reason", "again"]) == 1


# ---------------------------------------------------------------------------
# 3. An unreadable registry fails closed
# ---------------------------------------------------------------------------


class TestUnreadableRegistryFailsClosed:
    @pytest.mark.parametrize(
        "text",
        [
            "{not json\n",
            json.dumps({"schema": 99, "instruments": {}}),
            json.dumps({"schema": 1, "instruments": {"X": {"status": "halted"}}}),
            json.dumps(
                {
                    "schema": 1,
                    "instruments": {
                        "X": {
                            "status": "suspended",
                            "since": "yesterday",
                            "source": "seed",
                            "reason": "r",
                        }
                    },
                }
            ),
        ],
        ids=["not-json", "wrong-schema", "unknown-status", "bad-date"],
    )
    def test_every_lookup_answers_suspended_and_logs(
        self, midas_data_root, caplog, text
    ):
        path = status.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger="engine.instrument_status"):
            assert status.status_of("AAPL") == status.SUSPENDED
            assert status.status_of("ANYTHING") == status.SUSPENDED
        assert any("unreadable" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize(
        "raw",
        [
            b"\xff\xfe",
            json.dumps(
                {
                    "schema": 1,
                    "instruments": {
                        "X": {
                            "status": ["suspended"],
                            "since": "2026-10-01",
                            "source": "seed",
                            "reason": "r",
                        }
                    },
                }
            ).encode(),
            json.dumps(
                {
                    "schema": 1,
                    "instruments": {
                        "X": {
                            "status": {"s": 1},
                            "since": "2026-10-01",
                            "source": "seed",
                            "reason": "r",
                        }
                    },
                }
            ).encode(),
        ],
        ids=["not-utf8", "list-status", "dict-status"],
    )
    def test_a_file_that_breaks_the_parser_still_fails_closed(
        self, midas_data_root, caplog, raw
    ):
        """Regression (review of feat/stage1-asof-reads, 2026-10-03): an
        unhashable status raised TypeError at the ``in STATUSES`` test, and a
        non-UTF-8 file raised UnicodeDecodeError (a ValueError, not the
        OSError ``load`` caught), so the lookup raised instead of answering
        ``suspended``: fill_day aborted part-way and every watcher fire
        counted as an error."""
        path = status.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        with caplog.at_level(logging.ERROR, logger="engine.instrument_status"):
            assert status.status_of("AAPL") == status.SUSPENDED
        assert any("unreadable" in r.getMessage() for r in caplog.records)

    def test_a_writer_never_overwrites_it(self, midas_data_root):
        path = status.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json\n", encoding="utf-8")
        with pytest.raises(status.RegistryUnreadable):
            status.mark_suspended("CTVA", since="2026-10-01", source="t", reason="r")
        with pytest.raises(status.RegistryUnreadable):
            status.clear("CTVA", reason="r")
        assert path.read_text(encoding="utf-8") == "{not json\n"

    def test_the_tripwire_path_survives_it(self, midas_data_root, capsys):
        """A refusal against an unreadable registry still merges and reports.

        Raising here would crash the nightly run, and a crashed run commits
        nothing; the refusal already exits non-zero, and lookups fail closed.
        """
        path = status.registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json\n", encoding="utf-8")
        _write_jsonl(
            get_config().ohlcv_dir / "CTVA.jsonl",
            [{"date": "2026-09-30", "close": 77.65}],
        )
        merged = fo._write_rows(
            "CTVA", _frame({"2026-10-01": 12.57}), "2026-09-30", guard_anomalies=True
        )
        assert merged.quarantined == 1
        assert "registry unreadable" in capsys.readouterr().err
        assert status.status_of("CTVA") == status.SUSPENDED

    def test_a_missing_registry_is_empty(self, midas_data_root):
        """A fork or a fresh data root has refused nothing yet."""
        assert not status.registry_path().exists()
        assert status.status_of("AAPL") is None

    def test_a_missing_registry_beside_a_quarantine_fails_closed(
        self, midas_data_root, caplog
    ):
        """Regression (review of feat/stage1-asof-reads, finding 3): a deleted
        registry read as an empty one, so the broker, the watcher and the
        baseline-manager book all accepted CTVA at its frozen 09-30 close; only
        an advisory CI test noticed. A quarantine row proves a refusal was
        recorded, so the file's absence is a loss, not a fresh root."""
        path = status.registry_path()
        _write_jsonl(path.parent / "quarantine" / "CTVA.jsonl", [{"symbol": "CTVA"}])
        assert not path.exists()
        with caplog.at_level(logging.ERROR, logger="engine.instrument_status"):
            assert status.status_of("CTVA") == status.SUSPENDED
            assert status.status_of("AAPL") == status.SUSPENDED
        assert any("is missing" in r.getMessage() for r in caplog.records)

    def test_the_next_refusal_recreates_a_lost_registry(self, midas_data_root):
        """A tripwire refusal is not blocked by the loss it would repair: the
        file comes back, rebuilt from the quarantine, holding the new symbol."""
        path = status.registry_path()
        _write_jsonl(path.parent / "quarantine" / "CTVA.jsonl", _QUARANTINE["CTVA"])
        status.mark_suspended("CTVA", since="2026-10-01", source="t", reason="r")
        assert path.exists()
        assert status.status_of("AAPL") is None
        assert status.status_of("CTVA") == status.SUSPENDED

    def test_a_refusal_after_the_loss_keeps_every_lost_suspension(
        self, midas_data_root
    ):
        """Regression (review of feat/stage1-asof-reads, lost-registry
        writers): `mark` read the lost file as empty and wrote back only the
        new symbol, so the next tripwire refusal on any other symbol cleared
        CTVA, and the broker would fill it at its frozen 09-30 close."""
        path = status.registry_path()
        _write_jsonl(path.parent / "quarantine" / "CTVA.jsonl", _QUARANTINE["CTVA"])
        assert status.status_of("CTVA") == status.SUSPENDED  # the `_lost` rule
        status.mark_suspended(
            "XYZ", since="2026-10-04", source="tripwire", reason="refused"
        )
        assert status.status_of("CTVA") == status.SUSPENDED
        assert status.status_of("XYZ") == status.SUSPENDED
        assert status.load()["CTVA"].since == "2026-10-01"

    def test_a_clear_after_the_loss_keeps_every_other_suspension(
        self, midas_data_root
    ):
        """Same hole through `clear`: adjudicating one symbol must not write
        an empty registry over the lost one."""
        path = status.registry_path()
        quarantine = path.parent / "quarantine"
        _write_jsonl(quarantine / "CTVA.jsonl", _QUARANTINE["CTVA"])
        _write_jsonl(quarantine / "MRNA.jsonl", _QUARANTINE["MRNA"][:1])
        removed = status.clear("MRNA", reason="human: real repricing")
        assert removed is not None and removed.status == status.SUSPENDED
        assert path.exists()
        assert status.status_of("CTVA") == status.SUSPENDED
        assert status.status_of("MRNA") is None

    def test_a_lost_registry_that_cannot_be_rebuilt_is_not_overwritten(
        self, midas_data_root
    ):
        """A ledger the rebuild cannot read leaves the file absent, so every
        lookup still fails closed, and the tripwire path reports it."""
        path = status.registry_path()
        _write_jsonl(path.parent / "quarantine" / "CTVA.jsonl", _QUARANTINE["CTVA"])
        (path.parent / "corporate_actions.jsonl").write_text("{not json\n")
        with pytest.raises(status.RegistryUnreadable):
            status.mark_suspended("XYZ", since="2026-10-04", source="t", reason="r")
        with pytest.raises(status.RegistryUnreadable):
            status.clear("CTVA", reason="r")
        assert not path.exists()
        assert status.status_of("AAPL") == status.SUSPENDED


# ---------------------------------------------------------------------------
# 4. Seeding reads only unadjudicated quarantine rows (review SHOULD 4)
# ---------------------------------------------------------------------------

# Copied from data/market/quarantine/*.jsonl, corporate_actions.jsonl and the
# store on 2026-10-03 (a subset of rows; every symbol keeps its shape).
_QUARANTINE = {
    "MNST": [
        {
            "symbol": "MNST",
            "date": "2026-08-10",
            "kind": "revision",
            "stored_close": 91.43,
            "incoming_close": 45.715,
            "ratio": 0.5,
        },
        {
            "symbol": "MNST",
            "date": "2026-08-11",
            "kind": "new-row",
            "stored_close": 91.43,
            "incoming_close": 45.53,
            "ratio": 0.49798,
        },
    ],
    "JMAT.L": [
        {
            "symbol": "JMAT.L",
            "date": "2026-08-12",
            "kind": "revision",
            "stored_close": 22.12,
            "incoming_close": 29.4933,
            "ratio": 1.33333,
        },
    ],
    "BYND": [
        {
            "symbol": "BYND",
            "date": "2026-08-12",
            "kind": "revision",
            "stored_close": 0.4141,
            "incoming_close": 12.423,
            "ratio": 30.0,
        },
        {
            "symbol": "BYND",
            "date": "2026-08-13",
            "kind": "new-row",
            "stored_close": 0.4141,
            "incoming_close": 12.465,
            "ratio": 30.1014,
        },
    ],
    "MRNA": [
        {
            "symbol": "MRNA",
            "date": "2026-08-19",
            "kind": "new-row",
            "stored_close": 64.46,
            "incoming_close": 174.38,
            "ratio": 2.70524,
        },
        {
            "symbol": "MRNA",
            "date": "2026-08-20",
            "kind": "new-row",
            "stored_close": 64.46,
            "incoming_close": 133.32,
            "ratio": 2.06826,
        },
    ],
    "CTVA": [
        {
            "symbol": "CTVA",
            "date": "2026-10-01",
            "kind": "new-row",
            "stored_close": 77.65,
            "incoming_close": 12.57,
            "ratio": 0.16188,
        },
        {
            "symbol": "CTVA",
            "date": "2026-10-02",
            "kind": "new-row",
            "stored_close": 77.65,
            "incoming_close": 11.92,
            "ratio": 0.15351,
        },
    ],
}
_LEDGER = [
    {"symbol": "MNST", "effective": "2026-08-11", "shares_ratio": 2.0},
    {"symbol": "JMAT.L", "effective": "2026-08-17", "shares_ratio": 0.75},
    {"symbol": "BYND", "effective": "2026-08-14", "shares_ratio": 1 / 30},
]
# MNST's and BYND's stores still hold the pre-action basis on some refused
# dates (BYND 08-12 at 0.4141), which is why the ledger arm, not the store
# arm, is what adjudicates them. MRNA's store holds the refused side: a human
# re-merge on 2026-08-21 (8b6907458) landed it and, no corporate action having
# occurred, wrote no ledger row. CTVA's store stops at 09-30.
_STORE = {
    "MNST": {"2026-08-10": 91.43},
    "JMAT.L": {"2026-08-12": 29.4933},
    "BYND": {"2026-08-12": 0.4141, "2026-08-14": 13.47},
    "MRNA": {"2026-08-18": 62.96, "2026-08-19": 174.38, "2026-08-20": 133.32},
    "CTVA": {"2026-09-29": 77.87, "2026-09-30": 77.65},
}


def _seed_world(root: Path, *, ledger=_LEDGER, store=_STORE) -> tuple[Path, Path, Path]:
    market = root / "market"
    for symbol, rows in _QUARANTINE.items():
        _write_jsonl(market / "quarantine" / f"{symbol}.jsonl", rows)
    _write_jsonl(market / "corporate_actions.jsonl", list(ledger))
    for symbol, closes in store.items():
        _write_jsonl(
            market / "ohlcv" / f"{symbol}.jsonl",
            [{"date": d, "close": c} for d, c in closes.items()],
        )
    return market / "quarantine", market / "corporate_actions.jsonl", market / "ohlcv"


class TestSeeding:
    def test_adjudicated_symbols_are_not_suspended_and_ctva_is(self, tmp_path):
        entries = status.seed_entries(*_seed_world(tmp_path))
        assert set(entries) == {"CTVA"}
        assert entries["CTVA"].status == status.SUSPENDED
        assert entries["CTVA"].since == "2026-10-01"
        for adjudicated in ("MNST", "JMAT.L", "BYND", "MRNA"):
            assert adjudicated not in entries

    def test_ctva_with_its_ledger_row_is_not_suspended(self, tmp_path):
        """Plan 0.1 appends CTVA's spin-off row; the seed must follow it."""
        ledger = [
            *_LEDGER,
            {"symbol": "CTVA", "effective": "2026-10-01", "shares_ratio": 1.0},
        ]
        assert status.seed_entries(*_seed_world(tmp_path, ledger=ledger)) == {}

    def test_the_ledger_arm_is_what_clears_mnst_and_bynd(self, tmp_path):
        """Control: without their ledger rows, the stores alone do not clear them."""
        entries = status.seed_entries(*_seed_world(tmp_path, ledger=[]))
        assert {"MNST", "BYND", "CTVA"} <= set(entries)

    def test_the_store_arm_is_what_clears_mrna(self, tmp_path):
        """Control: with MRNA's store still on the refused-against side, it is
        unadjudicated (the review's ledger-only rule says so in every case)."""
        store = {**_STORE, "MRNA": {"2026-08-18": 62.96}}
        entries = status.seed_entries(*_seed_world(tmp_path, store=store))
        assert set(entries) == {"CTVA", "MRNA"}

    def test_an_old_ledger_row_does_not_explain_a_new_refusal(self, tmp_path):
        """A ledger row clears refusals near its effective date only: MNST
        refused again in 2027 is a new incident, not the 2026 split."""
        quarantine, ledger, ohlcv = _seed_world(tmp_path)
        _write_jsonl(
            quarantine / "MNST.jsonl",
            [
                {
                    "symbol": "MNST",
                    "date": "2027-03-01",
                    "kind": "new-row",
                    "stored_close": 45.0,
                    "incoming_close": 4.5,
                    "ratio": 0.1,
                }
            ],
        )
        assert "MNST" in status.seed_entries(quarantine, ledger, ohlcv)

    def test_the_cli_refuses_to_overwrite_a_non_empty_registry(self, midas_data_root):
        status.mark_suspended("CTVA", since="2026-10-01", source="human", reason="r")
        assert status.main(["seed"]) == 2
        assert status.load()["CTVA"].source == "human"

    def test_render_round_trips(self, tmp_path):
        entries = status.seed_entries(*_seed_world(tmp_path))
        path = tmp_path / "instrument_status.json"
        path.write_text(status.render(entries), encoding="utf-8")
        assert status.load(path) == entries
