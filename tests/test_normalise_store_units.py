"""The one-shot pence→pounds store migration, and the marker that guards it.

This script had **no test coverage at all**, which is uncomfortable for the
one piece of code in the repo whose failure mode is silent and permanent: a
store in pounds is indistinguishable by inspection from one in pence (1.17 is
a perfectly plausible penny stock), so a second pass would divide the history
by 100 again and nothing would look wrong.

The hole this file closes (review W5.2): a pence-quoted ticker fetched for the
FIRST time after 2026-08-07 is ISO-denominated at ingest, and is therefore
absent from the marker's `migrated` map — for the opposite of the reason the
map exists. Re-running the script would have selected it and halved its
history by two orders of magnitude.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.normalise_store_units import MARKER_NAME, main


def _seed(store: Path, symbol: str, rows: list[tuple[str, float]]) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{symbol}.jsonl"
    path.write_text(
        "\n".join(json.dumps({"date": d, "close": c}) for d, c in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _closes(path: Path) -> list[float]:
    return [json.loads(line)["close"] for line in path.read_text().splitlines()]


def _marker(store: Path) -> dict:
    return json.loads((store / MARKER_NAME).read_text())


@pytest.fixture
def store(midas_data_root):
    from engine.config import get_config

    return get_config().ohlcv_dir


def test_a_pence_store_is_converted_once(store, monkeypatch, capsys):
    """The migration's actual job, on a store with no marker yet."""
    path = _seed(store, "LLOY.L", [("2026-04-17", 116.0), ("2026-04-18", 117.5)])

    monkeypatch.setattr("sys.argv", ["normalise_store_units.py", "--apply"])
    assert main() == 0

    assert _closes(path) == pytest.approx([1.16, 1.175])
    marker = _marker(store)
    assert "LLOY.L" in marker["migrated"]
    assert marker["completed_at"]


def test_running_it_twice_is_a_no_op(store, monkeypatch, capsys):
    """The marker path itself, previously untested.

    Idempotence here cannot be inferred from the data — that is the entire
    reason the marker exists — so it has to be asserted against a second run.
    """
    path = _seed(store, "LLOY.L", [("2026-04-17", 116.0)])

    monkeypatch.setattr("sys.argv", ["normalise_store_units.py", "--apply"])
    assert main() == 0
    after_first = _closes(path)

    assert main() == 0
    assert _closes(path) == after_first == pytest.approx([1.16])


def test_a_ticker_arriving_after_the_migration_is_refused(store, monkeypatch, capsys):
    """W5.2: the double-divide hole.

    `WTB.L` here is already pounds — ingest normalised it — and it is missing
    from `migrated` only because it did not exist when the migration ran. The
    completion stamp is what tells those two situations apart.
    """
    monkeypatch.setattr("sys.argv", ["normalise_store_units.py", "--apply"])
    _seed(store, "LLOY.L", [("2026-04-17", 116.0)])
    assert main() == 0

    # A new pence-quoted listing appears, already ISO at ingest.
    newcomer = _seed(store, "WTB.L", [("2026-08-20", 3.12)])
    assert main() == 0

    assert _closes(newcomer) == pytest.approx([3.12]), (
        "a post-migration file was scaled again — the double-divide this stamp exists to stop"
    )
    out = capsys.readouterr().out
    assert "REFUSED" in out
    assert "WTB.L" in out


def test_without_a_completion_stamp_a_missing_symbol_is_still_migrated(
    store, monkeypatch, capsys
):
    """The control: the refusal must depend on the stamp, not on the symbol.

    A legacy store carrying a marker written before this change has no
    `completed_at`, and its unmigrated symbols must still be convertible —
    otherwise the fix silently strands them.
    """
    path = _seed(store, "LLOY.L", [("2026-04-17", 116.0)])
    (store / MARKER_NAME).write_text(
        json.dumps({"note": "legacy marker", "migrated": {"OTHER.L": "2026-08-07"}})
    )

    monkeypatch.setattr("sys.argv", ["normalise_store_units.py", "--apply"])
    assert main() == 0

    assert _closes(path) == pytest.approx([1.16])


def test_a_dry_run_writes_nothing(store, monkeypatch, capsys):
    path = _seed(store, "LLOY.L", [("2026-04-17", 116.0)])

    monkeypatch.setattr("sys.argv", ["normalise_store_units.py", "--dry-run"])
    assert main() == 0

    assert _closes(path) == pytest.approx([116.0])
    assert not (store / MARKER_NAME).exists()


def test_the_live_marker_carries_a_completion_stamp():
    """The committed marker must be able to answer the question at all.

    Without `completed_at`, the guard above is inert on the real store — a
    check that cannot fire. The stamp was backfilled from the migration's own
    latest per-symbol timestamp, not invented.
    """
    repo_marker = (
        Path(__file__).resolve().parents[1] / "data" / "market" / "ohlcv" / MARKER_NAME
    )
    if not repo_marker.exists():  # a fork that never ran the migration
        pytest.skip("no committed migration marker in this repo")
    marker = json.loads(repo_marker.read_text())
    assert marker.get("completed_at")
    assert marker["completed_at"] >= max(marker["migrated"].values())
