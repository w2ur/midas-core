"""A tripwire around a persona dispatch round.

Dispatched agents now hold live web tools, and what a page returns is untrusted
text that may try to make them write a file. The prompt says not to; this checks
afterwards whether one did. It is a tripwire against an agent writing files, not
containment: a subagent with Bash runs in the same sandbox as the session, so a
capable injection could act outside what is fingerprinted here, or reach the
network through its own tools. What bounds every order is the broker's rails.

``snapshot_data_tree`` records, and ``assert_data_tree_unchanged`` compares, two
classes of signal.

**Abort** — what the session never produces itself while a round runs, because
no helper writes between begin and end. Any change raises
``DispatchWroteDataError``:

- ``HEAD`` (commit and symbolic ref) and every ref under ``refs/heads/`` and
  ``refs/tags/``, so a commit made during the round, which leaves
  ``git status`` clean, is still seen. ``refs/remotes/`` and ``FETCH_HEAD``
  stay outside: a fetch is harmless;
- every non-ignored path ``git status`` lists (modified, staged, renamed,
  untracked — ``engine/``, ``scripts/`` and ``roster.yaml`` as much as
  ``data/``), by content and by its two-letter status code;
- the index entries marked skip-worktree or assume-unchanged, which hide a
  written tracked file from ``git status``;
- ``info/exclude``, ``info/attributes``, the repository config and every hook
  file, which can hide a new file or run code at the next commit.

**Report** — gitignored inputs the session reads after a round, where a write
is suspicious but not proof (the orchestrator keeps its dispatch results under
``data/session_state/``). A change is printed and appended, one JSON line per
change, to ``data/session_state/dispatch_guard_concerns.jsonl``; the session
commit turns that file into ``Concerns:`` trailers:

- everything under ``data/session_state/`` except what the orchestrator writes
  there itself (the dispatch ledger and the ``prompts/`` directory) and the
  concerns file. ``results/`` is in this class: the session persists a round's
  results there only after that round's end has run;
- ``data/market/today.json`` (the prices Step 4 writes immutable snapshots
  from);
- the ``.pth`` and ``sitecustomize``/``usercustomize`` files in the running
  interpreter's site-packages, compared only when begin and end ran under the
  same ``sys.prefix``.

The snapshot lives under the git dir, outside the checkout, keyed by round and
by the run's anchor when one is given. Each run from Step 0c is a new key, so
a snapshot an earlier run left behind is never read as this run's (its writes
were reverted by Step 0's reset, and comparing against it would false-abort). Begin takes a fresh baseline, written
atomically over any snapshot under the key, which also clears its pass state:
begin exactly once per dispatch, immediately before it. One exception guards a
round re-begun within the same run: when an anchored snapshot exists that has not passed (a round
interrupted mid-dispatch), begin first compares it with one capture of the
tree, the same capture its new baseline is built from. A changed abort signal
raises ``DispatchWroteDataError`` and keeps the old snapshot; a changed report
signal is recorded as a concern, as end would, before the re-baseline, so
neither class is absorbed silently. The two are independent: the abort
comparison reads ``abort`` alone, and a malformed ``report`` or missing
``prefix`` only skips the report diff. A missing, passed or unreadable
snapshot (one whose ``abort`` is not a mapping included), and any anchorless
begin, re-baselines without comparing.

End keeps the snapshot, so an end re-run after a downstream error evaluates
again, until an end passes. With an anchor it then sets ``"passed": true``
inside the snapshot, and a later end for the same round and anchor returns
without comparing, because the session's own post-round writes (outbox,
research files, manager book) would otherwise trip it. Without an anchor a
passing end deletes the snapshot instead: an anchorless snapshot is keyed by
round name alone, so one left behind would persist across ad-hoc runs and
answer "already verified" to a later end that had no begin. Anchorless runs are
local or manual, so an end re-run after a pass raising "did not run" is
deliberate. End refuses when there is no snapshot for its round and anchor, and
turns any error it meets (an unreadable snapshot included) into
``DispatchWroteDataError``: an end call that cannot evaluate is not a pass.

**Remaining limit.** A re-dispatch after a passing end MUST be preceded by a
new begin. An end after a pass returns "already verified" without comparing, so
a re-dispatch whose begin was forgotten is unfenced; nothing here can tell.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GUARD_DIRNAME = "midas-dispatch-guard"
_SESSION_STATE = "data/session_state"
CONCERNS_FILE = f"{_SESSION_STATE}/dispatch_guard_concerns.jsonl"
# Written by the orchestrator's own helpers, during a round as much as outside.
_ORCHESTRATOR_WRITTEN = (
    f"{_SESSION_STATE}/dispatch_ledger.jsonl",
    CONCERNS_FILE,
)
_ORCHESTRATOR_WRITTEN_DIRS = (
    f"{_SESSION_STATE}/prompts/",
)
_IGNORED_INPUTS = ("data/market/today.json",)
_WATCHED_REFS = ("refs/heads/", "refs/tags/")
_GIT_FILES = ("info/exclude", "info/attributes", "config", "config.worktree")
_STARTUP_FILES = ("sitecustomize.py", "usercustomize.py")
_SITE_PREFIX = "site-packages:"
_DELETED = "deleted"
_SUMMARY_LIMIT = 300


class DispatchWroteDataError(RuntimeError):
    """A dispatch round changed what the guard watches, or could not be checked."""


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True
    ).stdout


def _git_str(root: Path, *args: str) -> str:
    return os.fsdecode(_git(root, *args)).strip()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _show(key: str) -> str:
    """A key as printable text, whatever bytes its file name held."""
    return os.fsencode(key).decode("utf-8", "backslashreplace")


def _snapshot_path(round_name: str, anchor: str | None, root: Path) -> Path:
    git_dir = Path(_git_str(root, "rev-parse", "--absolute-git-dir"))
    name = f"{round_name}@{anchor}" if anchor else round_name
    return git_dir / _GUARD_DIRNAME / f"{name}.json"


def _git_path(root: Path, name: str) -> Path:
    path = Path(_git_str(root, "rev-parse", "--git-path", name))
    return path if path.is_absolute() else root / path


def _fingerprint(path: Path) -> str:
    if path.is_symlink():
        return _sha(b"symlink:" + os.fsencode(os.readlink(path)))
    if path.is_file():
        return _sha(path.read_bytes())
    return _DELETED


def _split_z(raw: bytes) -> list[str]:
    return [os.fsdecode(field) for field in raw.split(b"\0") if field]


def _status(root: Path) -> dict[str, str]:
    """``{path: XY}`` for every path ``git status`` lists, renames on both sides."""
    entries = _split_z(
        _git(root, "status", "--porcelain", "--untracked-files=all", "-z")
    )
    out: dict[str, str] = {}
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        code, path = entry[:2], entry[3:]
        out[path] = code
        if "R" in code or "C" in code:
            out[entries[i]] = code  # a rename/copy is followed by its origin path
            i += 1
    return out


def _hidden_index_flags(root: Path) -> dict[str, str]:
    """Index entries tagged assume-unchanged (lowercase) or skip-worktree (S)."""
    out: dict[str, str] = {}
    for entry in _split_z(_git(root, "ls-files", "-v", "-z")):
        tag, path = entry[:1], entry[2:]
        if tag.islower() or tag == "S":
            out[path] = tag
    return out


def _refs(root: Path) -> dict[str, str]:
    lines = _git_str(
        root, "for-each-ref", "--format=%(refname) %(objectname)", *_WATCHED_REFS
    ).splitlines()
    return dict(line.rsplit(" ", 1) for line in lines if line)


def _site_packages_dirs() -> list[Path]:
    return sorted(Path(sys.prefix).glob("lib/python*/site-packages"))


def _is_reported_path(rel: str) -> bool:
    return rel.startswith(f"{_SESSION_STATE}/") or rel in _IGNORED_INPUTS


def _is_orchestrator_written(rel: str) -> bool:
    return rel in _ORCHESTRATOR_WRITTEN or rel.startswith(_ORCHESTRATOR_WRITTEN_DIRS)


def _capture_abort(root: Path) -> dict[str, str]:
    snap: dict[str, str] = {}
    for rel, code in _status(root).items():
        if _is_reported_path(rel):
            continue  # judged by the report class, as when it is ignored
        snap[rel] = _fingerprint(root / rel)
        snap[f"index-status:{rel}"] = code

    head = _git_str(root, "rev-parse", "HEAD")
    head_ref = _git_str(root, "rev-parse", "--symbolic-full-name", "HEAD")
    snap["HEAD"] = f"{head} {head_ref}"
    for ref, sha in _refs(root).items():
        snap[f"ref:{ref}"] = sha
    for rel, tag in _hidden_index_flags(root).items():
        snap[f"index-flag:{rel}"] = tag

    for name in _GIT_FILES:
        snap[f"git:{name}"] = _fingerprint(_git_path(root, name))
    hooks = _git_path(root, "hooks")
    if hooks.is_dir():
        for hook in hooks.rglob("*"):
            if hook.is_file() or hook.is_symlink():
                snap[f"git:hooks/{hook.relative_to(hooks).as_posix()}"] = (
                    _fingerprint(hook)
                )
    return snap


def _capture_report(root: Path) -> dict[str, str]:
    rels: set[str] = set(_IGNORED_INPUTS)
    state_dir = root / _SESSION_STATE
    if state_dir.is_dir():
        rels.update(
            p.relative_to(root).as_posix()
            for p in state_dir.rglob("*")
            if p.is_file() or p.is_symlink()
        )
    snap = {
        rel: _fingerprint(root / rel)
        for rel in rels
        if not _is_orchestrator_written(rel)
    }
    for site in _site_packages_dirs():
        startup = [*site.glob("*.pth"), *(site / n for n in _STARTUP_FILES)]
        for path in startup:
            if path.exists() or path.is_symlink():
                snap[f"{_SITE_PREFIX}{path.name}"] = _fingerprint(path)
    return snap


def _write_snapshot(path: Path, doc: dict) -> None:
    """Write ``doc`` to ``path`` through a temp file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def snapshot_data_tree(
    round_name: str, repo_root: Path | None = None, anchor: str | None = None
) -> None:
    """Record what the guard watches before a dispatch round.

    ``anchor`` identifies one run (it includes the start time); the snapshot is keyed by it and by the
    round. An existing snapshot under the same key is replaced, atomically, and
    its pass state with it, except that an anchored snapshot that has not
    passed (a round interrupted mid-dispatch) is compared first: a changed
    abort signal raises ``DispatchWroteDataError`` and the snapshot is kept; a
    changed report signal is recorded as a concern before the re-baseline.
    """
    root = repo_root or _REPO_ROOT
    path = _snapshot_path(round_name, anchor, root)
    doc = {
        "anchor": anchor,
        "prefix": sys.prefix,
        "abort": _capture_abort(root),
        "report": _capture_report(root),
    }
    if anchor is not None:
        _compare_unfinished_round(round_name, anchor, path, doc, root)
    _write_snapshot(path, doc)
    count = len(doc["abort"]) + len(doc["report"])
    print(f"  dispatch guard [{round_name}]: snapshot taken ({count} entries)")


def _compare_unfinished_round(
    round_name: str, anchor: str, path: Path, now: dict, root: Path
) -> None:
    """Compare a round interrupted before its end with the capture ``now``.

    Two independent steps. The abort comparison reads ``abort`` alone: a
    snapshot that is missing or unreadable, or whose ``abort`` is not a
    mapping, has nothing to compare and is re-baselined. The report diff is a
    separate step whose own failure (a malformed ``report``, a missing
    ``prefix``) only skips recording and never prevents the abort check.
    Raising and recording sit outside the error handling, so their own
    failures are not mistaken for an unreadable snapshot.
    """
    try:
        before = json.loads(path.read_text(encoding="ascii"))
        if before.get("passed") is True:
            return
        aborts = _diff(before["abort"], now["abort"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return
    try:
        reported = _report_diff(round_name, before, now["report"])
    except (KeyError, TypeError, AttributeError):
        # Never lets a malformed report section skip the abort check above,
        # but the re-baseline must not absorb it without a trace either.
        reported = [("<report class>", "could not be evaluated")]
    _raise_on_aborts(
        round_name, aborts, " (found at begin, on a round that had not finished)"
    )
    _note_reported(round_name, anchor, reported, root)


def _diff(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str]]:
    out = [(p, "appeared") for p in sorted(after.keys() - before.keys())]
    out += [(p, "disappeared") for p in sorted(before.keys() - after.keys())]
    out += [
        (p, "changed")
        for p in sorted(before.keys() & after.keys())
        if before[p] != after[p]
    ]
    return out


def _without_site_packages(snap: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in snap.items() if not k.startswith(_SITE_PREFIX)}


def _raise_on_aborts(
    round_name: str, aborts: list[tuple[str, str]], where: str = ""
) -> None:
    if aborts:
        raise DispatchWroteDataError(
            f"dispatch round {round_name!r} wrote in the checkout{where}: "
            + "; ".join(f"{_show(p)} ({kind})" for p, kind in aborts)
        )


def _report_diff(
    round_name: str, before: dict, report_after: dict[str, str]
) -> list[tuple[str, str]]:
    """Changes to the report class, site-packages left out across prefixes."""
    report_before = before["report"]
    if before["prefix"] != sys.prefix:
        print(
            f"  dispatch guard [{round_name}]: snapshot taken under "
            f"{before['prefix']}, now under {sys.prefix}; site-packages not compared"
        )
        report_before = _without_site_packages(report_before)
        report_after = _without_site_packages(report_after)
    return _diff(report_before, report_after)


def _note_reported(
    round_name: str,
    anchor: str | None,
    reported: list[tuple[str, str]],
    root: Path,
) -> None:
    if reported:
        _record(round_name, anchor, reported, root)
        for rel, kind in reported:
            print(f"  dispatch guard [{round_name}]: reported {_show(rel)} ({kind})")


def _record(
    round_name: str, anchor: str | None, changes: list[tuple[str, str]], root: Path
) -> None:
    path = root / CONCERNS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rel, kind in changes:
            line = {"anchor": anchor, "round": round_name, "path": _show(rel), "kind": kind}
            fh.write(json.dumps(line, sort_keys=True) + "\n")


def _check(round_name: str, anchor: str | None, root: Path) -> None:
    """Compare the round against its snapshot, or return if already verified."""
    path = _snapshot_path(round_name, anchor, root)
    if not path.is_file():
        raise DispatchWroteDataError(
            f"no dispatch-guard snapshot for round {round_name!r} in this session "
            f"at {path}: the guard did not run, which is not the same as the "
            "round being clean"
        )
    before = json.loads(path.read_text(encoding="ascii"))
    if before.get("passed") is True:
        print(f"  dispatch guard [{round_name}]: already verified this session")
        return

    _raise_on_aborts(round_name, _diff(before["abort"], _capture_abort(root)))
    reported = _report_diff(round_name, before, _capture_report(root))
    _note_reported(round_name, anchor, reported, root)
    print(f"  dispatch guard [{round_name}]: no abort signal changed")
    if anchor is None:
        path.unlink()
    else:
        _write_snapshot(path, {**before, "passed": True})


def assert_data_tree_unchanged(
    round_name: str, repo_root: Path | None = None, anchor: str | None = None
) -> None:
    """Raise ``DispatchWroteDataError`` if the round changed an abort signal.

    Changes to a report signal are printed and recorded instead. Any other
    error met while checking is raised as ``DispatchWroteDataError`` too, with
    its cause. A failed end leaves the snapshot unmarked, so a repeated end
    evaluates again; after a pass, a repeated end returns without comparing
    (an anchorless run deletes the snapshot on a pass instead).
    """
    root = repo_root or _REPO_ROOT
    try:
        _check(round_name, anchor, root)
    except DispatchWroteDataError:
        raise
    except Exception as exc:
        raise DispatchWroteDataError(
            f"dispatch guard for round {round_name!r} could not be evaluated, "
            f"which is not a pass: {type(exc).__name__}: {exc}"
        ) from exc


def guard_concerns(anchor: str | None, repo_root: Path | None = None) -> list[str]:
    """One concern per round with reported changes in this session, bounded.

    Reads ``CONCERNS_FILE``; lines from another anchor are skipped and repeated
    lines (an end re-run) collapse. Raises on a malformed file: the caller
    decides what an unreadable record costs.
    """
    path = (repo_root or _REPO_ROOT) / CONCERNS_FILE
    if not path.is_file():
        return []
    by_round: dict[str, list[str]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        if line.get("anchor") != anchor:
            continue
        item = f"{line['path']} ({line['kind']})"
        items = by_round.setdefault(line["round"], [])
        if item not in items:
            items.append(item)
    concerns = []
    for round_name, items in by_round.items():
        head = f"dispatch guard [{round_name}] reported changes to ignored inputs: "
        shown: list[str] = []
        for item in items:
            if len(head) + len(", ".join([*shown, item])) > _SUMMARY_LIMIT:
                break
            shown.append(item)
        text = head + ", ".join(shown)
        if len(shown) < len(items):
            text += f" and {len(items) - len(shown)} more"
        concerns.append(text)
    return concerns
