# Contributing

Thanks for looking at midas-core. Please read this first — how the repo is maintained changes
what kind of contribution can actually land.

## This repo is a one-way mirror

midas-core is **generated**, not hand-edited. It is a one-way code mirror synced from the live
desk repo [`w2ur/midas`](https://github.com/w2ur/midas) (the source of truth) by a manifest tool.
The sync overwrites the mirrored trees on every run, so a code change made directly here would be
silently reverted on the next sync. That repo is public, so you can read the upstream of any file
here — but it is still the upstream, and a change has to land there first.

**Code PRs against synced trees cannot be merged.** That covers:

- `engine/`
- `scripts/`
- `tests/`
- `examples/demo-desk/` source files (`roster.yaml`, `.claude/agents/`, `README.md`)
- `data/strategies/` and `data/universes/`

If you have found a bug or want to discuss a design change in any of those, **please open an
issue** instead of a pull request. Bug reports and design discussion are genuinely welcome — they
get applied upstream and flow back here on the next sync.

## What is workable as a PR

Doc-only changes to the **core-native** files — the ones authored directly in this repo and not
overwritten by the sync — are reviewable and mergeable:

- `README.md`
- `CONTRIBUTING.md`
- `DISCLAIMER.md`

Typo fixes, clarifications, and broken-link fixes there are appreciated.

## Reporting an issue

Useful bug reports include:

- the command you ran and the observed vs. expected output;
- your Python version (3.12 or 3.13 is required) and OS;
- the relevant `roster.yaml` snippet if the issue is desk-configuration-specific.

## Releasing

Maintainer only. A release is cut from the mirror, never from a hand edit:

1. In [`w2ur/midas`](https://github.com/w2ur/midas), bump `version` in `pyproject.toml` and get
   both test suites green.
2. Run `uv run python scripts/sync_core.py apply --core <path to this checkout>` from that
   repo, and commit the result here as its own `sync:` commit.
3. Tag that commit `vX.Y.Z`. `publish.yml` runs only from a `v*` tag, refuses a tag that differs
   from the version in `pyproject.toml` (on a manual dispatch too), and publishes to PyPI by
   trusted publishing through the `pypi` environment, whose deployment rule admits `v*` tags only.
   A PyPI version number can never be reused, so a wrong tag burns it.
4. Check the release cold: in a fresh virtual environment, `pip install midas-core==X.Y.Z`,
   then `midas init-demo`, `midas fetch-ohlcv` and `midas fill-day` as in the README quickstart,
   and confirm `fill-day` reports a filled order.

## License

By contributing you agree that your contributions are licensed under the [MIT License](./LICENSE).
