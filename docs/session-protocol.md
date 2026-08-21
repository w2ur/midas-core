# Session protocol

How one autonomous trading session runs, start to finish, independent of which
model drives it or which harness schedules it.

`midas-core` gives you the pieces: a broker that enforces rails, portfolios that
value themselves, baselines, an output bundle. This document is the **contract
between those pieces and whatever is driving them** — a scheduler, a shell
script, an agent framework, or a person at a terminal. Implement it and you have
a desk. Ignore the ordering and you get a desk that silently does less than it
appears to.

The driver itself is harness-specific and is not part of the framework, so it
is not shipped here. A reader of `midas-core` will not find one in this
repository; a reader of the live desk repository can see a real
implementation at `docs/triggers/weekday-session.md`. What *is* here, in
either repository, is every step the driver calls, in
`scripts/daily_session.py`, and this description of how to call them.

## What the driver must supply

1. **A model, or a human, per agent.** Every persona in `roster.yaml` needs
   something that reads a prompt and returns structured output. Core does not
   care what. `engine.persona_dispatch.wrap_persona_prompt(agent_id, task)`
   wraps a persona file around a task prompt and hands back the wrapped text
   plus whatever model the persona declares; what you do with either is yours.
2. **Isolation between agents.** Agents must not see each other's reasoning
   within a session. This is a property of the experiment, not of the code —
   nothing in core enforces it. If your driver runs the trading round as one
   conversation, every agent is reading every other agent's thesis and the
   roster stops being a set of independent books.
3. **A clock and a working tree.** The session pins one date and one ledger base
   at the start and re-validates both before writing. The driver owns the
   working tree: core never clones, never checks out, and never chooses a branch.
4. **A parser per output shape.** The prompt-building steps return strings and
   the persisting steps take structured values. Turning one into the other —
   parsing an agent's JSON, a narrator response, a post payload — is the
   driver's job, and it must degrade rather than crash: a single unparseable
   response is one agent's silence, not a lost session.

## Invariants — the parts that are not optional

Each of the following is load-bearing, and each is here because a session broke
without it.

**Realign before anything else.** A driver that reuses a workspace across runs
must reset it to the current remote tip before it reads a single file. A session
that starts from a stale tip reads yesterday's portfolios, and its agents author
sells of positions that have already been sold and buys of cash that has already
been spent. Nothing downstream can detect this, because every artifact the
session produces is internally consistent — it is consistent with a world that
no longer exists.

**Anchor the date and the ledger base once, at the start.** `scripts/session_guard.py`
pins the session date, the base commit and the wall-clock start;
`assert_session_fresh` re-checks all three before the first irreversible write
and again before the push. A driver whose process can be suspended — a sandbox,
a container, a laptop lid — will eventually resume hours later and finish a
session against a snapshot that has been superseded, and every individual guard
in the pipeline will have been correct at the moment it ran. The anchor is the
only check that asks whether the world moved *since*.

**A `[SKIP]` you did not ask for is an abort condition.** `scripts/session_state.py`
records a done-marker per step, scoped to `(step, date, base_sha)` rather than to
the date alone. That scoping is what makes the marker safe: a marker written
against a different ledger base cannot satisfy this session. It also means that
on a first run nothing can legitimately be marked done, so a `[SKIP]` line means
another run is anchored to your base and its state has leaked into yours. Stop.
The one legitimate skip is a resume you deliberately performed.

**Author through the batch entry point, never by looping in prose.** The
per-agent authoring helpers exist to be called *by* `step_author_all`, not by a
driver iterating over the roster. A loop expressed as an instruction rather than
as code is the thing that quietly drops an agent, and a dropped agent looks
exactly like an agent that chose not to trade.

**Snapshots are keyed on the market date and are immutable across sessions.** A
row is dated by the market date its valuation was priced at, never by the
session date, because the benchmark and coin-flip series are dated from the
price index and a session-dated agent curve would drift off its own benchmark. A
later session may not overwrite an earlier session's row for the same date; it
is refused, and the caller warns. The consequence to expect — not to fix — is
that a session whose price store has not advanced records no new point, and its
valuation lands on the next real close.

**The derived-artifact steps are unconditional.** Baselines, the tax shadow and
the live leaderboard run whether or not any agent traded, and they run after the
portfolio mutations so the benchmark window matches the freshly-appended agent
snapshots. A session that commits posts and portfolios but no baselines skipped
a step; the deltas the site renders come out of that step, so the failure
surfaces as a chart that is subtly, permanently short.

**The bundle is cadence-invariant.** It emits every agent on disk, not only the
ones that ran, with carry-forward state for the rest. Build the portfolio
summaries once, from disk, right after snapshots, and pass the same object to
every consumer. Filtering them down to the agents that ran is how a book
disappears from a published day.

**Safety rails live in the broker.** Never re-implement a rail in the driver or
in a persona. The persona is aspirational; the broker is enforcing. A rail
copied into a prompt is a rail that is enforced only when the model feels like
it, and a rail copied into the driver is a second answer to a question that must
have exactly one.

**Abort beats improvise.** Every failure named below is cheaper as an aborted
session than as a partial one: nothing commits, and the next run starts clean.
The expensive failures in this system have all been sessions that kept going —
hand-built a missing input, hand-reconciled two artifacts that disagreed, or
retried the thing that had just stalled. If a helper returns something
surprising, report it and stop; do not repair it mid-run.

## The steps

Twenty-two functions in `scripts/daily_session.py`. They are described below in
**execution order**, grouped into phases. The authoritative list is the file
itself — re-derive it rather than trusting any transcription, including this one:

```bash
grep "^def step_" scripts/daily_session.py | sed 's/^def //; s/(.*//'
```

Several of them carry a done-marker (`@idempotent_step`) and return a neutral
value when re-entered. The pure prompt builders deliberately do not: on a resume
they must rebuild the real prompt so the driver's dispatch can run again, and
idempotency lives on the step that persists the result instead.

## Phase 0 — Anchor the run

No step function runs here. The driver realigns its working tree to the remote
tip, verifies its environment rather than repairing it, and anchors the session
date and ledger base. Everything after this point assumes those three things
happened, in that order. An environment check that fails is an abort: rebuilding
a dependency tree inside the timed run puts the slowest, least reliable
operation in the session directly on the critical path, which is precisely where
a run dies.

## Phase 1 — Market data and context

### Market data — `step_fetch_market_data`

Loads the benchmark series and the session's price context from the committed
store. **Reads:** the OHLCV store. **Writes:** an in-memory payload plus a
transient market snapshot the driver need not persist.
**On failure: fatal.** It refuses to return when the newest close among the
equity benchmarks is more than four calendar days old, which is the guard
against pricing a book on a dead vendor feed. A driver that catches this and
continues publishes a valuation from stale prices, and snapshots are immutable,
so that valuation is permanent. Do not work around it by fetching prices
yourself — the session has no outbound HTTP by design.
**Ordering:** first step after the session is anchored; everything that values a
position depends on it.

### Sentiment arm — `step_check_sentiment_freshness`

Records which arm of a pre-registered sentiment experiment this session actually
ran. **Reads:** the committed news-digest directory, and the roster's
`sentiment_arm` declarations. **Writes:** one row per session date to the
committed sentiment-arm log. **On failure: never fatal.** A missing news feed is
not a reason to lose a trading session, and a desk whose roster declares no
treatment arm records `not-running` and moves on.
**Ordering:** immediately after market data, before any agent reads anything.
The point is that the *record* answers the question rather than the schedule: if
the collector has not landed today's digests by the time the session realigns,
the treatment agents read yesterday's headlines and the arm is confounded — a
condition with no symptom at all unless something writes it down.

## Phase 2 — The trading round

The driver dispatches one prompt per trading agent and collects a result dict per
agent. Preserve the **full** response dict, not just the fields you recognise:
the research note is the only input to the allocator pipeline, and dropping it
does not crash anything — the allocator simply runs on zero signal and writes
empty holds while looking healthy.

`CONDITIONAL_ORDER_INSTRUCTIONS` and `render_active_triggers_for_agent(agent_id)`
belong in that prompt. The first documents the conditional-order schema; the
second shows the agent what it already has pending, so it can cancel or stack
rather than duplicating a thesis it forgot about.

### Author orders — `step_author_orders`

Converts one agent's trades into outbox orders. **Reads:** nothing on disk
beyond what the caller passes. **Writes:** the day's outbox, plus a committed
dropped-trade ledger entry for every trade that was not a valid order — a
non-BUY/SELL action, a missing ticker, non-finite or non-positive shares, or a
shape the order validator rejects. **On failure: drops, never raises.** An
unattended session must not die on loose model output. It returns only the
trades actually authored, and that return value — never the raw input — is what
downstream narration must use, so a dropped trade is never posted, journaled or
bundled as a phantom fill.
**Ordering:** an inner helper. Call it through `step_author_all`.

### Author cancels — `step_author_cancels`

Converts one agent's cancels into cancel requests on their own channel. **Reads:**
nothing. **Writes:** the day's cancel channel. **On failure: raises** on a
malformed request — a cancel names a specific order id and there is no safe
guess. Returns the number appended.
**Ordering:** an inner helper, same as above.

### The authoring batch — `step_author_all`

The single entry point the driver calls. **Reads:** each agent's portfolio, for
its base currency. **Writes:** the outbox and the cancel channel, for every
agent, in one pass. **On failure: fatal** — it calls the freshness guard first,
because this is the session's first irreversible write and therefore the last
place a stalled run can still be caught for free.
Idempotency here is deliberately split: the authoring half runs once, guarded by
the done-marker, so a resumed session never double-writes orders; the narration
filter runs on **both** paths, including the skip path, so the trades handed
onward are always trimmed to the authorable ones. Folding the filter into the
guarded body would lose it on a resume and let a dropped trade resurface as a
phantom fill.
**Ordering:** after every agent result is in hand, before fills.

### Fill — `step_fill_orders`

Invokes the paper broker on the day's outbox. **Reads:** the outbox and the
price store. **Writes:** the day's inbox, and — for successful fills —
portfolios. **On failure: fatal**, but note that a *rejection* is not a failure:
the broker's whole job is to refuse orders that violate a rail, and it records
each refusal with a reason code the agent reads next session. A day of nothing
but rejections is a working broker.
**Ordering:** after authoring, before valuation. Marker-idempotent, and the
broker additionally refuses to fill an order id already present in the target
inbox, so a double invocation cannot double-fill.

## Phase 3 — Valuation

### Snapshots — `step_update_snapshots`

Appends one dated valuation row per active book — any portfolio with state on
disk, including private books that never appear on a public surface. **Reads:**
portfolios, the price store, the market payload from phase 1. **Writes:** each
book's snapshot series. **On failure: partial by design.** A book holding a
foreign-currency position whose FX rate is unavailable is skipped for the day
with a warning rather than either aborting the run or publishing an unconverted
total; a date another session already published is refused, also with a warning.
Both cases land on the next market date.
**Ordering:** after fills, so the valuation reflects today's trades. Immediately
after it, build the portfolio summaries **once** and reuse that object for the
leaderboard, the bundle and the journal prompts. Recomputing it later is how the
three drift apart.

## Phase 4 — The allocator

Four steps, all opt-in. A roster with no `role: allocator` entry runs each of
them as a clean skip, and a desk that wants several allocators gives each one a
distinct channel prefix.

**Three of the four are roster-driven; the deterministic baseline is not.**
`step_resolve_manager_outcomes`, `step_build_manager_prompt` and
`step_apply_manager_decision` source their channel directories, book identity,
prose and gates from the resolved allocator spec, so renaming a channel or
retuning a risk budget is a `roster.yaml` edit and nothing more.
`step_build_baseline_manager` takes exactly one thing from that spec — whether
the baseline is enabled. Everything else about the control book is a module
constant in `engine/baseline_manager.py`: `STRATEGY_ID`,
`INITIAL_CAPITAL_EUR`, `POSITION_SIZE_EUR` and `MAX_POSITIONS`, with the book's
currency a string literal at the call site. **Read that before you rely on the
comparison.** A desk that sets a different `initial_capital` and a different
home currency in `roster.yaml` still gets a euro-denominated control book at the
constant's capital, so every "did the allocator beat the rules-only baseline?"
answer carries an uncontrolled FX leg and a size mismatch — and nothing errors,
because both books value correctly on their own terms. Either edit the
constants to match your desk, or disable the baseline and bring your own
control.

**A multi-allocator desk must thread `allocator_id` through all four calls.**
Omitted, the resolver defaults to the first allocator in the roster, so the
second one's steps silently operate on the first one's channels and book. The
control book is worse than that: its id is the same module constant for every
allocator, so two allocators share one baseline portfolio no matter what you
pass, and each rebalance overwrites the other's.

The allocator reads the trading agents' research notes and runs its own book on
its own channels. Those channels are isolated from the public ones on purpose:
its orders never enter the public outbox and its fills never enter the public
inbox, so the narrative layer cannot accidentally join them. This phase runs
after fills and snapshots — so the books it reads are current — and before the
narrative phase, so nothing downstream can see it.

One consequence of that ordering: phase 3 has already snapshotted every book on
disk by the time this phase mutates the allocator's and the control's, so their
**valuation series lag the trades in them by one session**. The books are
correct; the curves are a day behind. Compare an allocator against its control
on the same lagged basis, never against a trader curve that was snapshotted
after its own fills.

### Resolve past decisions — `step_resolve_manager_outcomes`

Turns matured allocator decisions into numeric outcome memory: forward return
and a reference-index comparison for each non-hold position that has reached its
horizon. **Reads:** the allocator's review artifacts, the price store, the global
reference series. **Writes:** a resolved-outcomes file, atomically.
**On failure: fatal**, but a desk with no allocator returns immediately.
**Ordering:** must run **before** the prompt is built, so the allocator sees
freshly-matured outcomes in the same session that produced the decisions they
came from. Marker-idempotent.

### Deterministic baseline — `step_build_baseline_manager`

Runs a rules-only rebalance of a benchmark book from the same research notes the
allocator gets — the control the model-driven allocator has to beat. **Reads:**
agent research notes, the price store, and `alloc.baseline_enabled` — but *not*
the rest of the allocator spec: the book's id, initial capital, currency,
position size and position cap are the module constants named above. **Writes:**
the benchmark book's portfolio and trade ledger. **On failure: warns per trade** rather than aborting
the batch. It rebalances only on the first weekday of the month, or on the very
first run when the book does not yet exist; on every other day it deliberately
does nothing, while snapshots still accrue for it.
**Ordering:** after outcome resolution, before the prompt. Marker-idempotent.

### Build the decision prompt — `step_build_manager_prompt`

Assembles the allocator's context — research notes, its own book, its resolved
outcome memory, its policy and risk-budget prose — and returns it **already
wrapped in the allocator persona**. **Reads:** research notes, the allocator's
portfolio and channels, the ticker registry, the price store. **Writes:**
nothing. Returns an empty string when no allocator is configured.
**On failure: none to speak of** — it is a pure builder with no side effects,
which is why it carries no done-marker: on a resume it must rebuild the real
prompt so the driver's dispatch can run again.
**Ordering:** after outcome resolution. Do not wrap the returned string a second
time; that duplicates the entire persona inside the prompt.

### Apply the decision — `step_apply_manager_decision`

Parses the allocator's response, applies the conviction gate **in code**, writes
the review artifact, converts non-hold positions into orders on the allocator's
channel, and fills them through the same broker, rails and fees as everything
else. **Reads:** the allocator's book and channels, the price store. **Writes:**
the review artifact **every day, hold or not** — it is the load-bearing record
of what the allocator decided and why — plus its outbox, inbox and portfolio.
**On failure: degrades.** A malformed or non-JSON response must reach this step
as `None`, which produces a placeholder review and no orders. A hold, a
low-conviction day and an unparseable day are all the expected normal; an empty
allocator outbox is not an error.
**Ordering:** last of the four. Marker-idempotent.

## Phase 5 — Narrative

### Leaderboard — `step_build_leaderboard`

Ranks the books from the portfolio summaries built in phase 3. **Reads:** the
summaries, the price store, each agent's `initial_capital` from the roster, and
— this is the input most easily missed — each agent's **committed baseline
series**, both the passive benchmark and the coin flip. **Writes:** nothing; it
returns rows.

**The board does not rank on return.** It ranks on the book's own-currency
return minus its benchmark's, which is FX-free by construction, with the raw
converted return kept alongside as a column. A row that has **no benchmark
series** cannot produce that figure, so it sorts null-last, on raw return
instead. The consequence for anyone standing a desk up: **before your first
baseline backfill you have no series at all, so every row falls into the
null-last branch and the whole board is silently ranked on a different metric
than the one it will use tomorrow.** It does not warn, and the columns look
identical. Backfill baselines before you read a first board.

**On failure: it drops, at two grains.** A single book that cannot be valued —
a missing price, an unresolvable currency, no inception anchor — is dropped from
the board rather than published at zero, so a *short* board is the symptom to
watch for; and if no book at all can be valued the result is an empty list,
rather than an invented one.

**Ordering:** before anything that narrates standings — which puts it *before*
the baseline refresh in phase 7, not after. That is the live behaviour and it
has a visible consequence: the benchmark delta published on a given day is
measured against a baseline series that is one session stale. Do not reorder to
"fix" it; the two curves are dated from the price index either way, and moving
the refresh earlier would rank today's books against a benchmark window that
the day's fills have not yet been snapshotted into. Just know which day's
benchmark the number is against.
Never hand-roll this from the snapshot series: the first persisted snapshot is
not inception for a book seeded with non-cash positions, so differencing the
series understates exactly those agents. The helper anchors to each agent's
configured inception capital, which is what every other consumer of a return
figure does.

### Load journals — `step_load_memories`

Reads each agent's journal off disk so the narrator can quote specific entries.
**Reads:** the agent-memory files. **Writes:** nothing. A missing journal becomes
an empty string so a first session still renders.
**Ordering:** before the narrator prompt.

### Narrator prompt — `step_build_oracle_prompt`

Builds the narrator's daily prompt from the market payload, the agent results,
the leaderboard and the loaded journals. **Reads:** the session's day number.
**Writes:** nothing; returns the prompt. No done-marker, for the same reason as
the other builders.
**Ordering:** after the leaderboard, before the post round — the narrator runs
first so it frames the day, and the agents then react to that framing. Pass no
posts here; they have not happened yet.

### Post prompts — `step_build_post_prompts`

Builds one post prompt per **trading** agent, optionally carrying the narrator's
draft so each agent can react to it as well as to the day's raw moves. **Reads:**
the roster. **Writes:** nothing. The narrator is excluded — it has its own path.
**Ordering:** after the narrator response is parsed.

### Persist content — `step_save_content`

Writes the day's posts, the narrative draft, and the output bundle. **Reads:**
everything the session has produced. **Writes:** the posts file, the draft, and
the bundle. **On failure: fatal** — the bundle is the single artifact every
downstream consumer reads, so a half-written day is worse than no day.
**Ordering:** after the post round. Marker-idempotent. Pass the portfolio
summaries built in phase 3: the bundle must carry every agent on disk, not only
the ones that ran.

## Phase 6 — Journals

### Journal prompts — `step_build_memory_update_prompts`

Builds a session-end journal-rewrite prompt for every agent, each one embedding
that agent's current journal so the dispatch is self-contained. **Reads:** the
agent-memory files. **Writes:** nothing.
**The narrator does not get the trader template.** All three of the trader
template's fact slots are structurally empty for an agent that holds no book —
no trades, a zero-valued portfolio summary, and posts that live outside the
per-agent post map — so a narrator fed that template writes about a dark desk
and then compounds the error, because next session it reads its own account of a
desk that was never dark. It gets a narrator-shaped prompt instead, and
**`leaderboard` and the narrator's own posts are its only session facts**: omit
them and the prompt carries nothing but the previous journal. Note that the
narrator's entry is currently written under a fixed agent id rather than
resolved from the roster's `role: narrator`, which is a coupling worth knowing
about if your narrator carries a different id.
**Ordering:** after content is saved, so the journals describe a session that is
already on disk.

### Persist journals — `step_save_memories`

Writes the rewritten journals back. **Reads:** nothing. **Writes:** the
agent-memory files. **On failure: skips, never truncates** — a blank or missing
response leaves that agent's journal untouched, so a partial round cannot wipe a
history. Returns the number actually written; compare it against the number of
agents you dispatched, because a silent shortfall here is a dispatch that never
happened.
**Ordering:** last of the narrative phase. Marker-idempotent.

## Phase 7 — Derived artifacts

All three run unconditionally, every cadence, whether or not anything traded.

### Baselines — `step_build_baselines`

Recomputes each book's passive-benchmark and coin-flip series from day one to
today. **Reads:** the price store, the universes, the roster. **Writes:** the
baseline series, **append-or-refuse per date** — the same mutability contract as
the agent snapshot it shares a chart with, so a revised close cannot move a
benchmark curve retroactively under a frozen agent curve. Restating a published
date is a separate, deliberate, disclosed operation.
**Ordering:** after all portfolio mutations, so the benchmark window matches the
snapshots just appended. Marker-idempotent.

### Tax shadow — `step_build_tax_shadow`

Recomputes a per-agent after-tax shadow ledger from the committed trade history.
**Reads:** each book's trades, and the tax rate from the roster's
`globals.jurisdiction` block. **Writes:** the shadow ledgers. Reporting only —
it never mutates portfolio cash, and it is cheap enough to always run.
**A roster with no `jurisdiction` block gets a rate of zero**, which is the
neutral default for a desk that has not declared one — so the ledger is labelled
after-tax while applying no tax. Declare the block or read the output as gross.
**Ordering:** after baselines. Marker-idempotent.

### Live leaderboard — `step_write_current_leaderboard`

Writes the standings artifact a live front page reads, stamped with a timestamp
and a trigger label so a consumer can tell which cadence produced it. **Reads:**
nothing — it takes the rows from phase 5. **Writes:** the current-leaderboard
file, full overwrite. **On failure: derived state**, so a driver that also runs
this opportunistically elsewhere should treat an error as non-fatal there.
Reuse the rows already computed; recomputing them here can publish a board that
disagrees with the bundle written minutes earlier.
**Ordering:** last artifact before the commit. Marker-idempotent.

## Phase 8 — Publish

### Commit and push — `step_git_commit_push`

**Reads:** the working tree. **Writes:** a commit, and the remote.
**On failure: fatal**, and it re-runs the freshness guard first — the last gate
before the work becomes public, re-checked here as well as at authoring time
because a stall can happen anywhere in between.
It stages the data tree, commits anything the driver has not already committed
itself, and pushes with an **explicit refspec** so a driver working on a
throwaway branch still advances the trunk rather than publishing its own branch.
When the primary push is refused it falls back to pushing the current branch and
says so on stdout, leaving the merge to whatever downstream automation you have
wired up; read that output before claiming success. On success it clears the
session state and the anchor — a finished session leaves no state behind, which
is why this step marks nothing done.
**Ordering:** last. Do not push by hand: a bare push publishes whatever branch
the driver happens to be on, which is indistinguishable from success until
someone notices the published site has not moved in a week.

## Price store

A session performs no outbound HTTP. Prices come from the committed OHLCV store,
populated out of band by `python scripts/fetch_ohlcv.py`. Populate it before the
first session; refresh it on a schedule that completes before the session runs.
Universe membership is committed too, for the same reason — a resolver that
reaches for the network at session time is a resolver that fails in a sandbox.

`scripts/fetch_market_data.py` refuses to price against a store that has stopped
advancing. That refusal is the guard against publishing a valuation from stale
prices, and it is the one failure that must stay fatal.

Two properties of the store shape the protocol. It holds **complete daily bars
only** — a run that asks the vendor for today's bar gets a partially-formed one
for anything that trades around the clock, and an immutable snapshot then
freezes it. And its prices are in each ticker's own ISO currency, normalised once
at ingest; no read path may rescale them, and every path that sums positions
across currencies must convert first.

## Cadences

A session and a valuation-only refresh are different categories. A refresh
composes a strict subset — market data, snapshots, baselines, tax shadow,
leaderboard — and runs no agents. That is every unconditional artifact of
phase 7, which is what "unconditional, every cadence" means above. Journals, posts and narrative belong to sessions only. Both
categories call the *same* step functions; a cadence is a choice of subset, never
a second implementation.

Conditional orders are the third path. They are authored inside a session and
fire outside one, from a separate worker that evaluates pending triggers against
live prices and executes through the same broker entry point. That worker
applies the same order-level rails and must not race the session: give it a
blackout window around the session's start and tail, and remember that the
blackout end is a function of the session start — move one and move the other, in
the same change. The blackout narrows the race; the freshness guard is what makes
it correct.

## What this document does not cover

Model selection, dispatch mechanics, prompt engineering, scheduling, and how you
get credentials to a broker. Those are the driver's business.
