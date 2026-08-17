# Training pipeline — audit and implementation plan

Written after reading `ai/`, `game/`, `config.py`, `model_manager.py`,
`param_bounds.py` and the model `config.json` files, plus a profiling run of a
single 96-simulation MCTS search on a 40-stone 9×9 position.

Everything below is either **measured** or **traced through the code**. Where a
claim is a judgement call it says so.

---

## Summary of what is wrong

| # | Finding | Kind | Severity | Status |
|---|---|---|---|---|
| B1 | `temperature_init` / `temperature_final` are never plumbed into self-play | legacy / dead setting | **High** | ✅ fixed |
| B2 | Policy target overflows to `inf` → `NaN` at low temperature | latent bug | **High** (armed by fixing B1) | ✅ fixed |
| B3 | `save_weights_now()` drops the champion from the checkpoint | bug | **High** | ✅ fixed |
| B4 | Replay buffer is never persisted — every restart discards it | design gap | **High** | ✅ fixed |
| B5 | Elo inflates ~5 points/iteration forever once win-rate saturates | bug | Medium | ✅ fixed |
| B6 | Gate and random-eval searches ignore configured `c_puct` / `fpu_reduction` | plumbing | Medium | ✅ fixed |
| B7 | `allow_pass=False` is applied at every tree depth, not just the root | bug | Low-Medium | ✅ fixed |
| B8 | `total_games` counts games that a force-stop cancelled | cosmetic | Low | ✅ fixed |
| P1 | `Board.neighbors()` — 152k calls/search, ~33% of search time | perf | **High** | ✅ fixed |
| P2 | MCTS builds a `GameState` for every legal move; ≤3% are ever visited | perf | **High** | ✅ fixed |
| P3 | Move legality is computed twice for every child | perf | Medium | ✅ fixed |
| P4 | Terminal nodes are fully re-scored on every visit | perf | Low | ✅ fixed |

All five phases are **implemented and tested** (549 tests pass, up from 297).
Phase 4 is the self-atari filter (§7), shipped off by default — and the
measurement that came with it contradicts HEURISTICS.md's ranking of it.
Phase 5 is the versioned input encoding (§6), with `v2_12` available for new
models and a behaviour-preserving migration for existing ones. See "What
shipped" at the end of this document.

Plus the two things you asked about: **17-plane history input** (§6) and
**prevent self-atari** (§7).

---

## 1. Where the time actually goes

Profile of one `MCTS.search(..., num_simulations=96)` on a mid-game 9×9 board,
`torch.set_num_threads(1)` (i.e. what a self-play worker actually does):

```
MCTS search @96 sims          0.878 s wall
  _expand                     97 %   of search time
    rules.is_legal_move       58 %   (11,090 calls)
    Board.neighbors           33 %   (152,067 calls; 0.40s + 0.17s is_on_board)
    play_move → apply_move    41 %   (3,233 calls — building child states)
  network.predict             16 %   (97 calls @ 1.85 ms)
```

**The neural network is 16% of self-play. The rules engine is ~84%.**

That inverts the usual assumption in this project — the comments in `config.py`
and `network.py` are all written as if the net were the constraint (MPS, filter
counts, presets). It is not, and it will not be until the engine is roughly 5×
faster. Every hour spent making the net smaller to go faster is buying ~16% of
a speedup.

The concrete causes:

- **P1 — `neighbors()`** allocates a fresh list and runs four bounds checks per
  call, 152,067 times per search. The board geometry is fixed at construction.
- **P2 — eager child construction.** `_expand` copies the state and plays the
  move for *every* legal move (3,233 child states per search) so it can hand
  each child a `GameState`. With 96 simulations at most 96 of those are ever
  visited. ~97% of that work is discarded.
- **P3 — double legality.** `get_legal_moves()` proves a move legal by
  simulating it; `play_move()` → `apply_move()` → `is_legal_move()` then
  simulates it again from scratch.
- **P4** — `_terminal_value` runs a full Chinese scoring pass every time a
  terminal node is visited, instead of once per node.

---

## 2. Correctness bugs

### B1 — `temperature_init` and `temperature_final` do nothing

The full path exists and is a dead end:

- `param_bounds.py` exposes both sliders,
- `model_manager.TrainingParams` stores them,
- `Config.from_model` reads them into `config.mcts`,
- `training_routes.apply_params` live-tunes them onto `trainer.config.mcts`
  and persists them to `config.json`,
- `Trainer.train()` passes **only `temperature_threshold`** to
  `run_self_play_batch()`, which does not accept the other two at all,
- so `play_self_play_game` falls back to its own signature defaults:
  `temperature_init=1.0, temperature_final=0.1`.

Every model has been training on a 1.0 → 0.1 schedule regardless of the UI.
`hot-boi` is configured 0.8 → 0.2 and is running 1.0 → 0.1: a hotter opening
than intended, which means noisier move selection *and* noisier policy targets
in exactly the phase that produces training data.

### B2 — the policy target overflows to NaN, and fixing B1 arms it

`ai/mcts.py:229`:

```python
policy[_action_index(action)] = float(child.visit_count) ** (1.0 / temperature)
```

`policy` is `np.float32`. Overflow happens when `(1/T)·log₁₀(N) > 38.5`.
Verified:

```
temperature_threshold=30, temp schedule 0.5→0.001
  move 28  temp=0.0343  exponent=29.2  visits=50  →  float32 inf
  move 29  temp=0.0176  exponent=56.7  visits=5   →  float32 inf
```

After `policy /= policy.sum()` that entry becomes `inf/inf = NaN`, the training
target is `[NaN, 0, 0, …]`, the policy cross-entropy is `NaN`, and one optimizer
step turns the whole network into `NaN`.

It has never fired, for one reason only: B1 pins `temperature_final` at 0.1 for
every model. But `param_bounds` lets the slider go to **0.001**, and
`MCTSConfig`'s own default (the `run_training.py` CLI path) *is* 0.001.

**So B1 and B2 must be fixed in the same commit.** Fixing the plumbing alone
hands the user a slider that destroys the network at the low end.

Fix: normalise before exponentiating, in float64 —
`exp((1/T)·(log N − log N_max))` — which is scale-invariant and cannot
overflow. Same treatment in `_select_action`.

### B3 — a manual weights save silently un-gates the champion

`Trainer.save_weights_now()` (the `/training/api/save_weights` route) calls
`save_weights(...)` **without** `champion_state_dict`. The written file then has
no champion entry. On the next restart, `_try_load_weights` finds none and does:

```python
self.eval_network.load_state_dict(self.network.state_dict())
```

— i.e. it promotes whatever un-gated candidate happened to be in the training
network to champion, and that network then generates all self-play data. One
click of "save weights" defeats the promotion gate across a restart. Fix is one
argument.

### B4 — the replay buffer dies on every restart

`Trainer.__init__` always constructs an empty `ReplayBuffer`. `_try_load_weights`
restores iteration, Elo, total games, optimizer state and champion — but not the
samples. And `web/app.py:switch_model()` builds a **new Trainer**, so switching
models and switching back also empties it.

Consequences, in the order they bite:

1. Up to 50,000 samples — many hours of self-play — are thrown away per restart.
2. The first iteration after a restart trains on one batch of games (5–20 games,
   ~400–1,600 samples) with a *restored* Adam state and a full-size step count.
3. That candidate then faces a 20-game promotion gate against a champion trained
   on the full buffer. It loses. The rejection streak counter climbs, and after
   5 rejections the stall breaker resets the training net — which looks like
   "training has stalled" when the real cause was a restart.

For someone driving this from the web UI, this is the single largest loss of
learning progress in the system.

Fix: persist the buffer next to `weights.pt` (`replay_buffer.pt`), save it in
`_save_weights()`, load it in `__init__`. Guard it with the same arch/board
signature as the checkpoint so a 9×9 buffer is never loaded into a 13×13 model,
and cap what is written so the file stays bounded.

### B5 — Elo is a function of iteration count, not strength

```python
self.elo = compute_elo_update(self.elo, 500, win_rate)   # K=32, fixed anchor
```

Once `win_rate` saturates at 1.0 against the random bot — which
`HEURISTICS.md` already documents as happening early — the update is
`32 × (1 − expected)`, which is **strictly positive forever**:

| current Elo | Elo gained per iteration at 100% win rate |
|---|---|
| 778 | +5.4 |
| 1,000 | +3.3 |
| 1,500 | +1.1 |

It never reaches zero. `hot-boi` at "778 Elo / 22k" has been accumulating a
participation trophy, not a rating. The displayed kyu rank is not measuring
anything past the point where the random bot stops being a test.

Fix: two things, both cheap.
- Freeze the anchor-based update when `win_rate >= 1.0` (no information in a
  saturated test) and say so in the metrics.
- Track a second, meaningful series: **gate Elo**, using the
  `compute_pairwise_elo` that already exists in `evaluator.py` (currently only
  used by the arena). The promotion gate already plays 20 rated games between
  candidate and champion every iteration — that measurement is being thrown
  away after a single boolean promote/reject decision.

### B6 — the gate does not search the way you configured

`evaluator._eval_worker` and `_play_gate_game` build `MCTS(...)` without
`c_puct` or `fpu_reduction`, so both always use 1.5 / 0.35. Tune `c_puct` and
self-play uses your value while the gate judges the result under a different
search. Both sides of the gate get the same wrong value so the *comparison* is
not biased — but you stop measuring the configuration you are actually
training. Fix: thread them through, same as `restrict_eye_fill` already is.

### B7 — pass is banned at every depth, not just at the root

Self-play disables pass before move ~29 (`min_pass_move`). But `_expand` takes
`allow_pass` as a per-*search* constant and applies it to every node, so during
an opening search the tree cannot end a game **at any depth** — a line 30 plies
deep still cannot pass. Terminal values are therefore unreachable during the
first ~29 moves of every game, and the value head's only grounding in that phase
is its own estimate.

Fix: decide per node from `node.state.move_number` rather than from the root's.

### B8 — `total_games` over-counts on force-stop

`self.total_games += num_self_play_games` runs unconditionally, before the
force-stop check that can have cancelled part of the batch. Cosmetic, one line.

---

## 3. Phased implementation

Each phase is independently shippable and independently testable. Phases 1–3
change no defaults and no model behaviour except where a bug is being removed.

### Phase 0 — safety net

Write failing tests first, so each fix is demonstrably a fix:

- `tests/test_policy_target_numerics.py` — assert `np.isfinite(policy).all()`
  and `policy.sum() ≈ 1` across `T ∈ {0.5 … 0.001}` and visit counts up to 800.
- `tests/test_temperature_plumbing.py` — a configured `temperature_init` /
  `temperature_final` reaches `play_self_play_game` (monkeypatch and capture the
  kwargs).
- `tests/test_checkpoint_champion.py` — `save_weights_now()` preserves the
  champion across a save/load round trip.
- `tests/test_replay_persistence.py` — buffer survives a Trainer rebuild; a
  buffer from a different board size is rejected.

### Phase 1 — correctness (B1, B2, B3, B7, B8, B6)

1. `mcts.py`: log-space policy construction (B2) — **before** the plumbing fix.
2. `self_play.py` + `trainer.py`: accept and forward `temperature_init` /
   `temperature_final` through `run_self_play_batch` into the worker task dicts
   (B1). Workers are separate processes and only see the dict.
3. `trainer.py` / `checkpoint.py`: pass `champion_state_dict` in
   `save_weights_now` (B3).
4. `mcts.py`: per-node pass legality (B7).
5. `trainer.py`: credit `total_games` from games actually completed (B8).
6. `evaluator.py`: thread `c_puct` / `fpu_reduction` into both worker paths (B6).

Expect a visible change in self-play behaviour from (2): the opening
temperature drops from 1.0 to whatever each model is configured for. This is
the correct behaviour but it *is* a change — worth one iteration of eyeballing
`policy` entropy before and after.

### Phase 2 — durability (B4, B5)

7. `ai/replay_store.py` (new): `save_buffer` / `load_buffer`, signature-guarded
   on `(board_size, num_input_planes)`, atomic temp-file write like
   `checkpoint.py`. Wire into `Trainer._save_weights` and `Trainer.__init__`.
8. `trainer.py` + `evaluator.py`: freeze anchor Elo at saturation, and record a
   gate-derived pairwise Elo series in `training_log.jsonl`. Add the series to
   the metrics chart.

### Phase 3 — throughput (P1–P4)

9. `game/board.py`: module-level neighbour table per board size, built once,
   `neighbors()` becomes a lookup. Pure win, no behaviour change (P1).
10. `game/rules.py`: extract
    `_simulate(board, color, r, c) -> (test_board, captured_any, own_group, own_libs)`
    shared by `is_legal_move`, `apply_move` and (later) the self-atari filter
    (P3).
11. `ai/mcts.py`: **lazy child expansion.** A child holds `(action, prior)` and
    materialises its `GameState` on first selection. This is the largest single
    win and the one change in this phase that needs care — `best_child`,
    `_backup` and the terminal check all currently assume `child.state` exists
    (P2).
12. `ai/mcts.py`: cache terminal values on the node (P4).

Target: **~2× self-play throughput**, i.e. double the games per hour at
identical quality. Verify with a fixed-seed benchmark script committed under
`tests/` (not in the pytest suite) so the number is reproducible before/after.

### Phase 4 — self-atari filter (§7)

Sequenced after Phase 3 deliberately: it needs the shared `_simulate` helper
from step 10, or it doubles the cost of the hottest function in the engine.

### Phase 5 — input feature versioning (§6)

Independent of everything above; can be done at any point after Phase 1.

---

## 6. Your question: AlphaZero's 17 planes / 8-step history

### What it actually is

AlphaGo Zero's input is 17 planes: the last **8** board positions for each
colour (8 + 8) plus one constant colour-to-play plane. There are no liberty
planes and no ko plane — the history *is* how the network sees ko.

Your encoder is different and, feature-for-feature, richer already: 2 stone
planes, 6 liberty planes, an explicit ko plane, and colour-to-play (10 total).

### Would it help you?

Partly — but not for the reason it helped AlphaGo Zero.

History planes buy two things in Go:

1. **Ko / repetition awareness.** This is most of why AGZ needed them. You
   already have an explicit ko plane *and* full positional superko in the rules
   engine. This benefit is largely already banked.
2. **Locality — where the opponent just played.** Go is intensely local; most
   good replies are within a few points of the last stone. This is real, it is
   not banked, and for a small network on 9×9 it is probably the single
   cheapest policy-head improvement available.

The locality benefit comes almost entirely from the **last one or two moves**.
Planes 3 through 8 of AGZ's history are doing very little that the current
position does not already say.

### Costs, specific to this codebase

- **Network cost: negligible.** Input conv 10→18 channels adds ~2% to the
  forward pass. This is not the constraint (see §1 — the net is 16% of runtime).
- **`GameState.copy()` cost: the real risk.** `copy()` runs ~3,200× per search.
  If history is stored as "8 grids copied per state" it will cost more than the
  network does. It must be an **append-only list of immutable grids**, so
  `copy()` copies references, not boards. Non-negotiable if this is built.
- **Two hardcodes must go**: `game_state.encode_for_nn` allocates
  `np.zeros((10, size, size))` literally, and `trainer._collapse_diagnostics`
  reads **plane index 9** as the turn-colour plane. Both need to come from a
  layout descriptor.
- **It cannot be a toggle on an existing model.** Changing plane count changes
  `input_conv.in_channels`, so `weights.pt` becomes shape-incompatible —
  `checkpoint.py` will (correctly) refuse to load it. This is a
  **create-model-time choice, frozen per model**, exactly like `NetworkParams`.
- **One new risk to watch.** History gives the value head more shortcut features
  (move count, phase) on top of the turn-colour plane it can already collapse
  onto. Watch `value_std_black` / `value_std_white` after enabling; the collapse
  guard is the right instrument and it already exists.

### Recommendation

Yes — add it, as a **versioned per-model feature set, defaulting to today's
encoding**, and start with the cheap variant rather than the full 17:

| id | planes | contents |
|---|---|---|
| `v1_10` (default) | 10 | current encoding — every existing model |
| `v2_12` | 12 | v1 + last move one-hot + opponent's previous move one-hot |
| `v3_18` | 18 | v1 + 4-step stone history (4 own + 4 opponent) |

`v2_12` is where I would put the effort: it captures the locality signal, adds
two planes, and keeps `GameState.copy()` cheap. `v3_18` exists for when you want
to A/B whether deeper history buys anything on 9×9 — my expectation is that it
buys very little over `v2_12`, and this framing lets you find out instead of
guessing. (Full AGZ-equivalent 8-step history would be 26 planes here, since you
keep the liberty and ko planes. I would not start there.)

Implementation shape:
- `game/features.py` — a `FeatureSet` descriptor: plane count, plane names,
  turn-colour plane index, and the encode function. One place that knows the
  layout.
- `NetworkParams.input_features: str = "v1_10"` + `num_input_planes` derived
  from it, stored per model, in the checkpoint arch signature (which already
  guards mismatches).
- Validate with the `HEURISTICS.md` protocol: `copy_model` a twin, train both,
  play them head-to-head.

---

## 7. Your question: prevent self-atari

`HEURISTICS.md` calls this the best sample-efficiency lever available and I
agree, for the reason it gives: unlike eye-filling, pointless self-atari is
*frequent* in exactly the weak-network regime you are in. Every one of those
moves costs a search slot and puts probability mass on a move that hands over
stones. Removing them sharpens the policy target and concentrates the search
budget in one change.

### Definition (as specced, unchanged)

A move by `C` at `p` is a pointless self-atari iff, after placing and resolving
captures:

1. it captured **nothing**, and
2. the group containing `p` has exactly **1 liberty**, and
3. that group is **larger than `k` stones** (`k` = `self_atari_max_stones`,
   default 1).

Condition 1 exempts ko captures and most snapbacks; condition 3 exempts
throw-ins, the main small-group tesuji.

### What can go wrong — read this part before enabling it

1. **It is a heuristic, not a theorem.** This is the important difference from
   `restrict_eye_fill`, which rests on a proof. There is no proof here, and the
   `k` threshold has nothing behind it but judgement. Real moves it can remove:
   nakade placement larger than `k`, eye-space reduction, seki manipulation,
   squeeze and sacrifice tesuji, and ko-threat creation.
2. **The promotion gate cannot detect the damage.** `HEURISTICS.md` is right
   about this and it bears repeating: both sides play under the same filter, so
   the gate measures relative strength *under the handicap* and is blind to the
   handicap's own cost. The gate-Elo ladder will keep climbing while the model
   gets quietly worse. Only the twin-model A/B — head-to-head with the filter
   **off for both sides** — can tell you.
3. **It can empty the move list.** In a filled endgame position every legal move
   can be a self-atari. The filter would then leave only pass, silently, at
   every node in the subtree. The spec in `HEURISTICS.md` does not cover this.
   **Guard: if the filter removes every move, return the unfiltered list.**
   Same guard is worth adding for `restrict_eye_fill` while we are there.
4. **Cost, if implemented naively.** Per §1, `is_legal_move` is already 58% of
   search time. A self-atari check that re-simulates each move would roughly
   double the cost of the hottest function in the engine and could easily eat
   more throughput than the sharper targets buy back. Riding it on the shared
   `_simulate` helper (Phase 3, step 10) makes it nearly free, because the
   simulation has *already* computed the resulting group and its liberties —
   conditions 1–3 are then three comparisons on values in hand. **This is why
   Phase 4 comes after Phase 3.**
5. **Superko corner case.** As with every filter in `HEURISTICS.md`: under
   positional superko a self-atari can occasionally be the only legal non-pass
   move. Pass is always available so there is no deadlock, but pass is not
   equivalent when a ko is live. Rare; accepted.

### How it will be built

Following the `HEURISTICS.md` plumbing recipe exactly, so it behaves like every
other optional restriction:

- `game/self_atari.py` — mirrors `game/eyes.py`: rationale in the docstring, one
  scan function, one point function.
- Filtered in `rules.get_legal_moves` alongside `restrict_eye_fill`, propagated
  through `GameState.copy()` so it holds at every depth of the tree — never in
  `is_legal()` / `play_move()`, so humans, replays and stored games are
  untouched.
- Two settings, **both defaulting to off/1**:
  `restrict_self_atari: bool = False`, `self_atari_max_stones: int = 1` (1–4).
- Applied to: self-play ✅, both sides of the gate ✅ (otherwise you measure the
  filter), the random bot ❌ (it is the Elo anchor).
- Off must reproduce current behaviour byte-for-byte — asserted by a test that
  compares filtered and unfiltered move lists.

### Tests (`tests/test_self_atari.py`)

The over-restriction cases matter more than the detection cases:

- throw-in into opponent eye space stays playable
- snapback setup stays playable
- ko capture stays playable (exempt via "captured something")
- a move that captures is never blocked, even at 1 resulting liberty
- ladder/atari positions where self-atari is the only defence
- large self-atari is blocked
- exact behaviour at the `k` boundary (`k` and `k+1` stones)
- the other colour's moves are unaffected
- **the filter never returns an empty move list**
- off-by-default: filtered and unfiltered lists are identical

### Expectation to set

Do not expect the gate ladder or `win_rate_vs_random` to validate this. Expect
a **discontinuous step** in `win_rate_vs_random` at the iteration you switch it
on (only the network is filtered there), and expect the gate ladder not to step
at all, because it is self-referential. The A/B rig is the answer, and it costs
two training runs.

---

## Order I would actually do this in

1. **Phase 0 + 1** — the bugs. B1/B2 together, B3 and B4 next; these are losing
   you real work right now.
2. **Phase 2** — buffer persistence is the highest-value single change for how
   you operate this thing day to day.
3. **Phase 3** — 2× throughput, and it is the prerequisite that makes the
   self-atari filter affordable.
4. **Phase 4** — self-atari, with the A/B rig, since it is the best remaining
   sample-efficiency lever.
5. **Phase 5** — `v2_12` features on a *new* model, A/B'd against a `v1_10`
   twin.

---

## What shipped (Phases 1 and 2)

345 tests pass, up from 297. All 8 existing models load unchanged.

**`ai/mcts.py`**
- New `visit_distribution()` builds the move distribution in log space, so the
  policy target cannot overflow at any temperature the sliders allow (B2).
- `search()` now samples the played move from the *same* distribution it returns
  as the training target, so a stored move always has support in its own label.
- `search(min_pass_move=...)` — the opening pass ban is evaluated per node
  instead of once per search (B7). `allow_pass` is unchanged for other callers.
- A root with no children returns a valid one-hot pass target instead of an
  all-zero vector.

**`ai/self_play.py`**
- `run_self_play_batch` accepts and forwards `temperature_init` /
  `temperature_final` (B1).
- New `build_self_play_task()` — one named place that constructs the dict
  crossing the process boundary, so it can be tested against the worker's
  signature without starting a pool. This is the drift that made B1 possible.
- Failed workers tag their stub record so a game that never happened is not
  counted (B8).

**`ai/trainer.py`**
- Forwards the temperature schedule, and `c_puct` / `fpu_reduction` to both the
  gate and the random-bot eval (B6).
- `save_weights_now()` writes the champion (B3).
- Replay buffer saved every iteration and restored on construction (B4).
- Tracks `gate_elo`, a rating that only moves when a candidate actually beats
  the champion (B5). Recorded in `training_log.jsonl` and `get_status()`.
- `total_games` credits completed games only (B8).

**`ai/replay_store.py`** (new) — signature-guarded, atomic buffer persistence.
Indicator planes are stored as uint8, which is lossless for the current encoding
and takes a full 50k buffer from ~179 MB to ~57 MB per save; a future encoding
with continuous planes falls back to float32 automatically.

**`ai/evaluator.py`** — `clamp_score()` and `performance_elo_gap()`;
`compute_elo_update(..., num_games=)` applies the half-game correction so the
anchor rating converges instead of drifting (B5).

**`ai/checkpoint.py`** — carries `gate_elo`; older checkpoints simply lack it.

**Tests added**: `test_policy_target_numerics.py`, `test_temperature_plumbing.py`,
`test_checkpoint_champion.py`, `test_replay_persistence.py`,
`test_elo_saturation.py`.

**One existing test repaired**: `test_game_move_counts_from_disk` globbed the
flat `iter_000001_game_0000.json` layout that `migrate_legacy_layout` moves games
out of, and sliced the *first* 20 games — always iteration 1, played by an
untrained network and the longest a model ever produces. It now reads through
`ai/game_store` and measures the most recent 50.

## What shipped (Phase 3)

**`game/board.py`** — precomputed adjacency (`neighbour_table`), built once per
board size and shared by every board including copies. `get_group`,
`get_liberties` and `liberty_count` bind the table and grid locally instead of
calling back through `self.neighbors()` per stone (P1).

**`game/rules.py`** — new `simulate_move()` answers legality, captures, the
resulting Zobrist hash and the new ko point from adjacency alone: no board copy,
no stone placement, no post-move re-derivation. `is_legal_move` is now a thin
wrapper over it, and `apply_move` reuses the same simulation instead of
validating and then redoing the identical work on the real board (P3).

Two facts make it exact, both resting on (row, col) being empty and adjacent to
the groups involved:

- (row, col) is necessarily a liberty of every adjacent group, so an opponent
  group is captured exactly when it has **1** liberty, and a friendly group
  keeps the merged group alive exactly when it has **more than 1**. No post-move
  board is needed to count them.
- Zobrist hashing is XOR of per-(point, colour) values, so the resulting hash is
  the current hash XOR the placed stone XOR each captured stone — an exact
  superko check with no board copy.

**`ai/mcts.py`** — children hold `(action, prior)` and materialise their position
only when the search selects them (`MCTSNode.ensure_state`), plus `__slots__`
and cached terminal values (P2, P4).

### Measured

The machine was running a training job at load 260 while this was verified, so
wall-clock is not comparable; these are the load-independent numbers. Same setup
as the original profile — 9x9, 40 stones, one 96-simulation search:

| operation | before | after | |
|---|---|---|---|
| `simulate_move` / `is_legal_move` | 11,090 | 7,953 | 1.4x fewer |
| `GameState.copy` | 3,233 | 97 | **33x fewer** |
| `Board.copy` | 11,090 | 97 | **114x fewer** |
| `network.predict` | 97 | 97 | unchanged |

`predict` staying at exactly 97 is the important one: the search performs the
same number of expansions and therefore the same amount of thinking. Only the
waste is gone.

A same-process CPU-time A/B of the old rules implementation against the new one
(both under identical load) measures **1.43x** on legality checks, on top of the
neighbour-table gain.

**Final wall-clock, `tests/benchmark_search.py` on an idle machine** (four
96-simulation searches at 0, 20, 40 and 60 stones):

| | before | after | |
|---|---|---|---|
| four searches | 1.079 s | **0.547 s** | **1.97x faster** |
| moves/sec, one worker | 3.71 | **7.31** | |
| `get_legal_moves` | 0.785 ms | 0.327 ms | 2.4x faster |
| neural network's share of search | ~30% | **~55%** | |

The last row is the durable result: the rules engine no longer dominates
self-play, so the network is now roughly half the cost and further engine work
has much less headroom. Any future speedup has to come from the network side
(batched evaluation) or from playing fewer wasted moves (the mercy rule).

### How this was kept honest

Rewriting the referee is the kind of change that passes every hand-written test
and is still wrong in a snapback, a multi-group capture or a superko that only
triggers after a particular capture. `tests/test_rules_equivalence.py` keeps a
verbatim copy of the ORIGINAL simulate-everything implementation as an oracle
and fuzzes thousands of positions from random and contact-biased playouts,
comparing the legality verdict for every point, the exact captured set, the
resulting board hash and the ko point. The oracle is deliberately not tidied —
it is the old code, kept as evidence.

`tests/test_mcts_lazy_expansion.py` pins the laziness itself: that unvisited
children really do stay unmaterialised (otherwise the optimisation could vanish
silently), that a materialised child's position is exactly parent-plus-one-move,
that `restrict_eye_fill` is still inherited at every depth, and that a cached
terminal score equals a freshly computed one.

`tests/benchmark_search.py` is committed but excluded from the pytest suite by
its filename. Run it before and after any further performance work.

## What shipped (Phase 5 — versioned input encoding)

**`game/features.py`** (new) — a registry of versioned encodings. `v1_10` is the
existing 10-plane encoding and remains the default; `v2_12` appends two planes:
the opponent's last move and the mover's own previous move.

The encoding is attached to the **network**, not the game state
(`GoNetwork.input_features`), so nothing has to be threaded through the dozens
of `GameState` construction sites. `features.encode_for_network(state, network)`
reads the layout off the network, which makes it impossible to pair a position
with the wrong plane count.

Also removed the last hardcoded layout assumption: the collapse guard was
reading `states[:, 9]` as the turn-colour plane, and now takes the index from
the feature set.

**Cost measured**: encoding is 0.082 ms either way (identical), and the wider
first convolution adds ~1,150 parameters on the Small preset — 320,701 to
321,853, +0.4%. This is not a throughput decision.

**Why `v2_12` and not AlphaGo Zero's 17**: AGZ's 17 planes are 8 board states per
colour plus colour-to-play, with no liberty or ko planes at all — history was
largely how it perceived ko. This project encodes ko explicitly and enforces
positional superko in the rules engine, so that motivation is already satisfied.
What history still buys is **locality**, and that signal lives almost entirely in
the last one or two moves.

### Migrations

**Nothing to migrate to keep what you have.** A `config.json` with no
`input_features` falls back to `v1_10`, which is what every existing model was
trained with. All 8 models load unchanged.

**To move a trained model onto `v2_12`** there is a widening migration
(`ai/feature_migration.py`, driven by `scripts/migrate_features.py`). Because
`v2_12` appends its planes and leaves 0-9 untouched, every existing weight still
addresses the right channel; the two new input weights are zero-initialised, so
the widened network computes the identical function on the day it runs. That
identity is asserted numerically in `tests/test_feature_migration.py`, including
against scrambled history planes to prove the new channels genuinely cannot
influence the output.

| | |
|---|---|
| Preserved | weights, gated champion, iteration, Elo, gate Elo, total games, stored games |
| Dropped | optimizer state (Adam re-warms in a few hundred steps) |
| Invalidated | the replay buffer — its positions were encoded with 10 planes, and a stored sample does not remember which move preceded it |

The buffer's signature guard rejects it automatically rather than silently
feeding a 12-plane network 10-plane data. `weights.pt` is backed up first, and
the migration is one-way.

`update_model` never accepts network params, so the encoding cannot be changed
on a trained model through the API — the freeze is structural, not just
documented.

### Documented in the app

`web/templates/info.html` gains an **Input Encoding** card next to the
architecture presets: the full plane table, why 12 rather than 17, what it is
expected to do to learning (and what to watch — the value head gains more
phase cues, which is what the collapse guard exists to catch), and the migration
story with the exact commands. Verified rendering in the browser.

### Honest status

`v2_12` is a well-supported idea, **not a measured result on this project**. It
has not been A/B'd here. The way to settle it is the same protocol as every
other change the promotion gate cannot see: copy a model, train one twin on each
encoding for the same number of iterations, then play them head-to-head.

---

## What shipped (Phase 4 — self-atari filter)

Two new settings, **both off by default**: `restrict_self_atari` and
`self_atari_max_stones` (1–4, default 1). Full plumbing per the HEURISTICS.md
recipe — `param_bounds` → `TrainingParams` → `Config.from_model` → trainer →
self-play / gate / random-eval workers → live-tuning API → play-vs-human →
info page. Applied to self-play and **both** sides of the gate; never to the
random bot, which is the Elo anchor.

Implementation is in `game/self_atari.py`; the filter runs in
`rules.get_legal_moves` and propagates through `GameState.copy()`, so a blocked
move is never expanded, never given a prior, and never appears in a policy
target at any depth.

**It never simulates anything itself.** `simulate_move(need_group_facts=True)`
retains the merged group's size and liberties from the walk it already does for
legality, reusing the flood fills the suicide check performed. Two early-outs
keep it off the hot path: a capturing move is exempt, and a move with two or
more empty neighbours provably has two liberties.

**One deliberate departure from the spec**: if the filter would remove every
legal move it hands them back rather than forcing a pass. The eye rule
deliberately does not get this guard — filling your own last two eyes is
provably useless, so passing really is at least as good. Self-atari has no such
proof, and a position where every move trips it is where the assumption is
weakest.

### Measured — and the prediction in HEURISTICS.md did not hold

HEURISTICS.md called this "the best sample-efficiency lever". Checking every
move of 60 stored self-play games per model against the filter, at the moment it
was played:

| phase | share of played moves that are pointless self-atari |
|---|---|
| first half — where training samples come from | **0.3 – 0.4%** |
| late game — past the sample cutoff | **5.5 – 7.1%** |

The moves it removes are overwhelmingly in the tail that produces **no training
data at all**. Overall incidence is 2.5–4.7% of moves, but only 1.8–2.2% land
inside the training window, and in the opening these networks already almost
never play one.

The cost, meanwhile, is paid on every move: ~15% on `get_legal_moves`, which is
~80% of search time — roughly **12% of self-play throughput**.

**So: a marginal trade for these models, not the headline win.** The late-game
tail it targets is better handled by the mercy rule, which deletes that phase
outright instead of filtering moves inside it, and costs nothing per move. Turn
on `resign_enabled` first. Reach for `restrict_self_atari` only if the twin-model
A/B shows it earning its 12% — and remember the gate cannot tell you, because
both sides play under the filter.

### Tests

`tests/test_self_atari.py` (21) and `tests/test_self_atari_plumbing.py` (13).
Every hand-built shape asserts its own preconditions against an oracle that
plays the move on a real board and reads the result back, so a test cannot
quietly stop testing what its name says. The over-restriction cases are the
point: throw-in, snapback, ko capture, connection-to-safety, the `k` boundary,
the other colour, and "never empties the move list". Fuzzing over real positions
confirms every blocked move satisfies all three conditions against a real board
(30+ seen) and that no capturing move is ever blocked (20+ seen).

### A finding that fell out of repairing that test

Moves past `board_size²` cost a full MCTS search and produce **zero** training
data. Measured across the stored self-play games:

| model | games | avg moves | >81 moves | search spent on moves that teach nothing |
|---|---|---|---|---|
| hero-of-time | 1033 | 113 | 92% | **29%** |
| hero-of-time-result-after-night | 380 | 117 | 95% | **32%** |
| night-model | 250 | 114 | 76% | **33%** |
| hot-boi-at-710 | 390 | 73 | 26% | 10% |
| hot-boi | 690 | 64 | 14% | 6% |

Roughly a third of self-play compute on the `hero-of-time` line is buying
nothing. This is exactly what the mercy rule was built for, and it is off by
default on every one of these models. Worth turning on (`resign_enabled`) and
watching `false_resign_rate` — independent of Phase 3.
