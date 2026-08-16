# Optional Play Heuristics — status, verdicts, implementation plans

Heuristics are **optional move filters and game-termination rules** that are not
rules of Go. They exist to stop the network wasting search and training capacity
on moves that are never correct, or on game phases that teach nothing.

Every one of them is opt-in, defaults to off, and off must reproduce the
previous behaviour exactly.

---

## Status at a glance

| Heuristic | Verdict | Status |
|---|---|---|
| Own-two-eye fill (`restrict_eye_fill`) | Keep — provably safe, small win | **Implemented** |
| No pointless self-atari | Built — but **measured smaller than predicted**, see below | **Implemented** |
| Mercy rule / resignation in self-play | Keep — best throughput lever | **Implemented** |
| Never fill *any* true eye | Skip — unprovable, marginal | Not implemented |
| No moves in own Benson pass-alive region | Skip — safe but net-negative cost | Not implemented |
| Root symmetry deduplication | Skip — redundant with replay augmentation | Not implemented |
| First-line ban before move N | Skip — strategy bias, no safety basis | Not implemented |

### The two currencies

Rank proposals by which bottleneck they attack. They are not interchangeable.

- **Sample efficiency** — better learning *per game*, by removing garbage from
  the policy target or sharpening it.
- **Throughput** — more games *per hour*, by cutting moves that cost a search
  and teach nothing.

`ai/self_play.py` discards every training sample past move `board_size²`, so on
9×9 everything after move 81 costs a full MCTS search and produces **zero**
training data. Anything that shortens that tail is pure throughput.

### Why the promotion gate cannot validate a move filter

Candidate and champion play under the *same* filter, so the gate measures
relative strength under the handicap and is blind to the handicap's own cost. A
filter that removes a needed tesuji produces no rejection and no warning — the
gate-Elo ladder keeps climbing while the model is quietly worse.

`win_rate_vs_random` is not a substitute: only the network is filtered there, so
it does see the effect, but it saturates at 100% early and is documented as
useless past that point. Expect a **discontinuous step** in that series at the
iteration where a filter is switched on mid-run. The gate ladder does not step,
because it is self-referential.

**The only valid test is outside the loop** — see [Validation protocol](#validation-protocol).

---

## Implemented

### `restrict_eye_fill` — own-two-eye fill

See `game/eyes.py` for the full rationale and the proof. Summary:

- An **eye of chain C** is a liberty of C whose every on-board neighbour is a
  stone of C. Strict by design: false eyes are exactly the points whose
  surrounding stones are not one chain, so they are never counted and the
  connecting move at a false eye is never removed.
- Forbidden iff the point is an eye of a chain with **exactly two** eyes.
- Safety basis: **theorem**, not heuristic. A chain with two such eyes cannot be
  captured (playing either eye is suicide for the opponent), so filling one is
  the move that ends unconditional life, and it can never capture, connect or
  defend.

**Known coverage limits** (deliberate — coverage was traded for provability):

- It counts eye *points*, not eye *regions*. Multi-point eye spaces contribute
  zero, so a group with two two-space eyes is never protected, and can be
  self-killed one move at a time without the rule firing.
- It catches the single-move blunder from the canonical two-single-point-eye
  shape and nothing else. It is **not** "the bot can never kill its own group".
- Counting larger eye spaces would drag in nakade (is a bulky five one eye or
  two? depends whose move it is), which is where a correctness claim collapses.
  If coverage is ever wanted, add it as a **separate, explicitly heuristic
  tier** so the provable rule can still be run alone.

**Residual exception**: under positional superko, filling your own territory is
sometimes the only legal non-pass move that avoids a repetition. Pass is always
available so there is no deadlock, but pass is not equivalent when a ko is live.
Rare; applies to every filter in this document.

---

### Mercy rule (`resign_enabled`)

Implemented in `ai/self_play.py::play_self_play_game`; see the block comment
there. Six settings under the **Mercy Rule (Resignation)** category:
`resign_enabled`, `resign_threshold`, `resign_consecutive`,
`resign_min_move_factor`, `resign_both_sides`, `resign_playout_fraction`.

Notes that only became clear once it was built and measured:

- **It is self-gating.** An untrained value head produces root evaluations
  averaging `|v| ≈ 0.12` and essentially never reaches the 0.90 default, so the
  rule does nothing until the network is genuinely confident. There is no
  early-training speedup to look for, and that is the correct behaviour — a
  network that knows nothing should not be resigning.
- **Streak semantics**: confirming moves are counted from the start of the
  game, not from the earliest-resign boundary. A side hopeless for forty moves
  resigns the moment the boundary opens. The boundary exists to protect
  training data, not to restart the evidence.
- **Collapse interlock**: the trainer suppresses the rule while the collapse
  guard is tripped, because a flat value head would resign every game on sight
  and the resulting two-move games would silence the guard's own pass-rate
  tripwire.
- **The tuning signal** is `false_resign_rate` in `training_log.jsonl`, and it
  is reported after each self-play phase. `None` means the rule never fired on
  a playout game — no evidence either way, not a clean bill of health.
- **Where to read it**: the Mercy Rule card under *Training Metrics*, served by
  `/training/api/resign_stats`. Bars are the benefit (games ended early), the
  red line is the cost (cumulative wrong-resignation rate) with a Wilson
  confidence band, and a verdict banner states the answer outright. The band
  matters more than the line: with a handful of checks a 0% rate is consistent
  with a true rate above 20%, so the endpoint refuses to report "good" until
  the interval's UPPER bound clears the 5% danger line. The playout games do
  double duty — they measure both the wrong-resignation rate and the moves
  saved, since they record where the rule would have stopped and then played
  the tail anyway.

---

## Worth building

### 1. No pointless self-atari — **now implemented, and the prediction was wrong**

Built as specified (`restrict_self_atari`, `self_atari_max_stones`), see
`game/self_atari.py`. The reasoning below is kept because the *rule* is sound;
what did not survive contact with measurement is the claim that this is the
best sample-efficiency lever.

**Measured on the stored self-play games of every trained model** (60 games
each, checking each move against the filter at the moment it was played):

| model | moves | self-atari played | of those, inside the training window |
|---|---|---|---|
| night-model | 6,629 | 4.72% | 1.78% |
| hero-of-time | 6,208 | 4.03% | 1.80% |
| hot-boi | 3,934 | 2.47% | 2.24% |

Split by phase, the picture is decisive:

| phase | share of moves that are pointless self-atari |
|---|---|
| first half (where training samples come from) | **0.3 – 0.4%** |
| late game (past the sample cutoff) | **5.5 – 7.1%** |

**The moves this filter removes are overwhelmingly in the tail that produces no
training data at all.** In the opening and midgame — the only phase whose
policy targets are kept — these networks already almost never play a pointless
self-atari, so there is very little garbage left to remove from the target.

**The cost is paid everywhere.** `restrict_self_atari` adds ~15% to
`get_legal_moves`, which is ~80% of search time, so roughly 12% of self-play
throughput — on every move of every game, including the opening where it finds
almost nothing.

**Conclusion**: on current evidence this is a marginal trade for these models,
not the headline win. The late-game tail it targets is better addressed by the
**mercy rule**, which deletes that whole phase instead of filtering moves within
it, and costs nothing per move. Enable `resign_enabled` first; only reach for
`restrict_self_atari` if the A/B rig shows it earning its 12%.

The original reasoning, for the record:

**Why**: highest expected sample-efficiency gain, because unlike eye-filling it
is *frequent* in exactly the weak-network regime that matters. Removes a large
share of the bad moves from both the search budget and the policy target.
(Frequent — yes. In the phase that matters — no.)

**Safety basis**: a tuned assumption, not a proof. Legitimate self-atari
includes throw-ins, snapbacks, nakade placement, ko captures, eye-space
reduction and seki manipulation. The soft form below preserves most of them, but
the stone-count threshold has nothing behind it except judgement.

**Definition.** A move by `C` at `p` is a pointless self-atari iff, after
placing the stone and resolving captures:

1. the move captured **nothing**, and
2. the group containing `p` has exactly **1 liberty**, and
3. that group is **larger than `k` stones** (`k` default 1).

Condition 1 exempts every ko capture and most snapbacks. Condition 3 exempts
throw-ins, which are the main small-group tesuji.

**Implementation notes** (as built)

- `game/self_atari.py` holds the rule; the filter runs inside
  `rules.get_legal_moves` next to `restrict_eye_fill` and propagates through
  `GameState.copy()`, so it holds at every depth of the search.
- It never simulates anything itself. `rules.simulate_move` already walks the
  played point's neighbours for legality; `need_group_facts=True` makes it
  retain the merged group's size and liberty set from that same walk, reusing
  the flood fills the suicide check already performed. Two early-outs keep it
  off the hot path: a move that captures is exempt, and a move with two or more
  empty neighbours provably has two liberties, so neither needs any group work.
- **One deliberate departure from the spec below**: if the filter would remove
  *every* legal move, it hands them back rather than forcing a pass. The eye
  rule does not get this guard and should not — filling your own last two eyes
  is provably useless, so passing really is at least as good there. Self-atari
  has no such proof, and a position where every move trips it (capturing race,
  seki, filled endgame) is exactly where the assumption is least reliable.

**Implementation strategy** (original plan)

- New module `game/self_atari.py`, mirroring the shape of `game/eyes.py`
  (module docstring carrying the rationale, one scan function, one point
  function).
- **Do not simulate the move twice.** `rules.is_legal_move` already builds
  `test_board` and resolves captures. Refactor that into a helper that returns
  the facts both callers need — something like
  `_simulate(board, color, r, c) -> (test_board, captured_any, own_group)` —
  and have `is_legal_move` and the self-atari check share it. A naive
  implementation that re-simulates inside `get_legal_moves` doubles the cost of
  the hottest function in the engine.
- Filter in `rules.get_legal_moves` alongside `restrict_eye_fill`, using the
  same flag-on-`GameState` → propagated-through-`copy()` mechanism, so the
  restriction holds at every depth of the MCTS tree.
- Two settings: `restrict_self_atari` (bool) and `self_atari_max_stones`
  (int, 1–4, default 1) so `k` is tunable without a code change.

**Tests** (`tests/test_self_atari.py`) — the over-restriction cases matter most:

- throw-in into opponent eye space stays playable
- snapback setup stays playable
- ko capture stays playable (exempt via "captured something")
- large self-atari is blocked
- a move that captures is never blocked, even at 1 resulting liberty
- exact behaviour at the `k` boundary (`k`, `k+1` stones)
- the other colour's moves are unaffected
- off-by-default: filtered and unfiltered move lists are identical
- ladder/atari positions where self-atari is the only defence — confirm the
  filter is not removing a forced move

---

### 2. Mercy rule — **now implemented**

Built as designed. See the "Mercy rule" entry under *Implemented* above for the
settings, and for the three things that only became clear once it was measured:
self-gating behaviour, streak semantics, and the collapse interlock.

---

## Rejected — with reasons, so they are not re-litigated

### Never fill *any* true eye (generalise from exactly-two to one-or-more)

Safe in practice — with the chain-based eye definition the point touches no
opponent stone, so it cannot capture, connect or threaten — but "I could not
construct a counterexample" is a weaker claim than the two-eye rule's theorem,
and generalising trades the proof for very little.

The principled objection: **filter mistakes whose punishment is delayed and hard
to attribute, not mistakes that are punished immediately.** Filling your last
two eyes qualifies (the group dies twenty moves later). Filling the sole eye of
a one-eyed group is punished almost at once and is cheap for the network to
learn unaided — filtering it removes a cheap lesson without removing an
expensive one.

### No moves inside your own Benson pass-alive region

**The naive version is broken in this codebase.** `game/scoring/chinese.py` is
naive area scoring with no dead-stone-removal phase — stones on the board count,
period, and `get_empty_regions` marks any empty region touching both colours as
neutral dame. So a white stone left uncaptured inside Black's eye space scores
+1 for White *and* turns Black's surrounding territory into dame. On a 6-point
region that is a ~6-point swing with 6.5 komi. The configuration is fully
reachable: Benson regions are defined over empty-*or-opponent* points, and a
throw-in does not break vitality, so the region stays pass-alive and the capture
stays forbidden.

**Restricted to regions containing no opponent stones, it is sound**, subject to
three conditions: regions must be enclosed by chains in the *final* pass-alive
set (not merely by any friendly stone); the analysis must be recomputed every
move so a throw-in immediately lifts the restriction for that whole connected
component; and "empty" must mean entirely empty. Under those conditions the move
provably cannot capture, connect, defend, form a ko or touch a seki, and is
score-neutral under area scoring.

**It is still not worth building**, for cost rather than safety reasons. Benson
is an iterative fixpoint over the whole board; enforced the way `restrict_eye_fill`
is (flag propagates through `GameState.copy()`), it would run ~96 times per move
at the default sim count — plausibly comparable to the small-net forward pass —
against a saving of maybe 10–20% of moves at the end of a game. **The throughput
win could come out negative.** Root-only enforcement fixes the cost but then buys
only game-shortening, which the mercy rule does better and more thoroughly.

Two further notes if it is ever revisited: it is a strict **superset** of
`restrict_eye_fill` (a chain with two single-point eyes is Benson-alive), so the
eye rule would become redundant; and the real risk is not the theorem but the
implementation — a *permissive* Benson bug (botching "every empty point of the
region is a liberty of that chain") declares not-actually-alive groups alive and
starts forbidding moves that genuinely defend, silently, in exactly the way the
gate cannot detect.

### Root symmetry deduplication

Exact, but it only bites while the position is symmetric — on an empty 9×9 board
that means the first move (81 points collapse to 15 orbits) and essentially
nothing after. `ReplayBuffer.sample` already applies 8-fold dihedral
augmentation, so the network is already being taught the symmetry this would
enforce. Largely redundant.

### First-line ban before move N

No safety basis: unlike the others it biases *strategy* rather than removing
self-destruction. Its apparent appeal comes from the same property that makes it
dangerous — on 9×9 the perimeter is 32 of 81 points, so it deletes ~40% of the
board, on a board where the third line is already the middle. Defensible only as
a strictly time-limited opening prior, and only with an A/B behind it.

---

## Validation protocol

Because the gate is blind to a filter's own cost, any new filter must be tested
outside the training loop:

1. `ModelManager.copy_model` to make a twin of an existing model, so both start
   from identical weights and hyperparameters.
2. Train both for the same number of iterations, one with the filter on, one
   with it off.
3. Play them head-to-head **with the filter OFF for both sides**.

Step 3 is the whole point. A head-to-head with the filter on for both measures
nothing — it reproduces the gate's blind spot.

The mercy rule is the exception: its false-positive rate is measurable inside a
single run via the playout fraction, and needs no A/B rig.

## Plumbing recipe

`restrict_eye_fill` establishes the path a new setting takes. In order:

1. `param_bounds.py` — add the spec (and a category if needed). This alone gives
   you the control in Create Model, Edit Model and Live Tuning; the sliders and
   toggles are generated from it, so **no JS or template changes are needed**.
2. `model_manager.py::TrainingParams` — add the field with the
   backwards-compatible default. Unknown keys in old `config.json` files are
   filtered out on load, and missing keys fall back to the dataclass default.
3. `config.py::TrainingConfig` — add the field, and wire it in
   `Config.from_model` via `_model_default(...)` so models predating the setting
   keep the documented default rather than a second hand-written copy of it.
4. `ai/trainer.py` — pass it to whichever phases need it (self-play, gate,
   random eval). Config is re-read at the start of each phase, so live tuning
   takes effect on the next iteration with no restart.
5. The phase functions (`ai/self_play.py`, `ai/evaluator.py`) — thread it
   through to `MCTS` and into the worker task dicts, since workers are separate
   processes and only see what is in the dict.
6. `web/routes/training_routes.py::apply_params` — add the key to the live-tune
   list so it is applied to the running trainer and persisted to `config.json`.
7. `web/routes/game_routes.py` — if the bot should behave the same against a
   human, read it from the live trainer config.
8. `web/templates/info.html` — document it as a param card and add a TOC entry.

Decide deliberately whether the setting applies to **both sides** of a gate match
(yes, or you measure the filter instead of the networks) and to the **random bot**
in the Elo eval (no — it is the anchor, and strengthening it makes the Elo curve
incomparable across iterations).
