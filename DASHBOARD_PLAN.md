# Dashboard — audit and redesign plan

Written after reading `web/templates/index.html`, `web/static/js/dashboard.js`,
`web/routes/model_routes.py`, `model_manager.py`, `ai/match.py`, `ai/players.py`
and every `models/*/config.json` + `models/*/logs/training_log.jsonl` on this
machine.

Every number below is **measured from the current `models/` directory**, not
estimated. The scaffold (§0) is already in place: `/dashboard_new` serves a copy
of the live dashboard so the redesign can land without touching `/`.

---

## Summary

| # | Finding | Kind | Severity |
|---|---|---|---|
| D1 | Page is a render-time snapshot; every action is `window.location.reload()` | staleness | **High** |
| D2 | Elo is the headline number and is not comparable between models | misleading | **High** |
| D3 | Model list can't distinguish forks — the actual daily problem | usability | **High** |
| D4 | Config panel shows 8 of 25 training params, omitting everything added since it was written | stale surface | **High** |
| D5 | No model history at all: 21.2h of training, 25/42 promotions, 1725 games, all invisible | missing | **High** |
| D6 | Health warnings already recorded in the logs are never surfaced | missing | **High** |
| D7 | 9 head-to-head match games sit on disk, fully invisible | missing | Medium |
| D8 | Dead surface: archived-training card, `Legacy Default (9x9)`, nav-duplicating action cards | legacy | Medium |
| D9 | Copy/delete are `prompt()`/`confirm()` with no idea what's at stake | usability | Medium |

The through-line: the dashboard was built to answer *"what is my model?"* when
there was one model. There are now **8**, they are **forks of each other**, and
the questions are *"which one is actually best"*, *"what did this fork change"*
and *"is that one still healthy"*. None of those can be asked of the current
page.

---

## 0. Scaffold (done)

* `web/templates/dashboard_new.html` — copy of `index.html`, tagged `new` in the
  page title so the two are never confused.
* `web/static/js/dashboard_new.js` — copy of `dashboard.js`.
* `web/app.py` — routes `/dashboard_new` and `/dashboard_new/`.

Both routes render the same 8 models and behave identically today. The redesign
lands in these two files; `/` keeps working the whole time.

---

## 1. What the current page actually does

```
/  →  index.html (Jinja, server-rendered from model_manager.list_models())
      ├── model list          name · board · ruleset · kyu · elo
      ├── active model card   4 stat tiles + 8 config rows
      ├── 3 action cards      links to /play, /training/, /training/review
      ├── archive card        link to /training_old
      └── create/edit modal   full PARAM_BOUNDS sliders  ← the only current part
```

`dashboard.js` is 324 lines and does exactly four things: select a model,
open the create/edit modal, copy, delete. Three of those four end in
`window.location.reload()`.

---

## 2. Findings

### D1 — The page is a snapshot, and never updates

Every value is stamped into HTML at render time. There is no polling and no
socket subscription, although `base.html` already opens a global SocketIO
connection and the trainer broadcasts `training_update` continuously.

Consequences today:

* Training can be running and the dashboard shows no sign of it.
* `iteration`, `elo`, `total_games` go stale the moment training advances.
* Clicking a different model while training runs produces a raw
  `alert('Stop training before switching models')` from a 400
  ([model_routes.py:205](web/routes/model_routes.py:205)) — a rule the page
  never told you about.

### D2 — Elo is the headline, and it is not comparable across models

Measured, in the order the dashboard sorts them:

| Model | Elo | Iterations | Gate matches | Promotions | Evidence behind the number |
|---|---|---|---|---|---|
| hot boi | **778** | 48 | 0 | 0 | random-bot eval only |
| hot boi at 710 | **725** | 29 | 2 | 0 | random-bot eval, 2 gates |
| hero of time | **591** | 42 | 42 | 25 | gated every iteration |
| hero of time result after night | 591 | 19 | 19 | 14 | gated every iteration |
| hero of time for danya | 540 | 7 | 7 | 5 | gated every iteration |
| Night Model | 370 | 10 | 0 | 0 | random-bot eval only |

The two highest-rated models are the two with **no promotion-gate evidence at
all**. Their Elo comes from the older random-bot path, which saturates — the
exact failure `IMPROVEMENT_PLAN.md` B5 describes. Meanwhile the head-to-head
games on disk say `hero of time` beat `hero of time for danya` **8–0**, and the
dashboard ranks them 591 vs 540 as if that were the same kind of fact.

A single number with no provenance is worse than no number. Fix: show Elo *with
what earned it*, and make real head-to-head evidence reachable (D7).

### D3 — The model list can't tell forks apart

The list shows `name · board · ruleset · kyu · elo`. Current names:

```
hero of time · hero of time for danya · hero of time result after night
hot boi      · hot boi at 710
```

All 9×9, all chinese, all "small" network. The name is the only distinguishing
feature and it is doing all the work.

**Lineage is fully reconstructible and nobody is using it.** `copy_model()`
deep-copies the directory, so a fork inherits the parent's
`logs/training_log.jsonl` prefix verbatim. Comparing `(iteration, timestamp)`
prefixes across models gives exact fork points — measured:

```
hero-of-time-for-danya        ⟷ hero-of-time-result-after-night : forked at iteration 7
hero-of-time-result-after-night ⟷ hero-of-time                  : forked at iteration 19
hot-boi-at-710                ⟷ hot-boi                         : forked at iteration 27
```

i.e. two families:

```
hero of time  ─┬─ @7  → for danya          (stopped, 0 further iterations)
               └─ @19 → result after night (stopped, 0 further)
               └─ continued to 42

hot boi       ─── @27 → hot boi at 710 (+2 iterations, then stopped)
               └─ continued to 48
```

### D4 — The config panel is the stale half of a solved problem

`index.html` hard-codes 8 config rows. `TrainingParams` has **25 fields** and
`PARAM_BOUNDS` describes **21** of them across 8 categories. Not shown:

```
gate_enabled, gate_games, gate_threshold, gate_simulations, gate_stall_warning,
resign_enabled, resign_threshold, resign_consecutive, resign_min_move_factor,
resign_both_sides, resign_playout_fraction, restrict_eye_fill,
num_parallel_workers, batch_size, num_epochs_per_iteration,
replay_buffer_size, reflection_interval_games
```

The promotion gate and the mercy rule — the two subsystems added most recently,
and the two with the biggest effect on what a model becomes — are invisible on
the page that claims to show "Training Configuration". The *edit modal* renders
all of them from `PARAM_BOUNDS` already; only the read-only display was left
behind. It should be generated from the same source.

### D5 — A model's history is entirely absent

Everything below is already on disk, and none of it is on the dashboard:

| Model | Iterations | Wall-clock | Promotions | Games on disk | Size | Last trained |
|---|---|---|---|---|---|---|
| hero of time | 42 | 21.2 h | 25/42 | 1725 | 29 MB | 2026-08-15 20:58 |
| hot boi | 48 | 9.6 h | 0/0 | 882 | 12 MB | 2026-08-14 06:52 |
| hero of time result after night | 19 | 13.1 h | 14/19 | 595 | 13 MB | 2026-08-15 09:58 |
| hot boi at 710 | 29 | 3.2 h | 0/2 | 506 | 9.7 MB | 2026-08-14 12:07 |
| night model | 10 | 9.3 h | 0/0 | 291 | 7.5 MB | 2026-08-13 10:45 |
| hero of time for danya | 7 | 2.5 h | 5/7 | 245 | 8.2 MB | 2026-08-14 23:18 |
| lucky thirteen | 0 | — | — | 0 | 4 KB | never |
| Legacy Default (9x9) | 0 | — | — | 0 | 4 KB | never |

`training_log.jsonl` carries ~25 fields per iteration (elo, losses, gate win
rate, timings, pass rates, value spread, resign stats). The training page charts
them for the *active* model only, and reaching them costs a model switch — which
rebuilds the whole `Trainer`.

### D6 — Recorded health warnings are never surfaced

`hot boi at 710`'s last logged iteration contains:

```
"collapse_warning": "White is passing 25% of its moves (> 15%) — under area
                     scoring that donates points every move"
```

The dashboard lists it as the **second-strongest model** with no hint that its
final iteration recorded a collapse. The trainer already computes
`collapse_warning`, `gate_rejections`, `false_resign_rate`, `pass_rate_*`,
`value_std_*` — a health verdict per model is a read of data that exists.

### D7 — Head-to-head results exist on disk and are invisible

**9 unique match games** are stored (18 files — each game is written into both
participants' `games/match/`, so any reader must dedupe on
`(match_id, game_index)`). Each record carries `black_player.rating_key` /
`white_player.rating_key` (`model:<id>`) and the Elo delta it caused:

```
model:hero-of-time            beat model:hero-of-time-for-danya : 8
model:hero-of-time-for-danya  beat model:night-model            : 1
```

The whole comparison machinery is built — `MatchRunner`, pairwise Elo, ratings
written back into `config.json` through `rating_sink`, a random-bot anchor fixed
at 500 — and the dashboard, the one page whose job is comparing models, neither
shows the results nor offers to start a match.

### D8 — Dead and duplicated surface

* **Archived Training Interface** card → `/training_old`. A footer card
  promoting a legacy page.
* **`Legacy Default (9x9)`** — a model with no `games/`, no log, 0 iterations,
  no `network` key. It occupies a list row identical in weight to a 42-iteration
  model.
* **Three action cards** linking to `/play`, `/training/`, `/training/review` —
  the same three links as the nav bar directly above them.
* Empty models (`lucky thirteen`, 0 iterations) are visually indistinguishable
  from trained ones.

### D9 — Model actions don't match how models are used

* **Copy** is `prompt('Enter name for the copied model:')`. What the user
  actually does is *fork a run at its current iteration* — the copy should say
  what it is forking from and at which iteration.
* **Delete** is a `confirm()` that never mentions it is about to remove 29 MB,
  1725 games and 42 iterations of history.
* No archive/retire — the only way to reduce list noise is deletion.
* No place to record *why* a fork exists. `ModelInfo` has no notes field.

---

## 3. What the new dashboard should be

**From "model detail page" to "model fleet console."** Three jobs, in priority
order:

1. **Compare** — which of my models is actually best, and on what evidence.
2. **Understand a lineage** — where did this fork split, what changed, did it help.
3. **Act** — resume training, run a head-to-head, fork, retire.

```
┌────────────────────────────────────────────────────────────────────┐
│ Fleet header    8 models · 2 families · 79 MB · ⚡ training: hero  │
├───────────────────────────────┬────────────────────────────────────┤
│ FLEET TABLE (sortable)        │ DETAIL — hero of time              │
│ ▸ hero of time   42  ▁▃▅▆ 591 │ identity · lineage · vitals        │
│   ├ for danya     7  ▁▃   540 │ ┌ health ──────────────────────┐   │
│   └ after night  19  ▁▃▅  591 │ │ ✓ gate promoting 25/42       │   │
│ ▸ hot boi        48  ▁▂▃  778 │ │ ⚠ 3 rejections in a row      │   │
│   └ at 710       29  ▁▂ ⚠ 725 │ └──────────────────────────────┘   │
│ ▸ night model    10  ▁    370 │ elo / loss / gate charts           │
│   lucky thirteen  0   —   —   │ full config by category            │
│                               │ head-to-head record + [Match ▸]    │
└───────────────────────────────┴────────────────────────────────────┘
```

Rules the layout follows:

* **Fork children are nested under their parent**, with the fork iteration shown
  — the family tree is the list.
* **Every model row carries an Elo sparkline**, not just the final number: a
  trajectory says whether the run is still learning.
* **Health is a first-class column**, not something you find by opening a chart.
* **Untrained models are visibly untrained** (dimmed, "never trained"), not a
  row that looks like a result.
* **The active model is a state, not the layout** — the detail column follows
  selection, and selecting is not the same as *activating* for training.

---

## 4. Implementation plan

### Phase 1 — data layer (no UI)

Everything the redesign needs must be readable **without switching the active
model** (a switch rebuilds `Trainer`, reloads weights and is not free).

New module `model_stats.py` at the repo root, next to `model_manager.py`:

| Function | Returns |
|---|---|
| `read_log(model_id)` | parsed `training_log.jsonl` rows, cached by mtime |
| `summarize(model_id)` | iterations, elo, elo_delta_10, last_trained, total_seconds, promotions/gates, last losses, health flags, games_on_disk, bytes_on_disk |
| `lineage()` | fork graph across all models, from shared `(iteration, timestamp)` log prefixes |
| `head_to_head()` | win matrix from `games/match/*.json`, deduped on `(match_id, game_index)` |
| `health(model_id)` | `{level, headline, reasons[]}` from collapse_warning / gate_rejections / false_resign_rate / pass_rate |

New endpoints in `web/routes/model_routes.py`:

```
GET /models/api/summary                    → one row per model (the fleet table)
GET /models/api/<id>/history?fields=…      → downsampled series for sparklines/charts
GET /models/api/lineage                    → fork graph
GET /models/api/head_to_head               → win matrix + per-pair game ids
```

Tests: `tests/test_model_stats.py` — fork detection against a synthetic
`models/` tree (shared prefix, divergent prefix, no log, corrupt line);
head-to-head dedupe; health thresholds; `summarize()` on a model with no
`games/` directory (the `Legacy Default` case).

**Deliverable:** the numbers in §2 of this document, reproducible from an API.

### Phase 2 — fleet table

Replace the model list in `dashboard_new.html` with a table rendered by
`dashboard_new.js` from `/models/api/summary`:

* columns: name (nested by lineage) · iterations · Elo + sparkline · health · last trained · size
* sortable; sort preference persisted in `localStorage`
* selecting a row updates the detail column **without a page reload**
* "Activate for training" is a separate explicit action, disabled with a reason
  when training is running

Removes: the archive card, the three nav-duplicating action cards.

### Phase 3 — live state

Subscribe `dashboard_new.js` to the existing `training_update` socket event:

* a fleet-header banner naming the model currently training and its stage
* live `iteration` / `elo` / `games` on that model's row
* the activate/delete/edit controls disable themselves with an explanatory
  tooltip instead of failing with an `alert` after the click

### Phase 4 — detail column

* **Identity**: name, board, komi, ruleset, network preset + parameter count,
  created, forked-from `<parent> @ iteration N`.
* **Health strip**: verdict + reasons, styled like the existing
  `.resign-verdict` banner on the training page.
* **Charts**: Elo, total loss, gate win rate over iterations, from
  `/models/api/<id>/history` (Chart.js is already loaded in `base.html`).
* **Full config**, generated from `PARAM_BOUNDS` categories — all 21 sliders'
  values plus the 4 extra keys, with non-default values highlighted. Deletes the
  hard-coded 8-row grid and the D4 problem permanently.

### Phase 5 — comparison

* **Head-to-head panel** in the detail column: record vs every other model, from
  `/models/api/head_to_head`, each result linking to the game in the review page.
* **Compare mode**: select two models → config diff (what the fork changed) +
  overlaid Elo/loss curves + head-to-head record + **[Run match]**, which POSTs
  to the existing `/api/match/new` and links to the live match view.

This is the feature the current dashboard most conspicuously lacks, and it is
almost entirely wiring to code that already exists.

### Phase 6 — lifecycle

* **Fork** replaces Copy: dialog states parent and iteration, offers to change
  hyperparameters at fork time (the reason forks exist), defaults the name to
  `<parent> @<iteration>`.
* **Delete** states the impact measured from disk: "removes 42 iterations,
  1725 games, 29 MB" and requires the model name typed to confirm.
* **Archive**: a `models/<id>/.archived` marker (or an `archived` flag in
  `config.json`) that collapses a model into a separate section instead of
  deleting it. Retires `Legacy Default (9x9)` without data loss.
* **Notes**: a `notes` field on `ModelInfo`, editable inline — why this fork
  exists, in the user's own words.

### Phase 7 — cutover

1. `/` renders `dashboard_new.html`; old template moves to `/dashboard_old`.
2. One week of both, then delete `index.html` + `dashboard.js` and rename
   `dashboard_new.*` → `dashboard.*` (keeping `/dashboard_new` as an alias so
   bookmarks survive).
3. Remove the `/training_old` archive card and, if it is equally unused, the
   route itself.

---

## 5. Feature ideas ranked by (value ÷ effort)

| Feature | Value | Effort | Note |
|---|---|---|---|
| Fork tree with divergence points | **High** | Low | Data already on disk; §2 D3 proves it |
| Health flags per model | **High** | Low | Trainer already computes all inputs |
| Full config from `PARAM_BOUNDS` | **High** | Low | Modal already does exactly this |
| Live training banner | **High** | Low | Socket already broadcasting |
| Head-to-head record | **High** | Medium | 9 games already stored |
| Elo sparkline per row | Medium | Low | One downsampled series per model |
| Config diff between forks | **High** | Medium | Answers "what did I change" |
| Run match from dashboard | Medium | Medium | `/api/match/new` exists |
| Model notes | Medium | Low | One field on `ModelInfo` |
| Archive instead of delete | Medium | Low | Fixes list noise without data loss |
| Disk usage + game pruning | Medium | Medium | 79 MB across 8 models today |
| Start/stop training from dashboard | Medium | Low | Socket events already exist |
| Export/import a model as zip | Low | Medium | Sharing a run off this machine |
| Provenance on the Elo number | **High** | Low | D2 — say what earned the rating |

---

## 6. What this plan deliberately does not do

* **No change to `ModelManager`'s on-disk format** beyond two optional fields
  (`notes`, `archived`), both defaulted so every existing `config.json` keeps
  loading — the same compatibility rule `NetworkParams` and `TrainingParams`
  already follow.
* **No new training behaviour.** Every number the dashboard grows is a *read* of
  data the pipeline already writes. The one exception is Phase 5's "Run match",
  which calls an endpoint that already exists.
* **No cross-model Elo rescaling.** D2 is fixed by showing provenance and real
  head-to-head results, not by inventing a new rating scheme. Recalibrating the
  Elo scale is a training-pipeline question, not a dashboard one.
