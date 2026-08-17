"""
overfit_stats.py — measuring whether a model is learning Go or learning itself.

A root peer of `model_stats.py` / `elo_history.py`, and deliberately separate:
`model_stats` answers "how is this model doing", which every existing chart is
built on. This module answers the one question none of those charts can, because
every input they have comes from the model playing itself.

THE FAILURE THIS EXISTS TO SHOW
-------------------------------
An AlphaZero loop can degrade with every internal signal looking healthy. The
policy prior narrows onto a small set of moves; self-play is unaffected because
both sides narrow together; the promotion gate reads ~50% because a network is
never much worse than itself; the value head stays sharp; losses keep falling.
The only evidence is external — losing to opponents it used to beat.

hero-of-time did exactly this. Between iteration ~95 and ~185 its win rate
against OGS bots fell from 79% to 40% while it beat its own iteration-128
snapshot 8-2, its gate sat at 0.511 over ~1800 games, and its Elo ledger rose.

FOUR MEASUREMENTS, IN ORDER OF WHEN THEY MOVE
---------------------------------------------
1. `label_health`     — entropy of the policy TARGETS in the replay buffer. The
                        root cause moves first: a tau<1 target teaches argmax.
2. `network_health`   — entropy of what the policy head actually outputs, over
                        legal moves. Follows the labels down over ~tens of
                        iterations.
3. `training_series`  — the same two, per iteration, as logged by the trainer.
4. `generalization`   — internal vs external win rate. Moves last and is the
                        only one that is unambiguous.

Nothing here writes anything. It is all read-only over `models/`.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

MODELS_ROOT = "models"

# Opponent kinds that are NOT evidence of generalization. A model's own
# ancestors are the worst possible judges of whether it has overfit to its own
# lines: they share its entire training history and its blind spots.
INTERNAL_KINDS = {"model", "random"}


# ---------------------------------------------------------------------------
# Entropy probes — live measurements against the model's own files
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=axis)


def label_health(model_id: str, sample: int = 4096) -> dict:
    """
    Entropy of the policy targets sitting in the replay buffer right now.

    This is the root-cause probe. A healthy tau=1 target on 9x9 with this
    repo's fpu_reduction measures ~0.95 nats over a support of ~7 moves. The
    buffer that produced hero-of-time's collapse measured 0.19 nats, 62% of
    labels one-hot, support 3.

    `one_hot_frac` is the number to look at first: it is the share of training
    labels that name a single move, i.e. the share of the buffer that teaches
    the policy head nothing except its own argmax.
    """
    path = os.path.join(MODELS_ROOT, model_id, "replay_buffer.pt")
    out = {'available': False, 'reason': None, 'samples': 0,
           'entropy': None, 'one_hot_frac': None, 'support': None,
           'max_prob': None, 'entropy_deciles': []}
    if not os.path.isfile(path):
        out['reason'] = 'no replay_buffer.pt'
        return out
    try:
        import torch
        payload = torch.load(path, map_location='cpu', weights_only=False)
        pol = np.asarray(payload['policies'], dtype=np.float64)
    except Exception as exc:
        out['reason'] = f'unreadable: {exc}'
        return out
    if pol.size == 0:
        out['reason'] = 'buffer is empty'
        return out

    # The buffer is FIFO, so slicing it in order is slicing it by age. Sample
    # evenly across the whole thing rather than taking a contiguous block, or
    # the numbers describe one training session instead of the buffer.
    if len(pol) > sample:
        pol = pol[np.linspace(0, len(pol) - 1, sample).astype(int)]

    ent = _entropy(pol)
    out.update(
        available=True,
        samples=int(len(pol)),
        entropy=round(float(ent.mean()), 4),
        one_hot_frac=round(float((pol.max(axis=1) > 0.99).mean()), 4),
        support=round(float((pol > 1e-6).sum(axis=1).mean()), 2),
        max_prob=round(float(pol.max(axis=1).mean()), 4),
        # Age profile: the buffer holds ~30 iterations, so a rising trend here
        # is the fix taking effect and a flat line is the problem persisting.
        entropy_deciles=[round(float(chunk.mean()), 4)
                         for chunk in np.array_split(ent, 10)],
    )
    return out


def network_health(model_id: str, positions: int = 256,
                   from_iteration: Optional[int] = None) -> dict:
    """
    Entropy of the policy head's own output, over LEGAL moves only.

    Restricting to legal moves matters: entropy over all 82 outputs is dominated
    by the occupied points the network correctly gives ~0, so it would fall as a
    game fills up no matter how the prior behaves.

    Positions come from the model's own recorded self-play games — the
    distribution it will actually meet — replayed to sample across game phases.
    """
    out = {'available': False, 'reason': None, 'positions': 0,
           'entropy': None, 'top1_mass': None, 'top5_mass': None,
           'effective_moves': None}
    try:
        import torch
        from ai.model_loader import load_model_network
        from game.features import encode_for_network
        from game.game_state import GameState
    except Exception as exc:
        out['reason'] = f'imports unavailable: {exc}'
        return out

    states = _sample_self_play_positions(model_id, positions, from_iteration)
    if not states:
        out['reason'] = 'no recorded self-play games to sample'
        return out

    try:
        net = load_model_network(model_id)
        net = net[0] if isinstance(net, tuple) else net
        net.eval()
    except Exception as exc:
        out['reason'] = f'could not load weights: {exc}'
        return out

    ents, top1, top5, eff = [], [], [], []
    with torch.no_grad():
        for state in states:
            tensor = encode_for_network(state, net)
            probs, _ = net.predict(tensor, "cpu")
            probs = np.asarray(probs, dtype=np.float64)

            mask = np.zeros_like(probs, dtype=bool)
            for (r, c) in state.get_legal_moves():
                mask[r * state.board_size + c] = True
            mask[-1] = True  # pass is always an option
            masked = probs * mask
            total = masked.sum()
            if total <= 0:
                continue
            masked /= total

            ent = float(_entropy(masked))
            ents.append(ent)
            ordered = np.sort(masked)[::-1]
            top1.append(float(ordered[0]))
            top5.append(float(ordered[:5].sum()))
            # perplexity: "the prior is effectively spread over N moves", which
            # is easier to reason about than nats.
            eff.append(float(np.exp(ent)))

    if not ents:
        out['reason'] = 'no scorable positions'
        return out

    out.update(
        available=True,
        positions=len(ents),
        entropy=round(float(np.mean(ents)), 4),
        top1_mass=round(float(np.mean(top1)), 4),
        top5_mass=round(float(np.mean(top5)), 4),
        effective_moves=round(float(np.mean(eff)), 2),
    )
    return out


def _sample_self_play_positions(model_id: str, limit: int,
                                from_iteration: Optional[int] = None) -> list:
    """
    Replay recorded self-play games, keeping positions spread across phases.

    Sampling every 6th move rather than all of them keeps consecutive positions
    (which are nearly identical) from dominating the average.
    """
    from game.game_state import GameState

    games_root = os.path.join(MODELS_ROOT, model_id, "games")
    if not os.path.isdir(games_root):
        return []
    iters = sorted(d for d in os.listdir(games_root) if d.startswith("iter_"))
    if not iters:
        return []
    if from_iteration is not None:
        want = f"iter_{from_iteration:06d}"
        iters = [d for d in iters if d == want] or iters[-1:]
    else:
        iters = iters[-3:]  # the most recent few, i.e. the current behaviour

    komi, size = _board_params(model_id)
    states = []
    for name in reversed(iters):
        phase_dir = os.path.join(games_root, name, "self-play")
        if not os.path.isdir(phase_dir):
            continue
        for fname in sorted(os.listdir(phase_dir)):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(phase_dir, fname)) as fh:
                    game = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            state = GameState(board_size=game.get("board_size", size),
                              komi=game.get("komi", komi))
            for i, move in enumerate(game.get("moves", [])):
                if i % 6 == 0:
                    states.append(state.copy())
                    if len(states) >= limit:
                        return states
                mv = move.get("move") or [-1, -1]
                if mv[0] < 0:
                    state.play_pass()
                elif not state.play_move(mv[0], mv[1]):
                    state.play_pass()
    return states


def _board_params(model_id: str) -> tuple:
    try:
        with open(os.path.join(MODELS_ROOT, model_id, "config.json")) as fh:
            cfg = json.load(fh)
        return float(cfg.get("komi", 6.5)), int(cfg.get("board_size", 9))
    except (json.JSONDecodeError, OSError, ValueError):
        return 6.5, 9


# ---------------------------------------------------------------------------
# Logged series
# ---------------------------------------------------------------------------

TRACKED_FIELDS = (
    'policy_entropy', 'policy_top5_mass', 'target_entropy',
    'target_one_hot_frac', 'target_support', 'iter_target_entropy',
    'gate_win_rate', 'policy_loss', 'value_loss', 'learning_rate',
)


def training_series(model_id: str, limit: int = 400) -> dict:
    """
    The per-iteration series, straight from the trainer's own log.

    Iterations logged before a field existed contribute None, which every chart
    here renders as a gap rather than a zero — a model with 185 iterations of
    history and 3 of policy-entropy data should look like exactly that.
    """
    from model_stats import read_log

    rows = read_log(model_id)[-limit:]
    return {
        'iterations': [r.get('iteration') for r in rows],
        'series': {f: [r.get(f) for r in rows] for f in TRACKED_FIELDS},
        'has_entropy_data': any(r.get('policy_entropy') is not None for r in rows),
    }


# ---------------------------------------------------------------------------
# Generalization gap — the measurement that actually settles it
# ---------------------------------------------------------------------------

def generalization(model_id: str, bucket: int = 20) -> dict:
    """
    Win rate against INTERNAL opponents versus EXTERNAL ones, over time.

    Internal means this model's own ancestors, siblings and the random anchor.
    External means anything that did not come out of this training loop — OGS
    bots, GNU Go. Only the external number can distinguish "stronger at Go" from
    "better at beating things that share my blind spots", and a widening gap
    between the two IS overfitting, stated in the only units that matter.

    Games are bucketed by the model's iteration at the time they were played, so
    the trend survives the fact that matches are played in irregular bursts.
    """
    from model_stats import _match_games

    rating_key = f"model:{model_id}"
    rows = []
    for game in _match_games():
        for side, other in (('black_player', 'white_player'),
                            ('white_player', 'black_player')):
            me, opp = game.get(side) or {}, game.get(other) or {}
            if me.get('rating_key') != rating_key:
                continue
            winner = game.get('winner')
            won = (winner == 1 and side == 'black_player') or \
                  (winner == 2 and side == 'white_player')
            rows.append({
                'iteration': me.get('iteration'),
                'external': (opp.get('kind') not in INTERNAL_KINDS),
                'won': bool(won),
                'opponent': opp.get('name') or opp.get('kind') or '?',
                'opponent_kind': opp.get('kind'),
                'sims': me.get('num_simulations'),
                'timestamp': game.get('timestamp'),
                'margin': game.get('margin'),
            })

    rows = [r for r in rows if r['iteration'] is not None]
    rows.sort(key=lambda r: r['iteration'])

    buckets: Dict[int, dict] = {}
    for row in rows:
        key = (row['iteration'] // bucket) * bucket
        cell = buckets.setdefault(key, {'iteration': key,
                                        'internal_w': 0, 'internal_n': 0,
                                        'external_w': 0, 'external_n': 0})
        prefix = 'external' if row['external'] else 'internal'
        cell[f'{prefix}_n'] += 1
        cell[f'{prefix}_w'] += int(row['won'])

    series = []
    for key in sorted(buckets):
        cell = buckets[key]
        series.append({
            'iteration': key,
            'internal_games': cell['internal_n'],
            'external_games': cell['external_n'],
            'internal_rate': (round(cell['internal_w'] / cell['internal_n'], 4)
                              if cell['internal_n'] else None),
            'external_rate': (round(cell['external_w'] / cell['external_n'], 4)
                              if cell['external_n'] else None),
        })

    return {
        'bucket': bucket,
        'series': series,
        'by_opponent': _by_opponent(rows),
        'totals': {
            'internal_games': sum(1 for r in rows if not r['external']),
            'internal_rate': _rate([r['won'] for r in rows if not r['external']]),
            'external_games': sum(1 for r in rows if r['external']),
            'external_rate': _rate([r['won'] for r in rows if r['external']]),
        },
        'games': rows,
    }


def _rate(flags: List[bool]) -> Optional[float]:
    return round(sum(flags) / len(flags), 4) if flags else None


def _by_opponent(rows: List[dict]) -> List[dict]:
    """
    Per-opponent record, split early vs late by the model's own iteration.

    Aggregate win rate confounds strength with who was played when — a model
    that faced easy bots early and hard ones late looks like it declined. Split
    at the median iteration and the comparison is within-opponent, which is the
    only fair reading of "it used to beat this bot".
    """
    external = [r for r in rows if r['external']]
    if not external:
        return []
    iters = sorted(r['iteration'] for r in external)
    midpoint = iters[len(iters) // 2]

    table: Dict[str, dict] = {}
    for row in external:
        cell = table.setdefault(row['opponent'], {
            'opponent': row['opponent'], 'kind': row['opponent_kind'],
            'early_w': 0, 'early_n': 0, 'late_w': 0, 'late_n': 0,
            'first_iteration': row['iteration'], 'last_iteration': row['iteration'],
        })
        phase = 'early' if row['iteration'] < midpoint else 'late'
        cell[f'{phase}_n'] += 1
        cell[f'{phase}_w'] += int(row['won'])
        cell['first_iteration'] = min(cell['first_iteration'], row['iteration'])
        cell['last_iteration'] = max(cell['last_iteration'], row['iteration'])

    out = []
    for cell in table.values():
        cell['early_rate'] = (round(cell['early_w'] / cell['early_n'], 4)
                              if cell['early_n'] else None)
        cell['late_rate'] = (round(cell['late_w'] / cell['late_n'], 4)
                             if cell['late_n'] else None)
        cell['games'] = cell['early_n'] + cell['late_n']
        if cell['early_rate'] is not None and cell['late_rate'] is not None:
            cell['delta'] = round(cell['late_rate'] - cell['early_rate'], 4)
        else:
            cell['delta'] = None
        out.append(cell)
    out.sort(key=lambda c: -c['games'])
    return out


# ---------------------------------------------------------------------------
# Bundle + verdict
# ---------------------------------------------------------------------------

def report(model_id: str, probe: bool = True) -> dict:
    """
    Everything above, plus a plain-language verdict.

    `probe=False` skips the two live measurements (which load weights and replay
    games, ~seconds) and returns only what can be read from logs — used by the
    dashboard poll, where the probes come from the trainer's own per-iteration
    numbers instead.
    """
    out = {
        'model_id': model_id,
        'training': training_series(model_id),
        'generalization': generalization(model_id),
        'labels': None,
        'network': None,
    }
    if probe:
        out['labels'] = label_health(model_id)
        out['network'] = network_health(model_id)
    out['findings'] = _findings(out)
    return out


def _findings(data: dict) -> List[dict]:
    """
    Turn the numbers into ranked statements, with the number attached.

    Severity is 'critical' for a measured root cause, 'warning' for a trend that
    needs more evidence, 'ok' for a check that passed. Anything not measurable
    is omitted rather than reported as fine.
    """
    findings = []
    labels = data.get('labels') or {}
    network = data.get('network') or {}
    gen = data.get('generalization') or {}
    totals = gen.get('totals') or {}

    if labels.get('available'):
        one_hot = labels['one_hot_frac']
        if one_hot >= 0.25:
            findings.append({
                'severity': 'critical',
                'title': 'Policy targets are near-deterministic',
                'detail': (
                    f"{one_hot:.0%} of the {labels['samples']:,} sampled training "
                    f"labels name a single move, mean entropy {labels['entropy']:.2f} "
                    f"nats over a support of {labels['support']:.1f} moves. The policy "
                    f"head can only clone its own argmax from this. Check that "
                    f"policy_target_temperature is 1.0 — if it was just fixed, this "
                    f"clears as the buffer turns over (~30 iterations)."
                ),
            })
        else:
            findings.append({
                'severity': 'ok',
                'title': 'Policy targets carry search information',
                'detail': (f"mean entropy {labels['entropy']:.2f} nats, support "
                           f"{labels['support']:.1f} moves, {one_hot:.0%} one-hot."),
            })

    if network.get('available'):
        eff = network['effective_moves']
        if network['top5_mass'] >= 0.93:
            findings.append({
                'severity': 'critical' if eff < 4 else 'warning',
                'title': 'Policy prior has narrowed',
                'detail': (
                    f"{network['top5_mass']:.0%} of the prior sits on five moves "
                    f"(effectively {eff:.1f} moves considered, entropy "
                    f"{network['entropy']:.2f} nats over {network['positions']} "
                    f"positions). Moves outside that set get a prior near zero, so "
                    f"PUCT cannot recover them however long it searches — which is "
                    f"why this shows up only against opponents that play them."
                ),
            })
        else:
            findings.append({
                'severity': 'ok',
                'title': 'Policy prior is still spread',
                'detail': (f"entropy {network['entropy']:.2f} nats, effectively "
                           f"{eff:.1f} moves, {network['top5_mass']:.0%} on the top five."),
            })

    internal, external = totals.get('internal_rate'), totals.get('external_rate')
    if internal is not None and external is not None and totals['external_games'] >= 8:
        gap = internal - external
        if gap >= 0.25:
            findings.append({
                'severity': 'critical',
                'title': 'Wins against itself, loses to everyone else',
                'detail': (
                    f"{internal:.0%} against its own lineage over "
                    f"{totals['internal_games']} games, {external:.0%} against external "
                    f"opponents over {totals['external_games']}. A {gap:.0%} gap is the "
                    f"definition of overfitting to self-play: internal opponents share "
                    f"the blind spots, so they cannot see them."
                ),
            })
        else:
            findings.append({
                'severity': 'ok',
                'title': 'Internal and external results agree',
                'detail': (f"{internal:.0%} internal vs {external:.0%} external — "
                           f"self-play strength is transferring."),
            })

    declines = [c for c in (gen.get('by_opponent') or [])
                if c.get('delta') is not None and c['delta'] <= -0.25
                and c['early_n'] >= 3 and c['late_n'] >= 3]
    if declines:
        worst = ", ".join(
            f"{c['opponent']} {c['early_rate']:.0%}→{c['late_rate']:.0%}"
            for c in declines[:4])
        findings.append({
            'severity': 'critical',
            'title': 'Regressed against specific opponents',
            'detail': (f"Same opponents, later iterations, worse results: {worst}. "
                       f"This is within-opponent, so it is not explained by a harder "
                       f"schedule."),
        })

    series = (data.get('training') or {}).get('series') or {}
    gates = [g for g in (series.get('gate_win_rate') or []) if g is not None][-40:]
    if len(gates) >= 20:
        mean_gate = sum(gates) / len(gates)
        if 0.45 <= mean_gate <= 0.56:
            findings.append({
                'severity': 'warning',
                'title': 'Promotion gate has no signal left',
                'detail': (
                    f"mean gate win rate {mean_gate:.3f} over the last {len(gates)} "
                    f"iterations. Candidates are indistinguishable from the champion, "
                    f"so promotions are being decided by noise rather than by "
                    f"improvement — and the gate cannot tell you that on its own."
                ),
            })

    order = {'critical': 0, 'warning': 1, 'ok': 2}
    findings.sort(key=lambda f: order.get(f['severity'], 3))
    return findings
