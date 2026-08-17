"""
evaluator.py — Head-to-head measurement inside the training loop.

What lives here is the PROMOTION GATE: candidate vs reigning champion, played
headless and in parallel, whose verdict decides which network generates the
next batch of self-play data.

The random-bot evaluation that used to share this file is gone. It rated a
model against a fixed anchor, which measures nothing once the model wins every
game — the score saturates at 1.0 and the rating then climbs with the iteration
count rather than with strength. Competitive Elo is earned in played matches
(`ai/match.py`, `compute_pairwise_elo`) and recorded in the model's
`elo_history.jsonl` ledger.

`performance_elo_gap` stays because the gate needs it: it turns a gate score
into the Elo gap that score implies, which is what moves `Trainer.gate_elo`.
That ladder is self-referential — it only advances when a candidate actually
beats a champion — so it does not have the anchor's saturation problem.
"""

import math
from typing import Callable, Optional
from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE
from game.scoring.base import get_scorer
from ai.mcts import MCTS
from ai.network import GoNetwork
from ai.game_store import PHASE_PROMOTION, save_game


def _play_gate_game(
    current_network: GoNetwork,
    opponent_network: GoNetwork,
    board_size: int,
    komi: float,
    num_simulations: int,
    device: str,
    game_idx: int,
    iteration: int,
    games_dir: Optional[str],
    record_games: bool = True,
    restrict_eye_fill: bool = False,
    c_puct: float = 1.5,
    fpu_reduction: float = 0.35,
    restrict_self_atari: bool = False,
    self_atari_max_stones: int = 1,
) -> int:
    """
    Play one candidate-vs-champion game and record it. Returns 1 if the
    candidate won, else 0.

    Colours alternate with `game_idx` so a network that only plays one side
    well cannot pass the gate.
    """
    import time
    from datetime import datetime

    game_start = time.time()
    # Both sides restricted: the gate compares the candidate with the champion
    # under the conditions they will actually play under, so an asymmetric
    # filter here would measure the filter instead of the networks.
    current_mcts = MCTS(network=current_network, num_simulations=num_simulations,
                        device=device, c_puct=c_puct, fpu_reduction=fpu_reduction,
                        restrict_eye_fill=restrict_eye_fill,
                        restrict_self_atari=restrict_self_atari,
                        self_atari_max_stones=self_atari_max_stones)
    opponent_mcts = MCTS(network=opponent_network, num_simulations=num_simulations,
                         device=device, c_puct=c_puct, fpu_reduction=fpu_reduction,
                         restrict_eye_fill=restrict_eye_fill,
                         restrict_self_atari=restrict_self_atari,
                         self_atari_max_stones=self_atari_max_stones)
    scorer = get_scorer("chinese")

    state = GameState(board_size=board_size, komi=komi)
    current_color = BLACK if game_idx % 2 == 0 else WHITE
    move_list = []

    # Opening randomisation, then deterministic play.
    #
    # Both sides used to run at temperature=0.1 for the WHOLE game, which was
    # the gate's only source of game-to-game variety (add_noise is off, so two
    # deterministic networks would otherwise replay one identical game per
    # colour assignment). Paying for that variety with a weighted coin flip on
    # every move — including the endgame — put most of the gate's variance in
    # random blunders rather than in the difference between the networks, which
    # is the thing being measured.
    #
    # Worse, the supply of variety shrinks exactly when it is most needed: a
    # network whose policy has collapsed produces peaked visit counts, so
    # tau=0.1 sampling stops diverging at all and 40 gate games converge on a
    # handful of distinct games. The gate then reads ~50% because it is
    # replaying near-duplicates, not because the networks are matched.
    #
    # Confining the randomness to the opening buys diversity where it is cheap
    # (many near-equal opening moves) and plays the rest of the game at full
    # strength, so the result reflects strength instead of luck.
    opening_moves = board_size
    for move_num in range(board_size * board_size * 3):
        if state.is_over:
            break
        temp = 0.6 if move_num < opening_moves else 0.0
        if state.current_player == current_color:
            action, _ = current_mcts.search(state, temperature=temp, add_noise=False)
        else:
            action, _ = opponent_mcts.search(state, temperature=temp, add_noise=False)

        move_list.append({
            'color': int(state.current_player),
            'move': list(action) if action != MOVE_PASS else [-1, -1],
            'move_num': move_num,
        })

        if action == MOVE_PASS:
            state.play_pass()
        else:
            if not state.play_move(action[0], action[1]):
                state.play_pass()

    if not state.is_over:
        state.play_pass()
        if not state.is_over:
            state.play_pass()

    winner, margin = scorer.determine_winner(state)
    candidate_won = (winner == current_color)

    if games_dir:
        elapsed = round(time.time() - game_start, 2)
        black_score, white_score = scorer.score(state)
        champion_color = WHITE if current_color == BLACK else BLACK
        save_game(games_dir, iteration, PHASE_PROMOTION, game_idx, {
            'board_size': board_size,
            'komi': komi,
            'moves': move_list,
            'num_moves': len(move_list),
            'winner': int(winner) if winner else 0,
            'black_score': black_score,
            'white_score': white_score,
            'margin': margin,
            'timestamp': datetime.now().isoformat(),
            'is_promotion': True,
            'candidate_color': int(current_color),
            'champion_color': int(champion_color),
            'candidate_won': candidate_won,
            'elapsed_seconds': elapsed,
        }, store_full=record_games)

    return 1 if candidate_won else 0


def _gate_worker(kwargs: dict) -> int:
    """Worker for parallel promotion-gate games. Returns 1 if candidate won."""
    import torch
    torch.set_num_threads(1)
    return _play_gate_game(**kwargs)


def evaluate_against_checkpoint(
    current_network: GoNetwork,
    opponent_network: GoNetwork,
    board_size: int = 9,
    komi: float = 6.5,
    num_simulations: int = 50,
    num_games: int = 10,
    device: str = "cpu",
    iteration: int = 0,
    games_dir: Optional[str] = None,
    record_games: bool = True,
    num_workers: int = 4,
    stop_checker: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable] = None,
    restrict_eye_fill: bool = False,
    c_puct: float = 1.5,
    fpu_reduction: float = 0.35,
    restrict_self_atari: bool = False,
    self_atari_max_stones: int = 1,
) -> float:
    """
    Play the promotion-gate match between the candidate (`current_network`) and
    the reigning champion (`opponent_network`). Returns the candidate's win rate.

    Games are independent, so they are played across a process pool the same
    way self-play is — the gate is the most expensive phase of an iteration
    (two MCTS searches per move instead of one), so running it sequentially was
    leaving most of the machine idle. `num_workers=1` forces the old in-process
    path, which is also the automatic fallback if the pool cannot be created.

    Every game is recorded to `games_dir` (when given) under the iteration's
    `promotion/` directory, tagged with which side each network played, so the
    match that decided a promotion can be replayed move by move afterwards.
    `record_games=False` keeps the games in the index — and therefore in every
    statistic — but skips those records, which is the bulk of what an iteration
    writes to disk.
    """
    if num_games <= 0:
        return 0.0

    import concurrent.futures
    import os

    def _task(game_idx: int) -> dict:
        return {
            'current_network': current_network,
            'opponent_network': opponent_network,
            'board_size': board_size,
            'komi': komi,
            'num_simulations': num_simulations,
            'device': device,
            'game_idx': game_idx,
            'iteration': iteration,
            'games_dir': games_dir,
            'record_games': record_games,
            'restrict_eye_fill': restrict_eye_fill,
            'c_puct': c_puct,
            'fpu_reduction': fpu_reduction,
            'restrict_self_atari': restrict_self_atari,
            'self_atari_max_stones': self_atari_max_stones,
        }

    workers = max(1, min(num_games, num_workers, os.cpu_count() or 4))

    if workers > 1:
        try:
            wins = 0
            completed = 0
            active_futures = {}  # future -> game_idx
            next_task_idx = 0

            def _notify_progress():
                if not progress_callback:
                    return
                active_nums = sorted([idx + 1 for idx in active_futures.values()])
                try:
                    progress_callback(completed, num_games, active_games=active_nums, num_workers=workers, candidate_wins=wins)
                except TypeError:
                    try:
                        progress_callback(completed, num_games, active_nums, workers)
                    except TypeError:
                        progress_callback(completed, num_games)

            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                while next_task_idx < min(workers, num_games):
                    f = executor.submit(_gate_worker, _task(next_task_idx))
                    active_futures[f] = next_task_idx
                    next_task_idx += 1

                _notify_progress()

                while active_futures:
                    if stop_checker and stop_checker():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    done, _ = concurrent.futures.wait(
                        set(active_futures.keys()), timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        game_idx = active_futures.pop(future)
                        try:
                            wins += future.result()
                        except Exception as e:
                            print(f"Gate worker failed: {e}")

                        completed += 1

                        if next_task_idx < num_games and not (stop_checker and stop_checker()):
                            new_f = executor.submit(_gate_worker, _task(next_task_idx))
                            active_futures[new_f] = next_task_idx
                            next_task_idx += 1

                        _notify_progress()

            # Score against the games that actually finished, so an interrupted
            # or partially failed match is not read as a pile of losses.
            return wins / max(completed, 1)
        except Exception as e:
            print(f"Gate pool unavailable ({e}); falling back to sequential")

    wins = 0
    played = 0
    for game_idx in range(num_games):
        if stop_checker and stop_checker():
            break
        wins += _play_gate_game(**_task(game_idx))
        played += 1
        if progress_callback:
            try:
                progress_callback(played, num_games, active_games=[game_idx + 1], num_workers=1, candidate_wins=wins)
            except Exception:
                pass

    return wins / max(played, 1)



def compute_pairwise_elo(elo_a: float, elo_b: float, score_a: float,
                         k: float = 16.0) -> tuple:
    """
    Symmetric Elo update for one head-to-head result between two rated players.

    This is now the ONLY thing that moves a model's competitive rating. `score_a`
    is A's score in the game: 1.0 win, 0.5 draw, 0.0 loss. Both ratings move by
    the same amount in opposite directions — points won have to come from
    somewhere, which is exactly the property the old fixed-anchor update lacked
    and why that one could inflate forever.

    K is 16 rather than a developing player's 32: bot-vs-bot matches are played
    in batches, so a per-game K of 32 would swing ratings wildly over a 20-game
    series.

    Returns (new_elo_a, new_elo_b), both floored at 0.
    """
    expected_a = 1.0 / (1.0 + math.pow(10, (elo_b - elo_a) / 400))
    delta = k * (score_a - expected_a)
    return max(0.0, elo_a + delta), max(0.0, elo_b - delta)


def clamp_score(win_rate: float, num_games: int) -> float:
    """
    Pull a perfect (or perfectly bad) score in by half a game.

    A clean sweep of N games does not demonstrate infinite superiority; it
    demonstrates "at least as good as N-0.5 out of N". Without this correction
    any rating derived from a saturated score runs away, because
    `performance_elo_gap` sends a score of exactly 1.0 to +infinity.

    With N=4 the score is capped at 0.875, with N=20 at 0.975 — so gathering
    more evidence is what buys the right to claim a bigger gap, which is the
    behaviour you want.
    """
    if num_games <= 0:
        return win_rate
    margin = 1.0 / (2.0 * num_games)
    return min(max(win_rate, margin), 1.0 - margin)


def performance_elo_gap(win_rate: float, num_games: int) -> float:
    """
    Elo difference implied by a head-to-head score between equally rated players.

        gap = -400 * log10(1/score - 1)

    This is what the promotion gate measures: it plays `gate_games` rated games
    between candidate and champion every iteration, and reducing that to one
    promote/reject boolean throws the size of the win away. A 60% score over 20
    games is ~+70 Elo of evidence.

    This is the RAW gap implied by the score. It is not safe to add straight to
    a running ladder — see `promotion_elo_gain`.
    """
    score = clamp_score(win_rate, num_games)
    if score <= 0.0:
        return -4000.0
    if score >= 1.0:
        return 4000.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def promotion_elo_gain(win_rate: float, num_games: int, threshold: float) -> float:
    """
    The part of a gate result that is evidence of progress rather than of the
    selection rule that let it through.

    `performance_elo_gap` used to be added to `Trainer.gate_elo` directly, on the
    reasoning that a rejected candidate leaves the champion untouched, so the
    ladder "cannot drift upward on its own". That reasoning is wrong, and it is
    expensive.

    The ladder only ever sees results that CLEARED the threshold. Two networks of
    identical strength still score 60%+ some of the time — with 40 games at a 0.6
    bar, 13% of the time — and each of those coin flips was credited at face
    value, ~+110 Elo, while the 87% that fell short contributed nothing. That is
    a one-way ratchet driven by the winner's curse, not by strength. A real run
    shows it plainly: mean gate win rate 0.511 over the last 45 iterations
    (~1800 games, i.e. no measurable improvement at all) while gate_elo climbed
    from 1986 to 3998.

    The correction is to credit only the EXCESS over the bar the result had to
    clear. A candidate landing exactly on the threshold has demonstrated nothing
    beyond passing and earns 0; one winning 80% of 40 games still earns the
    difference between an 80% gap and a 60% gap. Noise near the bar now
    contributes ~0 instead of ~+110, while a genuine improvement is still paid.

    Returns 0.0 rather than a negative number for a below-threshold score: those
    are rejections and never reach the ladder anyway, so clamping just makes this
    safe to call unconditionally.
    """
    if win_rate < threshold:
        return 0.0
    earned = performance_elo_gap(win_rate, num_games)
    bar = performance_elo_gap(threshold, num_games)
    return max(0.0, earned - bar)
