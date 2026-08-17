"""
trainer.py — Main training loop for the Go AI.

Implements the AlphaZero training pipeline:
1. Self-play: generate games with current network + MCTS
2. Train: update network on the collected data, then put the candidate through
   the promotion gate against the reigning champion
3. Save weights: persist model state
4. Repeat

Training does NOT rate the model. Competitive Elo is earned in played matches
(`ai/match.py`) and recorded in the model's `elo_history.jsonl`; the only
rating training maintains is `gate_elo`, its own self-referential measure of
whether candidates are still beating champions.

The training loop runs in a background thread when started from the web UI.
It emits progress events via a callback so the dashboard can update in real-time.

REPLAY BUFFER:
Training samples are stored in a fixed-size buffer. When it's full, old samples
are dropped. This ensures the network trains on recent, relevant data rather
than ancient games from when it was much weaker.
"""

import os
import time
import json
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Callable, Dict
from collections import deque
from datetime import datetime

from config import Config, elo_to_rank, simulations_for_board
from ai.network import GoNetwork
from ai.self_play import run_self_play_batch, TrainingSample
from ai.evaluator import (evaluate_against_checkpoint, performance_elo_gap,
                          promotion_elo_gain)
from elo_history import DEFAULT_ELO
from ai.checkpoint import save_weights, load_weights
from ai.game_store import migrate_legacy_layout
from ai.replay_store import buffer_path, load_buffer, save_buffer


class ReplayBuffer:
    """
    Fixed-size buffer of training samples.
    
    When full, oldest samples are dropped (FIFO).
    Samples are (state_tensor, mcts_policy, game_outcome).
    """
    
    def __init__(self, max_size: int = 50_000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, samples: List[TrainingSample]) -> None:
        """Add a batch of samples to the buffer."""
        self.buffer.extend(samples)
    
    def sample(self, batch_size: int, augment: bool = True) -> tuple:
        """
        Sample a random mini-batch for training with optional 8-fold symmetry augmentation.
        
        Returns:
            (states, policies, values) — batched tensors ready for training.
        """
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        
        states = []
        policies = []
        values = []
        
        for idx in indices:
            s, p, v = self.buffer[idx]
            
            if augment:
                # 8-fold dihedral symmetry (4 rotations x 2 flips)
                k = random.randint(0, 7)
                rot_k = k % 4
                flip = (k >= 4)
                
                # Augment state tensor (C, H, W)
                s_aug = s.clone()
                if rot_k > 0:
                    s_aug = torch.rot90(s_aug, k=rot_k, dims=(1, 2))
                if flip:
                    s_aug = torch.flip(s_aug, dims=(2,))
                
                # Augment policy array (H*W + 1)
                board_size = s.shape[1]
                p_tensor = torch.from_numpy(p)
                p_board = p_tensor[:-1].reshape(board_size, board_size)
                p_pass = p_tensor[-1:]
                
                if rot_k > 0:
                    p_board = torch.rot90(p_board, k=rot_k, dims=(0, 1))
                if flip:
                    p_board = torch.flip(p_board, dims=(1,))
                
                p_aug = torch.cat([p_board.reshape(-1), p_pass])
                
                states.append(s_aug)
                policies.append(p_aug)
            else:
                states.append(s)
                policies.append(torch.from_numpy(p))
            
            values.append(v)
        
        return (
            torch.stack(states),
            torch.stack(policies),
            torch.tensor(values, dtype=torch.float32),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class Trainer:
    """
    Main training orchestrator.
    
    Manages the self-play → train → evaluate cycle and reports progress.
    
    Args:
        config: Global configuration (built from model config).
        progress_callback: Called with progress dicts for the web dashboard.
            The dict contains keys like 'type', 'iteration', 'elo', 'loss', etc.
        model_id: The model ID this trainer is associated with (for state updates).
    """
    
    def __init__(self, config: Config, progress_callback: Optional[Callable] = None,
                 model_id: Optional[str] = None):
        self.config = config
        self.progress_callback = progress_callback
        self.model_id = model_id
        self.is_running = False
        self._stop_requested = False
        self._force_stop_requested = False
        
        # Initialize network
        self.network = GoNetwork(
            board_size=config.board.size,
            num_input_planes=config.network.num_input_planes,
            input_features=config.network.input_features,
            num_res_blocks=config.network.num_res_blocks,
            num_filters=config.network.num_filters,
            value_head_hidden=config.network.value_head_hidden,
        )
        
        # Dedicated eval network for fast CPU inference (self-play & human games)
        self.eval_network = GoNetwork(
            board_size=config.board.size,
            num_input_planes=config.network.num_input_planes,
            input_features=config.network.input_features,
            num_res_blocks=config.network.num_res_blocks,
            num_filters=config.network.num_filters,
            value_head_hidden=config.network.value_head_hidden,
        ).to("cpu")
        self.eval_network.eval()
        
        device = config.training.device
        # MPS doesn't support all operations; fallback gracefully
        try:
            self.network = self.network.to(device)
            test_input = torch.randn(1, config.network.num_input_planes,
                                      config.board.size, config.board.size).to(device)
            self.network(test_input)  # Test forward pass
            self.device = device
        except Exception:
            self.device = "cpu"
            self.network = self.network.to("cpu")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.lr_decay_steps,
            eta_min=1e-5,
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.training.replay_buffer_size)
        
        # Training state
        self.iteration = 0
        self.total_games = 0
        # Competitive rating, owned by the model's `elo_history.jsonl` ledger
        # and moved ONLY by played games (see ai/match.py). Training no longer
        # touches it: the anchor-based evaluation that used to nudge it every
        # iteration measured nothing once the random bot stopped winning.
        self.elo = DEFAULT_ELO
        self.best_elo = self.elo
        # Rating on the promotion-gate ladder — a separate, self-referential
        # scale that only moves when a candidate actually beats the champion.
        # This is training's own progress measure and is not the competitive
        # Elo above.
        self.gate_elo = DEFAULT_ELO
        
        # Metrics history (for charts). `_history_loaded` tracks whether the
        # on-disk log has been merged in yet — see get_metrics_history().
        self.metrics_history: List[dict] = []
        self._history_loaded = False
        self.recent_logs = deque(maxlen=100)

        # Promotion-gate bookkeeping
        self.gate_rejections = 0          # Consecutive candidates refused
        self.stall_breaks = 0             # Times the stall breaker has fired
        self.last_gate_win_rate = None    # Candidate's score in the last gate match
        # Per-iteration pass tallies, refreshed each self-play phase and read
        # by the collapse guard. {colour: [passes, total_moves]}
        self._pass_stats = {1: [0, 0], 2: [0, 0]}
        # Per-iteration policy-LABEL tallies, refreshed each self-play phase.
        # Fresh counterpart to the buffer-wide numbers in the collapse guard.
        self._label_stats = self._empty_label_stats()
        # Per-iteration mercy-rule tallies, refreshed each self-play phase.
        self._resign_stats = self._empty_resign_stats()
        # Set by the collapse guard each iteration. While it is True the mercy
        # rule is suppressed: a collapsed value head reports ±1 from the turn
        # colour alone, so its "I have lost" verdict carries no information and
        # would end every game on sight.
        self._collapse_active = False

        # Comprehensive live stage state for real-time tracking & refresh persistence
        self.current_stage = {
            'stage': 'idle',
            'stage_name': 'Idle',
            'stage_index': 0,
            'total_stages': 4,
            'iteration': self.iteration,
            'completed_items': 0,
            'total_items': 0,
            'percent': 0,
            'active_games': [],
            'num_workers': 0,
            'detail': 'Waiting to start training',
            'stages_overview': self._build_stages_overview('idle'),
        }
        
        # Fold any pre-existing flat game files into the per-iteration layout
        # so old and new games show up in the same history tree.
        try:
            moved = migrate_legacy_layout(config.paths.games_dir)
            if moved:
                self._emit('info', f'Reorganised {moved} stored games into per-iteration folders')
        except OSError as e:
            self._emit('warning', f'Game history migration skipped: {e}')

        # Load weights if available
        self._try_load_weights()
        # ...and the samples those weights were trained on. Without this every
        # restart starts the next iteration from a few hundred samples and the
        # candidate it produces reliably loses its gate match.
        self._try_load_replay_buffer()
        # A model with no weights.pt never reaches _try_load_weights' refresh,
        # but it still has a rating if it has ever played a match.
        self._refresh_elo()
    
    @staticmethod
    def _empty_resign_stats() -> dict:
        return {
            'games': 0,          # Self-play games completed this iteration
            'resigned': 0,       # Games actually stopped by the mercy rule
            'moves_total': 0,    # Moves played across all games
            'playout_games': 0,  # Games that ignored the rule to measure it
            'would_resign': 0,   # ...of those, ones where it would have fired
            'false_resigns': 0,  # ...of those, ones where it would have been WRONG
        }

    def _resign_metrics(self) -> dict:
        """
        Mercy-rule summary for the metrics log.

        `false_resign_rate` is the number that matters: it is measured on the
        games that overruled the rule and played on, so it is the only evidence
        that the threshold is not throwing away winnable games. None means the
        rule never fired on a playout game this iteration — no evidence either
        way, not a clean bill of health.
        """
        s = self._resign_stats
        if not self.config.training.resign_enabled:
            return {'resign_rate': None, 'false_resign_rate': None,
                    'resign_playout_games': 0}
        return {
            'resign_rate': (round(s['resigned'] / s['games'], 4)
                            if s['games'] else None),
            'false_resign_rate': (round(s['false_resigns'] / s['would_resign'], 4)
                                  if s['would_resign'] else None),
            'resign_playout_games': s['playout_games'],
            'resign_checked_games': s['would_resign'],
            'resign_suppressed': self._collapse_active,
        }

    def _policy_entropy_trend(self, current: Optional[float]) -> dict:
        """
        Compare the current policy entropy against a window of history.

        The baseline is the MEAN of the five logged iterations ending
        `collapse_entropy_trend_lookback` back, not the single value at that
        point: iteration-to-iteration entropy is noisy enough (the probe samples
        128 buffer positions) that one reading makes a poor anchor and would
        produce both false alarms and misses.

        Returns {'baseline': float|None, 'drop': float|None} where drop is the
        relative fall, so 0.25 means a quarter of the entropy is gone. None
        means not enough history yet — the check is skipped, not passed.
        """
        cfg = self.config.training
        out = {'baseline': None, 'drop': None}
        if current is None or current <= 0:
            return out
        look = max(1, cfg.collapse_entropy_trend_lookback)
        # metrics_history is append-per-iteration, so index -look is `look`
        # iterations ago. A run resumed from a checkpoint starts with an empty
        # history, so this silently no-ops until the window fills — which is
        # correct, since the pre-restart entropy was measured under whatever
        # config that run used.
        if len(self.metrics_history) < look:
            return out
        window = self.metrics_history[-look:-look + 5] or self.metrics_history[-look:-look + 1]
        vals = [m.get('policy_entropy') for m in window]
        vals = [v for v in vals if isinstance(v, (int, float)) and v > 0]
        if not vals:
            return out
        baseline = sum(vals) / len(vals)
        out['baseline'] = round(baseline, 4)
        out['drop'] = round(max(0.0, (baseline - current) / baseline), 4)
        return out

    @staticmethod
    def _empty_label_stats() -> dict:
        return {'entropy_sum': 0.0, 'support_sum': 0.0,
                'max_prob_sum': 0.0, 'games': 0}

    def _label_metrics(self) -> dict:
        """
        Entropy of the policy labels THIS iteration produced, averaged per game.

        Distinct from the collapse guard's `target_entropy`, which reads the
        replay buffer: at 50k samples the buffer holds ~30 iterations, so a
        change to the label schedule takes that long to show up there. These
        three describe only the games just played.
        """
        s = self._label_stats
        if not s['games']:
            return {'iter_target_entropy': None, 'iter_target_support': None,
                    'iter_target_max_prob': None}
        g = s['games']
        return {
            'iter_target_entropy': round(s['entropy_sum'] / g, 4),
            'iter_target_support': round(s['support_sum'] / g, 2),
            'iter_target_max_prob': round(s['max_prob_sum'] / g, 4),
        }

    def _build_stages_overview(self, current_stage_key: str) -> list:
        """Build metadata for the 4-stage learning pipeline."""
        gate_enabled = bool(getattr(self.config.training, 'gate_enabled', True) and getattr(self.config.training, 'gate_games', 0) > 0)

        stage_order = [
            ('self_play', 'Self-Play', '🎲', 'Generate MCTS training games'),
            ('training', 'NN Training', '🧠', 'Optimize Policy & Value Loss'),
            ('gate', 'Champion Gate', '🛡️', 'Candidate vs Champion Match' if gate_enabled else 'Gate Disabled'),
            ('saving', 'Checkpoint', '💾', 'Save Weights & Collapse Guard'),
        ]

        keys = [s[0] for s in stage_order]
        current_idx = keys.index(current_stage_key) if current_stage_key in keys else (-1 if current_stage_key == 'idle' else 99)

        stages = []
        for idx, (key, label, icon, desc) in enumerate(stage_order):
            is_skipped = (key == 'gate' and not gate_enabled)

            if is_skipped:
                status = 'skipped'
            elif current_stage_key == 'idle':
                status = 'idle'
            elif current_stage_key == 'completed':
                status = 'completed'
            elif idx < current_idx:
                status = 'completed'
            elif idx == current_idx:
                status = 'active'
            else:
                status = 'pending'
                
            stages.append({
                'key': key,
                'label': label,
                'icon': icon,
                'description': desc,
                'status': status,
                'index': idx + 1,
            })
        return stages

    def _set_stage(
        self,
        stage: str,
        stage_name: str,
        stage_index: int,
        completed_items: int = 0,
        total_items: int = 0,
        active_games: list = None,
        num_workers: int = 0,
        detail: str = "",
        extra_data: dict = None,
    ) -> None:
        """Update current stage status atomically."""
        if active_games is None:
            active_games = []
        percent = round((completed_items / total_items * 100)) if total_items > 0 else 0
        state = {
            'stage': stage,
            'stage_name': stage_name,
            'stage_index': stage_index,
            'total_stages': 4,
            'iteration': self.iteration,
            'completed_items': completed_items,
            'total_items': total_items,
            'percent': percent,
            'active_games': active_games,
            'num_workers': num_workers,
            'detail': detail,
            'stages_overview': self._build_stages_overview(stage),
        }
        if extra_data:
            state.update(extra_data)
        self.current_stage = state

    def _try_load_weights(self) -> None:
        """Try to resume from saved weights."""
        weights_path = self.config.paths.weights_path
        if os.path.isfile(weights_path):
            try:
                meta = load_weights(weights_path, self.network, self.optimizer,
                                    self.device, scheduler=self.scheduler)
                if meta:
                    if meta.get('scheduler_restored'):
                        self._emit('info',
                                   'Restored LR schedule at '
                                   f"{self.optimizer.param_groups[0]['lr']:.2e}")
                    else:
                        # Pre-fix checkpoints have no scheduler state. Fast-forward
                        # the cosine to where this model's iteration count says it
                        # should be, so a long-running model does not get handed a
                        # full-strength LR on every resume for the rest of its life.
                        resume_at = int(meta.get('iteration', 0))
                        if resume_at > 0:
                            self.scheduler.last_epoch = resume_at - 1
                            # Torch warns when scheduler.step() precedes any
                            # optimizer.step(), which is unavoidable here: this
                            # runs at construction, before training starts. The
                            # warning is about skipping the schedule's first LR,
                            # and skipping ahead is exactly the intent.
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', UserWarning)
                                self.scheduler.step()
                            self._emit('info',
                                       'No saved LR schedule — fast-forwarded cosine to '
                                       f"iteration {resume_at} "
                                       f"(lr {self.optimizer.param_groups[0]['lr']:.2e})")
                    self.iteration = meta.get('iteration', 0)
                    # The checkpoint's `elo` is a snapshot from whenever it was
                    # written; the ledger may have moved since. Use it only as
                    # the fallback when there is no ledger to read.
                    if self._refresh_elo() is None:
                        self.elo = meta.get('elo', DEFAULT_ELO)
                    self.total_games = meta.get('total_games', 0)
                    # Checkpoints predating the gate ladder seed it from the
                    # anchor rating rather than restarting it at zero.
                    self.gate_elo = float(meta.get('gate_elo') or self.elo)
                    self.current_stage['iteration'] = self.iteration
                    # Restore the gated champion as the self-play generator.
                    # Checkpoints written before gating existed have no champion
                    # entry, so fall back to the training weights.
                    champion = meta.get('champion_state_dict')
                    if champion:
                        self.eval_network.load_state_dict(champion)
                        self._emit('info', 'Restored gated champion for self-play')
                    else:
                        self.eval_network.load_state_dict(self.network.state_dict())
                    self._emit('info', f"Resumed from weights: iteration {self.iteration}, Elo {self.elo:.0f}")
            except Exception as e:
                self._emit('warning', f"Failed to load weights: {e}")
    
    @property
    def _replay_path(self) -> str:
        return buffer_path(self.config.paths.data_dir)

    def _try_load_replay_buffer(self) -> None:
        """Restore the replay buffer saved next to weights.pt, if it fits."""
        try:
            samples, reason = load_buffer(
                self._replay_path,
                board_size=self.config.board.size,
                num_input_planes=self.config.network.num_input_planes,
            )
        except Exception as e:
            self._emit('warning', f'Failed to load replay buffer: {e}')
            return

        if reason:
            self._emit('warning', reason)
            return
        if samples:
            self.replay_buffer.add(samples)
            self._emit('info',
                       f'Restored {len(samples)} replay samples '
                       f'({len(self.replay_buffer)} in buffer)')

    def _save_replay_buffer(self) -> None:
        """
        Persist the replay buffer. Called from the same place as the weights
        save so the two never drift apart by more than one iteration.
        """
        try:
            save_buffer(
                self.replay_buffer.buffer,
                self._replay_path,
                board_size=self.config.board.size,
                num_input_planes=self.config.network.num_input_planes,
                max_samples=self.config.training.replay_buffer_size,
            )
        except Exception as e:
            # Never let this take down a training run — the buffer is a
            # performance asset, not a correctness requirement.
            self._emit('warning', f'Failed to save replay buffer: {e}')

    def _emit(self, event_type: str, message: str = "", data: dict = None) -> None:
        """Send a progress event to the web dashboard."""
        if self.progress_callback:
            event = {
                'type': event_type,
                'message': message,
                # Lets the browser notice that the series it is charting now
                # belongs to a different model and reset instead of appending.
                'model_id': self.model_id,
                'iteration': self.iteration,
                'total_games': self.total_games,
                'elo': self.elo,
                'kyu_rank': elo_to_rank(self.elo),
                'current_stage': self.current_stage,
                'timestamp': datetime.now().isoformat(),
            }
            if data:
                event.update(data)
            
            # Store log for web UI refreshes
            if message:
                self.recent_logs.append(event)
                
            self.progress_callback(event)

    
    def _check_force_stop(self) -> bool:
        """Helper callback passed to workers to check if force stop requested."""
        return self._force_stop_requested

    def force_stop(self) -> None:
        """
        Immediately stop learning without corrupting weights.
        
        Halts the active iteration immediately and reloads clean weights
        from the last saved checkpoint on disk.
        """
        self._stop_requested = True
        self._force_stop_requested = True
        self._emit('warning', '⚡ Force stop requested! Halting immediately and restoring last clean iteration weights...')

    def train(self, max_iterations: Optional[int] = None) -> None:
        """
        Run the full training loop.
        
        Args:
            max_iterations: Stop after this many iterations (None = run forever).
        """
        self.is_running = True
        self._stop_requested = False
        
        self._set_stage('self_play', 'Starting Training', 1, 0, 0, [], 0, 'Initializing iteration')
        self._emit('training_started', 'Training started')
        
        try:
            while not self._stop_requested and not self._force_stop_requested:
                if max_iterations and self.iteration >= max_iterations:
                    break
                
                iteration_start = time.time()
                self.iteration += 1
                
                self._emit('iteration_start', f'Starting iteration {self.iteration}')
                
                # --- Phase 1: Self-play ---
                num_sp = self.config.training.num_self_play_games
                sp_workers = max(1, min(num_sp, self.config.training.num_parallel_workers, os.cpu_count() or 4))
                self._set_stage(
                    'self_play', 'Self-Play Data Generation', 1,
                    completed_items=0, total_items=num_sp,
                    active_games=[], num_workers=sp_workers,
                    detail=f'Generating {num_sp} self-play games'
                )
                self._emit('self_play_start', f'Self-play: generating {num_sp} games')

                # Reset per-iteration pass tallies for the collapse guard.
                self._pass_stats = {1: [0, 0], 2: [0, 0]}
                self._label_stats = self._empty_label_stats()
                self._resign_stats = self._empty_resign_stats()

                # Mercy rule is suppressed while the collapse guard is tripped.
                # A flat value head emits ±1 from the turn-colour plane alone,
                # so every game would "resign" almost immediately — and because
                # the games would then be a handful of moves long, the guard's
                # own pass-rate tripwire would go quiet and hide the collapse
                # that caused it.
                resign_on = bool(self.config.training.resign_enabled
                                 and not self._collapse_active)
                if self.config.training.resign_enabled and self._collapse_active:
                    self._emit('warning',
                               '⚠️ Mercy rule suppressed this iteration — the collapse '
                               'guard is tripped, so the value head\'s resignation '
                               'verdicts cannot be trusted.')


                # Effective sim count scales with board area so policy targets
                # stay sharp on larger boards (see config.simulations_for_board).
                effective_sims = simulations_for_board(
                    self.config.mcts, self.config.board.size)

                sp_start = time.time()
                # Use eval_network on CPU for self-play (much faster for batch_size=1)
                samples = run_self_play_batch(
                    network=self.eval_network,
                    num_games=self.config.training.num_self_play_games,
                    board_size=self.config.board.size,
                    komi=self.config.board.komi,
                    num_simulations=effective_sims,
                    c_puct=self.config.mcts.c_puct,
                    temperature_threshold=self.config.mcts.temperature_threshold,
                    temperature_init=self.config.mcts.temperature_init,
                    temperature_final=self.config.mcts.temperature_final,
                    policy_target_temperature=self.config.mcts.policy_target_temperature,
                    device="cpu",
                    game_store_every_n=self.config.training.game_store_every_n,
                    games_dir=self.config.paths.games_dir,
                    record_games=self.config.training.record_self_play_games,
                    iteration=self.iteration,
                    progress_callback=self._on_game_complete,
                    fpu_reduction=self.config.mcts.fpu_reduction,
                    value_target_outcome_weight=self.config.training.value_target_outcome_weight,
                    stop_checker=self._check_force_stop,
                    num_workers=self.config.training.num_parallel_workers,
                    restrict_eye_fill=self.config.training.restrict_eye_fill,
                    restrict_self_atari=self.config.training.restrict_self_atari,
                    self_atari_max_stones=self.config.training.self_atari_max_stones,
                    resign_enabled=resign_on,
                    resign_threshold=self.config.training.resign_threshold,
                    resign_consecutive=self.config.training.resign_consecutive,
                    resign_min_move_factor=self.config.training.resign_min_move_factor,
                    resign_both_sides=self.config.training.resign_both_sides,
                    resign_playout_fraction=self.config.training.resign_playout_fraction,
                )
                sp_seconds = round(time.time() - sp_start, 2)
                
                if self._force_stop_requested:
                    break

                # Credit the games that actually finished, not the games that
                # were requested — a force-stop cancels part of the batch, and
                # counting the cancelled ones drifts total_games away from the
                # number of games on disk for good.
                self.total_games += self._resign_stats['games']
                self.replay_buffer.add(samples)
                
                self._set_stage(
                    'self_play', 'Self-Play Done', 1,
                    completed_items=num_sp, total_items=num_sp,
                    active_games=[], num_workers=sp_workers,
                    detail=f'Generated {len(samples)} samples, replay buffer: {len(self.replay_buffer)}'
                )
                self._emit('self_play_done',
                           f'Self-play done: {len(samples)} samples, buffer size: {len(self.replay_buffer)}')

                # Mercy-rule report. The false-resign rate is the tuning signal:
                # if it climbs above ~5% the threshold is ending games that the
                # "hopeless" side would have gone on to win, and every such game
                # trains ~board_size² samples on a label that was never true.
                rs = self._resign_stats
                if resign_on and rs['resigned']:
                    fr = self._resign_metrics()['false_resign_rate']
                    fr_txt = ('no playout games triggered it — no evidence yet'
                              if fr is None else f'{fr:.0%} of checked games')
                    self._emit('resign_summary',
                               f'🏳️ Mercy rule: {rs["resigned"]}/{rs["games"]} games '
                               f'resigned · wrong-resignation rate: {fr_txt}',
                               data=self._resign_metrics())
                
                # --- Phase 2: Training ---
                train_metrics = {}
                gate_seconds = 0.0
                if len(self.replay_buffer) >= self.config.training.batch_size:
                    steps_per_epoch = max(1, min(200, len(self.replay_buffer) // self.config.training.batch_size))
                    total_train_steps = steps_per_epoch * self.config.training.num_epochs_per_iteration
                    self._set_stage(
                        'training', 'Neural Network Training', 2,
                        completed_items=0, total_items=total_train_steps,
                        active_games=[], num_workers=0,
                        detail='Optimizing policy & value loss on replay buffer samples'
                    )
                    self._emit('training_start', 'Training network...')

                    train_metrics = self._train_network()

                    if self._force_stop_requested:
                        break

                    # --- Promotion gate ---
                    gate_start = time.time()
                    promoted, gate_wr = self._run_promotion_gate()
                    gate_seconds = round(time.time() - gate_start, 2)
                    train_metrics['gate_win_rate'] = gate_wr
                    train_metrics['gate_promoted'] = promoted
                    train_metrics['champion_gate_seconds'] = gate_seconds

                    if self._force_stop_requested:
                        break

                    self._emit('training_done', 'Training done', data=train_metrics)
                
                if self._force_stop_requested:
                    break

                # The random-bot evaluation phase used to sit here. It is
                # gone: a fixed anchor stops carrying information the moment
                # the bot wins every game, and the rating it produced then
                # counted iterations rather than strength. Competitive Elo now
                # moves only in `ai/match.py`, one played game at a time, and
                # is recorded in the model's elo_history.jsonl ledger.
                kyu = elo_to_rank(self.elo)

                # --- Phase 3: Save weights (EVERY iteration to prevent data loss) ---
                self._set_stage(
                    'saving', 'Saving Checkpoint & Diagnostics', 4,
                    completed_items=0, total_items=1,
                    active_games=[], num_workers=0,
                    detail=f'Saving iteration {self.iteration} weights & evaluating collapse guard'
                )
                self._save_weights()
                
                # --- Collapse guard ---
                # Runs after the champion is settled so it measures the network
                # that will actually generate the next batch of self-play games.
                diagnostics = self._collapse_diagnostics()

                # --- Record metrics ---
                elapsed = time.time() - iteration_start
                metrics = {
                    'iteration': self.iteration,
                    'total_games': self.total_games,
                    'elo': self.elo,
                    'kyu_rank': kyu,
                    'buffer_size': len(self.replay_buffer),
                    'elapsed_seconds': round(elapsed, 1),
                    'self_play_seconds': sp_seconds,
                    'champion_gate_seconds': gate_seconds,
                    'timestamp': datetime.now().isoformat(),
                    'gate_win_rate': self.last_gate_win_rate,
                    'gate_rejections': self.gate_rejections,
                    'gate_elo': round(self.gate_elo, 1),
                    **diagnostics,
                    **self._label_metrics(),
                    **self._resign_metrics(),
                }
                if train_metrics:
                    metrics.update(train_metrics)
                
                self.metrics_history.append(metrics)
                self._save_metrics_log(metrics)
                
                # --- Reflection milestone ---
                if self.total_games % self.config.training.reflection_interval_games < \
                   self.config.training.num_self_play_games:
                    self._emit_reflection()
                
                self._set_stage(
                    'completed', f'Iteration {self.iteration} Completed', 4,
                    completed_items=1, total_items=1,
                    active_games=[], num_workers=0,
                    detail=f'Iteration {self.iteration} finished in {elapsed:.1f}s | Elo {self.elo:.0f} ({kyu})'
                )
                self._emit('iteration_done',
                           f'Iteration {self.iteration} done in {elapsed:.1f}s | '
                           f'Elo: {self.elo:.0f} ({kyu})',
                           data=metrics)
                
                # Update model state in config.json
                self._update_model_state()
        
        except Exception as e:
            self._set_stage('idle', 'Training Error', 0, 0, 0, [], 0, f'Error: {str(e)}')
            self._emit('error', f'Training error: {str(e)}')
            raise
        finally:
            self.is_running = False
            if self._force_stop_requested:
                # Reload clean weights from disk from last saved completed iteration
                self._try_load_weights()
                self._update_model_state()
                self._set_stage('idle', 'Force Stopped', 0, 0, 0, [], 0, f'Restored weights from iteration {self.iteration}')
                self._emit('training_stopped', f'⚡ Force stopped! Restored weights from iteration {self.iteration} (Elo {self.elo:.0f}).')
                self._force_stop_requested = False
            else:
                # Always save weights on normal stop
                self._save_weights()
                self._set_stage('idle', 'Idle', 0, 0, 0, [], 0, f'Ready · Iteration {self.iteration} · Elo {self.elo:.0f}')
                self._emit('training_stopped', 'Training stopped')
    
    def _cpu_copy(self, source: GoNetwork) -> GoNetwork:
        """Build a CPU-resident clone of `source` for head-to-head evaluation."""
        clone = GoNetwork(
            board_size=self.config.board.size,
            num_input_planes=self.config.network.num_input_planes,
            input_features=self.config.network.input_features,
            num_res_blocks=self.config.network.num_res_blocks,
            num_filters=self.config.network.num_filters,
            value_head_hidden=self.config.network.value_head_hidden,
        )
        clone.load_state_dict({k: v.detach().cpu() for k, v in source.state_dict().items()})
        clone.to("cpu").eval()
        return clone

    def _run_promotion_gate(self) -> tuple:
        """
        Decide whether the freshly trained candidate replaces the champion.

        The champion (`self.eval_network`) is the network that generates all
        self-play data, so letting a worse network take that role poisons every
        subsequent iteration. The candidate must beat the champion head-to-head
        by `gate_threshold` before it is promoted.

        Returns:
            (promoted, win_rate) — win_rate is None when gating is disabled.
        """
        cfg = self.config.training

        if not cfg.gate_enabled or cfg.gate_games <= 0:
            # Legacy behaviour: accept every update unconditionally.
            self.eval_network.load_state_dict(self.network.state_dict())
            self._set_stage(
                'gate', 'Promotion Gate (Disabled)', 3,
                completed_items=0, total_items=0,
                active_games=[], num_workers=0,
                detail='Gating disabled — candidate accepted automatically'
            )
            return True, None

        workers = max(1, min(cfg.num_parallel_workers, cfg.gate_games, os.cpu_count() or 4))

        # The gate is a measurement, and it is only worth anything if it
        # measures the regime the model actually plays in. Running it at 50 sims
        # while self-play runs at 200 and real matches at 400 selects for
        # whichever network happens to be best at a search depth nobody uses —
        # policy-heavy networks look strong at low sim counts because there is
        # not enough search to overrule the prior. Warn once per iteration
        # rather than silently overriding the user's number.
        if cfg.gate_simulations < self.config.mcts.num_simulations / 2:
            self._emit('warning',
                       f'Gate runs at {cfg.gate_simulations} sims but self-play runs at '
                       f'{self.config.mcts.num_simulations} — the gate is selecting for a '
                       f'search depth the model never plays at. Raise gate_simulations to '
                       f'at least {self.config.mcts.num_simulations // 2}.')

        self._set_stage(
            'gate', 'Promotion Gate (Candidate vs Champion)', 3,
            completed_items=0, total_items=cfg.gate_games,
            active_games=[], num_workers=workers,
            detail=f'Playing candidate vs champion ({cfg.gate_games} games @ {cfg.gate_simulations} sims, {workers} workers)'
        )
        self._emit('gate_start',
                   f'Promotion gate: candidate vs champion '
                   f'({cfg.gate_games} games @ {cfg.gate_simulations} sims, '
                   f'{workers} workers)')

        candidate = self._cpu_copy(self.network)
        champion = self._cpu_copy(self.eval_network)

        def _on_gate_game_progress(completed, total, active_games=None, num_workers=None, candidate_wins=0):
            self._set_stage(
                'gate', 'Promotion Gate (Candidate vs Champion)', 3,
                completed_items=completed, total_items=total,
                active_games=active_games or [], num_workers=num_workers or workers,
                detail=f'Gate match: {completed}/{total} games completed ({candidate_wins} candidate wins)'
            )
            self._emit('gate_progress', f'Gate match: {completed}/{total} ({len(active_games or [])} parallel)', data={'current_stage': self.current_stage})

        try:
            win_rate = evaluate_against_checkpoint(
                current_network=candidate,
                opponent_network=champion,
                board_size=self.config.board.size,
                komi=self.config.board.komi,
                num_simulations=cfg.gate_simulations,
                num_games=cfg.gate_games,
                device="cpu",
                iteration=self.iteration,
                games_dir=self.config.paths.games_dir,
                record_games=cfg.record_gate_games,
                num_workers=cfg.num_parallel_workers,
                stop_checker=self._check_force_stop,
                progress_callback=_on_gate_game_progress,
                restrict_eye_fill=cfg.restrict_eye_fill,
                c_puct=self.config.mcts.c_puct,
                fpu_reduction=self.config.mcts.fpu_reduction,
                restrict_self_atari=cfg.restrict_self_atari,
                self_atari_max_stones=cfg.self_atari_max_stones,
            )
        except Exception as e:
            # Never let a gate failure take down training — fall back to
            # keeping the existing champion, which is the safe direction.
            self._emit('warning', f'Promotion gate failed ({e}); keeping champion')
            return False, None

        if self._force_stop_requested:
            # The match was cut short, so its win rate is not a verdict on the
            # candidate. Leave the champion (and the rejection streak) alone.
            return False, None

        self.last_gate_win_rate = win_rate

        if win_rate >= cfg.gate_threshold:
            self.eval_network.load_state_dict(
                {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
            )
            # The candidate started as a copy of the champion, so their rating
            # gap is what this match just measured — but only the part of it
            # that exceeds the promotion bar is evidence of strength rather than
            # of the selection rule. See promotion_elo_gain.
            gap = promotion_elo_gain(win_rate, cfg.gate_games, cfg.gate_threshold)
            self.gate_elo += gap
            self.gate_rejections = 0
            self._set_stage(
                'gate', 'Promotion Gate (Promoted)', 3,
                completed_items=cfg.gate_games, total_items=cfg.gate_games,
                active_games=[], num_workers=workers,
                detail=f'✅ Candidate promoted (beat champion {win_rate:.0%} ≥ {cfg.gate_threshold:.0%})'
            )
            self._emit('gate_promoted',
                       f'✅ Candidate promoted (beat champion {win_rate:.0%} '
                       f'≥ {cfg.gate_threshold:.0%}) · gate Elo '
                       f'{self.gate_elo:.0f} ({gap:+.0f})',
                       data={'gate_win_rate': win_rate,
                             'gate_elo': self.gate_elo,
                             'gate_elo_gain': round(gap, 1)})
            return True, win_rate

        # Rejected: the champion keeps generating self-play data.
        self.gate_rejections += 1
        self._set_stage(
            'gate', 'Promotion Gate (Rejected)', 3,
            completed_items=cfg.gate_games, total_items=cfg.gate_games,
            active_games=[], num_workers=workers,
            detail=f'⛔ Candidate rejected ({win_rate:.0%} < {cfg.gate_threshold:.0%}) — champion retained'
        )
        self._emit('gate_rejected',
                   f'⛔ Candidate rejected ({win_rate:.0%} < {cfg.gate_threshold:.0%}) — '
                   f'champion retained [{self.gate_rejections} in a row]',
                   data={'gate_win_rate': win_rate,
                         'gate_rejections': self.gate_rejections})


        # A long rejection streak means the training network has drifted
        # somewhere it cannot recover from on its own. Snap it back to the
        # champion so the next iterations retrain from a known-good point
        # instead of deadlocking forever.
        if self.gate_rejections >= cfg.gate_stall_warning:
            self.network.load_state_dict(
                {k: v.to(self.device) for k, v in self.eval_network.state_dict().items()}
            )
            self.gate_rejections = 0
            self.stall_breaks += 1
            self._emit('warning',
                       f'⚠️ {cfg.gate_stall_warning} consecutive rejections — '
                       f'training network reset to the champion to break the stall '
                       f'(stall #{self.stall_breaks}).')

            # Repeated stalls are qualitatively different from one bad streak:
            # resetting to the champion only helps if the DATA can produce a
            # better candidate. If it cannot — e.g. self-play is so one-sided
            # that every candidate inherits a degenerate value target — this
            # loop is stable and training silently makes no progress forever.
            # Escalate rather than let that look like normal operation.
            if self.stall_breaks >= 3:
                self._emit('collapse_warning',
                           f'🚨 Training has stalled {self.stall_breaks} times: no candidate '
                           f'can beat the champion. This usually means the champion\'s '
                           f'self-play is too one-sided to learn from — check the colour '
                           f'balance and pass rates rather than waiting it out.')

        return False, win_rate

    def _collapse_diagnostics(self) -> dict:
        """
        Tripwires for the two ways this loop degrades without anything failing.

        VALUE-HEAD COLLAPSE. When one colour wins nearly every game, the value
        head can drive its loss to ~0 by reading only the turn-colour plane
        (`v = +1 if Black to move else -1`), ignoring the board entirely. MCTS
        then sees a constant Q for every child and stops evaluating — 800
        simulations of nothing. The spread must be measured WITHIN each colour:
        across mixed positions a collapsed head still looks bimodal (±1) and
        would appear healthy.

        POLICY-HEAD COLLAPSE (overfitting to self-play). The prior narrows onto
        a handful of moves until the model can only play — and only knows how to
        answer — its own lines. Nothing else in the loop can see this: self-play
        is unaffected because both sides share the narrowed move set, the gate
        reads ~50% because a network is no worse than itself, and the value head
        stays healthy throughout. The first external evidence is losing to
        opponents it used to beat.

        Measured as mean Shannon entropy over LEGAL moves, in nats, at two
        points in the loop, because they narrow in this order:
          - `target_entropy`: the training LABEL, straight from the replay
            buffer. Narrows first if policy_target_temperature < 1.
          - `policy_entropy`: what the champion network actually outputs. Follows
            the label down over the next few dozen iterations.
        Also reported: `policy_top5_mass` (share of probability on the best five
        moves) and `target_one_hot_frac` (labels that are a single move), which
        are the two numbers that make the state of it obvious at a glance.
        """
        cfg = self.config.training
        out = {
            'value_std_black': None,
            'value_std_white': None,
            'pass_rate_black': None,
            'pass_rate_white': None,
            'policy_entropy': None,
            'policy_top5_mass': None,
            'policy_entropy_baseline': None,
            'policy_entropy_drop': None,
            'target_entropy': None,
            'target_one_hot_frac': None,
            'target_support': None,
            'collapse_warning': None,
        }

        # --- Value-head spread, split by side to move ---
        # Needs the turn-colour plane (index 9) to separate the sides.
        from game.features import resolve as _resolve_features
        turn_plane = _resolve_features(self.config.network.input_features).turn_colour_plane
        if (len(self.replay_buffer) >= 16
                and self.config.network.num_input_planes > turn_plane):
            n = min(cfg.collapse_probe_positions, len(self.replay_buffer))
            # Policies come out of the same draw as the states, so the label and
            # network numbers below describe the same positions and can be
            # compared against each other directly.
            states, policies, _ = self.replay_buffer.sample(n, augment=False)
            with torch.no_grad():
                # predict_batch returns softmaxed PROBABILITIES, not logits.
                prior_probs, values = self.eval_network.predict_batch(states, "cpu")
            # The turn-colour plane is all-ones exactly when Black is to move.
            # Its index comes from the feature set — hardcoding 9 here would
            # silently mis-split the probe if a set ever reorders its planes.
            is_black = states[:, turn_plane, 0, 0] > 0.5
            for key, mask in (('value_std_black', is_black),
                              ('value_std_white', ~is_black)):
                sel = values[mask]
                if sel.numel() >= 2:
                    out[key] = round(float(sel.std().item()), 5)

            # --- Policy-head spread, over LEGAL moves only ---
            # Entropy over all 82 outputs would be dominated by the occupied
            # points the network correctly gives ~0, so it would drift down as a
            # game fills up regardless of how the prior is behaving. Planes 0
            # and 1 are the two stone planes in every feature set (they are
            # written by features._fill_base_planes), so their sum marks the
            # occupied points; the remainder plus pass is a close enough stand-in
            # for legality here — it over-counts only ko and suicide.
            with torch.no_grad():
                occupied = (states[:, 0] + states[:, 1]).reshape(n, -1) > 0.5
                legal = torch.ones_like(prior_probs, dtype=torch.bool)
                legal[:, :occupied.shape[1]] = ~occupied
                # Renormalise over the legal subset so the number means "how
                # spread out is the prior across the moves available here".
                priors = prior_probs * legal
                priors = priors / priors.sum(dim=1, keepdim=True).clamp_min(1e-12)
                ent = -(priors * priors.clamp_min(1e-12).log()).sum(dim=1)
                out['policy_entropy'] = round(float(ent.mean()), 4)
                top5 = priors.topk(min(5, priors.shape[1]), dim=1).values.sum(dim=1)
                out['policy_top5_mass'] = round(float(top5.mean()), 4)

                # --- Training-label spread, same positions ---
                pol = policies.float()
                lab_ent = -(pol * pol.clamp_min(1e-12).log()).sum(dim=1)
                out['target_entropy'] = round(float(lab_ent.mean()), 4)
                out['target_one_hot_frac'] = round(
                    float((pol.max(dim=1).values > 0.99).float().mean()), 4)
                out['target_support'] = round(
                    float((pol > 1e-6).sum(dim=1).float().mean()), 2)

        # --- Pass rates from this iteration's self-play ---
        for colour, key in ((1, 'pass_rate_black'), (2, 'pass_rate_white')):
            passes, total = self._pass_stats[colour]
            if total > 0:
                out[key] = round(passes / total, 4)

        if not cfg.collapse_guard_enabled:
            # Guard off means no verdict, not a standing one — otherwise a
            # trip recorded before it was disabled would suppress the mercy
            # rule forever.
            self._collapse_active = False
            return out

        # --- Evaluate tripwires ---
        problems = []
        stds = [v for v in (out['value_std_black'], out['value_std_white']) if v is not None]
        if stds and max(stds) < cfg.collapse_value_std_min:
            problems.append(
                f"value head is flat (max within-colour std {max(stds):.4f} < "
                f"{cfg.collapse_value_std_min}) — it is predicting from turn colour alone, "
                f"so MCTS has no positional signal"
            )

        for key, name in (('pass_rate_black', 'Black'), ('pass_rate_white', 'White')):
            rate = out[key]
            if rate is not None and rate > cfg.collapse_pass_rate_max:
                problems.append(
                    f"{name} is passing {rate:.0%} of its moves "
                    f"(> {cfg.collapse_pass_rate_max:.0%}) — under area scoring that "
                    f"donates points every move"
                )

        # Read by the next iteration's self-play phase to suppress the mercy
        # rule while the value head cannot be trusted to judge a lost game.
        # Deliberately computed from the value/pass tripwires ONLY: a narrow
        # policy says nothing about whether the value head can judge a lost
        # game, and suppressing the mercy rule for it would just make every
        # iteration slower while the real problem went unfixed.
        self._collapse_active = bool(problems)

        # --- Policy-head narrowing (reported, does not gate the mercy rule) ---
        if (out['target_entropy'] is not None
                and out['target_entropy'] < cfg.collapse_target_entropy_min):
            problems.append(
                f"training LABELS are near-deterministic (mean entropy "
                f"{out['target_entropy']:.2f} nats < {cfg.collapse_target_entropy_min}, "
                f"{out['target_one_hot_frac']:.0%} one-hot, mean support "
                f"{out['target_support']:.1f} moves) — the policy head is being taught "
                f"its own argmax instead of the search's judgement. Check that "
                f"policy_target_temperature is 1.0"
            )

        if (out['policy_entropy'] is not None
                and out['policy_entropy'] < cfg.collapse_policy_entropy_min):
            problems.append(
                f"policy head has narrowed (mean legal-move entropy "
                f"{out['policy_entropy']:.2f} nats < {cfg.collapse_policy_entropy_min}, "
                f"{out['policy_top5_mass']:.0%} of the prior on 5 moves) — the model is "
                f"overfitting to its own self-play lines and will lose to opponents "
                f"that play outside them, while self-play and the gate still look fine"
            )

        # The trend check. An absolute floor cannot be calibrated from the runs
        # available, but a sustained decline is unambiguous whatever the healthy
        # value turns out to be for a given board size and sim budget.
        drift = self._policy_entropy_trend(out['policy_entropy'])
        out['policy_entropy_baseline'] = drift['baseline']
        out['policy_entropy_drop'] = drift['drop']
        if drift['drop'] is not None and drift['drop'] >= cfg.collapse_entropy_trend_drop:
            problems.append(
                f"policy entropy is trending down: {drift['baseline']:.2f} -> "
                f"{out['policy_entropy']:.2f} nats ({drift['drop']:.0%} fall over "
                f"{cfg.collapse_entropy_trend_lookback} iterations). The prior is "
                f"narrowing onto its own self-play lines — expect losses to external "
                f"opponents before self-play or the gate shows anything"
            )

        if problems:
            out['collapse_warning'] = '; '.join(problems)
            self._emit('collapse_warning',
                       '🚨 COLLAPSE GUARD: ' + out['collapse_warning'],
                       data=out)
            if cfg.collapse_auto_stop:
                self._stop_requested = True
                self._emit('warning', 'collapse_auto_stop is on — halting training.')

        return out

    def _save_weights(self) -> None:
        """Save current model weights."""
        try:
            save_weights(
                model=self.network,
                optimizer=self.optimizer,
                iteration=self.iteration,
                elo=self.elo,
                kyu_rank=elo_to_rank(self.elo),
                total_games=self.total_games,
                weights_path=self.config.paths.weights_path,
                champion_state_dict=self.eval_network.state_dict(),
                gate_elo=self.gate_elo,
                scheduler=self.scheduler,
            )
            self._emit('info', f'Weights saved (iteration {self.iteration})')
        except Exception as e:
            self._emit('warning', f'Failed to save weights: {e}')

        self._save_replay_buffer()

    def save_weights_now(self) -> str:
        """
        Force an immediate weights save. Called from API.

        The champion has to be written here too. Without it the file carries no
        champion entry, and the next restart falls back to treating the training
        network as champion — silently handing self-play generation to a
        candidate that never passed the gate. One click of "save weights" used
        to be enough to defeat gating across a restart.
        """
        save_weights(
            model=self.network,
            optimizer=self.optimizer,
            iteration=self.iteration,
            elo=self.elo,
            kyu_rank=elo_to_rank(self.elo),
            total_games=self.total_games,
            weights_path=self.config.paths.weights_path,
            champion_state_dict=self.eval_network.state_dict(),
            gate_elo=self.gate_elo,
            scheduler=self.scheduler,
        )
        self._emit('info', f'Weights saved manually (iteration {self.iteration})')
        return self.config.paths.weights_path

    def _update_model_state(self) -> None:
        """
        Write back what training owns — iteration and total games — and pick up
        any rating the model earned in matches while it was training.

        The rating is READ here, never written: `ai/match.py` owns it now. A
        long training run and a bot-vs-bot series can overlap, and the trainer's
        in-memory `self.elo` goes stale the moment the model plays a game.
        """
        if not self.model_id:
            return
        try:
            from model_manager import ModelManager
            mgr = ModelManager()
            mgr.update_training_state(self.model_id, self.iteration, self.total_games)
            self._refresh_elo(mgr)
        except Exception:
            pass  # Non-critical

    def _refresh_elo(self, manager=None) -> Optional[float]:
        """
        Reload the competitive rating from the model's Elo ledger.

        Returns None — leaving `self.elo` untouched — when there is no model
        directory to read from. A Trainer built straight from a Config, as the
        CLI and the tests do, is not attached to anything that holds a ledger,
        and its rating then has to come from the checkpoint as before.
        """
        if not self.model_id:
            return None
        try:
            import elo_history
            from model_manager import ModelManager
            mgr = manager or ModelManager()
            info = mgr.get_model(self.model_id)
            if info is None:
                return None
            self.elo = elo_history.current_elo(
                mgr.get_model_dir(self.model_id), fallback=info.elo)
            return self.elo
        except Exception:
            return None
    
    def stop(self) -> None:
        """Request graceful stop of the training loop."""
        self._stop_requested = True
        self._emit('info', 'Stop requested. Will halt after current iteration finishes...')

    def log(self, message) -> None:
        """Log a message to the console and emit it to the UI."""
        self._emit('info', message)

    def _train_network(self) -> dict:
        """
        Train the network on samples from the replay buffer.
        
        Returns dict with training metrics (losses).
        """
        train_start = time.time()
        self.network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        # Calculate steps per epoch based on buffer size.
        # We cap it at 200 steps to ensure training doesn't take forever, 
        # but is large enough to actually learn (previously it was taking exactly 1 step per epoch!)
        steps_per_epoch = max(1, len(self.replay_buffer) // self.config.training.batch_size)
        steps_per_epoch = min(steps_per_epoch, 200)
        num_epochs = self.config.training.num_epochs_per_iteration
        total_train_steps = steps_per_epoch * num_epochs
        
        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                if self._force_stop_requested:
                    return {}
                states, target_policies, target_values = self.replay_buffer.sample(
                    self.config.training.batch_size
                )
                
                states = states.to(self.device)
                target_policies = target_policies.to(self.device)
                target_values = target_values.to(self.device)
                
                # Forward pass
                pred_policies, pred_values = self.network(states)
                pred_values = pred_values.squeeze(-1)
                
                # Policy loss: cross-entropy between MCTS policy and predicted policy
                log_probs = torch.log_softmax(pred_policies, dim=1)
                policy_loss = -torch.mean(torch.sum(target_policies * log_probs, dim=1))
                
                # Value loss: MSE between game outcome and predicted value
                value_loss = torch.mean((target_values - pred_values) ** 2)
                
                # Combined loss
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

                completed_steps = epoch * steps_per_epoch + step + 1
                if completed_steps % 5 == 0 or completed_steps == total_train_steps:
                    self._set_stage(
                        'training', 'Neural Network Training', 2,
                        completed_items=completed_steps, total_items=total_train_steps,
                        active_games=[], num_workers=0,
                        detail=f'Epoch {epoch+1}/{num_epochs} · Step {step+1}/{steps_per_epoch} · Policy: {policy_loss.item():.4f} · Value: {value_loss.item():.4f}'
                    )
                    self._emit('training_step',
                               f'Training: step {completed_steps}/{total_train_steps} (Loss {loss.item():.4f})',
                               data={'current_stage': self.current_stage})
        
        # Step the learning rate scheduler — but only through ONE cosine descent.
        #
        # CosineAnnealingLR is periodic: lr = eta_min + (base - eta_min) *
        # (1 + cos(pi * last_epoch / T_max)) / 2. It reaches eta_min at
        # last_epoch = T_max and then CLIMBS BACK, returning to the full base rate
        # at 2 * T_max. With lr_decay_steps = 100 that means a run annealing to
        # ~0 by iteration 100 is back at 0.002 by iteration 200 — a warm-restart
        # cycle nobody asked for, in the second half of every long run.
        #
        # This was invisible until now: the scheduler's state was not checkpointed,
        # so restarts kept resetting last_epoch and the schedule never advanced far
        # enough to turn around. Persisting it correctly exposed the periodicity.
        # Holding at the floor once the descent completes is what "anneal the
        # learning rate" is meant to do; a genuine warm restart should be an
        # explicit choice, not a side effect of T_max being shorter than the run.
        if self.scheduler.last_epoch < self.config.training.lr_decay_steps:
            self.scheduler.step()


        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        
        train_res = {
            'policy_loss': round(avg_policy_loss, 4),
            'value_loss': round(avg_value_loss, 4),
            'total_loss': round(avg_policy_loss + avg_value_loss, 4),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'nn_train_seconds': round(time.time() - train_start, 2),
        }
        self._set_stage(
            'training', 'Neural Network Training Done', 2,
            completed_items=total_train_steps, total_items=total_train_steps,
            active_games=[], num_workers=0,
            detail=f"Trained {total_train_steps} steps · Loss: {train_res['total_loss']:.4f} (P: {train_res['policy_loss']:.4f}, V: {train_res['value_loss']:.4f})"
        )
        return train_res
    
    def _on_game_complete(
        self,
        game_num: int,
        total: int,
        record: dict = None,
        active_games: list = None,
        num_workers: int = None,
    ) -> None:
        """Callback for self-play game progress and completions."""
        active_list = active_games or []
        w_count = num_workers or self.config.training.num_parallel_workers

        if record is None:
            # Initial batch launch notification
            self._set_stage(
                'self_play', 'Self-Play Data Generation', 1,
                completed_items=0, total_items=total,
                active_games=active_list, num_workers=w_count,
                detail=f'Starting self-play ({total} games)'
            )
            self._emit('game_progress', f'Self-play launched with {len(active_list)} parallel workers', data={
                'current_stage': self.current_stage,
                'active_games': active_list,
                'num_workers': w_count,
            })
            return

        if record.get('failed'):
            # The worker died; there is no game here to tally. Progress still
            # advances (the slot is done) but nothing is counted as played.
            self._set_stage(
                'self_play', 'Self-Play Data Generation', 1,
                completed_items=game_num, total_items=total,
                active_games=active_list, num_workers=w_count,
                detail=f'Generated {game_num}/{total} games (1 worker failed)'
            )
            return

        # Tally this iteration's label entropy. The collapse guard's
        # `target_entropy` reads the whole replay buffer, which at 50k samples
        # is ~30 iterations deep and therefore lags a config change by weeks of
        # training. This one is fresh: it describes the labels produced RIGHT
        # NOW, so flipping policy_target_temperature shows up on the next
        # iteration instead of slowly fading in as the FIFO turns over.
        te = record.get('target_entropy')
        if te is not None:
            self._label_stats['entropy_sum'] += float(te)
            self._label_stats['support_sum'] += float(record.get('target_support') or 0.0)
            self._label_stats['max_prob_sum'] += float(record.get('target_max_prob') or 0.0)
            self._label_stats['games'] += 1

        # Tally passes per colour for the collapse guard. A colour that starts
        # passing heavily is handing away points under area scoring.
        for m in record.get('moves', []):
            colour = m.get('color')
            if colour in self._pass_stats:
                self._pass_stats[colour][1] += 1
                if m.get('move', [0, 0])[0] < 0:
                    self._pass_stats[colour][0] += 1

        # Tally mercy-rule outcomes. `would_resign_color` is only ever set on
        # playout games, which are the ones that can tell us whether the rule
        # would have been wrong.
        rs = self._resign_stats
        rs['games'] += 1
        rs['moves_total'] += record.get('num_moves', 0)
        if record.get('resigned'):
            rs['resigned'] += 1
        if record.get('resign_playout'):
            rs['playout_games'] += 1
            if record.get('would_resign_color') is not None:
                rs['would_resign'] += 1
                if record.get('false_resign'):
                    rs['false_resigns'] += 1

        self._set_stage(
            'self_play', 'Self-Play Data Generation', 1,
            completed_items=game_num, total_items=total,
            active_games=active_list, num_workers=w_count,
            detail=f'Generated {game_num}/{total} games'
        )

        self._emit('game_complete', f'Game {game_num}/{total}', data={
            'game_num': game_num,
            'total': total,
            'winner': record.get('winner', 0),
            'num_moves': record.get('num_moves', 0),
            'elapsed': record.get('elapsed_seconds', 0),
            'total_games': self.total_games + game_num,
            'buffer_size': len(self.replay_buffer),
            'active_games': active_list,
            'num_workers': w_count,
            'current_stage': self.current_stage,
        })
    
    def _save_metrics_log(self, metrics: dict) -> None:
        """Append metrics to the JSON lines log file."""
        log_path = os.path.join(self.config.paths.logs_dir, 'training_log.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def _emit_reflection(self) -> None:
        """
        Emit a reflection milestone event.
        
        This is a summary emitted every N games that tracks how
        the model is improving. Shown inline in the training log.
        """
        if len(self.metrics_history) < 2:
            return
        
        recent = self.metrics_history[-1]
        oldest = self.metrics_history[0]
        
        elo_gain = recent['elo'] - oldest['elo']
        
        report = {
            'total_games': self.total_games,
            'iterations': self.iteration,
            'starting_elo': oldest['elo'],
            'current_elo': recent['elo'],
            'elo_gain': round(elo_gain, 1),
            'starting_rank': oldest.get('kyu_rank', '?'),
            'current_rank': recent.get('kyu_rank', '?'),
            'latest_gate_win_rate': recent.get('gate_win_rate'),
            'latest_policy_loss': recent.get('policy_loss', 0),
            'latest_value_loss': recent.get('value_loss', 0),
        }
        
        # Check for rank milestones (only when rank genuinely changes and improves from the previous iteration/milestone)
        milestone = None
        current_rank = recent.get('kyu_rank')
        last_rank = getattr(self, '_last_milestone_rank', None)
        if last_rank is None and len(self.metrics_history) >= 2:
            last_rank = self.metrics_history[-2].get('kyu_rank')

        if last_rank and current_rank and current_rank != last_rank:
            # Verify that Elo actually increased
            if recent.get('elo', 0) > oldest.get('elo', 0):
                milestone = f"Rank improved from {last_rank} to {current_rank}!"
                report['milestone'] = milestone
            self._last_milestone_rank = current_rank
        elif not hasattr(self, '_last_milestone_rank') and current_rank:
            self._last_milestone_rank = current_rank
        
        msg = f"📊 Progress: {self.total_games} games | Elo {recent['elo']:.0f} ({recent.get('kyu_rank', '?')}) | Gain: {elo_gain:+.0f}"
        self._emit('reflection', msg, data=report)
    
    def get_status(self) -> dict:
        """Get current training status for the web API."""
        return {
            'is_running': self.is_running,
            'stop_requested': self._stop_requested,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'elo': self.elo,
            'gate_elo': round(self.gate_elo, 1),
            'kyu_rank': elo_to_rank(self.elo),
            'buffer_size': len(self.replay_buffer),
            'device': self.device,
            'current_stage': self.current_stage,
            'recent_logs': list(self.recent_logs),
            'metrics_history': self.metrics_history[-50:],  # Last 50 entries
        }

    
    def get_metrics_history(self) -> List[dict]:
        """
        Get full metrics history for charts.

        Reads the on-disk log exactly once per trainer and merges it with
        whatever this run has produced, keyed by iteration.

        The previous version only touched disk `if not self.metrics_history`.
        If the training loop appended even one row before a client first asked
        for history, that guard was already false and the entire on-disk
        history was silently dropped — the charts would show a single point.
        """
        if not self._history_loaded:
            self._history_loaded = True
            log_path = os.path.join(self.config.paths.logs_dir, 'training_log.jsonl')
            disk_rows = []
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            disk_rows.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue

            # Merge disk + in-memory, newest wins per iteration, then order.
            merged = {}
            for row in disk_rows + self.metrics_history:
                merged[row.get('iteration')] = row
            self.metrics_history = [
                merged[k] for k in sorted(merged, key=lambda x: (x is None, x))
            ]

        return self.metrics_history
