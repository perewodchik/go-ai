"""
trainer.py — Main training loop for the Go AI.

Implements the AlphaZero training pipeline:
1. Self-play: generate games with current network + MCTS
2. Train: update network on the collected data
3. Evaluate: compare new network vs previous version
4. Save weights: persist model state
5. Repeat

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
from ai.evaluator import (
    evaluate_against_random,
    evaluate_against_checkpoint,
    compute_elo_update,
)
from ai.checkpoint import save_weights, load_weights


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
            num_res_blocks=config.network.num_res_blocks,
            num_filters=config.network.num_filters,
            value_head_hidden=config.network.value_head_hidden,
        )
        
        # Dedicated eval network for fast CPU inference (self-play & human games)
        self.eval_network = GoNetwork(
            board_size=config.board.size,
            num_input_planes=config.network.num_input_planes,
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
        self.elo = config.training.elo_anchor  # Start at random bot level
        self.best_elo = self.elo
        
        # Metrics history (for charts). `_history_loaded` tracks whether the
        # on-disk log has been merged in yet — see get_metrics_history().
        self.metrics_history: List[dict] = []
        self._history_loaded = False
        self.recent_logs = deque(maxlen=100)

        # Promotion-gate bookkeeping
        self.gate_rejections = 0          # Consecutive candidates refused
        self.last_gate_win_rate = None    # Candidate's score in the last gate match
        # Per-iteration pass tallies, refreshed each self-play phase and read
        # by the collapse guard. {colour: [passes, total_moves]}
        self._pass_stats = {1: [0, 0], 2: [0, 0]}
        
        # Load weights if available
        self._try_load_weights()
    
    def _try_load_weights(self) -> None:
        """Try to resume from saved weights."""
        weights_path = self.config.paths.weights_path
        if os.path.isfile(weights_path):
            try:
                meta = load_weights(weights_path, self.network, self.optimizer, self.device)
                if meta:
                    self.iteration = meta.get('iteration', 0)
                    self.elo = meta.get('elo', 500)
                    self.total_games = meta.get('total_games', 0)
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
        
        self._emit('training_started', 'Training started')
        
        try:
            while not self._stop_requested and not self._force_stop_requested:
                if max_iterations and self.iteration >= max_iterations:
                    break
                
                iteration_start = time.time()
                self.iteration += 1
                
                self._emit('iteration_start', f'Starting iteration {self.iteration}')
                
                # --- Phase 1: Self-play ---
                self._emit('self_play_start', f'Self-play: generating {self.config.training.num_self_play_games} games')

                # Reset per-iteration pass tallies for the collapse guard.
                self._pass_stats = {1: [0, 0], 2: [0, 0]}
                
                # Effective sim count scales with board area so policy targets
                # stay sharp on larger boards (see config.simulations_for_board).
                effective_sims = simulations_for_board(
                    self.config.mcts, self.config.board.size)

                # Use eval_network on CPU for self-play (much faster for batch_size=1)
                samples = run_self_play_batch(
                    network=self.eval_network,
                    num_games=self.config.training.num_self_play_games,
                    board_size=self.config.board.size,
                    komi=self.config.board.komi,
                    num_simulations=effective_sims,
                    c_puct=self.config.mcts.c_puct,
                    temperature_threshold=self.config.mcts.temperature_threshold,
                    device="cpu",
                    game_store_every_n=self.config.training.game_store_every_n,
                    games_dir=self.config.paths.games_dir,
                    iteration=self.iteration,
                    progress_callback=self._on_game_complete,
                    fpu_reduction=self.config.mcts.fpu_reduction,
                    stop_checker=self._check_force_stop,
                )
                
                if self._force_stop_requested:
                    break

                self.total_games += self.config.training.num_self_play_games
                self.replay_buffer.add(samples)
                
                self._emit('self_play_done',
                           f'Self-play done: {len(samples)} samples, buffer size: {len(self.replay_buffer)}')
                
                # --- Phase 2: Training ---
                if len(self.replay_buffer) >= self.config.training.batch_size:
                    self._emit('training_start', 'Training network...')

                    train_metrics = self._train_network()

                    if self._force_stop_requested:
                        break

                    # --- Promotion gate ---
                    # The freshly trained candidate only replaces the champion
                    # (the network that generates self-play games) if it beats
                    # it head-to-head. Promoting unconditionally is what let a
                    # degenerate network take over the self-play loop with
                    # nothing able to detect or reject it.
                    promoted, gate_wr = self._run_promotion_gate()
                    train_metrics['gate_win_rate'] = gate_wr
                    train_metrics['gate_promoted'] = promoted

                    if self._force_stop_requested:
                        break

                    self._emit('training_done', 'Training done', data=train_metrics)
                
                if self._force_stop_requested:
                    break

                # --- Phase 3: Evaluation ---
                self._emit('eval_start', 'Evaluating against random bot...')
                
                win_rate = evaluate_against_random(
                    network=self.eval_network,
                    board_size=self.config.board.size,
                    komi=self.config.board.komi,
                    num_simulations=effective_sims,
                    num_games=self.config.training.eval_games,
                    device="cpu",
                    iteration=self.iteration,
                    games_dir=self.config.paths.games_dir,
                    stop_checker=self._check_force_stop,
                )

                if self._force_stop_requested:
                    break
                
                # Update Elo based on win rate against random bot
                old_elo = self.elo
                self.elo = compute_elo_update(
                    self.elo, self.config.training.elo_anchor, win_rate
                )
                
                kyu = elo_to_rank(self.elo)
                
                eval_data = {
                    'win_rate_vs_random': win_rate,
                    'elo_delta': self.elo - old_elo,
                }
                self._emit('eval_done', f'Win rate vs random: {win_rate:.1%}, Elo: {self.elo:.0f} ({kyu})',
                           data=eval_data)
                
                # --- Phase 4: Save weights (EVERY iteration to prevent data loss) ---
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
                    'win_rate_vs_random': win_rate,
                    'buffer_size': len(self.replay_buffer),
                    'elapsed_seconds': round(elapsed, 1),
                    'timestamp': datetime.now().isoformat(),
                    'gate_win_rate': self.last_gate_win_rate,
                    'gate_rejections': self.gate_rejections,
                    **diagnostics,
                }
                if 'train_metrics' in dir() and train_metrics:
                    metrics.update(train_metrics)
                
                self.metrics_history.append(metrics)
                self._save_metrics_log(metrics)
                
                # --- Reflection milestone ---
                if self.total_games % self.config.training.reflection_interval_games < \
                   self.config.training.num_self_play_games:
                    self._emit_reflection()
                
                self._emit('iteration_done',
                           f'Iteration {self.iteration} done in {elapsed:.1f}s | '
                           f'Elo: {self.elo:.0f} ({kyu})',
                           data=metrics)
                
                # Update model state in config.json
                self._update_model_state()
        
        except Exception as e:
            self._emit('error', f'Training error: {str(e)}')
            raise
        finally:
            self.is_running = False
            if self._force_stop_requested:
                # Reload clean weights from disk from last saved completed iteration
                self._try_load_weights()
                self._update_model_state()
                self._emit('training_stopped', f'⚡ Force stopped! Restored weights from iteration {self.iteration} (Elo {self.elo:.0f}).')
                self._force_stop_requested = False
            else:
                # Always save weights on normal stop
                self._save_weights()
                self._emit('training_stopped', 'Training stopped')
    
    def _cpu_copy(self, source: GoNetwork) -> GoNetwork:
        """Build a CPU-resident clone of `source` for head-to-head evaluation."""
        clone = GoNetwork(
            board_size=self.config.board.size,
            num_input_planes=self.config.network.num_input_planes,
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

        if not cfg.gate_enabled:
            # Legacy behaviour: accept every update unconditionally.
            self.eval_network.load_state_dict(self.network.state_dict())
            return True, None

        self._emit('gate_start',
                   f'Promotion gate: candidate vs champion ({cfg.gate_games} games)')

        candidate = self._cpu_copy(self.network)
        champion = self._cpu_copy(self.eval_network)

        try:
            win_rate = evaluate_against_checkpoint(
                current_network=candidate,
                opponent_network=champion,
                board_size=self.config.board.size,
                komi=self.config.board.komi,
                num_simulations=cfg.gate_simulations,
                num_games=cfg.gate_games,
                device="cpu",
            )
        except Exception as e:
            # Never let a gate failure take down training — fall back to
            # keeping the existing champion, which is the safe direction.
            self._emit('warning', f'Promotion gate failed ({e}); keeping champion')
            return False, None

        self.last_gate_win_rate = win_rate

        if win_rate >= cfg.gate_threshold:
            self.eval_network.load_state_dict(
                {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
            )
            self.gate_rejections = 0
            self._emit('gate_promoted',
                       f'✅ Candidate promoted (beat champion {win_rate:.0%} '
                       f'≥ {cfg.gate_threshold:.0%})',
                       data={'gate_win_rate': win_rate})
            return True, win_rate

        # Rejected: the champion keeps generating self-play data.
        self.gate_rejections += 1
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
            self._emit('warning',
                       f'⚠️ {cfg.gate_stall_warning} consecutive rejections — '
                       f'training network reset to the champion to break the stall.')

        return False, win_rate

    def _collapse_diagnostics(self) -> dict:
        """
        Tripwires for value-head collapse.

        When one colour wins nearly every game, the value head can drive its
        loss to ~0 by reading only the turn-colour plane (`v = +1 if Black to
        move else -1`), ignoring the board entirely. MCTS then sees a constant
        Q for every child and stops evaluating — 800 simulations of nothing.

        The spread must be measured WITHIN each colour: across mixed positions
        a collapsed head still looks bimodal (±1) and would appear healthy.
        """
        cfg = self.config.training
        out = {
            'value_std_black': None,
            'value_std_white': None,
            'pass_rate_black': None,
            'pass_rate_white': None,
            'collapse_warning': None,
        }

        # --- Value-head spread, split by side to move ---
        # Needs the turn-colour plane (index 9) to separate the sides.
        if len(self.replay_buffer) >= 16 and self.config.network.num_input_planes > 9:
            n = min(cfg.collapse_probe_positions, len(self.replay_buffer))
            states, _, _ = self.replay_buffer.sample(n, augment=False)
            with torch.no_grad():
                _, values = self.eval_network.predict_batch(states, "cpu")
            # Plane 9 is all-ones exactly when Black is to move.
            is_black = states[:, 9, 0, 0] > 0.5
            for key, mask in (('value_std_black', is_black),
                              ('value_std_white', ~is_black)):
                sel = values[mask]
                if sel.numel() >= 2:
                    out[key] = round(float(sel.std().item()), 5)

        # --- Pass rates from this iteration's self-play ---
        for colour, key in ((1, 'pass_rate_black'), (2, 'pass_rate_white')):
            passes, total = self._pass_stats[colour]
            if total > 0:
                out[key] = round(passes / total, 4)

        if not cfg.collapse_guard_enabled:
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
            )
            self._emit('info', f'Weights saved (iteration {self.iteration})')
        except Exception as e:
            self._emit('warning', f'Failed to save weights: {e}')

    def save_weights_now(self) -> str:
        """Force an immediate weights save. Called from API."""
        save_weights(
            model=self.network,
            optimizer=self.optimizer,
            iteration=self.iteration,
            elo=self.elo,
            kyu_rank=elo_to_rank(self.elo),
            total_games=self.total_games,
            weights_path=self.config.paths.weights_path,
        )
        self._emit('info', f'Weights saved manually (iteration {self.iteration})')
        return self.config.paths.weights_path

    def _update_model_state(self) -> None:
        """Update the model's config.json with current training state."""
        if self.model_id:
            try:
                from model_manager import ModelManager
                mgr = ModelManager()
                mgr.update_model_state(
                    self.model_id, self.elo, elo_to_rank(self.elo),
                    self.iteration, self.total_games
                )
            except Exception:
                pass  # Non-critical
    
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
        self.network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        # Calculate steps per epoch based on buffer size.
        # We cap it at 200 steps to ensure training doesn't take forever, 
        # but is large enough to actually learn (previously it was taking exactly 1 step per epoch!)
        steps_per_epoch = max(1, len(self.replay_buffer) // self.config.training.batch_size)
        steps_per_epoch = min(steps_per_epoch, 200)
        
        for epoch in range(self.config.training.num_epochs_per_iteration):
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
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        avg_policy_loss = total_policy_loss / max(num_batches, 1)
        avg_value_loss = total_value_loss / max(num_batches, 1)
        
        return {
            'policy_loss': round(avg_policy_loss, 4),
            'value_loss': round(avg_value_loss, 4),
            'total_loss': round(avg_policy_loss + avg_value_loss, 4),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
    
    def _on_game_complete(self, game_num: int, total: int, record: dict) -> None:
        """Callback for each completed self-play game."""
        # Tally passes per colour for the collapse guard. A colour that starts
        # passing heavily is handing away points under area scoring.
        for m in record.get('moves', []):
            colour = m.get('color')
            if colour in self._pass_stats:
                self._pass_stats[colour][1] += 1
                if m.get('move', [0, 0])[0] < 0:
                    self._pass_stats[colour][0] += 1

        self._emit('game_complete', f'Game {game_num}/{total}', data={
            'game_num': game_num,
            'total': total,
            'winner': record.get('winner', 0),
            'num_moves': record.get('num_moves', 0),
            'elapsed': record.get('elapsed_seconds', 0),
            'total_games': self.total_games + game_num,
            'buffer_size': len(self.replay_buffer),
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
            'latest_win_rate': recent.get('win_rate_vs_random', 0),
            'latest_policy_loss': recent.get('policy_loss', 0),
            'latest_value_loss': recent.get('value_loss', 0),
        }
        
        # Check for rank milestones
        milestone = None
        if oldest.get('kyu_rank') != recent.get('kyu_rank'):
            milestone = f"Rank improved from {oldest.get('kyu_rank')} to {recent.get('kyu_rank')}!"
            report['milestone'] = milestone
        
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
            'kyu_rank': elo_to_rank(self.elo),
            'buffer_size': len(self.replay_buffer),
            'device': self.device,
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
