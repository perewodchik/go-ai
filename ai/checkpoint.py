"""
checkpoint.py — Save and load model weights.

Uses a single `weights.pt` file per model that contains everything
needed to resume training:
  - Model weights
  - Optimizer state
  - Training iteration number
  - Elo rating and kyu rank

This replaces the old versioned checkpoint system. The file is
overwritten in-place each time, keeping things simple.
"""

import os
import torch
from typing import Optional, Dict
from datetime import datetime


def save_weights(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    elo: float,
    kyu_rank: str,
    total_games: int,
    weights_path: str,
    champion_state_dict: Optional[Dict] = None,
    gate_elo: Optional[float] = None,
) -> str:
    """
    Save training state to a single weights file.

    Args:
        model: The neural network.
        optimizer: The optimizer.
        iteration: Current training iteration number.
        elo: Current Elo rating.
        kyu_rank: Current estimated rank string.
        total_games: Total self-play games played.
        weights_path: Full path to the weights.pt file.
        champion_state_dict: Weights of the gated champion (the network that
            generates self-play games). Stored alongside the training network
            so a restart resumes from the champion rather than from an
            un-promoted candidate. Optional — older files simply lack it.
        gate_elo: Rating on the self-referential promotion-gate ladder. Kept
            separate from `elo` (which is measured against the random-bot
            anchor) because the two stop agreeing as soon as the anchor
            saturates. Optional — older files simply lack it.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # Tag the checkpoint with the architecture it was trained with, so a later
    # load into a mismatched network is caught with a clear message instead of
    # a cryptic tensor-shape error (or silent corruption).
    arch = model.arch_signature() if hasattr(model, "arch_signature") else None

    state = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'elo': elo,
        'kyu_rank': kyu_rank,
        'total_games': total_games,
        'gate_elo': gate_elo,
        'arch': arch,
        'timestamp': datetime.now().isoformat(),
    }

    if champion_state_dict is not None:
        state['champion_state_dict'] = {
            k: v.detach().cpu() for k, v in champion_state_dict.items()
        }


    # Write to a temporary file first, then rename for atomicity
    tmp_path = weights_path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, weights_path)
    
    return weights_path


def load_weights(
    weights_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict:
    """
    Load weights and restore model (and optionally optimizer) state.
    
    Args:
        weights_path: Path to the weights.pt file.
        model: The neural network to load weights into.
        optimizer: If provided, restore optimizer state too.
        device: Device to map tensors to.
    
    Returns:
        Dict with metadata (iteration, elo, etc.).
    """
    if not os.path.isfile(weights_path):
        return {}
    
    state = torch.load(weights_path, map_location=device, weights_only=False)

    # Guard against loading weights into a network with a different shape.
    # Only enforced when the checkpoint carries an arch tag (older files don't);
    # raising here lets the trainer warn and start fresh rather than crash.
    saved_arch = state.get('arch')
    if saved_arch and hasattr(model, "arch_signature"):
        current_arch = model.arch_signature()
        if saved_arch != current_arch:
            raise ValueError(
                "Checkpoint architecture does not match the current network. "
                f"Saved={saved_arch}, current={current_arch}. "
                "Refusing to load to avoid corrupting training."
            )

    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in state:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return {
        'iteration': state.get('iteration', 0),
        'elo': state.get('elo', 500),
        'kyu_rank': state.get('kyu_rank', '30k'),
        'total_games': state.get('total_games', 0),
        # None for checkpoints written before the gate ladder existed; the
        # trainer then seeds it from the anchor Elo.
        'gate_elo': state.get('gate_elo'),
        'arch': saved_arch,
        'timestamp': state.get('timestamp', ''),
        # None for checkpoints saved before gating existed; the caller then
        # falls back to using the training weights as the champion.
        'champion_state_dict': state.get('champion_state_dict'),
    }
