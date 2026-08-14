"""
network.py — Neural network for Go (policy + value heads).

Architecture: AlphaZero-style residual CNN.
- Input: multi-plane board encoding (from GameState.encode_for_nn)
- Body: stack of residual blocks with batch normalization
- Policy head: predicts probability of each move (board_size² + 1 for pass)
- Value head: predicts game outcome from current player's perspective (-1 to +1)

CPU/MPS OPTIMIZATIONS:
- Small architecture (4 res blocks, 64 filters) fits M2 MacBook Air
- Supports MPS (Apple Silicon GPU) when available for ~3-5x speedup
- Can export to ONNX for even faster inference during self-play
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Standard residual block: Conv → BN → ReLU → Conv → BN → skip connection → ReLU.
    
    The skip connection is what makes ResNets trainable at depth — gradients
    flow through the shortcut path, avoiding vanishing gradient problems.
    """
    
    def __init__(self, num_filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Skip connection
        out = F.relu(out)
        return out


class GoNetwork(nn.Module):
    """
    AlphaZero-style neural network for Go.
    
    Input shape:  (batch, num_input_planes, board_size, board_size)
    Output:       (policy_logits, value)
        policy_logits: (batch, board_size² + 1)  — log-probabilities for each move + pass
        value:         (batch, 1)                — predicted outcome ∈ [-1, +1]
    
    The network learns two things simultaneously:
    1. POLICY: "What move should I play?" → used to guide MCTS tree expansion
    2. VALUE: "Who's winning from this position?" → used to evaluate leaf nodes in MCTS
    
    Args:
        board_size: Size of the Go board (e.g. 9).
        num_input_planes: Number of input feature planes (default 10, matching
            GameState.encode_for_nn: 2 stone colors, liberty buckets ×2 colors,
            ko point, and turn color).
        num_res_blocks: Number of residual blocks in the body.
        num_filters: Number of convolutional filters per layer.
        value_head_hidden: Hidden layer size in the value head.
    """
    
    def __init__(self, board_size: int = 9, num_input_planes: int = 10,
                 num_res_blocks: int = 4, num_filters: int = 64,
                 value_head_hidden: int = 64):
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size + 1  # All positions + pass
        
        # --- Input block: project input planes to num_filters channels ---
        self.input_conv = nn.Conv2d(num_input_planes, num_filters, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_filters)
        
        # --- Residual body ---
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # --- Policy head ---
        # Conv 1×1 to reduce channels, then flatten and fully-connected
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # --- Value head ---
        # Conv 1×1, flatten, two FC layers, tanh output
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, value_head_hidden)
        self.value_fc2 = nn.Linear(value_head_hidden, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, num_input_planes, board_size, board_size)
        
        Returns:
            (policy_logits, value):
                policy_logits: (batch, action_size) — raw logits (apply softmax for probs)
                value: (batch, 1) — game outcome prediction in [-1, +1]
        """
        # Input block
        out = F.relu(self.input_bn(self.input_conv(x)))
        
        # Residual body
        for block in self.res_blocks:
            out = block(out)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)  # Raw logits
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Squash to [-1, +1]
        
        return policy, value
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total number of parameters in this network."""
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def arch_signature(self) -> dict:
        """
        Architecture-defining hyperparameters. Used to tag checkpoints so a
        saved weights.pt is never loaded into a mismatched network (which would
        either crash with a cryptic shape error or silently corrupt training).
        """
        return {
            "board_size": self.board_size,
            "num_input_planes": self.input_conv.in_channels,
            "num_res_blocks": len(self.res_blocks),
            "num_filters": self.input_conv.out_channels,
            "value_head_hidden": self.value_fc1.out_features,
        }

    @torch.no_grad()
    def predict(self, state_tensor: torch.Tensor, device: str = "cpu") -> Tuple[torch.Tensor, float]:
        """
        Single-state prediction for MCTS (no gradient tracking).
        
        Args:
            state_tensor: Shape (num_planes, board_size, board_size) — single state.
            device: Device to run on ("cpu", "mps", "cuda").
        
        Returns:
            (policy_probs, value):
                policy_probs: (action_size,) — probability for each action
                value: float — predicted game outcome
        """
        self.eval()
        x = state_tensor.unsqueeze(0).to(device)  # Add batch dim
        policy_logits, value = self(x)
        
        # Apply softmax to get probabilities
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0).cpu()
        value_scalar = value.item()
        
        return policy_probs, value_scalar
    
    @torch.no_grad()
    def predict_batch(self, state_tensors: torch.Tensor, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch prediction for parallel MCTS evaluations.
        
        Args:
            state_tensors: Shape (batch, num_planes, board_size, board_size)
            device: Device string.
        
        Returns:
            (policy_probs, values):
                policy_probs: (batch, action_size) — probabilities
                values: (batch,) — predicted outcomes
        """
        self.eval()
        x = state_tensors.to(device)
        policy_logits, values = self(x)
        policy_probs = F.softmax(policy_logits, dim=1).cpu()
        values = values.squeeze(-1).cpu()
        return policy_probs, values
