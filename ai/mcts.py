"""
mcts.py — Monte Carlo Tree Search (AlphaZero variant).

This is the "thinking" engine. Given a game position and a neural network,
MCTS explores possible move sequences to find the best action.

HOW IT WORKS (simplified):
1. SELECT: Walk down the tree picking the most promising branch (PUCT formula).
2. EXPAND: At a leaf node, ask the neural network for a policy + value estimate.
3. BACKUP: Propagate the value estimate back up the tree.
4. Repeat for N simulations.
5. Pick the move with the most visits as the actual move.

PUCT FORMULA (controls exploration vs exploitation):
    Q(s,a) + c_puct * P(s,a) * sqrt(sum_visits) / (1 + N(s,a))
    
    - Q(s,a): average value of taking action a from state s
    - P(s,a): prior probability from the neural network
    - N(s,a): visit count for this action
    - c_puct: exploration constant (higher = more exploration)

The neural network GUIDES the search (which moves to explore first),
while MCTS CORRECTS the network (spending more time on promising lines).
"""

import math
import numpy as np
import torch
from typing import Optional, Dict, Tuple, List
from game.game_state import GameState, MOVE_PASS
from game.board import BLACK, WHITE, opponent


class MCTSNode:
    """
    A node in the MCTS search tree.
    
    Each node represents a game state and tracks statistics for each
    child action (move). We DON'T store child states explicitly — they're
    created on-demand during expansion to save memory.
    
    Attributes:
        state: The GameState at this node.
        parent: Parent MCTSNode (None for root).
        parent_action: The action that led from parent to this node.
        children: Dict mapping action → child MCTSNode.
        visit_count: How many times this node has been visited.
        value_sum: Sum of all backpropagated values through this node.
        prior: Prior probability from the neural network (P(s,a) for this action).
        is_expanded: Whether we've run the neural network on this state.
    """
    
    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None,
                 parent_action: Optional[Tuple[int, int]] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior
        
        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    @property
    def q_value(self) -> float:
        """
        Mean value of this node = value_sum / visit_count.
        Returns 0 if unvisited (optimistic initialization).
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float, parent_visits: int,
                  parent_q: float = 0.0, fpu_reduction: float = 0.0) -> float:
        """
        Upper Confidence Bound score (PUCT variant).

        Balances exploitation (high Q) with exploration (high prior, low visits).
        The neural network prior P(s,a) ensures we explore "smart" moves first,
        while the visit-based term ensures we don't ignore any move forever.

        First Play Urgency (FPU): an unvisited child has no Q of its own. Instead
        of the optimistic 0.0 (which makes MCTS try every move once before it will
        deepen any of them), we estimate its value from the parent's perspective
        and subtract `fpu_reduction`. `parent_q` is the parent's mean value from
        the parent-node's own parent's perspective, so the value from the mover's
        perspective at this node is `-parent_q`.
        """
        if self.visit_count == 0:
            q = -parent_q - fpu_reduction
        else:
            q = self.q_value
        # Exploration bonus: high when visit_count is low relative to parent
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + exploration

    def best_child(self, c_puct: float, fpu_reduction: float = 0.0) -> 'MCTSNode':
        """Select child with highest UCB score."""
        parent_q = self.q_value
        return max(
            self.children.values(),
            key=lambda child: child.ucb_score(c_puct, self.visit_count,
                                              parent_q, fpu_reduction)
        )
    
    def most_visited_child(self) -> Tuple[Tuple[int, int], 'MCTSNode']:
        """Select child with most visits (used for final move selection)."""
        best_action = max(self.children, key=lambda a: self.children[a].visit_count)
        return best_action, self.children[best_action]


class MCTS:
    """
    Monte Carlo Tree Search engine.
    
    Args:
        network: The GoNetwork for position evaluation.
        config: MCTSConfig with simulation count and parameters.
        device: Torch device string ("cpu", "mps", "cuda").
    """
    
    def __init__(self, network, num_simulations: int = 200,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, device: str = "cpu",
                 fpu_reduction: float = 0.35,
                 restrict_eye_fill: bool = False):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device
        self.fpu_reduction = fpu_reduction
        # Optional playing restriction: hide moves that fill one of the mover's
        # own two eyes. Stamped onto the root copy in search(), from where every
        # node in the tree inherits it via GameState.copy() — so a restricted
        # move is never expanded, never given a prior, never visited, and never
        # appears in the policy target. See game/eyes.py.
        self.restrict_eye_fill = restrict_eye_fill
        # Root evaluation from the most recent search(), expressed from the
        # perspective of the player to move at the root (∈ [-1, +1]). Used for
        # win-rate tracking. Updated by every search() call.
        self.root_value = 0.0

    def search(self, state: GameState, temperature: float = 1.0,
               add_noise: bool = True, allow_pass: bool = True) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Run MCTS from the given state and return the best action.
        
        Args:
            state: Current game state.
            temperature: Controls randomness of move selection.
                - temperature=1.0: proportional to visit counts (exploratory)
                - temperature→0: pick the most-visited move (greedy)
            add_noise: Whether to add Dirichlet noise at root (for training).
            allow_pass: Whether MOVE_PASS is allowed as a legal action.
        
        Returns:
            (action, policy_vector):
                action: (row, col) or MOVE_PASS
                policy_vector: Visit count distribution over all actions
                              (used as training target for the policy head).
        """
        root_state = state.copy()
        # Applied to our own copy rather than to the caller's state: the same
        # GameState object is shared with opponents (the random bot in the Elo
        # eval, the other network in the gate match, the human in a web game),
        # and the restriction is a property of THIS searcher, not of the game.
        if self.restrict_eye_fill:
            root_state.restrict_eye_fill = True
        root = MCTSNode(root_state)

        # Expand root node
        self._expand(root, allow_pass=allow_pass)
        
        # Add Dirichlet noise to root priors for exploration during training
        if add_noise and root.children:
            self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            
            # SELECT: walk down tree picking best child
            while node.is_expanded and node.children and not node.state.is_over:
                node = node.best_child(self.c_puct, self.fpu_reduction)
            
            # If game is over at this node, use actual result
            if node.state.is_over:
                # Value from perspective of the node's current player
                value = self._terminal_value(node.state)
            else:
                # EXPAND: evaluate with neural network
                value = self._expand(node, allow_pass=allow_pass)
            
            # BACKUP: propagate value up the tree
            self._backup(node, value)

        # Capture the root evaluation for win-rate tracking. root.q_value is
        # accumulated from the opponent's perspective (see _backup — value_sum
        # is read by a node's parent, i.e. the other side), so negate it to get
        # the value from the root mover's own perspective.
        self.root_value = -root.q_value if root.visit_count > 0 else 0.0

        # Build policy vector from visit counts with temperature scaling
        # (AlphaZero formulation: pi_a ~ N_a^(1/tau)).
        action_size = state.board_size * state.board_size + 1  # +1 for pass
        policy = np.zeros(action_size, dtype=np.float32)

        def _action_index(action):
            if action == MOVE_PASS:
                return action_size - 1  # Pass is the last action
            return action[0] * state.board_size + action[1]

        if temperature < 0.01:
            # tau -> 0: the target is a *single* most-visited move. Setting every
            # child tied at max_visits to 1.0 (the old behavior) degenerates to a
            # near-uniform target whenever visits are flat (e.g. low sim counts),
            # which is exactly when the policy head can least afford a noisy label.
            # Pick one argmax deterministically and make the selected move match.
            best_action = max(root.children.items(),
                              key=lambda kv: kv[1].visit_count)[0]
            policy[_action_index(best_action)] = 1.0
            return best_action, policy

        for action, child in root.children.items():
            policy[_action_index(action)] = float(child.visit_count) ** (1.0 / temperature)

        # Apply temperature to visit counts for move selection
        action = self._select_action(root, temperature)

        # Normalize policy to probabilities (training target)
        total = policy.sum()
        if total > 0:
            policy = policy / total

        return action, policy
    
    def _expand(self, node: MCTSNode, allow_pass: bool = True) -> float:
        """
        Expand a leaf node: run neural network, create child nodes.
        
        Returns the value estimate for this position.
        """
        state = node.state
        
        # Get neural network prediction
        state_tensor = state.encode_for_nn()
        policy_probs, value = self.network.predict(state_tensor, self.device)
        
        # Get legal moves
        legal_moves = state.get_legal_moves()
        
        # Shuffle legal moves to break MCTS tie-breaking symmetry. 
        # When the network is untrained, priors are uniform. Without shuffling, 
        # MCTS will always evaluate and play the very first legal moves (e.g. A9, B9) 
        # because of deterministic iteration order.
        import random
        random.shuffle(legal_moves)
        
        # Allow pass only if allow_pass is True or if there are no legal board moves
        if allow_pass or not legal_moves:
            all_actions = legal_moves + [MOVE_PASS]
        else:
            all_actions = list(legal_moves)
        
        # Mask illegal moves and renormalize policy
        action_size = state.board_size * state.board_size + 1
        legal_mask = np.zeros(action_size, dtype=np.float32)
        
        for action in all_actions:
            if action == MOVE_PASS:
                legal_mask[action_size - 1] = 1.0
            else:
                legal_mask[action[0] * state.board_size + action[1]] = 1.0
        
        # Apply mask: zero out illegal moves
        masked_policy = policy_probs.numpy() * legal_mask
        policy_sum = masked_policy.sum()
        
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Fallback: uniform over legal moves (shouldn't happen normally)
            masked_policy = legal_mask / legal_mask.sum()
        
        # Create child nodes for each legal action
        for action in all_actions:
            if action == MOVE_PASS:
                prior = masked_policy[action_size - 1]
            else:
                prior = masked_policy[action[0] * state.board_size + action[1]]
            
            # Create child state
            child_state = state.copy()
            if action == MOVE_PASS:
                child_state.play_pass()
            else:
                success = child_state.play_move(action[0], action[1])
                if not success:
                    continue  # Skip if move somehow failed
            
            child = MCTSNode(child_state, parent=node, parent_action=action, prior=prior)
            node.children[action] = child
        
        node.is_expanded = True
        return value
    
    def _backup(self, node: MCTSNode, value: float) -> None:
        """
        Backpropagate value up the tree.
        
        IMPORTANT: Value is negated at each level because what's good for
        the current player is bad for the opponent. If the value is +0.8
        for the player who just moved, it should be -0.8 for the parent
        (who is the opponent).
        """
        current = node
        while current is not None:
            current.visit_count += 1
            # Negate BEFORE accumulating: `value` arrives from _expand() as the
            # value for the player about to move at `node` (its own perspective).
            # But `current.value_sum` is read by current's PARENT as "how good is
            # this action for me" (Q(s,a) from the parent's mover's perspective),
            # which is the opposite side. So we flip first, then add, and keep
            # alternating on the way up.
            value = -value
            current.value_sum += value
            current = current.parent
    
    def _terminal_value(self, state: GameState) -> float:
        """
        Get value for a terminal game state (game over).
        
        Returns +1 if current player wins, -1 if they lose.
        """
        if state.winner is not None:
            # Winner already determined (resignation)
            if state.winner == state.current_player:
                return 1.0
            else:
                return -1.0
        
        # Game ended by two passes — use scoring
        from game.scoring.base import get_scorer
        scorer = get_scorer("chinese")  # Training always uses Chinese rules
        winner, margin = scorer.determine_winner(state)
        
        if winner is None:
            return 0.0  # Draw
        elif winner == state.current_player:
            return 1.0
        else:
            return -1.0
    
    def _add_dirichlet_noise(self, node: MCTSNode) -> None:
        """
        Add Dirichlet noise to root node priors for exploration.
        
        This is crucial for training: without noise, the network would
        always explore the same moves and never discover new strategies.
        The noise is only added at the root, not deeper in the tree.
        
        new_prior = (1 - ε) * prior + ε * noise
        where noise ~ Dir(α) and ε = dirichlet_epsilon
        """
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior +
                self.dirichlet_epsilon * noise[i]
            )
    
    def _select_action(self, root: MCTSNode, temperature: float) -> Tuple[int, int]:
        """
        Select an action from the root based on visit counts and temperature.
        
        - temperature=1.0: sample proportional to visit counts
        - temperature→0: pick the most-visited move (argmax)
        """
        if not root.children:
            return MOVE_PASS
        
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions], dtype=np.float64)
        
        if temperature < 0.01:
            # Greedy: pick the most-visited action
            best_idx = np.argmax(visits)
            return actions[best_idx]
        
        # Apply temperature: raise visit counts to power 1/T, then normalize
        # Higher temperature → more uniform, lower → more peaked
        adjusted = visits ** (1.0 / temperature)
        total = adjusted.sum()
        
        if total == 0:
            return actions[np.random.randint(len(actions))]
        
        probs = adjusted / total
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]
