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
from game.features import encode_for_network


def visit_distribution(visits, temperature: float) -> np.ndarray:
    """
    AlphaZero's move distribution pi_a ~ N_a^(1/tau), computed in log space.

    The obvious formulation, `visits ** (1.0 / temperature)`, overflows: at
    tau = 0.034 the exponent is ~29, so 50 visits produces 3.8e49. Written into
    the float32 policy array that becomes +inf, normalising turns it into NaN,
    and one optimizer step on a NaN target destroys the network. The sliders
    allow tau down to 0.001, so this was reachable, not theoretical.

    Working in log space and subtracting the maximum first makes the result
    scale-invariant, so it cannot overflow at any temperature. The numbers it
    produces are identical to the naive formula wherever the naive one is
    representable — this is a numerics fix, not a change of target.

    Args:
        visits: Visit count per action.
        temperature: tau > 0. Lower = more peaked on the most-visited action.

    Returns:
        Probabilities over the given actions, summing to 1.
    """
    counts = np.asarray(visits, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        # No visits at all — nothing to prefer, so treat every action equally
        # rather than emitting NaN from a 0/0 normalisation.
        return np.full(counts.shape, 1.0 / max(len(counts), 1), dtype=np.float64)

    with np.errstate(divide='ignore'):
        # Unvisited actions go to -inf here and to exactly 0 after exp(), which
        # is what we want: they were never explored, so they get no mass.
        logits = np.log(counts) / max(temperature, 1e-12)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class MCTSNode:
    """
    A node in the MCTS search tree.
    
    Each node represents a game state and tracks statistics for each
    child action (move).

    LAZY STATE MATERIALISATION

    A node created during expansion holds only the action that reaches it and
    its prior; `state` stays None until the search actually selects it. The
    previous version built a full GameState for every legal move the moment a
    node was expanded — 3,233 of them per 96-simulation search on a mid-game
    9x9 board, of which at most 96 could ever be visited. Around 97% of that
    work, and all the memory behind it, was discarded.

    The saving is bigger than one board copy each: GameState.copy() also copies
    the superko hash-history set, which grows with every move played, so the
    cost of the abandoned children rose as games got longer.

    Attributes:
        state: The GameState at this node, or None until `ensure_state()` runs.
        parent: Parent MCTSNode (None for root).
        parent_action: The action that led from parent to this node.
        children: Dict mapping action → child MCTSNode.
        visit_count: How many times this node has been visited.
        value_sum: Sum of all backpropagated values through this node.
        prior: Prior probability from the neural network (P(s,a) for this action).
        is_expanded: Whether we've run the neural network on this state.
        terminal_value: Cached result for a finished position, so a terminal
            node is scored once instead of on every visit.
    """

    __slots__ = ('state', 'parent', 'parent_action', 'prior', 'children',
                 'visit_count', 'value_sum', 'is_expanded', 'terminal_value')

    def __init__(self, state: Optional[GameState], parent: Optional['MCTSNode'] = None,
                 parent_action: Optional[Tuple[int, int]] = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.parent_action = parent_action
        self.prior = prior

        self.children: Dict[Tuple[int, int], 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.terminal_value: Optional[float] = None

    def ensure_state(self) -> GameState:
        """
        Materialise this node's position, deriving it from the parent the first
        time it is needed.

        The move is known to be legal: it came from `get_legal_moves()` on the
        parent's own position, and since `rules.apply_move` and
        `rules.get_legal_moves` now share one simulation, they cannot disagree.
        A failure here means the tree is being built from a position that has
        drifted from its parent, which would silently corrupt every policy
        target derived from this subtree — so it is raised, not swallowed.
        """
        if self.state is None:
            state = self.parent.state.copy()
            if self.parent_action == MOVE_PASS:
                state.play_pass()
            elif not state.play_move(self.parent_action[0], self.parent_action[1]):
                raise RuntimeError(
                    f"MCTS child move {self.parent_action} was legal at expansion "
                    f"but illegal on materialisation — tree state is inconsistent"
                )
            self.state = state
        return self.state

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
                 restrict_eye_fill: bool = False,
                 restrict_self_atari: bool = False,
                 self_atari_max_stones: int = 1):
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
        # Same mechanism, different rule: hide moves that walk a group larger
        # than self_atari_max_stones into atari for nothing. See
        # game/self_atari.py — this one is a heuristic, not a theorem.
        self.restrict_self_atari = restrict_self_atari
        self.self_atari_max_stones = self_atari_max_stones
        # Root evaluation from the most recent search(), expressed from the
        # perspective of the player to move at the root (∈ [-1, +1]). Used for
        # win-rate tracking. Updated by every search() call.
        self.root_value = 0.0

    def search(self, state: GameState, temperature: float = 1.0,
               add_noise: bool = True, allow_pass: bool = True,
               min_pass_move: int = 0,
               target_temperature: float = 1.0) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Run MCTS from the given state and return the best action.

        Args:
            state: Current game state.
            temperature: Controls randomness of MOVE SELECTION only.
                - temperature=1.0: proportional to visit counts (exploratory)
                - temperature→0: pick the most-visited move (greedy)
            target_temperature: tau used to build the returned POLICY TARGET.
                AlphaZero trains on pi = N/sum(N), i.e. tau = 1, whatever
                temperature the played move was sampled at — the search's
                relative judgement of the runners-up is the entire training
                signal, and sharpening it throws that signal away. Keep this at
                1.0 unless you are deliberately experimenting; see the note
                below the Returns block for what happens when you don't.
            add_noise: Whether to add Dirichlet noise at root (for training).
            allow_pass: Whether MOVE_PASS is allowed as a legal action at all.
            min_pass_move: Game move number before which passing is forbidden.
                Evaluated PER NODE against that node's own move number, not once
                for the whole search. Passing a single bool down (the previous
                behaviour) banned passing at every depth of the tree whenever it
                was banned at the root, so during the first ~29 moves of a
                self-play game no line in the search — however deep — could ever
                reach a terminal position, and the value head's estimate was the
                only grounding the search had in that phase.

        Returns:
            (action, policy_vector):
                action: (row, col) or MOVE_PASS
                policy_vector: Visit count distribution over all actions
                              (used as training target for the policy head).

        WHY THE TWO TEMPERATURES ARE SEPARATE
        -------------------------------------
        They used to be one. A single `visit_distribution(visits, temperature)`
        served as both the sampling distribution and the training target, so a
        self-play schedule that (correctly) anneals temperature towards 0 to
        play its best move late in the game also (incorrectly) annealed the
        LABEL towards one-hot.

        With tau = 0.101 the exponent is 1/tau ~= 9.9, so a 60/50 visit split
        becomes a 0.86/0.14 target and a 60/40 split becomes 0.98/0.02. Measured
        on a real 50k-sample buffer produced this way: 62% of targets were
        literally one-hot and the median position had THREE actions with any
        mass at all, out of 200 simulations spread over ~60 legal moves.

        Training on that is behaviour cloning of the search's argmax, not policy
        improvement: the network is told which move was best but never how much
        better, so its policy entropy decays monotonically. The practical
        failure is invisible in self-play (both sides share the same shrinking
        move set and the gate reads ~50%) and shows up only against a foreign
        opponent, whose replies fall outside the few moves the prior still
        covers — where PUCT cannot recover them because their prior is ~0.

        Sampling at tau < 1 while targeting tau = 1 keeps the old invariant that
        a stored position's played move always has support in its own target:
        low-tau sampling only ever concentrates mass on moves that already hold
        mass at tau = 1, never on moves outside that support.
        """
        root_state = state.copy()
        # Applied to our own copy rather than to the caller's state: the same
        # GameState object is shared with opponents (the random bot in the Elo
        # eval, the other network in the gate match, the human in a web game),
        # and the restriction is a property of THIS searcher, not of the game.
        if self.restrict_eye_fill:
            root_state.restrict_eye_fill = True
        if self.restrict_self_atari:
            root_state.restrict_self_atari = True
            root_state.self_atari_max_stones = self.self_atari_max_stones
        root = MCTSNode(root_state)

        # Expand root node
        self._expand(root, allow_pass=allow_pass, min_pass_move=min_pass_move)

        # Add Dirichlet noise to root priors for exploration during training
        if add_noise and root.children:
            self._add_dirichlet_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root

            # SELECT: walk down tree picking best child. Each child's position
            # is built the first time the search commits to it, so the nodes
            # that are never selected never cost a board copy.
            while node.is_expanded and node.children and not node.state.is_over:
                node = node.best_child(self.c_puct, self.fpu_reduction)
                node.ensure_state()

            # If game is over at this node, use actual result
            if node.state.is_over:
                # Scoring a finished position is not free, and a terminal node
                # is revisited many times over a search, so keep the answer.
                if node.terminal_value is None:
                    node.terminal_value = self._terminal_value(node.state)
                value = node.terminal_value
            else:
                # EXPAND: evaluate with neural network
                value = self._expand(node, allow_pass=allow_pass,
                                     min_pass_move=min_pass_move)

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

        if not root.children:
            policy[action_size - 1] = 1.0
            return MOVE_PASS, policy

        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions],
                          dtype=np.float64)

        # --- Training target: built at target_temperature, independent of the
        # temperature the played move is sampled at (see the docstring). ---
        if target_temperature < 0.01:
            # tau -> 0: the target is a *single* most-visited move. Setting every
            # child tied at max_visits to 1.0 (the old behavior) degenerates to a
            # near-uniform target whenever visits are flat (e.g. low sim counts),
            # which is exactly when the policy head can least afford a noisy label.
            target_probs = np.zeros(len(actions), dtype=np.float64)
            target_probs[int(np.argmax(visits))] = 1.0
        else:
            target_probs = visit_distribution(visits, target_temperature)

        for action, p in zip(actions, target_probs):
            policy[_action_index(action)] = p

        # --- Move selection: sampled at `temperature`. ---
        if temperature < 0.01:
            # Deterministic argmax rather than a random pick among ties: with
            # low simulation counts visit ties are common, and breaking them
            # randomly is a coin flip on every move of a competitive game.
            action = actions[int(np.argmax(visits))]
        else:
            probs = visit_distribution(visits, temperature)
            action = actions[int(np.random.choice(len(actions), p=probs))]

        return action, policy
    
    def _expand(self, node: MCTSNode, allow_pass: bool = True,
                min_pass_move: int = 0) -> float:
        """
        Expand a leaf node: run neural network, create child nodes.

        Returns the value estimate for this position.
        """
        state = node.state

        # Get neural network prediction. The encoding is read off the network
        # itself, so a 12-plane model is never fed a 10-plane position.
        state_tensor = encode_for_network(state, self.network)
        policy_probs, value = self.network.predict(state_tensor, self.device)

        # Get legal moves
        legal_moves = state.get_legal_moves()

        # Shuffle legal moves to break MCTS tie-breaking symmetry.
        # When the network is untrained, priors are uniform. Without shuffling,
        # MCTS will always evaluate and play the very first legal moves (e.g. A9, B9)
        # because of deterministic iteration order.
        import random
        random.shuffle(legal_moves)

        # Pass legality is decided from THIS node's move number, not the root's,
        # so an opening search can still see a game end deep in a line.
        pass_allowed = allow_pass and state.move_number >= min_pass_move

        # Allow pass only if it is permitted here, or if there is nothing else
        # to play (the no-legal-moves fallback — never leave a node actionless).
        if pass_allowed or not legal_moves:
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
        
        # Create child nodes for each legal action. No board work happens here:
        # a child is an action plus a prior until the search selects it (see
        # MCTSNode.ensure_state).
        for action in all_actions:
            if action == MOVE_PASS:
                prior = masked_policy[action_size - 1]
            else:
                prior = masked_policy[action[0] * state.board_size + action[1]]

            node.children[action] = MCTSNode(
                None, parent=node, parent_action=action, prior=prior)

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

        Kept as a standalone entry point for callers that only want a move.
        search() no longer routes through it: it samples from the very same
        distribution it returns as the training target, so the stored move can
        never fall outside the support of the label it is stored with.
        """
        if not root.children:
            return MOVE_PASS

        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions],
                          dtype=np.float64)

        if temperature < 0.01:
            # Greedy: pick the most-visited action
            return actions[int(np.argmax(visits))]

        probs = visit_distribution(visits, temperature)
        return actions[int(np.random.choice(len(actions), p=probs))]
