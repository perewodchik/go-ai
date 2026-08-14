"""
estimator.py — Score estimation for display purposes only.

THIS MODULE IS DISPLAY-ONLY. The bots never see this information.
It provides a visual overlay showing estimated territory ownership.

Two complementary approaches:
1. Benson's algorithm — finds groups that are UNCONDITIONALLY alive
   (cannot be killed even if the opponent plays infinite consecutive moves).
   Very conservative but 100% accurate for the groups it identifies.

2. Flood-fill territory estimation — starting from alive groups, flood-fill
   outward to estimate territory. More speculative but gives a useful
   visual indication of who's ahead.

The UI can toggle this overlay on/off. When toggled on, each intersection
gets a color indicating estimated ownership:
  - Strong black territory: high confidence black owns this point
  - Strong white territory: high confidence white owns this point  
  - Neutral / contested: unclear ownership (dame or fighting)
"""

import numpy as np
from typing import Dict, Tuple, FrozenSet, Set, List
from game.board import Board, EMPTY, BLACK, WHITE, opponent
from game.game_state import GameState


class ScoreEstimator:
    """
    Estimates territory ownership for display overlay.
    
    IMPORTANT: This is NEVER used by the bots for decision-making.
    The AI uses only its neural network value head for evaluation.
    """
    
    def __init__(self):
        pass
    
    def estimate(self, state: GameState) -> dict:
        """
        Estimate the current score and territory map.
        
        Returns a dict with:
            'ownership_map': 2D array of floats (-1.0 = black, +1.0 = white, 0 = neutral)
            'black_estimate': estimated black score
            'white_estimate': estimated white score
            'alive_groups': dict of unconditionally alive groups per color
            'dead_candidates': groups that might be dead
        """
        board = state.board
        size = board.size
        
        # Step 1: Find unconditionally alive groups (Benson's algorithm)
        alive_black = self._bensons_alive(board, BLACK)
        alive_white = self._bensons_alive(board, WHITE)
        
        # Step 2: Build ownership map via flood-fill from alive groups
        ownership = np.zeros((size, size), dtype=np.float32)
        
        # Mark alive stones with strong ownership
        for group in alive_black:
            for r, c in group:
                ownership[r, c] = -1.0  # Black = negative
        for group in alive_white:
            for r, c in group:
                ownership[r, c] = 1.0   # White = positive
        
        # Step 3: Flood-fill territory from alive groups
        territory_map = self._flood_fill_territory(board, alive_black, alive_white)
        
        # Merge: empty points get territory color, non-alive stones get weaker signal
        for r in range(size):
            for c in range(size):
                if board.grid[r, c] == EMPTY:
                    ownership[r, c] = territory_map[r, c]
                elif ownership[r, c] == 0:
                    # Stone that isn't proven alive — slight ownership for its color,
                    # but mark as "uncertain" (weaker signal)
                    if board.grid[r, c] == BLACK:
                        ownership[r, c] = -0.5
                    else:
                        ownership[r, c] = 0.5
        
        # Step 4: Calculate score estimates
        black_est = float(np.sum(ownership < -0.1))
        white_est = float(np.sum(ownership > 0.1)) + state.komi
        
        # Identify potentially dead groups (non-alive stones in opponent territory)
        dead_candidates = self._find_dead_candidates(board, ownership)
        
        return {
            'ownership_map': ownership.tolist(),
            'black_estimate': black_est,
            'white_estimate': white_est,
            'alive_black_count': sum(len(g) for g in alive_black),
            'alive_white_count': sum(len(g) for g in alive_white),
            'dead_candidates': dead_candidates,
        }
    
    def _bensons_alive(self, board: Board, color: int) -> List[FrozenSet[Tuple[int, int]]]:
        """
        Benson's algorithm for unconditional life.
        
        A group is unconditionally alive if:
        1. It has at least two "vital" regions (potential eyes).
        2. Each vital region is completely surrounded by the group's stones
           (no opponent stones can intrude).
        
        Algorithm:
        1. Start with all groups of `color` as candidates.
        2. Find "healthy" regions: empty regions where ALL border stones
           belong to candidate groups (no opponent border stones).
        3. A group is "safe" if it borders at least 2 healthy regions.
        4. Remove unsafe groups from candidates.
        5. Repeat until stable (no more groups removed).
        
        This is conservative — it only finds groups that are PROVABLY alive.
        Groups in seki or complex life-and-death situations may not be detected.
        """
        # Get all groups of this color
        all_groups = board.get_all_groups(color)
        if not all_groups:
            return []
        
        # Candidate set: indices into all_groups that might be alive
        candidates = set(range(len(all_groups)))
        
        changed = True
        while changed:
            changed = False
            
            # Find healthy regions: empty regions bordered ONLY by candidate groups
            healthy_regions = self._find_healthy_regions(board, all_groups, candidates, color)
            
            # Check each candidate: does it border >= 2 healthy regions?
            new_candidates = set()
            for idx in candidates:
                group = all_groups[idx]
                # Count how many healthy regions this group borders
                bordering_healthy = 0
                for region in healthy_regions:
                    # Check if any stone in the group is adjacent to this region
                    if self._group_borders_region(board, group, region):
                        bordering_healthy += 1
                
                if bordering_healthy >= 2:
                    new_candidates.add(idx)
            
            if new_candidates != candidates:
                candidates = new_candidates
                changed = True
        
        return [all_groups[idx] for idx in candidates]
    
    def _find_healthy_regions(
        self, board: Board,
        all_groups: list,
        candidates: Set[int],
        color: int
    ) -> List[FrozenSet[Tuple[int, int]]]:
        """
        Find empty regions where every adjacent stone belongs to a candidate group.
        These are potential "eyes" for the candidate groups.
        """
        # Build set of all positions in candidate groups
        candidate_positions = set()
        for idx in candidates:
            candidate_positions |= all_groups[idx]
        
        healthy = []
        visited = set()
        
        for r in range(board.size):
            for c in range(board.size):
                if board.grid[r, c] == EMPTY and (r, c) not in visited:
                    # BFS to find connected empty region
                    region = set()
                    is_healthy = True
                    queue = [(r, c)]
                    region.add((r, c))
                    
                    while queue:
                        cr, cc = queue.pop()
                        for nr, nc in board.neighbors(cr, cc):
                            if (nr, nc) in region:
                                continue
                            if board.grid[nr, nc] == EMPTY:
                                region.add((nr, nc))
                                queue.append((nr, nc))
                            elif board.grid[nr, nc] == color:
                                # Must be part of a candidate group
                                if (nr, nc) not in candidate_positions:
                                    is_healthy = False
                            else:
                                # Opponent stone borders this region → not healthy
                                is_healthy = False
                    
                    visited |= region
                    if is_healthy and len(region) > 0:
                        healthy.append(frozenset(region))
        
        return healthy
    
    def _group_borders_region(
        self, board: Board,
        group: FrozenSet[Tuple[int, int]],
        region: FrozenSet[Tuple[int, int]]
    ) -> bool:
        """Check if any stone in `group` is adjacent to any point in `region`."""
        for r, c in group:
            for nr, nc in board.neighbors(r, c):
                if (nr, nc) in region:
                    return True
        return False
    
    def _flood_fill_territory(
        self, board: Board,
        alive_black: List[FrozenSet],
        alive_white: List[FrozenSet]
    ) -> np.ndarray:
        """
        Multi-source BFS from every stone to estimate territory.

        Each empty point gets a value based on which color's *nearest stone*
        is closer (BFS distance through empty points). Points equidistant
        from both colors — or unreachable from either — are neutral (dame).

        Proximity alone only decides *which* color owns a point. Whether
        that gets a strong (+/-0.8) or weak (+/-0.4) signal depends on
        whether the point directly borders a proven-alive (Benson) group
        of the owning color — this is a confidence modifier, not the gate
        for ownership. Gating ownership itself on "touches an alive group"
        (the previous approach) meant that whichever color had zero
        Benson-alive groups — extremely common mid-game, since Benson's
        requires two fully sealed eyes — lost its *entire* surrounding
        territory to the other color, even points sitting right next to
        its own stones.

        Returns array of floats: -1.0 (black) .. +1.0 (white), 0 = neutral.
        """
        size = board.size
        territory = np.zeros((size, size), dtype=np.float32)

        alive_black_positions = set()
        for g in alive_black:
            alive_black_positions |= g

        alive_white_positions = set()
        for g in alive_white:
            alive_white_positions |= g

        dist_black = self._bfs_distance_from_stones(board, BLACK)
        dist_white = self._bfs_distance_from_stones(board, WHITE)

        for r in range(size):
            for c in range(size):
                if board.grid[r, c] != EMPTY:
                    continue

                db, dw = dist_black[r, c], dist_white[r, c]
                if db == dw:
                    continue  # equidistant (or unreachable from both) — dame

                owner_black = db < dw
                # Strong signal if this point directly borders a proven-alive
                # group of the owning color; otherwise a weaker signal.
                borders_alive = any(
                    (nr, nc) in (alive_black_positions if owner_black else alive_white_positions)
                    for nr, nc in board.neighbors(r, c)
                )
                strength = 0.8 if borders_alive else 0.4
                territory[r, c] = -strength if owner_black else strength

        return territory

    def _bfs_distance_from_stones(self, board: Board, color: int) -> np.ndarray:
        """BFS distance (through empty points) from every point to the nearest stone of `color`."""
        from collections import deque

        size = board.size
        dist = np.full((size, size), size * size, dtype=np.int32)
        dq = deque()

        for r in range(size):
            for c in range(size):
                if board.grid[r, c] == color:
                    dist[r, c] = 0
                    dq.append((r, c))

        while dq:
            cr, cc = dq.popleft()
            d = dist[cr, cc]
            for nr, nc in board.neighbors(cr, cc):
                # Only hop through empty points — stones of either color block the path.
                if board.grid[nr, nc] == EMPTY and dist[nr, nc] > d + 1:
                    dist[nr, nc] = d + 1
                    dq.append((nr, nc))

        return dist
    
    def _find_dead_candidates(
        self, board: Board, ownership: np.ndarray
    ) -> List[dict]:
        """
        Find groups that might be dead: stones sitting in clearly
        opponent-controlled territory.
        
        Returns list of dicts with group info for the UI.
        """
        dead = []
        visited = set()
        
        for color in (BLACK, WHITE):
            for r in range(board.size):
                for c in range(board.size):
                    if board.grid[r, c] == color and (r, c) not in visited:
                        group = board.get_group(r, c)
                        visited |= group
                        
                        # Check if the group is in opponent territory
                        # (average ownership of surrounding empty points)
                        surround_vals = []
                        for gr, gc in group:
                            for nr, nc in board.neighbors(gr, gc):
                                if board.grid[nr, nc] == EMPTY:
                                    surround_vals.append(ownership[nr, nc])
                        
                        if not surround_vals:
                            continue
                        
                        avg_surround = sum(surround_vals) / len(surround_vals)
                        
                        # Black stone in white territory (positive ownership)
                        # or white stone in black territory (negative ownership)
                        is_dead = False
                        if color == BLACK and avg_surround > 0.3:
                            is_dead = True
                        elif color == WHITE and avg_surround < -0.3:
                            is_dead = True
                        
                        if is_dead:
                            dead.append({
                                'color': color,
                                'positions': [list(p) for p in group],
                                'size': len(group),
                            })
        
        return dead
