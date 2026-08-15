"""
benchmark_search.py — reproducible self-play throughput measurement.

NOT part of the pytest suite (the filename does not match `test_*`). Run it
directly, before and after a performance change:

    venv/bin/python tests/benchmark_search.py

WHY THIS EXISTS

Profiling a 96-simulation search on a mid-game 9x9 board showed the split is not
where the code comments assume it is:

    _expand                97 %  of search time
      rules.is_legal_move  58 %  (11,090 calls)
      Board.neighbors      33 %  (152,067 calls)
      building child states 41 % (3,233 GameStates, <=96 ever visited)
    network.predict        16 %

The rules engine is ~84% of self-play and the neural network is ~16%, so making
the network smaller buys almost nothing. Every number this script prints is
about the engine.

Positions are generated from a fixed seed so runs are comparable.
"""

import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from ai.mcts import MCTS
from ai.network import GoNetwork
from game.game_state import GameState


def make_position(board_size: int, num_moves: int, seed: int) -> GameState:
    """A reproducible mid-game position reached by random legal play."""
    rng = random.Random(seed)
    state = GameState(board_size=board_size, komi=6.5)
    for _ in range(num_moves):
        legal = state.get_legal_moves()
        if not legal:
            break
        state.play_move(*rng.choice(legal))
    return state


def _time(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    board_size = 9
    net = GoNetwork(board_size=board_size, num_input_planes=10)
    net.eval()

    print(f"board {board_size}x{board_size}, 4x64 network, 1 torch thread\n")

    # --- component costs -------------------------------------------------
    state = make_position(board_size, 40, seed=0)
    tensor = state.encode_for_nn()

    nn_ms = _time(lambda: net.predict(tensor, "cpu"), 200) * 1000
    enc_ms = _time(state.encode_for_nn, 500) * 1000
    legal_ms = _time(state.get_legal_moves, 200) * 1000
    copy_ms = _time(state.copy, 2000) * 1000

    print("component                       ms")
    print(f"  network.predict          {nn_ms:8.3f}")
    print(f"  encode_for_nn            {enc_ms:8.3f}")
    print(f"  get_legal_moves          {legal_ms:8.3f}")
    print(f"  GameState.copy           {copy_ms:8.3f}")

    # --- full searches across several game phases ------------------------
    sims = 96
    print(f"\nMCTS search @ {sims} sims")
    print("  stones   sec/search   NN share")
    total = 0.0
    for stones in (0, 20, 40, 60):
        pos = make_position(board_size, stones, seed=stones + 1)
        mcts = MCTS(network=net, num_simulations=sims, c_puct=1.5, device="cpu")
        elapsed = _time(lambda: mcts.search(pos, temperature=0.5, add_noise=True), 3)
        total += elapsed
        share = (sims * nn_ms / 1000) / elapsed
        print(f"  {stones:6d}   {elapsed:10.3f}   {share:7.1%}")

    print(f"\n  TOTAL (4 phases)         {total:.3f} s")
    print(f"  moves/sec (1 worker)     {4.0 / total:.2f}")


if __name__ == "__main__":
    main()
