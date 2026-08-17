"""
worker_pool.py — The one process pool the game-playing phases share.

Self-play and the promotion gate both deal independent games out to worker
processes. They used to each build their own `ProcessPoolExecutor` inside a
`with` block, so a pool was created and destroyed TWICE PER ITERATION.

WHY THAT MATTERS MORE ON WINDOWS. `fork` gives a worker its parent's memory for
free; Windows (and macOS since 3.8) has no fork, so every worker is a fresh
interpreter that must `import torch` — about 1-3 seconds each. At 4 workers
that was a rounding error. At 18 it is tens of seconds per iteration spent
starting processes that are then thrown away, and the cost grows with exactly
the setting a fast machine wants to raise. Keeping one pool alive across phases
and iterations pays it once per training run.

WHAT THE POOL DOES NOT DO. It is not kept alive between training runs:
`Trainer.train()` shuts it down when the loop exits, because ~18 idle
interpreters holding torch is a few gigabytes of resident memory to leave lying
around while nothing is training.

RANDOMNESS. `_init_worker` reseeds every worker from OS entropy. This is not
cosmetic. Under `fork` the child inherits the parent's RNG state, so
`np.random.dirichlet` (the root exploration noise), `np.random.choice` (move
sampling) and `random.randint` (the jittered pass cutoff) all produce the SAME
stream in every worker — and since the network is deterministic, a batch of N
self-play games played in parallel is N copies of one game. The pool never had
an initializer, so on any fork platform that is what a parallel batch was
producing. Reseeding costs nothing and makes the batch independent on every
platform, forked or spawned.
"""

import os
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Optional


def _init_worker() -> None:
    """
    Runs once in each worker process, before its first task.

    Two jobs: keep one worker to one core, and make sure this worker's random
    stream is its own (see the module docstring on fork).
    """
    import random
    import numpy as np
    import torch

    # A worker plays ONE game at batch size 1. Letting torch open a thread per
    # core inside each of N workers oversubscribes the machine by N x cores and
    # makes every game slower than it would be alone.
    torch.set_num_threads(1)

    seed = int.from_bytes(os.urandom(4), "little")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class _PoolState:
    """Module-level singleton, guarded by a lock (the trainer is threaded)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.executor: Optional[ProcessPoolExecutor] = None
        self.workers = 0


_state = _PoolState()


def _is_usable(executor: ProcessPoolExecutor) -> bool:
    """
    Whether an existing executor can still accept work.

    A pool whose worker was killed (OOM, a hard force-stop) is permanently
    broken and every submit raises BrokenProcessPool. These are CPython
    internals, hence the getattr dance — an unrecognised executor is treated as
    unusable, which costs one pool rebuild and never a crash.
    """
    if executor is None:
        return False
    if getattr(executor, "_broken", False):
        return False
    if getattr(executor, "_shutdown_thread", False):
        return False
    return True


def get_executor(max_workers: int) -> ProcessPoolExecutor:
    """
    The shared pool, sized to at least `max_workers`.

    The pool is REUSED when it is already at least this big: a phase that wants
    fewer workers than the pool has simply keeps fewer tasks in flight, which
    both callers already do (they submit `workers` tasks and top up as each
    completes). Only a request for MORE workers than the live pool has rebuilds
    it, so lowering the worker count mid-run costs nothing and raising it costs
    one rebuild.
    """
    max_workers = max(1, int(max_workers))
    with _state.lock:
        if _is_usable(_state.executor) and _state.workers >= max_workers:
            return _state.executor

        if _state.executor is not None:
            _state.executor.shutdown(wait=False, cancel_futures=True)

        _state.executor = ProcessPoolExecutor(
            max_workers=max_workers, initializer=_init_worker)
        _state.workers = max_workers
        return _state.executor


def shutdown(wait: bool = False) -> None:
    """
    Tear the pool down. Called when a training run ends.

    `wait=False` returns immediately; the workers still finish the game they are
    holding, because a game is the smallest thing this project can abandon
    cleanly — there is no way to interrupt an MCTS search mid-simulation.
    """
    with _state.lock:
        if _state.executor is not None:
            _state.executor.shutdown(wait=wait, cancel_futures=True)
        _state.executor = None
        _state.workers = 0


def active_workers() -> int:
    """Size of the live pool, or 0 when there isn't one. For status/tests."""
    with _state.lock:
        return _state.workers if _is_usable(_state.executor) else 0
