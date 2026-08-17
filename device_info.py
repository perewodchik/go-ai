"""
device_info.py — What this machine can actually train on.

One place that answers three questions the rest of the project keeps asking
separately:

  1. WHICH DEVICE does the training network run on (CUDA / MPS / CPU), and what
     is it called? `detect()` / `resolve_torch_device()`.
  2. HOW MANY WORKER PROCESSES may the game-playing phases use? `worker_ceiling()`
     / `recommended_workers()`. This is a CPU question, not a GPU one — see the
     note below.
  3. IS THE GPU ACTUALLY WORTH USING for this workload? `benchmark()`.

WHY WORKERS ARE A CPU NUMBER. Self-play and the promotion gate play independent
games across a process pool, and the trainer deliberately hands those pools its
CPU-resident champion (`device="cpu"`). Each worker runs one MCTS at batch size
1, which is latency-bound: a 320k-parameter forward pass on a 9x9 board is
faster on one CPU core than it is through a GPU launch. Twenty such workers also
cannot share one GPU without twenty CUDA contexts and twenty copies of the
weights. So the pool scales with cores, and the GPU is used for exactly one
thing — the gradient steps in `Trainer._train_network`, which run in the main
process on a real batch.

Nothing here imports torch at module level: `param_bounds` needs the core counts
to build its sliders, and making a slider definition depend on a 2-second torch
import (and on torch being installed at all) is not a trade worth making.
"""

import os
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Worker-count policy
# ---------------------------------------------------------------------------

# Never offer FEWER workers than the ceiling this project shipped with, even on
# a 2-core machine. The pools clamp to the core count and to the game count at
# run time anyway (`min(num_games, os.cpu_count(), num_workers)`), so a ceiling
# above the core count costs nothing and keeps an existing model's stored value
# inside the slider's range when its config moves between machines.
MIN_WORKER_CEILING = 8

# Sanity cap. Beyond this the per-process memory of a torch worker (~150-250 MB
# once torch is imported) matters more than the extra parallelism.
MAX_WORKER_CEILING = 64

# Logical cores deliberately left unallocated by the RECOMMENDED count: the web
# server thread, the trainer thread, and — on a GPU box — whatever feeds the
# GPU between iterations. The ceiling does not reserve these; a user who wants
# every core may still have them.
WORKER_HEADROOM = 2


def logical_cores() -> int:
    """Logical processors visible to this process (hyperthreads included)."""
    return os.cpu_count() or 4


def worker_ceiling() -> int:
    """
    Highest worker count the UI should offer on this host.

    This is what `param_bounds` uses as the slider maximum, so the answer has to
    be stable across processes on the same machine — it is derived from the core
    count alone, never from a live measurement.
    """
    return max(MIN_WORKER_CEILING, min(logical_cores(), MAX_WORKER_CEILING))


def recommended_workers() -> int:
    """
    Sensible default worker count for this host.

    Leaves WORKER_HEADROOM logical cores for everything that is not a game
    worker. Never returns more than `worker_ceiling()`, and never less than 1.
    """
    return max(1, min(worker_ceiling(), logical_cores() - WORKER_HEADROOM))


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """A resolved compute device, described well enough to put in a UI badge."""

    kind: str = "cpu"                 # "cuda" | "mps" | "cpu"
    torch_device: str = "cpu"         # what you pass to .to()
    label: str = "CPU"                # short badge text
    name: str = ""                    # "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
    detail: str = ""                  # one-line description for a tooltip
    is_gpu: bool = False
    total_memory_gb: Optional[float] = None
    capability: Optional[str] = None  # CUDA compute capability, e.g. "8.6"
    device_count: int = 0
    torch_version: str = ""
    cuda_runtime: Optional[str] = None
    # Why we are NOT on a GPU, when we are not. Empty when there is nothing to
    # explain. This is the field that turns "training is slow" into a diagnosis.
    note: str = ""
    # Populated only when torch itself could not be imported.
    torch_available: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _cpu_info() -> DeviceInfo:
    """The always-available fallback."""
    return DeviceInfo(
        kind="cpu",
        torch_device="cpu",
        label="CPU",
        name=f"{logical_cores()} logical cores",
        detail=f"CPU · {logical_cores()} logical cores",
        is_gpu=False,
    )


def _detect_uncached() -> DeviceInfo:
    """
    Pick the best available device: CUDA > MPS > CPU.

    CUDA is checked FIRST. The order used to be MPS-first because this project
    was written on an M2, and on any machine only one of the two is ever
    available — but `torch.backends.mps` does not exist on old builds and
    `is_available()` has been known to raise rather than return False on
    unusual ones, which would take the whole config import down with it. Both
    probes are therefore defensive, and neither can promote a device that
    cannot actually run a tensor.
    """
    try:
        import torch
    except Exception as e:                                  # torch not installed
        info = _cpu_info()
        info.torch_available = False
        info.note = f"PyTorch is not importable ({e}) — install it to train."
        return info

    info = _cpu_info()
    info.torch_version = getattr(torch, "__version__", "")

    # --- CUDA (NVIDIA) ---
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_gb = round(props.total_memory / (1024 ** 3), 1)
            capability = f"{props.major}.{props.minor}"
            return DeviceInfo(
                kind="cuda",
                torch_device="cuda",
                label="CUDA",
                name=props.name,
                detail=f"{props.name} · {total_gb} GB · compute {capability}",
                is_gpu=True,
                total_memory_gb=total_gb,
                capability=capability,
                device_count=torch.cuda.device_count(),
                torch_version=info.torch_version,
                cuda_runtime=getattr(torch.version, "cuda", None),
            )
        # An NVIDIA card with a CPU-only torch build is the single most common
        # "why is my GPU idle" case on Windows, and it is worth naming exactly.
        if getattr(torch.version, "cuda", None) is None:
            info.note = (
                "This PyTorch build has no CUDA support (CPU-only wheel). "
                "Reinstall from the CUDA index to use an NVIDIA GPU."
            )
        else:
            info.note = (
                f"PyTorch was built for CUDA {torch.version.cuda} but no GPU is "
                "visible — check the NVIDIA driver."
            )
    except Exception as e:
        info.note = f"CUDA probe failed ({e}); running on CPU."

    # --- MPS (Apple Silicon) ---
    try:
        backends = getattr(torch.backends, "mps", None)
        if backends is not None and backends.is_available():
            return DeviceInfo(
                kind="mps",
                torch_device="mps",
                label="MPS",
                name="Apple Silicon GPU",
                detail="Apple Silicon GPU (Metal Performance Shaders)",
                is_gpu=True,
                device_count=1,
                torch_version=info.torch_version,
            )
    except Exception:
        # Absent or broken MPS is not worth a note on a machine that already
        # explained itself under CUDA above.
        pass

    return info


_cached: Optional[DeviceInfo] = None


def detect(refresh: bool = False) -> DeviceInfo:
    """
    The best device on this machine, cached after the first call.

    Detection touches the CUDA driver, so it is not free; nothing about it
    changes while the process lives. Pass `refresh=True` in a test that has
    monkeypatched torch.
    """
    global _cached
    if _cached is None or refresh:
        _cached = _detect_uncached()
    return _cached


def resolve_torch_device(preferred: Optional[str] = None) -> tuple:
    """
    Confirm a device can really run a tensor, and fall back to CPU if not.

    A device string that only came from `is_available()` is a claim, not a
    guarantee: a CUDA build whose driver is too old, or an MPS build hitting an
    unimplemented op, both fail at the first real allocation. Everything that
    moves a network onto a device should go through here.

    Returns (device_string, note) where note is '' when the preferred device
    worked and an explanation when it did not.
    """
    target = preferred or detect().torch_device
    if target == "cpu":
        return "cpu", ""

    try:
        import torch
        probe = torch.zeros(8, 8, device=target)
        # Force a real kernel launch — allocation alone can succeed on a device
        # that cannot actually compute.
        float((probe + 1).sum().item())
        return target, ""
    except Exception as e:
        return "cpu", f"{target} was unusable ({e}); falling back to CPU."


# ---------------------------------------------------------------------------
# "Is the GPU actually worth using?"
# ---------------------------------------------------------------------------

# Below this speed-up over CPU the GPU is not buying anything worth the
# complexity of using it, and the honest answer is to say so.
GPU_USEFUL_SPEEDUP = 1.3
# Above this, the GPU is unambiguously the right place to train.
GPU_FAST_SPEEDUP = 2.5


def _bench_workload(device: str, batch_size: int, board_size: int,
                    num_filters: int, num_blocks: int, iters: int) -> float:
    """
    Time one training-shaped forward+backward, in milliseconds per step.

    Deliberately a synthetic residual conv stack rather than the real GoNetwork:
    this module must stay importable with nothing from `ai/` loaded, and the
    shape is what determines the timing, not the exact head layout.
    """
    import time
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    layers = []
    for _ in range(num_blocks):
        layers += [
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        ]
    model = nn.Sequential(
        nn.Conv2d(10, num_filters, 3, padding=1, bias=False),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True),
        *layers,
    ).to(device)
    x = torch.randn(batch_size, 10, board_size, board_size, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = model(x).square().mean()
        loss.backward()
        opt.step()

    # Warm-up: the first CUDA step pays for context creation and kernel
    # autotuning, which would otherwise be reported as the workload's cost.
    for _ in range(3):
        step()
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def benchmark(batch_size: int = 32, board_size: int = 9, num_filters: int = 64,
              num_blocks: int = 4, iters: int = 12,
              large_batch: int = 256) -> dict:
    """
    Measure whether this machine's GPU beats its CPU on a training step.

    Two batch sizes are measured, because the answer usually differs between
    them and the difference IS the advice. This project's default network is
    tiny (4x64 on a 9x9 board) and its default batch is 32, so a single training
    step is a handful of microseconds of arithmetic wrapped in kernel-launch
    overhead — a regime where a laptop GPU can genuinely lose to a CPU. The same
    GPU pulls far ahead at batch 256. Reporting only the configured batch would
    let someone conclude their GPU is useless when what is actually useless is
    the batch size they are feeding it.

    Returns a dict; `verdict` is one of:
        no_gpu     — nothing to compare against
        fast       — clear win at the configured batch, train on the GPU
        modest     — a win, but a small one
        cpu_better — the CPU is faster here; raise batch_size or network size
        failed     — the measurement itself blew up (message in `error`)
    """
    info = detect()
    out = {
        'device': info.torch_device,
        'kind': info.kind,
        'name': info.name,
        'batch_size': batch_size,
        'large_batch': large_batch,
        'gpu_ms': None,
        'cpu_ms': None,
        'speedup': None,
        'gpu_ms_large': None,
        'cpu_ms_large': None,
        'speedup_large': None,
        'verdict': 'no_gpu',
        'headline': '',
        'advice': '',
        'error': None,
    }

    if not info.is_gpu:
        out['headline'] = 'No GPU detected — training runs on the CPU.'
        out['advice'] = info.note or (
            'Self-play is CPU-bound either way; more parallel workers is the '
            'lever that matters on this machine.')
        return out

    try:
        for suffix, bs in (('', batch_size), ('_large', large_batch)):
            gpu_ms = _bench_workload(info.torch_device, bs, board_size,
                                     num_filters, num_blocks, iters)
            cpu_ms = _bench_workload('cpu', bs, board_size,
                                     num_filters, num_blocks, iters)
            out[f'gpu_ms{suffix}'] = round(gpu_ms, 2)
            out[f'cpu_ms{suffix}'] = round(cpu_ms, 2)
            out[f'speedup{suffix}'] = round(cpu_ms / gpu_ms, 2) if gpu_ms > 0 else None
    except Exception as e:
        out['verdict'] = 'failed'
        out['error'] = str(e)
        out['headline'] = f'GPU benchmark failed: {e}'
        return out

    speedup = out['speedup'] or 0.0
    big = out['speedup_large']

    if speedup >= GPU_FAST_SPEEDUP:
        out['verdict'] = 'fast'
        out['headline'] = f"{info.name} is {speedup:.1f}x faster than the CPU at batch {batch_size}."
        out['advice'] = 'Train on the GPU — it is already the right choice at your batch size.'
    elif speedup >= GPU_USEFUL_SPEEDUP:
        out['verdict'] = 'modest'
        out['headline'] = f"{info.name} is {speedup:.1f}x faster than the CPU at batch {batch_size}."
        out['advice'] = (
            f'A real but small win. This network is tiny, so most of a step is '
            f'launch overhead' +
            (f' — at batch {large_batch} the same GPU is {big:.1f}x faster. '
             f'Raising batch_size (or the network size) is what turns the GPU on.'
             if big else '.'))
    else:
        out['verdict'] = 'cpu_better'
        out['headline'] = (
            f"{info.name} is NOT faster than this CPU at batch {batch_size} "
            f"({speedup:.2f}x).")
        out['advice'] = (
            f'A 4x64 network on a {board_size}x{board_size} board at batch '
            f'{batch_size} is too small to fill a GPU' +
            (f'; at batch {large_batch} it reaches {big:.1f}x. Raise batch_size '
             f'or pick a larger network preset before the GPU pays for itself.'
             if big else '. Raise batch_size or the network size.'))
    return out


# ---------------------------------------------------------------------------
# One payload for the UI
# ---------------------------------------------------------------------------

def roles_for(label: str) -> dict:
    """
    The GPU/CPU split, described for a given training device.

    A function rather than a literal because the trainer has to rebuild it: a
    run that asked for CUDA and got demoted to CPU must not keep a sentence
    claiming the gradient steps are on a device nothing is using. Rebuilding it
    here beats patching the string at the call site.
    """
    return {
        'training': label,
        'self_play': 'CPU workers',
        'explanation': (
            f'Gradient steps run on {label}. Self-play and gate games run as '
            f'CPU worker processes — one MCTS at batch size 1 is faster on a '
            f'core than through a GPU launch, and they cannot share one GPU '
            f'without a context each.'
        ),
    }


def summary(include_benchmark: bool = False, **bench_kwargs) -> dict:
    """
    Everything a client needs to describe this machine, in one dict.

    The benchmark is opt-in because it costs seconds of real compute; page loads
    ask for the description only.
    """
    info = detect()
    out = {
        'device': info.to_dict(),
        'cpu': {
            'logical_cores': logical_cores(),
            'worker_ceiling': worker_ceiling(),
            'recommended_workers': recommended_workers(),
            'worker_headroom': WORKER_HEADROOM,
        },
        # Said once, here, so every surface that shows the device can explain
        # the split the same way instead of inventing its own wording.
        'roles': roles_for(info.label),
    }
    if include_benchmark:
        out['benchmark'] = benchmark(**bench_kwargs)
    return out
