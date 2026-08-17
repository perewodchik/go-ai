"""
param_bounds.py — Centralized bounds and categories for training & MCTS parameters.

Edit this file if you need to adjust parameter bounds (min, max, step, category, default, label, hint).
Both backend validation and frontend sliders (Create Model, Edit Model, Live Tuning)
derive their configuration from this single authoritative file.

ONE BOUND IS HOST-DEPENDENT. `num_parallel_workers` cannot have a fixed maximum:
the pools cap themselves at the core count anyway, so a hardcoded 8 was simply a
20-thread machine's slider refusing to offer 12 of its cores. Its max and
default come from device_info, which is deliberately torch-free so importing
this file stays cheap. Everything else here is a constant.
"""

import device_info

# Slider ceiling for the worker count on THIS machine. Floored at 8 so the
# bound never gets tighter than the one this project shipped with, which also
# means a model config carrying 8 workers stays inside the slider's range when
# it is opened on a 4-core laptop.
MAX_PARALLEL_WORKERS = device_info.worker_ceiling()
DEFAULT_PARALLEL_WORKERS = device_info.recommended_workers()

# --- Recording volume ------------------------------------------------------
# Bytes one stored game costs, measured on 9x9 self-play records (~12 KB) and
# scaled by board area for other sizes. Used by the UI to price the recording
# toggles before the disk fills rather than after.
BYTES_PER_STORED_GAME_9X9 = 12_000
# Games in ONE PHASE above which leaving that phase's recording on is a bad
# default rather than a preference. Per phase, not summed, because the toggles
# are per phase — "which one is expensive" is the actionable question. At 20
# workers a single iteration can produce more games than a whole afternoon used
# to, and nothing ever deletes them.
RECORDING_VOLUME_WARN_GAMES = 24


def format_bytes(num_bytes: float) -> str:
    """
    Bytes at a human scale, unit chosen by magnitude.

    Fixed units do not survive the range this is used over: one iteration of
    gate games is half a megabyte and a thousand iterations of them is half a
    gigabyte. Rounding the first to "0 MB" told the reader the cost was nothing
    in the same sentence that called it a problem. Mirrors formatBytes() in
    web/static/js/param_sliders.js.
    """
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.{0 if num_bytes >= 1e10 else 1}f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.{0 if num_bytes >= 1e7 else 1}f} MB"
    return f"{max(1, round(num_bytes / 1e3))} KB"

PARAM_BOUNDS = {
    "num_simulations": {
        "key": "num_simulations",
        "order": 10,
        "label": "MCTS Simulations",
        "category": "mcts",
        "category_label": "MCTS & Search Strategy",
        "min": 10,
        "max": 1000,
        "step": 1,
        "default": 96,
        "type": "int",
        "hint": "Look-ahead simulations per move — main search quality and time lever",
    },
    "c_puct": {
        "key": "c_puct",
        "order": 20,
        "label": "Exploration (c_puct)",
        "category": "mcts",
        "category_label": "MCTS & Search Strategy",
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "default": 1.5,
        "type": "float",
        "hint": "Exploration vs exploitation ratio in MCTS search",
    },
    "temperature_threshold": {
        "key": "temperature_threshold",
        "order": 30,
        "label": "Temp Decay Moves",
        "category": "temperature",
        "category_label": "Temperature & Randomness",
        "min": 0,
        "max": 100,
        "step": 1,
        "default": 30,
        "type": "int",
        "hint": "Number of early moves over which move randomness decays",
    },
    "temperature_init": {
        "key": "temperature_init",
        "order": 10,
        "label": "Temp Init (Early)",
        "category": "temperature",
        "category_label": "Temperature & Randomness",
        "min": 0.0,
        "max": 2.0,
        "step": 0.05,
        "default": 0.8,
        "type": "float",
        "hint": "Early-game move randomness temperature",
    },
    "temperature_final": {
        "key": "temperature_final",
        "order": 20,
        "label": "Temp Final (Late)",
        "category": "temperature",
        "category_label": "Temperature & Randomness",
        "min": 0.001,
        "max": 1.0,
        "step": 0.01,
        "default": 0.1,
        "type": "float",
        "hint": "Late-game move randomness temperature",
    },
    "policy_target_temperature": {
        "key": "policy_target_temperature",
        "order": 30,
        "label": "Policy Target Temp",
        "category": "temperature",
        "category_label": "Temperature & Randomness",
        # Floor of 0.5, unlike temperature_final's 0.001: this is the LABEL, and
        # tau below ~0.5 turns it into an argmax the policy head can only clone.
        "min": 0.5,
        "max": 1.5,
        "step": 0.05,
        "default": 1.0,
        "type": "float",
        "hint": "tau for the policy LABEL (1.0 = AlphaZero). Lower = the network "
                "is taught only the search's best move, not how much better it "
                "was — policy entropy then decays and the model overfits to its "
                "own self-play lines. Leave at 1.0.",
    },
    "num_self_play_games": {
        "key": "num_self_play_games",
        "order": 10,
        "label": "Self-Play Games",
        "category": "volume",
        "category_label": "Self-Play Volume",
        "min": 1,
        "max": 100,
        "step": 1,
        "default": 5,
        "type": "int",
        "hint": "Games the bot plays against itself per training iteration",
    },
    # --- Champion vs candidate (promotion gate) --------------------------
    # These decide whether a freshly trained candidate is allowed to replace
    # the champion that generates all self-play data. See the "Champion vs
    # Candidate" section of the Info page for what each one does to learning.
    "gate_enabled": {
        "key": "gate_enabled",
        "order": 10,
        "label": "Promotion Gate",
        "category": "gate",
        "category_label": "Champion vs Candidate",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": True,
        "type": "bool",
        "hint": "Off = every update becomes champion unchecked (regressions stick)",
    },
    "gate_games": {
        "key": "gate_games",
        "order": 20,
        "label": "Gate Games",
        "category": "gate",
        "category_label": "Champion vs Candidate",
        "min": 2,
        "max": 60,
        "step": 2,
        "default": 20,
        "type": "int",
        "hint": "Head-to-head games per gate — more games = fewer wrong verdicts",
    },
    "gate_threshold": {
        "key": "gate_threshold",
        "order": 30,
        "label": "Promotion Threshold",
        "category": "gate",
        "category_label": "Champion vs Candidate",
        "min": 0.5,
        "max": 0.8,
        "step": 0.01,
        "default": 0.55,
        "type": "float",
        "hint": "Win share the candidate must beat to become champion",
    },
    "gate_simulations": {
        "key": "gate_simulations",
        "order": 40,
        "label": "Gate Simulations",
        "category": "gate",
        "category_label": "Champion vs Candidate",
        "min": 10,
        "max": 400,
        "step": 5,
        "default": 50,
        "type": "int",
        "hint": "Sims per move in gate games — spend budget on games, not sims",
    },
    "gate_stall_warning": {
        "key": "gate_stall_warning",
        "order": 50,
        "label": "Stall Breaker",
        "category": "gate",
        "category_label": "Champion vs Candidate",
        "min": 2,
        "max": 20,
        "step": 1,
        "default": 5,
        "type": "int",
        "hint": "Consecutive rejections before the candidate resets to champion",
    },
    # --- Compute -----------------------------------------------------------
    # Both game-playing phases (self-play, gate) run their games across a
    # process pool on CPU, so one worker count governs both rather than each
    # phase carrying its own knob. The GPU, when there is one, trains the
    # network in the main process — it is not what this slider scales.
    "num_parallel_workers": {
        "key": "num_parallel_workers",
        "order": 10,
        "label": "Parallel Workers",
        "category": "compute",
        "category_label": "Compute & Parallelism",
        "min": 1,
        "max": MAX_PARALLEL_WORKERS,
        "step": 1,
        "default": DEFAULT_PARALLEL_WORKERS,
        "type": "int",
        "hint": "CPU processes playing games at once in self-play and the gate "
                "(1 = sequential). Capped at this machine's core count.",
    },
    # --- Move restrictions -------------------------------------------------
    # Heuristics that remove moves from the bot's action set. These are NOT
    # rules of Go: they never affect human play, stored games or replays, only
    # what the search is allowed to consider.
    "restrict_eye_fill": {
        "key": "restrict_eye_fill",
        "order": 10,
        "label": "No Own-Eye Filling",
        "category": "restrictions",
        "category_label": "Move Restrictions",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": False,
        "type": "bool",
        "hint": "On = bot may never fill one of its own two eyes (killing a live group)",
    },
    "restrict_self_atari": {
        "key": "restrict_self_atari",
        "order": 20,
        "label": "No Pointless Self-Atari",
        "category": "restrictions",
        "category_label": "Move Restrictions",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": False,
        "type": "bool",
        "hint": "On = bot may not walk a group into atari for nothing (heuristic, not a rule)",
    },
    "self_atari_max_stones": {
        "key": "self_atari_max_stones",
        "order": 30,
        "label": "Sacrifice Size Allowed",
        "category": "restrictions",
        "category_label": "Move Restrictions",
        "min": 1,
        "max": 4,
        "step": 1,
        "default": 1,
        "type": "int",
        "hint": "Self-ataris up to this many stones stay legal — keeps throw-ins playable",
    },
    # --- Mercy rule (self-play resignation) --------------------------------
    # Ends a decided self-play game early instead of paying an MCTS search per
    # move for an endgame nobody can lose. Self-play only — never the gate or
    # the random-bot eval. See the block comment in ai/self_play.py.
    "resign_enabled": {
        "key": "resign_enabled",
        "order": 10,
        "label": "Mercy Rule",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": False,
        "type": "bool",
        "hint": "On = stop a self-play game once a side's own search says it is lost",
    },
    "resign_threshold": {
        "key": "resign_threshold",
        "order": 20,
        "label": "Resign Confidence",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 0.5,
        "max": 0.99,
        "step": 0.01,
        "default": 0.90,
        "type": "float",
        "hint": "0.90 = resign at a 5% self-assessed win rate; higher is more cautious",
    },
    "resign_consecutive": {
        "key": "resign_consecutive",
        "order": 30,
        "label": "Confirming Moves",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 4,
        "type": "int",
        "hint": "Own moves in a row that must agree — filters one-move value spikes",
    },
    "resign_min_move_factor": {
        "key": "resign_min_move_factor",
        "order": 40,
        "label": "Earliest Resign",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 0.25,
        "max": 2.0,
        "step": 0.05,
        "default": 1.0,
        "type": "float",
        "hint": "x board area. 1.0 = never costs a training sample; below 1.0 trades data for speed",
    },
    "resign_both_sides": {
        "key": "resign_both_sides",
        "order": 50,
        "label": "Both Sides Must Agree",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": True,
        "type": "bool",
        "hint": "Winner must also see itself winning — one broken value head can't end games",
    },
    "resign_playout_fraction": {
        "key": "resign_playout_fraction",
        "order": 60,
        "label": "Playout Check",
        "category": "resign",
        "category_label": "Mercy Rule (Resignation)",
        "min": 0.0,
        "max": 0.5,
        "step": 0.05,
        "default": 0.1,
        "type": "float",
        "hint": "Share played out anyway to measure wrong resignations — 0 = flying blind",
    },
    "learning_rate": {
        "key": "learning_rate",
        "order": 10,
        "label": "Learning Rate",
        "category": "network",
        "category_label": "Neural Network Optimization",
        "min": 0.0001,
        "max": 0.02,
        "step": 0.0001,
        "default": 0.002,
        "type": "float",
        "hint": "Gradient update step size — lower if training is unstable",
    },
    # --- Game recording ----------------------------------------------------
    # What a phase writes to DISK, not what it plays. Every game is summarized
    # into games/index.jsonl either way, and that index is what the charts on
    # this page are drawn from — so turning these off costs the replay of those
    # games in the review page and nothing else.
    "record_self_play_games": {
        "key": "record_self_play_games",
        "order": 10,
        "label": "Record Self-Play Games",
        "category": "storage",
        "category_label": "Game Recording",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": True,
        "type": "bool",
        "hint": "Off = keep the stats, skip the replays (~12 KB/game on 9×9). "
                "Turn it off at high game counts — the charts do not read these files",
    },
    "record_gate_games": {
        "key": "record_gate_games",
        "order": 20,
        "label": "Record Gate Games",
        "category": "storage",
        "category_label": "Game Recording",
        "min": 0,
        "max": 1,
        "step": 1,
        "default": True,
        "type": "bool",
        "hint": "Champion vs candidate matches — the bulkiest phase, 20 games per iteration",
    },
}

CATEGORIES = [
    {"key": "mcts", "label": "MCTS & Search Strategy"},
    {"key": "temperature", "label": "Temperature & Randomness"},
    {"key": "volume", "label": "Self-Play Volume"},
    {"key": "gate", "label": "Champion vs Candidate"},
    {"key": "compute", "label": "Compute & Parallelism"},
    {"key": "restrictions", "label": "Move Restrictions"},
    {"key": "resign", "label": "Mercy Rule (Resignation)"},
    {"key": "network", "label": "Neural Network Optimization"},
    {"key": "storage", "label": "Game Recording"},
]

# Keys every editing surface (create / edit / live tune) accepts. Derived from
# PARAM_BOUNDS so adding a slider above is the only edit needed.
PARAM_KEYS = tuple(PARAM_BOUNDS.keys())


def sanitize_params(raw_params: dict) -> dict:
    """
    Sanitize and clamp input training parameters according to PARAM_BOUNDS.
    Returns a dict with clamped and typed values.
    """
    sanitized = {}
    for key, spec in PARAM_BOUNDS.items():
        if key not in raw_params or raw_params[key] is None:
            continue
        val = raw_params[key]
        if spec["type"] == "bool":
            # Accept real booleans, 0/1 from a slider, and "true"/"false" from
            # a JSON client that stringified the value.
            if isinstance(val, str):
                sanitized[key] = val.strip().lower() in ("1", "true", "yes", "on")
            else:
                sanitized[key] = bool(val)
            continue
        try:
            if spec["type"] == "int":
                num_val = int(round(float(val)))
            else:
                num_val = float(val)
        except (ValueError, TypeError):
            continue
        clamped = max(spec["min"], min(spec["max"], num_val))
        if spec["type"] == "int":
            sanitized[key] = int(clamped)
        else:
            sanitized[key] = round(clamped, 6)
    return sanitized
