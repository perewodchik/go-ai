"""
param_bounds.py — Centralized bounds and categories for training & MCTS parameters.

Edit this file if you need to adjust parameter bounds (min, max, step, category, default, label, hint).
Both backend validation and frontend sliders (Create Model, Edit Model, Live Tuning)
derive their configuration from this single authoritative file.
"""

PARAM_BOUNDS = {
    "num_simulations": {
        "key": "num_simulations",
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
    "num_self_play_games": {
        "key": "num_self_play_games",
        "label": "Self-Play Games",
        "category": "volume",
        "category_label": "Self-Play & Evaluation Volume",
        "min": 1,
        "max": 100,
        "step": 1,
        "default": 5,
        "type": "int",
        "hint": "Games the bot plays against itself per training iteration",
    },
    "eval_games": {
        "key": "eval_games",
        "label": "Eval Games",
        "category": "volume",
        "category_label": "Self-Play & Evaluation Volume",
        "min": 0,
        "max": 50,
        "step": 1,
        "default": 4,
        "type": "int",
        "hint": "Evaluation games vs random bot (0 = skip evaluation phase entirely)",
    },
    # --- Champion vs candidate (promotion gate) --------------------------
    # These decide whether a freshly trained candidate is allowed to replace
    # the champion that generates all self-play data. See the "Champion vs
    # Candidate" section of the Info page for what each one does to learning.
    "gate_enabled": {
        "key": "gate_enabled",
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
    # All three game-playing phases (self-play, gate, random eval) run their
    # games across a process pool on CPU, so one worker count governs all of
    # them rather than each phase carrying its own knob.
    "num_parallel_workers": {
        "key": "num_parallel_workers",
        "label": "Parallel Workers",
        "category": "compute",
        "category_label": "Compute & Parallelism",
        "min": 1,
        "max": 8,
        "step": 1,
        "default": 4,
        "type": "int",
        "hint": "Games played at once in self-play, gate and eval (1 = sequential)",
    },
    "learning_rate": {
        "key": "learning_rate",
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
}

CATEGORIES = [
    {"key": "mcts", "label": "MCTS & Search Strategy"},
    {"key": "temperature", "label": "Temperature & Randomness"},
    {"key": "volume", "label": "Self-Play & Evaluation Volume"},
    {"key": "gate", "label": "Champion vs Candidate"},
    {"key": "compute", "label": "Compute & Parallelism"},
    {"key": "network", "label": "Neural Network Optimization"},
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
