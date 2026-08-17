"""
Global configuration for the Go AI project.

All tuneable parameters live here. Change board size, network architecture,
training hyperparameters, and paths in one place.

DEVICE:
- The training device (CUDA / MPS / CPU) is detected in device_info.py, which
  also owns the worker-count policy. See its module docstring for why the
  worker count is a CPU number even on a GPU machine.
- Expect ~5-10 self-play games/min on 9x9 with 200 MCTS simulations on an M2.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Literal, Optional

import device_info


def _get_device() -> str:
    """
    Pick the best available device: CUDA > MPS > CPU.

    Delegates to device_info.detect() so the string here, the badge in the web
    UI and the hardware report all come from one probe.
    """
    return device_info.detect().torch_device


@dataclass
class BoardConfig:
    """Go board settings."""
    size: int = 9                          # Default board size (7, 9, 13, 15, 19)
    supported_sizes: tuple = (7, 9, 13, 15, 19)
    komi: float = 6.5                      # Compensation for white (standard)
    scoring_method: str = "chinese"        # "chinese" or "japanese"


@dataclass
class NetworkConfig:
    """Neural network architecture — kept small for CPU/MPS."""
    num_res_blocks: int = 4                # Residual blocks (4 is good for 9x9)
    num_filters: int = 64                  # Convolutional filters per layer
    # Versioned input encoding — see game/features.py. Frozen per model at
    # creation: changing it changes input_conv.in_channels, which makes an
    # existing weights.pt shape-incompatible.
    input_features: str = "v1_10"
    # Derived from input_features in __post_init__; never set it by hand.
    num_input_planes: int = 10
    value_head_hidden: int = 64            # Hidden layer size in value head

    def __post_init__(self):
        # One source of truth. Keeping the plane count derived means a config
        # can never claim 10 planes while asking for a 12-plane encoding.
        from game.features import num_planes
        self.num_input_planes = num_planes(self.input_features)


# ------------------------------------------------------------------
# Network size presets — chosen at model-creation time.
#
# Each preset maps to a concrete (num_res_blocks, num_filters,
# value_head_hidden). The approx_params figures are measured on a 9x9
# board; the residual body dominates and is board-size independent, so
# these are good ballpark numbers for other sizes too. The exact count
# for a given board size is computed live via GoNetwork.count_parameters().
#
# `small` is the historical default (≈320k params) — leaving it unchanged
# keeps every previously trained 4x64 model loadable.
# ------------------------------------------------------------------
NETWORK_PRESETS = {
    "tiny":   {"num_res_blocks": 3,  "num_filters": 32,  "value_head_hidden": 32,
               "approx_params": 74_749,    "label": "Tiny", "note": "fastest, weakest ceiling"},
    "small":  {"num_res_blocks": 4,  "num_filters": 64,  "value_head_hidden": 64,
               "approx_params": 320_701,   "label": "Small", "note": "default — good for 9×9"},
    "medium": {"num_res_blocks": 6,  "num_filters": 96,  "value_head_hidden": 96,
               "approx_params": 1_028_093, "label": "Medium", "note": "1M+ params, stronger, slower"},
    "large":  {"num_res_blocks": 8,  "num_filters": 128, "value_head_hidden": 128,
               "approx_params": 2_399_549, "label": "Large", "note": "much slower self-play on M2"},
    "huge":   {"num_res_blocks": 10, "num_filters": 160, "value_head_hidden": 128,
               "approx_params": 4_653_597, "label": "Huge", "note": "research-scale; very slow on M2"},
}

# Ordered from smallest to largest — the order a slider should present them.
NETWORK_PRESET_ORDER = ["tiny", "small", "medium", "large", "huge"]

DEFAULT_NETWORK_PRESET = "small"

# Bounds for custom (non-preset) architectures, to keep the web API from
# building a network that would exhaust memory or never finish an iteration.
NETWORK_ARCH_BOUNDS = {
    "num_res_blocks": (1, 20),
    "num_filters": (8, 256),
    "value_head_hidden": (8, 512),
}


def simulations_for_board(mcts_cfg, board_size: int) -> int:
    """
    Effective MCTS simulation count for a board size.

    When auto_scale_simulations is on, `num_simulations` is the 9x9 baseline
    and is scaled by board_area / 81, clamped to [min_simulations,
    max_simulations]. This keeps the visit distribution (and therefore the
    policy training target) sharp on larger boards without hand-tuning per size.
    """
    base = mcts_cfg.num_simulations
    if not getattr(mcts_cfg, "auto_scale_simulations", False):
        return base
    area = board_size * board_size
    scaled = int(round(base * area / 81.0))
    lo = getattr(mcts_cfg, "min_simulations", base)
    hi = getattr(mcts_cfg, "max_simulations", base)
    return max(lo, min(hi, scaled))


def resolve_network_preset(preset: str) -> dict:
    """
    Map a preset name to concrete architecture params.

    Returns a dict with num_res_blocks, num_filters, value_head_hidden.
    Falls back to the default preset for unknown names.
    """
    spec = NETWORK_PRESETS.get(preset) or NETWORK_PRESETS[DEFAULT_NETWORK_PRESET]
    return {
        "num_res_blocks": spec["num_res_blocks"],
        "num_filters": spec["num_filters"],
        "value_head_hidden": spec["value_head_hidden"],
    }


def clamp_network_arch(num_res_blocks: int, num_filters: int,
                       value_head_hidden: int) -> dict:
    """Clamp a custom architecture into the supported bounds."""
    def _clamp(val, lo, hi):
        return max(lo, min(hi, int(val)))
    rb_lo, rb_hi = NETWORK_ARCH_BOUNDS["num_res_blocks"]
    nf_lo, nf_hi = NETWORK_ARCH_BOUNDS["num_filters"]
    vh_lo, vh_hi = NETWORK_ARCH_BOUNDS["value_head_hidden"]
    return {
        "num_res_blocks": _clamp(num_res_blocks, rb_lo, rb_hi),
        "num_filters": _clamp(num_filters, nf_lo, nf_hi),
        "value_head_hidden": _clamp(value_head_hidden, vh_lo, vh_hi),
    }


@dataclass
class MCTSConfig:
    """Monte Carlo Tree Search parameters."""
    num_simulations: int = 96              # Base simulations per move (see auto_scale_simulations)
    # Scale simulations to board area so the visit distribution stays sharp
    # enough to be a usable policy target. On a 9x9 board (81 points) a low
    # budget spreads visits ~uniformly across ~80 legal moves, and the network
    # ends up training on near-random targets. When True, num_simulations is
    # treated as the 9x9 baseline and scaled by (board_area / 81).
    auto_scale_simulations: bool = True
    min_simulations: int = 48              # Floor after scaling (small boards)
    max_simulations: int = 800             # Ceiling after scaling (19x19 sanity cap)
    c_puct: float = 1.5                    # Exploration constant (PUCT formula)
    # First Play Urgency reduction: value assigned to not-yet-visited children,
    # relative to the parent's value. Optimistic init (0.0) forces MCTS to try
    # every legal move once before deepening — on a large board with a small
    # sim budget that keeps visits flat and the policy target near-uniform.
    # A positive reduction makes unvisited moves look slightly worse than the
    # parent, so search concentrates on promising moves and produces a sharper,
    # more learnable target. 0.0 recovers the original AlphaZero behavior.
    fpu_reduction: float = 0.35
    dirichlet_alpha: float = 0.3           # Exploration noise (0.03 for 19x19, 0.3 for 9x9)
    dirichlet_epsilon: float = 0.25        # Fraction of noise to mix in at root
    # Temperature controls move randomness:
    # - temp=0.5 early game → sharpens targets to combat low simulation counts
    # - temp→0 late game → pick the most-visited move (greedy)
    temperature_threshold: int = 15        # After this many moves, temp drops to near-zero
    temperature_init: float = 0.5
    temperature_final: float = 0.001
    # tau used to build the POLICY TRAINING TARGET, kept separate from the three
    # settings above (which govern only which move gets PLAYED). AlphaZero
    # trains on pi = N/sum(N), i.e. tau = 1.0, regardless of the play
    # temperature: the relative visit counts of the runners-up ARE the training
    # signal, and any tau < 1 discards them.
    #
    # This used to be tied to the play temperature. The result, measured on a
    # real 50k buffer: 62% of targets one-hot, median 3 actions with any mass.
    # The policy head then behaviour-clones its own argmax, its entropy decays
    # monotonically, and the model overfits to the narrow move set it shares
    # with its self-play opponent — invisible to the promotion gate, obvious
    # against any foreign opponent. Leave at 1.0 unless experimenting.
    policy_target_temperature: float = 1.0


@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""
    device: str = field(default_factory=_get_device)
    
    # Self-play
    num_self_play_games: int = 5           # Games per iteration before training (Reduced for faster iterations)
    # Worker processes for EVERY game-playing phase — self-play and the
    # promotion gate. Both play independent games on CPU, so they share one
    # setting. Capped at the CPU count and at the number of games in that phase.
    #
    # The default is derived from THIS machine rather than hardcoded: a model
    # config that stores no value (or a null) should run well on whatever
    # hardware it is opened on, and 4 workers on a 20-thread desktop leaves
    # most of the machine idle. device_info.recommended_workers() reserves a
    # couple of cores for the server and trainer threads.
    #
    # Evaluated at class-creation time, so this stays a plain int default and
    # `_model_default` can keep reading it off __dataclass_fields__.
    num_parallel_workers: int = device_info.recommended_workers()
    
    # Replay buffer
    replay_buffer_size: int = 50_000       # Max training samples to keep
    
    # Game storage — keep 1 out of every N self-play games for replay/review.
    # We want to save every single game per user request.
    game_store_every_n: int = 1

    # --- Game recording ---------------------------------------------------
    # Whether a phase writes its full game records (move lists, per-move win
    # rates) to disk. Off means the games are still PLAYED and still indexed —
    # every chart on the training page keeps every point, because those are
    # drawn from games/index.jsonl, not from the records (see
    # ai/game_index.py). What is lost is the ability to replay those games in
    # the review page.
    #
    # The bytes are the reason to turn them off: a 9x9 self-play record is
    # ~12 KB and a hundred-iteration model has produced over a gigabyte of
    # them. The gate is the bigger offender per iteration — it plays
    # gate_games (20 by default) to self-play's num_self_play_games.
    record_self_play_games: bool = True
    record_gate_games: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.002
    # Length of the ONE cosine descent, in iterations: the LR falls from
    # learning_rate to the scheduler's eta_min (1e-5) over this many iterations
    # and then holds there — Trainer stops stepping the scheduler at T_max
    # rather than letting CosineAnnealingLR's periodicity carry it back up to
    # full strength, which it does by 2 * T_max.
    #
    # Set this to roughly the number of iterations you actually intend to run.
    # Leaving it at 100 for a 400-iteration run is not broken, but it means three
    # quarters of the run happens at the 1e-5 floor.
    lr_decay_steps: int = 100
    weight_decay: float = 1e-4             # L2 regularization
    num_epochs_per_iteration: int = 3      # Training epochs per self-play iteration
    
    # Evaluation
    eval_games: int = 4                    # Games to play for evaluation matches (Reduced for speed)
    # Reflection report: generate a progress summary every N games.
    reflection_interval_games: int = 50

    # --- Promotion gate ---------------------------------------------------
    # A freshly trained candidate only becomes the self-play generator if it
    # actually beats the current champion head-to-head. Without this, every
    # update is accepted unconditionally and the loop can walk into a
    # degenerate network with nothing able to detect or reject it.
    gate_enabled: bool = True
    # Statistical power depends on the NUMBER of games, not sims per game, while
    # cost is roughly (games x sims). Measured on a 9x9 M2: ~35s/game at 100
    # sims, ~18s/game at 50. So 20 games x 50 sims costs the same wall-clock as
    # 10 x 100 but makes strictly better decisions at every effect size —
    # e.g. a clearly worse candidate (true p=0.35) is blocked 94.7% of the time
    # instead of 90.5%, and a genuinely better one (p=0.65) accepted 87.8%
    # instead of 75.1%.
    gate_games: int = 20                   # Head-to-head games (colors alternate)
    gate_threshold: float = 0.55           # Candidate must win >= this share
    # Sims per move during the gate match. Should track num_simulations: the
    # gate decides which network generates every future self-play game, so
    # measuring it at a search depth the model never plays at selects for the
    # wrong thing (a low sim budget can't overrule the prior, so it flatters
    # whichever candidate has the most confident policy — exactly the network
    # you don't want when its confidence is the problem). Trainer warns when
    # this drops below half of num_simulations.
    gate_simulations: int = 200
    # Consecutive rejections before warning that training has stalled.
    gate_stall_warning: int = 5

    # --- Value target blending --------------------------------------------
    # The value head is trained on `lambda * z + (1 - lambda) * q`, where z is
    # the final game outcome (±1) and q is the MCTS root evaluation at that
    # position (both from the mover's perspective).
    #
    # This is the fix for value-head collapse AT THE SOURCE. With outcome-only
    # targets, every position in a game shares one label, so once one colour
    # wins nearly every game the label is predictable from the turn-colour
    # plane alone and the network stops reading the board. Because q varies
    # position-to-position even when z is constant across the whole game, the
    # colour shortcut stops being a good fit and the network is forced to
    # evaluate the actual position. (This is what KataGo does.)
    #
    # 1.0 = pure game outcome (original AlphaZero, collapse-prone).
    value_target_outcome_weight: float = 0.6

    # --- Collapse guard ---------------------------------------------------
    # Tripwires for value-head collapse: when one colour wins nearly every
    # game, the value head can drive its loss to ~0 by reading only the
    # turn-colour plane, which makes MCTS blind. These detect that state.
    collapse_guard_enabled: bool = True
    collapse_value_std_min: float = 0.05   # Min spread of value head over positions
    # Calibrated against hot-boi's 48-iteration history: White's pass rate was
    # 6-20% while healthy and 31-46% once collapsed. 0.15 fired on 6 of the 15
    # healthy iterations (alarm fatigue); 0.25 sits in the empty band between
    # the two regimes — 0 false alarms, 0 missed collapses, and it still fires
    # at iteration 22, well before the confirmed collapse at 29.
    collapse_pass_rate_max: float = 0.25   # Max share of a colour's moves that are passes
    collapse_probe_positions: int = 128    # Buffer positions sampled for the probe
    collapse_auto_stop: bool = False       # True = halt training when tripped
    # --- Policy-head collapse ---------------------------------------------
    # The value-head tripwires above miss the failure that actually degraded
    # hero-of-time: the POLICY head narrowing until only a handful of moves
    # carry any prior. Nothing else in the loop sees it. Self-play looks fine
    # (both sides share the shrinking move set), the gate reads ~50% (a network
    # is no worse than itself), and the value head stays perfectly healthy. It
    # is visible only against a foreign opponent, whose replies fall outside the
    # narrow prior — where PUCT cannot recover them because their prior is ~0.
    #
    # Two separate probes, because they fail in that order: the LABEL narrows
    # first (a tau < 1 policy target, now fixed), and the NETWORK follows.
    #
    # LABEL floor, measured on 9x9 with this repo's own settings rather than
    # assumed. Re-running iteration 185's positions at tau=1 gives a mean of
    # 0.95 nats over a support of ~7 moves — lower than a textbook AlphaZero
    # target because fpu_reduction (0.35) deliberately concentrates the search.
    # The collapsed buffer this was found on measured 0.19 with 62% of labels
    # literally one-hot and a support of 3. 0.55 sits in the empty band between
    # those two regimes; anything nearer 0.9 would fire on healthy runs, since
    # 42% of individual healthy positions are already below 0.9.
    collapse_target_entropy_min: float = 0.55
    # NETWORK floor. Deliberately low: the only measurements available are from
    # a run that was already degrading (1.40 nats at iteration 128, 1.05 at
    # 185), so there is no honest "healthy" value to calibrate against, and a
    # guessed threshold either cries wolf or fires far too late. This is a
    # last-resort floor meaning "definitely broken", not a health check — the
    # trend tripwire below is what actually catches the failure.
    collapse_policy_entropy_min: float = 0.5
    # TREND tripwire, which needs no calibration and is what would have caught
    # hero-of-time. A policy head that is narrowing gives up entropy steadily,
    # so compare the current value against the mean of the window ending
    # `collapse_entropy_trend_lookback` iterations ago and fire on a sustained
    # relative drop. The real regression was -25% over 57 iterations.
    collapse_entropy_trend_lookback: int = 25
    collapse_entropy_trend_drop: float = 0.15  # Relative fall that trips it

    # --- Move restrictions (heuristics, not rules of Go) -------------------
    # When True, the bot may not play into one of its own chain's only two
    # eyes — the move that turns a provably-alive group into a dead one. It is
    # enforced by removing the move from the action set, so MCTS never expands
    # it, never gives it a prior and never puts it in a policy target.
    # See game/eyes.py for the exact eye definition and its proof.
    # Default False = identical behaviour to before this setting existed.
    restrict_eye_fill: bool = False

    # When True, the bot may not play a move that captures nothing and leaves
    # the group it joins on a single liberty, provided that group is larger
    # than self_atari_max_stones. Enforced the same way as restrict_eye_fill —
    # the move is removed from the action set, so MCTS never expands it, never
    # gives it a prior and never puts it in a policy target.
    #
    # UNLIKE restrict_eye_fill THIS IS NOT PROVABLY SAFE. It is a tuned
    # assumption: self-atari is genuinely correct in nakade, seki manipulation,
    # eye-space reduction and ko-threat creation. The promotion gate CANNOT
    # detect over-restriction, because candidate and champion both play under
    # the filter — validate it with the twin-model A/B in HEURISTICS.md.
    # Default False = identical behaviour to before this setting existed.
    restrict_self_atari: bool = False
    # Sacrifices of at most this many stones stay playable, which is what keeps
    # throw-ins — the main small-group tesuji — available.
    self_atari_max_stones: int = 1

    # --- Mercy rule (self-play resignation) -------------------------------
    # Stop a self-play game once one side's own search says it is lost, instead
    # of paying a full MCTS search per move for a decided endgame. Self-play
    # ONLY — the gate and the random-bot eval never resign, because there a
    # wrong verdict corrupts a measurement rather than one training game.
    #
    # The cost of the tail being cut is zero at the default factor: training
    # samples stop at move board_size², so nothing that produces data is ever
    # reached. What resignation does risk is the outcome LABEL, which is why
    # resign_playout_fraction exists — see ai/self_play.py.
    resign_enabled: bool = False
    resign_threshold: float = 0.90         # Root value <= -this counts as lost
    resign_consecutive: int = 4            # Consecutive own moves that must agree
    resign_min_move_factor: float = 1.0    # Earliest resign = factor x board area
    resign_both_sides: bool = True         # Winner must agree it is winning
    resign_playout_fraction: float = 0.1   # Share played out to measure mistakes

    # Elo
    elo_anchor: int = 500                  # Random bot starts at this Elo (≈30 kyu)


def _model_default(training_params, key: str):
    """
    Read an optional per-model training override, falling back to the default
    declared on TrainingConfig for that same field.
    """
    fallback = TrainingConfig.__dataclass_fields__[key].default
    value = getattr(training_params, key, None)
    return fallback if value is None else value


@dataclass
class PathConfig:
    """File system paths — can point into a model directory."""
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    model_dir: Optional[str] = None  # If set, all data goes here
    
    @property
    def data_dir(self) -> str:
        if self.model_dir:
            return self.model_dir
        return os.path.join(self.project_root, "data")
    
    @property
    def games_dir(self) -> str:
        return os.path.join(self.data_dir, "games")
    
    @property
    def logs_dir(self) -> str:
        return os.path.join(self.data_dir, "logs")
    
    @property
    def weights_path(self) -> str:
        """Path to the single weights file."""
        return os.path.join(self.data_dir, "weights.pt")
    
    def ensure_dirs(self):
        """Create all data directories if they don't exist."""
        for d in [self.data_dir, self.games_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    """Top-level config that bundles everything."""
    board: BoardConfig = field(default_factory=BoardConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def __post_init__(self):
        """Validate and create directories."""
        assert self.board.size in self.board.supported_sizes, \
            f"Board size {self.board.size} not in {self.board.supported_sizes}"
        self.paths.ensure_dirs()

    @classmethod
    def from_model(cls, model_info, model_dir: str) -> "Config":
        """
        Build a Config from a ModelInfo (from model_manager).
        
        Routes all file paths into the model's directory.
        """
        board = BoardConfig(
            size=model_info.board_size,
            komi=model_info.komi,
            scoring_method=model_info.ruleset,
        )
        mcts = MCTSConfig(
            num_simulations=model_info.training.num_simulations,
            c_puct=model_info.training.c_puct,
            temperature_threshold=getattr(model_info.training, "temperature_threshold", 30),
            temperature_init=getattr(model_info.training, "temperature_init", 0.8),
            temperature_final=getattr(model_info.training, "temperature_final", 0.1),
            # Models created before the split have no value here and get 1.0 —
            # the AlphaZero default — NOT their old temperature_final. Their
            # existing buffer of sharpened labels ages out of the FIFO on its
            # own; re-sharpening new ones to match it would just extend the
            # problem.
            policy_target_temperature=getattr(
                model_info.training, "policy_target_temperature", 1.0),
        )
        training = TrainingConfig(
            num_self_play_games=model_info.training.num_self_play_games,
            eval_games=model_info.training.eval_games,
            learning_rate=model_info.training.learning_rate,
            batch_size=model_info.training.batch_size,
            num_epochs_per_iteration=model_info.training.num_epochs_per_iteration,
            replay_buffer_size=model_info.training.replay_buffer_size,
            reflection_interval_games=model_info.training.reflection_interval_games,
            # Gate / collapse-guard settings are optional per-model overrides;
            # models that predate them fall back to the defaults declared on
            # TrainingConfig itself — never to a second, hand-written copy of
            # them, which is how gate_games/gate_simulations previously drifted
            # away from the values the dataclass documents.
            gate_enabled=_model_default(model_info.training, "gate_enabled"),
            gate_games=_model_default(model_info.training, "gate_games"),
            gate_threshold=_model_default(model_info.training, "gate_threshold"),
            gate_simulations=_model_default(model_info.training, "gate_simulations"),
            gate_stall_warning=_model_default(model_info.training, "gate_stall_warning"),
            num_parallel_workers=_model_default(model_info.training, "num_parallel_workers"),
            record_self_play_games=_model_default(model_info.training, "record_self_play_games"),
            record_gate_games=_model_default(model_info.training, "record_gate_games"),
            collapse_guard_enabled=_model_default(model_info.training, "collapse_guard_enabled"),
            collapse_auto_stop=_model_default(model_info.training, "collapse_auto_stop"),
            restrict_eye_fill=_model_default(model_info.training, "restrict_eye_fill"),
            restrict_self_atari=_model_default(model_info.training, "restrict_self_atari"),
            self_atari_max_stones=_model_default(model_info.training, "self_atari_max_stones"),
            resign_enabled=_model_default(model_info.training, "resign_enabled"),
            resign_threshold=_model_default(model_info.training, "resign_threshold"),
            resign_consecutive=_model_default(model_info.training, "resign_consecutive"),
            resign_min_move_factor=_model_default(model_info.training, "resign_min_move_factor"),
            resign_both_sides=_model_default(model_info.training, "resign_both_sides"),
            resign_playout_fraction=_model_default(model_info.training, "resign_playout_fraction"),
        )
        # Network architecture is stored per-model so that a saved weights.pt
        # is always rebuilt into the matching network. Fall back to defaults
        # (the historical 4x64 "small" net) when a model predates this field.
        net_info = getattr(model_info, "network", None)
        if net_info is not None:
            network = NetworkConfig(
                num_res_blocks=net_info.num_res_blocks,
                num_filters=net_info.num_filters,
                value_head_hidden=net_info.value_head_hidden,
                # Models created before the registry existed have no such field
                # and fall back to the 10-plane encoding they were trained with.
                input_features=getattr(net_info, "input_features", None) or "v1_10",
            )
        else:
            network = NetworkConfig()

        paths = PathConfig(model_dir=model_dir)
        return cls(board=board, network=network, mcts=mcts,
                   training=training, paths=paths)


# ------------------------------------------------------------------
# Elo-to-Kyu conversion table (approximate, based on EGD scale).
# 100 Elo points ≈ 1 rank. Random bot anchored at 500 Elo ≈ 30 kyu.
# ------------------------------------------------------------------
ELO_TO_KYU = {
    500: "30k", 600: "29k", 700: "28k", 800: "27k", 900: "26k",
    1000: "25k", 1100: "24k", 1200: "23k", 1300: "22k", 1400: "21k",
    1500: "20k", 1600: "19k", 1700: "18k", 1800: "17k", 1900: "16k",
    2000: "15k", 2100: "14k", 2200: "13k", 2300: "12k", 2400: "11k",
    2500: "10k", 2600: "9k",  2700: "8k",  2800: "7k",  2900: "6k",
    3000: "5k",  3100: "4k",  3200: "3k",  3300: "2k",  3400: "1k",
    3500: "1d",  3600: "2d",  3700: "3d",  3800: "4d",  3900: "5d",
}


def elo_to_rank(elo: float) -> str:
    """Convert an Elo rating to an approximate Go rank string."""
    # Clamp to table range
    clamped = max(500, min(3900, int(round(elo / 100) * 100)))
    return ELO_TO_KYU.get(clamped, f"~{clamped} Elo")
