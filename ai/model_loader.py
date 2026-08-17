"""
model_loader.py — Load any model's network by id, outside the Trainer.

The Trainer owns the network for the ACTIVE model only. Bot-vs-bot matches need
arbitrary models loaded side by side, on CPU, without touching training state,
so the loading lives here and is shared by anything that needs a playable
opponent (matches today; an OGS bridge later).

Networks are cached per (model_id, weights mtime) — a model whose weights.pt has
been rewritten by a training iteration is reloaded automatically, and repeated
matches against the same unchanged model reuse one network.
"""

import os
import threading
from typing import Optional, Tuple

import torch

from config import Config
from elo_history import EloEntry
from model_manager import ModelManager
from ai.network import GoNetwork
from ai.checkpoint import load_weights
from ai.mercy_rule import MercyRule
from ai.players import ModelPlayer


# (model_id, mtime) -> network. Guarded by _cache_lock; matches run on their own
# threads and two of them may load the same model at once.
_network_cache = {}
_cache_lock = threading.Lock()


def _weights_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def load_model_network(model_id: str, manager: Optional[ModelManager] = None):
    """
    Load a model's playing network (the gated champion when one is stored).

    Returns (network, model_info, config). The network is on CPU and in eval
    mode — matches play unbatched single positions, where CPU beats MPS.

    An untrained model (no weights.pt yet) yields a randomly initialised
    network rather than an error: it is a legitimate, very weak opponent, and
    that is exactly what its ~500 Elo says.
    """
    mgr = manager or ModelManager()
    info = mgr.get_model(model_id)
    if info is None:
        raise ValueError(f"Model not found: {model_id}")

    model_dir = mgr.get_model_dir(model_id)
    config = Config.from_model(info, model_dir)
    weights_path = config.paths.weights_path
    cache_key = (model_id, _weights_mtime(weights_path))

    with _cache_lock:
        cached = _network_cache.get(cache_key)
    if cached is not None:
        return cached, info, config

    network = GoNetwork(
        board_size=config.board.size,
        num_input_planes=config.network.num_input_planes,
        input_features=config.network.input_features,
        num_res_blocks=config.network.num_res_blocks,
        num_filters=config.network.num_filters,
        value_head_hidden=config.network.value_head_hidden,
    ).to("cpu")

    if os.path.isfile(weights_path):
        # Raises on an architecture mismatch — the caller surfaces that as a
        # clear error rather than playing a match with garbage weights.
        meta = load_weights(weights_path, network, optimizer=None, device="cpu")
        champion = meta.get('champion_state_dict') if meta else None
        if champion:
            # The champion is what actually plays: it is the gated network the
            # model uses for its own self-play and human games.
            network.load_state_dict(champion)

    network.eval()

    with _cache_lock:
        # Drop stale entries for this model (old weights are not coming back).
        for key in [k for k in _network_cache if k[0] == model_id]:
            _network_cache.pop(key, None)
        _network_cache[cache_key] = network

    return network, info, config


def invalidate_model_network(model_id: Optional[str] = None) -> None:
    """Drop cached networks — for one model, or all of them."""
    with _cache_lock:
        if model_id is None:
            _network_cache.clear()
            return
        for key in [k for k in _network_cache if k[0] == model_id]:
            _network_cache.pop(key, None)


def persist_model_rating(model_id: str, rating: float,
                         event: Optional[EloEntry] = None) -> None:
    """
    Record a model's new Elo — as a ledger entry, not an overwritten float.

    The entry names the opponent, the outcome and the game record it came from,
    so the rating is reconstructible and every point on the model's Elo curve
    links to the game that earned it. `ModelManager.record_elo_result` also
    refreshes the cached `elo` in config.json, leaving iteration and
    total_games (which training owns) alone.

    A bare `rating` with no event still works — a manual correction has nothing
    to link to — and is written as a noted entry rather than silently.
    """
    mgr = ModelManager()
    info = mgr.get_model(model_id)
    if info is None:
        return

    if event is None:
        event = EloEntry(new_elo=rating,
                         elo_delta=rating - float(info.elo),
                         note='Rating set directly')
    mgr.record_elo_result(model_id, event)


def build_model_player(model_id: str, num_simulations: Optional[int] = None,
                       board_size: Optional[int] = None,
                       label: Optional[str] = None,
                       manager: Optional[ModelManager] = None,
                       mercy_resign: Optional[bool] = None) -> ModelPlayer:
    """
    Build a ready-to-play ModelPlayer for a stored model.

    `board_size`, when given, is the board the match will actually be played on;
    a model trained for a different size cannot play it, so that is rejected
    here rather than producing a shape error mid-game.

    `mercy_resign` turns the mercy rule on or off for this player, overriding
    the model's own `resign_enabled` (a training setting); None defers to it.
    """
    network, info, config = load_model_network(model_id, manager=manager)

    if board_size is not None and int(board_size) != info.board_size:
        raise ValueError(
            f"{info.name} is a {info.board_size}x{info.board_size} model and "
            f"cannot play on {board_size}x{board_size}"
        )

    sims = num_simulations or config.mcts.num_simulations
    return ModelPlayer(
        model_id=model_id,
        name=label or info.name,
        network=network,
        board_size=info.board_size,
        num_simulations=int(sims),
        c_puct=config.mcts.c_puct,
        restrict_eye_fill=bool(config.training.restrict_eye_fill),
        device="cpu",
        rating=info.elo,
        iteration=info.iteration,
        meta={'kyu_rank': info.kyu_rank, 'ruleset': info.ruleset, 'komi': info.komi},
        rating_sink=persist_model_rating,
        # Where this model's copy of a match record lands, so its ledger entry
        # can point at a game the review page can actually open.
        games_dir=config.paths.games_dir,
        mercy=MercyRule.from_config(config, info.board_size, enabled=mercy_resign),
    )
