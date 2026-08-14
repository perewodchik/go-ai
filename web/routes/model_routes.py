"""
model_routes.py — REST API for model CRUD operations.
"""

import os
from flask import Blueprint, jsonify, request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model_manager import ModelManager
from config import (
    NETWORK_PRESETS, NETWORK_PRESET_ORDER, DEFAULT_NETWORK_PRESET,
    resolve_network_preset, clamp_network_arch,
)

model_bp = Blueprint('models', __name__)


def _get_manager() -> ModelManager:
    from web.app import model_manager
    return model_manager


def _estimate_params(board_size: int, num_res_blocks: int,
                     num_filters: int, value_head_hidden: int) -> int:
    """Exact parameter count for a given architecture + board size."""
    from ai.network import GoNetwork
    net = GoNetwork(
        board_size=board_size,
        num_input_planes=10,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        value_head_hidden=value_head_hidden,
    )
    return net.count_parameters()


def _resolve_network_params(data: dict, board_size: int):
    """
    Build the per-model network params dict from request data.

    Accepts either a `network_size` preset name, or explicit
    num_res_blocks/num_filters/value_head_hidden for a custom architecture
    (clamped to safe bounds). Returns (network_params_dict, error_or_None).
    """
    preset = data.get('network_size', DEFAULT_NETWORK_PRESET)

    if preset == 'custom':
        try:
            arch = clamp_network_arch(
                num_res_blocks=data.get('num_res_blocks', 4),
                num_filters=data.get('num_filters', 64),
                value_head_hidden=data.get('value_head_hidden', 64),
            )
        except (TypeError, ValueError):
            return None, 'Invalid custom network architecture'
        arch['size_preset'] = 'custom'
        return arch, None

    if preset not in NETWORK_PRESETS:
        return None, f'Invalid network size: {preset}'

    arch = resolve_network_preset(preset)
    arch['size_preset'] = preset
    return arch, None


@model_bp.route('/api/network_presets')
def network_presets():
    """
    Return the available network-size presets with live parameter counts for
    the requested board size, so the creation UI can show real numbers.
    """
    try:
        board_size = int(request.args.get('board_size', 9))
    except (TypeError, ValueError):
        board_size = 9
    if board_size not in (7, 9, 13, 15, 19):
        board_size = 9

    presets = []
    for key in NETWORK_PRESET_ORDER:
        spec = NETWORK_PRESETS[key]
        presets.append({
            'key': key,
            'label': spec['label'],
            'note': spec['note'],
            'num_res_blocks': spec['num_res_blocks'],
            'num_filters': spec['num_filters'],
            'value_head_hidden': spec['value_head_hidden'],
            'params': _estimate_params(
                board_size, spec['num_res_blocks'],
                spec['num_filters'], spec['value_head_hidden'],
            ),
        })
    return jsonify({
        'board_size': board_size,
        'default': DEFAULT_NETWORK_PRESET,
        'presets': presets,
    })


@model_bp.route('/api/list')
def list_models():
    mgr = _get_manager()
    models = mgr.list_models()
    return jsonify([m.to_dict() for m in models])


@model_bp.route('/api/active')
def get_active():
    mgr = _get_manager()
    model = mgr.get_active_model()
    if model:
        return jsonify(model.to_dict())
    return jsonify(None)


@model_bp.route('/api/create', methods=['POST'])
def create_model():
    data = request.get_json() or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    board_size = data.get('board_size', 9)
    if board_size not in (7, 9, 13, 15, 19):
        return jsonify({'error': f'Invalid board size: {board_size}'}), 400

    komi = data.get('komi', 6.5)
    ruleset = data.get('ruleset', 'chinese')
    if ruleset not in ('chinese', 'japanese'):
        return jsonify({'error': f'Invalid ruleset: {ruleset}'}), 400

    training_params = {}
    for key in ('num_self_play_games', 'eval_games', 'num_simulations',
                'c_puct', 'learning_rate', 'batch_size',
                'num_epochs_per_iteration', 'replay_buffer_size',
                'reflection_interval_games', 'temperature_threshold',
                'temperature_init', 'temperature_final'):
        if key in data:
            training_params[key] = data[key]

    network_params, net_err = _resolve_network_params(data, board_size)
    if net_err:
        return jsonify({'error': net_err}), 400

    mgr = _get_manager()
    info = mgr.create_model(
        name=name,
        board_size=board_size,
        komi=komi,
        ruleset=ruleset,
        training_params=training_params,
        network_params=network_params,
    )

    # Auto-select if it's the only model or no model is active
    if not mgr.get_active_model_id():
        mgr.set_active_model_id(info.id)
        # Trigger model switch in app
        from web.app import switch_model
        switch_model(info.id)

    return jsonify(info.to_dict()), 201


@model_bp.route('/api/<model_id>/select', methods=['POST'])
def select_model(model_id):
    mgr = _get_manager()
    model = mgr.get_model(model_id)
    if not model:
        return jsonify({'error': 'Model not found'}), 404

    from web.app import switch_model, trainer
    if trainer and trainer.is_running:
        return jsonify({'error': 'Stop training before switching models'}), 400

    mgr.set_active_model_id(model_id)
    switch_model(model_id)
    return jsonify({'message': f'Switched to model: {model.name}', 'model': model.to_dict()})


@model_bp.route('/api/<model_id>/rename', methods=['POST'])
def rename_model(model_id):
    data = request.get_json() or {}
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({'error': 'Name is required'}), 400

    mgr = _get_manager()
    info = mgr.rename_model(model_id, new_name)
    if not info:
        return jsonify({'error': 'Model not found'}), 404
    return jsonify(info.to_dict())


@model_bp.route('/api/<model_id>/update', methods=['POST'])
def update_model(model_id):
    data = request.get_json() or {}

    mgr = _get_manager()
    existing = mgr.get_model(model_id)
    if not existing:
        return jsonify({'error': 'Model not found'}), 404

    # Don't allow editing a model whose training is currently running.
    from web.app import trainer
    if trainer and trainer.is_running and trainer.model_id == model_id:
        return jsonify({'error': 'Stop training before editing this model'}), 400

    # Validate fields that are present.
    name = data.get('name')
    if name is not None:
        name = name.strip()
        if not name:
            return jsonify({'error': 'Name cannot be empty'}), 400

    board_size = data.get('board_size')
    if board_size is not None and board_size not in (7, 9, 13, 15, 19):
        return jsonify({'error': f'Invalid board size: {board_size}'}), 400

    ruleset = data.get('ruleset')
    if ruleset is not None and ruleset not in ('chinese', 'japanese'):
        return jsonify({'error': f'Invalid ruleset: {ruleset}'}), 400

    komi = data.get('komi')

    training_params = {}
    for key in ('num_self_play_games', 'eval_games', 'num_simulations',
                'c_puct', 'learning_rate', 'batch_size',
                'num_epochs_per_iteration', 'replay_buffer_size',
                'reflection_interval_games', 'temperature_threshold',
                'temperature_init', 'temperature_final'):
        if key in data:
            training_params[key] = data[key]

    # Warn (but allow) if changing board size on a model that already has weights.
    board_size_changed = board_size is not None and board_size != existing.board_size
    warning = None
    if board_size_changed and existing.iteration > 0:
        warning = ('Board size changed on a trained model — existing weights are '
                   'incompatible and training will restart from scratch for this model.')

    info = mgr.update_model(
        model_id,
        name=name,
        board_size=board_size,
        komi=komi,
        ruleset=ruleset,
        training_params=training_params,
    )
    if not info:
        return jsonify({'error': 'Model not found'}), 404

    # If this is the active model, rebuild the trainer so the new config
    # (hyperparameters, board size, komi, ruleset) takes effect immediately.
    if mgr.get_active_model_id() == model_id:
        from web.app import switch_model
        switch_model(model_id)

    return jsonify({'model': info.to_dict(), 'warning': warning})


@model_bp.route('/api/<model_id>/copy', methods=['POST'])
def copy_model(model_id):
    data = request.get_json() or {}
    new_name = data.get('name', '').strip()
    if not new_name:
        return jsonify({'error': 'Name is required'}), 400

    mgr = _get_manager()
    info = mgr.copy_model(model_id, new_name)
    if not info:
        return jsonify({'error': 'Source model not found'}), 404
    return jsonify(info.to_dict()), 201


@model_bp.route('/api/<model_id>/delete', methods=['DELETE'])
def delete_model(model_id):
    mgr = _get_manager()

    from web.app import trainer
    if trainer and trainer.is_running and trainer.model_id == model_id:
        return jsonify({'error': 'Stop training before deleting the active model'}), 400

    if not mgr.delete_model(model_id):
        return jsonify({'error': 'Model not found'}), 404

    # If deleted model was active, clear trainer
    from web.app import clear_trainer
    active = mgr.get_active_model_id()
    if active is None:
        clear_trainer()

    return jsonify({'message': 'Model deleted'})
