"""
api.py — General REST API endpoints.
"""

from flask import Blueprint, jsonify

api_bp = Blueprint('api', __name__)


@api_bp.route('/health')
def health():
    return jsonify({'status': 'ok'})


@api_bp.route('/config')
def get_config():
    from web.app import model_manager, trainer
    active = model_manager.get_active_model()
    if active:
        return jsonify({
            'model_id': active.id,
            'model_name': active.name,
            'board_size': active.board_size,
            'komi': active.komi,
            'ruleset': active.ruleset,
            'device': trainer.device if trainer else 'cpu',
        })
    return jsonify({
        'model_id': None,
        'model_name': None,
        'board_size': 9,
        'komi': 6.5,
        'ruleset': 'chinese',
        'device': 'cpu',
    })
