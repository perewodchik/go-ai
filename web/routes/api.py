"""
api.py — General REST API endpoints.
"""

from flask import Blueprint, jsonify, request

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


@api_bp.route('/system')
def system_info():
    """
    What this machine trains on: device, cores, and worker limits.

    `?benchmark=1` additionally times a training-shaped forward+backward on the
    GPU and on the CPU and returns a verdict on whether the GPU is worth using
    at the current batch size. That measurement costs a few seconds of real
    compute, so it never runs on a plain page load.

    The device reported is the one the ACTIVE TRAINER resolved, not merely the
    best one the machine has: a CUDA build that fails its first forward pass
    demotes the run to CPU, and this endpoint has to show that rather than the
    card sitting unused.
    """
    import device_info
    from web.app import trainer

    want_bench = request.args.get('benchmark', '').lower() in ('1', 'true', 'yes', 'on')

    if trainer is not None and not want_bench:
        payload = dict(trainer.hardware)
        payload['active_device'] = trainer.device
        return jsonify(payload)

    bench_kwargs = {}
    if want_bench and trainer is not None:
        # Measure at the settings this model actually trains with, so the
        # verdict is about the user's run and not about a generic shape.
        bench_kwargs = {
            'batch_size': trainer.config.training.batch_size,
            'board_size': trainer.config.board.size,
            'num_filters': trainer.config.network.num_filters,
            'num_blocks': trainer.config.network.num_res_blocks,
        }

    payload = device_info.summary(include_benchmark=want_bench, **bench_kwargs)
    payload['active_device'] = trainer.device if trainer else payload['device']['torch_device']
    return jsonify(payload)
