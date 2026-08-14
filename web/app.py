"""
app.py — Flask + SocketIO application factory.

Creates the web server that serves:
- The play page (human vs bot)
- The training dashboard
- REST API endpoints
- WebSocket events for real-time training updates
"""

import os
import sys
import threading
from flask import Flask
from flask_socketio import SocketIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from ai.trainer import Trainer
from model_manager import ModelManager

# Global instances (shared across routes)
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')
model_manager = ModelManager()
trainer = None  # Initialized when a model is selected
training_thread = None


def _emit_progress(event: dict):
    """SocketIO callback for training events."""
    socketio.emit('training_update', event)


def switch_model(model_id: str) -> bool:
    """
    Switch the active trainer to a different model.
    
    Stops any running training, creates a new Config from the model,
    and initializes a new Trainer.
    """
    global trainer

    info = model_manager.get_model(model_id)
    if not info:
        return False

    model_dir = model_manager.get_model_dir(model_id)
    config = Config.from_model(info, model_dir)
    trainer = Trainer(config=config, progress_callback=_emit_progress, model_id=model_id)
    return True


def clear_trainer():
    """Clear the trainer when no model is active."""
    global trainer
    trainer = None


def create_app() -> Flask:
    """Flask application factory."""
    global trainer

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
        static_folder=os.path.join(os.path.dirname(__file__), 'static'),
    )
    app.config['SECRET_KEY'] = 'go-ai-secret-key'

    socketio.init_app(app)

    # Auto-select active model on startup
    active_id = model_manager.get_active_model_id()
    if active_id:
        switch_model(active_id)

    # Register routes
    from web.routes.game_routes import game_bp
    from web.routes.training_routes import training_bp
    from web.routes.model_routes import model_bp
    from web.routes.api import api_bp

    app.register_blueprint(game_bp)
    app.register_blueprint(training_bp, url_prefix='/training')
    app.register_blueprint(model_bp, url_prefix='/models')
    app.register_blueprint(api_bp, url_prefix='/api')

    # Index route
    @app.route('/')
    def index():
        from flask import render_template
        active_model = model_manager.get_active_model()
        models = model_manager.list_models()
        return render_template('index.html',
                               active_model=active_model,
                               models=models)

    # Info / parameter guide route
    @app.route('/info')
    def info():
        from flask import render_template
        from config import NETWORK_PRESETS, NETWORK_PRESET_ORDER
        presets = [dict(key=k, **NETWORK_PRESETS[k]) for k in NETWORK_PRESET_ORDER]
        return render_template('info.html', network_presets=presets)

    @socketio.on('start_training')
    def handle_start_training(data=None):
        max_iter = data.get('max_iterations') if isinstance(data, dict) else None
        start_training_job(max_iter)

    @socketio.on('stop_training')
    def handle_stop_training():
        if trainer:
            trainer.stop()

    @socketio.on('force_stop_training')
    def handle_force_stop_training():
        if trainer:
            trainer.force_stop()

    @socketio.on('request_status')
    def handle_status():
        active_model = model_manager.get_active_model()
        status_data = trainer.get_status() if trainer else {
            'is_running': False,
            'stop_requested': False,
            'iteration': 0,
            'total_games': 0,
            'elo': 500,
            'kyu_rank': '30k',
            'buffer_size': 0,
            'device': 'cpu',
            'recent_logs': [],
        }
        status_data['active_model'] = active_model.to_dict() if active_model else None
        socketio.emit('training_update', {
            'type': 'status',
            'data': status_data,
        })

    return app


def start_training_job(max_iterations=None):
    global training_thread, trainer
    if trainer is None:
        socketio.emit('training_update', {'type': 'error', 'message': 'No model selected. Create or select a model from the Dashboard.'})
        return False, 'No model selected'
    if trainer.is_running:
        socketio.emit('training_update', {'type': 'info', 'message': 'Training already running'})
        return False, 'Training already running'

    training_thread = threading.Thread(target=trainer.train, args=(max_iterations,), daemon=True)
    training_thread.start()
    return True, 'Training started'
