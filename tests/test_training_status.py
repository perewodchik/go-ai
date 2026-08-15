"""
Tests for training status enhancements, parallel games tracking, and current_stage API.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.trainer import Trainer
from config import Config


def test_trainer_initial_stage():
    config = Config()
    trainer = Trainer(config=config)
    status = trainer.get_status()

    assert 'current_stage' in status
    stage = status['current_stage']
    assert stage['stage'] == 'idle'
    assert stage['stage_index'] == 0
    assert stage['total_stages'] == 5
    assert isinstance(stage['stages_overview'], list)
    assert len(stage['stages_overview']) == 5


def test_trainer_set_stage_transitions():
    config = Config()
    trainer = Trainer(config=config)

    trainer._set_stage(
        'self_play', 'Self-Play Data Generation', 1,
        completed_items=4, total_items=20,
        active_games=[5, 6, 7, 8], num_workers=4,
        detail='Generating self-play games'
    )

    stage = trainer.get_status()['current_stage']
    assert stage['stage'] == 'self_play'
    assert stage['stage_index'] == 1
    assert stage['completed_items'] == 4
    assert stage['total_items'] == 20
    assert stage['percent'] == 20
    assert stage['active_games'] == [5, 6, 7, 8]
    assert stage['num_workers'] == 4

    # Check stages overview
    overview = stage['stages_overview']
    assert overview[0]['key'] == 'self_play'
    assert overview[0]['status'] == 'active'
    assert overview[1]['status'] == 'pending'


def test_trainer_on_game_complete_with_active_games():
    config = Config()
    trainer = Trainer(config=config)

    # Initial launch notification
    trainer._on_game_complete(0, 10, record=None, active_games=[1, 2, 3, 4], num_workers=4)
    status = trainer.get_status()
    assert status['current_stage']['active_games'] == [1, 2, 3, 4]
    assert status['current_stage']['completed_items'] == 0

    # Finished game 1
    mock_record = {
        'winner': 1,
        'num_moves': 30,
        'elapsed_seconds': 1.2,
        'moves': [{'color': 1, 'move': [2, 3]}],
    }
    trainer._on_game_complete(1, 10, record=mock_record, active_games=[2, 3, 4, 5], num_workers=4)
    status = trainer.get_status()
    assert status['current_stage']['active_games'] == [2, 3, 4, 5]
    assert status['current_stage']['completed_items'] == 1
    assert status['current_stage']['percent'] == 10
