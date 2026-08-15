"""
test_resignation_reason.py — Tests for ai/resignation.py.

A stored game can end in a resignation for three unrelated reasons (the mercy
rule, an engine conceding a match, the user conceding on the Play page), and
one near-miss worth surfacing (a mercy-rule check: the rule fired but the game
was played out to measure it). `describe_resignation()` is the single place
that tells them apart, so these tests pin down:

  * each source is identified, and named in wording a reader can act on;
  * a resigned game reports B+R / W+R rather than a point margin it does not
    have;
  * a check is never reported as an early end, and a WRONG check is flagged —
    that is the one case that means the threshold is costing training data;
  * games that predate `resign_evidence` still get an explanation, without
    inventing numbers;
  * a normally finished game gets nothing at all.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.resignation import (
    SOURCE_HUMAN,
    SOURCE_MERCY_RULE,
    SOURCE_PLAYER,
    annotate_resignation,
    describe_resignation,
)
from game.board import BLACK, WHITE


def mercy_record(**overrides):
    record = {
        'phase': 'self-play',
        'winner': WHITE,
        'num_moves': 84,
        'margin': 12.5,
        'resigned': True,
        'resign_color': BLACK,
        'resign_move': 84,
        'resign_evidence': {
            'root_value': -0.94,
            'opponent_value': 0.97,
            'streak': 5,
            'threshold': 0.90,
            'required_streak': 4,
            'both_sides': True,
            'min_move': 81,
        },
    }
    record.update(overrides)
    return record


class TestMercyRule:

    def test_identifies_the_mercy_rule_and_the_resigning_side(self):
        info = describe_resignation(mercy_record())
        assert info['resigned'] is True
        assert info['checked'] is False
        assert info['source'] == SOURCE_MERCY_RULE
        assert info['color'] == BLACK
        assert info['move'] == 84

    def test_result_is_go_notation_not_a_margin(self):
        """A resigned game has no point margin — `margin` is a board score."""
        assert describe_resignation(mercy_record())['result'] == 'W+R'
        assert describe_resignation(
            mercy_record(winner=BLACK, resign_color=WHITE))['result'] == 'B+R'

    def test_reason_names_the_numbers_that_fired_it(self):
        reason = describe_resignation(mercy_record())['reason']
        assert '-0.94' in reason          # the losing side's own evaluation
        assert '-0.90' in reason          # the threshold it passed
        assert '5' in reason              # consecutive confirmations
        assert '+0.97' in reason          # the winner agreeing

    def test_reason_warns_that_the_outcome_is_a_prediction(self):
        """The label risk is the whole reason to look at a resigned game."""
        assert 'LABEL' in describe_resignation(mercy_record())['reason']

    def test_facts_expose_the_trigger_conditions(self):
        facts = describe_resignation(mercy_record())['facts']
        labels = ' '.join(f['label'] for f in facts)
        values = ' '.join(f['value'] for f in facts)
        assert 'Black' in labels and 'White' in labels
        assert '-0.94' in values and '+0.97' in values
        assert '5 (needs 4)' in values

    def test_win_percentage_is_derived_from_the_root_value(self):
        facts = describe_resignation(mercy_record())['facts']
        own = next(f for f in facts if f['label'].startswith("Black's"))
        assert '3% win' in own['value']   # -0.94 -> 50 - 47

    def test_older_games_are_explained_without_inventing_numbers(self):
        info = describe_resignation(mercy_record(resign_evidence=None))
        assert info['resigned'] is True
        assert info['facts'] == []
        assert 'not available' in info['reason']
        assert 'move 84' in info['reason']

    def test_one_sided_resignation_says_confirmation_was_off(self):
        record = mercy_record()
        record['resign_evidence'].update(opponent_value=None, both_sides=False)
        info = describe_resignation(record)
        assert 'switched off' in info['reason']
        assert any(f['value'] == 'not required' for f in info['facts'])


class TestMercyRuleCheck:

    def check_record(self, **overrides):
        record = {
            'phase': 'self-play',
            'winner': WHITE,
            'num_moves': 130,
            'resigned': False,
            'resign_color': None,
            'resign_playout': True,
            'would_resign_color': BLACK,
            'would_resign_move': 84,
            'false_resign': False,
            'resign_evidence': mercy_record()['resign_evidence'],
        }
        record.update(overrides)
        return record

    def test_a_check_is_not_reported_as_an_early_end(self):
        info = describe_resignation(self.check_record())
        assert info['resigned'] is False
        assert info['checked'] is True
        assert info['result'] is None      # the game was scored on the board

    def test_a_correct_check_says_the_rule_would_have_been_right(self):
        info = describe_resignation(self.check_record())
        assert info['false_resign'] is False
        assert 'would have been correct' in info['reason']

    def test_a_wrong_check_is_flagged_loudly(self):
        info = describe_resignation(
            self.check_record(winner=BLACK, false_resign=True))
        assert info['false_resign'] is True
        assert 'WIN' in info['reason']
        assert 'mislabelled' in info['reason']


class TestPlayerAndHumanResignations:

    def test_human_resignation_is_attributed_to_the_user(self):
        info = describe_resignation({
            'phase': 'human',
            'winner': WHITE,
            'num_moves': 40,
            'human_color': BLACK,
            'resigned_by': BLACK,
        })
        assert info['source'] == SOURCE_HUMAN
        assert info['result'] == 'W+R'
        assert info['badge'] == 'You resigned'
        assert 'Play page' in info['reason']

    def test_bot_resignation_in_a_recorded_game_is_attributed_to_the_bot(self):
        info = describe_resignation({
            'phase': 'human',
            'winner': BLACK,
            'num_moves': 40,
            'human_color': BLACK,
            'resigned_by': WHITE,
        })
        assert info['badge'] == 'The bot resigned'
        assert info['result'] == 'B+R'

    def test_match_resignation_names_the_engine(self):
        info = describe_resignation({
            'phase': 'match',
            'winner': BLACK,
            'num_moves': 60,
            'resigned_by': WHITE,
            'black_player': {'name': 'iter 40'},
            'white_player': {'name': 'iter 12'},
        })
        assert info['source'] == SOURCE_PLAYER
        assert info['badge'] == 'iter 12 resigned'
        assert 'iter 12 resigned as White' in info['reason']

    def test_unnamed_resignation_falls_back_to_the_colour(self):
        info = describe_resignation({
            'phase': 'eval', 'winner': BLACK, 'num_moves': 20, 'resigned_by': WHITE,
        })
        assert info['badge'] == 'White resigned'
        assert info['facts'] == []


class TestNonResignations:

    def test_a_scored_game_gets_no_description(self):
        assert describe_resignation({
            'phase': 'self-play', 'winner': BLACK, 'margin': 7.5,
            'resigned': False, 'resign_color': None,
            'would_resign_color': None, 'false_resign': False,
        }) is None

    def test_annotate_leaves_scored_games_untouched(self):
        record = {'winner': BLACK, 'margin': 7.5}
        assert 'resignation' not in annotate_resignation(record)

    def test_annotate_attaches_the_description_in_place(self):
        record = annotate_resignation(mercy_record())
        assert record['resignation']['result'] == 'W+R'

    def test_garbage_in_is_not_an_error(self):
        assert describe_resignation(None) is None
        assert describe_resignation('not a game') is None
