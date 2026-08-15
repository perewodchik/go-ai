"""
resignation.py — One explanation of why a stored game ended in a resignation.

A game record can carry a resignation for three unrelated reasons, and each
writes a different set of fields:

  * **the mercy rule** (self-play) — `resigned` / `resign_color` /
    `resign_move`, plus `resign_evidence` on games recorded after the evidence
    was added. Nobody chose to resign here: the losing side's own search called
    the game hopeless for several moves running and the loop stopped early.
  * **a player** (bot vs bot matches) — `resigned_by`, set when an engine
    returned MOVE_RESIGN. Named by `black_player` / `white_player`.
  * **the user** (games recorded from the Play page) — `resigned_by` again,
    with `human_color` telling us which side that was.

Plus a fourth case that is not a resignation at all but is worth surfacing in
the same place: a **mercy-rule check**. A share of self-play games
(`resign_playout_fraction`) ignore the rule and play on so that wrong
resignations can be counted. Those records carry `would_resign_color` /
`would_resign_move` and, once the game is scored, `false_resign` — the single
most useful thing the review UI can show about the mercy rule, because it is
the one place you can see the rule being wrong.

`describe_resignation()` folds all four into one shape so the API, the review
page and the training page never have to know which of them produced a game.
Records written before any given field existed still get an explanation — a
vaguer one, which is the honest output when the evidence was not recorded.
"""

from typing import Optional

BLACK = 1
WHITE = 2

# Emitted as `source`, and used by the UI to pick an icon.
SOURCE_MERCY_RULE = 'mercy-rule'
SOURCE_PLAYER = 'player'
SOURCE_HUMAN = 'human'
SOURCE_UNKNOWN = 'unknown'

_COLOR_NAMES = {BLACK: 'Black', WHITE: 'White'}


def _color_name(color) -> str:
    return _COLOR_NAMES.get(color, 'Someone')


def _win_pct(value: float) -> int:
    """A root value in [-1, +1], as the win probability it represents."""
    return int(round(50.0 + 50.0 * float(value)))


def _fact(label: str, value: str) -> dict:
    return {'label': label, 'value': value}


def _player_name(record: dict, color) -> Optional[str]:
    """Name of the engine that played `color` in a bot vs bot match."""
    slot = 'black_player' if color == BLACK else 'white_player'
    player = record.get(slot) or {}
    name = player.get('name')
    return name or None


def _mercy_facts(evidence: dict, color) -> list:
    """The trigger conditions, as label/value pairs for a UI to lay out."""
    facts = []
    who = _color_name(color)

    root = evidence.get('root_value')
    if root is not None:
        facts.append(_fact(
            f"{who}'s own evaluation",
            f"{root:+.2f} (~{_win_pct(root)}% win)",
        ))

    threshold = evidence.get('threshold')
    if threshold is not None:
        facts.append(_fact('Resign threshold', f"{-abs(threshold):+.2f}"))

    streak = evidence.get('streak')
    required = evidence.get('required_streak')
    if streak is not None:
        needed = f" (needs {required})" if required is not None else ''
        facts.append(_fact('Hopeless moves in a row', f"{streak}{needed}"))

    opponent = evidence.get('opponent_value')
    if opponent is not None:
        opp_who = _color_name(WHITE if color == BLACK else BLACK)
        facts.append(_fact(
            f"{opp_who}'s own evaluation",
            f"{opponent:+.2f} (~{_win_pct(opponent)}% win)",
        ))
    elif evidence.get('both_sides') is False:
        facts.append(_fact('Opponent confirmation', 'not required'))

    min_move = evidence.get('min_move')
    if min_move is not None:
        facts.append(_fact('Earliest allowed resign', f"move {min_move}"))

    return facts


def _mercy_reason(record: dict, color, move, evidence: Optional[dict]) -> str:
    """Prose explanation of a mercy-rule trigger."""
    who = _color_name(color)
    at = f" at move {move}" if move is not None else ''

    if not evidence:
        # Recorded before the evidence was kept — say what is known and no
        # more. What happened NEXT is the caller's sentence, not this one's:
        # the same trigger either ends the game or gets overruled by a check.
        return (
            f"The mercy rule fired{at}: {who}'s own search had called the "
            f"position lost for several moves running. This game predates the "
            f"recording of the underlying evaluations, so the exact numbers "
            f"are not available."
        )

    root = evidence.get('root_value')
    threshold = evidence.get('threshold')
    streak = evidence.get('streak')

    parts = []
    if root is not None and threshold is not None and streak is not None:
        parts.append(
            f"{who}'s own search rated the position {root:+.2f} "
            f"(~{_win_pct(root)}% win) for {streak} of its moves in a row, "
            f"past the {-abs(threshold):+.2f} mercy-rule threshold"
        )
    else:
        parts.append(f"{who}'s own search had called the position lost")

    opponent = evidence.get('opponent_value')
    if opponent is not None:
        opp_who = _color_name(WHITE if color == BLACK else BLACK)
        parts.append(
            f"{opp_who} agreed it was winning at {opponent:+.2f} "
            f"(~{_win_pct(opponent)}%)"
        )
    elif evidence.get('both_sides') is False:
        parts.append("opponent confirmation was switched off, so the losing "
                     "side's opinion was enough on its own")

    return f"The mercy rule fired{at}: " + ', and '.join(parts) + '.'


def describe_resignation(record: dict) -> Optional[dict]:
    """
    Explain how a stored game ended, if it ended in (or flirted with) a
    resignation. Returns None for every other game.

    The returned dict is what both the review page and the training page
    render, so all of the wording lives here:

        resigned      True if the game actually ENDED by resignation.
        checked       True for a mercy-rule check — the rule fired but the game
                      was played out anyway to test it. Never True with
                      `resigned`.
        color         Colour that resigned (or would have), 1/2.
        move          Move number the resignation happened on, if known.
        source        Who ended it: mercy-rule / player / human / unknown.
        result        Go notation for a resigned result ('B+R'), or None. This
                      is what a game list should show instead of a point
                      margin, which a resigned game does not have.
        badge         One short line for a list row.
        headline      One short line for the detail header.
        reason        The full explanation.
        facts         [{label, value}] — the numbers behind it, if recorded.
        false_resign  For checks only: True if the side the rule wanted to
                      resign went on to WIN, i.e. the rule would have been
                      wrong. None when it is not a check.
    """
    if not isinstance(record, dict):
        return None

    resigned_by = record.get('resigned_by')
    mercy_color = record.get('resign_color')
    mercy = bool(record.get('resigned')) and mercy_color is not None
    check_color = record.get('would_resign_color')

    if not resigned_by and not mercy and check_color is None:
        return None

    phase = record.get('phase')
    evidence = record.get('resign_evidence') or None

    # --- The rule fired but the game was played out anyway (a check) --------
    if not mercy and not resigned_by and check_color is not None:
        move = record.get('would_resign_move')
        who = _color_name(check_color)
        false_resign = bool(record.get('false_resign'))
        verdict = (
            f"{who} went on to WIN, so resigning would have thrown away a won "
            f"game and mislabelled every training sample in it."
            if false_resign else
            f"{who} did go on to lose, so the resignation would have been "
            f"correct — this game cost some wall-clock to confirm it."
        )
        return {
            'resigned': False,
            'checked': True,
            'color': int(check_color),
            'move': move,
            'source': SOURCE_MERCY_RULE,
            'result': None,
            'badge': 'Mercy-rule check' + (f' · move {move}' if move is not None else ''),
            'headline': 'Mercy-rule check' + (f' at move {move}' if move is not None else ''),
            'reason': (
                _mercy_reason(record, check_color, move, evidence)
                + f" This game was one of the sampled playouts, so the rule was "
                f"overruled and the game played on to test it: {verdict}"
            ),
            'facts': _mercy_facts(evidence, check_color) if evidence else [],
            'false_resign': false_resign,
        }

    # --- An actual resignation ---------------------------------------------
    color = int(mercy_color) if mercy else int(resigned_by)
    winner = record.get('winner')
    result = None
    if winner in (BLACK, WHITE):
        result = 'B+R' if winner == BLACK else 'W+R'

    if mercy:
        move = record.get('resign_move')
        who = _color_name(color)
        reason = _mercy_reason(record, color, move, evidence)
        reason += (
            " Nobody chose to resign — self-play stops a decided game to buy "
            "iterations. The cost is the outcome LABEL: every training sample "
            "kept from this game is trained on a predicted result rather than "
            "a played-out one."
        )
        return {
            'resigned': True,
            'checked': False,
            'color': color,
            'move': move,
            'source': SOURCE_MERCY_RULE,
            'result': result,
            'badge': f'{who} resigned' + (f' · move {move}' if move is not None else ''),
            'headline': f'{who} resigned (mercy rule)',
            'reason': reason,
            'facts': _mercy_facts(evidence, color) if evidence else [],
            'false_resign': None,
        }

    move = record.get('resign_move')
    if move is None:
        # Neither the Play page nor a match records the move a resignation
        # landed on, but it is always the position after the last recorded
        # move — the resignation itself is not a board move.
        move = record.get('num_moves')

    who = _color_name(color)

    if phase == SOURCE_HUMAN or record.get('human_color') is not None:
        human_color = record.get('human_color')
        by_human = (human_color is not None and int(human_color) == color)
        name = 'You' if by_human else 'The bot'
        reason = (
            f"{name} resigned the game from the Play page{f' after move {move}' if move else ''}. "
            + ("A resignation is a choice, not a scored result — there is no "
               "point margin to show." if by_human else
               "The bot conceded rather than playing the game out.")
        )
        return {
            'resigned': True,
            'checked': False,
            'color': color,
            'move': move,
            'source': SOURCE_HUMAN,
            'result': result,
            'badge': f'{name} resigned',
            'headline': f'{name} resigned ({who})',
            'reason': reason,
            'facts': [],
            'false_resign': None,
        }

    name = _player_name(record, color)
    label = name or who
    return {
        'resigned': True,
        'checked': False,
        'color': color,
        'move': move,
        'source': SOURCE_PLAYER if name else SOURCE_UNKNOWN,
        'result': result,
        'badge': f'{label} resigned',
        'headline': f'{label} resigned as {who}',
        'reason': (
            f"{label} resigned as {who}"
            f"{f' after move {move}' if move else ''}. The engine offered a "
            f"resignation rather than playing the game out, so the result is a "
            f"concession and carries no point margin."
        ),
        'facts': [],
        'false_resign': None,
    }


def annotate_resignation(record: dict) -> dict:
    """
    Attach `describe_resignation(record)` to a record as `resignation`, in
    place, and return it. Games that ended normally get no key at all, so a
    client can treat presence as "something resignation-related happened here".
    """
    info = describe_resignation(record)
    if info is not None:
        record['resignation'] = info
    return record
