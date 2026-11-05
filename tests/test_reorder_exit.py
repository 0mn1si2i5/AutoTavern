import json

import numpy as np
import taverngym as tg

from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision
from omnibob.reorder_exit import ReorderExitPolicy


class PreferMove:
    version = "prefer-move"
    runtime_config = {}

    def act(self, obs):
        moves = np.flatnonzero(obs.action_mask[196:]) + 196
        return Decision(int(moves[0]) if len(moves) else int(np.flatnonzero(obs.action_mask)[0]), "test")


def game_with_board(count=2):
    game = tg.Game(42)
    for seat in range(8):
        game.step(seat, 30)
    for i in range(count):
        game.debug(json.dumps(dict(cmd="add_minion", p=0, zone="board", card="BG25_354")))
        game.debug(json.dumps(dict(cmd="edit_minion", p=0, zone="board", index=i, atk=100+i)))
    return game


def test_actual_insertion_cycle_exits_to_legal_nonmove():
    game = game_with_board()
    policy = ReorderExitPolicy(PreferMove())
    first = policy.act(ObservationV2.from_game(game, 0))
    assert first.action == 197
    game.step(0, first.action)
    second = policy.act(ObservationV2.from_game(game, 0))
    assert second.action == 0  # Reversing the pair via either move alias must stop.
    assert game.legal_mask(0)[second.action]
    policy.reset()
    assert policy.act(ObservationV2.from_game(game, 0)).action == 197


def test_public_opponent_updates_do_not_hide_a_board_cycle():
    game = game_with_board()
    policy = ReorderExitPolicy(PreferMove())
    game.step(0, policy.act(ObservationV2.from_game(game, 0)).action)
    obs = ObservationV2.from_game(game, 0)
    f = obs.features.copy()
    f[3656] += 1  # Public opponent summary; irrelevant to a pure own-board insertion.
    assert policy.act(ObservationV2(f, obs.cards, obs.action_mask)).action == 0


def test_changed_board_and_nonmove_allow_new_reordering():
    game = game_with_board()
    policy = ReorderExitPolicy(PreferMove())
    game.step(0, policy.act(ObservationV2.from_game(game, 0)).action)
    game.debug(json.dumps(dict(cmd="edit_minion", p=0, zone="board", index=0, atk=777)))
    assert policy.act(ObservationV2.from_game(game, 0)).action == 197
    obs = ObservationV2.from_game(game, 0)
    mask = np.zeros(245, dtype=bool)
    mask[2] = True  # An actual non-move ends this movement-only segment.
    assert policy.act(ObservationV2(obs.features, obs.cards, mask)).action == 2
    assert policy.act(obs).action == 197


def test_no_illegal_fallback_when_only_moves_exist():
    game = game_with_board()
    policy = ReorderExitPolicy(PreferMove())
    game.step(0, policy.act(ObservationV2.from_game(game, 0)).action)
    obs = ObservationV2.from_game(game, 0)
    mask = obs.action_mask.copy()
    mask[:196] = False
    assert mask[policy.act(ObservationV2(obs.features, obs.cards, mask)).action]


def test_eight_distinct_reorders_are_not_a_seven_move_cap():
    game = game_with_board(4)
    actions = [197, 205, 213, 197, 205, 197, 213, 205]
    start_count = ObservationV2.from_game(game, 0).features[3227]

    class Route(PreferMove):
        def act(self, obs):
            action = actions[round(float(obs.features[3227]-start_count))]
            assert obs.action_mask[action]
            return Decision(action, "distinct board route")

    policy = ReorderExitPolicy(Route())
    observed = set()
    for action in actions:
        obs = ObservationV2.from_game(game, 0)
        signature = tuple(obs.features[68:285].reshape(7, 31)[:4, 1])
        assert signature not in observed
        observed.add(signature)
        assert policy.act(obs).action == action
        game.step(0, action)
