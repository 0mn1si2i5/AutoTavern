import json

import numpy as np
import pytest
import taverngym as tg

from omnibob.move_guard import MoveGuardPolicy
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision


class PreferMove:
    version = "test-prefer-move"

    def act(self, obs):
        moves = np.flatnonzero(obs.action_mask[196:])+196
        return Decision(int(moves[0]) if len(moves) else 0, "test controller")


@pytest.mark.parametrize("budget", [1, 7])
def test_actual_board_move_cannot_cycle_and_policy_can_reset(budget):
    game = tg.Game(42)
    for seat in range(8):
        game.step(seat, 30)
    for _ in range(2):
        game.debug(json.dumps(dict(cmd="add_minion", p=0, zone="board", card="BG25_354")))
    game.debug(json.dumps(dict(cmd="edit_minion", p=0, zone="board", index=0, atk=123)))
    guard = MoveGuardPolicy(PreferMove(), budget)
    first = guard.act(ObservationV2.from_game(game, 0))
    assert 196 <= first.action < 245
    game.step(0, first.action)
    second = guard.act(ObservationV2.from_game(game, 0))
    assert second.action == 0 and game.legal_mask(0)[second.action]
    guard.reset()
    assert 196 <= guard.act(ObservationV2.from_game(game, 0)).action < 245
