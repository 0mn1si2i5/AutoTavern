import json
import numpy as np
import pytest
import taverngym as tg
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision
from omnibob.repeat_guard import RepeatGuardPolicy, clone_runtime


class PlayThenCancel:
    version = 'test-repeat'
    runtime_config = {}
    repeat_limit = 0
    move_guard_budget = 0
    def act(self, obs):
        action = 7 if obs.action_mask[7] else 46 if obs.action_mask[46] else 0
        return Decision(action, 'deliberate native spell cancellation cycle')


@pytest.mark.parametrize('limit', [1, 2])
def test_real_spell_cancel_cycle_is_bounded_and_resettable(limit):
    game = tg.Game(42)
    for seat in range(8):
        game.step(seat, 30)
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card='BG25_354')))
    game.debug(json.dumps(dict(cmd='add_spell',p=0,zone='hand',card='BG20_GEM')))
    guard = RepeatGuardPolicy(PlayThenCancel(), limit)
    initial = ObservationV2.from_game(game, 0)
    for _ in range(limit):
        for expected in (46, 7):
            obs = ObservationV2.from_game(game, 0)
            action = guard.act(obs).action
            assert action == expected and obs.action_mask[action]
            game.step(0, action)
    current = ObservationV2.from_game(game, 0)
    changed = np.flatnonzero(initial.features != current.features)
    assert changed.tolist() == [guard.action_count_index]
    assert guard.act(current).action == 0
    other = clone_runtime(guard)
    assert other.base is guard.base and other.act(current).action == 46
    guard.reset()
    assert guard.act(current).action == 46
