import json

import numpy as np
import taverngym as tg

from omnibob.capacity_trade import CapacityTradePolicy
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision
from omnibob.repeat_guard import RepeatGuardPolicy


class PreferRefresh:
    version = 'fixture'
    runtime_config = {}
    config = {'max_actions_per_turn': 100}

    def act(self, obs):
        for action in (1, 0):
            if obs.action_mask[action]:
                return Decision(action, 'fixture')
        return Decision(int(np.flatnonzero(obs.action_mask)[0]), 'fixture')


def full_game():
    game = tg.Game(42)
    for seat in range(8):
        game.step(seat, 30)
    # Golden board copies and pairs in hand avoid automatic triple conversion.
    for _ in range(7):
        game.debug(json.dumps({'cmd': 'add_minion', 'p': 0, 'zone': 'board', 'card': 'BG25_354', 'golden': True}))
    for card in ['BG25_354', 'BG20_100', 'BG29_611', 'BG26_146', 'BG36_372']:
        for _ in range(2):
            game.debug(json.dumps({'cmd': 'add_minion', 'p': 0, 'zone': 'hand', 'card': card}))
    return game


def observe(game):
    return ObservationV2.from_game(game, 0)


def test_real_engine_sale_then_play_returns_control_and_frees_hand():
    game = full_game()
    policy = RepeatGuardPolicy(CapacityTradePolicy(PreferRefresh()))
    before = observe(game)
    assert before.legacy().blocks()['hand'][:, 0].sum() == 10
    assert not before.action_mask[46:196].any()
    sale = policy.act(before)
    assert 16 <= sale.action < 23
    game.step(0, sale.action)
    after_sale = observe(game)
    assert after_sale.legacy().blocks()['board'][:, 0].sum() == 6
    play = policy.act(after_sale)
    assert 46 <= play.action < 126 and after_sale.action_mask[play.action]
    game.step(0, play.action)
    after_play = observe(game)
    assert after_play.legacy().blocks()['hand'][:, 0].sum() == 9
    assert after_play.legacy().blocks()['board'][:, 0].sum() == 7
    assert policy.act(after_play).action == 1
    game.debug(json.dumps({'cmd': 'add_minion', 'p': 0, 'zone': 'hand', 'card': 'BG35_342'}))
    assert policy.act(observe(game)).action == 1  # Once per turn, even when full again.


def test_failed_sale_does_not_force_illegal_play_and_reset_clears_chain():
    obs = observe(full_game())
    policy = CapacityTradePolicy(PreferRefresh())
    assert policy.act(obs).action == 16
    assert policy.act(obs).action == 1  # Unchanged state: sale did not take effect.
    assert policy.last_event == 'aborted_after_sale'
    policy.reset()
    assert policy.act(obs).action == 16
    policy.reset()
    f = obs.features.copy()
    f[0] += .05
    f[68] = 0  # New turn with board space; no stale forced play.
    assert policy.act(ObservationV2(f, obs.cards, obs.action_mask)).action == 1


def test_capacity_only_and_four_actions_reserved():
    obs = observe(full_game())
    for field, value in [(3227, 97), (68, 0), (285, 0)]:
        f = obs.features.copy()
        f[field] = value
        assert CapacityTradePolicy(PreferRefresh()).act(ObservationV2(f, obs.cards, obs.action_mask)).action == 1
    for field in (3, 6):  # Locked or doomed hand minions cannot justify selling.
        f = obs.features.copy()
        f[285+field:685:40] = 1
        assert CapacityTradePolicy(PreferRefresh()).act(ObservationV2(f, obs.cards, obs.action_mask)).action == 1
    for action in (8, 46, 126):
        mask = obs.action_mask.copy()
        mask[action] = True
        assert CapacityTradePolicy(PreferRefresh()).act(ObservationV2(obs.features, obs.cards, mask)).action == 1
