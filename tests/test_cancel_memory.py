import json

import numpy as np
import taverngym as tg

from omnibob.cancel_memory import CancelMemoryPolicy, cancelled_play_context
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision
from omnibob.repeat_guard import RepeatGuardPolicy
from omnibob.reorder_exit import ReorderExitPolicy


class PlayCancelMove:
    version = 'test-play-cancel-move'
    runtime_config = {}

    def act(self, obs):
        if obs.action_mask[7]:
            return Decision(7, 'cancel target selection')
        plays = np.flatnonzero(obs.action_mask[46:54])+46
        moves = np.flatnonzero(obs.action_mask[196:])+196
        return Decision(int(plays[0]) if len(plays) else int(moves[0]) if len(moves) else 0, 'test')


def fixture_game():
    game = tg.Game(42)
    for seat in range(8):
        game.step(seat, 30)
    for i in range(2):
        game.debug(json.dumps(dict(cmd='add_minion', p=0, zone='board', card='BG25_354')))
        game.debug(json.dumps(dict(cmd='edit_minion', p=0, zone='board', index=i, atk=100+i)))
    game.debug(json.dumps(dict(cmd='add_spell', p=0, zone='hand', card='BG20_GEM')))
    return game


def stack(memory):
    policy = RepeatGuardPolicy(ReorderExitPolicy(PlayCancelMove()), 1)
    return CancelMemoryPolicy(policy) if memory else policy


def take(game, policy):
    obs = ObservationV2.from_game(game, 0)
    action = policy.act(obs).action
    assert obs.action_mask[action]
    game.step(0, action)
    return action


def test_actual_play_cancel_move_reopens_without_memory_but_exits_with_it():
    game, old = fixture_game(), stack(False)
    assert [take(game, old) for _ in range(4)] == [46, 7, 197, 46]
    game, new = fixture_game(), stack(True)
    assert [take(game, new) for _ in range(4)] == [46, 7, 197, 0]


def test_cancel_is_allowed_and_stat_change_reopens_a_useful_play():
    game, policy = fixture_game(), stack(True)
    assert [take(game, policy) for _ in range(3)] == [46, 7, 197]
    game.debug(json.dumps(dict(cmd='edit_minion', p=0, zone='board', index=0, atk=777)))
    assert take(game, policy) == 46
    assert take(game, policy) == 7  # Cancellation itself remains available.
    policy.reset()
    assert take(game, policy) == 46


def test_all_play_position_aliases_are_blocked_but_no_illegal_fallback():
    game, policy = fixture_game(), CancelMemoryPolicy(PlayCancelMove())
    assert [take(game, policy) for _ in range(2)] == [46, 7]
    obs = ObservationV2.from_game(game, 0)
    mask = obs.action_mask.copy()
    mask[46:54] = True
    assert not 46 <= policy.act(ObservationV2(obs.features, obs.cards, mask)).action < 54
    # Future contracts with only an excluded action must retain a legal fallback.
    mask[:] = False
    mask[46] = True
    assert policy.act(ObservationV2(obs.features, obs.cards, mask)).action == 46


def test_successfully_completed_target_selection_is_not_a_cancelled_play():
    class Complete(PlayCancelMove):
        def act(self, obs):
            if obs.action_mask[7]:
                return Decision(int(np.flatnonzero(obs.action_mask[30:46])[0])+30, 'commit target')
            return super().act(obs)
    game = fixture_game()
    game.debug(json.dumps(dict(cmd='add_spell', p=0, zone='hand', card='BG20_GEM')))
    policy = CancelMemoryPolicy(Complete())
    assert take(game, policy) == 46
    assert 30 <= take(game, policy) < 46
    assert take(game, policy) == 46  # A second gem must remain playable.


def test_context_ignores_only_board_order_and_count_but_keeps_new_information():
    game = fixture_game()
    before = ObservationV2.from_game(game, 0)
    game.step(0, 197)
    after = ObservationV2.from_game(game, 0)
    assert cancelled_play_context(before) == cancelled_play_context(after)
    for index in (0, 7, 285, 3656):  # Turn, gold, hand, public opponent information.
        f = after.features.copy()
        f[index] += 1
        assert cancelled_play_context(after) != cancelled_play_context(ObservationV2(f, after.cards, after.action_mask))


def test_non_restoring_cancel_does_not_blacklist_a_play():
    game, policy = fixture_game(), CancelMemoryPolicy(PlayCancelMove())
    assert [take(game, policy) for _ in range(2)] == [46, 7]
    # An actual own-board change between opening and the returned observation.
    game.debug(json.dumps(dict(cmd='edit_minion', p=0, zone='board', index=0, atk=777)))
    assert take(game, policy) == 46
