import json

import numpy as np
import taverngym as tg

from omnibob.end_trade import EndTradePolicy
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision


class PreferEnd:
    version='test-prefer-end'
    runtime_config={}
    config={'max_actions_per_turn':100,'turn_action_caps':[100]}

    def act(self, obs):
        return Decision(0 if obs.action_mask[0] else int(np.flatnonzero(obs.action_mask)[0]),'test')


def game_with_stronger_hand():
    game=tg.Game(42)
    for seat in range(8):game.step(seat,30)
    for card in ['BG25_354','BG20_100','BG29_611','BG26_146','BG36_372','BG35_342','BG28_583']:
        game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card=card)))
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='hand',card='BG25_354')))
    game.debug(json.dumps(dict(cmd='edit_minion',p=0,zone='hand',index=0,atk=200,hp=200)))
    return game


def test_one_actual_legal_sale_opens_space_without_repeating_in_same_turn():
    game=game_with_stronger_hand();policy=EndTradePolicy(PreferEnd())
    first=policy.act(ObservationV2.from_game(game,0));assert first.action==16
    game.step(0,first.action)
    assert sum(ObservationV2.from_game(game,0).features[68:285:31])==6
    # Refill the slot, then confirm that a second forced sale cannot occur this turn.
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card='BG25_354')))
    assert policy.act(ObservationV2.from_game(game,0)).action==0
    policy.reset()
    assert policy.act(ObservationV2.from_game(game,0)).action==16


def test_does_not_sell_at_last_operations_or_when_no_qualified_hand_exists():
    game=game_with_stronger_hand();obs=ObservationV2.from_game(game,0)
    f=obs.features.copy();f[3227]=97
    assert EndTradePolicy(PreferEnd()).act(ObservationV2(f,obs.cards,obs.action_mask)).action==0
    f=obs.features.copy();f[285+3]=1  # The strong owned card is locked.
    assert EndTradePolicy(PreferEnd()).act(ObservationV2(f,obs.cards,obs.action_mask)).action==0
    f=obs.features.copy();f[285+6]=1  # Dies if played this turn.
    assert EndTradePolicy(PreferEnd()).act(ObservationV2(f,obs.cards,obs.action_mask)).action==0
