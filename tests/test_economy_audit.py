import json

import taverngym as tg

from omnibob.economy_audit import inspect_end
from omnibob.observation_v2 import ObservationV2


def test_flags_describe_real_legal_hand_options_and_full_board():
    game=tg.Game(42)
    for seat in range(8):
        game.step(seat,30)
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card='BG25_354')))
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='hand',card='BG20_100')))
    game.debug(json.dumps(dict(cmd='add_spell',p=0,zone='hand',card='BG20_GEM')))
    catalog={c['cid']+1:c for c in json.loads(tg.catalog_json())}
    obs=ObservationV2.from_game(game,0)
    result=inspect_end(obs,catalog)
    assert result['flags']['playable_minion_with_space']
    assert result['flags']['free_playable_spell']
    assert not result['flags']['full_board_held_battlecry']
    assert result['hand'][0]['name']=='剃刀沼泽地卜师'
    for _ in range(6):
        game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card='BG25_354')))
    full=inspect_end(ObservationV2.from_game(game,0),catalog)
    assert full['board_size']==7
    assert full['flags']['full_board_held_battlecry']
    assert not full['flags']['playable_minion_with_space']
    assert full['flags']['free_playable_spell']  # Board space does not gate a targeted gem.
