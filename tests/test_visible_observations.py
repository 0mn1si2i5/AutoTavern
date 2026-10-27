"""Player observations must be complete for the owner and invariant to hidden zones."""
import json
import numpy as np
import pytest
import taverngym as tg


def recruit():
    game = tg.Game(1)
    for seat in range(8):
        game.step(seat, 30)
    return game


def test_observe2_real_extensions_and_legacy_prefix():
    game = recruit()
    game.debug(json.dumps(dict(cmd="add_minion", p=0, zone="board", card="BG25_354")))
    game.debug(json.dumps(dict(cmd="edit_minion", p=0, zone="board", index=0, attach="BG36_MidGameEffect_000t73")))
    f1,c1 = game.observe(0)
    f2,c2 = game.observe2(0)
    assert (len(f2)//4,len(c2)//4) == (4017,447)
    assert f2[:len(f1)] == f1 and c2[:len(c1)] == c1
    assert np.any(np.frombuffer(c2,dtype="<i4")[124:])
    assert sum(n for _,n in tg.obs2_blocks()[0]) == 790
    assert sum(n for _,n in tg.obs2_blocks()[1]) == 323
    own=json.loads(game.player_view(0))["me"]
    assert own["board"][0]["attached"]
    assert all(key in own for key in ["hand","shop","hero_powers","quests","rewards","secrets","effects","scheduled","stats","counters","levels","pending","dark_gift","buddy_button"])


def test_hidden_opponent_zones_never_change_observations():
    game = recruit()
    before = (game.observe(0),game.observe2(0),game.player_view(0))
    for zone in ("board","hand","shop"):
        game.debug(json.dumps(dict(cmd="add_minion", p=1, zone=zone, card="BG25_354")))
    assert (game.observe(0),game.observe2(0),game.player_view(0)) == before
    view=json.loads(game.player_view(0))
    assert not {"seed","config","combats","log","replay"} & view.keys()
    for op in view["opponents"]:
        assert not {"board","hand","shop","hand_size","pending","secrets","gold"} & op.keys()
        assert op["seen_board"] == []
    def inspect(value):
        if isinstance(value,dict):
            assert not {"eid","hid","sid","pool_copies","pool_parts","turn_events"} & value.keys()
            for v in value.values():inspect(v)
        elif isinstance(value,list):
            for v in value:inspect(v)
    inspect(view)


def test_other_hero_selection_is_hidden_until_barrier():
    game=tg.Game(1)
    before=(game.observe(0),game.observe2(0),game.player_view(0))
    game.step(1,30)
    assert (game.observe(0),game.observe2(0),game.player_view(0)) == before


def test_observation_seat_validation():
    game=tg.Game(1)
    for method in (game.observe2,game.player_view):
        with pytest.raises(ValueError):method(8)


def test_gym_observation_v2_selection():
    env=tg.BattlegroundsEnv(seed=1,observation_version=2)
    obs,_=env.reset(seed=1)
    assert obs['features'].shape==(4017,) and obs['cards'].shape==(447,)
    assert env.observation_sizes['features']==4017
    assert env.player_view()['seat']==0


def test_visible_target_locations_include_spells_without_entity_ids():
    game = recruit()
    game.debug(json.dumps(dict(cmd="add_spell", p=0, zone="shop", card="BG20_GEM")))
    game.debug(json.dumps(dict(cmd="add_spell", p=0, zone="hand", card="BG20_GEM")))
    state=json.loads(game.state_json())
    pl=state['players'][0]
    shop_index=len(pl['shop'])-1
    hand_index=len(pl['hand'])-1
    ids=[pl['shop'][shop_index]['sid'],pl['hand'][hand_index]['hid']]
    pl['pending']=[{'Target':{'options':ids,'cont':{'HandSpell':{'hid':ids[1],'opt':0}}}}]
    game=tg.Game.from_json(json.dumps(state))
    view=json.loads(game.player_view(0))
    assert [t['location'] for t in view['me']['pending']['targets']]==[
        {'zone':'shop','slot':shop_index},{'zone':'hand','slot':hand_index}]
    assert view['me']['casting_hand_index']==hand_index
