import json

from omnibob.capacity_selector import NumericSalePolicy
from omnibob.capacity_trade import CapacityTradePolicy
from omnibob.observation_v2 import ObservationV2
from test_capacity_trade import PreferRefresh, full_game, observe


def test_low_numeric_sale_changes_only_sale_then_real_play_continues():
    game = full_game()
    for slot in range(7):
        game.debug(json.dumps({'cmd': 'edit_minion', 'p': 0, 'zone': 'board',
                              'index': slot, 'atk': 100, 'hp': 100}))
    game.debug(json.dumps({'cmd': 'edit_minion', 'p': 0, 'zone': 'board',
                          'index': 5, 'atk': 1, 'hp': 1}))
    obs = observe(game)
    assert CapacityTradePolicy(PreferRefresh()).act(obs).action == 16
    candidate = NumericSalePolicy(PreferRefresh())
    sell = candidate.act(obs)
    assert sell.action == 21
    game.step(0, sell.action)
    play = candidate.act(observe(game))
    assert 46 <= play.action < 126
    game.step(0, play.action)
    assert observe(game).legacy().blocks()['hand'][:, 0].sum() == 9
    assert candidate.act(observe(game)).action == 1


def test_ties_use_original_selector_and_guard_excluded_sales_stay_excluded():
    obs = observe(full_game())
    assert NumericSalePolicy(PreferRefresh()).act(obs).action == CapacityTradePolicy(PreferRefresh()).act(obs).action
    mask = obs.action_mask.copy()
    mask[16:22] = False
    assert NumericSalePolicy(PreferRefresh()).act(ObservationV2(obs.features, obs.cards, mask)).action == 22


def test_original_no_intervention_boundary_unchanged():
    obs = observe(full_game())
    for field, value in ((285, 0), (3227, 97), (68, 0)):
        f = obs.features.copy()
        f[field] = value
        altered = ObservationV2(f, obs.cards, obs.action_mask)
        assert NumericSalePolicy(PreferRefresh()).act(altered) == CapacityTradePolicy(PreferRefresh()).act(altered)
