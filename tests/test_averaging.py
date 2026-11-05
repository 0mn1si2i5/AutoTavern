import copy

import pytest
import torch

from omnibob.average import average
from omnibob.evaluate import run_game
from omnibob.export import export
from omnibob.learned import load_policy
from omnibob.train import TrainConfig, atomic_checkpoint, load_checkpoint, train


def test_weight_average_preserves_provenance_and_runs_complete_game(tmp_path):
    cfg = TrainConfig(seed=880000, n_envs=1, horizon=2, epochs=1, minibatch=2, cpu_threads=1)
    train(tmp_path/'initial', cfg, updates=1, max_seconds=60)
    for name, seed in [('a', 881000), ('b', 882000)]:
        cfg.seed = seed
        train(tmp_path/name, cfg, initialize=tmp_path/'initial/latest.pt', updates=1, max_seconds=60)
        export(tmp_path/name/'latest.pt', tmp_path/(name+'-deployment'), move_budget=7, repeat_limit=1)
    inputs = [tmp_path/(name+'-deployment')/'policy.pt' for name in ('a','b')]
    average(inputs, tmp_path/'average')
    a, b, merged = [load_checkpoint(p) for p in [*inputs, tmp_path/'average/policy.pt']]
    for key, tensor in merged['model'].items():
        expected = ((a['model'][key].double()+b['model'][key].double())/2).to(tensor.dtype) if tensor.is_floating_point() else a['model'][key]
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    assert merged['steps'] == 0 and merged['runtime_config'] == {}
    assert len(merged['weight_averaging']['parents']) == 2
    assert merged['weight_averaging']['weights'] == [.5,.5]
    export(tmp_path/'average/policy.pt', tmp_path/'deployment', move_budget=7, repeat_limit=1)
    exported = load_checkpoint(tmp_path/'deployment/policy.pt')
    assert exported['weight_averaging'] == merged['weight_averaging']
    policy = load_policy(tmp_path/'deployment/policy.pt')
    assert 'uniform-average-2' in policy.version
    result, _ = run_game('ppo', 881234, 0, policy.config, policy_override=policy)
    assert result['status'] == 'completed' and result['illegal_actions'] == 0
    with pytest.raises(ValueError, match='distinct'):
        average([inputs[0], inputs[0]], tmp_path/'duplicate')
    with pytest.raises(ValueError, match='nested'):
        average([inputs[0], tmp_path/'average/policy.pt'], tmp_path/'nested')
    changed = copy.deepcopy(b)
    changed['parent_checkpoint']['sha256'] = 'different'
    atomic_checkpoint(tmp_path/'different-parent.pt', changed)
    with pytest.raises(ValueError, match='same recorded initialization'):
        average([inputs[0], tmp_path/'different-parent.pt'], tmp_path/'bad-parent')
    changed = copy.deepcopy(b)
    changed['model'][next(iter(changed['model']))].flatten()[0] = float('nan')
    atomic_checkpoint(tmp_path/'nonfinite.pt', changed)
    with pytest.raises(ValueError, match='nonfinite'):
        average([inputs[0], tmp_path/'nonfinite.pt'], tmp_path/'bad-numerics')


def test_resumed_branches_require_verified_common_ancestor(tmp_path):
    cfg = TrainConfig(seed=883000, n_envs=1, horizon=2, epochs=1, minibatch=2, cpu_threads=1)
    train(tmp_path/'root', cfg, updates=1, max_seconds=60)
    ancestor = tmp_path/'root/latest.pt'
    inputs = []
    for name, seed in [('a', 884000), ('b', 885000)]:
        cfg.seed = seed
        train(tmp_path/name, cfg, initialize=ancestor, updates=1, max_seconds=60)
        train(tmp_path/(name+'-resume'), resume=tmp_path/name/'latest.pt', updates=1, max_seconds=60)
        inputs.append(tmp_path/(name+'-resume')/'latest.pt')
    with pytest.raises(ValueError, match='same recorded initialization'):
        average(inputs, tmp_path/'old-default')
    average(inputs, tmp_path/'merged', initialization=ancestor)
    merged = load_checkpoint(tmp_path/'merged/policy.pt')
    definition = merged['weight_averaging']
    assert len(definition['ancestry']) == 2
    assert all(len(chain) == 3 for chain in definition['ancestry'])
    states = [load_checkpoint(p) for p in inputs]
    for key, value in merged['model'].items():
        expected = torch.stack([s['model'][key].double() for s in states]).mean(0).to(value.dtype)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='ancestor'):
        average(inputs, tmp_path/'wrong-root', initialization=tmp_path/'a/latest.pt')
    parent = load_checkpoint(tmp_path/'a/latest.pt'); parent['steps'] += 1
    atomic_checkpoint(tmp_path/'a/latest.pt', parent)
    with pytest.raises(ValueError, match='hash'):
        average(inputs, tmp_path/'modified-parent', initialization=ancestor)
