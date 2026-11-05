import json
import os

import pytest
import taverngym as tg

from omnibob import arena
from omnibob.compat import ROOT, effective_config
from omnibob.export import export
from omnibob.observation_v2 import ObservationV2
from omnibob.offline_check import check
from omnibob.train import TrainConfig, train


@pytest.fixture(scope='module')
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp('offline-check')
    rules = effective_config(json.loads((ROOT/'configs/arena-v2.json').read_text()))
    train(root/'training', config=TrainConfig(n_envs=2, horizon=4, epochs=1, minibatch=8),
          rules=rules, updates=1, device='cpu', max_seconds=60)
    export(root/'training/latest.pt', root/'deployment', repeat_limit=1, move_budget=0)
    obs = ObservationV2.from_game(tg.Game(42, json.dumps(rules)), 0)
    payload = {key: getattr(obs, key).tolist() for key in ('features', 'cards', 'action_mask')}
    rows = [{'match_id': match, 'seat': 0, 'turn': 1, 'observation': payload} for match in ('a', 'b', 'a', 'b')]
    recording = root/'own.jsonl'
    recording.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return root/'deployment/policy.pt', recording


def test_real_export_adapter_and_interleaved_contexts(inputs, tmp_path, monkeypatch):
    monkeypatch.setenv('OMNIBOB_CHECKPOINT', 'original-setting')
    monkeypatch.setenv('OMNIBOB_JOURNAL_DIR', str(tmp_path/'unwanted-journal'))
    prior = arena._learned
    result = check(*inputs, tmp_path/'result')
    assert result['status'] == 'passed'
    assert result['checked_frames'] == 4 and result['match_seat_contexts'] == 2
    assert result['independent_context_state_verified']
    assert not result['action_mismatches'] and result['illegal_actions'] == 0
    assert arena._learned is prior
    assert os.environ['OMNIBOB_CHECKPOINT'] == 'original-setting'
    assert not (tmp_path/'unwanted-journal').exists()
    with pytest.raises(FileExistsError):
        check(*inputs, tmp_path/'result')


def test_adapter_failure_is_retained_and_environment_restored(inputs, tmp_path, monkeypatch):
    original = arena.act
    calls = 0
    def wrong_action(observation, context):
        nonlocal calls
        action = original(observation, context)
        calls += 1
        return action if calls == 1 else 245  # First call establishes isolation history.
    monkeypatch.setattr(arena, 'act', wrong_action)
    monkeypatch.delenv('OMNIBOB_CHECKPOINT', raising=False)
    with pytest.raises(ValueError, match='equivalence'):
        check(*inputs, tmp_path/'failed')
    result = json.loads((tmp_path/'failed/summary.json').read_text())
    assert result['status'] == 'failed' and result['illegal_actions'] == 4
    assert result['checked_frames'] == 4
    assert 'OMNIBOB_CHECKPOINT' not in os.environ
