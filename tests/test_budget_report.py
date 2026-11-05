"""Budget bookkeeping tests; synthetic placements are not strength evidence."""
import json

import pytest

from test_opponent_report import experiment as opposition_experiment, write
from omnibob.budget_report import digest, read, report


@pytest.fixture
def experiment(opposition_experiment):
    root = opposition_experiment
    protocol = read(root/'PROTOCOL.json')
    protocol.update(control_updates=50, updates_per_candidate=200,
                    decisions_per_candidate=819200, additional_updates=150)
    protocol.pop('training_opponent_sha256')
    control = protocol['roles']['96000000-control']
    candidate = protocol['roles']['96000000-candidate']
    prefix = root/'96000000-control/training/latest.pt'
    prefix.write_bytes(b'50-update-optimizer-and-environments')
    latest = root/'96000000-candidate/training/latest.pt'
    latest.write_bytes(b'200-update-optimizer-and-environments')
    for spec in (control, candidate):
        spec.update(resume_checkpoint=str(prefix), resume_sha256=digest(prefix))
    candidate.update(control_role='96000000-control', probe_dir=None)
    cp = root/'96000000-control/manifest.json'
    short = read(cp)
    short.update(source_checkpoint=str(prefix), source_checkpoint_sha256=digest(prefix))
    write(cp, short)
    path = root/'96000000-candidate/manifest.json'
    long = read(path)
    long.update(ppo_config=short['ppo_config'], frozen_opponent=None, update=200, steps=819200,
                parent_checkpoint={'sha256': digest(prefix)}, source_checkpoint=str(latest),
                source_checkpoint_sha256=digest(latest))
    write(path, long)
    path = root/'96000000-candidate/training/manifest.json'
    training = read(path)
    training.update(train_config=short['ppo_config'], resume=str(prefix), initialize=None,
                    updates_requested=150, parent_checkpoint={'sha256': digest(prefix)})
    write(path, training)
    path = root/'96000000-candidate/training/summary.json'
    summary = read(path)
    summary.update(updates=200, steps=819200)
    write(path, summary)
    rows = [{'update': i, 'steps': i*4096, 'truncated_episodes': 0,
             'discarded_decisions': 0} for i in range(51, 201)]
    (latest.parent/'metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    write(root/'PROTOCOL.json', protocol)
    return root


def test_only_long_budget_candidate_can_advance_and_final_cannot_switch(experiment):
    result = report(experiment, 'dev')
    assert result['completed_games'] == 144
    assert result['selected_role'] == '96000000-candidate'
    assert result['comparisons']['96000000-candidate']['pooled']['mean_delta'] == -1
    assert result['comparisons']['96000000-control']['pooled']['mean_delta'] == -3
    write(experiment/'dev-decision.json', result)
    roles = read(experiment/'PROTOCOL.json')['roles']
    write(experiment/'CANDIDATE_FREEZE.json', {
        'selected_role': result['selected_role'], 'control_role': '96000000-control',
        'policy_sha256': {role: digest(spec['checkpoint']) for role, spec in roles.items() if role != 'v02'},
        'protocol_sha256': digest(experiment/'PROTOCOL.json'),
        'development_decision_sha256': digest(experiment/'dev-decision.json'),
        'reporter_sha256': result['reporter_sha256']})
    final = report(experiment, 'final')
    assert final['selected_role'] == '96000000-candidate'
    assert not final['gate_passed']
    assert final['comparisons']['96000000-control']['gate_passed']


@pytest.mark.parametrize('defect', ['prefix_changed', 'extra_factor', 'short_checkpoint',
                                   'missing_update', 'wrong_resume', 'failed_game', 'duplicate_game'])
def test_invalid_continuation_or_evaluation_is_rejected(experiment, defect):
    if defect == 'prefix_changed':
        (experiment/'96000000-control/training/latest.pt').write_bytes(b'changed')
    elif defect in ('extra_factor', 'short_checkpoint'):
        path = experiment/'96000000-candidate/manifest.json'
        data = read(path)
        if defect == 'extra_factor':
            data['ppo_config']['learning_rate'] *= 2
        else:
            data.update(update=199, steps=199*4096)
        write(path, data)
    elif defect == 'wrong_resume':
        path = experiment/'96000000-candidate/training/manifest.json'
        data = read(path)
        data['resume'] = 'a-different-checkpoint.pt'
        write(path, data)
    else:
        path = (experiment/'96000000-candidate/training/metrics.jsonl' if defect == 'missing_update'
                else experiment/'dev-scripted-96000000-candidate/games.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if defect == 'missing_update':
            rows.pop(5)
        elif defect == 'failed_game':
            rows[0]['status'] = 'failed'
        else:
            rows[-1] = rows[0]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):
        report(experiment, 'dev')
