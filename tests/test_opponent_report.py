"""Synthetic bookkeeping checks; fixtures are not strength evidence."""
import json

import pytest

from omnibob.evaluate import schedule
from omnibob.opponent_report import digest, read, report


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path
    baseline = root/'baseline/policy.pt';baseline.parent.mkdir();baseline.write_bytes(b'v02')
    heldout = root/'heldout.pt';heldout.write_bytes(b'heldout')
    runtime = {'repeat_guard': {'limit': 1}}
    same = {'model_config': {'width': 64}, 'environment': 'same', 'config': {'limit': 100}, 'runtime_config': runtime}
    write(baseline.with_name('manifest.json'), same)
    roles = {'v02': {'kind': 'baseline', 'checkpoint': str(baseline), 'sha256': digest(baseline)}}
    source = {'model.py': 'same'}
    for kind in ('control', 'candidate'):
        role = f'96000000-{kind}';folder = root/role;folder.mkdir();(folder/'policy.pt').write_bytes(role.encode())
        probe = folder/'probe';probe.mkdir();(probe/'latest.pt').write_bytes(f'probe-{kind}'.encode())
        roles[role] = {'kind': kind, 'seed': 96000000, 'checkpoint': str(folder/'policy.pt'),
                       'training_dir': str(folder/'training'), 'probe_dir': str(probe)}
        if kind == 'control':roles[role]['sha256'] = digest(folder/'policy.pt')
        config = {'seed': 96000000, 'learning_rate': .00003, 'learner_repeat_limit': 0,
                  'opponents': 'mixed' if kind == 'control' else 'mixed_learned',
                  'opponent_checkpoint': None if kind == 'control' else str(baseline)}
        write(folder/'manifest.json', {**same, 'policy_sha256': digest(folder/'policy.pt'), 'update': 50, 'steps': 204800,
              'ppo_config': config, 'reference_checkpoint': {'sha256': digest(baseline)},
              'parent_checkpoint': {'sha256': digest(probe/'latest.pt')},
              'frozen_opponent': None if kind == 'control' else {'sha256': digest(baseline)}})
        training = {'train_config': config, 'source_files': source, 'model_config': same['model_config'],
                    'environment': 'same', 'config_sha256': 'same', 'torch': 'same', 'device': 'mps',
                    'python': 'same', 'numpy': 'same', 'machine': 'arm64', 'parent_checkpoint': {'sha256': digest(baseline)}}
        summary = {'status': 'updates_completed', 'error': None, 'updates': 50, 'steps': 204800,
                   'truncated_episodes_this_run': 0, 'discarded_decisions_this_run': 0}
        write(folder/'training/manifest.json', training);write(folder/'training/summary.json', summary)
        write(probe/'manifest.json', training);write(probe/'summary.json', {**summary, 'updates': 3, 'steps': 12288})
    protocol = {'roles': roles, 'training_seeds': [96000000], 'initial_checkpoint': str(baseline),
                'initial_sha256': digest(baseline), 'training_opponent_sha256': digest(baseline),
                'common_config': {'learning_rate': .00003, 'learner_repeat_limit': 0},
                'updates_per_candidate': 50, 'decisions_per_candidate': 204800,
                'training_source_hashes': source, 'source_hashes': source,
                'dev': {'offset': 101000, 'games_per_policy_per_profile': 16,
                        'profiles': ['scripted', 'heuristic', 'mixed_learned'], 'opponent_checkpoint': str(heldout)},
                'final': {'offset': 180000, 'games_per_policy_per_profile': 16,
                          'bootstrap': {'samples': 10000, 'rng_seed': 20261033}}}
    write(root/'PROTOCOL.json', protocol)
    for stage in ('dev', 'final'):
        for role, spec in roles.items():
            for profile in protocol['dev']['profiles']:
                folder = root/f'{stage}-{profile}-{role}'
                write(folder/'manifest.json', {
                    'split': stage, 'opponents': profile, 'seed_offset': protocol[stage]['offset'],
                    'games_requested': 16, 'checkpoint_sha256': digest(spec['checkpoint']), 'runtime': runtime,
                    'source_files': source, 'environment': 'same', 'config_sha256': 'same', 'scheduling': 'shuffled_rounds',
                    'opponent_checkpoint_sha256': digest(heldout) if profile == 'mixed_learned' else None})
                placement = 4 if role == 'v02' else 1 if spec['kind'] == 'control' else 3 if stage == 'dev' else 5
                rows = [{'seed': seed, 'seat': seat, 'status': 'completed', 'placement': placement,
                         'illegal_actions': 0, 'elapsed_seconds': 1, 'decisions': 10, 'decision_ms_total': 10}
                        for seed, seat in schedule(stage, protocol[stage]['offset'], 16)]
                (folder/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return root


def test_reused_control_never_selected_and_final_never_switches(experiment):
    result = report(experiment, 'dev')
    assert result['selected_role'] == '96000000-candidate'
    assert result['comparisons']['96000000-control']['pooled']['mean_delta'] == -3
    assert result['completed_games'] == 144
    write(experiment/'dev-decision.json', result)
    roles = read(experiment/'PROTOCOL.json')['roles']
    write(experiment/'CANDIDATE_FREEZE.json', {
        'selected_role': result['selected_role'], 'control_role': '96000000-control',
        'policy_sha256': {role: digest(spec['checkpoint']) for role, spec in roles.items() if role != 'v02'},
        'protocol_sha256': digest(experiment/'PROTOCOL.json'),
        'development_decision_sha256': digest(experiment/'dev-decision.json'), 'reporter_sha256': result['reporter_sha256']})
    final = report(experiment, 'final')
    assert final['selected_role'] == '96000000-candidate'
    assert not final['gate_passed']
    assert final['comparisons']['96000000-control']['gate_passed']


@pytest.mark.parametrize('defect', ['control_modified', 'extra_factor', 'heldout_training', 'missing_game', 'failed_game'])
def test_invalid_experiment_evidence_is_rejected(experiment, defect):
    if defect == 'control_modified':
        (experiment/'96000000-control/policy.pt').write_bytes(b'changed')
    elif defect == 'extra_factor':
        path = experiment/'96000000-candidate/manifest.json';manifest = read(path)
        manifest['ppo_config']['gamma'] = .9;write(path, manifest)
        for suffix in ('training', 'probe'):
            path = experiment/f'96000000-candidate/{suffix}/manifest.json';training = read(path)
            training['train_config']['gamma'] = .9;write(path, training)
    elif defect == 'heldout_training':
        path = experiment/'PROTOCOL.json';protocol = read(path)
        protocol['dev']['opponent_checkpoint'] = protocol['initial_checkpoint'];write(path, protocol)
    else:
        path = experiment/'dev-scripted-96000000-candidate/games.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if defect == 'missing_game':rows.pop()
        else:rows[0]['status'] = 'failed'
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError):report(experiment, 'dev')
