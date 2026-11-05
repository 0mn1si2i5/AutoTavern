"""Synthetic results exercise gate bookkeeping, not evidence of policy strength."""
import json

import pytest

from omnibob.guarded_report import digest, identity, opposite, report
from omnibob.evaluate import schedule


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path
    baseline = root/'baseline'/'policy.pt'
    baseline.parent.mkdir()
    baseline.write_bytes(b'baseline')
    opponent = root/'opponent.pt'
    opponent.write_bytes(b'opponent')
    runtime = {'repeat_guard': {'limit': 1}}
    write(baseline.parent/'manifest.json', {'runtime_config': runtime})
    protocol = {
        'training_seeds': [96000000, 97000000, 98000000], 'arms': [0, 1],
        'updates_per_arm': 50, 'decisions_per_arm': 204800,
        'initial_checkpoint': str(baseline), 'initial_sha256': digest(baseline),
        'config_except_seed_and_arm': {'learning_rate': .00003}, 'source_hashes': {'model.py': 'same'},
        'dev': {'offset': 91000, 'games_per_policy_per_profile': 16,
                'profiles': ['scripted', 'heuristic', 'mixed_learned'], 'opponent_checkpoint': str(opponent)},
        'final': {'offset': 160000, 'games_per_policy_per_profile': 16,
                  'bootstrap': {'samples': 10000, 'rng_seed': 20261032}}}
    write(root/'PROTOCOL.json', protocol)
    roles = ['v02', *[f'{s}-arm{a}' for s in protocol['training_seeds'] for a in (0, 1)]]
    for role in roles:
        checksum = digest(baseline)
        if role != 'v02':
            seed, arm = identity(role)
            folder = root/f'policy-{role}'
            folder.mkdir()
            (folder/'policy.pt').write_bytes(role.encode())
            checksum = digest(folder/'policy.pt')
            write(folder/'manifest.json', {
                'policy_sha256': checksum, 'update': 50, 'steps': 204800,
                'ppo_config': {'seed': seed, 'learner_repeat_limit': arm, 'learning_rate': .00003},
                'parent_checkpoint': {'sha256': digest(baseline)},
                'reference_checkpoint': {'sha256': digest(baseline)}, 'runtime_config': runtime})
            write(root/f'seed{role}'/'summary.json', {
                'status': 'updates_completed', 'error': None, 'updates': 50, 'steps': 204800,
                'truncated_episodes_this_run': 0, 'discarded_decisions_this_run': 0})
            write(root/f'seed{role}'/'manifest.json', {
                'train_config': {'seed': seed, 'learner_repeat_limit': arm, 'learning_rate': .00003},
                'source_files': protocol['source_hashes']})
            if seed == 96000000:
                probe_folder = root/f'probe-{arm}'
                write(root/f'probe-{arm}'/'summary.json', {
                    'status': 'updates_completed', 'error': None, 'updates': 3, 'steps': 12288,
                    'truncated_episodes_this_run': 0, 'discarded_decisions_this_run': 0})
                (probe_folder/'latest.pt').write_bytes(f'probe{arm}'.encode())
                write(probe_folder/'manifest.json', {
                    'parent_checkpoint': {'sha256': digest(baseline)},
                    'train_config': {'seed': seed, 'learner_repeat_limit': arm, 'learning_rate': .00003},
                    'source_files': protocol['source_hashes']})
                deployment = json.loads((folder/'manifest.json').read_text())
                deployment['parent_checkpoint'] = {'sha256': digest(probe_folder/'latest.pt')}
                write(folder/'manifest.json', deployment)
        for stage in ('dev', 'final'):
            for profile in protocol['dev']['profiles']:
                folder = root/f'{stage}-{profile}-{role}'
                write(folder/'manifest.json', {
                    'split': stage, 'opponents': profile, 'seed_offset': protocol[stage]['offset'],
                    'games_requested': 16, 'checkpoint_sha256': checksum, 'runtime': runtime,
                    'source_files': protocol['source_hashes'], 'environment': 'same',
                    'config_sha256': 'same', 'scheduling': 'shuffled_rounds',
                    'opponent_checkpoint_sha256': digest(opponent) if profile == 'mixed_learned' else None})
                # All development candidates tie. Control0/lowest seed must win.
                # That selected candidate loses at final, while its opposite arm wins.
                place = 4 if role == 'v02' else 3
                if stage == 'final' and role == '96000000-arm0':
                    place = 5
                rows = [{'seed': seed, 'seat': seat, 'status': 'completed', 'placement': place,
                         'illegal_actions': 0, 'elapsed_seconds': 1, 'decisions': 10, 'decision_ms_total': 10}
                        for seed, seat in schedule(stage, protocol[stage]['offset'], 16)]
                (folder/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return root


def test_paired_selection_and_final_cannot_switch_to_opposite(experiment):
    result = report(experiment, 'dev')
    assert result['selected_role'] == '96000000-arm0'
    assert result['completed_games'] == 336
    assert len(result['paired_alignment_minus_control']) == 3
    assert all(pair['pooled']['paired_seed_bootstrap_95ci'] == [0, 0]
               for pair in result['paired_alignment_minus_control'].values())
    write(experiment/'dev-decision.json', result)
    selected = result['selected_role']
    write(experiment/'CANDIDATE_FREEZE.json', {
        'selected_role': selected, 'opposite_role': opposite(selected),
        'policy_sha256': {role: digest(experiment/f'policy-{role}'/'policy.pt') for role in (selected, opposite(selected))},
        'protocol_sha256': digest(experiment/'PROTOCOL.json'),
        'development_decision_sha256': digest(experiment/'dev-decision.json'),
        'reporter_sha256': result['reporter_sha256']})
    final = report(experiment, 'final')
    assert final['comparisons'][opposite(selected)]['gate_passed']
    assert not final['gate_passed']
    assert final['selected_role'] == selected
    assert final['decision'] == 'no promotion; keep frozen v0.2'


@pytest.mark.parametrize('defect', ['missing_game', 'failed_game', 'wrong_runtime', 'extra_training_change', 'changed_weight', 'changed_probe'])
def test_invalid_evidence_is_rejected(experiment, defect):
    folder = experiment/'dev-scripted-97000000-arm1'
    if defect in ('missing_game', 'failed_game'):
        rows = [json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
        if defect == 'missing_game':
            rows.pop()
        else:
            rows[0]['status'] = 'failed'
        (folder/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    elif defect == 'wrong_runtime':
        path = folder/'manifest.json'; manifest = json.loads(path.read_text())
        manifest['runtime'] = {}; write(path, manifest)
    elif defect == 'extra_training_change':
        path = experiment/'policy-97000000-arm1'/'manifest.json'; manifest = json.loads(path.read_text())
        manifest['ppo_config']['gamma'] = .9; write(path, manifest)
    elif defect == 'changed_weight':
        (experiment/'policy-97000000-arm1'/'policy.pt').write_bytes(b'changed')
    else:
        (experiment/'probe-0'/'latest.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):
        report(experiment, 'dev')
