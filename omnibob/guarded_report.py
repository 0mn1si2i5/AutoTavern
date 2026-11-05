"""Placement selection for the preregistered paired training-mask experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .continuation_report import eligible
from .evaluate import schedule


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def identity(role):
    seed, arm = role.split('-arm')
    return int(seed), int(arm)


def opposite(role):
    seed, arm = identity(role)
    return f'{seed}-arm{1-arm}'


def interval(values, bootstrap):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('need at least two finite complete seed blocks')
    samples = np.random.default_rng(bootstrap['rng_seed']).choice(
        values, (bootstrap['samples'], len(values)), replace=True).mean(axis=1)
    return {'mean_delta': float(values.mean()),
            'paired_seed_bootstrap_95ci': np.percentile(samples, [2.5, 97.5]).tolist()}


def select(comparisons):
    passing = [role for role, result in comparisons.items() if result['gate_passed']]
    return min(passing, key=lambda role: (
        comparisons[role]['pooled']['mean_delta'], identity(role)[1], identity(role)[0])) if passing else None


def report(root, stage):
    root = Path(root)
    protocol = read(root/'PROTOCOL.json')
    roles = ['v02', *[f'{seed}-arm{arm}' for seed in protocol['training_seeds'] for arm in protocol['arms']]]
    frozen = None
    if stage == 'final':
        frozen = read(root/'CANDIDATE_FREEZE.json')
        selected = frozen['selected_role']
        if selected not in roles[1:] or frozen['opposite_role'] != opposite(selected):
            raise ValueError('unknown candidate or wrong paired arm')
        if frozen['protocol_sha256'] != digest(root/'PROTOCOL.json'):
            raise ValueError('protocol changed after selection')
        if frozen['development_decision_sha256'] != digest(root/'dev-decision.json'):
            raise ValueError('development decision changed after selection')
        if read(root/'dev-decision.json')['selected_role'] != selected:
            raise ValueError('candidate differs from development selection')
        if frozen['reporter_sha256'] != digest(__file__):
            raise ValueError('reporter changed after selection')
        roles = ['v02', selected, opposite(selected)]
    elif stage != 'dev':
        raise ValueError('unknown stage')
    expected = set(schedule(stage, protocol[stage]['offset'], protocol[stage]['games_per_policy_per_profile']))
    seeds = sorted({seed for seed, _ in expected})
    if len(expected) != 8*len(seeds) or len(seeds) < 2:
        raise ValueError('evaluation requires complete eight-seat blocks')
    profiles = protocol['dev']['profiles']
    bootstrap = protocol['final']['bootstrap']
    baseline = read(Path(protocol['initial_checkpoint']).parent/'manifest.json')
    if digest(protocol['initial_checkpoint']) != protocol['initial_sha256']:
        raise ValueError('frozen baseline modified')
    expected_runtime = baseline['runtime_config']
    manifests, means, metrics, deployments, inputs = [], {}, {}, {}, {}
    for role in roles:
        checkpoint_digest = protocol['initial_sha256']
        if role != 'v02':
            folder = root/f'policy-{role}'
            deployment = read(folder/'manifest.json')
            deployments[role] = deployment
            checkpoint_digest = digest(folder/'policy.pt')
            if checkpoint_digest != deployment['policy_sha256']:
                raise ValueError('deployment checkpoint modified')
            if frozen and checkpoint_digest != frozen['policy_sha256'][role]:
                raise ValueError('frozen candidate modified')
            if (deployment['update'], deployment['steps']) != (protocol['updates_per_arm'], protocol['decisions_per_arm']):
                raise ValueError('not the fixed-update checkpoint')
            config = deployment['ppo_config']
            if (config['seed'], config['learner_repeat_limit']) != identity(role):
                raise ValueError('training role mismatch')
            if any(config[k] != v for k, v in protocol['config_except_seed_and_arm'].items()):
                raise ValueError('training configuration mismatch')
            if deployment['reference_checkpoint']['sha256'] != protocol['initial_sha256']:
                raise ValueError('training reference mismatch')
            if deployment['runtime_config'] != expected_runtime:
                raise ValueError('deployment runtime mismatch')
            summary = read(root/f'seed{role}'/'summary.json')
            if (summary['status'], summary['error'], summary['updates'], summary['steps']) != (
                    'updates_completed', None, protocol['updates_per_arm'], protocol['decisions_per_arm']):
                raise ValueError('training did not complete normally')
            if summary['truncated_episodes_this_run'] or summary['discarded_decisions_this_run']:
                raise ValueError('training truncation/discard detected')
            training = read(root/f'seed{role}'/'manifest.json')
            if training['train_config'] != config:
                raise ValueError('exported config differs from training manifest')
            for file, checksum in protocol['source_hashes'].items():
                if training['source_files'].get(file) != checksum:
                    raise ValueError('training source mismatch: '+file)
            if identity(role)[0] == protocol['training_seeds'][0]:
                probe_folder = root/f'probe-{identity(role)[1]}'
                probe = read(probe_folder/'summary.json')
                if (probe['status'], probe['error'], probe['updates'], probe['steps'],
                        probe['truncated_episodes_this_run'], probe['discarded_decisions_this_run']) != (
                        'updates_completed', None, 3, 12288, 0, 0):
                    raise ValueError('resumed probe was not a successful three-update run')
                probe_manifest = read(probe_folder/'manifest.json')
                if (deployment['parent_checkpoint']['sha256'] != digest(probe_folder/'latest.pt')
                        or probe_manifest['parent_checkpoint']['sha256'] != protocol['initial_sha256']
                        or probe_manifest['train_config'] != config):
                    raise ValueError('training resume lineage mismatch')
                for file, checksum in protocol['source_hashes'].items():
                    if probe_manifest['source_files'].get(file) != checksum:
                        raise ValueError('probe source mismatch: '+file)
            elif deployment['parent_checkpoint']['sha256'] != protocol['initial_sha256']:
                raise ValueError('training initialization mismatch')
        means[role], metrics[role] = {}, {}
        for profile in profiles:
            folder = root/f'{stage}-{profile}-{role}'
            manifest = read(folder/'manifest.json')
            if (manifest['split'], manifest['opponents'], manifest['seed_offset'], manifest['games_requested'], manifest['checkpoint_sha256']) != (
                    stage, profile, protocol[stage]['offset'], len(expected), checkpoint_digest):
                raise ValueError('evaluation contract mismatch')
            if manifest['runtime'] != expected_runtime:
                raise ValueError('evaluation runtime mismatch')
            for file, checksum in protocol['source_hashes'].items():
                if manifest['source_files'].get(file) != checksum:
                    raise ValueError('frozen source mismatch: '+file)
            for prior in manifests:
                for key in ('environment', 'config_sha256', 'scheduling', 'runtime'):
                    if prior[key] != manifest[key]:
                        raise ValueError('different environment/runtime: '+key)
            if profile == 'mixed_learned' and manifest['opponent_checkpoint_sha256'] != digest(protocol['dev']['opponent_checkpoint']):
                raise ValueError('heldout opponent mismatch')
            manifests.append(manifest)
            rows = [json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
            index = {(row['seed'], row['seat']): row for row in rows}
            if len(index) != len(rows) or set(index) != expected:
                raise ValueError('missing or duplicate games')
            if any(row['status'] != 'completed' or row['illegal_actions'] or not 1 <= row['placement'] <= 8 for row in rows):
                raise ValueError('failed, illegal or incomplete games')
            means[role][profile] = np.array([np.mean([index[(seed, seat)]['placement'] for seat in range(8)]) for seed in seeds])
            decisions = sum(row['decisions'] for row in rows)
            metrics[role][profile] = {
                'mean_placement': float(means[role][profile].mean()),
                'win_rate': float(np.mean([row['placement'] == 1 for row in rows])),
                'top4_rate': float(np.mean([row['placement'] <= 4 for row in rows])),
                'completed': len(rows), 'failed': 0, 'interrupted': 0, 'illegal_actions': 0,
                'elapsed_seconds_sum': sum(row['elapsed_seconds'] for row in rows),
                'decision_count': decisions,
                'decision_mean_ms': sum(row['decision_ms_total'] for row in rows)/decisions}
            inputs[str(folder)] = {'manifest_sha256': digest(folder/'manifest.json'),
                                   'games_sha256': digest(folder/'games.jsonl')}
    for role, deployment in deployments.items():
        other = deployments.get(opposite(role))
        if other:
            a = {k: v for k, v in deployment['ppo_config'].items() if k != 'learner_repeat_limit'}
            b = {k: v for k, v in other['ppo_config'].items() if k != 'learner_repeat_limit'}
            if a != b:
                raise ValueError('matched training pair differs in more than mask alignment')

    def contrast(candidate, control):
        deltas = {profile: means[candidate][profile]-means[control][profile] for profile in profiles}
        per = {profile: interval(values, bootstrap) for profile, values in deltas.items()}
        pooled = interval(np.mean(list(deltas.values()), axis=0), bootstrap)
        return {'profiles': per, 'pooled': pooled, 'gate_passed': eligible(per, pooled, stage)}

    comparisons = {role: contrast(role, 'v02') for role in roles[1:]}
    paired = {str(seed): contrast(f'{seed}-arm1', f'{seed}-arm0') for seed in protocol['training_seeds']
              if f'{seed}-arm1' in roles and f'{seed}-arm0' in roles}
    # Secondary comparisons cannot replace the one candidate frozen on development.
    selected = select(comparisons) if stage == 'dev' else frozen['selected_role']
    passed = selected is not None and comparisons[selected]['gate_passed']
    return {
        'stage': stage, 'completed_games': len(manifests)*len(expected), 'seed_blocks': len(seeds), 'seeds': seeds,
        'metrics': metrics, 'comparisons': comparisons, 'paired_alignment_minus_control': paired,
        'selected_role': selected, 'gate_passed': passed, 'inputs': inputs,
        'protocol_sha256': digest(root/'PROTOCOL.json'), 'reporter_sha256': digest(__file__),
        'decision': ('freeze candidate for independent final' if stage == 'dev' else 'eligible for engineering/package checks')
                    if passed else 'no promotion; keep frozen v0.2',
        'limitations': 'Local synthetic opponents. Development intervals do not correct for six-way selection. '
                       'Three training pairs share one parent. Secondary arm contrast is not the promotion gate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--stage', required=True, choices=['dev', 'final'])
    args = parser.parse_args()
    result = report(args.root, args.stage)
    with (args.root/f'{args.stage}-decision.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ('metrics', 'inputs', 'seeds')}, indent=2))


if __name__ == '__main__':
    main()
