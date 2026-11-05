"""Fixed-budget learned-opponent experiment with immutable reused controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from .continuation_report import eligible
from .evaluate import schedule
from .guarded_report import digest, interval, read


def control_for(protocol, candidate):
    spec = protocol['roles'][candidate]
    if spec['kind'] != 'candidate':
        raise ValueError('only newly trained candidates may be selected')
    controls = [role for role, item in protocol['roles'].items()
                if item['kind'] == 'control' and item['seed'] == spec['seed']]
    if len(controls) != 1:
        raise ValueError('need exactly one predefined matched control')
    return controls[0]


def verify_training(spec, protocol, baseline):
    path = Path(spec['checkpoint'])
    deployment = read(path.with_name('manifest.json'))
    checksum = digest(path)
    if checksum != deployment['policy_sha256'] or spec.get('sha256', checksum) != checksum:
        raise ValueError('deployment or reused control modified')
    if (deployment['update'], deployment['steps']) != (protocol['updates_per_candidate'], protocol['decisions_per_candidate']):
        raise ValueError('wrong fixed training budget')
    config = deployment['ppo_config']
    if config['seed'] != spec['seed'] or any(config[key] != value for key, value in protocol['common_config'].items()):
        raise ValueError('common training configuration mismatch')
    if spec['kind'] == 'control':
        if config['opponents'] != 'mixed' or config['opponent_checkpoint'] is not None or deployment['frozen_opponent'] is not None:
            raise ValueError('reused control is not the native-mask mixed-opponent arm')
    elif spec['kind'] == 'candidate':
        if (config['opponents'] != 'mixed_learned'
                or Path(config['opponent_checkpoint']).resolve() != Path(protocol['initial_checkpoint']).resolve()
                or deployment['frozen_opponent']['sha256'] != protocol['training_opponent_sha256']):
            raise ValueError('wrong frozen training opposition')
    else:
        raise ValueError('unknown training role')
    for key in ('model_config', 'environment', 'config', 'runtime_config'):
        if deployment[key] != baseline[key]:
            raise ValueError('model/rules/deployment runtime mismatch: '+key)
    if deployment['reference_checkpoint']['sha256'] != protocol['initial_sha256']:
        raise ValueError('wrong reference policy')
    folder = Path(spec['training_dir'])
    training = read(folder/'manifest.json')
    summary = read(folder/'summary.json')
    if (summary['status'], summary['error'], summary['updates'], summary['steps'],
            summary['truncated_episodes_this_run'], summary['discarded_decisions_this_run']) != (
            'updates_completed', None, protocol['updates_per_candidate'], protocol['decisions_per_candidate'], 0, 0):
        raise ValueError('training incomplete or truncated')
    if training['train_config'] != config:
        raise ValueError('training/export configuration mismatch')
    sources = [training]
    if spec['probe_dir']:
        probe_folder = Path(spec['probe_dir'])
        probe = read(probe_folder/'summary.json')
        probe_manifest = read(probe_folder/'manifest.json')
        if (probe['status'], probe['error'], probe['updates'], probe['steps'],
                probe['truncated_episodes_this_run'], probe['discarded_decisions_this_run']) != (
                'updates_completed', None, 3, 12288, 0, 0):
            raise ValueError('invalid three-update prefix')
        if (deployment['parent_checkpoint']['sha256'] != digest(probe_folder/'latest.pt')
                or probe_manifest['parent_checkpoint']['sha256'] != protocol['initial_sha256']
                or probe_manifest['train_config'] != config):
            raise ValueError('resume lineage mismatch')
        sources.append(probe_manifest)
    elif deployment['parent_checkpoint']['sha256'] != protocol['initial_sha256']:
        raise ValueError('initialization mismatch')
    for source in sources:
        for file, expected in protocol['training_source_hashes'].items():
            if source['source_files'].get(file) != expected:
                raise ValueError('training source differs: '+file)
    return checksum, deployment, training


def report(root, stage):
    root = Path(root)
    protocol = read(root/'PROTOCOL.json')
    roles = list(protocol['roles'])
    frozen = None
    if stage == 'final':
        frozen = read(root/'CANDIDATE_FREEZE.json')
        development = read(root/'dev-decision.json')
        selected = frozen['selected_role']
        if (frozen['protocol_sha256'] != digest(root/'PROTOCOL.json')
                or frozen['development_decision_sha256'] != digest(root/'dev-decision.json')
                or frozen['reporter_sha256'] != digest(__file__)
                or not development['gate_passed'] or development['selected_role'] != selected):
            raise ValueError('frozen selection/protocol/reporter changed')
        control = control_for(protocol, selected)
        if frozen['control_role'] != control:
            raise ValueError('wrong frozen control')
        roles = ['v02', selected, control]
    elif stage != 'dev':
        raise ValueError('unknown stage')
    baseline = read(Path(protocol['initial_checkpoint']).with_name('manifest.json'))
    if digest(protocol['initial_checkpoint']) != protocol['initial_sha256']:
        raise ValueError('frozen v0.2 modified')
    opponent_sha = digest(protocol['dev']['opponent_checkpoint'])
    if opponent_sha == protocol['training_opponent_sha256']:
        raise ValueError('heldout evaluation opponent also used for training')
    expected = set(schedule(stage, protocol[stage]['offset'], protocol[stage]['games_per_policy_per_profile']))
    seeds = sorted({seed for seed, _ in expected})
    if len(seeds) < 2 or len(expected) != len(seeds)*8:
        raise ValueError('need complete eight-seat seed blocks')
    profiles = protocol['dev']['profiles']
    metrics, means, training, deployments, inputs, manifests = {}, {}, {}, {}, {}, []
    for role in roles:
        spec = protocol['roles'][role]
        checksum = protocol['initial_sha256']
        if role != 'v02':
            checksum, deployments[role], training[role] = verify_training(spec, protocol, baseline)
            if frozen and checksum != frozen['policy_sha256'][role]:
                raise ValueError('frozen policy changed')
        elif spec['kind'] != 'baseline' or digest(spec['checkpoint']) != checksum:
            raise ValueError('baseline role mismatch')
        metrics[role], means[role] = {}, {}
        for profile in profiles:
            folder = root/f'{stage}-{profile}-{role}'
            manifest = read(folder/'manifest.json')
            if (manifest['split'], manifest['opponents'], manifest['seed_offset'], manifest['games_requested'], manifest['checkpoint_sha256']) != (
                    stage, profile, protocol[stage]['offset'], len(expected), checksum):
                raise ValueError('evaluation contract mismatch')
            if manifest['runtime'] != baseline['runtime_config']:
                raise ValueError('evaluation runtime changed')
            for file, expected_sha in protocol['source_hashes'].items():
                if manifest['source_files'].get(file) != expected_sha:
                    raise ValueError('evaluation source changed: '+file)
            for previous in manifests:
                for key in ('environment', 'config_sha256', 'scheduling', 'runtime'):
                    if previous[key] != manifest[key]:
                        raise ValueError('different evaluation conditions: '+key)
            if profile == 'mixed_learned' and manifest['opponent_checkpoint_sha256'] != opponent_sha:
                raise ValueError('wrong heldout evaluation opponent')
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
                'elapsed_seconds_sum': sum(row['elapsed_seconds'] for row in rows), 'decisions': decisions,
                'decision_mean_ms': sum(row['decision_ms_total'] for row in rows)/decisions}
            inputs[str(folder)] = {'manifest_sha256': digest(folder/'manifest.json'), 'games_sha256': digest(folder/'games.jsonl')}
    candidates = [role for role in roles if protocol['roles'][role]['kind'] == 'candidate']
    for candidate in candidates:
        control = control_for(protocol, candidate)
        a, b = deployments[candidate]['ppo_config'], deployments[control]['ppo_config']
        ignored = {'opponents', 'opponent_checkpoint'}
        if {k: v for k, v in a.items() if k not in ignored} != {k: v for k, v in b.items() if k not in ignored}:
            raise ValueError('paired configuration differs beyond opposition')
        for key in ('model_config', 'environment', 'config_sha256', 'torch', 'device', 'python', 'numpy', 'machine'):
            if training[candidate][key] != training[control][key]:
                raise ValueError('paired training conditions differ: '+key)

    def contrast(candidate, control):
        deltas = {profile: means[candidate][profile]-means[control][profile] for profile in profiles}
        per = {profile: interval(values, protocol['final']['bootstrap']) for profile, values in deltas.items()}
        pooled = interval(np.mean(list(deltas.values()), axis=0), protocol['final']['bootstrap'])
        return {'profiles': per, 'pooled': pooled, 'gate_passed': eligible(per, pooled, stage)}

    comparisons = {role: contrast(role, 'v02') for role in roles if role != 'v02'}
    paired = {role: contrast(role, control_for(protocol, role)) for role in candidates}
    passing = [role for role in candidates if comparisons[role]['gate_passed']]
    selected = (min(passing, key=lambda role: (comparisons[role]['pooled']['mean_delta'], protocol['roles'][role]['seed']))
                if passing else None) if stage == 'dev' else frozen['selected_role']
    passed = selected is not None and comparisons[selected]['gate_passed']
    return {'stage': stage, 'completed_games': len(manifests)*len(expected), 'seed_blocks': len(seeds), 'seeds': seeds,
            'metrics': metrics, 'comparisons': comparisons, 'candidate_minus_matched_control': paired,
            'selected_role': selected, 'gate_passed': passed, 'inputs': inputs,
            'protocol_sha256': digest(root/'PROTOCOL.json'), 'reporter_sha256': digest(__file__),
            'decision': ('freeze candidate for independent final' if stage == 'dev' else 'eligible for engineering/package checks')
                        if passed else 'no promotion; keep frozen v0.2',
            'limitations': 'All three predefined controls reused, never eligible for reselection. Training times are not '
                           'contemporaneous causal comparisons. Local synthetic opponents; no real-match calibration. '
                           'Development intervals do not correct for selection among three candidates or repeated research.'}


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
