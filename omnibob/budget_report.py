"""Paired 50/200-update continuation; only new fixed-budget weights may advance."""
import argparse
import json
from pathlib import Path

import numpy as np

from .continuation_report import eligible
from .evaluate import schedule
from .guarded_report import digest, interval, read
from .opponent_report import control_for, verify_training as verify_short_training


def verify_training(spec, protocol, baseline):
    short = {**protocol, 'updates_per_candidate': protocol['control_updates'],
             'decisions_per_candidate': protocol['control_updates'] * 4096}
    if spec['kind'] == 'control':
        checksum, deployment, training = verify_short_training(spec, short, baseline)
        resume = Path(spec['resume_checkpoint'])
        if (digest(resume) != spec['resume_sha256']
                or deployment['source_checkpoint_sha256'] != spec['resume_sha256']
                or Path(deployment['source_checkpoint']).resolve() != resume.resolve()):
            raise ValueError('control training state differs from exported short-budget policy')
        return checksum, deployment, training
    if spec['kind'] != 'candidate':
        raise ValueError('unknown training role')
    control = protocol['roles'][spec['control_role']]
    if control['kind'] != 'control' or control['seed'] != spec['seed']:
        raise ValueError('wrong short-budget prefix')
    _, prefix, _ = verify_training(control, protocol, baseline)
    if any(spec[k] != control[k] for k in ('resume_checkpoint', 'resume_sha256')):
        raise ValueError('candidate resumed a different prefix')
    path = Path(spec['checkpoint'])
    deployment = read(path.with_name('manifest.json'))
    checksum = digest(path)
    if checksum != deployment['policy_sha256']:
        raise ValueError('candidate export modified')
    target, steps = protocol['updates_per_candidate'], protocol['decisions_per_candidate']
    if target != protocol['control_updates'] + protocol['additional_updates'] or steps != target * 4096:
        raise ValueError('inconsistent budget protocol')
    if (deployment['update'], deployment['steps']) != (target, steps):
        raise ValueError('candidate is not the fixed long-budget checkpoint')
    config = deployment['ppo_config']
    if config != prefix['ppo_config'] or config['opponents'] != 'mixed' or config['opponent_checkpoint'] is not None:
        raise ValueError('training configuration changed beyond update budget')
    if deployment['frozen_opponent'] is not None:
        raise ValueError('unexpected learned training opponent')
    for key in ('model_config', 'environment', 'config', 'runtime_config'):
        if deployment[key] != baseline[key]:
            raise ValueError('model/rules/deployment runtime mismatch: ' + key)
    if deployment['reference_checkpoint']['sha256'] != protocol['initial_sha256']:
        raise ValueError('reference policy changed')
    folder = Path(spec['training_dir'])
    training, summary = read(folder/'manifest.json'), read(folder/'summary.json')
    if (summary['status'], summary['error'], summary['updates'], summary['steps'],
            summary['truncated_episodes_this_run'], summary['discarded_decisions_this_run']) != (
            'updates_completed', None, target, steps, 0, 0):
        raise ValueError('long-budget training incomplete or truncated')
    if (training['train_config'] != config or training['initialize'] is not None
            or Path(training['resume']).resolve() != Path(spec['resume_checkpoint']).resolve()
            or training['updates_requested'] != protocol['additional_updates']):
        raise ValueError('continuation configuration or resume contract mismatch')
    if (training['parent_checkpoint']['sha256'] != spec['resume_sha256']
            or deployment['parent_checkpoint']['sha256'] != spec['resume_sha256']):
        raise ValueError('continuation parent changed')
    if (deployment['source_checkpoint_sha256'] != digest(folder/'latest.pt')
            or Path(deployment['source_checkpoint']).resolve() != (folder/'latest.pt').resolve()):
        raise ValueError('export differs from completed continuation state')
    for file, expected in protocol['training_source_hashes'].items():
        if training['source_files'].get(file) != expected:
            raise ValueError('training implementation changed: ' + file)
    rows = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
    expected_updates = list(range(protocol['control_updates'] + 1, target + 1))
    if [row['update'] for row in rows] != expected_updates:
        raise ValueError('missing, duplicate or extra continuation updates')
    if any(row['steps'] != row['update'] * 4096 or row['truncated_episodes']
           or row['discarded_decisions'] for row in rows):
        raise ValueError('continuation metrics contain invalid training decisions')
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
    if opponent_sha == protocol['initial_sha256']:
        raise ValueError('evaluation opponent must differ from initialization/reference')
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
        ignored = set()
        if {k: v for k, v in a.items() if k not in ignored} != {k: v for k, v in b.items() if k not in ignored}:
            raise ValueError('paired configuration differs beyond update budget')
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
            'limitations': 'All three 50-update prefixes reused, never eligible for reselection. Only total update budget changes. Times are not '
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
