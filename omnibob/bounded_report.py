"""Paired placement bounds for pre-registered local experiments with censored games.

A turn-cap result has unknown placement in [1, 8]. Never use the engine's
health-based forced rank. Missing games, errors and combat cutoffs remain invalid.
The pessimistic bound, not complete-case performance, controls promotion.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import schedule


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def placement_bounds(row):
    if row.get('illegal_actions') != 0 or row.get('error') is not None:
        raise ValueError('invalid action or failed game')
    if row.get('truncated_combats') != 0:
        raise ValueError('combat cutoff is not an ordinary turn-cap censor')
    if row['status'] == 'completed' and row.get('turn_limit_end') is False:
        place = row['placement']
        if type(place) is not int or not 1 <= place <= 8:
            raise ValueError('invalid placement')
        if sorted(row['placements']) != list(range(1, 9)) or row['placements'][row['seat']] != place:
            raise ValueError('inconsistent placements')
        return place, place
    if row['status'] == 'truncated' and row.get('turn_limit_end') is True:
        return 1, 8
    raise ValueError('incomplete or inconsistent game')


def indexed_bounds(rows, expected):
    result = {}
    for row in rows:
        key = row['seed'], row['seat']
        if key in result:
            raise ValueError('duplicate game')
        result[key] = placement_bounds(row)
    if set(result) != set(expected):
        raise ValueError('missing or unexpected seed/seat')
    return np.array([result[key] for key in expected], dtype=float)


def paired_bounds(candidate, baseline, indices):
    lower = (candidate[:, 0] - baseline[:, 1]).reshape(-1, 8).mean(1)
    upper = (candidate[:, 1] - baseline[:, 0]).reshape(-1, 8).mean(1)
    return {
        'mean_delta_bounds': [float(lower.mean()), float(upper.mean())],
        'pessimistic_bootstrap_95ci': np.percentile(upper[indices].mean(1), [2.5, 97.5]).tolist(),
        'interval_scope': 'Whole eight-seat blocks; every censored candidate rank is 8 and every censored baseline rank is 1.',
    }, lower, upper


def gate(comparisons, rules, censor_rates):
    pooled = comparisons['pooled']
    return bool(
        max(censor_rates) <= rules['max_censored_rate']
        and pooled['mean_delta_bounds'][1] <= rules['pooled_delta_max']
        and (not rules['require_superiority_ci'] or pooled['pessimistic_bootstrap_95ci'][1] < 0)
        and all((item['pessimistic_bootstrap_95ci'][1] if rules['profile_use_ci']
                 else item['mean_delta_bounds'][1]) < rules['profile_delta_max']
                for item in comparisons['profiles'].values())
    )


def report(root, stage):
    root = Path(root)
    protocol = read(root/'PROTOCOL.json')
    plan = protocol[stage]
    if sha(__file__) != protocol['reporter_sha256']:
        raise ValueError('reporter changed')
    for path, expected_hash in protocol['source_hashes'].items():
        if sha(path) != expected_hash:
            raise ValueError('source changed: '+path)
    for role, item in protocol['roles'].items():
        if sha(item['checkpoint']) != item['sha256']:
            raise ValueError('checkpoint changed: '+role)
    opponent = protocol['opponent']
    if sha(opponent['checkpoint']) != opponent['sha256']:
        raise ValueError('opponent changed')
    if stage == 'final':
        freeze = read(root/'FINAL_FREEZE.json')
        dev = report(root, 'dev')
        if not dev['gate_passed'] or dev != read(root/'dev-decision.json'):
            raise ValueError('development not passed or changed')
        if freeze != {'protocol_sha256': sha(root/'PROTOCOL.json'),
                      'development_sha256': sha(root/'dev-decision.json'),
                      'candidate_sha256': protocol['roles']['candidate']['sha256']}:
            raise ValueError('final freeze changed')
    pairs = schedule(stage, plan['offset'], plan['games_per_role_profile'])
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    count = len(pairs)//8
    indices = rng.integers(0, count, size=(protocol['bootstrap_replicates'], count))
    profiles, summaries, rates, lowers, uppers = {}, {}, [], [], []
    shared = None
    for profile in protocol['profiles']:
        bounds = {}
        for role, item in protocol['roles'].items():
            directory = root/f'{stage}-{profile}-{role}'
            manifest = read(directory/'manifest.json')
            expected = {'checkpoint_sha256': item['sha256'], 'opponent_checkpoint_sha256':
                        opponent['sha256'] if profile == 'mixed_learned' else None,
                        'games_requested': len(pairs), 'split': stage, 'seed_offset': plan['offset'],
                        'policy_name': 'baseline', 'opponents': profile, 'scheduling': 'shuffled_rounds',
                        'runtime': item['runtime']}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError('evaluation identity mismatch: '+str(directory))
            if manifest['environment'] != protocol['environment'] or manifest['config_sha256'] != protocol['config_sha256']:
                raise ValueError('registered environment or rules changed')
            common = {k: manifest[k] for k in ('environment', 'config_sha256', 'python', 'numpy', 'machine', 'native_build')}
            if shared is not None and shared != common:
                raise ValueError('environment mismatch')
            shared = common
            if any(manifest['source_files'].get(path) != value for path, value in protocol['source_hashes'].items()):
                raise ValueError('evaluation source mismatch')
            paths = sorted((directory/'games').glob('*.json.gz'))
            rows = []
            for path in paths:
                with gzip.open(path, 'rt') as stream:
                    row = json.load(stream)['game']
                if path.name != f"{row['seed']}-{row['seat']}.json.gz":
                    raise ValueError('artifact identity mismatch')
                rows.append(row)
            bounds[role] = indexed_bounds(rows, pairs)
            censored = sum(r['status'] == 'truncated' for r in rows)
            completed = [r for r in rows if r['status'] == 'completed']
            rates.append(censored/len(pairs))
            summaries[f'{profile}/{role}'] = {
                'attempted': len(rows), 'completed': len(completed), 'censored': censored,
                'complete_case_mean_placement': float(np.mean([r['placement'] for r in completed])) if completed else None,
                'complete_case_win_rate': sum(r['placement'] == 1 for r in completed)/len(completed) if completed else None,
                'complete_case_top4_rate': sum(r['placement'] <= 4 for r in completed)/len(completed) if completed else None,
                'illegal_actions': sum(r['illegal_actions'] for r in rows),
            }
        profile_result, lower, upper = paired_bounds(bounds['candidate'], bounds['baseline'], indices)
        profiles[profile] = profile_result
        lowers.append(lower)
        uppers.append(upper)
    lower, upper = np.mean(lowers, axis=0), np.mean(uppers, axis=0)
    comparisons = {'profiles': profiles, 'pooled': {
        'mean_delta_bounds': [float(lower.mean()), float(upper.mean())],
        'pessimistic_bootstrap_95ci': np.percentile(upper[indices].mean(1), [2.5, 97.5]).tolist()}}
    return {'stage': stage, 'protocol_sha256': sha(root/'PROTOCOL.json'), 'summaries': summaries,
            'comparisons': comparisons, 'gate_passed': gate(comparisons, plan['gate'], rates),
            'warning': 'Synthetic opponents only. Complete-case metrics are descriptive, not promotion evidence. This interval does not correct selection across earlier research campaigns.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--stage', choices=('dev', 'final'), required=True)
    args = p.parse_args()
    result = report(args.root, args.stage)
    with (args.root/f'{args.stage}-decision.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
