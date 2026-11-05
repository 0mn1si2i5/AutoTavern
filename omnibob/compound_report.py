"""Placement-based decision for the preregistered two-profile cancellation experiment."""
import argparse
import json
from pathlib import Path

import numpy as np


def interval(differences):
    values = np.asarray(differences, dtype=float)
    if len(values) < 2:
        raise ValueError('at least two complete seed blocks required')
    rng = np.random.default_rng(20261029)
    samples = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {'mean_delta': float(values.mean()),
            'paired_seed_bootstrap_95ci': np.percentile(samples, [2.5, 97.5]).tolist(),
            'seed_blocks': len(values)}


def report(scripted, heuristic, stage):
    profiles, manifests = {}, []
    for name, directory in [('scripted', scripted), ('heuristic', heuristic)]:
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        manifests.append(manifest)
        if manifest['opponents'] != name or manifest['split'] != stage:
            raise ValueError('wrong opponent profile or split')
        if set(manifest['variants']) != {'baseline', 'cycle_exit', 'compound'}:
            raise ValueError('three frozen comparison variants required')
        rows = [json.loads(line) for line in (directory/'games.jsonl').read_text().splitlines()]
        expected = manifest['games_per_variant']*3
        keys = [(r['policy'], r['seed'], r['seat']) for r in rows]
        if len(rows) != expected or len(set(keys)) != expected:
            raise ValueError('missing or duplicate results; cannot promote')
        if any(r['status'] != 'completed' or r['illegal_actions'] for r in rows):
            raise ValueError('invalid or incomplete games; cannot promote')
        index = dict(zip(keys, rows))
        seeds = sorted({r['seed'] for r in rows})
        blocks = {p: {seed: float(np.mean([index[(p, seed, seat)]['placement'] for seat in range(8)]))
                       for seed in seeds} for p in manifest['variants']}
        profiles[name] = {'directory': str(directory), 'blocks': blocks,
                          'metrics': json.loads((directory/'summary.json').read_text())['metrics']}
    for key in ('environment', 'config_sha256', 'split', 'scheduling', 'checkpoint_sha256', 'runtimes'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError(f'incompatible profiles: {key}')
    for name in ('model.py', 'learned.py', 'repeat_guard.py', 'move_guard.py',
                 'reorder_exit.py', 'cancel_memory.py', 'policies.py'):
        if manifests[0]['source_files']['omnibob/'+name] != manifests[1]['source_files']['omnibob/'+name]:
            raise ValueError('policy/opponent source changed across profiles')
    seeds = sorted(profiles['scripted']['blocks']['baseline'])
    if seeds != sorted(profiles['heuristic']['blocks']['baseline']):
        raise ValueError('profiles must share complete seed blocks')
    comparisons = {}
    for control in ('baseline', 'cycle_exit'):
        deltas = {name: [p['blocks']['compound'][s]-p['blocks'][control][s] for s in seeds]
                  for name, p in profiles.items()}
        # Both profiles share the same seed. Keep them together in resampling.
        pooled = np.mean(list(deltas.values()), axis=0)
        comparisons[control] = {'profiles': {name: interval(values) for name, values in deltas.items()},
                                'equal_profile_pooled': interval(pooled)}
    dev_gate = all(x['mean_delta'] <= .15 for x in comparisons['cycle_exit']['profiles'].values())
    strength = all(c['equal_profile_pooled']['paired_seed_bootstrap_95ci'][1] < 0
                   for c in comparisons.values())
    profile_margin = all(x['paired_seed_bootstrap_95ci'][1] < .15
                         for x in comparisons['baseline']['profiles'].values())
    return {'stage': stage, 'games_completed': sum(m['games_per_variant']*3 for m in manifests),
            'checkpoint_sha256': manifests[0]['checkpoint_sha256'], 'seeds': seeds,
            'comparisons': comparisons, 'metrics': {name:p['metrics'] for name,p in profiles.items()},
            'development_gate_passed': dev_gate if stage == 'dev' else None,
            'promotion_criteria_met': strength and profile_margin if stage == 'final' else None,
            'primary_both_pooled_comparisons_improved': strength,
            'both_profiles_noninferior_to_baseline_margin_0_15': profile_margin,
            'decision': ('advance unchanged to reserved final' if dev_gate else 'stop; reserve final seeds')
                        if stage == 'dev' else ('eligible for local promotion' if strength and profile_margin
                                               else 'do not promote; keep frozen v0.2'),
            'scope': 'Conditional local placement evidence for frozen policies; no new server matches. '
                     'Moves, cancellations and operation caps are diagnostic only.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scripted', type=Path, required=True)
    parser.add_argument('--heuristic', type=Path, required=True)
    parser.add_argument('--stage', choices=['dev', 'final'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.scripted, args.heuristic, args.stage)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:result[k] for k in ('stage', 'games_completed', 'comparisons', 'decision')}, indent=2))


if __name__ == '__main__':
    main()
