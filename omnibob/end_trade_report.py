"""Apply the fixed end-trade experiment's placement gates to complete paired games."""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import schedule


def statistics(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('need at least two complete seed blocks')
    rng = np.random.default_rng(20261030)
    samples = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {'mean_delta': float(values.mean()),
            'paired_seed_bootstrap_95ci': np.percentile(samples, [2.5, 97.5]).tolist()}


def gates(profiles, pooled, stage):
    if stage == 'dev':
        return pooled['mean_delta'] <= -.05 and all(p['mean_delta'] <= .15 for p in profiles.values())
    return (pooled['mean_delta'] <= -.10 and pooled['paired_seed_bootstrap_95ci'][1] < 0
            and all(p['paired_seed_bootstrap_95ci'][1] < .15 for p in profiles.values()))


def report(root, stage):
    root = Path(root)
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    spec = protocol[stage]
    pairs = schedule(stage, spec['offset'], spec['games_per_policy_per_profile'])
    expected = set(pairs)
    seeds = sorted({s for s, _ in pairs})
    differences, metrics, manifests = {}, {}, []
    for profile in protocol['profiles']:
        blocks, metrics[profile] = {}, {}
        for policy in ('baseline', 'end_trade'):
            folder = root/f'{stage}-{profile}-{policy}'
            manifest = json.loads((folder/'manifest.json').read_text())
            if any(manifest[k] != v for k,v in dict(split=stage, opponents=profile,
                    policy_name=policy, seed_offset=spec['offset'],
                    games_requested=len(expected)).items()):
                raise ValueError('experiment contract mismatch')
            for filename, digest in protocol['policy_source_hashes'].items():
                if manifest['source_files'][filename] != digest:
                    raise ValueError('source differs from frozen protocol: '+filename)
            for prior in manifests:
                for key in ('environment','config_sha256','scheduling','checkpoint_sha256'):
                    if prior[key] != manifest[key]:
                        raise ValueError('comparison identity mismatch: '+key)
                for filename in ('evaluate.py','opponents.py','observation.py','observation_v2.py','move_guard.py'):
                    if prior['source_files']['omnibob/'+filename] != manifest['source_files']['omnibob/'+filename]:
                        raise ValueError('runner/opponent implementation mismatch')
                if prior['opponents'] == profile and prior['opponent_checkpoint_sha256'] != manifest['opponent_checkpoint_sha256']:
                    raise ValueError('opponent checkpoint mismatch')
                if prior['policy_name'] == policy and prior['runtime'] != manifest['runtime']:
                    raise ValueError('runtime mismatch across profiles')
            manifests.append(manifest)
            rows = [json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
            index = {(r['seed'],r['seat']):r for r in rows}
            if len(index) != len(rows) or set(index) != expected:
                raise ValueError('missing, unexpected or duplicate games')
            if any(r['policy'] != policy or r['status'] != 'completed' or r['illegal_actions']
                   or not 1 <= r['placement'] <= 8 for r in rows):
                raise ValueError('invalid/incomplete games; cannot advance')
            blocks[policy] = np.array([np.mean([index[(s,seat)]['placement'] for seat in range(8)]) for s in seeds])
            metrics[profile][policy] = {'mean_placement':float(blocks[policy].mean()),
                'win_rate':float(np.mean([r['placement']==1 for r in rows])),
                'top4_rate':float(np.mean([r['placement']<=4 for r in rows])),
                'completed':len(rows), 'illegal_actions':0,
                'diagnostics':json.loads((folder/'summary.json').read_text())}
        differences[profile] = blocks['end_trade']-blocks['baseline']
    profile_stats = {name:statistics(d) for name,d in differences.items()}
    pooled = statistics(np.mean(list(differences.values()), axis=0))
    passed = gates(profile_stats, pooled, stage)
    return {'stage':stage, 'games_completed':len(manifests)*len(expected),
            'seeds':seeds,'profile_statistics':profile_stats,'equal_profile_pooled':pooled,
            'gate_passed':passed,'metrics':metrics,
            'decision':('advance unchanged to final' if stage=='dev' else 'eligible for engineering and packaging checks')
                        if passed else 'reject; retain frozen v0.2',
            'scope':'Local placement only. Shared profiles and eight seats remain together in bootstrap.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--stage',choices=['dev','final'],required=True)
    args=parser.parse_args()
    result=report(args.root,args.stage)
    with (args.root/f'{args.stage}-decision.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))


if __name__=='__main__':
    main()
