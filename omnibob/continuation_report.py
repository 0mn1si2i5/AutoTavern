"""Fixed-update three-seed PPO selection and independent placement acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate import schedule


def interval(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2:raise ValueError('at least two seed blocks required')
    samples=np.random.default_rng(20261031).choice(a,(10000,len(a)),replace=True).mean(axis=1)
    return {'mean_delta':float(a.mean()),'paired_seed_bootstrap_95ci':np.percentile(samples,[2.5,97.5]).tolist()}


def eligible(profiles,pooled,stage):
    if stage=='dev':
        return pooled['mean_delta']<=-.05 and all(p['mean_delta']<=.15 for p in profiles.values())
    return pooled['mean_delta']<=-.10 and pooled['paired_seed_bootstrap_95ci'][1]<0 and all(
        p['paired_seed_bootstrap_95ci'][1]<.15 for p in profiles.values())


def report(root,stage):
    root=Path(root);protocol=json.loads((root/'PROTOCOL.json').read_text())
    roles=['v02',*map(str,protocol['training_seeds'])]
    frozen=None
    if stage=='final':
        frozen=json.loads((root/'CANDIDATE_FREEZE.json').read_text())
        if frozen['training_seed'] not in roles[1:]:raise ValueError('unknown frozen candidate')
        roles=['v02',frozen['training_seed']]
    expected=set(schedule(stage,protocol[stage]['offset'],protocol[stage]['games_per_policy_per_profile']))
    seeds=sorted({s for s,_ in expected});profiles=protocol['dev']['profiles']
    manifests=[];means={};metrics={}
    for role in roles:
        if role=='v02':digest=protocol['initial_sha256']
        else:
            deployment=json.loads((root/f'policy{role}'/'manifest.json').read_text())
            digest=deployment['policy_sha256']
            if hashlib.sha256((root/f'policy{role}'/'policy.pt').read_bytes()).hexdigest()!=digest:
                raise ValueError('deployment checkpoint modified')
            if deployment['update']!=50 or deployment['steps']!=204800:
                raise ValueError('candidate is not the fixed 50-update checkpoint')
            config=deployment['ppo_config']
            if config['seed']!=int(role) or any(config[k]!=v for k,v in protocol['config_except_seed'].items()):
                raise ValueError('training configuration mismatch')
            if deployment['reference_checkpoint']['sha256']!=protocol['initial_sha256']:
                raise ValueError('reference initialization mismatch')
            if frozen and frozen['policy_sha256']!=digest:raise ValueError('selected candidate changed')
        means[role]={};metrics[role]={}
        for profile in profiles:
            folder=root/f'{stage}-{profile}-{role}'
            m=json.loads((folder/'manifest.json').read_text())
            if (m['split'],m['opponents'],m['seed_offset'],m['games_requested'],m['checkpoint_sha256'])!=(
                    stage,profile,protocol[stage]['offset'],len(expected),digest):
                raise ValueError('evaluation contract mismatch')
            for file,h in protocol['source_hashes'].items():
                if m['source_files'][file]!=h:raise ValueError('frozen source changed: '+file)
            for prior in manifests:
                for key in ('environment','config_sha256','scheduling','runtime'):
                    if prior[key]!=m[key]:raise ValueError('different environment/runtime: '+key)
                if prior['opponents']==profile and prior['opponent_checkpoint_sha256']!=m['opponent_checkpoint_sha256']:
                    raise ValueError('opponent checkpoint mismatch')
            manifests.append(m)
            rows=[json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
            index={(r['seed'],r['seat']):r for r in rows}
            if len(index)!=len(rows) or set(index)!=expected:raise ValueError('missing or duplicate games')
            if any(r['status']!='completed' or r['illegal_actions'] or not 1<=r['placement']<=8 for r in rows):
                raise ValueError('failed, illegal or incomplete games')
            means[role][profile]=np.array([np.mean([index[(s,seat)]['placement'] for seat in range(8)]) for s in seeds])
            metrics[role][profile]={'mean_placement':float(means[role][profile].mean()),
                'win_rate':float(np.mean([r['placement']==1 for r in rows])),
                'top4_rate':float(np.mean([r['placement']<=4 for r in rows])),
                'completed':len(rows),'illegal_actions':0,
                'elapsed_seconds_sum':sum(r['elapsed_seconds'] for r in rows),
                'decision_count':sum(r['decisions'] for r in rows),
                'decision_mean_ms':sum(r['decision_ms_total'] for r in rows)/sum(r['decisions'] for r in rows)}
    comparisons={}
    for role in roles[1:]:
        deltas={profile:means[role][profile]-means['v02'][profile] for profile in profiles}
        per={k:interval(v) for k,v in deltas.items()}
        pooled=interval(np.mean(list(deltas.values()),axis=0))
        comparisons[role]={'profiles':per,'pooled':pooled,'gate_passed':eligible(per,pooled,stage)}
    passing=[r for r in roles[1:] if comparisons[r]['gate_passed']]
    selected=min(passing,key=lambda r:(comparisons[r]['pooled']['mean_delta'],int(r))) if passing else None
    return {'stage':stage,'completed_games':len(manifests)*len(expected),'seed_blocks':len(seeds),'seeds':seeds,
            'metrics':metrics,'comparisons':comparisons,'selected_training_seed':selected,
            'decision':('freeze selected candidate for independent final' if stage=='dev' else 'eligible for engineering/package checks')
                        if selected else 'no promotion; keep frozen v0.2',
            'limitations':'Three seeds share one parent. Development intervals do not correct for model selection. Local synthetic opponents only.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--stage',choices=['dev','final'],required=True);a=p.parse_args()
    result=report(a.root,a.stage)
    with (a.root/f'{a.stage}-decision.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('metrics','seeds')},indent=2))


if __name__=='__main__':main()
