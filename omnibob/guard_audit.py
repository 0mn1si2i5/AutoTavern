"""Observation-only diagnosis of native-mask versus deployed repetition filtering.

The actual action is always returned by the unmodified frozen RepeatGuard.
Unfiltered logits are read on the same own-visible input, without game/RNG access.
This measures mismatch along deployed trajectories, not strength or its cause.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .compat import provenance
from .evaluate import run_game, schedule
from .learned import LearnedPolicy, load_policy
from .repeat_guard import RepeatGuardPolicy


def category(action):
    if action>=196:return 'move'
    if action>=126:return 'magnet'
    if action>=46:return 'play'
    if action>=30:return 'choose'
    if action>=23:return 'activate'
    if action>=16:return 'sell'
    if action>=8:return 'buy'
    return ('end','refresh','freeze','upgrade','hero_power','hero_power','gift','cancel')[action]


class CaptureMask:
    def __init__(self,base):self.base=base
    def __getattr__(self,name):return getattr(self.base,name)
    def act(self,observation):
        self.mask=observation.action_mask.copy()
        return self.base.act(observation)


class GuardObserver:
    observation_version=2

    def __init__(self,policy):
        if not isinstance(policy,RepeatGuardPolicy) or not isinstance(policy.base,LearnedPolicy):
            raise ValueError('requires repeat-only learned policy')
        self.guard,self.raw=policy,policy.base
        self.capture=CaptureMask(self.raw)
        self.guard.base=self.capture
        self.reset()

    def __getattr__(self,name):return getattr(self.guard,name)

    def reset(self):
        self.guard.reset();self.records=[]

    def act(self,observation):
        raw=self.raw
        with torch.inference_mode():
            logits,_=raw.model(torch.tensor(observation.features[None],device=raw.device),
                torch.tensor(observation.cards[None],dtype=torch.long,device=raw.device),
                torch.tensor(observation.action_mask[None],dtype=torch.bool,device=raw.device))
            probabilities=logits.softmax(-1)[0].cpu().numpy()
            preferred=int(logits.argmax(-1).item())
        decision=self.guard.act(observation)
        blocked=observation.action_mask & ~self.capture.mask
        self.records.append({'turn':round(float(observation.features[0])*20),
            'raw_action':preferred,'actual_action':decision.action,
            'blocked_actions':np.flatnonzero(blocked).tolist(),
            'removed_probability':float(probabilities[blocked].sum()),
            'visible_sha256':hashlib.sha256(observation.features.tobytes()+observation.cards.tobytes()+observation.action_mask.tobytes()).hexdigest()})
        return decision


def audit(checkpoint,output,seed=93000000,games=16,max_seconds=180):
    output=Path(output);pairs=schedule('train',seed,games)
    output.mkdir(parents=True,exist_ok=False)
    policy=GuardObserver(load_policy(checkpoint));start=time.monotonic()
    manifest={**provenance(policy.config),'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'split':'train','seed_offset':seed,'games_per_profile':games,'profiles':['scripted','heuristic','mixed'],
        'scope':'same visible state native-mask probabilities versus frozen deployed filtering; no training or strength claim',
        'max_seconds':max_seconds,'diagnostic_latency_not_a_benchmark':True}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result={}
    for profile in manifest['profiles']:
        rows=[];records=[]
        for game_seed,seat in pairs:
            if time.monotonic()-start>max_seconds:raise TimeoutError('audit budget reached; existing games retained')
            row,_=run_game('v02-instrumented',game_seed,seat,policy.config,policy_override=policy,
                           opponents=profile,scheduler='shuffled_rounds',timeout=30)
            record={'game':row,'decisions':policy.records}
            with (output/f'{profile}-{game_seed}-{seat}.json').open('x') as stream:json.dump(record,stream)
            rows.append(row);records.extend(policy.records)
            if row['status']!='completed' or row['illegal_actions']:raise RuntimeError('non-completed game retained; do not summarize as complete')
        changed=[r for r in records if r['raw_action']!=r['actual_action']]
        result[profile]={'completed':len(rows),'decisions':len(records),
            'filtered_decisions':sum(bool(r['blocked_actions']) for r in records),
            'changed_argmax':len(changed),'changed_argmax_rate':len(changed)/len(records),
            'removed_probability_mean':float(np.mean([r['removed_probability'] for r in records])),
            'changed_raw_categories':dict(Counter(category(r['raw_action']) for r in changed)),
            'actual_categories':dict(Counter(category(r['actual_action']) for r in records)),
            'mean_placement_diagnostic_only':float(np.mean([r['placement'] for r in rows]))}
        print(profile,json.dumps(result[profile]),flush=True)
    summary={'profiles':result,'seconds':time.monotonic()-start,'limitations':[
        'Only two environment seed blocks per profile by default; exploratory diagnosis, not promotion.',
        'States come from deployed greedy trajectories, not stochastic PPO rollouts.',
        'Native-mask probability removed describes a distribution difference, not measured ranking loss.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=93000000)
    p.add_argument('--games',type=int,default=16);p.add_argument('--max-seconds',type=float,default=180)
    audit(**vars(p.parse_args()))


if __name__=='__main__':main()
