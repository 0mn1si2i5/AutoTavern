"""Read-only economy diagnosis of frozen policies on NEW development seeds.

Only own visible observations are inspected. Flags are hypotheses, not labels of
bad play. Atomic per-game files allow quota-safe continuation without rerunning
completed games. No training, search, credential loading or network operations.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import taverngym as tg

from .compat import provenance
from .evaluate import run_game, schedule, summarize
from .learned import load_policy
from .opponents import LEARNED_PROFILES
from .policies import OmniBobPolicy


def inspect_end(obs, catalog):
    b = obs.legacy().blocks()
    f, hand, board, mask = obs.features, b['hand'], b['board'], obs.action_mask
    nboard = int(board[:, 0].sum())
    held = []
    for i, item in enumerate(hand):
        if not item[0]:
            continue
        card = catalog[int(b['hand_cards'][i])]
        held.append({'slot': i, 'card': card['id'], 'name': card['name_zh'],
                     'text': card['text'], 'minion': bool(item[1]),
                     'locked': int(item[3]), 'temporary': bool(item[4]),
                     'cost': round(float(item[2])*5),
                     'playable': bool(mask[46+8*i:54+8*i].any()),
                     'magnetizable': bool(mask[126+7*i:133+7*i].any()),
                     'visible_value': OmniBobPolicy.minion_value(item[9:]) if item[1] else None})
    ready = [h for h in held if not h['locked']]
    board_values = [OmniBobPolicy.minion_value(m) for m in board if m[0]]
    flags = {
        'gold_at_least_3': float(f[7])*10 >= 2.99,
        'playable_minion_with_space': nboard < 7 and any(h['minion'] and h['playable'] for h in ready),
        'free_playable_spell': any(not h['minion'] and h['cost'] == 0 and h['playable'] for h in ready),
        'playable_magnetic': any(h['magnetizable'] for h in ready),
        'full_board_held_battlecry': nboard == 7 and any(h['minion'] and 'battlecry' in h['text'].lower() for h in ready),
        'full_board_stronger_numeric_hand': nboard == 7 and any(h['minion'] and h['visible_value'] > min(board_values)+2 for h in ready),
        'temporary_playable_hand': any(h['temporary'] and h['playable'] for h in ready),
    }
    return {'turn': round(float(f[0])*20), 'gold': round(float(f[7])*10),
            'tier': round(float(f[10])*6), 'board_size': nboard, 'actions_used': int(f[3227]),
            'can_upgrade': bool(mask[3]), 'upgrade_cost': round(float(f[11])*10),
            'flags': flags, 'hand': held,
            'board': [{'slot': i, 'card': catalog[int(b['board_cards'][i])]['id'],
                       'name': catalog[int(b['board_cards'][i])]['name_zh'],
                       'attack': round(float(m[1])*10), 'health': round(float(m[2])*10),
                       'visible_value': OmniBobPolicy.minion_value(m)} for i,m in enumerate(board) if m[0]]}


class EconomyObserver:
    observation_version = 2

    def __init__(self, policy, catalog):
        self.base, self.catalog = policy, catalog

    def reset(self):
        self.base.reset()
        self.ends, self.trades = [], []

    def act(self, obs):
        decision = self.base.act(obs)
        if self.trades and 'next_action' not in self.trades[-1]:
            self.trades[-1]['next_action'] = decision.action
        if 'end trade: free one slot' in decision.reason:
            self.trades.append({**inspect_end(obs, self.catalog), 'action':decision.action,
                                'reason':decision.reason})
        if decision.action == 0:
            self.ends.append({**inspect_end(obs, self.catalog), 'reason': decision.reason,
                              'observation': {k:getattr(obs,k).tolist() for k in ('features','cards','action_mask')}})
        return decision


def audit(output, checkpoint, games=128, seed=61000, opponents='scripted',
          opponent_checkpoint=None, max_seconds=600, resume=False, stop_file=None,
          policy_name='baseline', split='dev'):
    output, checkpoint = Path(output), Path(checkpoint)
    if max_seconds <= 0:
        raise ValueError('positive time budget required')
    pairs = schedule(split, seed, games)
    if (opponents in LEARNED_PROFILES) != (opponent_checkpoint is not None):
        raise ValueError('learned opponents require an explicit frozen checkpoint')
    if policy_name == 'end_trade':
        from .end_trade import load_end_trade_policy
        policy = load_end_trade_policy(checkpoint)
    elif policy_name == 'baseline':
        policy = load_policy(checkpoint)
    else:
        raise ValueError('unknown economy policy')
    opponent = load_policy(opponent_checkpoint) if opponent_checkpoint else None
    if opponent and opponent.config != policy.config:
        raise ValueError('opponent rules mismatch')
    identity = provenance(policy.config)
    contract = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'opponent_checkpoint_sha256': hashlib.sha256(Path(opponent_checkpoint).read_bytes()).hexdigest() if opponent_checkpoint else None,
                'games_requested': games, 'seed_offset': seed, 'split': split, 'policy_name':policy_name,
                'opponents': opponents, 'scheduling': 'shuffled_rounds', 'runtime':policy.runtime_config}
    if resume:
        old = json.loads((output/'manifest.json').read_text())
        for key in (*contract, 'environment', 'config_sha256'):
            if old[key] != {**identity, **contract}[key]:
                raise ValueError(f'resume mismatch: {key}')
        used_files = ['economy_audit.py','evaluate.py','learned.py','model.py','repeat_guard.py',
                      'move_guard.py','observation.py','observation_v2.py','opponents.py','policies.py','compat.py']
        if policy_name == 'end_trade':
            used_files.append('end_trade.py')
        if any(old['source_files']['omnibob/'+n] != identity['source_files']['omnibob/'+n] for n in used_files):
            raise ValueError('restore recorded implementation before resuming this audit')
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output/'games').mkdir()
        (output/'manifest.json').write_text(json.dumps({**identity, **contract,
            'scope':'own visible end-of-turn flags; not error labels or training data',
            'primary_records':'atomic games/*.json.gz; summary and games.jsonl are derived',
            'max_seconds_initial':max_seconds}, indent=2))
    catalog = {c['cid']+1:c for c in json.loads(tg.catalog_json())}
    observer = EconomyObserver(policy, catalog)
    started, newly_run = time.monotonic(), 0
    with (output/'invocations.jsonl').open('a') as stream:
        stream.write(json.dumps({'kind':'started','pid':os.getpid(),'time_unix':time.time(),'resume':resume})+'\n')
    for game_seed, seat in pairs:
        target = output/'games'/f'{game_seed}-{seat}.json.gz'
        if target.exists():
            continue
        if time.monotonic()-started >= max_seconds or stop_file and Path(stop_file).exists():
            break
        row, latencies = run_game(policy_name, game_seed, seat, policy.config, policy_override=observer,
                          opponents=opponents, opponent_policy=opponent, scheduler='shuffled_rounds', timeout=45)
        with tempfile.NamedTemporaryFile(dir=output/'games', suffix='.partial', delete=False) as temp:
            temporary = Path(temp.name)
        with gzip.open(temporary, 'wt') as stream:
            json.dump({'game':row,'ends':observer.ends,'trades':observer.trades,'latencies_ms':latencies}, stream)
        os.replace(temporary, target)
        newly_run += 1
        if newly_run % 8 == 0:
            print(f'{newly_run} new games, {time.monotonic()-started:.1f}s', flush=True)
    records = []
    for game_seed, seat in pairs:
        target = output/'games'/f'{game_seed}-{seat}.json.gz'
        if target.exists():
            with gzip.open(target, 'rt') as stream:
                record = json.load(stream)
            if (record['game']['seed'],record['game']['seat']) != (game_seed,seat):
                raise ValueError('atomic record identity mismatch')
            records.append(record)
    rows = [r['game'] for r in records]
    ends = [e for r in records for e in r['ends']]
    flags = Counter(k for e in ends for k,v in e['flags'].items() if v)
    cards = Counter(h['card'] for e in ends for h in e['hand'] if not h['locked'])
    summary = {'games':summarize(rows,[policy_name],games,{policy_name:[v for r in records for v in r['latencies_ms']]})[policy_name],
               'newly_run':newly_run,'invocation_seconds':time.monotonic()-started,
               'end_observations':len(ends),'flags':dict(flags),'unlocked_held_cards':dict(cards.most_common()),
               'interventions':sum(len(r['trades']) for r in records),
               'flags_after_turn_7':dict(Counter(k for e in ends if e['turn']>=8 for k,v in e['flags'].items() if v)),
               'warning':'Flags suggest hypotheses; they are not proofs of erroneous decisions. Hand counts are observed slots, not unique physical cards.'}
    (output/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output/'invocations.jsonl').open('a') as stream:
        stream.write(json.dumps({'kind':'finished','pid':os.getpid(),'time_unix':time.time(),'newly_run':newly_run,
                                'completed':summary['games']['completed']})+'\n')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--games',type=int,default=128)
    p.add_argument('--seed',type=int,default=61000)
    p.add_argument('--split',choices=['dev','final'],default='dev')
    p.add_argument('--policy-name',choices=['baseline','end_trade'],default='baseline')
    p.add_argument('--opponents',choices=['scripted','heuristic','mixed_learned'],default='scripted')
    p.add_argument('--opponent-checkpoint',type=Path)
    p.add_argument('--max-seconds',type=float,default=600)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-file',type=Path)
    args=p.parse_args()
    result=audit(**vars(args))
    print(json.dumps(result,indent=2),flush=True)
    m=result['games']
    if m['completed']!=m['requested'] or m['illegal_actions']:
        raise SystemExit('Incomplete/failed audit retained; use --resume for unattempted games only.')


if __name__=='__main__':
    main()
