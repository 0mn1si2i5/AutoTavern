"""Frozen three-role placement test: env32 must beat v0.2 AND matched env8.

Keeps the existing bounded reporter and its historical evidence unchanged.
"""
import gzip
import json
from pathlib import Path

import numpy as np

from . import bounded_report
from .bounded_report import gate, indexed_bounds, paired_bounds, read, sha
from .evaluate import schedule


def report(root, stage):
    root = Path(root)
    protocol = read(root/'PROTOCOL.json')
    if protocol['comparison_reporter_sha256'] != sha(__file__):
        raise ValueError('comparison reporter changed')
    if set(protocol['roles']) != {'baseline', 'control', 'candidate'}:
        raise ValueError('requires frozen baseline, env8 control and env32 candidate')
    if stage == 'final':
        dev = report(root, 'dev')
        if not dev['gate_passed'] or dev != read(root/'dev-comparison.json'):
            raise ValueError('joint development gate not passed or changed')
        expected = {'protocol_sha256': sha(root/'PROTOCOL.json'),
                    'joint_development_sha256': sha(root/'dev-comparison.json'),
                    'control_sha256': protocol['roles']['control']['sha256'],
                    'candidate_sha256': protocol['roles']['candidate']['sha256']}
        if read(root/'COMPARISON_FINAL_FREEZE.json') != expected:
            raise ValueError('joint final freeze changed')
    primary = bounded_report.report(root, stage)
    plan = protocol[stage]
    pairs = schedule(stage, plan['offset'], plan['games_per_role_profile'])
    count = len(pairs)//8
    indices = np.random.default_rng(protocol['bootstrap_seed']).integers(
        0, count, size=(protocol['bootstrap_replicates'], count))
    profiles, lowers, uppers, rates = {}, [], [], []
    for profile in protocol['profiles']:
        bounds = {}
        for role in ('candidate', 'control'):
            rows = []
            for path in sorted((root/f'{stage}-{profile}-{role}'/'games').glob('*.json.gz')):
                with gzip.open(path, 'rt') as stream:
                    rows.append(json.load(stream)['game'])
            bounds[role] = indexed_bounds(rows, pairs)
            rates.append(sum(r['status'] == 'truncated' for r in rows)/len(pairs))
        profiles[profile], lower, upper = paired_bounds(bounds['candidate'], bounds['control'], indices)
        lowers.append(lower)
        uppers.append(upper)
    lower, upper = np.mean(lowers, axis=0), np.mean(uppers, axis=0)
    comparison = {'profiles': profiles, 'pooled': {
        'mean_delta_bounds': [float(lower.mean()), float(upper.mean())],
        'pessimistic_bootstrap_95ci': np.percentile(upper[indices].mean(1), [2.5, 97.5]).tolist()}}
    control_passed = gate(comparison, plan['control_gate'], rates)
    return {'stage': stage, 'protocol_sha256': sha(root/'PROTOCOL.json'),
            'baseline_comparison': primary, 'control_comparison': comparison,
            'baseline_gate_passed': primary['gate_passed'], 'control_gate_passed': control_passed,
            'gate_passed': bool(primary['gate_passed'] and control_passed),
            'warning': 'Local synthetic opponents only. Both placement gates must pass; no selection of the better arm.'}
