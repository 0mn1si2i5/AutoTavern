import io
import json
from pathlib import Path

import pytest

from omnibob.evaluate import run_game
from omnibob.guard_audit import GuardObserver
from omnibob.learned import load_policy


def test_instrumentation_preserves_complete_game_actions_and_placement():
    checkpoint=Path('runs/releases/omnibob-v0.2-20261028/policy/policy.pt')
    if not checkpoint.exists():pytest.skip('local frozen deployment required for behavioral audit')
    stock=load_policy(checkpoint);observed=GuardObserver(load_policy(checkpoint))
    traces=[];results=[]
    for policy in (stock,observed):
        trace=io.StringIO()
        result,_=run_game('audit-equivalence',93000000,0,policy.config,policy_override=policy,
                          opponents='scripted',scheduler='shuffled_rounds',trace=trace,timeout=30)
        assert result['status']=='completed' and result['illegal_actions']==0
        results.append(result)
        traces.append([(r['seat'],r['turn'],r['action'],r['visible_sha256'])
                       for r in map(json.loads,trace.getvalue().splitlines())])
    assert traces[0]==traces[1] and results[0]['placements']==results[1]['placements']
    assert len(observed.records)==results[1]['decisions']
    assert all(0<=r['removed_probability']<=1.000001 for r in observed.records)
    assert all(r['actual_action'] not in r['blocked_actions'] for r in observed.records)
