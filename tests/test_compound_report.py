import json

import pytest

from omnibob.compound_report import report


def reports(tmp_path, opposite=False):
    paths = []
    for profile in ('scripted', 'heuristic'):
        root = tmp_path/profile
        root.mkdir()
        paths.append(root)
        manifest = dict(opponents=profile, split='final', variants=['baseline', 'cycle_exit', 'compound'],
                        games_per_variant=16, environment={}, config_sha256='config',
                        scheduling='shuffled_rounds', checkpoint_sha256='weights', runtimes={},
                        source_files={'omnibob/'+n:'same' for n in (
                            'model.py', 'learned.py', 'repeat_guard.py', 'move_guard.py',
                            'reorder_exit.py', 'cancel_memory.py', 'policies.py')})
        (root/'manifest.json').write_text(json.dumps(manifest))
        (root/'summary.json').write_text(json.dumps({'metrics': {}}))
        rows = []
        for seed in (2000110000, 2000110001):
            for seat in range(8):
                for policy in manifest['variants']:
                    placement = 4 if policy != 'compound' else 5 if opposite and profile == 'heuristic' else 3
                    rows.append(dict(policy=policy, seed=seed, seat=seat, placement=placement,
                                     status='completed', illegal_actions=0))
        (root/'games.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    return paths


def test_same_seed_opponent_profiles_stay_together_in_bootstrap(tmp_path):
    paths = reports(tmp_path, opposite=True)
    result = report(*paths, 'final')
    # A one-place gain and one-place loss on EVERY shared seed average to exactly zero.
    pooled = result['comparisons']['baseline']['equal_profile_pooled']
    assert pooled['mean_delta'] == 0 and pooled['paired_seed_bootstrap_95ci'] == [0, 0]
    assert not result['promotion_criteria_met']


def test_placement_criteria_require_both_controls_and_complete_evidence(tmp_path):
    paths = reports(tmp_path)
    result = report(*paths, 'final')
    assert result['promotion_criteria_met']
    assert result['comparisons']['cycle_exit']['equal_profile_pooled']['mean_delta'] == -1
    for path in paths:
        journal = path/'games.jsonl'
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        for row in rows:
            if row['policy'] == 'cycle_exit':
                row['placement'] = 3  # Candidate improves v0.2 but only ties the nearer control.
        journal.write_text('\n'.join(json.dumps(r) for r in rows))
    assert not report(*paths, 'final')['promotion_criteria_met']
    journal = paths[0]/'games.jsonl'
    journal.write_text('\n'.join(journal.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError, match='missing or duplicate'):
        report(*paths, 'final')
