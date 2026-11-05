import numpy as np
import pytest
import gzip
import json

from omnibob import bounded_report
from omnibob.bounded_report import gate, indexed_bounds, paired_bounds, placement_bounds


def row(seat=0):
    return dict(seed=12, seat=seat, placement=seat+1, placements=list(range(1, 9)),
                status='completed', illegal_actions=0, error=None, truncated_combats=0, turn_limit_end=False)


def test_forced_first_place_never_counts_as_a_win():
    game = row()
    game.update(status='truncated', turn_limit_end=True)
    assert placement_bounds(game) == (1, 8)
    c = np.ones((8, 2))*2
    b = np.ones((8, 2))*3
    b[0] = placement_bounds(game)
    result, lo, hi = paired_bounds(c, b, np.zeros((100, 1), dtype=int))
    assert result['mean_delta_bounds'] == [-1.625, -0.75]
    assert result['pessimistic_bootstrap_95ci'] == [-0.75, -0.75]


def test_both_censored_are_not_treated_as_equal():
    b = np.tile([1, 8], (8, 1))
    result, _, _ = paired_bounds(b, b, np.zeros((100, 1), dtype=int))
    assert result['mean_delta_bounds'] == [-7, 7]


@pytest.mark.parametrize('change', [dict(status='interrupted'), dict(error='panic'),
    dict(illegal_actions=1), dict(truncated_combats=1), dict(placement=0), dict(placement=2),
    dict(status='completed', turn_limit_end=True)])
def test_invalid_outcomes_do_not_become_censors(change):
    game = row(); game.update(change)
    with pytest.raises(ValueError): placement_bounds(game)


def test_missing_and_duplicate_seats_rejected():
    rows = [row(i) for i in range(8)]; expected = [(12, i) for i in range(8)]
    assert indexed_bounds(rows, expected).shape == (8, 2)
    for invalid in [rows[:-1], rows+[rows[0]], rows[:-1]+[rows[0]]]:
        with pytest.raises(ValueError): indexed_bounds(invalid, expected)


def test_gate_uses_worst_case_and_profile_regression():
    rules = dict(max_censored_rate=.01, pooled_delta_max=-.1, require_superiority_ci=True,
                 profile_use_ci=True, profile_delta_max=.15)
    result = {'pooled': {'mean_delta_bounds': [-.3, -.2], 'pessimistic_bootstrap_95ci': [-.4, -.01]},
              'profiles': {'a': {'mean_delta_bounds': [-.3, -.2], 'pessimistic_bootstrap_95ci': [-.4, .1]}}}
    assert gate(result, rules, [0, .001])
    assert not gate(result, rules, [.02])
    result['pooled']['mean_delta_bounds'][1] = -.09
    assert not gate(result, rules, [0])
    result['pooled']['mean_delta_bounds'][1] = -.2
    result['profiles']['a']['pessimistic_bootstrap_95ci'][1] = .16
    assert not gate(result, rules, [0])


def test_report_reads_atomic_games_and_rejects_changed_identity(tmp_path):
    # Deliberately false summary files must not override the primary game evidence.
    for name in ('baseline', 'candidate', 'opponent'):
        (tmp_path/name).write_text(name)
    rules = dict(max_censored_rate=.01, pooled_delta_max=-.1, require_superiority_ci=True,
                 profile_use_ci=True, profile_delta_max=.15)
    protocol = dict(reporter_sha256=bounded_report.sha(bounded_report.__file__), source_hashes={},
                    profiles=['scripted'], bootstrap_seed=123, bootstrap_replicates=100,
                    environment={'test': 1}, config_sha256='rules',
                    dev=dict(offset=42, games_per_role_profile=8, gate=rules),
                    roles={name:dict(checkpoint=str(tmp_path/name), sha256=bounded_report.sha(tmp_path/name),
                                    runtime={}) for name in ('baseline', 'candidate')},
                    opponent=dict(checkpoint=str(tmp_path/'opponent'), sha256=bounded_report.sha(tmp_path/'opponent')))
    (tmp_path/'PROTOCOL.json').write_text(json.dumps(protocol))
    for name, place in [('baseline', 3), ('candidate', 2)]:
        directory = tmp_path/f'dev-scripted-{name}'; (directory/'games').mkdir(parents=True)
        manifest = dict(checkpoint_sha256=protocol['roles'][name]['sha256'], opponent_checkpoint_sha256=None,
                        games_requested=8, split='dev', seed_offset=42, policy_name='baseline',
                        opponents='scripted', scheduling='shuffled_rounds', runtime={},
                        environment={'test': 1}, config_sha256='rules', python='test', numpy='test',
                        machine='test', native_build={}, source_files={})
        (directory/'manifest.json').write_text(json.dumps(manifest))
        (directory/'summary.json').write_text('{"gate_passed": false}')
        for seat in range(8):
            game = row(seat); game.update(seed=1000000042, placement=place)
            game['placements'][seat], game['placements'][place-1] = place, seat+1
            with gzip.open(directory/'games'/f'1000000042-{seat}.json.gz', 'wt') as stream:
                json.dump({'game':game}, stream)
    actual = bounded_report.report(tmp_path, 'dev')
    assert actual['gate_passed']
    assert actual['comparisons']['pooled']['mean_delta_bounds'] == [-1, -1]
    directory = tmp_path/'dev-scripted-candidate'
    manifest = json.loads((directory/'manifest.json').read_text()); manifest['seed_offset'] = 43
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity mismatch'):
        bounded_report.report(tmp_path, 'dev')
