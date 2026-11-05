import gzip
import json

import pytest

from omnibob import bounded_report, parallel_report


def setup_results(root, control_place):
    for name in ('baseline', 'candidate', 'control', 'opponent'):
        (root/name).write_text(name)
    rule = dict(max_censored_rate=.01, pooled_delta_max=-.1, require_superiority_ci=True,
                profile_use_ci=True, profile_delta_max=.15)
    protocol = dict(reporter_sha256=bounded_report.sha(bounded_report.__file__),
                    comparison_reporter_sha256=bounded_report.sha(parallel_report.__file__), source_hashes={},
                    profiles=['scripted'], bootstrap_seed=3, bootstrap_replicates=100,
                    environment={'test': 1}, config_sha256='rules',
                    dev=dict(offset=42, games_per_role_profile=8, gate=rule, control_gate={**rule, 'pooled_delta_max': -.05}),
                    roles={name: dict(checkpoint=str(root/name), sha256=bounded_report.sha(root/name), runtime={})
                           for name in ('baseline', 'control', 'candidate')},
                    opponent=dict(checkpoint=str(root/'opponent'), sha256=bounded_report.sha(root/'opponent')))
    (root/'PROTOCOL.json').write_text(json.dumps(protocol))
    for name, place in [('baseline', 4), ('candidate', 3), ('control', control_place)]:
        directory = root/f'dev-scripted-{name}'
        (directory/'games').mkdir(parents=True)
        manifest = dict(checkpoint_sha256=protocol['roles'][name]['sha256'], opponent_checkpoint_sha256=None,
                        games_requested=8, split='dev', seed_offset=42, policy_name='baseline', opponents='scripted',
                        scheduling='shuffled_rounds', runtime={}, environment={'test': 1}, config_sha256='rules',
                        python='test', numpy='test', machine='test', native_build={}, source_files={})
        (directory/'manifest.json').write_text(json.dumps(manifest))
        for seat in range(8):
            placements = list(range(1, 9))
            placements[seat], placements[place-1] = place, seat+1
            row = dict(seed=1000000042, seat=seat, placement=place, placements=placements, status='completed',
                       illegal_actions=0, error=None, truncated_combats=0, turn_limit_end=False)
            with gzip.open(directory/'games'/f'1000000042-{seat}.json.gz', 'wt') as stream:
                json.dump({'game': row}, stream)


@pytest.mark.parametrize('control_place,accepted', [(2, False), (3, False), (4, True)])
def test_beating_baseline_is_insufficient_without_beating_control(tmp_path, control_place, accepted):
    setup_results(tmp_path, control_place)
    result = parallel_report.report(tmp_path, 'dev')
    assert result['baseline_gate_passed']
    assert result['gate_passed'] is accepted
    assert result['control_comparison']['pooled']['mean_delta_bounds'] == [3-control_place]*2


def test_final_cannot_bypass_joint_development_rejection(tmp_path):
    setup_results(tmp_path, 2)
    with pytest.raises(ValueError, match='joint development gate'):
        parallel_report.report(tmp_path, 'final')


def test_control_missing_game_is_invalid_even_when_baseline_gate_would_pass(tmp_path):
    setup_results(tmp_path, 4)
    (tmp_path/'dev-scripted-control/games/1000000042-7.json.gz').unlink()
    with pytest.raises(ValueError, match='missing or unexpected'):
        parallel_report.report(tmp_path, 'dev')
