from omnibob.end_trade_report import gates, statistics


def test_profile_improvement_and_loss_cancel_with_shared_seed_blocks():
    # Independent resampling of profiles would create spurious uncertainty.
    a = [-1, -2, -3, -4]
    b = [1, 2, 3, 4]
    pooled = statistics([(x+y)/2 for x,y in zip(a,b)])
    assert pooled == {'mean_delta':0., 'paired_seed_bootstrap_95ci':[0.,0.]}
    assert not gates({'a':statistics(a),'b':statistics(b)}, pooled, 'final')


def test_promotion_needs_effect_size_certainty_and_each_profile_margin():
    good={'mean_delta':-.2,'paired_seed_bootstrap_95ci':[-.3,-.1]}
    assert gates({'a':good},good,'final')
    assert not gates({'a':good},{**good,'mean_delta':-.09},'final')
    assert not gates({'a':good},{**good,'paired_seed_bootstrap_95ci':[-.4,0]},'final')
    assert not gates({'a':{**good,'paired_seed_bootstrap_95ci':[-.5,.15]}},good,'final')
    assert gates({'a':good},good,'dev')
    assert not gates({'a':{**good,'mean_delta':.151}},good,'dev')


def test_actual_report_rejects_incomplete_pairs_and_changed_opponents(tmp_path):
    import json
    import pytest
    from omnibob.end_trade_report import report
    from omnibob.evaluate import schedule
    profiles=['scripted','heuristic','mixed_learned']
    protocol={'dev':{'offset':71000,'games_per_policy_per_profile':16},
              'profiles':profiles,'policy_source_hashes':{'omnibob/model.py':'frozen'}}
    (tmp_path/'PROTOCOL.json').write_text(json.dumps(protocol))
    for profile in profiles:
        for policy in ('baseline','end_trade'):
            folder=tmp_path/f'dev-{profile}-{policy}';folder.mkdir()
            manifest=dict(split='dev',opponents=profile,policy_name=policy,seed_offset=71000,
                games_requested=16,environment={},config_sha256='rules',scheduling='shuffled_rounds',
                checkpoint_sha256='v02',opponent_checkpoint_sha256='opponent',runtime={'policy':policy},
                source_files={'omnibob/'+name:'frozen' for name in ('model.py','evaluate.py','opponents.py',
                    'observation.py','observation_v2.py','move_guard.py')})
            (folder/'manifest.json').write_text(json.dumps(manifest))
            rows=[dict(seed=s,seat=seat,policy=policy,status='completed',illegal_actions=0,
                       placement=4 if policy=='baseline' else 3) for s,seat in schedule('dev',71000,16)]
            (folder/'games.jsonl').write_text('\n'.join(map(json.dumps,rows)))
            (folder/'summary.json').write_text('{}')
    assert report(tmp_path,'dev')['gate_passed']
    path=tmp_path/'dev-mixed_learned-end_trade'/'manifest.json'
    original=path.read_text();manifest=json.loads(original);manifest['opponent_checkpoint_sha256']='different'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='opponent checkpoint mismatch'):report(tmp_path,'dev')
    path.write_text(original)
    path=path.with_name('games.jsonl');path.write_text('\n'.join(path.read_text().splitlines()[:-1]))
    with pytest.raises(ValueError,match='missing, unexpected or duplicate'):report(tmp_path,'dev')
