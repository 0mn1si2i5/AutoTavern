from omnibob.continuation_report import eligible,interval


def test_complete_seed_profile_average_cancels_opposite_effects():
    a=[-1,-2,-3];b=[1,2,3]
    pooled=interval([(x+y)/2 for x,y in zip(a,b)])
    assert pooled['mean_delta']==0 and pooled['paired_seed_bootstrap_95ci']==[0,0]
    assert not eligible({'a':interval(a),'b':interval(b)},pooled,'final')


def test_fixed_selection_and_final_thresholds():
    strong={'mean_delta':-.2,'paired_seed_bootstrap_95ci':[-.4,-.01]}
    small={'mean_delta':-.06,'paired_seed_bootstrap_95ci':[-.15,.03]}
    assert eligible({'a':small},small,'dev')
    assert not eligible({'a':small},small,'final')
    assert eligible({'a':strong},strong,'final')
    assert not eligible({'a':{**strong,'paired_seed_bootstrap_95ci':[-.4,.15]}},strong,'final')
    assert not eligible({'a':{**strong,'mean_delta':.16}},strong,'dev')
