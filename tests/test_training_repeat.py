import json

import numpy as np
import pytest
import taverngym as tg
import torch

from omnibob.compat import effective_config
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import Decision
from omnibob.repeat_guard import RepeatGuardPolicy
from omnibob.training_repeat import LearnerRepeatMask,GuardedVectorGames
from omnibob.train import TrainConfig,train,load_checkpoint
from omnibob.export import export
from omnibob.learned import load_policy


class Cycle:
    version='cycle';runtime_config={}
    def act(self,obs):
        return Decision(7 if obs.action_mask[7] else 46 if obs.action_mask[46] else 0,'cycle')


def test_native_cancel_loop_matches_deployment_and_peek_is_idempotent():
    game=tg.Game(42)
    for seat in range(8):game.step(seat,30)
    game.debug(json.dumps(dict(cmd='add_minion',p=0,zone='board',card='BG25_354')))
    game.debug(json.dumps(dict(cmd='add_spell',p=0,zone='hand',card='BG20_GEM')))
    deployment=RepeatGuardPolicy(Cycle(),1);learner=LearnerRepeatMask(1)
    for expected in (46,7,0):
        obs=ObservationV2.from_game(game,0)
        before=learner.save()
        first=learner.mask(obs);after=learner.save()
        assert np.array_equal(first,learner.mask(obs)) and learner.save()==after
        assert before['counts']==after['counts']  # Peeking never records an action.
        action=Cycle().act(ObservationV2(obs.features,obs.cards,first)).action
        assert action==deployment.act(obs).action==expected
        game.step(0,action);learner.commit(obs,action)
    saved=learner.save();restored=LearnerRepeatMask(1);restored.restore(json.loads(json.dumps(saved)))
    assert restored.save()==saved


def test_rejected_batch_does_not_advance_engine_and_saved_history_restores():
    env=GuardedVectorGames(2,94000000,effective_config(),learner_repeat_limit=1)
    try:
        obs=env.observe();before=[g.game.state_json() for g in env.games]
        with pytest.raises(ValueError,match='excluded'):env.step([-1,30])
        assert before==[g.game.state_json() for g in env.games]
        actions=[int(np.flatnonzero(m)[0]) for m in obs['action_mask']];env.step(actions)
        saved=env.save();copy=GuardedVectorGames(2,94000000,effective_config(),state=saved,learner_repeat_limit=1)
        try:
            assert copy.save()==saved
            for key,value in env.observe().items():assert np.array_equal(value,copy.observe()[key])
        finally:copy.close()
        saved['learner_repeat']['source_sha256']='bad'
        with pytest.raises(ValueError,match='incompatible'):
            GuardedVectorGames(2,94000000,effective_config(),state=saved,learner_repeat_limit=1)
    finally:env.close()


def test_cpu_guarded_ppo_resume_matches_uninterrupted_updates(tmp_path):
    cfg=TrainConfig(seed=95000000,n_envs=2,horizon=8,epochs=1,minibatch=8,cpu_threads=1,learner_repeat_limit=1)
    uninterrupted=train(tmp_path/'full',cfg,updates=2,device='cpu',max_seconds=60)
    first=train(tmp_path/'first',cfg,updates=1,device='cpu',max_seconds=60)
    resumed=train(tmp_path/'resume',resume=tmp_path/'first/latest.pt',updates=1,device='cpu',max_seconds=60)
    assert all(s['error'] is None for s in (uninterrupted,first,resumed))
    a=load_checkpoint(tmp_path/'full/latest.pt');b=load_checkpoint(tmp_path/'resume/latest.pt')
    assert a['environments']==b['environments']
    assert torch.equal(a['sampling_rng'],b['sampling_rng'])
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert a['environments']['learner_repeat']['stats']['decisions']==32
    export(tmp_path/'resume/latest.pt',tmp_path/'exported')
    assert load_policy(tmp_path/'exported/policy.pt').repeat_limit==1
