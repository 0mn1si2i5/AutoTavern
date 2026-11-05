"""Optional learner masks using the unchanged frozen deployment RepeatGuard.

Peek stops before the guard records an action. Commit records only successfully
stepped actions. PPO samples, stores log probabilities and updates using these
same masks; there is no post-sampling action replacement.
"""
import hashlib
from pathlib import Path

import numpy as np

from .observation_v2 import ObservationV2
from .policies import Decision
from .repeat_guard import RepeatGuardPolicy, runtime_config
from .runtime_state import save_runtime_state, restore_runtime_state
from .training_env import VectorGames


class _MaskReady(Exception):
    pass


class _ActionBridge:
    version='training-repeat-bridge-v1'
    runtime_config={}

    def __init__(self):self.reset()
    def reset(self):self.action=None;self.mask=None

    def act(self,obs):
        self.mask=obs.action_mask.copy()
        if self.action is None:raise _MaskReady
        if not 0<=self.action<len(self.mask) or not self.mask[self.action]:raise ValueError('action excluded by learner repeat mask')
        return Decision(self.action,'record successfully stepped learner action')


class LearnerRepeatMask:
    def __init__(self,limit):
        self.bridge=_ActionBridge()
        self.guard=RepeatGuardPolicy(self.bridge,limit)

    def mask(self,obs):
        self.bridge.action=None
        try:self.guard.act(obs)
        except _MaskReady:return self.bridge.mask.copy()
        raise RuntimeError('learner mask peek unexpectedly chose an action')

    def commit(self,obs,action):
        self.mask(obs)  # Establish the current turn before the guard's reset path.
        self.bridge.action=int(action)
        try:self.guard.act(obs)
        finally:self.bridge.action=None

    def reset(self):self.guard.reset()
    def save(self):return save_runtime_state(self.guard)
    def restore(self,state):restore_runtime_state(self.guard,state)


class GuardedVectorGames(VectorGames):
    def __init__(self,*args,learner_repeat_limit=1,**kwargs):
        runtime_config(learner_repeat_limit)
        if learner_repeat_limit<1:raise ValueError('guarded training requires a positive repeat limit')
        state=kwargs.get('state')
        self.learner_repeat_limit=learner_repeat_limit
        self.learner_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        super().__init__(*args,**kwargs)
        self.learner_guards=[LearnerRepeatMask(learner_repeat_limit) for _ in self.games]
        self.learner_mask_stats={'decisions':0,'filtered_decisions':0,'blocked_actions':0}
        if state is not None:
            saved=state.get('learner_repeat')
            if not saved or saved['limit']!=learner_repeat_limit or saved['source_sha256']!=self.learner_source_sha256:
                self.close();raise ValueError('saved learner controller missing or incompatible')
            if len(saved['guards'])!=self.n_games:
                self.close();raise ValueError('saved learner controller count mismatch')
            for guard,item in zip(self.learner_guards,saved['guards']):guard.restore(item)
            self.learner_mask_stats=dict(saved['stats'])

    @staticmethod
    def _observations(batch):
        return [ObservationV2(**{name:values[i] for name,values in batch.items()}) for i in range(len(batch['features']))]

    def observe(self):
        batch=super().observe()
        batch['action_mask']=np.stack([guard.mask(obs) for guard,obs in zip(self.learner_guards,self._observations(batch))])
        return batch

    def step(self,actions):
        if len(actions)!=self.n_games:raise ValueError('training action batch has wrong length')
        native=self._observations(super().observe())
        masks=[guard.mask(obs) for guard,obs in zip(self.learner_guards,native)]
        for action,mask in zip(actions,masks):
            if not isinstance(action,(int,np.integer)) or isinstance(action,(bool,np.bool_)) or not 0<=action<len(mask) or not mask[action]:
                raise ValueError('action excluded by learner repeat mask')
        rewards,dones,records=super().step(actions)
        for guard,obs,mask,action,done in zip(self.learner_guards,native,masks,actions,dones):
            blocked=int(np.count_nonzero(obs.action_mask & ~mask))
            self.learner_mask_stats['decisions']+=1
            self.learner_mask_stats['filtered_decisions']+=bool(blocked)
            self.learner_mask_stats['blocked_actions']+=blocked
            guard.commit(obs,action)
            if done:guard.reset()
        return rewards,dones,records

    def save(self):
        state=super().save()
        state['learner_repeat']={'limit':self.learner_repeat_limit,'source_sha256':self.learner_source_sha256,
            'guards':[guard.save() for guard in self.learner_guards],'stats':dict(self.learner_mask_stats)}
        return state
