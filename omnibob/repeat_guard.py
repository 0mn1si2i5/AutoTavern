"""Optional observation-only repetition guard for reversible play/cancel loops.

Exact fixed observation equality is not proof of identical full game state. This
is an explicit inference heuristic: ignore only the documented own action count,
limit repeated (visible state, action) pairs, and retain the native legal mask.
"""
from collections import Counter
import hashlib
from pathlib import Path

from .move_guard import MoveGuardPolicy
from .observation_v2 import ObservationV2, extension_slices
from .policies import Decision


def runtime_config(limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= 100:
        raise ValueError("repeat limit must be an integer in 0..100")
    return ({"repeat_guard": {"version": "visible-repeat-guard-v1", "limit": limit,
                                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
            if limit else {})


class RepeatGuardPolicy:
    observation_version = 2

    def __init__(self, base, limit=1):
        if limit < 1:
            raise ValueError("repeat guard limit must be positive")
        self.base, self.repeat_limit = base, limit
        self.runtime_config = {**base.runtime_config, **runtime_config(limit)}
        self.version = f"{base.version}+visible-repeat-guard-v1-limit{limit}"
        self.action_count_index = extension_slices()[0]['g2'].start
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.turn, self.counts = None, Counter()
        if callable(getattr(self.base, 'reset', None)):
            self.base.reset()

    def act(self, observation):
        turn = round(float(observation.features[0])*20)
        if turn != self.turn:
            self.reset()
            self.turn = turn
        features = observation.features.copy()
        features[self.action_count_index] = 0
        state = hashlib.sha256(features.tobytes()+observation.cards.tobytes()
                               +observation.action_mask.tobytes()).digest()
        mask = observation.action_mask.copy()
        for action in range(len(mask)):
            if mask[action] and self.counts[(state, action)] >= self.repeat_limit:
                mask[action] = False
        blocked = int(observation.action_mask.sum()-mask.sum())
        if not mask.any():
            mask, blocked = observation.action_mask, 0
        result = self.base.act(ObservationV2(observation.features, observation.cards, mask))
        self.counts[(state, result.action)] += 1
        return Decision(result.action, f"repeat guard: blocked={blocked}; {result.reason}")


def clone_runtime(policy):
    """Share immutable inference weights while giving each seat its own controllers."""
    if isinstance(policy, RepeatGuardPolicy):
        return RepeatGuardPolicy(clone_runtime(policy.base), policy.repeat_limit)
    if isinstance(policy, MoveGuardPolicy):
        return MoveGuardPolicy(clone_runtime(policy.base), policy.move_guard_budget)
    return policy
