"""Experimental visible-state sell/play chain; never enabled by deployment defaults.

The frozen network chooses both the sale and, from the next real observation,
the play. This supplies action sequencing, not a claim about replacement value.
RepeatGuard remains outside this controller and records only emitted actions.
"""
import hashlib
from pathlib import Path

from .observation_v2 import ObservationV2
from .policies import Decision


def eligible_slots(hand):
    return [i for i, h in enumerate(hand)
            if h[0] and h[1] and not h[3] and not h[6] and h[2] == 0]


class CapacityTradePolicy:
    observation_version = 2

    def __init__(self, base):
        self.base = base
        self.version = base.version + '+capacity-trade-v1'
        self.runtime_config = {**base.runtime_config, 'capacity_trade': {
            'version': 'capacity-trade-v1', 'max_chains_per_turn': 1,
            'minimum_remaining_actions': 4,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.turn, self.used, self.await_play = None, False, False
        self.last_event = None

    def restricted(self, obs, mask, event):
        decision = self.base.act(ObservationV2(obs.features, obs.cards, mask))
        self.last_event = event
        return Decision(decision.action, f'capacity trade: {event}; {decision.reason}')

    def act(self, obs):
        turn = round(float(obs.features[0]) * 20)
        self.last_event = None
        if turn != self.turn:
            self.turn, self.used, self.await_play = turn, False, False
        blocks = obs.legacy().blocks()
        board, hand = blocks['board'], blocks['hand']
        if self.await_play:
            self.await_play = False  # One next-observation attempt, never a stale retry.
            mask = obs.action_mask.copy()
            mask[:] = False
            if int(board[:, 0].sum()) == 6:
                for slot in eligible_slots(hand):
                    start = 46 + slot * 8
                    mask[start:start+8] = obs.action_mask[start:start+8]
            if mask.any():
                return self.restricted(obs, mask, 'play')
            self.last_event = 'aborted_after_sale'
        original = self.base.act(obs)
        cap = self.base.config['max_actions_per_turn']
        caps = self.base.config.get('turn_action_caps', [])
        if caps:
            cap = min(cap, caps[min(max(turn, 1)-1, len(caps)-1)])
        if (self.used or original.action not in (0, 1)
                or cap-int(obs.features[3227]) < 4
                or int(board[:, 0].sum()) != 7 or int(hand[:, 0].sum()) != 10
                or obs.action_mask[8:16].any() or obs.action_mask[46:196].any()
                or not eligible_slots(hand)):
            return original
        mask = obs.action_mask.copy()
        mask[:] = False
        mask[16:23] = obs.action_mask[16:23]
        if not mask.any():
            return original
        self.used, self.await_play = True, True
        return self.restricted(obs, mask, 'sell')


def load_capacity_policy(checkpoint):
    from .learned import load_policy
    from .repeat_guard import RepeatGuardPolicy
    frozen = load_policy(checkpoint, 'cpu', 1)
    if not isinstance(frozen, RepeatGuardPolicy) or frozen.base.runtime_config:
        raise ValueError('capacity experiment requires a repeat-only frozen policy')
    return RepeatGuardPolicy(CapacityTradePolicy(frozen.base), frozen.repeat_limit)
