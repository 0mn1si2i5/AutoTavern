"""Experimental one-slot replacement before a voluntary EndTurn.

Same frozen network, with one inference-only intervention: when its EndTurn
leaves a full board and a substantially higher numeric-value minion in hand,
ask that network which qualifying minion to sell. Then normal play continues.
The numeric score is the frozen v0.1 heuristic, NOT a combat-value oracle. It can
undervalue engines and synergies; promotion requires complete-game placement.
"""
import hashlib
from pathlib import Path

from .observation_v2 import ObservationV2
from .policies import Decision, OmniBobPolicy


class EndTradePolicy:
    observation_version = 2

    def __init__(self, base):
        self.base = base  # Stateless network; outer RepeatGuard records only the returned action.
        self.version = f'{base.version}+visible-end-trade-v1'
        self.runtime_config = {**base.runtime_config, 'end_trade': {
            'version':'visible-end-trade-v1', 'minimum_gain':10, 'value_ratio':1.5,
            'max_interventions_per_turn':1, 'minimum_remaining_actions':4,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'score_source_sha256':hashlib.sha256(Path(__file__).with_name('policies.py').read_bytes()).hexdigest()}}
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.turn, self.used = None, False

    def act(self, observation):
        f = observation.features
        turn = round(float(f[0])*20)
        if turn != self.turn:
            self.turn, self.used = turn, False
        original = self.base.act(observation)
        cap = self.base.config['max_actions_per_turn']
        caps = self.base.config.get('turn_action_caps', [])
        if caps:
            cap = min(cap, caps[min(max(turn,1)-1,len(caps)-1)])
        if original.action != 0 or self.used or cap-int(f[3227]) < 4:
            return original
        blocks = observation.legacy().blocks()
        board, hand = blocks['board'], blocks['hand']
        if int(board[:,0].sum()) != 7:
            return original
        held = [OmniBobPolicy.minion_value(h[9:]) for h in hand
                if h[0] and h[1] and not h[3] and not h[6] and h[2] == 0]
        if not held:
            return original
        best = max(held)
        mask = observation.action_mask.copy()
        mask[:] = False
        for slot, minion in enumerate(board):
            value = OmniBobPolicy.minion_value(minion)
            if observation.action_mask[16+slot] and best-value >= max(10., .5*value):
                mask[16+slot] = True
        if not mask.any():
            return original
        chosen = self.base.act(ObservationV2(f, observation.cards, mask))
        self.used = True
        return Decision(chosen.action, f'end trade: free one slot, held numeric value={best:.2f}; {chosen.reason}')


def load_end_trade_policy(checkpoint, device='cpu', cpu_threads=1):
    from .learned import load_policy
    from .repeat_guard import RepeatGuardPolicy
    frozen = load_policy(checkpoint, device, cpu_threads)
    if not isinstance(frozen, RepeatGuardPolicy) or frozen.base.runtime_config:
        raise ValueError('end-trade experiment requires repeat-only frozen deployment')
    return RepeatGuardPolicy(EndTradePolicy(frozen.base), frozen.repeat_limit)
