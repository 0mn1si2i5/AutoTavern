"""Experimental sale-selector ablation of the unchanged capacity trade chain.

Lowest visible numeric value is a hypothesis, not a combat-value oracle: utility
engines can have low stats. Keep this out of deployment until placement gates pass.
"""
import hashlib
from pathlib import Path

from .capacity_trade import CapacityTradePolicy
from .policies import OmniBobPolicy


class NumericSalePolicy(CapacityTradePolicy):
    def __init__(self, base):
        super().__init__(base)
        self.version += '+numeric-sale-v1'
        self.runtime_config = {**self.runtime_config, 'capacity_selector': {
            'version': 'numeric-sale-v1', 'selection': 'minimum frozen minion_value',
            'ties': 'same network legal argmax',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'score_source_sha256': hashlib.sha256(Path(__file__).with_name('policies.py').read_bytes()).hexdigest()}}

    def restricted(self, obs, mask, event):
        if event == 'sell':
            board = obs.legacy().blocks()['board']
            values = {a: OmniBobPolicy.minion_value(board[a-16])
                      for a in range(16, 23) if mask[a]}
            minimum = min(values.values())
            mask = mask.copy()
            for action, value in values.items():
                mask[action] = value == minimum
        return super().restricted(obs, mask, event)


def load_numeric_sale_policy(checkpoint):
    from .learned import load_policy
    from .repeat_guard import RepeatGuardPolicy
    frozen = load_policy(checkpoint, 'cpu', 1)
    if not isinstance(frozen, RepeatGuardPolicy) or frozen.base.runtime_config:
        raise ValueError('selector experiment requires a repeat-only frozen policy')
    return RepeatGuardPolicy(NumericSalePolicy(frozen.base), frozen.repeat_limit)
