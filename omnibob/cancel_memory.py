"""Experimental memory of confirmed, immediately cancelled hand plays.

Remember only a visible roundtrip: Play -> Cancel -> the same observation, modulo
own action count and board order. While that canonical observation is unchanged,
exclude another Play of the same hand slot (including its position aliases).
Cancel and target choices remain legal. Every other visible change can reopen the
play, including an opponent update; this deliberately conservative rule is not a
proof of strategic equivalence. Reordering can change adjacency-spell value.

Wrap the full existing controller, and invoke it exactly once per actual decision.
No simulator, saved random state, hidden information, or weight modification.
"""
import hashlib
from pathlib import Path

from .observation_v2 import ObservationV2, extension_slices
from .policies import Decision
from .reorder_exit import board_layout, load_reorder_policy


def cancelled_play_context(observation):
    fs, cs = extension_slices()
    features, cards = observation.features.copy(), observation.cards.copy()
    units = board_layout(observation)
    features[3227] = 0  # Own actions_this_turn, the sole ignored scalar.
    features[68:285] = 0
    features[fs['board_x']] = 0
    features[fs['board_att_n']] = 0
    cards[19:26] = 0
    cards[cs['board_att']] = 0
    return hashlib.sha256(features.tobytes()+cards.tobytes()+b''.join(sorted(units))).digest()


class CancelMemoryPolicy:
    observation_version = 2

    def __init__(self, base):
        self.base = base
        self.version = f'{base.version}+visible-cancel-memory-v1'
        self.runtime_config = {**base.runtime_config, 'cancel_memory': {
            'version': 'visible-cancel-memory-v1',
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.turn, self.last_play, self.pending_return = None, None, None
        self.cancelled, self.last_blocked = set(), 0
        if callable(getattr(self.base, 'reset', None)):
            self.base.reset()

    def act(self, observation):
        turn = round(float(observation.features[0])*20)
        if turn != self.turn:
            self.reset()
            self.turn = turn
        context = cancelled_play_context(observation)
        if self.pending_return is not None and self.pending_return[0] == context:
            self.cancelled.add(self.pending_return)
        self.pending_return = None
        mask = observation.action_mask.copy()
        for hand in range(10):
            if (context, hand) in self.cancelled:
                mask[46+8*hand:54+8*hand] = False
        self.last_blocked = int(observation.action_mask.sum()-mask.sum())
        if not mask.any():
            mask, self.last_blocked = observation.action_mask, 0
        decision = self.base.act(ObservationV2(observation.features, observation.cards, mask))
        if decision.action == 7 and self.last_play is not None:
            self.pending_return = self.last_play
        self.last_play = ((context, (decision.action-46)//8)
                          if 46 <= decision.action < 126 else None)
        return Decision(decision.action, f'cancel memory: blocked={self.last_blocked}; {decision.reason}')


def load_compound_policy(checkpoint, device='cpu', cpu_threads=1):
    return CancelMemoryPolicy(load_reorder_policy(checkpoint, device, cpu_threads))
