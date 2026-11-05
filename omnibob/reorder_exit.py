"""Experimental move-cycle exit around stateless inference, not trained weights.

Within a consecutive move-only segment, stop reordering when the preferred move
would revisit a visible board arrangement. Ask the same model for a non-move
instead. No fixed move budget, simulator access, save copying, or future RNG.
Keep this INSIDE RepeatGuardPolicy: its action history must record only actions
actually returned, never a discarded proposal. The base must be stateless.
"""
import hashlib
from pathlib import Path

from .observation_v2 import ObservationV2, extension_slices
from .policies import Decision


def board_layout(observation):
    """Include stats and attachments, distinguishing different copies of one card."""
    fs, cs = extension_slices()
    f, c = observation.features, observation.cards
    board = f[68:285].reshape(7, 31)
    extra = f[fs["board_x"]].reshape(7, 9)
    attachments = c[cs["board_att"]].reshape(7, 6)
    counts = f[fs["board_att_n"]].reshape(7, 6)
    return tuple(board[i].tobytes() + extra[i].tobytes() + c[19+i:20+i].tobytes()
                 + attachments[i].tobytes() + counts[i].tobytes()
                 for i in range(7) if board[i, 0])


class ReorderExitPolicy:
    observation_version = 2

    def __init__(self, base):
        self.base = base
        self.version = f"{base.version}+visible-reorder-exit-v1"
        self.runtime_config = {**base.runtime_config, "reorder_exit": {
            "version": "visible-reorder-exit-v1",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.context, self.previous_move, self.seen = None, False, set()

    def act(self, observation):
        units = board_layout(observation)
        context = (round(float(observation.features[0])*20), tuple(sorted(units)))
        if not self.previous_move or context != self.context:
            self.seen = set()
        self.context = context
        self.seen.add(units)
        decision = self.base.act(observation)
        cycle = False
        if 196 <= decision.action < 245:
            source, target = divmod(decision.action-196, 7)
            reordered = list(units)
            reordered.insert(target, reordered.pop(source))
            cycle = tuple(reordered) in self.seen
            if cycle and observation.action_mask[:196].any():
                mask = observation.action_mask.copy()
                mask[196:] = False
                decision = self.base.act(ObservationV2(observation.features, observation.cards, mask))
        self.previous_move = 196 <= decision.action < 245
        return Decision(decision.action, f"reorder exit: cycle={cycle}; {decision.reason}")


def load_reorder_policy(checkpoint, device="cpu", cpu_threads=1):
    """Explicit local candidate; the frozen deployment loader/defaults stay intact."""
    from .learned import load_policy
    from .repeat_guard import RepeatGuardPolicy
    frozen = load_policy(checkpoint, device, cpu_threads)
    if not isinstance(frozen, RepeatGuardPolicy) or frozen.base.runtime_config:
        raise ValueError("reorder experiment requires repeat-only frozen deployment")
    return RepeatGuardPolicy(ReorderExitPolicy(frozen.base), frozen.repeat_limit)
