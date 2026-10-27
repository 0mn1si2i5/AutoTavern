"""Optional explicit heuristic controller around a learned policy, not an RL gain.

Rejects already observed board permutations within unchanged visible own resources,
and caps move recommendations per turn. Predicting a deterministic board insertion
does not call the simulator, copy a save, or inspect unknown/random outcomes.
"""
import hashlib
from pathlib import Path

from .observation_v2 import ObservationV2, extension_slices
from .policies import Decision


def runtime_config(budget):
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 <= budget <= 100:
        raise ValueError("move budget must be an integer in 0..100")
    return ({"move_guard": {"version": "visible-move-guard-v1", "budget": budget,
                             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
            if budget else {})


class MoveGuardPolicy:
    observation_version = 2

    def __init__(self, base, budget=7):
        if not 1 <= budget <= 100:
            raise ValueError("move budget must be in 1..100")
        self.base, self.budget = base, budget
        self.move_guard_budget = budget
        self.runtime_config = runtime_config(budget)
        self.version = f"{base.version}+visible-move-guard-v1-budget{budget}"
        self.fs, self.cs = extension_slices()
        self.reset()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def reset(self):
        self.turn, self.context, self.moves = None, None, 0
        self.seen = set()

    def act(self, observation):
        f, c = observation.features, observation.cards
        turn = round(float(f[0])*20)
        if turn != self.turn:
            self.reset()
            self.turn = turn
        board = f[68:285].reshape(7, 31)
        extra = f[self.fs["board_x"]].reshape(7, 9)
        attachments = c[self.cs["board_att"]].reshape(7, 6)
        counts = f[self.fs["board_att_n"]].reshape(7, 6)
        units = tuple(board[i].tobytes()+extra[i].tobytes()+c[19+i:20+i].tobytes()
                      +attachments[i].tobytes()+counts[i].tobytes() for i in range(7) if board[i, 0])
        context = (turn, float(f[7]), f[285:1645].tobytes(), c[26:60].tobytes(), tuple(sorted(units)))
        if context != self.context:
            self.context, self.seen = context, set()
        self.seen.add(units)
        mask = observation.action_mask.copy()
        blocked = 0
        for action in range(196, 245):
            if not mask[action]:
                continue
            source, target = divmod(action-196, 7)
            reordered = list(units)
            # Native legal-mask validation guarantees both slots exist.
            reordered.insert(target, reordered.pop(source))
            if self.moves >= self.budget or tuple(reordered) in self.seen:
                mask[action] = False
                blocked += 1
        # Never manufacture an illegal fallback if a future rule has only moves.
        if not mask.any():
            mask = observation.action_mask
            blocked = 0
        decision = self.base.act(ObservationV2(f, c, mask))
        if 196 <= decision.action < 245:
            self.moves += 1
        return Decision(decision.action, f"move guard: blocked={blocked}, moves={self.moves}/{self.budget}; {decision.reason}")
