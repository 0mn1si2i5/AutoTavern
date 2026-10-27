"""Explicit frozen synthetic opponents; not a real-game-calibrated opponent library."""
from .observation import Observation
from .policies import Decision, OmniBobPolicy

DESCRIPTIONS = {
    "scripted": "seven unmodified upstream scripted bots",
    "heuristic": "seven frozen tempo-v0.1 observation-only opponents",
    "mixed": "frozen tempo-v0.1 and upstream bots, selected by (seed+seat) parity",
    "learned": "seven identical frozen observation-only learned opponents",
    "mixed_learned": "upstream, frozen tempo-v0.1 and frozen learned opponents, selected by (seed+seat) modulo 3",
}

LEARNED_PROFILES = ("learned", "mixed_learned")


class OpponentSet:
    def __init__(self, game, seed, controlled_seat, profile="scripted", learned_policy=None):
        if profile not in DESCRIPTIONS:
            raise ValueError(f"unsupported opponent profile: {profile}")
        self.game = game
        self.profile = profile
        self.learned = {}
        if profile in LEARNED_PROFILES:
            if learned_policy is None:
                raise ValueError("learned opponents require an explicit frozen policy")
            from .repeat_guard import clone_runtime
            for seat in range(8):
                if seat != controlled_seat and (profile == "learned" or (seed+seat) % 3 == 2):
                    self.learned[seat] = clone_runtime(learned_policy)
        elif learned_policy is not None:
            raise ValueError("opponent checkpoint requires a learned opponent profile")
        self.policies = {seat: OmniBobPolicy() for seat in range(8) if seat != controlled_seat
                         and (profile == "heuristic" or profile == "mixed" and (seed+seat) % 2 == 0
                              or profile == "mixed_learned" and (seed+seat) % 3 == 1)}

    def act(self, seat):
        if seat in self.learned:
            from .observation_v2 import ObservationV2
            result = self.learned[seat].act(ObservationV2.from_game(self.game, seat))
            return Decision(result.action, "frozen learned opponent: "+result.reason)
        if seat in self.policies:
            result = self.policies[seat].act(Observation.from_game(self.game, seat))
            return Decision(result.action, "frozen tempo opponent: "+result.reason)
        return Decision(int(self.game.bot_action(seat)), "upstream opponent")

    def save(self):
        heuristic = {str(seat): {"turn": p.turn, "actions": p.actions, "refreshes": p.refreshes,
                                "used_buttons": sorted(p.used_buttons)} for seat, p in self.policies.items()}
        if self.learned:
            from .runtime_state import save_runtime_state
            saved = {"learned": {str(seat): save_runtime_state(policy) for seat, policy in self.learned.items()}}
            if self.profile == "mixed_learned":
                saved["heuristic"] = heuristic
            return saved
        return heuristic

    def restore(self, state):
        if self.learned:
            from .runtime_state import restore_runtime_state
            expected = {"learned", "heuristic"} if self.profile == "mixed_learned" else {"learned"}
            if set(state) != expected or set(state["learned"]) != {str(seat) for seat in self.learned}:
                raise ValueError("saved learned opponent seats do not match")
            for seat, policy in self.learned.items():
                restore_runtime_state(policy, state["learned"][str(seat)])
            if self.profile != "mixed_learned":
                return
            state = state["heuristic"]
        if set(state) != {str(seat) for seat in self.policies}:
            raise ValueError("saved opponent profile does not match")
        for seat, policy in self.policies.items():
            saved = state[str(seat)]
            policy.turn, policy.actions, policy.refreshes = saved["turn"], saved["actions"], saved["refreshes"]
            policy.used_buttons = set(saved["used_buttons"])
