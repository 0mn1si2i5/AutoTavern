"""Frozen v0.1 baselines. No model has been trained.

All policies implement act(Observation) -> Decision. Only the upstream comparator
has a native chooser callback; OmniBob and Random never receive a game handle.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import taverngym as tg

from .observation import Observation


@dataclass(frozen=True)
class Decision:
    action: int
    reason: str


class Policy(Protocol):
    def act(self, observation: Observation) -> Decision: ...


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def act(self, observation):
        return Decision(int(self.rng.choice(observation.legal)), "uniform over legal actions")


class ScriptedPolicy:
    """Unmodified upstream bot, not an OmniBob training result.

    bot.rs consults own zones and the legal mask. Hero/quest/ChooseOne tie breaks
    use the engine RNG; the binding restores that RNG after choosing. This
    comparator is therefore local-only and not a deployable observation policy.
    """
    def __init__(self, chooser):
        self.chooser = chooser

    def act(self, observation):
        return Decision(int(self.chooser()), "upstream scripted bot (fb61201)")


class OmniBobPolicy:
    """Explainable tempo/economy heuristic using only visible numeric features.

    Unknown spell/target semantics are deliberately generic. This is a baseline,
    not a claim of optimal play or a reproduction of TavernAgent.
    """
    version = "tempo-v0.1"

    def __init__(self):
        width, raw = tg.card_static()
        self.static = np.frombuffer(raw, dtype="<f4").reshape(tg.N_CARDS, width)
        self.turn = None
        self.actions = 0
        self.refreshes = 0
        self.used_buttons = set()

    @staticmethod
    def minion_value(m):
        if not m[0]:
            return 0.0
        attack, health = float(m[1] * 10), float(m[2] * 10)
        kw = m[8:20]
        return (attack + health + min(attack, health) * 0.35 + float(m[4]) * 6
                + float(kw[1]) * (attack + 3) * 0.65 + float(kw[3]) * 4
                + float(kw[4] + kw[5]) * 12 + float(kw[9]) * attack * 0.6
                + float(kw[2]) * attack * 0.3)

    def act(self, observation):
        legal = observation.legal
        if not legal:
            raise ValueError("policy called when no action is legal")
        b = observation.blocks()
        g, board, hand, shop = (b[x] for x in ("global", "board", "hand", "shop"))
        turn = round(float(g[0]) * 20)
        if turn != self.turn:
            self.turn, self.actions, self.refreshes = turn, 0, 0
            self.used_buttons.clear()
        self.actions += 1

        def choose(action, reason):
            return Decision(int(action), reason)

        def best(actions, score):
            return max(actions, key=lambda a: (score(a), -a))

        choices = [a for a in legal if 30 <= a < 46]
        if choices:
            kind = int(np.argmax(b["pending"]))
            def score(a):
                i = a - 30
                item = b["choice"][i]
                card = int(b["choice_cards"][i])
                if kind == 1:  # Hero: static armor only, no human win-rate table.
                    return float(self.static[card, 14])
                if item[1]:
                    return self.minion_value(item[9:])
                return float(item[13]) * 6 - float(item[2]) * 5 * 0.2
            a = best(choices, score)
            return choose(a, f"pending kind={kind}: visible value/armor score={score(a):.2f}")
        if self.actions > 55 and 0 in legal:
            return choose(0, "bounded turn: stop after 55 decisions")

        # Put owned resources onto the board before buying more.
        magnets = [a for a in legal if 126 <= a < 196]
        if magnets:
            a = best(magnets, lambda a: self.minion_value(hand[(a-126)//7, 9:])
                     + 0.1 * self.minion_value(board[(a-126) % 7]))
            return choose(a, "magnetize: add owned stats without consuming a board slot")
        plays = [a for a in legal if 46 <= a < 126]
        if plays:
            def score_play(a):
                h, pos = divmod(a-46, 8)
                row = hand[h]
                value = self.minion_value(row[9:]) if row[1] else 5.0
                # Prefer appending; movement below orders attackers once purchases end.
                return value - abs(pos - round(float(g[33]) * 7)) * 0.01
            return choose(best(plays, score_play), "play owned card for immediate tempo")

        nboard = int(np.count_nonzero(board[:, 0]))
        nhand = int(np.count_nonzero(hand[:, 1]))
        weakest = min(range(nboard), key=lambda i: self.minion_value(board[i])) if nboard else None
        weak_value = self.minion_value(board[weakest]) if weakest is not None else 0.0
        owned_cards = list(b["board_cards"][board[:, 0] > 0]) + list(b["hand_cards"][hand[:, 1] > 0])
        def buy_value(a):
            i = a - 8
            row = shop[i]
            if not row[1]:
                return 4.0 + float(row[13]) * 6 * 0.2
            copies = owned_cards.count(b["shop_cards"][i]) if not row[14] else 0
            pair = 10 if copies >= 2 else (2 if copies == 1 else 0)
            synergy = sum(float(np.minimum(row[29:40], m[20:31]).sum()) for m in board if m[0])
            return self.minion_value(row[9:]) + pair + min(synergy, 5) * 0.5

        buys = [a for a in legal if 8 <= a < 16]
        # Avoid holding a better minion forever on a full board.
        held_value = max((self.minion_value(h[9:]) for h in hand if h[1]), default=0)
        if nboard == 7 and weakest is not None and 16 + weakest in legal:
            if held_value > weak_value + 2:
                return choose(16 + weakest, "sell weakest to play a stronger owned minion")

        gold, tier, price = float(g[7]) * 10, round(float(g[10]) * 6), float(g[11]) * 10
        if 3 in legal and nboard >= min(turn, 4) and (
            price <= 2 or (turn >= 2 * tier + 1 and gold - price >= 3)
        ):
            return choose(3, "upgrade: board established and affordable economy curve")

        candidates = []
        for a in buys:
            i = a - 8
            if not shop[i, 1] or nboard + nhand < 7:
                candidates.append(a)
            elif owned_cards.count(b["shop_cards"][i]) >= 2 and not shop[i, 14]:
                candidates.append(a)
            elif nboard == 7 and nhand == 0 and buy_value(a) > weak_value + 5:
                if weakest is not None and 16 + weakest in legal:
                    return choose(16 + weakest, "sell weakest to make room for visible shop upgrade")
        if candidates:
            a = best(candidates, buy_value)
            return choose(a, f"buy: visible tempo/pair/synergy score={buy_value(a):.2f}")

        for a in [4, 5, 6, *range(23, 30)]:
            if a in legal and a not in self.used_buttons:
                self.used_buttons.add(a)
                return choose(a, "use available power/gift/activate once per turn")
        if 1 in legal and gold >= 3 and self.refreshes < 6:
            self.refreshes += 1
            return choose(1, "refresh: retain purchase budget, at most six refreshes per turn")
        # Strictly improving adjacent swaps terminate; no repeated move cycle.
        for i in range(max(0, nboard - 1)):
            a = 196 + (i + 1) * 7 + i
            if a in legal and float(board[i+1, 1]) > float(board[i, 1]):
                return choose(a, "position: move higher attack left")
        if 0 in legal:
            return choose(0, "end turn: no preferred economical action remains")
        return choose(legal[0], "mandatory legal fallback")


def make_policy(name, seed, chooser=None):
    if name == "random":
        return RandomPolicy(seed)
    if name == "scripted":
        if chooser is None:
            raise ValueError("scripted comparator requires a local native chooser")
        return ScriptedPolicy(chooser)
    if name == "omnibob":
        return OmniBobPolicy()
    raise ValueError(f"unknown policy: {name}")
