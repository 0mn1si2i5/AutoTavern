"""The policy boundary deliberately excludes Game, saves, seeds and debug views.

Feature semantics below are for upstream fb61201 / observe v1 only. A changed engine
must be explicitly reviewed before updating the frozen compatibility manifest.
"""
from dataclasses import dataclass

import numpy as np
import taverngym as tg


@dataclass(frozen=True)
class Observation:
    features: np.ndarray
    cards: np.ndarray
    action_mask: np.ndarray

    def __post_init__(self):
        for name, dtype, size in (
            ("features", np.float32, tg.OBS_FLOATS),
            ("cards", np.int32, tg.OBS_CARDS),
            ("action_mask", bool, tg.N_ACTIONS),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.shape != (size,):
                raise ValueError(f"{name}: expected {(size,)}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name}: non-finite input")
            if name == "cards" and ((value < 0) | (value >= tg.N_CARDS)).any():
                raise ValueError("card index outside the vocabulary")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_game(cls, game, seat):
        f, c = game.observe(seat)
        return cls(np.frombuffer(f, dtype="<f4"), np.frombuffer(c, dtype="<i4"),
                   np.frombuffer(game.legal_mask(seat), dtype=np.uint8))

    @property
    def legal(self):
        return np.flatnonzero(self.action_mask).tolist()

    def blocks(self):
        layout = dict(tg.layout())
        offset = layout["GLOBAL_F"] + layout["N_QUESTS"] * layout["QUEST_F"]
        pending = self.features[offset:offset + layout["PENDING_KINDS"]]
        offset += layout["PENDING_KINDS"]
        blocks = {"global": self.features[:layout["GLOBAL_F"]], "pending": pending}
        card_offset = 1 + layout["N_POWERS"] + 2 + 3 * layout["N_QUESTS"] + layout["N_SECRETS"]
        for name, count, width in (
            ("board", layout["N_BOARD"], layout["MINION_F"]),
            ("hand", layout["N_HAND"], layout["ITEM_F"]),
            ("shop", layout["N_SHOP"], layout["ITEM_F"]),
            ("choice", layout["N_CHOICE"], layout["ITEM_F"]),
        ):
            blocks[name] = self.features[offset:offset + count * width].reshape(count, width)
            blocks[name + "_cards"] = self.cards[card_offset:card_offset + count]
            offset += count * width
            card_offset += count
        return blocks
