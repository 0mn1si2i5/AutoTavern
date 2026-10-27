"""Reviewed observe2 contract, with an explicit legacy baseline projection."""
from dataclasses import dataclass

import numpy as np
import taverngym as tg


@dataclass(frozen=True)
class ObservationV2:
    features: np.ndarray
    cards: np.ndarray
    action_mask: np.ndarray

    def __post_init__(self):
        for name, dtype, shape in (("features", np.float32, (4017,)),
                                   ("cards", np.int32, (447,)), ("action_mask", bool, (245,))):
            raw = np.asarray(getattr(self, name))
            if raw.shape != shape or not np.isfinite(raw).all():
                raise ValueError(f"invalid observe2 {name}")
            if name == "cards" and ((raw < 0) | (raw >= tg.N_CARDS) | (raw != np.floor(raw))).any():
                raise ValueError("invalid card vocabulary index")
            if name == "action_mask" and not np.isin(raw, [0, 1]).all():
                raise ValueError("invalid action mask")
            value = np.array(raw, dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_game(cls, game, seat):
        f, c = game.observe2(seat)
        return cls(np.frombuffer(f, dtype="<f4"), np.frombuffer(c, dtype="<i4"),
                   np.frombuffer(game.legal_mask(seat), dtype=np.uint8))

    def legacy(self):
        """Only for the frozen v0.1 baseline; v1 is a verified, exact prefix of v2.

        Learned models consume v2. This projection never establishes compatibility:
        the full engine/schema/vocabulary contract must already have been checked.
        """
        from .observation import Observation
        return Observation(self.features[:3227], self.cards[:124], self.action_mask)


def extension_slices():
    result = []
    for start, blocks in zip((3227, 124), tg.obs2_blocks()):
        mapping = {}
        for name, size in blocks:
            mapping[name] = slice(start, start+size)
            start += size
        result.append(mapping)
    return tuple(result)
