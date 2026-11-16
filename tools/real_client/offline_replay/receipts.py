"""Corroborate a sent option within its own epoch, turn and option window."""

import re
from .state import entity_id

BLOCK = re.compile(
    r"BLOCK_START BlockType=(\w+) Entity=(.*?) EffectCardId=.*? Target=(.*?) SubOption=(-?\d+)"
)


class Receipt:
    def __init__(self, row, player_eid, counter):
        self.row = row
        self.player = player_eid
        self.counter = counter
        self.same_turn = True
        self.data = row["execution"] = {
            "status": "pending",
            "cooccurrence": False,
            "player_entity": player_eid,
            "counter_before": counter,
        }

    def block(self, msg, ev):
        m = BLOCK.match(msg)
        if not m:
            return
        if m[1] in ("PLAY", "POWER", "MOVE_MINION", "ATTACK"):
            self.data.setdefault("first_effect", ev.start)
            self.data.setdefault("first_effect_kind", m[1])
        bind = self.row["binding"]
        expected = "MOVE_MINION" if self.row["mapping"]["kind"] in ("move", "shop_move") else "PLAY"
        if (
            self.same_turn
            and bind["source"] is not None
            and m[1] == expected
            and entity_id(m[2]) == bind["source"]
            and entity_id(m[3]) == bind["target"]
            and int(m[4]) == bind["sub"]
        ):
            self.data.setdefault("bound_effect", ev.start)
            self.data.setdefault("latency_s", round(ev.seconds - self.row["seconds"], 6))

    def tag(self, change, ev):
        eid, tag, old, value = change
        if tag != "NUM_OPTIONS_PLAYED_THIS_TURN":
            return
        self.data.setdefault("first_counter_tag", ev.start)
        if (
            self.same_turn
            and "bound_effect" in self.data
            and self.player is not None
            and eid == self.player
            and int(value) == int(old or 0) + 1
            and int(old or 0) == self.counter
        ):
            self.data.setdefault("own_counter_increment", ev.start)
            self.data.setdefault("counter_after", int(value))

    def finish(self, reason):
        d = self.data
        d["cooccurrence"] = "first_effect" in d and "first_counter_tag" in d
        d["status"] = (
            "confirmed" if "bound_effect" in d and "own_counter_increment" in d else "unconfirmed"
        )
        d["window_closed_by"] = reason
