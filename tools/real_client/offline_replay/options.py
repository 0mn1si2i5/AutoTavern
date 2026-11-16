"""Parse the option/target hierarchy, preserving log indices and entity bindings."""

import re

ENTRY = re.compile(
    r"^( +)(option|subOption|target) (\d+) (?:type=(\w+) )?(?:mainEntity|entity)=(.*?) error=(\w+)"
)
REF = re.compile(r" id=(\d+) zone=(\w+) zonePos=(\d+) cardId=([^\s\]]*) player=(\d+)")


def reference(text):
    match = REF.search(text)
    if not match:
        return {}
    return dict(
        id=int(match[1]), zone=match[2], pos=int(match[3]), card=match[4], controller=int(match[5])
    )


class OptionTree:
    def __init__(self, number, offset):
        self.number = number
        self.offset = offset
        self.options = {}
        self.complete = False
        self.parent = None
        self.child = None
        self.stale = []

    def add(self, message):
        m = ENTRY.match(message)
        if not m:
            return None
        ref = reference(m[5])
        node = dict(index=int(m[3]), entity=ref, error=m[6])
        if m[2] == "option":
            self.parent = node
            self.child = None
            node.update(type=m[4], targets=[], subs=[])
            self.options[node["index"]] = node
        elif m[2] == "subOption" and self.parent:
            self.child = node
            node["targets"] = []
            self.parent["subs"].append(node)
        elif m[2] == "target" and self.parent:
            dest = self.child if len(m[1]) == 6 else self.parent
            if dest is None:
                raise ValueError("orphan subOption target")
            dest["targets"].append(node)
        return node if m[2] == "option" else None


def mapping(kind, sequence, status="mapped", reason=None):
    return dict(kind=kind, native_action_sequence=sequence, status=status, reason=reason)


def map_option(option, send, view, build, catalog):
    src = option["entity"]
    eid, cid = src.get("id"), src.get("card", "")
    target, pos, sub = send["target"], send["position"], send["sub"]
    groups = {z: [e["id"] for e in view[z]] for z in ("board", "shop", "hand", "powers")}
    board, shop, hand, powers = (groups[z] for z in ("board", "shop", "hand", "powers"))
    if option["error"] != "NONE":
        return mapping("illegal", [], "unresolved", option["error"])
    choices = [c for c in option["subs"] if c["error"] == "NONE"]
    if choices and sub < 0:
        return mapping("choose_one", [], "unresolved", "required subOption missing")
    node = option
    if sub >= 0:
        node = next((c for c in choices if c["index"] == sub), None)
        if node is None:
            return mapping("choose_one", [], "unresolved", "subOption absent or illegal")
    valid = {t["entity"].get("id") for t in node["targets"] if t["error"] == "NONE"}
    if target and target not in valid:
        return mapping("target", [], "unresolved", "target absent or illegal")
    if "DragBuy" in cid and target in shop[:8]:
        return mapping("buy", [8 + shop.index(target)])
    if "DragSell" in cid:
        if target in board[:7]:
            return mapping("sell", [16 + board.index(target)])
        if target in hand and build == 253216:
            return mapping(
                "sell_hand", [], "registered_excluded", "legacy hand sale has no native slot"
            )
    for fragment, kind, code in [
        ("Reroll_Button", "refresh", 1),
        ("LockAll", "freeze", 2),
        ("TechUp", "upgrade", 3),
        ("Button_DarkGift", "dark_gift", 6),
    ]:
        if fragment in cid:
            return mapping(kind, [code])
    if eid in shop and pos:
        return mapping(
            "shop_move", [], "registered_excluded", "native tavern has no shop reposition"
        )
    if eid in powers[:2]:
        kind, action = "hero_power", 4 + powers.index(eid)
    elif eid in hand[:10]:
        if pos > 7 or pos < 0:
            return mapping("play", [], "unresolved", "position out of range")
        action = 46 + 8 * hand.index(eid) + max(0, pos - 1)
        kind = "targeted_play" if target else "play"
    elif eid in board[:7] and (
        target
        or (
            pos == 0
            and option.get("type") == "POWER"
            and re.search(r"\bActivate\b", catalog.get(cid, {}).get("text", ""), re.I)
        )
    ):
        kind, action = "activate", 23 + board.index(eid)
    elif eid in board[:7] and 1 <= pos <= len(board) <= 7:
        if board.index(eid) == pos - 1:
            return mapping(
                "move", [], "unsupported", "client same-slot move; native mask forbids from == to"
            )
        return mapping("move", [196 + 7 * board.index(eid) + pos - 1])
    else:
        return mapping("unresolved", [], "unresolved", "no public source/slot binding")
    seq = [action]
    if sub >= 0:
        if sub >= 16:
            return mapping("choose_one", seq, "unsupported", "too many subOptions")
        kind = "choose_one"
        seq.append(30 + sub)
    if target:
        # Native legal_targets enumerates eligible board slots, then shop slots;
        # discard choices enumerate eligible hand slots. Never use raw entity-ID order.
        ids = hand if target in hand else board + shop
        candidates = [i for i in ids if i in valid]
        if target not in candidates or candidates.index(target) >= 16 or valid - set(candidates):
            return mapping(kind, seq, "unsupported", "target vocabulary cannot be aligned")
        seq.append(30 + candidates.index(target))
    return mapping(kind, seq)
