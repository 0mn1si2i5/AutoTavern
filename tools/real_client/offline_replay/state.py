"""Store explicit zone tags. Export only current shop and local player entities."""

import re

IDENT = re.compile(r" id=(\d+) zone=")
NEW = re.compile(r"FULL_ENTITY - Creating ID=(\d+) CardID=(\S*)")
UPDATE = re.compile(
    r"(?:FULL_ENTITY|SHOW_ENTITY|CHANGE_ENTITY) - Updating (?:Entity=)?(.*?) CardID=(\S*)"
)
PLAYER = re.compile(r"Player EntityID=(\d+) PlayerID=(\d+)")
CHANGE = re.compile(r"(?:TAG_CHANGE |HIDE_ENTITY - )Entity=(.*?) tag=(\S+) value=(\S+)")
TAGS = set(
    "TAG_SCRIPT_DATA_NUM_1 TAG_SCRIPT_DATA_NUM_2 TAG_SCRIPT_DATA_NUM_3 ZONE CONTROLLER ZONE_POSITION CARDTYPE ATK HEALTH ARMOR DAMAGE COST TECH_LEVEL EXHAUSTED BACON_LOCKED_MULLIGAN_HERO PLAYER_TECH_LEVEL RESOURCES RESOURCES_USED TEMP_RESOURCES NUM_OPTIONS_PLAYED_THIS_TURN TURN STEP STATE BACON_IN_COMBAT_PHASE ATTACHED CREATOR MODULAR".split()
)
# State needed by the read-only advisor. This remains a whitelist: identities,
# account data, game seeds and private random state are never retained here.
TAGS.update(
    """
CARDRACE TAUNT DIVINE_SHIELD WINDFURY REBORN VENOMOUS POISONOUS STEALTH MAGNETIC
MEGA_WINDFURY CLEAVE IMMUNE IMMUNE_ATTACKING FROZEN PREMIUM BATTLEGROUNDS_PREMIUM
HAS_ACTIVATE_POWER NUM_TURNS_IN_PLAY BACON_TRIPLE_UPGRADE_MINION_ID
BACON_TRIPLED_BASE_MINION_ID BATTLECRY DEATHRATTLE TAG_ONE_TURN_EFFECT
NUM_CARDS_PLAYED_THIS_TURN NUM_MINIONS_PLAYED_THIS_TURN NUM_RESOURCES_SPENT_THIS_GAME
NUM_HERO_POWER_USES_THIS_TURN HERO_ENTITY CURRENT_PLAYER PLAYSTATE
NEXT_OPPONENT_PLAYER_ID BACON_TRINKETS_ACTIVE BACON_DARK_GIFTS_ACTIVE
BACON_PLAYER_LEADERBOARD_PLACE PLAYER_LEADERBOARD_PLACE 2442
""".split()
)


def entity_id(token):
    if token.isdecimal():
        return int(token)
    m = IDENT.search(token)
    return int(m[1]) if m else None


class ZoneState:
    def __init__(self):
        self.entities = {}
        self.players = {}
        self.names = {}  # Private lookup for TAG_CHANGE only, never serialized.
        self.game = {}
        self.game_id = None
        self.initial = None
        self.initial_snapshot = False
        self.complete_entities = set()

    def identity(self, token):
        if token == "GameEntity":
            return self.game_id
        if token in self.names:
            return self.players.get(self.names[token])
        return entity_id(token)

    def apply(self, raw):
        msg = raw.strip()
        if msg.startswith("tag="):
            if self.initial is not None:
                tag, sep, value = msg[4:].partition(" value=")
                if sep and tag in TAGS:
                    previous = self.entities[self.initial].get(tag)
                    self.entities[self.initial][tag] = value
                    if self.initial == self.game_id:
                        self.game[tag] = value
                    if tag in ("CREATOR", "MODULAR", "ATTACHED"):
                        return self.initial, tag, previous, value
            return None
        if self.initial is not None and self.initial_snapshot:
            self.complete_entities.add(self.initial)
        self.initial = None
        self.initial_snapshot = False
        if msg.startswith("GameEntity EntityID="):
            self.game_id = int(msg.split("=")[1])
            self.initial = self.game_id
            self.entities[self.initial] = {}
        elif m := PLAYER.match(msg):
            self.initial = int(m[1])
            self.players[int(m[2])] = self.initial
            self.entities[self.initial] = {}
        elif m := NEW.match(msg):
            self.initial = int(m[1])
            self.complete_entities.discard(self.initial)
            self.entities[self.initial] = {"card": m[2]}
            self.initial_snapshot = True
        elif m := UPDATE.match(msg):
            self.initial = self.identity(m[1])
            if self.initial is not None:
                self.complete_entities.discard(self.initial)
                self.entities.setdefault(self.initial, {})["card"] = m[2]
                self.initial_snapshot = msg.startswith(("FULL_ENTITY", "SHOW_ENTITY"))
        elif m := CHANGE.match(msg):
            eid, tag, value = self.identity(m[1]), m[2], m[3]
            if eid is not None and tag in TAGS:
                tags = self.entities.setdefault(eid, {})
                previous = tags.get(tag)
                tags[tag] = value
                if eid == self.game_id:
                    self.game[tag] = value
                return eid, tag, previous, value
        return None

    def zones(self, own, shop):
        result = {zone: [] for zone in ("board", "hand", "shop", "powers", "hero")}
        for eid, tags in self.entities.items():
            ctrl, zone = tags.get("CONTROLLER"), tags.get("ZONE")
            pos = int(tags.get("ZONE_POSITION", 0))
            typ = tags.get("CARDTYPE")
            if own is not None and ctrl == str(own):
                if zone == "HAND" and pos > 0:
                    dest = "hand"
                elif zone == "PLAY" and typ in ("MINION", "HERO_POWER", "HERO"):
                    dest = {"MINION": "board", "HERO_POWER": "powers", "HERO": "hero"}[typ]
                    if dest == "board" and pos <= 0:
                        continue
                else:
                    continue
            elif (
                self.game.get("BACON_IN_COMBAT_PHASE") != "1"
                and shop is not None
                and ctrl == str(shop)
                and zone == "PLAY"
                and pos > 0
                and typ in ("MINION", "SPELL", "BATTLEGROUND_SPELL")
            ):
                dest = "shop"
            else:
                continue
            result[dest].append(
                {
                    "id": eid,
                    "card": tags.get("card", ""),
                    "pos": pos,
                    **{
                        k: tags[k]
                        for k in ("ATK", "HEALTH", "DAMAGE", "COST", "EXHAUSTED")
                        if k in tags
                    },
                }
            )
        for items in result.values():
            items.sort(key=lambda e: (e["pos"], e["id"]))
        player = self.entities.get(self.players.get(own), {})
        result["player"] = {
            k: player[k]
            for k in ("PLAYER_TECH_LEVEL", "RESOURCES", "RESOURCES_USED", "TEMP_RESOURCES")
            if k in player
        }
        return result
