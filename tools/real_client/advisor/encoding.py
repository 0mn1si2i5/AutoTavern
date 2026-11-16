"""Visible-state transfer into frozen observe2, with explicit missingness.

This is an inference adapter, not a claim of equivalence across client patches.
It never reconstructs a Game or reads hidden opponents. The frozen network has
no missingness channel; every imputed position is reported to the caller.
"""

from __future__ import annotations

import copy
import re
import numpy as np
import taverngym as tg

from omnibob.observation_v2 import ObservationV2, extension_slices
from .state import metadata

PROFILE = "readonly-visible-observe2-v2"
RACES = (
    "BEAST",
    "DEMON",
    "DRAGON",
    "ELEMENTAL",
    "MECHANICAL",
    "MURLOC",
    "PIRATE",
    "QUILBOAR",
    "UNDEAD",
    "ABERRATION",
    "NAGA",
)
KEYWORDS = (
    "TAUNT",
    "DIVINE_SHIELD",
    "WINDFURY",
    "REBORN",
    "VENOMOUS",
    "POISONOUS",
    "STEALTH",
    "MAGNETIC",
    "MEGA_WINDFURY",
    "CLEAVE",
    "IMMUNE",
    "IMMUNE_ATTACKING",
)


class AdapterGap(ValueError):
    pass


def name(entity):
    return entity.get("name") or entity.get("card_id") or "未知卡牌"


def located(entity):
    """Disambiguate repeated names using the current visible zone and slot."""
    zone = entity.get("zone")
    position = int(entity.get("position", 0))
    label = {"board": "场上", "hand": "手牌", "shop": "商店", "powers": "技能"}.get(zone, "")
    prefix = f"{label}第 {position} 位" if label and position > 0 else label
    if zone == "choices" and "choice_index" in entity:
        prefix = f"第 {entity['choice_index'] + 1} 项"
    result = (prefix + " ") if prefix else ""
    result += "「" + name(entity) + "」"
    tags = entity.get("tags", {})
    if entity.get("card_type") == "MINION" and "ATK" in tags and "HEALTH" in tags:
        result += f"（{int(tags['ATK'])}/{int(tags['HEALTH']) - int(tags.get('DAMAGE', 0))}）"
    elif entity.get("card_type") == "MINION" and entity.get("snapshot_complete"):
        result += f"（{int(tags.get('ATK', 0))}/{int(tags.get('HEALTH', 0)) - int(tags.get('DAMAGE', 0))}）"
    return result


def describe(binding):
    kind = binding["kind"]
    item = binding.get("entity", {})
    if kind == "end":
        return "停止经营，等待战斗"
    if kind in ("refresh", "upgrade", "freeze", "dark_gift"):
        return {
            "refresh": "刷新酒馆",
            "upgrade": "升级酒馆",
            "freeze": "切换酒馆冻结状态",
            "dark_gift": "使用黑暗发现",
        }[kind]
    if kind == "move":
        return f"将 {located(item)} 移到场上第 {binding['position'] + 1} 位"
    if kind == "play":
        return f"打出 {located(item)}" + (
            f"，放在第 {binding['position'] + 1} 位" if item.get("card_type") == "MINION" else ""
        )
    if kind == "magnetize":
        return f"将 {located(item)} 磁力融合到 {located(binding['target'])}"
    return (
        {
            "buy": "购买",
            "sell": "出售",
            "power": "使用技能",
            "activate": "发动",
            "choose": "选择",
            "target": "选择目标",
            "suboption": "选择分支",
        }.get(kind, kind)
        + " "
        + located(item)
    )


class Encoder:
    def __init__(self, catalog):
        self.catalog = catalog
        self.ids = {tg.card_info(i)[0]: i for i in range(1, tg.N_CARDS)}
        self.fs, self.cs = extension_slices()
        self.aliases = {}

    def index(self, entity):
        if not entity:
            return 0
        cid = entity.get("model_card_id", entity.get("card_id", ""))
        if cid in self.ids:
            return self.ids[cid]
        card = self.catalog.get(cid) or {}
        parent = card.get("battlegroundsSkinParentId")
        if parent:
            base = self.catalog.get(parent) or {}
            if base.get("id") in self.ids and base.get("heroPowerDbfId") == card.get(
                "heroPowerDbfId"
            ):
                self.aliases[cid] = base["id"]
                return self.ids[base["id"]]
        raise AdapterGap("冻结模型不认识卡牌：" + (cid or "身份未读到"))

    def actions(self, view):
        mask = np.zeros(245, bool)
        bindings = {}

        def add(action, **binding):
            if binding.get("entity") and not binding["entity"].get("zone"):
                entity = binding["entity"]
                for zone in ("board", "hand", "shop", "powers", "choices"):
                    for index, candidate in enumerate(view.get(zone, [])):
                        if candidate["entity_id"] == entity["entity_id"]:
                            binding["entity"] = dict(entity, zone=zone, position=index + 1)
            binding["text"] = describe(binding)
            mask[action] = True
            bindings[str(action)] = binding

        choices = view.get("choices", [])
        if choices:
            for slot, entity in enumerate(choices):
                i = entity.get("choice_index", slot)
                if not entity.get("locked") and 0 <= i < 16:
                    add(30 + i, kind="choose", entity=entity, slot=i)
            return mask, bindings
        if not view.get("ready") or view.get("game_complete"):
            return mask, bindings
        board, hand, shop = (view[k] for k in ("board", "hand", "shop"))
        powers = view.get("powers", [])
        add(0, kind="end")
        for node in view["last_options"]["options"]:
            if node["error"] != "NONE":
                continue
            entity = node["entity"]
            eid, cid = entity["entity_id"], entity["card_id"]
            tags = entity["tags"]
            targets = [t["entity"] for t in node["targets"] if t["error"] == "NONE"]
            subs = [s for s in node.get("subs", []) if s["error"] == "NONE"]
            common = {"targets": targets, "suboptions": subs, "option_index": node["index"]}
            if "Reroll_Button" in cid:
                add(1, kind="refresh", entity=entity)
            elif "LockAll" in cid:
                add(2, kind="freeze", entity=entity)
            elif "TechUp" in cid:
                add(3, kind="upgrade", entity=entity)
            elif "Button_DarkGift" in cid:
                add(6, kind="dark_gift", entity=entity, **common)
            elif "DragBuy" in cid and len(hand) < 10:
                for i, item in enumerate(shop):
                    if int(tags.get("2442", -1)) == item["entity_id"]:
                        add(8 + i, kind="buy", entity=item, cost=int(tags.get("COST", 0)))
            elif "DragSell" in cid:
                ids = {t["entity_id"] for t in targets}
                for i, item in enumerate(board):
                    if item["entity_id"] in ids:
                        add(16 + i, kind="sell", entity=item)
            else:
                for i, power in enumerate(powers[:2]):
                    if power["entity_id"] == eid:
                        add(4 + i, kind="power", entity=power, **common)
                for i, item in enumerate(hand):
                    if item["entity_id"] != eid:
                        continue
                    if item.get("card_type") == "MINION":
                        if len(board) < 7:
                            for pos in range(len(board) + 1):
                                add(
                                    46 + i * 8 + pos,
                                    kind="play",
                                    entity=item,
                                    position=pos,
                                    **common,
                                )
                        # A magnetize alternative is exposed only if the current
                        # client option explicitly allows that target.
                        magnetic = (
                            int(item["tags"].get("MAGNETIC", 0))
                            or "MAGNETIC" in metadata(self.catalog, item["card_id"])["mechanics"]
                        )
                        if magnetic:
                            for j, target in enumerate(board):
                                if target["entity_id"] in {t["entity_id"] for t in targets}:
                                    add(
                                        126 + i * 7 + j,
                                        kind="magnetize",
                                        entity=item,
                                        target=target,
                                        position=j,
                                    )
                    else:
                        add(46 + i * 8, kind="play", entity=item, position=0, **common)
                for i, item in enumerate(board):
                    if item["entity_id"] != eid:
                        continue
                    is_activate = (
                        targets
                        or int(item["tags"].get("HAS_ACTIVATE_POWER", 0))
                        or re.search(
                            r"\bActivate\b",
                            metadata(self.catalog, item["card_id"]).get(
                                "text_en", metadata(self.catalog, item["card_id"]).get("text", "")
                            ),
                            re.I,
                        )
                    )
                    if node["type"] == "POWER" and is_activate:
                        add(23 + i, kind="activate", entity=item, **common)
                    # The archived client labels owned minion options POWER,
                    # even when the subsequent effect is MOVE_MINION. Movement
                    # is still limited to an explicitly available board source.
                    if node["type"] in ("POWER", "MOVE_MINION"):
                        for j in range(len(board)):
                            if i != j:
                                add(196 + i * 7 + j, kind="move", entity=item, position=j)
        return mask, bindings

    def encode(self, view):
        if not view.get("ready"):
            raise AdapterGap("当前画面尚未同步或不在可决策阶段")
        client_build = view.get("build")
        catalog_build = getattr(self.catalog, "meta", {}).get("build")
        if (
            client_build is not None
            and catalog_build is not None
            and str(client_build) != str(catalog_build)
        ):
            raise AdapterGap(
                f"卡表版本 {catalog_build} 与当前客户端 {client_build} 不一致，请加载对应版本卡表"
            )
        self.aliases = {}
        features = np.zeros(4017, np.float32)
        cards = np.zeros(447, np.int32)
        known = np.zeros(4017, bool)
        known_cards = np.zeros(447, bool)
        mask, bindings = self.actions(view)
        if not mask.any():
            raise AdapterGap("当前没有可映射的合法建议")

        def put(i, value):
            features[i] = value
            known[i] = True

        def ci(i, item):
            cards[i] = self.index(item)
            known_cards[i] = True

        def minion(item, start):
            if item is None:
                known[start : start + 31] = True
                return
            data, tags = metadata(self.catalog, item["card_id"]), item["tags"]
            put(start, 1)
            for at, key, fallback, scale in (
                (1, "ATK", "attack", 10),
                (3, "HEALTH", "health", 10),
                (4, "TECH_LEVEL", "techLevel", 6),
            ):
                # A static card's printed Attack/Health cannot replace a missing
                # dynamic value after buffs. Tier may use version-pinned data.
                value = tags.get(key, data.get(fallback) if key == "TECH_LEVEL" else None)
                if value is None and item.get("snapshot_complete") and key in ("ATK", "HEALTH"):
                    # Independently implement the client's zero-tag semantics,
                    # also used by HearthSim HDT Entity.GetTag/Attack/Health:
                    # https://github.com/HearthSim/Hearthstone-Deck-Tracker/blob/master/Hearthstone%20Deck%20Tracker/Hearthstone/Entities/Entity.cs
                    # Only after a completed FULL/SHOW snapshot; never fill from
                    # the card's printed/base stats or an incomplete fragment.
                    value = 0
                if value is None:
                    raise AdapterGap(f"{name(item)} 的 {key} 尚未读到")
                put(start + at, int(value) / scale)
            put(start + 2, features[start + 3] - int(tags.get("DAMAGE", 0)) / 10)
            # PREMIUM is cosmetic foil; normal/golden card linkage is semantic.
            put(start + 5, bool(data.get("battlegroundsNormalDbfId") or data.get("normal")))
            for i, key in enumerate(KEYWORDS):
                if key in tags:
                    put(start + 8 + i, int(tags[key]) != 0)
                else:
                    put(start + 8 + i, key in data["mechanics"])
            races = set(data.get("races", [])) | ({data["race"]} if data.get("race") else set())
            # CARDRACE's numeric client enum is not a race string; preserve the
            # named catalog races instead of silently adding an unmatched number.
            for i, race in enumerate(RACES):
                put(start + 20 + i, race in races or "ALL" in races)

        def item(value, start, *, cost=0, choice_kind=0):
            if value is None:
                known[start : start + 40] = True
                return
            put(start, 1)
            if choice_kind in (1, 3, 5):
                # obs.rs uses presence + card identity, not a pretend spell or
                # minion, for HeroSelect, ChooseOne and HeroPowerChoice.
                known[start : start + 40] = True
                return
            is_minion = value.get("card_type") == "MINION"
            put(start + 1, is_minion)
            if cost is not None:
                put(start + 2, cost / 5)
            if is_minion:
                minion(value, start + 9)
            else:
                put(start + 9, 1)
                data = metadata(self.catalog, value["card_id"])
                if data.get("techLevel") is not None:
                    put(start + 13, int(data["techLevel"]) / 6)
                # Data (e.g. Triple Reward's tier) is not the card's Attack.
                known[start + 11 : start + 13] = True
                known[start + 14 : start + 40] = True

        board, hand, shop = (view[k] for k in ("board", "hand", "shop"))
        if len(board) > 7 or len(hand) > 10 or len(shop) > 8:
            raise AdapterGap("当前区域超过冻结模型的槽位容量")
        choices = view.get("choices", [])
        kind = view.get("choice_kind", 2) if choices else 0
        put(0, view["round"] / 20)
        put(1, kind == 1)
        put(2, view["game_complete"])
        put(4, not view["game_complete"])
        hero = view.get("hero")
        if hero:
            tags = hero["tags"]
            if "HEALTH" in tags:
                put(5, (int(tags["HEALTH"]) - int(tags.get("DAMAGE", 0))) / 40)
            if "ARMOR" in tags:
                put(6, int(tags["ARMOR"]) / 20)
        if view.get("gold") is None and kind != 1:
            raise AdapterGap("当前金币未读到")
        if view.get("gold") is not None:
            put(7, view["gold"] / 10)
        player = view.get("player", {})
        if view.get("tier") is not None:
            put(10, int(view["tier"]) / 6)
        elif kind != 1:
            raise AdapterGap("当前酒馆等级未读到")
        put(33, len(board) / 7)
        put(34, len(hand) / 10)
        for i in range(9):
            put(59 + i, i == kind)
        ci(0, hero)
        powers = view.get("powers", [])
        for i in range(2):
            power = powers[i] if i < len(powers) else None
            ci(1 + i, power)
            put(16 + 3 * i, bool(mask[4 + i]))
            if power and "COST" in power["tags"]:
                put(17 + 3 * i, int(power["tags"]["COST"]) / 5)
            elif power is None:
                put(17 + 3 * i, 0)
                put(18 + 3 * i, 0)
        prices = {}
        for button in view.get("buttons", []):
            cid, tags = button["card_id"], button["tags"]
            if "COST" not in tags:
                continue
            cost = int(tags["COST"])
            if "DragBuy" in cid and "2442" in tags:
                prices[int(tags["2442"])] = cost
            if "Reroll_Button" in cid:
                put(12, cost / 2)
            if "TechUp" in cid:
                put(11, cost / 10)
        for i in range(7):
            m = board[i] if i < len(board) else None
            minion(m, 68 + i * 31)
            ci(19 + i, m)
        for items, size, start, card_start in ((hand, 10, 285, 26), (shop, 8, 685, 36)):
            for i in range(size):
                value = items[i] if i < len(items) else None
                cost = prices.get(value["entity_id"]) if value and items is shop else 0
                item(value, start + 40 * i, cost=cost)
                ci(card_start + i, value)
        by_index = {e.get("choice_index", i): e for i, e in enumerate(choices)}
        if any(i < 0 or i >= 16 for i in by_index):
            raise AdapterGap("选择数量超过冻结模型容量")
        for i in range(16):
            value = by_index.get(i)
            cost = (
                int(value["tags"]["COST"]) if value and kind == 6 and "COST" in value["tags"] else 0
            )
            item(value, 1005 + 40 * i, cost=cost, choice_kind=kind)
            ci(44 + i, value)
        if view.get("choice_source"):
            ci(4, view["choice_source"])
        gs = self.fs["g2"].start
        mapped_contracts = []
        for slot, power in enumerate(powers[:2]):
            if power["card_id"] != "BG21_HERO_000p":
                continue
            tags = power["tags"]
            data = metadata(self.catalog, power["card_id"])
            # heroes_special.rs: cariel returns 1 + values[0/1/2].
            # obs2.rs exposes only the first two values, never the hp increment.
            if data.get("cost") == 0:
                for at, tag in ((65, "TAG_SCRIPT_DATA_NUM_3"), (66, "TAG_SCRIPT_DATA_NUM_1")):
                    if tag in tags and int(tags[tag]) >= 1:
                        put(gs + slot * 7 + at, int(tags[tag]) - 1)
                        mapped_contracts.append(
                            {
                                "card": power["card_id"],
                                "tag": tag,
                                "float_index": gs + slot * 7 + at,
                                "transform": "visible total minus base 1",
                            }
                        )
        for offset, tag in (
            (0, "NUM_OPTIONS_PLAYED_THIS_TURN"),
            (32, "NUM_RESOURCES_SPENT_THIS_GAME"),
            (58, "NUM_CARDS_PLAYED_THIS_TURN"),
            (59, "NUM_MINIONS_PLAYED_THIS_TURN"),
        ):
            if tag in player:
                put(gs + offset, int(player[tag]))
        # RESOURCES_USED decreases on selling/coins; it is NOT gross gold spent.
        put(gs + 15, bool(mask[3]))
        for block in ("phase", "item_key", "cond"):
            known[self.fs[block]] = True
        known_cards[self.cs["goal_cards"]] = True
        report = {
            "profile": PROFILE,
            "bindings": bindings,
            "exact_training_observation": False,
            "known_float_count": int(known.sum()),
            "float_count": len(features),
            "known_card_count": int(known_cards.sum()),
            "card_count": len(cards),
            "imputed_float_indices": np.flatnonzero(~known).tolist(),
            "imputed_card_indices": np.flatnonzero(~known_cards).tolist(),
            "imputation": "zero for explicitly listed unknowns; frozen model has no missingness input",
            "card_aliases": dict(self.aliases),
            "client_build": view.get("build"),
            "catalog_build": catalog_build,
            "mapped_power_contracts": mapped_contracts,
            "warnings": [
                "冻结模型跨客户端版本迁移，建议强度尚未验证",
                "对手历史、附加效果及部分计数未对齐；未知值以零占位，并在输入记录逐项标注",
            ],
        }
        if client_build is None or catalog_build is None:
            report["warnings"].append("客户端或卡表版本未知，无法核对二者是否一致")
        return ObservationV2(features, cards, mask), report

    def complete_advice(self, policy, view, report, action):
        """Resolve legal subchoices without sending or assuming a client action.

        These extra evaluations use the visible PRE-action state. They are
        explicitly transfer approximations, never a simulated post-battlecry
        result. The selected entities remain bound to the original option tree.
        """
        binding = copy.deepcopy(report["bindings"][str(action)])
        pieces = [binding["text"]]
        decisions = []
        source = binding.get("entity")
        targets = binding.get("targets", [])
        subs = binding.get("suboptions", [])
        if subs:
            choices = [
                dict(s["entity"], model_card_id=source["card_id"], choice_index=s["index"])
                for s in subs
            ]
            pending = {**view, "choices": choices, "choice_kind": 3, "choice_source": source}
            obs, _ = self.encode(pending)
            decision = policy.act(obs)
            selected = next(s for s in subs if 30 + s["index"] == decision.action)
            binding["selected_suboption"] = selected["index"]
            pieces.append(
                "选择分支 "
                + located(dict(selected["entity"], zone="choices", choice_index=selected["index"]))
            )
            targets = [t["entity"] for t in selected["targets"] if t["error"] == "NONE"]
            decisions.append({"kind": "suboption", "action": decision.action})
        if targets:
            allowed = {t["entity_id"] for t in targets}
            order = (
                view["hand"]
                if allowed <= {e["entity_id"] for e in view["hand"]}
                else view["board"] + view["shop"]
            )
            choices = [
                dict(e, choice_index=i)
                for i, e in enumerate(e for e in order if e["entity_id"] in allowed)
            ]
            if {e["entity_id"] for e in choices} != allowed:
                raise AdapterGap("目标无法对齐冻结模型的可见槽位")
            pending = {**view, "choices": choices, "choice_kind": 4, "choice_source": source}
            obs, _ = self.encode(pending)
            decision = policy.act(obs)
            selected = choices[decision.action - 30]
            binding["selected_target"] = selected
            if not selected.get("zone"):
                for zone in ("board", "hand", "shop"):
                    for index, candidate in enumerate(view[zone]):
                        if selected["entity_id"] == candidate["entity_id"]:
                            selected = dict(selected, zone=zone, position=index + 1)
                binding["selected_target"] = selected
            pieces.append("目标：" + located(selected))
            decisions.append({"kind": "target", "action": decision.action})
        binding["text"] = "；".join(pieces)
        binding["subdecisions"] = decisions
        if decisions:
            binding["selection_caveat"] = "分支/目标基于操作前可见状态，未预测战吼等执行后的变化"
        return binding
