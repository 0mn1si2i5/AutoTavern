"""Project only rendered local zones into the advisor's public state.

The option tree supplies availability, never the authoritative zone or position.
Server-only offers are withheld until the recorded presentation event.
"""

from __future__ import annotations

from tools.real_client.offline_replay.state import TAGS


def metadata(catalog, card_id):
    raw = catalog.get(card_id) or {}
    types = {
        "minion": "MINION",
        "hero": "HERO",
        "hero_power": "HERO_POWER",
        "spell": "SPELL",
        "trinket": "BATTLEGROUND_TRINKET",
    }
    return {
        **raw,
        "type": raw.get("type", types.get(raw.get("kind"))),
        "name": raw.get("name_zh") or raw.get("name") or card_id,
        "techLevel": raw.get("techLevel", raw.get("tier")),
        "attack": raw.get("attack", raw.get("atk")),
        "health": raw.get("health", raw.get("hp")),
        "mechanics": raw.get("mechanics", raw.get("keywords", [])),
    }


def public_view(replay, catalog):
    """Return a JSON-safe prefix view; never infer current zones from options."""
    projected = replay.display()
    own, shop = replay.own, replay.shop
    state = replay.pt
    game = state.game
    player = state.entities.get(state.players.get(own), {})
    warnings = []

    def entity(eid, fallback_card="", *, choice=False):
        tags = state.entities.get(eid, {})
        # Presented choices have a public identity even when the render stream
        # does not repeat SHOW_ENTITY. No server entity tags are copied here.
        cid = tags.get("card") or fallback_card
        card = metadata(catalog, cid)
        visible = {k: v for k, v in tags.items() if k in TAGS}
        zero_default_tags = []
        if tags.get("CARDTYPE", card["type"]) == "MINION" and eid in state.complete_entities:
            for tag in ("ATK", "HEALTH"):
                if tag not in visible:
                    visible[tag] = "0"
                    zero_default_tags.append(tag)
        if choice:
            zone = "choices"
        elif tags.get("CONTROLLER") == str(own):
            zone = (
                "hand"
                if tags.get("ZONE") == "HAND"
                else {"MINION": "board", "HERO": "hero", "HERO_POWER": "powers"}.get(
                    tags.get("CARDTYPE", card["type"]), "buttons"
                )
            )
        else:
            zone = "shop"
        summary = None
        if cid == "BG21_HERO_000p":
            values = [tags.get("TAG_SCRIPT_DATA_NUM_" + str(i)) for i in (1, 2, 3)]
            if all(v is not None for v in values):
                attack, health, targets = map(int, values)
                summary = f"随机 {targets} 个友方随从获得 +{attack}/+{health}"
        return {
            "entity_id": eid,
            "card_id": cid,
            "name": card["name"],
            "tags": visible,
            "position": int(tags.get("ZONE_POSITION", 0)),
            "zone": zone,
            "card_type": tags.get("CARDTYPE", card["type"]),
            "metadata_known": bool(catalog.get(cid)),
            "snapshot_complete": eid in state.complete_entities,
            "zero_default_tags": zero_default_tags,
            "presented_choice": choice,
            "power_summary": summary,
        }

    zones = {
        z: [entity(e["id"], e["card"]) for e in projected[z]]
        for z in ("board", "hand", "shop", "powers", "hero")
    }
    # Choices are not ordinary playable cards in hand.
    offer_ids = {e["id"] for e in replay.offered(replay.active)}
    zones["hand"] = [e for e in zones["hand"] if e["entity_id"] not in offer_ids]
    choices = [
        dict(
            entity(e["id"], e["card"], choice=True),
            choice_index=e["index"],
            locked=e.get("locked", False),
        )
        for e in projected["offered"]
    ]
    active = replay.choices.get(replay.active, {})
    choice_kind = (
        1
        if active.get("kind") == "MULLIGAN"
        else 6
        if choices and all(e["card_type"] == "BATTLEGROUND_TRINKET" for e in choices)
        else 5
        if choices and all(e["card_type"] == "HERO_POWER" for e in choices)
        else 2
    )
    public_ids = {e["entity_id"] for es in zones.values() for e in es}
    buttons = []
    if own is not None:
        for eid, tags in state.entities.items():
            if tags.get("CONTROLLER") != str(own) or tags.get("ZONE") != "PLAY":
                continue
            cid = tags.get("card", "")
            if any(x in cid for x in ("Button", "DragBuy", "DragSell")):
                buttons.append(entity(eid))
                public_ids.add(eid)
    options = []
    if replay.tree:
        for node in replay.tree.options.values():
            eid = node["entity"].get("id")
            if eid not in public_ids:
                continue

            def targets(n):
                return [
                    {"error": t["error"], "entity": entity(t["entity"]["id"])}
                    for t in n["targets"]
                    if t["entity"].get("id") in public_ids
                ]

            options.append(
                {
                    "index": node["index"],
                    "type": node["type"],
                    "entity": entity(eid),
                    "error": node["error"],
                    "targets": targets(node),
                    "subs": [
                        {
                            "index": n["index"],
                            "error": n["error"],
                            "entity": {
                                "entity_id": n["entity"].get("id"),
                                "card_id": n["entity"].get("card", ""),
                                "name": metadata(catalog, n["entity"].get("card", ""))["name"],
                                "tags": {},
                                "card_type": "SPELL",
                                "zone": "choices",
                                "choice_index": n["index"],
                            },
                            "targets": targets(n),
                        }
                        for n in node["subs"]
                    ],
                }
            )
    completed = game.get("STATE") == "COMPLETE" or game.get("STEP") == "FINAL_GAMEOVER"
    turn = int(game.get("TURN", 0))
    phase = (
        "finished"
        if completed
        else (
            "choice"
            if choices
            else "waiting_choice"
            if replay.active is not None
            else "recruit"
            if game.get("STEP") == "MAIN_ACTION" and game.get("BACON_IN_COMBAT_PHASE") != "1"
            else "combat"
            if game.get("BACON_IN_COMBAT_PHASE") == "1"
            else "waiting"
        )
    )
    gold = None
    if "RESOURCES" in player:
        gold = (
            int(player["RESOURCES"])
            - int(player.get("RESOURCES_USED", 0))
            + int(player.get("TEMP_RESOURCES", 0))
        )
    aligned = game.get("TURN") == replay.gs.game.get("TURN") and game.get(
        "STEP"
    ) == replay.gs.game.get("STEP")
    if not aligned and phase == "recruit":
        warnings.append("服务器与画面仍在同步，请稍后刷新")
    ready = bool(choices) or bool(
        phase == "recruit" and aligned and replay.tree and replay.tree.complete
    )
    # An old option list can survive SendOption until a new one arrives. Never
    # present its availability as the current board after an animation.
    if (
        not choices
        and replay.last_option
        and replay.tree
        and replay.last_option.get("options_offset") == replay.tree.offset
    ):
        ready = False
        warnings.append("操作已发送，等待新的可用操作列表")
    if phase == "waiting_choice":
        warnings.append("选择尚未确认展示，未读取服务器提前给出的选项")
    if phase == "recruit" and gold is None:
        ready = False
        warnings.append("当前金币未读到")
    hero = next(iter(zones["hero"]), None)
    tier_candidates = {
        source: int(tags["PLAYER_TECH_LEVEL"])
        for source, tags in (("hero", (hero or {}).get("tags", {})), ("player", player))
        if "PLAYER_TECH_LEVEL" in tags
    }
    tier = next(iter(tier_candidates.values()), None)
    if len(set(tier_candidates.values())) > 1:
        tier = None
        ready = False
        warnings.append("英雄与玩家记录的酒馆等级冲突，请等待同步")
    elif tier is not None and not 1 <= tier <= 7:
        tier = None
        ready = False
        warnings.append("酒馆等级超出已支持范围")
    elif tier is None and phase in ("recruit", "choice") and choice_kind != 1:
        ready = False
        warnings.append("当前酒馆等级未读到")
    return {
        **zones,
        "hero": hero,
        "choices": choices,
        "choice_kind": choice_kind,
        "active_choice": replay.active,
        "buttons": buttons,
        "player": {k: v for k, v in player.items() if k in TAGS},
        "game": {k: v for k, v in game.items() if k in TAGS},
        "gold": gold,
        "tier": tier,
        "tier_sources": tier_candidates,
        "round": (turn + 1) // 2,
        "game_complete": completed,
        "build": replay.build,
        "game_key": f"epoch-{replay.epoch}",
        "phase": phase,
        "ready": ready,
        "warnings": warnings,
        "local_controller": own,
        "shop_controller": shop,
        "last_options": {
            "id": replay.tree.number if replay.tree else None,
            "offset": replay.tree.offset if replay.tree else None,
            "complete": bool(replay.tree and replay.tree.complete),
            "options": options,
        },
    }
