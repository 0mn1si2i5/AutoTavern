"""Replay one bounded archive. No live file tailing, game or model imports."""

import re

from .options import OptionTree, reference, map_option, mapping
from .receipts import Receipt
from .state import ZoneState

SEND = re.compile(
    r"selectedOption=(\d+) selectedSubOption=(-?\d+) selectedTarget=(\d+) selectedPosition=(\d+)"
)
PLAYER_NAME = re.compile(r"PlayerID=(\d+), PlayerName=(.+)")
CHOICE = re.compile(r"id=(\d+) .*?ChoiceType=(\w+)")
CHOICE_ITEM = re.compile(r"(?:m_chosenEntities|Entities)\[(\d+)\]=(.*)")


class Replay:
    def __init__(self, catalog):
        self.catalog = catalog
        self.rows = []
        self.blocks = []
        self.automatic = []
        self.epochs = []
        self.epoch = 0
        self.build = None
        self.gs, self.pt = ZoneState(), ZoneState()
        self.reset()

    def reset(self):
        self.tree = None
        self.own = self.shop = None
        self.choices = {}
        self.active = self.sent_choice = self.result_choice = self.auto_result = None
        self.receipt = None
        self.last_option = None
        self.choose_ui = {}
        self.gs_step = self.pt_step = None

    def close(self, reason):
        if self.receipt:
            self.receipt.finish(reason)
            self.receipt = None

    def apply(self, ev):
        src, method, msg = ev.stream, ev.method, ev.message
        if src == "ChoiceCardMgr":
            if method == "WaitThenShowChoices" and (m := re.match(r"id=(\d+) (WAIT|BEGIN)", msg)):
                self.choose_ui[int(m[1])] = (m[2], ev.seconds, ev.start)
            return
        if src == "GameState" and self.tree and method != "DebugPrintOptions":
            self.tree.complete = True
        state = self.gs if src == "GameState" else self.pt
        if method == "DebugPrintGame":
            if m := PLAYER_NAME.match(msg):
                state.names[m[2].strip()] = int(m[1])
                if src == "GameState":
                    self.pt.names[m[2].strip()] = int(m[1])
            if msg.startswith("BuildNumber="):
                self.build = int(msg.split("=")[1])
            return
        if method == "DebugPrintPower":
            if msg.strip() == "CREATE_GAME":
                if src == "GameState":
                    self.close("CREATE_GAME")
                    self.pt = ZoneState()
                    self.gs = ZoneState()
                    self.reset()
                    self.epoch += 1
                    self.epochs.append(
                        {
                            "epoch": self.epoch,
                            "start": ev.start,
                            "first_turn": None,
                            "terminal": False,
                            "match_relation": "unresolved",
                        }
                    )
                else:
                    self.pt = ZoneState()
                    self.pt.names.update(self.gs.names)
                    self.pt_step = None
                return
            change = state.apply(msg)
            if change:
                eid, tag, _, value = change
                if eid == state.game_id:
                    if tag == "STEP" and value == "MAIN_ACTION":
                        stamp = (state.game.get("TURN"), ev.seconds, ev.start)
                        if src == "GameState":
                            self.gs_step = stamp
                        else:
                            self.pt_step = stamp
                    if src == "GameState" and self.epochs:
                        epoch = self.epochs[-1]
                        if tag == "TURN" and epoch["first_turn"] is None:
                            epoch["first_turn"] = int(value)
                        if (tag == "STATE" and value == "COMPLETE") or (
                            tag == "STEP" and value == "FINAL_GAMEOVER"
                        ):
                            epoch["terminal"] = True
                if self.receipt and src == "GameState":
                    if eid == state.game_id and (
                        tag == "TURN"
                        and int(value) != self.last_option["turn"]
                        or tag == "STEP"
                        and value == "MAIN_END"
                    ):
                        self.receipt.same_turn = False
                        self.last_option["execution"].setdefault(
                            "turn_boundary",
                            dict(
                                offset=ev.start,
                                delay_s=round(ev.seconds - self.last_option["seconds"], 5),
                            ),
                        )
                    binding = self.last_option["binding"]
                    if (
                        tag == "ZONE_POSITION"
                        and eid == binding["source"]
                        and int(value) == binding["position"]
                        and self.receipt.same_turn
                        and "bound_effect" in self.receipt.data
                        and "own_counter_increment" not in self.receipt.data
                    ):
                        self.receipt.data["source_position_offset"] = ev.start
                    self.receipt.tag(change, ev)
                    if tag in ("ATTACHED", "MODULAR", "CREATOR"):
                        enchant = state.entities[eid]
                        if (
                            enchant.get("MODULAR") == "1"
                            and enchant.get("CARDTYPE") == "ENCHANTMENT"
                            and enchant.get("CREATOR") == str(self.last_option["binding"]["source"])
                            and enchant.get("ATTACHED")
                        ):
                            self.last_option["execution"].update(
                                attached_to=int(enchant["ATTACHED"]),
                                enchantment=eid,
                                enchantment_offset=ev.start,
                            )
                    if (
                        tag == "ZONE"
                        and value == "SETASIDE"
                        and eid == self.last_option["binding"]["source"]
                    ):
                        self.last_option["execution"]["source_setaside"] = True
            if src == "GameState" and self.receipt:
                self.receipt.block(msg, ev)
            return
        if src != "GameState":
            return
        if method == "DebugPrintOptions":
            if msg.startswith("id="):
                self.close("next_options")
                self.tree = OptionTree(int(msg[3:]), ev.start)
                self.blocks.append(
                    {
                        "epoch": self.epoch,
                        "offset": ev.start,
                        "id": self.tree.number,
                        "stale": self.tree.stale,
                    }
                )
            elif self.tree:
                option = self.tree.add(msg)
                if option and (ref := option["entity"]):
                    tags = self.gs.entities.get(ref["id"], {})
                    # Determine controllers only from actual tavern buttons and Bob.
                    if "DragBuy" in ref["card"] or "Reroll_Button" in ref["card"]:
                        self.own = int(tags["CONTROLLER"]) if "CONTROLLER" in tags else None
                    if ref["card"].startswith("TB_BaconShopBob") and "CONTROLLER" in tags:
                        self.shop = int(tags["CONTROLLER"])
                    if ref["zone"] != tags.get("ZONE") or str(ref["controller"]) != tags.get(
                        "CONTROLLER"
                    ):
                        self.tree.stale.append(
                            dict(
                                entity=ref["id"],
                                offset=ev.start,
                                annotation=ref["zone"],
                                explicit=tags.get("ZONE"),
                                controller=tags.get("CONTROLLER"),
                            )
                        )
            return
        if method == "SendOption" and (m := SEND.search(msg)):
            self.send_option(ev, *map(int, m.groups()))
            return
        if method in ("DebugPrintEntityChoices", "SendChoices", "DebugPrintEntitiesChosen"):
            self.choice_event(ev)

    def offered(self, number):
        return [
            {
                **item,
                "locked": self.gs.entities.get(item["id"], {}).get("BACON_LOCKED_MULLIGAN_HERO")
                == "1",
            }
            for item in self.choices.get(number, {}).get("items", [])
        ]

    def display(self):
        view = self.pt.zones(self.own, self.shop)
        gate = self.choose_ui.get(self.active)
        offered = self.offered(self.active)
        shown = bool(gate and gate[0] == "BEGIN")
        if self.active is not None and not shown:
            pending = {e["id"] for e in offered}
            view["hand"] = [e for e in view["hand"] if e["id"] not in pending]
        view["offered"] = offered if shown else []
        view["offer_visibility"] = (
            ("presented" if shown else "unknown_or_waiting")
            if self.active is not None
            else "no_active_offer"
        )
        return view

    def snapshot(self):
        # This projection can be requested at a partial byte prefix. Never publish
        # the server's option tree as already rendered on screen.
        return {
            "epoch": self.epoch,
            "server": self.gs.zones(self.own, self.shop),
            "display": self.display(),
            "gs_step": self.gs.game.get("STEP"),
            "pt_step": self.pt.game.get("STEP"),
            "server_turn": self.gs.game.get("TURN"),
            "options_complete": bool(self.tree and self.tree.complete),
            "server_options": (
                [self.public_node(o) for o in self.tree.options.values() if o["error"] == "NONE"]
                if self.tree and self.tree.complete
                else []
            ),
            "active_choice": self.active,
            "server_choice_offered": self.offered(self.active),
            "choice_presentation": self.choose_ui.get(self.active),
        }

    def base(self, ev):
        turn = int(self.gs.game.get("TURN", 0))
        view = self.gs.zones(self.own, self.shop)
        paired = self.gs_step and self.pt_step and self.gs_step[0] == self.pt_step[0] == str(turn)
        choice_gate = self.choose_ui.get(self.active)
        ready = (
            (choice_gate is not None and choice_gate[0] == "BEGIN")
            if self.active is not None
            else (paired and self.gs.game.get("STEP") == self.pt.game.get("STEP") == "MAIN_ACTION")
        )
        return {
            "epoch": self.epoch,
            "build": self.build,
            "offset": ev.start,
            "seconds": ev.seconds,
            "turn": turn,
            "round": (turn + 1) // 2,
            "server_before": view,
            "display_before": self.display(),
            "timing": {
                "presentation_gate_observed": bool(ready),
                "gs_step": self.gs.game.get("STEP"),
                "pt_step": self.pt.game.get("STEP"),
                "pt_turn": self.pt.game.get("TURN"),
                "choice_ui": choice_gate,
                "step_lead_s": round(self.pt_step[1] - self.gs_step[1], 4) if paired else None,
            },
            "active_choice": self.active,
            "server_choice_offered": self.offered(self.active),
            "options_complete": bool(self.tree and self.tree.complete),
        }

    def send_option(self, ev, idx, sub, target, pos):
        self.close("next_send")
        row = self.base(ev)
        node = self.tree.options.get(idx) if self.tree else None
        bind = dict(
            source=node["entity"].get("id") if node else None,
            option=idx,
            sub=sub,
            target=target,
            position=pos,
        )
        row.update(
            event="SendOption",
            binding=bind,
            options_offset=self.tree.offset if self.tree else None,
            option_id=self.tree.number if self.tree else None,
        )
        row["mapping"] = (
            map_option(node, bind, row["server_before"], self.build, self.catalog)
            if node
            else mapping("unresolved", [], "unresolved", "option index missing")
        )
        # Preserve public option identities/targets, independently of 245 mapping success.
        row["legal_options"] = (
            [self.public_node(o) for o in self.tree.options.values() if o["error"] == "NONE"]
            if self.tree
            else []
        )
        row["input"] = "client_sent_origin_unattributed"
        row["passive_end"] = (
            "WAIT_TIMER"
            if self.tree
            and any(
                o["type"] == "END_TURN" and o["error"] == "INVALID"
                for o in self.tree.options.values()
            )
            else None
        )
        self.rows.append(row)
        eid = self.gs.players.get(self.own)
        self.receipt = Receipt(
            row, eid, int(self.gs.entities.get(eid, {}).get("NUM_OPTIONS_PLAYED_THIS_TURN", 0))
        )
        self.last_option = row

    def public_node(self, node):
        def public_ref(ref):
            # Only expose identities visible in own hand/board or the tavern.
            eid = ref.get("id")
            tags = self.gs.entities.get(eid, {})
            own_zone = tags.get("CONTROLLER") == str(self.own) and tags.get("ZONE") in (
                "PLAY",
                "HAND",
            )
            shop_zone = (
                self.gs.game.get("BACON_IN_COMBAT_PHASE") != "1"
                and tags.get("CONTROLLER") == str(self.shop)
                and tags.get("ZONE") == "PLAY"
            )
            return (
                {"id": eid, "card": tags.get("card", "")} if own_zone or shop_zone else {"id": eid}
            )

        def targets(n):
            return [public_ref(t["entity"]) for t in n["targets"] if t["error"] == "NONE"]

        return {
            "index": node["index"],
            "entity": public_ref(node["entity"]),
            "targets": targets(node),
            "subs": [
                {"index": s["index"], "card": s["entity"].get("card", ""), "targets": targets(s)}
                for s in node["subs"]
                if s["error"] == "NONE"
            ],
        }

    def choice_event(self, ev):
        method, msg = ev.method, ev.message.strip()
        if method == "DebugPrintEntityChoices":
            if m := CHOICE.match(msg):
                self.active = int(m[1])
                self.choices[self.active] = {
                    "id": self.active,
                    "kind": m[2],
                    "items": [],
                    "offset": ev.start,
                }
                return
            if self.active is not None and (m := CHOICE_ITEM.match(msg)):
                ref = reference(m[2])
                if ref:
                    self.choices[self.active]["items"].append(
                        {
                            "index": int(m[1]),
                            "id": ref["id"],
                            "card": ref["card"],
                            "locked": self.gs.entities.get(ref["id"], {}).get(
                                "BACON_LOCKED_MULLIGAN_HERO"
                            )
                            == "1",
                        }
                    )
                    if self.own is None and self.choices[self.active]["kind"] == "MULLIGAN":
                        self.own = ref["controller"]
            return
        if msg.startswith("id="):
            number = int(msg.split()[0][3:])
            if method == "SendChoices":
                row = self.base(ev)
                offered = self.offered(number)
                row.update(
                    event=method,
                    binding={"choice": number, "chosen": []},
                    offered=list(offered),
                    mapping=mapping("choice", [], "unresolved", "selected entity pending"),
                    execution={"status": "awaiting_EntitiesChosen"},
                    input="client_sent_origin_unattributed",
                )
                self.rows.append(row)
                self.sent_choice = row
            else:
                candidate = self.sent_choice
                self.auto_result = None
                if candidate and candidate["binding"]["choice"] == number:
                    self.result_choice = candidate
                    count = re.search(r"EntitiesCount=(\d+)", msg)
                    candidate["execution"]["declared_count"] = int(count[1]) if count else None
                    self.sent_choice = None
                else:
                    self.result_choice = None
                    self.auto_result = dict(
                        epoch=self.epoch,
                        choice=number,
                        offset=ev.start,
                        evidence="result_without_observed_send",
                        result_ids=[],
                    )
                    self.automatic.append(self.auto_result)
                if self.active == number:
                    self.active = None
            return
        if m := CHOICE_ITEM.match(msg.replace("m_chosenEntities", "Entities")):
            ref = reference(m[2])
            if not ref:
                return
            eid = ref["id"]
            if method == "SendChoices" and self.sent_choice:
                row = self.sent_choice
                row["binding"]["chosen"].append(eid)
                selected = [
                    x["index"] for x in row["offered"] if x["id"] == eid and not x["locked"]
                ]
                if len(row["binding"]["chosen"]) > 1:
                    row["mapping"] = mapping(
                        "multi_choice", [], "unsupported", "multiple entities selected"
                    )
                elif len(selected) == 1 and selected[0] < 16:
                    row["mapping"] = mapping("choice", [30 + selected[0]])
            elif method == "DebugPrintEntitiesChosen" and self.auto_result is not None:
                self.auto_result["result_ids"].append(eid)
            elif method == "DebugPrintEntitiesChosen" and self.result_choice:
                row = self.result_choice
                ids = row["execution"].setdefault("result_ids", [])
                ids.append(eid)
                row["execution"].update(
                    status="confirmed"
                    if len(ids) == row["execution"]["declared_count"]
                    and ids == row["binding"]["chosen"]
                    else "unconfirmed",
                    result_offset=ev.start,
                )

    def finish(self):
        self.close("end_of_archive")
        previous = None
        for epoch in self.epochs:
            rows = [r for r in self.rows if r["epoch"] == epoch["epoch"]]
            heroes = [r["server_before"]["hero"][0] for r in rows if r["server_before"]["hero"]]
            first = rows[0]["turn"] if rows else None
            epoch.update(
                first_action_turn=first, match_key=epoch["epoch"], match_relation="ambiguous"
            )
            if epoch["first_turn"] == 1 or (first is not None and first <= 1):
                epoch["match_relation"] = "opening"
            elif (
                previous
                and heroes
                and previous["hero"] == (heroes[0]["id"], heroes[0]["card"])
                and not previous["terminal"]
                and first is not None
                and 0 <= first - previous["last_turn"] <= 2
            ):
                epoch.update(
                    match_key=previous["key"], match_relation="reconnect_same_hero_entity_and_turn"
                )
            for row in rows:
                row["match_key"] = epoch["match_key"]
            previous = dict(
                key=epoch["match_key"],
                terminal=epoch["terminal"],
                last_turn=rows[-1]["turn"] if rows else -99,
                hero=(heroes[-1]["id"], heroes[-1]["card"]) if heroes else None,
            )
        for row in self.rows:
            effect = row["execution"]
            if (
                row["event"] == "SendOption"
                and effect["status"] == "confirmed"
                and effect.get("enchantment")
                and effect.get("attached_to")
                and effect.get("source_setaside")
            ):
                board = [e["id"] for e in row["server_before"]["board"]]
                hand = [e["id"] for e in row["server_before"]["hand"]]
                if effect["attached_to"] in board and row["binding"]["source"] in hand:
                    slot = board.index(effect["attached_to"])
                    row["resolved_magnetic"] = mapping(
                        "magnetize", [126 + 7 * hand.index(row["binding"]["source"]) + slot]
                    )
