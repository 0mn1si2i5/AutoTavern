import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import taverngym as tg

from tools.real_client.advisor.encoding import AdapterGap, Encoder
from tools.real_client.advisor.state import public_view
from tools.real_client.offline_replay.__main__ import card_metadata
from tools.real_client.offline_replay.ledger import Replay
from tools.real_client.offline_replay.stream import LogStream


def unit(eid=10, card="BG25_354", **tags):
    return {
        "entity_id": eid,
        "card_id": card,
        "name": card,
        "card_type": "MINION",
        "tags": {"ATK": "11", "HEALTH": "17", "DAMAGE": "2", "TECH_LEVEL": "5", **tags},
        "position": 1,
    }


def view():
    return {
        "ready": True,
        "round": 4,
        "game_complete": False,
        "gold": 7,
        "tier": 3,
        "board": [unit()],
        "hand": [],
        "shop": [],
        "powers": [],
        "hero": None,
        "choices": [],
        "buttons": [],
        "player": {"PLAYER_TECH_LEVEL": "3"},
        "last_options": {"options": []},
    }


def test_prefix_uses_rendered_entities_not_stale_options_and_hides_server_offers():
    replay = Replay(card_metadata())
    replay.own, replay.shop = 4, 12
    for state in (replay.gs, replay.pt):
        state.game.update(STEP="MAIN_ACTION", TURN="7")
        state.entities[10] = {
            "card": "BG25_354",
            "CONTROLLER": "4",
            "ZONE": "PLAY",
            "ZONE_POSITION": "1",
            "CARDTYPE": "MINION",
            "ATK": "11",
            "HEALTH": "17",
        }
        state.entities[90] = {
            "card": "SECRET",
            "CONTROLLER": "8",
            "ZONE": "HAND",
            "ZONE_POSITION": "1",
        }
    replay.active = 2
    replay.choices[2] = {"kind": "GENERAL", "items": [{"id": 20, "card": "FUTURE", "index": 0}]}
    replay.choose_ui[2] = ("WAIT", 1.0, 20)
    result = public_view(replay, card_metadata())
    assert result["board"][0]["entity_id"] == 10
    assert result["board"][0]["tags"]["ATK"] == "11"
    assert result["choices"] == []
    assert "SECRET" not in json.dumps(result) and "FUTURE" not in json.dumps(result)
    replay.choose_ui[2] = ("BEGIN", 2.0, 40)
    assert public_view(replay, card_metadata())["choices"][0]["card_id"] == "FUTURE"


def test_tavern_tier_comes_from_rendered_hero_and_conflicts_block_advice():
    replay = Replay(card_metadata())
    replay.own, replay.shop = 4, 12
    for state in (replay.gs, replay.pt):
        state.game.update(STEP="MAIN_ACTION", TURN="7")
    replay.pt.entities[120] = {
        "card": "BG21_HERO_000",
        "CONTROLLER": "4",
        "ZONE": "PLAY",
        "CARDTYPE": "HERO",
        "PLAYER_TECH_LEVEL": "2",
    }
    replay.pt.players[4] = 11
    replay.pt.entities[11] = {"RESOURCES": "5"}
    result = public_view(replay, card_metadata())
    assert result["tier"] == 2
    assert result["tier_sources"] == {"hero": 2}
    result.update(ready=True)
    result["last_options"]["options"] = []
    assert Encoder(card_metadata()).encode(result)[0].features[10] == pytest.approx(2 / 6)
    replay.pt.entities[11]["PLAYER_TECH_LEVEL"] = "3"
    conflicted = public_view(replay, card_metadata())
    assert conflicted["tier"] is None and not conflicted["ready"]
    assert any("等级冲突" in warning for warning in conflicted["warnings"])


def test_core_minion_features_match_native_observe2_not_old_view_assumptions():
    game = tg.Game(1)
    for seat in range(8):
        game.step(seat, 30)
    game.debug(json.dumps({"cmd": "add_minion", "p": 0, "zone": "board", "card": "BG25_354"}))
    # Independent native state mutation provides the expected normalized encoding.
    raw = json.loads(game.state_json())
    m = raw["players"][0]["board"][0]
    m.update(atk=11, hp=15, max_hp=17)
    game = tg.Game.from_json(json.dumps(raw))
    expected = np.frombuffer(game.observe2(0)[0], dtype="<f4")
    obs, report = Encoder(card_metadata()).encode(view())
    np.testing.assert_array_equal(obs.features[68:74], expected[68:74])
    np.testing.assert_array_equal(obs.features[76:99], expected[76:99])
    assert obs.cards[19] == np.frombuffer(game.observe2(0)[1], dtype="<i4")[19]
    assert report["exact_training_observation"] is False
    assert report["imputed_float_indices"]


def test_spending_is_not_net_resource_usage_and_keyword_removal_is_kept():
    v = view()
    v["player"].update(RESOURCES_USED="2", NUM_OPTIONS_PLAYED_THIS_TURN="9")
    v["board"][0]["tags"]["TAUNT"] = "0"
    catalog = card_metadata()
    catalog["BG25_354"] = {**catalog["BG25_354"], "keywords": ["TAUNT"]}
    enc = Encoder(catalog)
    obs, report = enc.encode(v)
    assert obs.features[76] == 0
    assert obs.features[enc.fs["g2"].start] == 9
    assert enc.fs["g2"].start + 1 in report["imputed_float_indices"]


def test_hero_choice_is_presence_only_and_locked_slot_keeps_index():
    catalog = card_metadata()
    heroes = [k for k, c in catalog.items() if c["kind"] == "hero"][:3]
    v = view()
    v.update(
        choice_kind=1,
        choices=[
            {
                "entity_id": i + 30,
                "card_id": h,
                "tags": {},
                "card_type": "HERO",
                "choice_index": i,
                "locked": i == 1,
            }
            for i, h in enumerate(heroes)
        ],
    )
    obs, _ = Encoder(catalog).encode(v)
    assert np.flatnonzero(obs.action_mask).tolist() == [30, 32]
    assert obs.features[1005] == 1
    assert not obs.features[1006:1045].any()
    assert obs.features[60] == 1


def test_unknown_card_rejected_instead_of_empty_slot():
    v = view()
    v["board"][0]["card_id"] = "NEW_UNSEEN_CARD"
    with pytest.raises(AdapterGap, match="冻结模型不认识"):
        Encoder(card_metadata()).encode(v)


def test_missing_dynamic_stats_are_not_replaced_by_printed_card_stats():
    v = view()
    del v["board"][0]["tags"]["ATK"]
    with pytest.raises(AdapterGap, match="ATK 尚未读到"):
        Encoder(card_metadata()).encode(v)
    v["board"][0]["snapshot_complete"] = True
    assert Encoder(card_metadata()).encode(v)[0].features[69] == 0


def test_complete_entity_is_marked_only_after_full_snapshot_ends():
    from tools.real_client.offline_replay.state import ZoneState

    state = ZoneState()
    state.apply("FULL_ENTITY - Creating ID=10 CardID=BG25_354")
    state.apply("tag=HEALTH value=17")
    assert 10 not in state.complete_entities
    state.apply("BLOCK_END")
    assert 10 in state.complete_entities
    state.apply("TAG_CHANGE Entity=11 tag=HEALTH value=2")
    assert 11 not in state.complete_entities
    state.apply("SHOW_ENTITY - Updating Entity=10 CardID=BG25_354")
    assert 10 not in state.complete_entities


def test_catalog_build_mismatch_blocks_advice_and_unknown_build_is_reported():
    class Catalog(dict):
        meta = {"build": "255790"}

    v = view()
    v["build"] = 253216
    encoder = Encoder(Catalog(card_metadata()))
    with pytest.raises(AdapterGap, match="版本.*不一致"):
        encoder.encode(v)
    v["build"] = 255790
    assert encoder.encode(v)[1]["catalog_build"] == "255790"
    v["build"] = None
    assert any("版本未知" in x for x in encoder.encode(v)[1]["warnings"])


def test_identical_names_are_disambiguated_by_zone_slot_and_live_stats():
    v = view()
    v["board"] = [unit(10), unit(11, ATK="5", HEALTH="9", DAMAGE="0")]
    v["last_options"]["options"] = [
        {
            "index": 1,
            "type": "POWER",
            "error": "NONE",
            "entity": {"entity_id": 99, "card_id": "TB_BaconShop_DragSell", "tags": {}},
            "targets": [{"error": "NONE", "entity": e} for e in v["board"]],
            "subs": [],
        }
    ]
    _, report = Encoder(card_metadata()).encode(v)
    assert "场上第 1 位" in report["bindings"]["16"]["text"]
    assert "（11/15）" in report["bindings"]["16"]["text"]
    assert "场上第 2 位" in report["bindings"]["17"]["text"]
    assert "（5/9）" in report["bindings"]["17"]["text"]


def test_cariel_visible_total_maps_to_native_upgrade_values():
    catalog = card_metadata()
    encoder = Encoder(catalog)
    v = view()
    v["powers"] = [
        {
            "entity_id": 90,
            "card_id": "BG21_HERO_000p",
            "name": "定罪",
            "card_type": "HERO_POWER",
            "tags": {
                "COST": "0",
                "TAG_SCRIPT_DATA_NUM_1": "5",
                "TAG_SCRIPT_DATA_NUM_2": "8",
                "TAG_SCRIPT_DATA_NUM_3": "3",
            },
        }
    ]
    obs, report = encoder.encode(v)
    game = tg.Game(1)
    for seat in range(8):
        game.step(seat, 30)
    raw = json.loads(game.state_json())
    power = raw["players"][0]["powers"][0]
    power.update(card=encoder.ids["BG21_HERO_000p"] - 1, values=[2, 4, 7])
    native = np.frombuffer(tg.Game.from_json(json.dumps(raw)).observe2(0)[0], dtype="<f4")
    start = encoder.fs["g2"].start + 65
    np.testing.assert_array_equal(obs.features[start : start + 2], native[start : start + 2])
    assert obs.features[start : start + 2].tolist() == [2, 4]
    assert len(report["mapped_power_contracts"]) == 2
    # Health upgrade has no third values slot in the frozen observation contract.
    assert {x["tag"] for x in report["mapped_power_contracts"]} == {
        "TAG_SCRIPT_DATA_NUM_1",
        "TAG_SCRIPT_DATA_NUM_3",
    }


def test_full_board_activate_can_select_hand_target_without_defaulting_first():
    v = view()
    v["board"] = [unit(i) for i in range(7)]
    v["hand"] = [unit(20), unit(21)]
    v["last_options"]["options"] = [
        {
            "index": 4,
            "type": "POWER",
            "error": "NONE",
            "entity": v["board"][0],
            "targets": [{"error": "NONE", "entity": e} for e in v["hand"]],
            "subs": [],
        }
    ]
    encoder = Encoder(card_metadata())
    obs, report = encoder.encode(v)
    assert obs.action_mask[23]
    assert obs.action_mask[197]  # POWER is also the client's move source.
    assert not obs.action_mask[196]  # same-slot gesture is not a model action.
    policy = SimpleNamespace(act=lambda observation: SimpleNamespace(action=31))
    advice = encoder.complete_advice(policy, v, report, 23)
    assert advice["selected_target"]["entity_id"] == 21
    assert advice["subdecisions"] == [{"kind": "target", "action": 31}]
    assert "操作前" in advice["selection_caveat"]
    assert v["hand"] == [unit(20), unit(21)]


def test_real_demo_uses_same_prefix_for_state_and_encoder():
    root = Path(__file__).resolve().parents[1]
    log = root / "runs/raw-client-logs-20261106/Hearthstone_2026_11_05_23_31_31/Power.log"
    pointer = root / "runs/real-client-data/CURRENT.json"
    if not log.exists() or not pointer.exists():
        pytest.skip("local archived demonstration is not distributed")
    from tools.real_client.catalog import Catalog

    catalog = Catalog(json.loads(pointer.read_text())["catalog"])
    replay, stream, encoder = Replay(card_metadata()), LogStream(), Encoder(catalog)
    sent = encoded = late_nonempty = 0
    last_options_at = None
    with log.open("rb") as f:
        while chunk := f.read(65536):
            for event in stream.feed(chunk):
                if event.stream == "GameState" and event.method == "DebugPrintOptions":
                    last_options_at = event.seconds
                if (
                    event.stream == "GameState"
                    and event.method in ("SendOption", "SendChoices")
                    and event.message.startswith(("selectedOption=", "id="))
                ):
                    sent += 1
                    # The recorded elapsed silence is the replay equivalent of
                    # a stable tail. No contents from the send event enter view.
                    if (
                        replay.tree
                        and last_options_at is not None
                        and event.seconds - last_options_at >= 0.3
                    ):
                        replay.tree.complete = True
                    v = public_view(replay, catalog)
                    if v["ready"]:
                        obs, report = encoder.encode(v)
                        encoded += 1
                        assert np.count_nonzero(obs.features[68:285:31]) == len(v["board"])
                        assert obs.features[7] == pytest.approx(v["gold"] / 10)
                        assert obs.features[10] == pytest.approx(v["tier"] / 6)
                        if sent == 16:
                            assert v["tier"] == 2
                        assert report["bindings"]
                    if v["round"] >= 11 and v["phase"] == "recruit":
                        assert v["board"]
                        late_nonempty += 1
                replay.apply(event)
    catalog.close()
    assert sent == 242
    assert encoded == 235
    assert late_nonempty >= 90
