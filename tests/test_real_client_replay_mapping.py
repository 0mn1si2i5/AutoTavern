from tools.real_client.offline_replay.options import OptionTree, map_option


def test_sub_target_not_attached_to_parent():
    tree = OptionTree(1, 0)
    tree.add("  option 2 type=POWER mainEntity= error=NONE errorParam=")
    tree.add("    subOption 0 entity= error=NONE errorParam=")
    tree.add("      target 1 entity= error=NONE errorParam=")
    node = tree.options[2]
    assert node["targets"] == []
    assert node["subs"][0]["targets"][0]["index"] == 1


def test_discard_uses_hand_order_and_explicit_eligibility():
    node = {
        "entity": {"id": 42, "card": "BG36_300"},
        "error": "NONE",
        "subs": [],
        "targets": [{"entity": {"id": i}, "error": "NONE"} for i in (17, 11)],
    }
    view = {"powers": [], "board": [{"id": 42}], "hand": [{"id": 11}, {"id": 17}], "shop": []}
    assert map_option(node, dict(target=17, position=0, sub=-1), view, 255790, {})[
        "native_action_sequence"
    ] == [23, 31]


def test_same_position_reorder_cannot_be_executed_as_a_native_move():
    node = {
        "entity": {"id": 42, "card": "A"},
        "error": "NONE",
        "type": "POWER",
        "subs": [],
        "targets": [],
    }
    view = {"powers": [], "board": [{"id": 41}, {"id": 42}], "hand": [], "shop": []}
    result = map_option(node, dict(target=0, position=2, sub=-1), view, 253216, {})
    assert result["status"] == "unsupported"
    assert result["kind"] == "move" and result["native_action_sequence"] == []


def test_incomplete_choose_one_cannot_be_mapped_to_a_plain_cast():
    node = {
        "entity": {"id": 17, "card": "A"},
        "error": "NONE",
        "targets": [],
        "subs": [{"index": 0, "error": "NONE", "targets": []}],
    }
    view = {"powers": [], "board": [], "shop": [], "hand": [{"id": 17}]}
    assert (
        map_option(node, dict(sub=-1, target=0, position=0), view, 255790, {})["status"] != "mapped"
    )
