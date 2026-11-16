from tools.real_client.offline_replay.stream import LogStream
from tools.real_client.offline_replay.state import ZoneState


def test_byte_boundaries_and_midnight():
    raw = (
        "D 23:59:59.9000000 GameState.DebugPrintPower() - CREATE_GAME\n"
        "D 00:00:00.1000000 GameState.DebugPrintPower() - 中文\n"
    ).encode()

    def events(chunks):
        parser = LogStream()
        return [event for chunk in chunks for event in parser.feed(chunk)]

    assert events([raw]) == events(bytes([byte]) for byte in raw)
    first, second = events([raw])
    assert second.seconds > first.seconds
    assert second.end == len(raw)


def test_annotation_does_not_override_zone():
    state = ZoneState()
    state.apply("FULL_ENTITY - Creating ID=42 CardID=A")
    state.apply("    tag=ZONE value=HAND")
    state.apply("    tag=CONTROLLER value=4")
    state.apply(
        "TAG_CHANGE Entity=[entityName=X id=42 zone=PLAY zonePos=1 cardId=A player=12] tag=ZONE_POSITION value=2"
    )
    assert state.entities[42]["ZONE"] == "HAND"
    assert state.entities[42]["CONTROLLER"] == "4"


def test_prefix_does_not_publish_partial_tree_or_future_state(tmp_path):
    from tools.real_client.offline_replay.archive import prefixes

    raw = (
        "D 15:00:00.0000000 GameState.DebugPrintPower() - CREATE_GAME\n"
        "D 15:00:00.0000000 GameState.DebugPrintOptions() - id=1\n"
        "D 15:00:00.0000000 GameState.DebugPrintOptions() -   option 1 type=POWER mainEntity= error=NONE errorParam=\n"
    ).encode()
    ending = b"D 15:00:01.0000000 GameState.DebugPrintPower() - BLOCK_END\n"
    p = tmp_path / "Power.log"
    p.write_bytes(raw + ending)
    a, b, c = [v for _, v in prefixes(p, [len(raw) - 2, len(raw), len(raw + ending)], {})]
    assert a["server_options"] == b["server_options"] == []
    assert c["options_complete"] and len(c["server_options"]) == 1


def test_private_zone_and_combat_entities_do_not_appear_as_shop():
    z = ZoneState()
    z.entities = {
        1: {"card": "hidden", "CONTROLLER": "8", "ZONE": "HAND", "ZONE_POSITION": "1"},
        2: {
            "card": "combatant",
            "CONTROLLER": "8",
            "ZONE": "PLAY",
            "ZONE_POSITION": "1",
            "CARDTYPE": "MINION",
        },
    }
    z.game["BACON_IN_COMBAT_PHASE"] = "1"
    assert "hidden" not in str(z.zones(4, 8))
    assert not z.zones(4, 8)["shop"]
