from types import SimpleNamespace
from tools.real_client.offline_replay.receipts import Receipt
from tools.real_client.offline_replay.ledger import Replay
from tools.real_client.offline_replay.stream import Event


def received():
    return Receipt(
        dict(seconds=1, binding=dict(source=17, target=21, sub=-1), mapping={"kind": "activate"}),
        2,
        4,
    )


def block(r, eid):
    r.block(
        f"BLOCK_START BlockType=PLAY Entity={eid} EffectCardId=x Target=21 SubOption=-1",
        SimpleNamespace(start=100, seconds=1.3),
    )


def increment(r, eid=2, old=4, new=5):
    r.tag((eid, "NUM_OPTIONS_PLAYED_THIS_TURN", str(old), str(new)), SimpleNamespace(start=123))


def test_strict_receipts_need_same_entity_player_turn_and_increment():
    for eid, player, old, new, same, expected in [
        (17, 2, 4, 5, True, True),
        (71, 2, 4, 5, True, False),
        (17, 3, 4, 5, True, False),
        (17, 2, 4, 0, True, False),
        (17, 2, 4, 5, False, False),
    ]:
        r = received()
        block(r, eid)
        r.same_turn = same
        increment(r, player, old, new)
        r.finish("next_options")
        assert (r.data["status"] == "confirmed") == expected
    r = received()
    increment(r)
    block(r, 17)
    r.finish("next_options")
    assert r.data["status"] == "unconfirmed"


def test_choice_without_send_is_not_a_receipt_and_incomplete_count_never_confirms():
    replay = Replay({})

    def event(method, msg):
        replay.apply(Event(0, 1, 123, "GameState", method, msg))

    ref = "[entityName=Visible id=17 zone=HAND zonePos=1 cardId=A player=1]"
    event("DebugPrintEntitiesChosen", "id=9 Player=local EntitiesCount=1")
    event("DebugPrintEntitiesChosen", "  Entities[0]=" + ref)
    assert not replay.rows
    assert replay.automatic[0]["result_ids"] == [17]
    event("SendChoices", "id=10 ChoiceType=GENERAL")
    event("SendChoices", "  m_chosenEntities[0]=" + ref)
    event("DebugPrintEntitiesChosen", "id=10 Player=local EntitiesCount=2")
    event("DebugPrintEntitiesChosen", "  Entities[0]=" + ref)
    assert replay.rows[0]["execution"]["status"] != "confirmed"
