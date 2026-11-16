"""Optional real-archive regression. Filenames alone do not identify the fixtures."""

import hashlib
from pathlib import Path

import pytest

from tools.real_client.offline_replay.archive import prefixes

ARCHIVE = Path(__file__).resolve().parents[1] / "runs/raw-client-logs-20261106"


def verified(session, digest):
    path = ARCHIVE / session / "Power.log"
    if not path.exists():
        pytest.skip("local, untracked historical archive absent")
    with path.open("rb") as handle:
        assert hashlib.file_digest(handle, "sha256").hexdigest() == digest
    return path


def test_visible_fix_restores_actual_own_minions_and_all_five_shop_entities():
    path = verified(
        "Hearthstone_2026_11_05_14_35_57",
        "920725ef949ac46d4eb9062cf968b9e74f863738e991235b2b042757316a1b8a",
    )
    snapshots = dict(prefixes(path, [3089821, 4704026, 4950942, 5046462], {}))
    for off, v in snapshots.items():
        assert [e["id"] for e in v["server"]["board"]] == (
            [1277, 1278] if off == 3089821 else [1858, 1859]
        )
    for off in (4950942, 5046462):
        assert [e["id"] for e in snapshots[off]["server"]["shop"]] == [2336, 2335, 2334, 2333, 2330]


def test_demo_false_empty_has_presented_warband():
    path = verified(
        "Hearthstone_2026_11_05_23_31_31",
        "0b73ccb4557c3e2317368a4e9628482d976636c5d113bc1c60e8ed5d1cd2b727",
    )
    v = next(prefixes(path, [26237772], {}))[1]
    assert v["server"]["board"] and v["display"]["board"]


def test_server_choices_do_not_enter_player_projection_during_real_wait():
    path = verified(
        "Hearthstone_2026_11_05_23_31_31",
        "0b73ccb4557c3e2317368a4e9628482d976636c5d113bc1c60e8ed5d1cd2b727",
    )
    waiting, shown = [v for _, v in prefixes(path, [966372, 1253363], {})]
    assert len(waiting["server_choice_offered"]) == 3
    assert not waiting["display"]["offered"]
    assert waiting["display"]["offer_visibility"] == "unknown_or_waiting"
    assert [e["id"] for e in shown["display"]["offered"]] == [784, 785, 786]
