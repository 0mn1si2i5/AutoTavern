"""Exercise log prefixes, stale advice, HTTP controls, and refresh semantics."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from tools.real_client.advisor import runtime
from tools.real_client.advisor.__main__ import handler_for
from tools.real_client.advisor.source import LogSource, ReplaySource
from http.server import HTTPServer


def line(time, method, message):
    return f"D 01:00:{time:09.6f} GameState.{method}() - {message}\n".encode()


def test_live_partial_lines_and_log_replacement_do_not_reuse_previous_game(tmp_path):
    path = tmp_path / "Power.log"
    full = line(0, "DebugPrintPower", "CREATE_GAME")
    path.write_bytes(full[:-1])
    feed = LogSource({}, path)
    feed.poll()
    assert feed.replay.epoch == 0
    with path.open("ab") as stream:
        stream.write(b"\n")
    feed.poll()
    assert feed.replay.epoch == 1
    replacement = tmp_path / "replacement"
    replacement.write_bytes(line(2, "DebugPrintPower", "CREATE_GAME"))
    replacement.replace(path)
    feed.poll()
    assert feed.replay.epoch == 1 and feed.generation == 2
    assert feed.offset == path.stat().st_size


def test_replay_prefix_never_contains_action_or_its_future_effect(tmp_path):
    path = tmp_path / "Power.log"
    before = line(0, "DebugPrintPower", "CREATE_GAME")
    before += line(.1, "DebugPrintPower", "GameEntity EntityID=1")
    before += line(.2, "DebugPrintPower", "tag=TURN value=1")
    send = line(1, "SendOption", "selectedOption=0 selectedSubOption=-1 selectedTarget=0 selectedPosition=0")
    after = line(2, "DebugPrintPower", "TAG_CHANGE Entity=1 tag=TURN value=3")
    path.write_bytes(before + send + after)
    source = ReplaySource({}, path)
    assert source.total == 1 and source.goto(0)
    assert source.offset == len(before)
    assert source.replay.gs.game["TURN"] == "1"
    assert source.replay.rows == []
    source.close()


def test_log_disappearance_clears_state_and_reappearance_reloads(tmp_path):
    path = tmp_path / "Power.log"
    path.write_bytes(line(0, "DebugPrintPower", "CREATE_GAME"))
    feed = LogSource({}, path)
    feed.poll()
    assert feed.available and feed.replay.epoch == 1
    path.unlink()
    assert feed.poll() and not feed.available
    assert feed.replay.epoch == 0 and feed.info()["available"] is False
    path.write_bytes(line(1, "DebugPrintPower", "CREATE_GAME"))
    feed.poll()
    assert feed.available and feed.replay.epoch == 1


class Catalog(dict):
    meta = {"build": "test"}


@pytest.fixture
def advisor(tmp_path, monkeypatch):
    view = {"ready": True, "round": 1, "game_key": "test", "gold": 3, "warnings": []}
    monkeypatch.setattr(runtime, "public_view", lambda *_: dict(view))
    source = SimpleNamespace(replay=None, poll=lambda: False, info=lambda: {"path": "test", "offset": 0})
    app = runtime.Advisor(source, Catalog(), tmp_path / "session")
    obs = SimpleNamespace(features=np.zeros(4017, np.float32), cards=np.zeros(447, np.int32),
                          action_mask=np.ones(245, bool))
    binding = {"kind": "refresh", "text": "刷新酒馆"}
    app.encoder = SimpleNamespace(encode=lambda _: (obs, {"bindings": {"1": binding}}),
                                 complete_advice=lambda *args: binding)
    calls = []
    app.policy = SimpleNamespace(act=lambda _: (calls.append(1) or SimpleNamespace(action=1, reason="test")))
    yield app, view, calls
    app.close()


def test_same_snapshot_refresh_is_idempotent_and_state_change_invalidates(advisor):
    app, view, calls = advisor
    assert app.refresh()["advice"]["text"] == "刷新酒馆"
    first = dict(app.state["advice"])
    app.refresh()
    assert calls == [1] and app.state["advice"] == first
    view["gold"] = 2
    app.observe()
    assert app.state["status"] == "stale" and app.state["advice"] is None
    assert calls == [1]  # Polling must never call inference.
    app.refresh()
    assert calls == [1, 1]


def test_not_ready_cannot_keep_previous_advice_and_notes_preserve_human_input(advisor):
    app, view, calls = advisor
    app.refresh()
    view["ready"] = False
    app.refresh()
    assert app.state["status"] == "waiting" and app.state["advice"] is None
    assert len(calls) == 1
    app.note("我选择升本；模型建议刷新")
    events = [json.loads(row) for row in (app.session / "events.jsonl").read_text().splitlines()]
    assert events[-1]["kind"] == "human_note"
    assert events[-1]["text"] == "我选择升本；模型建议刷新"


def test_inference_result_is_discarded_if_visible_state_changes_mid_call(advisor):
    app, view, _ = advisor

    def act(_):
        view["gold"] = 0
        return SimpleNamespace(action=1, reason="test")

    app.policy = SimpleNamespace(act=act)
    app.refresh()
    assert app.state["status"] == "stale" and app.state["advice"] is None
    assert (app.session / "decisions/000001/DISCARDED.json").exists()


def test_proposal_guards_do_not_mutate_until_a_matching_confirmed_action(advisor):
    from omnibob.repeat_guard import RepeatGuardPolicy
    from omnibob.observation_v2 import ObservationV2
    app, _, _ = advisor
    base = SimpleNamespace(runtime_config={}, version="test", act=lambda _: SimpleNamespace(action=1, reason="test"))
    original = RepeatGuardPolicy(base, 1)
    copy = runtime.proposal_runtime(original)
    features = np.zeros(4017, np.float32)
    features[0] = .05
    obs = ObservationV2(features, np.zeros(447, np.int32), np.ones(245, bool))
    copy.act(obs)
    assert len(copy.counts) == 1 and not original.counts
    assert copy.base is original.base
    app.policy = original
    app.source.replay = SimpleNamespace(rows=[])
    app.sync_actual()
    app.pending = {"generation": 0, "row": 0, "policy": copy,
                   "binding": {"kind": "refresh", "native_action": 1}, "sequence": 1}
    app.source.replay.rows.append({"binding": {"option": 4},
                                   "mapping": {"native_action_sequence": [1]},
                                   "execution": {"status": "awaiting"}})
    app.sync_actual()
    assert app.policy is original
    app.source.replay.rows[-1]["execution"]["status"] = "confirmed"
    app.sync_actual()
    assert app.policy is copy


def test_receipt_distinguishes_shared_activate_move_option_and_magnetic_attachment():
    proposal = {"kind": "activate", "option_index": 5, "native_action": 23,
                "entity": {"entity_id": 777}}
    move = {"binding": {"option": 5, "source": 777, "position": 2},
            "mapping": {"native_action_sequence": [197]}}
    assert runtime.Advisor.matches(proposal, move) is False
    magnetic = {"kind": "magnetize", "native_action": 126, "entity": {"entity_id": 888},
                "target": {"entity_id": 2119}}
    sent = {"binding": {"source": 888, "target": 0, "position": 1},
            "mapping": {"native_action_sequence": [46]}, "execution": {}}
    assert runtime.Advisor.matches(magnetic, sent) is None
    sent["execution"]["attached_to"] = 2119
    assert runtime.Advisor.matches(magnetic, sent) is True
    sent["execution"]["attached_to"] = 2120
    assert runtime.Advisor.matches(magnetic, sent) is False


def test_archived_state_to_real_frozen_model_recommendation(tmp_path):
    from tools.real_client.advisor.__main__ import Catalog, catalog_path
    root = Path(__file__).resolve().parents[1]
    log = root / "runs/raw-client-logs-20261106/Hearthstone_2026_11_05_23_31_31/Power.log"
    if not log.exists() or not (root / "runs/releases/current.json").exists():
        pytest.skip("Local frozen model and archive are not distributed")
    catalog = Catalog(catalog_path(None))
    source = ReplaySource(catalog, log)
    source.goto(15)
    app = runtime.Advisor(source, catalog, tmp_path / "real")
    try:
        result = app.refresh()
        assert result["status"] == "ready", result["message"]
        assert result["advice"]["text"]
        assert (app.session / "decisions/000001/INPUT.npz").exists()
        first = result["advice"]["text"]
        assert app.refresh()["advice"]["text"] == first
    finally:
        app.close()


def test_incremental_live_tail_and_replay_encode_the_identical_real_prefix(tmp_path, monkeypatch):
    from tools.real_client.advisor import source as sources
    from tools.real_client.advisor.__main__ import Catalog, catalog_path
    from tools.real_client.advisor.state import public_view
    from tools.real_client.advisor.encoding import Encoder
    root = Path(__file__).resolve().parents[1]
    log = root / "runs/raw-client-logs-20261106/Hearthstone_2026_11_05_23_31_31/Power.log"
    if not log.exists():
        pytest.skip("Archived demonstration is not distributed")
    catalog = Catalog(catalog_path(None))
    offline = ReplaySource(catalog, log)
    offline.goto(15)
    live_path = tmp_path / "Power.log"
    with log.open("rb") as handle:
        prefix = handle.read(offline.offset)
    clock = [1.0]
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock[0])
    live = LogSource(catalog, live_path)
    # A line split by filesystem writes must not become a partial observation.
    live_path.write_bytes(prefix[:-10])
    live.poll()
    with live_path.open("ab") as handle:
        handle.write(prefix[-10:])
    live.poll()
    clock[0] += .31
    live.poll()
    expected = Encoder(catalog).encode(public_view(offline.replay, catalog))[0]
    actual = Encoder(catalog).encode(public_view(live.replay, catalog))[0]
    for name in ("features", "cards", "action_mask"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert live.replay.rows[-1]["offset"] < offline.offset
    offline.close()


def test_local_api_requires_session_token_and_has_no_execution_route(advisor):
    app, _, calls = advisor
    server = HTTPServer(("127.0.0.1", 0), handler_for(app, "token"))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/refresh", data=b"{}"))
        assert error.value.code == 403 and not calls
        request = Request(base + "/api/refresh", data=b"{}", headers={"X-Omnibob-Token": "token"})
        with urlopen(request) as response:
            assert json.load(response)["status"] == "ready"
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/execute", data=b"{}", headers={"X-Omnibob-Token": "token"}))
        assert error.value.code == 404
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


def test_retired_entry_points_are_absent():
    root = Path(__file__).resolve().parents[1] / "tools/real_client"
    for name in ("autoplay.py", "autoplay.command", "model_autoplay.py", "act.py", "play_step.py",
                 "ClickOnce.swift", "GameInput.swift"):
        assert not (root / name).exists()
