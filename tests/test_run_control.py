import json

import pytest

from omnibob import run_control
from omnibob.run_control import RunControl

torch = pytest.importorskip("torch", reason="optional training dependencies: install requirements-omnibob-mps.lock")

from omnibob.train import TrainConfig, load_checkpoint, train  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def control(path, clock, threads=4, **kwargs):
    seen = []
    c = RunControl(path, threads, poll_seconds=2.0, quantum_seconds=.5, clock=clock,
                   sleep=kwargs.pop("sleep", clock.sleep), set_threads=seen.append, **kwargs)
    return c, seen


def test_settings_are_validated_and_written_atomically(tmp_path):
    path = tmp_path/"control.json"
    assert run_control.read(path) == run_control.DEFAULTS
    stored = run_control.write(path, {"mode": "low", "low_duty": .5})
    assert stored["mode"] == "low" and json.loads(path.read_text())["low_duty"] == .5
    assert not list(tmp_path.glob("*.tmp"))
    for bad in ({"mode": "fast"}, {"threads": 0}, {"low_duty": 0}, {"low_duty": 2}, {"speed": 1}, []):
        with pytest.raises(ValueError):
            run_control.validate(bad)


def test_missing_file_runs_normally_and_invalid_edits_keep_last_valid_mode(tmp_path):
    clock, path = Clock(), tmp_path/"control.json"
    c, seen = control(path, clock)
    assert c.mode == "normal" and seen == [4]
    run_control.write(path, {"mode": "low", "low_threads": 1})
    assert c.refresh() == "normal"  # within the poll interval
    clock.now += 2
    assert c.refresh() == "low" and seen == [4, 1]
    path.write_text('{"mode": "sto')  # half-written edit
    clock.now += 2
    assert c.refresh() == "low" and c.error
    run_control.write(path, {"mode": "normal", "threads": 6})
    clock.now += 2
    assert c.refresh() == "normal" and seen == [4, 1, 6] and c.error is None


def test_low_mode_sleeps_for_the_duty_cycle(tmp_path):
    clock, path = Clock(), tmp_path/"control.json"
    run_control.write(path, {"mode": "low", "low_duty": .25})
    c, _ = control(path, clock)
    clock.now += .2
    assert c.throttle() == 0  # below the quantum
    clock.now += .8
    assert c.throttle() == pytest.approx(3.0)  # 1 s of work at 25 % duty
    assert c.throttled_seconds == pytest.approx(3.0)
    run_control.write(path, {"mode": "normal"})
    clock.now += 2
    c.refresh()
    clock.now += 5
    assert c.throttle() == 0


def test_pause_saves_once_and_waits_until_resumed(tmp_path):
    clock, path = Clock(), tmp_path/"control.json"
    run_control.write(path, {"mode": "pause"})
    saves, checks = [], []

    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 6:
            run_control.write(path, {"mode": "normal"})

    c, _ = control(path, clock, sleep=sleep)
    assert c.wait_while_paused(lambda: saves.append(1), lambda: checks.append(1))
    assert saves == [1] and len(checks) == 3 and c.mode == "normal"
    assert c.paused_seconds == pytest.approx(6.0)
    assert not c.wait_while_paused(lambda: saves.append(1), lambda: None)


TINY = dict(seed=410000, n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=1)


def weights(path):
    return load_checkpoint(path)["model"]


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_control_file_in_normal_and_low_mode_does_not_change_training(tmp_path):
    plain = train(tmp_path/"plain", TrainConfig(**TINY), updates=2, max_seconds=60)
    run_control.write(tmp_path/"normal.json", {"mode": "normal"})
    normal = train(tmp_path/"normal", TrainConfig(**TINY), updates=2, max_seconds=60,
                   control=tmp_path/"normal.json")
    clock = Clock()
    run_control.write(tmp_path/"low.json", {"mode": "low", "low_threads": 1, "low_duty": .5})
    low = RunControl(tmp_path/"low.json", 1, quantum_seconds=0, sleep=clock.sleep)
    throttled = train(tmp_path/"low", TrainConfig(**TINY), updates=2, max_seconds=60, control=low)
    assert plain["status"] == normal["status"] == throttled["status"] == "updates_completed"
    assert_same(weights(tmp_path/"plain/latest.pt"), weights(tmp_path/"normal/latest.pt"))
    assert_same(weights(tmp_path/"plain/latest.pt"), weights(tmp_path/"low/latest.pt"))
    rows = [json.loads(line) for line in (tmp_path/"low/metrics.jsonl").read_text().splitlines()]
    assert all(r["control_mode"] == "low" and r["threads"] == 1 for r in rows)
    assert low.throttled_seconds > 0 and sum(r["throttled_seconds"] for r in rows) == pytest.approx(low.throttled_seconds)
    assert "control_mode" not in (tmp_path/"plain/metrics.jsonl").read_text()


def test_stop_mode_saves_a_resumable_checkpoint(tmp_path):
    path = tmp_path/"control.json"
    run_control.write(path, {"mode": "stop"})
    stopped = train(tmp_path/"stopped", TrainConfig(**TINY), updates=3, max_seconds=60, control=path)
    assert stopped["status"] == "control stop" and stopped["updates"] == 0 and stopped["error"] is None
    run_control.write(path, {"mode": "normal"})
    resumed = train(tmp_path/"resumed", updates=3, resume=tmp_path/"stopped/latest.pt", max_seconds=60, control=path)
    continuous = train(tmp_path/"continuous", TrainConfig(**TINY), updates=3, max_seconds=60)
    assert resumed["updates"] == continuous["updates"] == 3
    assert_same(weights(tmp_path/"resumed/latest.pt"), weights(tmp_path/"continuous/latest.pt"))


def test_pause_checkpoints_then_continues(tmp_path):
    path = tmp_path/"control.json"
    run_control.write(path, {"mode": "pause"})
    polls = []

    def sleep(seconds):
        polls.append(seconds)
        assert (tmp_path/"paused/latest.pt").exists()  # saved before waiting
        run_control.write(path, {"mode": "normal"})

    paused = RunControl(path, 1, poll_seconds=.01, sleep=sleep)
    summary = train(tmp_path/"paused", TrainConfig(**TINY), updates=2, max_seconds=60, control=paused)
    assert summary["status"] == "updates_completed" and polls
    continuous = train(tmp_path/"continuous", TrainConfig(**TINY), updates=2, max_seconds=60)
    assert continuous["updates"] == 2
    assert_same(weights(tmp_path/"paused/latest.pt"), weights(tmp_path/"continuous/latest.pt"))


def test_latest_and_archive_intervals_are_separate(tmp_path):
    train(tmp_path/"run", TrainConfig(**TINY), updates=4, max_seconds=60, checkpoint_every=1, archive_every=2)
    assert sorted(p.name for p in (tmp_path/"run").glob("update-*.pt")) == ["update-000002.pt", "update-000004.pt"]
    manifest = json.loads((tmp_path/"run/manifest.json").read_text())
    assert manifest["checkpointing"] == {"latest_every_updates": 1, "archive_every_updates": 2}
