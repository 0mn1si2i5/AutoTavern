import json
import os
import signal
import subprocess
import sys
import time

import pytest

torch = pytest.importorskip("torch", reason="optional training dependencies: install requirements-omnibob-mps.lock")

from omnibob import campaign, run_control  # noqa: E402
from omnibob.compat import ROOT  # noqa: E402

TINY = dict(seed=420000, n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=1)


def make(tmp_path, updates):
    config = tmp_path/"config.json"
    config.write_text(json.dumps(TINY))
    directory = tmp_path/"campaign"
    # Archive rarely: every archived checkpoint is a separate file.
    campaign.init(directory, updates, config, checkpoint_every=1, archive_every=100000, segment_hours=1)
    return directory


def cli(*args):
    return subprocess.run([sys.executable, "-m", "omnibob.campaign", *map(str, args)], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)


def start(directory, **kwargs):
    return subprocess.Popen([sys.executable, "-m", "omnibob.campaign", "run", str(directory)], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **kwargs)


def wait_for_updates(directory, count, process, timeout=120):
    metrics = directory/"segments/0001/metrics.jsonl"
    deadline = time.time()+timeout
    while time.time() < deadline:
        if metrics.exists() and len(metrics.read_text().splitlines()) >= count:
            return
        assert process.poll() is None, process.stdout.read()
        time.sleep(.05)
    pytest.fail("training did not start")


def test_stop_mode_ends_the_run_and_run_again_finishes_it(tmp_path):
    directory = make(tmp_path, 400)
    process = start(directory)
    wait_for_updates(directory, 2, process)
    assert campaign.is_running(directory)
    assert cli("mode", directory, "stop").returncode == 0
    assert process.wait(timeout=120) == 0, process.stdout.read()
    stopped = campaign.summary_of(directory/"segments/0001")
    assert stopped["status"] == "control stop" and 2 <= stopped["updates"] < 400
    finished = cli("run", directory)
    assert finished.returncode == 0, finished.stdout+finished.stderr
    assert run_control.read(directory/"control.json")["mode"] == "normal"
    assert [p.name for p in campaign.segment_dirs(directory)] == ["0001", "0002"]
    manifest = json.loads((directory/"segments/0002/manifest.json").read_text())
    assert manifest["resume"].endswith("0001/latest.pt")
    assert json.loads((directory/"DONE.json").read_text())["updates"] == 400
    status = json.loads(cli("status", directory, "--json").stdout)
    assert status["done"] and status["updates"] == 400 and not status["running"]
    assert cli("run", directory).returncode == 0 and len(campaign.segment_dirs(directory)) == 2
    events = [json.loads(line)["event"] for line in (directory/"events.jsonl").read_text().splitlines()]
    assert events.count("segment_started") == 2 and events[-1] == "done"


@pytest.mark.skipif(os.name == "nt", reason="kills a POSIX process group")
def test_a_hard_kill_resumes_from_the_last_saved_checkpoint(tmp_path):
    directory = make(tmp_path, 400)
    process = start(directory, start_new_session=True)
    wait_for_updates(directory, 3, process)
    os.killpg(process.pid, signal.SIGKILL)  # like a power cut: no summary, no final save
    process.wait(timeout=60)
    assert campaign.summary_of(directory/"segments/0001") is None
    assert not campaign.is_running(directory)
    saved = campaign.newest_checkpoint(directory)[2]["update"]
    assert saved >= 2
    finished = cli("run", directory)
    assert finished.returncode == 0, finished.stdout+finished.stderr
    assert json.loads((directory/"segments/0002/manifest.json").read_text())["resume"].endswith("0001/latest.pt")
    assert json.loads((directory/"DONE.json").read_text())["updates"] == 400


def test_a_second_runner_is_refused_while_the_lock_is_held(tmp_path):
    directory = make(tmp_path, 1)
    lock = campaign.Lock(directory/"runner.lock")
    assert lock.acquire()
    try:
        assert campaign.is_running(directory)
        refused = cli("run", directory)
        assert refused.returncode == 2 and not campaign.segment_dirs(directory)
    finally:
        lock.release()
    assert not campaign.is_running(directory)


def test_init_checks_inputs_and_mode_command_edits_the_control_file(tmp_path):
    directory = make(tmp_path, 2)
    with pytest.raises(FileExistsError):
        campaign.init(directory, 2, tmp_path/"config.json")
    bad = tmp_path/"bad.json"
    bad.write_text(json.dumps({"n_env": 2}))
    with pytest.raises(ValueError):
        campaign.init(tmp_path/"other", 2, bad)
    changed = cli("mode", directory, "low", "--low-threads", "2", "--low-duty", "0.5")
    assert changed.returncode == 0, changed.stderr
    assert run_control.read(directory/"control.json") == {"mode": "low", "threads": None, "low_threads": 2, "low_duty": .5}
    (directory/"control.json").write_text("{broken")
    assert cli("mode", directory, "pause").returncode == 0
    assert run_control.read(directory/"control.json")["mode"] == "pause"
    text = cli("status", directory).stdout
    assert "0 / 2" in text and "pause" in text
