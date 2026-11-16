import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch", reason="optional training dependencies: install requirements-omnibob-mps.lock")

from omnibob import campaign, panel, run_control  # noqa: E402
from omnibob.compat import ROOT  # noqa: E402


@pytest.fixture
def directory(tmp_path):
    config = tmp_path/"config.json"
    config.write_text(json.dumps(dict(seed=430000, n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=2)))
    campaign.init(tmp_path/"campaign", 5, config)
    return tmp_path/"campaign"


def test_levels_write_the_control_file_and_are_read_back(directory):
    assert panel.configured_threads(directory) == 2
    assert panel.level_of(run_control.read(directory/"control.json"), 2) is None  # 2 threads: no such level
    for key, _, _, mode, threads in panel.LEVELS:
        stored = panel.apply_level(directory, key)
        assert stored["mode"] == mode and (threads is None or stored["threads"] == threads)
        assert panel.level_of(run_control.read(directory/"control.json"), 2) == key
    assert run_control.read(directory/"control.json")["threads"] == 4  # low and pause keep the normal setting
    assert panel.level_of({**run_control.DEFAULTS, "mode": "stop"}, 8) is None
    assert panel.level_of(dict(run_control.DEFAULTS), 8) == "high"
    events = [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]
    assert [e["mode"] for e in events if e["event"] == "mode"] == ["normal", "normal", "normal", "low", "pause"]


def test_status_lines_cover_idle_paused_and_done(directory):
    lines = panel.describe(campaign.status(directory))
    assert lines[0] == "状态：未运行（按配置 线程）" and "0 / 5" in lines[1] and "- 秒" in lines[2]
    panel.apply_level(directory, "low")
    assert "1 线程，35% 时间在算" in panel.describe(campaign.status(directory))[0]
    status = campaign.status(directory)
    assert panel.describe({**status, "running": True, "mode": "pause"})[0].startswith("状态：已暂停")
    assert panel.describe({**status, "running": True, "mode": "normal", "seconds_per_update": 26.2,
                           "remaining_days_at_recent_speed": 5.61})[2] == "最近每次更新 26 秒，还需约 5.6 天"
    assert panel.describe({**status, "done": True})[0].startswith("状态：已完成")


def test_panel_starts_without_torch_or_a_window():
    code = "import sys, omnibob.panel; print('torch' in sys.modules, 'tkinter' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.stdout.split() == ["False", "False"], result.stderr
