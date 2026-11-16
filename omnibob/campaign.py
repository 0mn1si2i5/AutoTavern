"""A long training run split into resumable segments, with a lock, live control and status.

    python -m omnibob.campaign init   DIR --initialize CKPT --config CFG.json --rules RULES.json --updates N
    python -m omnibob.campaign run    DIR [--priority below_normal|idle|normal]
    python -m omnibob.campaign status DIR [--json]
    python -m omnibob.campaign mode   DIR normal|low|pause|stop [--threads N] [--low-threads N] [--low-duty F]

Each segment is one ``omnibob.train`` process writing ``segments/NNNN/``; its console output is
``segments/NNNN.log``. ``run`` always continues from the newest saved checkpoint, so after a
stop, Ctrl+C, a closed window or a reboot the same command resumes the run. ``mode`` changes a
running job within a few seconds (see run_control.py).
"""
import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from . import run_control
from .compat import ROOT

FORMAT = "omnibob-campaign-v1"
RESTARTABLE = ("time budget reached", "RSS or MPS driver memory budget reached")
PRIORITIES = ("below_normal", "idle", "normal")


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stored_path(path):
    """Repository-relative POSIX path when inside the checkout, else absolute."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve(stored):
    path = Path(stored)
    return path if path.is_absolute() else ROOT/path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    run_control.replace(temporary, path)


def event(directory, kind, **data):
    with (Path(directory)/"events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": utc(), "event": kind, **data}, ensure_ascii=False)+"\n")


def load(directory):
    meta = json.loads((Path(directory)/"campaign.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT:
        raise ValueError("not an OmniBob training campaign")
    return meta


def init(directory, updates, config, initialize=None, rules=None, device="cpu", checkpoint_every=10,
         archive_every=250, max_rss_gib=4.0, segment_hours=24.0):
    """Freeze the run's inputs into ``directory`` and check that they load together."""
    from .train import TrainConfig  # here, so that status and mode start without loading torch
    directory = Path(directory)
    if (directory/"campaign.json").exists():
        raise FileExistsError(f"{directory} already holds a campaign; use run or status")
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"{directory} is not empty")
    if updates < 1 or checkpoint_every < 1 or archive_every < 1 or max_rss_gib <= 0 or segment_hours <= 0:
        raise ValueError("positive updates, checkpoint intervals, memory and segment length required")
    overrides = json.loads(Path(config).read_text(encoding="utf-8"))
    unknown = sorted(set(overrides)-{f.name for f in fields(TrainConfig)})
    if unknown:
        raise ValueError(f"unknown TrainConfig fields: {unknown}")
    rules_value = json.loads(Path(rules).read_text(encoding="utf-8")) if rules else None
    if initialize:
        # Fail now rather than at the first segment: same checks as train(--initialize).
        from dataclasses import asdict
        from .compat import effective_config
        from .model import ModelConfig
        from .train import load_checkpoint
        initial = load_checkpoint(initialize)
        if initial.get("error"):
            raise ValueError("initial checkpoint records a failed training state")
        if initial["config"] != effective_config(rules_value):
            raise ValueError("initial checkpoint was trained under different rules")
        if initial["model_config"] != asdict(ModelConfig()):
            raise ValueError("initial checkpoint architecture differs from the default model")
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"segments").mkdir()
    shutil.copyfile(config, directory/"train_config.json")
    if rules:
        shutil.copyfile(rules, directory/"rules.json")
    meta = {"format": FORMAT, "created_utc": utc(), "updates_total": updates, "device": device,
            "checkpoint_every": checkpoint_every, "archive_every": archive_every,
            "max_rss_gib": max_rss_gib, "segment_seconds": segment_hours*3600,
            "decisions_per_update": TrainConfig(**overrides).n_envs*TrainConfig(**overrides).horizon,
            "initialize": ({"path": stored_path(initialize), "sha256": sha256(initialize)} if initialize else None),
            "train_config_sha256": sha256(directory/"train_config.json"),
            "rules_sha256": sha256(directory/"rules.json") if rules else None}
    write_json(directory/"campaign.json", meta)
    run_control.write(directory/"control.json", {})
    event(directory, "init", **{k: meta[k] for k in ("updates_total", "device", "initialize")})
    return meta


def read_control(path):
    """Control settings, or the defaults with mode kept when the file was hand-edited badly."""
    try:
        return run_control.read(path)
    except ValueError as exc:
        print(f"控制文件无效，按默认设置处理：{exc}", flush=True)
        return dict(run_control.DEFAULTS)


def segment_dirs(directory):
    return sorted(p for p in (Path(directory)/"segments").iterdir() if p.is_dir() and p.name.isdigit())


def newest_checkpoint(directory):
    """(segment, latest.pt, saved state summary) of the newest segment that saved, else None."""
    import torch
    for segment in reversed(segment_dirs(directory)):
        latest = segment/"latest.pt"
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=True)
            return segment, latest, {k: state.get(k) for k in ("update", "steps", "error", "stop_reason")}
    return None


def summary_of(segment):
    path = Path(segment)/"summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Lock:
    """An OS file lock, released by the operating system if the holder dies."""

    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def acquire(self):
        stream = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self):
        if self.stream is None:
            return
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        self.stream.close()
        self.stream = None


def is_running(directory):
    lock = Lock(Path(directory)/"runner.lock")
    if lock.acquire():
        lock.release()
        return False
    return True


def lower_priority(level):
    """Lower this process's CPU priority; segments inherit it."""
    if level == "normal":
        return
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        value = {"below_normal": 0x4000, "idle": 0x40}[level]
        if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), value):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        target = {"below_normal": 10, "idle": 19}[level]
        current = os.nice(0)
        if target > current:
            os.nice(target-current)


def keep_awake(on):
    """On Windows, keep the system from sleeping when idle while segments run; the display can
    still turn off. Sleep only suspends training, but a long run would otherwise stall overnight."""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes
    es_continuous, es_system_required = 0x80000000, 0x00000001
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]  # 0x80000000 overflows a C int
    kernel32.SetThreadExecutionState.restype = wintypes.DWORD
    kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def run(directory, priority="below_normal", python=sys.executable):
    """Run segments until the campaign is done, stopped, interrupted or fails; returns an exit code."""
    directory = Path(directory).resolve()
    meta = load(directory)
    lock = Lock(directory/"runner.lock")
    if not lock.acquire():
        print(f"另一个进程正在运行这个训练：{directory}", flush=True)
        return 2
    interrupted = []
    handlers = {}

    def on_signal(signum, frame):
        interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGBREAK", None)):
        if signum is not None:
            handlers[signum] = signal.signal(signum, on_signal)
    try:
        lower_priority(priority)
        keep_awake(True)
        control_path = directory/"control.json"
        settings = read_control(control_path)
        if settings["mode"] == "stop":
            run_control.write(control_path, {**settings, "mode": "normal"})
            event(directory, "mode", mode="normal", reason="run restarted after stop")
        stalled = 0
        while True:
            newest = newest_checkpoint(directory)
            if newest and newest[2]["error"]:
                print(f"训练失败：{newest[0].name}，{newest[2]['error']}。日志：{newest[0]}.log", flush=True)
                return 1
            done = newest[2]["update"] if newest else 0
            if done >= meta["updates_total"]:
                if not (directory/"DONE.json").exists():
                    write_json(directory/"DONE.json", {"utc": utc(), "updates": done, "steps": newest[2]["steps"],
                                                       "checkpoint": stored_path(newest[1]), "sha256": sha256(newest[1])})
                    event(directory, "done", updates=done)
                print(f"训练已完成：{done} 次更新。最终检查点：{newest[1]}", flush=True)
                return 0
            if interrupted:
                return 130
            existing = segment_dirs(directory)
            segment = directory/"segments"/f"{(int(existing[-1].name) if existing else 0)+1:04d}"
            command = [python, "-B", "-m", "omnibob.train", "--output", str(segment),
                       "--updates", str(meta["updates_total"]-done), "--device", meta["device"],
                       "--max-seconds", str(meta["segment_seconds"]), "--max-rss-gib", str(meta["max_rss_gib"]),
                       "--checkpoint-every", str(meta["checkpoint_every"]),
                       "--archive-every", str(meta["archive_every"]), "--control", str(control_path),
                       "--max-truncated-episodes", "0"]
            if newest:
                command += ["--resume", str(newest[1])]
            else:
                command += ["--config", str(directory/"train_config.json")]
                if meta["initialize"]:
                    command += ["--initialize", str(resolve(meta["initialize"]["path"]))]
                if (directory/"rules.json").exists():
                    command += ["--rules", str(directory/"rules.json")]
            event(directory, "segment_started", segment=segment.name, from_update=done,
                  parent=stored_path(newest[1]) if newest else None)
            print(f"开始第 {segment.name} 段：从第 {done} 次更新继续，目标 {meta['updates_total']}。"
                  f"日志：{segment}.log", flush=True)
            env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
            # Where the engine writes the game when its stack guard stops a runaway effect chain.
            env.setdefault("TAVERNGYM_GUARD_DIR", str(directory/"guards"))
            Path(env["TAVERNGYM_GUARD_DIR"]).mkdir(parents=True, exist_ok=True)
            flags = {"below_normal": getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
                     "idle": getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)}.get(priority, 0) if os.name == "nt" else 0
            with open(f"{segment}.log", "ab") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         creationflags=flags)
                forwarded = False
                while True:
                    try:
                        child.wait(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    if interrupted and not forwarded and os.name != "nt":
                        # A console Ctrl+C reaches the child directly; a plain kill of this
                        # process does not, so pass it on.
                        child.send_signal(signal.SIGTERM)
                        forwarded = True
            summary = summary_of(segment)
            event(directory, "segment_ended", segment=segment.name, returncode=child.returncode,
                  status=summary and summary["status"], updates=summary and summary["updates"])
            if summary is None:
                print(f"第 {segment.name} 段异常退出（返回码 {child.returncode}），没有写出 summary.json。"
                      f"日志：{segment}.log", flush=True)
                return 1
            status = summary["status"]
            print(f"第 {segment.name} 段结束：{status}，累计 {summary['updates']} 次更新。", flush=True)
            if status == "control stop":
                print("已按控制文件停止。再次运行同一命令即可续跑。", flush=True)
                return 0
            if status.startswith("signal") or interrupted:
                print("已中断。再次运行同一命令即可续跑。", flush=True)
                return 130
            if status == "updates_completed":
                continue
            if status in RESTARTABLE:
                stalled = stalled+1 if summary["updates"] <= done else 0
                if stalled >= 3:
                    print("连续 3 段没有进展，停止。请检查日志。", flush=True)
                    return 1
                continue
            print(f"训练失败：{summary.get('error')}。日志：{segment}.log", flush=True)
            return 1
    finally:
        keep_awake(False)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
        lock.release()


def metrics_rows(directory, limit=50):
    rows = []
    for segment in reversed(segment_dirs(directory)):
        path = segment/"metrics.jsonl"
        if path.exists():
            parsed = []
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    parsed.append(json.loads(line))
                except ValueError:
                    pass  # the line being written right now
            rows = parsed[-(limit-len(rows)):]+rows
        if len(rows) >= limit:
            break
    return rows[-limit:]


def status(directory):
    directory = Path(directory).resolve()
    meta = load(directory)
    settings = read_control(directory/"control.json")
    segments = segment_dirs(directory)
    rows = metrics_rows(directory)
    last_summary = summary_of(segments[-1]) if segments else None
    latest = next((s/"latest.pt" for s in reversed(segments) if (s/"latest.pt").exists()), None)
    done = rows[-1]["update"] if rows else 0
    recent = rows[-20:]
    placements = [p for r in rows for p in r.get("placements", [])][-200:]
    wall = sum(r["seconds"] for r in recent)/len(recent) if recent else None
    active = (sum(r.get("active_seconds", r["seconds"]) for r in recent)/len(recent)) if recent else None
    result = {
        "directory": str(directory), "running": is_running(directory),
        "done": (directory/"DONE.json").exists(), "mode": settings["mode"], "control": settings,
        "updates": done, "updates_total": meta["updates_total"],
        "decisions": rows[-1]["steps"] if rows else 0, "segments": len(segments),
        "last_segment_status": last_summary and last_summary["status"],
        "last_segment_error": last_summary and last_summary["error"],
        "seconds_per_update": wall, "active_seconds_per_update": active,
        "active_decisions_per_second": meta["decisions_per_update"]/active if active else None,
        "recent_mean_placement": sum(placements)/len(placements) if placements else None,
        "recent_episodes": len(placements),
        "remaining_days_at_recent_speed": ((meta["updates_total"]-done)*wall/86400) if wall else None,
        "latest_checkpoint": stored_path(latest) if latest else None,
        "latest_checkpoint_age_minutes": (time.time()-latest.stat().st_mtime)/60 if latest else None,
    }
    return result


def format_status(s):
    def num(value, pattern):
        return "-" if value is None else pattern.format(value)
    state = ("已完成" if s["done"] else "运行中" if s["running"] else
             "失败" if s["last_segment_error"] else "未运行")
    control = s["control"]
    lines = [
        f"训练目录：{s['directory']}",
        f"状态：{state}（最近一段：{s['last_segment_status'] or '-'}）",
        f"模式：{s['mode']}（正常线程 {control['threads'] or '按配置'}，低功耗线程 {control['low_threads']}，"
        f"低功耗占空比 {control['low_duty']}）",
        f"进度：{s['updates']} / {s['updates_total']} 次更新（{100*s['updates']/s['updates_total']:.1f}%），"
        f"{s['decisions']:,} 次决策，{s['segments']} 段",
        f"最近 20 次更新：每次 {num(s['seconds_per_update'], '{:.1f}')} 秒，"
        f"计算时每秒 {num(s['active_decisions_per_second'], '{:.0f}')} 次决策",
        f"最近 {s['recent_episodes']} 局平均名次：{num(s['recent_mean_placement'], '{:.3f}')}",
        f"按最近速度剩余：{num(s['remaining_days_at_recent_speed'], '{:.1f}')} 天",
        f"最新检查点：{s['latest_checkpoint'] or '-'}（{num(s['latest_checkpoint_age_minutes'], '{:.0f}')} 分钟前）",
    ]
    if s["last_segment_error"]:
        lines.append(f"错误：{s['last_segment_error']}")
    return "\n".join(lines)


def set_mode(directory, mode, threads=None, low_threads=None, low_duty=None):
    directory = Path(directory)
    load(directory)
    path = directory/"control.json"
    settings = read_control(path)
    settings["mode"] = mode
    if threads is not None:
        settings["threads"] = threads
    if low_threads is not None:
        settings["low_threads"] = low_threads
    if low_duty is not None:
        settings["low_duty"] = low_duty
    stored = run_control.write(path, settings)
    event(directory, "mode", **stored)
    return stored


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m omnibob.campaign", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("init")
    c.add_argument("directory", type=Path)
    c.add_argument("--updates", type=int, required=True)
    c.add_argument("--config", type=Path, required=True, help="full TrainConfig JSON")
    c.add_argument("--initialize", type=Path)
    c.add_argument("--rules", type=Path)
    c.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    c.add_argument("--checkpoint-every", type=int, default=10)
    c.add_argument("--archive-every", type=int, default=250)
    c.add_argument("--max-rss-gib", type=float, default=4.0)
    c.add_argument("--segment-hours", type=float, default=24.0)
    r = commands.add_parser("run")
    r.add_argument("directory", type=Path)
    r.add_argument("--priority", choices=PRIORITIES, default="below_normal")
    s = commands.add_parser("status")
    s.add_argument("directory", type=Path)
    s.add_argument("--json", action="store_true")
    m = commands.add_parser("mode")
    m.add_argument("directory", type=Path)
    m.add_argument("mode", choices=run_control.MODES)
    m.add_argument("--threads", type=int)
    m.add_argument("--low-threads", type=int)
    m.add_argument("--low-duty", type=float)
    args = p.parse_args(argv)
    if args.command == "init":
        meta = init(args.directory, args.updates, args.config, args.initialize, args.rules, args.device,
                    args.checkpoint_every, args.archive_every, args.max_rss_gib, args.segment_hours)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return 0
    if args.command == "run":
        return run(args.directory, args.priority)
    if args.command == "status":
        result = status(args.directory)
        print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else format_status(result))
        return 0
    stored = set_mode(args.directory, args.mode, args.threads, args.low_threads, args.low_duty)
    print(f"已设为 {stored['mode']}，训练进程几秒内生效。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
