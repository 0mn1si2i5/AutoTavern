"""A small window that sets how hard a training campaign works.

    pythonw -m omnibob.panel DIR

It only writes the campaign's control file (the same as ``omnibob.campaign mode``) and reads its
status; closing the window changes nothing. Thread counts and speeds in the labels were measured
on the 8-core machine the long run uses (docs/OMNIBOB_WINDOWS_TRAINING.md).
"""
import argparse
import os
from pathlib import Path
import subprocess

from . import campaign
from .compat import ROOT

# key, button text, what it costs, control mode, PyTorch threads in normal mode
LEVELS = (
    ("full", "全速", "12 线程，每次更新约 23 秒", "normal", 12),
    ("high", "高", "8 线程，约 26 秒，占 4 到 5 个核", "normal", 8),
    ("medium", "中", "4 线程，约 35 秒，占 2 到 3 个核", "normal", 4),
    ("low", "低", "1 线程，算一会儿歇一会儿，占不到半个核", "low", None),
    ("pause", "暂停", "保存后停住，不占 CPU", "pause", None),
)


def level_of(settings, configured_threads):
    """The LEVELS key matching the control settings, or None for a custom or stopped state."""
    mode = settings["mode"]
    if mode in ("low", "pause"):
        return mode
    if mode == "normal":
        threads = settings["threads"] or configured_threads
        return next((key for key, _, _, level_mode, n in LEVELS if level_mode == "normal" and n == threads), None)
    return None


def apply_level(directory, key):
    _, _, _, mode, threads = next(level for level in LEVELS if level[0] == key)
    return campaign.set_mode(directory, mode, threads=threads)


def configured_threads(directory):
    import json
    config = json.loads((Path(directory)/"train_config.json").read_text(encoding="utf-8"))
    return config.get("cpu_threads", 4)


def describe(status):
    """Status lines for the window."""
    def num(value, pattern):
        return "-" if value is None else pattern.format(value)
    if status["done"]:
        state = "已完成"
    elif not status["running"]:
        state = "失败，见日志" if status["last_segment_error"] else "未运行"
    else:
        state = {"pause": "已暂停", "stop": "正在停止"}.get(status["mode"], "运行中")
    control = status["control"]
    work = {"normal": f"{control['threads'] or '按配置'} 线程", "pause": "暂停", "stop": "停止",
            "low": f"{control['low_threads']} 线程，{round(100*control['low_duty'])}% 时间在算"}[status["mode"]]
    return [
        f"状态：{state}（{work}）",
        f"进度：{status['updates']} / {status['updates_total']}（{100*status['updates']/status['updates_total']:.1f}%）",
        f"最近每次更新 {num(status['seconds_per_update'], '{:.0f}')} 秒，"
        f"还需约 {num(status['remaining_days_at_recent_speed'], '{:.1f}')} 天",
        f"检查点：{num(status['latest_checkpoint_age_minutes'], '{:.0f}')} 分钟前",
    ]


def start_training(directory):
    """Open the campaign runner in its own console window (Windows)."""
    script = ROOT/"tools/windows/train-run.cmd"
    env = {**os.environ, "OMNIBOB_RUN": str(Path(directory).resolve())}
    return subprocess.Popen(["cmd.exe", "/k", str(script)], cwd=ROOT, env=env,
                            creationflags=subprocess.CREATE_NEW_CONSOLE)


def build(root, directory):
    """Create the widgets in ``root``; returns the function that reloads the status."""
    import tkinter as tk
    from tkinter import ttk

    default_threads = configured_threads(directory)
    root.title("OmniBob 训练功耗")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=14)
    frame.grid()
    info = tk.StringVar()
    ttk.Label(frame, textvariable=info, justify="left").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
    chosen = tk.StringVar()

    def choose():
        try:
            apply_level(directory, chosen.get())
        except (OSError, ValueError) as exc:
            info.set(f"没有改成：{exc}")
        else:
            refresh()

    for row, (key, name, cost, _, _) in enumerate(LEVELS, start=1):
        ttk.Radiobutton(frame, text=name, value=key, variable=chosen, command=choose, width=6).grid(
            row=row, column=0, sticky="w", pady=3)
        ttk.Label(frame, text=cost, foreground="#666666").grid(row=row, column=1, sticky="w", padx=(8, 0))

    def launch():
        start.state(["disabled"])  # until the next status shows whether it is running
        start_training(directory)

    start = ttk.Button(frame, text="启动 / 续跑训练", command=launch)
    start.grid(row=len(LEVELS)+1, column=0, columnspan=2, sticky="we", pady=(12, 0))
    on_top = tk.BooleanVar(value=False)
    ttk.Checkbutton(frame, text="窗口置顶", variable=on_top,
                    command=lambda: root.attributes("-topmost", on_top.get())).grid(
        row=len(LEVELS)+2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def refresh():
        try:
            status = campaign.status(directory)
            info.set("\n".join(describe(status)))
            chosen.set(level_of(status["control"], default_threads) or "")
            can_start = os.name == "nt" and not status["running"] and not status["done"]
            start.state(["!disabled"] if can_start else ["disabled"])
        except (OSError, ValueError, KeyError) as exc:
            info.set(f"读不到训练状态：{exc}")

    refresh()
    return refresh


def main(argv=None):
    import tkinter as tk

    p = argparse.ArgumentParser(prog="python -m omnibob.panel", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    args = p.parse_args(argv)
    directory = args.directory.resolve()
    root = tk.Tk()
    try:
        campaign.load(directory)
        refresh = build(root, directory)
    except (OSError, ValueError) as exc:
        # pythonw has no console to print to.
        from tkinter import messagebox
        root.withdraw()
        messagebox.showerror("OmniBob 训练功耗", f"打不开训练目录 {directory}：\n{exc}")
        return 1

    def tick():
        refresh()
        root.after(4000, tick)

    root.after(4000, tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
