"""python -m tools.real_client.offline_replay --archive ... --out NEW_DIR [--session ...]"""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import time

from .stream import LogStream
from .ledger import Replay
from .dispatch import join_actions


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def render(path, rows):
    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>离线对局账本</title>',
        "<style>body{font:16px/1.5 system-ui;max-width:1000px;margin:auto;padding:1em}details{border-top:1px solid #bbb;padding:.4em}summary{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>",
        "<h1>逐操作账本</h1><p>点击操作可查看实体、时序和原始字节位置。server_before 为服务器已接收状态，display_before 为已播放日志投影。</p>",
    ]
    for i, r in enumerate(rows, 1):
        action = r.get("resolved_magnetic", r["mapping"])
        names = {
            "play": "打出",
            "activate": "发动",
            "move": "移位",
            "choice": "选择",
            "choose_one": "抉择",
            "buy": "购买",
            "sell": "出售",
            "targeted_play": "指向打出",
            "dark_gift": "黑暗发现",
            "hero_power": "英雄技能",
            "refresh": "刷新",
            "upgrade": "升本",
            "freeze": "冻结",
            "shop_move": "商店内移位",
            "magnetize": "磁力",
        }
        label = f"#{i} · 局头 {r['epoch']} · 回合 {r['round']} · {names.get(action['kind'], action['kind'])} · {action['status']} · {r['execution']['status']}"
        overview = {"board": "场上", "hand": "手牌", "shop": "商店"}
        zones = "　".join(
            f"{name} {len(r['display_before'][zone])}" for zone, name in overview.items()
        )
        parts.append(
            f'<details id="row-{i}"><summary>{html.escape(label)}</summary><p>展示层 · {zones}　原始字节 {r["offset"]}</p><pre>{html.escape(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details>'
        )
    path.write_text("\n".join(parts) + "</html>")


def card_metadata():
    path = Path(__file__).resolve().parents[3] / "crates/taverngym/data/cards.json"
    cards = json.loads(path.read_text())["cards"]
    return {card["id"]: card for card in cards}


def run(log, destination, expected, dispatch=None, demo=None):
    with log.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != expected:
        raise ValueError("archive hash mismatch: " + log.parent.name)
    destination.mkdir()
    stream = LogStream()
    replay = Replay(card_metadata())
    samples = []
    with log.open("rb") as f:
        while chunk := f.read(65536):
            start = time.perf_counter()
            for ev in stream.feed(chunk):
                replay.apply(ev)
            samples.append(1000 * (time.perf_counter() - start))
    replay.finish()
    for i, row in enumerate(replay.rows, 1):
        row.update(row=i, log_sha256=expected)
    if demo:
        metadata = json.loads((demo / "SESSION.json").read_text())
        with (demo / "events.jsonl").open() as events:
            first = json.loads(next(events))
        if (
            first.get("event") == "log_source"
            and Path(first["path"]).parent.name == log.parent.name
            and metadata["mode"].startswith("human demonstration")
            and metadata["input_events"] == 0
        ):
            for row in replay.rows:
                row["input"] = "human_demonstration"
    if dispatch:
        archived = json.loads((dispatch / "steps/000001/VISIBLE_VIEW.json").read_text())
        if Path(archived["source_log"]).parent.name == log.parent.name:
            attempts = join_actions(dispatch, replay.rows)
            write_json(destination / "dispatch.json", attempts)
            for attempt in attempts:
                if "row" in attempt:
                    row = replay.rows[attempt["row"] - 1]
                    row["input"] = attempt["input_status"]
                    row["dispatch"] = attempt
    with (destination / "ledger.jsonl").open("x") as out:
        for row in replay.rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(destination / "option-blocks.json", replay.blocks)
    write_json(destination / "automatic-choices.json", replay.automatic)
    render(destination / "index.html", replay.rows)
    counts = {}
    for row in replay.rows:
        m = row.get("resolved_magnetic", row["mapping"])
        counts.setdefault(m["kind"], Counter())[m["status"]] += 1
    options = [r for r in replay.rows if r["event"] == "SendOption"]
    report = dict(
        session=log.parent.name,
        sha256=expected,
        rows=len(replay.rows),
        options=len(options),
        stale_blocks=sum(bool(b["stale"]) for b in replay.blocks),
        blocks=len(replay.blocks),
        mapping=counts,
        automatic=len(replay.automatic),
        epochs=replay.epochs,
        cooccurrence=sum(r["execution"]["cooccurrence"] for r in options),
        legacy_play_move=sum(
            r["execution"]["cooccurrence"]
            and r["execution"].get("first_effect_kind") in ("PLAY", "MOVE_MINION")
            for r in options
        ),
        strict=sum(r["execution"]["status"] == "confirmed" for r in options),
        tail_bytes=len(stream.tail),
        chunks=len(samples),
        p99_ms=sorted(samples)[int(0.99 * (len(samples) - 1))],
        replay_seconds=sum(samples) / 1000,
    )
    write_json(destination / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dispatch", type=Path, help="archived visible-fix controller session")
    parser.add_argument("--demo", type=Path, help="archived read-only human recorder session")
    parser.add_argument("--session", action="append")
    args = parser.parse_args()
    manifest = {
        path: h
        for h, path in (
            s.split(None, 1) for s in (args.archive / "SHA256SUMS").read_text().splitlines()
        )
    }
    logs = sorted(args.archive.glob("Hearthstone_*/Power.log"))
    if args.session:
        logs = [p for p in logs if p.parent.name in args.session]
        if set(args.session) != {p.parent.name for p in logs}:
            parser.error("requested session absent")
    if not logs:
        parser.error("no archived Power.log files")
    args.out.mkdir(parents=True, exist_ok=False)
    for log in logs:
        relative = str(log.relative_to(args.archive))
        report = run(log, args.out / log.parent.name, manifest[relative], args.dispatch, args.demo)
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
