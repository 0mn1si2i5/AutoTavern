"""Build a readable index of the demonstration; each link resolves to the full evidence row."""

import argparse
import html
import json
import os
from pathlib import Path

from .__main__ import card_metadata


def write_timeline(ledger, out):
    catalog = card_metadata()
    kinds = {
        "buy": "购买",
        "sell": "出售",
        "activate": "发动",
        "play": "打出",
        "targeted_play": "指向打出",
        "choice": "选择",
        "choose_one": "抉择",
        "magnetize": "磁力",
        "hero_power": "英雄技能",
        "dark_gift": "黑暗发现",
        "upgrade": "升本",
        "refresh": "刷新",
        "freeze": "冻结",
        "move": "移位",
        "shop_move": "商店内移位",
    }
    statuses = {
        "mapped": "已映射",
        "registered_excluded": "登记排除",
        "unsupported": "原生不支持",
        "unresolved": "待定",
    }
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    target = os.path.relpath(ledger.with_name("index.html"), out.parent)

    def name(card):
        return catalog.get(card, {}).get("name_zh") or card or "身份未知"

    with out.open("x") as f:
        f.write(
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>人类示范局逐操作账本</title>'
            "<style>body{font:16px/1.55 system-ui;max-width:72rem;margin:2rem auto;padding:1rem}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:.4rem;border-bottom:1px solid #bbb}thead{position:sticky;top:0;background:Canvas;color:CanvasText}small{display:block}</style>"
            "<h1>人类示范局逐操作账本</h1><p>按客户端发送顺序排列。点击序号可查看展示层前态、服务器选项、原生命令与执行回执。卡名取本地词表，缺少译名时显示卡牌 ID。单局记录仅描述实际动作。</p>"
            "<table><thead><tr><th>序号</th><th>本地时间</th><th>回合</th><th>操作</th><th>对象或落点</th><th>映射与回执</th></tr></thead><tbody>"
        )
        for row in rows:
            mapping = row.get("resolved_magnetic", row["mapping"])
            view = row["server_before"]
            b = row["binding"]
            public = {
                c["id"]: c["card"]
                for zone in ("hand", "board", "shop", "hero", "powers")
                for c in view[zone]
            }
            public.update({c["id"]: c["card"] for c in row.get("offered", [])})
            kind = mapping["kind"]
            entity = b.get("target") if kind in ("buy", "sell") else b.get("source")
            if row["event"] == "SendChoices":
                entity = next(iter(b["chosen"]), None)
            detail = name(public[entity]) if entity in public else ""
            if kind not in ("sell", "buy") and b.get("target"):
                detail += " → " + name(public.get(b["target"]))
            if b.get("position"):
                detail += f" · 第 {b['position']} 位"
            if b.get("sub", -1) >= 0:
                node = next(o for o in row["legal_options"] if o["index"] == b["option"])
                sub = next((s for s in node["subs"] if s["index"] == b["sub"]), {})
                detail += " · " + name(sub.get("card"))
            clock = int(row["seconds"]) % 86400
            hhmmss = f"{clock // 3600:02}:{clock // 60 % 60:02}:{clock % 60:02}"
            ack = "已确认" if row["execution"]["status"] == "confirmed" else "待核实"
            f.write(
                f'<tr><td><a href="{html.escape(target)}#row-{row["row"]}">{row["row"]}</a></td>'
                f"<td>{hhmmss}</td><td>{row['round']}</td><td>{html.escape(kinds.get(kind, kind))}</td>"
                f"<td>{html.escape(detail)}</td><td>{statuses[mapping['status']]} · {ack}</td></tr>"
            )
        f.write("</tbody></table></html>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    write_timeline(args.ledger, args.out)
