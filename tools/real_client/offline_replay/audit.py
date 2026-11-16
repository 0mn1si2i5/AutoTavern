"""Audit ledger coverage against raw headers, corroborating client logs and pinned historic views."""

import argparse
from collections import Counter, defaultdict
import html
import json
from pathlib import Path
import re
from .__main__ import write_json, card_metadata
from .historical import compare, sha, read
from .fields import inventory
from .performance import benchmark


def signals(path, regex):
    hits = []
    with path.open("rb") as f:
        offset = 0
        for line in f:
            m = regex.search(line)
            if m:
                hour, minute, second = line[2:18].split(b":")
                hits.append(
                    (int(hour) * 3600 + int(minute) * 60 + float(second), offset, m.groups())
                )
            offset += len(line)
    return hits


def exceptions(rows, folder):
    missing = [
        r for r in rows if r["event"] == "SendOption" and r["execution"]["status"] != "confirmed"
    ]
    if not missing:
        return []
    rejected = signals(
        folder / "Hearthstone.log", re.compile(rb"OnOptionRejected.*rejected option id \((\d+)\)")
    )
    disconnect = signals(folder / "GameNetLogger.log", re.compile(rb"DisconnectAfterFailedPings"))
    out = []
    for r in missing:
        ex = r["execution"]
        t = r["seconds"] % 86400
        item = dict(
            row=r["row"],
            offset=r["offset"],
            session=folder.name,
            kind=r["mapping"]["kind"],
            option_id=r["option_id"],
            turn=r["turn"],
            window_end=ex["window_closed_by"],
            evidence=ex,
        )
        reject = next(
            (h for h in rejected if int(h[2][0]) == r["option_id"] and 0 <= h[0] - t < 40), None
        )
        drop = next((h for h in disconnect if 0 <= h[0] - t < 40), None)
        if reject:
            item.update(
                context="explicit_option_rejection",
                source="Hearthstone.log",
                source_offset=reject[1],
            )
        elif drop and ex["window_closed_by"] == "CREATE_GAME":
            item.update(
                context="disconnect_then_reconnect",
                source="GameNetLogger.log",
                source_offset=drop[1],
            )
        elif "turn_boundary" in ex and ex.get("first_effect_kind") == "ATTACK":
            item["context"] = "turn_ended_before_bound_effect"
        else:
            item["context"] = "unresolved"
        out.append(item)
    return out


def receipt_crosscheck(log, rows):
    """Read referenced bytes independently. Resolve private player names only in memory."""
    counter = {}
    players = {}
    names = {}
    offsets = {r["offset"]: r for r in rows if r["event"] == "SendOption"}
    active = None
    effect = None
    correct = set()
    equivalent = []
    baseline = None
    initial = None

    def entity(raw):
        if raw.isdigit():
            return int(raw)
        ref = re.search(rb" id=(\d+) zone=", raw)
        return int(ref[1]) if ref else players.get(names.get(raw))

    with log.open("rb") as f:
        pos = 0
        for line in f:
            start, pos = pos, pos + len(line)
            if b" GameState." not in line:
                continue
            text = line.partition(b"() - ")[2].rstrip()
            if text == b"CREATE_GAME":
                counter.clear()
                players.clear()
                names.clear()
                active = None
                initial = None
            elif b" GameState.DebugPrintGame()" in line and text.startswith(b"PlayerID="):
                m = re.match(rb"PlayerID=(\d+), PlayerName=(.*)", text)
                if m:
                    names[m[2]] = int(m[1])
            elif b" GameState.DebugPrintOptions() - id=" in line:
                active = None
            elif start in offsets:
                active = offsets[start]
                effect = None
                baseline = counter.get(active["execution"]["player_entity"], 0)
            elif b" GameState.DebugPrintPower()" in line:
                item = text.lstrip()
                if item.startswith(b"Player EntityID="):
                    m = re.match(rb"Player EntityID=(\d+) PlayerID=(\d+)", item)
                    initial = int(m[1])
                    players[int(m[2])] = initial
                elif item.startswith(b"tag="):
                    if initial is not None and item.startswith(
                        b"tag=NUM_OPTIONS_PLAYED_THIS_TURN value="
                    ):
                        counter[initial] = int(item.split(b"value=")[1])
                else:
                    initial = None
                if (
                    active
                    and b"TAG_CHANGE Entity=GameEntity tag=" in text
                    and (b"tag=STEP value=MAIN_END" in text or b"tag=TURN value=" in text)
                ):
                    active = None
                if active and text.startswith(b"BLOCK_START BlockType="):
                    m = re.match(
                        rb"BLOCK_START BlockType=(\w+) Entity=(.*?) EffectCardId=.*? Target=(.*?) SubOption=(-?\d+)",
                        text,
                    )
                    bind = active["binding"]
                    expected = (
                        b"MOVE_MINION"
                        if active["mapping"]["kind"] in ("move", "shop_move")
                        else b"PLAY"
                    )
                    if (
                        m
                        and m[1] == expected
                        and entity(m[2]) == bind["source"]
                        and entity(m[3]) == bind["target"]
                        and int(m[4]) == bind["sub"]
                    ):
                        effect = start
                if b" tag=NUM_OPTIONS_PLAYED_THIS_TURN value=" in item:
                    m = re.fullmatch(
                        rb"TAG_CHANGE Entity=(.*?) tag=NUM_OPTIONS_PLAYED_THIS_TURN value=(\d+)",
                        item,
                    )
                    if m:
                        owner = entity(m[1])
                        new = int(m[2])
                        old = counter.get(owner, 0)
                        if (
                            active
                            and effect is not None
                            and owner is not None
                            and owner == active["execution"]["player_entity"]
                            and old == baseline
                            and new == old + 1
                        ):
                            correct.add(active["offset"])
                            ex = active["execution"]
                            if (effect, start, old, new) != (
                                ex.get("bound_effect"),
                                ex.get("own_counter_increment"),
                                ex.get("counter_before"),
                                ex.get("counter_after"),
                            ):
                                raise ValueError("raw receipt witnesses differ")
                            active = None
                        if owner is not None:
                            counter[owner] = new
    for offset, row in offsets.items():
        ex = row["execution"]
        strict = offset in correct
        if strict != (ex["status"] == "confirmed"):
            raise ValueError(f"raw receipt verdict differs at {offset}")
        old = ex["cooccurrence"] and ex.get("first_effect_kind") in ("PLAY", "MOVE_MINION")
        if strict != old or strict != ex["cooccurrence"]:
            equivalent.append(
                dict(
                    offset=offset,
                    row=row["row"],
                    legacy_play_move=old,
                    broad=ex["cooccurrence"],
                    strict=strict,
                )
            )
    return dict(
        verified=len(correct),
        unconfirmed=len(offsets) - len(correct),
        per_row_differences=equivalent,
    )


def inspect_raw(log, rows):
    # Independent of the replay tokenizer. Exact byte offsets detect missing, added or duplicate events.
    raw = []
    ends = 0
    invalid = 0
    with log.open("rb") as f:
        offset = 0
        for line in f:
            if b" GameState.SendOption() - selectedOption=" in line:
                raw.append((offset, "SendOption"))
            elif b" GameState.SendChoices() - id=" in line:
                raw.append((offset, "SendChoices"))
            elif b" GameState.DebugPrintOptions() - " in line and b"type=END_TURN " in line:
                ends += 1
                invalid += b"error=INVALID " in line
            offset += len(line)
    return dict(
        rows_equal=raw == [(r["offset"], r["event"]) for r in rows],
        partition=dict(Counter(event for _, event in raw)),
        end_turn=ends,
        invalid_end_turn=invalid,
    )


def report(root, archive, ledgers, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        name: digest
        for digest, name in (
            s.split(None, 1) for s in (archive / "SHA256SUMS").read_text().splitlines()
        )
    }
    baselines = {name: sha(archive / name) for name in manifest}
    if baselines != manifest:
        raise ValueError("archive does not match supplied hashes")
    proc = {}
    metrics = defaultdict(Counter)
    failures = []
    moves = []
    perf = {}
    cards = card_metadata()
    for log in sorted(archive.glob("*/Power.log")):
        session = log.parent.name
        dest = ledgers / session
        rows = [json.loads(line) for line in (dest / "ledger.jsonl").read_text().splitlines()]
        report = read(dest / "summary.json")
        raw = inspect_raw(log, rows)
        if not raw["rows_equal"] or report["sha256"] != manifest[str(log.relative_to(archive))]:
            raise ValueError("raw ledger mismatch for " + session)
        witness_audit = receipt_crosscheck(log, rows)
        opts = [r for r in rows if r["event"] == "SendOption"]

        def strict(r):
            ex = r["execution"]
            return (
                "bound_effect" in ex
                and "own_counter_increment" in ex
                and r["offset"] < ex["bound_effect"] < ex["own_counter_increment"]
            )

        if any(strict(r) != (r["execution"]["status"] == "confirmed") for r in opts):
            raise ValueError("inconsistent effect/counter offsets")
        for r in rows:
            m = r.get("resolved_magnetic", r["mapping"])
            metrics[m["kind"]][m["status"]] += 1
            if m["kind"] == "move":
                b = r["binding"]
                before = next(
                    e["pos"] for e in r["server_before"]["board"] if e["id"] == b["source"]
                )
                moves.append(
                    dict(
                        session=session,
                        row=r["row"],
                        offset=r["offset"],
                        source=b["source"],
                        before=before,
                        requested=b["position"],
                        status=m["status"],
                        confirmed=r["execution"]["status"],
                        changed_position_offset=r["execution"].get("source_position_offset"),
                    )
                )
            if m["status"] == "mapped" and (
                not m["native_action_sequence"]
                or not all(0 <= a < 245 for a in m["native_action_sequence"])
            ):
                raise ValueError("invalid native sequence")
        proc[session] = dict(
            raw=raw,
            rows=len(rows),
            ordinary=len(opts),
            confirmed=sum(map(strict, opts)),
            raw_receipts=witness_audit,
            legacy=report["legacy_play_move"],
            broad=report["cooccurrence"],
            choice=len(rows) - len(opts),
            choices_confirmed=sum(
                r["execution"]["status"] == "confirmed" for r in rows if r["event"] == "SendChoices"
            ),
            auto=report["automatic"],
            stale=report["stale_blocks"],
            blocks=report["blocks"],
            p99_64k_ms=report["p99_ms"],
            load_replay_s=report["replay_seconds"],
        )
        failures.extend(exceptions(rows, log.parent))
        perf[session] = benchmark(log, cards)
    demos = next(ledgers.glob("*23_31_31"))
    vf = next(ledgers.glob("*14_35_57"))
    attempts = read(vf / "dispatch.json")
    hist = compare(root, archive, ledgers)
    write_json(out / "HISTORICAL_REGRESSIONS.json", hist)
    write_json(out / "SERVER_EXCEPTIONS.json", failures)
    write_json(out / "PERFORMANCE.json", perf)
    inventory(out / "OBSERVE2_FIELDS.csv")
    write_json(out / "MOVES.json", moves)
    late = hist["late"]["prefixes"]
    mapped = sum(c["mapped"] for c in metrics.values())
    excluded = sum(c["registered_excluded"] for c in metrics.values())
    total = sum(q["rows"] for q in proc.values())
    contexts = Counter(x["context"] for x in failures)
    dispatch = Counter(a["input_status"] for a in attempts)
    gates = {
        "archive_and_raw_row_coverage": len(baselines) == 81
        and len(proc) == 6
        and all(p["raw"]["rows_equal"] for p in proc.values()),
        "mapping_per_observed_class": all(
            (c["mapped"] + c["registered_excluded"]) / c.total() >= 0.95 for c in metrics.values()
        ),
        "move_boundaries_audited": all(
            (m["before"] != m["requested"]) == (m["changed_position_offset"] is not None)
            and (m["status"] == "mapped") == (m["before"] != m["requested"])
            for m in moves
        ),
        "demo_late_board": len(late) > 0
        and hist["late"]["legacy_example_board"] == 0
        and all(r["server_count"] and r["presented_count"] for r in late),
        "visible_fix_views": all(
            v["legacy_reproduced"] and v["old"] != v["replay"] for v in hist["visible_fix"].values()
        ),
        "stale_blocks": (
            proc[vf.name]["stale"],
            proc[vf.name]["blocks"],
            proc[demos.name]["stale"],
            proc[demos.name]["blocks"],
        )
        == (4, 40, 36, 232),
        "choices": all(p["choices_confirmed"] == p["choice"] for p in proc.values()),
        "old_failures": bool(
            hist["bob_skin"]["old_error"]
            and hist["locked_hero"]["legacy_included_locked"]
            and len(set(hist["reconnect"]["legacy_keys"]) - {None}) == 3
            and len({e["match_key"] for e in hist["reconnect"]["epochs"]}) == 1
            and not hist["overlay"]["old_has_subs"]
            and len(hist["overlay"]["offered_subs"]) == 2
            and all(hist["overlay"]["unchanged_old_choice_confirms"].values())
        ),
        "dispatch_negative_cases": {
            (a["step"], a["label"]) for a in attempts if a["input_status"] == "no_send"
        }
        == {
            (8, "dark_gift"),
            (23, "dark_gift"),
            (57, "model_choice"),
            (62, "play"),
            (65, "refresh"),
            (68, "buy"),
            (71, "dark_gift"),
            (74, "buy"),
            (77, "upgrade"),
        },
        "dispatch_bindings": dispatch == {"matched": 27, "binding_mismatch": 1, "no_send": 9}
        and [a["step"] for a in attempts if a["input_status"] == "binding_mismatch"] == [1],
        "legacy_receipt_regression": sum(p["legacy"] for p in proc.values()) == 1237
        and sum(p["ordinary"] for p in proc.values()) == 1246,
        "raw_receipt_witnesses": all(
            p["raw_receipts"]["verified"] == p["confirmed"] for p in proc.values()
        ),
        "server_exceptions_explained": not contexts["unresolved"],
        "p99_under_50_ms": all(
            p["p99_64k_ms"] < 50 and perf[k]["p99_ms"] < 50 for k, p in proc.items()
        ),
    }
    verdict = "PASS" if all(gates.values()) else "HOLD"
    result = dict(
        status="paused",
        gate=verdict,
        scope="offline archived evidence",
        checks=gates,
        logs=proc,
        rows=total,
        mapped=mapped,
        registered_excluded=excluded,
        unsupported=sum(c["unsupported"] for c in metrics.values()),
        unresolved_mapping=sum(c["unresolved"] for c in metrics.values()),
        per_kind=metrics,
        receipt_exceptions=contexts,
        dispatch_visible_fix=dispatch,
        demo_late_prefixes=len(late),
        late_false_empty_removed=sum(not r["old_count"] and bool(r["server_count"]) for r in late),
        model_or_game_operations=0,
        next_stage_started=False,
    )
    write_json(out / "RESULT.json", result)
    write_json(out / "ARCHIVE_SHA256.json", baselines)
    # Paths and byte offsets stay relative and all hyperlinks work offline.
    import os

    links = []
    for name, stats in proc.items():
        href = html.escape(os.path.relpath(ledgers / name / "index.html", out))
        links.append(
            f'<li><a href="{href}">{name}</a>　{stats["rows"]} 条发送，普通操作确认 {stats["confirmed"]}/{stats["ordinary"]}</li>'
        )
    (out / "index.html").write_text(
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>OmniBob 离线回放</title>'
        "<style>body{font:17px/1.8 system-ui;max-width:56rem;margin:3rem auto;padding:1rem}code{overflow-wrap:anywhere}</style>"
        f"<h1>六份归档的逐操作账本</h1><p>状态 {verdict} · 共 {total} 条发送。下列链接可在本地打开。</p><ul>"
        + "".join(links)
        + '</ul><h2>回执与证据</h2><p><a href="SERVER_EXCEPTIONS.json">普通操作未确认明细</a> · '
        '<a href="HISTORICAL_REGRESSIONS.json">五类历史故障</a> · <a href="OBSERVE2_FIELDS.csv">字段表</a> · '
        '<a href="PERFORMANCE.json">时延</a> · <a href="RESULT.json">各项验收</a></p>'
        + f'<p><a href="{os.path.relpath(vf / "dispatch.json", out)}">37 条脚本输入与客户端发送的逐操作对照</a></p></html>'
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--ledgers", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = report(Path(__file__).resolve().parents[3], args.archive, args.ledgers, args.out)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "gate", "checks", "rows", "receipt_exceptions")}
        )
    )


if __name__ == "__main__":
    main()
