"""Local journal for future human-executed real games; no game control or screen scraping.

Suggestions are supplied by the human/assistant after inspecting visible evidence.
This is a recording tool, not an automatic real-client observation encoder.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def now():
    return datetime.now(timezone.utc).isoformat()


def initialize(directory, game_version, rules, policy):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/"session.json").write_text(json.dumps({"format": "omnibob-manual-v1", "created_at": now(),
        "source": "human-executed real game; user-reported facts", "game_version": game_version,
        "rules": rules, "policy": policy, "automatic_game_integration": False}, indent=2)+"\n")
    (directory/"events.jsonl").touch()
    return str(directory)


def events(directory):
    directory = Path(directory)
    metadata = json.loads((directory/"session.json").read_text())
    if metadata.get("format") != "omnibob-manual-v1":
        raise ValueError("unsupported manual journal")
    return [json.loads(line) for line in (directory/"events.jsonl").read_text().splitlines()]


def append(directory, event):
    with (Path(directory)/"events.jsonl").open("a") as stream:
        stream.write(json.dumps({"at": now(), **event}, ensure_ascii=False)+"\n")


def record(directory, kind, **fields):
    directory = Path(directory)
    history = events(directory)
    if any(row["kind"] == "result" for row in history):
        raise ValueError("session already finished")
    suggestions = {r["decision"]: r for r in history if r["kind"] == "suggestion"}
    if kind == "suggestion":
        if fields["turn"] < 1 or not fields["advice"].strip():
            raise ValueError("positive turn and nonempty advice required")
        evidence = Path(fields.pop("evidence"))
        if not evidence.is_file():
            raise ValueError("visible evidence must be a local file")
        decision = len(suggestions)+1
        filename = f"visible-{decision:05d}{evidence.suffix}"
        shutil.copyfile(evidence, directory/filename)
        fields.update(decision=decision, evidence=filename,
                      evidence_sha256=hashlib.sha256((directory/filename).read_bytes()).hexdigest())
    elif kind == "execution":
        decision = fields["decision"]
        if decision not in suggestions or any(r["kind"] == kind and r["decision"] == decision for r in history):
            raise ValueError("execution needs one existing, unexecuted suggestion")
        if not fields["actual"].strip():
            raise ValueError("actual action required; use an explicit description if skipped")
        if fields["actual"] != suggestions[decision]["advice"] and not fields["deviation"].strip():
            raise ValueError("describe why actual action differs from advice")
    elif kind == "result":
        if fields["placement"] not in range(1, 9):
            raise ValueError("placement must be 1..8")
    else:
        raise ValueError("unsupported manual event")
    row = {"kind": kind, **fields}
    append(directory, row)
    return row


def report(directory):
    rows = events(directory)
    suggested = {r["decision"] for r in rows if r["kind"] == "suggestion"}
    executed = {r["decision"] for r in rows if r["kind"] == "execution"}
    return {"suggestions": len(suggested), "recorded_executions": len(executed),
            "missing_execution_records": sorted(suggested-executed),
            "deviations": sum(bool(r.get("deviation")) for r in rows if r["kind"] == "execution"),
            "result": next((r for r in rows if r["kind"] == "result"), None),
            "note": "Manual records are not independent verification of live play or automatic policy operation."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("init", "suggest", "execute", "finish", "report"):
        command = sub.add_parser(name)
        command.add_argument("directory", type=Path)
        if name == "init":
            for field in ("game-version", "rules", "policy"):
                command.add_argument("--"+field, required=True)
        elif name == "suggest":
            command.add_argument("--turn", type=int, required=True)
            command.add_argument("--evidence", type=Path, required=True, help="only information visible to your player")
            command.add_argument("--advice", required=True)
            command.add_argument("--reason", required=True)
        elif name == "execute":
            command.add_argument("--decision", type=int, required=True)
            command.add_argument("--actual", required=True)
            command.add_argument("--deviation", default="")
        elif name == "finish":
            command.add_argument("--placement", type=int, required=True)
            command.add_argument("--notes", default="")
    args = vars(p.parse_args())
    command, directory = args.pop("command"), args.pop("directory")
    result = (initialize(directory, **args) if command == "init" else report(directory) if command == "report"
              else record(directory, {"suggest": "suggestion", "execute": "execution", "finish": "result"}[command], **args))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
