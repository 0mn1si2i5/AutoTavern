"""Compare at archived byte cursors. Execute only named pure functions from pinned sources."""

import ast
import hashlib
import json
import re
import sqlite3
from types import SimpleNamespace

from .archive import prefixes


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pure(source, name, functions, namespace):
    tree = ast.parse((source / name).read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.Assign) or isinstance(n, ast.FunctionDef) and n.name in functions
    ]
    tree.body = nodes
    exec(compile(tree, str(source / name), "exec"), namespace)
    return SimpleNamespace(**namespace)


def legacy_view(source, prefix, catalog):
    log = pure(source, "log_view.py", {"entity", "snapshot"}, {"re": re})
    live = pure(
        source, "live_view.py", {"snapshot"}, {"log_view": log, "re": re, "hashlib": hashlib}
    )
    return live.snapshot(prefix.decode(errors="replace"), catalog)


def compare(root, archive, final):
    base = root / "runs/real-client-autoplay"
    dirs = [
        base / ("v03-transfer-live-002" + suffix) for suffix in ("", "-hero-fix", "-visible-fix")
    ]
    source_info = {}
    for path in dirs:
        expected = read(path / "SESSION.json")["source_hashes"]
        actual = {f: sha(path / "source" / f) for f in expected}
        if expected != actual:
            raise ValueError("historic source changed: " + path.name)
        source_info[path.name] = actual
    visible = dirs[-1]
    log = next(archive.glob("*14_35_57/Power.log"))
    data = log.read_bytes()
    old = {
        n: read(visible / f"steps/{n:06d}/VISIBLE_VIEW.json")
        for n in (1, 8, 21, 22, 23, 57, 62, 65, 68, 71, 74, 77, 79)
    }
    hero = read(dirs[0] / "steps/000011/VISIBLE_VIEW.json")
    offsets = [v["source_bytes"] for v in old.values()] + [hero["source_bytes"]]
    now = dict(prefixes(log, offsets, {}))
    db = root / "runs/real-client-data/revisions/255790-cdeb4e540bd4cc0d/catalog.sqlite"
    conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    cache = {k: json.loads(v) for k, v in conn.execute("select card_id,data from cards")}
    conn.close()

    class Reference(dict):
        meta = {"build": "255790"}

    catalog = Reference(cache)
    reg = {}
    for n in (1, 21, 22, 23):
        off = old[n]["source_bytes"]
        was = legacy_view(visible / "source", data[:off], catalog)
        equal = all(
            [c["entity_id"] for c in was[z]] == [c["entity_id"] for c in old[n][z]]
            for z in ("board", "hand", "shop")
        )
        new = now[off]["server"]
        reg[n] = dict(
            offset=off,
            legacy_reproduced=equal,
            old={z: [e["entity_id"] for e in was[z]] for z in ("board", "shop")},
            replay={z: [e["id"] for e in new[z]] for z in ("board", "shop")},
        )

    def historic_method():
        # The old Encoder.actions uses only np.zeros in its hero-offer branch.
        tree = ast.parse((dirs[0] / "source/model_bridge.py").read_text())
        method = next(
            node
            for cl in tree.body
            if isinstance(cl, ast.ClassDef) and cl.name == "Encoder"
            for node in cl.body
            if isinstance(node, ast.FunctionDef) and node.name == "actions"
        )
        ns = {"np": SimpleNamespace(zeros=lambda n, dtype: [False] * n)}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "historical.actions", "exec"), ns)
        return ns["actions"]

    _, bindings = historic_method()(None, hero, hero["active_choice"]["entities"], 1)
    offered = now[hero["source_bytes"]]["server_choice_offered"]
    locked = [e["id"] for e in offered if e["locked"]]
    legacy = [v["entity"]["entity_id"] for v in bindings.values()]
    # Both legacy snapshots receive exactly the same prefix. The hero-fix implementation
    # counted some Bob skins as a second player; the live parser swallowed this exception.
    off = old[1]["source_bytes"]
    try:
        legacy_view(dirs[1] / "source", data[:off], catalog)
        bob_error = None
    except ValueError as exc:
        bob_error = str(exc)
    # Do not compile module-level initialization from a controller.
    tree = ast.parse((visible / "source/model_autoplay.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "confirmed"]
    ns = {}
    exec(compile(tree, "archived.confirmed", "exec"), ns)
    repeated = {}
    for step in (8, 23):
        binding = read(visible / f"steps/{step:06d}/MODEL_DECISION.json")["binding"]
        repeated[step] = ns["confirmed"](binding, old[step], old[step])
    replay62 = now[old[62]["source_bytes"]]
    option = next(o for o in replay62["server_options"] if o["entity"]["card"] == "BG31_320")
    demo = root / "runs/human-demonstration-20261106-001"
    history = {}
    for file in (demo / "views").glob("*.json"):
        record = read(file)
        v = record["view"]
        if v.get("round") in (11, 12, 13) and v["game"]["STEP"] == "MAIN_ACTION":
            history[record["source_bytes"]] = v
    demo_log = next(archive.glob("*23_31_31/Power.log"))
    late = []
    for offset, view in prefixes(demo_log, history, {}):
        oldview = history[offset]
        late.append(
            dict(
                offset=offset,
                round=oldview["round"],
                old_count=len(oldview["board"]),
                server_count=len(view["server"]["board"]),
                presented_count=len(view["display"]["board"]),
            )
        )
    example = next(item for item in late if item["old_count"] == 0)
    with demo_log.open("rb") as handle:
        old_example = legacy_view(visible / "source", handle.read(example["offset"]), catalog)
    reconnect = next(final.glob("*09_27*/summary.json"))
    ep = read(reconnect)["epochs"]
    reconlog = archive / reconnect.parent.name / "Power.log"
    keys = []
    with reconlog.open("rb") as handle:
        # Old game_key is a digest of the first three lines from the last CREATE_GAME.
        # Exercise the historical snapshot even when the prefix contains multiple headers.
        raw = handle.read()
        for epoch in ep:
            stop = raw.index(b"GameState.DebugPrintOptions() - id=", epoch["start"])
            end = raw.index(b"PowerTaskList.", stop)
            try:
                saved_view = legacy_view(visible / "source", raw[:end], catalog)
                keys.append(saved_view["game_key"])
            except ValueError:
                keys.append(None)
    return dict(
        verified_source_hashes=source_info,
        catalog_sha256=sha(db),
        visible_fix=reg,
        locked_hero=dict(
            offset=hero["source_bytes"],
            legacy_allowed=legacy,
            locked=locked,
            replay_allowed=[e["id"] for e in offered if not e["locked"]],
            legacy_included_locked=bool(set(legacy) & set(locked)),
        ),
        bob_skin=dict(prefix=off, old_error=bob_error, replay_board=reg[1]["replay"]["board"]),
        overlay=dict(
            offset=old[62]["source_bytes"],
            offered_subs=option["subs"],
            old_has_subs=any("subs" in o for o in old[62]["last_options"]["options"]),
            unchanged_old_choice_confirms=repeated,
        ),
        late=dict(
            prefixes=late,
            legacy_example=example["offset"],
            legacy_example_board=len(old_example["board"]),
        ),
        reconnect=dict(epochs=ep, legacy_keys=keys),
    )
