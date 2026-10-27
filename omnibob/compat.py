"""Exact local compatibility checks and reproducible run identities."""
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import tempfile

import numpy as np
import taverngym as tg

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {"1.0.0": "contract-v1.json", "1.1.0": "contract-arena-v2.json"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def effective_config(overrides=None):
    """Let the engine validate and normalize default/omitted rule fields."""
    game = tg.Game(0, json.dumps(overrides or {}))
    return json.loads(game.state_json())["cfg"]


def environment_contract():
    return {
        "package_version": tg.version(),
        "game_version": tg.game_version(),
        "server_rules_version": tg.server_rules_version(),
        "engine_source_hash": tg.build_info()["source_hash"],
        "layout": dict(tg.layout()),
        "actions_sha256": digest([tg.action_name(i) for i in range(tg.N_ACTIONS)]),
        "catalog_sha256": hashlib.sha256(tg.catalog_json().encode()).hexdigest(),
        "vocabulary_sha256": digest([None] + [tg.card_info(i) for i in range(1, tg.N_CARDS)]),
        "n_cards": tg.N_CARDS,
    }


def check_compatibility(expected=None):
    if expected is None:
        name = CONTRACTS.get(tg.version())
        if name is None:
            raise ValueError(f"unreviewed TavernGym version: {tg.version()}")
        expected = json.loads((ROOT / "omnibob" / name).read_text())
    actual = environment_contract()
    mismatches = [key for key in expected if expected[key] != actual.get(key)]
    if tg.build_is_current(ROOT) is not True:
        mismatches.append("native build does not match this checkout")
    if mismatches:
        raise ValueError("incompatible TavernGym: " + ", ".join(mismatches))
    return actual


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_manifest():
    paths = [p for base in ("omnibob", "python", "crates", "configs", "tools", "tests") for p in (ROOT/base).rglob("*")
             if p.is_file() and p.suffix in (".py", ".rs", ".json", ".toml", ".sh")]
    paths += [ROOT/p for p in ("Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "pyproject.toml")]
    paths += list(ROOT.glob("requirements-omnibob*.lock"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def source_snapshot(sources):
    """Content-addressed local source overlay; no credentials, runs, native binaries or environments."""
    directory = ROOT/"runs/source-snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory/(digest(sources)+".tar.gz")
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with tarfile.open(temporary, "w:gz") as archive:
                for name, expected in sources.items():
                    data = (ROOT/name).read_bytes()
                    if hashlib.sha256(data).hexdigest() != expected:
                        raise RuntimeError("source changed while snapshotting; rerun with stable source files")
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mode = len(data), 0o644
                    archive.addfile(entry, io.BytesIO(data))
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
    return {"path": str(destination.relative_to(ROOT)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "restore": "overlay onto the recorded git base; install local dependencies separately"}


def provenance(config):
    from . import __version__
    sources = source_manifest()
    return {
        "omnibob_version": __version__, "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "source_files": sources,
        "source_sha256": digest(sources), "source_archive": source_snapshot(sources),
        "environment": check_compatibility(),
        "native_build": tg.build_info(), "config": config, "config_sha256": digest(config),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
    }
