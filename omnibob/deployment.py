"""Select a verified local inference release without networking or training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .compat import ROOT, check_compatibility, digest

RELEASES = ROOT / "runs/releases"


def _read(path):
    return json.loads(path.read_text())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(version=None):
    """Validate the selected bytes and exact runtime before returning a checkpoint."""
    catalog = _read(RELEASES / "catalog.json")
    selected = _read(RELEASES / "current.json") if version is None else {"version": version}
    version = selected["version"]
    if version not in catalog:
        raise ValueError(f"unknown local release: {version}")
    entry = catalog[version]
    if "manifest_sha256" in selected and selected["manifest_sha256"] != entry["sha256"]:
        raise ValueError("active release manifest changed")
    manifest = ROOT / entry["path"]
    if _sha(manifest) != entry["sha256"]:
        raise ValueError("release manifest checksum mismatch")
    release = _read(manifest)
    if release["version"] != version:
        raise ValueError("release version mismatch")
    checkpoint = ROOT / release["checkpoint"]
    if _sha(checkpoint) != release["checkpoint_sha256"]:
        raise ValueError("release checkpoint checksum mismatch")
    from .train import load_checkpoint
    from .learned import load_policy
    state = load_checkpoint(checkpoint)
    if state.get("format") != "omnibob-policy-v1" or any(
        k in state for k in ("optimizer", "environments", "sampling_rng", "torch_rng", "reference_model")
    ):
        raise ValueError("local release must contain inference-only weights")
    check_compatibility(release["environment"])
    policy = load_policy(checkpoint, device="cpu", cpu_threads=1)
    if policy.runtime_config != release["runtime_config"] or digest(policy.config) != release["rules_sha256"]:
        raise ValueError("release runtime or rules mismatch")
    return {"version": version, "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": release["checkpoint_sha256"],
            "manifest_sha256": entry["sha256"], "runtime_config": policy.runtime_config,
            "scheduler": release["scheduler"], "device": "cpu", "cpu_threads": 1}


def activate(version):
    """Atomic local pointer switch; immutable release directories stay untouched."""
    selected = resolve(version)
    pointer = {"version": version, "manifest_sha256": selected["manifest_sha256"]}
    fd, name = tempfile.mkstemp(prefix=".current-", suffix=".json", dir=RELEASES)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(pointer, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, RELEASES / "current.json")
    finally:
        Path(name).unlink(missing_ok=True)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    use = commands.add_parser("use")
    use.add_argument("version", help="local release, e.g. v0.2 or v0.3")
    args = parser.parse_args()
    try:
        result = activate(args.version) if args.command == "use" else resolve()
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Local deployment unavailable: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
