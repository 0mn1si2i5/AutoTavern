"""Read-only public arena preflight and a local policy adapter for the reviewed connector.

This module does not register, queue or send actions. The official connector is
run separately only after the user authorizes a match and supplies a local key.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import taverngym as tg

from .compat import ROOT, check_compatibility, digest, effective_config
from .observation_v2 import ObservationV2
from .policies import OmniBobPolicy


def catalog_hash():
    cards = [{"index": i, "id": info[0], "name": info[1], "name_zh": info[2], "tier": info[3]}
             for i in range(1, tg.N_CARDS) for info in (tg.card_info(i),)]
    return digest({"empty_index": 0, "cards": cards, "details": json.loads(tg.catalog_json())})


def validate_manifest(manifest, checkpoint_config=None):
    local = check_compatibility()
    errors = []
    if manifest.get("protocol_version") != "2":
        errors.append("protocol version")
    observation = manifest["observation"]
    if [observation[k]["length"] for k in ("features", "cards", "action_mask")] != [4017, 447, 245]:
        errors.append("observation dimensions")
    if digest(observation) != manifest["observation_sha256"]:
        errors.append("observation checksum")
    if digest(manifest["actions"]) != manifest["action_layout_sha256"]:
        errors.append("remote action checksum")
    if local["actions_sha256"] != manifest["action_layout_sha256"]:
        errors.append("action encoding")
    if local["layout"] != observation["layout"]:
        errors.append("layout")
    if local["engine_source_hash"] != manifest["environment"]["engine_source_hash"]:
        errors.append("engine source")
    if local["package_version"] != manifest["environment"]["version"]:
        errors.append("package version")
    if catalog_hash() != manifest["catalog_sha256"]:
        errors.append("card vocabulary/catalog")
    if checkpoint_config is not None and effective_config(checkpoint_config) != effective_config(manifest["rules"]):
        errors.append("checkpoint rules")
    if errors:
        raise ValueError("arena incompatibility: " + ", ".join(errors))
    return {"compatible": True, "ruleset_id": manifest["ruleset_id"], "observation_version": 2,
            "engine_source_hash": local["engine_source_hash"], "catalog_sha256": manifest["catalog_sha256"],
            "rules_sha256": digest(effective_config(manifest["rules"])), "queued": False,
            "checked_at": datetime.now(timezone.utc).isoformat()}


def preflight(url="https://rizling.top", checkpoint=None):
    origin = urlsplit(url)
    if origin.scheme != "https" or not origin.hostname or origin.path not in ("", "/") or any(
        (origin.username, origin.password, origin.query, origin.fragment)
    ):
        raise ValueError("use an HTTPS origin without credentials, path, query or fragment")
    request = Request(url.rstrip("/")+"/api/v1/manifest",
                      headers={"Accept": "application/json", "User-Agent": "OmniBob-Preflight/0.2"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(2**20+1)
    if len(raw) > 2**20:
        raise ValueError("unexpected manifest size")
    rules = None
    if checkpoint:
        from .learned import load_policy
        rules = load_policy(checkpoint).config
    return validate_manifest(json.loads(raw), rules)


_learned = None


def prepare(checkpoint=None):
    """Load before the connector enters a match; reject mismatched reviewed rules."""
    global _learned
    checkpoint = checkpoint or os.environ.get("OMNIBOB_CHECKPOINT")
    if checkpoint:
        from .learned import load_policy
        policy = load_policy(checkpoint)
        validate_manifest(json.loads((ROOT/"configs/arena-manifest-v2.json").read_text()), policy.config)
        _learned = policy
    return _learned


def act(observation, context):
    """Official connector entry point: act(observation, context) -> legal integer.

    OMNIBOB_CHECKPOINT optionally chooses a learned policy. Otherwise frozen v0.1.
    No environment variables holding credentials are read here.
    """
    global _learned
    obs = ObservationV2(**{k: observation[k] for k in ("features", "cards", "action_mask")})
    state = context["state"]
    checkpoint = os.environ.get("OMNIBOB_CHECKPOINT")
    if checkpoint:
        if _learned is None or _learned.checkpoint != str(Path(checkpoint).resolve()):
            prepare(checkpoint)
        if state.get("learned_definition") is not _learned:
            from .repeat_guard import clone_runtime
            policy = clone_runtime(_learned)
            state["learned_definition"], state["learned_policy"] = _learned, policy
        policy = state["learned_policy"]
        decision = policy.act(obs)
    else:
        if "omnibob_policy" not in state:
            state["omnibob_policy"] = OmniBobPolicy()
        policy = state["omnibob_policy"]
        decision = policy.act(obs.legacy())
    if not obs.action_mask[decision.action]:
        raise ValueError("policy proposed an illegal arena action")
    journal = os.environ.get("OMNIBOB_JOURNAL_DIR")
    if journal:
        directory = Path(journal)
        directory.mkdir(parents=True, exist_ok=True)
        # Filenames never contain untrusted match IDs; JSON holds only the provided own view.
        record = {"match_id": context["match_id"], "seat": context["seat"], "turn": context["turn"],
                  "action": decision.action, "action_name": tg.action_name(decision.action),
                  "reason": decision.reason, "policy": getattr(policy, "version", "tempo-v0.1"),
                  "remaining_ms": context.get("remaining_ms"), "player_view": context.get("player_view"),
                  "observation": {"features": obs.features.tolist(), "cards": obs.cards.tolist(),
                                  "action_mask": obs.action_mask.astype(int).tolist()}}
        with (directory/"decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False)+"\n")
    return int(decision.action)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="https://rizling.top")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = preflight(args.url, args.checkpoint)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
elif os.environ.get("OMNIBOB_CHECKPOINT"):
    # The official connector imports the policy module before queueing.
    prepare()
