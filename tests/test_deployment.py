import hashlib
import json
from dataclasses import asdict

import pytest
import torch

from omnibob import deployment
from omnibob.compat import ROOT, check_compatibility, digest, effective_config
from omnibob.model import EntityActorCritic, ModelConfig
from omnibob.repeat_guard import runtime_config


@pytest.fixture
def releases(tmp_path, monkeypatch):
    # Random inference weights only: no optimizer, training, or external files.
    config = ModelConfig(width=16, layers=1, heads=2)
    environment = check_compatibility()
    rules = effective_config()
    state = {"format": "omnibob-policy-v1", "model": EntityActorCritic(config).state_dict(),
             "model_config": asdict(config), "environment": environment, "config": rules,
             "model_source_sha256": hashlib.sha256((ROOT / "omnibob/model.py").read_bytes()).hexdigest(),
             "runtime_config": runtime_config(1), "steps": 0}
    catalog = {}
    for version in ("v0.2", "v0.3"):
        checkpoint = tmp_path / (version + ".pt")
        torch.save(state, checkpoint)
        manifest = tmp_path / (version + ".json")
        manifest.write_text(json.dumps({"version": version, "checkpoint": str(checkpoint),
            "checkpoint_sha256": deployment._sha(checkpoint), "environment": environment,
            "runtime_config": runtime_config(1), "rules_sha256": digest(rules), "scheduler": "shuffled_rounds"}))
        catalog[version] = {"path": str(manifest), "sha256": deployment._sha(manifest)}
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    monkeypatch.setattr(deployment, "RELEASES", tmp_path)
    return tmp_path


def test_switch_rollback_and_reject_corrupt_candidate(releases):
    original = (releases / "v0.2.pt").read_bytes()
    deployment.activate("v0.2")
    deployment.activate("v0.3")
    assert deployment.resolve()["version"] == "v0.3"
    deployment.activate("v0.2")
    before = (releases / "current.json").read_bytes()
    (releases / "v0.3.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint checksum"):
        deployment.activate("v0.3")
    assert (releases / "current.json").read_bytes() == before
    assert deployment.resolve()["version"] == "v0.2"
    assert (releases / "v0.2.pt").read_bytes() == original


def test_missing_unknown_and_modified_manifest_fail_closed(releases):
    with pytest.raises(FileNotFoundError):
        deployment.resolve()
    with pytest.raises(ValueError, match="unknown local release"):
        deployment.activate("unknown")
    deployment.activate("v0.2")
    with (releases / "v0.2.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="manifest checksum"):
        deployment.resolve()


def test_runtime_mismatch_rejected_before_switch(releases):
    deployment.activate("v0.2")
    path = releases / "v0.3.json"
    manifest = json.loads(path.read_text())
    manifest["runtime_config"] = {}
    path.write_text(json.dumps(manifest))
    catalog = json.loads((releases / "catalog.json").read_text())
    catalog["v0.3"]["sha256"] = deployment._sha(path)
    (releases / "catalog.json").write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="runtime or rules"):
        deployment.activate("v0.3")
    assert deployment.resolve()["version"] == "v0.2"


def test_default_cli_selects_release_and_preserves_explicit_baseline(releases, tmp_path, monkeypatch):
    from omnibob import __main__, evaluate
    deployment.activate("v0.3")
    calls = []
    def local_evaluate(output, policies, *args):
        output.mkdir()
        calls.append((policies, args))
        return {"policies": {policies[0]: {"completed": 1, "requested": 1}}}
    monkeypatch.setattr(evaluate, "evaluate", local_evaluate)
    monkeypatch.setattr("sys.argv", ["omnibob", "play", "--output", str(tmp_path / "default")])
    assert __main__.main() == 0
    assert calls[-1][0] == ["ppo"]
    assert calls[-1][1][8].name == "v0.3.pt"
    assert calls[-1][1][10] == "shuffled_rounds"
    monkeypatch.setattr("sys.argv", ["omnibob", "play", "--policy", "omnibob", "--output", str(tmp_path / "heuristic")])
    assert __main__.main() == 0
    assert calls[-1][0] == ["omnibob"] and calls[-1][1][8] is None
    assert calls[-1][1][10] == "round_robin"
