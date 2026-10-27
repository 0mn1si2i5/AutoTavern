"""Uniform parameter averaging of compatible PPO weights, not additional training.

Useful only as an explicit experiment: independently initialized networks can have
unaligned hidden units. Prefer continuations from the exact same initialization;
even then averaging need not improve policy quality. Evaluate before deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .compat import ROOT, provenance
from .train import atomic_checkpoint, load_checkpoint


def average(checkpoints, output):
    paths = [Path(p).resolve() for p in checkpoints]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("at least two distinct checkpoints required")
    states = [load_checkpoint(p) for p in paths]
    first = states[0]
    for state in states:
        if state.get("error") or state.get("training_method", "ppo") != "ppo":
            raise ValueError("only successful PPO weights can be averaged")
        if state.get("weight_averaging"):
            raise ValueError("nested averaging is not supported; provide original equal-weight inputs")
        for key in ("environment", "model_config", "config"):
            if state[key] != first[key]:
                raise ValueError(f"incompatible averaging {key}")
        if set(state["model"]) != set(first["model"]):
            raise ValueError("incompatible parameter names")
    parents = []
    for path, state in zip(paths, states):
        parents.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "steps": state["steps"], "update": state.get("update"),
                        "source_sha256": state.get("source_sha256"),
                        "ppo_config": state.get("train_config", state.get("ppo_config")),
                        "initialization": state.get("parent_checkpoint"),
                        "truncation_handling": state.get("truncation_handling")})
    initializations = {(p["initialization"] or {}).get("sha256") for p in parents}
    if len(initializations) != 1 or None in initializations:
        raise ValueError("averaging requires the same recorded initialization checkpoint")
    weights = {}
    for key, template in first["model"].items():
        values = [s["model"][key] for s in states]
        if any(t.shape != template.shape or t.dtype != template.dtype for t in values):
            raise ValueError(f"incompatible tensor {key}")
        if template.is_floating_point():
            if not all(torch.isfinite(t).all().item() for t in values):
                raise ValueError(f"nonfinite input tensor {key}")
            weights[key] = torch.stack([t.double() for t in values]).mean(0).to(template.dtype)
        else:
            if not all(torch.equal(t, template) for t in values):
                raise ValueError(f"nonfloating buffers differ: {key}")
            weights[key] = template.clone()
    definition = {"method": "uniform arithmetic mean of compatible parameters, float64 accumulation",
                  "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "parents": parents, "weights": [1/len(paths)]*len(paths),
                  "note": "No optimizer steps or new trajectories; parent step counts are per-stage and overlap in ancestry. "
                          "Runtime controllers are not inherited. This is a post-training experiment, not a paper reproduction."}
    result = {key: first[key] for key in ("environment", "model_config", "config")}
    identity = provenance(first["config"])
    result.update(format="omnibob-policy-v1", model=weights, error=None, training_method="ppo",
                  steps=0, update=0, policy_label=f"uniform-average-{len(paths)}-ppo-policies",
                  source_sha256=identity["source_sha256"],
                  model_source_sha256=hashlib.sha256((ROOT/"omnibob/model.py").read_bytes()).hexdigest(),
                  runtime_config={}, weight_averaging=definition)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_checkpoint(out/"policy.pt", result)
    manifest = {key: value for key, value in result.items() if key != "model"}
    manifest.update(policy_sha256=hashlib.sha256((out/"policy.pt").read_bytes()).hexdigest(),
                    source_archive=identity["source_archive"],
                    policy_bytes=(out/"policy.pt").stat().st_size)
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(average(args.checkpoints, args.output), indent=2))


if __name__ == "__main__":
    main()
