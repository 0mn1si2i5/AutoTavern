"""Export observation-only deployment weights without optimizer or simulator saves."""
import argparse
import hashlib
import json
from pathlib import Path

from .compat import ROOT
from .train import atomic_checkpoint, load_checkpoint


def export(checkpoint, output, move_budget=None, repeat_limit=None):
    checkpoint, output = Path(checkpoint), Path(output)
    state = load_checkpoint(checkpoint)
    if state.get("error"):
        raise ValueError("cannot export a failed training checkpoint")
    fields = ("model", "model_config", "environment", "config", "source_sha256", "steps", "update",
              "training_method", "imitation_epoch", "teacher", "validation_accuracy", "parent_checkpoint",
              "policy_label", "weight_averaging",
              "reference_checkpoint", "objective", "runtime_config", "truncation_handling", "frozen_opponent")
    deployment = {key: state[key] for key in fields if key in state}
    deployment.update(format="omnibob-policy-v1", error=None,
                      model_source_sha256=hashlib.sha256((ROOT/"omnibob/model.py").read_bytes()).hexdigest(),
                      source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    if state.get("training_method", "ppo") == "ppo":
        deployment["ppo_config"] = state.get("train_config", state.get("ppo_config"))
    runtime = dict(deployment.get("runtime_config", {}))
    if move_budget is not None:
        from .move_guard import runtime_config
        runtime.pop("move_guard", None)
        runtime.update(runtime_config(move_budget))
    if repeat_limit is not None:
        from .repeat_guard import runtime_config
        runtime.pop("repeat_guard", None)
        runtime.update(runtime_config(repeat_limit))
    deployment["runtime_config"] = runtime
    output.mkdir(parents=True, exist_ok=False)
    atomic_checkpoint(output/"policy.pt", deployment)
    metadata = {key: value for key, value in deployment.items() if key != "model"}
    metadata.update(source_checkpoint=str(checkpoint.resolve()),
                    policy_sha256=hashlib.sha256((output/"policy.pt").read_bytes()).hexdigest(),
                    policy_bytes=(output/"policy.pt").stat().st_size,
                    use="evaluate/play/arena inference or PPO weight initialization; not exact training resume")
    (output/"manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--move-budget", type=int, help="record an explicit heuristic controller; 0 disables")
    p.add_argument("--repeat-limit", type=int, help="record an explicit repeat controller; 0 disables")
    args = p.parse_args()
    print(json.dumps(export(args.checkpoint, args.output, args.move_budget, args.repeat_limit), indent=2))


if __name__ == "__main__":
    main()
