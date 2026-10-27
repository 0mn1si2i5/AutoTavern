"""Optional warm start from the frozen heuristic, explicitly distinct from PPO gains.

Collects complete training-domain games with player-visible observe2 and legal
teacher actions. Game-level validation avoids near-identical adjacent-state splits.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .benchmark import memory
from .compat import effective_config, provenance
from .evaluate import emit
from .model import EntityActorCritic, ModelConfig
from .policies import OmniBobPolicy
from .train import TrainConfig, atomic_checkpoint, tensors
from .training_env import GameEpisode, VectorGames


@contextmanager
def stop_signals():
    requested = []
    old = {sig: signal.signal(sig, lambda number, frame: requested.append(number))
           for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: bool(requested)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def collect(output, games=128, seed=10000000, rules=None, max_seconds=120):
    if games < 1 or max_seconds <= 0 or seed < 0 or seed+games >= 1_000_000_000:
        raise ValueError("positive budgets and train-domain seeds required")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = effective_config(rules)
    identity = provenance(cfg)
    identity.update(games_requested=games, first_seed=seed, teacher="frozen tempo-v0.1",
                    observation_version=2, source="local simulator, train partition only")
    (out/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    started = time.time()
    data = {k: [] for k in ("features", "cards", "action_mask", "actions", "game_id")}
    records = []
    for i in range(games):
        if time.time()-started >= max_seconds:
            break
        episode, teacher = GameEpisode(seed+i, i % 8, cfg), OmniBobPolicy()
        for _ in range(10000):
            obs = episode.observe()
            action = teacher.act(obs.legacy()).action
            if not obs.action_mask[action]:
                raise ValueError("teacher proposed an illegal action")
            for key in ("features", "cards", "action_mask"):
                data[key].append(getattr(obs, key))
            data["actions"].append(action)
            data["game_id"].append(i)
            record = episode.step(action)
            if record is not None:
                records.append(record)
                break
        else:
            raise RuntimeError("teacher game exceeded decision bound")
    if not records:
        raise RuntimeError("no teacher games completed")
    arrays = {k: np.asarray(v, dtype=(np.int32 if k in ("actions", "game_id", "cards") else
                                    bool if k == "action_mask" else np.float32)) for k, v in data.items()}
    np.savez_compressed(out/"dataset.npz", **arrays)
    summary = {"games_completed": len(records), "games_requested": games, "decisions": len(arrays["actions"]),
               "mean_teacher_placement": float(np.mean([r["placement"] for r in records])),
               "seconds": time.time()-started, "dataset_sha256": hashlib.sha256((out/"dataset.npz").read_bytes()).hexdigest(),
               **memory()}
    (out/"games.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def fit(dataset, output, epochs=8, device="mps", max_seconds=600, deadline=None, seed=1234):
    from .compat import check_compatibility
    dataset, out = Path(dataset), Path(output)
    if epochs < 1 or max_seconds <= 0:
        raise ValueError("positive imitation budgets required")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; choose CPU explicitly")
    manifest = json.loads((dataset/"manifest.json").read_text())
    check_compatibility(manifest["environment"])
    if manifest["first_seed"] < 0 or manifest["first_seed"]+manifest["games_requested"] >= 1_000_000_000:
        raise ValueError("imitation dataset is outside the training partition")
    cfg = manifest["config"]
    with np.load(dataset/"dataset.npz", allow_pickle=False) as packed:
        data = {key: packed[key] for key in packed.files}
    n = len(data["actions"])
    if (data["features"].shape != (n, 4017) or data["cards"].shape != (n, 447)
            or data["action_mask"].shape != (n, 245)):
        raise ValueError("invalid imitation dataset shapes")
    if not data["action_mask"][np.arange(n), data["actions"]].all():
        raise ValueError("dataset contains illegal teacher actions")
    # Every fifth complete game is held back from gradient updates.
    validation = np.flatnonzero(data["game_id"] % 5 == 0)
    training = np.flatnonzero(data["game_id"] % 5 != 0)
    if not len(training) or not len(validation):
        raise ValueError("need at least two games for game-level validation")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model_config = ModelConfig()
    model = EntityActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    out.mkdir(parents=True, exist_ok=False)
    identity = provenance(cfg)
    identity.update(dataset=str(dataset.resolve()), dataset_sha256=hashlib.sha256((dataset/"dataset.npz").read_bytes()).hexdigest(),
                    model_config=asdict(model_config), train_examples=len(training), validation_examples=len(validation),
                    objective="masked teacher-action cross entropy; not RL improvement", device=device, seed=seed)
    (out/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    started = time.time()
    stop_at = min(started+max_seconds, deadline or float("inf"))
    best_accuracy, batches, status = -1., 0, "epochs_completed"
    completed = 0

    def batch(ids):
        observations = {k: data[k][ids] for k in ("features", "cards", "action_mask")}
        return tensors(observations, device), torch.tensor(data["actions"][ids], dtype=torch.long, device=device)

    with stop_signals() as stopped, (out/"metrics.jsonl").open("w") as metrics:
        for epoch in range(epochs):
            if time.time() >= stop_at or stopped():
                status = "signal requested" if stopped() else "time budget reached"
                break
            t = time.perf_counter()
            losses, hits, n = [], 0, 0
            model.train()
            order = rng.permutation(training)
            for start in range(0, len(order), 128):
                if time.time() >= stop_at or stopped():
                    status = "signal requested" if stopped() else "time budget reached"
                    break
                obs, labels = batch(order[start:start+128])
                logits, _ = model(*obs)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                hits += int((logits.argmax(-1) == labels).sum().item())
                n += len(labels)
                batches += 1
            if not losses:
                break
            model.eval()
            validation_hits, validation_loss = 0, 0.
            with torch.no_grad():
                for start in range(0, len(validation), 128):
                    obs, labels = batch(validation[start:start+128])
                    logits, _ = model(*obs)
                    validation_hits += int((logits.argmax(-1) == labels).sum().item())
                    validation_loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum").cpu())
            accuracy = validation_hits/len(validation)
            row = {"epoch": epoch+1, "train_accuracy": hits/max(n, 1), "train_loss": float(np.mean(losses)) if losses else None,
                   "validation_accuracy": accuracy, "validation_loss": validation_loss/len(validation),
                   "seconds": time.perf_counter()-t, **memory(device)}
            emit(metrics, row)
            print(json.dumps(row), flush=True)
            completed += 1
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                # A fresh environment/optimizer is used if starting PPO from this policy.
                # --initialize loads only weights; this artifact is not a PPO resume.
                ppo_cfg = TrainConfig(seed=20000000)
                env = VectorGames(ppo_cfg.n_envs, ppo_cfg.seed, cfg)
                try:
                    atomic_checkpoint(out/"best.pt", {"format": "omnibob-ppo-v1", "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "model_config": asdict(model_config),
                        "train_config": asdict(ppo_cfg), "environment": identity["environment"], "config": cfg,
                        "source_sha256": identity["source_sha256"], "update": 0, "steps": 0,
                        "model_source_sha256": identity["source_files"]["omnibob/model.py"],
                        "sampling_rng": torch.Generator().manual_seed(ppo_cfg.seed+1).get_state(),
                        "torch_rng": torch.get_rng_state(), "environments": env.save(), "error": None,
                        "stop_reason": "imitation warm start", "training_method": "behavior_cloning",
                        "imitation_epoch": epoch+1, "imitation_batches": batches,
                        "teacher": "tempo-v0.1", "validation_accuracy": accuracy})
                finally:
                    env.close()
            if status != "epochs_completed":
                break
            if max(row["rss_bytes"], row.get("mps_driver_bytes", 0)) > 4*1024**3:
                status = "RSS or MPS driver memory budget reached"
                break
    summary = {"status": status, "epochs": completed, "best_validation_accuracy": best_accuracy,
               "seconds": time.time()-started, "checkpoint": str(out/"best.pt") if (out/"best.pt").exists() else None,
               "note": "Imitation accuracy is not game strength or an improvement over the teacher."}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--games", type=int, default=128)
    c.add_argument("--seed", type=int, default=10000000)
    c.add_argument("--rules", type=Path, required=True)
    c.add_argument("--max-seconds", type=float, default=120)
    f = sub.add_parser("fit")
    f.add_argument("--dataset", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--epochs", type=int, default=8)
    f.add_argument("--device", choices=("cpu", "mps"), default="mps")
    f.add_argument("--max-seconds", type=float, default=600)
    f.add_argument("--deadline", type=float)
    args = p.parse_args()
    if args.command == "collect":
        result = collect(args.output, args.games, args.seed, json.loads(args.rules.read_text()), args.max_seconds)
    else:
        result = fit(args.dataset, args.output, args.epochs, args.device, args.max_seconds, args.deadline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
