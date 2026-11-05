"""Bounded on-policy PPO, atomic checkpoints, and exact boundary resume on CPU."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .benchmark import memory
from .compat import ROOT, check_compatibility, effective_config, provenance
from .evaluate import emit
from .model import EntityActorCritic, ModelConfig
from .training_env import VectorGames
from .opponents import DESCRIPTIONS, LEARNED_PROFILES


@dataclass
class TrainConfig:
    seed: int = 100000
    n_envs: int = 32
    workers: int = 1
    horizon: int = 128
    epochs: int = 2
    minibatch: int = 128
    learning_rate: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = .95
    clip: float = .2
    target_kl: float = .03
    entropy_coef: float = .01
    value_coef: float = .5
    max_grad_norm: float = 1.0
    cpu_threads: int = 4
    decision_cost: float = 0.0
    reference_kl_coef: float = 0.0
    learner_repeat_limit: int = 0
    opponents: str = "scripted"
    opponent_checkpoint: str | None = None
    scheduler: str = "round_robin"
    terminate_on_elimination: bool = False


def gae(rewards, values, dones, bootstrap, gamma, lam):
    """Time-major arrays; done prevents both bootstrapping and cross-episode traces."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    carry = np.zeros_like(bootstrap, dtype=np.float32)
    for step in reversed(range(len(rewards))):
        next_value = bootstrap if step == len(rewards)-1 else values[step+1]
        alive = 1.0-dones[step].astype(np.float32)
        delta = rewards[step] + gamma*next_value*alive - values[step]
        carry = delta + gamma*lam*alive*carry
        advantages[step] = carry
    return advantages, advantages+values


def valid_fragments(dones, invalid_ends):
    """Exclude a censored episode's current-rollout tail, never invent a terminal reward.

    Earlier updates from this long episode used ordinary value bootstrapping and
    cannot be retroactively undone. Completed episodes and subsequent resets stay
    independent through the done boundaries.
    """
    valid = np.ones_like(dones, dtype=bool)
    start = np.zeros(dones.shape[1], dtype=int)
    for t in range(len(dones)):
        for env in np.flatnonzero(invalid_ends[t]):
            valid[start[env]:t+1, env] = False
        start[dones[t]] = t+1
    return valid


def tensors(obs, device):
    return (torch.as_tensor(obs["features"], device=device),
            torch.as_tensor(obs["cards"], dtype=torch.long, device=device),
            torch.as_tensor(obs["action_mask"], dtype=torch.bool, device=device))


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") not in ("omnibob-ppo-v1", "omnibob-policy-v1"):
        raise ValueError("unsupported OmniBob checkpoint")
    check_compatibility(state["environment"])
    implementation = state.get("model_source_sha256")
    if implementation is None:
        # Early local v0.2 probes stored the per-file hashes in the run manifest.
        manifest = Path(path).parent/"manifest.json"
        if manifest.exists():
            implementation = json.loads(manifest.read_text()).get("source_files", {}).get("omnibob/model.py")
    actual = hashlib.sha256((ROOT/"omnibob/model.py").read_bytes()).hexdigest()
    if implementation != actual:
        raise ValueError("missing or incompatible model implementation hash; retain the original manifest or migrate explicitly")
    return state


def train(output, config=None, model_config=None, updates=15, device="cpu", max_seconds=600,
          deadline=None, max_rss_gib=4, resume=None, initialize=None, checkpoint_every=5, rules=None,
          max_truncated_episodes=0):
    import taverngym as tg
    output = Path(output)
    if updates < 1 or max_seconds <= 0 or max_rss_gib <= 0 or checkpoint_every < 1:
        raise ValueError("positive training budgets required")
    if resume and initialize:
        raise ValueError("resume and initialize are mutually exclusive")
    if max_truncated_episodes < 0:
        raise ValueError("truncation budget must be nonnegative")
    torch.set_num_threads((config or TrainConfig()).cpu_threads)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable; explicitly choose --device cpu")
    started = time.time()
    stop_at = min(started+max_seconds, deadline if deadline is not None else float("inf"))
    if stop_at <= started:
        raise ValueError("training deadline already passed")
    prior = load_checkpoint(resume) if resume else None
    if prior and prior.get("error"):
        raise ValueError("failed training state; resume an earlier safe checkpoint")
    if prior and prior["format"] != "omnibob-ppo-v1":
        raise ValueError("deployment weights require --initialize; --resume requires a training checkpoint")
    if prior and prior.get("training_method", "ppo") != "ppo":
        raise ValueError("imitation weights require --initialize, not PPO --resume")
    config = TrainConfig(**prior["train_config"]) if prior else (config or TrainConfig())
    model_config = ModelConfig(**prior["model_config"]) if prior else (model_config or ModelConfig())
    if config.n_envs < 1 or config.horizon < 1 or config.epochs < 1 or config.minibatch < 1:
        raise ValueError("invalid training sizes")
    if config.opponents not in DESCRIPTIONS:
        raise ValueError("unknown training opponent profile")
    if (config.opponents in LEARNED_PROFILES) != (config.opponent_checkpoint is not None):
        raise ValueError("learned opponents require their frozen checkpoint")
    if config.opponent_checkpoint:
        config.opponent_checkpoint = str(Path(config.opponent_checkpoint).resolve())
    if not np.isfinite(config.decision_cost) or config.decision_cost < 0:
        raise ValueError("decision cost must be finite and nonnegative")
    if not np.isfinite(config.reference_kl_coef) or config.reference_kl_coef < 0:
        raise ValueError("reference KL coefficient must be finite and nonnegative")
    from .repeat_guard import runtime_config as repeat_runtime_config
    repeat_runtime_config(config.learner_repeat_limit)
    torch.set_num_threads(config.cpu_threads)
    torch.manual_seed(config.seed)
    rng = torch.Generator(device="cpu").manual_seed(config.seed+1)
    cfg = effective_config(prior["config"] if prior and rules is None else rules)
    identity = provenance(cfg)
    identity["frozen_opponent"] = ({"path": config.opponent_checkpoint,
        "sha256": hashlib.sha256(Path(config.opponent_checkpoint).read_bytes()).hexdigest()}
        if config.opponent_checkpoint else None)
    if prior and config.opponents != "scripted" and prior.get("opponent_policy_sha256") != identity["source_files"]["omnibob/policies.py"]:
        raise ValueError("frozen opponent implementation changed since checkpoint")
    model = EntityActorCritic(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
    update, steps = 0, 0
    if prior:
        if prior["config"] != cfg:
            raise ValueError("resume rules configuration mismatch")
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        rng.set_state(prior["sampling_rng"])
        torch.set_rng_state(prior["torch_rng"])
        update, steps = prior["update"], prior["steps"]
    elif initialize:
        initial = load_checkpoint(initialize)
        if initial.get("error"):
            raise ValueError("failed training state; initialize from an earlier safe checkpoint")
        if initial["model_config"] != asdict(model_config) or initial["config"] != cfg:
            raise ValueError("initial checkpoint architecture/configuration mismatch")
        model.load_state_dict(initial["model"])
    reference = None
    if config.reference_kl_coef > 0:
        reference_weights = prior.get("reference_model") if prior else (initial["model"] if initialize else None)
        if reference_weights is None:
            raise ValueError("reference KL requires --initialize or a saved reference model")
        torch_state = torch.get_rng_state()
        reference = EntityActorCritic(model_config).to(device)
        torch.set_rng_state(torch_state)
        reference.load_state_dict(reference_weights)
        reference.eval().requires_grad_(False)
    output.mkdir(parents=True, exist_ok=False)
    identity.update(train_config=asdict(config), model_config=asdict(model_config),
                    torch=str(torch.__version__), device=device,
                    parameters=sum(p.numel() for p in model.parameters()),
                    deadline_unix=stop_at, max_rss_gib=max_rss_gib, updates_requested=updates,
                    resume=str(resume) if resume else None, initialize=str(initialize) if initialize else None,
                    objective=f"own terminal utility 1-2*(place-1)/7 minus {config.decision_cost} per decision",
                    observation_version=2, scheduler=config.scheduler, opponents=config.opponents,
                    episode_end="own placement known" if config.terminate_on_elimination else "full game over")
    identity["truncation_handling"] = {"max_truncated_episodes": max_truncated_episodes,
        "policy": "stop" if max_truncated_episodes == 0 else "record, reset, discard current-rollout episode fragment",
        "limitation": "Earlier bootstrapped updates cannot be retroactively removed; no forced placement becomes a reward."}
    parent = resume or initialize
    identity["parent_checkpoint"] = ({"path": str(Path(parent).resolve()),
                                      "sha256": hashlib.sha256(Path(parent).read_bytes()).hexdigest()}
                                     if parent else None)
    identity["reference_checkpoint"] = ((prior.get("reference_checkpoint") if prior else identity["parent_checkpoint"])
                                        if reference is not None else None)
    identity["reference_regularizer"] = "KL(frozen initialization policy || current policy), legal actions only"
    identity["learner_runtime"] = repeat_runtime_config(config.learner_repeat_limit)
    (output/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    environment_type, learner_options = VectorGames, {}
    if config.learner_repeat_limit:
        from .training_repeat import GuardedVectorGames
        environment_type = GuardedVectorGames
        learner_options = {"learner_repeat_limit": config.learner_repeat_limit}
    env = environment_type(config.n_envs, config.seed, cfg, config.workers,
                      state=prior["environments"] if prior else None,
                      opponents=config.opponents, scheduler=config.scheduler,
                      terminate_on_elimination=config.terminate_on_elimination,
                      allow_truncated=max_truncated_episodes > 0, opponent_checkpoint=config.opponent_checkpoint,
                      **learner_options)
    obs = env.observe()
    target_update = update+updates
    stop_reason, error, completed_games, completed_episodes = "updates_completed", None, 0, 0
    truncated_episodes, discarded_decisions = 0, 0
    requested_signal = []
    original_handlers = {}

    def request_stop(signum, frame):
        requested_signal.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        original_handlers[signum] = signal.signal(signum, request_stop)

    def check_stop():
        if requested_signal:
            raise InterruptedError(f"signal {requested_signal[-1]}")
        if time.time() >= stop_at:
            raise InterruptedError("time budget reached")

    def save(archive=False):
        atomic_checkpoint(output/"latest.pt", {
            "format": "omnibob-ppo-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "training_method": "ppo",
            "model_config": asdict(model_config), "train_config": asdict(config),
            "environment": identity["environment"], "config": cfg, "source_sha256": identity["source_sha256"],
            "model_source_sha256": identity["source_files"]["omnibob/model.py"],
            "opponent_policy_sha256": identity["source_files"]["omnibob/policies.py"],
            "frozen_opponent": identity["frozen_opponent"],
            "parent_checkpoint": identity["parent_checkpoint"],
            "reference_model": reference.state_dict() if reference is not None else None,
            "reference_checkpoint": identity["reference_checkpoint"],
            "objective": identity["objective"],
            "runtime_config": repeat_runtime_config(config.learner_repeat_limit),
            "truncation_handling": identity["truncation_handling"],
            "update": update, "steps": steps, "sampling_rng": rng.get_state(),
            "torch_rng": torch.get_rng_state(), "environments": env.save(),
            "stop_reason": stop_reason, "error": error,
        })
        if archive:
            os.link(output/"latest.pt", output/f"update-{update:06d}.pt")

    try:
        with (output/"metrics.jsonl").open("w") as metrics, (output/"games.jsonl").open("w") as games:
            while update < target_update:
                check_stop()
                iteration_start = time.perf_counter()
                samples = {k: [] for k in ("features", "cards", "action_mask", "actions", "logp", "values", "rewards", "dones", "invalid_ends")}
                if reference is not None:
                    samples["reference_logp"] = []
                iteration_places = []
                iteration_full_games = 0
                iteration_truncated = 0
                model.eval()
                for t in range(config.horizon):
                    check_stop()
                    with torch.no_grad():
                        logits, values = model(*tensors(obs, device))
                        log_probs = logits.log_softmax(-1)
                        actions = torch.multinomial(log_probs.exp().cpu(), 1, generator=rng).squeeze(-1)
                        logp = log_probs.cpu().gather(1, actions[:, None]).squeeze(-1)
                        if reference is not None:
                            ref_logits, _ = reference(*tensors(obs, device))
                            samples["reference_logp"].append(ref_logits.log_softmax(-1).cpu().numpy())
                    for key in ("features", "cards", "action_mask"):
                        samples[key].append(obs[key])
                    samples["actions"].append(actions.numpy())
                    samples["logp"].append(logp.numpy())
                    samples["values"].append(values.cpu().numpy())
                    rewards, dones, records = env.step(actions.tolist())
                    rewards = rewards - config.decision_cost
                    samples["rewards"].append(rewards)
                    samples["dones"].append(dones)
                    invalid = np.zeros(config.n_envs, dtype=bool)
                    steps += config.n_envs
                    for record in records:
                        emit(games, {"step": steps, **record})
                        if record["status"] == "truncated":
                            invalid[record["environment_index"]] = True
                            iteration_truncated += 1
                            truncated_episodes += 1
                            continue
                        iteration_places.append(record["placement"])
                        iteration_full_games += int(record["full_game_completed"])
                    samples["invalid_ends"].append(invalid)
                    completed_episodes += sum(r["status"] == "completed" for r in records)
                    completed_games += sum(int(r["full_game_completed"]) for r in records)
                    obs = env.observe()
                    if truncated_episodes > max_truncated_episodes:
                        raise RuntimeError("simulator truncation budget exceeded; inspect recorded episodes")
                with torch.no_grad():
                    _, boot = model(*tensors(obs, device))
                samples = {k: np.stack(v) for k, v in samples.items()}
                valid = valid_fragments(samples["dones"], samples["invalid_ends"]).reshape(-1)
                discarded_decisions += int((~valid).sum())
                advantages, returns = gae(samples["rewards"], samples["values"], samples["dones"],
                                          boot.cpu().numpy(), config.gamma, config.gae_lambda)
                collection_seconds = time.perf_counter()-iteration_start
                collected = config.horizon*config.n_envs
                total = int(valid.sum())
                flat = {k: v.reshape(collected, *v.shape[2:])[valid] for k, v in samples.items()
                        if k not in ("rewards", "dones", "values", "invalid_ends")}
                f, c, mask = tensors(flat, device)
                actions = torch.as_tensor(flat["actions"], dtype=torch.long, device=device)
                old_logp = torch.as_tensor(flat["logp"], device=device)
                ref_logp = torch.as_tensor(flat["reference_logp"], device=device) if reference is not None else None
                adv = torch.as_tensor(advantages.reshape(-1)[valid], device=device)
                if total:
                    adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
                targets = torch.as_tensor(returns.reshape(-1)[valid], device=device)
                training_metrics, early_stop = [], False
                model.train()
                for epoch in range(config.epochs):
                    order = torch.randperm(total, generator=rng)
                    for start in range(0, total, config.minibatch):
                        check_stop()
                        ids = order[start:start+config.minibatch].to(device)
                        logits, value = model(f[ids], c[ids], mask[ids])
                        log_probs = logits.log_softmax(-1)
                        logp = log_probs.gather(1, actions[ids, None]).squeeze(-1)
                        logratio = logp-old_logp[ids]
                        ratio = logratio.exp()
                        actor_loss = -torch.minimum(ratio*adv[ids], ratio.clamp(1-config.clip, 1+config.clip)*adv[ids]).mean()
                        value_loss = (value-targets[ids]).square().mean()
                        entropy = -(log_probs.exp()*log_probs).sum(-1).mean()
                        kl = ((ratio-1)-logratio).mean()
                        if float(kl.detach().cpu()) > config.target_kl:
                            early_stop = True
                            break
                        loss = actor_loss+config.value_coef*value_loss-config.entropy_coef*entropy
                        reference_kl = torch.zeros_like(actor_loss)
                        if ref_logp is not None:
                            reference_kl = (ref_logp[ids].exp()*(ref_logp[ids]-log_probs)).sum(-1).mean()
                            loss = loss+config.reference_kl_coef*reference_kl
                        if not bool(torch.isfinite(loss).item()):
                            raise FloatingPointError("nonfinite PPO loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                                  error_if_nonfinite=True)
                        optimizer.step()
                        training_metrics.append([float(x.detach().cpu()) for x in
                                                 (actor_loss, value_loss, entropy, kl, grad_norm,
                                                  ((ratio-1).abs()>config.clip).float().mean(), reference_kl)])
                    if early_stop:
                        break
                update += 1
                resource = memory(device)
                duration = time.perf_counter()-iteration_start
                mean = np.mean(training_metrics, axis=0).tolist() if training_metrics else [None]*7
                row = {"update": update, "steps": steps, "seconds": duration,
                       "collection_seconds": collection_seconds, "decisions_per_second": collected/duration,
                       "trainable_decisions": total, "discarded_decisions": collected-total,
                       "truncated_episodes": iteration_truncated,
                       "completed_games": iteration_full_games, "completed_episodes": len(iteration_places),
                       "placements": iteration_places,
                       "mean_placement": float(np.mean(iteration_places)) if iteration_places else None,
                       "reward_sum": float(samples["rewards"].sum()), "early_kl_stop": early_stop,
                       "optimizer_batches": len(training_metrics), **resource,
                       **dict(zip(("actor_loss", "value_loss", "entropy", "approx_kl", "grad_norm", "clip_fraction", "reference_kl"), mean))}
                if config.learner_repeat_limit:
                    row["learner_mask_stats_cumulative"] = dict(env.learner_mask_stats)
                emit(metrics, row)
                print(json.dumps({k: row[k] for k in ("update", "steps", "seconds", "completed_episodes", "completed_games", "mean_placement", "approx_kl")}), flush=True)
                if max(resource["rss_bytes"], resource.get("mps_driver_bytes", 0)) > max_rss_gib*1024**3:
                    raise InterruptedError("RSS or MPS driver memory budget reached")
                if update % checkpoint_every == 0:
                    save(archive=True)
    except InterruptedError as exc:
        stop_reason = str(exc)
    except BaseException as exc:
        stop_reason, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        save()
        env.close()
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        summary = {"status": stop_reason, "error": error, "updates": update, "steps": steps,
                   "completed_games_this_run": completed_games, "seconds": time.time()-started,
                   "completed_episodes_this_run": completed_episodes,
                   "truncated_episodes_this_run": truncated_episodes,
                   "discarded_decisions_this_run": discarded_decisions,
                   "checkpoint": str(output/"latest.pt"), "device": device, **memory(device)}
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=15)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    p.add_argument("--max-seconds", type=float, default=600)
    p.add_argument("--deadline", type=float)
    p.add_argument("--max-rss-gib", type=float, default=4)
    p.add_argument("--max-truncated-episodes", type=int, default=0,
                   help="default stop; positive budget records/resets truncations and discards their rollout fragments")
    p.add_argument("--resume", type=Path)
    p.add_argument("--initialize", type=Path)
    p.add_argument("--config", type=Path, help="JSON TrainConfig overrides (resume uses saved config)")
    p.add_argument("--rules", type=Path, help="JSON game rules, e.g. configs/arena-v2.json")
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--horizon", type=int, default=128)
    p.add_argument("--opponents", choices=tuple(DESCRIPTIONS), default="scripted")
    p.add_argument("--opponent-checkpoint", type=Path)
    p.add_argument("--scheduler", choices=("round_robin", "shuffled_rounds"), default="round_robin")
    args = p.parse_args()
    overrides = json.loads(args.config.read_text()) if args.config else {}
    cfg = TrainConfig(**{**dict(seed=args.seed, n_envs=args.envs, workers=args.workers,
                               cpu_threads=args.cpu_threads,
                               horizon=args.horizon, opponents=args.opponents, scheduler=args.scheduler,
                               opponent_checkpoint=str(args.opponent_checkpoint) if args.opponent_checkpoint else None), **overrides})
    result = train(args.output, cfg, updates=args.updates, device=args.device,
                   max_seconds=args.max_seconds, deadline=args.deadline, max_rss_gib=args.max_rss_gib,
                   resume=args.resume, initialize=args.initialize,
                   max_truncated_episodes=args.max_truncated_episodes,
                   rules=json.loads(args.rules.read_text()) if args.rules else None)
    print(json.dumps(result, indent=2))
    return 1 if result["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
