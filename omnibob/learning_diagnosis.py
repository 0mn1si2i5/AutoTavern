"""Bounded, offline PPO diagnostics; never exports or promotes updated policies.

The diagnostic loss intentionally matches train.py. Saved rollouts contain only
player observations, sampled actions and training rewards, not engine saves.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .model import EntityActorCritic, ModelConfig
from .train import gae, load_checkpoint, tensors
from .training_env import VectorGames


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_from(state, device, key="model"):
    model = EntityActorCritic(ModelConfig(**state["model_config"])).to(device)
    model.load_state_dict(state[key])
    return model.eval()


def loss_parts(model, data, ids, value_coef=.5, reference_coef=.03):
    logits, values = model(data["features"][ids], data["cards"][ids], data["action_mask"][ids])
    logp = logits.log_softmax(-1)
    selected = logp.gather(1, data["actions"][ids, None]).squeeze(-1)
    logratio = selected-data["logp"][ids]
    ratio = logratio.exp()
    advantage = data["advantages"][ids]
    return {
        "actor": -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean(),
        "critic": value_coef*(values-data["returns"][ids]).square().mean(),
        "entropy": .001*(logp.exp()*logp).sum(-1).mean(),
        "reference": reference_coef*(data["reference_logp"][ids].exp()
                      *(data["reference_logp"][ids]-logp)).sum(-1).mean(),
    }, logp, values, ((ratio-1)-logratio).mean()


def future_terminal(rewards, dones, cost):
    """Ground-truth remaining return only where a terminal is inside this rollout."""
    targets = np.full(rewards.shape, np.nan)
    distance = np.full(rewards.shape, -1, dtype=np.int32)
    for j in range(rewards.shape[1]):
        end = None
        for t in reversed(range(rewards.shape[0])):
            if dones[t, j]:
                end = t
            if end is not None:
                distance[t, j] = end-t
                targets[t, j] = rewards[end, j]+cost-cost*(end-t+1)
    return targets, distance


def collect(state, model, reference, seed, horizon, device, deadline):
    env = VectorGames(8, seed, state["config"], workers=1, opponents="mixed",
                      scheduler="shuffled_rounds", terminate_on_elimination=True)
    rng = torch.Generator().manual_seed(seed)
    samples = {k: [] for k in ("features", "cards", "action_mask", "actions", "logp", "values",
                               "rewards", "dones", "reference_logp")}
    records = []
    try:
        obs = env.observe()
        for t in range(horizon):
            if time.monotonic() > deadline:
                raise TimeoutError("diagnostic time budget exhausted")
            with torch.no_grad():
                logits, values = model(*tensors(obs, device))
                logp = logits.log_softmax(-1).cpu()
                ref, _ = reference(*tensors(obs, device))
                actions = torch.multinomial(logp.exp(), 1, generator=rng).squeeze(-1)
            for k in obs:
                samples[k].append(obs[k])
            samples["actions"].append(actions.numpy())
            samples["logp"].append(logp.gather(1, actions[:, None]).squeeze(-1).numpy())
            samples["values"].append(values.cpu().numpy())
            samples["reference_logp"].append(ref.log_softmax(-1).cpu().numpy())
            rewards, dones, finished = env.step(actions.tolist())
            # Independently verify that the reward exactly follows the declared placement objective.
            for record in finished:
                idx = record["environment_index"]
                assert record["status"] == "completed" and dones[idx]
                assert np.isclose(rewards[idx], 1-2*(record["placement"]-1)/7)
                records.append({"rollout_step": t, **record})
            assert np.all(rewards[~dones] == 0)
            samples["rewards"].append(rewards-.0003)
            samples["dones"].append(dones)
            obs = env.observe()
        with torch.no_grad():
            _, boot = model(*tensors(obs, device))
        samples = {k: np.stack(v) for k, v in samples.items()}
        samples["bootstrap"] = boot.cpu().numpy()
        return samples, records
    finally:
        env.close()


def prepare(samples, device):
    adv, ret = gae(samples["rewards"], samples["values"], samples["dones"], samples["bootstrap"], 1., .995)
    total = adv.size
    flat = {k: v.reshape(total, *v.shape[2:]) for k, v in samples.items() if k != "bootstrap"}
    flat["advantages"] = (adv.reshape(-1)-adv.mean())/(adv.std()+1e-8)
    flat["returns"] = ret.reshape(-1)
    data = {k: torch.as_tensor(v, device=device) for k, v in flat.items()}
    data["cards"], data["actions"] = data["cards"].long(), data["actions"].long()
    return data, adv, ret


def gradient_summary(model, data, ids):
    """Full-batch and minibatch gradients; report shared-backbone conflict separately."""
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]
    heads = {"general", "buy", "sell", "activate", "choose", "play", "magnetize", "move"}
    group = ["actor_heads" if n.split('.')[0] in heads else "value_head" if n.startswith("value.")
             else "backbone" for n in names]
    ranges, start = [], 0
    for p in params:
        ranges.append(slice(start, start+p.numel()))
        start += p.numel()
    sums = {k: torch.zeros(start) for k in ("actor", "critic", "entropy", "reference")}
    per_batch, scalars = [], []
    for chunk in ids.split(128):
        losses, _, _, _ = loss_parts(model, data, chunk)
        grads = {}
        for k, loss in losses.items():
            raw = torch.autograd.grad(loss, params, retain_graph=k != "reference", allow_unused=True)
            vector = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
                                for p, g in zip(params, raw)])
            grads[k] = vector
            sums[k] += vector*(len(chunk)/len(ids))
        per_batch.append({k: float(g.norm()) for k, g in grads.items()})
        scalars.append({k: float(v.detach().cpu()) for k, v in losses.items()})

    def stats(g):
        a = g["actor"]
        aa = float(a.dot(a))
        return {
            "norms": {k: float(v.norm()) for k, v in g.items()},
            "cosine_with_actor": {k: float(a.dot(v)/(a.norm()*v.norm()+1e-20)) for k, v in g.items()},
            "projection_on_actor": {k: float(a.dot(v))/(aa+1e-20) for k, v in g.items()},
        }

    result = {"all": stats(sums), "minibatch_norm_median": {k: float(np.median([x[k] for x in per_batch]))
                                                              for k in sums},
              "loss_mean": {k: float(np.mean([x[k] for x in scalars])) for k in sums}}
    for name in set(group):
        result[name] = stats({k: torch.cat([v[s] for s, g in zip(ranges, group) if g == name])
                             for k, v in sums.items()})
    return result


@torch.no_grad()
def evaluate(model, data, ids, initial_logp=None):
    totals = {k: 0. for k in ("actor_loss", "value_mse", "reference_kl", "entropy", "old_policy_kl",
                              "greedy_changed", "mean_abs_probability_change")}
    for chunk in ids.split(128):
        parts, lp, v, _ = loss_parts(model, data, chunk)
        weight = len(chunk)/len(ids)
        totals["actor_loss"] += weight*float(parts["actor"].cpu())
        totals["value_mse"] += weight*float(((v-data["returns"][chunk])**2).mean().cpu())
        totals["reference_kl"] += weight*float((parts["reference"]/.03).cpu())
        totals["entropy"] -= weight*float((parts["entropy"]/.001).cpu())
        if initial_logp is not None:
            old = initial_logp[chunk]
            totals["old_policy_kl"] += weight*float((old.exp()*(old-lp)).sum(-1).mean().cpu())
            totals["greedy_changed"] += weight*float((old.argmax(-1) != lp.argmax(-1)).float().mean().cpu())
            totals["mean_abs_probability_change"] += weight*float((old.exp()-lp.exp()).abs().sum(-1).mean().cpu())
    return totals


def fit_normalization(raw, fit):
    center, scale = raw[fit].mean(), raw[fit].std(unbiased=False)+1e-8
    return (raw-center)/scale, center, scale


def short_updates(state, data, device, seed, deadline, raw_advantages):
    """One fixed two-epoch optimizer pass, same rows/order/state for each ablation.

    Whole environment streams 0..5 are fit; streams 6..7 are held out. This is a
    surrogate-objective diagnostic, not a full-game policy comparison.
    """
    n = len(data["actions"])
    fit = torch.arange(n, device=device)[torch.arange(n, device=device) % 8 < 6]
    held = torch.arange(n, device=device)[torch.arange(n, device=device) % 8 >= 6]
    # Held-out rewards must not enter even the fitting batch's normalization constants.
    raw = torch.as_tensor(raw_advantages.reshape(-1), device=device)
    normalized, center, scale = fit_normalization(raw, fit)
    data = {**data, "advantages": normalized}
    original = model_from(state, device)
    with torch.no_grad():
        old = torch.cat([original(data["features"][i:i+128], data["cards"][i:i+128],
                                 data["action_mask"][i:i+128])[0].log_softmax(-1) for i in range(0, n, 128)])
    baseline = {"fit": evaluate(original, data, fit, old), "held": evaluate(original, data, held, old)}
    del original
    result = {"before": baseline, "fit_rows": len(fit), "held_rows": len(held), "variants": {},
              "normalization": {"source": "fit streams only", "mean": float(center.cpu()), "scale": float(scale.cpu())}}
    for label, vc, rc in (("original", .5, .03), ("no_reference", .5, 0.), ("no_critic", 0., .03)):
        model = model_from(state, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, eps=1e-5)
        # Preserve Adam moments for continuation checkpoints. Frozen policy exports have no optimizer.
        if "optimizer" in state:
            optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        rng = torch.Generator().manual_seed(seed+555)
        count, stopped, norms = 0, False, []
        for _ in range(2):
            order = fit[torch.randperm(len(fit), generator=rng).to(device)]
            for chunk in order.split(128):
                if time.monotonic() > deadline:
                    raise TimeoutError("diagnostic time budget exhausted")
                parts, _, _, kl = loss_parts(model, data, chunk, vc, rc)
                if float(kl.detach().cpu()) > .02:
                    stopped = True
                    break
                optimizer.zero_grad(set_to_none=True)
                sum(parts.values()).backward()
                norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).cpu()))
                optimizer.step()
                count += 1
            if stopped:
                break
        result["variants"][label] = {"optimizer_batches": count, "early_kl_stop": stopped,
                                    "mean_grad_norm": float(np.mean(norms)) if norms else None,
                                    "fit": evaluate(model, data, fit, old), "held": evaluate(model, data, held, old)}
        del model, optimizer
    result["optimizer_state"] = "retained" if "optimizer" in state else "fresh"
    return result


def rollout_stats(samples, adv, ret, records):
    truth, distance = future_terminal(samples["rewards"], samples["dones"], .0003)
    known = np.isfinite(truth)
    stats = {"rows": int(adv.size), "terminals": len(records), "terminal_covered_fraction": float(known.mean()),
             "advantage_std": float(adv.std()), "value_std": float(samples["values"].std()),
             "gae_return_std": float(ret.std()), "reward_sum": float(samples["rewards"].sum()),
             "value_mse_gae": float(((samples["values"]-ret)**2).mean()), "distance_bins": {}}
    for name, condition in (("all", known), ("0_31", known & (distance < 32)),
                             ("32_127", known & (distance >= 32) & (distance < 128)),
                             ("128_plus", known & (distance >= 128))):
        if not condition.any():
            continue
        y, v = truth[condition], samples["values"][condition]
        stats["distance_bins"][name] = {"n": int(condition.sum()), "target_std": float(y.std()),
            "value_mse_terminal_return": float(np.mean((y-v)**2)),
            "value_bias": float(np.mean(v-y)), "explained_variance": float(1-np.var(y-v)/(np.var(y)+1e-20)),
            "value_correlation": float(np.corrcoef(y, v)[0, 1]) if y.std() > 0 and v.std() > 0 else None,
            "gae_target_terminal_mse": float(np.mean((ret[condition]-y)**2))}
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="mps", choices=("mps", "cpu"))
    parser.add_argument("--horizon", type=int, default=512)
    parser.add_argument("--max-seconds", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.horizon <= 512 or not 1 <= args.max_seconds <= 1200:
        raise ValueError("bounded diagnostics require horizon <=512 and seconds <=1200")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+args.max_seconds
    write_json(out/"manifest.json", {**vars(args), "checkpoint_sha256": sha(args.checkpoint),
                                    "reference_sha256": sha(args.reference), "diagnostic_source_sha256": sha(__file__),
                                    "training_semantics": "mixed/shuffled/raw-mask/own-elimination; gamma1 lambda.995 cost.0003",
                                    "weights_saved": False, "strength_evidence": False})
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    state, ref_state = load_checkpoint(args.checkpoint), load_checkpoint(args.reference)
    model, reference = model_from(state, args.device), model_from(ref_state, args.device)
    if state.get("reference_model") is not None:
        for key, value in state["reference_model"].items():
            if not torch.equal(value, ref_state["model"][key]):
                raise ValueError("supplied reference differs from checkpoint reference")
    try:
        samples, records = collect(state, model, reference, args.seed, args.horizon, args.device, deadline)
        np.savez_compressed(out/"rollout.npz", **samples)
        write_json(out/"terminals.json", records)
        print(json.dumps({"stage": "collected", "terminals": len(records), "seconds": time.monotonic()-started}), flush=True)
        data, adv, ret = prepare(samples, args.device)
        stats = rollout_stats(samples, adv, ret, records)
        write_json(out/"rollout-stats.json", stats)
        gradients = gradient_summary(model, data, torch.arange(adv.size, device=args.device))
        write_json(out/"gradients.json", gradients)
        print(json.dumps({"stage": "gradients", "all": gradients["all"], "seconds": time.monotonic()-started}), flush=True)
        del model, reference
        updates = short_updates(state, data, args.device, args.seed, deadline, adv)
        write_json(out/"short-updates.json", updates)
        write_json(out/"summary.json", {"status": "completed", "seconds": time.monotonic()-started,
                                        "rows": adv.size, "terminals": len(records), "weights_saved": False})
    except BaseException as exc:
        write_json(out/"failure.json", {"error": repr(exc), "seconds": time.monotonic()-started})
        raise


if __name__ == "__main__":
    main()
