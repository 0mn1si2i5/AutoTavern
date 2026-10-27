"""Bounded simulator and synthetic learning-kernel probes; never long training."""
import json
import os
import platform
import resource
import subprocess
import time

import numpy as np
import taverngym as tg

from .compat import check_compatibility, provenance


def memory(device=None):
    rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
    result = {"rss_bytes": rss, "process_peak_rss_bytes": peak}
    if device == "mps":
        import torch
        result.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_bytes=torch.mps.driver_allocated_memory())
    return result


def machine():
    result = {"architecture": platform.machine(), "platform": platform.platform(),
              "logical_cpus": os.cpu_count(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        for key in ("hw.memsize", "hw.physicalcpu", "machdep.cpu.brand_string"):
            result[key] = subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        # Select only hardware fields, never serialize serial numbers or hardware UUIDs.
        raw = json.loads(subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"], text=True))
        result["gpus"] = [{k: v for k, v in gpu.items()
                            if k in ("sppci_model", "sppci_cores", "spdisplays_metal")}
                           for gpu in raw["SPDisplaysDataType"]]
    return result


def simulator_probe():
    check_compatibility()
    rows = []
    tg.run_bot_games(4, seed=100)
    start = time.perf_counter()
    turns = tg.run_bot_games(128, seed=100)
    elapsed = time.perf_counter() - start
    native = {"games": len(turns), "seconds": elapsed, "games_per_second": len(turns)/elapsed,
              "mean_turns": float(np.mean(turns)), **memory(),
              "note": "native bot runner; not Python policy throughput or live play"}
    for n, threads in ((1, 1), (8, 1), (8, 2), (32, 4)):
        env = tg.BatchEnv(n_games=n, seed=10000, controlled=(0,), threads=threads)
        start, ready_count, done_count, rejected = time.perf_counter(), 0, 0, 0
        for _ in range(128):
            obs = env.observe()
            actions = obs["action_mask"].argmax(axis=1)
            ready_count += int(obs["ready"].sum())
            _, done, info = env.step(actions)
            done_count += int(done.sum())
            rejected += int(info["rejected"].sum())
        seconds = time.perf_counter()-start
        rows.append({"n_games": n, "threads": threads, "batch_steps": 128,
                     "ready_decisions": ready_count, "episode_ends": done_count,
                     "rejected": rejected, "seconds": seconds,
                     "ready_decisions_per_second": ready_count/seconds, **memory()})
    return {"native_bots": native, "batch_first_legal": rows,
            "note": "Batch episodes stop at the learner's elimination; these are not full-game counts."}


def torch_probe():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        return {"available": False, "error": str(exc), "fallback": "CPU simulator remains usable"}
    torch.set_num_threads(4)
    torch.manual_seed(42)
    # Real player observations for the kernel probe; no reward or opponent fitting.
    env = tg.BatchEnv(8, seed=20000, threads=1)
    for _ in range(12):
        obs = env.observe()
        env.step(obs["action_mask"].argmax(axis=1))
    obs = env.observe()
    layout = dict(tg.layout())
    start = layout["GLOBAL_F"] + layout["N_QUESTS"] * layout["QUEST_F"] + layout["PENDING_KINDS"]
    board_end = start + 7 * 31
    item_end = board_end + 34 * 40
    card_start = 1 + layout["N_POWERS"] + 2 + 3*layout["N_QUESTS"] + layout["N_SECRETS"]

    class TinyEntityProbe(nn.Module):
        def __init__(self):
            super().__init__()
            self.card = nn.Embedding(tg.N_CARDS, 64, padding_idx=0)
            self.glob, self.board, self.item, self.opp = (nn.Linear(n, 64) for n in (53, 31, 40, 9))
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0, batch_first=True, norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.policy, self.value = nn.Linear(64, 245), nn.Linear(64, 1)

        def forward(self, f, c, mask):
            symlog = lambda x: x.sign() * torch.log1p(x.abs())
            own = torch.cat((self.board(symlog(f[:, start:board_end].reshape(-1, 7, 31))),
                             self.item(symlog(f[:, board_end:item_end].reshape(-1, 34, 40)))), 1)
            own = own + self.card(c[:, card_start:card_start+41])
            opponents = self.opp(symlog(f[:, item_end:item_end+63].reshape(-1, 7, 9)))
            tokens = torch.cat((self.glob(symlog(f[:, :53])).unsqueeze(1), own, opponents), 1)
            state = self.trunk(tokens)[:, 0]
            return self.policy(state).masked_fill(~mask, -1e9), self.value(state).squeeze(-1)

    base = TinyEntityProbe()
    results = {"torch": torch.__version__, "mps_built": torch.backends.mps.is_built(),
               "mps_available": torch.backends.mps.is_available(),
               "parameters": sum(p.numel() for p in base.parameters()),
               "architecture": "49 tokens, d=64, 2 layers, 4 heads, 128 FF, card identities; probe only",
               "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"), "runs": []}
    devices = ["cpu"] + (["mps"] if results["mps_available"] else [])
    reference = None
    for device in devices:
        for size in (8, 32, 128):
            record = {"device": device, "batch": size}
            try:
                model = TinyEntityProbe().to(device)
                model.load_state_dict(base.state_dict())
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                indices = np.arange(size) % 8
                f = torch.tensor(obs["features"][indices], device=device)
                c = torch.tensor(obs["cards"][indices], dtype=torch.long, device=device)
                mask = torch.tensor(obs["action_mask"][indices], device=device)
                actions = mask.int().argmax(dim=1)
                sync = torch.mps.synchronize if device == "mps" else lambda: None
                with torch.no_grad():
                    logits, value = model(f, c, mask)
                    if size == 8:
                        visible = logits.cpu().numpy()[obs["action_mask"]]
                        if device == "cpu":
                            reference = visible
                        else:
                            record["cpu_max_abs_logit_difference"] = float(np.max(np.abs(visible-reference)))
                    old_logp = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1).detach()
                advantages = torch.linspace(-1, 1, size, device=device)

                def update():
                    optimizer.zero_grad(set_to_none=True)
                    logits, values = model(f, c, mask)
                    logp = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
                    ratio = (logp-old_logp).exp()
                    policy = -torch.minimum(ratio*advantages, ratio.clamp(.8, 1.2)*advantages).mean()
                    loss = policy + .5*(values-advantages).square().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    return loss

                for _ in range(2):
                    update()
                sync()
                t = time.perf_counter()
                for _ in range(8):
                    loss = update()
                sync()
                elapsed = time.perf_counter()-t
                finite = bool(torch.isfinite(loss).item()) and all(
                    bool(torch.isfinite(p.grad).all().item()) for p in model.parameters() if p.grad is not None)
                record.update(seconds=elapsed, updates=8, update_ms=elapsed/8*1000,
                              samples_per_second=size*8/elapsed, finite_loss_and_gradients=finite,
                              final_synthetic_loss=float(loss.detach().cpu()), **memory())
                if device == "mps":
                    record.update(mps_allocated_bytes=torch.mps.current_allocated_memory(),
                                  mps_driver_bytes=torch.mps.driver_allocated_memory())
                del optimizer, model, f, c, mask
                if device == "mps":
                    torch.mps.empty_cache()
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            results["runs"].append(record)
    results["note"] = "Synthetic clipped-policy loss + value MSE on real observations; compatibility/speed only. No agent training, returns, GAE, checkpoint or strength claim. Peak RSS is cumulative within this process."
    return results


def benchmark(include_torch=True):
    result = {"machine": machine(), "provenance": provenance(json.loads(tg.default_config_json())),
              "environment": check_compatibility(), "simulator": simulator_probe()}
    if include_torch:
        result["torch"] = torch_probe()
    return result
