"""Read saved own-visible PPO diagnostic batches; measure independent-stream signal."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .learning_diagnosis import loss_parts, model_from, prepare, write_json
from .train import gae, load_checkpoint


def actor_gradient(model, data, ids):
    params = list(model.parameters())
    total = None
    for chunk in ids.split(128):
        loss = loss_parts(model, data, chunk)[0]['actor']
        raw = torch.autograd.grad(loss, params, allow_unused=True)
        grad = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten().cpu()
                          for p, g in zip(params, raw)])
        total = grad*len(chunk)/len(ids) if total is None else total+grad*len(chunk)/len(ids)
    return total


def independent_gae(samples):
    """Direct sum of TD residuals, independent of trainer's backward recurrence."""
    r, v, done = (samples[k] for k in ('rewards', 'values', 'dones'))
    result = np.zeros(r.shape, dtype=np.float64)
    for t in range(len(r)):
        alive = np.ones(r.shape[1], dtype=bool)
        weight = 1.
        for k in range(t, len(r)):
            nxt = samples['bootstrap'] if k == len(r)-1 else v[k+1]
            delta = r[k]+(~done[k])*nxt-v[k]
            result[t] += alive*weight*delta
            alive &= ~done[k]
            if not alive.any():
                break
            weight *= .995
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = root/'signal.json'
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    results = {}
    for path in sorted(root.glob('*/rollout.npz')):
        manifest = json.loads((path.parent/'manifest.json').read_text())
        samples = dict(np.load(path))
        model = model_from(load_checkpoint(manifest['checkpoint']), 'mps')
        data, adv, _ = prepare(samples, 'mps')
        n = len(data['actions'])
        ids = torch.arange(n, device='mps')
        gs = torch.stack([actor_gradient(model, data, ids[ids % 8 == i]) for i in range(8)])
        norms = gs.norm(dim=1)
        cosines = (gs @ gs.T)/(norms[:, None]*norms[None, :]+1e-20)
        pairs = cosines[torch.triu_indices(8, 8, 1).unbind()]
        avg = gs.mean(0)
        noise_of_mean = float((gs-avg).square().sum(1).sum()/(8*7))
        squared_mean = float(avg.square().sum())
        fit, held = gs[:6].mean(0), gs[6:].mean(0)
        fit_held_cos = float(fit.dot(held)/(fit.norm()*held.norm()+1e-20))
        exact = independent_gae(samples)
        max_error = float(np.max(np.abs(exact-adv)))
        assert max_error < 2e-5
        results[path.parent.name] = dict(
            pairwise_cosine_mean=float(pairs.mean()), pairwise_cosine_median=float(pairs.median()),
            pairwise_positive_fraction=float((pairs > 0).float().mean()),
            fit_held_actor_cosine=fit_held_cos,
            environment_gradient_norms=norms.tolist(), cosine_matrix=cosines.tolist(),
            squared_mean_gradient=squared_mean, estimated_noise_variance_of_mean=noise_of_mean,
            estimated_signal_squared=squared_mean-noise_of_mean,
            signal_to_noise=(squared_mean-noise_of_mean)/(noise_of_mean+1e-20),
            note='8 stream estimates, includes seat/episode heterogeneity; noisy signal estimate can be negative',
            independent_gae_max_abs_error=max_error,
        )
        print(path.parent.name, json.dumps({k: v for k, v in results[path.parent.name].items()
                                         if k not in ('environment_gradient_norms', 'cosine_matrix')}), flush=True)
    write_json(output, results)


if __name__ == '__main__':
    main()
