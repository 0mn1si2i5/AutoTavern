"""Synthetic visible-state learnability probe, not a game-strength benchmark.

Task: conditional on buying, prefer the strictly larger copy of the SAME card at
the SAME price. Labels use only own visible shop stats. All other cards, effects
and stats in that shop match. The real legal mask is preserved; the supervised
loss is conditional over legal BUY actions. No trained policy is exported.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import taverngym as tg

from .learning_diagnosis import model_from, sha, write_json
from .observation_v2 import ObservationV2
from .train import load_checkpoint, tensors


def make_states(seed, count, card, high_range=(15, 26)):
    rng = np.random.default_rng(seed)
    result = {k: [] for k in ("features", "cards", "action_mask", "labels")}
    metadata = []
    for i in range(count):
        game = tg.Game(seed+i)
        for seat in range(8):
            game.step(seat, 30)
            if seat == 0:
                # Set the controlled hero BEFORE the hero-selection barrier starts turn1.
                # Otherwise Faelin/A.F.Kay legitimately disallow buying in this fixture.
                game.debug(json.dumps(dict(cmd="set_hero", p=0, hero="TB_BaconShop_HERO_11")))
        for _ in range(16):
            if json.loads(game.player_view(0))["me"]["pending"] is None:
                break
            legal = np.flatnonzero(np.frombuffer(game.legal_mask(0), dtype=np.uint8))
            choices = legal[(legal >= 30) & (legal < 46)]
            game.step(0, int(choices[0]))
        else:
            raise RuntimeError("unresolved starting choice")
        for zone in ("board", "hand", "shop"):
            game.debug(json.dumps(dict(cmd="clear_zone", p=0, zone=zone)))
        game.debug(json.dumps(dict(cmd="set_player", p=0, gold=3)))
        size = int(rng.integers(2, 7))
        target = int(rng.integers(size))
        small = int(rng.integers(1, 11))
        large = int(rng.integers(*high_range))
        for slot in range(size):
            stat = large if slot == target else small
            game.debug(json.dumps(dict(cmd="add_minion", p=0, zone="shop", card=card, atk=stat, hp=stat)))
        obs = ObservationV2.from_game(game, 0)
        own = json.loads(game.player_view(0))["me"]
        assert obs.action_mask[8:8+size].all(), f"buy unavailable in fixture seed{seed+i}"
        assert len({c['id'] for c in own['shop']}) == 1
        assert {c['price'] for c in own['shop']} == {3}
        assert all(own['shop'][target][k] > c[k] for s, c in enumerate(own['shop']) if s != target
                   for k in ('atk', 'hp'))
        # Execute the labeled action once: check the observation/action encoding against the engine.
        game.step(0, 8+target)
        after = json.loads(game.player_view(0))["me"]
        assert len(after['hand']) == 1 and after['hand'][0]['atk'] == large and after['hand'][0]['hp'] == large
        for key in ("features", "cards", "action_mask"):
            result[key].append(getattr(obs, key))
        result['labels'].append(target)
        metadata.append(dict(seed=seed+i, target=target, shop_size=size, small=small, large=large, card=card))
    return {k: np.stack(v) for k, v in result.items()}, metadata


@torch.no_grad()
def score(model, data):
    correct, probability, full_buy, full_correct, nll = [], [], [], [], []
    for i in range(0, len(data['labels']), 64):
        logits, _ = model(data['features'][i:i+64], data['cards'][i:i+64], data['action_mask'][i:i+64])
        labels = data['labels'][i:i+64]
        lp = logits[:, 8:16].log_softmax(-1)
        correct.extend((lp.argmax(-1) == labels).cpu().tolist())
        target = lp.gather(1, labels[:, None]).squeeze(-1)
        probability.extend(target.exp().cpu().tolist())
        nll.extend((-target).cpu().tolist())
        best = logits.argmax(-1)
        full_buy.extend(((best >= 8) & (best < 16)).cpu().tolist())
        full_correct.extend((best == labels+8).cpu().tolist())
    return dict(conditional_buy_accuracy=float(np.mean(correct)), target_probability=float(np.mean(probability)),
                nll=float(np.mean(nll)), full_policy_buys=float(np.mean(full_buy)),
                full_policy_buys_strongest=float(np.mean(full_correct)), rows=len(correct))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        raise ValueError('at most200 diagnostic updates')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(127000000)
    started = time.monotonic()
    protocol = dict(checkpoint=args.checkpoint, checkpoint_sha256=sha(args.checkpoint),
                    source_sha256=sha(__file__), steps=args.steps, lr=.0003, batch_size=64,
                    objective='conditional BUY cross entropy; strictly dominant same-card stats',
                    controlled_hero='Ragnaros, set before turn1; no skipped-turn heroes',
                    weights_saved=False, strength_evidence=False,
                    datasets={'fit': [127000000, 256, 'BG26_OG_248', [15, 26]],
                              'held_seed': [128000000, 256, 'BG26_OG_248', [15, 26]],
                              'held_card': [129000000, 256, 'BG28_603t', [15, 26]],
                              'held_scale': [130000000, 256, 'BG26_OG_248', [30, 81]]})
    write_json(out/'protocol.json', protocol)
    try:
        datasets = {}
        for name, (seed, count, card, high) in protocol['datasets'].items():
            raw, metadata = make_states(seed, count, card, high)
            np.savez_compressed(out/(name+'.npz'), **raw)
            write_json(out/(name+'-states.json'), metadata)
            f, c, m = tensors(raw, 'mps')
            datasets[name] = dict(features=f, cards=c, action_mask=m,
                                  labels=torch.as_tensor(raw['labels'], device='mps').long())
        state = load_checkpoint(args.checkpoint)
        model = model_from(state, 'mps')
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003, eps=1e-5)
        rng = torch.Generator().manual_seed(127000001)
        history = []
        for step in range(args.steps+1):
            if time.monotonic()-started > 300:
                raise TimeoutError('microdiagnosis five minute limit')
            if step in (0, 25, 50, 100, args.steps):
                record = dict(step=step, seconds=time.monotonic()-started,
                              scores={name: score(model, data) for name, data in datasets.items()})
                history.append(record)
                with (out/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                print(json.dumps(record), flush=True)
            if step == args.steps:
                break
            ids = torch.randperm(256, generator=rng)[:64].to('mps')
            data = datasets['fit']
            logits, _ = model(data['features'][ids], data['cards'][ids], data['action_mask'][ids])
            loss = torch.nn.functional.cross_entropy(logits[:, 8:16], data['labels'][ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        write_json(out/'summary.json', dict(status='completed', seconds=time.monotonic()-started,
                                           history=history, weights_saved=False))
    except BaseException as exc:
        write_json(out/'failure.json', dict(error=repr(exc), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
