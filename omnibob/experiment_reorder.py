"""Bounded LOCAL paired comparison of frozen deployment and movement controllers."""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

from .compare import compare
from .compat import provenance
from .evaluate import emit, run_game, schedule, summarize
from .learned import load_policy
from .reorder_exit import load_reorder_policy


def load_variant(checkpoint, variant):
    if variant == "compound":
        from .cancel_memory import load_compound_policy
        return load_compound_policy(checkpoint)
    if variant == "cycle_exit":
        return load_reorder_policy(checkpoint)
    if variant in ("baseline", "cap7"):
        return load_policy(checkpoint, **({"move_budget": 7} if variant == "cap7" else {}))
    raise ValueError(f"unknown experiment variant: {variant}")


class MeasuredPolicy:
    observation_version = 2

    def __init__(self, base):
        self.base = base
        self.reset()

    def __getattr__(self, key):
        return getattr(self.base, key)

    def reset(self):
        self.base.reset()
        self.metrics = Counter()
        self.turns, self.cap_turns = set(), set()

    def act(self, observation):
        decision = self.base.act(observation)
        turn = round(float(observation.features[0])*20)
        self.turns.add(turn)
        self.metrics["moves"] += 196 <= decision.action < 245
        self.metrics["cancels"] += decision.action == 7
        self.metrics["hand_plays"] += 46 <= decision.action < 126
        self.metrics["cancel_memory_masked_decisions"] += getattr(self.base, 'last_blocked', 0) > 0
        self.metrics["cycle_exits"] += "reorder exit: cycle=True" in decision.reason and decision.action < 196
        # Count reaching the platform's 100-operation boundary, not an illegal action.
        if observation.features[3227] >= 99 and decision.action != 0:
            if turn not in self.cap_turns:
                self.metrics["gold_when_reaching_cap"] += round(float(observation.features[7])*10)
            self.cap_turns.add(turn)
        if observation.features[3227] >= 100:
            self.metrics["decisions_at_or_above_cap"] += 1
        if decision.action == 0:
            self.metrics["voluntary_ends"] += 1
            self.metrics["unspent_gold_at_end"] += round(float(observation.features[7])*10)
            self.metrics["underfilled_ends"] += int(observation.features[68:285:31].sum()) < 7
        return decision


def initialize(checkpoint, variants, deadline, opponents, scheduler):
    global POLICIES, DEADLINE, OPPONENTS, SCHEDULER
    DEADLINE, OPPONENTS, SCHEDULER = deadline, opponents, scheduler
    POLICIES = {}
    for variant in variants:
        base = load_variant(checkpoint, variant)
        POLICIES[variant] = MeasuredPolicy(base)


def play(task):
    variant, seed, seat = task
    policy = POLICIES[variant]
    row, latencies = run_game(variant, seed, seat, policy.config, policy_override=policy,
                      opponents=OPPONENTS, scheduler=SCHEDULER, timeout=45,
                      stop_requested=lambda: time.monotonic() >= DEADLINE)
    row["movement"] = {**policy.metrics, "turns": len(policy.turns),
                       "cap_turns": len(policy.cap_turns)}
    return row, latencies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=["baseline", "cycle_exit", "cap7", "compound"],
                        default=["baseline", "cycle_exit", "cap7"])
    parser.add_argument("--opponents", choices=["scripted", "heuristic"], default="scripted")
    parser.add_argument("--scheduler", choices=["round_robin", "shuffled_rounds"], default="shuffled_rounds")
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, default=64, help="per variant, full eight-seat blocks")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    if args.max_seconds <= 0 or len(args.variants) != len(set(args.variants)):
        parser.error("positive time budget and unique variants required")
    pairs = schedule(args.split, args.seed, args.games)
    baseline = load_policy(args.checkpoint)
    identity = provenance(baseline.config)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {**identity, "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "split": args.split, "seed_offset": args.seed, "games_per_variant": args.games,
                "variants": args.variants, "scheduling": args.scheduler, "opponents": args.opponents,
                "opponent_profile": args.opponents, "workers": args.workers,
                "max_seconds": args.max_seconds, "frozen_runtime": baseline.runtime_config,
                "scope": "local simulation only; no credentials, queue, network or server replay",
                "comparison": "same weights/config/seed/seat/opponents; controllers differ"}
    manifest["runtimes"] = {
        variant: load_variant(args.checkpoint, variant).runtime_config
        for variant in args.variants}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    # Interleave variants within each seed/seat, sharing no mutable policy histories.
    tasks = [(variant, seed, seat) for seed, seat in pairs for variant in args.variants]
    started = time.monotonic()
    rows = []
    all_latencies = {variant: [] for variant in args.variants}
    with (args.output/"games.jsonl").open("x") as journal:
        with mp.get_context("spawn").Pool(args.workers, initialize,
                (str(args.checkpoint), args.variants, started+args.max_seconds,
                 args.opponents, args.scheduler)) as pool:
            for row, latencies in pool.imap_unordered(play, tasks):
                rows.append(row)
                all_latencies[row["policy"]].extend(latencies)
                emit(journal, row)
                if len(rows) % 24 == 0:
                    print(f"{len(rows)}/{len(tasks)} games, {time.monotonic()-started:.1f}s", flush=True)
    summary = summarize(rows, args.variants, args.games, all_latencies)
    for variant in args.variants:
        own = [r for r in rows if r["policy"] == variant]
        metrics = Counter()
        for row in own:
            metrics.update(row["movement"])
        summary[variant]["movement_totals"] = dict(metrics)
        summary[variant]["mean_moves_per_game"] = metrics["moves"]/len(own)
        summary[variant]["games_reaching_cap"] = sum(r["movement"]["cap_turns"] > 0 for r in own)
        total = sum(r["decisions"] for r in own)
        summary[variant]["weighted_decision_mean_ms"] = sum(r["decision_ms_total"] for r in own)/total if total else None
    result = {"elapsed_seconds": time.monotonic()-started, "metrics": summary,
              "paired": {variant: compare(args.output, args.output, variant, "baseline")
                         for variant in args.variants if variant != "baseline" and "baseline" in args.variants}}
    if "compound" in args.variants and "cycle_exit" in args.variants:
        result["incremental"] = compare(args.output, args.output, "compound", "cycle_exit")
    (args.output/"summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if any(r["status"] != "completed" or r["illegal_actions"] for r in rows):
        raise SystemExit("Incomplete or invalid games retained; comparison is not acceptance.")


if __name__ == "__main__":
    main()
