"""Paired placement differences using complete common eight-seat seed blocks."""
import argparse
import json
from pathlib import Path

import numpy as np


def compare(candidate, baseline, candidate_policy="ppo", baseline_policy="omnibob"):
    roots = [Path(candidate), Path(baseline)]
    manifests = [json.loads((p/"manifest.json").read_text()) for p in roots]
    for key in ("environment", "config_sha256", "split", "scheduling", "opponents"):
        if manifests[0].get(key) != manifests[1].get(key):
            raise ValueError(f"cannot compare different {key}")
    # A friendly opponent label is not a version: the frozen heuristic can
    # change while retaining its name. Existing manifests already contain the
    # exact source digest, including for the heterogeneous learned pool.
    if any(m.get("opponent_profile") in ("heuristic", "mixed", "mixed_learned") for m in manifests):
        hashes = [m.get("source_files", {}).get("omnibob/policies.py") for m in manifests]
        if not all(hashes) or hashes[0] != hashes[1]:
            raise ValueError("cannot compare missing or different frozen heuristic opponent implementations")
    indices = []
    for path, policy in zip(roots, (candidate_policy, baseline_policy)):
        rows = [json.loads(line) for line in (path/"games.jsonl").read_text().splitlines()]
        own = [r for r in rows if r["policy"] == policy]
        if not own:
            raise ValueError(f"no results for {policy} in {path}")
        index = {(r["seed"], r["seat"]): r for r in own}
        if len(index) != len(own):
            raise ValueError("duplicate seed/seat results")
        indices.append(index)
    common = set(indices[0]) & set(indices[1])
    complete = [s for s in sorted({s for s, _ in common})
                if all((s, seat) in common and all(idx[(s, seat)]["status"] == "completed"
                                                  for idx in indices) for seat in range(8))]
    blocks = [[np.mean([idx[(s, seat)]["placement"] for seat in range(8)])
               for idx in indices] for s in complete]
    differences = np.asarray([c-b for c, b in blocks])
    interval = None
    if len(differences) >= 2:
        rng = np.random.default_rng(20261028)
        samples = rng.choice(differences, (5000, len(differences)), replace=True).mean(axis=1)
        interval = np.percentile(samples, [2.5, 97.5]).tolist()
    return {"candidate": str(roots[0]), "baseline": str(roots[1]),
            "candidate_policy": candidate_policy, "baseline_policy": baseline_policy,
            "split": manifests[0]["split"], "complete_common_seed_blocks": len(complete),
            "paired_games": len(complete)*8, "observed_common_pairs": len(common),
            "excluded_common_pairs": len(common)-len(complete)*8,
            "unmatched_candidate_pairs": len(set(indices[0])-common),
            "unmatched_baseline_pairs": len(set(indices[1])-common),
            "candidate_mean": float(np.mean([r[0] for r in blocks])) if blocks else None,
            "baseline_mean": float(np.mean([r[1] for r in blocks])) if blocks else None,
            "candidate_minus_baseline": float(differences.mean()) if blocks else None,
            "paired_seed_bootstrap_95ci": interval, "seeds": complete,
            "note": "Negative placement difference favors candidate. Development selection is exploratory; "
                    "this interval does not correct for repeated model selection."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("candidate", type=Path)
    p.add_argument("baseline", type=Path)
    p.add_argument("--candidate-policy", default="ppo")
    p.add_argument("--baseline-policy", default="omnibob")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = compare(args.candidate, args.baseline, args.candidate_policy, args.baseline_policy)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
