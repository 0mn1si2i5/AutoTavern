"""TavernBench direct-action diagnostic (80/368 tasks), never a full benchmark score.

Uses only tasks with a published action-to-answer map. No state stepping, search,
combat simulation, future RNG use, labels-as-input, or paid model calls.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import taverngym as tg

from .compat import ROOT, effective_config, provenance
from .evaluate import emit
from .observation_v2 import ObservationV2
from .policies import OmniBobPolicy


def diagnose(output, checkpoint=None):
    dataset = ROOT/"tavernbench/data/eval"
    if checkpoint:
        from .learned import load_policy
        learned = load_policy(checkpoint)
        config = learned.config
    else:
        learned = None
        config = effective_config(json.loads((ROOT/"configs/arena-v2.json").read_text()))
    identity = provenance(config)
    identity.update(mode="published direct action maps only; constrained MCQ policy",
                    model=learned.version if learned else "tempo-v0.1", training_use=False,
                    dataset={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.iterdir() if p.is_file()})
    if checkpoint:
        identity["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        identity["runtime_config"] = learned.runtime_config
    with gzip.open(dataset/"states.jsonl.gz", "rt") as stream:
        selected = [json.loads(line) for line in stream if '"map"' in line]
    selected = [row for row in selected if "map" in row]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out/"manifest.json").write_text(json.dumps(identity, indent=2)+"\n")
    answers = []
    with (out/"answers.jsonl").open("w") as stream:
        for task in selected:
            row = {"id": task["id"], "model": identity["model"], "response": None, "error": None}
            started = time.perf_counter()
            try:
                # Full serialized state belongs only to the diagnostic driver. Policy
                # receives the same owner-visible boundary used during ordinary play.
                game = tg.Game.from_json(task["state"])
                obs = ObservationV2.from_game(game, task.get("seat", 0))
                mapping = {int(a): answer for a, answer in task["map"].items()}
                allowed = np.zeros(245, dtype=bool)
                for action in mapping:
                    if not 0 <= action < 245 or not obs.action_mask[action]:
                        raise ValueError("published candidate action is not legal in current engine")
                    allowed[action] = True
                constrained = ObservationV2(obs.features, obs.cards, allowed)
                if learned and hasattr(learned, "reset"):
                    learned.reset()  # Published tasks are independent states, not consecutive turns.
                decision = learned.act(constrained) if learned else OmniBobPolicy().act(constrained.legacy())
                row.update(action=decision.action, response=json.dumps({"answer": mapping[decision.action]}),
                           reason=decision.reason)
                task_config = effective_config(json.loads(task["state"])["cfg"])
                row["task_rule_differences"] = [k for k in set(config)|set(task_config)
                                                if config.get(k) != task_config.get(k)]
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["latency_s"] = time.perf_counter()-started
            answers.append(row)
            emit(stream, row)
    # Load labels only after every policy answer has been written.
    sys.path.insert(0, str(ROOT/"tavernbench"))
    from tavernbench.score import score_answer
    labels = {r["id"]: r for r in map(json.loads, (dataset/"labels.jsonl").read_text().splitlines())}
    scores = defaultdict(list)
    for answer in answers:
        scores[answer["id"].split("-")[0]].append(score_answer(labels[answer["id"]], answer))
    summary = {"attempted": len(answers), "total_formal_tasks": len(labels),
               "unsupported": len(labels)-len(answers), "errors": sum(r["error"] is not None for r in answers),
               "tasks_with_different_rules": sum(bool(r.get("task_rule_differences")) for r in answers),
               "scenarios": {s: {"tasks": len(v), "accuracy": float(np.mean([a for a, _ in v])),
                                  "graded_score": float(np.mean([b for _, b in v]))} for s, v in sorted(scores.items())},
               "limitations": "80 mapped single-action questions only; choices restricted to supplied options. "
               "Saved task rules may differ from arena training. No rule-text reasoning or full TavernBench score. "
               "Public diagnostic only, never independent generalization or model-selection evidence."}
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    print(json.dumps(diagnose(args.output, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
