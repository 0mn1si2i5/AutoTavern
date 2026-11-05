"""Read-only own-observation practice diagnosis; no server calls or hidden-state replay."""
import argparse
import hashlib
import json
from pathlib import Path

from .learned import load_policy
from .observation_v2 import ObservationV2
from .reorder_exit import board_layout, load_reorder_policy


def review(session, checkpoint):
    session = Path(session)
    rows = [json.loads(line) for line in (session/"policy/decisions.jsonl").read_text().splitlines()]
    frozen, candidate = load_policy(checkpoint), load_reorder_policy(checkpoint)
    turns, bursts, seen, context = {}, [], set(), None
    burst = None
    first_difference = None
    mismatches, replay_reason_mismatches, tested_successors, matching_successors = [], [], 0, 0
    for index, row in enumerate(rows):
        obs = ObservationV2(**row["observation"])
        actual = frozen.act(obs)
        if actual.action != row["action"]:
            mismatches.append(index)
        if actual.reason != row["reason"]:
            replay_reason_mismatches.append(index)
        if first_difference is None:
            proposed = candidate.act(obs)
            if proposed.action != actual.action:
                first_difference = {"record_index_zero_based": index, "turn": row["turn"],
                                    "baseline_action": actual.action, "candidate_action": proposed.action,
                                    "candidate_reason": proposed.reason}
        me = row["player_view"]["me"]
        turn = turns.setdefault(row["turn"], {"decisions": 0, "moves": 0, "cycle_returns": 0})
        turn["decisions"] += 1
        turn.update(final_action=row["action_name"], gold_before_final_action=me["gold"],
                    tier_before_final_action=me["tier"], board_size_before_final_action=len(me["board"]),
                    hand_before_final_action=[{"name": c.get("name_zh"), "locked": c.get("locked"),
                                              "type": c.get("type")} for c in me["hand"]])
        units = board_layout(obs)
        current = (row["turn"], tuple(sorted(units)))
        if context != current:
            if burst is not None:
                bursts.append(burst)
            burst, seen = None, set()
        context = current
        if row["action"] >= 196:
            if burst is None:
                burst = {"turn": row["turn"], "start_record": index, "moves": 0, "cycle_returns": 0}
            turn["moves"] += 1
            burst["moves"] += 1
            seen.add(units)
            source, target = divmod(row["action"]-196, 7)
            reordered = list(units)
            reordered.insert(target, reordered.pop(source))
            returning = tuple(reordered) in seen
            burst["cycle_returns"] += returning
            turn["cycle_returns"] += returning
            if index+1 < len(rows) and rows[index+1]["turn"] == row["turn"]:
                tested_successors += 1
                matching_successors += tuple(reordered) == board_layout(ObservationV2(**rows[index+1]["observation"]))
        else:
            if burst is not None:
                bursts.append(burst)
            burst, seen, context = None, set(), None
    if burst is not None:
        bursts.append(burst)
    terminal = json.loads((session/"terminal.json").read_text())
    match = terminal["match"]
    return {"session": str(session.resolve()), "input_sha256": {
                name: hashlib.sha256((session/name).read_bytes()).hexdigest()
                for name in ("policy/decisions.jsonl", "http/requests.jsonl", "terminal.json")},
            "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            "match_id": match["id"], "category": match["category"],
            "human_count": match["human_count"], "bot_count": match["bot_count"],
            "placement": match["participants"][0]["placement"], "decisions": len(rows),
            "moves": sum(t["moves"] for t in turns.values()),
            "cycle_returns": sum(t["cycle_returns"] for t in turns.values()),
            "per_turn": turns, "move_bursts": bursts,
            "visible_move_successor_checks": {"tested": tested_successors, "matching": matching_successors},
            "frozen_policy_replay": {"action_mismatches": mismatches, "reason_mismatches": replay_reason_mismatches},
            "candidate_first_divergence": first_difference,
            "limitations": ["Only policy replay; server seed and combat RNG unavailable.",
                            "Candidate evaluated only up to first divergence, not as a counterfactual full match.",
                            "Cycle returns are confirmed repeated visible arrangements, not a measured placement loss.",
                            "Six-unit ends and held cards require strategic evaluation; neither is automatically an error."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = review(args.session, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("decisions", "moves", "cycle_returns",
          "visible_move_successor_checks", "frozen_policy_replay", "candidate_first_divergence")}, ensure_ascii=False))
    if result["frozen_policy_replay"]["action_mismatches"] or result["frozen_policy_replay"]["reason_mismatches"]:
        raise SystemExit("Frozen replay mismatch; retained the failed report.")


if __name__ == "__main__":
    main()
