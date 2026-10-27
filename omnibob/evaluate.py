"""Complete games, matched seed/seat blocks, and an append-only outcome journal."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import threading
import time

import numpy as np
import taverngym as tg

from .compat import digest, effective_config, provenance
from .observation import Observation
from .opponents import DESCRIPTIONS, LEARNED_PROFILES, OpponentSet
from .policies import make_policy

SPLITS = {"train": (0, 1_000_000_000), "dev": (1_000_000_000, 2_000_000_000),
          "final": (2_000_000_000, 3_000_000_000)}
SCHEDULERS = {"round_robin": "one action per ready seat, ascending seat order",
              "shuffled_rounds": "one action per ready seat, independently seeded shuffled order each round"}


def schedule(split, seed, games, seat=None):
    lo, hi = SPLITS[split]
    if games < 1 or seed < 0 or lo + seed + (games - 1) // 8 >= hi:
        raise ValueError("seed offset / game count outside the reserved split")
    if seat is None and games % 8:
        raise ValueError("evaluation games must be a multiple of 8 for full seat rotation")
    if seat is not None and (games != 1 or seat not in range(8)):
        raise ValueError("single-game mode requires one game and a seat in 0..7")
    return [(lo + seed + i // 8, i % 8 if seat is None else seat) for i in range(games)]


def emit(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def latency_summary(values):
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {"count": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(max(values))}


def run_game(policy_name, seed, seat, config, max_steps=100_000, timeout=30.0,
             trace=None, policy_override=None, opponents="scripted", scheduler="round_robin", opponent_policy=None,
             stop_requested=None):
    started = time.perf_counter()
    row = {"policy": policy_name, "seed": seed, "seat": seat, "status": "failed",
           "placement": None, "placements": None, "steps": 0, "decisions": 0,
           "illegal_actions": 0, "error": None, "truncated_combats": 0,
           "turn_limit_end": False}
    latencies, input_latencies = [], []
    game = None
    try:
        game = tg.Game(seed, json.dumps(config))
        opposition = OpponentSet(game, seed, seat, opponents, opponent_policy)
        if scheduler not in SCHEDULERS:
            raise ValueError("unknown action scheduler")
        scheduler_seed = int.from_bytes(hashlib.sha256(f"scheduler/{seed}".encode()).digest()[:8], "little")
        scheduler_rng = np.random.default_rng(scheduler_seed)
        # A private policy RNG, never passed into the game or exposed as future engine state.
        policy_seed = int.from_bytes(hashlib.sha256(f"policy/{seed}/{seat}".encode()).digest()[:8], "little")
        policy = policy_override or make_policy(policy_name, policy_seed, lambda: game.bot_action(seat))
        if callable(getattr(policy, "reset", None)):
            policy.reset()
        while not game.is_over():
            progressed = False
            order = (range(game.num_players) if scheduler == "round_robin"
                     else scheduler_rng.permutation(game.num_players).tolist())
            for p in order:
                if stop_requested is not None and stop_requested():
                    raise KeyboardInterrupt
                if not game.needs_action(p):
                    continue
                if row["steps"] >= max_steps:
                    row["status"] = "interrupted"
                    raise TimeoutError("driver decision limit reached")
                if time.perf_counter() - started >= timeout:
                    row["status"] = "interrupted"
                    raise TimeoutError("game wall-clock limit reached between native calls")
                t = time.perf_counter()
                visible_hash = None
                if p == seat:
                    observation_type = Observation
                    if getattr(policy, "observation_version", 1) == 2:
                        from .observation_v2 import ObservationV2
                        observation_type = ObservationV2
                    obs = observation_type.from_game(game, p)
                    input_latencies.append((time.perf_counter()-t)*1000)
                    t = time.perf_counter()
                    decision = policy.act(obs)
                    latency = (time.perf_counter()-t)*1000
                    latencies.append(latency)
                    action, reason = decision.action, decision.reason
                    row["decisions"] += 1
                    mask = obs.action_mask
                    if trace is not None:
                        visible_hash = hashlib.sha256(obs.features.tobytes() + obs.cards.tobytes()
                                                      + obs.action_mask.tobytes()).hexdigest()
                else:
                    opposing_decision = opposition.act(p)
                    action, reason = opposing_decision.action, opposing_decision.reason
                    latency = (time.perf_counter()-t)*1000
                    mask = game.legal_mask(p)
                valid = isinstance(action, (int, np.integer)) and not isinstance(action, bool)
                valid = valid and 0 <= action < tg.N_ACTIONS and bool(mask[action])
                if trace is not None:
                    emit(trace, {"policy": policy_name, "seed": seed, "candidate_seat": seat,
                                 "step": row["steps"], "turn": game.turn, "seat": p,
                                 "action": int(action) if isinstance(action, (int, np.integer)) else repr(action),
                                 "action_name": tg.action_name(int(action)) if valid else "INVALID",
                                 "reason": reason, "decision_ms": latency, "legal": bool(valid),
                                 "observation_version": getattr(policy, "observation_version", 1) if p == seat else None,
                                 "visible_sha256": visible_hash})
                if not valid:
                    row["illegal_actions"] += 1
                    raise ValueError(f"illegal action {action!r} at seat {p}")
                try:
                    game.step(p, int(action))
                except ValueError:
                    row["illegal_actions"] += 1
                    raise
                row["steps"] += 1
                progressed = True
            if not progressed and not game.is_over():
                raise RuntimeError("engine stalled: no seat needs action")
        placements = list(game.placements())
        if sorted(placements) != list(range(1, 9)):
            raise RuntimeError(f"engine did not produce eight final placements: {placements}")
        row.update(placements=placements, placement=placements[seat], status="completed")
        if game.truncated_combats or game.turn_limit_end:
            row["status"] = "truncated"
    except KeyboardInterrupt:
        row.update(status="interrupted", error="KeyboardInterrupt")
    except BaseException as exc:
        # Also retain PyO3 PanicException, which is a BaseException rather than Exception.
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["elapsed_seconds"] = time.perf_counter() - started
        if game is not None:
            row.update(turns=game.turn, truncated_combats=game.truncated_combats,
                       turn_limit_end=game.turn_limit_end)
        row["decision_latency"] = latency_summary(latencies)
        row["observation_latency"] = latency_summary(input_latencies)
        row["decision_ms_total"] = sum(latencies)
    return row, latencies


def summarize(rows, policies, games, all_latencies):
    result = {}
    for policy in policies:
        own = [r for r in rows if r["policy"] == policy]
        completed = [r for r in own if r["status"] == "completed"]
        places = [r["placement"] for r in completed]
        counts = Counter(r["status"] for r in own)
        block_means = []
        for seed in sorted({r["seed"] for r in completed}):
            block = [r for r in completed if r["seed"] == seed]
            if len(block) == 8 and {r["seat"] for r in block} == set(range(8)):
                block_means.append(float(np.mean([r["placement"] for r in block])))
        ci = None
        if len(block_means) >= 2:
            rng = np.random.default_rng(20261028)
            samples = rng.choice(block_means, size=(2000, len(block_means)), replace=True).mean(axis=1)
            ci = np.percentile(samples, [2.5, 97.5]).tolist()
        result[policy] = {
            "requested": games, "completed": counts["completed"], "failed": counts["failed"],
            "interrupted": counts["interrupted"], "truncated": counts["truncated"],
            "not_started": games - len(own), "metric_denominator": len(completed),
            "mean_placement": float(np.mean(places)) if places else None,
            "win_rate": sum(p == 1 for p in places)/len(places) if places else None,
            "top4_rate": sum(p <= 4 for p in places)/len(places) if places else None,
            "completed_seed_blocks": len(block_means), "mean_placement_seed_bootstrap_95ci": ci,
            "complete_block_metric_denominator": len(block_means)*8,
            "complete_block_mean_placement": float(np.mean(block_means)) if block_means else None,
            "complete_block_mean_placement_seed_bootstrap_95ci": ci,
            "confidence_interval_scope": "The interval estimates complete_block_mean_placement using whole eight-seat blocks; "
                                         "mean_placement uses every normally completed game and can have a different denominator. "
                                         "mean_placement_seed_bootstrap_95ci is retained as a legacy alias.",
            "illegal_actions": sum(r["illegal_actions"] for r in own),
            "decision_latency": latency_summary(all_latencies[policy]),
            "elapsed_seconds": sum(r["elapsed_seconds"] for r in own),
            "seats_attempted": dict(Counter(str(r["seat"]) for r in own)),
        }
    return result


def evaluate(output, policies, split="dev", seed=0, games=8, seat=None, config=None,
             max_steps=100_000, timeout=30.0, trace=False, checkpoint=None, opponents="scripted",
             scheduler="round_robin", move_budget=None, opponent_checkpoint=None, repeat_limit=None):
    if opponents not in DESCRIPTIONS:
        raise ValueError(f"unsupported opponent profile: {opponents}")
    if scheduler not in SCHEDULERS:
        raise ValueError("unknown action scheduler")
    if (opponents in LEARNED_PROFILES) != (opponent_checkpoint is not None):
        raise ValueError("learned opponents and --opponent-checkpoint must be specified together")
    if move_budget is not None and not 0 <= move_budget <= 100:
        raise ValueError("move budget must be in 0..100 (zero disables the optional controller)")
    pairs = schedule(split, seed, games, seat)
    if len(set(policies)) != len(policies) or any(p not in ("random", "scripted", "omnibob", "ppo", "bc") for p in policies):
        raise ValueError("policies must be unique: random, scripted, omnibob, ppo, bc")
    if "ppo" in policies and "bc" in policies:
        raise ValueError("evaluate PPO and imitation checkpoints in separate matched runs")
    learned = None
    if "ppo" in policies or "bc" in policies:
        if checkpoint is None:
            raise ValueError("ppo requires --checkpoint")
        from .learned import load_policy
        learned = load_policy(checkpoint, move_budget=move_budget, repeat_limit=repeat_limit)
        expected_method = "behavior_cloning" if "bc" in policies else "ppo"
        if learned.training_method != expected_method:
            raise ValueError(f"checkpoint method is {learned.training_method}; use the matching policy name")
    elif checkpoint is not None:
        raise ValueError("--checkpoint requires a ppo or bc policy; omnibob is the frozen heuristic baseline")
    elif move_budget or repeat_limit:
        raise ValueError("the optional move controller requires a learned checkpoint")
    opponent_policy = None
    if opponent_checkpoint is not None:
        from .learned import load_policy
        opponent_policy = load_policy(opponent_checkpoint)
    inferred_config = learned.config if learned else opponent_policy.config if opponent_policy else None
    cfg = effective_config(inferred_config if config is None else config)
    if opponent_policy and effective_config(opponent_policy.config) != cfg:
        raise ValueError("evaluation rules differ from the frozen opponent checkpoint")
    if cfg["num_players"] != 8 or cfg["board_size"] != 7 or cfg.get("experimental_mode", False):
        raise ValueError("v0.1 supports eight players, seven board slots, non-experimental rules")
    # Validate configuration through the actual engine before creating the run directory.
    tg.Game(pairs[0][0], json.dumps(cfg))
    identity = provenance(cfg)
    opponent_description = DESCRIPTIONS[opponents]
    if opponent_policy:
        opponent_sha = hashlib.sha256(Path(opponent_checkpoint).read_bytes()).hexdigest()
        opponent_description += "; checkpoint_sha256="+opponent_sha
        identity["opponent_checkpoint"] = {"path": str(Path(opponent_checkpoint).resolve()),
                                           "sha256": opponent_sha, "version": opponent_policy.version,
                                           "runtime_config": opponent_policy.runtime_config}
    if learned:
        if effective_config(learned.config) != cfg:
            raise ValueError("evaluation rules differ from the checkpoint's training rules")
        identity["checkpoint"] = {"path": str(Path(checkpoint).resolve()),
                                  "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                                  "training_method": learned.training_method,
                                  "training_steps": learned.training_steps}
        identity["inference"] = {"device": learned.device, "cpu_threads": learned.cpu_threads,
                                 "selection": "greedy legal argmax", "move_guard_budget": learned.move_guard_budget,
                                 "repeat_limit": learned.repeat_limit,
                                 "runtime_config": learned.runtime_config,
                                 "controller_note": "explicit heuristic controller; report separately from PPO gains" if learned.runtime_config else None}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    identity.update(created_at=datetime.now(timezone.utc).isoformat(), policies=policies,
                    split=split, seed_offset=seed, games_per_policy=games,
                    schedule=[{"seed": s, "seat": p} for s, p in pairs],
                    seed_partitions=SPLITS, seat_balanced=seat is None,
                    max_steps=max_steps, timeout_seconds=timeout,
                    scheduling=SCHEDULERS[scheduler], scheduler=scheduler,
                    opponents=opponent_description, opponent_profile=opponents,
                    trace=trace, policy_versions={"random": "uniform-v1", "scripted": "fb61201",
                                                 "omnibob": "tempo-v0.1"})
    if learned:
        identity["policy_versions"]["bc" if "bc" in policies else "ppo"] = learned.version
    (output/"manifest.json").write_text(json.dumps(identity, indent=2, ensure_ascii=False)+"\n")
    rows, latencies = [], {p: [] for p in policies}
    started, interrupted = time.perf_counter(), False
    trace_file = (output/"decisions.jsonl").open("w") if trace else None
    summary = {}
    termination_signal = None
    previous_term = None

    def request_interrupt(signum, frame):
        nonlocal termination_signal
        termination_signal = signum

    if threading.current_thread() is threading.main_thread():
        previous_term = signal.signal(signal.SIGTERM, request_interrupt)
    try:
        with (output/"games.jsonl").open("w") as results, (output/"events.jsonl").open("w") as events:
            # Interleave strategies within each seed/seat to reduce temporal performance bias.
            for game_seed, game_seat in pairs:
                for policy in policies:
                    if termination_signal is not None:
                        interrupted = True
                        break
                    emit(events, {"event": "started", "policy": policy, "seed": game_seed, "seat": game_seat})
                    row, timing = run_game(policy, game_seed, game_seat, cfg, max_steps, timeout, trace_file,
                                           policy_override=learned if policy in ("ppo", "bc") else None,
                                           opponents=opponents, scheduler=scheduler, opponent_policy=opponent_policy,
                                           stop_requested=lambda: termination_signal is not None)
                    rows.append(row)
                    latencies[policy].extend(timing)
                    emit(results, row)
                    emit(events, {"event": "finished", "policy": policy, "seed": game_seed,
                                  "seat": game_seat, "status": row["status"]})
                    if row["error"] == "KeyboardInterrupt":
                        interrupted = True
                        break
                if interrupted:
                    break
    except KeyboardInterrupt:
        interrupted = True
    finally:
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if trace_file:
            trace_file.close()
        summary = {"policies": summarize(rows, policies, games, latencies),
                   "elapsed_seconds": time.perf_counter() - started, "user_interrupted": interrupted,
                   "termination_signal": termination_signal,
                   "source_sha256": identity["source_sha256"], "config_sha256": digest(cfg),
                   "metrics_note": "Only non-truncated complete games enter placement metrics. "
                   "Failures are retained; seed blocks, not seats, are the resampling unit. "
                   "These frozen synthetic-opponent results are not paper or live-game results."}
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def replay(directory, on_visible_decision=None):
    """Reapply the full action trace and verify candidate-visible hashes and final results."""
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    from .compat import check_compatibility
    check_compatibility(manifest["environment"])
    expected = {(r["policy"], r["seed"], r["seat"]): r
                for r in map(json.loads, (directory/"games.jsonl").read_text().splitlines())}
    current, game, steps, verified = None, None, 0, 0

    def finish(key, g, n):
        record = expected[key]
        if record["status"] != "completed":
            raise ValueError(f"cannot certify incomplete/truncated trace {key}")
        if not g.is_over() or list(g.placements()) != record["placements"] or n != record["steps"]:
            raise ValueError(f"replay result mismatch: {key}")

    with (directory/"decisions.jsonl").open() as stream:
        for line in stream:
            event = json.loads(line)
            key = (event["policy"], event["seed"], event["candidate_seat"])
            if key != current:
                if current is not None:
                    finish(current, game, steps)
                    verified += 1
                game = tg.Game(key[1], json.dumps(manifest["config"]))
                current, steps = key, 0
            if event["step"] != steps or event["turn"] != game.turn:
                raise ValueError("replay sequence mismatch")
            if event["visible_sha256"] is not None:
                observation_type = Observation
                if event.get("observation_version", 1) == 2:
                    from .observation_v2 import ObservationV2
                    observation_type = ObservationV2
                obs = observation_type.from_game(game, event["seat"])
                actual = hashlib.sha256(obs.features.tobytes() + obs.cards.tobytes()
                                        + obs.action_mask.tobytes()).hexdigest()
                if actual != event["visible_sha256"]:
                    raise ValueError("replay visible observation mismatch")
                if on_visible_decision is not None:
                    # Consumers receive only this player's ordinary visible view,
                    # never the driver-owned game, full save, or random state.
                    on_visible_decision(event, json.loads(game.player_view(event["seat"])))
            game.step(event["seat"], event["action"])
            steps += 1
    if current is not None:
        finish(current, game, steps)
        verified += 1
    if verified != len(expected):
        raise ValueError("trace missing games")
    return {"verified_games": verified, "observation_hashes_verified": True}
