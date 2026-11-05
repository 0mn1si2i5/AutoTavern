import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="OmniBob local policies, evaluation and compatibility checks")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("play", "evaluate"):
        p = commands.add_parser(name)
        p.add_argument("--output", type=Path, required=True, help="new directory; existing results are never overwritten")
        p.add_argument("--split", choices=("train", "dev", "final"), default="dev")
        p.add_argument("--seed", type=int, default=0, help="offset within the reserved seed split")
        p.add_argument("--config", type=Path, help="JSON RulesConfig overrides")
        p.add_argument("--max-steps", type=int, default=100_000)
        p.add_argument("--timeout", type=float, default=30)
        p.add_argument("--checkpoint", type=Path, help="local PPO checkpoint for --policy/--policies ppo")
        p.add_argument("--opponents", choices=("scripted", "heuristic", "mixed", "learned", "mixed_learned"), default="scripted")
        p.add_argument("--opponent-checkpoint", type=Path, help="frozen seven-opponent model for --opponents learned")
        p.add_argument("--scheduler", choices=("round_robin", "shuffled_rounds"))
        p.add_argument("--move-budget", type=int, help="override checkpoint runtime; optional move controller, 0 disables")
        p.add_argument("--repeat-limit", type=int, help="override checkpoint runtime; limit repeated visible-state actions, 0 disables")
        if name == "play":
            p.add_argument("--policy", choices=("deployed", "random", "scripted", "omnibob", "ppo", "bc"), default="deployed")
            p.add_argument("--seat", type=int, choices=range(8), default=0)
        else:
            p.add_argument("--policies", nargs="+", choices=("deployed", "random", "scripted", "omnibob", "ppo", "bc"),
                           default=["deployed"])
            p.add_argument("--games", type=int, default=64, help="games PER policy, multiple of eight")
            p.add_argument("--trace", action="store_true", help="save all accepted actions and candidate observation hashes")
    p = commands.add_parser("replay")
    p.add_argument("directory", type=Path)
    commands.add_parser("check")
    p = commands.add_parser("benchmark")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-torch", action="store_true")
    args = parser.parse_args()
    try:
        if args.command in ("play", "evaluate"):
            from .evaluate import evaluate
            policies = [args.policy] if args.command == "play" else args.policies
            deployment = None
            if "deployed" in policies:
                if args.checkpoint or args.move_budget is not None or args.repeat_limit is not None or any(
                    p in policies for p in ("ppo", "bc")
                ):
                    raise ValueError("deployed uses its frozen checkpoint/runtime; use explicit --policy ppo for overrides")
                from .deployment import resolve
                deployment = resolve()
                args.checkpoint = Path(deployment["checkpoint"])
                args.scheduler = args.scheduler or deployment["scheduler"]
                policies = ["ppo" if p == "deployed" else p for p in policies]
            args.scheduler = args.scheduler or "round_robin"
            if args.timeout <= 0 or args.max_steps <= 0:
                raise ValueError("timeout and max-steps must be positive")
            result = evaluate(args.output, policies,
                              args.split, args.seed, 1 if args.command == "play" else args.games,
                              args.seat if args.command == "play" else None,
                              json.loads(args.config.read_text()) if args.config else None,
                              args.max_steps, args.timeout, args.command == "play" or args.trace,
                              args.checkpoint, args.opponents, args.scheduler, args.move_budget, args.opponent_checkpoint,
                              args.repeat_limit)
            if deployment is not None:
                (args.output / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
                result["deployment"] = deployment
            print(json.dumps(result, indent=2))
            return 0 if all(p["completed"] == p["requested"] for p in result["policies"].values()) else 2
        if args.command == "replay":
            from .evaluate import replay
            result = replay(args.directory)
        elif args.command == "benchmark":
            from .benchmark import benchmark
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as stream:
                result = benchmark(not args.no_torch)
                json.dump(result, stream, indent=2)
                stream.write("\n")
        else:
            from .compat import check_compatibility
            result = check_compatibility()
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
