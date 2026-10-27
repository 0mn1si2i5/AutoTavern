"""Player trajectories with explicit frozen opponents and reproducible scheduling.

Unlike native BatchEnv's hashed reset seeds, explicit bounded train-domain seeds
cannot overlap the dev/final domains. Engine saves belong only to the trainer.
Full games are the default; optional early termination uses the learner's already
assigned placement, never an estimate of the remaining players' results.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import taverngym as tg

from .observation_v2 import ObservationV2
from .opponents import LEARNED_PROFILES, OpponentSet


class SimulatorTruncation(RuntimeError):
    def __init__(self, episode):
        self.record = {"status": "truncated", "seed": episode.seed, "seat": episode.seat,
                       "placement": None, "placements": None, "turns": episode.game.turn,
                       "steps": episode.steps, "full_game_completed": False,
                       "episode_end": "simulator_truncation", "reward_excluded": True,
                       "forced_placements": list(episode.game.placements()),
                       "truncated_combats": episode.game.truncated_combats,
                       "turn_limit_end": episode.game.turn_limit_end}
        super().__init__(f"simulator truncation: seed={episode.seed}, turn={episode.game.turn}")


class GameEpisode:
    def __init__(self, seed, seat, config, opponents="scripted", scheduler="round_robin", terminate_on_elimination=False,
                 learned_policy=None):
        if not 0 <= seed < 1_000_000_000:
            raise ValueError("training game seed outside reserved train domain")
        self.seed, self.seat = seed, seat
        self.game = tg.Game(seed, json.dumps(config))
        self.opponent_profile, self.scheduler = opponents, scheduler
        self.terminate_on_elimination = terminate_on_elimination
        self.opposition = OpponentSet(self.game, seed, seat, opponents, learned_policy)
        if scheduler not in ("round_robin", "shuffled_rounds"):
            raise ValueError("unknown training scheduler")
        schedule_seed = int.from_bytes(hashlib.sha256(f"scheduler/{seed}".encode()).digest()[:8], "little")
        self.scheduler_rng = np.random.default_rng(schedule_seed)
        self.order = list(range(8)) if scheduler == "round_robin" else self.scheduler_rng.permutation(8).tolist()
        self.position, self.steps = 0, 0
        self.cursor = self.order[self.position]
        self.advance()

    def next_seat(self):
        self.position = (self.position+1) % 8
        if self.position == 0 and self.scheduler == "shuffled_rounds":
            self.order = self.scheduler_rng.permutation(8).tolist()
        self.cursor = self.order[self.position]

    def advance(self):
        idle = 0
        while not self.game.is_over():
            if self.terminate_on_elimination and self.game.placements()[self.seat] > 0:
                return
            if self.game.needs_action(self.cursor):
                if self.cursor == self.seat:
                    return
                self.game.step(self.cursor, self.opposition.act(self.cursor).action)
                self.steps += 1
                idle = 0
            else:
                idle += 1
            if self.steps > 100_000 or idle > 16:
                raise RuntimeError(f"training simulator stalled: seed={self.seed}, seat={self.seat}")
            self.next_seat()

    def observe(self):
        return ObservationV2.from_game(self.game, self.seat)

    def step(self, action):
        if self.cursor != self.seat or not self.game.needs_action(self.seat):
            raise RuntimeError("trainer attempted an action outside its decision point")
        mask = self.game.legal_mask(self.seat)
        if not 0 <= action < 245 or not mask[action]:
            raise ValueError(f"illegal training action: {action}")
        self.game.step(self.seat, action)
        self.steps += 1
        self.next_seat()
        self.advance()
        if self.game.truncated_combats or self.game.turn_limit_end:
            raise SimulatorTruncation(self)
        if self.game.is_over() or self.terminate_on_elimination and self.game.placements()[self.seat] > 0:
            places = list(self.game.placements())
            if places[self.seat] not in range(1, 9) or self.game.is_over() and sorted(places) != list(range(1, 9)):
                raise RuntimeError("invalid completed-game placements")
            return {"status": "completed", "seed": self.seed, "seat": self.seat, "placement": places[self.seat],
                    "placements": places, "turns": self.game.turn, "steps": self.steps,
                    "full_game_completed": self.game.is_over(),
                    "episode_end": "game_over" if self.game.is_over() else "learner_eliminated"}
        return None

    def save(self):
        return {"seed": self.seed, "seat": self.seat, "cursor": self.cursor, "steps": self.steps,
                "opponent_profile": self.opponent_profile, "opponent_states": self.opposition.save(),
                "terminate_on_elimination": self.terminate_on_elimination,
                "scheduler": self.scheduler, "order": self.order, "position": self.position,
                "scheduler_rng": self.scheduler_rng.bit_generator.state,
                "game": self.game.state_json()}

    @classmethod
    def restore(cls, state, learned_policy=None):
        episode = cls.__new__(cls)
        episode.seed, episode.seat = state["seed"], state["seat"]
        episode.cursor, episode.steps = state["cursor"], state["steps"]
        episode.game = tg.Game.from_json(state["game"])
        episode.opponent_profile = state.get("opponent_profile", "scripted")
        episode.scheduler = state.get("scheduler", "round_robin")
        episode.terminate_on_elimination = state.get("terminate_on_elimination", False)
        episode.opposition = OpponentSet(episode.game, episode.seed, episode.seat, episode.opponent_profile, learned_policy)
        episode.opposition.restore(state.get("opponent_states", {}))
        episode.order, episode.position = state.get("order", list(range(8))), state.get("position", episode.cursor)
        schedule_seed = int.from_bytes(hashlib.sha256(f"scheduler/{episode.seed}".encode()).digest()[:8], "little")
        episode.scheduler_rng = np.random.default_rng(schedule_seed)
        if "scheduler_rng" in state:
            episode.scheduler_rng.bit_generator.state = state["scheduler_rng"]
        if episode.order[episode.position] != episode.cursor:
            raise ValueError("saved scheduler position mismatch")
        return episode


class VectorGames:
    def __init__(self, n_games, seed, config, workers=1, state=None, opponents="scripted", scheduler="round_robin",
                 terminate_on_elimination=False, allow_truncated=False, opponent_checkpoint=None):
        if n_games < 1 or seed < 0:
            raise ValueError("positive environment count and nonnegative seed required")
        self.n_games, self.seed, self.config = n_games, seed, config
        self.opponents, self.scheduler = opponents, scheduler
        self.terminate_on_elimination = terminate_on_elimination
        self.allow_truncated = allow_truncated
        self.learned_policy, self.opponent_checkpoint_sha256 = None, None
        if (opponents in LEARNED_PROFILES) != (opponent_checkpoint is not None):
            raise ValueError("learned opponents require their frozen checkpoint")
        if opponent_checkpoint is not None:
            import torch
            from .compat import effective_config
            from .learned import load_policy
            self.opponent_checkpoint_sha256 = hashlib.sha256(Path(opponent_checkpoint).read_bytes()).hexdigest()
            if state and state.get("opponent_checkpoint_sha256") != self.opponent_checkpoint_sha256:
                raise ValueError("frozen opponent checkpoint changed since training save")
            # Loading immutable CPU weights must not perturb learner RNG or thread settings.
            rng, threads = torch.get_rng_state(), torch.get_num_threads()
            try:
                self.learned_policy = load_policy(opponent_checkpoint)
            finally:
                torch.set_rng_state(rng)
                torch.set_num_threads(threads)
            if effective_config(self.learned_policy.config) != effective_config(config):
                raise ValueError("frozen opponent rules differ from training rules")
        self.pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        if state is None:
            self.episodes = [0] * n_games
            self.games = [self.fresh(i) for i in range(n_games)]
        else:
            if len(state["games"]) != n_games:
                raise ValueError("checkpoint environment count mismatch")
            self.episodes = state["episodes"]
            self.games = [GameEpisode.restore(s, self.learned_policy) for s in state["games"]]
            if any(g.opponent_profile != opponents or g.scheduler != scheduler
                   or g.terminate_on_elimination != terminate_on_elimination for g in self.games):
                raise ValueError("saved opponent/scheduler configuration mismatch")

    def fresh(self, i):
        return GameEpisode(self.seed + self.episodes[i]*self.n_games + i,
                           (i+self.episodes[i]) % 8, self.config, self.opponents, self.scheduler,
                           self.terminate_on_elimination, self.learned_policy)

    def observe(self):
        observations = [g.observe() for g in self.games]
        return {name: np.stack([getattr(o, name) for o in observations])
                for name in ("features", "cards", "action_mask")}

    def step(self, actions):
        if len(actions) != self.n_games:
            raise ValueError("training action batch has wrong length")
        tasks = [(i, int(a)) for i, a in enumerate(actions)]

        def act(pair):
            i, a = pair
            try:
                return self.games[i].step(a)
            except SimulatorTruncation as exc:
                if not self.allow_truncated:
                    raise
                return exc.record

        records = list(self.pool.map(act, tasks)) if self.pool else [act(pair) for pair in tasks]
        rewards, dones = np.zeros(self.n_games, np.float32), np.zeros(self.n_games, bool)
        for i, record in enumerate(records):
            if record is not None:
                # Independent OmniBob objective, not an assertion of the paper's utility.
                if record["status"] == "completed":
                    rewards[i] = 1 - 2*(record["placement"]-1)/7
                record["environment_index"] = i
                dones[i] = True
                self.episodes[i] += 1
                self.games[i] = self.fresh(i)
        return rewards, dones, [r for r in records if r is not None]

    def save(self):
        return {"episodes": self.episodes.copy(), "games": [g.save() for g in self.games],
                "opponent_checkpoint_sha256": self.opponent_checkpoint_sha256}

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
