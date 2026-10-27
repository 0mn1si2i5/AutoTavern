import json
import subprocess
import sys
import time

import numpy as np
import pytest
import taverngym as tg

torch = pytest.importorskip("torch", reason="optional training dependencies: install requirements-omnibob-mps.lock")

from omnibob.evaluate import run_game
from omnibob.model import EntityActorCritic
from omnibob.observation_v2 import ObservationV2
from omnibob.policies import OmniBobPolicy
from omnibob.train import TrainConfig, gae, load_checkpoint, tensors, train
from omnibob.training_env import GameEpisode, VectorGames


def test_gae_terminal_boundary_and_horizon_bootstrap():
    rewards = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
    values = np.array([[.5, .25], [100., .5]], dtype=np.float32)
    dones = np.array([[True, False], [False, True]])
    adv, ret = gae(rewards, values, dones, np.array([101., 999.]), 1., 1.)
    np.testing.assert_allclose(adv, [[.5, 1.75], [1., 1.5]])
    np.testing.assert_allclose(ret, [[1., 2.], [101., 2.]])


def test_training_game_matches_full_evaluator_scheduler():
    for seat in (0, 3, 7):
        seed = 300+seat
        episode = GameEpisode(seed, seat, {})
        policy = OmniBobPolicy()
        for _ in range(5000):
            decision = policy.act(episode.observe().legacy())
            record = episode.step(decision.action)
            if record:
                break
        else:
            pytest.fail("training episode did not complete")
        reference, _ = run_game("omnibob", seed, seat, {})
        assert record["placements"] == reference["placements"]
        assert record["steps"] == reference["steps"]


def test_mixed_shuffled_training_matches_evaluator_and_restores_opponent_state():
    seed, seat = 901000, 3
    episode = GameEpisode(seed, seat, {}, opponents="mixed", scheduler="shuffled_rounds")
    policy = OmniBobPolicy()
    for step in range(5000):
        if step == 30:
            episode = GameEpisode.restore(episode.save())
        record = episode.step(policy.act(episode.observe().legacy()).action)
        if record:
            break
    else:
        pytest.fail("mixed shuffled episode did not finish")
    reference, _ = run_game("omnibob", seed, seat, {}, opponents="mixed", scheduler="shuffled_rounds")
    assert record["placements"] == reference["placements"]
    assert record["steps"] == reference["steps"]


def test_early_elimination_preserves_visible_trajectory_and_exact_own_placement():
    observed_early = False
    for seed in range(321, 325):
        full = GameEpisode(seed, 2, {})
        early = GameEpisode(seed, 2, {}, terminate_on_elimination=True)
        policy = OmniBobPolicy()
        for _ in range(5000):
            left, right = full.observe(), early.observe()
            for name in ("features", "cards", "action_mask"):
                np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
            action = policy.act(left.legacy()).action
            finished, known = full.step(action), early.step(action)
            assert bool(finished) == bool(known)
            if finished:
                assert finished["placement"] == known["placement"]
                assert finished["full_game_completed"]
                assert known["steps"] <= finished["steps"]
                observed_early |= not known["full_game_completed"]
                break
        else:
            pytest.fail("elimination probe failed to finish")
    assert observed_early


def test_vector_save_restore_and_seed_partitions():
    env = VectorGames(2, 200000, {}, workers=1)
    try:
        saved = env.save()
        second = VectorGames(2, 200000, {}, state=saved)
        try:
            for _ in range(10):
                before, restored = env.observe(), second.observe()
                for name in before:
                    np.testing.assert_array_equal(before[name], restored[name])
                actions = before["action_mask"].argmax(-1)
                one, two = env.step(actions), second.step(actions)
                np.testing.assert_array_equal(one[0], two[0])
                assert one[2] == two[2]
        finally:
            second.close()
    finally:
        env.close()
    with pytest.raises(ValueError, match="train domain"):
        GameEpisode(1_000_000_000, 0, {})


def test_entity_model_masks_illegal_actions_and_uses_v2():
    torch.set_num_threads(2)
    env = VectorGames(2, 300000, {})
    try:
        obs = env.observe()
        model = EntityActorCritic()
        logits, value = model(*tensors(obs, "cpu"))
        assert logits.shape == (2, 245) and value.shape == (2,)
        probs = logits.softmax(-1).detach().numpy()
        assert (probs[~obs["action_mask"]] == 0).all()
        assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(-1), 1., atol=1e-6)
        # Own v2 counter is a real model input, not ignored or silently padded away.
        changed = {k: v.copy() for k, v in obs.items()}
        changed["features"][:, 3227+20] += 50
        with torch.no_grad():
            different, _ = model(*tensors(changed, "cpu"))
        assert not torch.allclose(logits[torch.tensor(obs["action_mask"])],
                                  different[torch.tensor(obs["action_mask"])])
    finally:
        env.close()


def test_real_ppo_updates_and_resume_match_uninterrupted_cpu(tmp_path):
    cfg = TrainConfig(seed=400000, n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=1)
    first = train(tmp_path/"one", cfg, updates=1, max_seconds=60)
    assert first["status"] == "updates_completed" and first["steps"] == 8
    resumed = train(tmp_path/"resume", updates=1, resume=tmp_path/"one/latest.pt", max_seconds=60)
    continuous = train(tmp_path/"two", cfg, updates=2, max_seconds=60)
    assert resumed["steps"] == continuous["steps"] == 16
    r, c = (load_checkpoint(tmp_path/name/"latest.pt") for name in ("resume", "two"))
    for key in r["model"]:
        torch.testing.assert_close(r["model"][key], c["model"][key], rtol=0, atol=0)
    assert r["environments"] == c["environments"]
    r["model_source_sha256"] = "0"*64
    torch.save(r, tmp_path/"incompatible.pt")
    with pytest.raises(ValueError, match="implementation hash"):
        load_checkpoint(tmp_path/"incompatible.pt")


def test_visible_v2_prefix_and_invalid_inputs():
    game = tg.Game(77)
    obs = ObservationV2.from_game(game, 0)
    f, c = game.observe(0)
    assert obs.legacy().features.tobytes() == f
    assert obs.legacy().cards.tobytes() == c
    invalid = obs.cards.astype(float)
    invalid[0] = .5
    with pytest.raises(ValueError, match="vocabulary"):
        ObservationV2(obs.features, invalid, obs.action_mask)


def test_training_sigterm_preserves_loadable_checkpoint(tmp_path):
    output = tmp_path/"signal-run"
    log = tmp_path/"signal.log"
    with log.open("w") as stream:
        process = subprocess.Popen([sys.executable, "-m", "omnibob.train", "--output", str(output),
                                    "--envs", "2", "--horizon", "64", "--updates", "1000",
                                    "--max-seconds", "60"], stdout=stream, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+30
            while not (output/"metrics.jsonl").exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.02)
            assert (output/"metrics.jsonl").exists(), log.read_text()
            process.terminate()
            assert process.wait(timeout=15) == 0, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    summary = json.loads((output/"summary.json").read_text())
    assert summary["status"] == "signal 15" and summary["error"] is None
    saved = load_checkpoint(output/"latest.pt")
    assert len(saved["environments"]["games"]) == 2


@pytest.mark.parametrize('opponents', ['mixed', 'learned', 'mixed_learned'])
def test_frozen_reference_kl_and_resumed_optimizer_match(tmp_path, opponents):
    cfg = TrainConfig(seed=700000, n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=1)
    train(tmp_path/"initial", cfg, updates=1, max_seconds=60)
    initial = tmp_path/"initial/latest.pt"
    cfg.reference_kl_coef, cfg.decision_cost = .1, .0003
    cfg.opponents, cfg.scheduler = opponents, "shuffled_rounds"
    if opponents in ('learned', 'mixed_learned'):
        from omnibob.export import export
        export(initial, tmp_path/'opponent', move_budget=7, repeat_limit=1)
        cfg.opponent_checkpoint = str(tmp_path/'opponent/policy.pt')
        cfg.workers = 2
    cfg.terminate_on_elimination = True
    train(tmp_path/"anchored", cfg, initialize=initial, updates=1, max_seconds=60)
    train(tmp_path/"resumed", resume=tmp_path/"anchored/latest.pt", updates=1, max_seconds=60)
    train(tmp_path/"continuous", cfg, initialize=initial, updates=2, max_seconds=60)
    original = load_checkpoint(initial)
    resumed = load_checkpoint(tmp_path/"resumed/latest.pt")
    continuous = load_checkpoint(tmp_path/"continuous/latest.pt")
    for key in original["model"]:
        torch.testing.assert_close(original["model"][key], resumed["reference_model"][key], rtol=0, atol=0)
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    assert resumed["environments"] == continuous["environments"]
    assert torch.equal(resumed["torch_rng"], continuous["torch_rng"])


def test_native_truncation_is_not_a_terminal_reward_and_resets_cleanly():
    from omnibob.train import valid_fragments
    rules = {'max_turns':1, 'max_actions_per_turn':1, 'turn_action_caps':[1]}
    env = VectorGames(2, 990000, rules, allow_truncated=True)
    try:
        flags, invalid = [], []
        for _ in range(2):
            actions = env.observe()['action_mask'].argmax(-1)
            rewards, dones, records = env.step(actions)
            assert not rewards.any()
            flags.append(dones)
            row = np.zeros(2, dtype=bool)
            for record in records:
                assert record['status'] == 'truncated' and record['placement'] is None
                assert not record['full_game_completed'] and record['reward_excluded']
                row[record['environment_index']] = True
            invalid.append(row)
        assert records and not valid_fragments(np.stack(flags), np.stack(invalid)).any()
        assert all(g.game.needs_action(g.seat) for g in env.games)
        assert env.episodes == [1, 1]
    finally:
        env.close()
    # A prior complete trajectory and a following fresh episode remain trainable.
    dones = np.array([[True], [False], [True], [False]])
    bad = np.array([[False], [False], [True], [False]])
    np.testing.assert_array_equal(valid_fragments(dones, bad)[:, 0], [True, False, False, True])


def test_truncation_budget_preserves_records_without_learning_forced_placement(tmp_path):
    rules = {'max_turns':1, 'max_actions_per_turn':1, 'turn_action_caps':[1]}
    cfg = TrainConfig(seed=990000, n_envs=2, horizon=2, epochs=1, minibatch=4, cpu_threads=1)
    summary = train(tmp_path/'censored', cfg, rules=rules, updates=2, max_seconds=60, max_truncated_episodes=8)
    assert summary['error'] is None and summary['truncated_episodes_this_run'] > 0
    assert summary['completed_games_this_run'] == summary['completed_episodes_this_run'] == 0
    assert summary['discarded_decisions_this_run'] > 0
    records = [json.loads(line) for line in (tmp_path/'censored/games.jsonl').read_text().splitlines()]
    assert all(r['placement'] is None and r['reward_excluded'] for r in records)
    stopped = train(tmp_path/'stopped', cfg, rules=rules, updates=2, max_seconds=60)
    assert stopped['status'] == 'failed' and 'truncation' in stopped['error']
    with pytest.raises(ValueError, match='earlier safe checkpoint'):
        train(tmp_path/'invalid-resume', resume=tmp_path/'stopped/latest.pt')
