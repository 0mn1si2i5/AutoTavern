import json

import numpy as np
import pytest
import taverngym as tg

from omnibob.compat import check_compatibility, environment_contract
from omnibob.evaluate import evaluate, replay, run_game, schedule, summarize
from omnibob.observation import Observation
from omnibob.policies import Decision, OmniBobPolicy


def test_three_policies_complete_full_balanced_lobbies_and_replay(tmp_path):
    out = tmp_path / "run"
    result = evaluate(out, ["random", "scripted", "omnibob"], split="train", seed=71,
                      games=8, trace=True)
    for policy in result["policies"].values():
        assert policy["completed"] == 8
        assert policy["illegal_actions"] == 0
        assert policy["seats_attempted"] == {str(i): 1 for i in range(8)}
        assert 1 <= policy["mean_placement"] <= 8
    assert replay(out)["verified_games"] == 24
    with pytest.raises(FileExistsError):
        evaluate(out, ["omnibob"], split="train", games=8)


def test_seed_split_and_seat_pairing():
    all_seeds = []
    for split in ("train", "dev", "final"):
        pairs = schedule(split, 9, 16)
        assert len(set(pairs)) == 16
        assert len({s for s, _ in pairs}) == 2
        assert [p for _, p in pairs] == list(range(8)) * 2
        all_seeds.append({s for s, _ in pairs})
    assert not (all_seeds[0] & all_seeds[1] or all_seeds[1] & all_seeds[2])
    for split, seed, games in (("dev", -1, 8), ("train", 999_999_999, 16), ("dev", 0, 9)):
        with pytest.raises(ValueError):
            schedule(split, seed, games)


def test_hidden_opponent_zones_and_future_rng_do_not_enter_policy():
    game = tg.Game(3)
    for p in range(8):
        game.step(p, 30)
    # Set one non-revealing hero. Manipulation is test-only, never used by the policy.
    game.debug(json.dumps({"cmd": "set_hero", "p": 0, "hero": "TB_BaconShop_HERO_35"}))
    before = Observation.from_game(game, 0)
    game.debug(json.dumps({"cmd": "add_minion", "p": 1, "zone": "hand", "card": "BG28_300"}))
    game.debug(json.dumps({"cmd": "add_minion", "p": 1, "zone": "board", "card": "BG25_354"}))
    state = json.loads(game.state_json())
    state["rng"]["s"] = [1, 2, 3, 4]
    game = tg.Game.from_json(json.dumps(state))
    after = Observation.from_game(game, 0)
    np.testing.assert_array_equal(before.features, after.features)
    np.testing.assert_array_equal(before.cards, after.cards)
    np.testing.assert_array_equal(before.action_mask, after.action_mask)
    assert OmniBobPolicy().act(before) == OmniBobPolicy().act(after)
    assert not hasattr(before, "game") and not hasattr(before, "seed")
    with pytest.raises(ValueError):
        before.features[0] = 99


def test_stronger_owned_minion_can_replace_full_board():
    game = tg.Game(4)
    for p in range(8):
        game.step(p, 30)
    # Start from an actual recruit observation and use valid fixed-shape visible input.
    obs = Observation.from_game(game, 0)
    f, c = obs.features.copy(), obs.cards.copy()
    layout = dict(tg.layout())
    offset = layout["GLOBAL_F"] + 6 + 9
    f[offset:offset+7*31] = 0
    for i in range(7):
        f[offset+i*31:offset+i*31+5] = [1, .1, .1, .1, 1/6]
    h = offset+7*31
    f[h:h+10*40] = 0
    f[h:h+2] = [1, 1]
    f[h+9:h+14] = [1, 5, 5, 5, .5]
    f[33], f[34] = 1, .1
    mask = np.zeros(245, dtype=bool)
    mask[[0, 16, 17, 18, 19, 20, 21, 22]] = True
    decision = OmniBobPolicy().act(Observation(f, c, mask))
    assert 16 <= decision.action <= 22
    assert "stronger owned" in decision.reason


def test_illegal_policy_and_budget_interrupt_are_retained():
    class Invalid:
        def act(self, observation):
            return Decision(999, "deliberately invalid")
    row, _ = run_game("omnibob", 1, 0, {}, policy_override=Invalid())
    assert row["status"] == "failed" and row["illegal_actions"] == 1
    assert row["placement"] is None and "illegal action" in row["error"]
    row, _ = run_game("random", 1, 0, {}, max_steps=1)
    assert row["status"] == "interrupted" and row["placement"] is None
    class Interrupt:
        def act(self, observation):
            raise KeyboardInterrupt
    row, _ = run_game("omnibob", 1, 0, {}, policy_override=Interrupt())
    assert row["status"] == "interrupted" and row["error"] == "KeyboardInterrupt"


def test_truncated_games_excluded_and_not_started_reported():
    row, latency = run_game("scripted", 2, 0, {"max_turns": 1})
    assert row["status"] == "truncated" and row["turn_limit_end"]
    summary = summarize([row], ["scripted"], 8, {"scripted": latency})["scripted"]
    assert summary["mean_placement"] is None and summary["metric_denominator"] == 0
    assert summary["truncated"] == 1 and summary["not_started"] == 7


def test_compatibility_rejects_same_shape_different_vocabulary():
    expected = environment_contract()
    expected["vocabulary_sha256"] = "different vocabulary, identical dimensions"
    with pytest.raises(ValueError, match="vocabulary_sha256"):
        check_compatibility(expected)
    with pytest.raises(ValueError, match="features"):
        Observation(np.zeros(4017), np.zeros(124), np.zeros(245))


def test_repeated_game_same_seed_reproduces_actions():
    from io import StringIO
    traces = [StringIO(), StringIO()]
    outcomes = [run_game("omnibob", 13, 4, {}, trace=t)[0] for t in traces]
    assert outcomes[0]["placements"] == outcomes[1]["placements"]
    actions = [[(r["seat"], r["action"], r["visible_sha256"])
                for r in map(json.loads, t.getvalue().splitlines())] for t in traces]
    assert actions[0] == actions[1]


def test_evaluation_sigterm_preserves_partial_outcomes(tmp_path):
    import json
    import subprocess
    import sys
    import time
    output = tmp_path/'interrupted-evaluation'
    with (tmp_path/'evaluation.log').open('w') as log:
        process = subprocess.Popen([sys.executable,'-m','omnibob','evaluate','--policies','random',
                                    '--games','8000','--output',str(output)],stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic()+20
            while not (output/'events.jsonl').exists() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(.01)
            assert (output/'events.jsonl').exists()
            process.terminate()
            assert process.wait(timeout=15) == 2
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    summary = json.loads((output/'summary.json').read_text())
    assert summary['user_interrupted'] and summary['termination_signal'] == 15
    counts = summary['policies']['random']
    assert sum(counts[k] for k in ['completed','failed','interrupted','truncated','not_started']) == 8000
    assert counts['completed'] < 8000
def test_checkpoint_cannot_be_silently_ignored_by_a_baseline(tmp_path):
    import pytest
    from omnibob.evaluate import evaluate

    with pytest.raises(ValueError, match="checkpoint requires a ppo or bc"):
        evaluate(tmp_path/"wrong-policy", ["omnibob"], games=8, checkpoint=tmp_path/"missing.pt")
    assert not (tmp_path/"wrong-policy").exists()

