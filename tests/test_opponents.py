import pytest

from omnibob.compare import compare
from omnibob.evaluate import evaluate, replay, run_game


def test_frozen_heuristic_lobby_seat_balance_and_profile_separation(tmp_path):
    same = evaluate(tmp_path/"heur", ["omnibob"], split="train", games=8, opponents="heuristic")
    result = same["policies"]["omnibob"]
    assert result["completed"] == 8 and result["illegal_actions"] == 0
    assert result["mean_placement"] == 4.5
    assert result["win_rate"] == .125 and result["top4_rate"] == .5
    evaluate(tmp_path/"native", ["omnibob"], split="train", games=8)
    with pytest.raises(ValueError, match="opponents"):
        compare(tmp_path/"heur", tmp_path/"native", "omnibob")


def test_mixed_opponent_game_is_reproducible_and_trace_replays(tmp_path):
    one, _ = run_game("omnibob", 850000, 3, {}, opponents="mixed")
    two, _ = run_game("omnibob", 850000, 3, {}, opponents="mixed")
    assert one["status"] == two["status"] == "completed"
    assert one["placements"] == two["placements"]
    assert one["steps"] == two["steps"]
    evaluate(tmp_path/"trace", ["omnibob"], split="train", seed=850000, games=1, seat=3,
             trace=True, opponents="mixed")
    assert replay(tmp_path/"trace")["verified_games"] == 1


def test_shuffled_scheduler_is_seat_balanced_and_replayable(tmp_path):
    summary = evaluate(tmp_path/"shuffled", ["omnibob"], split="train", games=8,
                       opponents="heuristic", scheduler="shuffled_rounds", trace=True)
    assert summary["policies"]["omnibob"]["mean_placement"] == 4.5
    assert summary["policies"]["omnibob"]["completed"] == 8
    assert replay(tmp_path/"shuffled")["verified_games"] == 8


@pytest.mark.parametrize("profile", ["learned", "mixed_learned"])
def test_learned_opponent_histories_are_isolated_and_only_receive_observations(profile):
    import json
    import numpy as np
    import taverngym as tg
    from omnibob.move_guard import MoveGuardPolicy
    from omnibob.observation_v2 import ObservationV2
    from omnibob.opponents import OpponentSet
    from omnibob.policies import Decision

    class VisibleOnly:
        version = 'test-visible'
        def act(self, obs):
            assert isinstance(obs, ObservationV2)
            assert not obs.features.flags.writeable
            moves = np.flatnonzero(obs.action_mask[196:])+196
            return Decision(int(moves[0]) if len(moves) else 0, 'test legal move')

    game = tg.Game(400)
    for seat in range(8):
        game.step(seat, 30)
    for seat in (1, 4):
        for _ in range(2):
            game.debug(json.dumps(dict(cmd='add_minion',p=seat,zone='board',card='BG25_354')))
        game.debug(json.dumps(dict(cmd='edit_minion',p=seat,zone='board',index=0,atk=123)))
    opponent = MoveGuardPolicy(VisibleOnly(), 7)
    group = OpponentSet(game, 400, 0, profile, opponent)
    group.act(1)
    assert group.learned[1].moves == 1
    assert group.learned[4].moves == 0 and opponent.moves == 0
    assert group.learned[1].base is group.learned[4].base
    if profile == 'mixed_learned':
        assert set(group.learned) == {1, 4, 7}
        assert set(group.policies) == {3, 6}
        group.act(3)
        assert group.policies[3].actions > 0
        assert group.policies[6].actions == 0
    saved = group.save()
    restored = OpponentSet(game, 400, 0, profile, opponent)
    restored.restore(saved)
    assert restored.learned[1].moves == 1 and restored.learned[4].moves == 0
    assert restored.save() == saved
    with pytest.raises(ValueError, match='specified together'):
        evaluate('unused-invalid-directory', ['omnibob'], opponents=profile)
