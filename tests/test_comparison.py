import json

import pytest

from omnibob.compare import compare
from omnibob.evaluate import summarize


def test_pairing_uses_whole_common_blocks_and_rejects_rule_changes(tmp_path):
    folders = [tmp_path/"candidate", tmp_path/"baseline"]
    for folder, policy, place in zip(folders, ("ppo", "omnibob"), (2, 4)):
        folder.mkdir()
        (folder/"manifest.json").write_text(json.dumps({"config_sha256": "same", "split": "dev"}))
        rows = [{"policy": policy, "seed": seed, "seat": seat, "placement": place,
                 "status": "completed" if (seed, seat, policy) != (3, 0, "ppo") else "failed"}
                for seed in (1, 2, 3) for seat in range(8)]
        (folder/"games.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    result = compare(*folders)
    assert result["candidate_minus_baseline"] == -2
    assert result["paired_seed_bootstrap_95ci"] == [-2, -2]
    assert result["paired_games"] == 16 and result["excluded_common_pairs"] == 8
    (folders[1]/"manifest.json").write_text(json.dumps({"config_sha256": "different", "split": "dev"}))
    with pytest.raises(ValueError, match="config_sha256"):
        compare(*folders)


def test_incomplete_seed_block_keeps_descriptive_and_interval_denominators_explicit():
    rows = [{"policy": "ppo", "seed": seed, "seat": seat,
             "placement": 2 if seed < 3 else 8,
             "status": "truncated" if (seed,seat) == (3,0) else "completed",
             "illegal_actions": 0, "elapsed_seconds": 1}
            for seed in (1,2,3) for seat in range(8)]
    result = summarize(rows, ["ppo"], 24, {"ppo": []})["ppo"]
    assert result['completed'] == result['metric_denominator'] == 23
    assert result['truncated'] == 1
    assert result['mean_placement'] == pytest.approx((16*2+7*8)/23)
    assert result['complete_block_metric_denominator'] == 16
    assert result['complete_block_mean_placement'] == 2
    assert result['mean_placement_seed_bootstrap_95ci'] == [2,2]
    assert result['complete_block_mean_placement_seed_bootstrap_95ci'] == [2,2]
    assert 'complete_block_mean_placement' in result['confidence_interval_scope']


@pytest.mark.parametrize("profile", ["heuristic", "mixed", "mixed_learned"])
def test_same_opponent_name_cannot_hide_changed_frozen_policy(tmp_path, profile):
    folders = [tmp_path/"candidate", tmp_path/"baseline"]
    for folder, policy in zip(folders, ("ppo", "omnibob")):
        folder.mkdir()
        manifest = {"config_sha256": "same", "split": "dev", "opponent_profile": profile,
                    "opponents": "same friendly label", "source_files": {"omnibob/policies.py": "frozen-v1"}}
        (folder/"manifest.json").write_text(json.dumps(manifest))
        rows = [{"policy": policy, "seed": 1, "seat": seat, "placement": 4,
                 "status": "completed"} for seat in range(8)]
        (folder/"games.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    assert compare(*folders)["paired_games"] == 8
    for sources in ({"omnibob/policies.py": "changed-v2"}, {}):
        manifest["source_files"] = sources
        (folders[1]/"manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="frozen heuristic opponent"):
            compare(*folders)
