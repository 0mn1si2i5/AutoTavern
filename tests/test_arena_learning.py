import copy
import json
from pathlib import Path

import pytest
import taverngym as tg

pytest.importorskip("torch", reason="optional training dependencies")

from omnibob import arena
from omnibob.compat import ROOT, effective_config
from omnibob.imitation import collect, fit
from omnibob.export import export
from omnibob.learned import LearnedPolicy, load_policy
from omnibob.move_guard import MoveGuardPolicy
from omnibob.observation_v2 import ObservationV2
from omnibob.train import TrainConfig, load_checkpoint, train


def test_arena_contract_checks_encoding_catalog_and_rules():
    manifest = json.loads((ROOT/"configs/arena-manifest-v2.json").read_text())
    assert arena.validate_manifest(manifest, manifest["rules"])["queued"] is False
    for key in ("catalog_sha256", "action_layout_sha256"):
        broken = copy.deepcopy(manifest)
        broken[key] = "0"*64
        with pytest.raises(ValueError, match="incompatibility"):
            arena.validate_manifest(broken)
    with pytest.raises(ValueError, match="checkpoint rules"):
        arena.validate_manifest(manifest, {})


def test_arena_adapter_records_only_supplied_view(tmp_path, monkeypatch):
    monkeypatch.delenv("OMNIBOB_CHECKPOINT", raising=False)
    monkeypatch.setenv("OMNIBOB_JOURNAL_DIR", str(tmp_path))
    game = tg.Game(400)
    obs = ObservationV2.from_game(game, 0)
    context = dict(state={}, match_id="../untrusted", seat=0, turn=1,
                   player_view={"visible_marker": 9})
    action = arena.act({key: getattr(obs, key) for key in ("features", "cards", "action_mask")}, context)
    assert obs.action_mask[action]
    record = json.loads((tmp_path/"decisions.jsonl").read_text())
    assert record["player_view"] == context["player_view"]
    assert list(tmp_path.iterdir()) == [tmp_path/"decisions.jsonl"]


def test_imitation_checkpoint_is_deployable_and_only_initializes_ppo(tmp_path, monkeypatch):
    rules = effective_config(json.loads((ROOT/"configs/arena-v2.json").read_text()))
    dataset = tmp_path/"data"
    assert collect(dataset, games=2, seed=810000, rules=rules)["games_completed"] == 2
    result = fit(dataset, tmp_path/"bc", epochs=1, device="cpu", max_seconds=60)
    checkpoint = tmp_path/"bc/best.pt"
    assert result["checkpoint"] and checkpoint.exists()
    policy = LearnedPolicy(checkpoint)
    assert policy.training_method == "behavior_cloning"
    observation = ObservationV2.from_game(tg.Game(811000), 0)
    assert observation.action_mask[policy.act(observation).action]
    export(checkpoint, tmp_path/"deployment")
    portable = tmp_path/"deployment/policy.pt"
    assert LearnedPolicy(portable).act(observation).action == policy.act(observation).action
    assert not {"optimizer", "environments", "sampling_rng", "torch_rng"} & load_checkpoint(portable).keys()
    export(checkpoint, tmp_path/"guarded", move_budget=7)
    guarded = tmp_path/"guarded/policy.pt"
    assert isinstance(load_policy(guarded), MoveGuardPolicy)
    assert not isinstance(load_policy(guarded, move_budget=0), MoveGuardPolicy)
    monkeypatch.setenv("OMNIBOB_CHECKPOINT", str(guarded))
    monkeypatch.delenv("OMNIBOB_JOURNAL_DIR", raising=False)
    contexts = [dict(state={}, match_id=str(i), seat=0, turn=1) for i in range(2)]
    for context in contexts:
        action = arena.act({k: getattr(observation, k) for k in ("features", "cards", "action_mask")}, context)
        assert observation.action_mask[action]
    first, second = [context["state"]["learned_policy"] for context in contexts]
    assert first is not second and first.base is second.base
    first.moves = 7
    assert second.moves == 0
    broken = load_checkpoint(guarded)
    broken["runtime_config"]["move_guard"]["source_sha256"] = "0"*64
    import torch
    torch.save(broken, tmp_path/"broken-runtime.pt")
    with pytest.raises(ValueError, match="controller implementation"):
        load_policy(tmp_path/"broken-runtime.pt")
    export(checkpoint, tmp_path/"combined", move_budget=7, repeat_limit=1)
    combined = tmp_path/"combined/policy.pt"
    assert load_policy(combined).repeat_limit == 1
    assert load_policy(combined, repeat_limit=0).move_guard_budget == 7
    monkeypatch.setenv("OMNIBOB_CHECKPOINT", str(combined))
    combined_contexts = [dict(state={}, match_id=str(i), seat=0, turn=1) for i in range(2)]
    for context in combined_contexts:
        arena.act({k: getattr(observation, k) for k in ("features", "cards", "action_mask")}, context)
    one, two = [context["state"]["learned_policy"] for context in combined_contexts]
    assert one is not two and one.counts is not two.counts
    assert one.base is not two.base and one.base.base is two.base.base
    with pytest.raises(ValueError, match="deployment weights"):
        train(tmp_path/"bad-portable-resume", resume=portable)
    with pytest.raises(ValueError, match="initialize"):
        train(tmp_path/"bad-resume", resume=checkpoint)
    cfg = TrainConfig(n_envs=2, horizon=4, epochs=1, minibatch=8, cpu_threads=1)
    summary = train(tmp_path/"ppo", config=cfg, initialize=checkpoint, rules=rules, updates=1, max_seconds=60)
    assert summary["steps"] == 8
    assert LearnedPolicy(tmp_path/"ppo/latest.pt").training_method == "ppo"
