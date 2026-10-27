"""Deployable observation-only policy loaded from a verified local checkpoint."""
from pathlib import Path

import torch

from .model import EntityActorCritic, ModelConfig
from .policies import Decision
from .train import load_checkpoint


class LearnedPolicy:
    observation_version = 2

    def __init__(self, checkpoint, device="cpu", cpu_threads=1):
        torch.set_num_threads(cpu_threads)
        self.cpu_threads = cpu_threads
        state = load_checkpoint(checkpoint)
        if state.get("error"):
            raise ValueError("checkpoint belongs to a failed training state; select an earlier safe checkpoint")
        self.config = state["config"]
        self.device = device
        self.model = EntityActorCritic(ModelConfig(**state["model_config"])).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.checkpoint = str(Path(checkpoint).resolve())
        self.training_steps = state["steps"]
        self.runtime_config = state.get("runtime_config", {})
        self.move_guard_budget = 0
        self.repeat_limit = 0
        self.training_method = state.get("training_method", "ppo")
        self.version = (f"bc-epoch-{state['imitation_epoch']}" if self.training_method == "behavior_cloning"
                        else f"ppo-{state['steps']}-steps")
        self.version = state.get("policy_label", self.version)

    def act(self, observation):
        with torch.inference_mode():
            # Copy the policy boundary's read-only arrays before wrapping in torch.
            f = torch.tensor(observation.features[None], device=self.device)
            c = torch.tensor(observation.cards[None], dtype=torch.long, device=self.device)
            mask = torch.tensor(observation.action_mask[None], dtype=torch.bool, device=self.device)
            logits, value = self.model(f, c, mask)
            action = int(logits.argmax(-1).item())
        return Decision(action, f"{self.version}: highest legal logit; value estimate={value.item():.3f}")


def load_policy(checkpoint, device="cpu", cpu_threads=1, move_budget=None, repeat_limit=None):
    """Load a deployment's recorded controller, unless an explicit ablation overrides it."""
    from .move_guard import MoveGuardPolicy, runtime_config
    from .repeat_guard import RepeatGuardPolicy, runtime_config as repeat_config
    base = LearnedPolicy(checkpoint, device, cpu_threads)
    if set(base.runtime_config)-{"move_guard", "repeat_guard"}:
        raise ValueError("unsupported deployment runtime")
    if move_budget is None:
        recorded = base.runtime_config.get("move_guard")
        move_budget = recorded["budget"] if recorded else 0
        if recorded and runtime_config(move_budget).get("move_guard") != recorded:
            raise ValueError("incompatible move controller implementation; migrate or explicitly override for an ablation")
    runtime_config(move_budget)  # Validate explicit overrides too.
    if repeat_limit is None:
        recorded = base.runtime_config.get("repeat_guard")
        repeat_limit = recorded["limit"] if recorded else 0
        if recorded and repeat_config(repeat_limit).get("repeat_guard") != recorded:
            raise ValueError("incompatible repeat controller implementation")
    repeat_config(repeat_limit)
    base.runtime_config = {}
    policy = MoveGuardPolicy(base, move_budget) if move_budget else base
    return RepeatGuardPolicy(policy, repeat_limit) if repeat_limit else policy
