"""Evaluation-only payload interventions and strict loading; no simulator imports."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / (
    "logs/rsl_rl/payload_comparison/PayloadEstimator/2026-10-16_21-12-27/model_4999.pt"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(checkpoint, device="cpu"):
    """Use saved architecture, not mutable runner defaults; require every weight."""
    checkpoint = Path(checkpoint).resolve()
    agent = yaml.safe_load((checkpoint.parent / "params/agent.yaml").read_text())
    if agent["empirical_normalization"]:
        raise ValueError("This diagnostic requires the saved non-normalized PayloadEstimator run.")
    config = dict(agent["policy"])
    if config.pop("class_name") != "DreamWaQPayloadActorCritic":
        raise ValueError("Expected a PayloadEstimator checkpoint.")
    # Same tensor-only import technique as test_payload_estimator.py; no task registration.
    name = "_payload_diagnostic_agents"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / "source/dreamwaq_go2/dreamwaq_go2/tasks/manager_based/dreamwaq_go2/agents")]
        sys.modules[name] = package
    cls = importlib.import_module(name + ".payload_actor_critic").DreamWaQPayloadActorCritic
    policy = cls(45 * (1 + config["payload_history_length"]), 239, 12, **config).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    # The RSL-RL-compatible override returns a resume boolean, not _IncompatibleKeys.
    # Module's implementation performs the identical recursive strict tensor load.
    loaded = torch.nn.Module.load_state_dict(policy, saved["model_state_dict"], strict=True)
    policy.eval()
    if not policy.actor.use_payload_input:
        raise ValueError("Checkpoint configuration disables the mass input.")
    info = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                iteration=saved["iter"], missing_keys=loaded.missing_keys,
                unexpected_keys=loaded.unexpected_keys, agent=agent,
                mass_column_l2=policy.actor.base_actor.policy_head[0].weight[:, -1].norm().item())
    return policy, info


class ActorMassInput:
    """Replace only column 64 at the original policy-head boundary.

    The original forward still computes the CENet and payload estimator normally.
    A no-op predicted mode returns the original arguments, without copying actions.
    Never write observations, parameters, physics, velocity or latent features.
    """

    def __init__(self, policy, mode="predicted", value_kg=0.0):
        if mode not in ("predicted", "oracle", "fixed", "offset"):
            raise ValueError(mode)
        self.policy, self.mode, self.value_kg = policy, mode, value_kg
        self.scale = policy.actor.payload_scale_kg
        self.true_kg = None
        self.native_input = self.used_input = None
        self.handle = policy.actor.base_actor.policy_head.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        features = args[0]
        if features.shape[-1] != 65:
            raise ValueError(f"Unexpected actor head input: {features.shape}")
        self.native_input = features.detach().clone()
        if self.mode == "predicted":
            self.used_input = self.native_input
            return None
        changed = features.clone()
        if self.mode == "oracle":
            if self.true_kg is None:
                raise ValueError("Oracle requires measured simulation mass before inference.")
            changed[:, -1:] = self.true_kg.reshape(-1, 1).to(features) / self.scale
        elif self.mode == "fixed":
            changed[:, -1] = self.value_kg / self.scale
        else:
            changed[:, -1] += self.value_kg / self.scale
        if not torch.equal(changed[:, :-1], features[:, :-1]):
            raise AssertionError("Intervention changed non-mass features.")
        self.used_input = changed.detach().clone()
        return (changed,)

    def close(self):
        self.handle.remove()


@torch.inference_mode()
def probe_same_state(policy, features, true_kg):
    """Evaluate counterfactual actions from one frozen 64-dimensional context.

    Invoke the layers, bypassing the enclosing Sequential's intervention hook.
    This keeps a running closed-loop intervention from contaminating the probe.
    """
    scale = policy.actor.payload_scale_kg
    predicted = features[:, -1:] * scale
    masses = {"predicted": predicted, "oracle": true_kg.reshape(-1, 1).to(features)}
    masses.update({f"fixed_{v:g}": torch.full_like(predicted, v) for v in (0., 4., 8.)})
    masses.update({f"offset_{v:+g}": predicted + v for v in (-2., -.5, .5, 2.)})
    outputs = {}
    for mode, mass in masses.items():
        changed = features.clone()
        changed[:, -1:] = mass / scale
        assert torch.equal(changed[:, :-1], features[:, :-1])
        value = changed
        for layer in policy.actor.base_actor.policy_head:
            value = layer(value)
        outputs[mode] = (mass, value)
    return outputs


def tracking_metrics(rows):
    """First episode only; include terminal pre-reset state and disclose censoring."""
    selected = [r for r in rows if r["scored"]]
    result = {"scored_samples": len(selected), "failed": int(any(r["terminated"] for r in rows)),
              "timeout": int(any(r["timeout"] for r in rows)),
              "observed_s": rows[-1]["time_after_s"] if rows else 0.0}
    for key in ("vx_error", "vy_error", "wz_error", "roll", "pitch"):
        result[key + ("_rmse" if "error" in key else "_rms")] = (
            float(np.sqrt(np.mean([r[key] ** 2 for r in selected]))) if selected else None
        )
    result["linear_xy_vector_rmse"] = (
        float(np.sqrt(np.mean([r["vx_error"] ** 2 + r["vy_error"] ** 2 for r in selected])))
        if selected else None
    )
    for label, subset in (("all", rows), ("steady", selected)):
        errors = np.asarray([r["mass_error_kg"] for r in subset])
        result[f"mass_mae_{label}_kg"] = float(np.abs(errors).mean()) if len(errors) else None
        result[f"mass_bias_{label}_kg"] = float(errors.mean()) if len(errors) else None
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
