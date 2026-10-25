"""Inference-only contracts for the unified three-model payload benchmark."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from payload_diagnostics import ActorMassInput, ROOT, load_policy, sha256, tracking_metrics

RUNS = {
    "Asym_PPO": "Asym_PPO/2026-09-09_21-44-06",
    "DreamWaQ_CENet": "DreamWaQ_CENet/2026-09-11_20-51-03",
    "PayloadEstimator": "PayloadEstimator/2026-10-16_21-12-27",
}
CHECKPOINTS = {name: ROOT / "logs/rsl_rl/payload_comparison" / run / "model_4999.pt" for name, run in RUNS.items()}
OFFSETS = (-1., -.5, -.25, 0., .25, .5, 1.)
WIDTHS = (3, 3, 3, 12, 12, 12)


class PhysicalMassOffset(ActorMassInput):
    """Apply delta in kg before the saved normalization, preserving all context."""

    def _hook(self, module, args):
        features = args[0]
        self.native_input = features.detach().clone()
        if self.value_kg == 0.:
            self.used_input = self.native_input
            return None
        physical_kg = features[:, -1:] * self.scale + self.value_kg
        changed = features.clone()
        changed[:, -1:] = physical_kg / self.scale
        assert torch.equal(changed[:, :-1], features[:, :-1])
        self.used_input = changed.detach().clone()
        return (changed,)


class SavedConfigLoader(yaml.SafeLoader):
    pass


SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", lambda loader, node: slice(*loader.construct_sequence(node)))


def saved_env(model):
    return yaml.load((CHECKPOINTS[model].parent / "params/env.yaml").read_text(), Loader=SavedConfigLoader)


def canonical(value):
    return json.loads(json.dumps(value, default=str))


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        rows = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            rows.extend([dict(path=path, left=left.get(key), right=right.get(key))] if key not in left or key not in right
                        else differences(left[key], right[key], path))
        return rows
    return [] if left == right else [dict(path=prefix, left=left, right=right)]


def validate_saved_contracts():
    reference = canonical(saved_env("PayloadEstimator"))
    result = {}
    current_names = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
    for model in RUNS:
        cfg = canonical(saved_env(model))
        diff = differences(reference, cfg)
        if any(not row["path"].startswith("observations.") for row in diff):
            raise ValueError(f"Saved physical environments differ: {model}: {diff}")
        for term in current_names:
            assert cfg["observations"]["policy"][term] == reference["observations"]["policy"][term], term
            if model != "Asym_PPO":
                history = dict(cfg["observations"]["policy"]["history_" + term])
                long_history = dict(reference["observations"]["policy"]["history_" + term])
                assert history.pop("history_length") == (25 if model == "PayloadEstimator" else 5)
                long_history.pop("history_length")
                assert history == long_history
        result[model] = dict(differences=diff, env_sha256=sha256(CHECKPOINTS[model].parent / "params/env.yaml"))
    return result


def load_comparison_policy(model, device="cpu"):
    path = CHECKPOINTS[model]
    if model == "PayloadEstimator":
        return load_policy(path, device)
    agent = yaml.safe_load((path.parent / "params/agent.yaml").read_text())
    if agent["empirical_normalization"]:
        raise ValueError("Normalizer state must be handled explicitly.")
    config = dict(agent["policy"])
    class_name = config.pop("class_name")
    if model == "Asym_PPO":
        from rsl_rl.modules import ActorCritic
        assert class_name == "ActorCritic"
        policy = ActorCritic(45, 238, 12, **config)
    else:
        # Reuse the established tensor-only import and original CENet class.
        payload, _ = load_policy(CHECKPOINTS["PayloadEstimator"], "cpu")
        package = type(payload).__module__.rsplit(".", 1)[0]
        cls = importlib.import_module(package + ".dreamwaq_actor_critic").DreamWaQActorCritic
        assert class_name == "DreamWaQActorCritic"
        policy = cls(270, 238, 12, **config)
    policy.to(device).eval()
    data = torch.load(path, map_location=device, weights_only=True)
    match = torch.nn.Module.load_state_dict(policy, data["model_state_dict"], strict=True)
    return policy, dict(checkpoint=str(path), checkpoint_sha256=sha256(path), iteration=data["iter"],
                        missing_keys=match.missing_keys, unexpected_keys=match.unexpected_keys, agent=agent)


def model_observation(observation, model):
    if observation.shape[-1] != 1170:
        raise ValueError(observation.shape)
    if model == "PayloadEstimator":
        return observation
    if model == "Asym_PPO":
        return observation[:, :45]
    parts, offset = [observation[:, :45]], 45
    for width in WIDTHS:
        history = observation[:, offset:offset + 25 * width].reshape(-1, 25, width)
        parts.append(history[:, -5:].flatten(1))
        offset += 25 * width
    return torch.cat(parts, dim=1)


def initial_samples(base_seed, count):
    """One independent PRNG seed per trajectory; identical table for every model."""
    rows = []
    for index in range(count):
        seed = int(np.random.SeedSequence([base_seed, index]).generate_state(1)[0])
        rng = np.random.default_rng(seed)
        row = dict(trial=index, evaluation_seed=seed, x=rng.uniform(-.05, .05), y=rng.uniform(-.08, .08),
                   roll=rng.uniform(-.02, .02), pitch=rng.uniform(-.02, .02), yaw=rng.uniform(-.10, .10),
                   vx=rng.uniform(-.03, .03), vy=rng.uniform(-.02, .02), wz=rng.uniform(-.02, .02))
        row.update({f"joint_offset_{i}": rng.uniform(-.02, .02) for i in range(12)})
        row.update({f"joint_velocity_{i}": rng.uniform(-.02, .02) for i in range(12)})
        rows.append(row)
    return rows


def course_mesh(kind, size=(16., 20.)):
    """Deterministic walking surface; origin is on the flat approach at x=2 m."""
    import trimesh
    sx, sy = size
    start = 2.8
    if kind == "flat":
        profile = [(0., 0.), (sx, 0.)]
    elif kind == "slope":
        height = 2.4 * np.tan(np.deg2rad(10.))
        profile = [(0., 0.), (start, 0.), (start + 2.4, height), (sx, height)]
    elif kind == "stairs":
        profile = [(0., 0.), (start, 0.)]
        for i in range(6):
            x = start + i * .3
            if i:
                profile.append((x, i * .1))
            profile.append((x, (i + 1) * .1))
        profile.append((sx, .6))
    else:
        raise ValueError(kind)
    vertices = np.array([(x, y, z) for x, z in profile for y in (0., sy)])
    faces = np.array([(2*i, 2*i+2, 2*i+3) for i in range(len(profile)-1)] +
                     [(2*i, 2*i+3, 2*i+1) for i in range(len(profile)-1)])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False), np.array([2., sy/2, 0.])


def episode_metrics(rows, duration, goal_x):
    result = tracking_metrics(rows)
    failed = bool(result["failed"])
    survived = bool(rows) and not failed and rows[-1]["time_after_s"] >= duration - 1e-6
    reached = bool(rows) and max(r["x_progress"] for r in rows) >= goal_x
    result.update(survived=int(survived), reached_goal=int(reached), passed=int(survived and reached),
                  incomplete=int(survived and not reached), valid_runtime_s=result["observed_s"],
                  scored_runtime_s=result["scored_samples"] * .02,
                  mass_input_mae_kg=float(np.mean([abs(r["actor_mass_error_kg"]) for r in rows if r["scored"]]))
                  if any(r["scored"] for r in rows) else None,
                  mass_input_outside_fraction=float(np.mean([r["actor_mass_outside_0_8"] for r in rows])) if rows else None)
    return result


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
