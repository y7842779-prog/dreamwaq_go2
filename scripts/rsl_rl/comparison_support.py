"""Inference-only contracts and metrics for paired payload evaluation.

Reuses the October 25 strict loading, term-major observation adapter, initial
sampling and physical pairing. No training implementation is modified.
"""
from __future__ import annotations
import csv
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
RUNS = {
    "Asym_PPO": "Asym_PPO/2026-09-09_21-44-06",
    "DreamWaQ_CENet": "DreamWaQ_CENet/2026-09-11_20-51-03",
    "PayloadEstimator": "PayloadEstimator/2026-10-16_21-12-27",
}
CHECKPOINTS = {name: ROOT / "logs/rsl_rl/payload_comparison" / run / "model_4999.pt" for name, run in RUNS.items()}
WIDTHS = (3, 3, 3, 12, 12, 12)

class SavedConfigLoader(yaml.SafeLoader):
    pass


SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice", lambda loader, node: slice(*loader.construct_sequence(node)))


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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


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
        expected_terms = list(current_names)
        if model != "Asym_PPO":
            expected_terms += ["history_"+name for name in current_names]
        actual_terms = [name for name in cfg["observations"]["policy"]
                        if name in current_names or name.startswith("history_") and name != "history_length"]
        assert actual_terms == expected_terms, (model, actual_terms)
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
        result[model] = dict(differences=diff, observation_terms=actual_terms,
                            env_sha256=sha256(CHECKPOINTS[model].parent / "params/env.yaml"))
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


def validate_initial_pair(initial, reference):
    """PhysX quaternion normalization can differ at sub-nanoradian precision.

    Masses, inertias, materials and gains remain exact. Only states and derived
    observations allow 1e-7 absolute floating-point error, with actual errors saved.
    """
    checks = {}
    for key, left in initial.items():
        right = reference[key]
        if key in ("root_state", "full_policy_observation"):
            a, b = np.asarray(left), np.asarray(right)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-7, err_msg=key)
            checks[key] = dict(max_abs_difference=float(np.max(np.abs(a-b))), absolute_tolerance=1e-7)
        else:
            if left != right:
                raise AssertionError(f"Physical initial-condition mismatch: {key}")
            checks[key] = dict(exactly_equal=True)
    return checks


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


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


METRICS = ("base_roll_deviation_rad", "base_flatness_penalty", "velocity_tracking_error_m_s")


def trajectory_metrics(rows, duration=12., warmup=2., dt=.02):
    """One statistic per first episode; terminal pre-reset samples are retained.

    Post-step states on intervals beginning at >= warmup are scored: (2,12] s
    at 50 Hz, exactly 500 samples for a complete formal trial. Failed episodes
    are retained and identified as censored; missing/invalid metrics stay blank.
    Motion checks are diagnostics and never select or remove primary samples.
    """
    selected = [r for r in rows if r["time_before_s"] >= warmup - 1e-9]
    fields = ("roll", "gx", "gy", "gz", "vx", "vy", "cmd_vx", "cmd_vy")
    invalid = sum(not all(np.isfinite(r[k]) for k in fields) for r in selected)
    gravity_errors = [abs(np.linalg.norm([r[k] for k in ("gx", "gy", "gz")]) - 1.)
                      for r in selected if all(np.isfinite(r[k]) for k in ("gx", "gy", "gz"))]
    gravity_norm_error = max(gravity_errors, default=0.)
    invalid += int(gravity_norm_error > 1e-5)
    observed = rows[-1]["time_after_s"] if rows else 0.
    failed = int(any(r["terminated"] for r in rows))
    timeout = int(any(r["timeout"] for r in rows))
    complete = int(bool(rows) and not failed and not timeout and observed >= duration - 1e-6)
    result = dict(samples=len(rows), scored_samples=len(selected), observed_s=observed,
                  scored_s=len(selected)*dt, failed=failed, timeout=timeout, complete=complete,
                  censored=int(not complete), invalid_samples=invalid,
                  max_gravity_norm_error=gravity_norm_error,
                  **{key: None for key in METRICS})
    if selected and not invalid:
        result.update(base_roll_deviation_rad=float(np.sqrt(np.mean([r["roll"]**2 for r in selected]))),
                      base_flatness_penalty=float(np.mean([r["gx"]**2+r["gy"]**2 for r in selected])),
                      velocity_tracking_error_m_s=float(np.sqrt(np.mean([
                          (r["vx"]-r["cmd_vx"])**2+(r["vy"]-r["cmd_vy"])**2 for r in selected]))))
    vx = [r["vx"] for r in selected]
    longest = consecutive = 0
    for v in vx:
        consecutive = consecutive + 1 if np.isfinite(v) and v < .1 else 0
        longest = max(longest, consecutive)
    mean_vx = float(np.mean(vx)) if vx and np.isfinite(vx).all() else None
    result.update(mean_forward_velocity_m_s=mean_vx,
                  low_forward_speed_fraction=float(np.mean(np.asarray(vx)<.1)) if vx else None,
                  longest_low_forward_speed_s=longest*dt,
                  stall_flag=int(mean_vx is not None and (mean_vx < .1 or longest*dt >= 2.)),
                  final_world_x_progress_m=rows[-1]["x_progress"] if rows else None)
    return result
