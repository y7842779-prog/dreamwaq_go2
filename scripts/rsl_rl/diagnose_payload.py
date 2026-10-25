"""Small Isaac Lab payload diagnostics using the trained policy and original task.

Run one mass/mode/seed per process to isolate PhysX state. Outputs are CSV/JSON/PNG.
This entry point performs inference only and never saves/exports model weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--mass-kg", type=float, default=4.)
parser.add_argument("--mode", choices=("predicted", "oracle", "fixed", "offset"), default="predicted")
parser.add_argument("--value-kg", type=float, default=0.)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--duration", type=float, default=12., help="Total duration including warmup, seconds.")
parser.add_argument("--warmup", type=float, default=2.)
parser.add_argument("--command", type=float, nargs=3, default=(.6, 0., 0.), metavar=("VX", "VY", "WZ"))
parser.add_argument("--probe-stride", type=int, default=10, help="0 disables same-state probes.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs < 1 or args.mass_kg < 0 or not 0 <= args.warmup < args.duration or args.probe_stride < 0:
    parser.error("Invalid environment count, mass, warmup, duration or probe stride.")
args.output_dir = args.output_dir.resolve()
args.output_dir.mkdir(parents=True, exist_ok=True)
if (args.output_dir / "manifest.json").exists() or (args.output_dir / "trajectory.csv").exists():
    parser.error("Use a new output directory; diagnostics never overwrite a run.")

app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
import yaml

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/dreamwaq_go2"))
import dreamwaq_go2.tasks  # noqa: F401, E402

from payload_diagnostics import ActorMassInput, load_policy, probe_same_state, sha256, tracking_metrics, write_json

TASK = "Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0"
ROOT = Path(__file__).resolve().parents[2]


class SnapshotRecorder(RecorderTerm):
    """Read post-physics state before automatic resets using Isaac Lab's recorder API."""

    def record_post_step(self):
        robot = self._env.scene["robot"]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        self._env.payload_diagnostic_state = dict(
            lin=robot.data.root_lin_vel_b.clone(), ang=robot.data.root_ang_vel_b.clone(),
            roll=wrap_to_pi(roll).clone(), pitch=wrap_to_pi(pitch).clone(),
            height=robot.data.root_pos_w[:, 2].clone(),
            terminated=self._env.termination_manager.terminated.clone(),
            timeout=self._env.termination_manager.time_outs.clone(),
        )
        return None, None


class SavedConfigLoader(yaml.SafeLoader):
    pass


SavedConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))
SavedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice", lambda loader, node: slice(*loader.construct_sequence(node))
)


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(dict(path=path, saved=left.get(key), current=right.get(key)))
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [dict(path=prefix, saved=left, current=right)]


def canonical(value):
    return json.loads(json.dumps(value, default=str))


def evaluation_cfg():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    saved = yaml.load((args.checkpoint.parent / "params/env.yaml").read_text(), Loader=SavedConfigLoader)
    current = canonical(cfg.to_dict())
    raw_drift = differences(canonical(saved), current)
    # Training YAML was dumped after scene/generator construction resolved these fields.
    for name in ("robot", "contact_forces", "height_scanner"):
        current["scene"][name]["prim_path"] = current["scene"][name]["prim_path"].replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    terrain = current["scene"]["terrain"]
    terrain["num_envs"], terrain["env_spacing"] = current["scene"]["num_envs"], current["scene"]["env_spacing"]
    for subterrain in terrain["terrain_generator"]["sub_terrains"].values():
        subterrain["size"] = terrain["terrain_generator"]["size"]
        if "slope_threshold" in subterrain:
            subterrain["slope_threshold"] = terrain["terrain_generator"]["slope_threshold"]
    drift = differences(canonical(saved), current)
    write_json(args.output_dir / "training_config_differences.json", dict(raw=raw_drift, after_scene_resolution=drift))
    # Fail instead of silently changing a saved observation/actuator/reward contract.
    allowed = ("seed", "sim.device", "scene.num_envs", "viewer.")
    critical = [d for d in drift if not any(d["path"] == p or d["path"].startswith(p) for p in allowed)]
    if critical:
        raise ValueError(f"Training config drift; inspect training_config_differences.json: {critical}")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum.terrain_levels = None
    cfg.curriculum.payload_dr_levels = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (args.mass_kg, args.mass_kg)
    # Keep the training mass event's additive mass and recompute_inertia semantics.
    for event in ("base_com", "push_robot", "base_external_force_torque", "physics_material"):
        if hasattr(cfg.events, event):
            setattr(cfg.events, event, None)
    # Paired seeds vary reset yaw (and only yaw); command is expressed in body frame.
    cfg.events.reset_base.params["pose_range"] = {"x": (0., 0.), "y": (0., 0.), "yaw": (-.15, .15)}
    for group in (cfg.observations.policy, cfg.observations.estimator, cfg.observations.critic):
        group.enable_corruption = False
    command = cfg.commands.base_velocity
    command.resampling_time_range = (1.e6, 1.e6)
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = command.debug_vis = False
    command.ranges.lin_vel_x = (args.command[0], args.command[0])
    command.ranges.lin_vel_y = (args.command[1], args.command[1])
    command.ranges.ang_vel_z = (args.command[2], args.command[2])
    cfg.episode_length_s = args.duration + 1.
    cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    cfg.recorders.snapshot = RecorderTermCfg(class_type=SnapshotRecorder)
    dump_yaml(str(args.output_dir / "effective_env.yaml"), cfg)
    write_json(args.output_dir / "evaluation_overrides.json", differences(canonical(saved), canonical(cfg.to_dict())))
    return cfg


def open_csv(name, fields, files):
    handle = (args.output_dir / name).open("w", newline="")
    files.append(handle)
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    return writer


def main():
    torch.set_num_threads(1)
    cfg = evaluation_cfg()
    policy, load_info = load_policy(args.checkpoint, cfg.sim.device)
    write_json(args.output_dir / "checkpoint_load.json", load_info)
    env = gym.make(TASK, cfg=cfg).unwrapped
    intervention = ActorMassInput(policy, args.mode, args.value_kg)
    files = []
    try:
        with torch.inference_mode():
            obs, _ = env.reset(seed=args.seed)
            robot = env.scene["robot"]
            base_ids = robot.find_bodies("base")[0]
            if len(base_ids) != 1 or obs["policy"].shape != (args.num_envs, 1170):
                raise AssertionError("Unexpected base or policy observations")
            if obs["critic"].shape != (args.num_envs, 239) or not math.isclose(env.step_dt, .02):
                raise AssertionError("Unexpected critic observations or control period")
            actual = robot.root_physx_view.get_masses().clone()
            true_kg = (actual[:, base_ids] - robot.data.default_mass.cpu()[:, base_ids]).sum(-1)
            torch.testing.assert_close(true_kg, torch.full_like(true_kg, args.mass_kg), rtol=0, atol=2e-6)
            torch.testing.assert_close(obs["critic"][:, -1].cpu(), true_kg, rtol=0, atol=2e-6)
            intervention.true_kg = true_kg
            initial = dict(root_state=robot.data.root_state_w.cpu().tolist(),
                           joint_pos=robot.data.joint_pos.cpu().tolist(), joint_vel=robot.data.joint_vel.cpu().tolist(),
                           masses=actual.tolist(), inertias=robot.root_physx_view.get_inertias().tolist(),
                           coms=robot.root_physx_view.get_coms().tolist(),
                           materials=robot.root_physx_view.get_material_properties().tolist(),
                           policy_observation=obs["policy"].cpu().tolist(),
                           command=env.command_manager.get_command("base_velocity").cpu().tolist())
            write_json(args.output_dir / "initial_state.json", initial)
            # Noise is off: newest history entries must equal their current term.
            current_offset, history_offset = 0, 45
            for width in (3, 3, 3, 12, 12, 12):
                history = obs["policy"][:, history_offset:history_offset + 25 * width].reshape(args.num_envs, 25, width)
                torch.testing.assert_close(history[:, -1], obs["policy"][:, current_offset:current_offset + width])
                current_offset += width
                history_offset += 25 * width
            trajectory_fields = ["seed", "env_id", "step", "time_before_s", "time_after_s", "mode", "true_mass_kg",
                                 "predicted_mass_kg", "actor_mass_kg", "mass_error_kg", "abs_mass_error_kg",
                                 "actor_mass_outside_0_8", "scored", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz",
                                 "vx_error", "vy_error", "wz_error", "roll", "pitch", "height", "terminated", "timeout"]
            trajectory_fields += [f"action_{i}" for i in range(12)]
            trace = open_csv("trajectory.csv", trajectory_fields, files)
            feature_writer = probe_writer = observation_writer = None
            if args.probe_stride:
                feature_writer = open_csv("actor_features.csv", ["seed", "env_id", "step", "time_s", "true_mass_kg"] +
                                          [f"feature_{i}" for i in range(65)], files)
                observation_writer = open_csv("observations.csv", ["env_id", "step"] + [f"obs_{i}" for i in range(1170)], files)
                probe_writer = open_csv("action_sensitivity.csv", ["seed", "env_id", "step", "time_s", "mode", "true_mass_kg",
                                        "predicted_mass_kg", "actor_mass_kg", "normalized_mass", "outside_0_8", "scored",
                                        "action_delta_l2", "action_delta_rms", "action_delta_max", "target_delta_rms_rad"] +
                                        [f"action_{i}" for i in range(12)] + [f"delta_{i}" for i in range(12)], files)
            alive = np.ones(args.num_envs, dtype=bool)
            rows = [[] for _ in range(args.num_envs)]
            max_forward_error = 0.
            count = round(args.duration / env.step_dt)
            for step in range(count):
                command = env.command_manager.get_command("base_velocity").cpu().numpy().copy()
                actions = policy.act_inference(obs["policy"])
                features = intervention.native_input
                prediction = features[:, -1].cpu().numpy() * policy.actor.payload_scale_kg
                used_mass = intervention.used_input[:, -1].cpu().numpy() * policy.actor.payload_scale_kg
                if not torch.isfinite(actions).all() or not torch.isfinite(features).all():
                    raise FloatingPointError("Non-finite inference")
                if args.probe_stride and step % args.probe_stride == 0:
                    # Features are captured once; probes do not rerun encoders/history.
                    probes = probe_same_state(policy, features, true_kg)
                    native = probes["predicted"][1]
                    if args.mode == "predicted":
                        error = (actions - native).abs().max().item()
                        max_forward_error = max(max_forward_error, error)
                        torch.testing.assert_close(actions, native, rtol=0, atol=1e-6)
                    for e in np.flatnonzero(alive):
                        feature_writer.writerow(dict(seed=args.seed, env_id=e, step=step, time_s=step * env.step_dt,
                                                    true_mass_kg=float(true_kg[e]), **{f"feature_{i}": v for i, v in enumerate(features[e].tolist())}))
                        observation_writer.writerow(dict(env_id=e, step=step, **{f"obs_{i}": v for i, v in enumerate(obs["policy"][e].tolist())}))
                    for mode, (masses, counterfactual) in probes.items():
                        delta = counterfactual - native
                        for e in np.flatnonzero(alive):
                            mass = masses[e].item()
                            rms = delta[e].square().mean().sqrt().item()
                            probe_writer.writerow(dict(seed=args.seed, env_id=e, step=step, time_s=step * env.step_dt,
                                mode=mode, true_mass_kg=float(true_kg[e]), predicted_mass_kg=float(prediction[e]),
                                actor_mass_kg=mass, normalized_mass=mass / policy.actor.payload_scale_kg,
                                outside_0_8=int(mass < 0. or mass > 8.), scored=int(step * env.step_dt >= args.warmup),
                                action_delta_l2=delta[e].norm().item(), action_delta_rms=rms,
                                action_delta_max=delta[e].abs().max().item(), target_delta_rms_rad=rms * cfg.actions.joint_pos.scale,
                                **{f"action_{i}": v for i, v in enumerate(counterfactual[e].tolist())},
                                **{f"delta_{i}": v for i, v in enumerate(delta[e].tolist())}))
                action_values = actions.cpu().numpy()
                obs, _, _, _, _ = env.step(actions)
                state = {k: v.cpu().numpy() for k, v in env.payload_diagnostic_state.items()}
                for e in np.flatnonzero(alive):
                    row = dict(seed=args.seed, env_id=e, step=step, time_before_s=step * env.step_dt,
                               time_after_s=(step + 1) * env.step_dt, mode=args.mode, true_mass_kg=float(true_kg[e]),
                               predicted_mass_kg=float(prediction[e]), actor_mass_kg=float(used_mass[e]),
                               mass_error_kg=float(prediction[e] - true_kg[e]), abs_mass_error_kg=float(abs(prediction[e] - true_kg[e])),
                               actor_mass_outside_0_8=int(used_mass[e] < 0 or used_mass[e] > 8),
                               scored=int(step * env.step_dt >= args.warmup),
                               cmd_vx=float(command[e, 0]), cmd_vy=float(command[e, 1]), cmd_wz=float(command[e, 2]),
                               vx=float(state["lin"][e, 0]), vy=float(state["lin"][e, 1]), wz=float(state["ang"][e, 2]),
                               vx_error=float(state["lin"][e, 0] - command[e, 0]), vy_error=float(state["lin"][e, 1] - command[e, 1]),
                               wz_error=float(state["ang"][e, 2] - command[e, 2]), roll=float(state["roll"][e]),
                               pitch=float(state["pitch"][e]), height=float(state["height"][e]),
                               terminated=int(state["terminated"][e]), timeout=int(state["timeout"][e]),
                               **{f"action_{i}": float(v) for i, v in enumerate(action_values[e])})
                    trace.writerow(row)
                    rows[e].append(row)
                    if row["terminated"] or row["timeout"]:
                        alive[e] = False
                if not alive.any():
                    break
            # The mass event may run on failures, but must reapply the same mass/inertia.
            torch.testing.assert_close(robot.root_physx_view.get_masses(), actual, rtol=0, atol=2e-6)
            summary = [dict(seed=args.seed, env_id=e, mode=args.mode, value_kg=args.value_kg, true_mass_kg=args.mass_kg,
                            **tracking_metrics(items)) for e, items in enumerate(rows)]
            writer = open_csv("episode_metrics.csv", list(summary[0]), files)
            writer.writerows(summary)
            for handle in files:
                handle.flush()
            if sha256(args.checkpoint) != load_info["checkpoint_sha256"]:
                raise AssertionError("Checkpoint changed during evaluation")
            manifest = dict(status="complete", args=vars(args), checkpoint=load_info,
                            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                            git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                            torch_version=torch.__version__, numpy_version=np.__version__,
                            step_dt=env.step_dt, physics_dt=env.physics_dt, action_scale=cfg.actions.joint_pos.scale,
                            joint_order=robot.joint_names, base_body_ids=base_ids, actual_mass_kg=true_kg.tolist(),
                            max_probe_forward_error=max_forward_error,
                            initial_state_sha256=sha256(args.output_dir / "initial_state.json"),
                            failed_envs=sum(r["failed"] for r in summary),
                            metrics_window="t_before >= warmup, first episode only; post-physics state before auto-reset",
                            prediction_window="mass prediction and label before action; both all-time and post-warmup MAE")
            write_json(args.output_dir / "manifest.json", manifest)
            print("DIAGNOSTIC_COMPLETE", json.dumps(summary), flush=True)
    finally:
        for handle in files:
            handle.close()
        intervention.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        error = traceback.format_exc()
        write_json(args.output_dir / "error.json", dict(status="failed", traceback=error))
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        app.close()
