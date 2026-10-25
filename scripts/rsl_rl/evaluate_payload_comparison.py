"""Paired inference of existing models on identical courses and initial conditions.

Each case creates a fresh scene. Task 1 runs three models; task 2 runs mass offsets.
No training, exports, checkpoint writes, or training configuration edits occur.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--phase", choices=("comparison", "offset"), default="comparison")
parser.add_argument("--models", nargs="+", default=["Asym_PPO", "DreamWaQ_CENet", "PayloadEstimator"])
parser.add_argument("--terrain", choices=("flat", "slope", "stairs"), default="flat")
parser.add_argument("--mass-kg", type=float, default=4.)
parser.add_argument("--yaw-rate", type=float, default=0.)
parser.add_argument("--num-trials", type=int, default=10)
parser.add_argument("--evaluation-seed", type=int, default=20261025)
parser.add_argument("--duration", type=float, default=12.)
parser.add_argument("--warmup", type=float, default=2.)
parser.add_argument("--goal-x", type=float, default=3.2)
parser.add_argument("--offsets", type=float, nargs="+", default=[-1., -.5, -.25, 0., .25, .5, 1.])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output_dir = args.output_dir.resolve()
if args.output_dir.exists():
    parser.error("Use a new output directory; existing experiment data are never overwritten.")
if not 0 <= args.warmup < args.duration or args.num_trials < 1 or args.mass_kg < 0:
    parser.error("Invalid duration, warmup, trial count or payload.")
args.output_dir.mkdir(parents=True)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaacsim.core.utils.stage import create_new_stage
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi
from isaaclab_tasks.utils import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source/dreamwaq_go2"))
import dreamwaq_go2.tasks  # noqa: F401, E402

from comparison_support import (
    CHECKPOINTS, PhysicalMassOffset, canonical, course_mesh, differences, episode_metrics,
    initial_samples, load_comparison_policy, model_observation, save_csv, saved_env, validate_saved_contracts, validate_initial_pair,
)
from payload_diagnostics import ROOT, sha256, write_json

TASK = "Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0"


def terrain_function(difficulty, cfg):
    mesh, origin = course_mesh(cfg.kind, cfg.size)
    return [mesh], origin


@configclass
class CourseCfg(SubTerrainBaseCfg):
    function = terrain_function
    kind: str = "flat"


def independent_reset(env, env_ids, samples):
    robot = env.scene["robot"]
    ids = env_ids.tolist()
    selected = [samples[i] for i in ids]
    states = robot.data.default_root_state[env_ids].clone()
    states[:, :3] += env.scene.env_origins[env_ids]
    states[:, :2] += torch.tensor([[r["x"], r["y"]] for r in selected], device=env.device)
    rpy = torch.tensor([[r[k] for k in ("roll", "pitch", "yaw")] for r in selected], device=env.device)
    states[:, 3:7] = quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    states[:, 7:9] = torch.tensor([[r["vx"], r["vy"]] for r in selected], device=env.device)
    states[:, 12] = torch.tensor([r["wz"] for r in selected], device=env.device)
    positions = robot.data.default_joint_pos[env_ids].clone()
    positions += torch.tensor([[r[f"joint_offset_{i}"] for i in range(12)] for r in selected], device=env.device)
    velocities = torch.tensor([[r[f"joint_velocity_{i}"] for i in range(12)] for r in selected], device=env.device)
    robot.write_root_state_to_sim(states, env_ids)
    robot.write_joint_state_to_sim(positions, velocities, env_ids=env_ids)


class StateRecorder(RecorderTerm):
    def record_post_step(self):
        robot = self._env.scene["robot"]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
        self._env.comparison_state = dict(lin=robot.data.root_lin_vel_b.clone(), ang=robot.data.root_ang_vel_b.clone(),
            position=robot.data.root_pos_w.clone(), roll=wrap_to_pi(roll).clone(), pitch=wrap_to_pi(pitch).clone(),
            terminated=self._env.termination_manager.terminated.clone(), timeout=self._env.termination_manager.time_outs.clone())
        return None, None


def config_for_run(samples, directory):
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    saved = canonical(saved_env("PayloadEstimator"))
    current = canonical(cfg.to_dict())
    for name in ("robot", "contact_forces", "height_scanner"):
        current["scene"][name]["prim_path"] = current["scene"][name]["prim_path"].replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
    terrain = current["scene"]["terrain"]
    terrain["num_envs"], terrain["env_spacing"] = current["scene"]["num_envs"], current["scene"]["env_spacing"]
    for sub in terrain["terrain_generator"]["sub_terrains"].values():
        sub["size"] = terrain["terrain_generator"]["size"]
        if "slope_threshold" in sub:
            sub["slope_threshold"] = terrain["terrain_generator"]["slope_threshold"]
    drift = differences(saved, current)
    if any(d["path"] != "seed" for d in drift):
        raise ValueError(f"Unexpected saved-config drift: {drift}")
    cfg.scene.num_envs = args.num_trials
    cfg.seed = args.evaluation_seed
    cfg.sim.device = args.device or "cuda:0"
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(seed=args.evaluation_seed, size=(16., 20.),
        num_rows=1, num_cols=args.num_trials, border_width=0., use_cache=False,
        sub_terrains={"course": CourseCfg(kind=args.terrain)})
    cfg.curriculum.terrain_levels = cfg.curriculum.payload_dr_levels = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (args.mass_kg, args.mass_kg)
    for name in ("base_com", "push_robot", "base_external_force_torque", "physics_material", "reset_robot_joints"):
        if hasattr(cfg.events, name):
            setattr(cfg.events, name, None)
    cfg.events.reset_base.func = independent_reset
    cfg.events.reset_base.params = {"samples": samples}
    for group in (cfg.observations.policy, cfg.observations.estimator, cfg.observations.critic):
        group.enable_corruption = False
    command = cfg.commands.base_velocity
    command.resampling_time_range = (1e6, 1e6)
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = command.debug_vis = False
    command.ranges.lin_vel_x, command.ranges.lin_vel_y = (.6, .6), (0., 0.)
    command.ranges.ang_vel_z = (args.yaw_rate, args.yaw_rate)
    cfg.episode_length_s = args.duration + 1.
    cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    cfg.recorders.state = RecorderTermCfg(class_type=StateRecorder)
    dump_yaml(str(directory / "effective_env.yaml"), cfg)
    write_json(directory / "config_overrides.json", differences(saved, canonical(cfg.to_dict())))
    return cfg


def run_case(model, offset, samples, directory, reference):
    directory.mkdir()
    create_new_stage()
    cfg = config_for_run(samples, directory)
    policy, loaded = load_comparison_policy(model, cfg.sim.device)
    env = gym.make(TASK, cfg=cfg).unwrapped
    intervention = PhysicalMassOffset(policy, "offset", offset) if model == "PayloadEstimator" else None
    try:
        with torch.inference_mode():
            obs, _ = env.reset(seed=args.evaluation_seed)
            robot = env.scene["robot"]
            ids = robot.find_bodies("base")[0]
            masses = robot.root_physx_view.get_masses().clone()
            true_kg = (masses[:, ids] - robot.data.default_mass.cpu()[:, ids]).sum(-1)
            torch.testing.assert_close(true_kg, torch.full_like(true_kg, args.mass_kg), rtol=0, atol=2e-6)
            torch.testing.assert_close(obs["critic"][:, -1].cpu(), true_kg, rtol=0, atol=2e-6)
            assert obs["policy"].shape == (args.num_trials, 1170)
            assert math.isclose(env.step_dt, .02)
            initial = dict(root_state=robot.data.root_state_w.cpu().tolist(),
                joint_pos=robot.data.joint_pos.cpu().tolist(), joint_vel=robot.data.joint_vel.cpu().tolist(),
                masses=masses.tolist(), inertias=robot.root_physx_view.get_inertias().tolist(),
                coms=robot.root_physx_view.get_coms().tolist(), materials=robot.root_physx_view.get_material_properties().tolist(),
                joint_stiffness=robot.data.joint_stiffness.cpu().tolist(), joint_damping=robot.data.joint_damping.cpu().tolist(),
                command=env.command_manager.get_command("base_velocity").cpu().tolist(),
                full_policy_observation=obs["policy"].cpu().tolist(), joint_order=robot.joint_names,
                terrain=args.terrain, goal_x=args.goal_x)
            write_json(directory / "initial_state.json", initial)
            initial_hash = sha256(directory / "initial_state.json")
            if reference is None:
                reference = (initial, initial_hash)
            pairing = validate_initial_pair(initial, reference[0])
            write_json(directory / "initial_pairing_checks.json", pairing)
            start_position = robot.data.root_pos_w.cpu().numpy().copy()
            rows = [[] for _ in samples]
            alive = np.ones(args.num_trials, dtype=bool)
            for step in range(round(args.duration / env.step_dt)):
                cmd = env.command_manager.get_command("base_velocity").cpu().numpy().copy()
                actions = policy.act_inference(model_observation(obs["policy"], model))
                if not torch.isfinite(actions).all():
                    raise FloatingPointError("Non-finite actions")
                if intervention:
                    predicted = intervention.native_input[:, -1].cpu().numpy() * intervention.scale
                    actor_mass = intervention.used_input[:, -1].cpu().numpy() * intervention.scale
                else:
                    predicted = actor_mass = np.full(args.num_trials, np.nan)
                action_values = actions.cpu().numpy()
                obs, _, _, _, _ = env.step(actions)
                state = {k: v.cpu().numpy() for k, v in env.comparison_state.items()}
                for e in np.flatnonzero(alive):
                    p = state["position"][e] - start_position[e]
                    row = dict(model=model, offset_kg=offset, terrain=args.terrain, true_mass_kg=float(true_kg[e]),
                        trial=e, evaluation_seed=samples[e]["evaluation_seed"], training_seed=loaded["agent"]["seed"], step=step,
                        time_before_s=step * env.step_dt, time_after_s=(step+1) * env.step_dt,
                        scored=int(step * env.step_dt >= args.warmup), cmd_vx=float(cmd[e, 0]), cmd_vy=float(cmd[e, 1]), cmd_wz=float(cmd[e, 2]),
                        vx=float(state["lin"][e, 0]), vy=float(state["lin"][e, 1]), wz=float(state["ang"][e, 2]),
                        vx_error=float(state["lin"][e, 0]-cmd[e, 0]), vy_error=float(state["lin"][e, 1]-cmd[e, 1]),
                        wz_error=float(state["ang"][e, 2]-cmd[e, 2]), roll=float(state["roll"][e]), pitch=float(state["pitch"][e]),
                        x_progress=float(p[0]), y_progress=float(p[1]), height=float(state["position"][e, 2]),
                        predicted_mass_kg=float(predicted[e]), actor_mass_kg=float(actor_mass[e]),
                        mass_error_kg=float(predicted[e]-true_kg[e]), actor_mass_error_kg=float(actor_mass[e]-true_kg[e]),
                        actor_mass_outside_0_8=int(actor_mass[e]<0 or actor_mass[e]>8) if intervention else 0,
                        terminated=int(state["terminated"][e]), timeout=int(state["timeout"][e]),
                        **{f"action_{i}":float(v) for i,v in enumerate(action_values[e])})
                    rows[e].append(row)
                    if row["terminated"] or row["timeout"]:
                        alive[e] = False
                if not alive.any():
                    break
            save_csv(directory / "trajectory.csv", [row for trial in rows for row in trial])
            metrics = [dict(model=model, offset_kg=offset, terrain=args.terrain, true_mass_kg=args.mass_kg,
                yaw_command=args.yaw_rate, trial=i, evaluation_seed=samples[i]["evaluation_seed"], training_seed=loaded["agent"]["seed"],
                **episode_metrics(values, args.duration, args.goal_x)) for i,values in enumerate(rows)]
            save_csv(directory / "episode_metrics.csv", metrics)
            torch.testing.assert_close(robot.root_physx_view.get_masses(), masses, rtol=0, atol=2e-6)
            assert sha256(CHECKPOINTS[model]) == loaded["checkpoint_sha256"]
            manifest = dict(status="complete", model=model, offset_kg=offset, terrain=args.terrain, mass_kg=args.mass_kg,
                yaw_rate=args.yaw_rate, initial_state_sha256=initial_hash, checkpoint=loaded,
                paired_reference_sha256=reference[1], initial_pairing_verified=True,
                num_trials=args.num_trials, evaluation_seed=args.evaluation_seed, duration=args.duration, warmup=args.warmup,
                goal_x=args.goal_x, physics_dt=env.physics_dt, control_dt=env.step_dt,
                action_scale=cfg.actions.joint_pos.scale, joint_order=robot.joint_names,
                initial_conditions_csv_sha256=sha256(args.output_dir / "initial_conditions.csv"),
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True),
                passed=sum(r["passed"] for r in metrics), failures=sum(r["failed"] for r in metrics),
                torch_version=torch.__version__, numpy_version=np.__version__)
            write_json(directory / "manifest.json", manifest)
            print("CASE_COMPLETE",model,offset,"passed",manifest["passed"],"failed",manifest["failures"],flush=True)
            return reference
    finally:
        if intervention:
            intervention.close()
        # close() unsubscribes Isaac Lab's "keep rendering when stopped" callback.
        # Stopping first would intentionally enter its standalone viewer wait loop.
        simulation = env.sim
        env.close()
        simulation.stop()
        del env
        gc.collect()


def main():
    torch.set_num_threads(1)
    write_json(args.output_dir / "saved_model_contracts.json", validate_saved_contracts())
    samples = initial_samples(args.evaluation_seed, args.num_trials)
    save_csv(args.output_dir / "initial_conditions.csv", samples)
    mesh, _ = course_mesh(args.terrain)
    save_csv(args.output_dir / "terrain_vertices.csv", [dict(index=i,x=v[0],y=v[1],z=v[2]) for i,v in enumerate(mesh.vertices)])
    save_csv(args.output_dir / "terrain_faces.csv", [dict(index=i,a=v[0],b=v[1],c=v[2]) for i,v in enumerate(mesh.faces)])
    cases = [(model,0.) for model in args.models] if args.phase == "comparison" else [("PayloadEstimator",v) for v in args.offsets]
    reference = None
    for model, offset in cases:
        name = model if args.phase == "comparison" else f"offset_{offset:+g}"
        reference = run_case(model, offset, samples, args.output_dir / name, reference)
    write_json(args.output_dir / "manifest.json",dict(status="complete",args=vars(args),cases=cases,
        initial_state_sha256=reference[1],terrain_vertices_sha256=sha256(args.output_dir / "terrain_vertices.csv")))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        error=traceback.format_exc()
        write_json(args.output_dir / "error.json",dict(status="failed",traceback=error))
        print(error,file=sys.stderr,flush=True)
        raise
    finally:
        app.close()
