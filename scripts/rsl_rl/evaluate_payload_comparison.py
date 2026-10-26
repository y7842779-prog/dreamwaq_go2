"""Paired, deterministic flat-ground evaluation of three final checkpoints.

Only evaluation configuration is overridden. Every model sees the same sampled
physical initial states; term-major history adapts to its saved training layout.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--masses-kg", type=float, nargs="+", default=[2., 4., 6., 8.])
parser.add_argument("--num-trials", type=int, default=10)
parser.add_argument("--evaluation-seed", type=int, default=20261026)
parser.add_argument("--duration", type=float, default=12.)
parser.add_argument("--warmup", type=float, default=2.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output_dir = args.output_dir.resolve()
if args.output_dir.exists():
    parser.error("Use a new output directory; previous experiments are never overwritten.")
if (not 0 <= args.warmup < args.duration or args.num_trials < 1
        or not all(math.isfinite(m) and 0 <= m <= 8 for m in args.masses_kg)
        or len(set(args.masses_kg)) != len(args.masses_kg)):
    parser.error("Invalid duration, warmup, trial count or payloads.")
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
    ROOT, RUNS, CHECKPOINTS, canonical, course_mesh, differences, initial_samples,
    load_comparison_policy, model_observation, save_csv, saved_env, sha256,
    trajectory_metrics, validate_saved_contracts, validate_initial_pair, write_json,
)

TASK = "Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0"


def terrain_function(difficulty, cfg):
    mesh, origin = course_mesh("flat", cfg.size)
    return [mesh], origin


@configclass
class CourseCfg(SubTerrainBaseCfg):
    function = terrain_function
    kind: str = "flat"


class StateRecorder(RecorderTerm):
    def record_post_step(self):
        # Called after physics, before automatic reset, using simulator truth.
        robot = self._env.scene["robot"]
        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        self._env.comparison_state = dict(
            lin=robot.data.root_lin_vel_b.clone(), ang=robot.data.root_ang_vel_b.clone(),
            position=robot.data.root_pos_w.clone(), quat=robot.data.root_quat_w.clone(),
            gravity=robot.data.projected_gravity_b.clone(),
            roll=wrap_to_pi(roll).clone(), pitch=wrap_to_pi(pitch).clone(), yaw=wrap_to_pi(yaw).clone(),
            masses=robot.root_physx_view.get_masses().clone(),
            terminated=self._env.termination_manager.terminated.clone(),
            timeout=self._env.termination_manager.time_outs.clone())
        return None, None


# The deterministic reset and config_for_run below are reused from the October
# 25 evaluator; the mass is now an explicit function argument.
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


def config_for_run(samples, directory, mass_kg):
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
        sub_terrains={"course": CourseCfg(kind="flat")})
    cfg.curriculum.terrain_levels = cfg.curriculum.payload_dr_levels = None
    cfg.events.add_base_mass.params["mass_distribution_params"] = (mass_kg, mass_kg)
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
    command.ranges.ang_vel_z = (0., 0.)
    cfg.episode_length_s = args.duration + 1.
    cfg.recorders = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    cfg.recorders.state = RecorderTermCfg(class_type=StateRecorder)
    dump_yaml(str(directory / "effective_env.yaml"), cfg)
    write_json(directory / "config_overrides.json", differences(saved, canonical(cfg.to_dict())))
    return cfg




def run_case(model, mass_kg, samples, directory, reference):
    directory.mkdir()
    create_new_stage()
    cfg = config_for_run(samples, directory, mass_kg)
    policy, loaded = load_comparison_policy(model, cfg.sim.device)
    assert loaded["iteration"] == 4999
    assert loaded["missing_keys"] == loaded["unexpected_keys"] == []
    assert loaded["agent"]["clip_actions"] is None
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        with torch.inference_mode():
            obs, _ = env.reset(seed=args.evaluation_seed)
            robot = env.scene["robot"]
            base_ids = robot.find_bodies("base")[0]
            assert len(base_ids) == 1
            masses = robot.root_physx_view.get_masses().clone()
            nominal = robot.data.default_mass.cpu()
            true_kg = (masses[:, base_ids] - nominal[:, base_ids]).sum(-1)
            torch.testing.assert_close(true_kg, torch.full_like(true_kg, mass_kg), rtol=0, atol=2e-6)
            torch.testing.assert_close(obs["critic"][:, -1].cpu(), true_kg, rtol=0, atol=2e-6)
            non_base = [i for i in range(masses.shape[1]) if i not in base_ids]
            torch.testing.assert_close(masses[:, non_base], nominal[:, non_base], rtol=0, atol=0)
            assert obs["policy"].shape == (args.num_trials, 1170)
            terms = ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions"]
            assert env.observation_manager.active_terms["policy"] == terms+["history_"+name for name in terms]
            assert robot.joint_names == [f"{leg}_{joint}_joint" for joint in ("hip", "thigh", "calf")
                                         for leg in ("FL", "FR", "RL", "RR")]
            assert math.isclose(env.step_dt, .02) and math.isclose(env.physics_dt, .005)
            assert cfg.actions.joint_pos.scale == .25
            # Explicit DCMotor computes PD torques in the actuator model; PhysX
            # drive gains are intentionally zero and must not be read as PD gains.
            actuator = robot.actuators["base_legs"]
            torch.testing.assert_close(actuator.stiffness, torch.full_like(actuator.stiffness, 25.))
            torch.testing.assert_close(actuator.damping, torch.full_like(actuator.damping, .5))
            initial = dict(root_state=robot.data.root_state_w.cpu().tolist(),
                joint_pos=robot.data.joint_pos.cpu().tolist(), joint_vel=robot.data.joint_vel.cpu().tolist(),
                masses=masses.tolist(), inertias=robot.root_physx_view.get_inertias().tolist(),
                coms=robot.root_physx_view.get_coms().tolist(), materials=robot.root_physx_view.get_material_properties().tolist(),
                joint_stiffness=actuator.stiffness.cpu().tolist(), joint_damping=actuator.damping.cpu().tolist(),
                physx_drive_stiffness=robot.data.joint_stiffness.cpu().tolist(),
                physx_drive_damping=robot.data.joint_damping.cpu().tolist(),
                command=env.command_manager.get_command("base_velocity").cpu().tolist(),
                full_policy_observation=obs["policy"].cpu().tolist(), joint_order=robot.joint_names)
            write_json(directory / "initial_state.json", initial)
            if reference is None:
                reference = initial
            write_json(directory / "initial_pairing_checks.json", validate_initial_pair(initial, reference))
            # Confirm reset re-applies the SAME nominal-plus-payload mass, without
            # accumulating it or allowing a training curriculum to override it.
            obs, _ = env.reset(seed=args.evaluation_seed)
            torch.testing.assert_close(robot.root_physx_view.get_masses(), masses, rtol=0, atol=0)
            torch.testing.assert_close(obs["policy"], torch.tensor(initial["full_policy_observation"], device=env.device), rtol=0, atol=1e-7)
            start = robot.data.root_pos_w.cpu().numpy().copy()
            rows = [[] for _ in samples]
            alive = np.ones(args.num_trials, dtype=bool)
            expected_width = {"Asym_PPO": 45, "DreamWaQ_CENet": 270, "PayloadEstimator": 1170}[model]
            for step in range(round(args.duration / env.step_dt)):
                command = env.command_manager.get_command("base_velocity")
                torch.testing.assert_close(command, torch.tensor([.6, 0., 0.], device=env.device).expand_as(command))
                policy_obs = model_observation(obs["policy"], model)
                assert policy_obs.shape == (args.num_trials, expected_width)
                actions = policy.act_inference(policy_obs)
                if not torch.isfinite(actions).all():
                    raise FloatingPointError("Non-finite actions")
                action_values = actions.cpu().numpy()
                cmd = command.cpu().numpy().copy()
                obs, _, _, _, _ = env.step(actions)
                torch.testing.assert_close(env.comparison_state["masses"], masses, rtol=0, atol=0)
                torch.testing.assert_close(robot.root_physx_view.get_masses(), masses, rtol=0, atol=0)
                state = {k: v.cpu().numpy() for k, v in env.comparison_state.items() if k != "masses"}
                np.testing.assert_allclose(np.linalg.norm(state["gravity"], axis=1), 1., rtol=0, atol=1e-5)
                for e in np.flatnonzero(alive):
                    row = dict(model=model, payload_kg=mass_kg, trial=int(e), evaluation_seed=samples[e]["evaluation_seed"],
                        step=step, time_before_s=step*env.step_dt, time_after_s=(step+1)*env.step_dt,
                        scored=int(step*env.step_dt >= args.warmup-1e-9),
                        actual_added_base_mass_kg=float(true_kg[e]), actual_base_mass_kg=float(masses[e, base_ids[0]]),
                        actual_robot_total_mass_kg=float(masses[e].sum()),
                        cmd_vx=float(cmd[e, 0]), cmd_vy=float(cmd[e, 1]), cmd_wz=float(cmd[e, 2]),
                        vx=float(state["lin"][e, 0]), vy=float(state["lin"][e, 1]), wz=float(state["ang"][e, 2]),
                        roll=float(state["roll"][e]), pitch=float(state["pitch"][e]), yaw=float(state["yaw"][e]),
                        **{k: float(v) for k, v in zip(("gx", "gy", "gz"), state["gravity"][e])},
                        **{k: float(v) for k, v in zip(("qw", "qx", "qy", "qz"), state["quat"][e])},
                        x_progress=float(state["position"][e, 0]-start[e, 0]),
                        y_progress=float(state["position"][e, 1]-start[e, 1]), height=float(state["position"][e, 2]),
                        terminated=int(state["terminated"][e]), timeout=int(state["timeout"][e]),
                        **{f"action_{i}":float(v) for i, v in enumerate(action_values[e])})
                    rows[e].append(row)
                    if row["terminated"] or row["timeout"]:
                        alive[e] = False
                if not alive.any():
                    break
            metrics = []
            for i, values in enumerate(rows):
                save_csv(directory / f"trial_{i:02d}.csv", values)
                metrics.append(dict(model=model, payload_kg=mass_kg, trial=i,
                    evaluation_seed=samples[i]["evaluation_seed"], training_seed=loaded["agent"]["seed"],
                    **trajectory_metrics(values, args.duration, args.warmup, env.step_dt)))
            save_csv(directory / "episode_metrics.csv", metrics)
            assert sha256(CHECKPOINTS[model]) == loaded["checkpoint_sha256"]
            write_json(directory / "manifest.json", dict(status="complete", model=model, payload_kg=mass_kg,
                checkpoint=loaded, initial_pairing_verified=True, initial_state_sha256=sha256(directory/"initial_state.json"),
                num_trials=args.num_trials, evaluation_seed=args.evaluation_seed, duration=args.duration, warmup=args.warmup,
                physics_dt=env.physics_dt, control_dt=env.step_dt, action_scale=cfg.actions.joint_pos.scale,
                joint_order=robot.joint_names, base_body_ids=base_ids, nominal_base_mass_kg=nominal[:, base_ids].tolist(),
                true_added_mass_kg=true_kg.tolist(), deterministic_inference=True,
                mass_checked_every_step_before_and_after_reset=True, repeated_reset_mass_check=True,
                completed=sum(r["complete"] for r in metrics), failed=sum(r["failed"] for r in metrics),
                stalls=sum(r["stall_flag"] for r in metrics), torch_version=torch.__version__, numpy_version=np.__version__))
            print("CASE_COMPLETE", model, mass_kg, "completed", sum(r["complete"] for r in metrics),
                  "mean_vx", float(np.mean([r["mean_forward_velocity_m_s"] for r in metrics])), flush=True)
            return reference
    finally:
        simulation = env.sim
        env.close()
        simulation.stop()
        del env
        gc.collect()


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # act_inference() uses the policy mean and latent mean without sampling.
    # Do not enable PyTorch's global deterministic-algorithms switch: in this
    # bundled Torch 2.7 CUDA build it breaks CircularBuffer's boolean assignment.
    write_json(args.output_dir / "saved_model_contracts.json", validate_saved_contracts())
    samples = initial_samples(args.evaluation_seed, args.num_trials)
    previous = initial_samples(20261025, 10)
    assert not ({r["evaluation_seed"] for r in samples} & {r["evaluation_seed"] for r in previous})
    save_csv(args.output_dir / "initial_conditions.csv", samples)
    write_json(args.output_dir / "protocol.json", dict(args=vars(args), models=list(RUNS), terrain="flat",
        command=[.6, 0., 0.], deterministic_policy=True, observation_corruption=False,
        stall_definition="mean scored body vx < 0.1 m/s OR >=2 consecutive scored seconds with vx <0.1; diagnostic only",
        formulas=dict(roll="sqrt(mean(roll**2)) [rad]", flatness="mean(gx**2+gy**2), norm(g)=1",
                      velocity="sqrt(mean((vx-.6)**2+vy**2)) [m/s]"),
        sampling="post-step simulator truth on intervals beginning >=2 s; formal (2,12] s, 500 samples",
        aggregation=f"trajectory metrics, then mean and sample SD (ddof=1) across {args.num_trials} trajectories",
        interpretation="Experiment-defined metrics; no claimed numeric equivalence to unspecified paper metrics.",
        prior_seed=20261025, old_initial_states_disjoint=True,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)))
    for mass_kg in args.masses_kg:
        directory = args.output_dir / f"mass_{mass_kg:g}kg"
        directory.mkdir()
        reference = None
        for model in RUNS:
            reference = run_case(model, mass_kg, samples, directory/model, reference)
    write_json(args.output_dir / "manifest.json", dict(status="complete", cases=3*len(args.masses_kg),
        trajectories=3*len(args.masses_kg)*args.num_trials, args=vars(args)))


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
