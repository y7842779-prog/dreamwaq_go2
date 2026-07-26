#!/usr/bin/env python3

"""Evaluate DreamWaQ velocity estimation error for a single training run."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-Play-v0"
DEFAULT_RUN_DIR = (
    Path(__file__).resolve().parent / "logs" / "rsl_rl" / "unitree_go2_dreamwaq_cenet" / "2026-06-25_22-56-17"
)


parser = argparse.ArgumentParser(description="Figure 5-style estimation error evaluation for one DreamWaQ run.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Task name used to instantiate the environment.")
parser.add_argument(
    "--run-dir",
    type=Path,
    default=DEFAULT_RUN_DIR,
    help="Run directory containing model checkpoints plus params/agent.pkl and params/env.pkl.",
)
parser.add_argument(
    "--checkpoint",
    type=Path,
    default=None,
    help="Checkpoint to evaluate. Defaults to the latest model_*.pt under --run-dir.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--steps", type=int, default=1200, help="Maximum number of control steps to simulate.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the evaluation environment.")
parser.add_argument(
    "--stairs-subterrain",
    type=str,
    default="pyramid_stairs",
    help="Stair sub-terrain name to keep active during evaluation.",
)
parser.add_argument("--fixed-lin-vel-x", type=float, default=0.6, help="Fixed forward velocity command in m/s.")
parser.add_argument("--fixed-lin-vel-y", type=float, default=0.0, help="Fixed lateral velocity command in m/s.")
parser.add_argument("--fixed-ang-vel-z", type=float, default=0.0, help="Fixed yaw-rate command in rad/s.")
parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Directory for CSV/PNG/JSON outputs. Defaults to <run-dir>/fig5_eval/<checkpoint-stem>/.",
)
parser.add_argument(
    "--use-saved-env",
    action="store_true",
    help="Reuse params/env.pkl from the run directory instead of the current registered Play env config.",
)
parser.add_argument(
    "--keep-going-after-done",
    action="store_true",
    help="Continue stepping after the first reset instead of stopping at the first done.",
)
parser.add_argument(
    "--hold-after-finish",
    action="store_true",
    help="Keep the viewer open after evaluation finishes so the final scene stays visible.",
)
parser.add_argument(
    "--disable-fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# clear out sys.argv for Hydra-driven imports
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import matplotlib

matplotlib.use("Agg")

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.dreamwaq_actor_critic import DreamWaQActorCritic
from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.dreamwaq_ppo import DreamWaQPPO
import dreamwaq_go2.tasks  # noqa: F401

rsl_rl_on_policy_runner.DreamWaQActorCritic = DreamWaQActorCritic
rsl_rl_on_policy_runner.DreamWaQPPO = DreamWaQPPO


def _resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        checkpoint_path = checkpoint.expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
        return checkpoint_path

    model_paths = sorted(
        run_dir.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not model_paths:
        raise FileNotFoundError(f"No model_*.pt checkpoint found under: {run_dir}")
    return model_paths[-1]


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def _load_cfgs(run_dir: Path, task_name: str, device: str, num_envs: int):
    params_dir = run_dir / "params"
    env_pkl = params_dir / "env.pkl"
    agent_pkl = params_dir / "agent.pkl"

    env_cfg = None
    if args_cli.use_saved_env and env_pkl.is_file():
        try:
            env_cfg = _load_pickle(env_pkl)
            print(f"[INFO] Loaded saved env cfg from: {env_pkl}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load saved env cfg from {env_pkl}: {exc}")
            print("[WARN] Falling back to the current task registry Play env config.")

    if env_cfg is None:
        env_cfg = parse_env_cfg(task_name, device=device, num_envs=num_envs, use_fabric=not args_cli.disable_fabric)
        print("[INFO] Loaded env cfg from the current task registry.")

    if agent_pkl.is_file():
        try:
            agent_cfg = _load_pickle(agent_pkl)
            print(f"[INFO] Loaded saved agent cfg from: {agent_pkl}")
            return env_cfg, agent_cfg
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to load saved agent cfg from {agent_pkl}: {exc}")
            print("[WARN] Falling back to the current task registry agent config.")

    agent_cfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    print("[INFO] Loaded agent cfg from the current task registry.")
    return env_cfg, agent_cfg


def _prepare_env_cfg(env_cfg, stairs_subterrain: str):
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    env_cfg.scene.num_envs = args_cli.num_envs

    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.origin_type = "asset_body"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.body_name = "base"
        env_cfg.viewer.eye = (2.8, 2.8, 1.6)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.3)

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "estimator"):
        env_cfg.observations.estimator.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False

    for event_name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    env_cfg.scene.terrain.max_init_terrain_level = None
    terrain_generator = getattr(env_cfg.scene.terrain, "terrain_generator", None)
    if terrain_generator is None:
        raise RuntimeError("The evaluation environment has no terrain generator, so stairs-only testing is impossible.")

    terrain_generator.curriculum = False
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    found_stairs = False
    for name, cfg in terrain_generator.sub_terrains.items():
        use_this = name == stairs_subterrain
        cfg.proportion = 1.0 if use_this else 0.0
        found_stairs = found_stairs or use_this
    if not found_stairs:
        available = ", ".join(sorted(terrain_generator.sub_terrains.keys()))
        raise ValueError(f"Unknown stairs sub-terrain '{stairs_subterrain}'. Available: {available}")

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False
    command_cfg.ranges.lin_vel_x = (args_cli.fixed_lin_vel_x, args_cli.fixed_lin_vel_x)
    command_cfg.ranges.lin_vel_y = (args_cli.fixed_lin_vel_y, args_cli.fixed_lin_vel_y)
    command_cfg.ranges.ang_vel_z = (args_cli.fixed_ang_vel_z, args_cli.fixed_ang_vel_z)
    if hasattr(command_cfg.ranges, "heading"):
        command_cfg.ranges.heading = (0.0, 0.0)


def _runner_cfg_to_dict(agent_cfg) -> dict:
    runner_cfg = agent_cfg.to_dict() if hasattr(agent_cfg, "to_dict") else dict(agent_cfg)
    alg_cfg = runner_cfg["algorithm"]
    policy_cfg = runner_cfg["policy"]
    alg_class_name = alg_cfg["class_name"]

    # Older saved DreamWaQ runs were serialized with `algorithm.class_name: PPO`
    # while still keeping DreamWaQ-only kwargs like `est_loss_coef`.
    uses_dreamwaq_ppo = alg_class_name == "DreamWaQPPO" or (
        policy_cfg.get("class_name") == "DreamWaQActorCritic"
        and any(
            key in alg_cfg
            for key in (
                "est_loss_coef",
                "recon_loss_coef",
                "kl_loss_coef",
                "est_target_start_idx",
                "est_target_dim",
            )
        )
    )

    if uses_dreamwaq_ppo:
        rsl_rl_on_policy_runner.PPO = DreamWaQPPO
        alg_cfg["class_name"] = "PPO"
    else:
        rsl_rl_on_policy_runner.PPO = RslRlPPO
    return runner_cfg


def _get_actor_critic(ppo_runner: OnPolicyRunner):
    try:
        return ppo_runner.alg.policy
    except AttributeError:
        return ppo_runner.alg.actor_critic


def _build_output_dir(run_dir: Path, checkpoint_path: Path) -> Path:
    if args_cli.output_dir is not None:
        return args_cli.output_dir.expanduser().resolve()
    return run_dir / "fig5_eval" / checkpoint_path.stem


def _save_records(output_dir: Path, records: list[dict[str, float | int]]):
    csv_path = output_dir / "estimation_trace.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return csv_path


def _save_plot(output_dir: Path, times_ms: np.ndarray, squared_error: np.ndarray):
    plot_path = output_dir / "squared_estimation_error.png"
    plt.figure(figsize=(10, 3.6))
    plt.plot(times_ms, squared_error, color="#d84b3a", linewidth=1.5, label="DreamWaQ / CENet")
    plt.xlabel("Timestep [ms]")
    plt.ylabel("Squared est. error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path


def _save_summary(output_dir: Path, checkpoint_path: Path, num_steps: int, squared_error: np.ndarray, stopped_on_done: bool):
    summary_path = output_dir / "summary.json"
    summary = {
        "task": args_cli.task,
        "run_dir": str(args_cli.run_dir.resolve()),
        "checkpoint": str(checkpoint_path),
        "stairs_subterrain": args_cli.stairs_subterrain,
        "fixed_command": {
            "lin_vel_x": args_cli.fixed_lin_vel_x,
            "lin_vel_y": args_cli.fixed_lin_vel_y,
            "ang_vel_z": args_cli.fixed_ang_vel_z,
        },
        "num_envs": args_cli.num_envs,
        "num_steps_recorded": num_steps,
        "stopped_on_done": stopped_on_done,
        "squared_error_mean": float(np.mean(squared_error)),
        "squared_error_std": float(np.std(squared_error)),
        "squared_error_max": float(np.max(squared_error)),
        "squared_error_min": float(np.min(squared_error)),
        "squared_error_final": float(squared_error[-1]),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main():
    run_dir = args_cli.run_dir.expanduser().resolve()
    checkpoint_path = _resolve_checkpoint(run_dir, args_cli.checkpoint)
    output_dir = _build_output_dir(run_dir, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg, agent_cfg = _load_cfgs(
        run_dir=run_dir,
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    _prepare_env_cfg(env_cfg, stairs_subterrain=args_cli.stairs_subterrain)

    print(f"[INFO] Evaluating checkpoint: {checkpoint_path}")
    print(f"[INFO] Output directory: {output_dir}")
    if hasattr(env_cfg, "viewer"):
        print(
            "[INFO] Viewer follow: "
            f"origin_type={env_cfg.viewer.origin_type}, "
            f"asset_name={env_cfg.viewer.asset_name}, "
            f"body_name={env_cfg.viewer.body_name}, "
            f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}"
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    ppo_runner = OnPolicyRunner(env, _runner_cfg_to_dict(agent_cfg), log_dir=None, device=args_cli.device)
    ppo_runner.load(str(checkpoint_path))

    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    actor_critic = _get_actor_critic(ppo_runner)
    actor_critic.eval()

    obs, _ = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    print(f"[INFO] Scene entities: {list(env.unwrapped.scene.keys())}")
    print(f"[INFO] Robot bodies: {robot.body_names}")
    print(f"[INFO] Initial robot root position (world): {robot.data.root_pos_w[0].detach().cpu().tolist()}")
    records: list[dict[str, float | int]] = []
    stopped_on_done = False

    try:
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                print("[INFO] Simulation app stopped early.")
                break

            with torch.inference_mode():
                estimator_outputs = actor_critic.predict_estimator(obs)
                v_pred = estimator_outputs["velocity_estimate"]
                v_gt = robot.data.root_lin_vel_b
                command = env.unwrapped.command_manager.get_command("base_velocity")
                squared_error_xyz = torch.square(v_pred[:, :3] - v_gt[:, :3])
                squared_error = torch.sum(squared_error_xyz, dim=-1)

                records.append(
                    {
                        "step": step,
                        "time_s": float(step * env.unwrapped.step_dt),
                        "time_ms": float(step * env.unwrapped.step_dt * 1000.0),
                        "command_x": float(command[0, 0].cpu()),
                        "command_y": float(command[0, 1].cpu()),
                        "command_yaw": float(command[0, 2].cpu()),
                        "v_pred_x": float(v_pred[0, 0].cpu()),
                        "v_pred_y": float(v_pred[0, 1].cpu()),
                        "v_pred_z": float(v_pred[0, 2].cpu()),
                        "v_gt_x": float(v_gt[0, 0].cpu()),
                        "v_gt_y": float(v_gt[0, 1].cpu()),
                        "v_gt_z": float(v_gt[0, 2].cpu()),
                        "sq_err_x": float(squared_error_xyz[0, 0].cpu()),
                        "sq_err_y": float(squared_error_xyz[0, 1].cpu()),
                        "sq_err_z": float(squared_error_xyz[0, 2].cpu()),
                        "squared_error": float(squared_error[0].cpu()),
                    }
                )

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)

            if bool(dones[0].item()) and not args_cli.keep_going_after_done:
                stopped_on_done = True
                print(f"[INFO] The first environment reached done at step {step}.")
                break

        if args_cli.hold_after_finish:
            print("[INFO] Evaluation finished. Keeping the viewer open. Close the Isaac Sim window to exit.")
            while simulation_app.is_running():
                env.unwrapped.sim.render()
    finally:
        env.close()

    if not records:
        raise RuntimeError("No evaluation records were collected.")

    times_ms = np.asarray([row["time_ms"] for row in records], dtype=np.float64)
    squared_error = np.asarray([row["squared_error"] for row in records], dtype=np.float64)

    csv_path = _save_records(output_dir, records)
    plot_path = _save_plot(output_dir, times_ms, squared_error)
    summary_path = _save_summary(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        num_steps=len(records),
        squared_error=squared_error,
        stopped_on_done=stopped_on_done,
    )

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved plot: {plot_path}")
    print(f"[INFO] Saved summary: {summary_path}")
    print(
        "[INFO] Squared error stats: "
        f"mean={np.mean(squared_error):.6f}, std={np.std(squared_error):.6f}, max={np.max(squared_error):.6f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
