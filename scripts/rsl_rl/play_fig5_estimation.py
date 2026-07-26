# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Play a DreamWaQ checkpoint and record Fig.5-style velocity estimation error.

Fig.5 metric:
    squared estimation error = || v_pred - v_gt ||^2

where:
    v_pred: velocity estimated by CENet / estimator network
    v_gt:   simulator ground-truth base linear velocity in body frame

Save this file as:
    /home/hyc/dreamwaq_go2/scripts/rsl_rl/play_fig5_estimation.py
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# ----------------------------
# CLI arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Play DreamWaQ checkpoint and record Fig.5 estimation error.")

parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")

# Fig.5 test arguments
parser.add_argument("--fig5_test", action="store_true", help="Record Fig.5-style velocity estimation error.")
parser.add_argument("--fig5_steps", type=int, default=1200, help="Number of steps for Fig.5 evaluation.")
parser.add_argument(
    "--fig5_out",
    type=str,
    default=None,
    help="Output directory for Fig.5 logs. Defaults to <checkpoint-dir>/fig5_eval/<checkpoint-stem>/.",
)
parser.add_argument(
    "--fig5_stairs_subterrain",
    type=str,
    default="pyramid_stairs",
    help="Sub-terrain name used for stairs-only Fig.5 test.",
)
parser.add_argument(
    "--fig5_no_force_stairs",
    action="store_true",
    help="Do not force terrain to stairs. Useful for checking whether the dog is visible.",
)
parser.add_argument(
    "--fig5_fixed_command",
    action="store_true",
    help="Use fixed command during Fig.5 test.",
)
parser.add_argument("--fig5_vx", type=float, default=0.6, help="Fixed forward velocity command.")
parser.add_argument("--fig5_vy", type=float, default=0.0, help="Fixed lateral velocity command.")
parser.add_argument("--fig5_wz", type=float, default=0.0, help="Fixed yaw velocity command.")
parser.add_argument(
    "--fig5_hold_after_finish",
    action="store_true",
    help="Keep Isaac Sim viewer open after Fig.5 evaluation finishes.",
)
parser.add_argument(
    "--fig5_raw_obs_for_estimator",
    action="store_true",
    help="Feed raw obs to estimator instead of normalized obs. Only use this if your estimator was trained on raw obs.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ----------------------------
# Imports after app launch
# ----------------------------
import csv
import json
import os
import time

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_rl_on_policy_runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.dreamwaq_actor_critic import DreamWaQActorCritic
from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.dreamwaq_ppo import DreamWaQPPO
import dreamwaq_go2.tasks  # noqa: F401


# Register custom classes into RSL-RL runner module.
rsl_rl_on_policy_runner.DreamWaQActorCritic = DreamWaQActorCritic
rsl_rl_on_policy_runner.DreamWaQPPO = DreamWaQPPO

DEFAULT_FIG5_LOAD_RUN = "2026-06-25_22-56-17"


def _get_runner_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> dict:
    """Adapt custom DreamWaQ config names to the stock RSL-RL runner.

    The stock OnPolicyRunner uses eval(class_name). To make class_name="PPO"
    instantiate DreamWaQPPO, we monkey-patch the PPO symbol in the runner module.
    This is the same style as your working play.py.
    """
    runner_cfg = agent_cfg.to_dict()
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


def _prepare_fig5_viewer(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Make the viewer stay with env_0/robot so the dog remains visible."""
    if not hasattr(env_cfg, "viewer"):
        return

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = None
    env_cfg.viewer.eye = (2.8, 2.8, 1.6)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.3)


def _prepare_fig5_eval_cfg(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Disable play-time noise/randomization that hurts reproducibility."""
    _prepare_fig5_viewer(env_cfg)

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    if hasattr(env_cfg.observations, "estimator"):
        env_cfg.observations.estimator.enable_corruption = False
    if hasattr(env_cfg.observations, "critic"):
        env_cfg.observations.critic.enable_corruption = False

    for event_name in ("push_robot", "base_external_force_torque", "add_base_mass", "base_com"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)


def _force_stairs_env(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Force terrain generator to use only one stairs sub-terrain."""
    terrain = getattr(env_cfg.scene, "terrain", None)
    terrain_generator = getattr(terrain, "terrain_generator", None)

    if terrain_generator is None:
        print("[WARN] No terrain generator found. Skip forcing stairs terrain.")
        return

    # More stable than None when num_rows=num_cols=1
    if hasattr(env_cfg.scene.terrain, "max_init_terrain_level"):
        env_cfg.scene.terrain.max_init_terrain_level = 0

    terrain_generator.curriculum = False
    terrain_generator.num_rows = 1
    terrain_generator.num_cols = 1

    available = list(terrain_generator.sub_terrains.keys())
    print("[INFO] Available sub-terrains:", available)

    target = args_cli.fig5_stairs_subterrain
    found = False

    for name, cfg in terrain_generator.sub_terrains.items():
        if name == target:
            cfg.proportion = 1.0
            found = True
        else:
            cfg.proportion = 0.0

    if found:
        print(f"[INFO] Fig.5 terrain forced to stairs sub-terrain: {target}")
    else:
        print(f"[WARN] Stairs sub-terrain '{target}' not found.")
        print("[WARN] Not changing terrain proportions. Please choose one from available sub-terrains.")


def _set_fixed_command(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Use a fixed command so the robot consistently walks on stairs."""
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "base_velocity"):
        print("[WARN] No base_velocity command config found. Skip fixed command.")
        return

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False

    command_cfg.ranges.lin_vel_x = (args_cli.fig5_vx, args_cli.fig5_vx)
    command_cfg.ranges.lin_vel_y = (args_cli.fig5_vy, args_cli.fig5_vy)
    command_cfg.ranges.ang_vel_z = (args_cli.fig5_wz, args_cli.fig5_wz)

    if hasattr(command_cfg.ranges, "heading"):
        command_cfg.ranges.heading = (0.0, 0.0)

    print(
        "[INFO] Fig.5 fixed command: "
        f"vx={args_cli.fig5_vx}, vy={args_cli.fig5_vy}, wz={args_cli.fig5_wz}"
    )


def _extract_velocity_from_estimator_output(estimator_outputs):
    """Robustly extract velocity estimate from different estimator output formats."""
    if isinstance(estimator_outputs, dict):
        for key in (
            "velocity_estimate",
            "base_lin_vel",
            "lin_vel",
            "v_pred",
            "velocity",
            "estimated_velocity",
        ):
            if key in estimator_outputs:
                return estimator_outputs[key]
        raise RuntimeError(
            "predict_estimator() returned a dict, but no known velocity key was found. "
            f"Available keys: {list(estimator_outputs.keys())}"
        )

    if isinstance(estimator_outputs, (tuple, list)):
        # Usually first output is velocity estimate.
        return estimator_outputs[0]

    if torch.is_tensor(estimator_outputs):
        # Assume first 3 dimensions are velocity.
        return estimator_outputs[:, :3]

    raise RuntimeError(f"Unsupported estimator output type: {type(estimator_outputs)}")


def _predict_velocity_estimate(policy_nn, obs_for_network: torch.Tensor):
    """Call CENet / estimator and return predicted body velocity."""
    if hasattr(policy_nn, "predict_estimator"):
        estimator_outputs = policy_nn.predict_estimator(obs_for_network)
        return _extract_velocity_from_estimator_output(estimator_outputs)

    if hasattr(policy_nn, "estimator"):
        estimator_outputs = policy_nn.estimator(obs_for_network)
        return _extract_velocity_from_estimator_output(estimator_outputs)

    if hasattr(policy_nn, "cenet"):
        estimator_outputs = policy_nn.cenet(obs_for_network)
        return _extract_velocity_from_estimator_output(estimator_outputs)

    raise RuntimeError(
        "This policy has no predict_estimator(), estimator, or cenet. "
        "So it is not a CENet/EstimatorNet model and cannot reproduce Fig.5."
    )


def _save_fig5_outputs(fig5_records: list[dict], resume_path: str):
    """Save CSV, summary JSON, and error curve."""
    resume_path = Path(resume_path).expanduser().resolve()
    if args_cli.fig5_out:
        out_dir = Path(args_cli.fig5_out).expanduser().resolve()
    else:
        out_dir = resume_path.parent / "fig5_eval" / resume_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "fig5_estimation_error.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fig5_records[0].keys())
        writer.writeheader()
        writer.writerows(fig5_records)

    times_ms = np.asarray([r["time_ms"] for r in fig5_records], dtype=np.float64)
    squared_error = np.asarray([r["squared_error"] for r in fig5_records], dtype=np.float64)

    summary = {
        "checkpoint": str(resume_path),
        "num_steps": len(fig5_records),
        "mean_squared_error": float(np.mean(squared_error)),
        "std_squared_error": float(np.std(squared_error)),
        "max_squared_error": float(np.max(squared_error)),
        "min_squared_error": float(np.min(squared_error)),
        "final_squared_error": float(squared_error[-1]),
        "stairs_subterrain": args_cli.fig5_stairs_subterrain,
        "force_stairs": not args_cli.fig5_no_force_stairs,
        "fixed_command": args_cli.fig5_fixed_command,
        "fixed_vx": args_cli.fig5_vx,
        "fixed_vy": args_cli.fig5_vy,
        "fixed_wz": args_cli.fig5_wz,
        "output_dir": str(out_dir),
    }

    summary_path = out_dir / "fig5_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(10, 3.6))
    plt.plot(times_ms, squared_error, linewidth=1.5, label="DreamWaQ / CENet")
    plt.xlabel("Timestep [ms]")
    plt.ylabel("Squared estimation error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = out_dir / "fig5_estimation_error.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved Fig.5 CSV: {csv_path}")
    print(f"[INFO] Saved Fig.5 summary: {summary_path}")
    print(f"[INFO] Saved Fig.5 plot: {fig_path}")
    print("[INFO] Fig.5 summary:")
    print(summary)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent and optionally record Fig.5 estimation error."""

    # task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set seed and device
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Fig.5 environment settings
    if args_cli.fig5_test:
        print("[INFO] Enabled Fig.5 velocity estimation error test.")
        _prepare_fig5_eval_cfg(env_cfg)

        if args_cli.load_run is None and args_cli.checkpoint is None and not args_cli.use_pretrained_checkpoint:
            agent_cfg.load_run = DEFAULT_FIG5_LOAD_RUN
            print(f"[INFO] Fig.5 default load_run: {agent_cfg.load_run}")

        if not args_cli.fig5_no_force_stairs:
            _force_stairs_env(env_cfg)

        if args_cli.fig5_fixed_command:
            _set_fixed_command(env_cfg)

    # logging path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    if args_cli.fig5_test and hasattr(env_cfg, "viewer"):
        print(
            "[INFO] Viewer follow: "
            f"origin_type={env_cfg.viewer.origin_type}, "
            f"asset_name={env_cfg.viewer.asset_name}, "
            f"eye={env_cfg.viewer.eye}, lookat={env_cfg.viewer.lookat}"
        )

    # create Isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent if needed
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # video wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # load runner
    ppo_runner = OnPolicyRunner(env, _get_runner_cfg(agent_cfg), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # inference policy
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # neural network module
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    policy_nn.eval()

    # export policy
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    fig5_records = []

    # get robot
    robot = env.unwrapped.scene["robot"]

    if args_cli.fig5_test:
        print("[INFO] Scene entities:", list(env.unwrapped.scene.keys()))
        print("[INFO] Robot body names:", robot.body_names)
        print("[INFO] Initial robot root position:", robot.data.root_pos_w[0].detach().cpu().tolist())

    # simulate
    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            if args_cli.fig5_test:
                # For estimator, usually use the same normalized observation as policy network.
                obs_for_estimator = obs
                if (not args_cli.fig5_raw_obs_for_estimator) and (ppo_runner.obs_normalizer is not None):
                    obs_for_estimator = ppo_runner.obs_normalizer(obs)

                # predicted body linear velocity from CENet / estimator
                v_pred = _predict_velocity_estimate(policy_nn, obs_for_estimator)

                # simulator ground-truth body-frame base linear velocity
                v_gt = robot.data.root_lin_vel_b

                sq_err_xyz = torch.square(v_pred[:, :3] - v_gt[:, :3])
                sq_err = torch.sum(sq_err_xyz, dim=-1)

                fig5_records.append(
                    {
                        "step": timestep,
                        "time_s": float(timestep * dt),
                        "time_ms": float(timestep * dt * 1000.0),
                        "v_pred_x": float(v_pred[0, 0].cpu()),
                        "v_pred_y": float(v_pred[0, 1].cpu()),
                        "v_pred_z": float(v_pred[0, 2].cpu()),
                        "v_gt_x": float(v_gt[0, 0].cpu()),
                        "v_gt_y": float(v_gt[0, 1].cpu()),
                        "v_gt_z": float(v_gt[0, 2].cpu()),
                        "sq_err_x": float(sq_err_xyz[0, 0].cpu()),
                        "sq_err_y": float(sq_err_xyz[0, 1].cpu()),
                        "sq_err_z": float(sq_err_xyz[0, 2].cpu()),
                        "squared_error": float(sq_err[0].cpu()),
                    }
                )

            # policy action
            actions = policy(obs)

            # environment step
            obs, _, _, _ = env.step(actions)

        timestep += 1

        # video exit
        if args_cli.video and timestep == args_cli.video_length:
            break

        # Fig.5 exit
        if args_cli.fig5_test and timestep >= args_cli.fig5_steps:
            print(f"[INFO] Finished Fig.5 evaluation at step {timestep}.")
            break

        # real-time delay
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # save Fig.5 outputs
    if args_cli.fig5_test and fig5_records:
        _save_fig5_outputs(fig5_records, resume_path)

    # hold viewer after finish
    if args_cli.fig5_test and args_cli.fig5_hold_after_finish:
        print("[INFO] Holding viewer. Close Isaac Sim window to exit.")
        while simulation_app.is_running():
            env.unwrapped.sim.render()

    # close simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
