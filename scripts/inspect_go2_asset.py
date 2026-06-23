# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspect a Go2 asset against the current manager-based locomotion config."""

"""Launch Isaac Sim Simulator first."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Inspect the loaded Go2 asset names and config assumptions.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-v0",
    help="Name of the task to inspect.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import dreamwaq_go2.tasks  # noqa: F401


def print_header(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def print_match_result(kind: str, pattern: str, entity):
    try:
        ids, names = entity.find_bodies(pattern, preserve_order=True)
        print(f"{kind:>12} | pattern={pattern!r} | ids={ids} | names={names}")
    except Exception as exc:
        print(f"{kind:>12} | pattern={pattern!r} | ERROR: {type(exc).__name__}: {exc}")


def print_joint_match_result(pattern: str, robot):
    try:
        ids, names = robot.find_joints(pattern, preserve_order=True)
        print(f"{'joint':>12} | pattern={pattern!r} | ids={ids} | names={names}")
    except Exception as exc:
        print(f"{'joint':>12} | pattern={pattern!r} | ERROR: {type(exc).__name__}: {exc}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    print_header("Config Summary")
    print(f"task: {args_cli.task}")
    print(f"robot usd_path: {env_cfg.scene.robot.spawn.usd_path}")
    print(f"robot prim_path: {env_cfg.scene.robot.prim_path}")
    print(f"height_scanner prim_path: {getattr(env_cfg.scene.height_scanner, 'prim_path', None)}")
    print(f"action scale: {env_cfg.actions.joint_pos.scale}")
    print(f"action use_default_offset: {env_cfg.actions.joint_pos.use_default_offset}")
    print(f"base external force body_names: {env_cfg.events.base_external_force_torque.params['asset_cfg'].body_names}")
    print(f"termination base_contact body_names: {env_cfg.terminations.base_contact.params['sensor_cfg'].body_names}")
    print(f"reward feet_air_time body_names: {env_cfg.rewards.feet_air_time.params['sensor_cfg'].body_names}")

    env = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg)
        env.reset()

        robot = env.unwrapped.scene["robot"]
        contact_sensor = env.unwrapped.scene["contact_forces"]

        print_header("Robot Names")
        print(f"num_bodies: {robot.num_bodies}")
        print(f"body_names: {robot.body_names}")
        print(f"num_joints: {robot.num_joints}")
        print(f"joint_names: {robot.joint_names}")

        print_header("Robot Body Matches")
        for pattern in ["base", ".*_foot", ".*FOOT", ".*_thigh", ".*THIGH", ".*_calf", ".*_hip"]:
            print_match_result("robot_body", pattern, robot)

        print_header("Robot Joint Matches")
        for pattern in [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]:
            print_joint_match_result(pattern, robot)

        print_header("Contact Sensor Bodies")
        print(f"sensor body_names: {contact_sensor.body_names}")
        for pattern in ["base", ".*_foot", ".*FOOT"]:
            print_match_result("sensor_body", pattern, contact_sensor)

        print_header("Quick Read")
        print("If 'base' matches 0 bodies, base contact termination and external force injection are misconfigured.")
        print("If '.*_foot' matches 0 bodies, feet_air_time reward is misconfigured.")
        print("If hip/thigh/calf joint patterns match 0 joints, the action/actuator assumptions do not fit this USD.")

    except Exception as exc:
        print_header("Creation Failed")
        print(f"{type(exc).__name__}: {exc}")
        print(traceback.format_exc())
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
