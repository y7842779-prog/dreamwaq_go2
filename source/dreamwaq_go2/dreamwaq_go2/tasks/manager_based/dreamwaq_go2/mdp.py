# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401,F403


def applied_base_wrench(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Return the currently buffered external wrench applied to the selected body.

    This exposes the disturbance injected through ``set_external_force_and_torque``
    as privileged information for the critic.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    if body_ids is None:
        forces = asset._external_force_b
        torques = asset._external_torque_b
    else:
        forces = asset._external_force_b[:, body_ids]
        torques = asset._external_torque_b[:, body_ids]

    return torch.cat((forces.reshape(env.num_envs, -1), torques.reshape(env.num_envs, -1)), dim=-1)


def applied_base_force(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Return only the buffered external force applied to the selected body."""

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    if body_ids is None:
        forces = asset._external_force_b
    else:
        forces = asset._external_force_b[:, body_ids]

    return forces.reshape(env.num_envs, -1)


def joint_power_l1(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute joint power usage."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_power = asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(joint_power), dim=1)


def joint_position_penalty(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position deviation from the default pose."""

    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    penalty = torch.linalg.norm(asset.data.joint_pos - asset.data.default_joint_pos, dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), penalty, stand_still_scale * penalty)


def foot_clearance_penalty(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
) -> torch.Tensor:
    """Penalize feet whose swing height deviates from the desired clearance."""

    asset: Articulation = env.scene[asset_cfg.name]
    foot_height_error = torch.square(target_height - asset.data.body_pos_w[:, asset_cfg.body_ids, 2])
    foot_planar_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    return torch.sum(foot_height_error * foot_planar_speed, dim=1)


def power_distribution_penalty(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize uneven squared power distribution across joints."""

    asset: Articulation = env.scene[asset_cfg.name]
    joint_power_sq = torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids] * asset.data.joint_vel[:, asset_cfg.joint_ids])
    return torch.var(joint_power_sq, dim=1, unbiased=False)


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize the second-order difference of actions."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        action_dim = env.action_manager.total_action_dim
        self._prev_prev_action = torch.zeros((env.num_envs, action_dim), device=env.device)
        self._action_age = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_prev_action[env_ids] = 0.0
        self._action_age[env_ids] = 0

    def __call__(self, env: ManagerBasedEnv) -> torch.Tensor:
        action = env.action_manager.action
        prev_action = env.action_manager.prev_action
        smoothness = torch.sum(torch.square(action - 2.0 * prev_action + self._prev_prev_action), dim=1)
        smoothness *= (self._action_age >= 2).float()

        self._prev_prev_action[:] = prev_action
        self._action_age += 1
        return smoothness
