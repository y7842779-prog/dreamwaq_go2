# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

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
