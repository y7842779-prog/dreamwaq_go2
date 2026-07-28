# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class DreamWaQObservationsCfg:
    """Observation groups used by the DreamWaQ-style environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor-facing partial observations with explicit history packing.

        The term order is important here:
        1. Current proprioceptive observation ``o_t`` (45 dims).
        2. History observation ``o_t^H`` flattened afterwards (5 x 45 dims).

        This keeps the actor proprioception-only while making the CENet input
        available inside the same ``policy`` tensor consumed by RSL-RL.
        """

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        history_base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=5,
            flatten_history_dim=True,
        )
        history_projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
            flatten_history_dim=True,
        )
        history_velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=5,
            flatten_history_dim=True,
        )
        history_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=5,
            flatten_history_dim=True,
        )
        history_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=5,
            flatten_history_dim=True,
        )
        history_actions = ObsTerm(
            func=mdp.last_action,
            history_length=5,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class EstimatorCfg(ObsGroup):
        """History observations for the CENet input."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations available only during training."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        disturbance_force = ObsTerm(
            func=mdp.applied_base_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    estimator: EstimatorCfg = EstimatorCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2DreamWaQEnvCfg(UnitreeGo2RoughEnvCfg):
    """DreamWaQ-style Go2 rough-terrain environment.

    The actor now receives a single ``policy`` tensor packed as
    ``[o_t, o_t^H]`` so a custom DreamWaQ policy can recover the current
    proprioceptive observation and the temporal history from one input.

    The paper's CENet auxiliary loss and adaptive bootstrapping still need
    custom PPO/runner changes.
    """

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        terrain_gen = self.scene.terrain.terrain_generator
        if terrain_gen is not None:
            terrain_gen.curriculum = True
            terrain_gen.num_rows = 10
            terrain_gen.num_cols = 20
            self.scene.terrain.max_init_terrain_level = 2

            stairs = terrain_gen.sub_terrains["pyramid_stairs"]
            stairs_inv = terrain_gen.sub_terrains["pyramid_stairs_inv"]
            stairs.step_height_range = (0.04, 0.12)
            stairs.step_width = 0.30
            stairs.proportion = 0.18
            stairs_inv.step_height_range = (0.04, 0.12)
            stairs_inv.step_width = 0.30
            stairs_inv.proportion = 0.10

            stairs_short = copy.deepcopy(stairs)
            stairs_short.step_width = 0.24
            stairs_short.proportion = 0.14
            terrain_gen.sub_terrains["pyramid_stairs_short_tread"] = stairs_short

            stairs_long = copy.deepcopy(stairs)
            stairs_long.step_width = 0.36
            stairs_long.proportion = 0.08
            terrain_gen.sub_terrains["pyramid_stairs_long_tread"] = stairs_long

            stairs_inv_short = copy.deepcopy(stairs_inv)
            stairs_inv_short.step_width = 0.24
            stairs_inv_short.proportion = 0.07
            terrain_gen.sub_terrains["pyramid_stairs_inv_short_tread"] = stairs_inv_short

            stairs_inv_long = copy.deepcopy(stairs_inv)
            stairs_inv_long.step_width = 0.36
            stairs_inv_long.proportion = 0.03
            terrain_gen.sub_terrains["pyramid_stairs_inv_long_tread"] = stairs_inv_long

            terrain_gen.sub_terrains["boxes"].proportion = 0.14
            terrain_gen.sub_terrains["random_rough"].proportion = 0.16
            terrain_gen.sub_terrains["hf_pyramid_slope"].proportion = 0.05
            terrain_gen.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.05

        self.rewards.joint_pos = RewTerm(
            func=mdp.joint_position_penalty,
            weight=-0.7,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.3,
            },
        )
        self.rewards.foot_clearance = RewTerm(
            func=mdp.foot_clearance_penalty,
            weight=-0.03,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": 0.12,
            },
        )

        # Keep a per-episode disturbance force on the base as privileged critic input.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.base_external_force_torque.params["torque_range"] = (0.0, 0.0)

        self.observations.policy.enable_corruption = True
        self.observations.estimator.enable_corruption = True
        self.observations.critic.enable_corruption = False


@configclass
class UnitreeGo2DreamWaQEnvCfg_PLAY(UnitreeGo2DreamWaQEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.observations.estimator.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
