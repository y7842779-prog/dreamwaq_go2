# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
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
        """Actor-facing proprioceptive observations.

        This group intentionally excludes terrain height scans and simulator
        ground-truth linear velocity so the actor stays proprioception-only.
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

    This file only prepares the environment-side observation split.
    The paper's CENet loss and adaptive bootstrapping still need a custom
    runner/model implementation.
    """

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep a per-episode disturbance force on the base as privileged critic input.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["force_range"] = (-15.0, 15.0)
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
