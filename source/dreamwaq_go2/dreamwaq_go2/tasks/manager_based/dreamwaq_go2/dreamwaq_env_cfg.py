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
from .rough_env_cfg import UnitreeGo2RoughAsymEnvCfg


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
class UnitreeGo2DreamWaQEnvCfg(UnitreeGo2RoughAsymEnvCfg):
    """DreamWaQ-style policy on the asymmetric PPO baseline environment.

    The actor now receives a single ``policy`` tensor packed as
    ``[o_t, o_t^H]`` so a custom DreamWaQ policy can recover the current
    proprioceptive observation and the temporal history from one input.
    """

    observations: DreamWaQObservationsCfg = DreamWaQObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

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
