# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DreamWaQPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for PPO with an auxiliary DreamWaQ velocity loss."""

    class_name: str = "PPO"
    est_loss_coef: float = 0.1
    recon_loss_coef: float = 0.1
    kl_loss_coef: float = 1.0e-4
    est_target_start_idx: int = 0
    est_target_dim: int = 3


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_go2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2DreamWaQPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Runner config for the DreamWaQ-style task.

    In the current Isaac Lab version used by this project, the RSL-RL wrapper
    binds actor observations to the env's ``policy`` group and privileged
    critic observations to the env's ``critic`` group directly.
    """

    experiment_name = "unitree_go2_dreamwaq_cenet"
    policy = RslRlPpoActorCriticCfg(
        class_name="DreamWaQActorCritic",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = DreamWaQPPOAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        est_loss_coef=0.1,
        recon_loss_coef=0.1,
        kl_loss_coef=1.0e-4,
        est_target_start_idx=0,
        est_target_dim=3,
    )


@configclass
class UnitreeGo2RoughAsymPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    """Runner config for the privileged-critic rough Go2 baseline."""

    max_iterations = 5000
    experiment_name = "unitree_go2_rough_asym_payload_curriculum"


@configclass
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
