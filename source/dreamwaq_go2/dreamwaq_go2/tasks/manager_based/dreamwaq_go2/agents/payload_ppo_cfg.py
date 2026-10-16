# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fair-comparison settings: inherit baseline PPO settings and training budget."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_ppo_cfg import DreamWaQPPOAlgorithmCfg, UnitreeGo2DreamWaQPPORunnerCfg


@configclass
class PayloadActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "DreamWaQPayloadActorCritic"
    history_length: int = 5
    payload_history_length: int = 25
    payload_hidden_dims: list[int] = [64, 32]
    payload_scale_kg: float = 8.0
    detach_payload: bool = True
    use_payload_input: bool = True


@configclass
class PayloadPPOAlgorithmCfg(DreamWaQPPOAlgorithmCfg):
    class_name: str = "DreamWaQPayloadPPO"
    payload_loss_coef: float = 0.1


@configclass
class UnitreeGo2DreamWaQPayloadPPORunnerCfg(UnitreeGo2DreamWaQPPORunnerCfg):
    log_root_path: str = "logs/rsl_rl/payload_comparison"
    experiment_name = "PayloadEstimator"

    def __post_init__(self):
        # Copy actual baseline fields, including inherited hyperparameters.
        baseline_policy = self.policy.to_dict()
        baseline_policy.pop("class_name")
        self.policy = PayloadActorCriticCfg(**baseline_policy)
        baseline_algorithm = self.algorithm.to_dict()
        baseline_algorithm.pop("class_name")
        self.algorithm = PayloadPPOAlgorithmCfg(**baseline_algorithm)
