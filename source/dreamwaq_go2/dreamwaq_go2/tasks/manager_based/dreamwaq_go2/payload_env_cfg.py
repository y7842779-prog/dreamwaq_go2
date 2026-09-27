# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Payload estimator task on the unchanged DreamWaQ/Asym curriculum environment."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils import configclass

from .dreamwaq_env_cfg import (
    DreamWaQObservationsCfg,
    UnitreeGo2DreamWaQEnvCfg,
    UnitreeGo2DreamWaQEnvCfg_PLAY,
)


class BasePayloadMass(ManagerTermBase):
    """Read actual added base mass after reset; never change the physical parameters.

    The shared curriculum randomizes mass only on reset. Observation-manager reset
    invalidates this cache, and the next observation reads mass after reset events.
    The label travels with the *current* critic observation, before the action.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._dirty = True
        self._payload = None

    def reset(self, env_ids=None):
        self._dirty = True

    def __call__(self, env, asset_cfg: SceneEntityCfg):
        if self._dirty:
            asset = env.scene[asset_cfg.name]
            actual = asset.root_physx_view.get_masses()[:, asset_cfg.body_ids]
            default = asset.data.default_mass[:, asset_cfg.body_ids].to(actual.device)
            self._payload = (actual - default).sum(dim=-1, keepdim=True).to(env.device)
            self._dirty = False
        return self._payload


@configclass
class PayloadObservationsCfg(DreamWaQObservationsCfg):
    @configclass
    class PolicyCfg(DreamWaQObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            # Preserve current terms and the original term-major history layout.
            # The model extracts the most recent five entries of *each* term.
            for name in (
                "history_base_ang_vel", "history_projected_gravity", "history_velocity_commands",
                "history_joint_pos", "history_joint_vel", "history_actions",
            ):
                getattr(self, name).history_length = 25

    @configclass
    class CriticCfg(DreamWaQObservationsCfg.CriticCfg):
        # Transport-only label: the value network explicitly excludes this column.
        payload_mass = ObsTerm(
            func=BasePayloadMass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class UnitreeGo2DreamWaQPayloadEnvCfg(UnitreeGo2DreamWaQEnvCfg):
    observations: PayloadObservationsCfg = PayloadObservationsCfg()


@configclass
class UnitreeGo2DreamWaQPayloadEnvCfg_PLAY(UnitreeGo2DreamWaQEnvCfg_PLAY):
    observations: PayloadObservationsCfg = PayloadObservationsCfg()
