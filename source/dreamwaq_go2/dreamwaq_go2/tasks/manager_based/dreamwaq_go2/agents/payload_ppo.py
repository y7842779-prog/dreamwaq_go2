# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO auxiliary supervision for the independent payload estimator."""

import torch.nn.functional as F

from .dreamwaq_ppo import DreamWaQPPO


class DreamWaQPayloadPPO(DreamWaQPPO):
    def __init__(self, *args, payload_loss_coef: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if payload_loss_coef < 0:
            raise ValueError("payload_loss_coef must be nonnegative")
        if self.symmetry:
            raise ValueError("Payload task requires a dedicated history/label symmetry mapping; symmetry is disabled.")
        self.payload_loss_coef = payload_loss_coef
        # Decoder still predicts only next current(45), not the long history.
        self.next_obs_dim = self.policy.actor.current_obs_dim

    def additional_auxiliary_loss(self, obs_batch, critic_obs_batch):
        prediction = self.policy.predict_payload(obs_batch)
        scale = self.policy.actor.payload_scale_kg
        target = critic_obs_batch[:, -1:].detach() / scale
        loss = F.huber_loss(prediction, target)
        return self.payload_loss_coef * loss, {
            "payload": loss.detach().item(),
            "payload_mae_kg": ((prediction.detach() - target).abs().mean() * scale).item(),
        }
