# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Original five-frame CENet plus an independent supervised payload estimator."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from .dreamwaq_actor_critic import DreamWaQActor, DreamWaQActorCritic, _InputSpec


class PayloadActor(nn.Module):
    """Actor input: [current(45), term-major history(25 * 45)].

    Only the most recent five frames reach the original CENet. The payload MLP
    consumes the full history and predicts added mass divided by payload_scale_kg.
    True mass is never an actor input. No recurrent state lives inside the model,
    so shuffled PPO minibatches, inference and export use the same computation.
    """

    def __init__(
        self, base_actor: DreamWaQActor, num_actor_obs: int,
        payload_history_length: int, payload_hidden_dims: list[int],
        payload_scale_kg: float, detach_payload: bool, use_payload_input: bool,
        activation: str,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.num_actor_obs = num_actor_obs
        self.current_obs_dim = base_actor.current_obs_dim
        self.history_obs_dim = base_actor.history_obs_dim
        self.payload_history_length = payload_history_length
        self.payload_scale_kg = payload_scale_kg
        self.detach_payload = detach_payload
        self.use_payload_input = use_payload_input
        expected = self.current_obs_dim * (1 + payload_history_length)
        if self.current_obs_dim != 45 or num_actor_obs != expected:
            raise ValueError(f"PayloadActor requires current(45) + history; expected {expected}, got {num_actor_obs}")
        if payload_history_length < base_actor.history_length or payload_scale_kg <= 0:
            raise ValueError("Payload history must cover CENet history, and mass scale must be positive.")

        indices = []
        offset = self.current_obs_dim
        for width in (3, 3, 3, 12, 12, 12):
            end = offset + payload_history_length * width
            indices.extend(range(end - base_actor.history_length * width, end))
            offset = end
        self.register_buffer("short_history_indices", torch.tensor(indices, dtype=torch.long), persistent=False)

        self.payload_estimator = DreamWaQActor._build_mlp(
            self.current_obs_dim * payload_history_length, payload_hidden_dims, 1,
            resolve_nn_activation(activation),
        )
        # Preserve the original initial action mapping; PPO learns the new column.
        old_first = base_actor.policy_head[0]
        new_first = nn.Linear(old_first.in_features + 1, old_first.out_features)
        with torch.no_grad():
            new_first.weight[:, :-1].copy_(old_first.weight)
            new_first.weight[:, -1].zero_()
            new_first.bias.copy_(old_first.bias)
        base_actor.policy_head[0] = new_first

    def __getitem__(self, index: int):
        if index == 0:
            return _InputSpec(self.num_actor_obs)
        raise IndexError(index)

    def split_observations(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return observations[:, :self.current_obs_dim], observations.index_select(1, self.short_history_indices)

    def short_observations(self, observations: torch.Tensor) -> torch.Tensor:
        current, history = self.split_observations(observations)
        return torch.cat((current, history), dim=-1)

    def estimate_context(self, observations: torch.Tensor):
        return self.base_actor.estimate_context(self.short_observations(observations))

    def predict_payload(self, observations: torch.Tensor) -> torch.Tensor:
        return self.payload_estimator(observations[:, self.current_obs_dim:])

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        current, history = self.split_observations(observations)
        encoded = self.base_actor.encoder(history)
        velocity = self.base_actor.velocity_head(encoded)
        mu = self.base_actor.latent_mu_head(encoded)
        payload = self.predict_payload(observations)
        if self.detach_payload:
            payload = payload.detach()
        if not self.use_payload_input:
            payload = torch.zeros_like(payload)
        return self.base_actor.policy_head(torch.cat((current, velocity, mu, payload), dim=-1))


class DreamWaQPayloadActorCritic(DreamWaQActorCritic):
    """Value input is unchanged; the final privileged column is only a mass label."""

    def __init__(
        self, num_actor_obs: int, num_critic_obs: int, num_actions: int,
        payload_history_length: int = 25, payload_hidden_dims: list[int] | None = None,
        payload_scale_kg: float = 8.0, detach_payload: bool = True,
        use_payload_input: bool = True, **kwargs,
    ):
        if num_critic_obs != 239:
            raise ValueError(f"Expected original critic(238) + mass label(1), got {num_critic_obs}")
        history_length = kwargs.get("history_length", 5)
        super().__init__(45 * (1 + history_length), num_critic_obs - 1, num_actions, **kwargs)
        self.actor = PayloadActor(
            self.actor, num_actor_obs, payload_history_length,
            payload_hidden_dims if payload_hidden_dims is not None else [64, 32],
            payload_scale_kg, detach_payload, use_payload_input, kwargs.get("activation", "elu"),
        )
        print(f"Payload actor: observation={num_actor_obs}, CENet={45 * history_length}, actor head=65")
        print(f"Payload estimator: {self.actor.payload_estimator}")

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(critic_observations[:, :-1])

    def predict_payload(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.predict_payload(observations)
