# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# 模型长什么样？
# 输入怎么流？
# 输出 action 怎么算？
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class _InputSpec:
    """Lightweight compatibility shim for Isaac Lab's ONNX exporter."""

    def __init__(self, in_features: int):
        self.in_features = in_features


class DreamWaQActor(nn.Module):
    """Actor module with a built-in CENet-style encoder.

    The incoming policy observation is expected to be packed as:
    ``[current_proprio_obs, flattened_history_obs]``.
    """

    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        policy_hidden_dims: list[int],
        activation_name: str,
        history_length: int = 5,
        estimator_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        velocity_dim: int = 3,
        latent_dim: int = 16,
    ):
        super().__init__()

        if estimator_hidden_dims is None:
            estimator_hidden_dims = [128, 64]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [64, 128]
        if history_length <= 0:
            raise ValueError(f"history_length must be positive, got: {history_length}")
        if num_actor_obs % (history_length + 1) != 0:
            raise ValueError(
                "DreamWaQActor expects the policy observation to be packed as "
                "[current_obs, flattened_history_obs] with matching per-step dimensions. "
                f"Received num_actor_obs={num_actor_obs}, history_length={history_length}."
            )

        activation = resolve_nn_activation(activation_name)
        self.num_actor_obs = num_actor_obs
        self.history_length = history_length
        self.velocity_dim = velocity_dim
        self.latent_dim = latent_dim
        self.current_obs_dim = num_actor_obs // (history_length + 1)
        self.history_obs_dim = num_actor_obs - self.current_obs_dim

        if self.history_obs_dim != self.current_obs_dim * history_length:
            raise ValueError(
                "The history slice does not match current_obs_dim * history_length: "
                f"{self.history_obs_dim} != {self.current_obs_dim} * {history_length}."
            )

        self.encoder = self._build_mlp(
            input_dim=self.history_obs_dim,
            hidden_dims=estimator_hidden_dims,
            output_dim=None,
            activation=activation,
        )
        encoder_feature_dim = estimator_hidden_dims[-1]
        self.velocity_head = nn.Linear(encoder_feature_dim, velocity_dim)
        self.latent_mu_head = nn.Linear(encoder_feature_dim, latent_dim)
        self.latent_logvar_head = nn.Linear(encoder_feature_dim, latent_dim)
        self.decoder = self._build_mlp(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=self.current_obs_dim,
            activation=activation,
        )
        self.policy_head = self._build_mlp(
            input_dim=self.current_obs_dim + velocity_dim + latent_dim,
            hidden_dims=policy_hidden_dims,
            output_dim=num_actions,
            activation=activation,
        )

    @staticmethod
    def _build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None,
        activation: nn.Module,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim
        if output_dim is not None:
            layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def __getitem__(self, index: int):
        if index == 0:
            return _InputSpec(self.num_actor_obs)
        raise IndexError(index)

    def split_observations(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_obs = observations[:, : self.current_obs_dim]
        history_obs = observations[:, self.current_obs_dim :]
        return current_obs, history_obs

    def encode_history(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current_obs, history_obs = self.split_observations(observations)
        encoded_history = self.encoder(history_obs)
        return current_obs, encoded_history

    @staticmethod
    def sample_latent(latent_mu: torch.Tensor, latent_logvar: torch.Tensor) -> torch.Tensor:
        # Clamp log-variance before exp() to keep the decoder-side sampling numerically stable.
        latent_logvar = torch.clamp(latent_logvar, min=-10.0, max=10.0)
        latent_std = torch.exp(0.5 * latent_logvar)
        latent_eps = torch.randn_like(latent_std)
        return latent_mu + latent_eps * latent_std

    def estimate_context(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_obs, encoded_history = self.encode_history(observations)
        velocity_estimate = self.velocity_head(encoded_history)
        latent_mu = self.latent_mu_head(encoded_history)
        latent_logvar = self.latent_logvar_head(encoded_history)
        latent_sample = self.sample_latent(latent_mu, latent_logvar)
        reconstructed_next_obs = self.decoder(latent_sample)
        return current_obs, velocity_estimate, latent_mu, latent_logvar, reconstructed_next_obs

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        current_obs, encoded_history = self.encode_history(observations)
        velocity_estimate = self.velocity_head(encoded_history)
        latent_mu = self.latent_mu_head(encoded_history)
        actor_input = torch.cat((current_obs, velocity_estimate, latent_mu), dim=-1)
        return self.policy_head(actor_input)


class DreamWaQActorCritic(nn.Module):
    """DreamWaQ-style actor-critic skeleton for RSL-RL.

    This version wires the history encoder into the actor forward pass so the
    exported policy already contains the CENet-style inference path.
    The auxiliary estimator/VAE losses are intentionally left for a later step.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        history_length: int = 5,
        estimator_hidden_dims: list[int] | None = None,
        decoder_hidden_dims: list[int] | None = None,
        velocity_dim: int = 3,
        latent_dim: int = 16,
        **kwargs,
    ):
        if kwargs:
            print(
                "DreamWaQActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        activation_module = resolve_nn_activation(activation)
        self.actor = DreamWaQActor(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            policy_hidden_dims=actor_hidden_dims,
            activation_name=activation,
            history_length=history_length,
            estimator_hidden_dims=estimator_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
            velocity_dim=velocity_dim,
            latent_dim=latent_dim,
        )
        self.critic = self._build_critic(
            input_dim=num_critic_obs,
            hidden_dims=critic_hidden_dims,
            activation=activation_module,
        )

        print(
            "DreamWaQ actor input split: "
            f"current={self.actor.current_obs_dim}, history={self.actor.history_obs_dim}, "
            f"velocity={velocity_dim}, latent={latent_dim}"
        )
        print(f"DreamWaQ Actor Head: {self.actor.policy_head}")
        print(f"Critic MLP: {self.critic}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    @staticmethod
    def _build_critic(input_dim: int, hidden_dims: list[int], activation: nn.Module) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for layer_index in range(len(hidden_dims)):
            if layer_index == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_index], 1))
            else:
                layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _compute_action_std(self, mean: torch.Tensor) -> torch.Tensor:
        """Return a numerically safe positive standard deviation tensor."""

        if self.noise_std_type == "scalar":
            raw_std = torch.nan_to_num(self.std, nan=1.0, posinf=1.0e3, neginf=-1.0e3)
            std = raw_std.abs().clamp_min(1.0e-6)
        elif self.noise_std_type == "log":
            raw_log_std = torch.nan_to_num(self.log_std, nan=0.0, posinf=5.0, neginf=-20.0)
            std = torch.exp(torch.clamp(raw_log_std, min=-20.0, max=5.0)).clamp_min(1.0e-6)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return std.expand_as(mean)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(observations)
        std = self._compute_action_std(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor):
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs):
        return self.critic(critic_observations)

    def split_actor_observations(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor.split_observations(observations)

    def predict_estimator(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        current_obs, velocity_estimate, latent_mu, latent_logvar, reconstructed_next_obs = self.actor.estimate_context(
            observations
        )
        return {
            "current_obs": current_obs,
            "velocity_estimate": velocity_estimate,
            "latent_mu": latent_mu,
            "latent_logvar": latent_logvar,
            "reconstructed_next_obs": reconstructed_next_obs,
        }

    def load_state_dict(self, state_dict, strict=True):
        remapped_state_dict = dict(state_dict)
        if self.noise_std_type == "log":
            legacy_std = remapped_state_dict.pop("std", None)
            if "log_std" not in remapped_state_dict and legacy_std is not None:
                remapped_state_dict["log_std"] = torch.log(legacy_std.abs().clamp_min(1.0e-6))
        elif self.noise_std_type == "scalar":
            legacy_log_std = remapped_state_dict.pop("log_std", None)
            if "std" not in remapped_state_dict and legacy_log_std is not None:
                remapped_state_dict["std"] = torch.exp(legacy_log_std)
        super().load_state_dict(remapped_state_dict, strict=strict)
        return True
