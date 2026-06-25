# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.algorithms import PPO


class DreamWaQPPO(PPO):
    """PPO variant with an auxiliary velocity estimation loss.

    This keeps the rollout/storage path unchanged and simply supervises the
    CENet velocity head from the privileged critic observations already
    available in each mini-batch. It also supports a simplified reconstruction
    loss that teaches the decoder to reconstruct the current partial
    observation, without requiring next-step rollout storage changes, plus a
    simplified KL regularization on the latent distribution parameters.
    """

    def __init__(
        self,
        *args,
        est_loss_coef: float = 0.1,
        recon_loss_coef: float = 0.1,
        kl_loss_coef: float = 1.0e-4,
        est_target_start_idx: int = 0,
        est_target_dim: int = 3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.est_loss_coef = est_loss_coef
        self.recon_loss_coef = recon_loss_coef
        self.kl_loss_coef = kl_loss_coef
        self.est_target_start_idx = est_target_start_idx
        self.est_target_dim = est_target_dim

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_estimation_loss = 0.0
        mean_reconstruction_loss = 0.0
        mean_kl_loss = 0.0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0.0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0.0
        else:
            mean_symmetry_loss = None

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:
            num_aug = 1
            original_batch_size = obs_batch.shape[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if not hasattr(self.policy, "predict_estimator"):
                raise AttributeError("DreamWaQPPO requires the policy to implement `predict_estimator(obs_batch)`.")

            estimator_outputs = self.policy.predict_estimator(obs_batch)
            current_obs = estimator_outputs["current_obs"]
            velocity_estimate = estimator_outputs["velocity_estimate"]
            latent_mu = estimator_outputs["latent_mu"]
            latent_logvar = estimator_outputs["latent_logvar"]
            reconstructed_obs = estimator_outputs["reconstructed_next_obs"]
            target_start = self.est_target_start_idx
            target_stop = target_start + self.est_target_dim
            velocity_target = critic_obs_batch[:, target_start:target_stop].detach()
            current_obs_target = current_obs.detach()

            if velocity_estimate.shape != velocity_target.shape:
                raise RuntimeError(
                    f"velocity_estimate shape {velocity_estimate.shape} != "
                    f"velocity_target shape {velocity_target.shape}"
                )
            if reconstructed_obs.shape != current_obs_target.shape:
                raise RuntimeError(
                    f"reconstructed_obs shape {reconstructed_obs.shape} != "
                    f"current_obs shape {current_obs_target.shape}"
                )
            if latent_mu.shape != latent_logvar.shape:
                raise RuntimeError(
                    f"latent_mu shape {latent_mu.shape} != latent_logvar shape {latent_logvar.shape}"
                )

            estimation_loss = F.mse_loss(velocity_estimate, velocity_target)
            reconstruction_loss = F.mse_loss(reconstructed_obs, current_obs_target)
            # Clamp log-variance before exp() to avoid rare numerical blow-ups during early training.
            latent_logvar_for_kl = torch.clamp(latent_logvar, min=-10.0, max=10.0)
            kl_loss = -0.5 * torch.sum(
                1.0 + latent_logvar_for_kl - latent_mu.pow(2) - latent_logvar_for_kl.exp(),
                dim=-1,
            ).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.est_loss_coef * estimation_loss
                + self.recon_loss_coef * reconstruction_loss
                + self.kl_loss_coef * kl_loss
            )

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = F.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_estimation_loss += estimation_loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_kl_loss += kl_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_estimation_loss /= num_updates
        mean_reconstruction_loss /= num_updates
        mean_kl_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "estimation": mean_estimation_loss,
            "reconstruction": mean_reconstruction_loss,
            "kl": mean_kl_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
