# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU contract/regression tests; run in env_isaaclab without launching Isaac Sim."""

import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import tempfile
import types
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
AGENTS = ROOT / "source/dreamwaq_go2/dreamwaq_go2/tasks/manager_based/dreamwaq_go2/agents"
# Import tensor-only modules without package-level Isaac Sim task registration.
package = types.ModuleType("_payload_test_agents")
package.__path__ = [str(AGENTS)]
sys.modules[package.__name__] = package
Base = importlib.import_module(package.__name__ + ".dreamwaq_actor_critic").DreamWaQActorCritic
Payload = importlib.import_module(package.__name__ + ".payload_actor_critic").DreamWaQPayloadActorCritic
PPO = importlib.import_module(package.__name__ + ".dreamwaq_ppo").DreamWaQPPO
PayloadPPO = importlib.import_module(package.__name__ + ".payload_ppo").DreamWaQPayloadPPO


def model(payload=True):
    with contextlib.redirect_stdout(io.StringIO()):
        return (Payload(1170, 239, 12) if payload else Base(270, 238, 12))


def rollout_update(policy, algorithm, actor_dim, critic_dim):
    alg = algorithm(policy, num_learning_epochs=2, num_mini_batches=2, device="cpu")
    alg.init_storage("rl", 4, 4, [actor_dim], [critic_dim], [12])
    torch.manual_seed(71)
    with torch.no_grad():
        obs, critic = torch.randn(4, actor_dim), torch.randn(4, critic_dim)
        for step in range(4):
            alg.act(obs, critic)
            next_obs, next_critic = torch.randn_like(obs), torch.randn_like(critic)
            dones = torch.tensor([step == 1, False, False, step == 3])
            alg.process_env_step(torch.randn(4), dones, {"observations": {"policy": next_obs}})
            torch.testing.assert_close(alg.storage.privileged_observations[step], critic)
            obs, critic = next_obs, next_critic
        alg.compute_returns(critic)
    torch.manual_seed(72)
    return alg, alg.update()


class PayloadContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_history_layout_and_initial_baseline_actions(self):
        baseline = model(False)
        torch.manual_seed(42)
        policy = model()
        terms = [torch.randn(3, 25, width) for width in (3, 3, 3, 12, 12, 12)]
        current = torch.randn(3, 45)
        long_obs = torch.cat([current] + [term.flatten(1) for term in terms], 1)
        short_obs = torch.cat([current] + [term[:, -5:].flatten(1) for term in terms], 1)
        torch.testing.assert_close(policy.actor.short_observations(long_obs), short_obs)
        torch.testing.assert_close(policy.act_inference(long_obs), baseline.act_inference(short_obs))
        ctx = policy.predict_estimator(long_obs)
        self.assertEqual(ctx["velocity_estimate"].shape, (3, 3))
        self.assertEqual(ctx["latent_mu"].shape, (3, 16))
        self.assertEqual(ctx["reconstructed_next_obs"].shape, (3, 45))
        self.assertEqual(policy.predict_payload(long_obs).shape, (3, 1))
        altered = [term.clone() for term in terms]
        for term in altered:
            term[:, :20] += 5
        changed = torch.cat([current] + [term.flatten(1) for term in altered], 1)
        torch.testing.assert_close(policy.actor.short_observations(changed), short_obs)
        self.assertFalse(torch.allclose(policy.predict_payload(changed), policy.predict_payload(long_obs)))

    def test_supervision_and_actor_gradient_are_separate(self):
        policy = model()
        obs, critic = torch.randn(4, 1170), torch.randn(4, 239)
        with torch.no_grad():
            policy.actor.base_actor.policy_head[0].weight[:, -1].fill_(0.1)
        policy.act_inference(obs).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in policy.actor.payload_estimator.parameters()))
        self.assertIsNotNone(policy.actor.base_actor.encoder[0].weight.grad)
        policy.zero_grad(set_to_none=True)
        alg = PayloadPPO(policy, device="cpu")
        loss, metrics = alg.additional_auxiliary_loss(obs, critic)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in policy.actor.payload_estimator.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.actor.base_actor.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        self.assertGreater(metrics["payload_mae_kg"], 0)
        other = critic.clone()
        other[:, -1] += 8
        torch.testing.assert_close(policy.evaluate(critic), policy.evaluate(other), rtol=0, atol=0)
        self.assertNotEqual(loss.item(), alg.additional_auxiliary_loss(obs, other)[0].item())
        policy.actor.use_payload_input = False
        before = policy.act_inference(obs).detach()
        with torch.no_grad():
            policy.actor.payload_estimator[-1].bias.add_(50)
        torch.testing.assert_close(policy.act_inference(obs), before, rtol=0, atol=0)

    def test_payload_ppo_update_and_checkpoint(self):
        policy = model()
        initial = policy.actor.payload_estimator[0].weight.detach().clone()
        alg, metrics = rollout_update(policy, PayloadPPO, 1170, 239)
        self.assertEqual(alg.storage.next_observations.shape, (4, 4, 45))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))
        self.assertFalse(torch.equal(initial, policy.actor.payload_estimator[0].weight))
        self.assertIn("payload_mae_kg", metrics)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model": policy.state_dict(), "optimizer": alg.optimizer.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            restored = model()
            self.assertTrue(restored.load_state_dict(saved["model"]))
            restored_alg = PayloadPPO(restored, device="cpu")
            restored_alg.optimizer.load_state_dict(saved["optimizer"])
            obs = torch.randn(2, 1170)
            torch.testing.assert_close(restored.act_inference(obs), policy.act_inference(obs), rtol=0, atol=0)

    def test_baseline_update_matches_pre_feature_commit(self):
        relative = (AGENTS / "dreamwaq_ppo.py").relative_to(ROOT)
        source = subprocess.check_output(["git", "show", f"753b8a4:{relative}"], cwd=ROOT, text=True)
        previous = types.ModuleType("_previous_dreamwaq_ppo")
        exec(compile(source, str(relative), "exec"), previous.__dict__)
        old_policy = model(False)
        torch.manual_seed(42)
        new_policy = model(False)
        old_alg, old_metrics = rollout_update(old_policy, previous.DreamWaQPPO, 270, 238)
        new_alg, new_metrics = rollout_update(new_policy, PPO, 270, 238)
        self.assertEqual(old_metrics, new_metrics)
        self.assertEqual(new_alg.storage.next_observations.shape, old_alg.storage.next_observations.shape)
        for key, value in old_policy.state_dict().items():
            torch.testing.assert_close(value, new_policy.state_dict()[key], rtol=0, atol=0)

    def test_torchscript_and_onnx(self):
        policy = model().eval()
        obs = torch.randn(2, 1170)
        scripted = torch.jit.script(policy.actor)
        torch.testing.assert_close(scripted(obs), policy.act_inference(obs))
        import onnx
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.onnx"
            torch.onnx.export(policy.actor, obs[:1], str(path), opset_version=11, dynamo=False)
            onnx.checker.check_model(onnx.load(str(path)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
