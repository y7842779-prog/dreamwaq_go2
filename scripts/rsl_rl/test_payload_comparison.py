"""Actual checkpoint and observation/physics-contract checks without Isaac Sim."""

import unittest

import numpy as np
import torch

from comparison_support import (
    PhysicalMassOffset, RUNS, course_mesh, initial_samples, load_comparison_policy,
    model_observation, validate_saved_contracts,
)


class ComparisonContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policies = {name: load_comparison_policy(name)[0] for name in RUNS}

    def test_saved_environment_and_real_checkpoint_forward(self):
        validate_saved_contracts()
        torch.manual_seed(19)
        obs = torch.randn(2, 1170)
        with torch.inference_mode():
            for name, policy in self.policies.items():
                output = policy.act_inference(model_observation(obs, name))
                self.assertEqual(output.shape, (2, 12))
                self.assertTrue(torch.isfinite(output).all())

    def test_term_major_adapter_matches_original_payload_short_history(self):
        obs = torch.arange(2*1170).reshape(2, 1170).float()
        actor = self.policies["PayloadEstimator"].actor
        torch.testing.assert_close(model_observation(obs, "DreamWaQ_CENet"), actor.short_observations(obs), rtol=0, atol=0)
        torch.testing.assert_close(model_observation(obs, "Asym_PPO"), obs[:, :45], rtol=0, atol=0)
        self.assertIs(model_observation(obs, "PayloadEstimator"), obs)

    def test_independent_reproducible_initial_seeds_and_courses(self):
        a, b = initial_samples(20261025, 10), initial_samples(20261025, 10)
        self.assertEqual(a, b)
        self.assertEqual(len({r["evaluation_seed"] for r in a}), 10)
        self.assertEqual(len({r["yaw"] for r in a}), 10)
        self.assertTrue(all(r["evaluation_seed"] != 42 for r in a))
        for terrain in ("flat", "slope", "stairs"):
            mesh, origin = course_mesh(terrain)
            self.assertEqual(tuple(origin), (2., 10., 0.))
            self.assertTrue(np.isfinite(mesh.vertices).all())
            self.assertTrue((mesh.area_faces > 0).all())
        stairs, _ = course_mesh("stairs")
        self.assertAlmostEqual(stairs.vertices[:, 2].max(), .6)
        slope, _ = course_mesh("slope")
        self.assertAlmostEqual(slope.vertices[:, 2].max(), 2.4*np.tan(np.deg2rad(10.)))

    def test_physical_offset_only_changes_last_feature_and_preserves_noop(self):
        policy = self.policies["PayloadEstimator"]
        torch.manual_seed(20)
        obs = torch.randn(2, 1170)
        with torch.inference_mode():
            original = policy.act_inference(obs)
            for delta in (-1., -.5, -.25, 0., .25, .5, 1.):
                hook = PhysicalMassOffset(policy, "offset", delta)
                try:
                    actual = policy.act_inference(obs)
                    torch.testing.assert_close(hook.used_input[:, :-1], hook.native_input[:, :-1], rtol=0, atol=0)
                    torch.testing.assert_close(hook.used_input[:, -1]*8, hook.native_input[:, -1]*8+delta, rtol=0, atol=0)
                    if delta == 0:
                        torch.testing.assert_close(actual, original, rtol=0, atol=0)
                finally:
                    hook.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
