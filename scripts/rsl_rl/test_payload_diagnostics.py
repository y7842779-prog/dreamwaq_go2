"""CPU checks of the actual checkpoint and intervention, without launching Isaac Sim."""

import unittest

import torch

from payload_diagnostics import ActorMassInput, DEFAULT_CHECKPOINT, load_policy, probe_same_state, tracking_metrics


class DiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy, cls.info = load_policy(DEFAULT_CHECKPOINT)

    def test_strict_trained_weights_and_noop(self):
        self.assertEqual(self.info["iteration"], 4999)
        self.assertEqual(self.info["missing_keys"], [])
        self.assertEqual(self.info["unexpected_keys"], [])
        torch.manual_seed(17)
        obs = torch.randn(3, 1170)
        with torch.inference_mode():
            expected = self.policy.act_inference(obs)
            intervention = ActorMassInput(self.policy)
            try:
                actual = self.policy.act_inference(obs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                probes = probe_same_state(self.policy, intervention.native_input, torch.tensor([0., 4., 8.]))
                torch.testing.assert_close(probes["predicted"][1], actual, rtol=0, atol=0)
            finally:
                intervention.close()

    def test_only_mass_changes_and_scale_and_no_clamp(self):
        torch.manual_seed(21)
        obs = torch.randn(3, 1170)
        untouched = obs.clone()
        saved = {k: v.clone() for k, v in self.policy.state_dict().items()}
        with torch.inference_mode():
            capture = ActorMassInput(self.policy)
            self.policy.act_inference(obs)
            original = capture.native_input
            capture.close()
            for mode, value in (("oracle", 0.), ("fixed", 4.), ("offset", 2.)):
                intervention = ActorMassInput(self.policy, mode, value)
                intervention.true_kg = torch.tensor([0., 4., 8.])
                try:
                    actual = self.policy.act_inference(obs)
                    torch.testing.assert_close(intervention.native_input, original, rtol=0, atol=0)
                    torch.testing.assert_close(intervention.used_input[:, :-1], original[:, :-1], rtol=0, atol=0)
                    expected_mass = {"oracle": torch.tensor([0., .5, 1.]), "fixed": torch.full((3,), .5),
                                     "offset": original[:, -1] + .25}[mode]
                    torch.testing.assert_close(intervention.used_input[:, -1], expected_mass, rtol=0, atol=0)
                    # Probe bypasses an active oracle/fixed/offset hook.
                    probe = probe_same_state(self.policy, original, intervention.true_kg)
                    equivalent = {"oracle": "oracle", "fixed": "fixed_4", "offset": "offset_+2"}[mode]
                    torch.testing.assert_close(actual, probe[equivalent][1], rtol=0, atol=0)
                finally:
                    intervention.close()
            for key, value in self.policy.state_dict().items():
                torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
            torch.testing.assert_close(obs, untouched, rtol=0, atol=0)

    def test_metric_definition_and_failure_accounting(self):
        rows = [dict(scored=0, terminated=0, timeout=0, time_after_s=.02, mass_error_kg=3.),
                dict(scored=1, terminated=1, timeout=0, time_after_s=.04, mass_error_kg=-1.,
                     vx_error=3., vy_error=4., wz_error=2., roll=-.1, pitch=.2)]
        metric = tracking_metrics(rows)
        self.assertEqual(metric["linear_xy_vector_rmse"], 5.)
        self.assertEqual(metric["mass_mae_all_kg"], 2.)
        self.assertEqual(metric["mass_mae_steady_kg"], 1.)
        self.assertEqual(metric["failed"], 1)
        self.assertEqual(metric["timeout"], 0)
        self.assertEqual(metric["scored_samples"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
