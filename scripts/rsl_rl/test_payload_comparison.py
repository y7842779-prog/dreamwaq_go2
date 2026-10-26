"""Small, meaningful checks of metric definitions and saved-model adapters."""
import copy
import math
import unittest
import numpy as np
import torch
from comparison_support import (RUNS, METRICS, initial_samples, load_comparison_policy,
    model_observation, trajectory_metrics, validate_saved_contracts)


class ComparisonChecks(unittest.TestCase):
    def test_metric_formulas_and_window(self):
        rows = []
        for i in range(600):
            roll = .2 if i >= 100 else 1.0
            rows.append(dict(time_before_s=i*.02, time_after_s=(i+1)*.02,
                roll=roll, gx=0., gy=math.sin(roll), gz=-math.cos(roll),
                vx=.3, vy=.4, cmd_vx=.6, cmd_vy=0., terminated=0, timeout=0,
                x_progress=(i+1)*.02*.3))
        result = trajectory_metrics(rows)
        self.assertEqual(result["scored_samples"], 500)
        self.assertAlmostEqual(result[METRICS[0]], .2)
        self.assertAlmostEqual(result[METRICS[1]], math.sin(.2)**2)
        self.assertAlmostEqual(result[METRICS[2]], .5)
        bad = copy.deepcopy(rows)
        for row in bad:
            for key in ("gx", "gy", "gz"):
                row[key] *= 9.81
        self.assertGreater(trajectory_metrics(bad)["invalid_samples"], 0)
        self.assertIsNone(trajectory_metrics(bad)[METRICS[1]])
        rows[74]["terminated"] = 1
        censored = trajectory_metrics(rows[:75])
        self.assertEqual(censored["failed"], 1)
        self.assertEqual(censored["censored"], 1)
        self.assertEqual(censored["scored_samples"], 0)
        self.assertIsNone(censored[METRICS[0]])

    def test_stall_retained(self):
        rows = [dict(time_before_s=i*.02, time_after_s=(i+1)*.02,
            roll=0., gx=0., gy=0., gz=-1., vx=0., vy=0., cmd_vx=.6, cmd_vy=0.,
            terminated=0, timeout=0, x_progress=0.) for i in range(600)]
        metrics = trajectory_metrics(rows)
        self.assertEqual(metrics["stall_flag"], 1)
        self.assertEqual(metrics[METRICS[0]], 0.)
        self.assertAlmostEqual(metrics[METRICS[2]], .6)

    def test_seed_and_term_major_history(self):
        old = {r["evaluation_seed"] for r in initial_samples(20261025, 10)}
        new = {r["evaluation_seed"] for r in initial_samples(20261026, 10)}
        self.assertFalse(old & new)
        obs = torch.arange(2*1170, dtype=torch.float32).reshape(2, 1170)
        expected = [obs[:, :45]]
        offset = 45
        for width in (3, 3, 3, 12, 12, 12):
            end = offset+25*width
            expected.append(obs[:, end-5*width:end])
            offset = end
        torch.testing.assert_close(model_observation(obs, "DreamWaQ_CENet"), torch.cat(expected, 1))
        torch.testing.assert_close(model_observation(obs, "Asym_PPO"), obs[:, :45])

    def test_strict_loading_and_deterministic_actor(self):
        validate_saved_contracts()
        for model in RUNS:
            policy, loaded = load_comparison_policy(model)
            self.assertEqual(loaded["iteration"], 4999)
            self.assertEqual(loaded["missing_keys"], [])
            self.assertEqual(loaded["unexpected_keys"], [])
            with torch.inference_mode():
                obs = torch.randn(2, 1170)*.05
                adapted = model_observation(obs, model)
                a = policy.act_inference(adapted)
                b = policy.act_inference(adapted)
                self.assertEqual(a.shape, (2, 12))
                self.assertTrue(torch.equal(a, b))
                if model == "PayloadEstimator":
                    torch.testing.assert_close(policy.actor.short_observations(obs),
                                               model_observation(obs, "DreamWaQ_CENet"))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
