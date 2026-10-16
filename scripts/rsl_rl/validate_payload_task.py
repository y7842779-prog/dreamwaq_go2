# Copyright (c) 2026, The DreamWaQ Go2 Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check fair configs, reset labels and two PPO iterations in Isaac Lab.

This is a smoke test, not a locomotion performance comparison. Outputs go to an
ignored log directory; default training configurations remain unchanged.
"""

import argparse
import ast
import json
from pathlib import Path
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", default="logs/payload_estimator_validation")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
import dreamwaq_go2.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from rsl_rl.algorithms import PPO as RslRlPPO
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module
from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.dreamwaq_ppo import DreamWaQPPO
from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.payload_actor_critic import DreamWaQPayloadActorCritic
from dreamwaq_go2.tasks.manager_based.dreamwaq_go2.agents.payload_ppo import DreamWaQPayloadPPO

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["Isaac-Velocity-Rough-Unitree-Go2-" + suffix + "-v0"
         for suffix in ("Asym", "DreamWaQ", "DreamWaQ-PayloadEstimator")]
OUT = Path(args.output_dir).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def serial(value):
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(differences(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def entry_adapter(filename):
    # Check actual train/play dispatch without executing their CLI/app launch.
    tree = ast.parse((ROOT / "scripts/rsl_rl" / filename).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_get_runner_cfg")
    namespace = dict(globals(), rsl_rl_on_policy_runner=runner_module,
                     RslRlOnPolicyRunnerCfg=object)
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, "exec"), namespace)
    return namespace["_get_runner_cfg"]


def main():
    env_cfgs = [load_cfg_from_registry(task, "env_cfg_entry_point") for task in TASKS]
    agents = [load_cfg_from_registry(task, "rsl_rl_cfg_entry_point") for task in TASKS]
    env_diffs = [differences(serial(env_cfgs[1].to_dict()), serial(cfg.to_dict())) for cfg in env_cfgs]
    for diff in env_diffs:
        assert all(path.startswith("observations.") for path in diff), diff
    expected_payload_diff = {
        "observations.critic.payload_mass",
        *[f"observations.policy.history_{term}.history_length" for term in
          ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")],
    }
    assert set(env_diffs[2]) == expected_payload_diff, env_diffs[2]
    agent_diffs = [differences(serial(agents[1].to_dict()), serial(cfg.to_dict())) for cfg in agents]
    allowed_agent = {"experiment_name", "log_root_path", "policy.class_name", "policy.history_length",
                     "policy.payload_history_length", "policy.payload_hidden_dims", "policy.payload_scale_kg",
                     "policy.detach_payload", "policy.use_payload_input", "algorithm.class_name",
                     "algorithm.payload_loss_coef"}
    assert set(agent_diffs[2]) <= allowed_agent, agent_diffs[2]
    assert all(agent.max_iterations == 5000 and agent.num_steps_per_env == 24 for agent in agents)
    for filename in ("train.py", "play.py"):
        adapter = entry_adapter(filename)
        for cfg, expected in zip(agents, (RslRlPPO, DreamWaQPPO, DreamWaQPayloadPPO)):
            assert adapter(cfg)["algorithm"]["class_name"] == "PPO"
            assert runner_module.PPO is expected
    play_cfg = load_cfg_from_registry(TASKS[2].replace("-v0", "-Play-v0"), "env_cfg_entry_point")
    assert play_cfg.observations.policy.history_actions.history_length == 25
    assert not play_cfg.observations.policy.enable_corruption
    print("PASS: three environment configs match outside observations; train/play dispatch matches.", flush=True)
    for task, cfg, agent in zip(TASKS, env_cfgs, agents):
        dump_yaml(str(OUT / task / "env_default.yaml"), cfg)
        dump_yaml(str(OUT / task / "agent_default.yaml"), agent)

    cfg, agent = env_cfgs[2], agents[2]
    cfg.scene.num_envs = 16
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 4
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.seed = agent.seed = 42
    cfg.sim.device = agent.device = args.device or "cuda:0"
    dump_yaml(str(OUT / "env_smoke.yaml"), cfg)
    dump_yaml(str(OUT / "agent_smoke.yaml"), agent)
    env = RslRlVecEnvWrapper(gym.make(TASKS[2], cfg=cfg), clip_actions=agent.clip_actions)
    raw = env.unwrapped
    try:
        runner_module.DreamWaQPayloadActorCritic = DreamWaQPayloadActorCritic
        runner = OnPolicyRunner(env, entry_adapter("train.py")(agent), log_dir=str(OUT / "training"), device=agent.device)
        assert isinstance(runner.alg, DreamWaQPayloadPPO)
        policy = runner.alg.policy
        obs, extras = env.get_observations()
        assert obs.shape == (16, 1170)
        assert extras["observations"]["critic"].shape == (16, 239)
        assert policy.critic[0].in_features == 238
        robot = raw.scene["robot"]
        base_ids = robot.find_bodies("base")[0]
        stage_samples = []
        with torch.no_grad():
            for iteration, maximum in ((0, 3.0), (1500, 5.0), (3500, 8.0)):
                raw.common_step_counter = iteration * 24
                obs, extras = env.reset()
                labels = extras["observations"]["critic"][:, -1]
                actual = (robot.root_physx_view.get_masses()[:, base_ids]
                          - robot.data.default_mass[:, base_ids].cpu()).sum(-1).to(raw.device)
                torch.testing.assert_close(labels, actual, rtol=0, atol=1e-6)
                assert labels.min() >= -1e-6 and labels.max() <= maximum + 1e-6
                assert raw.event_manager.get_term_cfg("add_base_mass").params["mass_distribution_params"] == (0., maximum)
                expected_com = cfg.curriculum.payload_dr_levels.params["com_ranges"][len(stage_samples)]
                assert raw.event_manager.get_term_cfg("base_com").params["com_range"] == expected_com
                offset = 45
                for width in (3, 3, 3, 12, 12, 12):
                    history = obs[:, offset:offset + 25 * width].reshape(16, 25, width)
                    torch.testing.assert_close(history, history[:, -1:].expand_as(history))
                    offset += 25 * width
                stage_samples.append({"iteration": iteration, "range": [0, maximum],
                                      "sample_min_kg": labels.min().item(), "sample_max_kg": labels.max().item()})
            raw.common_step_counter = 0
            env.reset()
        initial = policy.actor.payload_estimator[0].weight.detach().clone()
        runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)
        assert not torch.equal(initial, policy.actor.payload_estimator[0].weight)
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        obs, _ = env.get_observations()
        with torch.no_grad():
            before = policy.act_inference(obs).clone()
        checkpoint = OUT / "smoke_checkpoint.pt"
        runner.save(str(checkpoint))
        runner.load(str(checkpoint))
        with torch.no_grad():
            torch.testing.assert_close(policy.act_inference(obs), before, rtol=0, atol=0)
        export_policy_as_jit(policy, runner.obs_normalizer, str(OUT / "exported"))
        export_policy_as_onnx(policy, str(OUT / "exported"), runner.obs_normalizer)
        scripted = torch.jit.load(str(OUT / "exported/policy.pt"))
        torch.testing.assert_close(scripted(obs.cpu()), before.cpu(), rtol=1e-4, atol=1e-5)
        report = {"tasks": TASKS, "environment_differences_from_dreamwaq": env_diffs,
                  "agent_differences_from_dreamwaq": agent_diffs, "stage_samples": stage_samples,
                  "validation": "16 envs, 2 iterations x 24 steps, 5 epochs, 4 minibatches; reset/labels/export/restore passed",
                  "checkpoint": str(checkpoint), "repo": str(ROOT),
                  "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)}
        (OUT / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("PASS: curriculum, physical labels, reset histories, PPO update, checkpoint and exports.", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    app.close()
