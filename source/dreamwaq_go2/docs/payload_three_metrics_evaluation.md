# Three-metric payload evaluation

Only `/home/hyc/dreamwaq_go2` evaluation code changes. Existing training source,
saved configs, final weights, and October 25 results are protected. The evaluator
and support functions reuse the scene, strict loading, initial sampling and
term-major adapters from `d69725d87cb4cd938318911fa07275e7e6e23134`; these scripts
were absent after the later revert. This version is a focused evaluation CLI,
with no mass-input interventions, training or ablations.

Protocol: flat ground; added base masses 2, 4, 6, 8 kg; body commands
`[0.6, 0, 0]`; 10 matched initial states per condition; 12 s; 2 s warmup.
Evaluation base seed 20261026 creates seeds disjoint from the October 25 set.
The same initial samples are reused at all masses. Every model uses its final
`model_4999.pt`, strict loading, saved architecture, `eval()` and deterministic
`act_inference()` (policy and latent means). One training seed (42) per method.

The common environment produces current 45 values followed by 25-frame histories
in term order: angular velocity, projected gravity, commands, joint positions,
joint velocities, previous actions. Asym receives current 45; CENet receives
current 45 plus the final five entries of **each** history term (270 total);
Ours receives 1170. Runtime shapes and saved config contracts are checked.
No observation scaling/order, action scaling, rewards or training parameters
are changed. PD gains 25/0.5, action scale 0.25, physics 0.005 s, control 0.02 s.
Joint names and actual gains are recorded and paired between models.

Mass uses the original `add` event on body `base`, with its range fixed to
`(payload, payload)`, and its original inertia update. It is NOT total robot
mass. Non-base masses stay nominal. COM randomization and mass/terrain curricula
are disabled. Two explicit resets must preserve mass and observations; actual
PhysX masses are checked every step before and after any automatic reset.
Pushes, external force/torque, joint reset randomization, contact-material
randomization and observation corruption are disabled. Fixed USD materials,
COMs, masses, inertias, PD gains, joints, commands, root states and observations
are recorded and compared across models, with `1e-7` root/observation tolerance
for PhysX normalization. The terrain has a 16 m long flat walking surface.
The recorded robot material in the smoke and formal protocol is fixed at
static/dynamic friction 1.0/1.0 and restitution 0.0, using the USD default after
disabling the material event; this is an explicit evaluation override, not a
change to saved training configs (their material event uses 0.8/0.6).

Ground truth is captured by the post-physics recorder BEFORE automatic reset.
At 50 Hz the scoring intervals start at >=2 s; their post-step samples span
`(2,12] s`, exactly 500 samples. Each trajectory independently yields:

- Roll: `sqrt(mean(roll**2))` in rad, Euler roll wrapped to [-pi,pi].
- Flatness: `mean(gx**2 + gy**2)` with `g = R(q)^T [0,0,-1]`.
  Isaac Lab's `GRAVITY_VEC_W` is normalized at construction. Norm is checked
  during rollout and raw quaternions independently reproduce g during analysis.
- Velocity: `sqrt(mean((vx-0.6)**2 + vy**2))` in m/s, body-frame true velocity.

These are experiment-defined metrics. No equivalence to unspecified paper
metrics is implied. Aggregation is per-trajectory metrics, then mean and sample
SD (`ddof=1`) over 10 trajectories; time samples are not independent trials.
Relative improvement is `100*(baseline_mean-ours_mean)/baseline_mean`.
No metric/sample is changed or excluded because it favors a model.

Early terminations retain the pre-reset terminal state and stop that trial's
recording. Their truncated metrics and runtime are flagged, not silently dropped.
Empty post-warmup windows or invalid gravity/data yield blank metrics and anomaly
records. Motion diagnostics never select primary samples: a stall is mean scored
body vx <0.1 m/s or >=2 consecutive scored seconds below 0.1 m/s. Mean forward
velocity and world-x progress are also reported; no additional success threshold
is used to rank these three metrics.

Run from `/home/hyc/dreamwaq_go2` (choose new output paths for every repetition):

```bash
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python

/home/hyc/isaac-sim/python.sh scripts/rsl_rl/test_payload_comparison.py

# First: small smoke run (12 trajectories). Do not reuse an existing directory.
/home/hyc/isaac-sim/python.sh scripts/rsl_rl/evaluate_payload_comparison.py \
  --headless --device cuda:0 \
  --output-dir logs/payload_three_metrics/2026-10-26/smoke_v4 \
  --masses-kg 2 8 --num-trials 2 --duration 3 --warmup 2

# Formal: 120 entirely new trajectories.
/home/hyc/isaac-sim/python.sh scripts/rsl_rl/evaluate_payload_comparison.py \
  --headless --device cuda:0 \
  --output-dir logs/payload_three_metrics/2026-10-26/formal_v1

MPLCONFIGDIR=/tmp/payload_three_metrics_matplotlib \
/home/hyc/miniconda3/envs/env_unitree/bin/python \
  scripts/rsl_rl/summarize_payload_comparison.py \
  --run-root logs/payload_three_metrics/2026-10-26/formal_v1 \
  --output-dir logs/payload_three_metrics/2026-10-26/summary_v1
```

Require `manifest.json` with `status=complete` and summary `audit.json` passed;
Isaac Sim shutdown can return exit 0 after a Python exception, so process exit
alone is insufficient. The initial failed smoke is retained as `smoke_v1` with
its `error.json`: PyTorch 2.7's **global** deterministic-algorithms switch breaks
the CUDA boolean assignment used by the history circular buffer. The minimal
same-shape CUDA reproduction fails only with that switch enabled. The final
evaluator leaves that switch at its framework default; policy inference still
uses deterministic means. Bitwise repeatability of GPU physics is not claimed.
The second smoke (`smoke_v2`) intentionally failed a PD-gain assertion because
it read PhysX drive gains. Go2 uses the saved explicit DCMotor: its actuator-model
gains are 25/0.5 while PhysX drive gains are correctly zero. The final check reads
the actuator gains and records both sets; no physics parameters were changed.

Output contains per-trial raw CSV (quaternion, unit body gravity, body velocity,
commands, pose, mass and actions), initial physical states, effective env YAML,
overrides, checkpoint SHA256, per-trial metrics, mean/SD summary CSV, signed
improvement CSV, anomaly report, independent quaternion/metric audit and three
figures in PNG/PDF. Raw training weights are never copied into the report.
This comparison establishes performance of whole policies at fixed static
payloads; attributing a gain specifically to the explicit mass channel still
requires trained ablations and multiple training seeds in later experiments.
