"""Summarize completed diagnostic CSVs; never pool time samples as independent trials."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from payload_diagnostics import write_json


def read_csv(path):
    with Path(path).open() as handle:
        return [{k: (v if k == "mode" else float(v) if v else np.nan) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    runs, traces, probes = {}, {}, {}
    for manifest_path in sorted(args.run_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(manifest_path)
        config = manifest["args"]
        label = config["mode"] + (f"_{config['value_kg']:g}" if config["mode"] in ("fixed", "offset") else "")
        key = (config["mass_kg"], config["seed"], label)
        if key in runs:
            raise ValueError(f"Duplicate trial: {key}")
        runs[key] = (manifest, read_csv(manifest_path.parent / "episode_metrics.csv"))
        traces[key] = read_csv(manifest_path.parent / "trajectory.csv")
        if (manifest_path.parent / "action_sensitivity.csv").exists():
            probes[key] = read_csv(manifest_path.parent / "action_sensitivity.csv")
    if not runs:
        raise ValueError("No completed diagnostic runs.")
    checkpoints = {m["checkpoint"]["checkpoint_sha256"] for m, _ in runs.values()}
    if len(checkpoints) != 1:
        raise ValueError("Mixed checkpoints")
    # Require exactly paired physical initial states, including history, materials and inertia.
    pairing = []
    for (mass, seed, label), (manifest, _) in runs.items():
        reference = runs[(mass, seed, "predicted")][0]
        same = manifest["initial_state_sha256"] == reference["initial_state_sha256"]
        pairing.append(dict(mass_kg=mass, seed=seed, mode=label, initial_state_identical=same))
        if not same:
            raise ValueError(f"Unpaired initial states: {mass}, {seed}, {label}")
    write_json(out / "pairing_checks.json", pairing)
    # Accuracy uses only natural predicted-mass closed-loop trajectories.
    accuracy, episodes = [], []
    for mass in sorted({key[0] for key in runs}):
        selected_keys = [k for k in runs if k[0] == mass and k[2] == "predicted"]
        records = [r for k in selected_keys for r in traces[k]]
        for phase in ("all", "warmup", "steady"):
            subset = [r for r in records if phase == "all" or bool(r["scored"]) == (phase == "steady")]
            if not subset:
                continue
            error = np.array([r["mass_error_kg"] for r in subset])
            prediction = np.array([r["predicted_mass_kg"] for r in subset])
            accuracy.append(dict(true_mass_kg=mass, phase=phase, seeds=len(selected_keys), frames=len(subset),
                                 prediction_mean_kg=prediction.mean(), prediction_std_kg=prediction.std(),
                                 mae_kg=np.abs(error).mean(), bias_kg=error.mean(), rmse_kg=np.sqrt(np.mean(error ** 2)),
                                 abs_error_p95_kg=np.quantile(np.abs(error), .95)))
        episodes.extend(dict(true_mass_kg=mass, seed=k[1], env_id=r["env_id"],
                             mae_all_kg=r["mass_mae_all_kg"], mae_steady_kg=r["mass_mae_steady_kg"],
                             bias_steady_kg=r["mass_bias_steady_kg"]) for k in selected_keys for r in runs[k][1])
    save_csv(out / "accuracy_summary.csv", accuracy)
    save_csv(out / "accuracy_per_episode.csv", episodes)
    sensitivity_groups = defaultdict(list)
    for (mass, seed, label), values in probes.items():
        if label == "predicted":
            for row in values:
                if row["scored"]:
                    sensitivity_groups[(mass, row["mode"])].append(row)
    sensitivity = []
    for (mass, mode), records in sorted(sensitivity_groups.items()):
        rms = np.array([r["action_delta_rms"] for r in records])
        sensitivity.append(dict(true_mass_kg=mass, mode=mode, frames=len(records),
                                action_delta_rms_mean=rms.mean(), action_delta_rms_p95=np.quantile(rms, .95),
                                action_delta_max=max(r["action_delta_max"] for r in records),
                                target_delta_rms_mean_rad=np.mean([r["target_delta_rms_rad"] for r in records]),
                                fraction_outside_label_range=np.mean([r["outside_0_8"] for r in records])))
    save_csv(out / "sensitivity_summary.csv", sensitivity)
    measures = ["linear_xy_vector_rmse", "vx_error_rmse", "vy_error_rmse", "wz_error_rmse", "roll_rms", "pitch_rms",
                "mass_mae_steady_kg"]
    closed, paired, seed_means = [], [], []
    modes = sorted({k[2] for k in runs if k[0] == 4.})
    for mode in modes:
        values = [r for k in runs if k[0] == 4. and k[2] == mode for r in runs[k][1]]
        item = dict(mode=mode, episodes=len(values), failures=sum(r["failed"] for r in values),
                    timeouts=sum(r["timeout"] for r in values))
        for measure in measures:
            array = np.array([r[measure] for r in values])
            item[measure + "_mean"] = array.mean()
            item[measure + "_std"] = array.std(ddof=1) if len(array) > 1 else 0.
        closed.append(item)
        for key in [k for k in runs if k[0] == 4. and k[2] == mode]:
            by_env = {r["env_id"]: r for r in runs[(4., key[1], "predicted")][1]}
            values = runs[key][1]
            seed_means.append(dict(mode=mode, seed=key[1], **{m: np.mean([r[m] for r in values]) for m in measures}))
            for row in values:
                reference = by_env[row["env_id"]]
                paired.append(dict(mode=mode, seed=key[1], env_id=row["env_id"],
                                   reference_failed=reference["failed"], intervention_failed=row["failed"],
                                   reference_observed_s=reference["observed_s"], intervention_observed_s=row["observed_s"],
                                   **{m + "_delta": row[m] - reference[m] for m in measures}))
    save_csv(out / "closed_loop_summary.csv", closed)
    save_csv(out / "closed_loop_per_seed.csv", seed_means)
    save_csv(out / "paired_differences.csv", paired)
    # Comparison curves include transient and show between-episode spread, not a confidence interval.
    masses = sorted({k[0] for k in runs if k[2] == "predicted"})
    fig, axes = plt.subplots(len(masses), 1, figsize=(9, 2.8 * len(masses)), squeeze=False)
    for mass, ax in zip(masses, axes[:, 0]):
        time_groups = defaultdict(list)
        for key in [k for k in runs if k[0] == mass and k[2] == "predicted"]:
            for row in traces[key]:
                time_groups[row["time_before_s"]].append(row["predicted_mass_kg"])
        times = sorted(time_groups)
        mean = np.array([np.mean(time_groups[t]) for t in times])
        std = np.array([np.std(time_groups[t]) for t in times])
        ax.plot(times, mean, label="Predicted mean")
        ax.fill_between(times, mean - std, mean + std, alpha=.2, label="Episode SD")
        ax.axhline(mass, color="black", linestyle="--", label="True added mass")
        warmup = next(m[0]["args"]["warmup"] for k, m in runs.items() if k[0] == mass)
        ax.axvspan(0, warmup, color="grey", alpha=.12)
        ax.set(title=f"Actual payload {mass:g} kg", ylabel="Added mass [kg]", xlabel="Time [s]")
        ax.grid(alpha=.25)
        ax.legend(loc="best", ncol=3)
    fig.tight_layout()
    fig.savefig(out / "mass_prediction.png", dpi=200)
    fig.savefig(out / "mass_prediction.pdf")
    plt.close(fig)
    if sensitivity_groups:
        labels = [m for mass, m in sorted(sensitivity_groups) if mass == 4. and m != "predicted"]
        if labels:
            heatmap = np.array([[np.sqrt(np.mean([r[f"delta_{i}"] ** 2 for r in sensitivity_groups[(4., mode)]]))
                                 for i in range(12)] for mode in labels])
            save_csv(out / "per_action_sensitivity.csv", [dict(mode=label, **{f"action_{i}_delta_rms": v for i, v in enumerate(row)})
                                                          for label, row in zip(labels, heatmap)])
            fig, ax = plt.subplots(figsize=(10, 4.8))
            im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
            ax.set(yticks=range(len(labels)), yticklabels=labels, xticks=range(12),
                   xlabel="Action index (saved joint_order in manifest)", title="Same-state action differences, actual mass 4 kg")
            fig.colorbar(im, ax=ax, label="RMS action difference from predicted")
            fig.tight_layout()
            fig.savefig(out / "action_sensitivity.png", dpi=200)
            plt.close(fig)
    if closed:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, metric, title in zip(axes, ("linear_xy_vector_rmse", "wz_error_rmse", "mass_mae_steady_kg"),
                                      ("Planar velocity RMSE [m/s]", "Yaw rate RMSE [rad/s]", "Estimator MAE [kg]")):
            ax.bar(modes, [r[metric + "_mean"] for r in closed], color="steelblue")
            for index, mode in enumerate(modes):
                points = [r[metric] for r in seed_means if r["mode"] == mode]
                ax.scatter([index] * len(points), points, color="black", s=18, zorder=3)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=40)
            ax.grid(axis="y", alpha=.2)
        fig.suptitle("4 kg closed-loop pilot; dots are seed means")
        fig.tight_layout()
        fig.savefig(out / "closed_loop.png", dpi=200)
        plt.close(fig)
    write_json(out / "summary_manifest.json", dict(runs=len(runs), checkpoint_sha256=list(checkpoints)[0],
               run_directories=[str(v[0]["args"]["output_dir"]) for v in runs.values()],
               interpretation="Descriptive pilot only. Time samples and parallel environments are correlated. "
               "No training-seed replication or significance claim. Input label bounds are not measured actor training support."))
    print(f"Summarized {len(runs)} completed runs into {out}")


if __name__ == "__main__":
    main()
