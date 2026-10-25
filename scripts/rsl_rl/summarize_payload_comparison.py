"""Descriptive, paired summaries for the unified benchmark and kg-offset sweep."""

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

from comparison_support import RUNS, save_csv
from payload_diagnostics import write_json

MEASURES = ("linear_xy_vector_rmse", "wz_error_rmse", "roll_rms", "pitch_rms", "valid_runtime_s",
            "mass_input_mae_kg", "mass_mae_steady_kg", "mass_bias_steady_kg")


def read_csv(path):
    with path.open() as handle:
        return [{key: value if key in ("model", "terrain") else float(value) if value else np.nan
                 for key, value in row.items()} for row in csv.DictReader(handle)]


def finite_mean(values):
    a = np.asarray(values, dtype=float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def summarize(group):
    result = dict(trajectories=len(group), pass_rate=np.mean([r["passed"] for r in group]),
                  failure_rate=np.mean([r["failed"] for r in group]), incomplete_rate=np.mean([r["incomplete"] for r in group]),
                  survival_rate=np.mean([r["survived"] for r in group]))
    for measure in MEASURES:
        a = np.asarray([r[measure] for r in group])
        a = a[np.isfinite(a)]
        result[measure + "_mean"] = float(a.mean()) if len(a) else np.nan
        result[measure + "_sd"] = float(a.std(ddof=1)) if len(a)>1 else np.nan
        result[measure + "_n"] = len(a)
    return result


def paired_summary(pairs):
    result = dict(pairs=len(pairs), complete_pairs=sum(r["both_survived"] for r in pairs),
                  pass_rate_difference=np.mean([r["passed_delta"] for r in pairs]),
                  failure_rate_difference=np.mean([r["failed_delta"] for r in pairs]))
    for metric in MEASURES[:4]:
        # Avoid ranking a short failed rollout against a complete rollout by its RMSE.
        a=np.asarray([r[metric + "_delta"] for r in pairs if r["both_survived"]])
        a=a[np.isfinite(a)]
        result[metric + "_complete_pair_mean_delta"] = float(a.mean()) if len(a) else np.nan
        result[metric + "_complete_pair_win_fraction"] = float(np.mean(a < 0)) if len(a) else np.nan
        if len(a)>1:
            rng=np.random.default_rng(932)
            means=a[rng.integers(0,len(a),size=(4000,len(a)))].mean(axis=1)
            low,high=np.quantile(means,[.025,.975])
        else:
            low=high=np.nan
        result[metric + "_bootstrap_low"] = low
        result[metric + "_bootstrap_high"] = high
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--phase",choices=("comparison","offset"),required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new summary directory.")
    args.output_dir.mkdir(parents=True)
    groups=defaultdict(list)
    manifests={}
    for path in sorted(args.run_root.rglob("manifest.json")):
        manifest=json.loads(path.read_text())
        if "model" not in manifest:
            continue
        if manifest["status"] != "complete":
            raise ValueError(path)
        key=(manifest["terrain"],manifest["mass_kg"],manifest["yaw_rate"],manifest["model"],manifest["offset_kg"])
        if key in manifests:
            raise ValueError(f"Duplicate condition {key}")
        manifests[key]=manifest
        groups[key]=read_csv(path.parent/"episode_metrics.csv")
    if not groups:
        raise ValueError("No completed cases")
    condition_rows=[]
    for (terrain,mass,yaw,model,offset),rows in sorted(groups.items()):
        condition_rows.append(dict(terrain=terrain,true_mass_kg=mass,yaw_command=yaw,model=model,offset_kg=offset,**summarize(rows)))
    save_csv(args.output_dir/"condition_summary.csv",condition_rows)
    save_csv(args.output_dir/"all_episode_metrics.csv",[r for group in groups.values() for r in group])
    pairs,checks=[],[]
    for key,rows in sorted(groups.items()):
        terrain,mass,yaw,model,offset=key
        reference_key=(terrain,mass,yaw,"DreamWaQ_CENet",0.) if args.phase=="comparison" else (terrain,mass,yaw,"PayloadEstimator",0.)
        reference=manifests[reference_key]
        assert reference["initial_state_sha256"]==manifests[key]["initial_state_sha256"],key
        checks.append(dict(terrain=terrain,mass=mass,yaw=yaw,model=model,offset=offset,initial_physics_history_identical=True))
        by_trial={r["trial"]:r for r in groups[reference_key]}
        for row in rows:
            ref=by_trial[row["trial"]]
            assert row["evaluation_seed"]==ref["evaluation_seed"]
            pairs.append(dict(terrain=terrain,true_mass_kg=mass,yaw_command=yaw,model=model,offset_kg=offset,
                trial=row["trial"],evaluation_seed=row["evaluation_seed"],both_survived=int(row["survived"] and ref["survived"]),
                passed_delta=row["passed"]-ref["passed"],failed_delta=row["failed"]-ref["failed"],
                runtime_delta_s=row["valid_runtime_s"]-ref["valid_runtime_s"],
                **{m+"_delta":row[m]-ref[m] for m in MEASURES[:4]}))
    save_csv(args.output_dir/"paired_differences.csv",pairs)
    write_json(args.output_dir/"pairing_checks.json",checks)
    paired_groups=defaultdict(list)
    for r in pairs:
        paired_groups[(r["terrain"],r["true_mass_kg"],r["yaw_command"],r["model"],r["offset_kg"])].append(r)
    save_csv(args.output_dir/"paired_condition_summary.csv",[
        dict(terrain=k[0],true_mass_kg=k[1],yaw_command=k[2],model=k[3],offset_kg=k[4],**paired_summary(v))
        for k,v in sorted(paired_groups.items())])
    if args.phase=="comparison":
        # Plot each yaw-command set separately, preserving every mass/terrain condition.
        for yaw in sorted({key[2] for key in groups}):
            conditions=sorted({key[:2] for key in groups if key[2]==yaw})
            fig,axes=plt.subplots(2,3,figsize=(15,8))
            metrics=("linear_xy_vector_rmse_mean","wz_error_rmse_mean","roll_rms_mean","pitch_rms_mean","pass_rate","valid_runtime_s_mean")
            titles=("Planar velocity RMSE [m/s]","Yaw RMSE [rad/s]","Roll RMS [rad]","Pitch RMS [rad]","Pass rate","Valid runtime [s]")
            for ax,metric,title in zip(axes.flat,metrics,titles):
                for index,model in enumerate(RUNS):
                    selected=[next(r for r in condition_rows if r["terrain"]==t and r["true_mass_kg"]==m and r["yaw_command"]==yaw and r["model"]==model) for t,m in conditions]
                    ax.bar(np.arange(len(conditions))+(index-1)*.25,[r[metric] for r in selected],width=.24,label=model)
                ax.set_xticks(range(len(conditions)),[f"{t}\n{m:g} kg" for t,m in conditions],rotation=35)
                ax.set_title(title)
                ax.grid(axis="y",alpha=.2)
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f"Unified pilot: vx=0.6 m/s, wz={yaw:g} rad/s; 10 initial conditions per bar")
            fig.tight_layout()
            fig.savefig(args.output_dir/f"comparison_wz_{yaw:g}.png",dpi=180)
            fig.savefig(args.output_dir/f"comparison_wz_{yaw:g}.pdf")
            plt.close(fig)
        macro=[]
        for model in RUNS:
            rows=[r for k,g in groups.items() if k[3]==model for r in g]
            macro.append(dict(model=model,**summarize(rows)))
        save_csv(args.output_dir/"macro_summary.csv",macro)
    else:
        for terrain,mass,yaw in sorted({key[:3] for key in groups}):
            selected=sorted([r for r in condition_rows if r["terrain"]==terrain and r["true_mass_kg"]==mass and r["yaw_command"]==yaw],key=lambda r:r["offset_kg"])
            fig,axes=plt.subplots(2,3,figsize=(12,7))
            metrics=MEASURES[:4]+("mass_input_mae_kg","mass_bias_steady_kg")
            for ax,metric in zip(axes.flat,metrics):
                x=[r["offset_kg"] for r in selected]
                # Last panel shows the unmodified estimator bias; input bias adds offset.
                y=[r[metric+"_mean"]+(r["offset_kg"] if metric=="mass_bias_steady_kg" else 0.) for r in selected]
                ax.errorbar(x,y,yerr=[r[metric+"_sd"] for r in selected],marker="o",capsize=3)
                ax.axvline(0,color="grey",ls="--",lw=.8)
                ax.set(title=metric.replace("mass_bias_steady_kg","actor_mass_input_bias_kg"),xlabel="Actor mass offset [kg]")
                ax.grid(alpha=.25)
            fig.suptitle(f"{terrain}, actual payload {mass:g} kg, wz={yaw:g}; mean +/- episode SD")
            fig.tight_layout()
            fig.savefig(args.output_dir/f"offset_{terrain}_{mass:g}kg.png",dpi=180)
            fig.savefig(args.output_dir/f"offset_{terrain}_{mass:g}kg.pdf")
            plt.close(fig)
    write_json(args.output_dir/"summary_manifest.json",dict(phase=args.phase,cases=len(groups),trajectories=sum(map(len,groups.values())),
        checkpoint_sha256={k[3]:v["checkpoint"]["checkpoint_sha256"] for k,v in manifests.items()},
        training_seeds=sorted({r["training_seed"] for g in groups.values() for r in g}),
        evaluation_seeds=sorted({r["evaluation_seed"] for g in groups.values() for r in g}),
        analysis_bootstrap_seed=932,bootstrap_repetitions=4000,
        limitations="10 independent initial-condition trials per condition; one trained checkpoint/seed per method. "
        "Bootstrap intervals describe initial-condition uncertainty, with no multiple-comparison correction. "
        "Failed-rollout RMSE is censored; paired RMSE summaries use both-survived pairs only."))
    print("SUMMARY_COMPLETE",args.phase,len(groups),"cases",sum(map(len,groups.values())),"trajectories")


if __name__=="__main__":
    main()
