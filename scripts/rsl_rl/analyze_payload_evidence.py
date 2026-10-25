"""Offline, exploratory evidence audit of the completed 540+210 payload trials.

No simulator, policy loading, fitting, or parameter updates. All outputs require a
new directory. Initial-condition uncertainty is distinct from training variance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("Asym_PPO", "DreamWaQ_CENet", "PayloadEstimator")
BASELINES = MODELS[:2]
OUR = MODELS[2]
KEY = ("terrain", "true_mass_kg", "yaw_command")
ERRORS = ("linear_xy_vector_rmse", "wz_error_rmse", "roll_rms", "pitch_rms")
EXTRA = ("passed", "final_x_m", "max_x_m", "path_xy_m", "body_forward_distance_m",
         "mean_vx", "mean_speed_xy", "mean_wz", "slow_fraction_0p1", "slow_fraction_0p3",
         "moving_fraction_0p45", "vx_error_rmse", "vy_error_rmse", "mass_mae_steady_kg")
HIGHER = ("passed", "final_x_m", "max_x_m", "path_xy_m", "body_forward_distance_m")
RAW = ("step", "time_before_s", "scored", "evaluation_seed", "vx", "vy", "wz",
       "vx_error", "vy_error", "wz_error", "roll", "pitch", "x_progress", "y_progress",
       "predicted_mass_kg", "mass_error_kg", "actor_mass_error_kg", "actor_mass_outside_0_8")


def read_rows(path):
    with path.open() as handle:
        return [{k: (v if k in ("model", "terrain") else float(v) if v else np.nan)
                 for k, v in r.items()} for r in csv.DictReader(handle)]


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(values):
    a = np.asarray(values, dtype=float)
    return a[np.isfinite(a)]


def avg(values):
    a = finite(values)
    return float(a.mean()) if len(a) else np.nan


def condition(row):
    return tuple(row[k] for k in KEY)


def measurements(a, mask):
    if not mask.any():
        return {k: np.nan for k in ERRORS}
    return dict(linear_xy_vector_rmse=float(np.sqrt(np.mean(a["vx_error"][mask]**2+a["vy_error"][mask]**2))),
                wz_error_rmse=float(np.sqrt(np.mean(a["wz_error"][mask]**2))),
                roll_rms=float(np.sqrt(np.mean(a["roll"][mask]**2))),
                pitch_rms=float(np.sqrt(np.mean(a["pitch"][mask]**2))))


def summary(rows, metrics=ERRORS + EXTRA):
    result = {"n": len(rows)}
    for metric in metrics:
        a = finite([r[metric] for r in rows])
        result[metric+"_mean"] = float(a.mean()) if len(a) else np.nan
        result[metric+"_sd"] = float(a.std(ddof=1)) if len(a)>1 else np.nan
    return result


def difference(ours, baseline, lower=True):
    d = np.asarray(ours)-np.asarray(baseline)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0, mean_delta=np.nan, ci_low=np.nan, ci_high=np.nan, wins=0)
    # Same resampled initial-condition indices as the previous descriptive audit.
    rng = np.random.default_rng(932)
    lo, hi = np.quantile(d[rng.integers(0, len(d), (4000, len(d)))].mean(axis=1), [.025, .975])
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(d))))
    p = float(np.mean(np.abs((signs*d).mean(axis=1)) >= abs(d.mean())-1e-14))
    return dict(n=len(d), mean_delta=float(d.mean()), median_delta=float(np.median(d)),
                ci_low=float(lo), ci_high=float(hi), wins=int(np.sum(d<0 if lower else d>0)),
                ties=int(np.sum(d==0)), sign_flip_two_sided_p=p)


def correlations(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[valid], y[valid])[0, 1]) if valid.sum()>2 and min(np.std(x[valid]), np.std(y[valid]))>1e-12 else np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite an existing output directory")
    root, out = args.input_root.resolve(), args.output_dir.resolve()
    if root == out or root in out.parents:
        parser.error("Output must be outside the completed experiment directory")
    if not root.is_dir():
        parser.error("Input directory does not exist")
    args.output_dir.mkdir(parents=True)
    protected = sorted(p for p in root.rglob("*") if p.is_file())
    protected += sorted(Path("source/dreamwaq_go2/docs").glob("payload_*2026-10-25.md"))
    protected += sorted(Path("logs/rsl_rl/payload_comparison").glob("*/*/model_4999.pt"))
    before = {str(p.resolve()): digest(p) for p in protected}
    (out/"input_sha256_before.json").write_text(json.dumps(before, indent=2))
    episodes, phases, offset_episodes, audit, cases = [], [], [], [], {}
    for phase in ("task1_v2", "task2"):
        for path in sorted((root/phase).rglob("episode_metrics.csv")):
            manifest = json.loads((path.parent/"manifest.json").read_text())
            assert manifest["status"] == "complete" and manifest["initial_pairing_verified"]
            existing = read_rows(path)
            samples = read_rows(path.parent.parent/"initial_conditions.csv")
            raw_rows = read_rows(path.parent/"trajectory.csv")
            assert len(existing)==10 and len(raw_rows)==6000
            assert len({r["evaluation_seed"] for r in existing})==10
            arrays = {}
            for r in existing:
                trial = int(r["trial"])
                raw = [x for x in raw_rows if int(x["trial"])==trial]
                a = {k: np.array([x[k] for x in raw]) for k in RAW}
                assert np.array_equal(a["step"], np.arange(600))
                assert np.all(a["evaluation_seed"]==r["evaluation_seed"])
                assert r["training_seed"]==42 and r["survived"]==1
                for k in RAW[:14]:
                    assert np.isfinite(a[k]).all(), (path, k)
                scored = a["scored"]==1
                recalc = measurements(a, scored)
                max_error = max(abs(recalc[k]-r[k]) for k in ERRORS)
                assert max_error<1e-10, (path, max_error)
                x, y = a["x_progress"], a["y_progress"]
                path_length = float(np.hypot(np.diff(np.r_[0., x]), np.diff(np.r_[0., y])).sum())
                r.update(final_x_m=float(x[-1]), max_x_m=float(x.max()), path_xy_m=path_length,
                         body_forward_distance_m=float(a["vx"].sum()*.02),
                         mean_vx=float(a["vx"][scored].mean()),
                         mean_speed_xy=float(np.hypot(a["vx"][scored], a["vy"][scored]).mean()),
                         mean_wz=float(a["wz"][scored].mean()),
                         slow_fraction_0p1=float(np.mean(a["vx"][scored]<.1)),
                         slow_fraction_0p3=float(np.mean(a["vx"][scored]<.3)),
                         moving_fraction_0p45=float(np.mean(a["vx"][scored]>.45)))
                r["source_csv"] = str(path.parent/"trajectory.csv")
                audit.append(dict(source=str(path), trial=trial, max_metric_error=max_error))
                arrays[trial] = a
                if phase=="task2":
                    offset_episodes.append(r)
                    continue
                episodes.append(r)
                identity = {k:r[k] for k in KEY+("model", "trial", "evaluation_seed")}
                for start in range(0, 12, 2):
                    mask = (a["time_before_s"]>=start) & (a["time_before_s"]<start+2)
                    phases.append(dict(**identity, phase_type="clock", stage=f"{start}-{start+2}s",
                        samples=int(mask.sum()), duration_s=float(mask.sum()*.02),
                        mean_vx=float(a["vx"][mask].mean()),
                        mse_xy_contribution=float(np.sum(a["vx_error"][mask]**2+a["vy_error"][mask]**2)/500) if start else np.nan,
                        mass_mae=avg(np.abs(a["mass_error_kg"][mask])), **measurements(a, mask)))
                local_x = x+samples[trial]["x"]
                for label, spatial in (("approach_base", local_x<.8), ("obstacle_base", (local_x>=.8)&(local_x<3.2)), ("post_obstacle_base", local_x>=3.2)):
                    mask = scored & spatial
                    phases.append(dict(**identity, phase_type="position", stage=label,
                        samples=int(mask.sum()), duration_s=float(mask.sum()*.02),
                        mean_vx=avg(a["vx"][mask]),
                        mass_mae=avg(np.abs(a["mass_error_kg"][mask])), **measurements(a, mask)))
            if phase=="task1_v2":
                key = (manifest["terrain"], manifest["mass_kg"], manifest["yaw_rate"])
                cases[key+(manifest["model"],)] = (existing, arrays, manifest)
    assert len(episodes)==540 and len(offset_episodes)==210 and len(cases)==54
    write_rows(out/"trajectory_metrics.csv", episodes)
    write_rows(out/"offset_trajectory_metrics.csv", offset_episodes)
    write_rows(out/"phase_metrics.csv", phases)
    write_rows(out/"raw_metric_validation.csv", audit)
    conditions = sorted(set(condition(r) for r in episodes))
    grouped = defaultdict(list)
    for r in episodes:
        grouped[condition(r)+(r["model"],)].append(r)
    condition_rows = [dict(zip(KEY,k[:3]), model=k[3], **summary(v)) for k,v in sorted(grouped.items())]
    write_rows(out/"condition_metrics.csv", condition_rows)
    paired, paired_rows, moving, masked_episodes = [], [], [], []
    for c in conditions:
        ref = cases[c+(OUR,)][2]
        for baseline in BASELINES:
            bm = cases[c+(baseline,)][2]
            assert ref["paired_reference_sha256"]==bm["paired_reference_sha256"]
            assert ref["initial_conditions_csv_sha256"]==bm["initial_conditions_csv_sha256"]
            ours = sorted(grouped[c+(OUR,)], key=lambda r:r["trial"])
            base = sorted(grouped[c+(baseline,)], key=lambda r:r["trial"])
            for a,b in zip(ours,base):
                assert a["evaluation_seed"]==b["evaluation_seed"]
                paired_rows.append(dict(zip(KEY,c), baseline=baseline, trial=a["trial"], evaluation_seed=a["evaluation_seed"],
                    **{m+"_delta":a[m]-b[m] for m in ERRORS+EXTRA}))
            for metric in ERRORS+EXTRA:
                paired.append(dict(zip(KEY,c), baseline=baseline, metric=metric,
                    ours_mean=avg([r[metric] for r in ours]), baseline_mean=avg([r[metric] for r in base]),
                    **difference([r[metric] for r in ours],[r[metric] for r in base], metric not in HIGHER)))
        # Same clock times with all three models moving. This is a descriptive
        # post-treatment selection, not a causal speed-matched estimator.
        for threshold in (.2, .3, .45):
            for trial in range(10):
                aa = [cases[c+(m,)][1][trial] for m in MODELS]
                mask = np.logical_and.reduce([(a["scored"]==1)&(a["vx"]>threshold) for a in aa])
                for model,a in zip(MODELS,aa):
                    masked_episodes.append(dict(zip(KEY,c), model=model, trial=trial, threshold=threshold,
                        retained_fraction=float(mask.sum()/500), n_samples=int(mask.sum()),
                        **measurements(a, mask)))
    # Correct exploratory endpoint families: 18 conditions x two baselines.
    for metric in ERRORS:
        family = [r for r in paired if r["metric"]==metric]
        ordered = sorted(family, key=lambda r:r["sign_flip_two_sided_p"])
        running = 0.
        for index,r in enumerate(ordered):
            running = max(running, (len(ordered)-index)*r["sign_flip_two_sided_p"])
            r["holm_p_36_comparisons"] = min(1., running)
    write_rows(out/"paired_condition_effects.csv", paired)
    write_rows(out/"paired_trajectory_differences.csv", paired_rows)
    write_rows(out/"common_moving_trajectory_metrics.csv", masked_episodes)
    for c in conditions:
        for threshold in (.2,.3,.45):
            selection = [r for r in masked_episodes if condition(r)==c and r["threshold"]==threshold]
            for baseline in BASELINES:
                ours = sorted([r for r in selection if r["model"]==OUR],key=lambda r:r["trial"])
                base = sorted([r for r in selection if r["model"]==baseline],key=lambda r:r["trial"])
                moving.append(dict(zip(KEY,c), threshold=threshold, baseline=baseline,
                    mean_retained_fraction=avg([r["retained_fraction"] for r in ours]),
                    ours_yaw_rmse=avg([r["wz_error_rmse"] for r in ours]),
                    baseline_yaw_rmse=avg([r["wz_error_rmse"] for r in base]),
                    **difference([r["wz_error_rmse"] for r in ours], [r["wz_error_rmse"] for r in base])))
    write_rows(out/"common_moving_yaw_effects.csv", moving)
    # Group comparisons resample the ten initial-condition indices, preserving
    # their reuse across load/terrain/command conditions; never pool timestamps.
    aggregate = []
    strata = [("all",lambda c:True)]
    strata += [(f"mass_{m}",lambda c,m=m:c[1]==m) for m in (0,4,8)]
    strata += [(f"terrain_{t}",lambda c,t=t:c[0]==t) for t in ("flat","slope","stairs")]
    strata += [(f"yaw_{y}",lambda c,y=y:c[2]==y) for y in (0,.15)]
    strata += [(f"{t}_yaw_{y}",lambda c,t=t,y=y:c[0]==t and c[2]==y) for t in ("flat","slope","stairs") for y in (0,.15)]
    strata += [("flat_slope_turn",lambda c:c[0]!="stairs" and c[2]==.15)]
    for name,predicate in strata:
        cs = [c for c in conditions if predicate(c)]
        for baseline in BASELINES:
            for metric in ERRORS+EXTRA:
                a = [avg([next(r[metric] for r in grouped[c+(OUR,)] if int(r["trial"])==trial) for c in cs]) for trial in range(10)]
                b = [avg([next(r[metric] for r in grouped[c+(baseline,)] if int(r["trial"])==trial) for c in cs]) for trial in range(10)]
                aggregate.append(dict(stratum=name, conditions=len(cs), baseline=baseline, metric=metric,
                    ours_mean=avg(a), baseline_mean=avg(b), **difference(a,b,metric not in HIGHER)))
    write_rows(out/"stratified_paired_effects.csv", aggregate)
    # Full 18-condition inventory; metric improvements do not imply dominance.
    inventory = []
    for c in conditions:
        table = {m:next(r for r in condition_rows if condition(r)==c and r["model"]==m) for m in MODELS}
        r = dict(zip(KEY,c))
        for metric in ERRORS+("passed","mean_vx","final_x_m","path_xy_m"):
            for baseline in BASELINES:
                delta=table[OUR][metric+"_mean"]-table[baseline][metric+"_mean"]
                r[metric+"_vs_"+baseline]=delta
        r["both_baselines_xy_better"]=int(all(table[OUR][ERRORS[0]+"_mean"]<table[b][ERRORS[0]+"_mean"] for b in BASELINES))
        r["both_baselines_yaw_better"]=int(all(table[OUR][ERRORS[1]+"_mean"]<table[b][ERRORS[1]+"_mean"] for b in BASELINES))
        r["strict_all_six_metric_dominance"]=int(all(all(table[OUR][k+"_mean"]<=table[b][k+"_mean"] for k in ERRORS) and table[OUR]["passed_mean"]>=table[b]["passed_mean"] and table[OUR]["final_x_m_mean"]>=table[b]["final_x_m_mean"] for b in BASELINES))
        r["ours_mean_vx"]=table[OUR]["mean_vx_mean"]
        r["ours_slow_fraction"]=table[OUR]["slow_fraction_0p3_mean"]
        inventory.append(r)
    write_rows(out/"advantage_inventory.csv", inventory)
    # Mass/control clues are correlations within a condition, across ten trials.
    clues = []
    for c in conditions:
        ours = grouped[c+(OUR,)]
        for metric in ERRORS+("final_x_m","mean_vx"):
            x=np.array([r["mass_mae_steady_kg"] for r in ours])
            y=np.array([r[metric] for r in ours])
            clues.append(dict(zip(KEY,c), predictor="mass_mae_steady_kg", outcome=metric,
                n=10, pearson_r=correlations(x,y), predictor_min=float(x.min()), predictor_max=float(x.max())))
    write_rows(out/"mass_control_correlations.csv", clues)
    phase_groups=defaultdict(list)
    for r in phases:
        phase_groups[condition(r)+(r["phase_type"],r["stage"],r["model"])].append(r)
    write_rows(out/"phase_condition_summary.csv", [dict(zip(KEY,k[:3]),phase_type=k[3],stage=k[4],model=k[5],
        **summary(v,ERRORS+("duration_s","mean_vx","mass_mae"))) for k,v in sorted(phase_groups.items())])
    phase_effects=[]
    for c in (("flat",8.,0.),("slope",8.,0.)):
        for phase_type,stage in sorted({(r["phase_type"],r["stage"]) for r in phases}):
            ours=sorted(phase_groups[c+(phase_type,stage,OUR)],key=lambda r:r["trial"])
            for baseline in BASELINES:
                base=sorted(phase_groups[c+(phase_type,stage,baseline)],key=lambda r:r["trial"])
                for metric in ERRORS:
                    phase_effects.append(dict(zip(KEY,c),phase_type=phase_type,stage=stage,baseline=baseline,metric=metric,
                        ours_mean=avg([r[metric] for r in ours]),baseline_mean=avg([r[metric] for r in base]),
                        **difference([r[metric] for r in ours],[r[metric] for r in base])))
    write_rows(out/"heavy_phase_paired_effects.csv",phase_effects)
    moving_strata=[]
    for terrain in ("flat","slope","flat_slope"):
        for yaw in (0.,.15):
            for threshold in (.2,.3,.45):
                selected=[r for r in masked_episodes if (r["terrain"]==terrain or terrain=="flat_slope" and r["terrain"]!="stairs") and r["yaw_command"]==yaw and r["threshold"]==threshold]
                for baseline in BASELINES:
                    a=[avg([r["wz_error_rmse"] for r in selected if r["model"]==OUR and r["trial"]==trial]) for trial in range(10)]
                    b=[avg([r["wz_error_rmse"] for r in selected if r["model"]==baseline and r["trial"]==trial]) for trial in range(10)]
                    moving_strata.append(dict(terrain=terrain,yaw_command=yaw,threshold=threshold,baseline=baseline,
                        ours_mean=avg(a),baseline_mean=avg(b),
                        retained_fraction=avg([r["retained_fraction"] for r in selected]),**difference(a,b)))
    write_rows(out/"common_moving_stratified_effects.csv",moving_strata)
    # Load-related contrast uses paired initial conditions rather than treating
    # the same trial across loads as independent evidence.
    contrasts = []
    for terrain in ("flat","slope"):
        for baseline in BASELINES:
            for metric in ERRORS:
                def vals(model,mass):
                    return np.array([r[metric] for r in sorted(grouped[(terrain,mass,0.,model)], key=lambda r:r["trial"])])
                contrasts.append(dict(terrain=terrain, baseline=baseline, metric=metric,
                    comparison="(ours-baseline)_8kg minus (ours-baseline)_0kg",
                    **difference(vals(OUR,8)-vals(baseline,8),vals(OUR,0)-vals(baseline,0))))
    write_rows(out/"load_interaction_contrasts.csv", contrasts)
    # Offset sweep: reuse all 210 trials, acknowledge the 30 zero-offset replays.
    offset_rows = []
    for c in sorted(set(condition(r) for r in offset_episodes)):
        original=sorted([r for r in offset_episodes if condition(r)==c and r["offset_kg"]==0], key=lambda r:r["trial"])
        for offset in (-1.,-.5,-.25,0.,.25,.5,1.):
            group=sorted([r for r in offset_episodes if condition(r)==c and r["offset_kg"]==offset], key=lambda r:r["trial"])
            row=dict(zip(KEY,c), offset_kg=offset, **summary(group,ERRORS+("mass_input_mae_kg","passed","mean_vx","final_x_m")))
            for metric in ERRORS:
                delta=difference([r[metric] for r in group],[r[metric] for r in original])
                row.update({metric+"_"+k:v for k,v in delta.items()})
            offset_rows.append(row)
    write_rows(out/"offset_evidence.csv", offset_rows)
    make_figures(out, condition_rows, grouped, cases, phases, offset_rows)
    after={str(p.resolve()):digest(p) for p in protected}
    assert before==after, "Protected input changed during analysis"
    info=dict(input_root=str(root),source_files_verified_unchanged=len(before),
        formal_trajectories=750,comparison_trajectories=540,offset_trajectories=210,
        zero_offset_replays=30,raw_rows=450000,training_seed=42,
        evaluation_base_seed=20261025,analysis_seed=932,bootstrap_repetitions=4000,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        max_metric_recalculation_error=max(r["max_metric_error"] for r in audit),
        missing_fields=["latent_mu", "estimated_velocity", "full_observation", "joint_state", "foot_contacts", "yaw_angle"],
        limitations=["Exploratory, selected after seeing previous results; no holdout confirmation.",
            "One trained checkpoint per method; intervals cover initial conditions only.",
            "750 is a rollout count, including 30 deterministic zero-offset replays.",
            "Common-moving masks condition on policy outcomes and are not causal estimates.",
            "Geometric phases track base position, not actual foot/terrain contact.",
            "Sign-flip p values assume symmetric paired effects; Holm families have 36 tests per endpoint.",
            "More accurate mass, direct mass sensitivity, and method performance are distinct claims."])
    (out/"analysis_manifest.json").write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2),flush=True)


def make_figures(out, rows, grouped, cases, phases, offsets):
    colors=("#515c70","#18a9bd","#da258c")
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","pdf"):
            fig.savefig(out/f"{name}.{ext}",dpi=180)
        plt.close(fig)
    conditions=sorted(set(condition(r) for r in rows))
    fig,axes=plt.subplots(2,2,figsize=(13,10))
    for ax,metric in zip(axes.flat,ERRORS):
        data=[]
        for c in conditions:
            data.append([100*(next(r for r in rows if condition(r)==c and r["model"]==OUR)[metric+"_mean"]/
                next(r for r in rows if condition(r)==c and r["model"]==b)[metric+"_mean"]-1) for b in BASELINES])
        im=ax.imshow(data,cmap="RdBu_r",vmin=-60,vmax=60,aspect="auto")
        ax.set(xticks=[0,1],xticklabels=["vs Asym","vs CENet"],yticks=range(18),
            yticklabels=[f"{t}, {m:g}kg, wz={y:g}" for t,m,y in conditions],title=metric+" (%; lower is better)")
        for i,row in enumerate(data):
            for j,value in enumerate(row):
                ax.text(j,i,f"{value:+.0f}",ha="center",va="center",fontsize=7)
    fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.02,pad=.02)
    fig.suptitle("Exploratory condition inventory; error improvements require movement/progress audit")
    # Avoid constrained colorbar placement overlapping the right panels.
    fig.subplots_adjust(right=.87,top=.93,wspace=.65,hspace=.23)
    for ext in ("png","pdf"):
        fig.savefig(out/f"condition_effects.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,terrain in enumerate(("flat","slope")):
        c=(terrain,8.,0.)
        for ax,metric in zip(axes[i],(ERRORS[0],"final_x_m",ERRORS[1])):
            for j,model in enumerate(MODELS):
                v=[r[metric] for r in sorted(grouped[c+(model,)],key=lambda r:r["trial"])]
                ax.plot(range(10),v,"o-",label=model,color=colors[j],lw=1)
            ax.set(title=f"{terrain}, 8kg: {metric}",xlabel="Paired initial-condition index")
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    save(fig,"heavy_load_pairs")
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,terrain in enumerate(("flat","slope")):
        c=(terrain,8.,0.)
        for j,model in enumerate(MODELS):
            aa=cases[c+(model,)][1]
            vx=np.array([aa[t]["vx"] for t in range(10)])
            progress=np.array([aa[t]["x_progress"] for t in range(10)])
            mass=np.array([aa[t]["predicted_mass_kg"] for t in range(10)])
            for ax,v,title in zip(axes[i],(vx,progress,mass),("actual vx [m/s]","world +x progress [m]","predicted mass [kg]")):
                if not np.isfinite(v).any():
                    continue
                x=np.arange(600)*.02
                ax.plot(x,v.mean(axis=0),color=colors[j],label=model,lw=1)
                ax.fill_between(x,v.mean(axis=0)-v.std(axis=0),v.mean(axis=0)+v.std(axis=0),color=colors[j],alpha=.12)
                ax.set(title=f"{terrain}, 8kg: {title}",xlabel="Time [s]")
                ax.grid(alpha=.2)
        axes[i,0].axhline(.6,color="grey",ls="--")
        axes[i,2].axhline(8,color="grey",ls="--")
    axes[0,0].legend(fontsize=7)
    save(fig,"heavy_load_timecourses")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,terrain in zip(axes,("flat","slope")):
        for color,model in zip(colors,MODELS):
            selected=[r for r in phases if r["terrain"]==terrain and r["true_mass_kg"]==8 and r["yaw_command"]==0 and r["model"]==model and r["phase_type"]=="clock" and r["stage"]!="0-2s"]
            labels=[f"{t}-{t+2}s" for t in range(2,12,2)]
            ax.plot(range(5),[avg([r[ERRORS[0]] for r in selected if r["stage"]==label]) for label in labels],"o-",color=color,label=model)
        ax.set(title=f"8kg {terrain}: fixed-clock phases",xticks=range(5),xticklabels=labels,ylabel="Planar velocity RMSE [m/s]")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    save(fig,"heavy_load_phases")
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for i,yaw in enumerate((0.,.15)):
        for j,terrain in enumerate(("flat","slope","stairs")):
            ax=axes[i,j]
            for color,model in zip(colors,MODELS):
                selected=sorted([r for r in rows if r["terrain"]==terrain and r["yaw_command"]==yaw and r["model"]==model],key=lambda r:r["true_mass_kg"])
                ax.plot([r["true_mass_kg"] for r in selected],[r["wz_error_rmse_mean"] for r in selected],"o-",color=color,label=model)
            ax.set(title=f"{terrain}; wz={yaw:g}",xlabel="True payload [kg]",ylabel="Yaw RMSE [rad/s]",xticks=[0,4,8])
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    save(fig,"yaw_by_terrain_command")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,terrain in zip(axes,("flat","slope","stairs")):
        for color,model in zip(colors,MODELS):
            selected=sorted([r for r in rows if r["terrain"]==terrain and r["yaw_command"]==.15 and r["model"]==model],key=lambda r:r["true_mass_kg"])
            ax.scatter([r["mean_vx_mean"] for r in selected],[r["wz_error_rmse_mean"] for r in selected],color=color,label=model)
            for r in selected:
                ax.annotate(f"{r['true_mass_kg']:g}kg",(r["mean_vx_mean"],r["wz_error_rmse_mean"]),fontsize=7)
        ax.set(title=terrain,xlabel="Mean actual vx [m/s]",ylabel="Yaw RMSE [rad/s]")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    save(fig,"turn_yaw_vs_actual_speed")
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,c in zip(axes,sorted(set(condition(r) for r in offsets))):
        selected=sorted([r for r in offsets if condition(r)==c],key=lambda r:r["offset_kg"])
        ax.plot([r["mass_input_mae_kg_mean"] for r in selected],[r[ERRORS[0]+"_mean"] for r in selected],"o-")
        for r in selected:
            ax.annotate(f"{r['offset_kg']:+g}",(r["mass_input_mae_kg_mean"],r[ERRORS[0]+"_mean"]),fontsize=7)
        ax.set(title=f"{c[0]}, {c[1]:g}kg (labels: offset kg)",xlabel="Actor mass-input MAE [kg]",ylabel="Planar RMSE [m/s]")
        ax.grid(alpha=.2)
    save(fig,"mass_accuracy_control_tradeoff")


if __name__=="__main__":
    main()
