"""Audit every raw trajectory, aggregate per-trial metrics, export three figures."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comparison_support import (RUNS, METRICS, save_csv, sha256, trajectory_metrics,
                                validate_initial_pair, write_json)

LABELS = {"Asym_PPO": "Asym PPO", "DreamWaQ_CENet": "DreamWaQ-CENet", "PayloadEstimator": "Ours"}
TITLES = ("Base Roll Deviation", "Base Flatness Penalty", "Velocity Tracking Error")
YLABELS = ("Base Roll Deviation (rad)", "Base Flatness Penalty (dimensionless)", "Velocity Tracking Error (m/s)")


def display(value, signed=False):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:+.2f}" if signed else f"{value:.6f}"


def read_rows(path):
    with path.open() as handle:
        return [{k: v if k == "model" else float(v) if v else None for k, v in r.items()}
                for r in csv.DictReader(handle)]


def aggregate(rows):
    forward = [r["mean_forward_velocity_m_s"] for r in rows
               if r["mean_forward_velocity_m_s"] is not None and np.isfinite(r["mean_forward_velocity_m_s"])]
    result = dict(trajectories=len(rows), complete=sum(r["complete"] for r in rows),
        failed=sum(r["failed"] for r in rows), timeout=sum(r["timeout"] for r in rows),
        stalls=sum(r["stall_flag"] for r in rows), invalid_trials=sum(r["invalid_samples"] > 0 for r in rows),
        scored_samples_min=min(r["scored_samples"] for r in rows),
        scored_samples_max=max(r["scored_samples"] for r in rows),
        mean_forward_velocity_m_s=float(np.mean(forward)) if forward else None,
        final_progress_min_m=min(r["final_world_x_progress_m"] for r in rows),
        final_progress_max_m=max(r["final_world_x_progress_m"] for r in rows))
    for metric in METRICS:
        values = [r[metric] for r in rows if r[metric] is not None and np.isfinite(r[metric])]
        result.update({metric+"_mean": float(np.mean(values)) if values else None,
                       metric+"_sd": float(np.std(values, ddof=1)) if len(values)>1 else None,
                       metric+"_n": len(values)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new summary directory; no result is overwritten.")
    top = json.loads((args.run_root/"manifest.json").read_text())
    assert top["status"] == "complete"
    protocol = json.loads((args.run_root/"protocol.json").read_text())
    if not args.allow_smoke:
        assert sorted(top["args"]["masses_kg"]) == [2., 4., 6., 8.]
        assert top["trajectories"] == 120 and top["args"]["num_trials"] == 10
        assert top["args"]["duration"] == 12. and top["args"]["warmup"] == 2.
    args.output_dir.mkdir(parents=True)
    groups, summary, episodes, anomalies, inputs = {}, [], [], [], []
    max_recalculation_error = max_gravity_error = max_roll_error = 0.
    manifests = {}
    for mass in sorted(top["args"]["masses_kg"]):
        reference = None
        for model in RUNS:
            directory = args.run_root/f"mass_{mass:g}kg"/model
            manifest = json.loads((directory/"manifest.json").read_text())
            manifests[mass, model] = manifest
            assert manifest["status"] == "complete" and manifest["initial_pairing_verified"]
            initial = json.loads((directory/"initial_state.json").read_text())
            if reference is None:
                reference = initial
            validate_initial_pair(initial, reference)
            saved = read_rows(directory/"episode_metrics.csv")
            assert len(saved) == top["args"]["num_trials"]
            rows = []
            for trial in range(len(saved)):
                path = directory/f"trial_{trial:02d}.csv"
                raw = read_rows(path)
                assert raw
                assert [r["step"] for r in raw] == list(range(len(raw)))
                assert all(r["model"] == model and r["trial"] == trial and r["payload_kg"] == mass for r in raw)
                for r in raw:
                    assert abs(r["actual_added_base_mass_kg"]-mass) <= 2e-6
                    np.testing.assert_allclose([r["cmd_vx"],r["cmd_vy"],r["cmd_wz"]],[.6,0,0],rtol=0,atol=1e-6)
                    assert r["scored"] == int(r["time_before_s"] >= top["args"]["warmup"]-1e-9)
                    # Independent quaternion equations: gravity direction R(q)^T*[0,0,-1].
                    w,x,y,z = (r[k] for k in ("qw","qx","qy","qz"))
                    expected_g = np.array([2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))])
                    max_gravity_error = max(max_gravity_error, float(np.max(np.abs(expected_g-[r["gx"],r["gy"],r["gz"]]))))
                    independent_roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
                    max_roll_error = max(max_roll_error, abs(float(independent_roll)-r["roll"]))
                calculated = trajectory_metrics(raw, top["args"]["duration"],top["args"]["warmup"],manifest["control_dt"])
                for key, value in calculated.items():
                    if value is None:
                        assert saved[trial][key] is None
                    else:
                        error = abs(saved[trial][key]-value)
                        max_recalculation_error = max(max_recalculation_error, error)
                        assert error < 1e-10, (path,key,error)
                row = dict(model=model, payload_kg=mass, trial=trial,
                    evaluation_seed=raw[0]["evaluation_seed"], training_seed=manifest["checkpoint"]["agent"]["seed"],
                    **calculated)
                rows.append(row)
                if not row["complete"] or row["stall_flag"] or row["invalid_samples"]:
                    anomalies.append(row)
                inputs.append(dict(path=str(path.resolve()),sha256=sha256(path),samples=len(raw)))
            groups[mass,model] = rows
            episodes.extend(rows)
            summary.append(dict(payload_kg=mass,model=model,**aggregate(rows)))
    assert len(episodes) == top["trajectories"]
    assert max_gravity_error < 2e-6 and max_roll_error < 2e-6
    save_csv(args.output_dir/"episode_metrics.csv",episodes)
    save_csv(args.output_dir/"condition_summary.csv",summary)
    save_csv(args.output_dir/"raw_inputs.csv",inputs)
    # Always create the anomalies file, including a header when no anomaly exists.
    if anomalies:
        save_csv(args.output_dir/"anomalies.csv",anomalies)
    else:
        (args.output_dir/"anomalies.csv").write_text("model,payload_kg,trial,failed,timeout,stall_flag,invalid_samples\n")
    lookup = {(r["payload_kg"],r["model"]):r for r in summary}
    improvements = []
    for mass in sorted(top["args"]["masses_kg"]):
        for baseline in ("Asym_PPO","DreamWaQ_CENet"):
            ours, other = lookup[mass,"PayloadEstimator"],lookup[mass,baseline]
            for metric in METRICS:
                a,b = ours[metric+"_mean"], other[metric+"_mean"]
                paired = [(o[metric], r[metric]) for o,r in zip(groups[mass,"PayloadEstimator"],groups[mass,baseline])
                          if o["complete"] and r["complete"] and o[metric] is not None and r[metric] is not None]
                improvements.append(dict(payload_kg=mass,baseline=baseline,metric=metric,ours_mean=a,baseline_mean=b,
                    relative_improvement_percent=100*(b-a)/b if a is not None and b else None,
                    complete_pairs=len(paired),paired_win_count=sum(o<r for o,r in paired),
                    interpretation="positive=Ours lower; negative=Ours worse; descriptive, one training seed"))
    save_csv(args.output_dir/"relative_improvements.csv",improvements)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
        "axes.spines.right":False,"pdf.fonttype":42,"ps.fonttype":42,"axes.linewidth":.8})
    colors = ("#0072B2","#E69F00","#009E73")
    styles = ("o","s","^")
    for metric,title,ylabel in zip(METRICS,TITLES,YLABELS):
        fig,ax = plt.subplots(figsize=(5.4,3.8),layout="constrained")
        for model,color,marker in zip(RUNS,colors,styles):
            selected = [r for r in summary if r["model"]==model]
            ax.errorbar([r["payload_kg"] for r in selected], [r[metric+"_mean"] for r in selected],
                yerr=[r[metric+"_sd"] for r in selected], label=LABELS[model],color=color,marker=marker,
                markersize=5,linewidth=1.5,capsize=3,elinewidth=1)
        ax.set(xlabel="Payload Mass (kg)",ylabel=ylabel,title=title,xticks=sorted(top["args"]["masses_kg"]))
        ax.grid(axis="y",alpha=.22,linewidth=.6)
        ax.legend(frameon=False,fontsize=9)
        fig.savefig(args.output_dir/f"{metric}.png",dpi=300)
        fig.savefig(args.output_dir/f"{metric}.pdf")
        plt.close(fig)
    report = ["# 三模型平地附加负载对比\n",
        f"独立新评估 {len(episodes)} 条轨迹；种子 {top['args']['evaluation_seed']}。三个训练 checkpoint 均为 seed 42。",
        "每条 12 秒，过渡 2 秒；以 50 Hz 记录 (2,12] 秒的 500 个仿真真值样本。下表为先逐轨迹计算，再对 10 条轨迹求均值 ± 样本标准差（ddof=1）。",
        "附加质量仅加到 base，惯量按原事件更新；固定 COM，关闭推力、材料随机化和观测噪声。每个负载下三模型初始物理状态配对。",
        "本实验自定义统一口径，不声称与任何论文的同名指标可直接数值对应。",
        "\n|负载 kg|模型|Roll RMS rad|Flatness|平面速度 RMSE m/s|平均前进速度 m/s|完整/停滞/无效|",
        "|---:|---|---:|---:|---:|---:|---|" ]
    for r in summary:
        values = [display(r[m+'_mean'])+" ± "+display(r[m+'_sd']) for m in METRICS]
        report.append(f"|{r['payload_kg']:g}|{LABELS[r['model']]}|"+"|".join(values)+
            f"|{display(r['mean_forward_velocity_m_s'])}|{int(r['complete'])}/{int(r['stalls'])}/{int(r['invalid_trials'])}|")
    report.extend(["\n相对改善 = 100 × (基线均值 − Ours 均值) / 基线均值，正数为改善。",
        "\n|负载 kg|相对基线|Roll 改善 %|Flatness 改善 %|速度改善 %|", "|---:|---|---:|---:|---:|"])
    for mass in sorted(top["args"]["masses_kg"]):
        for baseline in ("Asym_PPO","DreamWaQ_CENet"):
            entries = [r for r in improvements if r["payload_kg"]==mass and r["baseline"]==baseline]
            report.append(f"|{mass:g}|{LABELS[baseline]}|"+"|".join(display(r['relative_improvement_percent'],signed=True) for r in entries)+"|")
    report.extend(["\n运动检查：停滞标记为统计窗口平均机体前向速度 <0.1 m/s，或连续 ≥2 秒前向速度 <0.1 m/s。仅用于诊断，不删数据、不改变三个指标。",
        f"异常轨迹总计 {len(anomalies)}；详见 anomalies.csv。提前终止轨迹保留，统计窗口被截断时不可把较小误差解读为提升。",
        "这组试验检验三个完整策略在固定平地及四种静态附加负载下的表现。10 次重复描述初始状态变化，并非 10 个独立训练种子。",
        "即使 Ours 部分指标更好，也不能仅凭三模型对比把因果归于显式质量输入；该机制仍需要训练消融与多训练种子验证。",
        f"\n原始数据位于 {args.run_root.resolve()}（每条轨迹独立 CSV）；完整配置、初始状态、checkpoint SHA256 和质量检查见各 case 的 manifest.json。"])
    (args.output_dir/"README.md").write_text("\n".join(report)+"\n")
    write_json(args.output_dir/"audit.json",dict(status="passed",trajectories=len(episodes),anomalies=len(anomalies),
        max_metric_recalculation_error=max_recalculation_error, max_quaternion_gravity_error=max_gravity_error,
        max_quaternion_roll_error=max_roll_error, input_protocol_sha256=sha256(args.run_root/"protocol.json"),
        formulas=protocol["formulas"], aggregation=protocol["aggregation"],
        checkpoint_sha256={model:manifests[next(iter(top["args"]["masses_kg"])),model]["checkpoint"]["checkpoint_sha256"] for model in RUNS}))
    print("SUMMARY_COMPLETE",len(episodes),"trajectories; anomalies",len(anomalies),flush=True)


if __name__ == "__main__":
    main()
