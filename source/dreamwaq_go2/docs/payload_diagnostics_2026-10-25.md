# PayloadEstimator 质量通道诊断（2026-10-25）

本报告针对已训练的 `model_4999.pt`，不包含重新训练、算法新增或 checkpoint 修改。
结论限于 Isaac Lab 平地、恒定前进指令的小规模诊断，不能直接作为最终论文结论。

## 1. 模型、训练配置与评估入口核对

使用的 checkpoint：

`/home/hyc/dreamwaq_go2/logs/rsl_rl/payload_comparison/PayloadEstimator/2026-10-16_21-12-27/model_4999.pt`

内部 iteration 为 4999，SHA256 为
`32b9145c3e554a322ad0f493c5db2eb287baa80868b6fe8b3796aacaaeb0180a`。
诊断从该运行的 `params/agent.yaml` 构造模型，严格匹配全部参数；missing/unexpected keys 均为空。
保存配置明确为 `empirical_normalization=false`、`detach_payload=true`、`use_payload_input=true`。
运行前后重新校验 checkpoint SHA256。

另外读取了 Asym_PPO 和 DreamWaQ_CENet 的最终 checkpoint 元数据，均为 iteration 4999。
本次问题是同一个 PayloadEstimator 的质量输入干预，未将两种 baseline 混入因果比较。

| 组件 | 实际输入与输出 |
| --- | --- |
| 当前观测 | 角速度 3、重力投影 3、速度指令 3、关节相对位置 12、关节速度 12、上一动作 12，共 45 维 |
| 外部 Actor observation | 当前 45 + 历史 25×45 = 1170 维 |
| 历史布局 | term-major；每个项内部从旧到新，最新帧包含当前时刻；reset 后重复首帧填充 |
| CENet | 从每个历史项取最近 5 帧，共 225 维；encoder 225→128→64，输出速度 3、μ 16、logvar 16 |
| 负载估计器 | 独立 MLP：1125→64→32→1，使用完整 25 帧历史 |
| Actor head | `[当前观测45, 估计速度3, μ16, 预测质量1]`，65→512→256→128→12 |
| Critic | 外部 239 维，最后质量标签被剔除，实际 value 网络输入 238 维 |
| 推理动作 | 确定性动作均值；仍使用原 0.25 action scale、关节位置控制和 50 Hz 控制频率 |

质量标签是 **base 实际质量减去默认 base 质量**，不是整机质量。由 PhysX mass tensor 读取，reset 后失效重读。
训练目标为 `q_true = m_added / 8 kg`，网络直接输出 `q_hat`；记录物理预测值时乘 8 kg。
末层线性，无 sigmoid 或裁剪。质量监督为 `0.1 * Huber(q_hat, q_true)`，默认 delta=1，mean reduction。
这相当于归一化误差尺度上的 Huber，不能把损失数值直接解释为 kg。

`detach_payload=true` 截断 **Actor/PPO 到质量估计器** 的梯度，未阻止 Actor 学习最后一列权重。
质量 Huber 仅直接训练质量 MLP；PPO Actor 路径仍训练 CENet encoder、速度头、μ 头及 policy head。
原速度 MSE、下一观测重构和 KL 辅助损失保持原样；所有网络仍共享 optimizer 和全局梯度裁剪。
因此不应描述成所有模块完全独立优化。真实质量只作监督，不直接进入训练 Actor 或 value 网络。

最终 checkpoint 的 Actor 首层质量列 L2 范数为 4.186605，已不同于零初始化。
这是参数层面证据，是否影响动作由第三阶段实测回答。

核对的原文件：

- `agents/payload_actor_critic.py`：历史切片、预测和拼接、detach、value 剔除标签。
- `agents/payload_ppo.py`、`agents/dreamwaq_ppo.py`：监督、辅助损失及优化路径。
- `payload_env_cfg.py`、`dreamwaq_env_cfg.py`、`rough_env_cfg.py`：观测布局及物理负载事件。
- `scripts/rsl_rl/play.py`：已经注册 Payload 类并支持显式 checkpoint，正常推理可以加载该模型；没有质量覆盖参数，也没有所需 CSV 诊断。
- `scripts/rsl_rl/play_fig5_estimation.py`：只注册原 DreamWaQ 类，默认日志根路径也未适配 payload_comparison；不可直接当作 PayloadEstimator 的有效质量评估脚本。
- `scripts/evaluate_command_tracking_policy_ppo.py`：MuJoCo 旧接口只支持 235/270 维，不支持本模型 1170 维。

因此新增独立评估入口，复用原模型类、任务配置、Actor 前向和 Isaac Lab recorder 接口。
无需改动训练代码：只在 `policy_head` 的 forward pre-hook 修改最后一列。
模型的 CENet、速度、μ 和质量预测仍照常计算；每次干预检查前 64 列逐元素不变。

## 2. 实验协议与指标

| 项目 | 实际设置 |
| --- | --- |
| 仿真 | Isaac Sim 5.0、Isaac Lab 0.44.9，GPU PhysX，CUDA:0 |
| 推理 | PyTorch 2.7.0+cu128，确定性动作均值，未进行参数更新 |
| 质量准确性 | 真实附加质量 0、4、8 kg，正常预测质量控制 |
| 同状态敏感性 | 正常预测、真实质量、固定 0/4/8 kg、预测值 ±0.5/±2 kg |
| 闭环干预 | 真实负载固定 4 kg，对比 predicted、oracle、fixed_0、fixed_4、fixed_8 |
| 指令 | 机体系 vx=0.6 m/s、vy=0、wz=0；无重采样和 heading controller |
| 地形与干扰 | 平地；关闭观测噪声、COM 随机化、接触材料随机化、外力和推搡 |
| 初始状态 | 默认关节姿态和零速度；reset yaw∈[-0.15,0.15] rad，配对时保持完全一致 |
| 种子与规模 | 11/22/33，每次 4 个环境，每条轨迹 12 s |
| 计分窗口 | 前 2 s 单列过渡段；主指标使用随后 10 s；质量 MAE 另外报告完整 12 s |
| 时间步 | physics dt=0.005 s，decimation=4，policy dt=0.02 s |
| 物理负载 | 沿用训练 add_base_mass 事件及 recompute_inertia=True；同负载各干预不改变质量或惯量 |

每个条件用独立进程创建物理场景，防止跨条件残留接触/求解器状态。
配对检查包含 root state、joint state、1170 维初始历史观测、质量、惯量、COM、材料和指令的完整哈希。
`fixed_4` 与 `oracle` 在真实 4 kg 时输入相同，是实现校验条件；错误质量条件是 `fixed_0`、`fixed_8`。
所有替换值统一除以 8，扰动不裁剪，同时记录是否超出 [0,8] kg 标签范围。

敏感性每 10 个控制步采样一次。一次捕获原 Actor 的 65 维输入，然后保持前 64 维不变，计算所有质量条件的 12 维动作。
原观测、冻结特征、各条件动作及逐维差值均存 CSV；探测动作不送回仿真。
正常预测的重算动作与原前向动作逐步核对。

闭环性能使用 **物理步之后、自动 reset 之前** 的真实速度和姿态，质量预测与真值则在动作之前对齐。
线速度指标为 `sqrt(mean((vx-vx_cmd)^2 + (vy-vy_cmd)^2))`，另存 vx、vy 分量 RMSE。
角速度指标为偏航 wz RMSE；Roll/Pitch RMS 使用 wrap 到 [-π,π] 的弧度。
失败按原任务 base_contact termination 记录，timeout 单列；每个环境只计首个 episode，失败后不把 reset 数据混入。
若出现提前失败，RMSE 只覆盖存活窗口，必须与失败率和运行时长一起解读。

统计表中闭环均值先计算每条轨迹指标，再对 12 条轨迹平均；另保存按种子汇总和逐对差值。
曲线阴影是轨迹间标准差，不是置信区间。三组种子主要改变初始 yaw，扰动种类很少；同种子内并行环境及时间点不视为独立训练重复。

## 3. 质量预测准确性

以下每个负载均来自 3 个种子×4 环境，7200 个完整时刻、6000 个去除过渡段后的时刻。
这不是 6000 次独立实验。0/4/8 kg 都在训练负载范围内，未测试超出 8 kg 的外推。

| 真实附加质量 kg | 稳态预测均值 kg | 稳态 MAE kg | 稳态有符号偏差 kg | 稳态绝对误差 P95 kg | 完整 12 s MAE kg |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -0.0466 | 0.0670 | -0.0466 | 0.1512 | 0.1220 |
| 4 | 3.2449 | 0.7551 | -0.7551 | 0.9333 | 0.7644 |
| 8 | 7.3997 | 0.6003 | -0.6003 | 0.8043 | 0.6479 |

三种负载等权的稳态 MAE 为 0.4741 kg，但单一平均值掩盖了 4 kg 下约 18.9% 的系统低估；8 kg 的相对 MAE 约 7.5%。
4 kg 稳态误差的三个 seed 均值为 0.7571、0.7562、0.7520 kg；8 kg 为 0.5857、0.6062、0.6091 kg。
这更像这些条件下的稳定偏差，而不是由个别异常轨迹造成的结果。

起始时刻三个真实负载的观测历史缺少动力学响应，预测均接近 3.6 kg，随后才区分负载。
因此曲线保留过渡段；不能只用 4 kg 起始时“恰好接近真值”说明估计准确。
空载稳态约 77.3% 的逐时刻预测略为负值，符合末层无边界约束的实现；本次没有为了改善数字加截断。
这提示物理标定和输出约束值得研究，但是否裁剪应另行验证其控制影响。

曲线：`summary/mass_prediction.png` 和可导出的 `mass_prediction.pdf`。
数据：`summary/accuracy_summary.csv`、`accuracy_per_episode.csv`，原始值在各条件 `trajectory.csv`。

## 4. Actor 对质量输入的同状态敏感性

在真实负载 4 kg 的自然轨迹上，去除过渡段后共有 600 个冻结状态。
表中动作差 RMS 指先对 12 个动作维度求 RMS，再对这些状态平均；原动作单位尚未乘 action scale。

| 输入质量 | 平均动作差 RMS | 动作差 RMS 的 P95 | 平均关节目标差 RMS rad |
| --- | ---: | ---: | ---: |
| 正常预测 | 0 | 0 | 0 |
| 真实质量 / 固定 4 kg | 0.02964 | 0.04317 | 0.00741 |
| 固定 0 kg | 0.12713 | 0.18860 | 0.03178 |
| 固定 8 kg | 0.19338 | 0.26858 | 0.04834 |
| 预测 +0.5 kg | 0.01998 | 0.02859 | 0.00500 |
| 预测 -0.5 kg | 0.01986 | 0.02834 | 0.00497 |
| 预测 +2 kg | 0.08056 | 0.11483 | 0.02014 |
| 预测 -2 kg | 0.07869 | 0.11311 | 0.01967 |

真实质量替换在这 12 个动作维度上均产生非零差异；逐维 RMS 约为 0.0151–0.0630，前腿 calf 通道较敏感。
动作顺序为 FL/FR/RL/RR 的 hip、thigh、calf 三组，完整名称存于 manifest。
0 kg 和 8 kg 真实负载下也观察到非零敏感性，数据包含于 `sensitivity_summary.csv`。

**可以确认 Actor 实际使用显式质量输入，而不是只在结构中保留一个无效通道。**
但“存在动作敏感性”不等于“依赖该通道才能稳定行走”，更不等于“显式质量带来优于 CENet 的性能收益”。
后两种主张仍需要闭环证据和另行设计的完整方法对照；本次没有训练消融模型。

分布偏移必须单列：4 kg 自然轨迹的预测质量 1%–99% 分位约为 3.015–3.469 kg；oracle=4 kg 改变了当前状态与质量通道的对应关系。
这只是本次自然轨迹的经验分布，未测量训练期间 Actor 的条件输入分布，不能直接断言训练分布之外。
固定 0/8 kg 都在训练**标签范围**内，也可能与同一时刻的 μ、速度和本体观测不匹配。
空载预测 -0.5/-2 kg 干预以及 8 kg 条件的 +2 kg 干预超出标签范围；8 kg 的 +0.5 kg 约 22.2% 冻结状态超出该范围。
这些数据可作为敏感性压力测试，不能全部解释为正常估计误差的影响。

曲线/数据：`summary/action_sensitivity.png`、`per_action_sensitivity.csv`、`sensitivity_summary.csv`；
所有 12 维原始动作和差值均在 `suite/*predicted*/action_sensitivity.csv`。

## 5. 4 kg 闭环性能与五个研究问题

全部 21 次短仿真完成，共 84 条 12 s 轨迹。这里的 4 kg 对比包含每种输入 12 条轨迹；
predicted 的 12 条同时用于质量准确性和闭环比较，没有重复计为新实验。

| Actor 质量输入 | 平面速度 RMSE m/s | 偏航 RMSE rad/s | Roll RMS rad | Pitch RMS rad | 原估计器 MAE kg | 失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 正常预测 | 0.05533 | 0.05632 | 0.01292 | 0.06280 | 0.75510 | 0/12 |
| 真实质量 | 0.06407 | 0.05388 | 0.01219 | 0.06713 | 0.81503 | 0/12 |
| 固定 4 kg | 0.06407 | 0.05388 | 0.01219 | 0.06713 | 0.81503 | 0/12 |
| 固定 0 kg（错误） | 0.04972 | 0.07251 | 0.01594 | 0.04966 | 0.72035 | 0/12 |
| 固定 8 kg（错误） | 0.09189 | 0.04385 | 0.01246 | 0.07028 | 0.34187 | 0/12 |

表中“原估计器 MAE”始终是仍在正常运行的质量 MLP 的误差，**不是** oracle/fixed 替换后的输入误差。
Actor 实际质量输入 MAE 在 oracle/fixed_4 为 0，在 fixed_0/fixed_8 为 4 kg。
闭环轨迹改变后，估计器看到的历史也改变，所以不同条件下原估计器 MAE 可以不同。
尤其 fixed_8 下估计器 MAE 更低、速度跟踪却更差，不能把跨条件 MAE 差异解释为模型参数变好了。

相对正常预测输入：

- 真实质量：平面速度 RMSE **增加 15.79%**，偏航 RMSE 降低 4.33%，Roll RMS 降低 5.69%，Pitch RMS 增加 6.88%。
- 错误 0 kg：平面速度 RMSE 降低 10.14%，偏航 RMSE **增加 28.75%**，Roll RMS **增加 23.36%**，Pitch RMS 降低 20.92%。
- 错误 8 kg：平面速度 RMSE **增加 66.08%**，偏航 RMSE 降低 22.14%，Roll RMS 降低 3.60%，Pitch RMS 增加 11.91%。

这些变化方向在三个种子的均值上均一致，但这不是统计显著性检验。
所有条件均未触发 base_contact failure，也无 timeout；短时、单一指令、无外扰条件不足以测定鲁棒失效边界。
oracle 和 fixed_4 三个种子各 2400 行轨迹中的所有 38 个数值列逐项相同；这是实现与复现校验，不是额外独立的性能证据。

本次可以明确回答：

1. **质量预测是否准确？** 能区分 0/4/8 kg，但不能称为全范围高精度。空载稳态误差小，4/8 kg 下存在稳定低估，起步阶段还有明显不确定性；恒定负载下也有步态相关波动。
2. **Actor 是否依赖显式质量？** 已确认动作对该通道有非零、可重复的响应，12 个动作维度都受影响，闭环性能也随之变化。因此不是“质量估计只作辅助损失而未进入 Actor”。尚不能称为不可替代或已证明优于纯 CENet。
3. **真实质量是否更好？** 本条件下没有全面改善。它改善偏航/横滚指标，却使平面跟踪与 Pitch 更差；不能把 oracle 自动当作该已训练策略的性能上界。
4. **错误质量是否使控制下降？** 会恶化部分关键指标，且方向取决于质量值。8 kg 错误输入明显恶化速度跟踪；0 kg 错误输入恶化偏航与 Roll。没有观察到所有指标单调下降或直接跌倒。
5. **主要不足及改进价值？** 首要问题是条件性的质量预测偏差，以及尚未建立“物理估计更准确→多指标控制更好”的证据链。当前并非 Actor 忽略质量，因此不宜先盲目增强质量通道权重。估计器标定、步态相关误差与显隐特征融合的一致性值得进一步研究，但单纯降低全局 MAE 不保证收益。

一种待验证的解释是 Actor 适应了训练时带偏差的预测质量，并与 μ/速度特征共同使用；也可能包含多目标奖励产生的控制取舍。
本次干预同时改变预测误差和质量与其余上下文的对应关系，**不能区分这些机制，也不能把所有性能变化归因于分布偏移**。
同状态动作测试隔离了输入的直接作用；闭环结果还包括历史、姿态和估计误差随反馈演化的间接作用。

在开展任何新算法前，建议先用现有脚本补充少量不同速度、非零转向、地形/噪声条件，检查偏差与指标取舍是否保持。
再围绕已有预测附近的小扰动或恒定标定偏移做闭环扫值，区分“误差大小”和“条件输入匹配”的影响；这只是后续建议，本次没有执行额外算法或训练。
只有在更广诊断中正确质量稳定改善目标指标，才能将估计精度视为主要控制瓶颈；
若仍出现类似取舍，应优先研究融合对误差/不确定性的处理和控制目标一致性，而不是仅扩大负载 MLP。

原假设目前只获得了“显式质量通道被策略使用”的支持，**还没有获得“它优于 DreamWaQ_CENet 的完整运动控制效果”的证明**。
该比较需后续在统一评估协议下直接评估三个模型，并纳入更多训练种子；不应拿本次单一 PayloadEstimator 的替换实验替代完整方法对比。

数据：`summary/closed_loop_summary.csv`（均值及标准差）、`closed_loop_per_seed.csv`、`paired_differences.csv`、`closed_loop_bias.csv`。
图：`summary/closed_loop.png`。全部结果为描述性诊断，不作论文最终结论。

## 6. 文件、复现与 Git 记录

原始结果根目录：`/home/hyc/dreamwaq_go2/logs/payload_diagnostics/2026-10-25/`。
`suite/<条件>/` 包含 `trajectory.csv`、`episode_metrics.csv`、初始状态、有效配置、配置差异和 manifest。
正常预测条件另有 `observations.csv`、`actor_features.csv`、`action_sensitivity.csv`。
`summary/` 包含准确性、同状态敏感性、闭环汇总和逐对差值 CSV，以及真实/预测质量曲线、动作热图、闭环指标图。
`runtime/` 保留包括早期启动和配置检查失败在内的完整日志；早期失败不计入实验结果。

运行环境通过已有 Isaac Sim 的 `python.sh` 补齐 IsaacLab conda 环境缺失的依赖，没有安装或修改 Python 环境。
GPU 仿真需要沙箱外执行；CPU 模型校验可在沙箱内执行。

单次复现示例（输出目录必须是新的）：

```bash
cd /home/hyc/dreamwaq_go2
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/payload-mpl \
PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python \
/home/hyc/isaac-sim/python.sh scripts/rsl_rl/diagnose_payload.py \
  --headless --device cuda:0 \
  --checkpoint logs/rsl_rl/payload_comparison/PayloadEstimator/2026-10-16_21-12-27/model_4999.pt \
  --output-dir logs/payload_diagnostics/reproduce_mass4_predicted_seed11 \
  --mass-kg 4 --mode predicted --seed 11 --num-envs 4 \
  --duration 12 --warmup 2 --probe-stride 10
```

真实质量条件改 `--mode oracle`；固定质量改 `--mode fixed --value-kg 0`（或 4、8）；都保持 `--mass-kg 4`。
批次精确命令存 `stage2_commands.json` 和 `stage4_commands.json`。
汇总入口：`scripts/rsl_rl/summarize_payload_diagnostics.py --run-root <suite> --output-dir <summary>`，使用同一 Python 启动方式。

只修改仓库 `/home/hyc/dreamwaq_go2`，branch=`feature/dreamwaq-cenet-skeleton`：

| 新增文件 | 目的 |
| --- | --- |
| `scripts/rsl_rl/payload_diagnostics.py` | 严格加载 checkpoint，质量单列干预，同状态动作探测，明确指标定义 |
| `scripts/rsl_rl/diagnose_payload.py` | 复用 Isaac Lab 任务进行配对短仿真，记录 CSV 和复现状态 |
| `scripts/rsl_rl/summarize_payload_diagnostics.py` | 校验配对、汇总 CSV、生成图表 |
| `scripts/rsl_rl/test_payload_diagnostics.py` | 实际 checkpoint 的无干预一致性、仅质量列修改、归一化、指标和权重不变检查 |
| 本报告 | 记录证据、结论边界和复现方法 |

提交：

- `3848efb` — `chore: mark baseline before payload input diagnostics`：修改代码前的空基线提交，不包含用户已有文件。
- `1e711c5` — `feat: add paired payload estimation and actor input diagnostics`：四个评估/验证脚本的完整逻辑提交。
- 报告提交的最终 hash 与最终 status 见本次交付回复及结果根目录的 `delivery_manifest.json`（避免在提交内容中自引用其 hash）。

评估代码版本为 `1e711c5`。现有 checkpoint 的保存配置：seed=42、4096 环境、24 steps/iteration、5000 iterations、PPO 5 epochs/4 minibatches、无 empirical normalization。
负载课程为 0–3 / 0–5 / 0–8 kg，阶段分界 1500/3500 iterations；这些训练参数本次未改动。

外部依赖 `/home/hyc/IsaacLab` 为 detached HEAD `46dff135f44683f031edf346e544fcfd8456b2bb`。
该仓库原有五个未提交文件（键盘接口两个、官方 Go2 的 runner/flat 配置、velocity rewards），本次未修改或提交。
其状态与相关文件 SHA256 已记录于 `repository_state_at_evaluation.json`。本任务使用项目自己的 Payload task；外部 rewards 中已有更改可能改变 reward 数值，本文不使用 reward 作为性能指标。
没有使用或修改 `unitree_mujoco`。

当前仓库原有未提交内容保留：被跟踪的 `scripts/rsl_rl/__pycache__/cli_args.cpython-311.pyc`，以及四个未跟踪的 `dreamwaq_*framework` 图文件。
执行设置 `PYTHONDONTWRITEBYTECODE=1`；这些缓存/图文件不参与策略推理，不作为本次提交的一部分。
CSV、图片、运行日志位于已忽略的 `logs/`，按用户 Git 规则留作实验产物，不提交生成数据。

交付时当前仓库的 `git status --short`（均为任务开始前已有内容）：

```text
 M scripts/rsl_rl/__pycache__/cli_args.cpython-311.pyc
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.dot
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.png
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.svg
?? source/dreamwaq_go2/docs/dreamwaq_paper_framework.svg
```

本次完成的代码和报告全部独立提交，没有 push，也没有重写历史。没有需跨仓库联动回滚的代码提交。

验证结果：

- 现有 `test_payload_estimator.py` 的 5 项 CPU 合约测试全部通过，包括历史切片、梯度隔离、保存/加载及导出接口；其中 PPO 测试仅对随机初始化小张量模型执行，未更新现有 checkpoint。
- 新增 `test_payload_diagnostics.py` 的 3 项测试全部通过：真实 checkpoint 严格加载、正常路径逐元素一致、仅质量列改变且统一 /8、模型权重/观测未被修改、指标和失败计数定义正确。
- 2 环境×2 s 的 predicted/oracle 冒烟均通过；实际质量标签为 4 kg，初始物理/历史状态哈希相同。
- 21 个正式运行均完整生成 manifest；正常预测模式的原动作与冻结特征重算动作最大差为 **0**。
- 从三个负载、三个种子的真实 GPU 数据中选取起步、2 s、11.8 s，共 108 份观测在 CPU 重放，最大动作绝对差 **1.66893e-6**；结果存 `replay_validation.csv`。
- 所有闭环干预与对应自然轨迹的初始状态哈希一致；oracle 与 fixed_4 的 7200 行完整轨迹逐项相同，见 `oracle_fixed4_equivalence.csv`。
- 运行前后 checkpoint SHA256 相同；没有 checkpoint、训练日志、缓存或生成图表被纳入提交。
- Python AST/syntax、`git diff --check` 和 staged diff 检查通过。最终仓库状态只保留任务开始前的缓存与四个图文件；详见 `delivery_manifest.json`。
