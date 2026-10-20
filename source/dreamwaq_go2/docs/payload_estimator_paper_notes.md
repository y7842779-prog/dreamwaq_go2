# PayloadEstimator：实际框架与论文表述

核对日期：2026-10-21。本文针对已经完成的具体实现，不按早期设想补画不存在的模块。

## 图与核对对象

- 任务：`Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0`。
- 模型：`DreamWaQPayloadActorCritic`；算法：`DreamWaQPayloadPPO`。
- 运行：`logs/rsl_rl/payload_comparison/PayloadEstimator/2026-10-16_21-12-27/`。
- checkpoint：该目录的 `model_4999.pt`，内部 `iter=4999`；TensorBoard 也记录至 4999。
- 图源：[payload_estimator_paper_framework.dot](payload_estimator_paper_framework.dot)。
- 论文矢量图：仓库根目录下 `logs/payload_paper_figures/framework.svg`、`framework.pdf`。
- 预览：`logs/payload_paper_figures/framework.png`。
- 蓝色代表沿用 DreamWaQ 的结构，绿色代表本项目新增路径；不代表蓝色部分被冻结，也不代表绿色部分在文献中具有首创性。

代码来源（相对于 `source/dreamwaq_go2/dreamwaq_go2/tasks/manager_based/dreamwaq_go2/`）：

| 文件 | 核对内容 |
| --- | --- |
| `agents/payload_actor_critic.py` | 历史切片、质量 MLP、Actor 拼接、detach、Critic 剔除标签 |
| `agents/dreamwaq_actor_critic.py` | CENet、μ/logvar、采样、decoder、Actor/Critic |
| `agents/payload_ppo.py` | 归一化质量 Huber、MAE、辅助损失权重 |
| `agents/dreamwaq_ppo.py` | PPO、速度、下一观测重构、KL 的共同优化 |
| `payload_env_cfg.py` | 25 帧历史、reset 后真实附加质量标签 |
| 该运行 `params/agent.yaml`、`params/env.yaml` | 实际保存的超参数和环境配置 |

读取三个最终 checkpoint 的张量元数据核对了图中的层尺寸。本次没有重新训练、运行仿真或修改训练代码。当前 `env_isaaclab` 的 PyTorch/NumPy/PyYAML 不可导入，因此没有声称执行 checkpoint 前向或 strict load；元数据通过白名单反序列化读取，日志通过现有 TensorBoard protobuf 解析，配置通过系统 Python 的 YAML 读取。没有安装依赖或改动该环境。

## 实际前向计算

单帧观测为 45 维：角速度 3、重力投影 3、速度指令 3、关节位置 12、关节速度 12、上一动作 12。因此“本体感知历史”也包含指令与动作历史，不应写成只有传感器测量。

令 $H_t^{25}$ 表示环境提供的 25 帧历史，$H_t^5$ 表示从每个观测项取出的最近 5 帧。实际存储是 term-major，每项内部旧到新；不能简单取整个 1125 维向量的末尾 225 维。本文不额外假设历史首尾与当前观测的时间偏移。

\[
(\hat v_t,\mu_t,\ell_t)=E_\psi(H_t^5),
\qquad \hat q_t=f_\eta(H_t^{25}),
\qquad \hat m_t=(8\,\mathrm{kg})\hat q_t.
\]

\[
\bar a_t=\pi_\theta\!\left([o_t,\hat v_t,\mu_t,\operatorname{sg}(\hat q_t)]\right).
\]

这里 $\bar a_t$ 是动作均值，训练从对应的 12 维高斯分布采样；部署使用动作均值。动作后续仍由原有 action scale 和关节位置控制处理。

| 组件 | 实际尺寸 / 行为 |
| --- | --- |
| Actor 外部观测缓冲区 | 当前 45 + 历史 25×45 = 1170 |
| CENet encoder | 225 → 128 → 64，隐藏层 ELU |
| CENet 三个线性头 | 64→3 速度、64→16 μ、64→16 logvar |
| VAE 训练分支 | μ/logvar 重参数采样 z；decoder 16→64→128→45 |
| Payload MLP | 1125→64→32→1，隐藏层 ELU，末层线性 |
| 质量输出 | 网络直接输出归一化质量；乘 8 kg 得附加质量，无硬截断 |
| Actor 拼接 | 45 + 3 + 16 + 1 = 65；使用 μ，不使用 VAE 采样 z |
| Actor head | 65→512→256→128→12 |
| Critic 网络 | 238→512→256→128→1 |
| Critic 数据缓冲区 | 239 维；最后 1 维质量标签在进入 Critic 前剔除 |

新分支有 74,177 个参数，Actor 首层额外增加 512 个参数，合计比原 DreamWaQ 多 74,689 个参数。可以据此描述结构规模；没有测量推理延时，不能把参数少直接写成实机速度提升。

两个分支都以 50 Hz 更新，5 帧和 25 帧约对应 0.1 s / 0.5 s 的窗口。它们是不同历史长度，不是两个更新频率，也没有实现慢变量约束、循环网络或严格的动力学解耦。

## 损失与梯度

真实标签是 $m_t=m_{\mathrm{base,actual}}-m_{\mathrm{base,default}}$，由 PhysX 实际质量读取。它不是整机质量、课程上限或 COM。当前负载在 episode reset 时更新，标签随当前动作前的 observation 保存。

\[
\mathcal L_m=\operatorname{Huber}\left(\hat q_t,\frac{m_t}{8\,\mathrm{kg}}\right).
\]

\[
\mathcal L=\mathcal L_{\mathrm{surrogate}}+\mathcal L_V-0.01\mathcal H
+0.1\mathcal L_v+0.1\mathcal L_{\mathrm{rec}}+10^{-4}\mathcal L_{\mathrm{KL}}
+0.1\mathcal L_m.
\]

其中 surrogate 是代码中要最小化的 PPO 策略损失；速度监督使用 MSE，重构预测下一时刻的当前观测 45 维，并排除 done 后的无效跨回合目标。VAE 在采样及 KL 中对 logvar 做 [-10,10] 数值截断。

| 损失 | 直接接收该损失梯度的参数 |
| --- | --- |
| PPO surrogate | Actor head、CENet encoder、速度头、μ 头、动作 std |
| Value loss | Critic |
| Entropy | 状态无关的动作 std |
| Velocity MSE | CENet encoder、速度头 |
| Reconstruction MSE | CENet encoder、μ/logvar 头、decoder |
| KL | CENet encoder、μ/logvar 头 |
| Payload Huber | 独立 Payload MLP |

`detach_payload=True` 使 PPO 的 Actor 路径不向质量估计器回传梯度，Actor 仍可学习质量输入的权重。真实质量既不直接输入 Actor，也不输入 Critic 网络；它只用于监督。训练和部署的 Actor 都接收预测值，从结构上避免把真实质量输入策略训练、部署时才切换为估计值的差异。

这些网络仍使用同一个 optimizer 和全局梯度裁剪。正确说法是“对质量通道实施策略梯度隔离”；不应声称完全独立优化、已经消除所有任务干扰或 CENet 被冻结。

## 论文主线与贡献表述

建议中文方法名：**显式负载条件化的上下文感知四足运动控制**。

建议英文题目草案：**Explicit Payload Conditioning for Context-Aware Quadrupedal Locomotion**。

核心研究问题：在已有短历史隐式上下文估计的基础上，增加具有物理监督的低维附加质量条件，能否帮助负载变化下的运动控制？

动机应写为“原有速度与重构目标不直接约束隐变量的质量可读性”，而不是“DreamWaQ 的 μ 不含动力学信息”。之前的负载 probe 也不能证明 μ 不含任何动力学信息。

可以直接用于方法介绍的段落：

> 针对附加载荷变化下的本体感知运动控制，我们在 DreamWaQ 的上下文估计框架上引入显式负载条件。原有 CENet 从短历史中提取速度估计与隐式上下文；并行的轻量 MLP 从较长观测历史中回归归一化附加质量，并将预测值与原有特征共同输入运动策略。训练阶段利用仿真中的实际附加质量监督该估计器，并在质量通道进入策略前停止梯度，使其直接优化物理估计目标；策略在训练和部署时均使用预测质量。该设计保留原有单策略控制结构和 CENet 目标，在统一负载课程下研究显式质量条件对运动控制的作用。

建议将设计贡献集中为两项，不把每个常见算子拆成一个“创新点”：

1. **面向负载的显式—隐式互补条件。** 在保留短历史速度/上下文估计的同时，引入由较长历史监督回归的单标量质量条件，形成质量变化可解释的策略输入。贡献是面向该问题的具体结构选择；MLP、拼接和不同窗口本身不是新算法。
2. **物理估计与策略利用之间的梯度边界。** 质量估计器由实际质量监督，控制策略通过 PPO 学习利用预测质量；通过 stop-gradient 避免策略目标直接改变质量回归分支的语义。stop-gradient 本身是常见方法，此处是设计依据，不单独宣称首创。

“在统一负载/COM课程、奖励及训练预算下进行比较”可作为实验贡献的组织方式，但课程是三组共用条件，不能归为新方法独有的创新。当前只有单 seed 训练日志，不能提前写“充分验证显著鲁棒性提升”。

## 三个对比实验应如何定位

| 比较维度 | Asym PPO | DreamWaQ + CENet | PayloadEstimator |
| --- | --- | --- | --- |
| 当前观测 | 45 | 45 | 45 |
| 历史表示 | 无显式历史分支 | 5 帧 CENet | 5 帧 CENet + 25 帧质量 MLP |
| 估计输出用于 Actor | 无 | 速度 3 + μ 16 | 速度 3 + μ 16 + 归一化预测质量 1 |
| Actor head 输入 | 45 | 64 | 65 |
| 真实质量监督 | 无 | 无 | 有，仅监督质量分支 |
| Critic 网络输入 | 238 | 238 | 238 |
| 主要比较问题 | 整体方法相对当前观测策略的表现 | 直接考察新增负载路径的整体增益 | 待研究方法 |

Asym PPO 不是“没有负载训练”的弱 baseline；DreamWaQ 也不是“完全不会适应负载”。它们与新方法均在相同负载随机化课程下训练。论文应明确这是本项目的配置对齐实现，不把本项目复现结果冒充原作者的官方结果。

与 DreamWaQ 相比，变化同时包含较长历史、额外参数和质量监督。因此三组主实验可以比较完整方法，不能仅据此把性能变化全部因果归因于“显式质量语义”。如以后只补一个最直接的消融，可优先用已有 `use_payload_input=False` 开关并重新训练，保留质量估计与监督但将 Actor 的质量通道置零。它检验质量通道是否有用，仍不单独证明 25 帧优于 5 帧。本文不启动额外实验。

## 保存配置与训练日志已经支持的事实

三个运行：

| 方法 | 运行目录（相对于 `logs/rsl_rl/payload_comparison/`） | 最终 checkpoint |
| --- | --- | --- |
| Asym PPO | `Asym_PPO/2026-09-09_21-44-06/` | `model_4999.pt` |
| DreamWaQ + CENet | `DreamWaQ_CENet/2026-09-11_20-51-03/` | `model_4999.pt` |
| PayloadEstimator | `PayloadEstimator/2026-10-16_21-12-27/` | `model_4999.pt` |

对三个实际保存的 `params/env.yaml` 作结构化比较：**除 observations 外没有差异**。三个保存的 agent 配置均为 seed 42、24 steps/iteration、5000 iterations、`resume=false`。环境数均为 4096。负载课程为 0–3 / 0–5 / 0–8 kg，分界为 iteration 1500、3500，末期日志均为阶段 3、质量上限 8 kg。奖励、COM 与地形课程、控制频率及动作 scale 保持相同。参数量和额外质量监督不同，应明确披露。

以下数据是 TensorBoard 中 step 4500–4999 的 **500 个 iteration 标量的算术平均**。排除 `/time` 后缀曲线；不是独立测试集、不是多 seed 均值、不是 checkpoint 4999 的重新评测。

| 训练指标 | Asym PPO | DreamWaQ + CENet | PayloadEstimator |
| --- | ---: | ---: | ---: |
| `Train/mean_reward` ↑ | 12.696490 | 14.189305 | 14.731007 |
| `Metrics/base_velocity/error_vel_xy` ↓ | 0.527702 | 0.478416 | 0.483861 |
| `Metrics/base_velocity/error_vel_yaw` ↓ | 0.428088 | 0.407452 | 0.403762 |
| `Train/mean_episode_length` ↑ | 980.310541 | 980.583321 | 984.968680 |
| `Curriculum/terrain_levels` | 5.321052 | 5.438238 | 5.431507 |
| `Loss/payload_mae_kg` ↓ | 不适用 | 不适用 | 0.501080 |

新方法相对 DreamWaQ 的平均训练 reward 约增加 3.82%，但平面速度误差约增加 1.14%，偏航误差约降低 0.91%。相对 Asym 的平均 reward 约增加 16.02%。没有置信区间，不使用“统计显著”。不应只展示奖励上涨而略去速度误差的相反变化。

`Loss/payload_mae_kg` 来自 PPO 更新所用训练 minibatch 的均值，不能称为未见负载预测精度。训练地形由策略表现驱动，三组虽课程规则相同，实际访问分布也不完全一样。`time_out` 计数不能直接充当跌倒率。

可直接使用的当前结果描述：

> 在相同环境配置与训练预算下，所提方法的训练末期平均回报高于两个对照方法，质量估计分支的训练 MAE 约为 0.50 kg。与 DreamWaQ 相比，偏航跟踪误差略低，平面速度跟踪误差略高。这些训练结果体现了显式质量分支的可学习性，但尚不支持各项控制指标全面改善的结论。

## 与已有工作的重合及差异

以下是针对这张网络图的定向核对，不是穷尽性新颖性检索。只引用原论文或作者项目页。

| 工作 | 已有思想 | 本方法可讨论的具体差异 |
| --- | --- | --- |
| [DreamWaQ, 2023](https://arxiv.org/abs/2301.10602) | CENet 联合速度估计、隐式上下文和 VAE 训练，服务于本体感知策略 | 原结构保留，新增直接监督的附加质量条件；不能将 CENet 或 VAE 归为本方法创新 |
| [HIMLoco, ICLR 2024](https://junfenglong.com/HIMLoco/) | 显式速度与隐式系统响应表征结合，使用对比学习 | 此处显式变量是附加质量，保留原 CENet 重构目标；“显式+隐式”这一泛化概念已有先例 |
| [RMA, RSS 2021](https://www.roboticsproceedings.org/rss17/p011.pdf) | 历史驱动适应模块与控制策略结合 | 历史推断并条件化控制不是首次提出 |
| [MULE, RA-L 11(1), 2026](https://ieeexplore.ieee.org/document/11242040/) / [公开预印本](https://arxiv.org/abs/2505.00488) | 名义策略加输出纠正动作的适应策略，处理负载变化 | 当前方法使用显式质量条件和一个动作策略，没有残差动作分支；不能据结构差异直接宣称性能优于 MULE |
| [Rapid Embodiment Adaptation, 2026 预印本](https://arxiv.org/abs/2608.01506) / [作者项目](https://embodiment-adaptation.github.io/) | 从历史识别显式物理参数，包括躯干质量，再用于条件化控制；先用真实参数训练基础策略，再训练适应模块 | 当前实现在已有 CENet 上并行增加质量分支，并在 PPO 训练期间始终向 Actor 提供预测质量；这是具体实现差异，不能写成显式质量适应的首创 |

最后一项于 2026-08-02 首次提交，早于本文核对日期。它是与本方法概念非常接近的工作；作者页面列为预印本，本文不将它称为已录用顶会/顶刊。

当前最稳妥定位是 **有物理解释的、面向未知附加质量的结构改进**。普通 MLP、单标量拼接、stop-gradient 和较长历史不能各算一项独立算法突破。是否达到目标期刊/会议的新颖性要求，不能仅凭这张图保证。

当前代码也不足以支持以下说法：完整动力学辨识、质量与 COM 严格解耦、摩擦/执行器故障辨识、任意负载下泛化、episode 中突变负载的快速恢复、实机零样本部署、已证明无性能损失。训练仅在 reset 时改变负载；质量输出仅覆盖附加 base mass。

## 英文图注草案

> Overview of the implemented payload-conditioned locomotion framework. A five-frame CENet provides a velocity estimate and an implicit context mean, while an additional MLP predicts normalized added mass from a 25-frame observation history. The policy receives the current observation, CENet outputs, and the stop-gradient mass prediction. Simulator mass is used only as a supervised target and is excluded from both actor and value-network inputs. The inherited CENet objectives and asymmetric PPO are retained. Blue blocks denote inherited structure; green blocks denote additions in this implementation.

## 导出与本次变更记录

```bash
cd /home/hyc/dreamwaq_go2
mkdir -p logs/payload_paper_figures
dot -Tsvg source/dreamwaq_go2/docs/payload_estimator_paper_framework.dot -o logs/payload_paper_figures/framework.svg
dot -Tpdf source/dreamwaq_go2/docs/payload_estimator_paper_framework.dot -o logs/payload_paper_figures/framework.pdf
dot -Tpng -Gdpi=110 source/dreamwaq_go2/docs/payload_estimator_paper_framework.dot -o logs/payload_paper_figures/framework.png
```

- 仓库：`/home/hyc/dreamwaq_go2`；分支：`feature/dreamwaq-cenet-skeleton`。
- 本次只新增这份文档与 DOT 图源，不修改项目网络、配置、checkpoint、日志数据或外部仓库。
- 修改前源码 HEAD：`f4f4fb8`；空提交检查点：`271ad12`，`chore: mark trained payload model before paper documentation`。
- 最终图源/说明提交可用 `git log -1 --format='%H %s' -- source/dreamwaq_go2/docs/payload_estimator_paper_notes.md` 查询。
- 图中的层尺寸同时对照源码、实际保存配置和 checkpoint 元数据；SVG/PDF/PNG 已渲染并检查 PNG 布局。
- 日志与导出的图片保留在忽略目录，不提交训练产物；可由 DOT 重建图片。
- 本次核对不更改实验状态。训练 git 快照中没有精确 HEAD hash，不能只凭该快照重建训练时全部依赖的精确版本。

开始任务时已有、结束时保留的 `git status --short` 项如下，不纳入本次提交：

```text
 M scripts/rsl_rl/__pycache__/cli_args.cpython-311.pyc
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.dot
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.png
?? source/dreamwaq_go2/docs/dreamwaq_algorithm_framework.svg
?? source/dreamwaq_go2/docs/dreamwaq_paper_framework.svg
```

它们是已有 pycache 和图文件；本次没有新增影响实验运行的未提交代码，也不把现有工作区描述为完全干净。
