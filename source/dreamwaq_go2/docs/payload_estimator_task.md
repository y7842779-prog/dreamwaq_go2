# DreamWaQ + 独立负载估计分支

在 `feature/dreamwaq-cenet-skeleton` 新增两个 RSL-RL 任务：

- 训练：`Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0`
- 回放：`Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-Play-v0`
- 网络：`DreamWaQPayloadActorCritic`
- 实际算法：`DreamWaQPayloadPPO`（入口适配为 runner 识别的 `PPO` 名称）
- 独立实验目录：`unitree_go2_dreamwaq_payload_estimator_curriculum`

原 `DreamWaQ-v0` / `Asym-v0`、其 Play 任务、网络和配置保留。
主对比默认从头训练，不加载旧 `model_4999.pt`；这保证训练预算不额外增加旧模型的训练量。

## 网络和监督

```text
当前观测 45 ────────────────────────────┐
25 帧历史 ──按每个观测项取最近 5 帧──原 CENet──vhat 3、mu 16──┤
25 帧历史 (1125)──MLP 64→32→1──预测附加质量 / 8───────────┤
                                                       ↓
                                  actor (65→512→256→128→12)
```

- 输入共 1170 维：当前 45 + 历史 25×45。保持 baseline 的观测项顺序：角速度、重力投影、速度指令、关节位置、关节速度、上一动作，维数依次 3/3/3/12/12/12。
- 历史沿用 Isaac Lab 的 **term-major** 排列：每一项内部按旧→新展开。不能简单截取整个 1125 维向量的最后 225 维给 CENet。
- CENet 仍接收 225 维，预测速度 3、mu 16、logvar 16，原单步 decoder 仍预测下一时刻当前观测 45。原速度、重构和 KL 损失不变。
- 新分支仅为 ELU MLP `1125→64→32→1`。输出是附加质量除以 8 的归一化估计；乘 8 得 kg，不强制截断到训练范围。
- actor 输入为 `[当前观测, vhat, mu, 预测质量/8]`，仍输出原顺序的 12 维关节位置动作。首层新增质量列初始为 0，其余初始化与同 seed 的原网络一致。
- 真实标签是 **PhysX 实际 base mass − 默认 base mass**，不是课程上限，不是整机总质量，也不是 COM。每次 episode reset 后更新。
- 标签暂存于 critic observation 的最后一列，与动作前的 observation 一起存入 rollout。critic 网络只接收原 238 维，显式剔除标签；actor 只读预测质量。不会把 reset 后新负载错配给上一帧样本。
- 新损失：`0.1 × Huber(predicted_mass/8, actual_added_mass/8)`。记录 `Loss/payload`、`Loss/payload_mae_kg`。
- 默认 `detach_payload=True`：PPO 不反向更新质量 MLP，该分支由真实质量监督；actor 通过 PPO 学习使用预测质量。原 CENet 仍按 baseline 更新。
- next observation 存储在新任务中仅保留 decoder 所需的 45 维，避免重复保存完整 25 帧历史；原任务的存储不变。
- 不含动作补偿分支；动作 scale、频率、奖励不变。

## 公平比较配置

新环境直接继承课程版 DreamWaQ，而 DreamWaQ 继承 Asym 环境。验证脚本对三个完整配置进行逐项比较：除观测差异以外，环境配置完全一致。

| 配置 | 三个任务共同设置 |
| --- | --- |
| 负载课程 | iteration <1500: 0–3 kg；1500–3499: 0–5 kg；≥3500: 0–8 kg |
| 采样时机 | episode reset，附加质量，重新计算惯量 |
| COM 第一阶段 | x ±0.02、y ±0.015、z ±0.005 m |
| COM 后两阶段 | x ±0.04、y ±0.025、z ±0.01 m |
| 课程计数 | `common_step_counter // 24` |
| 训练预算 | 4096 envs × 24 steps × 5000 iterations |
| seed | 默认 42；其他 seed 的重复实验须对三个任务一致设置 |
| PPO | 5 epochs、4 minibatches、初始 lr 0.001、gamma 0.99、lambda 0.95 |
| actor / critic 隐藏层 | 512 / 256 / 128 |
| 控制 | dt 0.005 s，decimation 4，50 Hz，action scale 0.25 |
| 其他 | 奖励、地形及地形课程、指令、终止条件、噪声、摩擦和执行器均继承 baseline |

新方法有意增加的是历史输入、负载监督和预测负载通道；不宣称三个方法参数量或监督信息相同。
Asym 仍运行标准 PPO；DreamWaQ 仍运行原 DreamWaQPPO。
新任务默认 `empirical_normalization=False`，保证质量标签保持 kg；不要单独开启 normalizer 后仍按原单位计算监督。

## 启动

在仓库根目录运行；主实验不要带 `--resume` 或旧 checkpoint：

```bash
cd /home/hyc/dreamwaq_go2
source /home/hyc/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export PYTHONPATH="/home/hyc/dreamwaq_go2/source/dreamwaq_go2:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

python -B scripts/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-v0 \
  --headless --device cuda:0 --num_envs 4096 --max_iterations 5000 --seed 42
```

两个 baseline 的训练使用同一命令，分别替换为 `Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-v0` 和 `Isaac-Velocity-Rough-Unitree-Go2-Asym-v0`。

回放必须指定新任务对应的 checkpoint，例如：

```bash
python -B scripts/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-PayloadEstimator-Play-v0 \
  --num_envs 16 --checkpoint /absolute/path/to/new_run/model_4999.pt
```

导出 actor 输入仍是 1170 维，包含历史；外部控制程序需要按上述顺序管理历史。
Play 沿用 baseline 的回放设置：关闭观测噪声等，**新建环境的负载课程从第一阶段开始，并不因加载 model_4999.pt 自动到 8 kg**。正式负载评测须统一显式设置测试负载及 COM 条件。
同样，已有 runner 的 checkpoint 不保存完整仿真状态/课程计数；保存恢复测试仅证明权重和 optimizer 可恢复，不能把默认 resume 当作完整课程续训。

消融可使用 Hydra 覆盖（追加在训练命令后）：

- `agent.policy.use_payload_input=False`：actor 的质量通道置零，保留监督分支，检验预测质量是否帮助控制。
- `agent.algorithm.payload_loss_coef=0.0`：关闭质量监督；在默认 detach 下该 MLP 不学习，属于无监督信号的对照。
- `agent.policy.detach_payload=False`：允许 PPO 梯度进入估计器，输出未必仍保持纯物理意义，属于独立消融。

25 帧历史在环境与网络两侧配套设置；不能只改网络 history 参数。

## 验证与产物

```bash
python -B scripts/rsl_rl/test_payload_estimator.py
python -B scripts/rsl_rl/validate_payload_task.py --headless --device cuda:0
```

本次实现已通过：

1. 5 项 CPU 测试：逐项历史切片、初始化映射、梯度/标签隔离、PPO 更新与 checkpoint、JIT/ONNX；包含与修改前提交 `753b8a4` 的原 DreamWaQPPO 对照，损失和更新权重逐元素一致。
2. 三任务实际配置核对；新训练/回放注册与实际算法分派正确。
3. Isaac Lab：16 envs、2 iterations×24 steps，保留 5 epochs / 4 minibatches；为节省启动资源，测试地形网格缩为 2×4，未改变正式默认配置。
4. 在课程 iteration 0/1500/3500 直接 reset，核对质量标签、COM 范围及历史填充；真实附加质量抽样范围分别为约 0.082–2.840、0.311–4.990、0.780–7.795 kg。
5. 新 checkpoint 保存/恢复、Isaac Lab JIT/ONNX exporter、导出策略与原策略输出一致性。
6. 原 DreamWaQ 和 Asym 的两个 `model_4999.pt` 均可 strict load，输出有限的 12 维动作。

`logs/payload_estimator_validation/` 保存实际配置、console.log、result.json、baseline_checkpoint_check.json、短测试 checkpoint 和导出模型，均不进入 Git。这些结果仅证明实现链路可用，不是负载预测精度或行走性能结论。没有执行 5000 次完整训练。

验证依赖本机 IsaacLab `/home/hyc/IsaacLab`，HEAD `46dff135f44683f031edf346e544fcfd8456b2bb`（detached）。它原有 5 个未提交文件：两个 keyboard、Go2 runner 配置、flat 配置、velocity/mdp/rewards.py；本次未修改或提交它们。rewards.py 中已有新增 helper 在本任务中由项目本地同名实现覆盖；三个任务使用相同的本机框架。
修改前主仓库标记为 `753b8a4`；实际实现提交见本文件的 Git 历史及任务完成报告。旧 checkpoint 只用于兼容性检查：

- DreamWaQ：`scripts/rsl_rl/logs/rsl_rl/unitree_go2_dreamwaq_cenet_payload_curriculum/2026-09-11_20-51-03/model_4999.pt`
- Asym：`scripts/rsl_rl/logs/rsl_rl/unitree_go2_rough_asym_payload_curriculum/2026-09-09_21-44-06/model_4999.pt`
