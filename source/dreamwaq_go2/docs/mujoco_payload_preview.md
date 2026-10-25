# 三个已训练模型的 MuJoCo C++ 手动预览

已准备目录：`/home/hyc/dreamwaq_go2/logs/mujoco_preview/2026-10-25_v2/`。
这是手动 sim-to-sim 预览，不替代 Isaac Lab 中物理参数配对的正式评测。
没有重新训练或修改原 checkpoint；未启动 DDS、控制器主循环或 MuJoCo 仿真。

原 `unitree_rl_lab` 控制器配置指向 2026-07-26 的旧模型。
本次为 Asym_PPO、DreamWaQ_CENet 和 PayloadEstimator 分别创建独立目录，复制现有控制器可执行文件，并生成其配置和 ONNX。
可执行文件必须是实际副本：现有程序通过 `/proc/self/exe` 查找配置，符号链接仍会读原目录。
未修改 `unitree_rl_lab` 或 `unitree_mujoco` 的代码、配置和模型 XML。

## 启动方法

终端 1，先用现有平地场景：

```bash
cd ~/unitree_mujoco/simulate/build
./unitree_mujoco -i 0 -n lo -r go2 -s scene_flat.xml
```

终端 2，只选择下面一个模型，不同时运行多个控制器。

Asym PPO：

```bash
cd ~/dreamwaq_go2/logs/mujoco_preview/2026-10-25_v2/Asym_PPO
./go2_ctrl --network lo
```

DreamWaQ-CENet：

```bash
cd ~/dreamwaq_go2/logs/mujoco_preview/2026-10-25_v2/DreamWaQ_CENet
./go2_ctrl --network lo
```

PayloadEstimator：

```bash
cd ~/dreamwaq_go2/logs/mujoco_preview/2026-10-25_v2/PayloadEstimator
./go2_ctrl --network lo
```

将键盘焦点放在终端 2：按 `f` 进入固定站立，等约 2–3 s，然后按 `v` 进入策略控制。
按住 `w/s` 前后运动（±0.6 m/s）、`a/d` 左右平移（±0.2 m/s）、`q/e` 原地转向（±0.3 rad/s）。松开方向键停止对应指令；`p` 切到被动阻尼。
现有键盘接口每次只读一个键，不支持同时按住前进与转向构成精确的组合命令。
按键不需要回车，控制器可执行文件不需要激活 Python 环境。

切换模型：在终端 2 按 Ctrl+C 停止原控制器，关闭并重新启动 MuJoCo，再启动另一个模型。
这样避免沿用上一模型造成的机器人姿态和位置。手动输入仍不构成严格配对实验。
终端显示异常时可在退出后执行 `stty sane` 恢复终端输入显示。

测试地形时，将终端 1 的 `-s scene_flat.xml` 改成 `-s scene_terrain.xml`，三个模型使用相同场景。
本次未增加附加负载，物理质量沿用现有 XML；不能将这组指令当成新增 4/8 kg 负载测试。

## 准备内容和验证

- 原模型：`logs/rsl_rl/payload_comparison/<方法>/<日期>/model_4999.pt`，严格加载对应 `params/agent.yaml`。
- 输入维度分别为 45、270、1170，输出均为 12；没有把旧 235 维 PPO 接口混入。
- 按训练观测函数、term-major 历史及首帧填充规则生成 C++ 配置；速度/质量估计留在原 Actor 内部。
- 关节顺序显式映射为 `[3,0,9,6,4,1,10,7,5,2,11,8]`；默认关节姿态、25/0.5 PD gains、0.25 action scale 和 50 Hz 策略周期取自保存配置。
- 手动命令范围是本次预览设置，不改训练范围或物理负载。
- 独立配置仅启用 Passive/FixStand/Velocity，不加载不相关的 Parkour/PIE 模型；CSV 输出写入各自预览目录。
- 不沿用旧部署模板的额外目标关节限幅或 ±100 原动作限幅；训练原动作 clip 为 null，MuJoCo 自身物理约束仍由其 XML 决定。
- 使用现有 C++ ObservationManager 和 OrtRunner 编译一个无 DDS 的检查程序，40 帧合成输入的历史拼接逐元素完全一致。
- C++ ONNX 相对原 PyTorch Actor 的最大动作差：Asym `1.19e-6`、CENet `8.34e-7`、Payload `8.79e-7`。
- 三个独立控制器副本均通过 `--help` 检查，该选项在 DDS 初始化前退出。
- 三个 checkpoint、原控制器二进制和原配置的 SHA256 前后相同。当前只验证导出/输入契约，没有验证新场景中的闭环行走表现。

完整元数据：预览目录下 `manifest.json`，另有 `cpp_observations.txt`、`cpp_actions.txt` 可检查。
早期失败的 `2026-10-25_v1` 目录保留，不用于启动。

## 再生成

必须使用新的输出目录，脚本拒绝覆盖既有目录。

```bash
cd ~/dreamwaq_go2
PYTHONDONTWRITEBYTECODE=1 \
PYTHONEXE=/home/hyc/miniconda3/envs/env_isaaclab/bin/python \
/home/hyc/isaac-sim/python.sh scripts/rsl_rl/prepare_mujoco_preview.py \
  --output-dir logs/mujoco_preview/reproduce_new
```

这个 Python 入口只借助已有 Python/ONNX 依赖，不启动 Isaac Sim 应用。
检查程序链接本机已有的 C++ ONNX Runtime，没有安装依赖，也没有重编译外部控制器。

## Git 记录

只修改 `/home/hyc/dreamwaq_go2`，分支 `feature/dreamwaq-cenet-skeleton`。
新增 `scripts/rsl_rl/prepare_mujoco_preview.py`（准备及 CPU 验证）和本文（手动启动方法与验证边界）。
生成的 ONNX、配置、二进制副本和验证文件位于已忽略的 `logs/`，不提交。
修改前基线：`736a4e7`，`chore: mark baseline before isolated MuJoCo preview preparation`。
实现提交及最终仓库状态记录在预览目录的 `delivery_manifest.json` 和交付回复。

`unitree_rl_lab` HEAD 为 `b0e50bd`，本轮前后均 clean；`unitree_mujoco` HEAD 为 `fcad09e`，已有多处修改，本轮未改。
不涉及外部仓库提交或联动回滚。没有 push。
主仓库原有一个缓存改动和四个未跟踪框架图文件保留，本轮不提交。
