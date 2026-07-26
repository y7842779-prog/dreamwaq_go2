# CENet 接入过程笔记

## 1. 目标

本项目的目标不是一步到位复现论文的完整 DreamWaQ，而是基于 IsaacLab 当前已有的 `manager_based locomotion` 框架，采用“先接线、再加 loss、再补训练细节”的方式，逐层把 CENet 接进现有 Go2 速度跟踪任务。

整体策略是：

1. 先保留 IsaacLab / RSL-RL 原有训练主干可以跑通。
2. 先把 `history -> encoder -> actor` 的前向链路接进去。
3. 再逐步补上 `Lest`、`Lrecon`、`KL`。
4. 最后再处理论文里更完整的部分，例如 AdaBoot 和更严格的训练细节。

因此，整个接入过程是一个“分阶段逼近论文”的过程，而不是一次性重写整个训练框架。

## 2. 接入前的基础状态

在接入 CENet 之前，项目已经具备：

- 基于 IsaacLab 的 Go2 rough terrain velocity 任务。
- 基于 RSL-RL 的标准 PPO 训练链路。
- actor 用 `policy` 观测，critic 用 `critic` 观测。
- 非对称 PPO 结构已经成立：actor 看部分观测，critic 看 privileged observations。

换句话说，CENet 接入之前，项目已经有了一个可运行的“非对称 PPO 底座”。

## 3. 新建和主要修改的文件

### 3.1 新建文件

- [`dreamwaq_env_cfg.py`](./dreamwaq_env_cfg.py)
  - 新建 DreamWaQ 专用环境配置。
  - 重新定义 `policy / estimator / critic` 三组观测。
- [`agents/dreamwaq_actor_critic.py`](./agents/dreamwaq_actor_critic.py)
  - 新建 CENet 风格 actor-critic。
  - 把历史编码器、速度头、潜变量头、decoder、actor 头接到一个可导出的策略里。
- [`agents/dreamwaq_ppo.py`](./agents/dreamwaq_ppo.py)
  - 新建自定义 PPO。
  - 在 PPO 主损失之外，加入 DreamWaQ 所需的辅助损失。
- [`agents/dreamwaq_actor_critic.md`](./agents/dreamwaq_actor_critic.md)
  - 单独说明 `dreamwaq_actor_critic.py` 的结构和作用。

### 3.2 主要修改文件

- [`__init__.py`](./__init__.py)
  - 注册新任务：
    - `Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-v0`
    - `Isaac-Velocity-Rough-Unitree-Go2-DreamWaQ-Play-v0`
- [`agents/rsl_rl_ppo_cfg.py`](./agents/rsl_rl_ppo_cfg.py)
  - 新增 DreamWaQ 专用 runner / algorithm 配置。
- [`rough_env_cfg.py`](./rough_env_cfg.py)
  - 修改 rough 环境的资产、奖励、事件、离线可用性等配置。
- [`mdp.py`](./mdp.py)
  - 新增 `applied_base_force()` / `applied_base_wrench()` 等辅助函数，用于 critic 的 privileged observations。
- [`scripts/rsl_rl/train.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/train.py)
  - 注入自定义 `DreamWaQActorCritic` 和 `DreamWaQPPO`。
- [`scripts/rsl_rl/play.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/play.py)
  - 同样注入自定义策略类，保证 play / 导出 `policy.pt` 一致。

## 4. 分阶段接入过程

## 4.1 第一阶段：新建 DreamWaQ 专用环境配置

第一步不是先改 PPO，而是先把环境侧观测盘子搭好。

在 [`dreamwaq_env_cfg.py`](./dreamwaq_env_cfg.py) 中，定义了三组关键观测：

- `policy`
  - 给 actor 使用。
  - 结构为 `current_obs + history_obs`。
- `estimator`
  - 给 CENet 输入使用。
  - 本质上是历史部分观测的单独描述。
- `critic`
  - 给 value network 使用。
  - 保留 privileged information。

这里最重要的设计是把 actor 输入重新打包成：

`policy = [o_t, o_t^H]`

其中：

- `o_t` 是当前 45 维部分观测。
- `o_t^H` 是长度为 5 的历史部分观测，展平后 225 维。

最终 actor 输入总维度变为：

`45 + 225 = 270`

这一阶段的关键收获是：

- 不需要另写新的 env class。
- 可以继续使用 IsaacLab 的 manager-based 任务接口。
- 只要控制好 `policy` 观测的拼接顺序，就能把历史直接塞进 actor。

## 4.2 第二阶段：先确认非对称 PPO 仍然成立

当时的一个关键疑问是：

`obs_groups = {"actor": ["policy"], "critic": ["critic"]}` 是否真的有意义？

后面确认到，在当前 IsaacLab 使用的 RSL-RL 封装里，真正决定 actor / critic 输入的，是环境中名为：

- `policy`
- `critic`

的观测组本身。

也就是说：

- actor 实际绑定 env 里的 `policy`
- critic 实际绑定 env 里的 `critic`

因此，CENet 接入并没有破坏非对称 PPO；相反，是在现有非对称 PPO 上继续扩展。

## 4.3 第三阶段：新建自定义 actor-critic，把 CENet 接到 actor 前面

在 [`agents/dreamwaq_actor_critic.py`](./agents/dreamwaq_actor_critic.py) 中，新建了 `DreamWaQActor` 和 `DreamWaQActorCritic`。

这一阶段的思路不是先做完整论文，而是先把前向“接线”跑通：

1. 从 `policy` 中切出：
   - `current_obs`
   - `history_obs`
2. `history_obs -> encoder`
3. `encoder -> velocity_head`
4. `encoder -> latent_mu_head`
5. `encoder -> latent_logvar_head`
6. `latent_mu` 与 `velocity_estimate` 拼接回 actor 输入
7. actor 输出 action

当前主干网络大致为：

- encoder: `225 -> 128 -> 64`
- velocity head: `64 -> 3`
- latent mu head: `64 -> 16`
- latent logvar head: `64 -> 16`
- decoder: `16 -> 64 -> 128 -> 45`
- actor head: `64 -> 512 -> 256 -> 128 -> 12`

这里有一个重要取舍：

- actor 当前走的是 `latent_mu` 主线
- 不是直接把采样后的 `z` 送进 actor

这是为了先保证训练稳定，把 stochastic latent 的随机性先限制在 decoder / reconstruction 分支中。

## 4.4 第四阶段：新建 DreamWaQ PPO，把辅助损失接进训练

只改 actor-critic 不够，因为论文里的 CENet 不只是推理结构，还要靠额外监督训练。

因此新建了 [`agents/dreamwaq_ppo.py`](./agents/dreamwaq_ppo.py)。

### 第一步：先加 `Lest`

最先加入的是速度监督损失：

`Lest = MSE(v_hat, v_t)`

其中：

- `v_hat` 来自 `velocity_head`
- `v_t` 来自 critic 观测中的 `base_lin_vel`

这是最容易接入、风险最低的一步，因为它不要求改 rollout storage。

### 第二步：先做简化版 `Lrecon`

为了快速验证 decoder 支路是否能学到东西，先用过一个过渡版：

`Lrecon = MSE(reconstructed_obs, current_obs)`

这个版本不是论文严格版，但它有两个好处：

- 不用改 storage
- 可以先确认 decoder、latent 分支确实被训练起来了

### 第三步：升级成严格版 `Lrecon(next_obs)`

后面进一步改成更接近论文的版本：

`Lrecon = MSE(reconstructed_next_obs, next_obs)`

这里遇到的核心问题是：

- 默认 PPO rollout 里没有单独存 `next policy obs`

因此，不能只改 loss，还必须扩展 storage。

解决方案是：

- 在 [`dreamwaq_ppo.py`](./agents/dreamwaq_ppo.py) 中新建 `DreamWaQRolloutStorage`
- 在 `Transition` 里增加 `next_observations`
- 在 `process_env_step()` 中从 `infos["observations"]["policy"]` 里取出下一时刻 actor 观测
- 在 `mini_batch_generator()` 中把 `next_obs_batch` 一起送进 `update()`

这样严格版 `Lrecon(next_obs)` 才真正成立。

### 第四步：加入 `KL loss`

为了让 latent 分支更接近 VAE 结构，又加入了：

`KL(q(z|h) || N(0, I))`

这里的实现依赖：

- `latent_mu`
- `latent_logvar`

并加入了数值稳定性保护：

- 在 `sample_latent()` 里对 `latent_logvar` 做 `clamp(-10, 10)`
- 在 `KL loss` 里也对 `latent_logvar` 做同样的 `clamp`

这样可以避免训练早期 `exp(logvar)` 数值爆炸。

## 4.5 第五阶段：让 RSL-RL 真正认识自定义类

只写好 `DreamWaQActorCritic` / `DreamWaQPPO` 还不够，训练一开始会直接报错：

`Training type not found for algorithm DreamWaQPPO`

原因是：

- `OnPolicyRunner` 默认并不认识用户自定义类

解决方式是在：

- [`scripts/rsl_rl/train.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/train.py)
- [`scripts/rsl_rl/play.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/play.py)

里，把自定义类显式注入到 `rsl_rl.runners.on_policy_runner` 命名空间中。

这是接入过程里一个非常关键的工程问题：不是算法写错，而是 runner 根本找不到这个类。

## 5. 接入过程中遇到的问题与解决

## 5.1 问题：`obs_groups` 看起来改了，但不确定是否真的生效

表现：

- 一开始担心 `obs_groups = {"actor": ["policy"], "critic": ["critic"]}` 只是配置摆设。

结论：

- 在当前 IsaacLab 封装里，真正关键的是 env 输出的 `policy` 和 `critic` 两个观测组。
- 因此只要环境配置端的 `policy / critic` 盘子定义正确，非对称 PPO 就是成立的。

## 5.2 问题：actor 用 `latent_mu` 还是采样后的 `z`

表现：

- 论文写的是 `pi(a_t | o_t, v_t, z_t)`，容易误以为 actor 必须立刻使用 sampled latent。

解决：

- 先让 actor 使用 `latent_mu`
- 让 sampled latent 先只用于 decoder reconstruction 分支

原因：

- 训练更稳
- 先把结构跑通
- 等 `Lrecon`、`KL` 稳定后再考虑完全随机 latent 驱动 actor

## 5.3 问题：严格版 `Lrecon(next_obs)` 无法直接实现

表现：

- 默认 mini-batch 里只有当前 `obs_batch`
- 没有直接的 `next_policy_obs`

解决：

- 自定义 `DreamWaQRolloutStorage`
- 在 rollout 阶段显式保存 `next_observations`

## 5.4 问题：终止帧的 reconstruction 不该计入 loss

表现：

- 对 episode 末帧直接做 `next_obs` reconstruction 会把无效目标也算进 loss

解决：

- 使用 `dones_batch` 构建 `reconstruction_valid_mask`
- 只统计非终止样本的 reconstruction error

同时还补了：

- `view(-1, 1)`，防止 mask 广播维度出错

## 5.5 问题：`latent_logvar.exp()` 可能数值爆炸

表现：

- KL 和重参数化采样阶段都依赖 `exp(logvar)`
- 如果 `logvar` 偶尔过大，会导致训练不稳定甚至崩溃

解决：

- 在采样分支和 KL 分支都对 `latent_logvar` 做 `clamp(-10, 10)`

## 5.6 问题：环境里持续 base 外力会把步态训矮

表现：

- 训练出的策略长期低重心、弯腿、保守
- 到 MuJoCo 里更容易失稳

排查结果：

- `base_external_force_torque` 是 `mode="reset"` 采样一次后整回合持续生效
- 且 `force_range` 会同时作用于 `Fx, Fy, Fz`
- 因此不仅有水平推力，还存在持续 `z` 向扰动

处理：

- 后面先在 [`dreamwaq_env_cfg.py`](./dreamwaq_env_cfg.py) 中把 `self.events.base_external_force_torque = None`
- 先验证这是不是造成“低趴步态”的重要因素

注意：

- 关掉 event 以后，critic 的 `disturbance_force` 维度不变，只是数值会退化成全 0

## 5.7 问题：本地资产和远程依赖影响离线可运行性

表现：

- 早期 rough 环境引用了远程 sky / material / 资产路径
- 离线模式下存在资源拉取风险

解决：

- 在 [`rough_env_cfg.py`](./rough_env_cfg.py) 中改成本地可用的视觉配置
- 同时切换到官方 IsaacLab 的 Go2 资产配置

## 5.8 问题：MuJoCo sim2sim 时策略一开始正常，后面逐渐发疯

表现：

- 开始时还能跟踪指令
- 走一会后 raw action 越来越大
- 之后力矩饱和，姿态崩掉

定位过程里确认了几个关键原因：

- 历史观测喂法最开始和训练不一致，存在“落后一拍”
- history flatten 顺序一开始不对，训练端是 term-major，不是 frame-major
- base 角速度在部署端一开始使用了世界系，而训练里用的是机体系
- 策略本身已经偏低重心、稳定裕度不大
- 到 MuJoCo 里后，闭环误差和 CENet history 叠加，导致 actor 输出发散

这说明：

- “低重心”解释了策略为什么保守
- “raw action 发散 + torque 饱和”解释了为什么它一开始还能走、后面才发疯

## 6. 当前代码相对于论文的完成度

## 6.1 已完成的部分

- 非对称 PPO 结构
- `policy = [current_obs, history_obs]` 的环境侧打包
- CENet 风格 encoder + velocity head + latent head
- actor 内部集成 CENet 前向
- `Lest`
- 严格版 `Lrecon(next_obs)`
- `KL loss`
- decoder 支路
- 自定义 rollout storage 支持 next observation
- train / play / export 对自定义策略类的兼容

## 6.2 尚未完全复现的部分

- AdaBoot
- 论文里更严格的整体训练配方
- 是否要让 actor 最终使用 sampled latent 而不是 `latent_mu`
- 更系统的课程学习 / 稳定站姿奖励设计

因此，当前版本更准确的定位是：

**基于 IsaacLab 的 DreamWaQ / CENet 渐进式实现版，而不是完整论文等价复现版。**

## 7. 当前代码的核心数据流

可以把当前版本概括成下面这条链：

1. 环境构造 `policy = [current_obs(45), history_obs(225)]`
2. actor 从 `policy` 中切出 `current_obs` 与 `history_obs`
3. `history_obs -> encoder`
4. `encoder -> velocity_estimate / latent_mu / latent_logvar`
5. `latent_sample -> decoder -> reconstructed_next_obs`
6. `actor_input = [current_obs, velocity_estimate, latent_mu]`
7. `actor_input -> policy_head -> action`
8. PPO 更新时，同时优化：
   - surrogate loss
   - value loss
   - entropy
   - `Lest`
   - `Lrecon(next_obs)`
   - `KL loss`

## 8. 一句话总结

这次 CENet 接入不是“只新建了一个网络文件”，而是沿着：

`环境观测定义 -> actor 前向结构 -> PPO 辅助损失 -> rollout storage -> train/play 兼容 -> sim2sim 验证`

这整条链，采用“先接线、再监督、再严格化”的方式，一层一层把 DreamWaQ/CENet 接进了 IsaacLab 的现有项目。

如果后续继续推进，最自然的下一步是：

1. 继续改善训练环境，让策略先学会更自然的站姿与步态。
2. 再决定是否补 AdaBoot。
3. 最后再做更稳定的 sim2sim / sim2real 验证。
