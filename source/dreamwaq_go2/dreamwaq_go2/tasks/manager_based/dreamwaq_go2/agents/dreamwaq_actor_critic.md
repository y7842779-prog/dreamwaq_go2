# `dreamwaq_actor_critic.py` 说明文档

## 1. 文件作用

[`dreamwaq_actor_critic.py`](./dreamwaq_actor_critic.py) 实现了当前项目中 DreamWaQ 风格策略网络的骨架版本。

它的目标不是直接复现完整论文，而是先把下面这条推理链接到现有 IsaacLab + RSL-RL 框架里：

`policy obs -> 历史观测编码器 -> 速度估计/潜变量估计 -> actor -> action`

当前版本已经完成了：

- 将 CENet 风格的历史编码器接入 actor 前向。
- 将 actor 和 critic 保持在一个 RSL-RL 可识别的 `ActorCritic` 风格模块中。
- 保证导出 `policy.pt` 时，actor 内部已经包含 estimator/CENet 前向路径。

当前版本还没有完成：

- `Lest = MSE(v_hat, v_t)` 的训练损失接入。
- `LVAE = recon + KL` 的 VAE 损失接入。
- AdaBoot 自适应 bootstrapping。

因此，这个文件现在更准确的定位是：

**DreamWaQ 的结构接线版模型，而不是完整论文训练版模型。**

## 2. 文件中各个类的作用

### 2.1 `_InputSpec`

这是一个很小的兼容类，用来兼容 IsaacLab 的 ONNX 导出逻辑。

原因是 IsaacLab 的导出器会访问 `policy.actor[0].in_features` 这类属性。  
当前 actor 不是普通 `nn.Sequential`，因此这里用 `_InputSpec` 模拟了一个“第 0 层输入描述符”。

### 2.2 `DreamWaQActor`

这是当前文件最核心的类，负责：

- 从 `policy` 观测里拆出当前帧观测 `o_t`
- 从 `policy` 观测里拆出历史观测 `o_t^H`
- 用历史观测经过 encoder 估计：
  - `v_hat`：机体线速度估计
  - `z`：环境上下文潜变量
- 将 `[o_t, v_hat, z]` 拼接后送入 actor MLP，输出动作均值

它相当于把论文里的：

- Context-aided Estimator
- Policy Network(Actor)

两部分前向逻辑串在了一起。

### 2.3 `DreamWaQActorCritic`

这个类是给 RSL-RL 使用的总策略类，包含两部分：

- `self.actor = DreamWaQActor(...)`
- `self.critic = critic MLP`

它需要实现 RSL-RL 默认 PPO 所依赖的接口：

- `act`
- `act_inference`
- `evaluate`
- `get_actions_log_prob`
- `entropy`
- `action_mean`
- `action_std`

所以它本质上是一个“兼容 RSL-RL API 的 DreamWaQ 风格 actor-critic”。

## 3. 输入输出设计

## 3.1 actor 输入

当前 actor 不再直接吃单纯的 45 维当前本体观测，而是吃一个打包后的 `policy` 张量：

`[current_obs, history_obs]`

其中：

- `current_obs = o_t`
- `history_obs = o_t^H`

在当前环境配置下：

- `o_t` 是 45 维
- 历史长度 `H = 5`
- `o_t^H` 展平后是 `45 x 5 = 225` 维

所以 `policy` 总维度为：

`45 + 225 = 270`

## 3.2 actor 内部数据流

`DreamWaQActor.forward()` 的数据流如下：

1. 调用 `split_observations()` 拆出：
   - `current_obs`
   - `history_obs`
2. 调用 `estimate_context()`：
   - `history_obs -> encoder`
   - `encoder feature -> velocity_head -> velocity_estimate`
   - `encoder feature -> latent_mu_head -> latent_mu`
   - `encoder feature -> latent_logvar_head -> latent_logvar`
   - `latent_mu + latent_logvar -> reparameterization sampling -> z`
   - `z -> decoder -> reconstructed_next_obs`
3. 拼接 actor 输入：
   - `actor_input = [current_obs, velocity_estimate, latent_mu]`
4. 送入 `policy_head`
5. 输出动作均值

## 3.3 critic 输入

critic 不使用上述 estimator 结构，仍然保持普通 MLP。

critic 直接吃环境中的 `critic` 观测组，也就是 privileged observations。  
当前项目中 critic 输入通常包括：

- `base_lin_vel`
- `base_ang_vel`
- `projected_gravity`
- `velocity_commands`
- `joint_pos`
- `joint_vel`
- `actions`
- `disturbance_force`
- `height_scan`

## 4. 网络结构说明

当前默认结构大致对应如下：

### 4.1 estimator / CENet 部分

- encoder: `225 -> 128 -> 64`
- velocity head: `64 -> 3`
- latent mean head: `64 -> 16`
- latent logvar head: `64 -> 16`
- decoder: `16 -> 64 -> 128 -> 45`

说明：

- `3` 对应机体线速度估计 `v_hat`
- `16` 对应上下文潜变量 `z`
- decoder 当前重建的是本项目定义下的“下一时刻 partial observation”

### 4.2 actor 部分

actor 实际输入维度为：

`45 + 3 + 16 = 64`

然后进入策略头：

`64 -> 512 -> 256 -> 128 -> 12`

最终输出 12 维动作，对应 Go2 的 12 个关节目标动作。

### 4.3 critic 部分

critic 仍是标准 MLP：

`critic_obs_dim -> 512 -> 256 -> 128 -> 1`

## 5. 关键方法说明

### 5.1 `split_observations(observations)`

作用：

- 按固定顺序把 `policy` 张量切成当前帧和历史帧

假设环境侧 `policy` 观测顺序保持为：

`[o_t, o_t^H]`

那么这里的切分就是可靠的。

这也是为什么环境配置文件里必须显式控制 term 的排列顺序。

### 5.2 `estimate_context(observations)`

作用：

- 从历史观测中提取 estimator 所需的中间结果

返回内容包括：

- `current_obs`
- `velocity_estimate`
- `latent_mu`
- `latent_logvar`
- `reconstructed_next_obs`

这个函数是后续接辅助损失最重要的入口。

如果后面要加：

- 速度监督损失 `Lest`
- VAE 重建损失 `LVAE`

通常就会在训练过程中调用这个函数拿中间量。

### 5.3 `forward(observations)`

作用：

- 生成 actor 动作均值

它当前仍然使用 `latent_mu` 直接作为 actor 的潜变量输入。  
也就是说，控制分支保持确定性；reparameterization sampling 只用于 decoder 的重构分支。

### 5.4 `predict_estimator(observations)`

作用：

- 以字典形式暴露 estimator 的中间结果

这个接口是为后续 PPO/runner 改造预留的。  
后面如果要加 DreamWaQ 的辅助损失，最方便的做法就是在训练循环里调用它。

## 6. 与环境配置的对应关系

本文件强依赖环境侧的 `policy` 观测打包方式。

当前约定来自：

[`dreamwaq_env_cfg.py`](../dreamwaq_env_cfg.py)

其中 `policy` 组的顺序是：

1. 当前帧：
   - `base_ang_vel`
   - `projected_gravity`
   - `velocity_commands`
   - `joint_pos`
   - `joint_vel`
   - `actions`
2. 历史帧：
   - `history_base_ang_vel`
   - `history_projected_gravity`
   - `history_velocity_commands`
   - `history_joint_pos`
   - `history_joint_vel`
   - `history_actions`

因此本文件里的切分逻辑默认认为：

- 前 45 维是 `o_t`
- 后 225 维是 `o_t^H`

如果以后环境侧更改了 `policy` 组顺序或维度，这个文件也必须同步修改。

## 7. 与 RSL-RL 的对应关系

这个文件不是单独运行的，而是通过以下链路被 RSL-RL 调用：

1. 在 [`rsl_rl_ppo_cfg.py`](./rsl_rl_ppo_cfg.py) 中设置：
   - `policy.class_name = "DreamWaQActorCritic"`
2. 在训练/播放脚本中，把该类注册到 `OnPolicyRunner` 的作用域
3. `OnPolicyRunner` 通过 `eval(class_name)` 实例化模型

这也是为什么当前项目里除了本文件本身，还修改了：

- [`scripts/rsl_rl/train.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/train.py)
- [`scripts/rsl_rl/play.py`](/home/hyc/dreamwaq_go2/scripts/rsl_rl/play.py)

## 8. 为什么导出 `policy.pt` 时 CENet 不会丢

IsaacLab 默认导出策略时，会拷贝 `policy.actor`。

当前结构中：

- `self.actor` 不是单纯线性层
- 而是已经包含：
  - 历史切分
  - encoder
  - velocity head
  - latent head
  - policy head

因此导出的 `policy.pt` 里，CENet 风格前向链已经包含在 actor 中，不会在部署时丢掉 estimator。

## 9. 当前版本的限制

当前实现仍然有几个重要限制：

### 9.1 estimator 已开始训练，但还不是论文完整版

虽然网络里已经有：

- `velocity_head`
- `latent_mu_head`
- `latent_logvar_head`
- `decoder`

这些分支现在已经开始参与训练：

- `velocity_head` 受 `Lest` 监督
- `decoder` 受简化版 `Lrecon` 监督
- `latent_mu / latent_logvar` 受简化版 `KL loss` 约束

但它们还没有完全达到论文中的严格训练形式。

### 9.2 只在 decoder 分支接入了 VAE 采样

当前实现已经在 decoder 分支接入了 reparameterization sampling，  
但 actor 前向仍然直接使用 `latent_mu`，没有把采样后的 `z` 送进策略头。

因此它现在更接近一个“半接入版 VAE”。

### 9.3 decoder 还没有接入论文严格版重构目标

当前 `reconstructed_next_obs` 已经参与了简化版 `Lrecon`，  
但 target 还是 `current_obs`，还没有拿它和论文更严格的 `o_{t+1}` 做重建误差。

### 9.4 还没有 AdaBoot

论文中的自适应 bootstrapping 机制还没有接入。

## 10. 后续如何继续扩展

如果要继续向完整 DreamWaQ 靠近，推荐顺序如下：

### 第一步：先加 `Lest`

目标：

- 监督 `velocity_head`

做法：

- 在训练循环中拿到 `critic` 里的 `base_lin_vel`
- 调用 `predict_estimator()`
- 计算 `MSE(velocity_estimate, base_lin_vel)`

这是最容易先落地的一步。

### 第二步：再加 `LVAE`

目标：

- 训练 `decoder`
- 训练 `latent_mu / latent_logvar`

做法：

- 在 rollout storage 中额外保存下一时刻 `policy` 当前帧 `o_{t+1}`
- 计算：
  - reconstruction loss
  - KL divergence

### 第三步：最后再加 AdaBoot

目标：

- 在训练不同阶段自适应控制是否使用 estimator bootstrapping

这一步要改训练流程和调度逻辑，复杂度最高，建议放最后。

## 11. 一句话总结

`dreamwaq_actor_critic.py` 当前做的事情可以概括为：

**把 DreamWaQ 论文中的 CENet 推理路径嵌入到 RSL-RL 可训练、可导出的 actor-critic 结构中，为后续补 estimator loss 和 VAE loss 打好了模型接口基础。**
