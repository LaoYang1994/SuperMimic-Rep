# Design

## Background
- BONES-SEED 全量（129,511 条，`beyondmimic_seq`，1×RTX PRO 6000，迭代 = 更新）：旧采样器的 `bones_seq_fast@B` 88k 次迭代失败 339、回报 0.934；**196k 次失败 7,997、回报 0.697**——训练发散。
- 训练日志：B 从约 14.5 万次迭代起 critic 系统性高估，3,992 个记录点中 331 个 critic 平均估值 > 10.2 或 critic loss > 1；最高估值 36.8、critic loss 9×10⁴、策略梯度范数 356。物理上限 G_max = 0.1/(1−0.99) = 10（每步奖励 ≤ max_step_reward = 0.1）。高估的 V_H 通过 J 的自举项把策略推向错误状态，训练回报 12 万次后从 0.973 降到 0.955，评测随之崩溃。
- 新采样器 A（`bones_seq_sampler@A`）到 11.2 万次估值中位数 9.26，仍正常，但在 7.8 万次（critic loss 194）和约 9 万次（非有限 critic loss）出现过尖峰。
  之后（实现期间发现）：12.27 万次起 critic 向**负方向**爆炸（估值均值 −42.5 → −6.9×10⁶，GAE 目标均值低至 −1.8×10¹³，critic loss 最高 6.4×10³⁰）；
  **15.17 万次 critic 参数变成 NaN**（`ckpt_0152000` 中 225,409 个参数有 224,769 个非有限；`ckpt_0148000` 完好），此后每次迭代的
  策略与 critic 更新都被非有限保护跳过——策略冻结（`ckpt_0152000` 与 `ckpt_0168000` 的策略逐位相同），世界模型照常训练。
  在 NaN 之前 critic 的输入 / 目标都是有限的（`critic_skipped_nonfinite` 为 0）：critic 更新只检查输入与目标，`clip_grad_norm_`
  的返回值被丢弃，loss 巨大时反传梯度溢出为 inf，裁剪把它变成 NaN 后写入参数（由现象推断，未单独复现）。
- allmimic（独立复现）报告同一现象：不截断时 6.6 h 后发散、critic loss > 1e10；把 critic 估值截到 ±max_r/(1−γ) 后稳定。
- 梯度裁剪：论文 §3.3 的 J 不除以 H，表 A.5 写 "global norm 1.0, then element-wise 0.1"。我们（沿用上游）`loss = −J/H`（H = 32），记录的 `policy_grad_norm` 中位数 0.03、p99 约 0.35 → 两道裁剪从不触发。这是与论文表 A.1–A.5 唯一确认的不一致；allmimic 也独立发现。
- Adam 对损失整体缩放近似不变（除 eps 外），所以去掉 /H 的**唯一实际效果**就是裁剪开始生效：梯度范数 > 1 的步被缩放、单个元素 > 0.1 的被截断，大尖峰（如 356）不再整步进入 Adam 的矩估计。

## Goal
- critic 更新在梯度非有限时跳过该步，NaN / Inf 不再写入 critic 参数（与策略更新的保护一致）。
- critic 估值不再超出物理可达范围，长训练（> 14.5 万次迭代）不发散。
- 策略梯度裁剪按论文表 A.5 实际作用在 ∇J 上。
- 在相同迭代下，BONES-SEED 全量失败数不高于 A（新采样器基线），且 14.5 万次之后不出现 B 式崩溃；回报不下降。

## Non-goals
- 回报采样器（按访问次数平均、聚类均衡、困难上限）——另开 change。
- critic 自身的训练（MSE、学习率、critic 裁剪 1.0、epoch / minibatch）。
- 学习率、H、γ、λ、网络、奖励、终止、观测口径、评测协议。
- allmimic 的其他技巧（无噪声数据份额、关节加权 WM 损失、残差动作、latent 128）。

## Current behavior
- `PolicyTrainer.update_on`（`src/supermimic/imagination/policy_trainer.py`）：`v_H = critic(...)` 原样进入 `policy_objective`（`src/supermimic/imagination/objective.py`）；`loss = −J / cfg.horizon`；`clip_and_step(opt, params, grad_clip_norm=1.0, grad_clip_value=0.1)`。
- critic 目标：`values = critic(...)` 原样进入 `gae_targets`，critic 用 MSE 拟合这些目标。
- 记录：`policy_grad_norm`（裁剪前的全局范数）、`critic_value_mean`、`critic_target_mean`、`critic_loss`。

## Proposed behavior
1. **critic 值域截断**：新配置 `PolicyTrainerCfg.critic_value_clip: bool = True`。G_max 由奖励配置给出（`TrackingRewardCfg.max_step_reward / (1 − γ)`，现为 10），Trainer 传入 PolicyTrainer。
   - 策略自举：`v_H ← clamp(v_H, −G_max, G_max)`，再进入 J（饱和时该项对 s_H 的梯度为 0，即不再朝"更高估值"方向推策略）。
   - GAE：`values ← clamp(values, −G_max, G_max)` 后再计算目标。目标本身由有界奖励与截断后的估值递推得到，不再单独截断。
   - critic 的网络输出和 MSE 拟合不变（不截预测值，以保留拟合梯度）。
2. **按论文缩放与裁剪**：新配置 `PolicyTrainerCfg.objective_divide_by_horizon: bool = False`；`False` 时 `loss = −J`（论文写法），`True` 保留旧行为以复现旧 run。裁剪阈值不变（1.0 / 0.1）。
3. **critic 非有限梯度保护**：critic 每个 minibatch 在全局范数裁剪后检查梯度范数，非有限则跳过该步（fused Adam 的
   `found_inf`，权重与矩估计不变），与"输入 / 目标非有限"的跳过条件取或。复用 `clip_and_step`（加可选参数 `extra_skip`），
   不增加 host 同步；多卡时梯度先 all-reduce，各 rank 得到相同的范数与决定。日志 `policy/critic_skipped_nonfinite_grad`
   为本次更新中被跳过的 minibatch 比例。正常（有限）情况下行为与之前逐位相同，因此 B 已在跑的 `bones_seq_clamp@B` 仍代表最终代码。
4. **日志**：新增 `policy/critic_vH_clamped_frac`（V_H 被截断的比例）、`policy/critic_values_clamped_frac`（GAE 用的估值被截断的比例）、`policy/grad_clipped`（本步全局范数 > grad_clip_norm 的 0/1，按 log interval 平均即为触发比例）。`policy_grad_norm` 含义不变（裁剪前），但量级变为原来的 H 倍——在 PROJECT.md 注明。

## Design details
- 两个开关独立，默认都开（新行为）。`--resume` 只恢复权重、优化器与采样器状态，配置来自当次命令行（checkpoint 里的 `config` 不会被读回；评测只读其中的 `policy` 段，不受影响）——因此**用新代码 resume 旧 run 会切到新行为**，在 PROJECT.md 注明；要复现旧 run 需显式 `--policy-trainer.critic-value-clip False --policy-trainer.objective-divide-by-horizon True`。
- 下界取 −G_max（与 allmimic 相同）：惩罚项（动作平滑、关节限位）使每步奖励可以为负，但实际远小于 G_max 量级；对称界只是防止异常值。
- `torch.clamp` 用张量上的标量界，不引入 host 同步；编译 / CUDA Graphs 的 `ImagRolloutStep` 不受影响（截断发生在 rollout 之外）。
- 多卡：截断是逐样本的，与梯度 all-reduce 无关；裁剪在 all-reduce 之后（现状），每个 rank 决策一致。
- **训练实验**：B 上用本 change 的代码跑 BONES-SEED 全量，`bones_seq_clamp@B`（`beyondmimic_seq`、与 A 相同的 320k 调度与配置、每 4,000 次存 checkpoint）；与 A 的 `bones_seq_sampler@A` 在相同迭代（44k、88k，以及之后 A、B 都有的点，重点看 14.5 万次之后）做全量评测对比，并用 `eval_groups.py` 分组。

## Edge cases
- V_H 为 NaN/Inf：clamp 不改变 NaN；现有的非有限跳过逻辑（`clip_and_step` 的 found_inf、critic 的 `skip`）照旧生效。
- 诊断时 NaN 与阈值比较恒为假：检查"估值是否越界"必须同时检查是否有限（A 的 NaN 曾因此漏检）。
- `use_critic=False`（无 critic 变体）：v_H 为 None，截断不适用；GAE 不运行。
- 去掉 /H 后梯度范数中位数约 1.0：约一半的步被全局范数缩放；元素截断 0.1 的作用比例记录在日志里观察，若几乎所有步都被元素截断，在 validation 里报告（不在本 change 调参）。
- 从旧 checkpoint resume：Adam 的二阶矩是按 /H 的梯度累积的，切换后前几百步有效步长会偏大；新实验从头训练，不受影响。

## Clarifications
- Q: 验证实验怎么安排？ → A: 两项一起，在 B 上跑一组全量训练，与 A（新采样器基线，同代码只差这两项）按相同迭代对比；不做单项消融。
- Q:（实现期间）critic 非有限梯度保护要不要加进本 change？ → A: 加进本 change，B 不重启（该保护只在梯度非有限时起作用）。
- Q:（实现期间）A 的策略从 15.17 万次起冻结，怎么处理？ → A: 停 A，评测 `ckpt_0148000`（最后一个 critic 完好的点）作为 A 的后期对照，之后 A 空闲。

## Assumptions
- 论文的"global norm 1.0, then element-wise 0.1"作用在未除以 H 的 ∇J 上（§3.3 的 J 定义没有 1/H）。
- G_max 用每步最大奖励 / (1 − γ) = 10，与采样器的 `g_max` 同一定义。
- allmimic 截的是"用作目标和自举时"的估值；它的实现细节（是否也截 critic 预测）未知，我们只截使用处。
- main 的训练代码与 A 的 worktree 相同（之后只有文档提交）——plan 里用 diff 确认。

## Contract impact
none（结果报告仍按迭代对齐；`policy_grad_norm` 量级变化写入 PROJECT.md，不是契约）

## Project impact
`docs/PROJECT.md`：策略更新的缩放（−J，不除以 H）与裁剪按论文生效、critic 值域截断、两个新开关与新日志键、resume 旧 run 的行为变化；与论文的差异列表里删去"除以 H 导致裁剪不生效"、加上"critic 值域截断（论文未写）"。

## Experience impact
`docs/experiences/training/`：Adam 下损失整体缩放只通过梯度裁剪起作用——"裁剪阈值按损失尺度检查是否真的触发"；critic 估值超过 max_r/(1−γ) 是发散前兆（症状、诊断日志键、修法）。

## Open questions
- 裁剪生效后，早期学习速度（按迭代）是否变慢——看 A/B 在 4k–44k 的训练回报曲线。
- 截断后是否仍有少量状态长期饱和在 G_max（`critic_values_clamped_frac` 持续 > 0）——如果是，critic 拟合本身有问题，另查。

## Verification（在 plan 中细化）
- 单元测试：V_H 超界时被截断且 J 对饱和样本的梯度为 0；GAE 输入估值被截断；`objective_divide_by_horizon` 两种取值下损失相差 H 倍；开关关闭时与旧实现逐位相同。
- 已有测试全部通过（B 机）。
- B 上短跑（约 1,000 次迭代）检查：`policy_grad_norm` 中位数约为旧值 × 32、`grad_clipped` 比例约一半、截断比例接近 0、无报错、速度（次/s）与 A 相当。
- 全量训练 + 相同迭代全量评测，与 A 对比失败数、回报、分组；特别检查 14.5 万次之后 critic 估值始终 ≤ 10、无崩溃。
