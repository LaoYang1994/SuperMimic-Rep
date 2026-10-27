# Design

## Background
- **critic loss 的离谱量级来自"想象中已经摔倒"的样本。**
  - A（`bones_seq_sampler@A`）的 critic loss 在这些点上出现异常：94,750 次为 inf，122,650 次 8.4×10¹³，145,900 次 6.4×10³⁰，149,100 次 6.0×10²³。到 151.7k，critic 参数变成了 NaN。
  - critic 的物理值域只有 ±G_max = ±10。
  - 每个尖峰点上，critic 目标的均值都等于 `imag_r/action_rate_l2` 的均值：−1.9×10¹⁷、−2.2×10⁵、−5.0×10¹¹、−1.8×10¹³。同时 `action_abs_mean` 从 1.0 跳到 9.8×10⁵、1.7、1,029、5,534。
- **这些样本的 ĉ 已经是 0。** 同一批点上，想象回报 Σ_t ĉ_t r_{t+1} 保持在 2.93，策略梯度范数也正常（0.03）。奖励均值到了 −10¹³ 而 Σĉr 不受影响，说明这些样本的 ĉ 已为 0。
- **机制**：
  1. 世界模型判定摔倒（ĉ → 0）之后，rollout 仍然继续跑；
  2. 策略输出没有界，世界模型又在外推，状态和动作互相放大；
  3. action_rate 惩罚是 Δa 的平方、没有下界，单步奖励到 −10¹⁷；
  4. 策略目标按 ĉ 加权，所以不受影响；
  5. 但 critic 损失按论文 Eq. 3 实现，L_V = Σ_t (V_t − R̂_t)²，不加权。它把这些样本也拟合进去：输入是失控的状态，critic 对它们的估值到 −10⁸；目标到 −10¹⁷。
  6. 每次 critic 更新的 20 个小批量 Adam 步都被这几个样本主导；数值溢出后 NaN 被写进参数。
- **这些样本的估值本来就用不上。** 在 GAE 里它们要乘 (1 − d̂)，在 J 里 V_H 要乘 ĉ_H，两者都约等于 0。
- **活动 change 覆盖不到。** 它只截断估值（自举项）；R̂ 里的 r̂，以及 critic 对垃圾输入的预测，仍然没有界。B 到 44k 为止没有发生这类事件；A 第一次严重事件在 94,750 次。
- **可能也影响了策略。** A 在 120k 到 148k 之间出现了站立漂移：body_check 回报从 0.970 降到 0.868。这和 122,650 次、145,900 次两次事件在时间上重合；144k、124k 的评测正在区分两者的关系。

## Goal
- 长训练中 critic loss 不再出现数量级上的尖峰，不出现 NaN；想象中已摔倒的样本不影响 critic。
- 在相同迭代下，BONES-SEED 全量评测的失败数和回报不差于 A 和 B；148k 时没有站立漂移。

## Non-goals
- **策略目标和想象 rollout 本身。** 不在摔倒后停止 rollout，不限制动作范围，不改 J 的写法。J 已经按 ĉ 加权，0·inf 产生的 NaN 由现有的非有限跳过处理。
- critic 损失的形式（MSE 不变，不换 Huber）、学习率、epoch / minibatch，以及世界模型的终止头、采样器、奖励定义。
- 旧 B 的系统性高估（估值 15–37）由活动 change 的估值截断处理。

## Current behavior
`PolicyTrainer.update_on`（`src/supermimic/imagination/policy_trainer.py`）的流程：
1. 对全部 (t, n) 计算 `values = critic(...)`，截到 ±G_max；
2. `gae_targets(rewards, dhat, values)`（`objective.py`）对全部样本递推，奖励不截断；
3. critic 损失是全部 H×n 个样本上的 MSE；
4. 跳过条件：输入或目标里有任何非有限值，整次 critic 更新都跳过（`c417edd` 起，每个小批量还会检查梯度范数）。

日志在 `training/loop.py` 中按 log interval（50 次迭代）求和再取平均。因此每个记录点是 50 次迭代的平均，而每次迭代本身又是全部样本的均值。
事件不会被漏掉，但尾部会被稀释：单个样本的 −10¹⁷ 被平均进约 50×32×4096 个样本，只能看到量级，看不出是多少个样本、多大的单值。

## Proposed behavior
1. **定义"已摔倒"样本**：ĉ_t < ε。新配置 `PolicyTrainerCfg.critic_min_survival: float = 0.001`，设为 0 时回到论文写法，即全部样本都算存活。
2. **GAE**：已摔倒样本按"已终止"处理：r̂_{t+1} = 0、V_t = 0、d̂_t = 1（用替换实现，不用乘法；否则 d̂ 为 NaN 时，(1 − d̂)·0 仍是 NaN）。R̂_t 在递推的每一步截到 [−G_max, G_max]，作为兜底：防止世界模型没判定终止、状态却已失控的情况。这一步由现有开关 `critic_value_clip` 控制。
3. **critic 损失**：只在存活样本上计算 MSE。每个小批量取存活样本的均值；小批量里一个存活样本都没有时，跳过这一步。
4. **新日志**：
   - `policy/imag_dead_frac`：ĉ_t < ε 的样本比例；
   - `policy/imag_action_abs_max`：本次 rollout 里最大的 \|a\|；
   - `policy/imag_reward_min`：最小的 r̂，统计全部样本，用来看事件是否仍在发生；
   - `policy/critic_targets_clamped_frac`：存活样本里 R̂ 触顶的比例，大于 0 说明世界模型没判定终止、状态却已失控；
   - `policy/critic_loss_max`：本次更新 20 个小批量里最大的 loss。`policy/critic_loss` 的含义不变，仍是最后一个小批量的 loss。
5. **日志聚合**：以 `_max` 结尾的键，在 log interval 内取最大；以 `_min` 结尾的取最小；其余键照旧取平均。出现 NaN 时结果为 NaN。
   目前没有已有的键以 `_max` 或 `_min` 结尾，所以已有日志的含义不变。

## Design details
- **依赖与重叠**：本 change 用到 change 2026-10-26-critic-clamp-paper-grad-clip 的 `clamp_values`、`g_max` 和 critic 梯度保护，
  两者都改 `policy_trainer.py`、`objective.py`、`tests/test_policy_clamp.py`、`docs/PROJECT.md`。分支从
  `change/2026-10-26-critic-clamp-paper-grad-clip`（`c417edd`）拉出，合并排在它之后；它的验证（B 的 44k、88k、148k）照常进行。
- **为什么按 0 处理，而不是"按 ĉ 加权"：**
  - 加权挡不住平方误差。比如 ĉ = 10⁻¹² 时，ĉ·(V − R̂)² 仍可能是 10¹⁴。
  - 设为 0 之后，存活样本的目标只由存活段的奖励和截断后的估值递推而来，不会被摔倒后的部分泄漏进去。
  - 只有"下一步就已摔倒"的存活样本 k 的目标与论文写法不同：它少了 γ(1 − d̂_{k+1})[(1−λ)V_{k+1} + λR̂_{k+1}] 这一项，
    其上界是 (ε / ĉ_k)·G_max（因为 1 − d̂_{k+1} = ĉ_{k+1}/ĉ_k < ε/ĉ_k）。ĉ_k = 1 时，ε = 0.001 对应的偏差不超过 0.01；其余存活样本与论文写法逐位相同。
    观察到的失控样本 ĉ 实际为 0（Σĉr 前 4 位不变，而 r ~ −10¹³，说明 ĉ < 10⁻¹³），所以更小的 ε 拦截效果相同、偏差更小。
- **非有限值的处理**（见经验 `torch.where does not stop NaN gradients`）：
  - 已摔倒样本的输入可能是 inf 或 NaN，所以在 critic 前向之前，先把它们的输入替换成有限的占位值（0），再用 0 权重排除。只屏蔽输出会出现 0·NaN = NaN。
  - 非有限检查只看存活样本；替换后，全部输入都是有限值。
- **形状保持固定**：用权重屏蔽，不用布尔索引。这样没有 host 同步，小批量切分方式不变。
- **多卡**：每个 rank 在本地按存活样本求均值，再对梯度做 all-reduce，和现在一样。各 rank 的存活数不同带来的权重差异可以忽略（目前是单卡）。
- **`use_critic=False`**：屏蔽和目标截断都不适用；rollout 的尾部统计照常记录。
- **验证实验（A）**：
  - 用本 change 的代码在 A 上从头跑一次 BONES-SEED 全量，run 名 `bones_seq_critic_alive@A`。代码包含活动 change 的全部内容，再加本 change。启动参数和 A、B 相同（`beyondmimic_seq`，320k 调度，每 4,000 次存一次 checkpoint）。
  - 和 A（`bones_seq_sampler@A`）、B（`bones_seq_clamp@B`）在相同迭代下对比：44k、88k、148k 做全量评测，再用 `eval_groups.py` 分组。
  - 训练侧要检查：
    - `critic_loss_max` 全程保持在 1 以下，不出现 NaN 或跳过；
    - 越过 A 的事件点（94,750、122,650、145,900、149,100 次）时，`imag_dead_frac`、`imag_reward_min` 仍然会显示事件，但 critic 不受影响；
    - 148k 时没有站立漂移（body_check 回报约 0.97）。
- **B 不动**：继续作为"截断 + 按论文裁剪"的对照（代码 `211d661`，没有梯度保护）。

## Edge cases
- **ĉ 在 ε 附近**：ĉ ≥ ε 的样本按论文写法拟合。它们的未来部分如果已摔倒，按 0 处理，所以目标只包含存活段。
- **世界模型没判定终止、状态却已失控**：R̂ 截断限住目标；critic 对这类输入的预测可能很大，但目标有界，梯度方向是在纠正它；数值溢出由 `c417edd` 的保护处理。这种情况会体现在 `critic_targets_clamped_frac` 里。
- **一个小批量全部是已摔倒样本**：跳过这一步。如果照常做一次梯度为 0 的 Adam 步，参数仍会被动量推动。
- **`critic_min_survival = 0`**：和论文写法、以及活动 change 的行为一致；只有 R̂ 截断仍由 `critic_value_clip` 控制。
- **用新代码 `--resume` 旧 run**：没有需要迁移的状态，直接切到新行为。PROJECT.md 已注明 `--resume` 不读回配置。

## Clarifications
- （无；范围在批准本设计时确认）

## Assumptions
- 范围沿用上一轮分析给出的修法：只改 critic 侧，再补日志；不限制动作，也不在摔倒后停止 rollout。
- 爆炸只发生在 ĉ ≈ 0 的样本上。这是从"Σĉr 正常、奖励均值 −10¹³"推出来的，还没有逐个样本观察过；新日志会验证这一点。
- ε = 0.001（plan-review 时由 0.01 下调，理由见 Design details 中的偏差上界）。
- 论文的 L_V 对全部想象样本不加权（见 `docs/upstream/PAPER_SPEC.md` §9）。按 ĉ 排除样本属于"论文未写"的偏离，要写进 PROJECT.md。
- A 在 17:10 评测完之后空闲，可以用来跑验证实验。320k 次约需 37 h（2.4 次/s）。

## Contract impact
none（结果仍按迭代对齐报告；新日志键和新开关写进 PROJECT.md）

## Project impact
`docs/PROJECT.md`：
- 策略更新一节：说明 critic 只拟合存活样本，R̂ 截断；
- 新开关 `critic_min_survival`，以及新日志键的含义；
- 与论文的差异列表加上"critic 损失排除 ĉ < ε 的样本"。

## Experience impact
新增 `docs/experiences/training/imagination-dead-samples.md`（加进 INDEX），内容：
- 想象 rollout 在预测终止后仍然继续，按 ĉ 加权的 J 不受影响，但任何不加权的拟合（critic）都会吃进垃圾；
- 识别方法：critic 目标均值和某个惩罚项同步爆炸，同时 Σĉr 保持正常；
- 只看均值的日志看不到尾部，要记录 max 和 min。

## Open questions
- 已摔倒样本占多少（`imag_dead_frac`），失控事件多久出现一次。这只决定日志读法，不影响设计。
- A 在 120k 到 148k 之间的站立漂移，是不是 critic 事件造成的。144k、124k 的评测和本实验的 148k 会给出答案。
