# Review

## Summary
审查对象：分支 `change/2026-10-26-critic-alive-samples` 中 `c417edd..3f136d9` 的改动（代码 `f40bd1f`、`c888e07`，文档 `9597c99`、`3f136d9`）。
代码与冻结的设计一致，没有 P1。一个 P2（经验文档把推断写成了事实）已在审查中修复（`3f136d9`）。

**Verdict：not-ready。** 代码层面没有阻塞问题；唯一的原因是验证任务 T27、T28 还在等 `bones_seq_critic_alive@A` 跑到 44k、88k、148k。
A 上运行的是 `9597c99`，与 `3f136d9` 只差文档。

**更新（归档时）**：T27、T28 已完成，见 `validation.md`。结果：critic 侧目标达成；148k 性能目标未达成，原因转交采样器 change。按用户决定归档并合并。

## Coverage
- **CLEAN：设计逐项都有实现。**
  - "已摔倒"的定义（`alive_mask`）；
  - 已摔倒样本在 GAE 里按终止处理（`mask_dead`：r = 0、d̂ = 1、V = 0，用替换实现）；
  - R̂ 在递推中截断（`gae_targets(clip=)`，由 `critic_value_clip` 控制）；
  - critic 前向前把已摔倒样本的输入替换为 0；
  - 损失只算存活样本（`masked_mse`）；小批量里没有存活样本时跳过这一步，多卡时由各 rank 一致决定；
  - 5 个新日志键；critic 统计只算存活样本；
  - 日志聚合时 `*_max` / `*_min` 取区间最大 / 最小。
- **CLEAN：plan 中已完成的任务都对应真实产物。**
  - T1–T26 和 T29 都有对应的代码、测试、文档、远程脚本或提交；
  - 测试结果：B 上 CPU 测试 16 项通过；A 启动前的全量测试 62 项通过；
  - T25 测得速度变化（见 P3-1）；
  - T27、T28 未完成，属于验证，不是遗漏。
- **没有超出设计的实现。** 多卡下"空小批量"由所有 rank 一致决定（`allreduce_max_`），是设计中"多卡：各 rank 一致"的具体实现，不算新功能。
- **大文件检查：** 没有超过 1 MB 的文件。
- **`CONTRACTS.md`：** 不受影响。

## Findings
### P1
（无）

### P2
（见 Resolved）

### P3
- [P3] (8/10) `src/supermimic/imagination/policy_trainer.py:173-223` — **速度没有达到计划"可以忽略"的预期。**
  - 在同一台 A、0–4k 次迭代上测得：策略阶段 0.2328 s/次，旧代码 `bones_seq_sampler@A` 为 0.2212 s/次，多了 11.6 ms（+5%）；整体约慢 3%（2.29 对 2.36 次/s）。
  - 可能的来源：每个小批量额外的小 kernel（`masked_mse`、空批判断、loss 最大值），20 个小批量共约 300–400 次 launch；GAE 里 32 次 clamp；三次 `torch.where` 复制输入。
  - 修法：按 profile 实测的毫秒数决定是否优化。例如把 critic 小批量循环编译或合并 kernel，或者只在 `alive.all()` 为假时才复制输入（但这会引入一次 host 同步，需要权衡）。不在本 change 内处理。
- [P3] (7/10) `src/supermimic/imagination/policy_trainer.py:124-125, 159` — **`critic_vH_clamped_frac` 仍然统计全部 n 个 V_H，包括已摔倒的样本。**
  - 这和"critic 统计只算存活样本"不一致。
  - 实测影响很小：新 run 在 0–8k 次迭代中，它和只算存活样本的 `critic_values_clamped_frac` 是同一量级（2.0e-5 对 6.8e-6，2.6e-5 对 2.4e-5）。
  - 修法：改为只在 `alive[-1]` 上统计（需要把 `alive` 的计算移到 actor 步之前）。只影响日志，可以以后处理。
- [P3] (6/10) `tests/` — **`PolicyTrainer.update_on` 处理已摔倒样本的那段接线没有专门的测试。**
  - 接线指：输入替换、权重 `W`、空批跳过、只看存活样本的非有限检查。纯函数已有 4 项 CPU 测试；GPU 冒烟测试走过这条路径，但不保证真的出现已摔倒样本。
  - 修法：加一个测试，在 `update_on` 的 rollout 里注入一个 ĉ = 0、奖励 −1e17、状态含 inf 的样本，检查 critic 参数有限、`critic_loss_max` 有界。需要 GPU 和世界模型，可以借用冒烟测试的配置。

## Unverified
- B 的 `critic_vH_clamped_frac` 缓慢上升（5×10⁻⁶ → 1.2×10⁻⁵）。它可能部分来自已摔倒样本的 V_H，而不是高估。B 没有 `imag_dead_frac`，无法确认（置信度 3/10）。

## Resolved
- [P2] (8/10) `docs/experiences/training/imagination-dead-samples.md` "Root cause" 一节 — 把"失控样本的 ĉ 已为 0"写成了观察到的事实。这其实是从区间均值推出来的（Σĉr 正常而奖励均值为 −10¹³），还没有逐样本观察过；长期文档不能写未验证的结论。
  → 已注明是推断（`3f136d9`）；T28 用新日志确认后，可以再改回陈述句。

## Accepted risks
- 验证 run 用的是 `9597c99`，与最终代码只差文档。
- 设定 ε 后，"下一步就已摔倒"的存活样本，目标会少一项自举项，偏差上界为 (ε/ĉ_k)·G_max（见 design）。新 run 的 `imag_dead_frac` 从 0–1k 次迭代的 15%，降到 6–8k 次迭代的 2×10⁻⁴。
- 策略侧的 NaN（已摔倒样本的 V_H 为 inf 时，ĉ_H·V_H = NaN）仍由现有的整步跳过处理，属于设计的非目标。

## Follow-up items
- T27、T28：新 run 在 44k、88k、148k 的同迭代全量评测和 tb 检查 → `/change-validate`（`validation.md`）。
- P3-1 速度优化：属于混合精度和 profile 工作的一部分（memory `pending-mixed-precision`），需要另开 change。
- P3-2、P3-3：日志一致性和接线测试，下次修改 `policy_trainer.py` 时一起处理。
