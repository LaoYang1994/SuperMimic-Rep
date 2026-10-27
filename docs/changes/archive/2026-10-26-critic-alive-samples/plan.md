# Plan

代码基于 change 2026-10-26-critic-clamp-paper-grad-clip 的分支（`c417edd`）。下文"存活"指 ĉ_t ≥ ε（ε = `critic_min_survival`），
"已摔倒"指 ĉ_t < ε；ĉ 即 `ImagRollout.survival`，形状 (H+1, n)；`rewards[t]` 是 r̂_{t+1}，在 J 里乘 ĉ_t。

## Implementation steps

准备
- [x] T1 — 从分支 `change/2026-10-26-critic-clamp-paper-grad-clip`（`c417edd`）建 worktree，分支 `change/2026-10-26-critic-alive-samples`（由 `/change-implement` 询问后创建）· `.worktrees/2026-10-26-critic-alive-samples`

纯函数（`objective.py`，CPU 可测）
- [x] T2 — 新增 `alive_mask(survival, eps) -> BoolTensor`：`survival >= eps`；eps = 0 时全为真 · `src/supermimic/imagination/objective.py`
- [x] T3 — 新增 `mask_dead(rewards, dhat_next, values, alive)`：用 `torch.where` 把已摔倒样本替换成 r̂ = 0（按 `alive[:-1]`）、d̂ = 1（按 `alive[1:]`）、V = 0（按 `alive`）；不用乘法 · `src/supermimic/imagination/objective.py`
- [x] T4 — `gae_targets` 加可选参数 `clip: float | None = None`：递推中每一步 `R = R.clamp(-clip, clip)`，`R_H` 也截；为 None 时与现在逐位相同 · `src/supermimic/imagination/objective.py`
- [x] T5 — 新增 `masked_mse(pred, target, weight) -> (loss, empty)`：`loss = Σ w·(p − y)² / max(Σ w, 1)`，`empty = (Σ w == 0)` 的 0 维 float 张量；不做 host 同步 · `src/supermimic/imagination/objective.py`

PolicyTrainer（`policy_trainer.py`）
- [x] T6 — `PolicyTrainerCfg` 加 `critic_min_survival: float = 0.001`，注释说明含义、0 即论文写法、为什么（想象中已摔倒样本的爆炸）· `src/supermimic/imagination/policy_trainer.py`
- [x] T7 — critic 块开头：`alive = alive_mask(ro.survival.detach(), cfg.critic_min_survival)`；在计算 `values` 之前，把已摔倒样本的 `feat_d`、`goals_d`、`e_root_d` 用 `torch.where(alive[..., None], x, 0)` 替换为 0 · `src/supermimic/imagination/policy_trainer.py`
- [x] T8 — 目标计算改为：`values`（截断，现有）→ `mask_dead(...)` → `gae_targets(..., clip=cfg.g_max if cfg.critic_value_clip else None)` · `src/supermimic/imagination/policy_trainer.py`
- [x] T9 — 构造逐样本权重 `W = alive[:-1].reshape(-1).float()`；非有限检查的 `Y` 改为 `torch.where(W > 0, Y, 0)`（输入在 T7 已替换，全部有限）· `src/supermimic/imagination/policy_trainer.py`
- [x] T10 — 小批量损失改用 `masked_mse(pred, Y[idx], W[idx])`；`extra_skip = torch.maximum(skip, empty)`；`grad_skips` 只计梯度非有限导致的跳过（减去 `max(skip, empty)`）；同时累计 `closs` 的最大值 · `src/supermimic/imagination/policy_trainer.py`
- [x] T11 — rollout 日志（始终记录）：`imag_dead_frac = (~alive[:-1]).float().mean()`（`use_critic=False` 时用 `ro.survival[:-1] < eps` 直接算）、`imag_action_abs_max = ro.actions.abs().max()`、`imag_reward_min = ro.rewards.min()` · `src/supermimic/imagination/policy_trainer.py`
- [x] T12 — critic 日志：`critic_loss_max`（20 个小批量的最大值）；`critic_value_clip` 开时，`critic_targets_clamped_frac` = 存活样本中 `|R̂| >= g_max` 的比例；已有的 `critic_value_mean`（截断前）、`critic_values_clamped_frac`、`critic_target_mean` 改为只在存活样本上统计（T7 之后，已摔倒样本的估值是 critic 对全 0 输入的输出，目标为 0，混进均值没有意义）· `src/supermimic/imagination/policy_trainer.py`

日志聚合
- [x] T13 — 新模块：`accumulate(agg, key, value)`（默认求和；`*_max` 取最大、`*_min` 取最小，NaN 传播）和 `interval_values(agg, n)`（求和键除以 n，`*_max` / `*_min` 与 `lr` 原样）· `src/supermimic/training/metrics.py`
- [x] T14 — 训练循环改用 `accumulate` / `interval_values` 代替内联的求和与除法（`wm/*`、`policy/*`、`lr` 三处；打印行里的 `agg.get(...)/agg_n` 不变）· `src/supermimic/training/loop.py`

测试
- [x] T15 — 新测试：死尾样本（第 4 步起 ĉ = 0，之后奖励 −1e17、估值 NaN、d̂ NaN）经 `mask_dead` + `gae_targets` 后，存活步的目标有限，且等于"在摔倒处终止"的干净 rollout 的目标；全程存活的样本与旧 `gae_targets` 逐位相同 · `tests/test_critic_alive.py`
- [x] T16 — 新测试：eps = 0、clip = None 时，随机输入下与旧 `gae_targets` 逐位相同 · `tests/test_critic_alive.py`
- [x] T17 — 新测试：存活样本带巨大正奖励时，`clip=10` 下所有目标 ∈ [−10, 10] · `tests/test_critic_alive.py`
- [x] T18 — 新测试：小 `Critic`（`Critic(CriticCfg(hidden=(16, 8)))`，默认 goal 维度 680 与默认 `PolicyCfg` 匹配）上，已摔倒样本输入含 inf / NaN，按 T7 替换后，`masked_mse` 的 loss 等于存活子集上的 MSE、梯度全部有限；权重全 0 时 `empty == 1`、loss 为 0 · `tests/test_critic_alive.py`
- [x] T19 — 新测试：`accumulate` / `interval_values` 的均值、最大、最小、NaN 传播与 `lr` 原样 · `tests/test_metric_agg.py`
- [x] T20 — 冒烟测试的 legacy 参数化再加 `critic_min_survival = 0.0`，让新旧两条路径都跑到 · `tests/test_imagination.py`

文档
- [x] T21 — 策略更新一节：critic 只拟合存活样本、R̂ 截断、`critic_min_survival`、新日志键、`*_max` / `*_min` 的聚合方式；标注"论文未写"（PROJECT.md 没有单独的差异列表，与上一条 critic 截断一样在正文内注明）；训练循环一节说明记录点是区间平均与 `*_max` / `*_min` 的聚合 · `docs/PROJECT.md`
- [x] T22 — 新经验文档（想象 rollout 预测终止后继续运行；不加权的拟合吃进垃圾；识别方法；均值日志看不到尾部），并在 INDEX 加一行 · `docs/experiences/training/imagination-dead-samples.md`、`docs/experiences/INDEX.md`

远程验证（按 remote-exp skill；worktree 运行设 `PYTHONPATH=<tree>/src`）
- [x] T23 — 同步 worktree 到远程，在 A 上先跑本 change 相关的测试（`tests/test_critic_alive.py tests/test_metric_agg.py tests/test_policy_clamp.py tests/test_clip_and_step.py tests/test_imagination.py`），记录通过数，并确认导入的是 worktree 的代码 · `tmp/A/alive_tests/`
- [x] T24 — 照 B 的启动脚本（B worktree 的 `tmp/B/bones_clamp/run.sh`）写 `tmp/A/critic_alive/run.sh`：先跑全量 `pytest -q tests`，不通过就不启动；记录导入的代码路径和提交号；然后 `sm-train --run-name bones_seq_critic_alive@A --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000`（seed 默认为 0，与 A、B 相同）。在 tmux 里启动 · `tmp/A/critic_alive/`
- [x] T25 — 前 1,000 次迭代检查：速度与 B 相当（约 2.4 次/s）、新日志键都在、`imag_dead_frac` 与 `critic_loss_max` 有限；不合格就停下来，回到计划 · `tmp/A/critic_alive/`
- [x] T26 — 在 A 上启动自动评测：44k、88k、148k 的 checkpoint 一出现，就做全量评测 + `eval_groups.py` · `tmp/A/critic_alive_eval/`
- [x] T27 — 44k、88k：与 A（210 / 0.934，67 / 0.9406）和 B 同迭代对比，比较失败数、回报、分组 · `tmp/research/night_2026-10-26.md`
  - 结果：44k 166 / 0.9333、88k 70 / 0.9413，与旧 A、对照分不出差别。见 validation.md
- [x] T28 — 148k：对比 A（80 / 0.9183，body_check 0.868）和 B；从 tb 检查全程 `critic_loss_max` < 1、所有值有限、`critic_skipped_*` 为 0，并看事件点附近的 `imag_reward_min` / `imag_action_abs_max` · `tmp/research/night_2026-10-26.md`
  - 结果：critic 全程有限、无跳过（通过）；`critic_loss_max` 有 8 个点 > 1（预测侧，孤立）；148k 125 / 0.8910，不如旧 A 的 80 / 0.9183，站立漂移不通过，转交采样器 change。见 validation.md

提交
- [x] T29 — 本地提交（每个逻辑单元一个提交）；提交前扫描被跟踪文件和提交内容，确认没有机器地址、端口、用户、路径；不推送 · `.worktrees/2026-10-26-critic-alive-samples`

## Files to change
- `src/supermimic/imagination/objective.py`：`alive_mask`、`mask_dead`、`masked_mse`；`gae_targets` 的 `clip` 参数
- `src/supermimic/imagination/policy_trainer.py`：`critic_min_survival`；critic 输入替换、目标屏蔽与截断、按权重计算损失、空小批量跳过；新日志键
- `src/supermimic/training/metrics.py`（新）：日志聚合
- `src/supermimic/training/loop.py`：改用聚合函数
- `tests/test_critic_alive.py`（新）、`tests/test_metric_agg.py`（新）、`tests/test_imagination.py`
- `docs/PROJECT.md`、`docs/experiences/training/imagination-dead-samples.md`（新）、`docs/experiences/INDEX.md`

**与活动 change 2026-10-26-critic-clamp-paper-grad-clip 重叠**：`policy_trainer.py`、`objective.py`、`tests/test_imagination.py`、`docs/PROJECT.md`。
本 change 建在它的分支上，合并顺序为它先、本 change 后；它的 T20（经验文档 `critic-overestimation-and-loss-scale.md`）和本 change 的 T22 内容相邻，写的时候互相链接，不要重复。

## Tests
- [x] Unit tests：T15–T19（CPU）
- [x] Integration tests：T20 冒烟测试（GPU，A）
- [x] Benchmark or performance tests：T25 速度与 B 相当（次/s）
- [x] Manual validation：T27、T28 同迭代全量评测与 tb 检查
  - T27、T28 已完成，结果见 validation.md

## Contract updates
none

## Project updates
`docs/PROJECT.md` 的策略更新一节：critic 只拟合存活样本（`critic_min_survival`，默认 0.001，0 即论文写法）、R̂ 截到 ±G_max、新日志键，以及 `*_max` / `*_min` 键在 log interval 内取最大 / 最小；与论文差异列表加一条。

## Experience updates
`docs/experiences/training/imagination-dead-samples.md`：想象 rollout 在预测终止后继续运行，按 ĉ 加权的目标不受影响，但不加权的拟合会吃进垃圾；
识别方法是 critic 目标均值和某个惩罚项同步爆炸、同时 Σĉr 正常；均值日志看不到尾部，要记录 max 和 min。

## Risks
- **世界模型没判定终止、状态却已失控**：存活样本里出现垃圾。目标由 R̂ 截断限住，但 critic 对这类输入的预测仍可能很大，loss 会变大。数值溢出由 `c417edd` 的保护处理，这种情况会体现在 `critic_targets_clamped_frac` 和 `critic_loss_max` 里。
- **ε 带来的偏差**：只有"下一步就已摔倒"的存活样本的目标有偏差，上界是 (ε/ĉ_k)·G_max（ĉ_k = 1 时不超过 0.01）。如果 `imag_dead_frac` 明显偏大（比如超过 1%），要回头看。
- **策略侧仍可能被跳过整步**：actor 的 V_H 仍然用未替换的输入计算；已摔倒样本的 V_H 如果是 inf，ĉ_H·V_H 会得到 NaN，整步策略更新会被现有保护跳过。这在设计的非目标范围内，可以从 `skipped_nonfinite` 观察到。
- **显存**：T7 的 `torch.where` 会复制 goals（33×4096×640 fp32，约 350 MB）和 feat，对单卡显存来说可以忽略。
- **实验混杂**：新 run 和 B 的差别是梯度保护加存活屏蔽，不是单独的存活屏蔽。保护只在出现非有限梯度时起作用，所以按 tb 里的跳过计数就能分开二者。
- **速度**：多了几次 `torch.where`，作用在 (33, 4096, 约 800) 的张量上，预计可以忽略；T25 会核对。
- **占用 A**：全量 run 约 37 h，期间 A 不能做 profile 或 TF32 实验。

## Rollback plan
- 设 `--policy-trainer.critic-min-survival 0` 即回到论文写法；R̂ 截断跟随 `critic_value_clip`。
- 新日志键和聚合函数不影响已有键。
- 必要时整个分支不合并，活动 change 不受影响。
