# Plan

基线：`main`（`6f8db6b`）。`git diff --stat dd91147 HEAD -- src scripts` 为空 → 训练代码与 A 的 `bones_seq_sampler@A`（新采样器 `dd91147`）相同。
远程操作按 `remote-exp` skill 与 `docs/PROJECT.md` 的 remote profile；机器信息只在 `.compute_units`。

## Implementation steps

### 代码
- [x] T1 — `PolicyTrainerCfg` 加 `critic_value_clip: bool = True`、`objective_divide_by_horizon: bool = False`（注释写明论文依据与旧行为开关）· `src/supermimic/imagination/policy_trainer.py`
- [x] T2 — `PolicyTrainerCfg` 加 `g_max: float = 10.0`（由 Trainer 覆盖，与现有 `obs_mode` 的复制方式相同）；在 `objective.py` 加纯函数 `clamp_values(v, g_max)`（返回截断值与越界比例），供 T4/T7 共用 · `src/supermimic/imagination/policy_trainer.py`、`src/supermimic/imagination/objective.py`
- [x] T3 — Trainer 在构造 PolicyTrainer 前设 `cfg.policy_trainer.g_max = heads.reward_cfg.max_step_reward / (1 − cfg.policy_trainer.gamma)`（`PolicyTrainer(` 只有 `loop.py:94` 一处构造）· `src/supermimic/training/loop.py`
- [x] T4 — `update_on` 的 actor 部分：`critic_value_clip` 时 `v_H = v_H.clamp(-g_max, g_max)`，并记录 `critic_vH_clamped_frac`（`(|v_H_raw| > g_max).float().mean()`）· `src/supermimic/imagination/policy_trainer.py`
- [x] T5 — 在 `objective.py` 加纯函数 `policy_loss(J, horizon, divide_by_horizon)`（`-J/H` 或 `-J`），`update_on` 用它替换 `loss = -J / cfg.horizon`（便于 CPU 测试 T11）· `src/supermimic/imagination/objective.py`、`src/supermimic/imagination/policy_trainer.py`
- [x] T6 — 记录 `grad_clipped = float(gn > cfg.grad_clip_norm)`（与 `policy_grad_norm` 同处，不额外同步：`gn` 已在 `float(gn)` 时同步；`gn` 非有限时该步被跳过，`grad_clipped` 记 1，与 `skipped_nonfinite` 一起看）· `src/supermimic/imagination/policy_trainer.py`
- [x] T7 — critic 部分：`critic_value_clip` 时 `values` 先 clamp 到 [−g_max, g_max] 再进 `gae_targets`；记录 `critic_values_clamped_frac`；`critic_value_mean` 仍记录**截断前**的均值（保持诊断意义）· `src/supermimic/imagination/policy_trainer.py`

- [x] T21 — `clip_and_step` 加可选参数 `extra_skip`（0 维 GPU 张量，与"梯度范数非有限"取或后传给 `fused_step`）；critic minibatch 循环改用 `clip_and_step(self.critic_opt, self.critic.parameters(), cfg.critic_grad_clip, None, extra_skip=skip)`，累计被跳过的 minibatch 数，记录 `critic_skipped_nonfinite_grad`（与 `critic_loss` 同处取值，不增加同步）· `src/supermimic/imagination/policy_trainer.py`

### 测试
- [x] T9 — 新测试（CPU）：`clamp_values` 截断超界值、越界比例正确；`policy_objective` 对饱和 V_H 的梯度为 0 · `tests/test_policy_clamp.py`
- [x] T10 — 新测试：GAE 输入估值截断（构造 values 含 > g_max 的值，目标 ≤ 有界奖励递推的上限）· `tests/test_policy_clamp.py`
- [x] T11 — 新测试（CPU）：`policy_loss` 两种取值下对同一参数的梯度相差 H 倍；把 H 倍梯度交给 `clip_and_step(max_norm=1.0, clip_value=0.1)` 时返回的 `gn` > 1（裁剪触发），/H 版本 < 1 · `tests/test_policy_clamp.py`
- [x] T22 — 新测试（CPU）：输入有限、但梯度范数为 inf 时 critic 式的一步被跳过（权重与 Adam 状态不变）；`extra_skip=1` 时即使梯度有限也跳过 · `tests/test_clip_and_step.py`
- [x] T12 — `test_training_smoke` 加一个参数化：两个开关设为旧值也能跑通（GPU，B 机；默认值路径由原测试覆盖）· `tests/test_imagination.py`

### 远程验证
- [x] T13 — 同步代码到 B 的 `$SM_REPO/.worktrees/2026-10-26-critic-clamp-paper-grad-clip`（`scripts/remote/sync_code.sh`，本地工具），确认 `supermimic.__file__` 指向该 worktree 的 `src/` · B 机
- [x] T14 — B 机跑全部测试（`PYTHONPATH=<tree>/src pytest tests`），日志写 `<tree>/tmp/B/tests/pytest.log` · B 机
- [x] T15 — B 机正式训练（tmux `bonesClamp`）：`sm-train --run-name bones_seq_clamp@B --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000`（与 A 的启动参数相同；不单独短跑——缩短 `total_iterations` 会改变余弦学习率曲线，前段就不再可比）· B 机
- [x] T16 — 正式训练前约 1,000 次迭代时检查（`scripts/tb_tail.py`，`--filter policy/`）：`policy_grad_norm` 中位数约为 A 同迭代段 × 32、`grad_clipped` 比例、两个 clamped_frac 接近 0、无报错、速度（次/s）与 A 相当；异常则停训并回到 plan · B 机
- [x] T17 — 全量评测 44k、88k（`scripts/eval_benchmark.py --library bones_seed_compact.pt`）+ `scripts/eval_groups.py`，与 A 同迭代结果（44k 210 / 88k 67）对比 · B 机
  - 结果：44k 199 / 0.9215（A 210 / 0.9340）；88k 108 / 0.9288 是 53.4k 时冻结的策略，不可比。见 validation.md
- [ ] T18 — A 已停（策略自 15.17 万次冻结）：评测 A 的 `ckpt_0148000`（最后一个 critic 完好的点）；B 到 14.8 万次时评测同迭代 checkpoint 对比；从 tb 检查 B 的 `critic_value_mean` 全程有限且 ≤ 10.2、`critic_skipped_nonfinite` / `skipped_nonfinite` 为 0、无崩溃（检查必须含 isfinite，NaN 与阈值比较恒为假）· A、B 机
  - 未完成，结果不通过：B 在 53,350 次 critic 爆炸，53,400 次起 critic 为 NaN、每步被跳过，101,850 次停止，没有 148k。由 change 2026-10-26-critic-alive-samples 补全。见 validation.md
- [x] T23 — 把 T21/T22 的代码同步到 B 的 worktree 并在 B 跑 `tests/test_clip_and_step.py` 与 `tests/test_policy_clamp.py`（不重启正在跑的训练；训练进程已加载旧代码，不受同步影响）· B 机

### 文档
- [x] T19 — 更新 PROJECT.md：策略目标 −J（不除以 H）与裁剪生效、critic 值域截断、两个开关、新日志键、`policy_grad_norm` 量级变化、新代码 resume 旧 run 会切换行为；与论文差异列表同步 · `docs/PROJECT.md`
- [ ] T20 — 记录经验：Adam 下损失整体缩放只通过梯度裁剪起作用；critic 估值 > max_r/(1−γ) 是发散前兆 · `docs/experiences/training/critic-overestimation-and-loss-scale.md` + `docs/experiences/INDEX.md`

## Files to change
- `src/supermimic/imagination/policy_trainer.py`: 两个开关、`g_max` 配置、V_H 与 GAE 估值截断、−J 缩放、三个新日志键；`clip_and_step` 的 `extra_skip` 与 critic 非有限梯度保护
- `tests/test_clip_and_step.py`: critic 非有限梯度跳过的测试
- `src/supermimic/imagination/objective.py`: 纯函数 `clamp_values`、`policy_loss`
- `src/supermimic/training/loop.py`: 设 `cfg.policy_trainer.g_max`
- `tests/test_policy_clamp.py`: 新测试
- `tests/test_imagination.py`: smoke 测试加旧开关参数化
- `docs/PROJECT.md`: 功能说明与论文差异
- `docs/experiences/training/critic-overestimation-and-loss-scale.md`、`docs/experiences/INDEX.md`: 新经验

## Tests
- [x] Unit tests — T9–T12（CPU，本地可跑的部分本地跑；全部在 B 机跑）
  - B 机 T14 全量 54 passed；T23 两个测试文件通过
- [x] Integration tests — 已有测试全部通过（T14）
  - T14：54 passed
- [x] Benchmark or performance tests — T16 速度与 A 相当（截断不应影响次/s）
  - 2.4 次/s，与 A 相同
- [ ] Manual validation — T16 日志量级检查；T17/T18 全量评测按迭代对比 A
  - T16、T17 已做；T18 未完成（B 崩溃）

## Contract updates
none

## Project updates
`docs/PROJECT.md`：策略目标缩放与裁剪、critic 值域截断、开关与日志键、resume 行为变化、论文差异列表（T19）

## Experience updates
`docs/experiences/training/`：损失整体缩放在 Adam 下只经由裁剪起作用；critic 越过可达上限是发散前兆（T20）

## Risks
- 裁剪生效后有效步长分布改变，早期（按迭代）学习可能变慢或变快——T17 的 44k 对比会暴露；不在本 change 调参。
- 元素截断 0.1 可能几乎每步都触发（梯度方向被改变）——T15 记录比例，若接近 100% 在 validation 报告并作为后续问题。
- A 已在 19.47 万次停止（策略自 15.17 万次冻结），T18 只能与 A 的 14.8 万次对比；14.8 万次之后只能对比 B 自身与旧 B 的 196k 崩溃。
- B 的训练进程运行的是 T21 之前的代码（`211d661`）：若 B 在本次运行中出现非有限 critic 梯度，保护不会生效——这时 B 的结果不能代表最终代码，需在 validation 注明并考虑重跑。
- B 机单卡：正式训练约 37 h 跑满 32 万次；T17 的 44k / 88k 分别约在开跑后 5 h / 10 h。

## Rollback plan
两个开关设回旧值（`--policy-trainer.critic-value-clip False --policy-trainer.objective-divide-by-horizon True`）即恢复旧行为；或不合并 `change/2026-10-26-critic-clamp-paper-grad-clip` 分支。
