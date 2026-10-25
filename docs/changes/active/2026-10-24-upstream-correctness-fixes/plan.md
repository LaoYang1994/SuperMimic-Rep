# Plan

约定（给零上下文的执行者）：
- 本地仓库 `<local repo>`（无 GPU）；GPU 机 `ssh <gpu-box>`，代码 `$SM_REPO`，
  环境 `source <env file>`（Python 在 `$UV_PROJECT_ENVIRONMENT/bin/python`）。代码用 `scripts/remote/sync_code.sh` 同步。
- **本 change 现在只实现（A–C 组）**；D 组的 GPU 验证要等 change `2026-10-24-paper-faithful-reproduction` 的 BONES-SEED 运行与评测结束
  （GPU 机 `tmp/phaseB/progress.log` 出现 `DONE phase C`）后才开始，期间**不得同步代码到 GPU 机**（会改动正在运行的代码目录）。
- 分支：从 `d24eda8` 新建 `change/2026-10-24-upstream-correctness-fixes`。（worktree：本地 `.worktrees/2026-10-24-upstream-correctness-fixes`）
- 远程运行按 `.claude/skills/remote-exp/SKILL.md`：用**本 worktree 里的** `scripts/remote/sync_code.sh` 同步到远程
  `$SM_REPO/.worktrees/2026-10-24-upstream-correctness-fixes`；在那里运行时
  `export PYTHONPATH=$PWD/src` 并打印 `supermimic.__file__` 确认；日志与 checkpoint 写该远程 tree 的 `logs/`
  （修复前的对照 `lafan1_now` 在远程主目录的 `logs/lafan1_now/`）。
- 结果按 `docs/CONTRACTS.md#结果报告与论文对照` 标注；实验记录写本 change 的 `validation.md`。

## Implementation steps

### A. WM 终止标签对齐（design 问题 1）
- [x] T1 — 新建分支 `change/2026-10-24-upstream-correctness-fixes`（从 `d24eda8`；是否用 worktree 由 `/change-implement` 决定）· git
- [x] T2 — `sample_windows` 的 `done` 改为取转移行自己的标签 `self.done[ee[:-1], tt[:-1]]`，并更新 `WindowBatch.done` 注释为"第 j 个转移结果的终止标签" · `src/supermimic/data/replay_buffer.py`
- [x] T3 — 更新模块 docstring：`r` 与 `done` 都是同一转移 (s_t, a_t)→s_{t+1} 的标签；终端行只作最后目标状态、不带标签 · `src/supermimic/data/replay_buffer.py`
- [x] T4 — `record_pre_reset` 写终端行时 `done` 置 0（终止标签已在上一行转移行里），并更新模块 docstring 与注释 · `src/supermimic/data/recorder.py`
- [x] T5 — 新增 CPU 单测：手工构造 `ReplayBuffer`（2 env、capacity 32），写一个以真实终止结束的 episode 和一个以超时结束的 episode，采样所有窗口，断言：只有"结果是终止状态"的那个转移 `done=1`，其余（含超时、含终止前一步）都为 0 · `tests/test_replay_labels.py`

### B. 四元数零误差梯度（design 问题 2）
- [x] T6 — 在 rewards 模块新增 `_quat_box_minus(q1, q2)`：与 mjlab `quat_box_minus` 同公式（符号折叠、`atan2`、小角度 Taylor 阈值），被丢弃分支的除数替换为 1 · `src/supermimic/envs/mdp/rewards.py`
- [x] T7 — `_sq_quat_err` 改用 `_quat_box_minus`，删去对 mjlab `quat_box_minus` 的 import · `src/supermimic/envs/mdp/rewards.py`
- [x] T8 — 新增单测：随机四元数（含 q1==q2、q1==−q2、极小角）上前向与 mjlab `quat_box_minus` 逐元素一致（atol 1e-7）；`_sq_quat_err` 在 q1==q2 处对两个输入的梯度有限且为 0 · `tests/test_quat_error.py`

### C. 续训 warmup 与非有限梯度保护（design 问题 3 + 保护）
- [x] T9 — `run()` 的 warmup 循环改为按"本次运行已采步数"计数（局部计数器），使续训后也先补采 `warmup_sim_steps` 步 · `src/supermimic/training/loop.py`
- [x] T10 — 在 policy_trainer 新增辅助函数 `clip_and_step(opt, params, max_norm, clip_value=None) -> (grad_norm, skipped)`：裁剪后若总范数非有限则清梯度、跳过 `opt.step()` · `src/supermimic/imagination/policy_trainer.py`
- [x] T11 — 策略更新改用 `clip_and_step`（范数裁剪 + 逐元素裁剪），把 `skipped` 计入指标 `skipped_nonfinite` · `src/supermimic/imagination/policy_trainer.py`
- [x] T12 — critic 拟合前检查一次输入（feat/goal/xi/e_root）与 GAE 目标是否全有限，否则跳过本次 critic 拟合并记 `critic_skipped_nonfinite`（每次更新 1 次同步，不在 20 个 minibatch 里逐个检查——见 design 修订）· `src/supermimic/imagination/policy_trainer.py`
- [x] T20 — （review P2）critic 的跳过判断在所有 rank 间 all-reduce（任一 rank 非有限则全部跳过），避免多卡下部分 rank 卡在 critic 梯度 all-reduce 里死锁 · `src/supermimic/imagination/policy_trainer.py`
- [x] T13 — 新增单测：玩具线性层 + Adam，构造 NaN 梯度时 `clip_and_step` 返回 skipped=True 且参数与优化器状态不变；有限梯度时正常更新 · `tests/test_clip_and_step.py`

### D. 验证（当前 BONES-SEED 运行结束后才做）
- [x] T14 — 确认 GPU 机 `tmp/phaseB/progress.log` 已出现 `DONE phase C`，再同步本分支代码 · `scripts/remote/sync_code.sh`
- [x] T15 — GPU 机跑全部测试（含新增 3 个测试文件），必须全通过、CUDA 测试不得因缺 CUDA/数据被跳过 · `tests/`
- [x] T16 — 续训冒烟：LAFAN1 训练 300 迭代后从 `ckpt_latest.pt` `--resume` 再训 100 迭代，确认日志先出现 warmup 采样、无空 buffer 报错 · `logs/resume_smoke/`
- [x] T17 — 对照重跑：`sm-train --run-name lafan1_now_fixed --data.npz-dir data/npz/lafan1 --env.obs-mode anchor_ori_now --schedule.max-hours 2 --schedule.checkpoint-interval 2000`（与修复前 `lafan1_now` 完全同配置）· `logs/lafan1_now_fixed/`
- [x] T18 — 训练结束后评测全部 checkpoint 与 `ckpt_latest`（`scripts/eval_benchmark.py --npz-dir data/npz/lafan1`）· `logs/lafan1_now_fixed/`
- [x] T19 — 与修复前 `lafan1_now` 逐点对比：评测回报（论文 §4）、失败数、训练日志的 `wm/term_acc`、`policy/imag_survival_H`、`policy/skipped_nonfinite` 计数，写入 `validation.md` · `docs/changes/active/2026-10-24-upstream-correctness-fixes/validation.md`

## Files to change
- `src/supermimic/data/replay_buffer.py`：窗口 `done` 取转移行标签；docstring
- `src/supermimic/data/recorder.py`：终端行 `done=0`；docstring
- `src/supermimic/envs/mdp/rewards.py`：本地 `_quat_box_minus`，`_sq_quat_err` 改用之
- `src/supermimic/training/loop.py`：warmup 按本次运行计数
- `src/supermimic/imagination/policy_trainer.py`：`clip_and_step`，策略与 critic 使用并记录跳过次数
- `tests/test_replay_labels.py`、`tests/test_quat_error.py`、`tests/test_clip_and_step.py`（新）
- `docs/changes/active/2026-10-24-upstream-correctness-fixes/validation.md`（验证阶段）

与活动 change `2026-10-24-paper-faithful-reproduction` 重叠：`training/loop.py`、`imagination/policy_trainer.py`。其代码已在 `d24eda8` 提交，本分支从它拉出，正常合并即可；注意它在 GPU 机上仍有运行中的任务，D 组之前不同步。

## Tests
- [x] Unit tests — `test_replay_labels.py`（标签对齐，CPU）、`test_quat_error.py`（前向一致 + 零误差梯度有限）、`test_clip_and_step.py`（非有限跳过）
- [x] Integration tests — 全部现有测试在 GPU 机通过（尤其 `test_world_model`、`test_imagination`、`test_analytic_heads` 的奖励对拍不变）
- [ ] Benchmark or performance tests — none（本 change 不涉及性能；`clip_and_step` 的有限性判断与现有 `float(gn)` 同处同步，不新增同步）
- [x] Manual validation — T16 续训冒烟；T17–T19 LAFAN1 修复前后对照

## Contract updates
none（若 review 认为需要，可在 CONTRACTS 增加数据约定："ReplayBuffer 窗口的 `done[j]` 是第 j 个转移结果的终止标签，超时截断不算终止"）

## Project updates
`docs/PROJECT.md`：世界模型 / 数据模块补充终止标签语义；Features 中 LAFAN1 结果注明"修复前 / 修复后"。

## Experience updates
`docs/experiences/`：横向对照同源 fork 能发现共同上游 bug；`torch.where` 的两支都参与反向，被丢弃分支的 NaN 仍会传播。

## Risks
- 终止标签修复后正样本约减半，终止头可能变得"更乐观"，想象回报尺度上升；若 LAFAN1 对照中回报反而下降，需分析是标签修复本身还是 critic 适配（design Open question）。
- `_quat_box_minus` 若与 mjlab 前向有任何差异，会同时改变仿真侧奖励与评测分数——T8 的逐元素一致测试是门槛。
- 单次 2 h 对照受评测噪声影响（同 checkpoint 两次评测单 clip 回报差可达 0.014，相邻 checkpoint 失败数波动大），结论按趋势而非单点下。

## Rollback plan
三类修复互不依赖、各自集中在 1–2 个文件：可按组 `git revert`。分支未合并前，回到 `change/2026-10-24-paper-faithful-reproduction`（`d24eda8`）即为修复前状态；修复前的实验结果已记录在 paper-faithful change 的 validation 中。
