# Plan

代码在 worktree `.worktrees/2026-10-25-sampler-attainable-return`（分支 `change/2026-10-25-sampler-attainable-return`，从
`change/2026-10-24-training-throughput` 的 `c83f72f` 拉出）；本文件和 design 在主 checkout。远程运行按 `remote-exp` skill 与
`docs/PROJECT.md` 的 remote profile：远程 tree `<根>/.worktrees/2026-10-25-sampler-attainable-return`，`PYTHONPATH=<tree>/src`，
启动时打印 `supermimic.__file__` 核对。结果按迭代对齐（`docs/CONTRACTS.md#结果报告与论文对照`）。

记号：t = episode 内的步序号（0 起）；n = episode 长度；`time_limit` = `ceil(episode_length_s / step_dt)`（10 s / 0.02 = 500）；
N* = episode 不失败时的长度：clip 播完或超时截断时 N* = n（精确），失败终止时 N* = min(n + rem_term, time_limit)，
rem_term = `last_frame_of(f) − f`，f 为 `_finalize_episodes` 被调用时的 `self.frame`（即终止帧，重采样在其后，`commands.py:289-295`）；
m_t = N* − t；A_t = g_max·(1 − γ^{m_t})；Ĝ_t = clip(R_t / A_t, 0, 1)。
（plan-review P1：不按帧号逐步推算剩余步数——奖励在 `frame += 1` 前按当前帧记录、clip 末帧记录两次、重置时可能先前进一帧，
逐步推算会差 1 步，偏差集中在 clip 末尾。）

## Implementation steps

### A. 归一化（核心）
- [x] T1 — 新建不依赖 mjlab 的小模块，函数 `normalized_returns_to_go(rtg, n_star, gamma, g_max)`：rtg (k, T)，n_star (k,)；m = n_star[:, None] − arange(T)，返回 clip(rtg / (g_max·(1 − γ^m)), 0, 1)（float64，`torch.pow`；m ≤ 0 的填充位置返回 0）· `src/supermimic/envs/mdp/sampler_math.py`
- [x] T2 — `MultiClipMotionCommand.__init__` 新增 `self.time_limit_steps = ceil(episode_length_s / step_dt)`；`bin_return` 初值由 `cfg.g_max` 改为 `1.0`（注释：归一化完成度） · `src/supermimic/envs/mdp/commands.py`
- [x] T3 — `_finalize_episodes`：按记号算每个 episode 的 N*（`terminated` 已在该函数中算出；rem_term 用 `self.library.last_frame_of(self.frame[env_ids]) − self.frame[env_ids]`） · `src/supermimic/envs/mdp/commands.py`
- [x] T4 — `_finalize_episodes`：在算出 `rtg`（及可能的自举）之后、按 bin 聚合之前，用 T1 把 `rtg` 换成 Ĝ；截断丢弃（`min_remaining_steps`）、按 bin 求平均、EMA 更新保持原样；`metrics["episode_return_to_go"]` 继续记录未归一化的 `rtg[:, 0]`（日志语义不变） · `src/supermimic/envs/mdp/commands.py`
- [x] T5 — `_sample_frames`：`hard = softmax(−cfg.softmax_temp * self.bin_return)`（去掉 `/ cfg.g_max`），注释写明对应论文 −25·G_b/G_max、G_max 取每步可达值 · `src/supermimic/envs/mdp/commands.py`
- [x] T6 — `g_max` 字段注释更新为"最大可达折现回报（每步奖励上限 / (1 − γ)），用于把 bin 回报归一化" · `src/supermimic/envs/mdp/commands.py`

### B. 诊断与日志
- [x] T7 — `MotionLibrary` 新增 `bin_is_clip_end`（bool, num_bins）：每条 clip 最后一个 bin 为 True（用已有的 `clip_bin_offset` 与 `bins_per_clip` 算） · `src/supermimic/envs/mdp/motion_library.py`
- [x] T8 — `_sample_frames` 在 `return` 采样时记录 `metrics["hard_mass_clip_end"][:] = (hard * lib.bin_is_clip_end).sum()`（GPU 上，无同步；`__init__` 的 metrics 键列表加入该键） · `src/supermimic/envs/mdp/commands.py`
- [x] T9 — 训练日志新增 `sampler/hard_mass_clip_end_bins`（与 `sampler/entropy` 同处、同一次 log interval） · `src/supermimic/training/loop.py`
- [x] T10 — 新增 CPU 诊断脚本：读 checkpoint 的 `bin_return`/`bin_memory` 与紧凑库元数据（`names`/`lengths`，`mmap=True`），按 `bin_frames=50` 重建 bin；自动识别新旧 checkpoint（`bin_return_normalized` 键）；按动作组（复用 `scripts/eval_groups.py` 的 `group_of`，再加 jump / floor / walk 组）输出：bin 数、均匀份额、实际采样概率、困难份额、记忆份额、G_b 均值/最小值；另输出困难份额在 clip 末尾 bin 上的占比与困难份额前 2,000 个 bin 的动作族分布 · `scripts/sampler_report.py`

### C. checkpoint 兼容
- [x] T11 — `Trainer.save` 写入 `"bin_return_normalized": True` · `src/supermimic/training/loop.py`
- [x] T12 — `Trainer.load`：经 `sampler_math.bin_return_from_checkpoint`（不依赖 mjlab，供诊断脚本与单测共用）读取；checkpoint 没有该键时，`bin_return` 按 `/ cfg g_max` 近似换算并截到 [0, 1]，打印 `[WARN] checkpoint predates normalized bin returns; converted approximately` · `src/supermimic/training/loop.py`

### D. 测试
- [x] T13 — CPU 单测（只导入 `sampler_math`）：完美跟踪（每步奖励 = 上限）且 N* = n 的 episode，所有步 Ĝ = 1（含最后一步，误差 < 1e-9）；失败终止（N* > n）的 episode，最后一步 Ĝ < 0.1、episode 开头 Ĝ 明显小于 1；不同长度的 episode 一起成批计算时填充位置为 0；负的 R_t 截到 0 · `tests/test_sampler_normalization.py`
- [x] T14 — CPU 单测：旧 checkpoint 换算（无键 → 除以 g_max 并截断；有键 → 原样） · `tests/test_sampler_normalization.py`

### E. 远程验证（A 机测试与实验）
执行顺序：T17 → T15（在空出的 A 上跑测试，不与 B 的基线训练抢 GPU）→ T16 → T18 → T19 → T20。
- [x] T15 — 同步 worktree 到远程 tree（`scripts/remote/sync_code.sh`），在 A 上跑全部测试（含 T13/T14、`test_env_smoke`、`test_imagination`） · `validation.md`
- [x] T16 — 用 `sampler_report.py`（CPU）读 `bones_seq_fast@B/ckpt_0044000.pt`（旧格式），复现诊断：末尾 bin 困难份额 ≈ 93.9%、空翻困难份额 ≈ 0 · `validation.md`
- [x] T17 — 停止 A 上的 `bones_seq_fix@A`：在 tmux `bonesA` 里找到 `sm-train` 的 PID（`ps -eo pid,cmd`，不要用 `pkill -f`）并 `kill`，记录最后一行迭代日志与已有 checkpoint · `validation.md`
- [x] T18 — 在 A 上启动 `bones_seq_sampler@A`：BONES-SEED 紧凑库、`--env.obs-mode beyondmimic_seq`、320k 调度（默认）、`--schedule.checkpoint-interval 4000`；同时挂 `tmp/eval_on_ckpt.sh` 监视 36k/40k/44k 全量评测 · `<tree>/tmp/A/bones_sampler/`
- [x] T19 — 训练到第 4,000 次迭代后，`sampler_report.py` 读 `ckpt_0004000.pt`：末尾 bin 困难份额显著低于 93.9%，空翻 / 侧滚困难份额 > 0；同时对照 B 的 `ckpt_0004000.pt` · `validation.md`
- [x] T20 — 36k/40k/44k 全量评测完成后，与 `bones_seq_fast@B` 同迭代对比回报、失败数与 `eval_groups.py` 分组（重点空翻、侧滚、跳跃、贴地组），写入 · `validation.md`

## Files to change
- `src/supermimic/envs/mdp/sampler_math.py`（新，不依赖 mjlab）：`normalized_returns_to_go`、`bin_return_from_checkpoint`
- `src/supermimic/envs/mdp/commands.py`：`time_limit_steps`；`_finalize_episodes` 按 N* 归一化；`bin_return` 初值 1；困难采样去掉 `/ g_max`；`hard_mass_clip_end` 指标
- `src/supermimic/envs/mdp/motion_library.py`：`bin_is_clip_end`
- `src/supermimic/training/loop.py`：保存 / 加载 `bin_return_normalized`（旧 checkpoint 近似换算）；日志 `sampler/hard_mass_clip_end_bins`
- `scripts/sampler_report.py`（新）：checkpoint 采样器诊断
- `tests/test_sampler_normalization.py`（新）：归一化与 checkpoint 换算
- `scripts/render_video.py`（已在主 checkout 写好，随本 change 一并提交到分支）：诊断视频

与活动 change 重叠：`2026-10-24-training-throughput` 的 `commands.py`（T16 `_finalize_episodes` 去同步，已推迟）与 `loop.py`（T13 日志累积，已推迟）——本 change 从其分支 `c83f72f` 拉出，那两项之后在本 change 之上做。`2026-10-24-upstream-correctness-fixes`、`2026-10-24-paper-faithful-reproduction` 改过 `loop.py` 但已提交在 `c83f72f` 的祖先里，无并行修改。

## Tests
- [x] Unit tests — `tests/test_sampler_normalization.py`（T13、T14，CPU）
- [x] Integration tests — A 机全部测试通过（原写 B 机，按执行顺序改在 A）（含 `test_env_smoke`：return 采样路径跑通；`test_imagination` 训练冒烟）
- [x] Benchmark or performance tests — none（每次采样多一次 (num_bins,) 的乘加与每次 episode 收尾多一次逐元素运算；以 A 上训练日志的 it/s 与 B 对照，确认无明显变慢）
- [x] Manual validation — T16 复现诊断；T19 新采样器份额；T20 同迭代全量评测对比

## Contract updates
none

## Project updates
`docs/PROJECT.md`：回报采样器 G_b 为归一化完成度（0–1）、`sampler/bin_return_mean` 含义变化、新日志键 `sampler/hard_mass_clip_end_bins`、工具 `scripts/sampler_report.py` 与 `scripts/render_video.py`；`docs/CONCEPTS.md`：补"回报采样 / G_b"词条

## Experience updates
`docs/experiences/training/return-sampler-clip-end-bins.md`：未归一化的回报-to-go 受剩余步数支配，大数据集上困难采样被 clip 末尾短 bin 吸走——症状（困难动作 0 进步、困难份额集中在末尾 bin）、诊断（`sampler_report.py`）、修法（按可达回报归一化）

## Risks
- 困难片段比例上升后，训练早期整体回报可能偏低（按迭代对齐观察，不按时间）。
- 若空翻 / 侧滚失败主要来自世界模型预测或探索，修复后这两组仍可能无进步——届时转查世界模型（design Open questions）。
- 停掉 A 的 `bones_seq_fix@A` 后，fix 版与加速版的 BONES-SEED 同迭代对比只剩 36k/40k/44k 三个点（已有）。
- 新旧 checkpoint 的 `bin_return` 含义不同；旧 checkpoint 续训只能近似换算。

## Rollback plan
本 change 的代码在独立分支；回退即继续使用 `c83f72f`（旧采样器）。A 上若新采样器明显更差（同迭代失败数或回报显著变差），停止 `bones_seq_sampler@A`，A 改回跑 fix 版或其他实验；B 的 `bones_seq_fast@B` 一直作为旧采样器基线保留。
