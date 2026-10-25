# Design

## Background
BONES-SEED 全量（129,511 条，`beyondmimic_seq`，fix+加速 `bones_seq_fast@B` 与 fix `bones_seq_fix@A`）在 36k/40k/44k 次迭代：回报 0.93（论文 3 h >0.90），失败 350–450（论文 25）。失败集中在空翻 54/54、侧滚 22/22（三个 checkpoint 一条不少，论文 3 h 共 3 条）、跳跃、贴地动作。

诊断（B 的 44k checkpoint）：
- 视频（`scripts/render_video.py`）：空翻 / 侧滚全部因**高度误差 > 0.25 m** 失败，倾斜误差仅 0.2–0.4（阈值 0.8）；画面里参考已下腰 / 下蹲，机器人一直直立不尝试——策略学会了"不去做"。
- 采样器（checkpoint 里的 `bin_return` / `bin_memory`）：空翻 bin 的 G_b 平均 8.40（满分 10，与走路 8.38 相同），困难采样（40%）中空翻份额 **0.0000%**，实际采样概率是均匀份额的一半。困难采样 **93.9%** 落在 clip 最后一个 bin（12.9 万个，中位长度 7 帧），前 2,000 个困难 bin 中 96.2% 是 clip 末尾 bin。
- 原因：G_b 是每步折现回报-to-go 的滑动平均，其上限取决于该步到 episode 结束的剩余步数（clip 结束不自举、超时截断丢弃最后 100 步）。clip 末尾的 bin 剩余步数少，G_b 天然很低，被当成"困难"；真正难的动作在失败前能先跟几十步，G_b 反而中等。

论文（附录 D.1）："each bin b keeps a running estimate G_b of the discounted return from it … 40% from softmax_b(−25 G_b / G_max), where G_max is the largest attainable discounted return"。没有说明 clip 末尾如何处理；论文 3 h 空翻只失败 1 条，说明其实现里困难采样确实覆盖了这些动作（推测，无直接证据）。

## Goal
- 困难采样落在策略回报低于**可达回报**的地方：clip 末尾 bin 在跟得好时不再被视为困难；空翻 / 侧滚等失败动作获得显著的困难采样份额。
- 在相同迭代次数下，BONES-SEED 全量失败数（尤其空翻、侧滚、跳跃组）相对旧采样器下降；回报不下降。
- 采样器状态可诊断：记录困难份额在 clip 末尾 bin 上的占比。

## Non-goals
- 采样比例（50/40/10）、softmax 温度 25、bin 长度 1 s、EMA 系数、记忆系数的改动。
- clip 结束时用 critic 自举、合并末尾短 bin、改变重置帧范围（方案 B/C，未选）。
- 网络、奖励、终止条件、观测口径、评测协议的改动。

## Current behavior
- `MultiClipMotionCommand._log_step_rewards` 记录每步奖励与所在 bin；`_finalize_episodes` 对结束的 episode 计算每步折现回报-to-go R_t，超时截断的 episode 丢弃最后 `min_remaining_steps`（100）步（`bootstrap_fn` 训练中为 None），按 bin 求平均后 EMA 更新 `bin_return`（初值 `g_max` = 10）。
- `_sample_frames`：`hard = softmax(−25 · bin_return / g_max)`，p = 0.5·均匀 + 0.4·hard + 0.1·memory。

## Proposed behavior
- 每步的可达折现回报 A_t = g_max · (1 − γ^{m_t})，m_t = 该步之后 episode 在不失败时还会持续的步数 = min(到 clip 结束的步数, 到 episode 时长上限的步数)。（g_max = max_step_reward / (1 − γ) = 10，与现定义一致。）
- 每步的归一化回报 Ĝ_t = clip(R_t / A_t, 0, 1)；`bin_return` 改存 Ĝ 的 bin 平均的 EMA，初值 1.0。
- 困难采样 `hard = softmax(−25 · bin_return)`：与论文公式 −25 G_b / G_max 同形式，只是 G_max 换成每步的可达值。
- 跟得好的 clip 末尾 bin：R_t ≈ A_t → Ĝ ≈ 1，不再困难；在空翻起跳处失败：R_t ≪ A_t → Ĝ 低，获得困难采样。

## Design details
- 剩余步数按 episode 的实际结构确定，不按帧号逐步推算（plan-review：奖励在 `frame += 1` 前按当前帧记录、clip 末帧会被记录两次、重置时可能先前进一帧，按帧号推算会差 1 步，偏差恰好集中在 clip 末尾）：在 `_finalize_episodes` 中，令 episode 的"不失败时长度" N* = n（clip 播完或超时截断），或 N* = min(n + 终止帧到 clip 结束的步数, episode 时长上限)（失败终止，终止帧取此时的 `self.frame`）；m_t = N* − t。前两种情况精确，失败终止至多差 1 步（这些步 Ĝ 本就很低）。
- 归一化为不依赖 mjlab 的纯函数（便于 CPU 单测）：输入每步回报-to-go、每个 episode 的 N*、γ、g_max → 输出每步 Ĝ_t；`_finalize_episodes` 调用它，其余（截断丢弃、按 bin 聚合、EMA）不变。
- checkpoint 兼容：保存 `bin_return_normalized: True`；加载没有此键的旧 checkpoint 时，`bin_return / g_max` 近似换算并打印警告（旧值没有剩余步数信息，无法精确换算）。
- 日志：`sampler/bin_return_mean` 键名不变，含义变为归一化完成度（0–1），在 PROJECT.md 注明；新增 `sampler/hard_mass_clip_end_bins`（困难份额在 clip 末尾 bin 上的占比），每个 log interval 计算一次。
- 诊断工具：`scripts/sampler_report.py`（CPU）——从 checkpoint 的 `bin_return`/`bin_memory` 与库元数据，按动作组输出采样概率、困难份额、平均 G_b、末尾 bin 占比（本次诊断用的逻辑整理入库）。
- 训练实验：停 A 的 `bones_seq_fix@A`；A 上用本 change 的代码跑 BONES-SEED 全量 `bones_seq_sampler@A`（`beyondmimic_seq`、320k 调度、每 4,000 次存 checkpoint），与 B 的 `bones_seq_fast@B`（同代码、旧采样器）在相同迭代下对比。

## Edge cases
- m_t ≥ 1（t < n ≤ N*）；γ^{m} 直接用 `torch.pow` 计算，不依赖 `_gamma_pow` 的表长。
- R_t 可能因惩罚项为负 → Ĝ 截到 0；R_t 理论上 ≤ A_t（每步奖励 ≤ max_step_reward），截到 1 只防数值误差。
- 多卡：bin 统计按 rank 各自更新（现状），不变。
- `sampling_mode` 为 `start`/`uniform`（评测、play）时不受影响。

## Clarifications
- Q: 用哪种修法？ → A: 方案 A——按可达回报归一化（不选 B 自举 / C 合并末尾 bin）。
- Q: 验证实验放哪里？ → A: 停 A 的 fix 版全量；A 跑修复后的全量；B 继续作为旧采样器基线。
- Q: change 文档放哪里？ → A: 按现规定放主 checkout 的 `docs/changes/active/`，代码在 worktree `.worktrees/2026-10-25-sampler-attainable-return`（分支自 `change/2026-10-24-training-throughput` 的 `6919975`）。

## Assumptions
- 论文的 G_max 指"可达的最大折现回报"，按每步剩余时长取值不违背其定义；论文实现细节未知。
- 采样器是空翻 / 侧滚学不会的主要原因之一；若修复后这两组仍 0 进步，再查世界模型在这些动作上的预测精度（下一项诊断）。
- 评测用全量 129,511 条，与之前的 36k/40k/44k 结果直接可比（同评测代码）。

## Contract impact
none（不改 CONTRACTS；结果报告仍按迭代对齐，并注明采样器版本）

## Project impact
`docs/PROJECT.md`：回报采样器的 G_b 定义（归一化完成度）、新诊断工具 `scripts/sampler_report.py`、可视化工具 `scripts/render_video.py`；`docs/CONCEPTS.md`：补"回报采样 / G_b"词条。

## Experience impact
`docs/experiences/`：大数据集上回报采样被 clip 末尾短 bin 吸走（未归一化的回报-to-go 受剩余步数支配）——症状、诊断方法（困难份额在末尾 bin 上的占比）、修法。

## Open questions
- （plan-review P3，置信 4/10，不属本 change）`Trainer._reset_episode_bounds` 用 `last_frame_of(frame) − frame + 1` 作 episode 可达长度；若重置时 frame 已先前进一帧，训练 / 评测的归一化回报分母可能差 1 步（约 0.2%），之后单独核实。
- 修复后空翻 / 侧滚若仍无进步，是世界模型预测精度还是探索问题。
- 新采样器下回报是否会因困难片段比例上升而在训练早期略低（按迭代对齐观察）。
