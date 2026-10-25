# Design

## Background
`Baojch/SuperMimic_Dyn`（同样基于 CuboiLeo `d9be385` 的 fork）在其 "Correctness" 改动中修了上游的若干问题。逐一对照本仓库代码（`2929b30`）核实，以下问题在本仓库同样存在：

1. **WM 终止标签错位一步**。`data/recorder.py` 写转移行时，`done` 是该行转移结果 s_{t+1} 的终止标签（`_write_rows(ids, done_next)`），终端行也写 `done=terminated`；但 `data/replay_buffer.py:135` 取窗口标签用 `done[ee[1:], tt[1:]]`（下一行的 `done`）。于是窗口中状态 s_{j+1} 拿到的是 s_{j+2} 的标签：终止前一步的 s_T 被标成终止，s_T 与 s_{T+1} 都成正样本。终止头因此学成"提前一步终止"，想象中的软存活权重 ĉ 偏悲观，进而影响 Eq. 2 目标与 critic 的 GAE 目标。
2. **四元数零误差处 NaN 梯度**。`envs/mdp/rewards.py:_sq_quat_err` 用 mjlab 的 `quat_box_minus`，其小角度分支用 `torch.where(angle > eps, sin(half)/angle, taylor)`：前向不受影响，但被丢弃分支在 angle=0 时为 `sin/0`，反向经 `where` 传回 NaN。该函数是根部朝向与刚体朝向奖励（仿真侧与想象侧共用）的核心，BPTT 梯度经过它。精确零误差少见，但一旦出现，NaN 会写进策略权重（`PolicyTrainer.update` 只做裁剪，无非有限保护）。
3. **续训时 replay 为空**。checkpoint 不保存 replay buffer；`Trainer.load` 恢复 `sim_steps` 后，`run()` 的 warmup 条件 `sim_steps < warmup_sim_steps` 不成立而被跳过，首次更新即从空 buffer 采样。上游（按迭代计数）同样如此。

## Goal
- 终止头训练标签与转移对齐：窗口中 s_{j+1} 的标签 = 第 j 个转移的结果是否终止；超时截断不算终止。
- 四元数误差函数前向结果不变、在零误差处梯度有限。
- 续训时先用当前策略重新采满 warmup 所需的仿真步，再开始更新。
- 策略 / critic 更新遇到非有限梯度范数时跳过该步并计数记录，而不是写坏权重。
- 量化修复对 LAFAN1 训练的影响（同配置对照）。

## Non-goals
- 训练吞吐优化（另开 change）。
- 任何超参、网络结构、算法目标的改动。
- 评测"完整播放 N vs N−1 步"的语义（本仓库实测为 N 步，分母已与之一致，见 paper-faithful change 的 validation）。
- BONES-SEED 重跑、SONIC 对照。
- SuperMimic_Dyn 的其他改动（数据筛选、均衡采样、机器人去手、ONNX 等）。

## Current behavior
- 窗口 `done[j]` = 第 j+1 行的 `done`（错位一步）；终端行 `done=terminated`。
- `_sq_quat_err` 在 q1==q2（±q）时反向为 NaN。
- `--resume` 后不做 warmup，buffer 为空。
- 策略与 critic 的 `opt.step()` 不检查梯度是否有限。

## Proposed behavior
- 窗口 `done[j]` 取第 j 行（转移行自己的 `done_next`）；终端行只作为最后一个目标状态，`done` 置 0、不携带转移标签。时间截断行 `done` 保持 0（现状已是）。
- 新增本地 `quat_box_minus` 等价实现：与 mjlab 前向逐元素一致，小角度分支的除数在被丢弃时替换为 1，使反向有限；`_sq_quat_err` 改用它。
- `run()` 在续训后、首次更新前补采 `warmup_sim_steps` 个仿真步（按"本次运行已采步数"判断，不依赖恢复的 `sim_steps`）。
- `PolicyTrainer.update`：策略梯度范数非有限时跳过 `opt.step()`；critic 在每次更新开始拟合前检查一次其输入与 GAE 目标是否全有限，否则跳过本次 critic 拟合；跳过次数进日志（`policy/skipped_nonfinite`、`policy/critic_skipped_nonfinite`）。（implement 中修订：原写"critic 每个 minibatch 检查"，每次更新会多 20 次 CPU-GPU 同步，而同步正是已知吞吐瓶颈；critic 的输入/目标是 detach 的 rollout 量，有限则其梯度经裁剪后有限，每次更新查一次即可。）

## Design details
- 标签修复同时改 `replay_buffer.sample_windows` 与 `recorder.record_pre_reset`（二者必须一起改，单改一处会丢正样本或重复）。WM 训练、`online_error` 等所有读 `WindowBatch.done` 的地方语义统一为"第 j 个转移结果的终止标签"。
- 四元数函数放在 `envs/mdp/rewards.py`（唯一调用者），不改 mjlab。
- 非有限保护：策略侧复用 `clip_grad_norm_` 返回的总范数（`torch.isfinite(gn)`），与现有 `float(gn)` 指标是同一处同步；critic 侧每次更新一次有限性检查。整体每次更新新增至多 1 次同步。
- 影响对照：修复后用与 `logs/lafan1_now` 完全相同的命令与配置（anchor_ori_now、2 h、ckpt 每 2000 迭代）重跑，训练后评测全部 checkpoint，与修复前逐点比较回报、失败数、终止头精度（`wm/` 日志中的终止分类准确率）。

## Edge cases
- 第一个转移即终止（episode 长 1）：窗口里唯一的转移行标签为 1，终端行为最后目标状态。
- 超时截断：转移行 `done=0`，终端行仍写入（作为最后目标状态）且 `done=0`。
- 四元数 q 与 −q 表示同一旋转：现有符号折叠保留，零误差判断对两者都成立。
- 多卡：跳过 `opt.step()` 必须所有 rank 一致，否则参数分叉——用 all-reduce 后的梯度判断（梯度已 all-reduce，故各 rank 的范数相同）。
- 续训时 return-based 采样器状态（bin_return）已从 checkpoint 恢复，补采的 warmup 数据按当前采样分布产生。

## Clarifications
- Q: 修复对结果的影响怎么验证？ → A: 单元测试 + LAFAN1 `anchor_ori_now` 2 h 重跑，与修复前的 `lafan1_now` 同配置对照（排在当前 BONES-SEED 运行之后）。
- Q: 实现与验证的时机？ → A: 现在只实现（本地写代码与测试，不碰 GPU 机）；所有需要 GPU 的验证（CUDA 单元测试、LAFAN1 对照重跑）等当前 BONES-SEED 运行与评测结束后再做，以免干扰其计时。

## Assumptions
- 续训 warmup 问题与非有限梯度保护一并纳入（都是正确性保护，改动小）。
- 终止头标签修复后，终止头的正样本数大约减半（每个终止 episode 从 2 个变 1 个），不调整 BCE 权重——论文 Eq. 1 是未加权 BCE。
- 修复不影响已完成实验的结论记录；它们按"修复前"标注保留在 paper-faithful change 中。
- 本 change 的代码在独立分支 `change/2026-10-24-upstream-correctness-fixes`（从 `2929b30`）上实现；是否用 worktree 在 `/change-implement` 时决定。

## Contract impact
none（不改 CONTRACTS 中的规则；可补一条"WindowBatch.done 是第 j 个转移结果的终止标签"作为数据约定，若 review 认为需要）

## Project impact
`docs/PROJECT.md` 世界模型 / 数据模块描述补充终止标签语义；Features 中注明 LAFAN1 结果分"修复前 / 修复后"。

## Experience impact
`docs/experiences/`：fork 之间横向对照能发现共同上游 bug；`torch.where` 两支都参与反向、被丢弃分支的 NaN 仍会传播。

## Open questions
- 修复后终止头若明显"更乐观"（存活权重变高），想象回报的尺度会变化；是否影响 critic / 学习率的适配，要看对照实验。
