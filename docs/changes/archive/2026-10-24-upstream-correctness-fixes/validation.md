# Validation

环境：GPU 机（见 `docs/PROJECT.md` remote profile），远程 tree `.worktrees/2026-10-24-upstream-correctness-fixes`
（修复后，`PYTHONPATH=<tree>/src`，打印的 `supermimic.__file__` 已核对）；"修复前" = 远程主目录 `src/`（`64df684`）。

## Bug 确认（CPU，`CUDA_VISIBLE_DEVICES=""`，2026-10-24 15:5x）
| 测试 | 修复前（64df684） | 修复后 |
|---|---|---|
| `tests/test_replay_labels.py`（窗口终止标签 = 转移结果的终止） | **FAIL**：正样本比真实终止状态早一行（s_T 被标为终止） | PASS |
| `_sq_quat_err` 在 q1 == q2 处的梯度（float64，8 个随机四元数） | **非有限**（部分行 NaN） | 全部有限（~1e-16） |
| `tests/test_quat_error.py`（前向与 mjlab 一致 + 零误差梯度） | 不适用（新函数不存在） | PASS（2 项） |
| `tests/test_clip_and_step.py`（非有限梯度跳过） | 不适用 | PASS（2 项） |

命令：`PYTHONPATH=<src> $PY -m pytest tests/test_replay_labels.py tests/test_quat_error.py tests/test_clip_and_step.py -q`（修复后 5 passed）。

影响判断：
- 终止标签：**每次训练都生效**——每个以真实终止结束的 episode 多一个提前一步的正样本，直接作用于终止头与想象存活权重 ĉ。
- 四元数 NaN：**潜在**——只在两个旋转逐位相等时出现；想象中预测姿态与参考几乎不会逐位相等，主要防止一旦出现就写坏权重。

## GPU 验证（机器 A，tmux `validateA`，脚本 `<tree>/tmp/chain_validate_fixes.sh`，2026-10-24 16:40 起）
运行环境：远程 tree `.worktrees/2026-10-24-upstream-correctness-fixes`，启动时打印 `supermimic.__file__` =
`…/.worktrees/2026-10-24-upstream-correctness-fixes/src/supermimic/__init__.py`（修复后）；"修复前" 打印
`…/Projects/SuperMimic/src/supermimic/__init__.py`。代码 = `64df684` + worktree 改动，之后提交为 `bc23265`（修复）+ `fba904c`（同步脚本），内容与运行时同步的一致（远程无 git 元数据，运行日志里 commit 记为 n/a）。

```bash
PYTHONPATH=<tree>/src $PY -m pytest tests -q -rs                                   # T15
PYTHONPATH=<main>/src sm-train --run-name resume_old@A   ... --schedule.total-iterations 150 --schedule.checkpoint-interval 50
PYTHONPATH=<main>/src sm-train --run-name resume_old@A   ... --schedule.total-iterations 200 --resume logs/resume_old@A/ckpt_latest.pt
PYTHONPATH=<tree>/src sm-train --run-name resume_fixed@A ... (same two legs)      # T16
```

| 检查 | 结果 |
|---|---|
| T15 全部测试（修复后，含 CUDA） | **rc=0：39 passed, 1 warning**（344.7 s）；无跳过 |
| T16a 续训，修复前 | 第一段 rc=0；续训 **rc=1：`RuntimeError: replay buffer has no sampleable rows yet`** —— bug 复现 |
| T16b 续训，修复后 | 第一段 rc=0；续训 **rc=0**：`resumed … at iteration 150 (415 sim steps)` → `it 200 | sim steps 580`（415 + 40 warmup 补采 + 50 × 2.5 = 580 ✓） |

## LAFAN1 修复前后对照（T17–T19，按迭代对齐）
配置：LAFAN1 40 clip，`anchor_ori_now`（+6 维朝向口径），1×RTX PRO 6000（机器 A），迭代 = 更新（UTD 0.4），
`--schedule.max-hours 2 --schedule.checkpoint-interval 2000`；修复前 = `logs/lafan1_now`（`64df684`），
修复后 = `logs/lafan1_now_fixed@A`（`bc23265`）。两者都在 2.0 h 停在**第 9600 次迭代**（24,040 仿真步，1.3 次/s），
所以训练进度和训练时间都相同。评测：`scripts/eval_benchmark.py`（论文 §F.1 协议），回报为论文 §4 归一化回报的 clip 均值。
每边只跑一个种子；同一 checkpoint 两次评测的单 clip 回报差可达约 0.014（MuJoCo Warp 不是逐位确定的，见 experiences）。

| 迭代 | 修复前 回报 / 失败 | 修复后 回报 / 失败 | Δ 回报 |
|---|---|---|---|
| 2000 | 0.406 / 17 | 0.412 / 18 | +0.006 |
| 4000 | 0.544 / 10 | 0.578 / 10 | +0.034 |
| 6000 | 0.654 / 8 | 0.666 / 10 | +0.012 |
| 8000 | 0.712 / 5 | 0.740 / 4 | +0.028 |
| 9600（latest） | 0.766 / 2 | 0.768 / 3 | +0.002 |

训练日志（TB，取 ≤ 该迭代的最后一个点）：

| 迭代 | `policy/imag_survival_H` 修复前 / 后 | `wm/term_acc` 修复前 / 后 |
|---|---|---|
| 1000 | 0.796 / **0.916** | 0.9977 / 0.9993 |
| 2000 | 0.972 / 0.971 | 0.9996 / 0.9998 |
| 4000 | 0.995 / 0.996 | 0.9999 / 1.0000 |
| 8000 | 0.9995 / 0.9996 | 1.0000 / 1.0000 |
| 9600 | 0.9996 / 0.9997 | 1.0000 / 1.0000 |

`policy/skipped_nonfinite` 与 `policy/critic_skipped_nonfinite`（只有修复后才有）：192 个记录点全部为 0，整次训练没有跳过任何一步。

结论：
- 修复后终止头早期更准（第 1k 次迭代 term_acc 0.9993 对 0.9977），想象存活率明显更高（0.916 对 0.796），
  符合"去掉提前一步的错误终止正样本"的预期；第 2k 次迭代之后两者趋同。
- 评测回报在 4k–8k 次迭代时修复后高 0.01–0.03，到 9600 次迭代基本相同（0.768 对 0.766），失败数在 ±2 以内。
  单种子下这个差距和噪声处在同一量级，**不能**说修复提高了最终性能；能确定的是修复**没有造成退化**，早期学习更快。
- 2 h / 9600 次迭代下 LAFAN1 仍是 0.77，没有达到论文的 0.82：瓶颈不在这三处 bug。

## change-validate（2026-10-26）
重新运行（代码 `fba904c`，远程 tree 已同步，`supermimic.__file__` 指向本 tree 的 `src/`，CPU，`CUDA_VISIBLE_DEVICES=""`，机器 B）：
```bash
PYTHONPATH=<tree>/src CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_replay_labels.py tests/test_quat_error.py tests/test_clip_and_step.py -q
```
rc=0：**5 passed, 1 warning**（5.9 s）。
GPU 集成测试（全部测试）在包含本 change 全部代码的后续分支上重新运行过：`2026-10-25-sampler-attainable-return` 的 `7c18a10`，
机器 A，2026-10-25 12:37–13:03，**49 passed**（见该 change 的 validation）。本 tree 上 GPU 全量测试的最近一次为 T15（39 passed）。

## Known gaps
- 修复前后对照为每边单种子，结论按迭代对齐的趋势下（见上文）。
- 四元数 NaN 在以往全部训练日志中从未出现（扫描 TB 非有限值数量为 0），修复只是防护，未观测到训练层面的影响。
- 多卡路径（critic 跳过判断的 all-reduce，review P2）只有代码审查与单卡测试，未在多卡上实测。
