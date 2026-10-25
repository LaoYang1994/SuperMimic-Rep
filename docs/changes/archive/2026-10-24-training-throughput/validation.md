# Validation

环境：机器 B（1×RTX PRO 6000，见 `docs/PROJECT.md` remote profile），远程 tree `.worktrees/2026-10-24-training-throughput`，
`PYTHONPATH=<tree>/src`（启动时打印的 `supermimic.__file__` 已核对）。LAFAN1 40 clip，`anchor_ori_now`，
起点权重 `lafan1_now_fixed@A/ckpt_0008000.pt`（第 8000 次迭代），迭代 = 更新。

## B. 基线（BASE = `0f4c446`，2026-10-24）

```bash
PY=scripts/perf/profile_train.py; CK=<correctness tree>/logs/lafan1_now_fixed@A/ckpt_0008000.pt
$PY --run-name tp_base@B --obs-mode anchor_ori_now --init-ckpt $CK --out logs/tp_base@B                                  # T6
$PY --run-name tp_base_stacks@B --obs-mode anchor_ori_now --init-ckpt $CK --sync-stacks --active 5 --out logs/tp_base_stacks@B  # T7
python scripts/perf/update_parity.py save --init-ckpt $CK --dir $SM_BANK/perf/parity_base                               # T8
python scripts/perf/update_parity.py run --dir $SM_BANK/perf/parity_base --out base_{a,b}.pt; ... compare base_a.pt base_b.pt
```

### T6 稳态速度
| 项 | BASE |
|---|---|
| 不开 profiler 的稳态（20 次迭代） | **754 ms/迭代（1.33 次/s）** |
| GPU kernel 时间（kernel 区间的并集） | 344 ms/迭代（约 46%） |
| kernel 发射 / 迭代（collect / wm / policy） | 2,030 / 12,580 / 41,480 |
| `cudaStreamSynchronize` / 迭代（collect / wm / policy） | 191 / 297 / 306 |

### T7 同步来源（`torch.cuda.set_sync_debug_mode("warn")`，5 次迭代平均）
torch 算子触发的同步：collect 173.5、wm 297、policy 306.5 次/迭代（和上表的 `cudaStreamSynchronize` 基本对得上）。

| 次/迭代 | 阶段 | 来源 |
|---|---|---|
| 280 | wm | `torch/optim/optimizer.py:102 _get_value`（Adam 每个参数读 `step`） |
| 280 | policy | 同上（策略 + critic 优化器） |
| 12 ×3 | collect | mjlab `reward_manager.reset`、`command_manager.reset`（两处） |
| 10 | wm | `supermimic/data/replay_buffer.py:119 _sample_from_mask`（每批一次） |
| 10 | policy | `supermimic/kinematics/rotations.py:104 projected_gravity` |
| 7.5 ×2 | collect | mjlab `circular_buffer.reset` |
| 7 | wm | `supermimic/world_model/wm_trainer.py:71 update_on`（指标） |
| 6 | collect | mjlab `termination_manager.reset` |
| 5.5 / 4 / 4 / 3 ×2 | collect | `replay_buffer.add`、`commands._log_step_rewards`、`recorder._write_rows`、`commands._finalize_episodes` |
| 3 ×7、2.5 ×13、… | collect | mjlab `entity/data.clear_state`、`tracking/mdp/rewards`、`manager_based_rl_env.step` 等 |

- Adam 的 560 次：`Trainer.load` 用 `torch.load(map_location=cuda)`（`training/loop.py:135`），优化器状态里的 `step` 被载入到 GPU；
  非 fused、非 capturable 的 Adam 不再移回 CPU，每个参数 `_get_value` 调用一次 `.item()` → 一次同步。从零开始的训练里 `step` 在 CPU，
  不会触发同步；从 checkpoint 起步或 resume 的训练都会触发。实际训练日志里 resume 前后的速度都是 1.3 次/s，说明这部分同步对墙钟时间影响不大
  （其他同步本来就会把 GPU 队列排空）。T9（fused Adam）会去掉这部分。
- 去掉 Adam 以后，我们自己代码里的同步：wm 约 17、policy 约 26.5、collect 约 15；其余 collect 里的同步来自 mjlab（仿真器 / 管理器内部，单独报告）。

### T8 对拍噪声基准
固定输入（`$SM_BANK/perf/parity_base/inputs.pt`）在两个**新进程**里各跑一次 WM 更新 + 策略/critic 更新：
所有 loss、梯度范数、想象指标的相对差 **0.00e+00**，三个网络参数的最大相对差 **0.00e+00**（逐位相同）。
对照：同一进程里先后跑两次（T5b selfcheck）相对差为 3.4e-3 / 0.84e-3 / 2.1e-3（wm / policy / critic）。

## C. 同步

```bash
bash tmp/chain_tp_candidate.sh <tag> <pytest paths>   # B 机：pytest → update_parity run/compare（对 base_a）→ profile_train --sync-stacks
```

### T9 fused Adam（tag `t9b`，2026-10-24 20:14）
第一次尝试（tag `t9`）**没有生效**：`Optimizer.load_state_dict` 用存档的 param_groups（`fused=None`）整个替换当前组，
从 checkpoint 起步后又回到默认实现（发射数、Adam 的 560 次同步都与 BASE 相同，对拍逐位相同，785 ms/迭代）。
加 `training/optim.py: load_optimizer_state`（保留 fused/foreach/capturable/differentiable）后重跑：

| 项 | BASE | T9 |
|---|---|---|
| pytest（optim_load、clip_and_step、world_model、imagination） | — | rc=0，12 passed |
| 不开 profiler 的稳态 | 754 ms（1.33 次/s） | **734 ms（1.36 次/s）** |
| GPU kernel 时间 | 344 ms | 331 ms |
| kernel 发射（collect / wm / policy） | 2,030 / 12,580 / 41,480 | 1,893 / 12,520 / 41,354 |
| torch 同步（collect / wm / policy） | 173.5 / 297 / 306.5 | 129 / **17** / **26.5** |

对拍（对 `base_a`，容差 = 同进程噪声基准 3.4e-3 / 0.84e-3 / 2.1e-3）：参数最大相对差 wm **2.7e-5**、policy **6.5e-4**、
critic **9.5e-4** → 通过。指标：WM loss 5.7e-6、`policy_grad_norm` 7.0e-3、`critic_loss` 1.2e-2（critic 拟合 20 个 minibatch
里的 loss 平均，参数微小差异逐步累积；绝对值 1.05e-2 对 1.07e-2）。

结论：去掉 560 次同步只省 20 ms（2.7%）——同步不是主要开销，CPU 发射（约 5.6 万次/迭代）才是；
后续收益主要应来自 D/E 组（编译融合、CUDA Graphs）。collect 阶段 173.5 → 129 是测量波动（collect 里 reset 次数随 episode 结束变化）。

### T10 + T11 无同步的非有限保护（tag `t10`，2026-10-24 20:25，提交 `b4848ee`）
- pytest（clip_and_step 含"跳过时权重、动量、`step` 不变"与"非 fused 优化器报错"、optim_load、imagination）：rc=0，7 passed。
- 对拍（对 `base_a`）：与 T9 **逐位相同**（无非有限梯度时计算不变）：参数最大相对差 wm 2.7e-5 / policy 6.5e-4 / critic 9.5e-4 → 通过。
- 稳态 739 ms（1.35 次/s），与 T9 的 734 ms 在测量波动内；torch 同步 wm 17、policy 26.5（不变：去掉的是 `bool(isfinite)` 两处，
  但 `float(skipped)` 等指标同步仍在，T12 才去掉）。collect 262.5（本次采样 episode 结束较多，mjlab reset 次数随之变化）。
- 结论：同步类优化收益已经接近为零，见下方"执行顺序重排"。

## 细分 profile（T9/T10 之后，2026-10-24 20:40，`profile_train.py` 子阶段标注：`policy.imagine`、`*.backward`）
实际耗时按 profiler 放大系数（profiled 1644 ms / unprofiled 764 ms ≈ 2.15）换算：

| 子阶段 | 发射/迭代 | 实际 CPU 时间（估） |
|---|---|---|
| policy.imagine（32 步前向） | 21,035 | ≈ 286 ms |
| policy.backward | 19,319 | ≈ 258 ms |
| policy 其余（critic 拟合、step、指标） | 1,028 | ≈ 18 ms |
| wm 前向 / wm.backward | 5,920 / 6,600 | ≈ 86 / 81 ms |
| collect | 2,148 | ≈ 35 ms |

想象前向的 eager 算子（每迭代）：`linear` 320、`silu` 256（策略 MLP 逐步 eager）、`slice`/`select` 2,675、`cross` 378（四元数 / goal / root error）；
反传：`t` 1,502、`add_` 1,430、`mm` 934。→ 先做 T18。另：`aten::multinomial` + `renormRows` ≈ 20 ms/迭代 GPU 时间（replay 采样，T14）。

## D. 编译融合

### T18 整个想象步一次编译调用（tag `t18`，2026-10-24 20:53）
预期：想象前向 + 反传 ≈ 544 ms → < 250 ms，整体 ≈ 450–550 ms/迭代。

| 项 | T10 | T18 |
|---|---|---|
| pytest（imagination 含新增 3 种口径的编译 vs eager、analytic_heads 含 goal 对仿真环境） | — | rc=0，12 passed（1333 s，编译测试慢） |
| 不开 profiler 的稳态 | 739 ms（1.35 次/s） | **640 ms（1.56 次/s）**，−99 ms |
| GPU kernel 时间 | 331 ms | 318 ms |
| 发射 policy.imagine / policy.backward | 21,035 / 19,319 | 19,121 / 16,380 |

对拍（对 `base_a`，容差 3.4e-3 / 0.84e-3 / 2.1e-3）：参数最大相对差 wm 2.7e-5、policy **3.2e-4**、critic **1.03e-3** → 通过。

实测只省 99 ms，**低于预期**：编译后的想象步每步确实只调用一次（Compiled Region 31 次/迭代），但 Inductor 生成的图本身仍有
数百个 kernel（30 个刚体的 FK 链、四元数、奖励与终止之间依赖多，融合不起来；反传图的 kernel 编号到 `_236`），每步前向 + 反传约 1,100 次发射。
→ 下一项改做 E 组 T22（CUDA Graphs 回放整个想象步，发射开销近 0）；预期策略阶段 ≈ 450 ms 中只剩 GPU 时间 ≈ 200 ms，整体 ≈ 390 ms/迭代。

## E. CUDA Graphs

### T22（想象步部分）CUDA Graphs 回放整个想象步（tag `t22`，2026-10-24 21:32）
`PolicyTrainerCfg.compile = "reduce-overhead"`（cudagraph trees：前向、反传各录一次），每次 rollout 开始 `cudagraph_mark_step_begin()`。
预期 ≈ 390 ms/迭代。

| 项 | T18 | T22 |
|---|---|---|
| pytest（`test_imagination.py -k compiled_imagined`，含 anchor_ori_now 的 reduce-overhead 对 eager） | — | rc=0，3 passed |
| 不开 profiler 的稳态 | 640 ms（1.56 次/s） | **435 ms（2.30 次/s）**，−205 ms |
| GPU kernel 时间 | 318 ms | 313 ms（GPU 忙碌 ≈ 72%） |
| 发射 policy.imagine / policy.backward | 19,121 / 16,380 | **1,394 / 1,262** |

对拍：与 T18 **逐位相同**（CUDA Graphs 回放同一批 kernel），参数最大相对差 wm 2.7e-5 / policy 3.2e-4 / critic 1.03e-3 → 通过。
日志中无 "skipping cudagraphs" 提示。剩余发射的大头变为 WM 阶段（前向 5,920 + 反传 6,600）→ 下一项 T19 + WM 窗口 CUDA Graphs。

### T19 + T22（WM 部分）整个 8 步窗口损失一次编译、CUDA Graphs 回放（tag `t19`，2026-10-24 22:00）
`WorldModel.enable_compile`：编译 `_window_loss`（k 步 + 指标），`WMTrainerCfg.compile = "reduce-overhead"`，每批前 `cudagraph_mark_step_begin()`，
指标 clone；`WorldModelTrainer.update` 的 10 批共用一个 `window_mask`（buffer 在批间不变，结果相同，无随机数）。预期 −80～−120 ms。

| 项 | T22 | T19 |
|---|---|---|
| pytest（world_model 含新增 CUDA Graphs 对 eager 的 loss / 指标 / 梯度对比、replay_labels、env_smoke） | — | rc=0，13 passed |
| 不开 profiler 的稳态 | 435 ms（2.30 次/s） | **371 ms（2.70 次/s）**，−64 ms |
| GPU kernel 时间 | 313 ms | 298 ms（GPU 忙碌 ≈ 80%） |
| 发射 wm / wm.backward | 5,920 / 6,600 | **549 / 20** |

对拍：参数最大相对差 wm **2.0e-5**、policy 3.3e-4、critic 1.3e-3 → 通过；无 "skipping cudagraphs"。
**达到设计目标 ≥ 2.5 次/s。** 剩余：371 − 298 ≈ 73 ms CPU 开销（collect 发射 1,932 与其中 mjlab 的同步、policy 其余 1,028）；
GPU 时间 298 ms 已成主要部分（上限约 3.4 次/s），其中 replay 的 `multinomial` + `renormRows` ≈ 20 ms（T14）。

## G. 训练级等价（T28，2026-10-24 22:22 – 10-25 00:42）
LAFAN1 40 clip，`anchor_ori_now`（+6 维朝向口径），1×RTX PRO 6000，迭代 = 更新（UTD 0.4），`total_iterations` 320k（学习率调度相同），
`--schedule.checkpoint-interval 2000`，评测 `scripts/eval_benchmark.py`（附录 F.1 协议），回报 = 论文 §4 归一化回报的 clip 均值 / 失败数。
- 快版 `lafan1_now_fast@B`：本 change `c83f72f`，机器 B，`--schedule.max-hours 2` → 停在第 17,350 次迭代（2.7 次/s，含启动编译）。
- 对照 1 `lafan1_now_fixed@A`：`c0df9d5`（修复，未加速），机器 A，2 h → 第 9,600 次迭代（1.3 次/s）。
- 对照 2 `lafan1_now_fixed_20k@A`：同 `c0df9d5`，机器 A，不限时，在第 20,000 次迭代后按 PID 停止（4.15 h）。

| 迭代 | 对照 1 | 对照 2 | 快版 | 快版 − 对照 2 |
|---|---|---|---|---|
| 2,000 | 0.412 / 18 | 0.415 / 16 | 0.450 / 13 | +0.035 |
| 4,000 | 0.578 / 10 | 0.575 / 11 | 0.644 / 4 | +0.069 |
| 6,000 | 0.666 / 10 | 0.703 / 4 | 0.699 / 6 | −0.004 |
| 8,000 | 0.740 / 4 | 0.738 / 3 | 0.693 / 8 | −0.045 |
| 9,600 | 0.768 / 3 | — | — | — |
| 10,000 | — | 0.758 / 5 | 0.716 / 6 | −0.042 |
| 12,000 | — | 0.776 / 5 | 0.768 / 5 | −0.008 |
| 14,000 | — | 0.765 / 7 | 0.754 / 4 | −0.011 |
| 16,000 | — | 0.796 / 1 | 0.815 / 2 | +0.019 |
| 17,350（快版 latest） | — | — | 0.792 / 2 | — |
| 18,000 | — | **0.821 / 1** | — | — |
| 20,000 | — | 0.818 / 2 | — | — |

结论：快版与未加速代码在相同迭代下的差值正负交替、均值 ≈ +0.002，幅度与两次未加速 run 之间的种子差异同量级（6k：0.666 对 0.703）
→ 未见系统性差异（每边单种子）。T28 通过。
附带结论（复现进度，不属本 change）：未加速代码在第 18,000 次迭代达到 0.821 / 1 失败（论文 1.2 h：0.82）——2 h 停在 0.77 是更新次数不够；
快版 2 h 只到 17,350 次，按 2.7 次/s 到 18,000 次需 ≈ 1.85 h。

## change-validate（2026-10-26）
重新运行（代码 `c83f72f`，远程 tree 已同步、`src` 无变化，`supermimic.__file__` 指向本 tree，CPU，`CUDA_VISIBLE_DEVICES=""`，机器 B）：
```bash
PYTHONPATH=<tree>/src CUDA_VISIBLE_DEVICES="" $PY -m pytest tests/test_optim_load.py tests/test_clip_and_step.py tests/test_replay_labels.py tests/test_quat_error.py tests/test_schedule.py -q
```
rc=0：**10 passed, 1 warning**（6.1 s）。GPU 全部测试：后续分支 `dd91147`（含本 change 全部代码）上 49 passed（2026-10-25，机器 A）。
速度与等价证据见上文（B 组基线、C/D/E 组每项对拍与 profile、G 组 T28），以及 BONES-SEED 36k/40k/44k fix 版 vs 加速版同迭代全量评测
（`2026-10-24-paper-faithful-reproduction` / 本 change 的全量运行：回报差 ±0.007、失败差 ±50，正负交替）。

## Known gaps
- 同步未降到个位数、编译缓存持久化 / 共享、重编译计数：推迟（plan"推迟 / 移出范围"）。
- `c83f72f` 本身未单独跑一次 GPU 全部测试（只在后续分支上跑过）。
- 多卡路径未实测。
- 训练级等价每边单种子。
