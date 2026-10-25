# Plan

约定（给零上下文的执行者）：
- 远程机器见仓库根目录 `.compute_units`（A、B 两台同配置 PRO 6000，共享 PFS），远程操作按 `remote-exp` skill 与
  `docs/PROJECT.md` 的 remote profile。**测速与开发在机器 B 上做**，A 上的正式实验不打断。
- 基线 = change `2026-10-24-upstream-correctness-fixes` 的提交 `fba904c`（下称 `FIXED`；修复本身是 `bc23265`，`fba904c` 另含远程同步脚本）；本 change 的第一个提交（T2–T5：工具 + 不改计算的 `update_on` 重构）记为 **`BASE`**，速度基线与对拍基线都在 `BASE` 上测。本 change 的分支
  `change/2026-10-24-training-throughput` 从 `FIXED` 拉出，worktree `.worktrees/2026-10-24-training-throughput`；远程 tree 同名，
  运行时 `export PYTHONPATH=$PWD/src` 并记录 `supermimic.__file__`。
- **速度的唯一依据**：`scripts/perf/profile_train.py` 的稳态测量——机器 B、从同一 checkpoint 权重起步、沉淀 60 次迭代后测 20 次，
  报告 ms/迭代（不开 profiler）、GPU kernel 时间、每阶段 kernel 发射与同步（`cudaStreamSynchronize` / `cudaMemcpyAsync`）次数。
  注意：`aten::item` 计数包括 CPU 张量（如 Adam 的 CPU `step`），不等于 GPU 同步；以 `cudaStreamSynchronize` 为同步指标。
- **数值等价的唯一依据**：`scripts/perf/update_parity.py`——用同一份保存下来的输入（WM 窗口批、策略起点批、模型/优化器状态、
  RNG 状态）在基线与候选代码上各跑一次 WM 更新与一次策略+critic 更新，比较 loss、梯度范数、更新后参数。
- 每一项优化（D 组的每个小节）都是：实现 → 等价对拍 → 测速 → 把数字写进 `validation.md`，再做下一项。

## Implementation steps

### A. 分支与工具
- [x] T1 — 从 `FIXED`（`fba904c`）新建分支与 worktree `.worktrees/2026-10-24-training-throughput` · git
- [x] T2 — 把 `tmp/scripts/profile_train.py` 移入仓库为 `scripts/perf/profile_train.py`，加 `--init-ckpt`、`--trace`（默认关）、按阶段的发射与同步计数（已在 tmp 版实现），并修正 GPU 忙碌时间只统计真实 kernel · `scripts/perf/profile_train.py`
- [x] T3 — 给 profile 工具加 `--sync-stacks`：在 `torch.cuda.set_sync_debug_mode("warn")` 下跑 `--stack-iters` 次迭代，把每个触发同步的 torch 算子归到最内层的 supermimic / mjlab 调用帧，按（阶段, 调用位置）聚合（原计划的 `with_stack=True` 方案：20 次迭代 OOM 到 133 GB，且 runtime 事件无法归因，已放弃） · `scripts/perf/profile_train.py`
- [x] T4 — 新增等价对拍脚本 `update_parity.py`：`--save <dir>` 从已训练 checkpoint 起步、填满 replay 后保存 10 个 WM 窗口批、1 个策略起点批、模型与三个优化器状态、RNG 状态；`--run <dir> --out <json>` 用这些输入跑一次 WM 更新与一次策略+critic 更新并记录 loss、梯度范数、参数摘要；`--compare a.json b.json` 输出最大相对误差 · `scripts/perf/update_parity.py`
- [x] T5 — 让 `WorldModelTrainer.update` 与 `PolicyTrainer.update` 可接受预先采样好的批（新增 `update_on(batches)` / `update_on(start)`，原 `update(buffer)` 内部采样后调用它），供 T4 使用；不改变计算 · `src/supermimic/world_model/wm_trainer.py`、`src/supermimic/imagination/policy_trainer.py`
- [x] T5b — 验证 T5 不改计算：同一 commit、同一 RNG 状态下，`update(buffer)` 与"先采样再 `update_on`"两条路径的输出逐元素相同；然后提交 T2–T5 为 `BASE` · `scripts/perf/update_parity.py`

### B. 基线测量（B 机）
- [x] T6 — 在 `BASE` 代码上跑稳态 profile（权重：`logs/lafan1_now_fixed@A/ckpt_0008000.pt`），记录基线数字 · `validation.md`
- [x] T7 — 跑 `--sync-stacks`，列出前 20 个同步来源及每次迭代的次数 · `validation.md`
- [x] T8 — 用 `update_parity.py --save` 在 `BASE` 上保存对拍输入（开着观测噪声，RNG 状态一并保存），并 `--run` 得到基线输出（`parity_base.json`）· `$BANK/perf/parity/`

### 执行顺序重排（2026-10-24，依据 T6/T9 的 profile）
T9 实测：去掉 560 次同步只省 20 ms（754 → 734 ms），同步不是主要开销；稳态 734 ms 里 GPU kernel 只占 331 ms，
其余约 400 ms 是 CPU 发射开销（约 5.6 万次/迭代：policy 4.1 万 ≈ 74%，wm 1.25 万 ≈ 22%，collect 0.19 万 ≈ 3%）。
因此按开销大小执行：**T18（想象步整体编译，policy 阶段）→ T20（critic minibatch 编译，policy 阶段）→ T19（WM 窗口编译）→ T21 → E 组（CUDA Graphs）**；
C 组剩余（T12–T17，T12 未测改动在分支 `wip/2026-10-24-tp-t12-metrics`）推迟到 E 组需要无同步时再做。每项开工前先写预期节省（ms），测后对照。

### C. 同步：每次迭代降到个位数
- [x] T9 — 三个优化器（WM、策略、critic）改用 `torch.optim.Adam(..., fused=True)`（`step` 与偏差修正在 GPU 上，单 kernel），对拍 + 测速；加载 checkpoint 时用 `load_optimizer_state` 保留当前优化器的实现标志（`Optimizer.load_state_dict` 会用存档里的 `fused=None` 覆盖，fused 悄悄失效；`Trainer.load` 与 `update_parity.py` 都改用它，测试 `tests/test_optim_load.py`）· `src/supermimic/world_model/wm_trainer.py`、`src/supermimic/imagination/policy_trainer.py`、`src/supermimic/training/optim.py`、`src/supermimic/training/loop.py`
- [x] T10 — 非有限梯度保护改为无同步：在 GPU 上算有限性标志，交给 fused Adam 的 `found_inf` 跳过机制（跳过时参数与动量不变），去掉 `bool(torch.isfinite(gn))`；`clip_and_step` 的测试相应更新 · `src/supermimic/imagination/policy_trainer.py`、`tests/test_clip_and_step.py`
- [x] T11 — critic 的有限性检查同样改为 GPU 标志 + `found_inf`，多卡时 all-reduce 这个标志张量（不转成 Python 数）· `src/supermimic/imagination/policy_trainer.py`

### D. kernel 数量：编译融合
- [x] T18 — 把策略前向、观测历史滚动、goal 构造并入 `ImagStep` 的编译图（一个想象步 = 一次编译调用）；观测噪声在 rollout 开始前于 eager 下一次生成全部 H 步并作为输入传入（保持现有随机数序列，见 design）· `src/supermimic/imagination/compiled.py`、`src/supermimic/imagination/rollout.py`
- [x] T19 — WM 的 k=8 步窗口损失整体编译为一个函数（现为每步一次编译调用 + eager 指标），连同 `wm_trainer.loss_on` 的参考帧准备（`lib.ref` 取数与 k 个 `ReferenceFrame` 拼装）· `src/supermimic/world_model/world_model.py`、`src/supermimic/world_model/wm_trainer.py`

### E. CUDA Graphs（视 D 组结果）
- [x] T22 — （已完成：想象步与 WM 窗口都用 `reduce-overhead`；优化器 step 未录进图）对 T18 的想象步与 T19 的 WM 窗口尝试 `torch.compile(mode="reduce-overhead")`（随机数已在图外生成）；若要把优化器 step 也录进图，学习率需改为张量（现 `set_lr` 每次迭代传 Python 浮点数）；对拍 + 测速，不达标或不等价则回退并记录原因 · `src/supermimic/imagination/compiled.py`、`src/supermimic/world_model/world_model.py`

### F. 编译缓存

### G. 验证
- [x] T25 — 全部测试在 B 上通过（含 T15 与更新后的 `test_clip_and_step`）· `tests/` （证据：包含本 change 全部代码的后续分支 `2026-10-25-sampler-attainable-return` `7c18a10` 上 A 机全部测试 49 passed，2026-10-25；本分支 `221ce2e` 上各项优化落地时分别跑过相关测试，见 validation）
- [x] T26 — 最终稳态 profile（LAFAN1 + BONES-SEED 紧凑库各一次），对比 T6 基线 · `validation.md` （证据：LAFAN1 最终稳态 = T19 的 371 ms / 2.70 次/s；BONES-SEED 用全量训练日志 `bones_seq_fast@B` 2.4 次/s，未单独跑 profile）
- [x] T27 — 最终 `update_parity` 对比 `parity_base.json` · `validation.md` （证据：最后一次代码改动 T19 的对拍——参数最大相对差 wm 2.0e-5 / policy 3.3e-4 / critic 1.3e-3，在噪声基准内）
- [x] T28 — LAFAN1 `anchor_ori_now` 2 h 训练（B 机，`lafan1_now_fast@B`），评测全部 checkpoint，按迭代次数与 `lafan1_now_fixed@A` 对齐比较 · `validation.md`

### 推迟 / 移出范围（2026-10-26，用户确认）
按 profile 开销排序后收益很小（T9 实测 560 次同步只值 20 ms），设计目标（≥ 2.5 次/s）已达到；转入后续 change。
`T12` 的半成品在分支 `wip/2026-10-24-tp-t12-metrics`；`T16` 需在 `2026-10-25-sampler-attainable-return` 之上做（同一函数）。
- T12 — 策略/critic/WM 的指标全部返回 GPU 张量（不再 `float(...)`、`.tolist()`），包括 `rollout.reward_terms` · `src/supermimic/imagination/policy_trainer.py`、`src/supermimic/imagination/rollout.py`、`src/supermimic/world_model/wm_trainer.py`
- T13 — 训练循环在 GPU 上累积指标与 episode 统计（含 `_collect` 里的 `done.any()` / `.tolist()`），只在 `log_interval` 时同步一次；日志键名与含义不变 · `src/supermimic/training/loop.py`
- T14 — replay 采样改为无同步：对有效掩码做 `cumsum` + `searchsorted(rand × total)` 在 GPU 上均匀抽样（等价于对有效行均匀抽样），去掉 `int(...item())` 与 `multinomial` · `src/supermimic/data/replay_buffer.py`
- T15 — 新增 CPU 单测：新采样只落在有效行上，且对有效行的经验分布与均匀分布一致（卡方检验，固定种子）· `tests/test_replay_sampling.py`
- T16 — command 的 episode 收尾（`_finalize_episodes`：`lens.max().item()`、`truncated.any()` 等）改为不需要 host 判断的写法或合并为一次同步 · `src/supermimic/envs/mdp/commands.py`
- T17 — 按 T7 的同步来源清单逐项处理**我们代码里**剩余的同步点（每处一个子任务记录在本行下方），直到我们代码引起的同步每次迭代 ≤ 10；仿真器（mjlab / MuJoCo Warp）内部的同步单独列出、不改 · 见清单
- T20 — critic 的单个 minibatch（前向、损失、反传、裁剪、step）编译为一个函数，循环 20 次调用 · `src/supermimic/imagination/policy_trainer.py`
- T21 — 记录每种 `obs_mode` 下的重编译次数（`torch._dynamo` 计数），确保训练稳态无重编译 · `scripts/perf/profile_train.py`
- T23 — 测量冷/热启动时间（清空与保留 `~/.cache/warp`、inductor 缓存各一次）· `validation.md`
- T24 — 在 `<env file>` 设定持久缓存目录（`TORCHINDUCTOR_CACHE_DIR` 放本地盘持久路径而非 `/tmp`），并写一个把缓存经 PFS 打成单个 tar 复制到另一台机器的脚本（同 venv 复制方式）· `scripts/remote/copy_compile_cache.sh`

## Files to change
- `scripts/perf/profile_train.py`（新，自 tmp 迁入）：稳态 profile、按阶段计数、同步调用栈
- `scripts/perf/update_parity.py`（新）：等价对拍
- `scripts/remote/copy_compile_cache.sh`（新）：编译缓存复制
- `src/supermimic/training/optim.py`（新）：`load_optimizer_state`，加载 checkpoint 时保留优化器实现标志（fused）
- `tests/test_optim_load.py`（新）：fused 在加载默认 Adam 的存档后仍生效
- `src/supermimic/world_model/wm_trainer.py`：`update_on`、fused Adam、GPU 指标
- `src/supermimic/world_model/world_model.py`：窗口损失整体编译
- `src/supermimic/imagination/policy_trainer.py`：`update_on`、fused Adam + `found_inf`、critic minibatch 编译、GPU 指标
- `src/supermimic/imagination/compiled.py`、`src/supermimic/imagination/rollout.py`：想象步整体编译、GPU 指标
- `src/supermimic/training/loop.py`：GPU 上累积统计，按 `log_interval` 同步
- `src/supermimic/data/replay_buffer.py`：无同步采样
- `src/supermimic/envs/mdp/commands.py`：episode 收尾去同步
- `tests/test_replay_sampling.py`（新）、`tests/test_clip_and_step.py`（更新）
- `docs/changes/archive/2026-10-24-training-throughput/validation.md`

与活动 change 重叠：`2026-10-24-upstream-correctness-fixes` 的 `policy_trainer.py`、`loop.py`、`replay_buffer.py`——本 change 建在其提交 `FIXED` 之上，不并行修改。

## Tests
- [x] Unit tests — （`test_replay_sampling.py` 随 T14/T15 推迟）`test_clip_and_step.py`（`found_inf` 跳过：参数与动量不变）
- [x] Integration tests — 全部现有测试通过（含奖励/观测对拍、世界模型、想象）；`update_parity.py` 候选对基线在容差内
- [x] Benchmark or performance tests — 稳态 profile：基线（T6）→ 每项优化后 → 最终（T26），LAFAN1 与 BONES-SEED；冷/热启动时间
- [x] Manual validation — LAFAN1 2 h 学习曲线按迭代次数与修复后基线对齐（T28）

## Contract updates
none

## Project updates
`docs/PROJECT.md`：remote profile 增加编译缓存位置与复制方式；Core modules 注明训练循环的同步策略（指标按 `log_interval` 同步）与 profile/对拍工具用法；Features 记录新的训练速度。

## Experience updates
`docs/experiences/`：PyTorch 中隐式同步的来源（非 fused Adam 的 CPU `step`、`.item()`、Python 分支上的 `.any()`、`multinomial`）与用带调用栈 profile 定位的方法；`torch.compile` / CUDA Graphs 在 BPTT 想象 rollout 上的限制。

## Risks
- fused Adam 与 foreach Adam 的浮点累加顺序不同，参数更新会有 ~1e-7 量级差异；训练级等价以 T28 判断。
- **对拍容差（2026-10-24 用户确认）**：冻结输入下新进程间逐位相同（T8 噪声基准 = 0），所以容差取**同进程噪声基准**——
  同一进程先后两次 `update` 的最大相对差（T5b：wm 3.4e-3 / policy 0.84e-3 / critic 2.1e-3）；候选对 BASE 的差不超过它即通过。
- `found_inf` 跳过依赖 fused Adam 的内部接口（GradScaler 使用的同一路径）；若接口在当前 torch 版本不可用，退回"每次迭代一次合并同步"的方案。
- 编译图变大后编译时间上升、重编译风险增加；动态形状（episode 末尾钳位、goal 维度）可能触发重编译。
- CUDA Graphs 与随机数、`torch.where` 跳过、BPTT 的反向图结合可能不可行；E 组允许部分或全部回退。
- 仿真器内部同步已有证据：采样阶段每次迭代约 209 次 `cudaStreamSynchronize`（每仿真步约 80 次），远多于我们代码的显式同步；本 change 不改 mjlab，速度上限可能低于 2.9。
- 无同步采样必须保证与原 `multinomial` 抽样同分布（均匀于有效行），T15 是门槛。

## Rollback plan
每一项优化单独提交，可逐项 `git revert`；任何一项等价对拍不通过即回退该项。最差情况回到 `FIXED`（正确性修复的 commit），其速度与已有实验一致。
