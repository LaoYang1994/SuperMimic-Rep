# Review

## Summary
单卡稳态训练从 754 ms/迭代（1.33 次/s）降到 371 ms（2.70 次/s，LAFAN1，机器 B），BONES-SEED 全量训练 2.4 次/s（旧代码 1.2），
达到设计目标 ≥ 2.5 次/s（LAFAN1）。手段：fused Adam（并修复 checkpoint 加载时 fused 失效）、无同步的非有限保护、整个想象步一次编译 +
CUDA Graphs、WM 8 步窗口一次编译 + CUDA Graphs。每项都做了单次更新对拍（均在同进程噪声基准内），训练级等价按迭代对齐验证
（LAFAN1 T28 8 个点差值均值 +0.002；BONES-SEED 36k/40k/44k fix 版对加速版回报差 ±0.007、失败差 ±50，正负交替）。
大文件检查（`git diff --name-only 0f4c446~1..c83f72f`，> 1 MB）：无输出。

**Verdict: archive-with-fixes** —— 无 P1/P2；design 中"同步降到个位数""编译缓存共享""重编译计数"三点未实现，已按用户确认移出范围（plan 的"推迟 / 移出范围"），记为 Accepted risks / Follow-up。

## Coverage
- MISSING（已确认推迟）— design Proposed behavior："每次迭代的 CPU-GPU 同步降到个位数"：现为 wm 17、policy 26.5、collect ≈ 130（多为 mjlab 内部）；T12–T17 推迟。
- MISSING（已确认推迟）— design Goal："冷启动编译时间显著缩短（复用编译缓存），两台机器间共享"：T23–T24 推迟。
- MISSING（已确认推迟）— design Edge cases："记录重编译次数"：T21 推迟；实测训练日志无重编译迹象、无 "skipping cudagraphs"。
- CLEAN — 其余 design 点：速度目标、计算结果不变（对拍 + 训练级对照）、TF32 不变、profile / 对拍工具入库、多卡调用顺序保持（见 P3）。
- DRIFT（计划内修订，已记入 plan）— 执行顺序按 profile 开销重排（C 组同步 → D/E 组编译与 CUDA Graphs）；T3 的同步归因由 `with_stack` 改为 `set_sync_debug_mode`；新增 `training/optim.py`、`imagination/goals.py`。

## Findings
### P1
无。
### P2
无。
### P3
[P3] (confidence 7/10) src/supermimic/training/loop.py:207,216,225,230 — 每次迭代 3 次 `torch.cuda.synchronize()` 只为日志计时，打断 CPU/GPU 重叠 → 改用 CUDA event 计时、在 `log_interval` 统一读出（原 T13 的一部分，已推迟；T28 实测训练速度 2.7 次/s 与 profile 一致，代价目前很小）。
[P3] (confidence 6/10) src/supermimic/world_model/world_model.py:90、src/supermimic/imagination/compiled.py:83 — `cudagraph_mark_step_begin()` 是全局的：在持有上一次想象 rollout 输出（`ImagRollout` 各张量）期间调用 `wm.window_loss`，会让这些输出被下一次回放覆盖。当前 `Trainer.run` / `update_parity` / `profile_train` 的调用顺序（WM 更新 → 策略更新，rollout 输出在策略更新内用完）不会触发 → 在 `PolicyTrainer.update_on` 文档里注明这一约束，或在 rollout 输出离开 `update_on` 前 clone。
[P3] (confidence 6/10) src/supermimic/imagination/policy_trainer.py:147,162 — 多卡路径（`allreduce_max_` 的跳过标志、`fused_step` 的 `found_inf`）只经代码审查与单卡测试，未在多卡上运行 → 有多卡机器时跑一次 `torchrun` 冒烟。
[P3] (confidence 8/10) tests/test_imagination.py:69-70 — 3 种口径的编译 vs eager 测试每个都要构造环境并冷编译，T18 时整组测试 1333 s → 默认只跑 `anchor_ori_now` 的 reduce-overhead 一项，其余加 `slow` 标记。
[P3] (confidence 6/10) scripts/perf/profile_train.py:71,80 — 子阶段标注靠替换 `torch.Tensor.backward` 与 `policy_trainer.imagine`（只在该脚本进程内）；若在同一进程里再跑别的东西会被影响 → 脚本说明里注明"独立进程运行"。

## Unverified
- （置信 4/10）`c83f72f` 本身未单独跑过一次"全部测试"；全部测试是在其后续分支（含采样器改动）上跑的 49 passed。采样器改动不触及本 change 的文件，判断风险很低。

## Resolved
- 第一次 fused Adam 没生效：`Optimizer.load_state_dict` 用存档的 `fused=None` 覆盖当前参数组 → `training/optim.py:load_optimizer_state`（`tests/test_optim_load.py`）。
- profile 的同步归因：`with_stack=True` 20 次迭代 OOM（133 GB）且 runtime 事件无法归因 → `set_sync_debug_mode("warn")` 按（阶段, 调用位置）计数。
- 对拍脚本恢复 RNG 状态时 `map_location` 把 CPU RNG 搬到 GPU → 恢复前 `.cpu()`。

## Accepted risks
- 同步未降到个位数、编译缓存未持久化共享、重编译计数未做（用户确认推迟）。
- 训练级等价每边单种子；结论按迭代对齐的趋势下。
- `PolicyTrainerCfg.compile` / `WMTrainerCfg.compile` 默认改为 `"reduce-overhead"`：显存占用 BONES-SEED 约 27.5 GB（96 GB 卡上无压力），更小显存的卡需要留意。

## Follow-up items
- T12–T17、T20、T21、T23、T24 → 新的吞吐 change（`T12` 半成品在分支 `wip/2026-10-24-tp-t12-metrics`；`T16` 需在 `2026-10-25-sampler-attainable-return` 之上做）。
- 上面 5 个 P3 → 同一个后续 change。
- 评测速度（每次全量评测与训练并行 1–1.7 h）不在本 change 范围，另开 change。
