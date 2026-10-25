# Design

## Background
稳态 profile（机器 B，LAFAN1 `anchor_ori_now`，从 `lafan1_now` 第 8000 次迭代的权重起步，`tmp/scripts/profile_train.py`，20 次迭代；数据见 change `2026-10-24-paper-faithful-reproduction` 的 validation）：
- 758 ms/迭代（1.32 次/秒），其中 **GPU kernel 时间 343 ms**，利用率约 45%；其余约 415 ms 是 CPU 侧开销。
- 每次迭代约 **60,200 次 kernel 发射**（策略 41,449 / WM 12,580 / 采样 2,107）、**约 631 次 `aten::item`**（WM 297 / 策略 291 / 采样 43）、约 810 次 `cudaStreamSynchronize`。
- 若完全隐藏 CPU 开销，速度约 1000/343 ≈ **2.9 次/秒**，与 vibed 在同款卡上测得的 3.1 次/秒吻合——差距来自实现（发射 + 同步），不是算力或显存带宽。
- 每次启动另有约 6–7 分钟的 MuJoCo Warp / `torch.compile` 冷编译（热缓存约 1 分钟），期间 GPU 利用率约 0。

吞吐直接限制实验：同样 2 h / 3 h 只能做约 0.4× 的更新，LAFAN1 未达 0.82、BONES-SEED 失败数偏高很可能与此有关；也决定了每张卡一天能跑多少组实验。

## Goal
- 稳态训练速度 **≥ 2.5 次/秒**（LAFAN1 `anchor_ori_now`，1×PRO 6000，同一 profile 方法测量），并报告 BONES-SEED 配置下的速度。
- **计算结果不变**：同样输入下，一次 WM 更新与一次策略/critic 更新的损失、梯度、参数更新与优化前在数值容差内一致；训练指标（日志）语义不变。
- 训练按迭代次数对齐的学习曲线（LAFAN1 2 h 内的各 checkpoint 评测回报）落在优化前同代码重跑的噪声范围内。
- 冷启动编译时间显著缩短（复用编译缓存），并在两台机器间共享。
- 保留一个可复用的 profile 工具与基准流程，后续 change 可以复测。

## Non-goals
- bf16 / fp16 计算或任何精度改变（保持论文的 TF32）。
- 超参、网络结构、算法、观测口径的任何改动。
- 评测（benchmark）速度与评测协议。
- 多卡训练与多卡吞吐。
- 显存优化（显存不是瓶颈）。

## Current behavior
- 每次迭代：采 2.5 仿真步 → 10 批 WM 更新（每批采样窗口、8 步展开、反传）→ 1 次策略更新（32 步想象 + BPTT）→ critic 拟合（5 epoch × 4 minibatch）。
- `ImagStep` 只编译了 WM 动力学 + 解析奖励/观测；策略 MLP 前向、观测历史拼接、goal 构造、噪声、critic 都在 eager 下逐个 kernel 执行。
- 同步来源（已定位的显式部分）：replay 采样 `int(mask.sum().item())`（每批一次）、指标 `float(...)`（WM 每次更新一次聚合，策略 ~10 处）、`reward_terms.tolist()`、command 重置中的 `.item()`、`clip_and_step` 的有限性判断、critic 的有限性判断等；其余（大部分）来源待带调用栈的 profile 定位。
- 编译缓存在各机器系统盘的默认位置，每台机器、每次环境变化都冷启动。

## Proposed behavior
- **我们自己代码里**每次迭代的 CPU-GPU 同步降到个位数（只在必要的决策点，并尽量合并）；日志指标改为在 GPU 上累积、到 `log_interval` 才统一同步一次。仿真器（mjlab / MuJoCo Warp）内部的同步不在本 change 范围，单独统计报告——稳态 profile 中采样阶段每次迭代约 209 次同步（每仿真步约 80 次），远多于我们代码在采样路径上的几处显式同步，这部分可能压低速度上限。
- 想象 rollout 的每一步（策略前向 + WM 一步 + 奖励/终止 + 观测与 goal 构造）作为一个编译单元；观测噪声**留在图外**：每次 rollout 开始前在 eager 下一次生成全部 H 步的噪声并传入（inductor 编译后的随机数与 eager 不同，图内生成会改变随机数序列、破坏等价对拍，也不利于 CUDA Graphs）；WM 的 k 步窗口展开同理；critic 的 minibatch 循环在编译后的函数里执行。在形状固定的部分评估 CUDA Graphs（`torch.compile(mode="reduce-overhead")` 或手动捕获），以消除逐 kernel 发射开销。
- replay 采样改为无同步实现（例如在有效索引上直接随机取整，或维护有效行计数在 GPU 上），去掉每批的 `multinomial` 归一化开销（约 21 ms/迭代）。
- Warp kernel 缓存与 inductor 缓存放在可复用、两台机器共享的位置（在共享文件系统上的小文件问题需评估——若不宜放 PFS，则每台机器本地持久化并预热）。
- profile 工具纳入仓库（`scripts/`），输出稳态 ms/迭代、GPU kernel 时间、发射与同步计数（按阶段）。

## Design details
- **方法：先测量再改，逐项验证。** 每一项优化单独落地、单独测速，并用数值对拍测试确认结果不变；以 profile 工具的稳态测量为唯一速度依据（同一机器 B、同一 checkpoint 起点、同一迭代数）。
- **数值等价的判定**：固定随机种子与同一份 replay 内容，比较优化前后单次 WM 更新与单次策略/critic 更新的 loss、梯度范数、更新后参数（TF32 下给容差，CUDA Graphs 不改变数学）。训练级等价用 LAFAN1 2 h 对照（与修复后基线 `lafan1_now_fixed@A` 同配置），按迭代次数对齐比较各 checkpoint 评测回报。
- **日志指标**：保留现有键名与含义；只改变同步时机（累积到 `log_interval`）。
- **非有限梯度保护**（来自 upstream-correctness-fixes）保留语义；可改为在 GPU 上判断并用 `torch.where` 风格跳过更新（不再每次同步），或与日志同步点合并——须保证跳过时参数与优化器状态不变。
- **编译缓存**：明确每台机器的缓存路径与环境变量（写入 remote profile），冷/热启动时间分别记录。

## Edge cases
- 编译后的函数遇到形状变化（如 episode 末尾、不同 `obs_mode` 的 goal 维度）触发重编译：每种 `obs_mode` 固定形状，记录重编译次数，避免训练中途反复编译。
- CUDA Graphs 要求静态内存地址与无 CPU 分支：涉及 `torch.where` 跳过、随机数（噪声）与可变长度循环的部分需单独处理，必要时只对部分区域捕获。
- 非有限梯度保护若改为无同步实现，必须仍保证"跳过时优化器动量不变"。
- 多卡路径（data parallel）不作为目标，但不能被破坏（all-reduce 的调用顺序与次数保持一致）。
- 训练循环的 `max_hours` / `checkpoint_at_hours` 判断依赖墙钟，不受影响。

## Clarifications
- （plan-review 修订）数值等价对拍需要能喂入预先采样的批：先做一个不改计算的纯重构（`update_on`）连同工具作为 `BASE` 提交，基线测量与对拍基线都在 `BASE` 上做。
- Q: 是否允许 bf16 等改变数值精度的提速？ → A: 不允许；保持 TF32，只做不改变计算结果的优化。
- Q: 基于哪份代码？ → A: 基于 `2026-10-24-upstream-correctness-fixes` 的分支（其 worktree 改动先提交）；之后所有重跑都用"修复 + 优化"版本。

## Assumptions
- 速度目标 ≥ 2.5 次/秒（稳态，LAFAN1，1×PRO 6000），理论上限约 2.9；若受限于 MuJoCo Warp 或框架而达不到，如实记录瓶颈与达到值。
- 测速以机器 B 为准（与 A 同配置），A 上的正式实验不被打断。
- 评测（benchmark）同样慢，但不在本 change 范围。
- 仿真侧（mjlab / MuJoCo Warp 的 env.step）只做调用方式层面的优化（如去掉我们自己代码里的同步），不改 mjlab。

## Contract impact
none（不改 CONTRACTS 规则；结果报告仍按"迭代 = 更新"语义，并在吞吐变化后的实验里注明代码版本与速度）

## Project impact
`docs/PROJECT.md`：训练吞吐与编译缓存位置写入 remote profile / Core modules；新增 profile 工具的用法。

## Experience impact
`docs/experiences/`：PyTorch 训练里隐式同步的常见来源与定位方法；`torch.compile` / CUDA Graphs 在 BPTT 想象 rollout 上的可用性与限制。

## Open questions
- 仿真器内部同步（每仿真步约 80 次）在不改 mjlab 的前提下能否绕开（例如调用方式、批量化），以及它对速度上限的影响。
- 剩余约 600 次 `.item()` 的具体来源（待带调用栈的 profile）。
- CUDA Graphs 能否覆盖带随机噪声与可变终止的想象 rollout，还是只能覆盖部分。
- 编译缓存放 PFS 是否会遇到小文件慢的问题（缓存是大量小文件）。
