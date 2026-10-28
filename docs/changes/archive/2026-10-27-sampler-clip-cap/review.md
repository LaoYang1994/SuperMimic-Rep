# Review

## Summary
审查对象：分支 `change/2026-10-27-sampler-clip-cap` 的 `dc7b29d..700ee1b`，包括代码 `0b4e2ab` 和文档 `700ee1b`。
代码与 design 一致，没有 P1 / P2。

**Verdict（10-28 更新）：ready-to-archive。** 验证已完成，见 `validation.md`：88k 30 / 0.9429、104k 21 / 0.9433（基线 70、58），上限全程生效，无塌陷。

原 verdict（10-27）：not-ready。代码层面没有阻塞问题，原因只是验证还没完成：
- T15、T16：`bones_seq_clipcap@A` 的 44k、88k、104k 全量评测；
- T18：经验文档要在 T15 有结论后再写；
- T19：最终提交。

## Coverage
- **CLEAN：design 的每一点都有实现。**
  - 每个 clip 的困难质量上限：`MultiClipMotionCommandCfg.hard_clip_cap`，默认 0.01，0 即论文写法；
  - 配置链：`EnvCfg` → `loop.py` → `SuperMimicEnvCfgArgs` → command cfg，CLI 为 `--env.hard-clip-cap`；
  - water-filling：`sampler_math.cap_group_mass`，被截的 clip 设为上限，其余按比例放大，clip 内部的相对权重不变；
  - 记忆项用截后的 `hard` 更新：`commands.py` 中 `bin_memory` 在截断之后计算；
  - 两个新日志键：`sampler/hard_top_clip_share`、`sampler/hard_eff_clips`；
  - 片段数少时自动放宽为 `max(cap, 1/num_groups)`；uniform 和 start 模式不受影响，因为截断只在 return 分支里。
- **实现方式与最初的 design 有一处不同，已先改 design 再改代码**：固定 20 轮迭代实测每次调用 +2.54 ms，超过 plan 的 2 ms 门槛，改为闭式解（排序加前缀和，+0.29 ms/次调用，结果与迭代版一致：有效 bin 367.1，有效类别 90.2）。design 和 plan 都记录了原因。
- **CLEAN：已勾选的 plan 任务都有对应产物。**
  - T1–T14、T17 对应提交、测试、A 机上的检查记录、启动脚本、tmux 会话；
  - T15、T16、T18、T19 未勾选，属于验证和收尾，不是遗漏。
- **没有超出 design 的实现。**
- **大文件检查**：没有超过 1 MB 的文件。
- **`CONTRACTS.md`**：不受影响，其中没有约束采样器的条款。

## Findings
### P1
（无）

### P2
（无）

### P3
- [P3] (6/10) `tests/test_sampler_cap.py` — **单元测试只覆盖了小规模的情况**（多为 float64，最多 200 个组），没有覆盖 float32、约 13 万组这种真实规模下的舍入。
  - 真实规模只在 T10 手动检查过一次（A 机，44k 的 `bin_return`：最大 clip 份额 0.01000，总和 1.000005）。
  - 修法：加一个 float32、10⁵ 组、软最大值极陡的 CPU 测试，断言最大组 ≤ cap·(1+1e-5)，总和与 1 相差不超过 1e-4，覆盖 `fits[-1] = True` 这个保护分支。
- [P3] (5/10) `src/supermimic/envs/mdp/commands.py:272` — **镜像片段按独立片段计算**，所以一个动作有 n 个 take × 2 个镜像时，最多能拿到 2n × cap。例如 `kick_back` 有 4 个 take，最多 8%；在 44k 的模拟中实际拿到 5%。
  - 这是 design 有意的选择（不解析片段名），记录在此，便于以后需要时改为按 take 分组（模拟中按 take 1% 时有效类别 99 个，按 clip 1% 时 90 个，差别很小）。

## Unverified
- 上限是否能消除 146k 那种训练回报崩溃（困难质量突然转到某个失败率高的 bin），要训练过 146k 才能看到。本 change 的验证只到 104k（置信度 4/10）。

## Resolved
- 固定 20 轮迭代的开销超出门槛（+2.54 ms/次调用）→ 改为闭式解（`0b4e2ab` 提交前已改，见 Coverage）。

## Accepted risks
- 效果可能有限：评测中失败位置在训练里只是"中等难"（G 中位数 0.844），上限后失败组只拿到困难质量的约 5%（design 中已说明）。
- `sampler/hard_mass_clip_end_bins` 改为统计截后的质量，和旧 run 的曲线不直接可比（PROJECT.md 已注明）。
- 单种子：44k 的失败数在已有 run 之间波动 166–210。

## Follow-up items
- T15、T16：44k、88k、104k 的全量评测与 tb 检查 → `/change-validate`（`validation.md`）。
- T18：经验文档，在 T15 有结论后写。
- P3 两项：下次修改 `sampler_math.py` 或采样器分组时一起处理。
