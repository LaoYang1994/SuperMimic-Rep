# Plan

基线：主分支 `dc7b29d`（已合并两个 critic change）。代码在 worktree `.worktrees/2026-10-27-sampler-clip-cap`，分支 `change/2026-10-27-sampler-clip-cap`。
远程操作按 `remote-exp` skill 与 `docs/PROJECT.md` 的 remote profile；机器信息只在 `.compute_units`。
配置链：`config.EnvCfg` → `training/loop.py:72` 的 `SuperMimicEnvCfgArgs(...)` → `envs/tracking_env_cfg.py`（参数类 :82，构造 :162）→ `MultiClipMotionCommandCfg`（`envs/mdp/commands.py:419` 一带）。

## Implementation steps

准备
- [x] T1 — 从主分支 `dc7b29d` 建 worktree，分支 `change/2026-10-27-sampler-clip-cap`（由 `/change-implement` 询问后创建）· `.worktrees/2026-10-27-sampler-clip-cap`

纯函数（CPU 可测）
- [x] T2 — 新增 `cap_group_mass(h, group, num_groups, cap) -> (h_capped, mass_before, mass_after)`：
  - `h` 是 (N,) 概率，`group` 是 (N,) 的组号；`mass_before`、`mass_after` 是截前、截后各组质量，形状 (num_groups,)；
  - `cap <= 0`，或没有组超限时，原样返回 `h`，保证逐位不变；
  - 实际上限取 `max(cap, 1/num_groups)`；
  - 闭式 water-filling：组质量降序排序，前缀和求 s_k，取第一个满足 m_k·s_k ≤ cap 的 k；前 k 组设为 cap，其余乘 s_k；不做 host 同步。最初的固定 20 轮迭代实测 2.54 ms/次，已在 T10 后改写；
  - docstring 写明"论文未写"和用途 · `src/supermimic/envs/mdp/sampler_math.py`

采样器
- [x] T3 — `MultiClipMotionCommandCfg` 加 `hard_clip_cap: float = 0.01`，注释写：每个片段（镜像单独计）最多拿困难质量的这一比例，0 即论文写法，以及为什么（BONES-SEED 上困难质量被单个片段捕获）· `src/supermimic/envs/mdp/commands.py`
- [x] T4 — `_sample_frames`：`hard = softmax(...)` 之后调用 `cap_group_mass(hard, lib.bin_clip, lib.num_clips, cfg.hard_clip_cap)`，后续的 `hard_mass_clip_end`、`bin_memory` 更新和 p 都用截后的 `hard` · `src/supermimic/envs/mdp/commands.py`
- [x] T5 — 新指标 `hard_top_clip_share`（`mass_before.max()`）和 `hard_eff_clips`（截后组质量 m 的 `1/Σm²`），写法同 `sampling_entropy`；在 `__init__` 的指标键列表里加上这两个键 · `src/supermimic/envs/mdp/commands.py`

配置接线
- [x] T6 — `SuperMimicEnvCfgArgs` 加 `hard_clip_cap: float = 0.01`，并在构造 `MultiClipMotionCommandCfg` 时传入 · `src/supermimic/envs/tracking_env_cfg.py`
- [x] T7 — `EnvCfg` 加 `hard_clip_cap: float = 0.01`（CLI：`--env.hard-clip-cap`），`loop.py:72` 处传给 `SuperMimicEnvCfgArgs` · `src/supermimic/config.py`、`src/supermimic/training/loop.py`
- [x] T8 — 日志：在 `sampler/hard_mass_clip_end_bins` 旁边加 `sampler/hard_top_clip_share`、`sampler/hard_eff_clips` · `src/supermimic/training/loop.py`

测试
- [x] T9 — 新测试（CPU），文件 `tests/test_sampler_cap.py`：
  - 截后每组 ≤ 实际上限（容差 1e-6），总和为 1；
  - 没有组超限时与输入逐位相同（`torch.equal`）；`cap = 0` 时逐位相同；
  - 组数 × cap < 1 时，各组质量相等（= 1/组数）；
  - 组内 bin 之间的比例不变；
  - 多出的质量按原权重比例分给未超限的组（构造一个大组加若干小组，算出解析值对比）。
- [x] T10 — 用真实规模做数值检查（GPU，A 或 B）：取 `bones_seq_critic_alive@A` 44k 的 `bin_return`，调用 `cap_group_mass`，最大组质量 ≤ 0.01，有效片段数与 design 模拟一致（约 90 类、约 367 个有效 bin），同时记录每次调用的耗时（ms）· `tmp/A/clipcap/`

远程验证（A 机；worktree 运行设 `PYTHONPATH=<tree>/src`）
- [x] T11 — 同步 worktree 到 A（`scripts/remote/sync_code.sh`），确认 `supermimic.__file__` 指向该 worktree 的 `src/` · A 机
- [x] T12 — 照 `tmp/A/critic_alive/run.sh`（critic-alive worktree）写 `tmp/A/clipcap/run.sh`：先跑全量 `pytest -q tests`，不通过就不启动；记录导入的代码路径和提交号；然后 `sm-train --run-name bones_seq_clipcap@A --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000`（seed 默认 0）。在 tmux `clipcap` 里启动 · `tmp/A/clipcap/`
  - 结果：10-27 15:48 CST 启动：测试 67 passed（exit 0），代码 `0b4e2ab`，导入该 worktree 的 `src/`
- [x] T13 — 约 1,000 次迭代时检查：
  - 速度与 `bones_seq_critic_alive@A` 相当（2.3 次/s，策略阶段以外的采集阶段 ≤ +2 ms/次）；
  - 新日志键都在；`hard_top_clip_share` 有记录；`hard_eff_clips` ≥ 50。
  - 不合格就停下，回到 plan · `tmp/A/clipcap/`
  - 结果：1,000 次：2.2 次/s；采集 0.094 s/次、策略 0.235 s/次（基线同迭代 0.093 / 0.233）；`hard_eff_clips` 100.7，截前最大 clip 份额 0.057；健康检查 OK
- [x] T14 — 在 A 上启动自动评测（tmux `clipcapEval`）：44k、88k、104k 的 checkpoint 一出现，就做全量评测 + `eval_groups.py`；同时在本地起健康监控（`tmp/research/health.py`，每 10 分钟一次，含训练回报下跌报警）· `tmp/A/clipcap_eval/`
  - 结果：tmux `clipcapEval` 等 44k / 88k / 104k；本地健康监控每 10 分钟一次（含训练回报下跌报警）
- [x] T15 — 44k：与 `bones_seq_critic_alive@A`（166 / 0.9333）以及四个 run 的波动范围（166–210 / 0.930–0.934）按迭代对比失败数、回报、分组，再按 design 的判定标准下结论 · `tmp/research/`
  - 结果（BONES-SEED 全量 129,511 片段，单卡 RTX PRO 6000，44k 次更新，评测 10-27 21:30–23:01 CST，`tmp/research/cmp_bench2.py`）：失败 165 / 回报 0.9329（基线 166 / 0.9333）。总数在"弱"区间（121–165）的边缘，等于没变；分组变了：空翻 25→9、侧滚 19→14、躺/起 16→11，fire-in-hole 9→13、其他 95→116；两个 run 共同失败只有 60 个。片段长度 ≥400 帧的回报 +0.001～+0.003，<200 帧 −0.002。训练中截前 `hard_top_clip_share` 0.34–0.50，`hard_eff_clips` 172–199，训练回报 0.72（基线 0.64）。按 design：效果弱，88k 再看一次（T16）。
- [x] T16 — 88k、104k：与基线（88k 70 / 0.9413；旧 A 104k 58 / 0.9390）和论文（88k 约 30，100k 27）对比；从 tb 检查训练回报没有出现单个片段造成的崩塌，`hard_top_clip_share` 和 `hard_eff_clips` 的走势 · `tmp/research/`
  - 88k 结果（BONES-SEED 全量 129,511 片段，单卡 RTX PRO 6000，88k 次更新，评测 10-28 03:12–04:43 CST）：失败 30 / 回报 0.9429，基线 70 / 0.9413，旧 A 67 / 0.9406，对照（−J/H）82 / 0.9394；论文 1×H100 88k 约 30（Fig. A.1 读图 ±3）。分组：侧滚 9→0、空翻 4→0、fire-in-hole 8→4、躺/起 3→3、其他 46→22；与基线共同失败 10 个。各长度区间回报都上升（+0.0014～+0.0044，越长越多）。剩余失败：high_jump 5、fire_in_the_hole_360 3、dance_retro_dancing_in_the_rain 3。tb 60k–100k：训练回报 0.70–0.73 无崩塌，截前最大片段份额 0.31–0.36（kick_back 一直被截），截后有效片段 180–204，8 步 nse 0.0086–0.0106，critic 正常，无跳过。
  - 104k 结果（同口径，评测 10-28 05:23–06:54 CST）：失败 21 / 回报 0.9433；旧 A 104k 58 / 0.9390；论文 1×H100 100k 约 27（3 h）。分组：侧滚 4→0、空翻 3→0、fire-in-hole 6→3、躺/起 1→4、其他 42→14；与旧 A 共同失败 5 个；各长度区间回报 +0.0036～+0.0082。剩余失败零散（rest 2、dance_basic_padeburee_270 2、fire_in_the_hole_360 2、hook_jump_off_wall 2）。tb 100k–116k：训练回报 0.71–0.73，截前最大片段份额 0.31–0.33，截后有效片段 198–225，8 步 nse 0.0080–0.0086。104k 用时 13.3 h（2.2 次/s），论文同迭代约 3 h。

文档
- [x] T17 — PROJECT.md 回报采样一节：每片段困难质量上限、默认 1%、"论文未写"、两个新日志键；配置 `--env.hard-clip-cap`（0 即论文写法）· `docs/PROJECT.md`
- [x] T18 — 在 T15 有结论后，更新经验文档：扩展为"困难采样被少数 bin 捕获"（片段末尾、难而学不动的片段两种），给出诊断方法（从 checkpoint 的 `bin_return` 还原 softmax 分配并按片段汇总）和按片段设上限的做法 · `docs/experiences/training/return-sampler-clip-end-bins.md`、`docs/experiences/INDEX.md`

提交
- [x] T19 — 本地提交（每个逻辑单元一个）；提交前扫描被跟踪文件和提交内容，确认没有机器地址、端口、用户、路径；不推送 · `.worktrees/2026-10-27-sampler-clip-cap`
  - 结果：代码 `0b4e2ab`、文档 `700ee1b`；10-28 扫描分支 diff 无机器信息；`--no-ff` 合并到本地 main `4a35416`，未推送

## Files to change
- `src/supermimic/envs/mdp/sampler_math.py`：`cap_group_mass`
- `src/supermimic/envs/mdp/commands.py`：`hard_clip_cap` 配置、在 `_sample_frames` 中截断、两个新指标
- `src/supermimic/envs/tracking_env_cfg.py`、`src/supermimic/config.py`、`src/supermimic/training/loop.py`：配置接线与日志
- `tests/test_sampler_cap.py`（新）
- `docs/PROJECT.md`、`docs/experiences/training/return-sampler-clip-end-bins.md`、`docs/experiences/INDEX.md`

与活动 change `2026-10-27-mixed-precision` 的重叠：两者都改 `config.py`、`training/loop.py`，但位置不同（本 change 改 `EnvCfg` 和 sampler 日志，另一个改 `TrainConfig` 和 Trainer 构造）。合并时注意冲突，先合并的一方不影响另一方。

## Tests
- [x] Unit tests：T9（CPU）——A 机 9 passed
- [x] Integration tests：全量测试在 A 上通过（T12 的门槛；`test_env_smoke` 会走默认上限）——67 passed
- [x] Benchmark or performance tests：T10 单次调用耗时；T13 训练速度——+0.29 ms/次调用；训练速度不变
- [x] Manual validation：T15、T16 按迭代的全量评测与 tb 检查

## Contract updates
none

## Project updates
`docs/PROJECT.md` 回报采样一节：上限、默认值、"论文未写"、新日志键与 CLI 开关（T17）；BONES-SEED 的 Not supported 列表按 T15、T16 的结果更新（归档时）。

## Experience updates
`docs/experiences/training/return-sampler-clip-end-bins.md` 扩展为困难采样被少数 bin 捕获的两种情形与按片段设上限（T18）。

## Risks
- **效果可能有限**：评测失败位置在训练中只是"中等难"（G 中位数 0.844），上限后失败组只拿到困难质量的约 5%，大部分质量流向训练中更难的 bin（舞蹈序列、高速转身跑等）。T15 的判定标准已经考虑到这一点。
- **单种子**：44k 的失败数在已有四个 run 之间波动 166–210，只能识别较大的效果。
- **开销**：最初的 20 轮迭代实测 +2.54 ms/次调用，已改为闭式解（+0.29 ms/次调用，T10）；T13 训练速度与基线相同。
- **`hard_mass_clip_end` 的含义变化**：改为统计截后的质量，与旧 run 的曲线不直接可比（PROJECT.md 注明）。
- **占用 A**：跑到 104k 约 13 h，加上 3 次评测约 4.5 h。

## Rollback plan
`--env.hard-clip-cap 0` 即回到论文写法，输出逐位相同（T9 覆盖）；或不合并分支。
