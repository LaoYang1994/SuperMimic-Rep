# Review

## Summary
采样器归一化按 design 实现：`sampler_math.normalized_returns_to_go`（每步回报-to-go ÷ 可达回报，N* 按 episode 实际结构确定）、
`bin_return` 改存完成度、困难采样去掉 `/ g_max`、末尾 bin 困难份额日志、checkpoint 兼容、诊断脚本与测试；远程验证 T15–T20 已完成
（A 上 49 项测试通过；同迭代全量评测失败数 −22% / −40% / −42%，回报持平）。只有 P3。
大文件检查（`git diff --name-only b660894..HEAD` + 未跟踪文件，> 1 MB）：无输出。

**Verdict: ready-to-archive**（archive 时完成 plan 的 Project / Experience 更新）。

## Coverage
- CLEAN — design 的 Proposed behavior / Design details 均有对应实现：N* 与归一化（`commands.py` `_finalize_episodes`，`sampler_math.py`）、
  `bin_return` 初值 1（`commands.py` `__init__`）、困难采样公式（`_sample_frames`）、`bin_is_clip_end`（`motion_library.py`）、
  日志 `sampler/hard_mass_clip_end_bins`（`loop.py`）、checkpoint 键与换算（`loop.py` save/load + `sampler_math.bin_return_from_checkpoint`）、
  `scripts/sampler_report.py`、`scripts/render_video.py`、训练实验（T17–T20）。
- plan T1–T20 都有对应产物；T12 的换算函数按 plan 修订放在 `sampler_math.py`（plan 已记录）。
- 未见超出 design 的改动。

## Findings
### P1
无。
### P2
无。
### P3
[P3] (confidence 6/10) src/supermimic/envs/mdp/sampler_math.py:19 — 每次 episode 收尾都用 `torch.tensor(gamma, device=rtg.device)` 新建一个 GPU 标量（一次小的主机到设备拷贝） → 改为 `gamma ** m`（`Tensor.__rpow__`，无需新建张量）。实测训练速度与旧采样器相同（2.4 次/s），影响可忽略。
[P3] (confidence 6/10) src/supermimic/envs/mdp/commands.py:237-241 — N* 的失败终止分支（`lens + rem_term`，受时长上限约束）只经 `test_env_smoke` / 训练间接执行，没有断言其数值的测试 → 以后补一个用小型 MotionLibrary 构造终止 episode 的测试，或把 N* 计算抽进 `sampler_math` 一并单测。
[P3] (confidence 5/10) src/supermimic/envs/mdp/commands.py:226-233 — 若启用 `bootstrap_fn`（训练中为 None），截断 episode 的 R_t 含 critic 自举的未来回报，按有限 N* 归一化会超过 1 并被截断 → 启用自举时改用"到 clip 结束"的可达值；当前不触发。
[P3] (confidence 9/10) src/supermimic/training/loop.py:28-30 — 移动 `bin_return_from_checkpoint` 后 `class Trainer` 前留下 3 个空行 → 删去 1 个。

## Unverified
- （置信 4/10）`Trainer._reset_episode_bounds` 用 `last_frame_of(frame) − frame + 1` 作 episode 可达长度；若重置时 frame 已先前进一帧，训练 / 评测的归一化回报分母可能差 1 步（约 0.2%）。已记入 design Open questions，不属本 change。
- （置信 4/10）fire-in-the-hole、立定跳远在新采样器下稳定略差的原因（推测：困难采样把练习集中到更难的动作上）。

## Resolved
- plan-review P1：剩余步数不再按帧号逐步推算（会差 1 步、偏差集中在 clip 末尾），改为按 episode 实际结构确定 N*（`commands.py:237-241`）。
- plan-review P3：归一化函数放在不依赖 mjlab 的 `sampler_math.py`，`tests/test_sampler_normalization.py` 只需 CPU。

## Accepted risks
- 失败终止的 N* 至多差 1 步（这些步完成度本就很低）。
- 旧 checkpoint 续训只能近似换算（`bin_return / g_max`），会打印警告。
- 结论为每边单种子；36k / 40k / 44k 三点一致，但没有多种子统计。
- 训练日志的 `train/episode_norm_return` 在新旧采样器之间不可比（训练分布不同），比较只用 benchmark（validation 已注明）。

## Follow-up items
- 上面 4 个 P3 → 下一个改动 `commands.py` / `sampler_math.py` 的 change 顺手处理。
- 稀少动作的访问与 EMA 更新慢（"按访问次数平均"候选修订）→ 新 change，视后续 checkpoint 而定。
- fire-in-the-hole 变差、剩余 139 条"其余动作"失败的分类 → 新 change（复现主线）。
- `2026-10-24-training-throughput` 推迟的 T16（`_finalize_episodes` 去同步）需在本 change 之上做。
