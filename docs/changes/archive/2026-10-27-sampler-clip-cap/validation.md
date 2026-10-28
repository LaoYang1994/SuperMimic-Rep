# Validation

口径：BONES-SEED 全量 benchmark（129,511 片段，`bones_seed_compact.pt`），单卡 RTX PRO 6000（A 机），`beyondmimic_seq`，
320k 调度、每 4k 存一次 checkpoint、seed 0；**迭代 = 更新**，所有对比按相同迭代。基线是同代码、不加上限的
`bones_seq_critic_alive@A`（critic 修复后的 main `dc7b29d`），以及更早的 `bones_seq_sampler@A`（旧 A）和 `bones_seq_alive_divH@B`（对照 −J/H）。
论文数字来自 p.18 Fig. A.1 矢量折线（1×H100，读图 ±3）。

## Commands
```bash
# 单元 + 全量测试（worktree 代码，PYTHONPATH 指向该 worktree 的 src）
cd $SM_REPO/.worktrees/2026-10-27-sampler-clip-cap
PYTHONPATH=$PWD/src $UV_PROJECT_ENVIRONMENT/bin/python -m pytest -q tests

# 训练（A 机，tmux `clipcap`，先跑全量测试，不通过不启动；见 tmp/A/clipcap/run.sh）
sm-train --run-name bones_seq_clipcap@A --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt \
  --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000

# 全量评测 + 分组（A 机，tmux `clipcapEval`，checkpoint 一出现就评；见 tmp/A/clipcap_eval/run.sh）
python scripts/eval_benchmark.py --ckpt logs/bones_seq_clipcap@A/ckpt_0044000.pt --library $SM_BANK/bones_seed/library/bones_seed_compact.pt
python scripts/eval_groups.py --benchmark logs/bones_seq_clipcap@A/benchmark_ckpt_0044000.json
# 同样对 0088000、0104000

# 跨 run 对比（本地 tmp/research/cmp_bench2.py，远程执行）
python cmp_bench2.py base=<critic_alive run dir>:44000 cap=<clipcap run dir>:44000 ...
```

## Results
- 全量测试，A 机，10-27 15:48 CST，代码 `0b4e2ab`：`67 passed, 1 warning in 212.56s`，exit 0（训练启动门槛，`tmp/A/tests/pytest.log`）。
- 全量测试，B 机，10-28 10:41 CST 起，同一份代码（归档前重跑）：见文末 "归档前重跑" 一行。
- 其中 `tests/test_sampler_cap.py` 5 个用例：上限与守恒、无超限时逐位相同、`cap=0` 逐位相同、组数 × cap < 1 时均分、
  组内比例不变与按比例再分配。
- 真实规模检查（T10，A 机，`bones_seq_critic_alive@A` 44k 的 `bin_return`，float32，约 13 万片段）：最大片段份额 0.01000，
  总和 1.000005，有效 bin 367.1、有效片段 90.2，与 design 模拟一致。

## Benchmark
### 失败数 / 回报（paper §4 口径），按迭代
| 迭代 | clipcap@A（本 change） | critic_alive@A（基线） | 旧 A | 对照 −J/H | 论文 1×H100 |
|---|---|---|---|---|---|
| 44k | 165 / 0.9329 | 166 / 0.9333 | 210 / 0.9340 | 192 / 0.9303 | 103 |
| 88k | **30 / 0.9429** | 70 / 0.9413 | 67 / 0.9406 | 82 / 0.9394 | ≈30 |
| 104k | **21 / 0.9433** | —（只有子集评测） | 58 / 0.9390 | — | 27（100k，Table A.8：25） |

训练用时（附带信息，不作比较依据）：clipcap@A 44k 5.5 h、88k 11.1 h、104k 13.3 h，2.2 次/s；论文 100k ≈ 3 h（9.5 次/s）。

### 分组失败数（`cmp_bench2.py` 分组）
| 迭代 / run | 侧滚 | 空翻 | fire-in-hole | 躺/起 | 其他 | 与基线共同失败 |
|---|---|---|---|---|---|---|
| 44k clipcap | 14 | 9 | 13 | 11 | 116 | 60 |
| 44k 基线 | 19 | 25 | 9 | 16 | 95 | |
| 88k clipcap | 0 | 0 | 4 | 3 | 22 | 10 |
| 88k 基线 | 9 | 4 | 8 | 3 | 46 | |
| 104k clipcap | 0 | 0 | 3 | 4 | 14 | 5（对旧 A） |
| 104k 旧 A | 4 | 3 | 6 | 1 | 42 | |

- 44k：总数不变，组成变了（空翻 25 → 9），"其他"类 +21；design 判定"弱"（121–165 的边缘）。
- 88k：各长度区间回报均上升（+0.0014 ～ +0.0044，越长越多）。104k 对旧 A：+0.0036 ～ +0.0082。
- 104k 剩余失败零散：rest 2、dance_basic_padeburee_270 2、fire_in_the_hole_360 2、hook_jump_off_wall 2，其余各 1。

### 采样器与训练健康（TensorBoard，各段中位数）
| 区间 | 训练回报 | 截前最大片段份额 | 截后有效片段 | 8 步 nse | critic_loss_max | 跳过 |
|---|---|---|---|---|---|---|
| 38k–60k | 0.71–0.74 | 0.34–0.50 | 172–199 | — | — | 0 |
| 64k–100k | 0.70–0.73 | 0.31–0.36 | 180–204 | 0.0086–0.0106 | 0.018–0.020 | 0 |
| 100k–116k | 0.71–0.73 | 0.31–0.33 | 198–225 | 0.0080–0.0086 | 0.018 | 0 |

基线同期训练回报 0.64，截前份额说明 `kick_back` 一直在被截。没有训练回报塌陷（基线类 run 在 146k 出现过），没有 NaN。

### 开销
`cap_group_mass` 每次调用 +0.29 ms（闭式解；弃用的 20 轮迭代版 +2.54 ms）。1,000 次迭代处训练速度 2.2 次/s，采集 0.094 s/次、
策略 0.235 s/次，基线 0.093 / 0.233。

## Manual checks
- 1,000 次迭代时新日志键 `sampler/hard_top_clip_share`、`sampler/hard_eff_clips` 都有记录，有效片段 100.7。
- `tmp/A/clipcap/progress.log` 记录了导入的代码路径指向该 worktree 的 `src/` 与提交号 `0b4e2ab`。
- 健康监控（`tmp/research/health.py`，每 10 分钟）全程 OK，无报警。

## Known gaps
- 单种子。44k 的失败数在四个旧 run 间波动 166–210；88k 的 70 → 30、104k 的 58 → 21 远超这个幅度，但仍没有第二个种子。
- 104k 之后（旧 run 退化、146k 训练回报塌陷的区间）尚未评测；148k 评测已排队（tmux `clipcapEval2`），结果写入 PROJECT.md，不改本 change 的结论。
- 单元测试只覆盖小规模（float64、≤200 组）；真实规模只手动检查过一次（review P3）。
- 镜像片段按独立片段计，一个动作最多 2n × cap（design 有意为之）。

## 归档前重跑
全量测试，B 机（GPU 空闲），10-28 10:41–10:45 CST，worktree 代码 `700ee1b`（与 main 合并提交 `4a35416` 内容相同）：
`67 passed, 1 warning in 192.27s (0:03:12)`，exit 0（`tmp/B/validate_tests.log`）。
