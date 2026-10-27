# Validation

结论：**critic 侧目标达成；148k 的性能目标未达成，且原因不在本 change。**
- critic 侧：新 run 跑满 169.6k，critic 全程没有出现非有限值，也没有被跳过；失控的想象样本全部落在"已摔倒"样本上，不再进入 critic。
- 148k 性能：不如旧 A，有站立漂移。下降从 104k 之后开始，两个 run 都有，并伴随重置采样器被单个片段捕获。后续交给采样器 change 处理。

所有数字的设定都是：BONES-SEED 全量 129,511 个片段，`beyondmimic_seq`，单卡 RTX PRO 6000，迭代 = 更新，单种子（seed 0）。回报是论文 §4 的归一化回报；失败指触发了训练终止条件。

## Commands
```bash
# 测试
#   A 机，启动前由 run.sh 执行（代码 9597c99）：
PYTHONPATH=<tree>/src pytest -q tests
#   B 机，合并前的最终代码 3f136d9（与 9597c99 相比 src/、tests/ 无差异）：
PYTHONPATH=<tree>/src pytest -q tests

# 训练
sm-train --run-name bones_seq_critic_alive@A --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt \
  --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000                 # A 机，代码 9597c99
sm-train --run-name bones_seq_alive_divH@B ... --policy-trainer.objective-divide-by-horizon   # B 机对照，其余参数相同

# 全量评测（T26、T27、T28）
python scripts/eval_benchmark.py --ckpt <run>/ckpt_00{44,88,148}000.pt --library $SM_BANK/bones_seed/library/bones_seed_compact.pt
python scripts/eval_groups.py --benchmark <run>/benchmark_ckpt_<k>.json

# 子集评测（随机 2000 个片段，协议与全量相同，用来定位下降起点）
python scripts/eval_benchmark.py --ckpt <run>/ckpt_<k>.pt --library $SM_BANK/bones_seed/library/bones_seed_subset2000.pt --out <file>

# tb 全程检查：读全部 policy/* 记录点，统计非有限值、最大值、超阈值点和跳过计数（本地分析脚本，不在仓库中）
```

## Results
### 测试
- A 机，启动前（`9597c99`）：62 passed，exit 0。
- B 机对照，启动前（同一代码）：62 passed，exit 0。
- B 机，最终代码（`3f136d9`）：62 passed，1 warning，exit 0，用时 220.65 s；导入的确实是该 worktree 的 `src/`。

### T25：速度
- 整体 2.3 次/s，旧 A（`bones_seq_sampler@A`）为 2.4 次/s。
- 0–4k 次迭代的策略阶段：0.2328 s/次，旧 A 0.2212 s/次，多 11.6 ms（+5%）。没有达到计划"可以忽略"的预期，见 review P3-1。

### T27：44k、88k 全量评测

| 迭代 | 新 run（本 change，−J） | 对照（本 change，−J/H） | 旧 A（无截断、无屏蔽，−J/H） | B（只截断，−J） |
|---|---|---|---|---|
| 44k | 166 / 0.9333 | 192 / 0.9303 | 210 / 0.9340 | 199 / 0.9215 |
| 88k | **70 / 0.9413** | 82 / 0.9394 | 67 / 0.9406 | （53.4k 时已冻结） |

- **88k 分组（Table A.8 口径）**：

  | 组 | 新 run | 对照 | 旧 A |
  |---|---|---|---|
  | 侧滚 | 9 | 8 | 2 |
  | 空翻 | 4 | 7 | 5 |
  | fire-in-the-hole | 8 | 7 | 6 |
  | 交叉腿 | 0 | 3 | 2 |
  | 躺 / 起 | 3 | 6 | 13 |
  | 其余动作 | 46 | 51 | 39 |

- **88k body_check 回报**：新 run 0.967，对照 0.975，旧 A 0.977。
- **结论**：到 88k 为止，三种配置的回报相差不到 0.002，各项完成度一致，分不出差别。小组之间的差异在单种子的波动范围内。

### T28：148k 全量评测和 tb 全程检查
- **148k 全量评测**：

  | | 失败 / 回报 | body_check 回报 | body_check 根位置完成度 |
  |---|---|---|---|
  | 新 run | 125 / 0.8910 | 0.834 | 0.20 |
  | 旧 A | 80 / 0.9183 | 0.868 | 0.29 |

  新 run 的失败分组：侧滚 2、空翻 10、fire-in-the-hole 7、躺/起 7、其余 99（其中跳高 37、够物跳 20）。各长度区间都比旧 A 低 0.017–0.034。**"148k 时没有站立漂移"这一条不通过。**
- **子集评测**（随机 2000 个片段；子集与全量的差在 88k 为 0.0014，在 148k 为 0.0002）：

  | 迭代 | 88k | 104k | 140k | 144k | 148k | 152k | 156k |
  |---|---|---|---|---|---|---|---|
  | 新 run | 0.9427 | 0.9401 | 0.9127 | 0.9069 | 0.8912 | 0.8880 | 0.8838 |
  | 旧 A | | | | 0.9209 | 0.9190 | | |

  - 下降从 104k 之后开始。
  - 88k → 156k 的降幅按项拆分：根位置 0.084 → 0.055（满分 0.1），约占一半；身体朝向 −0.015；两项速度合计 −0.016；action_rate 和 joint_limit 两项惩罚变大约 3 倍。
- **tb 全程检查（0–169.6k，3,392 个记录点，每点是 50 次迭代的区间值）**：
  - **通过**：`critic_value_mean`、`critic_loss`、`critic_loss_max`、`critic_target_mean` 没有任何非有限值；估值最大 9.60（153,250 次），低于 G_max = 10。
  - **通过**：`critic_skipped_nonfinite`、`critic_skipped_nonfinite_grad` 全程为 0，critic 没有被跳过。
  - **通过**：`critic_targets_clamped_frac` 最大 4.6×10⁻⁷，几乎全程为 0，说明失控奖励都落在已判定摔倒的样本上，被屏蔽掉了。
  - **不通过（字面标准"全程 < 1"）**：`critic_loss_max` 有 8 个点大于 1：

    | 迭代 | 93,700 | 100,850 | 125,200 | 132,050 | 134,700 | 143,950 | 151,100 | 151,850 |
    |---|---|---|---|---|---|---|---|---|
    | `critic_loss_max` | 75.5 | 63.6 | 2.65 | 11.9 | 3.16 | 1.46 | 1.35 | 4.99 |

    全程中位数 0.018。这些尖峰都孤立出现，目标没有触顶，来自预测侧：某个仍算存活、但状态已失控的样本，critic 对它的预测很大。这是 design 列出的边界情况，之后的区间都恢复正常。
  - **预期内**：想象统计 `imag_action_abs_max`、`imag_reward_min` 有 12–14 个点非有限，最小奖励 −1.7×10³⁴，`imag_action_abs_max` > 100 的区间有 74 个。这正是这些日志要记录的失控事件。
  - **预期内**：策略步被跳过约 15 次，分布在 14 个记录点，第一次在 84,650 次。原因是已摔倒样本的 V_H 或奖励为 ±inf，ĉ·inf 得到 NaN，属于 design 的非目标。
- **对照 `bones_seq_alive_divH@B`（0–123k）**：critic 全部有限，`critic_loss_max` 最大 0.52；策略步被跳过约 5 次；想象失控事件从 58,850 次开始。
- **146k–157k 训练回报崩溃（与本 change 无关）**：
  - 训练回报从 0.66 跌到 1k 区间中位数最低 0.07，158k 恢复。回合长度只从 117 降到 110，说明很少摔倒。
  - 原因：重置采样器的困难部分（占全部重置的 40%）转移到 `kneeling_start_001__A122` 的一个 bin 上（回报 0.10），156k 时占困难质量的 86%。此前两个 run 的困难质量 76–96% 一直压在 `kick_back_001` 上。
  - 同期确定性评测没有塌（152k 0.8880，156k 0.8838）。
  - 期间 critic 估值升到 9.60，`critic_values_clamped_frac` 最大 1.1%（156,950 次），截断起了作用。
  - 对照在 106–107.5k 也出现过同类的小幅下跌（0.71 → 0.56）。

## Benchmark
- 速度：2.3 次/s，旧 A 2.4 次/s，策略阶段 +11.6 ms/次（+5%）。

## Manual checks
- 44k、88k、148k 都做了 `eval_groups.py` 分组统计，结果见上。
- 失控事件逐个核对了 `imag_reward_min`、`imag_action_abs_max` 和 critic 指标，critic 指标都没有同步异常。
- 用 checkpoint 里的 `bin_return` 还原了困难采样的分布：用 softmax(−25·G_b) 按片段汇总。

## Known gaps
- **148k 不如旧 A 的原因没有完全排除本 change。** 在同一子集上，新 run 144k 为 0.9069，旧 A 为 0.9209，差 0.014。后期退化在旧 A 身上也有（120k 之后），但新 run 退化得更多。这只是单种子的一组对比，需要在采样器 change 之后用多种子复核。
- 对照只跑到 123k，没有 148k 的数据。
- `c417edd` 的梯度保护在两个 run 中都没有触发过。
- 子集评测里失败数太少（1–3 个），只能看回报，看不了失败数。
