# Validation

结论：**部分通过，由 change 2026-10-26-critic-alive-samples 补全后合并。**
- 只截断 critic 估值挡不住 critic 爆炸：B 在 53,350 次崩溃（T18 不通过）。
- 本 change 的代码（估值截断、−J 让裁剪生效、critic 小批量的非有限梯度保护）是后一个 change 的基础。合在一起后，新 run 跑满 169.6k，critic 没出现过 NaN，也没被跳过。
- −J 和 −J/H 在相同迭代下分不出差别，默认保留论文写法 −J。

所有数字的设定都是：BONES-SEED 全量 129,511 个片段，`beyondmimic_seq`，单卡 RTX PRO 6000，迭代 = 更新，单种子（seed 0）。

## Commands
```bash
# 测试（B 机，代码 211d661；按 remote profile 设 PYTHONPATH=<tree>/src）
PYTHONPATH=<tree>/src pytest tests                                           # T14
PYTHONPATH=<tree>/src pytest tests/test_clip_and_step.py tests/test_policy_clamp.py   # T23（c417edd）
# 最终代码（合并后的 3f136d9，包含本 change 全部提交）全量测试：见 change 2026-10-26-critic-alive-samples 的 validation

# 训练（B 机，tmux bonesClamp，代码 211d661，没有 c417edd 的梯度保护）
sm-train --run-name bones_seq_clamp@B --data.library $SM_BANK/bones_seed/library/bones_seed_compact.pt \
  --env.obs-mode beyondmimic_seq --schedule.checkpoint-interval 4000

# 评测（T17）
python scripts/eval_benchmark.py --ckpt logs/bones_seq_clamp@B/ckpt_00{44,88}000.pt --library $SM_BANK/bones_seed/library/bones_seed_compact.pt
python scripts/eval_groups.py --benchmark logs/bones_seq_clamp@B/benchmark_ckpt_00{44,88}000.json

# tb 全程检查（T18）：读 logs/bones_seq_clamp@B/tb 的 policy/* 全部记录点，统计非有限值、最大值和跳过计数
# （本地分析脚本，不在仓库中）
```

## Results
- **测试**：B 机全量 54 passed（T14）；T21/T22 的两个测试文件通过（T23）。
- **T16（约 1,000 次迭代时检查）**：
  - 速度 2.4 次/s，与 A（`bones_seq_sampler@A`）相同；策略阶段每 50 次迭代 11.0 s。
  - `policy_grad_norm` 全程中位数 1.0，旧 A 是 0.03，约 32 倍，与 H = 32 一致。
  - `grad_clipped` 全程中位数 0.04，说明裁剪会触发但不频繁。
  - 两个 clamped_frac 全程中位数都是 0（最大 0.0015 / 0.0018，都在 9,900 次）。
- **T17（全量评测）**：

  | 迭代 | B（本 change） | A（`bones_seq_sampler@A`，−J/H，无截断） |
  |---|---|---|
  | 44k | 199 / 0.9215 | 210 / 0.9340 |
  | 88k | 108 / 0.9288 | 67 / 0.9406 |

  - **B 的 88k 不能和 A 的 88k 比。** B 从 53,400 次起每一步都被跳过，策略冻结，所以 88k 实际是 53.4k 时的策略。
  - **B 的 44k 落后约 0.012。** 其他三个 run 的 44k 都在 0.930–0.934：A 0.9340；后一个 change 的新 run 0.9333，与 B 同样用 −J；对照 0.9303，用 −J/H。B 在 44k 之前的 critic 日志看不出异常，因此按单种子下 run 之间的差异处理。最初"B 落后来自梯度裁剪"的推测已经撤回，因为同样用 −J 的新 run 与 A 持平。
- **T18：不通过。**
  - **崩溃经过**：
    - 53,350 次，想象 rollout 失控：区间平均 |a| 1.5×10⁴，action_rate 项 −6.9×10¹³，与 critic 目标均值相同；critic loss 1.56×10²⁰。
    - 从 53,400 次起，`critic_value_mean` 和 `policy_grad_norm` 都是 NaN，共 970 个记录点。
    - `critic_skipped_nonfinite` 和 `skipped_nonfinite` 在这 970 个点上都是 1，约 48,450 次更新被跳过。
    - B 在 101,850 次手动停止（用户决定），所以没有 148k 的评测。
  - **原因**：已经摔倒的想象样本仍被 critic 拟合，奖励项 r̂ 没有下界，critic 对失控输入的预测也没有下界。估值截断只限住了自举项。
  - **另外**：B 跑的是 `211d661`，没有 `c417edd` 的逐小批量梯度保护。即使有保护，参数也只是不变成 NaN，失控样本照样会拖垮拟合。机制和修法见经验文档 `imagination-dead-samples`。
- **−J 与 −J/H（在后一个 change 中对比，同代码 `9597c99`，只差这一个开关）**：

  | 迭代 | −J（`bones_seq_critic_alive@A`） | −J/H（`bones_seq_alive_divH@B`） |
  |---|---|---|
  | 44k | 166 / 0.9333 | 192 / 0.9303 |
  | 88k | 70 / 0.9413 | 82 / 0.9394 |

  回报相差不到 0.003，各项完成度相同。两边都有想象失控事件：到 98.4k，独立事件约 14 次对 8 次，critic 都不受影响。

## Benchmark
- 速度：2.4 次/s，A 为 2.4 次/s，截断不影响速度。

## Manual checks
- T16 的日志量级检查已做，结果见上。
- T17 用 `eval_groups.py` 做了分组统计：B 44k 命名困难组失败 54、其余 145；A 分别为 71、139。

## Known gaps
- T18 的 148k 对比没做，因为 B 在 53,350 次崩溃，之后策略冻结。
- `c417edd` 的梯度保护从没在真正的崩溃里触发过：后一个 change 的两个 run 中 `critic_skipped_nonfinite_grad` 全程为 0。它只经过单元测试 T22 的验证。
- 只有单种子。B 的 44k 落后没有找到原因。
