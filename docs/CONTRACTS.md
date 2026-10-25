# Contracts

This file defines stable system contracts and hard rules. It is a living
document — add or amend rules whenever you discover the need, no active change
required.

## 结果报告与论文对照
### Rules
- **先按相同训练进度比较**：任何两组结果（我们之间、我们与论文 / vibed / 上游）先在**相同的迭代（= 更新）次数**下比较；
  训练时间作为附带信息单独报告，允许不同，不作为对比的依据。论文给的是按时间的数字（如"3 h"）时，先说明或估计它对应的
  迭代次数，在该迭代次数下比较。曲线横轴用迭代次数。
- 每个要和论文或其他实验对比的数字，必须同时写明：观测口径（论文口径 / +6 维朝向 / +9 维位姿，
  见 `CONCEPTS.md#观测口径`）、数据集与 clip 数、GPU 型号与卡数、训练时长、迭代语义
  （见 `CONCEPTS.md#迭代与-utd`）。缺任何一项的数字不得用于结论。
- 不同观测口径、不同数据集、不同评测协议的结果不能直接对比，也不能放在同一列里不加标注。
  观测口径取值：`beyondmimic_seq`（论文口径）/ `anchor_ori_now`（+6 维朝向）/ `proprio_only`（Table A.1 字面）/
  `anchor_pose9`（+9 维位姿，仅作参照）。迭代语义固定为"1 次迭代 = 1 次更新"（`CONCEPTS.md#迭代与-utd`）。
- BONES-SEED 结果与论文对照时注明过滤后条数：本仓库 129,511 条（论文 129,514，名字过滤 129,667 两边一致）。
- 训练时长只计训练主循环，不计 checkpoint 评测时间（与论文 §4.1 一致）。
- 子集评测（如 BONES-SEED 2000 条）必须注明是子集，不能当作全量 benchmark 的结果。

### Examples
- "LAFAN1 40 clip，+6 维朝向（`anchor_ori_now`），1×RTX PRO 6000，第 18,000 次迭代（3.7 h），评测回报 0.821，1/40 失败，迭代 = 更新"

### Invalid examples
- "1.4 h 达到 0.83，已对上论文"（缺口径、卡型、迭代语义）
- 把 +9 维位姿的 0.904 和论文口径的 0.82 并列称为"超过论文"

## Benchmark 协议（论文附录 F.1）
### Rules
- 每条 clip 只从首帧整条播放一次；足底摩擦 1.0；不加域随机化、观测噪声和动作噪声；物理确定性；
  策略确定性执行。
- 失败 = 触发训练终止条件（`CONCEPTS.md#失败failed-motion`）；回报 = 归一化回报：奖励和 /（`max_r` × 整条 clip 的步数），
  失败的 clip 只累计到终止为止（所以提前失败得分低）。
- 改动 benchmark 代码（`src/supermimic/eval/`、`scripts/eval_benchmark.py`）必须开 change 并做 validate，
  且要用改动前后的同一 checkpoint 对比结果。

## 数值约定
### Rules
- 四元数一律 **wxyz**（`kinematics/rotations.py`）。外部数据（如 G1 重定向 CSV 的 xyzw）
  只在 `data/motion_convert.py` 的边界处转换。
- 控制频率 50 Hz；动作库统一重采样到 50 Hz。
- Replay 窗口的终止标签 `done[j]` 是第 j 个转移（s_j → s_{j+1}）结果的终止标签，即 s_{j+1} 是否触发终止；
  超时截断（clip 结束、时长上限）不算终止。`tests/test_replay_labels.py` 守护这条约定。
