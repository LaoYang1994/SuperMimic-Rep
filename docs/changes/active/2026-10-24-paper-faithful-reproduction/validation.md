# Validation

GPU 机：1× RTX PRO 6000 Blackwell 96 GB，容器 64 核 / 128 GiB；torch 2.14.1+cu130、mujoco / mujoco-warp 3.16、warp-lang 1.18、mjlab `8bb2bc9`。

## 改动前基线（T8，未改动的上游代码）
- 命令：`$PY -m pytest tests -q`（2026-10-24 04:34–04:41）
- 结果：20 passed, 4 skipped。跳过项：3 个 `test_motion_convert`（当时测试路径 `data/raw/lafan1/g1` 下没有 CSV，之后已链接到 `$BANK/lafan1/csv`），1 个 `test_analytic_heads` 的条件跳过（短 rollout 内发生 reset）。

## 环境与数据记录
- LAFAN1：40 个 CSV，264,705 帧 @30 Hz → 40 个 npz，441,140 帧 @50 Hz（2.451 h），与论文 2.45 h 一致。
- BONES-SEED：`g1.tar.gz` 23,499,973,647 B，sha256 与 HF 一致；内含 142,220 个 CSV，与 metadata 行数一致。
- 名字过滤（仅 metadata）：SONIC 完整表 → 129,785；加补充词 `box_` / `fence` / `vault` → 129,667（= 论文），候选组合不唯一，待可行性过滤结果确定（T47）。

## 改动后测试（T36）
- 命令：`$PY -m pytest tests -q -rs`（改动后代码，05:34 与分母修正后 05:41 两次）
- 结果：33 passed, 0 skipped（新增：四种观测口径的仿真/想象 goal 逐元素对拍 ×4、`goal_layout` 维度 ×4、采集累加器 ×2；`test_motion_convert` 因链接了 LAFAN1 CSV 不再跳过）。

## benchmark 改动前后对比（T62，参照 checkpoint `logs/ref_upstream`，上游代码 8000 仿真步，口径 proprio_only）
- 改动前（上游 benchmark）：40 clips，失败 13，按执行步数归一化回报 均值 0.4477 / 步加权 0.4328。
- 改动后：失败 13；按执行步数（`norm_return_per_step`）0.4482 / 0.4330；**按论文定义（除以整条 clip 的最大可得奖励）均值 0.3570 / 长度加权 0.3721**。
- 发现：MuJoCo Warp 在 GPU 上逐次运行非逐位确定（同代码同 checkpoint 两次评测单 clip 回报差最大 0.0137、个别 clip 步数差 1）。T62 判据据此修订（见 plan T62）。
- 修订判据结果：(a) 改动前后单 clip 最大差 0.00534 ≤ 同代码两次评测间 0.00887；(b) 失败数 13 = 13；(c) 新分母算术 40/40 正确，未失败的 27 条 clip 完整播放步数 = 帧数（据此把分母从"帧数 − 1"更正为"帧数"，design 已同步）。→ **PASS**
- 含义：上游报告的 LAFAN1 回报（0.56 / 0.824 等）是按执行步数归一化的，与论文口径不同；同一个参照策略两种口径相差约 0.09。

## 训练吞吐（T38 起步，LAFAN1，beyondmimic_seq，4096 env，批 4096，H=32）
- 1.3 次更新/秒（= 迭代/秒），每次更新：采集 2.5 仿真步 0.06 s、WM 10 批 0.17 s、策略+critic 0.54 s；GPU 利用率 37%，显存 20 GB。
- 对照：vibed 同卡 3.1 次更新/秒；按此速度 2 h ≈ 9.4k 次更新，而 vibed 到 0.82 用了 ~12k 次更新 → 时间对齐判据（≤ 1.5 × 1.2 h）有风险。
- 训练期间 CPU 上同时在跑 BONES-SEED 流式转换（8 worker）；容器 64 核，训练进程单核满载，未见 CPU 节流（cpu.stat nr_throttled 6）。

## LAFAN1 主方案（T38/T41）：beyondmimic_seq，40 clip（2.45 h），1×RTX PRO 6000，迭代 = 更新，上限 2 h
命令：`sm-train --run-name lafan1_seq --data.npz-dir data/npz/lafan1 --env.obs-mode beyondmimic_seq --schedule.max-hours 2 --schedule.checkpoint-interval 2000`；评测 `scripts/eval_benchmark.py`（附录 F.1，整条 clip 播放），训练结束后进行。

| checkpoint | 训练时长 | 回报（论文 §4，均值 / 长度加权） | 回报（按执行步数，上游口径） | 失败 |
|---|---|---|---|---|
| 2000 | 0.43 h | 0.464 / 0.486 | 0.571 | 15/40 |
| 4000 | 0.85 h | 0.607 / 0.625 | 0.685 | 10/40 |
| 6000 | 1.28 h | 0.673 / 0.690 | 0.744 | 9/40 |
| 8000 | 1.70 h | 0.684 / 0.695 | 0.765 | 9/40 |
| latest（9450，停于 max_hours） | 2.00 h | **0.763 / 0.768** | 0.765 | **1/40** |

- 未达到论文 0.82 @ 1.2 h（H100）；2.0 h 时 0.763、仅 1/40 失败。相邻 checkpoint 间失败数波动大（8000 时 9 条 → 9450 时 1 条），单个 checkpoint 的评测噪声不可忽略。8 步 WM 误差（训练日志 `nse8`）1.7 h 时 0.023，高于论文 0.007–0.011（两者归一化尺度是否相同未确认）。
- 分项（ckpt 8000，每步均值，满分 0.02 / 根部 0.01）：根部朝向 0.0095（锚点朝向观测消除了偏航漂移）、刚体位置 0.0192、朝向 0.0152、线速度 0.0174、角速度 0.0169。
- 失败集中在长 clip 的早期终止（例：fight1_subject2 在 950/12244 步、fallAndGetUp1_subject5 在 666/8411 步），按论文定义丢掉其余全部奖励，是回报的主要损失来源。
- 按 design（Clarifications）：主方案未达 0.82 → 先跑 T39 `anchor_ori_now`、T40 `proprio_only` 再选口径。

## BONES-SEED 过滤（T46–T48），数据集 BONES-SEED G1 retarget，142,220 clip
- 流式转换（`stream_bones_seed.py`，120→50 Hz）：142,220 clip，0 失败，72 个分片（87 GB，PFS）。
- SONIC 完整关键词表 + 可行性过滤：名字过滤后 129,785（去 12,435）→ 可行性过滤后 129,605（去 180：elevated_support 124、ground_penetration 44、floating_feet 12）。
- 补充词搜索（17 个候选词、1–4 个的组合）：12 组能让名字过滤后**精确 = 129,667**；其中**无一**同时让可行性过滤后 = 129,514。最接近：`box_` + `jump_off_50cm` + `vault` → **129,511（−3）**，即我们的可行性过滤比论文多去 3 条（论文去 153，我们去 156）。差异归因于论文未给出的"直立 / 站立高度 / 静止"门限。选定该组（含论文举例词 box、jump off 50cm），不为凑数调可行性阈值。
- 时长：50 Hz 帧数折算 263.31 h（SONIC 表 + 可行性，未加补充词），论文 264.56 h；论文的时长折算口径未写明（按 120 Hz 原始帧数，SONIC 表后为 264.53 h）。
- **结论：名字过滤对上（129,667 ✓）；可行性过滤后 129,511 ≠ 129,514（差 3）——按 CONTRACTS 不得声称"对上"，与论文对比时注明。**

## Table A.8 分组大小（T57），在最终 129,511 条上
- 对上：side rolls 22 ✓（原 24，可行性去 2）、fire-in-the-hole 20 ✓、crossed legs 10 ✓、broad jumps 24 ✓。
- 对不上：flips 54（论文 44；flip_180_R 22、flip_360 20、flip_090 6、flip_360_hero_landing 6，任意组合凑不出 44）、lying/resting/standing up 64（论文 99）、dance routine 无恰好 8 条的 routine（论文 8）。论文未给分组规则，这三组的分组方式无法从公开信息确定；不影响训练与总失败数。

## 紧凑库（T49/T50）
- `bones_seed_compact.pt`：129,511 clip、47,353,779 帧（263.08 h @50 Hz）、6.72 GB fp16；`bones_seed_subset2000.pt`：2000 clip（seed 0）、4.27 h。
- 紧凑库的 `beyondmimic_seq` 输入（链式 FK 锚点朝向、关节窗口）与非紧凑库对拍（LAFAN1 6 clip，CPU）：锚点四元数最大差 3.8e-7，关节窗口 0；已补为测试 `test_compact_goal_windows_match_full`。

## LAFAN1 对照（T39）：anchor_ori_now，同上设置（40 clip、1×RTX PRO 6000、迭代 = 更新、上限 2 h）
| checkpoint | 训练时长 | 回报（论文 §4，均值 / 长度加权） | 回报（按执行步数） | 失败 |
|---|---|---|---|---|
| 2000 | ~0.42 h | 0.406 / 0.425 | 0.505 | 17/40 |
| 4000 | ~0.84 h | 0.544 / 0.559 | 0.608 | 10/40 |
| 6000 | 1.26 h | 0.654 / 0.671 | 0.712 | 8/40 |
| 8000 | ~1.68 h | 0.712 / 0.723 | 0.770 | 5/40 |
| latest（~9600） | 2.01 h | **0.766 / 0.773** | 0.775 | **2/40** |
- 训练回报（训练日志）2.0 h 时 0.818。与 beyondmimic_seq 终点（0.763，1/40）在评测噪声内持平；两者都未在 2 h 内达到 0.82。
- 注：T39 训练结束后原 tmux 链未执行评测（运行中被替换了脚本文件），评测由 `eval_run_lafan1.sh` 补跑，checkpoint 与命令相同。

## LAFAN1 对照（T40）：proprio_only（论文 Table A.1 字面），同上设置
| checkpoint | 回报（论文 §4，均值 / 长度加权） | 回报（按执行步数） | 失败 |
|---|---|---|---|
| 2000 | 0.315 / 0.327 | 0.456 | 20/40 |
| 4000 | 0.461 / 0.467 | 0.480 | 6/40 |
| 6000 | 0.510 / 0.514 | 0.521 | 4/40 |
| 8000 | 0.484 / 0.496 | 0.538 | 9/40 |
| 10000 | 0.544 / 0.552 | 0.568 | 4/40 |
| latest（2.0 h） | 0.522 / 0.535 | 0.590 | 8/40 |
- 平台在 0.52–0.54，与上游 e2/e7 的"论文口径约 0.56 封顶"一致（上游按执行步数）：无锚点朝向时长 clip 上偏航漂移使根部/世界系速度项失效。

## 观测口径选择（T41/T42）
| 口径 | 2 h 终点回报（论文 §4） | 失败 | 达到 0.82 |
|---|---|---|---|
| beyondmimic_seq | 0.763 | 1/40 | 否 |
| anchor_ori_now | **0.766** | 2/40 | 否 |
| proprio_only | 0.522 | 8/40 | 否 |
- 规则（design Clarifications）：无一达到 0.82 → 取终点回报最高 → anchor_ori_now。与 beyondmimic_seq 之差在噪声内。
- 与论文 Fig. 4a（0.82 @ 1.2 h，H100）比：同口径同协议下我们 2 h 为 ~0.77；时间对齐与终值都未达到。主要嫌疑：吞吐（1.3 次更新/秒 vs vibed 3.1，见上）——2 h 只做了 ~9.5k 次更新。
