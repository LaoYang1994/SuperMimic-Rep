# Design

## Background
目标是 100% 复现 SuperMimic（arXiv 2610.28480）。官方代码未发布（作者 GitHub、mjlab 分支/PR 均无），网页 demo 权重加密且许可证禁止逆向，X / alphaxiv / OpenReview 无作者补充说明。本仓库从 CuboiLeo/SuperMimic-Reproduced 导入，它在论文观测口径下 LAFAN1 回报封顶 0.56，加 9 维位姿观测才到 0.82+，且 BONES-SEED 过滤后 129,943 条与论文 129,514 不符。

调研发现，论文"按 BeyondMimic"的部分可由一作自己的公开代码确定：BeyondMimic（HybridRobotics/whole_body_tracking，已退役迁入 mjlab）与 mjlab `8bb2bc9` 的 `tasks/tracking`。据此上游有三处与论文不一致：
1. **迭代语义**：论文 Algorithm A.1 原文 "Both ω and θ update every iteration"，即 1 次迭代 = 1 次更新（UTD 0.4 → 每次迭代 2.5 个仿真步）。上游把 1 次迭代当成 1 个仿真步，320k 迭代只做了约 128k 次更新，lr 衰减节奏与 checkpoint 编号都对不上。
2. **观测口径**：BeyondMimic/mjlab 的不依赖状态估计版本（`G1FlatWoStateEstimationEnvCfg` / `has_state_estimation=False`）正好去掉 `motion_anchor_pos_b` 与 `base_lin_vel`、保留 `motion_anchor_ori_b`，与 Table A.1"无线速度、无全局位置"吻合；论文说把 BeyondMimic 的"单帧目标"换成 K=10 帧序列，单帧目标包含锚点朝向。上游 goal 只有纯参考量，没有任何相对朝向，导致全局漂移。
3. **名字过滤**：上游从 SONIC 关键词表（NVlabs/GR00T-WholeBodyControl `gear_sonic/data_process/filter_and_copy_bones_data.py`）删掉了杂技类词，名字过滤后 130,137 条；用 SONIC 完整表得 129,785，与论文 129,667 只差 118 条，正对应论文"plus a few terms it misses"。已用元数据找到多组补充词可精确得到 129,667。

另有一处时机差异：mjlab 的 CoM / 编码器零偏 / 足底摩擦都是 `startup` 采样，上游把后两者改成了每 episode 重采样。

plan-review 又发现一处指标定义不一致：
4. **归一化回报的分母**：论文 §4 定义为"整条 episode 的奖励和除以它能取到的最大值，提前终止的策略因此得分低"；上游 `eval/benchmark.py` 用 `max_r × 实际执行步数` 作分母，摔倒的 clip 几乎不被惩罚。上游报告的回报（0.56 / 0.824 / 0.934 等）因此与论文不同口径。

数据侧佐证（metadata）：SONIC 完整关键词表过滤后按 120 fps 计 264.53 h，与论文 264.56 h 吻合；被 SONIC 表删掉的 cartwheel（120 条）、handstand（162 条）也不在论文 Table A.8 的分组里——支持"用完整表"。

## Goal
在单卡 RTX PRO 6000 Blackwell 上，以明确标注的口径对上论文以下数字：
- **数据**：BONES-SEED 142,220 → 名字过滤 **129,667** → 可行性过滤 **129,514**（精确相等）；总时长 ≈ 264.56 h。
- **LAFAN1（Fig. 4a）**：40 条 clip（2.45 h），单卡评测回报达到 **0.82**，所需时间与论文 1.2 h（H100）同量级。
- **BONES-SEED（Fig. 3a / Table A.8）**：单卡 **3 h** 时全量 129,514 条 benchmark 评测回报 **> 0.90**（超过 SONIC 的 0.90），失败条数接近 **25**，且各组失败分布与 Table A.8 同形。
- **世界模型**：8 步预测误差落在 **0.007–0.011**；同批量下想象吞吐 > 10× MuJoCo Warp。
- 每个结果按 `CONTRACTS.md#结果报告与论文对照` 注明口径、数据、卡型、时长、迭代语义。

## Non-goals
- 重测 SONIC：0.90 / 179 直接引用论文。
- 论文的 2 卡 9.8 h 零失败结果（只有一张卡）。
- PPO-Sim / PPO-WM 基线、H × critic 消融、推力恢复、速度接口与蒸馏（§4.3–4.5）、数据量扫描（Fig. 3b/c 的 1/7/40 h 子集）、真机部署。
- 任何使用网页 demo 内容的做法。

## Current behavior
- 代码在本地，GPU 机上无环境；GPU 机访问不了 GitHub/HF，mjlab 是 git 依赖。
- 训练循环 `training/loop.py`：每次迭代 1 个仿真步，按 0.4 累加触发更新；`ScheduleCfg.total_iterations=320_000`、`lr_decay_fraction=250k/320k` 按仿真步计。
- 观测：`policy` 组 = Table A.1 的 93 维 × 10 帧；`goal` 组 = K=10 帧 × 68 维纯参考量（`motion_library.goal_features`：参考关节位置/速度、参考投影重力、参考根角速度、参考根线速度（航向系）、根高度）；`anchor_obs=True` 时额外加 9 维（位置 3 + 朝向 6）。
- 域随机化：CoM `startup`；编码器零偏、足底摩擦 `reset`。
- 名字过滤：SONIC 表去掉 8 个杂技词 + 7 个补充词。

## Proposed behavior
- GPU 机上 `uv` 环境可装（git 依赖经本地中转），单元测试通过。
- **迭代 = 一次更新**：每次迭代采 2.5 个仿真步（累加取整），然后 1 次世界模型更新（10 批）+ 1 次策略更新；320k 迭代、lr 在前 250k 迭代 cosine 衰减到 0.03×；checkpoint 编号与论文同义（Fig. A.1 每 4000 迭代评一次）。
- **观测口径可选三种**（配置项，默认 = 主方案）：
  - `beyondmimic_seq`（主方案）：goal 每个未来帧 = BeyondMimic 单帧目标 = 参考关节位置/速度 + 该帧参考锚点（torso_link）朝向在**当前**机器人锚点系下的 6D。
  - `anchor_ori_now`：goal 为纯参考 K 帧 + 仅当前帧 6D 锚点朝向（vibed 做法）。
  - `proprio_only`：论文字面，无任何相对朝向（上游 e2 口径）。
  - 三种都不含锚点位置、不含基座线速度。保留上游 9 维位姿作为非正式参照，不进结论。
- **域随机化时机**与 mjlab 一致：CoM、编码器零偏、足底摩擦全部 `startup`。
- **名字过滤** = SONIC 完整表 + 最小补充词集，补充词集由"名字过滤后 = 129,667 且可行性过滤后 = 129,514"共同确定。
- 世界模型与想象侧的解析观测头、critic 输入随观测口径同步。
- **归一化回报**按论文定义：分母 = `max_r ×` 整条 clip 完整播放的步数，失败的 clip 只累计到终止为止的奖励。（implement 中实测：本环境从首帧完整播放一条 clip 走 **帧数** 步，40/40 条未失败的 LAFAN1 clip 均如此，故分母 = `max_r × 帧数`；训练中从帧 f 起的 episode 上限 = min(末帧 − f + 1, episode 步数上限)。）旧的"按执行步数归一化"保留为附加字段，仅作与上游数字对照。训练日志里的训练回报同样改为除以该 episode 能达到的最大步数。

## Design details
- **迭代语义**改动集中在 `training/loop.py` 主循环与 `config.py:ScheduleCfg`：迭代计数器改为更新计数；采集步数用 1/UTD 的累加器；`warmup` 仍按仿真步填满 k=8 窗口。日志同时记录迭代数与仿真步数。
- **观测口径**：仿真侧在 `envs/tracking_env_cfg.py` 的 goal 组按口径组装；想象侧在 `world_model/heads.py` 由 FK 解析计算相同量（复用上游 `anchor_relative_obs`、`G1Kinematics`），二者须逐元素对拍。`policy.goal_dim` 按口径计算。
- **名字过滤**：`scripts/data/filter_bones_seed.py` 改为 SONIC 完整表 + 补充词；补充词集作为常量写入，并在脚本输出中打印两级计数以便核对。
- **对齐 checkpoint 口径**：LAFAN1 与 BONES-SEED 的评测都走 `eval/benchmark.py`（附录 F.1）；BONES-SEED 训练中用 2000 条子集跟踪，3 h checkpoint 跑全量 129,514 条。
- **数据**：LAFAN1 从 HF `lvhaidong/LAFAN1_Retargeting_Dataset` 经本地中转；BONES-SEED 已在传往 `DataBank/bones_seed`。数据处理产物放 `DataBank`，代码目录只放引用。

## Edge cases
- 2.5 步/迭代的累加取整：奇偶迭代分别采 2 / 3 步，长期平均 2.5。
- 想象 rollout 跨越 clip 末尾时，未来帧目标与锚点朝向的取值（沿用上游的截断/钳位规则，三种口径一致）。
- `startup` 随机化下 ξ 在环境生命周期内不变，replay buffer 中记录的 ξ 仍逐条保存。
- 名字过滤可能有多组补充词都能凑出 129,667，需由可行性过滤的 153 条进一步区分；若仍不唯一，取与论文举例词最一致的一组并记录其余候选。
- BONES-SEED 全量 benchmark 一次评 129,514 条的显存/时间开销（分批评测）。

## Clarifications
- Q: 一个 change 还是拆分？ → A: 一个 change 全包（环境 + 修正 + LAFAN1 + BONES-SEED）。
- Q: 锚点朝向怎么进 goal？ → A: 三种口径（每帧带 / 只当前帧 / 纯本体）都作为正式对照，在 LAFAN1 上并列跑，最接近论文曲线的口径用于 BONES-SEED。
- Q: SONIC 基线要不要重测？ → A: 不重测，引用论文的 0.90 / 179。
- Q: （implement 阶段，用户离开期间）GPU 时间不够跑完全部，优先级？ → A: 先冲 BONES-SEED 主结果：LAFAN1 先只跑主方案 `beyondmimic_seq`（上限 2 h），达到 0.82 即用它跑 BONES-SEED；另两种口径的 LAFAN1 对照排在其后。若主方案未达 0.82，回到"三种口径比较后再选"。
- （T42 结论，按上一条的规则执行）三种口径 LAFAN1 各 2 h 均未达 0.82；终点回报 anchor_ori_now 0.766（2/40 失败）、beyondmimic_seq 0.763（1/40）、proprio_only 0.522（8/40）。按"无一达标则取终点回报最高"选 **anchor_ori_now** 用于 BONES-SEED；它与 beyondmimic_seq 的差（0.003）在评测噪声内，二者实为持平。
- Q: 需要先征得同意的下载/传输/数据处理，用户离开期间可否直接做？ → A: 可以（LAFAN1 下载与传输、BONES-SEED 解压与转换、GPU 机装依赖）；不做 git commit / push。

## Assumptions
- "迭代 = 一次更新"以 Algorithm A.1 原文为准，推翻上游 D12 解读 A。
- 域随机化采样时机以 mjlab `8bb2bc9` 为准（全部 `startup`），论文的"per-episode sim context"理解为"每个 episode 都带着 ξ"而非"每个 episode 重采样"。
- 论文"Root Position/Orientation"奖励的根即 mjlab 的锚点 torso_link（上游已如此）；终止条件用 pelvis（论文明文）。
- LAFAN1 评测按附录 F.1 整条 clip 播放（与 CONTRACTS 一致）；若与论文 0.82 有系统差距，额外报告按 BONES-SEED 典型长度截断的数值作参考，但不替代正式结果。
- 时间对齐的判据：PRO 6000 上达到目标的时间在论文 H100 时间的 1.5× 以内即视为"同量级"，并如实报告比值。
- 上游其余"论文未指明"项（U1–U4、U10、U11、U14 等）沿用上游实现，仅当中间指标（8 步误差、LAFAN1 曲线）对不上时才回头调整。
- 大文件下载/传输（LAFAN1、BONES-SEED 后续步骤、checkpoint 回传）每次开始前征得同意。

## Contract impact
- `CONTRACTS.md#benchmark-协议论文附录-f1`：回报定义明确为"除以整条 clip 的最大可得奖励"；本 change 改 benchmark 代码，按该节要求用同一 checkpoint 做改动前后对比。
- `CONTRACTS.md#结果报告与论文对照`：观测口径的取值改为 `beyondmimic_seq` / `anchor_ori_now` / `proprio_only`（+ 非正式的 9 维位姿），迭代语义固定为"1 次迭代 = 1 次更新"。
- 新增"数据过滤对数"规则：BONES-SEED 过滤结果必须等于 129,667 / 129,514，否则不得用于与论文对照。

## Project impact
- `PROJECT.md`：环境节补 GPU 机安装方式；Core modules 中训练循环的迭代语义、环境层的观测口径与 DR 时机更新；Features 中 LAFAN1 与 BONES-SEED 的状态与结果更新。
- `CONCEPTS.md#观测口径`、`#迭代与-utd`：改为本 change 确定的定义。

## Experience impact
- `docs/experiences/`：论文"按 BeyondMimic"的细节应以一作公开代码（mjlab tracking）为准的经验；名字过滤用论文给出的精确计数反推关键词表的方法。
- `docs/workflows/`：GPU 机无 GitHub/HF 时的依赖与数据中转流程；BONES-SEED 数据准备流程。

## Open questions
- 主方案 `beyondmimic_seq` 中未来帧的参考关节量是否也应相对当前机器人状态（mjlab `MotionCommand.command` 是绝对参考关节位置与速度，按绝对值）。
- 论文 Fig. 3a 的 benchmark 回报是逐 clip 平均还是按步加权（两种都报告）。
- （implement 中发现）训练吞吐：上游实现在 PRO 6000 上约 1.3 次更新/秒（策略更新 0.54 s、WM 0.17 s，GPU 利用率 37%，CPU 侧 kernel 发射受限），vibed 在同卡约 3.1 次更新/秒（bf16 autocast + 更激进的 torch.compile）。按论文"迭代 = 更新"语义，时间对齐（1.2 h / 3 h）很可能受此拖累。是否做性能优化（不改算法）需另行决定，暂不在本 change 内改。
- 论文 Fig. 4a 的 LAFAN1 评测是否整条 220 s 播放（无法从原文确认）。
- 若三种口径在 LAFAN1 上都达不到 0.82 @ ~1.2 h，下一步优先排查哪一项未指明细节（届时在本 change 内修订 design）。
