# Plan

约定（给零上下文的执行者）：
- 本地 = Mac 上的 OrbStack VM，仓库 `<local repo>`，无 GPU，能访问 GitHub/HF。
- GPU 机 = `ssh <gpu-box>`；代码 `$SM_REPO`（下称 `$REPO`），
  数据 `$SM_BANK`（下称 `$BANK`）。GPU 机**访问不了 GitHub/HF**，PyPI / download.pytorch.org /
  pypi.nvidia.com 可用。所有训练、测试在 GPU 机上跑；代码在本地改，用 T1 的脚本同步。
- 大文件下载/传输（T6、T35 及 checkpoint 回传）开始前必须先征得用户同意。
- 每个要和论文比的数字都按 `docs/CONTRACTS.md#结果报告与论文对照` 标注；实验结果写进本 change 的 `validation.md`。
- 观测口径名：`beyondmimic_seq`（主方案）、`anchor_ori_now`、`proprio_only`、`anchor_pose9`（上游 9 维，仅作参照）。

## Implementation steps

### A. GPU 机环境（不改算法）
- [x] T1 — 新建本地→GPU 机代码同步脚本（rsync，排除 `.venv/ data/ logs/ tmp/ __pycache__/ .worktrees/`）· `scripts/remote/sync_code.sh`
- [x] T2 — 本地 `git clone --mirror https://github.com/mujocolab/mjlab`，确认含 commit `8bb2bc9e`，rsync 到 GPU 机 · `$BANK/mirrors/mjlab.git`
- [x] T3 — GPU 机设置 `git config --global url."file://$BANK/mirrors/mjlab.git".insteadOf https://github.com/mujocolab/mjlab`，使 uv 的 git 依赖走本地镜像 · GPU 机 `~/.gitconfig`
- [x] T4 — 用 T1 同步代码后，在 GPU 机 `$REPO` 执行 `uv sync --group dev`（不改 `pyproject.toml` / `uv.lock`；若 files.pythonhosted.org 慢，仅用环境变量切镜像）· `pyproject.toml`（只读）
> 存储与资源修订（implement 中发现）：PFS（PFS）建小文件约 3 个/秒，不能放 venv、解压的 CSV 或逐 clip npz；
> 容器上限是 **128 GiB 内存 / 64 核**（`/dev/shm` 计入内存配额，不用它放数据）；用户要求操作保守，数据处理 worker ≤ 16。
> venv 放本地盘 `<venv>`（`source <env file>`）；BONES-SEED CSV 解压到本地盘 `<box-local>/bones_csv`（~51 GB）；
> BONES-SEED 全量 npz（~93 GB）放不下，改为**按日期目录分批**：转 npz → 过滤 → 打包 71 维参考 → 删该批 npz（T63/T64）；
> BONES-SEED 评测改用紧凑库而非逐 clip npz（T58）；大文件产物（紧凑库、manifest、子集、checkpoint）落到 PFS。
> **再修订（用户要求系统盘不得存放任何数据）**：不再解压到系统盘。T63 改为 `scripts/data/stream_bones_seed.py`：从 PFS 上的 `g1.tar.gz` 流式读 CSV、内存中转换，每 2000 clip 写一个 npz 分片到 `$BANK/bones_seed/shards/`（tmux `bonesstream`，8 worker）；
> 后续过滤与打包（T44–T50、T64）改为读这些分片（`filter_bones_seed.py` 与 `merge_library_parts.py` 需支持分片输入）。`prepare_bones_seed.sh` 废弃。
> Python 环境（不是数据）经用户同意放系统盘 `<venv>`，装完清 uv 缓存。phase A 链（T4/T5/T7/T8/T55/T56）在 tmux `phaseA`，日志 `$REPO/tmp/phaseA/`。
- [x] T5 — GPU 机建数据目录 `$REPO/data -> <box-local>/sm_data`（本地盘）；`data/` 已在 `.gitignore` · GPU 机 `$REPO/data`
- [x] T6 — **（先征得同意）** 本地从 HF `lvhaidong/LAFAN1_Retargeting_Dataset` 下载 `g1/` 下 40 个 CSV，rsync 到 GPU 机 · `$BANK/lafan1/csv/`
- [x] T7 — GPU 机转换 LAFAN1：`uv run python scripts/data/csv_to_npz_batch.py --input-dir $BANK/lafan1/csv --output-dir data/npz/lafan1 --input-fps 30`，核对 40 条、2.45 h · `scripts/data/csv_to_npz_batch.py`（只读）
- [x] T8 — GPU 机跑改动前基线测试 `env -u PYTHONPATH uv run pytest tests -q`，把通过/跳过数记入 `validation.md` · `tests/`
- [x] T55 — 用**未改动**的代码在 LAFAN1 上训练一个参照 checkpoint（`--run-name ref_upstream --schedule.total-iterations 8000`，约 20 min）· `logs/ref_upstream/`
- [x] T56 — 用**未改动**的 `scripts/eval_benchmark.py` 评测 T55 的 checkpoint，保存逐 clip JSON 作为改动前基准（CONTRACTS 要求的前后对比）· `logs/ref_upstream/eval_before.json`

### B. 迭代语义：1 次迭代 = 1 次更新（Algorithm A.1）
- [x] T9 — 在 `schedule.py` 新增纯函数 `sim_steps_this_iter(acc, utd) -> (n, acc)`（按 1/UTD 累加取整，UTD=0.4 时交替 2/3 步）· `src/supermimic/training/schedule.py`
- [x] T10 — 新增单测：UTD=0.4 时 1000 次迭代总步数 = 2500、每次为 2 或 3；UTD=1 时恒为 1 · `tests/test_schedule.py`
- [x] T11 — `ScheduleCfg`：`total_iterations` / `lr_decay_fraction` / `checkpoint_interval` 注释改为"按更新计"，`warmup_iterations` 改名 `warmup_sim_steps`（默认 40，按仿真步）· `src/supermimic/config.py`
- [x] T12 — 重写 `Trainer.run` 主循环：每次迭代先用 T9 采 n 个仿真步，再做 1 次 WM 更新 + `policy_updates_per_wm_update` 次策略更新；`self.iteration` 计更新次数，新增 `self.sim_steps` · `src/supermimic/training/loop.py`
- [x] T13 — `save` / `load` 增加 `sim_steps` 与 UTD 累加器；旧 checkpoint 缺字段时按 `iteration*2.5` 估算并打印警告 · `src/supermimic/training/loop.py`
- [x] T14 — 日志新增 `time/sim_steps` 标量，控制台行同时打印迭代数与仿真步数 · `src/supermimic/training/loop.py`
- [x] T15 — `ScheduleCfg` 新增 `checkpoint_at_hours: tuple[float, ...] = ()`，主循环在训练时长首次越过该值时额外存 `ckpt_h{X}.pt`（用于取论文的 3 h 点）· `src/supermimic/config.py`
- [x] T16 — 在主循环实现 T15 的按时长存档 · `src/supermimic/training/loop.py`

### C. 观测口径（三种正式 + 上游 9 维参照）
- [x] T17 — `EnvCfg` 用 `obs_mode: Literal["beyondmimic_seq","anchor_ori_now","proprio_only","anchor_pose9"] = "beyondmimic_seq"` 替换 `anchor_obs`；新增 `obs_mode_from_config(cfg_dict)`（旧配置 `anchor_obs=True→anchor_pose9`、`False→proprio_only`）· `src/supermimic/config.py`
- [x] T18 — 新增 `goal_layout(obs_mode, k) -> (dim, scale_vector)`：`beyondmimic_seq` 每帧 = 参考关节位置 29 + 关节速度 29 + 6D 锚点朝向（共 64·K）；`anchor_ori_now` = 68·K + 6；`proprio_only` = 68·K；`anchor_pose9` = 68·K + 9 · `src/supermimic/policy/nets.py`
- [x] T19 — `MotionEncoder` / `Critic` 的 goal 维度与缩放改用 `goal_layout`，删掉 `divmod(goal_dim, 68)` 推断 · `src/supermimic/policy/nets.py`
- [x] T20 — `MotionLibrary` 新增 `anchor_quat_window(frame, k)`：帧 frame..frame+k−1（钳位到 clip 末尾）的参考锚点（torso_link）世界系四元数，紧凑库走在线 FK · `src/supermimic/envs/mdp/motion_library.py`
- [x] T21 — `MotionLibrary` 新增 `joint_window(frame, k)`：同一窗口的参考关节位置（**绝对值**，同 mjlab `MotionCommand.command`）与速度，供 `beyondmimic_seq` 使用 · `src/supermimic/envs/mdp/motion_library.py`
- [x] T22 — 新增 `anchor_ori_seq_obs(robot_anchor_quat, ref_anchor_quats)`：每帧参考锚点朝向在当前机器人锚点系下的 6D（与 `anchor_relative_obs` 的旋转部分同公式），输出 (N, K·6) · `src/supermimic/world_model/heads.py`
- [x] T23 — 新增仿真侧观测项 `goal_beyondmimic_seq`（按帧交错拼 [关节位置, 关节速度, 6D]）与 `goal_anchor_ori_now` · `src/supermimic/envs/mdp/observations.py`
- [x] T24 — `SuperMimicEnvCfgArgs.anchor_obs` 改为 `obs_mode`，goal 组按口径组装；朝向项噪声 U[−0.05, 0.05]、位置项 U[−0.25, 0.25]（同 mjlab）· `src/supermimic/envs/tracking_env_cfg.py`
- [x] T25 — `imagine()` 的 `anchor_obs` 参数改为 `obs_mode`；`make_goal` 按口径用 T20–T22 解析计算，锚点位姿取自 `step_fn` 返回值，噪声与仿真侧一致 · `src/supermimic/imagination/rollout.py`
- [x] T26 — `PolicyTrainerCfg.anchor_obs` 改为 `obs_mode` 并传给 `imagine` · `src/supermimic/imagination/policy_trainer.py`
- [x] T27 — `Trainer.__init__` 用 `goal_layout` 设 `policy.goal_dim`，把 `obs_mode` 传给环境与策略训练器 · `src/supermimic/training/loop.py`
- [x] T28 — `run_benchmark` 的 `anchor_obs` 参数改为 `obs_mode` · `src/supermimic/eval/benchmark.py`
- [x] T29 — 用 `obs_mode_from_config` 取口径 · `scripts/eval_benchmark.py`
- [x] T30 — 同上 · `scripts/play.py`
- [x] T31 — 同上 · `scripts/diag_drift.py`
- [x] T32 — 同上 · `scripts/eval_push.py`
- [x] T33 — 扩展对拍测试：四种口径下仿真 goal 组与想象侧 `make_goal`（无噪声）逐元素一致（atol 1e-4）· `tests/test_analytic_heads.py`
- [x] T34 — 按 `goal_layout` 更新 goal 切片断言 · `tests/test_env_smoke.py`

### D. 域随机化时机（对齐 mjlab `8bb2bc9`）
- [x] T35 — `encoder_bias`、`foot_friction` 的 `mode` 由 `reset` 改为 `startup`，并更新模块 docstring 的差异说明 · `src/supermimic/envs/tracking_env_cfg.py`

### D2. 归一化回报按论文定义
- [x] T59 — `run_benchmark` 的 `norm_return` 分母改为 `max_r ×` 完整播放步数（实测 = clip 帧数）；原"按执行步数"的值另存为 `norm_return_per_step` · `src/supermimic/eval/benchmark.py`
- [x] T60 — `summarize` 同时输出逐 clip 平均与按整条 clip 步数加权的回报（两种 × 新旧分母）· `src/supermimic/eval/benchmark.py`
- [x] T61 — 训练回报 `train/episode_norm_return` 的分母改为该 episode 能达到的最大步数（min(剩余 clip 步数, episode 上限)），旧值另记 `train/episode_norm_return_per_step` · `src/supermimic/training/loop.py`
- [x] T62 — 用改动后的 `scripts/eval_benchmark.py` 重评 T55 的 checkpoint（`obs_mode_from_config` → `proprio_only`），与 T56 比对。（implement 中修订判据：MuJoCo Warp 在 GPU 上逐次运行并非逐位确定——同代码同 checkpoint 两次评测单 clip 回报差最大 0.0137、4 条步数不同——故"逐 clip 完全一致"不可达。）判据改为：(a) 改动前后差异不超过同代码两次评测之间的差异；(b) 失败数一致；(c) 每条 clip 上 `norm_return × 分母 == norm_return_per_step × 执行步数`（新分母算术正确）· `validation.md`

### E. LAFAN1 对齐 Fig. 4a
- [x] T36 — 同步代码后在 GPU 机跑全部测试，必须全绿（CUDA 测试不得被跳过）· `tests/`
执行顺序（用户 2026-10-24 决定，见 design Clarifications）：T38 → T41（仅 seq）→ T42 → F 组（T43–T54）→ T39 → T40 → T41（补全）。
- [x] T38 — 训练 `beyondmimic_seq`：`sm-train --run-name lafan1_seq --data.npz-dir data/npz/lafan1 --env.obs-mode beyondmimic_seq --schedule.max-hours 2` · `logs/lafan1_seq/`
- [x] T39 — 训练 `anchor_ori_now`（同上参数，`--run-name lafan1_now`）· `logs/lafan1_now/`
- [x] T40 — 训练 `proprio_only`（同上参数，`--run-name lafan1_proprio`）· `logs/lafan1_proprio/`
- [x] T41 — 对已跑完各组的每个 checkpoint 跑 `scripts/eval_benchmark.py`（40 条整条播放），汇总回报、失败数、到达 0.82 的时长与 8 步误差 · `validation.md`
- [x] T42 — 若 `beyondmimic_seq` 达到 0.82 即选定它用于 BONES-SEED；否则先跑 T39/T40 再按"到 0.82 用时最短"选定。结论写入 `design.md` 的 Clarifications · `design.md`

### F. BONES-SEED 对齐 Fig. 3a / Table A.8
- [x] T43 — 确认 `$BANK/bones_seed/g1.tar.gz` 传输完成且 sha256 = `52580ea8…6d8a`，解压到 `$BANK/bones_seed/raw/` · `$BANK/bones_seed/`
- [x] T44 — `filter_bones_seed.py` 的关键词表改为 SONIC 完整表（含杂技类 8 词，来源 GR00T-WBC `gear_sonic/data_process/filter_and_copy_bones_data.py`）· `scripts/data/filter_bones_seed.py`
- [x] T45 — 给 `filter_bones_seed.py` 加 `--extra-keywords` 参数，并新增 `--name-only` 模式（只用 metadata CSV 计名字过滤数，不读 npz；匹配字段用 metadata 的 `filename` 列 = CSV/npz 文件名，含镜像后缀 `_M`，**不要**用 `move_name`，二者有 52k 行不同）· `scripts/data/filter_bones_seed.py`
- [x] T46 — 在 GPU 机跑 T63 的流式转换（tmux `bonesstream`，8 worker）得到 `$BANK/bones_seed/shards/`，再用 `filter_bones_seed.py --shards ... --parts ...` 逐分片做 SONIC 名字过滤 + 可行性过滤并打包（tmux `bonespost`）· `scripts/data/filter_bones_seed.py`
- [x] T63 — 新增流式转换脚本：从 PFS 上的 `g1.tar.gz` 直接读 CSV、内存中转 50 Hz（`--input-fps 120`），每 2000 clip 写一个 npz 分片，可续跑；系统盘零写入 · `scripts/data/stream_bones_seed.py`
- [x] T64 — 新增分片合并脚本：合并各批 manifest，按给定 clip 列表（全量或子集）从分片拼出单个紧凑库 · `scripts/data/merge_library_parts.py`
- [x] T47 — 在合并后的 manifest 上遍历候选补充词集（`box_` / `fence` / `look_over` / `jump_off_50cm` / `off_50cm` / `vault` / `_sit` / `50cm` 的组合），找出名字过滤后 = 129,667 **且** 可行性过滤后 = 129,514 的集合（可行性结果已对所有 SONIC 表幸存者算过，补充词只在名字上再筛）· `scripts/data/filter_bones_seed.py`
- [x] T48 — 把选定补充词集写成常量（注释列出其他候选），生成最终 clip 列表并核对 264.56 h · `scripts/data/filter_bones_seed.py`
- [x] T49 — 新增固定 2000 条子集生成脚本（seed 0，从 manifest 抽样）· `scripts/data/make_subset.py`
- [x] T50 — 用 T64 从分片拼出全量紧凑库 `$BANK/bones_seed/library/bones_seed_compact.pt` 与 2000 条子集库 `bones_seed_subset2000.pt` · `scripts/data/merge_library_parts.py`
- [x] T51 — 新增 Table A.8 分组脚本：按 clip 名关键词把 benchmark 结果归入 side rolls / flips / fire-in-the-hole / crossed legs / lying-resting-standing / broad jumps / dance routine / 其他，并输出各组失败数 · `scripts/eval_groups.py`
- [x] T58 — `scripts/eval_benchmark.py` 新增 `--library`（紧凑库路径）：用紧凑库评测（非紧凑全量约 60 GB 显存且需 14 万个 npz，不可行；紧凑库约 7 GB）· `scripts/eval_benchmark.py`
- [x] T52 — 训练 BONES-SEED（T42 选定口径，单卡）：`--data.library $BANK/bones_seed/library/bones_seed_compact.pt --schedule.checkpoint-at-hours 3.0 --schedule.max-hours 3.5`；**训练期间不在同卡上评测**（会拖慢训练、扭曲计时；论文也不计评测时间），checkpoint 在训练结束后用子集库评测 · `logs/bones_main/`
- [x] T53 — 用 `ckpt_h3.0.pt` 在全量紧凑库上跑 129,514 条 benchmark，得到回报与失败数 · `validation.md`
- [x] T54 — 用 T51 输出各组失败表，与 Table A.8 并列；记录 8 步误差 · `validation.md`

### 归档时未完成（2026-10-26，用户确认按已完成部分归档）
- T37 — 跑吞吐基准 `scripts/bench_sim_speed.py` 与 `scripts/bench_imagination.py`（2k/4k/8k/16k 环境），核对想象 > 10× MuJoCo Warp · `scripts/bench_imagination.py`（只读）
- T57 — 在过滤后的 manifest 上校验 T51 的分组大小**精确等于** Table A.8：22 / 44 / 20 / 10 / 99 / 24 / 8 / 129,287（原始名里 `flip` 58 条、`side_roll` 24 条，需由过滤与分组规则共同收敛到 44 / 22；对不上即说明过滤或分组有误）· `scripts/eval_groups.py`
  - T37：吞吐基准未跑（GPU 机一直在训练）；T57：flips 54 / lying 64 / dance routine 与 Table A.8 对不上，论文未给分组规则，无法从公开信息确定。
- 未达到的复现目标（LAFAN1 0.82 的时间、BONES-SEED 失败数接近 25）转入后续 change。

## Files to change
- `scripts/remote/sync_code.sh`（新）：本地→GPU 机代码同步
- `src/supermimic/training/schedule.py`：采集步数累加器
- `src/supermimic/training/loop.py`：迭代 = 更新；sim_steps 记账；按时长存档；`goal_layout` / `obs_mode` 接线
- `src/supermimic/config.py`：`ScheduleCfg` 语义与新字段；`EnvCfg.obs_mode`；`obs_mode_from_config`
- `src/supermimic/policy/nets.py`：`goal_layout`；编码器 / critic 维度与缩放
- `src/supermimic/envs/mdp/motion_library.py`：`anchor_quat_window`、`joint_window`
- `src/supermimic/world_model/heads.py`：`anchor_ori_seq_obs`
- `src/supermimic/envs/mdp/observations.py`：两个新 goal 观测项
- `src/supermimic/envs/tracking_env_cfg.py`：`obs_mode` 组装 goal；DR 改 `startup`
- `src/supermimic/imagination/rollout.py`、`imagination/policy_trainer.py`：`obs_mode`
- `src/supermimic/eval/benchmark.py`：`obs_mode`；归一化回报分母按论文定义，旧值另存；summary 双口径
- `scripts/eval_benchmark.py`：`--library` 紧凑库评测
- `scripts/data/prepare_bones_seed.sh`（新）：BONES-SEED 分批流水线
- `scripts/data/merge_library_parts.py`（新）：分片合并为紧凑库
- `scripts/eval_benchmark.py`、`scripts/play.py`、`scripts/diag_drift.py`、`scripts/eval_push.py`：读取 `obs_mode`
- `scripts/data/filter_bones_seed.py`：SONIC 完整表 + 补充词 + `--extra-keywords` / `--name-only`
- `scripts/data/make_subset.py`（新）、`scripts/eval_groups.py`（新）
- `tests/test_schedule.py`（新）、`tests/test_analytic_heads.py`、`tests/test_env_smoke.py`
- `docs/changes/archive/2026-10-24-paper-faithful-reproduction/validation.md`（实验记录，validate 阶段）

## Tests
- [x] Unit tests — `tests/test_schedule.py`（采集累加器）；四种口径的 `goal_layout` 维度（T36：33 passed；2026-10-26 CPU 重跑见 validation）
- [x] Integration tests — 四种口径仿真 goal 与想象 goal 逐元素对拍；`test_env_smoke`；全部原有 CUDA 测试在 GPU 机通过、不被跳过（T36：33 passed, 0 skipped；后续分支 `dd91147` 上 49 passed）
- [x] Benchmark or performance tests — LAFAN1 三口径 2 h 结果、BONES-SEED 3 h 回报 / 失败数、8 步误差 0.007–0.011 已测（见 validation）；想象 vs MuJoCo Warp 吞吐（T37）未做，见"归档时未完成"
- [x] Manual validation — 名字过滤 129,667 ✓ / 可行性过滤 129,511（论文 129,514，差 3）/ 263.08 h；翻跟头、侧滚的目视检查改用 `scripts/render_video.py` 录像完成（见 `2026-10-25-sampler-attainable-return`）

## Contract updates
`docs/CONTRACTS.md#benchmark-协议论文附录-f1`：写明回报 = 奖励和 /（`max_r` × 整条 clip 步数），失败 clip 只累计到终止；benchmark 改动的前后对比由 T55/T56/T62 完成。
`docs/CONTRACTS.md#结果报告与论文对照`：观测口径取值改为 `beyondmimic_seq` / `anchor_ori_now` / `proprio_only`（`anchor_pose9` 仅参照），迭代语义固定为"1 次迭代 = 1 次更新"；新增"BONES-SEED 过滤结果必须等于 129,667 / 129,514 才能与论文对照"。

## Project updates
`docs/PROJECT.md`：Environments 补 GPU 机安装流程（mjlab 镜像 + insteadOf）；Core modules 更新训练循环迭代语义、观测口径、DR 时机；Features 中 LAFAN1 与 BONES-SEED 状态与结果。`docs/CONCEPTS.md#观测口径`、`#迭代与-utd` 改为本 change 的定义。

## Experience updates
`docs/experiences/`：论文"按 BeyondMimic"的细节以一作公开代码（mjlab tracking）为准；用论文精确计数反推过滤关键词表。`docs/workflows/`：GPU 机无 GitHub/HF 时的依赖与数据中转；BONES-SEED 数据准备。

## Risks
- `uv sync` 在 GPU 机上可能因 files.pythonhosted.org 慢或 mjlab 镜像 insteadOf 不生效而失败（缓解：环境变量切 PyPI 镜像；必要时在本地 `uv export` 出 wheel 清单）。
- 改迭代语义后 lr 衰减变慢（相同步数下更新次数是上游的 2.5 倍），上游的经验超参未必仍最优；以论文 Table A.5 为准不调参。
- `beyondmimic_seq` 是对论文的推断，三种口径都可能达不到 1.2 h / 0.82；此时按 design 的 Open questions 在本 change 内修订 design。
- 补充词集可能无法同时精确命中 129,667 与 129,514（可行性过滤阈值也是推断）；此时记录最接近的组合与差值，不得声称"对上"。
- 单卡串行跑 3 组 LAFAN1（各 ≤3 h）+ BONES-SEED（3.5 h）+ 全量评测，GPU 总占用约 15 h；同时并行会扭曲计时，禁止并行。
- PRO 6000 与 H100 带宽差异可能使时间超过 1.5× 判据；如实报告比值。

## Rollback plan
- 代码改动都在 git 中，按任务分组提交；回退即 `git revert` 对应提交，或切回导入时的 `upstream/main`（`d9be385`）。
- 观测口径保留 `anchor_pose9` 与 `proprio_only`，旧 checkpoint 通过 `obs_mode_from_config` 仍可评测。
- 迭代语义改动若导致训练异常，可临时把 UTD 累加器设为"每迭代 1 步 + 0.4 更新"重现上游行为做对照（不作为正式结果）。
- 数据产物都在 `$BANK`，过滤清单按版本另存，不覆盖上游生成的 manifest。
