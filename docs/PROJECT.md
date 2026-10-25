# Project

## Overview
复现 **SuperMimic**（arXiv 2610.28480，CoRL 2026）：用一个从仿真器转移数据学到的可微世界模型，
加上短视野 BPTT actor-critic，训练 Unitree G1（29 自由度）全身动作跟踪策略。目标是在单卡上
尽量对上论文的数字：LAFAN1 单卡 1.2 h 达到回报 0.82（Fig. 4a），BONES-SEED 129,514 条
单卡 3 h 超过 SONIC、失败 25 条（Fig. 3a / Table A.8）。

代码基于 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
（MIT，git remote `upstream`，导入于其 `d9be385`），底座是 mjlab（MuJoCo Warp）+ PyTorch，
不依赖 rsl_rl。

## Users
本仓库维护者自己：在单卡 RTX PRO 6000 Blackwell（96 GB）上训练、评测、对照论文。

## Core workflows
1. **数据准备**：G1 重定向 CSV → 50 Hz npz（`scripts/data/csv_to_npz_batch.py`）→ BONES-SEED
   名字过滤 + 可行性过滤（`scripts/data/filter_bones_seed.py`，论文附录 E）→ 可选打包成 fp16
   紧凑库（`scripts/data/build_library.py`，常驻显存、在线 FK）。
2. **训练**：`sm-train`（`src/supermimic/scripts/train.py`）跑 Algorithm A.1 主循环；多卡用
   `torchrun`（每 rank 自带仿真与 replay buffer，梯度 all-reduce，lr × √world）。
3. **性能**：`scripts/perf/profile_train.py` 稳态 profile（按阶段的发射 / 同步计数、`--sync-stacks` 同步来源）；
   `scripts/perf/update_parity.py` 冻结一次迭代的输入，对比两个代码版本的单次更新（先测同进程噪声基准）。
4. **评测**：`scripts/eval_benchmark.py` 按附录 F.1 协议整条 clip 确定性播放，输出归一化回报与
   失败数；`scripts/eval_push.py` 推力恢复（§4.3）；`scripts/diag_drift.py` 全局漂移诊断。
5. **查看**：`scripts/play.py` 交互回放（参考 ghost），`scripts/tb_tail.py` 无 UI 读 TensorBoard，
   `scripts/render_video.py` 无显示器录 MP4（参考 ghost + 每步终止误差，`MUJOCO_GL=egl`），
   `scripts/sampler_report.py`（仅 CPU）按动作组查看 checkpoint 的回报采样份额。

## Environments
- **本地**（Mac 上的 OrbStack Linux，aarch64，无 GPU）：写代码、能访问 GitHub / HuggingFace；不训练、不跑 CUDA 测试。
- 依赖：Python 3.12、`uv`；torch ≥ 2.14（cu130）、mujoco / mujoco-warp 3.16、warp-lang ≥ 1.15、
  mjlab 固定在 commit `8bb2bc9`（见 `pyproject.toml` 的 `[tool.uv.sources]`）。

### 远程 GPU 机（remote profile，供 `remote-exp` skill 使用）
| 项 | 值 |
|---|---|
| 连接 | 见仓库根目录的 `.compute_units`（gitignored：每台机器的 ssh，以及远程项目根、数据根、env 文件、venv 的位置）；每个 shell 先 source 其中的 env 文件，它设置 `$SM_REPO`、`$SM_BANK`、`$UV_PROJECT_ENVIRONMENT` 并清空 `PYTHONPATH` |
| 机器 | 两台同配置单卡机（A、B），共享同一个 PFS；二者不能组成多卡任务 |
| 远程项目根 | `$SM_REPO`；worktree 对应 `<根>/.worktrees/<change-id>` |
| 数据根 | `$SM_BANK`（各远程 tree 的 `data` 软链到 `$SM_BANK/supermimic_data`）；数据只放这里，系统盘不放数据 |
| 同步 | 本地工具 `scripts/remote/sync_code.sh`（gitignored，不在仓库中；按所在 checkout 映射远程 tree，建 `logs/`、`tmp/`、`data`） |
| Python | 唯一 venv（位置见 `.compute_units`，系统盘，~7 GB），项目 editable 安装自 `$SM_REPO/src`；用 `$UV_PROJECT_ENVIRONMENT/bin/python` |
| 硬件 / 上限 | 1× RTX PRO 6000 Blackwell 96 GB；容器 128 GiB 内存 / 64 核 |
| 存储 | 本地系统盘 98 GB（不放数据）；代码、数据、日志都在各机器共享的 PFS 网络文件系统上（建小文件慢） |
| 网络 | 无 GitHub、无 HuggingFace；PyPI 走镜像（地址见 `.compute_units`）；git 依赖与数据在本地取得后 rsync |
| 环境 | Ubuntu 24.04 容器、驱动 580 / CUDA 13.0、无 Docker；git/curl/tmux/rsync 已装 |

相关坑见 `docs/experiences/INDEX.md` 的 remote/、sim/ 条目。

## Repository layout
```text
SuperMimic/
├── AGENTS.md / CLAUDE.md        agent 指令（CLAUDE.md 只引用 AGENTS.md）
├── pyproject.toml / uv.lock     依赖（uv）
├── src/supermimic/
│   ├── config.py                TrainConfig（tyro CLI）：data / env / world_model / wm_trainer /
│   │                            policy / critic / policy_trainer / schedule
│   ├── data/                    CSV→npz 转换、GPU 环形 replay buffer、mjlab recorder
│   ├── envs/                    mjlab ManagerBasedRlEnv 配置、带噪动作、sim context ξ、状态读写
│   │   └── mdp/                 动作库、多 clip 命令 + 回报采样重置、观测、奖励、终止
│   ├── kinematics/              G1 可微 FK（torch）与旋转工具（四元数 wxyz）
│   ├── world_model/             状态表示、动力学 MLP、终止头、解析观测/奖励头、WM 训练器
│   ├── imagination/             想象 rollout、Eq. 2 目标与 Eq. 3 GAE、BPTT 策略训练器、编译封装
│   ├── policy/                  MotionEncoder + 共享 Decoder（策略）、Critic
│   ├── training/                主循环、cosine lr、数据并行
│   ├── eval/                    附录 F benchmark
│   └── scripts/train.py         训练入口 sm-train
├── scripts/                     数据处理、评测、诊断、基准、可视化
├── tests/                       FK / 解析头 / 环境 / 世界模型 / 想象 / 转换 / 紧凑库 测试
├── docs/
│   ├── PROJECT.md / CONCEPTS.md / CONTRACTS.md     ChangeFlow 长期文档
│   ├── changes/ experiences/ workflows/            ChangeFlow
│   └── upstream/                上游文档（只读参考，见下）
├── data/  logs/                 不入库
└── tmp/                         本地临时文件，不入库
```

上游文档（`docs/upstream/`，导入时冻结，不再修改，内部互相引用的路径仍是旧的 `docs/X.md`）：
`PAPER_SPEC.md` 是论文全部数值与协议的逐条转录，含"论文未指明"清单 U1–U20；
`REPRODUCTION_PLAN.md` 是设计决策 D1–D20；`EXPERIMENTS.md` 是上游 e1–e8 的实验日志；
`REVIEW_LOG.md` 是上游计划的审查记录。本仓库的新结论写进 ChangeFlow 的 change 与长期文档。

## Core modules
### 环境层（`envs/`）
mjlab 环境：50 Hz 控制、4 个 5 ms 子步。策略观测是论文 Table A.1 的 93 维本体项 × 10 帧历史，
goal 是未来 K=10 帧参考，内容由 `EnvCfg.obs_mode` 决定（四种观测口径，见 `CONCEPTS.md#观测口径`；
默认实验用 `beyondmimic_seq` 与 `anchor_ori_now`）。域随机化在 startup 时按环境采样。奖励是 BeyondMimic 跟踪奖励（去掉自碰撞项，关节限位权重 −2），
终止条件是骨盆高度 > 0.25 m 或倾斜 > 0.8，clip 结束按超时处理。sim context ξ 共 33 维：足底摩擦 1、
编码器零偏 29、躯干质心偏移 3。重置按 1 s 分箱、基于回报采样：每个 bin 存"完成度"G_b（每步折现回报-to-go ÷
从该步起可达的最大折现回报，0–1，EMA 更新），重置 = 50% 均匀 + 40% `softmax(−25·G_b)` + 10% 历史困难权重
（见 `CONCEPTS.md#回报采样`）。日志 `sampler/bin_return_mean` 为完成度均值，`sampler/hard_mass_clip_end_bins`
为困难份额落在 clip 末尾 bin 上的比例。

### 世界模型（`world_model/`）
状态 71 维（qpos | qvel）。动力学 MLP（2048×3）预测速度增量后做显式积分；终止头是学出来的，
观测和奖励头是解析的（可微 FK 计算）。在 k=8 步自回归窗口上拟合；终止头的标签是每个转移结果状态的终止
（超时截断不算，见 `CONTRACTS.md#数值约定`）。

### 想象与策略更新（`imagination/`）
从 replay buffer 采起点，冻结世界模型想象 H=32 步，用软存活权重和 critic bootstrap 构造
Eq. 2 目标，再 BPTT。Critic 在 GAE(λ=0.95) 目标上训练，5 epoch × 4 minibatch。整个想象步（策略 + 世界模型 + 奖励 /
终止 + goal + 带噪观测历史）一次编译并以 CUDA Graphs 回放（`PolicyTrainerCfg.compile = "reduce-overhead"`）；
WM 的 8 步窗口损失同样（`WMTrainerCfg.compile`）；三个优化器为 fused Adam。单卡 RTX PRO 6000 稳态
LAFAN1 371 ms/迭代（2.70 次/s）、BONES-SEED 2.4 次/s（change 2026-10-24-training-throughput；原 1.33 / 1.2 次/s）。

### 训练循环（`training/loop.py`）
一次"迭代"等于一次更新（世界模型 + 策略 / critic 各一次），其间按 UTD=0.4 累加采集，平均 2.5 个仿真步
（`CONCEPTS.md#迭代与-utd`）。默认总长 320k 迭代，cosine 衰减到 0.03×，衰减区间占总长的 250k/320k；
学习率调度只依赖迭代次数，`--schedule.max-hours` 只决定何时停止。

## Features
### LAFAN1 跟踪训练（论文 Fig. 4a 设定）
Status: partial

Supported:
- 40 条 clip（2.45 h）的单卡与多卡训练、整条 clip benchmark。
- 上游结果（RTX 5090）：论文观测口径回报在 0.56 左右封顶（全局漂移）；加 9 维锚点观测
  1.66 h 达到 0.824、3.3 h 达到 0.904。
- 上游正确性修复（终止标签对齐、四元数误差零点梯度、resume 补 warmup、非有限梯度跳过，change
  2026-10-24-upstream-correctness-fixes）：同迭代对照下早期学得更快，9,600 次迭代时回报与修复前持平（0.768 vs 0.766）。

- 本仓库（1×RTX PRO 6000，`anchor_ori_now`，迭代 = 更新）：2 h / 9,600 次迭代 0.768；第 18,000 次迭代 0.821（1/40 失败），
  加速后约 1.85 h 可到 18,000 次（changes 2026-10-24-paper-faithful-reproduction、2026-10-24-training-throughput）。

Not supported:
- 论文口径 `beyondmimic_seq` 尚未跑到 18k 次迭代；达到 0.82 所需时间仍长于论文 1.2 h（H100）。

### BONES-SEED 全量训练（论文 Fig. 3a / Table A.8 设定）
Status: partial

Supported:
- 附录 E 两道过滤：名字过滤 129,667（与论文一致）、可行性过滤 129,511 / 263.08 h（论文 129,514 / 264.56 h）；
  流式转换、紧凑库（6.72 GB，常驻显存）、全量 benchmark（`--library`）、Table A.8 分组统计（`scripts/eval_groups.py`）。
- 全量结果（`beyondmimic_seq`，1×RTX PRO 6000，迭代 = 更新，修复 + 加速 + 回报采样修正）：第 44k 次迭代回报 0.934、
  失败 210（SONIC 0.90 / 179，论文 1 卡 3 h >0.90 / 25）。

Not supported:
- 失败数未达到论文（25）；Table A.8 中 flips / lying / dance routine 的分组规则无法从论文确定。

### 其他评测
Status: partial

Supported:
- 推力恢复测试（§4.3）、漂移诊断、仿真与想象吞吐基准。

Not supported:
- 速度指令接口与蒸馏对照（§4.5）、PPO-Sim / PPO-WM 基线、H × critic 消融、ONNX 导出、真机部署。
