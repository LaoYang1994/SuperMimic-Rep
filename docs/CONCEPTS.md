# Concepts

## 归一化回报（normalized return）
整条 episode 的每步奖励之和（不打折）除以它能取到的最大值，1.0 是完美跟踪。提前终止的
episode 因为步数少而得分低。论文 §4 的定义；代码里每步最大奖励是 `max_step_reward`
（= 0.1，见 `AnalyticHeads.reward_cfg`），即回报 = Σr / (0.1 × 步数)。论文里"return"一般就指它。
- **评测回报**：附录 F.1 benchmark 上各 clip 的均值（另记按帧加权的版本）。
- **训练回报**：训练环境 episode 的归一化回报。起点按回报采样重置，所以和评测回报不可直接比。

## 失败（failed motion）
benchmark 中一条 clip 从首帧整条播放时触发了训练终止条件（骨盆高度偏差 > 0.25 m 或倾斜
> 0.8），即记为失败。Table A.8 的"25 / 129,514"就是这个计数。SONIC 自己的判据不同
（根与末端高度 > 0.25 m），对照时以本定义为准。

## 观测口径
策略观测里有没有"参考相对机器人"的全局位姿信息，决定了能否纠正全局漂移：
- **论文口径**：只有 Table A.1 的 93 维本体项（关节位置/速度、IMU 角速度、投影重力、上一动作）× 10 帧，
  外加 goal（未来 10 帧参考）。没有朝向/位置反馈，长 clip 上会系统性转偏。
- **+6 维朝向**：goal 再加参考根朝向在机器人根系下的 6D 旋转（BeyondMimic 的
  `motion_anchor_ori_b`），IMU 可观测，不需要全局位置。vibed 等实现采用。
- **+9 维位姿**：再加 3 维参考锚点位置（`motion_anchor_pos_b`），需要状态估计。
  上游 `EnvCfg.anchor_obs=True` 即此口径。

## Sim context ξ
每个 episode 的域随机化参数，作为世界模型和 critic 的输入（论文 Table A.3）。本仓库为 33 维：
足底摩擦 1 + 关节编码器零偏 29 + 躯干质心偏移 3。世界模型是确定性的，随机性全部来自 ξ 和
在模型外添加的传感器噪声。

## 迭代与 UTD
**UTD**（update-to-data ratio）= 每个向量化仿真步对应的模型更新次数，论文取 0.4。
**迭代**的含义论文没写清，存在两种解读：
- 一次迭代 = 一个仿真步（上游 CuboiLeo，`training/loop.py`）：320k 迭代 ≈ 128k 次更新；
- 一次迭代 = 一次更新（含 2.5 个仿真步，vibed 的做法）：320k 迭代 = 320k 次更新。
它影响学习率衰减节奏和 checkpoint 编号能否与论文对齐。

## 世界模型 8 步预测误差
从 replay 中的真实状态出发，用记录的动作自回归滚 8 步，预测状态与真实状态之差按预设名义
尺度归一化后的平方误差，在策略自己的 rollout 上在线测得。论文 Fig. 3c 报 0.007–0.011；
TensorBoard 中为 `wm/online_nse_8step`。

## 回报采样
训练环境重置起点的选择（论文附录 D.1）。动作时间轴按 1 s 分 bin，每个 bin 维护完成度 **G_b**：
该 bin 内各步"折现回报-to-go ÷ 从该步起可达的最大折现回报"的滑动平均（0–1；可达值按 episode 不失败时
还会持续的步数计，到 clip 结束或时长上限）。重置 = 50% 均匀 + 40% `softmax(−25·G_b)`（困难）+ 10% 历史困难权重。
未归一化的回报-to-go 受剩余步数支配，会让 clip 末尾的短 bin 显得"困难"（见
`docs/experiences/training/return-sampler-clip-end-bins.md`）。

## 想象（imagination）
冻结世界模型，从 replay buffer 状态出发、由策略驱动、不调用仿真器的 H 步 rollout（默认 H=32）。
策略梯度经 BPTT 穿过这些步得到。
