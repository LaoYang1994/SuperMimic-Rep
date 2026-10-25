# Review

## Summary
本 change 建立了论文对齐的训练 / 评测基础：迭代 = 更新、四种观测口径（`beyondmimic_seq` / `anchor_ori_now` / `proprio_only` / `anchor_pose9`）
及其仿真 / 想象 goal 逐元素对拍、DR 在 startup 采样、论文定义的归一化回报（分母 = max_r × 整条 clip 步数）、BONES-SEED 流式转换 +
名字 / 可行性过滤（129,667 ✓ / 129,511）、紧凑库与全量评测、`eval_groups.py`。复现目标只部分达成（见 validation"复现目标达成情况"）：
数据过滤、8 步误差对上；LAFAN1 0.82 与 BONES-SEED 失败数在本 change 代码上未达到，后续 change（正确性修复、加速、采样器）
已把 LAFAN1 推到第 18k 次 0.821、BONES-SEED 44k 次 210 失败，仍未到论文 25。

**Verdict: archive-with-fixes** —— 按用户决定"按已完成部分归档"：未完成项（T37、T57）与未达目标写入 plan / validation，
复现剩余工作转入新 change；归档时同步 CONTRACTS / CONCEPTS / PROJECT（plan 的 Contract / Project updates 此前未做）。

## Coverage
- CLEAN — 迭代语义、观测口径、DR 时机、归一化回报、BONES-SEED 数据流水线与过滤、紧凑库与评测、分组脚本：均有实现与证据（validation T36、T46–T62）。
- MISSING（已确认）— design Goal：LAFAN1 0.82 @ ≈1.2 h、BONES-SEED 失败 ≈ 25 且分组同形、想象吞吐 > 10×（T37）、可行性过滤精确 129,514、Table A.8 分组精确（T57）。
- MISSING（本次归档补齐）— plan 的 Contract updates / Project updates 在实施时未落到 durable docs。

## Findings
### P1
无。
### P2
[P2] (confidence 9/10) docs/CONTRACTS.md、docs/CONCEPTS.md、docs/PROJECT.md — plan 要求的口径取值、迭代语义（1 次迭代 = 1 次更新）、回报分母定义、
Features 状态都没有写入 durable docs，PROJECT 仍写"一次迭代等于一个向量化仿真步" → 归档前就地更新（本次完成，见 Resolved）。
### P3
[P3] (confidence 6/10) plan Experience updates — "GPU 机无 GitHub/HF 时的依赖与数据中转""BONES-SEED 数据准备"两个 workflow 未写 → 下一次做数据准备时用 `/workflow-capture` 记录。

## Unverified
- （置信 4/10）BONES-SEED 可行性过滤差 3 条的来源（推测为论文未给的直立 / 站立高度 / 静止阈值）。

## Resolved
- P2：CONTRACTS（结果报告口径与迭代语义、BONES-SEED 条数注明、回报分母）、CONCEPTS（迭代与 UTD、观测口径）、PROJECT（训练循环、环境层观测、Features）已就地更新。

## Accepted risks
- 复现目标未全部达成即归档（用户决定）；剩余工作在新 change 中继续。
- 单种子结论。

## Follow-up items
- 新 change：缩小 BONES-SEED 失败数与论文的差距（fire-in-the-hole 变差、剩余失败分类、稀少动作的采样估计）。
- T37 吞吐基准（有空闲 GPU 时）。
- 两个 workflow 文档（P3）。
