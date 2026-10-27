# Experiences index

Reusable lessons and pitfalls. One line per experience — keep it scannable so
an agent can find the right lesson without opening every file.

<!-- - [Title](domain/slug.md) — the recurring problem it addresses -->

- [Editable venv runs the main tree's code](remote/editable-venv-runs-main-tree.md) — a run in a remote worktree silently imports the main tree's `src/`
- [Container limits and storage on the GPU box](remote/container-limits-and-storage.md) — `free`/`nproc` show the host, `/dev/shm` eats RAM, PFS is slow for small files, no data on `/`
- [Remote shell pitfalls](remote/remote-shell-pitfalls.md) — `pkill -f` over ssh kills itself; editing/syncing a running script breaks the chain
- [Pool.imap drains its input generator](python/pool-imap-drains-generator.md) — streaming job's memory grows to the whole input
- [MuJoCo Warp is not bitwise reproducible](sim/mujoco-warp-nondeterminism.md) — before/after benchmark comparisons need a same-code rerun as noise floor
- [Return-based reset sampling captured by clip-end bins](training/return-sampler-clip-end-bins.md) — hard-sampling mass goes to short clip-end bins, hard motions (flips) never get practiced
- [Imagined rollouts keep running after a predicted fall](training/imagination-dead-samples.md) — critic loss 1e13..inf and NaN weights late in long runs while the policy looks fine
- [Loss scale acts only through clipping; critic values above G_max precede collapse](training/critic-overestimation-and-loss-scale.md) — a loss rescale "has no effect" under Adam; critic mean creeps past max_r/(1−γ) before a late collapse
- [torch.where does not stop NaN gradients](python/torch-where-nan-gradient.md) — a "safe" branch still yields NaN gradients at singular inputs (quaternion error at zero)
- [Optimizer.load_state_dict drops fused=True](python/optimizer-load-drops-fused.md) — after loading a checkpoint the fused optimizer silently reverts (no speed-up, per-parameter syncs)
- [Profile in ms before optimizing a launch-bound loop](perf/launch-bound-training-profile.md) — sync counts mislead; compile + CUDA graphs cut launches; how to profile and verify
- [Reverse-engineer a data filter from exact counts](data/reverse-engineer-filter-from-counts.md) — paper gives filtered sizes but not the keyword list / thresholds
