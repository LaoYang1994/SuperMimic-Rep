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
