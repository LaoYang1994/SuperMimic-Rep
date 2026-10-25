# Container limits and storage on the GPU box are not what the usual commands say

## Problem
Planning memory/disk use from `free`, `nproc` or `df` led to choices that would have crashed the box: data
parked in `/dev/shm`, a 142k-file extraction onto the network filesystem, 180 workers on a 64-CPU container.

## Root cause
- In a container, `free` and `nproc` report the **host** (1.8 TB / 180 threads); the cgroup caps the
  container at **128 GiB / 64 CPUs**.
- `/dev/shm` is tmpfs: its contents count against the container's memory limit.
- PFS (PFS) is fast for big files but creates small files at ~3 per second (200 × 300 KB in 60 s,
  vs 0.17 s on local disk) — extracting 142k CSVs there would take about a day.
- The system disk `/` (98 GB) is local and fast but must hold no data (user rule): it is shared with the OS
  and filling it takes the box down.

## Pattern / Rule
Read the real limits and test the filesystem before heavy jobs; keep data on PFS as a few large files.

## How to apply
- Limits: `cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/cpu.max`; memory in use:
  `awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat` (not `memory.current`, which includes reclaimable
  page cache from reading PFS and gives false alarms).
- Many-small-file datasets: stream from the archive into large shards (`scripts/data/stream_bones_seed.py`)
  instead of extracting.
- Small **writes** are slow on PFS too, not only small files: `tar cf <file on PFS>` (10 KB records) wrote at
  ~1.5 MB/s, while `tar cf - … | dd of=<file on PFS> bs=64M iflag=fullblock` wrote 7 GB in 61 s (~115 MB/s).
  Push large single files through a big-block writer (`dd bs=64M`, or a library that buffers, like `np.savez`).
- Never use `/dev/shm` for datasets; cap workers (~8–16) and run CPU jobs under `ulimit -v`.

## Validation
`df -h /` unchanged after a data job; anonymous memory stays well under 128 GiB during it.

## Pitfalls
- Treating page cache (tens of GB after reading shards) as real memory pressure.
- The venv (~7 GB) is the only sanctioned large item on `/`; caches such as torch inductor's in `/tmp` should
  be cleared after a series of runs.

## Related changes
- 2026-10-24-paper-faithful-reproduction

## Related contracts
- none
