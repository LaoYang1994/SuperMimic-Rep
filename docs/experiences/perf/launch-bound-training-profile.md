# Profile in milliseconds before optimizing a launch-bound training loop

## Problem
A PyTorch training iteration with many small ops (BPTT through a 32-step imagined rollout, an 8-step world-model
window) runs at ~45% GPU utilization. Counting host syncs (hundreds per iteration) suggests removing them first,
but removing 560 syncs per iteration saved only 20 ms of 754 ms.

## Root cause
When GPU time per kernel is tiny, the CPU cost of launching ~56k kernels per iteration dominates; syncs only
stall a queue that is nearly empty anyway. Counts (syncs, `.item()` calls) do not say how much time they cost.

## Pattern / Rule
Rank work by measured milliseconds per phase, not by event counts, and re-rank after every measurement. For a
launch-bound loop the large wins are fewer launches: compile whole steps into one graph, then replay them as CUDA
graphs (`torch.compile(mode="reduce-overhead")`). Graph-friendly code needs RNG outside the graph (pre-draw the
noise in the same order to keep the random stream), static shapes and no host syncs inside.

## How to apply
- `scripts/perf/profile_train.py`: wall time vs union of kernel time per iteration; launches and syncs per phase
  and sub-phase (imagination forward, backward, critic, WM forward/backward); `--sync-stacks` attributes torch
  syncs with `torch.cuda.set_sync_debug_mode("warn")` (cheap; `with_stack=True` profiling OOMed at 133 GB).
- Before each optimization write down the expected saving in ms; after it, compare and re-plan.
- Verify every step numerically (`scripts/perf/update_parity.py`, tolerance = same-process noise floor) and
  the whole result at matched iterations.

## Validation
On RTX PRO 6000 / LAFAN1: 754 → 734 ms (fused Adam, syncs) → 640 ms (whole imagined step compiled) → 435 ms
(CUDA graphs) → 371 ms (WM window compiled + CUDA graphs); training-level results unchanged at matched iterations.

## Pitfalls
- A compiled graph can still hold hundreds of kernels (FK chains, reductions); compiling alone gave −99 ms,
  CUDA graphs on top −205 ms.
- `cudagraph_mark_step_begin()` is global: outputs of a previous replay are overwritten by the next one.

## Related changes
- 2026-10-24-training-throughput
