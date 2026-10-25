# multiprocessing.Pool.imap drains its input generator into memory

## Problem
A streaming converter fed `pool.imap(convert, generator_over_tar_members)`; its main process grew to 85 GB
in half an hour and the container approached its 128 GiB limit, although the workers were the slow part.

## Root cause
`Pool.imap`'s task-feeder thread consumes the input iterable as fast as it can and queues every item. A
generator that reads files (here every CSV in a 23 GB archive) is therefore read ahead entirely into memory;
"lazy" only applies to the results side.

## Pattern / Rule
Bound the read-ahead yourself: pull a fixed number of items, `pool.map` them, write results, repeat.

## How to apply
See `scripts/data/stream_bones_seed.py` (`_batches(..., read_ahead)` + `pool.map` per batch). Also start such
jobs under `ulimit -v` so a mistake kills the job, not the container.

## Validation
Main-process RSS stays flat (≈ one batch + one output shard) over the run.

## Pitfalls
- `imap_unordered` and `chunksize` do not change the read-ahead behaviour.

## Related changes
- 2026-10-24-paper-faithful-reproduction

## Related contracts
- none
