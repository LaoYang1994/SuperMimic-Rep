# MuJoCo Warp evaluations are not bitwise reproducible

## Problem
Comparing benchmark results "before vs after" a code change clip by clip reported 36/40 mismatches, although
the change did not touch the simulation; failure counts of neighbouring checkpoints also swing widely
(9 failures at one checkpoint, 1 at the next).

## Root cause
MuJoCo Warp on GPU is not deterministic run to run (parallel reductions / contact ordering): the same
checkpoint evaluated twice with the same code differs per clip by up to ~0.014 normalized return, and a few
clips end one step earlier or later.

## Pattern / Rule
Measure a same-code rerun as the noise floor and compare against it; never require exact equality.

## How to apply
- Evaluate the same checkpoint twice with the new code; accept a change if before-vs-after differences do not
  exceed the rerun differences (max per-clip return difference, number of clips with a different step count),
  failure counts agree, and any changed formula is checked arithmetically per clip (the T62 criteria of the
  change below).
- Judge training curves by trend over several checkpoints, not one checkpoint's failure count.

## Validation
The rerun spread is reported next to every before/after comparison.

## Pitfalls
- Reading a single checkpoint's failure count as a result.

## Related changes
- 2026-10-24-paper-faithful-reproduction

## Related contracts
- `docs/CONTRACTS.md#benchmark-协议论文附录-f1`
