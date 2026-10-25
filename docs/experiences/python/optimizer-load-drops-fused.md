# `Optimizer.load_state_dict` silently turns `fused=True` off

## Problem
After switching Adam to `fused=True`, a run resumed (or profiled) from a checkpoint shows no speed-up and the
same per-parameter host syncs as before; numbers are bitwise identical to the non-fused code.

## Root cause
`torch.optim.Optimizer.load_state_dict` replaces every param group with the saved one, including the
implementation flags (`fused`, `foreach`, `capturable`, `differentiable`). A checkpoint written by the default
Adam stores `fused=None`, so loading it reverts the optimizer to the default implementation. With
`map_location=cuda` its `step` tensors also land on the GPU, and the default Adam then calls `.item()` on them
for every parameter, every step.

## Pattern / Rule
Keep the current optimizer's implementation flags when loading state; take only hyperparameters and state from
the checkpoint.

## How to apply
Load through a helper that overwrites the saved param groups' implementation flags with the live optimizer's
(`src/supermimic/training/optim.py:load_optimizer_state`), everywhere a checkpoint is loaded (training resume,
benchmarks, parity tools).

## Validation
`tests/test_optim_load.py`: a fused Adam stays fused after loading a default-Adam state, the loaded lr and
moments are kept; plain `load_state_dict` drops `fused` (documents the torch behaviour).

## Pitfalls
- Judging an optimizer change by a profile that starts from a checkpoint without checking that the flag
  survived the load (kernel-launch count and sync count unchanged are the tell).

## Related changes
- 2026-10-24-training-throughput
