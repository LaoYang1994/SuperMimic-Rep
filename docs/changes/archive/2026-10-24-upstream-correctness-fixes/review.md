# Review

## Summary
Self-review of the worktree diff (`change/2026-10-24-upstream-correctness-fixes`, uncommitted, against
`64df684`) for tasks T1–T13 against `design.md`, `plan.md` and `docs/CONTRACTS.md`. The three fixes and the
guard match the design; one multi-GPU deadlock in the critic guard must be fixed; validation (T14–T19) has not
run yet (needs the GPU, deferred by the user until the current BONES-SEED run ends).
Verdict: **archive-with-fixes** → P2 fixed (T20); remaining gate is `/change-validate` (T14–T19).

Large-file guard: no added/changed file over 1 MB.

## Coverage
- Termination-label alignment (design problem 1): CLEAN — `replay_buffer.py:137` reads `done` from rows
  t..t+k−1 (same rows as `reward`); `recorder.py:104` writes `done=0` on the terminal row; the only reader of
  `WindowBatch.done` is `wm_trainer.py:50 → world_model.window_loss`; `sample_starts` excludes terminal rows.
  Test: `tests/test_replay_labels.py`.
- Zero-angle quaternion gradient (problem 2): CLEAN — `rewards.py:69` `_quat_box_minus`, used by
  `_sq_quat_err` (its only callers are the root/body orientation terms, `rewards.py:119,121`); the other
  `quat_error_magnitude` uses are metrics without gradients (`commands.py:186,188`). Test: `tests/test_quat_error.py`.
- Resume warmup (problem 3): CLEAN — `loop.py:204` collects `warmup_sim_steps` every run.
- Non-finite guard: CLEAN for the policy (`policy_trainer.py:97`, decision on all-reduced gradients);
  the critic guard has the multi-GPU defect below. Test: `tests/test_clip_and_step.py` (policy helper only).
- Validation T14–T19: MISSING by plan (deferred until the GPU is free).
- DRIFT: `scripts/remote/sync_code.sh` (remote tree per worktree) is in this worktree's diff but not in this
  change's design — it is the user-requested remote-workflow tooling, copied here so D-group validation syncs
  to `.worktrees/<id>` on the box. It should be committed with the tooling, not attributed to this change.

## Findings
### P1
none

### P2
none open (see Resolved)

### P3
[P3] (confidence 7/10) src/supermimic/imagination/policy_trainer.py:129 — the critic guard checks only its
inputs and targets; a finite input with critic weights that already overflowed would still step. Weights only
ever step on finite inputs here, so this needs an overflow first; accept, and watch `policy/critic_loss` for
NaN in validation.
[P3] (confidence 6/10) src/supermimic/imagination/policy_trainer.py:50,108 — the policy path now syncs twice
on the grad norm (`bool(torch.isfinite(gn))` and the existing `float(gn)` metric). Harmless here; fold into
one host read in the throughput change.

## Unverified
- `torch.linalg.norm` backward at an exactly zero vector is expected to return 0 (PyTorch masks zero norms),
  which `_quat_box_minus` relies on; `tests/test_quat_error.py::test_zero_error_gradient_is_finite` checks it
  but has not run yet (no torch locally).

## Resolved
- [P2] (confidence 9/10) src/supermimic/imagination/policy_trainer.py:129 — the critic skip decision `finite` is taken per rank from that rank's own rollout, but every critic minibatch calls `allreduce_grads(self.critic)` (`policy_trainer.py:142`, `dist.all_reduce` in `training/distributed.py`). If one rank skips and another does not, the non-skipping ranks block forever in `all_reduce` (deadlock in data-parallel runs, e.g. the paper's 2-GPU setting). → all-reduce the flag before deciding (`finite = allreduce_mean(float(finite)) == 1.0`, i.e. skip on every rank if any rank is non-finite); add a note in the docstring. Single-GPU runs are unaffected.
  **Fixed** (plan T20): `finite = allreduce_mean(float(finite)) == 1.0` before the critic loop; no-op on one GPU.

## Accepted risks
- Critic guard covers inputs/targets only (P3 above).

## Follow-up items
- Double host sync on the policy grad norm → throughput change.
- Commit `scripts/remote/sync_code.sh` with the remote-workflow tooling (outside this change).
