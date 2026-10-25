# `torch.where` does not stop NaN gradients from the discarded branch

## Problem
A loss built with `torch.where(cond, safe_value, f(x))` (or `x.clamp(...)` before `acos`, `sqrt`, `atan2`, …)
still produces NaN gradients exactly where `cond` selects the safe branch — e.g. a quaternion error
`2·acos(|w|)` / `sqrt(1 − w²)` at zero rotation error (two identical orientations).

## Root cause
Autograd differentiates both branches of `torch.where` and multiplies the unselected one by zero; `0 · NaN`
(or `0 · inf`) is still NaN. The forward value is fine, so only the backward pass shows the problem, and only
for inputs that hit the singular point exactly (rare in practice, catastrophic when it happens).

## Pattern / Rule
Make the input of the singular function safe, not just its output: compute `f` on a substituted input in the
singular region (`x_safe = torch.where(cond, harmless_constant, x)`), then select the outputs. Or use a
formulation without the singularity (e.g. `atan2(|v|, w)` for a quaternion angle, or the box-minus via
`log` with a small-angle series).

## How to apply
- For any `acos` / `asin` / `sqrt` / `log` / division in a loss, test the gradient at the singular input in
  float64: `torch.autograd.grad(loss(x_singular), x)` must be finite.
- Keep a regression test with the singular input (here `tests/test_quat_error.py`: zero-error gradient finite).

## Validation
`tests/test_quat_error.py`: forward equals mjlab's error, gradient at q1 == q2 finite (old code: NaN rows).

## Pitfalls
- Clamping the output of `acos` does not help; the derivative of `acos` at ±1 is already infinite.

## Related changes
- 2026-10-24-upstream-correctness-fixes
