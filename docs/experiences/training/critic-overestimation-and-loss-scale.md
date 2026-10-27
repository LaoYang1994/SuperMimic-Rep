# Loss scale acts only through clipping; critic values above G_max precede collapse

## Problem
Two recurring misreadings of the policy / critic update:
- A change to the overall scale of a loss (upstream divides the policy loss −J by the horizon H) is expected to change
  the learning dynamics, and an accuracy gap between runs gets attributed to it.
- The critic's mean value creeps above the attainable return G_max = max_step_reward / (1 − γ) (= 10 here) while every
  other training curve looks normal. On BONES-SEED (1×RTX PRO 6000, iteration = update) a run with unclamped values
  overestimated from ~145k iterations on (values 15–37, critic loss 1–28) and collapsed at 196k: failures 339 → 7,997,
  return 0.934 → 0.697 on the full benchmark.

## Root cause
- Adam divides each parameter's update by the root of its second-moment estimate. Scaling the loss by c scales the
  first moment and that root by c, so the update is unchanged (up to eps). Scale reaches the update only through
  gradient clipping: with −J/H the policy gradient norm is ~H× smaller (median 0.028 vs 0.72) and the paper's clipping
  (Table A.5: global norm 1.0, then element-wise 0.1) never fires.
- The critic is fit on bootstrapped targets whose bootstrap term is its own prediction; nothing bounds it. Once its
  values exceed G_max the targets inherit the excess, and the policy follows the optimistic V_H in Eq. 2.

## Pattern / Rule
- Before attributing an effect to a loss scale, check whether clipping fires (`policy/grad_clipped`); if it never
  does, the scale is a no-op under Adam.
- Clamp critic values to ±G_max wherever they are used (V_H in the objective, values inside the GAE recursion), and
  treat `critic_value_mean` near or above G_max as a precursor of collapse.

## How to apply
- `PolicyTrainerCfg.objective_divide_by_horizon` (default off = the paper's −J, so the clipping acts) and
  `critic_value_clip` (default on); `policy/critic_values_clamped_frac`, `policy/critic_vH_clamped_frac` show how
  often the clamp acts.
- Any threshold check on a logged value must also test `isfinite`: NaN compares false against every threshold, so a
  critic that has gone NaN passes "value ≤ 10.2".

## Validation
- −J vs −J/H, same code, one switch (BONES-SEED full benchmark, 1×RTX PRO 6000, iteration = update, single seed):
  44k 166 / 0.9333 vs 192 / 0.9303, 88k 70 / 0.9413 vs 82 / 0.9394 (failures / return) — no difference beyond
  run-to-run noise.
- With the clamp, a 169.6k-iteration run kept `critic_value_mean` ≤ 9.60; the clamp acted on at most 1.1% of the
  values (during a reset-sampler collapse).
- Not tested: whether the clamp alone prevents the 196k collapse (no clamped run went past 170k).

## Pitfalls
- Clamping values does not bound the targets: the reward term is still unbounded, and so is the critic's prediction on
  out-of-distribution inputs. A run with only the clamp crashed at 53k iterations (critic loss 1.6×10²⁰) — see
  [Imagined rollouts keep running after a predicted fall](imagination-dead-samples.md).
- Enabling the clipping (−J) changes the effective step distribution; compare at matched iterations, not wall-clock.

## Related changes
- 2026-10-26-critic-clamp-paper-grad-clip
- 2026-10-26-critic-alive-samples

## Related contracts
- none
