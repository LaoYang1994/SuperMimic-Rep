# Imagined rollouts keep running after a predicted fall

## Problem
Late in long runs the critic loss jumps by many orders of magnitude for one log interval (BONES-SEED, 1×RTX PRO 6000:
inf at ~95k iterations, 8.4×10¹³ at ~123k, 6.4×10³⁰ at ~146k) and the critic weights end up NaN (~152k). The critic's
attainable range is only ±max_step_reward/(1−γ) = ±10, so neither its predictions nor its targets should be anywhere
near that. The policy looks unaffected: policy gradient norm and imagined return stay normal in the same intervals.

## Root cause
The world model predicts a fall (survival ĉ_t → 0, exactly 0 once the sigmoid saturates in fp32), but the H-step
imagined rollout keeps going. Out of distribution, the unbounded policy actions and the extrapolating dynamics feed
each other (mean |a| 1 → 10³–10⁶); the action-rate penalty is quadratic in Δa and unbounded below, so r̂ reaches −10¹⁷.
- The policy objective Σ γᵗ ĉ_t r̂_{t+1} weights these steps by ĉ_t ≈ 0: nothing reaches the policy.
- The critic loss of the paper (Eq. 3, Σ_t (V_t − R̂_t)²) is unweighted: it fits garbage inputs (critic output −10⁸)
  to garbage targets (−10¹⁷). The few runaway samples dominate every Adam step of the update; when the loss overflows,
  NaN reaches the weights.
- Those values are never used anyway: V_{t+1} enters the targets times (1 − d̂_{t+1}) and V_H enters J times ĉ_H.

## Pattern / Rule
Anything fitted on imagined rollouts must apply the same survival the objective applies. Drop dead samples
(ĉ_t < ε) from the critic loss and treat them as terminated in the targets (r̂ = 0, V = 0, d̂ = 1). Use substitution,
not multiplication: their inputs and targets may be inf/NaN, and 0·inf = NaN (see
[torch.where does not stop NaN gradients](../python/torch-where-nan-gradient.md)). Weighting by ĉ_t is not enough:
ĉ = 10⁻¹² times an error of 10¹³ squared is still 10¹⁴.

## How to apply
- Diagnose: the critic target mean equals one penalty term's mean (here `imag_r/action_rate_l2`) in the spike
  intervals, `action_abs_mean` jumps, while Σĉr (`imag_return_undisc`) stays normal: the runaway samples are dead.
- Fix: `PolicyTrainerCfg.critic_min_survival` (ε = 0.001), plus the GAE targets clamped to ±G_max as the recursion
  goes, for samples the termination head misses.
- Log tails, not just means: a log interval averages 50 iterations × H × n samples, so one −10¹⁷ sample shows only as a
  magnitude. Keys ending in `_max` / `_min` keep the interval max / min (`imag_action_abs_max`, `imag_reward_min`,
  `critic_loss_max`).

## Validation
`tests/test_critic_alive.py`: a dead tail with −1e17 rewards, NaN values and NaN d̂ gives finite targets for the alive
steps, equal to an episode terminated at the fall; alive samples bitwise unchanged; masked loss and gradients finite
with inf/NaN inputs on dead samples. In training, check that `critic_loss_max` stays O(0.1) through the iterations
where the old run spiked, while `imag_reward_min` may still show the runaway samples.

## Pitfalls
- Clamping critic values alone (V_H, GAE inputs) does not bound the targets: the reward term r̂ is still unbounded,
  and the critic's prediction on garbage inputs is unbounded too.
- With ε > 0, an alive sample whose next state is dead loses the bootstrap term; the bias is at most (ε/ĉ_t)·G_max, so
  keep ε small (the runaway samples have ĉ ≈ 0, a smaller ε catches them equally).
- Every log point is an interval mean: a spike is never missed, but its size is divided by the interval length.

## Related changes
- 2026-10-26-critic-alive-samples
- 2026-10-26-critic-clamp-paper-grad-clip (critic value clamp, non-finite gradient guard)

## Related contracts
- none
