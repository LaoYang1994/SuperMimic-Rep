# Return-based reset sampling captured by clip-end bins

## Problem
On a large motion library (BONES-SEED, 129k clips) whole motion groups never improve across checkpoints
(flips and side rolls failed 54/54 and 22/22 at 36k, 40k and 44k iterations) although their reset bins are
"supposed" to be favored by the hard-sampling term. The sampler's per-bin return estimates for those motions
look as good as walking.

## Root cause
The per-bin estimate was the raw discounted return-to-go. Its upper bound depends on how many steps remain
before the episode ends (the clip end is not bootstrapped). The last bin of every clip (median 7 frames on
BONES-SEED) therefore looks "low return" even when tracked perfectly, and a sharp softmax (temperature 25 over
~1M bins) gives almost all hard mass to the lowest bins: 93.9% of the hard mass sat on clip-end bins, 0% on
flips. Hard motions that track for a few dozen steps before failing get a medium estimate and are never chosen.

## Pattern / Rule
Compare a bin's return with what was attainable from it: divide each step's return-to-go by the maximum
discounted return attainable over the steps the episode would still have run without failing (to the clip end
or the time limit). Determine those steps from the episode's actual structure (its length, plus the steps from
the terminal frame to the clip end for failed episodes), not from per-step frame arithmetic, which is off by one
around the clip end.

## How to apply
- Check where the hard mass goes before trusting a return-based sampler on a new dataset:
  `scripts/sampler_report.py --ckpt <ckpt> --library <compact library>` (CPU) prints the share on clip-end bins,
  per motion group, and the motion families of the top hard bins.
- Watch `sampler/hard_mass_clip_end_bins` in TensorBoard during training.
- With a sharp softmax, expect the hard mass to rotate between a handful of bins; a large clip-end share is only
  an artifact if those bins are the short ones (few frames) rather than genuinely failing clip endings.

## Validation
After the fix, clip-end share of the hard mass 99.9% → 1.7% at 4k iterations; at 36k/40k/44k the full-benchmark
failures fell 413 → 321, 425 → 253, 361 → 210 with unchanged return; flips 54 → 18–24.

## Pitfalls
- Reading the training-env episode return across sampler versions: a sampler that resets near clip ends gets
  near-perfect short episodes, so its training return looks much higher (0.97 vs 0.64) while the benchmark is
  the same or worse. Compare samplers on the benchmark only.
- Rare motions still update slowly (few visits, EMA 0.05 from an optimistic initial value of 1.0).

## Related changes
- 2026-10-25-sampler-attainable-return

## Related contracts
- `docs/CONTRACTS.md#结果报告与论文对照`
