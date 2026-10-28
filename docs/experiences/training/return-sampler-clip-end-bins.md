# Return-based reset sampling captured by a few bins

## Problem
On a large motion library (BONES-SEED, 129k clips) whole motion groups never improve across checkpoints
although their reset bins are "supposed" to be favored by the hard-sampling term
(softmax over `-25 * G_b`, 40% of resets). Two forms were seen:

1. **Clip-end bins** (fixed in `2026-10-25-sampler-attainable-return`): flips and side rolls failed 54/54 and
   22/22 at 36k, 40k and 44k iterations, while the sampler's per-bin return estimates for those motions looked as
   good as walking.
2. **Hard-but-unlearnable clips** (fixed in `2026-10-27-sampler-clip-cap`): after fix 1, 76–96% of the hard mass
   sat on a handful of bins of one clip family (`kick_back`, G ≈ 0.23, never improving) from 20k to 148k
   iterations; the effective number of hard bins was 2–5 and the effective number of motion categories about
   1.1. When the mass moved to another unlearnable clip (`kneeling_start`, G 0.10 at 146k; `kneeling_loop` at
   36k in another run) the training-env return collapsed (0.64 → 0.07, or 0.64 → 0.38) while the policy itself
   was fine. Motions that actually fail in the benchmark sat at G ≈ 0.84 (median) and received 0.5% of the hard
   mass.

## Root cause
1. The per-bin estimate was the raw discounted return-to-go. Its upper bound depends on how many steps remain
   before the episode ends (the clip end is not bootstrapped). The last bin of every clip (median 7 frames on
   BONES-SEED) therefore looks "low return" even when tracked perfectly, and a sharp softmax (temperature 25 over
   ~1M bins) gives almost all hard mass to the lowest bins.
2. A sharp softmax is winner-take-all: a bin whose return stays at 0.1–0.3 (physically unreachable from the
   reference state, or hard to learn) is `exp(25 * 0.6) ≈ 3e6` times more likely than a G = 0.84 bin. Nothing in
   the paper's formula bounds how much one clip can take, so one clip holds the hard budget for as long as it
   does not improve, and the memory term (EMA of the hard distribution) keeps it there.

## Pattern / Rule
- Compare a bin's return with what was attainable from it (fix 1): divide each step's return-to-go by the maximum
  discounted return attainable over the steps the episode would still have run without failing. Determine those
  steps from the episode's actual structure, not from per-step frame arithmetic.
- Cap the hard mass per clip (fix 2): no clip gets more than `hard_clip_cap` (default 1%) of the hard mass; the
  excess is redistributed proportionally to the uncapped clips (closed-form water-filling in
  `envs/mdp/sampler_math.cap_group_mass`). Update the memory term from the capped distribution. With no clip over
  the cap the result is bit-identical to the paper's formula. `hard_clip_cap=0` restores the paper's behaviour.

## How to apply
- Check where the hard mass goes before trusting a return-based sampler on a new dataset:
  `scripts/sampler_report.py --ckpt <ckpt> --library <compact library>` (CPU) prints the share on clip-end bins,
  per motion group, and the motion families of the top hard bins. Reconstruct the softmax from the checkpoint's
  `bin_return` and sum per clip; look at the effective number of clips `1 / sum(m_c^2)`.
- Watch `sampler/hard_mass_clip_end_bins`, `sampler/hard_top_clip_share` (share of the largest clip before the
  cap) and `sampler/hard_eff_clips` (effective clips after the cap) in TensorBoard. A top share of 0.3–0.5 that
  persists for tens of thousands of iterations is a capture.
- A sudden drop of the training-env return with healthy world-model / critic metrics is a sampler event (the hard
  mass moved to a lower-G clip), not a policy collapse: check `bin_return` before touching the learner.

## Validation
- Fix 1: clip-end share of the hard mass 99.9% → 1.7% at 4k iterations; at 36k/40k/44k the full-benchmark
  failures fell 413 → 321, 425 → 253, 361 → 210 with unchanged return; flips 54 → 18–24.
- Fix 2 (BONES-SEED full benchmark, 129,511 clips, one RTX PRO 6000, compared at equal updates against the same
  code without the cap): 44k 166 → 165 failures (return 0.9333 → 0.9329, group mix changed: flips 25 → 9), 88k
  70 → 30 (0.9413 → 0.9429; side rolls 9 → 0, flips 4 → 0), 104k 58 → 21 (0.9390 → 0.9433, old baseline). The
  paper reports ≈30 at 88k and 27 at 100k on one H100 (Fig. A.1). Effective clips after the cap 170–225; the
  uncapped top share stayed at 0.31–0.50 the whole time, i.e. the cap was active throughout. The cap does not
  help at 44k; the gain appears once the previously starved groups have had enough resets. At 148k: 14 failures /
  0.9428 (paper: 14), long standing clips at 0.970 — the late return drop of uncapped runs (0.918 / 80 and 0.891 / 125
  at 148k, standing clips drifting) was the capture, not the learner.

## Pitfalls
- Reading the training-env episode return across sampler versions: a sampler that resets near clip ends gets
  near-perfect short episodes, so its training return looks much higher (0.97 vs 0.64) while the benchmark is
  the same or worse; the cap raises it too (0.64 → 0.72) because fewer resets land on unreachable bins. Compare
  samplers on the benchmark only.
- Rare motions still update slowly (few visits, EMA 0.05 from an optimistic initial value of 1.0).
- Mirrored clips count as separate clips for the cap, so a motion with n takes × 2 mirrors can take up to
  `2n * cap`.
- `sampler/hard_mass_clip_end_bins` is computed after the cap since `2026-10-27-sampler-clip-cap`; curves from
  older runs are not directly comparable.

## Related changes
- 2026-10-25-sampler-attainable-return
- 2026-10-27-sampler-clip-cap

## Related contracts
- `docs/CONTRACTS.md#结果报告与论文对照`
