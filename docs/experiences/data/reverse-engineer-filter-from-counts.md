# Reverse-engineer an unpublished data filter from the paper's exact counts

## Problem
A paper reports exact dataset sizes after filtering (BONES-SEED: 142,220 → 129,667 after the name filter →
129,514 after the feasibility filter) but publishes only example keywords and no thresholds. Reproducing the
dataset by guessing the keyword list yields a different size, and every downstream comparison is off by an
unknown subset.

## Root cause
Filters described in prose ("we drop clips whose name contains a term denoting stairs, boxes, …") leave the
exact term list and thresholds unspecified; a related public filter (SONIC's keyword table) is close but not
identical.

## Pattern / Rule
Treat each published count as a constraint: start from the closest public filter, search small sets of extra
candidate terms (including the paper's own example words), and keep only combinations that hit the
intermediate count exactly. Do not tune the later stage's thresholds to force the final count; report the
residual difference instead.

## How to apply
- Count per stage on the full set, then search 1–4-term combinations of candidates against the first exact
  count (here 12 combinations hit 129,667).
- Among those, prefer the one closest at the next stage and containing the paper's examples
  (`box_`, `jump_off_50cm`, `vault` → 129,511, three short of 129,514).
- Record the chosen list and the residual (−3) wherever results are compared with the paper
  (`docs/CONTRACTS.md#结果报告与论文对照`).

## Validation
Name-filter count equals the paper's exactly; the final-count residual is stated next to every comparison.

## Pitfalls
- Stopping at "close enough" on the first stage hides which clips differ.
- Adjusting feasibility thresholds until the final number matches: it fits the count, not the paper's rule.

## Related changes
- 2026-10-24-paper-faithful-reproduction

## Related contracts
- `docs/CONTRACTS.md#结果报告与论文对照`
