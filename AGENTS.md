# Agent Instructions

## Read order
Before starting work, read:
1. `docs/PROJECT.md`
2. `docs/CONCEPTS.md` if domain terms are unclear
3. `docs/CONTRACTS.md` before changing data formats, APIs, metric semantics,
   tensor layouts, alignment logic, or config schemas
4. `docs/experiences/INDEX.md` to spot a relevant lesson, then read that file
5. `docs/workflows/INDEX.md` before a recurring multi-step task — if a listed
   procedure matches, read and follow it instead of improvising

Do not blind-search `docs/changes/archive/`. Read an archived change only when a
durable doc links to it by ID, or when investigating a regression the durable
docs don't explain.

## Documentation system
Do not create random markdown files. Use only these locations:
- `docs/PROJECT.md`: project overview, repository layout, workflows, feature
  status
- `docs/CONCEPTS.md`: stable domain vocabulary
- `docs/CONTRACTS.md`: stable contracts and hard rules (a living document — add
  rules whenever you discover the need, no active change required)
- `docs/workflows/`: reusable step-by-step procedures for recurring tasks
  ("to do X, first A then B") — one file per procedure, listed in `INDEX.md`
- `docs/changes/active/<YYYY-MM-DD>-<slug>/`: active changes
- `docs/changes/archive/<YYYY-MM-DD>-<slug>/`: completed changes (immutable)
- `docs/experiences/`: reusable lessons, pitfalls, debugging notes, listed in `INDEX.md`

Do not run markdown auto-formatters (mdformat, prettier) over `docs/changes/` —
`plan.md`'s task checkboxes are a state machine; reformatting reorders/corrupts
them. Exclude `docs/changes/` in your formatter / pre-commit config, and toggle a
box by matching its stable `T#`, not by position.

## When to open a change
- A change is human-initiated. Only open a change folder when a `/change-*`
  command is invoked. Never create one on your own.
- If a task would touch a contract, change observable behavior, or need
  validation, *suggest* opening a change before proceeding — but proceed
  without one if the user declines.

## Referencing changes
Refer to a change by its ID (`<YYYY-MM-DD>-<slug>`), never by path. The ID is
immutable. Link to the experience or contract distilled from a change for
long-term references, not to the change folder.

## Change workflow
For a tracked change:
1. `/change-design` — clarify requirements; create/update `design.md`.
2. `/change-plan` — create/update `plan.md` with steps and task checklist.
   - (optional, non-trivial changes) `/change-plan-review` — confront design +
     plan with the real code/data + contracts *before* implementing; fold fixes
     back into design/plan.
3. `/change-implement` — implement according to `plan.md`; keep checklist
   updated.
4. `/change-review` — review against `design.md`, `plan.md`, `CONTRACTS.md`;
   record findings in `review.md`.
5. `/change-validate` — required for correctness, performance, metric semantics,
   CUDA/TensorRT, data alignment, or benchmark changes; record commands,
   results, benchmark numbers, manual checks, and known gaps in `validation.md`.
6. `/change-archive` — move the active change folder to archive; update
   `PROJECT.md`, `CONTRACTS.md`, `CONCEPTS.md`, `experiences/`, and `workflows/`
   when needed.

Every `/change-*` command takes an optional change-id. If omitted: one active
change → use it; multiple → ask which.

`/change-implement` may, if you choose, isolate a change's code work in a git
worktree at `.worktrees/<change-id>/` (branch `change/<change-id>`);
`/change-archive` offers to remove it. ChangeFlow only creates/removes worktrees
— it never commits, merges, or pushes. Change docs always stay in the main tree.

## Entering implementation is human-triggered
Do not slide into the implement phase on your own. After design/plan — even if
the user approves the plan or asks for tweaks — do NOT start writing or changing
project code. Revising `design.md`/`plan.md` per feedback is fine; writing code
requires the user to explicitly start implementation (run `/change-implement`).
That step is also where the worktree choice is offered, so skipping it skips that.

## Discuss before modifying
When the user is designing or deciding something, settle the design and get
explicit approval before editing files. A directional "yes" confirms the
*direction*, not approval to implement — especially when the change would reverse
a prior decision or a contract. Discuss to clarity first, then modify.

## Durable knowledge rules
After tracked work, extract only durable learnings. Prefer this order:
1. Add or update a **test or hook** when the lesson can be enforced
   automatically.
2. Update `docs/CONTRACTS.md` when a stable hard rule changed.
3. Add or update `docs/experiences/` when a reusable debugging or implementation
   pattern was learned. Experiences are a holding pen: search before writing,
   update near-matches instead of duplicating, and graduate or prune over time.
4. Add or update `docs/workflows/` when a recurring multi-step procedure is
   worth following next time (a how-to, not a lesson).
5. Update `docs/CONCEPTS.md` only for stable project vocabulary.
6. Update this file only for short, stable, high-frequency rules.

Do not write task summaries, implementation diaries, chat logs, or unverified
ideas into long-term docs.

## Drift
Suggest `/docs-refresh` when you notice drift (random docs, conflicting
experiences or workflows, stale active changes, missing concepts). Run it at
milestones (before a release, after a batch of changes lands).

## Critical rules
- Compare runs at the same training progress (iterations = updates) first;
  wall-clock time is reported alongside but is never the basis of a comparison
  (`docs/CONTRACTS.md#结果报告与论文对照`).
- Every number compared with the paper or another run states its observation
  setting, dataset, GPU, training time and iteration semantics
  (`docs/CONTRACTS.md#结果报告与论文对照`).
- Training runs on the GPU machines listed in `.compute_units` (gitignored; code in
  `$SM_REPO`, data in `$SM_BANK`, shared by all machines). It cannot reach GitHub or HuggingFace:
  fetch git dependencies and datasets locally, then rsync them over. The local
  machine has no GPU — never train locally.
- Ask before starting any large download or transfer (datasets, checkpoints,
  multi-GB files), and before stopping or restarting one the user is waiting on.
- Scratch files go in the repo's `tmp/` (gitignored), never in `/tmp`; on the GPU
  box use `$SM_REPO/tmp/`.
- `docs/upstream/` is frozen upstream reference material: read it, don't edit it.
- Machine addresses, ports, login users and machine paths never go into tracked files (docs, plans, script
  defaults): they live only in `.compute_units`; refer to `$SM_REPO` / `$SM_BANK` / "see `.compute_units`".
  Scan tracked files and the commits to be pushed before every push (the repo is public).
- Anything on the GPU box (sync, train, evaluate, data processing) follows the
  `remote-exp` skill, with this project's remote profile in
  `docs/PROJECT.md#远程-gpu-机remote-profile供-remote-exp-skill-使用`: remote trees mirror
  local `.worktrees/`, worktree runs set `PYTHONPATH` to their own `src/`, no data on
  the system disk.
