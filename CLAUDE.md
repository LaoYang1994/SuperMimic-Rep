@AGENTS.md

# Claude Code

## Workflow
- Use plan mode when running `/change-design`.
- Use TodoWrite to track `plan.md` tasks during `/change-implement`.

## Memory
- Durable learnings belong in `docs/experiences/` (via `/experience-capture`),
  not left only in Claude's auto memory.
- When compacting, preserve the active change ID and `plan.md` task state.

## (Optional) Path-scoped contracts
Once `docs/CONTRACTS.md` grows large and is mostly path-specific, mirror
sections as path-scoped rules under `.claude/rules/` that **point to** (not copy)
the relevant `CONTRACTS.md` section, so they load only when editing matching
files. `CONTRACTS.md` remains the source of truth.
