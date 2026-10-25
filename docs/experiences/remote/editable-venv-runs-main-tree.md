# A shared editable venv runs the main tree's code from any worktree

## Problem
On the GPU box there is one venv with the project installed editable. Running a script from a remote worktree
(`<root>/.worktrees/<change-id>`) still imports `supermimic` from the remote **main** tree's `src/`, so a
"fixed" run silently executes the old code. Nothing errors; results just do not reflect the change.

## Root cause
An editable install puts the install-time source path (the main tree) on `sys.path` through a `.pth` file.
`cd` and running a script from another tree do not change which `src/` that path points at. `uv run` makes it
worse: it re-syncs the environment from the project it finds.

## Pattern / Rule
In any non-main remote tree, put that tree's `src/` first on the path and prove it at run start.

## How to apply
```bash
source <env file>              # it unsets PYTHONPATH, so export after it
cd <root>/.worktrees/<change-id>
export PYTHONPATH=$PWD/src
$UV_PROJECT_ENVIRONMENT/bin/python -c "import supermimic; print(supermimic.__file__)"
```
Call `$UV_PROJECT_ENVIRONMENT/bin/python` / `bin/sm-train` directly, not `uv run`. Log the printed path and
`git rev-parse HEAD` of the tree in the run's log.

## Validation
The printed `__file__` is under `<root>/.worktrees/<change-id>/src/`.

## Pitfalls
- Exporting `PYTHONPATH` before `source <env file>` — the env file unsets it.
- Entry-point scripts (`sm-train`) obey `PYTHONPATH` too, but only if it is set in the same shell.

## Related changes
- 2026-10-24-upstream-correctness-fixes

## Related contracts
- none
