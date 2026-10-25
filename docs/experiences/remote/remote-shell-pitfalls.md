# Remote shell pitfalls: pkill matching itself, editing a running script

## Problem
- `ssh box 'pkill -f stream_bones_seed.py; ...'` killed the ssh session itself (exit 255) before the rest ran.
- A tmux chain script was replaced (rsync) while bash was executing it; after the current step bash exited
  without running the remaining evaluation steps, and a downstream chain waited forever for its `DONE` marker.

## Root cause
- `pkill -f` matches full command lines; the remote shell's own command line contains the pattern.
- bash reads scripts incrementally; replacing or editing the file under a running bash leaves it reading a
  stale or shifted stream.

## Pattern / Rule
Kill by tmux session or explicit PID; never edit, rsync over, or sync into the directory of a script or tree a
job is still running — write a new script file instead.

## How to apply
- `tmux kill-session -t <name>`, or `pgrep -f '<pattern>' | grep -v $$` then `kill <pid>`.
- New behaviour for a running chain → a separate script that the chain (or a new tmux session) calls later.
- Before `scripts/remote/sync_code.sh`, check `tmux ls` for jobs using that tree.

## Validation
The job's progress log reaches its `DONE` marker; no unexpected `exit 255` from ssh.

## Pitfalls
- Dependent chains that wait on a marker hang silently when an upstream step dies — always grep for `FAIL`
  as well as `DONE`.

## Related changes
- 2026-10-24-paper-faithful-reproduction

## Related contracts
- none
