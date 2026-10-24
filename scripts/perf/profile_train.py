"""Steady-state throughput profile of one training configuration (change 2026-10-24-training-throughput).

Builds the Trainer, collects warmup data, runs `--settle` iterations so torch.compile / Warp kernels are built,
then profiles `--active` iterations with torch.profiler and prints:
  wall time per iteration, GPU busy fraction (union of kernel intervals / wall), kernels launched per
  iteration, host<->device sync calls and their time, and the time per phase (collect / wm / policy).
  python scripts/perf/profile_train.py --run-name profile@B --obs-mode anchor_ori_now \
      --init-ckpt logs/lafan1_now_fixed@A/ckpt_0008000.pt --out logs/profile@B [--sync-stacks]
Counts: `aten::item` also counts CPU-tensor items (e.g. non-fused Adam's CPU `step`), so the GPU sync indicator
is `cudaStreamSynchronize` / `cudaMemcpyAsync`. `--sync-stacks` additionally runs `--stack-iters` iterations under
`torch.cuda.set_sync_debug_mode("warn")` and attributes every synchronizing torch op to its innermost supermimic /
mjlab frame (syncs inside Warp kernels launched by the simulator are not torch ops and are not seen).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.training.schedule import sim_steps_this_iter

ap = argparse.ArgumentParser()
ap.add_argument("--run-name", default="profile")
ap.add_argument("--obs-mode", default="anchor_ori_now")
ap.add_argument("--npz-dir", default="data/npz/lafan1")
ap.add_argument("--settle", type=int, default=60)
ap.add_argument("--active", type=int, default=20)
ap.add_argument("--out", default="logs/profile")
ap.add_argument("--init-ckpt", default=None, help="start from a trained checkpoint (steady-state profile)")
ap.add_argument("--trace", action="store_true", help="also export a chrome trace (>1 GB; slow to write on PFS)")
ap.add_argument("--sync-stacks", action="store_true", help="attribute synchronizing torch ops to Python call sites")
ap.add_argument("--stack-iters", type=int, default=2)
a = ap.parse_args()

cfg = TrainConfig(run_name=a.run_name)
cfg.data.npz_dir = Path(a.npz_dir)
cfg.env.obs_mode = a.obs_mode
tr = Trainer(cfg)
if a.init_ckpt:
    tr.load(Path(a.init_ckpt))  # weights, optimizers, sampler state; the replay buffer refills below
env, sch = tr.env, cfg.schedule
stats = {"returns": [], "returns_per_step": [], "lens": []}
obs, _ = env.reset()
tr._reset_episode_bounds(torch.arange(env.num_envs, device=tr.device))
for _ in range(sch.warmup_sim_steps):
    obs = tr._collect(obs, stats)


phase = ""

# Sub-phases (profiler annotations only, no change to the computation): imagination rollout and every
# backward pass get their own range; the rest of a phase (WM forward, critic fit, optimizer steps, metrics)
# stays attributed to the enclosing phase.
import supermimic.imagination.policy_trainer as _pt

_imagine = _pt.imagine


def _imagine_annotated(*args, **kw):
    with record_function("phase/policy.imagine"):
        return _imagine(*args, **kw)


_pt.imagine = _imagine_annotated
_backward = torch.Tensor.backward


def _backward_annotated(self, *args, **kw):
    with record_function(f"phase/{phase}.backward"):
        return _backward(self, *args, **kw)


torch.Tensor.backward = _backward_annotated


def iteration():
    global obs, phase
    n, tr._utd_acc = sim_steps_this_iter(tr._utd_acc, sch.updates_per_env_step)
    phase = "collect"
    with record_function("phase/collect"):
        for _ in range(n):
            obs = tr._collect(obs, stats)
    phase = "wm"
    with record_function("phase/wm"):
        tr.wm_trainer.update(tr.buffer)
    phase = "policy"
    with record_function("phase/policy"):
        tr.policy_trainer.update(tr.buffer)


t0 = time.time()
for i in range(a.settle):
    iteration()
torch.cuda.synchronize()
print(f"settle: {a.settle} iterations in {time.time() - t0:.1f} s (includes compilation)", flush=True)

t0 = time.time()
for i in range(a.active):
    iteration()
torch.cuda.synchronize()
plain = (time.time() - t0) / a.active
print(f"unprofiled: {plain * 1000:.0f} ms / iteration ({1 / plain:.2f} it/s)", flush=True)

sync_sites = {}
if a.sync_stacks:
    import traceback
    import warnings

    def on_warning(message, category, filename, lineno, file=None, line=None):
        if "synchronizing" not in str(message):
            return
        site = f"{filename}:{lineno}"
        for fr in reversed(traceback.extract_stack()[:-1]):
            if "supermimic" in fr.filename or "mjlab" in fr.filename:
                site = f"{fr.filename.split('/site-packages/')[-1].split('/src/')[-1]}:{fr.lineno} {fr.name}"
                break
        k = (phase, site)
        sync_sites[k] = sync_sites.get(k, 0) + 1

    show, warnings.showwarning = warnings.showwarning, on_warning
    with warnings.catch_warnings():
        warnings.showwarning = on_warning
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        for i in range(a.stack_iters):
            iteration()
        torch.cuda.set_sync_debug_mode(0)
    warnings.showwarning = show

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
    t0 = time.perf_counter()
    for i in range(a.active):
        iteration()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

events = prof.events()
# GPU-side events include the record_function/compiled-region annotations: keep only real device work.
ANNOT = ("phase/", "## Call", "Torch-Compiled Region", "CompiledFunction")
kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA and not e.name.startswith(ANNOT)]
iv = sorted((e.time_range.start, e.time_range.end) for e in kernels)
busy, cur_s, cur_e = 0.0, None, None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
if cur_e is not None:
    busy += cur_e - cur_s
busy_s = busy / 1e6
sync_names = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "aten::item", "aten::_local_scalar_dense")
syncs = {}
for e in events:
    if e.device_type == torch.autograd.DeviceType.CPU and any(e.name.startswith(n) for n in sync_names):
        d = syncs.setdefault(e.name, [0, 0.0])
        d[0] += 1
        d[1] += e.cpu_time_total / 1e6 if hasattr(e, "cpu_time_total") else 0.0
phase = {}
ranges = []
for e in events:
    if e.name.startswith("phase/") and e.device_type == torch.autograd.DeviceType.CPU:
        phase[e.name] = phase.get(e.name, 0.0) + (e.time_range.end - e.time_range.start) / 1e6
        ranges.append((e.time_range.start, e.time_range.end, e.name))
ranges.sort()
import bisect

starts = [r[0] for r in ranges]


def phase_of(t):
    # Innermost range containing t: ranges nest, so walk back from the latest start until one contains t.
    i = bisect.bisect_right(starts, t) - 1
    while i >= 0:
        if t <= ranges[i][1]:
            return ranges[i][2]
        i -= 1
    return "other"


per_phase = {}
ops_per_phase = {}  # top-level aten ops (no aten parent) by call count, per innermost phase
for e in events:
    if e.device_type != torch.autograd.DeviceType.CPU:
        continue
    if e.name.startswith("aten::") and not (e.cpu_parent is not None and e.cpu_parent.name.startswith("aten::")):
        d = ops_per_phase.setdefault(phase_of(e.time_range.start), {})
        d[e.name] = d.get(e.name, 0) + 1
    key = None
    if e.name in ("aten::item", "cudaStreamSynchronize", "cudaMemcpyAsync"):
        key = e.name
    elif e.name in ("cuLaunchKernel", "cudaLaunchKernel", "cudaLaunchKernelExC"):
        key = "launches"
    if key:
        d = per_phase.setdefault(phase_of(e.time_range.start), {})
        d[key] = d.get(key, 0) + 1

summary = {
    "obs_mode": a.obs_mode,
    "init_ckpt": a.init_ckpt,
    "profiled_iterations": a.active,
    "wall_ms_per_iter_profiled": wall / a.active * 1000,
    "wall_ms_per_iter_unprofiled": plain * 1000,
    "it_per_s_unprofiled": 1 / plain,
    "gpu_busy_fraction": busy_s / wall,
    "kernels_per_iter": len(kernels) / a.active,
    "sync_calls_per_iter": {k: v[0] / a.active for k, v in syncs.items()},
    "phase_ms_per_iter": {k: v / a.active * 1000 for k, v in phase.items()},
    "per_phase_counts_per_iter": {ph: {k: v / a.active for k, v in d.items()} for ph, d in per_phase.items()},
    "gpu_kernel_ms_per_iter": busy_s / a.active * 1000,
    "top_ops_per_phase_per_iter": {ph: dict(sorted(((k, v / a.active) for k, v in d.items()), key=lambda kv: -kv[1])[:15]) for ph, d in ops_per_phase.items()},
}
if a.sync_stacks:
    top = sorted(sync_sites.items(), key=lambda kv: -kv[1])
    summary["torch_syncs_per_iter"] = {ph: sum(v for (p_, _), v in sync_sites.items() if p_ == ph) / a.stack_iters for ph in ("collect", "wm", "policy")}
    summary["torch_sync_sites_per_iter"] = [{"phase": k[0], "site": k[1], "per_iter": v / a.stack_iters} for k, v in top]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "summary.json").write_text(json.dumps(summary, indent=1))
(out / "top_cuda.txt").write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
(out / "top_cpu.txt").write_text(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
if a.trace:
    prof.export_chrome_trace(str(out / "trace.json"))
print(json.dumps(summary, indent=1), flush=True)
