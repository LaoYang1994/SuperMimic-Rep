"""Numerical-equivalence harness for training-throughput work (change 2026-10-24-training-throughput).

Freeze the inputs of one training iteration, then run that iteration's updates on two code versions and
compare. GPU backward passes use atomics, so first compare a version with itself (`--run` twice) to get the
noise floor; a candidate is equivalent if its differences to the baseline are within that floor.

  # 1. on the baseline code: train state + filled replay -> frozen inputs
  python scripts/perf/update_parity.py save --init-ckpt logs/lafan1_now_fixed@A/ckpt_0008000.pt --dir $SM_BANK/perf/parity
  # 2. on any code version: one WM update + one actor/critic update on those inputs
  python scripts/perf/update_parity.py run --dir $SM_BANK/perf/parity --out base_a.pt
  # 3. compare two runs
  python scripts/perf/update_parity.py compare base_a.pt base_b.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch


def _trainer(obs_mode: str, npz_dir: str, run_name: str):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name=run_name)
    cfg.data.npz_dir = Path(npz_dir)
    cfg.env.obs_mode = obs_mode
    return Trainer(cfg)


def save(a) -> None:
    tr = _trainer(a.obs_mode, a.npz_dir, "parity_save")
    if a.init_ckpt:
        tr.load(Path(a.init_ckpt))
    obs, _ = tr.env.reset()
    tr._reset_episode_bounds(torch.arange(tr.env.num_envs, device=tr.device))
    stats = {"returns": [], "returns_per_step": [], "lens": []}
    for _ in range(a.fill_steps):
        obs = tr._collect(obs, stats)
    wcfg, pcfg = tr.wm_trainer.cfg, tr.policy_trainer.cfg
    windows = [asdict(tr.buffer.sample_windows(wcfg.batch_size, wcfg.window)) for _ in range(wcfg.batches_per_update)]
    start = asdict(tr.buffer.sample_starts(pcfg.batch_size, pcfg.history))
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_mode": a.obs_mode,
            "npz_dir": a.npz_dir,
            "windows": windows,
            "start": start,
            "wm": tr.wm.state_dict(),
            "policy": tr.policy.state_dict(),
            "critic": tr.critic.state_dict(),
            "wm_opt": tr.wm_trainer.opt.state_dict(),
            "policy_opt": tr.policy_trainer.opt.state_dict(),
            "critic_opt": tr.policy_trainer.critic_opt.state_dict(),
            "lr": tr.wm_trainer.opt.param_groups[0]["lr"],
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        },
        d / "inputs.pt",
    )
    print("saved", d / "inputs.pt")


def run(a) -> None:
    from supermimic.data.replay_buffer import StartBatch, WindowBatch

    snap = torch.load(Path(a.dir) / "inputs.pt", map_location="cuda:0", weights_only=False)
    tr = _trainer(snap["obs_mode"], snap["npz_dir"], "parity_run")
    tr.wm.load_state_dict(snap["wm"])
    tr.policy.load_state_dict(snap["policy"])
    tr.critic.load_state_dict(snap["critic"])
    # Adam's load_state_dict moves `step` to the device a fused/capturable optimizer expects.
    for opt, key in ((tr.wm_trainer.opt, "wm_opt"), (tr.policy_trainer.opt, "policy_opt"), (tr.policy_trainer.critic_opt, "critic_opt")):
        opt.load_state_dict(snap[key])
    tr.wm_trainer.set_lr(snap["lr"])
    tr.policy_trainer.set_lr(snap["lr"])
    torch.set_rng_state(snap["rng_cpu"])
    torch.cuda.set_rng_state_all(snap["rng_cuda"])
    windows = [WindowBatch(**w) for w in snap["windows"]]
    start = StartBatch(**snap["start"])
    m_wm = tr.wm_trainer.update_on(windows)
    m_pi = tr.policy_trainer.update_on(start)
    torch.cuda.synchronize()
    to_f = lambda m: {k: float(v) for k, v in m.items()}  # noqa: E731
    torch.save(
        {
            "wm_metrics": to_f(m_wm),
            "policy_metrics": to_f(m_pi),
            "wm": {k: v.detach().cpu() for k, v in tr.wm.state_dict().items()},
            "policy": {k: v.detach().cpu() for k, v in tr.policy.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in tr.critic.state_dict().items()},
        },
        a.out,
    )
    print("saved", a.out, "| wm", to_f(m_wm), "| policy", {k: v for k, v in to_f(m_pi).items() if not k.startswith("imag_r/")})


def selfcheck(a) -> None:
    """T5b: on one code version, `update(buffer)` and "sample, then update_on" give the same result."""
    import copy

    tr = _trainer(a.obs_mode, a.npz_dir, "parity_selfcheck")
    if a.init_ckpt:
        tr.load(Path(a.init_ckpt))
    obs, _ = tr.env.reset()
    tr._reset_episode_bounds(torch.arange(tr.env.num_envs, device=tr.device))
    stats = {"returns": [], "returns_per_step": [], "lens": []}
    for _ in range(a.fill_steps):
        obs = tr._collect(obs, stats)
    mods = {"wm": tr.wm, "policy": tr.policy, "critic": tr.critic}
    opts = {"wm": tr.wm_trainer.opt, "policy": tr.policy_trainer.opt, "critic": tr.policy_trainer.critic_opt}
    m0 = {k: copy.deepcopy(m.state_dict()) for k, m in mods.items()}
    o0 = {k: copy.deepcopy(o.state_dict()) for k, o in opts.items()}
    rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())

    def restore():
        for k, m in mods.items():
            m.load_state_dict(m0[k])
        for k, o in opts.items():
            o.load_state_dict(o0[k])
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])

    def params():
        return {k: {n: v.detach().clone() for n, v in m.state_dict().items()} for k, m in mods.items()}

    restore()
    tr.wm_trainer.update(tr.buffer)
    tr.policy_trainer.update(tr.buffer)
    pa = params()
    restore()  # same path again: the run-to-run noise floor of this GPU code
    tr.wm_trainer.update(tr.buffer)
    tr.policy_trainer.update(tr.buffer)
    pa2 = params()
    restore()
    wcfg, pcfg = tr.wm_trainer.cfg, tr.policy_trainer.cfg
    windows = [tr.buffer.sample_windows(wcfg.batch_size, wcfg.window) for _ in range(wcfg.batches_per_update)]
    tr.wm_trainer.update_on(windows)
    tr.policy_trainer.update_on(tr.buffer.sample_starts(pcfg.batch_size, pcfg.history))
    pb = params()
    def worst(p, q, k):
        return max(((p[k][n] - q[k][n]).abs().max() / p[k][n].abs().max().clamp_min(1e-12)).item() for n in p[k] if torch.is_floating_point(p[k][n]))

    for k in pa:
        print(f"selfcheck {k:7s} max relative diff: update(buffer) vs update_on {worst(pa, pb, k):.2e} | update(buffer) twice (noise floor) {worst(pa, pa2, k):.2e}")


def compare(a) -> None:
    x, y = torch.load(a.a, weights_only=False), torch.load(a.b, weights_only=False)
    for part in ("wm_metrics", "policy_metrics"):
        for k in sorted(set(x[part]) & set(y[part])):
            u, v = x[part][k], y[part][k]
            rel = abs(u - v) / max(abs(u), abs(v), 1e-12)
            print(f"{part:15s} {k:32s} {u: .6e} {v: .6e} rel {rel:.2e}")
    for part in ("wm", "policy", "critic"):
        worst, worst_k = 0.0, ""
        for k in x[part]:
            if not torch.is_floating_point(x[part][k]):
                continue
            d = (x[part][k] - y[part][k]).abs().max().item()
            s = x[part][k].abs().max().item()
            r = d / max(s, 1e-12)
            if r > worst:
                worst, worst_k = r, k
        print(f"params {part:7s} max relative diff {worst:.2e} ({worst_k})")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("save")
s.add_argument("--dir", required=True)
s.add_argument("--init-ckpt", default=None)
s.add_argument("--obs-mode", default="anchor_ori_now")
s.add_argument("--npz-dir", default="data/npz/lafan1")
s.add_argument("--fill-steps", type=int, default=400)
r = sub.add_parser("run")
r.add_argument("--dir", required=True)
r.add_argument("--out", required=True)
sc = sub.add_parser("selfcheck")
sc.add_argument("--init-ckpt", default=None)
sc.add_argument("--obs-mode", default="anchor_ori_now")
sc.add_argument("--npz-dir", default="data/npz/lafan1")
sc.add_argument("--fill-steps", type=int, default=400)
c = sub.add_parser("compare")
c.add_argument("a")
c.add_argument("b")
args = ap.parse_args()
{"save": save, "run": run, "selfcheck": selfcheck, "compare": compare}[args.cmd](args)
