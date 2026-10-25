"""GAE targets, policy objective, and a short end-to-end training smoke run."""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data/npz/lafan1"


def test_gae_targets_match_hand_computation():
    from supermimic.imagination.objective import gae_targets

    H, n = 4, 3
    g, lam = 0.9, 0.8
    torch.manual_seed(0)
    r = torch.rand(H, n)
    d = torch.rand(H, n) * 0.3
    V = torch.rand(H + 1, n)
    out = gae_targets(r, d, V, g, lam)
    R = V[H].clone()
    for t in range(H - 1, -1, -1):
        R = r[t] + g * (1 - d[t]) * ((1 - lam) * V[t + 1] + lam * R)
        np.testing.assert_allclose(out[t].numpy(), R.numpy(), atol=1e-6)
    # lambda = 1, d = 0 -> Monte-Carlo discounted return with bootstrap.
    out1 = gae_targets(r, torch.zeros_like(d), V, g, 1.0)
    mc = r[0] + g * r[1] + g**2 * r[2] + g**3 * r[3] + g**4 * V[4]
    np.testing.assert_allclose(out1[0].numpy(), mc.numpy(), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
def test_training_smoke(tmp_path):
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="smoke", log_root=tmp_path, seed=1)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 128
    cfg.env.replay_capacity = 200
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    cfg.wm_trainer.batch_size = 256
    cfg.wm_trainer.batches_per_update = 2
    cfg.policy_trainer.batch_size = 128
    cfg.policy_trainer.horizon = 8
    cfg.schedule.total_iterations = 60
    cfg.schedule.warmup_iterations = 20
    cfg.schedule.log_interval = 20
    cfg.schedule.checkpoint_interval = 60
    tr = Trainer(cfg)
    try:
        tr.run()
        assert tr.num_updates > 0
        assert (tmp_path / "smoke" / "ckpt_latest.pt").exists()
        # Resume path works.
        cfg2 = TrainConfig(**{**cfg.__dict__, "resume": tmp_path / "smoke" / "ckpt_latest.pt", "run_name": "smoke2"})
        tr.env.close()
        tr2 = Trainer(cfg2)
        assert tr2.iteration == 60
        tr2.env.close()
    finally:
        pass


@pytest.mark.skipif(not torch.cuda.is_available() or not NPZ.exists(), reason="needs CUDA + LAFAN1 npz")
@pytest.mark.parametrize("obs_mode,mode", [("anchor_ori_now", "reduce-overhead"), ("beyondmimic_seq", "default"), ("anchor_pose9", "default")])
def test_compiled_imagined_step_matches_eager(tmp_path, obs_mode, mode):
    """The whole compiled imagined step (policy + world model + goal + noisy history) equals the eager one:
    same noise stream, same rollout, same policy gradient up to float fusion differences."""
    from supermimic.config import TrainConfig
    from supermimic.imagination.compiled import ImagRolloutStep
    from supermimic.imagination.objective import policy_objective
    from supermimic.imagination.rollout import imagine
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name="imag_compile", log_root=tmp_path, seed=1)
    cfg.data.clips = ["walk1_subject1"]
    cfg.env.num_envs = 64
    cfg.env.replay_capacity = 100
    cfg.env.obs_mode = obs_mode
    cfg.world_model.hidden = (256, 256)
    cfg.policy.encoder_hidden = (256, 128)
    cfg.policy.decoder_hidden = (256, 128)
    tr = Trainer(cfg)
    try:
        obs, _ = tr.env.reset()
        tr._reset_episode_bounds(torch.arange(tr.env.num_envs, device=tr.device))
        stats = {"returns": [], "returns_per_step": [], "lens": []}
        for _ in range(40):
            obs = tr._collect(obs, stats)
        start = tr.buffer.sample_starts(64, cfg.policy_trainer.history)
        q0 = tr.lib.default_joint_pos
        tr.wm.requires_grad_(False)
        out = {}
        for name, m in (("eager", None), ("compiled", mode)):
            step = ImagRolloutStep(tr.wm, tr.policy, q0, obs_mode, compile_mode=m)
            torch.manual_seed(123)
            ro = imagine(tr.wm, tr.policy, tr.lib, start, 8, cfg.policy_trainer.future_frames, obs_noise=True, step_fn=step, obs_mode=obs_mode)
            tr.policy.zero_grad(set_to_none=True)
            (-policy_objective(ro, None, 0.99)).backward()
            grad = torch.cat([p.grad.flatten() for p in tr.policy.parameters()])
            out[name] = (ro, grad.clone())
        (re, ge), (rc, gc) = out["eager"], out["compiled"]
        for f in ("states", "actions", "rewards", "dhat", "survival", "goals", "e_root"):
            torch.testing.assert_close(getattr(rc, f), getattr(re, f), rtol=1e-3, atol=1e-4, msg=f)
        assert torch.equal(rc.frames, re.frames)
        assert (gc - ge).abs().max() <= 1e-2 * ge.abs().max()
    finally:
        tr.wm.requires_grad_(True)
        tr.env.close()
