"""The critic fits only imagined samples still alive (c_t >= eps); dead ones count as terminated (CPU)."""

import math

import torch

from supermimic.imagination.objective import alive_mask, gae_targets, mask_dead, masked_mse
from supermimic.policy.nets import Critic, CriticCfg

GAMMA, LAM, EPS = 0.99, 0.95, 1e-3


def _rollout(H=8, n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(H, n, generator=g) * 0.1
    d = torch.rand(H, n, generator=g) * 0.05
    V = 8.0 + torch.rand(H + 1, n, generator=g)
    return r, d, V


def test_dead_tail_is_terminated_and_cannot_leak():
    H, k = 8, 4  # sample 1 is dead from s_4 on (c_4 = 0)
    r, d, V = _rollout(H)
    surv = torch.ones(H + 1, 3)
    surv[k:, 1] = 0.0
    r_bad, d_bad, V_bad = r.clone(), d.clone(), V.clone()
    r_bad[k:, 1] = -1e17  # r_{t+1} for dead c_t: the runaway action-rate penalty
    d_bad[k - 1 :, 1] = math.nan  # d_4 .. d_H of dead states
    V_bad[k:, 1] = math.nan
    out = gae_targets(*mask_dead(r_bad, d_bad, V_bad, alive_mask(surv, EPS)), GAMMA, LAM)

    # Sample 1 = an episode that terminates at s_4: R_3 = r_4, then the usual recursion.
    R = r[k - 1, 1]
    expected = [R]
    for t in range(k - 2, -1, -1):
        R = r[t, 1] + GAMMA * (1.0 - d[t, 1]) * ((1.0 - LAM) * V[t + 1, 1] + LAM * R)
        expected.insert(0, R)
    torch.testing.assert_close(out[:k, 1], torch.stack(expected))
    assert torch.equal(out[k:, 1], torch.zeros(H - k))
    # Samples alive throughout are bitwise the old targets.
    clean = gae_targets(r, d, V, GAMMA, LAM)
    assert torch.equal(out[:, [0, 2]], clean[:, [0, 2]])


def test_eps_zero_and_no_clip_reproduce_old_targets():
    r, d, V = _rollout(H=16, n=64, seed=1)
    surv = torch.rand(17, 64)
    surv[5:, :8] = 0.0  # even c_t = 0 is kept when eps = 0 (the paper's Eq. 3)
    alive = alive_mask(surv, 0.0)
    assert bool(alive.all())
    assert torch.equal(gae_targets(*mask_dead(r, d, V, alive), GAMMA, LAM), gae_targets(r, d, V, GAMMA, LAM))


def test_clip_bounds_targets_and_is_inactive_in_range():
    r, d, V = _rollout()
    assert torch.equal(gae_targets(r, d, V, GAMMA, LAM, clip=10.0), gae_targets(r, d, V, GAMMA, LAM))  # all |R| < 10
    r[2, 0] = 1e6  # alive sample, runaway reward the termination head missed
    V[5, 1] = -1e8
    out = gae_targets(r, d, V, GAMMA, LAM, clip=10.0)
    assert out.abs().max() <= 10.0
    assert gae_targets(r, d, V, GAMMA, LAM).abs().max() > 10.0


def test_masked_mse_on_substituted_inputs_is_finite():
    torch.manual_seed(0)
    critic = Critic(CriticCfg(hidden=(16, 8)))
    N = 12
    alive = torch.ones(N, dtype=torch.bool)
    alive[[2, 7, 11]] = False
    feat, goal, xi, er = torch.randn(N, 71), torch.randn(N, 680), torch.randn(N, 33), torch.randn(N, 4)
    feat[2], goal[7], er[11] = math.inf, math.nan, -math.inf
    y = torch.randn(N)
    y[[2, 7]] = -1e17
    y[11] = math.nan
    a = alive[:, None]
    pred = critic(torch.where(a, feat, 0.0), torch.where(a, goal, 0.0), xi, torch.where(a, er, 0.0))
    loss, empty = masked_mse(pred, y, alive.float())
    torch.testing.assert_close(loss, torch.nn.functional.mse_loss(pred[alive], y[alive]))
    assert empty.item() == 0.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters())

    critic.zero_grad()
    pred = critic(torch.zeros(N, 71), torch.zeros(N, 680), xi, torch.zeros(N, 4))
    loss, empty = masked_mse(pred, y, torch.zeros(N))
    assert empty.item() == 1.0 and loss.item() == 0.0
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in critic.parameters())
