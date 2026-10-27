"""Critic value clamping and the paper's objective scale / gradient clipping (CPU)."""

from types import SimpleNamespace

import torch

from supermimic.imagination.objective import clamp_values, gae_targets, policy_loss, policy_objective
from supermimic.imagination.policy_trainer import clip_and_step

G_MAX = 10.0


def _rollout(H=4, n=3, r=0.1):
    return SimpleNamespace(rewards=torch.full((H, n), r), survival=torch.ones(H + 1, n))


def test_clamp_values_bounds_and_fraction():
    v = torch.tensor([-12.0, -10.0, 3.0, 10.0, 36.8])
    out, frac = clamp_values(v, G_MAX)
    assert torch.equal(out, torch.tensor([-10.0, -10.0, 3.0, 10.0, 10.0]))
    assert abs(frac.item() - 2 / 5) < 1e-6


def test_saturated_bootstrap_has_no_gradient():
    v_H = torch.tensor([5.0, 20.0, -15.0], requires_grad=True)
    clamped, _ = clamp_values(v_H, G_MAX)
    policy_objective(_rollout(), clamped, 0.99).backward()
    g = v_H.grad
    assert g[0] > 0 and g[1] == 0 and g[2] == 0


def test_gae_targets_bounded_after_clamp():
    H, n, gamma, lam = 8, 5, 0.99, 0.95
    torch.manual_seed(0)
    r = torch.rand(H, n) * 0.1  # max step reward 0.1 -> g_max = 10
    d = torch.rand(H, n) * 0.2
    V = 10.0 + torch.rand(H + 1, n) * 30.0  # over-estimating critic, like B after ~145k iterations
    raw = gae_targets(r, d, V, gamma, lam)
    clamped, frac = clamp_values(V, G_MAX)
    out = gae_targets(r, d, clamped, gamma, lam)
    assert frac.item() == 1.0
    assert raw.max() > G_MAX
    assert out.max() <= 0.1 + gamma * G_MAX + 1e-5  # one bounded reward plus a discounted bounded value


def _grad_of(divide: bool, H: int) -> torch.Tensor:
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 1)
    J = m(torch.randn(16, 4)).mean() * 4.0  # global grad norm ~4.4 under -J, ~0.14 under -J / 32
    policy_loss(J, H, divide).backward()
    return torch.cat([p.grad.flatten() for p in m.parameters()]), m


def test_policy_loss_scale_and_clipping():
    H = 32
    g_paper, m_paper = _grad_of(False, H)
    g_upstream, m_up = _grad_of(True, H)
    torch.testing.assert_close(g_paper, g_upstream * H)
    opt_p = torch.optim.Adam(m_paper.parameters(), lr=1e-3, fused=True)
    opt_u = torch.optim.Adam(m_up.parameters(), lr=1e-3, fused=True)
    gn_p, _ = clip_and_step(opt_p, m_paper.parameters(), max_norm=1.0, clip_value=0.1)
    gn_u, _ = clip_and_step(opt_u, m_up.parameters(), max_norm=1.0, clip_value=0.1)
    assert gn_p > 1.0 > gn_u  # clipping triggers under the paper scale, not under -J / H
