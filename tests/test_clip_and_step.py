"""Optimizer step guard: a non-finite gradient leaves weights and optimizer state untouched (CPU)."""

import copy

import torch

from supermimic.imagination.policy_trainer import clip_and_step


def _model_and_opt():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    m(torch.randn(8, 4)).sum().backward()  # one normal step so Adam has moments
    opt.step()
    opt.zero_grad()
    return m, opt


def test_nonfinite_gradient_skips_the_step():
    m, opt = _model_and_opt()
    w0, s0 = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    (m(torch.randn(8, 4)).sum() * float("nan")).backward()
    _, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0, clip_value=0.1)
    assert skipped
    for k, v in m.state_dict().items():
        assert torch.equal(v, w0[k])
    for (_, a), (_, b) in zip(sorted(opt.state_dict()["state"].items()), sorted(s0["state"].items())):
        assert all(torch.equal(a[k], b[k]) for k in a)
    assert all(p.grad is None for p in m.parameters())


def test_finite_gradient_steps():
    m, opt = _model_and_opt()
    w0 = copy.deepcopy(m.state_dict())
    m(torch.randn(8, 4)).sum().backward()
    gn, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0, clip_value=0.1)
    assert not skipped and torch.isfinite(gn)
    assert any(not torch.equal(v, w0[k]) for k, v in m.state_dict().items())
