"""Optimizer step guard: a non-finite gradient leaves weights and optimizer state (incl. `step`) untouched (CPU)."""

import copy

import torch

import pytest

from supermimic.imagination.policy_trainer import clip_and_step, fused_step


def _model_and_opt():
    torch.manual_seed(0)
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, fused=True)
    m(torch.randn(8, 4)).sum().backward()  # one normal step so Adam has moments
    opt.step()
    opt.zero_grad()
    return m, opt


def test_nonfinite_gradient_skips_the_step():
    m, opt = _model_and_opt()
    w0, s0 = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    (m(torch.randn(8, 4)).sum() * float("nan")).backward()
    _, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0, clip_value=0.1)
    assert skipped.item() == 1.0
    for k, v in m.state_dict().items():
        assert torch.equal(v, w0[k])
    for (_, a), (_, b) in zip(sorted(opt.state_dict()["state"].items()), sorted(s0["state"].items())):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_finite_gradient_steps():
    m, opt = _model_and_opt()
    w0 = copy.deepcopy(m.state_dict())
    m(torch.randn(8, 4)).sum().backward()
    gn, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0, clip_value=0.1)
    assert skipped.item() == 0.0 and torch.isfinite(gn)
    assert any(not torch.equal(v, w0[k]) for k, v in m.state_dict().items())


def test_fused_step_rejects_a_non_fused_optimizer():
    m = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    with pytest.raises(ValueError):
        fused_step(opt, torch.zeros(()))


def test_finite_loss_with_overflowing_grad_norm_skips_the_step():
    """A huge but finite critic loss (A: 6.4e30) can overflow in backward; inputs and targets were finite there."""
    m, opt = _model_and_opt()
    with torch.no_grad():
        m.weight.fill_(1.0)
        m.bias.zero_()
    w0, s0 = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict())
    loss = torch.nn.functional.mse_loss(m(torch.full((1, 4), 3e18)), torch.zeros(1, 2))
    assert torch.isfinite(loss)
    loss.backward()
    gn, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0)
    assert not torch.isfinite(gn) and skipped.item() == 1.0
    for k, v in m.state_dict().items():
        assert torch.equal(v, w0[k])
    for (_, a), (_, b) in zip(sorted(opt.state_dict()["state"].items()), sorted(s0["state"].items())):
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_extra_skip_skips_a_finite_step():
    m, opt = _model_and_opt()
    w0 = copy.deepcopy(m.state_dict())
    m(torch.randn(8, 4)).sum().backward()
    gn, skipped = clip_and_step(opt, m.parameters(), max_norm=1.0, extra_skip=torch.ones(()))
    assert torch.isfinite(gn) and skipped.item() == 1.0
    for k, v in m.state_dict().items():
        assert torch.equal(v, w0[k])
