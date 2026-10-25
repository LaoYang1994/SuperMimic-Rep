"""Tracking-reward quaternion error: mjlab's forward values, finite gradients at zero error (CPU)."""

import torch
from mjlab.utils.lab_api.math import quat_box_minus

from supermimic.envs.mdp.rewards import _quat_box_minus, _sq_quat_err


def _unit(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True)


def test_forward_matches_mjlab():
    torch.manual_seed(0)
    q1 = _unit(torch.randn(512, 4, dtype=torch.float64))
    q2 = _unit(torch.randn(512, 4, dtype=torch.float64))
    tiny = _unit(q1 + 1e-8 * torch.randn_like(q1))  # below the Taylor threshold
    for a, b in [(q1, q2), (q1, q1), (q1, -q1), (q1, tiny)]:
        torch.testing.assert_close(_quat_box_minus(a, b), quat_box_minus(a, b), atol=1e-7, rtol=0)


def test_zero_error_gradient_is_finite():
    q = _unit(torch.randn(64, 4, dtype=torch.float64))
    for other in (q.clone(), -q.clone()):  # identical rotation, both quaternion signs
        a = q.clone().requires_grad_(True)
        b = other.requires_grad_(True)
        _sq_quat_err(a, b).sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, torch.zeros_like(a.grad), atol=1e-9, rtol=0)
