"""Return-based sampler: bin returns relative to the attainable return, and checkpoint conversion (CPU)."""

import torch

from supermimic.envs.mdp.sampler_math import bin_return_from_checkpoint, normalized_returns_to_go

GAMMA, G_MAX = 0.99, 10.0
R_MAX = G_MAX * (1 - GAMMA)  # per-step reward bound (0.1)


def _rtg(r: torch.Tensor) -> torch.Tensor:
    """Discounted return-to-go of each row (r already zero beyond each episode)."""
    T = r.shape[1]
    gp = GAMMA ** torch.arange(T, dtype=torch.float64)
    return torch.flip(torch.cumsum(torch.flip(r * gp, [1]), 1), [1]) / gp


def test_perfect_tracking_to_the_clip_end_is_complete_including_the_last_steps():
    # Short clip-end episode (7 steps) and a long one: both tracked at the reward bound.
    T = 300
    lens = torch.tensor([7, 300])
    r = (torch.arange(T)[None, :] < lens[:, None]).double() * R_MAX
    g = normalized_returns_to_go(_rtg(r), lens, GAMMA, G_MAX)
    for i, n in enumerate(lens.tolist()):
        assert torch.allclose(g[i, :n], torch.ones(n, dtype=torch.float64), atol=1e-9)
        assert (g[i, n:] == 0).all()  # padding


def test_failure_mid_clip_scores_low_near_the_failure():
    # Tracks perfectly for 60 steps, then fails although 240 more steps were attainable.
    T, n, n_star = 300, 60, 300
    r = (torch.arange(T)[None, :] < n).double() * R_MAX
    g = normalized_returns_to_go(_rtg(r), torch.tensor([n_star]), GAMMA, G_MAX)[0]
    assert g[n - 1] < 0.1
    assert g[0] < 0.6
    assert (g[:n].diff() <= 1e-12).all()  # completion only drops towards the failure


def test_negative_returns_clip_to_zero():
    r = torch.full((1, 10), -0.05, dtype=torch.float64)
    g = normalized_returns_to_go(_rtg(r), torch.tensor([10]), GAMMA, G_MAX)
    assert (g == 0).all()


def test_checkpoint_conversion():
    br = torch.tensor([10.0, 5.0, 0.0, 12.0])
    old = bin_return_from_checkpoint({"bin_return": br}, G_MAX)
    assert torch.allclose(old, torch.tensor([1.0, 0.5, 0.0, 1.0]))
    new = torch.tensor([0.9, 0.2])
    assert torch.equal(bin_return_from_checkpoint({"bin_return": new, "bin_return_normalized": True}, G_MAX), new)
