"""Return-based reset sampler: per-step returns relative to the attainable return (no mjlab dependency)."""

from __future__ import annotations

import torch


def normalized_returns_to_go(rtg: torch.Tensor, n_star: torch.Tensor, gamma: float, g_max: float) -> torch.Tensor:
    """Discounted return-to-go divided by the largest attainable one, clipped to [0, 1].

    rtg: (k, T) return-to-go R_t of k episodes (padded beyond each episode's length).
    n_star: (k,) length each episode would have had without failing (clip end or time limit).
    Step t can still earn at most g_max * (1 - gamma^m) with m = n_star - t remaining steps, so a clip-end bin
    tracked perfectly scores 1 instead of looking hard just because few steps remain. Steps with m <= 0
    (padding) return 0.
    """
    T = rtg.shape[1]
    m = n_star.to(torch.float64)[:, None] - torch.arange(T, device=rtg.device, dtype=torch.float64)[None, :]
    attainable = g_max * (1.0 - torch.pow(torch.tensor(gamma, dtype=torch.float64, device=rtg.device), m.clamp_min(0)))
    out = torch.where(m > 0, rtg.to(torch.float64) / attainable.clamp_min(1e-12), torch.zeros_like(m))
    return out.clamp(0.0, 1.0)


def bin_return_from_checkpoint(ck: dict, g_max: float):
    """Per-bin sampler returns of a checkpoint as completion in [0, 1]. Checkpoints from before the
    normalization stored raw discounted returns-to-go; dividing by g_max is only an approximation (their
    remaining-step information is lost)."""
    br = ck["bin_return"]
    if ck.get("bin_return_normalized"):
        return br
    print("[WARN] checkpoint predates normalized bin returns; converted approximately (bin_return / g_max)")
    return (br / g_max).clamp(0.0, 1.0)
