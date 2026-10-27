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


def cap_group_mass(
    h: torch.Tensor, group: torch.Tensor, num_groups: int, cap: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Limit every group's share of the probability vector ``h`` to ``cap`` (not in the paper).

    h: (N,) probabilities (the hard-sampling softmax over bins); group: (N,) group index of each entry (the clip
    of each bin). Water-filling: groups above the cap are set to exactly the cap, the removed mass goes to the
    other groups in proportion to their mass, and groups pushed over the cap by that are capped too. Relative
    weights inside a group are unchanged. The effective cap is max(cap, 1 / num_groups); cap <= 0, or no group
    above the cap, returns ``h`` unchanged (bitwise).

    Closed form (no iteration, no host sync): with group masses m sorted descending, total T and P_k the mass of
    the k largest, capping those k scales the rest by s_k = (T - k cap) / (T - P_k); the solution is the first k
    with m_k s_k <= cap (s_k does not decrease in k, so the k larger groups are indeed above the cap).

    On BONES-SEED the softmax of the paper (temperature 25 over ~1M bins) put 76-96% of the hard mass on the bins
    of a single clip for the whole run. Returns (capped h, group mass before, group mass after).
    """
    mass = torch.zeros(num_groups, dtype=h.dtype, device=h.device).index_add_(0, group, h)
    if cap <= 0:
        return h, mass, mass
    cap = max(cap, 1.0 / num_groups)
    m, order = torch.sort(mass, descending=True)
    k = torch.arange(num_groups, device=h.device, dtype=h.dtype)
    total = m.sum()
    rest = (total - (torch.cumsum(m, 0) - m)).clamp_min(torch.finfo(h.dtype).tiny)
    s = (total - k * cap) / rest
    fits = m * s <= cap
    fits[-1] = True  # with cap >= 1 / num_groups the last group always fits (guards against rounding)
    first = torch.argmax(fits.to(torch.int32))  # number of capped groups
    scale_sorted = torch.where(k < first, cap / m.clamp_min(torch.finfo(h.dtype).tiny), s[first])
    scale = torch.empty_like(scale_sorted).scatter_(0, order, scale_sorted)
    any_capped = first > 0
    return torch.where(any_capped, h * scale[group], h), mass, torch.where(any_capped, mass * scale, mass)


def bin_return_from_checkpoint(ck: dict, g_max: float):
    """Per-bin sampler returns of a checkpoint as completion in [0, 1]. Checkpoints from before the
    normalization stored raw discounted returns-to-go; dividing by g_max is only an approximation (their
    remaining-step information is lost)."""
    br = ck["bin_return"]
    if ck.get("bin_return_normalized"):
        return br
    print("[WARN] checkpoint predates normalized bin returns; converted approximately (bin_return / g_max)")
    return (br / g_max).clamp(0.0, 1.0)
