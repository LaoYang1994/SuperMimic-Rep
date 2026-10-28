"""Per-clip cap on the return sampler's hard mass (CPU)."""

import torch

from supermimic.envs.mdp.sampler_math import cap_group_mass


def _group_mass(h: torch.Tensor, group: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, dtype=h.dtype).index_add_(0, group, h)


def _softmax_case(seed: int = 0, n_groups: int = 200, bins_per_group: int = 5):
    """A sharp softmax over bins where one group holds almost all the mass (the captured-clip situation)."""
    g = torch.Generator().manual_seed(seed)
    group = torch.arange(n_groups).repeat_interleave(bins_per_group)
    G = 0.9 + 0.1 * torch.rand(len(group), generator=g, dtype=torch.float64)
    G[:bins_per_group] = 0.2  # group 0 never improves
    G[bins_per_group : 2 * bins_per_group] = 0.5
    return torch.softmax(-25 * G, 0), group, n_groups


def test_every_group_is_at_most_the_cap_and_mass_is_kept():
    h, group, n = _softmax_case()
    out, before, after = cap_group_mass(h, group, n, cap=0.01)
    assert before.max() > 0.9  # captured before
    m = _group_mass(out, group, n)
    assert m.max() <= 0.01 + 1e-9
    assert torch.allclose(out.sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-9)
    assert torch.allclose(after, m, atol=1e-12)


def test_no_group_over_the_cap_or_cap_zero_leaves_h_bitwise_unchanged():
    g = torch.Generator().manual_seed(1)
    group = torch.arange(100).repeat_interleave(3)
    h = torch.rand(300, generator=g)
    h = h / h.sum()  # every group ~1%, all below 5%
    out, _, _ = cap_group_mass(h, group, 100, cap=0.05)
    assert torch.equal(out, h)
    hs, gs, n = _softmax_case()
    out0, _, _ = cap_group_mass(hs, gs, n, cap=0.0)
    assert torch.equal(out0, hs)


def test_few_groups_relax_the_cap_to_an_even_split():
    # 4 groups with cap 0.01: the effective cap is 1/4, every group ends at exactly 1/4.
    group = torch.tensor([0, 0, 1, 2, 3, 3])
    h = torch.tensor([0.7, 0.2, 0.05, 0.03, 0.01, 0.01], dtype=torch.float64)
    out, _, _ = cap_group_mass(h, group, 4, cap=0.01)
    assert torch.allclose(_group_mass(out, group, 4), torch.full((4,), 0.25, dtype=torch.float64), atol=1e-12)


def test_relative_weights_inside_a_group_are_kept():
    h, group, n = _softmax_case()
    out, _, _ = cap_group_mass(h, group, n, cap=0.01)
    for k in (0, 1, 7):
        i = group == k
        assert torch.allclose(out[i] / out[i].sum(), h[i] / h[i].sum(), atol=1e-12)


def test_excess_goes_to_uncapped_groups_in_proportion_and_cascades():
    # Group 0 (0.6) is over the cap 0.3; redistributing its excess pushes group 1 (0.3) over too.
    # Capped: 0.3 + 0.3; the remaining 0.4 goes to groups 2-5 in proportion to their mass (equal here).
    group = torch.tensor([0, 0, 1, 2, 3, 4, 5])
    h = torch.tensor([0.4, 0.2, 0.3, 0.025, 0.025, 0.025, 0.025], dtype=torch.float64)
    out, _, _ = cap_group_mass(h, group, 6, cap=0.3)
    expect = torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1], dtype=torch.float64)
    assert torch.allclose(_group_mass(out, group, 6), expect, atol=1e-12)
    assert torch.allclose(out[:2], torch.tensor([0.2, 0.1], dtype=torch.float64), atol=1e-12)
