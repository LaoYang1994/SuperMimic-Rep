"""Policy objective (Eq. 2) and critic GAE targets (Eq. 3)."""

from __future__ import annotations

import torch

from supermimic.imagination.rollout import ImagRollout


def policy_objective(ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + (gamma**H) * ro.survival[-1] * terminal_values
    return J.mean()


def policy_loss(J: torch.Tensor, horizon: int, divide_by_horizon: bool) -> torch.Tensor:
    """-J as in paper §3.3, so Table A.5's clipping (global norm 1.0, then element-wise 0.1) acts on grad J.

    divide_by_horizon reproduces the upstream -J / H: Adam is (up to eps) invariant to that scale, so its only
    effect is that the gradient stays ~H times below the clipping thresholds and they never trigger.
    """
    return -J / horizon if divide_by_horizon else -J


def clamp_values(v: torch.Tensor, g_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Critic estimates clamped to the attainable discounted return [-g_max, g_max] (g_max = max step reward / (1 - gamma)).

    Returns (clamped values, fraction of entries outside the range). Gradients pass through unclamped entries only.
    """
    return v.clamp(-g_max, g_max), (v.abs() > g_max).float().mean()


@torch.no_grad()
def gae_targets(rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        out[t] = R
    return out
