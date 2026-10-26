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


def alive_mask(survival: torch.Tensor, eps: float) -> torch.Tensor:
    """Imagined samples still alive (c_t >= eps), (H+1, n) bool. eps <= 0 keeps every sample (the paper's Eq. 3).

    After the world model predicts a fall (c_t -> 0) the rollout keeps running: states and the unbounded actions
    feed each other and the action-rate penalty reaches -1e17. J weights these steps by c_t; the critic fit must
    drop them too.
    """
    if eps <= 0:
        return torch.ones_like(survival, dtype=torch.bool)
    return survival >= eps


@torch.no_grad()
def mask_dead(rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, alive: torch.Tensor):
    """Treat dead samples as terminated for the GAE targets: r_{t+1} = 0 (where c_t is dead), d_{t+1} = 1 (where
    c_{t+1} is dead), V_t = 0 (where c_t is dead). Substituted, not multiplied, so NaN/Inf there cannot leak.

    rewards (H, n), dhat_next (H, n), values (H+1, n), alive (H+1, n). Returns the three masked tensors.
    """
    rewards = torch.where(alive[:-1], rewards, 0.0)
    dhat_next = torch.where(alive[1:], dhat_next, 1.0)
    values = torch.where(alive, values, 0.0)
    return rewards, dhat_next, values


@torch.no_grad()
def gae_targets(rewards: torch.Tensor, dhat_next: torch.Tensor, values: torch.Tensor, gamma: float, lam: float, clip: float | None = None) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    clip: if set, every R_t (and R_H) is clamped to [-clip, clip] as the recursion goes, so one runaway reward
    cannot leak into the earlier targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    if clip is not None:
        R = R.clamp(-clip, clip)
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * ((1.0 - lam) * values[t + 1] + lam * R)
        if clip is not None:
            R = R.clamp(-clip, clip)
        out[t] = R
    return out


def masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MSE over the entries with weight > 0: sum w (p - y)^2 / max(sum w, 1). Returns (loss, empty), empty = 1.0 when
    no entry has weight (the caller skips that optimizer step). No host sync.

    Zero-weight targets may be non-finite; they are replaced before the difference, so 0 * inf never appears in the
    loss or its gradient (``pred`` itself must be finite: compute it on substituted inputs).
    """
    target = torch.where(weight > 0, target, 0.0)
    w_sum = weight.sum()
    loss = (weight * (pred - target) ** 2).sum() / w_sum.clamp_min(1.0)
    return loss, (w_sum == 0).float()
