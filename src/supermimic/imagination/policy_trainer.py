"""Short-horizon actor–critic through the frozen world model (paper §3.3, Table A.5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import ReplayBuffer, StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.objective import alive_mask, clamp_values, gae_targets, mask_dead, masked_mse, policy_loss, policy_objective
from supermimic.imagination.rollout import ImagRollout, imagine
from supermimic.policy.nets import Critic, Policy
from supermimic.training.distributed import allreduce_grads, allreduce_max_
from supermimic.world_model.world_model import WorldModel


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    # Clamp critic estimates to [-g_max, g_max] where they are used (V_H in the policy objective, values in the GAE
    # targets), and the GAE targets themselves as the recursion goes. Not in the paper; without it the critic
    # over-estimates a few states late in long runs and the policy follows (BONES-SEED: values up to 36.8 > 10
    # after ~145k iterations, then collapse).
    critic_value_clip: bool = True
    g_max: float = 10.0  # largest attainable discounted return; the Trainer sets max_step_reward / (1 - gamma)
    # The critic fits only imagined samples still alive (survival c_t >= this); dead ones count as terminated in the
    # GAE targets (r = 0, V = 0). After a predicted fall the rollout keeps running and the unbounded actions blow up
    # the action-rate penalty (to -1e17; critic loss 1e13 .. inf, then NaN weights at ~152k iterations). Not in the
    # paper (Eq. 3 fits every imagined step); 0 restores that.
    critic_min_survival: float = 0.001
    # True reproduces the upstream loss -J / H, under which the gradient clipping below never triggers.
    objective_divide_by_horizon: bool = False
    obs_noise: bool = True
    # torch.compile mode of the imagined step; "reduce-overhead" replays it as CUDA graphs (the compiled step
    # still has hundreds of kernels per call, so launch overhead dominates without graphs).
    compile: str | None = "reduce-overhead"
    obs_mode: str = "proprio_only"  # config.ObsMode; the Trainer copies EnvCfg.obs_mode here


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over ``mask`` (bool, same shape); 0 when the mask is empty. Masked-out entries may be non-finite."""
    m = mask.float()
    return torch.where(mask, x, 0.0).sum() / m.sum().clamp_min(1.0)


def fused_step(opt: torch.optim.Optimizer, skip: torch.Tensor) -> None:
    """``opt.step()`` that is a no-op where ``skip`` (0-dim float32 GPU tensor) is 1: fused Adam's ``found_inf``
    leaves weights, moments and ``step`` unchanged. No host sync."""
    if not opt.defaults.get("fused"):
        raise ValueError("fused_step needs a fused optimizer (found_inf is only supported there)")
    opt.found_inf = skip
    try:
        opt.step()
    finally:
        del opt.found_inf


def clip_and_step(
    opt: torch.optim.Optimizer, params, max_norm: float, clip_value: float | None = None, extra_skip: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip gradients and step, unless the total gradient norm is non-finite (or ``extra_skip`` is 1): then the
    step is skipped so one NaN/Inf cannot reach the weights or the optimizer moments.

    Returns (total grad norm before clipping, skipped as a 0-dim float tensor). The decision stays on the GPU.
    Gradients must already be all-reduced in data-parallel runs, so every rank takes the same decision.
    """
    params = [p for p in params if p.grad is not None]
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
    if clip_value is not None:
        torch.nn.utils.clip_grad_value_(params, clip_value)
    skipped = (~torch.isfinite(gn)).float()
    if extra_skip is not None:
        skipped = torch.maximum(skipped, extra_skip)
    fused_step(opt, skipped)
    return gn, skipped


class PolicyTrainer:
    def __init__(self, wm: WorldModel, policy: Policy, critic: Critic, lib: MotionLibrary, cfg: PolicyTrainerCfg):
        self.wm, self.policy, self.critic, self.lib, self.cfg = wm, policy, critic, lib, cfg
        # fused: see WorldModelTrainer.
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas, fused=True)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, fused=True)
        from supermimic.imagination.compiled import ImagRolloutStep

        self.step_fn = ImagRolloutStep(wm, policy, lib.default_joint_pos, cfg.obs_mode, compile_mode=cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        for g in self.critic_opt.param_groups:
            g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        return self.update_on(buffer.sample_starts(self.cfg.batch_size, self.cfg.history))

    def update_on(self, start: StartBatch) -> dict[str, float]:
        """One actor + critic update from pre-sampled imagination start states."""
        cfg = self.cfg
        xi_norm = self.wm.spec.normalize_xi(start.xi)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        ro = imagine(self.wm, self.policy, self.lib, start, cfg.horizon, cfg.future_frames, obs_noise=cfg.obs_noise, step_fn=self.step_fn, obs_mode=cfg.obs_mode)
        feat, goals, e_root = self._critic_inputs(ro)
        if cfg.use_critic:
            v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
            if cfg.critic_value_clip:
                v_H, vH_clamped = clamp_values(v_H, cfg.g_max)
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = policy_loss(J, cfg.horizon, cfg.objective_divide_by_horizon)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        gn, skipped = clip_and_step(self.opt, self.policy.parameters(), cfg.grad_clip_norm, cfg.grad_clip_value)
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        alive = alive_mask(ro.survival.detach(), cfg.critic_min_survival)  # (H+1, n)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_return_undisc": float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach()),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "grad_clipped": float(gn > cfg.grad_clip_norm),  # non-finite gn counts too: see skipped_nonfinite
            "skipped_nonfinite": float(skipped),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
            # Tails (means hide a single runaway sample); the loop keeps the interval max / min of *_max / *_min.
            "imag_dead_frac": float((~alive[:-1]).float().mean()),
            "imag_action_abs_max": float(ro.actions.detach().abs().max()),
            "imag_reward_min": float(ro.rewards.detach().min()),
        }
        if cfg.use_critic and cfg.critic_value_clip:
            metrics["critic_vH_clamped_frac"] = float(vH_clamped)
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                # Dead samples (c_t < critic_min_survival) are not fitted. Their inputs can be huge or non-finite, so
                # they are replaced by zeros before the critic sees them: a zero loss weight alone would still give
                # 0 * inf = NaN.
                a3 = alive[..., None]
                feat_d = torch.where(a3, feat.detach(), 0.0)
                goals_d = torch.where(a3, goals.detach(), 0.0)
                e_root_d = torch.where(a3, e_root.detach(), 0.0)
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(feat_d.reshape(H1 * n, -1), goals_d.reshape(H1 * n, -1), xi_rep.reshape(H1 * n, -1), e_root_d.reshape(H1 * n, -1)).reshape(H1, n)
                values_raw = values
                if cfg.critic_value_clip:
                    values, _ = clamp_values(values, cfg.g_max)
                r_m, d_m, v_m = mask_dead(ro.rewards.detach(), ro.dhat[1:].detach(), values, alive)
                targets = gae_targets(r_m, d_m, v_m, cfg.gamma, cfg.gae_lambda, clip=cfg.g_max if cfg.critic_value_clip else None)  # (H, n)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_er = e_root_d[:-1].reshape(-1, 4)
                Y = targets.reshape(-1)
                W = alive[:-1].reshape(-1).float()  # loss weight: 1 alive, 0 dead
                # One host sync per update (not per minibatch). Finite inputs and targets do not guarantee finite
                # gradients (a huge loss overflows in backward), so each minibatch also skips on a non-finite grad
                # norm. Every critic minibatch all-reduces its gradients, so all ranks must agree: skip if any rank
                # has non-finite inputs (dead inputs are already zeros; only alive targets count).
                skip = (~torch.stack([torch.isfinite(t).all() for t in (X_feat, X_goal, X_xi, X_er, torch.where(W > 0, Y, 0.0))]).all()).float()
                allreduce_max_(skip)
            metrics["critic_skipped_nonfinite"] = float(skip)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            loss_max = torch.zeros((), device=Y.device)
            grad_skips = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs, empty = masked_mse(pred, Y[idx], W[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    # A minibatch without alive samples has zero gradient, but Adam's momentum would still move the
                    # weights: skip it, only when it is empty on every rank (all ranks step together).
                    empty = 1.0 - allreduce_max_(1.0 - empty)
                    batch_skip = torch.maximum(skip, empty)
                    _, skipped_mb = clip_and_step(self.critic_opt, self.critic.parameters(), cfg.critic_grad_clip, extra_skip=batch_skip)
                    grad_skips += skipped_mb - batch_skip  # minibatches skipped only for their gradient
                    last = closs.detach()
                    loss_max = torch.maximum(loss_max, last)
            last = torch.where(skip > 0, torch.zeros_like(last), last)
            loss_max = torch.where(skip > 0, torch.zeros_like(loss_max), loss_max)
            metrics["critic_loss"] = float(last)
            metrics["critic_loss_max"] = float(loss_max)
            metrics["critic_skipped_nonfinite_grad"] = float(grad_skips) / (cfg.critic_epochs * cfg.critic_minibatches)
            # Statistics over alive samples only: dead ones carry the critic's output on zero inputs and zero targets.
            metrics["critic_value_mean"] = float(_masked_mean(values_raw, alive))  # before clamping: over-estimation stays visible
            metrics["critic_target_mean"] = float(_masked_mean(targets, alive[:-1]))
            if cfg.critic_value_clip:
                metrics["critic_values_clamped_frac"] = float(_masked_mean((values_raw.abs() > cfg.g_max).float(), alive))
                metrics["critic_targets_clamped_frac"] = float(_masked_mean((targets.abs() >= cfg.g_max).float(), alive[:-1]))
        return metrics
