"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagRolloutStep
from supermimic.imagination.goals import ANCHOR_NOISE_HALF, ANCHOR_ORI_NOISE_HALF, OBS_NOISE_HALF, goal_noise_shape, goal_static, imagined_goal  # noqa: F401 (re-exported)
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import ReferenceFrame, analytic_observation
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, 71)
    actions: torch.Tensor  # (H, n, 29)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal dim of obs_mode)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def imagine(
    wm: WorldModel,
    policy: Policy,
    lib: MotionLibrary,
    start: StartBatch,
    horizon: int,
    future_frames: int = 10,
    obs_noise: bool = True,
    step_fn: ImagRolloutStep | None = None,
    obs_mode: str = "proprio_only",
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagRolloutStep(wm, policy, q0, obs_mode)
    noise_half = OBS_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = torch.stack([add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])])
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals = [torch.zeros(n, device=device)], [c]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references and the frame-only goal inputs for every step in one batched call.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))
    goal_all = goal_static(obs_mode, lib, frames_all.reshape(-1), ref_all, future_frames, wm.heads.anchor_body, ai)

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    def goal_static_at(t):
        return {k: v[t * n : (t + 1) * n] for k, v in goal_all.items()}

    ref = ref_at(0)
    if obs_mode != "proprio_only":
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [imagined_goal(obs_mode, lib, frame, ref, anchor_pose, future_frames, wm.heads.anchor_body, ai, obs_noise)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)

    # Observation noise of every step, drawn in the eager order (step t: observation, then goal), so the
    # compiled step contains no random-number generation and sees the same noise as the eager loop did.
    g_shape = goal_noise_shape(obs_mode, n, future_frames)
    u_obs, u_goal = [None] * horizon, [None] * horizon
    if obs_noise:
        for t in range(horizon):
            u_obs[t] = torch.rand(n, hist.shape[-1], device=device)
            u_goal[t] = torch.rand(g_shape, device=device) if g_shape is not None else None

    for t in range(horizon):
        a, s, r, term_mean, d, c, hist, goal, e_root = step_fn(s, a_prev, hist, goals[-1], c, start.xi, ref_at(t), u_obs[t], goal_static_at(t + 1), ref_at(t + 1), u_goal[t])
        term_sum += term_mean
        a_prev = a
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(goal)
        e_roots.append(e_root)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=frames_all,
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )
