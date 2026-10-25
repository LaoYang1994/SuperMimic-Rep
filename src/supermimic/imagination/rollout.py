"""Imagined rollouts through the world model (paper §3.3)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from supermimic.data.replay_buffer import StartBatch
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.imagination.compiled import REWARD_TERM_NAMES, ImagStep
from supermimic.policy.nets import Policy
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, anchor_ori_obs, anchor_relative_obs, beyondmimic_seq_goal, flatten_history_term_major
from supermimic.world_model.wm_trainer import reference_at
from supermimic.world_model.world_model import WorldModel

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)  # anchor_pose9: position 3 + orientation 6D
ANCHOR_ORI_NOISE_HALF = 0.05  # 6D anchor orientation (mjlab motion_anchor_ori_b)


def imagined_goal(
    obs_mode: str,
    lib: MotionLibrary,
    frame: torch.Tensor,
    ref: ReferenceFrame,
    anchor_pose: torch.Tensor | None,
    future_frames: int,
    anchor_body: str,
    anchor_index: int,
    obs_noise: bool = False,
) -> torch.Tensor:
    """Goal observation from a (predicted) robot anchor pose (pos 3 | quat 4) and the reference, with the same
    content and noise as the simulator's goal group (envs.tracking_env_cfg._goal_terms)."""

    def ori_noise(x):
        return x + (torch.rand_like(x) * 2 - 1) * ANCHOR_ORI_NOISE_HALF if obs_noise else x

    if obs_mode == "beyondmimic_seq":
        jp, jv = lib.joint_window(frame, future_frames)
        ref_q = lib.anchor_quat_window(frame, future_frames, anchor_body)
        return beyondmimic_seq_goal(jp, jv, ori_noise(anchor_ori_obs(anchor_pose[..., 3:7], ref_q)))
    g = lib.goal_window(frame, future_frames)
    if obs_mode == "proprio_only":
        return g
    if obs_mode == "anchor_ori_now":
        return torch.cat([g, ori_noise(anchor_ori_obs(anchor_pose[..., 3:7], ref.body_quat_w[..., anchor_index, :]))], dim=-1)
    rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], ref.body_pos_w[..., anchor_index, :], ref.body_quat_w[..., anchor_index, :])
    if obs_noise:
        rel = rel + (torch.rand_like(rel) * 2 - 1) * ANCHOR_NOISE_HALF.to(rel.device)
    return torch.cat([g, rel], dim=-1)


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
    step_fn: ImagStep | None = None,
    obs_mode: str = "proprio_only",
) -> ImagRollout:
    device = start.state.device
    n = start.state.shape[0]
    q0 = lib.default_joint_pos
    step_fn = step_fn or ImagStep(wm, q0)
    noise_half = OBS_NOISE_HALF.to(device)
    ai = wm.heads.anchor_index

    def add_noise(o):
        if obs_noise:
            return o + (torch.rand_like(o) * 2 - 1) * noise_half
        return o

    def make_goal(frame_t, ref_t, anchor_pose):
        return imagined_goal(obs_mode, lib, frame_t, ref_t, anchor_pose, future_frames, wm.heads.anchor_body, ai, obs_noise)

    # Initial observation history from the logged states/actions (noise re-sampled per frame).
    hist = [add_noise(analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, q0)) for i in range(start.hist_states.shape[0])]
    s = start.state
    frame = start.frame.clone()
    a_prev = start.prev_action
    c = torch.ones(n, device=device)

    states = [s]
    actions, rewards = [], []
    dhats, survivals, frames = [torch.zeros(n, device=device)], [c], [frame]
    # All target frames of the rollout are known up front (frame + t, clamped at the clip end):
    # gather / FK the references for every step in one batched call.
    last = lib.last_frame_of(frame)
    frames_all = torch.minimum(frame[None, :] + torch.arange(horizon + 1, device=device)[:, None], last[None, :])  # (H+1, n)
    ref_all = reference_at(lib, frames_all.reshape(-1))

    def ref_at(t):
        sl = slice(t * n, (t + 1) * n)
        return ReferenceFrame(ref_all.body_pos_w[sl], ref_all.body_quat_w[sl], ref_all.body_lin_vel_w[sl], ref_all.body_ang_vel_w[sl], ref_all.root_pos_w[sl], ref_all.root_quat_w[sl])

    ref = ref_at(0)
    if obs_mode != "proprio_only":
        b0 = wm.heads.robot_bodies(s)
        anchor_pose = torch.cat([b0["pos"][..., ai, :], b0["quat"][..., ai, :]], dim=-1)
    else:
        anchor_pose = None
    goals = [make_goal(frame, ref, anchor_pose)]
    e_roots = [wm.heads.root_error_features(s, ref)]
    term_sum = torch.zeros(len(REWARD_TERM_NAMES), device=device)

    for t in range(horizon):
        obs_flat = flatten_history_term_major(torch.stack(hist, 0))
        a = policy(obs_flat, goals[-1])
        s_next, r, term_vec, logits, obs_clean, anchor_pose = step_fn(s, a, a_prev, start.xi, ref)
        d = torch.sigmoid(logits)
        c = c * (1.0 - d)
        term_sum += term_vec.detach().mean(-1)

        frame = frames_all[t + 1]
        ref = ref_at(t + 1)
        hist = hist[1:] + [add_noise(obs_clean)]
        a_prev = a
        s = s_next
        states.append(s)
        actions.append(a)
        rewards.append(r)
        dhats.append(d)
        survivals.append(c)
        goals.append(make_goal(frame, ref, anchor_pose))
        e_roots.append(wm.heads.root_error_features(s, ref))
        frames.append(frame)

    return ImagRollout(
        states=torch.stack(states),
        actions=torch.stack(actions),
        rewards=torch.stack(rewards),
        dhat=torch.stack(dhats),
        survival=torch.stack(survivals),
        goals=torch.stack(goals),
        e_root=torch.stack(e_roots),
        frames=torch.stack(frames),
        reward_terms={k: float(v / horizon) for k, v in zip(REWARD_TERM_NAMES, term_sum.tolist())},
    )
