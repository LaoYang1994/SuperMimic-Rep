"""One imagined step as a single compiled function (plan 3.6: launch-overhead control).

``ImagStep`` bundles world-model dynamics, analytic FK reward, termination logits and the
next noise-free observation into one call so ``torch.compile`` can fuse the hundreds of
small kernels of the per-body FK loop. Reference quantities are passed as plain tensors
(already gathered from the motion library) so the compiled graph has static shapes.
"""

from __future__ import annotations

import torch

from supermimic.imagination.goals import ANCHOR_NOISE_HALF, OBS_NOISE_HALF, goal_from_static
from supermimic.world_model.heads import ReferenceFrame, analytic_observation, flatten_history_term_major
from supermimic.world_model.world_model import WorldModel

REWARD_TERM_NAMES = (
    "motion_global_root_pos",
    "motion_global_root_ori",
    "motion_body_pos",
    "motion_body_ori",
    "motion_body_lin_vel",
    "motion_body_ang_vel",
    "action_rate_l2",
    "joint_limit",
)


class ImagStep:
    def __init__(self, wm: WorldModel, default_joint_pos: torch.Tensor, compile_mode: str | None = None):
        self.wm = wm
        self.q0 = default_joint_pos
        self._fn = self._step
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def _step(self, s, a, a_prev, xi, ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat):
        ref = ReferenceFrame(ref_body_pos, ref_body_quat, ref_body_lin, ref_body_ang, ref_root_pos, ref_root_quat)
        s_next = self.wm.step(s, a, xi)
        bodies = self.wm.heads.robot_bodies(s_next)
        terms = self.wm.heads.reward(s_next, a, a_prev, ref, bodies)
        r = terms["total"]
        term_vec = torch.stack([terms[k] for k in REWARD_TERM_NAMES], dim=0)  # (8, n)
        logits = self.wm.termination_logits(s_next, ref)
        obs_clean = analytic_observation(s_next, a, xi, self.q0)
        ai = self.wm.heads.anchor_index
        anchor_pose = torch.cat([bodies["pos"][..., ai, :], bodies["quat"][..., ai, :]], dim=-1)  # (n, 7)
        return s_next, r, term_vec, logits, obs_clean, anchor_pose

    def __call__(self, s, a, a_prev, xi, ref: ReferenceFrame):
        return self._fn(s, a, a_prev, xi, ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w)


def _ref_tensors(ref: ReferenceFrame) -> tuple[torch.Tensor, ...]:
    return ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w, ref.root_pos_w, ref.root_quat_w


class ImagRolloutStep:
    """A whole imagined step as one compiled call: observation history -> policy -> world-model step, reward,
    termination and survival -> next (noisy) observation history, next goal and critic root-error features.

    Only the world-model part used to be compiled; the policy MLP, goal and history handling ran eagerly as
    ~1,300 kernel launches per step (forward + backward). Frame-only goal inputs and the uniform noise samples
    are prepared outside (``imagine``), so the graph has static shapes and no random-number generation.
    """

    def __init__(self, wm: WorldModel, policy: torch.nn.Module, default_joint_pos: torch.Tensor, obs_mode: str, compile_mode: str | None = None):
        self.wm, self.policy, self.obs_mode = wm, policy, obs_mode
        self.core = ImagStep(wm, default_joint_pos)  # eager body; compiled as part of this step
        dev = default_joint_pos.device
        self.noise_half = OBS_NOISE_HALF.to(dev)
        self.anchor_noise_half = ANCHOR_NOISE_HALF.to(dev)
        self._fn = self._step
        self.cudagraphs = compile_mode == "reduce-overhead"
        if compile_mode:
            mode = None if compile_mode == "default" else compile_mode
            self._fn = torch.compile(self._step, mode=mode, dynamic=False)

    def begin_rollout(self) -> None:
        """Start of a rollout: CUDA-graph outputs of the previous rollout may be overwritten from here on."""
        if self.cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

    def _step(self, s, a_prev, hist, goal, c, xi, ref_t, u_obs, goal_next_static, ref_next, u_goal):
        a = self.policy(flatten_history_term_major(hist), goal)
        s_next, r, term_vec, logits, obs_clean, anchor_pose = self.core._step(s, a, a_prev, xi, *ref_t)
        d = torch.sigmoid(logits)
        c_next = c * (1.0 - d)
        o = obs_clean if u_obs is None else obs_clean + (u_obs * 2 - 1) * self.noise_half
        hist_next = torch.cat([hist[1:], o[None]], dim=0)
        goal_next = goal_from_static(self.obs_mode, goal_next_static, anchor_pose, u_goal, self.anchor_noise_half)
        ref_n = ReferenceFrame(*ref_next)
        e_root_next = self.wm.heads.root_error_features(s_next, ref_n)
        return a, s_next, r, term_vec.detach().mean(-1), d, c_next, hist_next, goal_next, e_root_next

    def __call__(self, s, a_prev, hist, goal, c, xi, ref_t: ReferenceFrame, u_obs, goal_next_static, ref_next: ReferenceFrame, u_goal):
        return self._fn(s, a_prev, hist, goal, c, xi, _ref_tensors(ref_t), u_obs, goal_next_static, _ref_tensors(ref_next), u_goal)
