"""Goal observations of imagined states, split so the imagined step can be compiled.

The goal of every observation setting has a part that depends only on the target frame (reference windows,
gathered for the whole rollout up front) and a part that depends on the predicted robot anchor pose (computed
inside the compiled step). Observation noise is passed in as pre-drawn uniform samples so the compiled graph
holds no random-number generation and the noise stream equals the eager one.
"""

from __future__ import annotations

import torch

from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.world_model.heads import ReferenceFrame, anchor_ori_obs, anchor_relative_obs, beyondmimic_seq_goal

# Paper Table A.1 sensor noise (uniform half-widths) in the 93-d observation order.
OBS_NOISE_HALF = torch.tensor([0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29)
ANCHOR_NOISE_HALF = torch.tensor([0.25] * 3 + [0.05] * 6)  # anchor_pose9: position 3 + orientation 6D
ANCHOR_ORI_NOISE_HALF = 0.05  # 6D anchor orientation (mjlab motion_anchor_ori_b)


def goal_static(obs_mode: str, lib: MotionLibrary, frame: torch.Tensor, ref: ReferenceFrame, future_frames: int, anchor_body: str, anchor_index: int) -> dict[str, torch.Tensor]:
    """Frame-only inputs of the goal of ``frame`` (N,): reference windows and the reference anchor pose."""
    if obs_mode == "beyondmimic_seq":
        jp, jv = lib.joint_window(frame, future_frames)
        return {"jp": jp, "jv": jv, "ref_q": lib.anchor_quat_window(frame, future_frames, anchor_body)}
    out = {"g": lib.goal_window(frame, future_frames)}
    if obs_mode in ("anchor_ori_now", "anchor_pose9"):
        out["ref_q"] = ref.body_quat_w[..., anchor_index, :]
    if obs_mode == "anchor_pose9":
        out["ref_pos"] = ref.body_pos_w[..., anchor_index, :]
    return out


def goal_noise_shape(obs_mode: str, n: int, future_frames: int) -> tuple[int, ...] | None:
    """Shape of the uniform noise sample the goal of ``obs_mode`` draws (None: noise-free goal)."""
    return {"beyondmimic_seq": (n, future_frames, 6), "anchor_ori_now": (n, 6), "anchor_pose9": (n, 9), "proprio_only": None}[obs_mode]


def goal_from_static(obs_mode: str, st: dict[str, torch.Tensor], anchor_pose: torch.Tensor | None, u: torch.Tensor | None, anchor_noise_half: torch.Tensor | None = None) -> torch.Tensor:
    """Goal from its frame-only inputs and the robot anchor pose (pos 3 | quat 4); ``u`` ~ U[0, 1) of
    ``goal_noise_shape`` adds the simulator's goal noise (None: no noise)."""
    if obs_mode == "proprio_only":
        return st["g"]
    if obs_mode == "anchor_pose9":
        rel = anchor_relative_obs(anchor_pose[..., 0:3], anchor_pose[..., 3:7], st["ref_pos"], st["ref_q"])
        if u is not None:
            rel = rel + (u * 2 - 1) * anchor_noise_half
        return torch.cat([st["g"], rel], dim=-1)
    ori = anchor_ori_obs(anchor_pose[..., 3:7], st["ref_q"])
    if u is not None:
        ori = ori + (u * 2 - 1) * ANCHOR_ORI_NOISE_HALF
    if obs_mode == "beyondmimic_seq":
        return beyondmimic_seq_goal(st["jp"], st["jv"], ori)
    return torch.cat([st["g"], ori], dim=-1)


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
    st = goal_static(obs_mode, lib, frame, ref, future_frames, anchor_body, anchor_index)
    shape = goal_noise_shape(obs_mode, frame.shape[0], future_frames)
    u = torch.rand(shape, device=frame.device) if obs_noise and shape is not None else None
    return goal_from_static(obs_mode, st, anchor_pose, u, ANCHOR_NOISE_HALF.to(frame.device))
