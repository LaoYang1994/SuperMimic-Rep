from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def goal_future_frames(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """(N, K*68) future reference-frame features (plan D5)."""
    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    return command.goal


def _ori_noise(x: torch.Tensor, half_width: float) -> torch.Tensor:
    return x + (torch.rand_like(x) * 2 - 1) * half_width if half_width > 0 else x


def goal_beyondmimic_seq(env: ManagerBasedRlEnv, command_name: str, ori_noise: float = 0.0) -> torch.Tensor:
    """(N, K*64) obs_mode ``beyondmimic_seq``: per future frame, BeyondMimic's single-frame target
    (absolute reference joint pos/vel + reference anchor orientation in the current robot anchor frame).
    ``ori_noise`` is the U[-x, x] sensor noise of mjlab's motion_anchor_ori_b, applied to the 6D part only."""
    from supermimic.world_model.heads import anchor_ori_obs, beyondmimic_seq_goal

    command = cast(MultiClipMotionCommand, env.command_manager.get_term(command_name))
    lib, k = command.library, command.cfg.future_frames
    jp, jv = lib.joint_window(command.frame, k)
    ref_q = lib.anchor_quat_window(command.frame, k, command.cfg.anchor_body_name)
    ori = _ori_noise(anchor_ori_obs(command.robot_anchor_quat_w, ref_q), ori_noise)
    return beyondmimic_seq_goal(jp, jv, ori)
