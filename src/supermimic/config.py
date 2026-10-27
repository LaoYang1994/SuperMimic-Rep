"""All training configuration in one dataclass tree (tyro-friendly)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from supermimic.imagination.policy_trainer import PolicyTrainerCfg
from supermimic.policy.nets import CriticCfg, PolicyCfg
from supermimic.world_model.wm_trainer import WMTrainerCfg
from supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/lafan1")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None  # csv with name,keep (filter_bones_seed.py output)
    library: Path | None = None  # packed compact library (.pt from build_library.py); overrides npz_dir


ObsMode = Literal["beyondmimic_seq", "anchor_ori_now", "proprio_only", "anchor_pose9"]


def obs_mode_from_config(cfg: dict) -> ObsMode:
    """Observation setting of a saved run config (`config.json` / checkpoint "config"); pre-obs_mode runs
    stored a bool `anchor_obs` (True = 9-d anchor pose, False = reference-only goal)."""
    env = cfg["env"]
    if "obs_mode" in env:
        return env["obs_mode"]
    return "anchor_pose9" if env.get("anchor_obs", False) else "proprio_only"


@dataclass
class EnvCfg:
    num_envs: int = 4096
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    # Largest share of the return sampler's hard mass per clip; 0 = the paper's softmax alone (not in the paper).
    hard_clip_cap: float = 0.01
    replay_capacity: int = 1000
    future_frames: int = 10
    compact_library: bool = False
    # Goal observation setting (docs/CONCEPTS.md#观测口径). beyondmimic_seq: each future frame is BeyondMimic's
    # single-frame target (reference joint pos/vel + reference anchor orientation in the robot anchor frame);
    # anchor_ori_now: reference-only frames + the current 6D anchor orientation; proprio_only: reference-only
    # frames (paper Table A.1 read literally); anchor_pose9: upstream's 9-d anchor pose (reference only).
    obs_mode: ObsMode = "beyondmimic_seq"


@dataclass
class ScheduleCfg:
    # One iteration = one world-model update + its policy update(s) (paper Algorithm A.1: "Both ω and θ update
    # every iteration"); each iteration first collects 1/UTD simulator steps on average.
    total_iterations: int = 320_000
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_sim_steps: int = 40  # simulator steps collected before the first iteration (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000  # of total_iterations
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    checkpoint_at_hours: tuple[float, ...] = ()  # extra ckpt_h{X}.pt when training time first passes X hours
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "debug"
    log_root: Path = Path("logs")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
