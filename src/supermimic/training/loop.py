"""SuperMimic training loop (paper Algorithm A.1): collect -> fit world model -> improve policy."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from supermimic.config import TrainConfig
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.mdp.sampler_math import bin_return_from_checkpoint
from supermimic.envs.tracking_env_cfg import ANCHOR_BODY, ROOT_BODY, TRACKED_BODIES
from supermimic.imagination.policy_trainer import PolicyTrainer
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.policy.nets import Critic, Policy, goal_layout
from supermimic.training.distributed import allreduce_mean, broadcast_params, finalize, init_distributed, world_size
from supermimic.training.optim import load_optimizer_state
from supermimic.training.schedule import cosine_lr, sim_steps_this_iter
from supermimic.world_model.heads import AnalyticHeads
from supermimic.world_model.wm_trainer import WorldModelTrainer
from supermimic.world_model.world_model import WorldModel



class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        cfg.policy.obs_mode = cfg.env.obs_mode
        cfg.policy.future_frames = cfg.env.future_frames
        cfg.policy.goal_dim = goal_layout(cfg.env.obs_mode, cfg.env.future_frames, cfg.policy.goal_term_scale)[0]
        cfg.policy_trainer.obs_mode = cfg.env.obs_mode
        self.rank, local_rank, self.world = init_distributed()
        self.is_main = self.rank == 0
        if self.world > 1:
            cfg.device = f"cuda:{local_rank}"
            # Paper: lr x sqrt(2) for 2 GPUs (each GPU keeps its own 4096 envs and buffer).
            cfg.wm_trainer.lr *= self.world**0.5
            cfg.policy_trainer.lr *= self.world**0.5
            cfg.policy_trainer.critic_lr *= self.world**0.5
        torch.manual_seed(cfg.seed + 1000 * self.rank)
        self.device = cfg.device
        self.log_dir = Path(cfg.log_root) / cfg.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.is_main:
            (self.log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        self.writer = SummaryWriter(str(self.log_dir / "tb")) if self.is_main else None

        if cfg.data.library is not None:
            files = [cfg.data.library]
            cfg.env.compact_library = True
        else:
            files = discover_npz(cfg.data.npz_dir, limit=cfg.data.limit, names=cfg.data.clips, manifest=cfg.data.manifest)
        if not files:
            raise FileNotFoundError(f"no clips under {cfg.data.npz_dir}")
        self.env = make_env(
            SuperMimicEnvCfgArgs(
                npz_files=[str(f) for f in files],
                num_envs=cfg.env.num_envs,
                future_frames=cfg.env.future_frames,
                episode_length_s=cfg.env.episode_length_s,
                action_noise_std=cfg.env.action_noise_std,
                obs_noise=cfg.env.obs_noise,
                reset_noise=cfg.env.reset_noise,
                domain_randomization=cfg.env.domain_randomization,
                sampling_mode=cfg.env.sampling_mode,
                replay_capacity=cfg.env.replay_capacity,
                compact_library=cfg.env.compact_library,
                obs_mode=cfg.env.obs_mode,
                seed=cfg.seed + 1000 * self.rank,
            ),
            device=self.device,
        )
        env = self.env
        self.command = env.command_manager.get_term("motion")
        self.lib = self.command.library
        self.recorder = env.recorder_manager.get_term("replay")
        self.buffer = self.recorder.buffer
        robot = env.scene["robot"]

        fk = G1Kinematics(load_g1_model(), device=self.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), robot.data.default_joint_pos[0].clone())
        self.wm = WorldModel(cfg.world_model, heads, self.device)
        self.policy = Policy(cfg.policy).to(self.device)
        self.critic = Critic(cfg.critic, pcfg=cfg.policy).to(self.device)
        for m in (self.wm, self.policy, self.critic):
            broadcast_params(m)
        self.wm_trainer = WorldModelTrainer(self.wm, self.lib, cfg.wm_trainer)
        self.policy_trainer = PolicyTrainer(self.wm, self.policy, self.critic, self.lib, cfg.policy_trainer)
        n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        if self.is_main:
            print(f"[INFO] params: world model {n_params(self.wm)/1e6:.2f}M, policy {n_params(self.policy)/1e6:.2f}M, critic {n_params(self.critic)/1e6:.3f}M")

        # Episode statistics of the training env (paper's "training return").
        self.ep_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.ep_len = torch.zeros(env.num_envs, device=self.device)
        self.ep_max_len = torch.ones(env.num_envs, device=self.device)  # steps the episode could last at most
        self.max_episode_steps = int(round(cfg.env.episode_length_s / env.step_dt))
        self.max_step_reward = heads.reward_cfg.max_step_reward
        self.iteration = 0  # = number of updates (paper Algorithm A.1)
        self.num_updates = 0
        self.sim_steps = 0  # vectorised simulator steps collected
        self._utd_acc = 0.0
        if cfg.resume is not None:
            self.load(cfg.resume)

    # ------------------------------------------------------------------ checkpoints

    def save(self, name: str = "ckpt_latest.pt") -> None:
        if not self.is_main:
            return
        torch.save(
            {
                "iteration": self.iteration,
                "num_updates": self.num_updates,
                "sim_steps": self.sim_steps,
                "utd_acc": self._utd_acc,
                "iteration_semantics": "update",
                "wm": self.wm.state_dict(),
                "policy": self.policy.state_dict(),
                "critic": self.critic.state_dict(),
                "wm_opt": self.wm_trainer.opt.state_dict(),
                "policy_opt": self.policy_trainer.opt.state_dict(),
                "critic_opt": self.policy_trainer.critic_opt.state_dict(),
                "bin_return": self.command.bin_return,
                "bin_memory": self.command.bin_memory,
                "bin_return_normalized": True,
                "config": asdict(self.cfg),
            },
            self.log_dir / name,
        )

    def load(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.wm.load_state_dict(ck["wm"])
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        load_optimizer_state(self.wm_trainer.opt, ck["wm_opt"])
        load_optimizer_state(self.policy_trainer.opt, ck["policy_opt"])
        load_optimizer_state(self.policy_trainer.critic_opt, ck["critic_opt"])
        self.command.bin_return.copy_(bin_return_from_checkpoint(ck, self.command.cfg.g_max))
        self.command.bin_memory.copy_(ck["bin_memory"])
        self.iteration = ck["iteration"]
        self.num_updates = ck["num_updates"]
        if ck.get("iteration_semantics") == "update":
            self.sim_steps = ck["sim_steps"]
            self._utd_acc = ck["utd_acc"]
        else:
            # Pre-change checkpoint: its "iteration" counted simulator steps, so the update count is num_updates.
            print("[WARN] checkpoint predates iteration = update; resuming at iteration = num_updates")
            self.sim_steps = self.iteration
            self.iteration = self.num_updates
        print(f"[INFO] resumed from {path} at iteration {self.iteration} ({self.sim_steps} sim steps)")

    # ------------------------------------------------------------------ main loop

    @torch.no_grad()
    def act(self, obs) -> torch.Tensor:
        return self.policy(obs["policy"], obs["goal"])

    def _reset_episode_bounds(self, ids: torch.Tensor) -> None:
        """Longest each (re)started episode can run: until its clip ends or the episode time limit."""
        frame = self.command.frame[ids]
        remaining = (self.lib.last_frame_of(frame) - frame + 1).float()  # a full playback takes frames steps
        self.ep_max_len[ids] = remaining.clamp_max(self.max_episode_steps)

    def _collect(self, obs, stats: dict):
        """One vectorised simulator step; returns the next observation."""
        action = self.act(obs)
        obs, rew, term, tout, extras = self.env.step(action)
        self.ep_reward_sum += rew
        self.ep_len += 1
        done = term | tout
        if done.any():
            ids = done.nonzero().squeeze(-1)
            lens = self.ep_len[ids]
            # Paper §4: normalize by the largest attainable sum, so early termination scores low.
            stats["returns"].extend((self.ep_reward_sum[ids] / (self.max_step_reward * self.ep_max_len[ids])).tolist())
            stats["returns_per_step"].extend((self.ep_reward_sum[ids] / (self.max_step_reward * lens.clamp_min(1))).tolist())
            stats["lens"].extend(lens.tolist())
            self.ep_reward_sum[ids] = 0.0
            self.ep_len[ids] = 0.0
            self._reset_episode_bounds(ids)  # auto-reset already moved these envs to their new start frame
        self.sim_steps += 1
        return obs

    def run(self) -> None:
        cfg = self.cfg
        sch = cfg.schedule
        env = self.env
        obs, _ = env.reset()
        self._reset_episode_bounds(torch.arange(env.num_envs, device=self.device))
        t_start = time.time()
        timers = {"sim": 0.0, "wm": 0.0, "policy": 0.0}
        agg: dict[str, float] = {}
        agg_n = 0
        stats: dict[str, list[float]] = {"returns": [], "returns_per_step": [], "lens": []}
        hours_pending = sorted(h for h in sch.checkpoint_at_hours if h * 3600 > 0)
        last_log = time.time()

        t0 = time.perf_counter()
        # Counted per run: the replay buffer is not checkpointed, so a resumed run must refill it too.
        for _ in range(sch.warmup_sim_steps):
            obs = self._collect(obs, stats)
        torch.cuda.synchronize()
        timers["sim"] += time.perf_counter() - t0

        while self.iteration < sch.total_iterations:
            # ---------------- (1) collect 1/UTD simulator steps on average
            t0 = time.perf_counter()
            n_steps, self._utd_acc = sim_steps_this_iter(self._utd_acc, sch.updates_per_env_step)
            for _ in range(n_steps):
                obs = self._collect(obs, stats)
            torch.cuda.synchronize()
            timers["sim"] += time.perf_counter() - t0

            # ---------------- (2) fit the world model, (3) improve the policy: one iteration = one update
            lr = cosine_lr(self.iteration, sch.total_iterations, cfg.wm_trainer.lr, sch.lr_final_ratio, sch.lr_decay_fraction)
            self.wm_trainer.set_lr(lr)
            self.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
            t0 = time.perf_counter()
            m_wm = self.wm_trainer.update(self.buffer)
            torch.cuda.synchronize()
            timers["wm"] += time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(sch.policy_updates_per_wm_update):
                m_pi = self.policy_trainer.update(self.buffer)
            torch.cuda.synchronize()
            timers["policy"] += time.perf_counter() - t0
            self.num_updates += 1
            self.iteration += 1
            for k, v in m_wm.items():
                agg[f"wm/{k}"] = agg.get(f"wm/{k}", 0.0) + v
            for k, v in m_pi.items():
                agg[f"policy/{k}"] = agg.get(f"policy/{k}", 0.0) + v
            agg["lr"] = lr
            agg_n += 1

            # ---------------- logging / checkpoints
            if self.iteration % sch.log_interval == 0:
                it = self.iteration
                elapsed = time.time() - t_start
                finished = stats["returns"]
                if self.world > 1:
                    # Every rank must issue the same collectives: reduce (sum, count) unconditionally.
                    tot = allreduce_mean(float(sum(finished))) * self.world
                    tot_ps = allreduce_mean(float(sum(stats["returns_per_step"]))) * self.world
                    cnt = allreduce_mean(float(len(finished))) * self.world
                    stats["returns"] = [tot / cnt] if cnt > 0 else []
                    stats["returns_per_step"] = [tot_ps / cnt] if cnt > 0 else []
                    n_finished = int(round(cnt))
                else:
                    n_finished = len(finished)
                if not self.is_main:
                    agg, agg_n = {}, 0
                    timers = {k: 0.0 for k in timers}
                elif agg_n > 0:
                    for k, v in agg.items():
                        self.writer.add_scalar(k, v / (agg_n if k != "lr" else 1), it)
                if self.is_main and stats["returns"]:
                    tr = sum(stats["returns"]) / len(stats["returns"])
                    self.writer.add_scalar("train/episode_norm_return", tr, it)
                    self.writer.add_scalar("train/episode_norm_return_per_step", sum(stats["returns_per_step"]) / len(stats["returns_per_step"]), it)
                    if stats["lens"]:
                        self.writer.add_scalar("train/episode_len", sum(stats["lens"]) / len(stats["lens"]), it)
                else:
                    tr = float("nan")
                with torch.no_grad():
                    if self.is_main and self.buffer.size > 2048:
                        err = self.wm_trainer.online_error(self.buffer, n=1024, k=8)
                        self.writer.add_scalar("wm/online_nse_1step", float(err[0]), it)
                        self.writer.add_scalar("wm/online_nse_8step", float(err[-1]), it)
                    else:
                        err = None
                if self.is_main:
                    for k, v in timers.items():
                        self.writer.add_scalar(f"time/{k}_s_per_iter", v / sch.log_interval, it)
                    self.writer.add_scalar("time/hours", elapsed / 3600, it)
                    self.writer.add_scalar("time/updates", self.num_updates, it)
                    self.writer.add_scalar("time/sim_steps", self.sim_steps, it)
                    self.writer.add_scalar("sampler/entropy", float(self.command.metrics["sampling_entropy"][0]), it)
                    self.writer.add_scalar("sampler/bin_return_mean", float(self.command.bin_return.mean()), it)
                    self.writer.add_scalar("sampler/hard_mass_clip_end_bins", float(self.command.metrics["hard_mass_clip_end"][0]), it)
                    dt_log = time.time() - last_log
                    last_log = time.time()
                    msg = (
                        f"it {it} | sim steps {self.sim_steps} | {elapsed/3600:.2f} h | {sch.log_interval / dt_log:.1f} it/s | "
                        f"train_ret {tr:.3f} (n={n_finished}) | "
                        f"wm nse1 {agg.get('wm/nse_1step', float('nan'))/max(agg_n,1):.4f} nse8 {(float(err[-1]) if err is not None else float('nan')):.4f} | "
                        f"imag_ret {agg.get('policy/imag_return_undisc', float('nan'))/max(agg_n,1):.3f} surv {agg.get('policy/imag_survival_H', float('nan'))/max(agg_n,1):.3f} | "
                        f"t sim/wm/pi {timers['sim']:.1f}/{timers['wm']:.1f}/{timers['policy']:.1f}s"
                    )
                    print(msg, flush=True)
                agg, agg_n = {}, 0
                stats = {k: [] for k in stats}
                timers = {k: 0.0 for k in timers}
            if self.iteration % sch.checkpoint_interval == 0:
                self.save("ckpt_latest.pt")
                self.save(f"ckpt_{self.iteration:07d}.pt")
            if hours_pending and (time.time() - t_start) / 3600 >= hours_pending[0]:
                h = hours_pending.pop(0)
                self.save(f"ckpt_h{h:g}.pt")
                if self.is_main:
                    print(f"[INFO] {h:g} h checkpoint at iteration {self.iteration}", flush=True)
            if sch.max_hours is not None and self.iteration % sch.log_interval == 0:
                stop = float((time.time() - t_start) / 3600 > sch.max_hours)
                if self.world > 1:
                    stop = allreduce_mean(stop)  # rank 0's clock decides (all ranks agree on the mean > 0)
                if stop > 0:
                    if self.is_main:
                        print("[INFO] max_hours reached, stopping")
                    break
        self.save("ckpt_latest.pt")
        if self.writer is not None:
            self.writer.flush()
        finalize()
