"""
DPPO Recovery-Aware Fine-Tuning Pipeline (Algorithm 1 from ECE285_MidTerm.pdf).

Executes on-policy rollouts with online stratified physical disturbances,
optimizes diffusion policies over reverse denoising chains with clipped PPO loss,
anchors updates to the pretrained nominal policy using analytical closed-form KL penalty,
and benchmarks recovery generalization on training and held-out disturbance distributions.
"""

from __future__ import annotations

import copy
import json
import random
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

from src.dppo.critic import ValueCritic
from src.dppo.sampler import DDPMSampler
from src.dppo.disturbances import DisturbanceManager, DisturbanceConfig, PHASE_APPROACHING, PHASE_PRE_GRASP, PHASE_GRASPED
from src.dppo.losses import compute_actor_loss, compute_critic_loss, compute_kl_regularization
from src.dppo.buffer import DPPORolloutBuffer

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_rng_state() -> Dict[str, Any]:
    state = {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


class DPPOTrainer:
    """
    Trainer executing Recovery-Aware DPPO Fine-Tuning.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.config = config or {}
        self.checkpoint_path = checkpoint_path

        # Device setup
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = TorchUtils.get_torch_device(try_to_use_cuda=True)

        logger.info(f"Initializing DPPO Trainer on device: {self.device}")

        # Global seeding (robosuite's placement sampler uses the global np.random)
        seed = self.config.get("experiment", {}).get("seed", None)
        if seed is not None:
            _seed_everything(int(seed))
            logger.info(f"Seeded RNGs with experiment.seed={seed}")

        # Load pretrained model and environment
        self._load_pretrained_model()
        self._setup_critic()
        self._setup_sampler()
        self._setup_disturbances()
        self._setup_parameter_partition_and_optimizers()

        # Buffer
        rl_cfg = self.config.get("rl", {})
        self.gamma = float(rl_cfg.get("gamma", 0.99))
        self.gae_lambda = float(rl_cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(rl_cfg.get("clip_eps", 0.2))
        self.beta = float(rl_cfg.get("kl_weight_beta", 0.01))
        self.cv = float(rl_cfg.get("critic_weight_cv", 0.5))
        self.epochs = int(rl_cfg.get("epochs", 5))
        self.batch_size = int(rl_cfg.get("batch_size", 128))
        tk = rl_cfg.get("target_kl", None)
        self.target_kl = float(tk) if tk is not None else None

        self.buffer = DPPORolloutBuffer(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            device=self.device,
        )

        # Output / checkpoint dir
        weights_cfg = self.config.get("weights", {})
        self.save_dir = Path(weights_cfg.get("save_dir", "results/dppo_checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_pretrained_model(self) -> None:
        """Load pretrained diffusion policy and create frozen reference anchor θ_0."""
        # PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects the
        # numpy objects robomimic pickles into checkpoints; load the dict ourselves.
        ckpt_dict = torch.load(
            os.path.expanduser(str(self.checkpoint_path)), map_location="cpu", weights_only=False
        )
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_dict=ckpt_dict,
            device=self.device,
            verbose=False,
        )
        self.algo = self.policy.policy
        self.algo.set_eval()

        # Build N independent training environments. Rollouts batch the denoising
        # chain across envs (one UNet call per denoising step for all envs), which
        # is the dominant cost: 100 steps take ~0.9 s at batch 1 and ~1.2 s at batch 16.
        self.num_envs = max(1, int(self.config.get("rollout", {}).get("num_envs", 1)))
        self.envs = []
        for _ in range(self.num_envs):
            env, _ = FileUtils.env_from_checkpoint(
                ckpt_dict=self.ckpt_dict,
                render=False,
                verbose=False,
            )
            self.envs.append(env)
        self.env = self.envs[0]
        logger.info(f"Created {self.num_envs} rollout environment(s)")

        # Pick θ_0. robomimic runs inference (and computed the checkpoint's success
        # rate) with the EMA weights, which differ from the raw `nets` by ~7% for the
        # epoch-150 checkpoint. Fine-tune and anchor to the policy that was evaluated.
        use_ema = bool(self.config.get("weights", {}).get("use_ema", True))
        if use_ema and getattr(self.algo, "ema", None) is not None:
            source_nets = self.algo.ema.averaged_model
            logger.info("Using EMA weights of the pretrained checkpoint as θ_0")
        else:
            if use_ema:
                logger.warning("weights.use_ema=true but checkpoint has no EMA weights; using raw nets")
            source_nets = self.algo.nets
            logger.info("Using raw (non-EMA) weights of the pretrained checkpoint as θ_0")

        # Clone and freeze reference policy θ_ref = θ_0
        self.ref_noise_pred_net = copy.deepcopy(source_nets["policy"]["noise_pred_net"]).to(self.device)
        self.ref_noise_pred_net.eval()
        for p in self.ref_noise_pred_net.parameters():
            p.requires_grad = False

        self.noise_pred_net = source_nets["policy"]["noise_pred_net"].to(self.device)
        # diffusers' EMAModel sets requires_grad=False on the averaged copy; re-enable it
        # on the trainable net (θ_time is re-frozen in the parameter partition step).
        self.noise_pred_net.requires_grad_(True)
        self.obs_encoder = source_nets["policy"]["obs_encoder"].to(self.device)
        self.noise_scheduler = self.algo.noise_scheduler

    def _setup_critic(self) -> None:
        """Instantiate MLP value function V_ϕ(s)."""
        input_dim = 38  # To * Dobs = 2 * 19
        self.critic = ValueCritic(input_dim=input_dim).to(self.device)

    def _setup_sampler(self) -> None:
        """Configure DDPM reverse stochastic step sampler."""
        rollout_cfg = self.config.get("rollout", {})
        k_ft_mode = rollout_cfg.get("k_ft_mode", "all")
        k_ft_stride = int(rollout_cfg.get("k_ft_stride", 1))

        # t=0 is the deterministic final step (a^0 = μ) and is never fine-tuned.
        all_timesteps = list(range(1, self.algo.algo_config.ddpm.num_train_timesteps))
        if k_ft_mode == "strided":
            self.k_ft_set = set(all_timesteps[::k_ft_stride])
        elif k_ft_mode == "tail":
            tail_m = int(rollout_cfg.get("k_ft_tail_m", 10))
            self.k_ft_set = set(all_timesteps[:tail_m])
        elif k_ft_mode == "all":
            self.k_ft_set = set(all_timesteps)
        else:
            raise ValueError(f"Unknown rollout.k_ft_mode '{k_ft_mode}' (expected all/strided/tail)")
        logger.info(f"Fine-tuned denoising steps K_ft ({k_ft_mode}): {sorted(self.k_ft_set)}")

        self.min_sampling_std = float(rollout_cfg.get("min_sampling_std", 0.1))
        self.min_logprob_std = float(rollout_cfg.get("min_logprob_std", 0.1))
        if self.min_logprob_std < self.min_sampling_std:
            raise ValueError("rollout.min_logprob_std must be >= rollout.min_sampling_std")

        self.sampler = DDPMSampler(
            noise_scheduler=self.noise_scheduler,
            obs_encoder=self.obs_encoder,
            noise_pred_net=self.noise_pred_net,
            obs_shapes=self.algo.obs_shapes,
            device=self.device,
            num_timesteps=self.algo.algo_config.ddpm.num_train_timesteps,
            prediction_horizon=self.algo.algo_config.horizon.prediction_horizon,
            action_horizon=self.algo.algo_config.horizon.action_horizon,
            observation_horizon=self.algo.algo_config.horizon.observation_horizon,
            action_dim=self.algo.ac_dim,
            min_sampling_std=self.min_sampling_std,
            min_logprob_std=self.min_logprob_std,
        )

    def _setup_disturbances(self) -> None:
        """Initialize disturbance configuration and manager."""
        dist_cfg = self.config.get("disturbances", {})
        rollout_cfg = self.config.get("rollout", {})

        cfg = DisturbanceConfig.from_config(rollout_cfg=rollout_cfg, dist_cfg=dist_cfg)
        for name in ("train_approach", "train_pre_grasp", "ho_approach", "ho_pre_grasp"):
            logger.info(f"Disturbance spec {name}: {getattr(cfg, name)}")
        self.dist_manager = DisturbanceManager(cfg)

    def _setup_parameter_partition_and_optimizers(self) -> None:
        """
        Partition parameters θ = (θ_time, θ_cond, θ_rest) per Section 2.5:
        - θ_time: FROZEN throughout training.
        - θ_cond: Trainable with learning rate decay schedule (η_0 -> η_min).
        - θ_rest: Trainable with standard learning rate.
        """
        rl_cfg = self.config.get("rl", {})
        self.actor_lr = float(rl_cfg.get("actor_lr", 1e-4))
        self.actor_lr_min = float(rl_cfg.get("actor_lr_min", 1e-5))
        self.critic_lr = float(rl_cfg.get("critic_lr", 3e-4))
        self.n1_decay_iters = int(rl_cfg.get("n1_decay_iters", 100))

        # 1. Freeze θ_time
        if hasattr(self.noise_pred_net, "diffusion_step_encoder"):
            for p in self.noise_pred_net.diffusion_step_encoder.parameters():
                p.requires_grad = False
            logger.info("Frozen θ_time (diffusion_step_encoder).")

        # 2. Collect θ_cond parameters
        cond_params: List[nn.Parameter] = []
        cond_param_ids = set()
        for name, module in self.noise_pred_net.named_modules():
            if "cond_encoder" in name:
                for p in module.parameters():
                    if p.requires_grad and id(p) not in cond_param_ids:
                        cond_params.append(p)
                        cond_param_ids.add(id(p))

        # 3. Collect θ_rest parameters
        rest_params: List[nn.Parameter] = []
        for p in self.noise_pred_net.parameters():
            if p.requires_grad and id(p) not in cond_param_ids:
                rest_params.append(p)

        logger.info(f"Parameter partition: θ_cond has {len(cond_params)} tensors, θ_rest has {len(rest_params)} tensors.")

        # Optimizers
        param_groups = [
            {"params": cond_params, "lr": self.actor_lr, "name": "cond"},
            {"params": rest_params, "lr": self.actor_lr, "name": "rest"},
        ]
        self.actor_optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-6)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def _decay_actor_lr(self, iteration: int) -> None:
        """Decay conditioning learning rate from η_0 to η_min over first N_1 iterations."""
        if iteration <= self.n1_decay_iters:
            frac = float(iteration) / float(self.n1_decay_iters)
            decayed_lr = self.actor_lr - frac * (self.actor_lr - self.actor_lr_min)
            for group in self.actor_optimizer.param_groups:
                if group.get("name") == "cond":
                    group["lr"] = decayed_lr

    def _get_current_stacked_obs(self, env: Optional[Any] = None) -> Dict[str, Any]:
        """
        Return the current observation in the same format env.step() returns.

        The env is wrapped in a FrameStackWrapper (frame_stack=To), whose
        `get_observation()` falls through to the inner env and returns a single
        unstacked frame. inject_disturbance() already rewrote obs_history with
        the post-disturbance frame, so rebuild the stacked obs from it.
        """
        env = self.env if env is None else env
        if getattr(env, "obs_history", None) is not None:
            return env._get_stacked_obs_from_history()
        return env.get_observation()

    @staticmethod
    def _seeded_reset(env: Any, seed: int) -> Dict[str, Any]:
        """env.reset() with robosuite's placement sampler (global np.random) seeded, without
        disturbing the caller's global RNG stream."""
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            return env.reset()
        finally:
            np.random.set_state(state)

    def load_finetuned_weights(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a DPPO checkpoint produced by save_checkpoint() into the trainable
        policy (and critic). The frozen KL anchor θ_0 is left untouched.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.noise_pred_net.load_state_dict(ckpt["policy_state_dict"])
        if "critic_state_dict" in ckpt:
            self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.noise_pred_net.eval()
        logger.info(f"Loaded DPPO fine-tuned weights from {path} (iteration {ckpt.get('iteration')})")
        return ckpt

    def collect_rollouts(
        self,
        n_episodes: int,
        mode: str = "train",
        min_sampling_std: Optional[float] = None,
        episode_plan: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        """
        Collect on-policy rollout episodes using stratified disturbance sampling (Algorithm 1 lines 4-8).

        Episodes from the plan are spread over self.num_envs environments. Each
        decision round, the observations of all running envs are denoised in one
        batched reverse chain; then every env executes its own Ta-step chunk. An env
        that finishes its episode immediately starts the next unassigned one.

        min_sampling_std: None uses the training exploration floor; 0.0 runs the
        true DDPM chain (used for evaluation).
        episode_plan: explicit plan (overrides n_episodes/mode stratification).
        """
        self.buffer.clear()
        if episode_plan is None:
            episode_plan = self.dist_manager.plan_iteration_episodes(n_episodes=n_episodes, mode=mode)

        max_horizon = int(self.config.get("rollout", {}).get("horizon", 400))
        Ta = self.algo.algo_config.horizon.action_horizon
        self.policy.start_episode()

        # One reset seed per planned episode, drawn up front from the global RNG.
        # With several envs, the order in which episodes start depends on which env
        # finishes first; seeding each reset ties episode j's initial object pose to
        # j (not to timing), so fixed-seed evals stay paired across checkpoints.
        reset_seeds = np.random.randint(0, 2**31 - 1, size=len(episode_plan))

        # Per-env slot state
        managers = [DisturbanceManager(self.dist_manager.cfg) for _ in self.envs]
        slots: List[Optional[Dict[str, Any]]] = [None] * len(self.envs)
        finished: List[Dict[str, Any]] = []  # (plan index, success, grasp step)
        next_ep = 0

        def start_slot(i: int) -> None:
            nonlocal next_ep
            if next_ep >= len(episode_plan):
                slots[i] = None
                return
            spec = episode_plan[next_ep]
            managers[i].reset_episode(spec)
            phase_type = spec["phase"] if spec is not None else "nominal"
            slots[i] = {
                "plan_idx": next_ep,
                "phase_type": phase_type,
                "obs": self._seeded_reset(self.envs[i], int(reset_seeds[next_ep])),
                "sim_step": 0,
                "success": False,
                "grasp_step": None,
                "episode": self.buffer.new_episode(is_disturbed=spec is not None, phase=phase_type),
            }
            next_ep += 1

        for i in range(len(self.envs)):
            start_slot(i)

        while any(sl is not None for sl in slots):
            active = [i for i, sl in enumerate(slots) if sl is not None]

            # Batched observation encoding, value estimate and reverse chain
            ob_tensors = [self.policy._prepare_observation(slots[i]["obs"]) for i in active]
            ob_batch = {k: torch.cat([ob[k] for ob in ob_tensors], dim=0) for k in ob_tensors[0]}
            with torch.no_grad():
                obs_cond = self.sampler.encode_observation(ob_batch)
                values = self.critic(obs_cond).view(-1).cpu().numpy()
                executed_actions, _, transitions = self.sampler.sample_action_chunk_with_trajectory(
                    obs_dict=ob_batch,
                    noise_pred_net=self.noise_pred_net,
                    k_ft_set=self.k_ft_set,
                    min_sampling_std=min_sampling_std,
                )
            chunk_actions = executed_actions.detach().cpu().numpy()  # (B, Ta, Da)

            for b, i in enumerate(active):
                sl, env, mgr = slots[i], self.envs[i], managers[i]
                chunk_reward, chunk_done = 0.0, False
                obs = sl["obs"]

                for sub_i in range(Ta):
                    act = chunk_actions[b, sub_i]
                    next_obs, r, d, _ = env.step(act)
                    sl["sim_step"] += 1

                    mgr.detect_phase(act, r, step_i=sl["sim_step"])
                    if mgr.should_perturb():
                        record = mgr.inject_disturbance(env, action=act)
                        if record is not None:
                            next_obs = self._get_current_stacked_obs(env)

                    is_succ = False
                    if hasattr(env, "is_success"):
                        res = env.is_success()
                        is_succ = bool(res.get("task", False)) if isinstance(res, dict) else bool(res)
                    elif r >= 1.0:
                        is_succ = True

                    obs = next_obs
                    if is_succ:
                        chunk_reward, chunk_done = 1.0, True
                        sl["success"], sl["grasp_step"] = True, sl["sim_step"]
                        break
                    if d:
                        chunk_done = True
                        break
                    if sl["sim_step"] >= max_horizon:
                        break

                sl["obs"] = obs
                self.buffer.add_step_to(
                    sl["episode"],
                    obs_cond=obs_cond[b],
                    reward=chunk_reward,
                    value=float(values[b]),
                    done=chunk_done,
                    denoising_transitions=[
                        {
                            "timestep": tr["timestep"],
                            "sample": tr["sample"][b:b + 1],
                            "prev_sample": tr["prev_sample"][b:b + 1],
                            "mu": tr["mu"][b:b + 1],
                            "sigma": tr["sigma"][b:b + 1],
                        }
                        for tr in transitions
                    ],
                )

                if chunk_done or sl["sim_step"] >= max_horizon:
                    # Bootstrap V(s_T) only when truncated by the horizon
                    final_val = 0.0
                    if not chunk_done:
                        with torch.no_grad():
                            s_final = self.sampler.encode_observation(self.policy._prepare_observation(obs))
                            final_val = float(self.critic(s_final).view(-1)[0].cpu().item())
                    self.buffer.finish_episode_obj(sl["episode"], success=sl["success"], final_value=final_val)
                    finished.append({
                        "plan_idx": sl["plan_idx"],
                        "phase_type": sl["phase_type"],
                        "success": sl["success"],
                        "grasp_step": sl["grasp_step"],
                    })
                    start_slot(i)

        # Aggregate metrics
        def rate(phase: str) -> float:
            eps = [f for f in finished if f["phase_type"] == phase]
            return (sum(f["success"] for f in eps) / len(eps) * 100.0) if eps else 0.0

        finished.sort(key=lambda f: f["plan_idx"])
        return {
            "nominal_success_rate": rate("nominal"),
            "approach_success_rate": rate(PHASE_APPROACHING),
            "pre_grasp_success_rate": rate(PHASE_PRE_GRASP),
            "total_episodes": len(episode_plan),
            "grasp_times": [f["grasp_step"] for f in finished if f["success"]],
        }

    def update_policy_and_critic(self) -> Dict[str, float]:
        """
        Execute E epochs of minibatch updates using Equation (7):
        L(θ, ϕ) = L_actor(θ) + c_v L_critic(ϕ) + β L_KL(θ)
        """
        flat_denoising, flat_env, buffer_stats = self.buffer.compute_gae_and_flatten()

        if len(flat_denoising) == 0:
            return {}

        # 1. Update Critic on environment transitions (s_t, R̂_t)
        env_states = torch.stack([x["obs_cond"] for x in flat_env]).to(self.device)
        env_targets = torch.tensor([x["target"] for x in flat_env], dtype=torch.float32, device=self.device)

        critic_dataset = TensorDataset(env_states, env_targets)
        critic_loader = DataLoader(critic_dataset, batch_size=min(self.batch_size, len(critic_dataset)), shuffle=True)

        critic_losses = []
        for _ in range(self.epochs):
            for b_states, b_targets in critic_loader:
                self.critic_optimizer.zero_grad()
                pred_v = self.critic(b_states).view(-1)
                c_loss, _ = compute_critic_loss(pred_v, b_targets)
                c_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
                critic_losses.append(c_loss.item())

        # 2. Update Actor on denoising transitions (s_t, a^k, k, a^{k-1}, μ_old, σ_k, A_t)
        b_conds = torch.stack([x["obs_cond"] for x in flat_denoising]).to(self.device)
        b_samples = torch.stack([x["sample"] for x in flat_denoising]).to(self.device).squeeze(1)
        b_prev_samples = torch.stack([x["prev_sample"] for x in flat_denoising]).to(self.device).squeeze(1)
        b_mu_olds = torch.stack([x["mu_old"] for x in flat_denoising]).to(self.device).squeeze(1)
        b_sigmas = torch.stack([x["sigma"] for x in flat_denoising]).to(self.device).squeeze(1)
        b_timesteps = torch.tensor([x["timestep"] for x in flat_denoising], dtype=torch.long, device=self.device)
        b_advantages = torch.tensor([x["advantage"] for x in flat_denoising], dtype=torch.float32, device=self.device)

        actor_dataset = TensorDataset(b_conds, b_samples, b_prev_samples, b_mu_olds, b_sigmas, b_timesteps, b_advantages)
        actor_loader = DataLoader(actor_dataset, batch_size=min(self.batch_size, len(actor_dataset)), shuffle=True)

        actor_losses = []
        kl_losses = []
        clip_fractions = []

        approx_kls = []
        epochs_run = 0
        self.noise_pred_net.train()

        for epoch in range(self.epochs):
            epoch_kls = []
            for cond, sample, prev_sample, mu_old, sigma, t_step, adv in actor_loader:
                self.actor_optimizer.zero_grad()

                # Current policy reverse mean μ_θ(a^k, k, s)
                # Compute μ for current batch
                B = sample.shape[0]
                noise_pred = self.noise_pred_net(sample=sample, timestep=t_step, global_cond=cond)

                alpha_prod_t = self.sampler.alpha_prod_t[t_step].view(B, 1, 1)
                beta_prod_t = self.sampler.beta_prod_t[t_step].view(B, 1, 1)

                pred_x0 = (sample - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

                c_orig = self.sampler.pred_x0_coeff[t_step].view(B, 1, 1)
                c_curr = self.sampler.current_sample_coeff[t_step].view(B, 1, 1)
                mu_curr = c_orig * pred_x0 + c_curr * sample

                # Reference pretrained policy reverse mean μ_{θ_0}
                with torch.no_grad():
                    noise_ref = self.ref_noise_pred_net(sample=sample, timestep=t_step, global_cond=cond)
                    pred_x0_ref = (sample - torch.sqrt(beta_prod_t) * noise_ref) / torch.sqrt(alpha_prod_t)
                    pred_x0_ref = torch.clamp(pred_x0_ref, -1.0, 1.0)
                    mu_ref = c_orig * pred_x0_ref + c_curr * sample

                # Losses
                act_loss, act_metrics = compute_actor_loss(
                    mu_curr=mu_curr,
                    mu_old=mu_old,
                    a_prev=prev_sample,
                    sigma_k=sigma,
                    advantages=adv,
                    clip_eps=self.clip_eps,
                )

                kl_loss, kl_metrics = compute_kl_regularization(
                    mu_curr=mu_curr,
                    mu_ref=mu_ref,
                    sigma_k=sigma,
                )

                total_actor_loss = act_loss + self.beta * kl_loss
                total_actor_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.noise_pred_net.parameters(), 1.0)
                self.actor_optimizer.step()

                actor_losses.append(act_loss.item())
                kl_losses.append(kl_loss.item())
                clip_fractions.append(act_metrics["clip_fraction"])
                epoch_kls.append(act_metrics["approx_kl"])

            epochs_run = epoch + 1
            approx_kls.extend(epoch_kls)
            # PPO early stopping: the policy has drifted far enough from the rollout
            # policy that further epochs would mostly hit the clip.
            if self.target_kl is not None and float(np.mean(epoch_kls)) > self.target_kl:
                logger.info(
                    f"Early stop after epoch {epochs_run}/{self.epochs}: "
                    f"approx_kl {np.mean(epoch_kls):.4f} > target_kl {self.target_kl}"
                )
                break

        self.noise_pred_net.eval()

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
            "epochs_run": float(epochs_run),
        }

    def evaluate(self, n_per_regime: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate current policy on 5 regimes with the SAME number of episodes each:
        nominal, train-approach, train-pre-grasp, held-out-approach, held-out-pre-grasp.

        Uses a fixed rl.eval_seed so every call sees identical initial object poses
        and disturbance draws (paired comparison across checkpoints), then restores
        the training RNG state so training randomness is unaffected.
        """
        rl_cfg = self.config.get("rl", {})
        n = int(n_per_regime if n_per_regime is not None else rl_cfg.get("eval_episodes_per_regime", 30))
        eval_seed = rl_cfg.get("eval_seed", 1234)
        logger.info(f"Running evaluation ({n} rollouts per regime, 5 regimes, eval_seed={eval_seed})...")
        self.noise_pred_net.eval()

        saved_rng = _get_rng_state()
        if eval_seed is not None:
            _seed_everything(int(eval_seed))
        try:
            train_plan = self.dist_manager.plan_episodes(n, n, n, mode="train")
            ho_plan = self.dist_manager.plan_episodes(0, n, n, mode="held_out")
            # Evaluate with the true DDPM chain (no exploration σ floor), matching how
            # the pretrained baseline is evaluated.
            train_stats = self.collect_rollouts(0, mode="train", min_sampling_std=0.0, episode_plan=train_plan)
            ho_stats = self.collect_rollouts(0, mode="held_out", min_sampling_std=0.0, episode_plan=ho_plan)
        finally:
            _set_rng_state(saved_rng)
            self.buffer.clear()

        return {
            "eval_nominal_success": train_stats["nominal_success_rate"],
            "eval_train_approach_success": train_stats["approach_success_rate"],
            "eval_train_pre_grasp_success": train_stats["pre_grasp_success_rate"],
            "eval_ho_approach_success": ho_stats["approach_success_rate"],
            "eval_ho_pre_grasp_success": ho_stats["pre_grasp_success_rate"],
            "eval_episodes_per_regime": float(n),
        }

    @staticmethod
    def _selection_score(m: Dict[str, float]) -> float:
        """Pareto score prioritizing both nominal and held-out disturbed recovery."""
        return (
            0.4 * m["eval_nominal_success"]
            + 0.3 * m["eval_ho_approach_success"]
            + 0.3 * m["eval_ho_pre_grasp_success"]
        )

    def _log_eval(self, tag: str, m: Dict[str, float]) -> None:
        logger.info(
            f"{tag}: Nominal={m['eval_nominal_success']:.1f}%, "
            f"TrainApp={m['eval_train_approach_success']:.1f}%, "
            f"TrainPre={m['eval_train_pre_grasp_success']:.1f}%, "
            f"HO App={m['eval_ho_approach_success']:.1f}%, "
            f"HO Pre={m['eval_ho_pre_grasp_success']:.1f}% "
            f"(n={int(m['eval_episodes_per_regime'])}/regime, score={self._selection_score(m):.2f})"
        )

    def save_checkpoint(
        self,
        path: Union[str, Path],
        iteration: int,
        metrics: Dict[str, Any],
        training_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Export fine-tuned policy checkpoint. If training_state is given, optimizer
        and RNG state are included so the run can be resumed (see resume_state()).
        Written to a temp file and renamed, so a kill mid-save never leaves a
        truncated checkpoint.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "iteration": iteration,
            "base_checkpoint": str(self.checkpoint_path),
            "policy_state_dict": self.noise_pred_net.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "metrics": metrics,
        }
        if training_state is not None:
            ckpt["training_state"] = {
                **training_state,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "rng": _get_rng_state(),
            }
        tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, save_path)
        logger.info(f"Saved DPPO checkpoint to {save_path}")
        return str(save_path)

    def resume_state(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Restore weights, optimizers and RNG from a checkpoint saved with training_state."""
        ckpt = self.load_finetuned_weights(path)
        state = ckpt.get("training_state")
        if state is None:
            raise ValueError(f"{path} has no training_state; resume from dppo_last.pt")
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        rng = state["rng"]
        if "cuda" in rng:
            rng["cuda"] = [t.cpu() for t in rng["cuda"]]
        rng["torch"] = rng["torch"].cpu()
        _set_rng_state(rng)
        logger.info(f"Resumed training state from {path} at iteration {ckpt['iteration']}")
        return {**state, "iteration": int(ckpt["iteration"])}

    def train(
        self,
        total_iterations: int = 100,
        episodes_per_iter: int = 20,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete DPPO training loop (Algorithm 1).

        Every iteration overwrites save_dir/dppo_last.pt (with optimizer + RNG state)
        and appends a line to save_dir/metrics.jsonl, so a killed run can continue
        with resume_from=save_dir/dppo_last.pt.
        """
        logger.info(f"Starting DPPO Training for {total_iterations} iterations, {episodes_per_iter} episodes/iter.")

        rl_cfg = self.config.get("rl", {})
        eval_interval = int(rl_cfg.get("eval_interval", 10))
        final_eval_n = int(rl_cfg.get("final_eval_episodes_per_regime", 0))
        metrics_path = self.save_dir / "metrics.jsonl"

        best_score = -1.0
        best_ckpt = ""
        history: List[Dict[str, Any]] = []
        baseline_metrics: Dict[str, float] = {}
        start_iteration = 1

        if resume_from is not None:
            state = self.resume_state(resume_from)
            start_iteration = state["iteration"] + 1
            best_score = float(state.get("best_score", -1.0))
            best_ckpt = state.get("best_ckpt", "")
            baseline_metrics = state.get("baseline_metrics", {})
            # LR schedule is a pure function of the iteration
            self._decay_actor_lr(state["iteration"])
            logger.info(f"Resuming at iteration {start_iteration} (best_score={best_score:.2f}, best={best_ckpt or 'none'})")
        elif rl_cfg.get("eval_at_start", True):
            # Baseline: pretrained policy on the exact same eval scenarios. A fine-tuned
            # checkpoint is only saved as "best" if it beats this.
            baseline_metrics = self.evaluate()
            self._log_eval("Baseline (pretrained, iter 0)", baseline_metrics)
            best_score = self._selection_score(baseline_metrics)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({"iteration": 0, **baseline_metrics}) + "\n")

        for iteration in range(start_iteration, total_iterations + 1):
            logger.info(f"\n--- Iteration {iteration} / {total_iterations} ---")

            # 1. Collect rollouts
            rollout_stats = self.collect_rollouts(n_episodes=episodes_per_iter, mode="train")
            logger.info(
                f"Rollout: Nominal={rollout_stats['nominal_success_rate']:.1f}%, "
                f"Approach={rollout_stats['approach_success_rate']:.1f}%, "
                f"Pre-Grasp={rollout_stats['pre_grasp_success_rate']:.1f}%"
            )

            # 2. Update policy and critic
            update_metrics = self.update_policy_and_critic()
            logger.info(
                f"Update: ActorLoss={update_metrics.get('actor_loss', 0.0):.4f}, "
                f"CriticLoss={update_metrics.get('critic_loss', 0.0):.4f}, "
                f"KLLoss={update_metrics.get('kl_loss', 0.0):.4f}, "
                f"ClipFrac={update_metrics.get('clip_fraction', 0.0):.3f}, "
                f"ApproxKL={update_metrics.get('approx_kl', 0.0):.4f}, "
                f"Epochs={int(update_metrics.get('epochs_run', 0))}"
            )

            # 3. Parameter schedule step
            self._decay_actor_lr(iteration)

            # 4. Periodic evaluation
            iter_summary = {**rollout_stats, **update_metrics, "iteration": iteration}
            if iteration % eval_interval == 0 or iteration == total_iterations:
                eval_metrics = self.evaluate()
                self._log_eval(f"Evaluation (iter {iteration})", eval_metrics)
                iter_summary.update(eval_metrics)

                score = self._selection_score(eval_metrics)
                if score > best_score:
                    best_score = score
                    best_ckpt = self.save_checkpoint(
                        self.save_dir / f"dppo_best_iter_{iteration}.pt",
                        iteration=iteration,
                        metrics=iter_summary,
                    )

            history.append(iter_summary)
            with open(metrics_path, "a") as f:
                f.write(json.dumps({k: v for k, v in iter_summary.items() if k != "grasp_times"}) + "\n")

            # Resumable snapshot every iteration (overwritten)
            self.save_checkpoint(
                self.save_dir / "dppo_last.pt",
                iteration=iteration,
                metrics=iter_summary,
                training_state={
                    "best_score": best_score,
                    "best_ckpt": best_ckpt,
                    "baseline_metrics": baseline_metrics,
                },
            )

        # Final, larger evaluation of the selected checkpoint (the periodic evals are
        # only for model selection; report these numbers instead).
        final_metrics: Dict[str, float] = {}
        if best_ckpt and final_eval_n > 0:
            self.load_finetuned_weights(best_ckpt)
            final_metrics = self.evaluate(n_per_regime=final_eval_n)
            self._log_eval(f"Final eval of {Path(best_ckpt).name}", final_metrics)
        elif not best_ckpt:
            logger.warning("No fine-tuned checkpoint beat the pretrained baseline score.")

        return {
            "best_checkpoint": best_ckpt,
            "best_score": best_score,
            "baseline_metrics": baseline_metrics,
            "final_metrics": final_metrics,
            "history": history,
        }
