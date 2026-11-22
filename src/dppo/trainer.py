"""
DPPO Recovery-Aware Fine-Tuning Pipeline (Algorithm 1 from ECE285_MidTerm.pdf).

Executes on-policy rollouts with online stratified physical disturbances,
optimizes diffusion policies over reverse denoising chains with clipped PPO loss,
anchors updates to the pretrained nominal policy using analytical closed-form KL penalty,
and benchmarks recovery generalization on training and held-out disturbance distributions.
"""

from __future__ import annotations

import copy
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
        self.policy, self.ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=self.checkpoint_path,
            device=self.device,
            verbose=False,
        )
        self.algo = self.policy.policy
        self.algo.set_eval()

        # Build training environment
        self.env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=self.ckpt_dict,
            render=False,
            verbose=False,
        )

        # Clone and freeze reference policy θ_ref = θ_0
        self.ref_noise_pred_net = copy.deepcopy(self.algo.nets["policy"]["noise_pred_net"]).to(self.device)
        self.ref_noise_pred_net.eval()
        for p in self.ref_noise_pred_net.parameters():
            p.requires_grad = False

        self.noise_pred_net = self.algo.nets["policy"]["noise_pred_net"]
        self.obs_encoder = self.algo.nets["policy"]["obs_encoder"]
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

        all_timesteps = list(range(self.algo.algo_config.ddpm.num_train_timesteps))
        if k_ft_mode == "strided":
            self.k_ft_set = set(all_timesteps[::k_ft_stride])
        elif k_ft_mode == "tail":
            tail_m = int(rollout_cfg.get("k_ft_tail_m", 25))
            self.k_ft_set = set(all_timesteps[:tail_m])
        else:
            self.k_ft_set = set(all_timesteps)

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
        )

    def _setup_disturbances(self) -> None:
        """Initialize disturbance configuration and manager."""
        dist_cfg = self.config.get("disturbances", {})
        rollout_cfg = self.config.get("rollout", {})

        cfg = DisturbanceConfig(
            nominal_fraction=float(rollout_cfg.get("nominal_fraction", 0.4)),
            disturbed_fraction=float(rollout_cfg.get("disturbed_fraction", 0.6)),
            phase_ratio=tuple(rollout_cfg.get("phase_ratio", (0.5, 0.5))),
        )
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

    def collect_rollouts(
        self,
        n_episodes: int,
        mode: str = "train",
    ) -> Dict[str, Any]:
        """
        Collect on-policy rollout episodes using stratified disturbance sampling (Algorithm 1 lines 4-8).
        """
        self.buffer.clear()
        episode_plan = self.dist_manager.plan_iteration_episodes(n_episodes=n_episodes, mode=mode)

        nominal_successes = 0
        nominal_count = 0
        approach_successes = 0
        approach_count = 0
        pre_grasp_successes = 0
        pre_grasp_count = 0
        grasp_times: List[int] = []

        max_horizon = int(self.config.get("rollout", {}).get("horizon", 400))
        Ta = self.algo.algo_config.horizon.action_horizon

        for ep_idx, dist_spec in enumerate(episode_plan):
            is_disturbed = dist_spec is not None
            phase_type = dist_spec["phase"] if is_disturbed else "nominal"
            self.dist_manager.reset_episode(dist_spec)

            self.policy.start_episode()
            obs = self.env.reset()
            sim_step = 0
            ep_success = False

            self.buffer.start_episode(is_disturbed=is_disturbed, phase=phase_type)

            while sim_step < max_horizon:
                # Prepare observation
                ob_tensor = self.policy._prepare_observation(obs)
                obs_cond = self.sampler.encode_observation(ob_tensor)

                # Query critic value V_ϕ(s_t)
                with torch.no_grad():
                    val = float(self.critic(obs_cond).cpu().item())

                # Generate action chunk and record denoising transitions
                with torch.no_grad():
                    executed_actions, _, denoising_transitions = self.sampler.sample_action_chunk_with_trajectory(
                        obs_dict=ob_tensor,
                        noise_pred_net=self.noise_pred_net,
                        k_ft_set=self.k_ft_set,
                    )

                chunk_actions = executed_actions.detach().squeeze(0).cpu().numpy()  # shape (Ta, Da)
                chunk_reward = 0.0
                chunk_done = False

                for sub_i in range(Ta):
                    act = chunk_actions[sub_i]
                    next_obs, r, d, info = self.env.step(act)
                    sim_step += 1

                    # Detect phase
                    curr_phase = self.dist_manager.detect_phase(act, r, step_i=sim_step)

                    # Inject disturbance if criteria match
                    if self.dist_manager.should_perturb():
                        self.dist_manager.inject_disturbance(self.env, action=act)
                        next_obs = self.env.get_observation() if hasattr(self.env, "get_observation") else next_obs

                    # Check success
                    is_succ = False
                    if hasattr(self.env, "is_success"):
                        res = self.env.is_success()
                        is_succ = bool(res.get("task", False)) if isinstance(res, dict) else bool(res)
                    elif r >= 1.0:
                        is_succ = True

                    if is_succ:
                        chunk_reward = 1.0
                        chunk_done = True
                        ep_success = True
                        grasp_times.append(sim_step)
                        obs = next_obs
                        break

                    obs = next_obs
                    if d:
                        chunk_done = True
                        break

                # Add chunk step to buffer
                self.buffer.add_step(
                    obs_cond=obs_cond.squeeze(0),
                    reward=chunk_reward,
                    value=val,
                    done=chunk_done,
                    denoising_transitions=denoising_transitions,
                )

                if chunk_done:
                    break

            # Bootstrap if truncated
            final_val = 0.0
            if not ep_success and sim_step >= max_horizon:
                with torch.no_grad():
                    ob_tensor = self.policy._prepare_observation(obs)
                    s_final = self.sampler.encode_observation(ob_tensor)
                    final_val = float(self.critic(s_final).cpu().item())

            self.buffer.finish_episode(success=ep_success, final_value=final_val)

            # Accumulate metrics
            if not is_disturbed:
                nominal_count += 1
                if ep_success:
                    nominal_successes += 1
            else:
                if phase_type == PHASE_APPROACHING:
                    approach_count += 1
                    if ep_success:
                        approach_successes += 1
                elif phase_type == PHASE_PRE_GRASP:
                    pre_grasp_count += 1
                    if ep_success:
                        pre_grasp_successes += 1

        stats = {
            "nominal_success_rate": (nominal_successes / nominal_count * 100.0) if nominal_count > 0 else 0.0,
            "approach_success_rate": (approach_successes / approach_count * 100.0) if approach_count > 0 else 0.0,
            "pre_grasp_success_rate": (pre_grasp_successes / pre_grasp_count * 100.0) if pre_grasp_count > 0 else 0.0,
            "total_episodes": len(episode_plan),
            "grasp_times": grasp_times,
        }
        return stats

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

        self.noise_pred_net.train()

        for _ in range(self.epochs):
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

        self.noise_pred_net.eval()

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "kl_loss": float(np.mean(kl_losses)) if kl_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        }

    def evaluate(self, n_episodes: int = 20) -> Dict[str, float]:
        """
        Evaluate current policy on:
        1. Nominal task (ξ = ∅)
        2. Training disturbances (Ξ_train)
        3. Held-out disturbances (Ξ_ho)
        """
        logger.info(f"Running evaluation ({n_episodes} rollouts per regime)...")
        self.noise_pred_net.eval()

        # Evaluate on training distribution
        train_stats = self.collect_rollouts(n_episodes=n_episodes, mode="train")
        # Evaluate on held-out distribution
        ho_stats = self.collect_rollouts(n_episodes=n_episodes, mode="held_out")

        eval_summary = {
            "eval_nominal_success": train_stats["nominal_success_rate"],
            "eval_train_approach_success": train_stats["approach_success_rate"],
            "eval_train_pre_grasp_success": train_stats["pre_grasp_success_rate"],
            "eval_ho_approach_success": ho_stats["approach_success_rate"],
            "eval_ho_pre_grasp_success": ho_stats["pre_grasp_success_rate"],
        }
        return eval_summary

    def save_checkpoint(self, path: Union[str, Path], iteration: int, metrics: Dict[str, float]) -> str:
        """Export fine-tuned policy checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "iteration": iteration,
            "policy_state_dict": self.noise_pred_net.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "metrics": metrics,
        }
        torch.save(ckpt, save_path)
        logger.info(f"Saved DPPO checkpoint to {save_path}")
        return str(save_path)

    def train(self, total_iterations: int = 100, episodes_per_iter: int = 20) -> Dict[str, Any]:
        """
        Run the complete DPPO training loop (Algorithm 1).
        """
        logger.info(f"Starting DPPO Training for {total_iterations} iterations, {episodes_per_iter} episodes/iter.")

        best_score = -1.0
        best_ckpt = ""
        history = []

        eval_interval = int(self.config.get("rl", {}).get("eval_interval", 10))
        eval_episodes = int(self.config.get("rl", {}).get("eval_episodes", 20))

        for iteration in range(1, total_iterations + 1):
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
                f"ClipFrac={update_metrics.get('clip_fraction', 0.0):.3f}"
            )

            # 3. Parameter schedule step
            self._decay_actor_lr(iteration)

            # 4. Periodic evaluation
            iter_summary = {**rollout_stats, **update_metrics, "iteration": iteration}
            if iteration % eval_interval == 0 or iteration == total_iterations:
                eval_metrics = self.evaluate(n_episodes=eval_episodes)
                logger.info(
                    f"Evaluation: Nominal={eval_metrics['eval_nominal_success']:.1f}%, "
                    f"HO Approach={eval_metrics['eval_ho_approach_success']:.1f}%, "
                    f"HO PreGrasp={eval_metrics['eval_ho_pre_grasp_success']:.1f}%"
                )
                iter_summary.update(eval_metrics)

                # Pareto score prioritizing both nominal and held-out disturbed recovery
                score = (
                    0.4 * eval_metrics["eval_nominal_success"]
                    + 0.3 * eval_metrics["eval_ho_approach_success"]
                    + 0.3 * eval_metrics["eval_ho_pre_grasp_success"]
                )
                if score > best_score:
                    best_score = score
                    best_ckpt = self.save_checkpoint(
                        self.save_dir / f"dppo_best_iter_{iteration}.pt",
                        iteration=iteration,
                        metrics=iter_summary,
                    )

            history.append(iter_summary)

        # Save last checkpoint
        self.save_checkpoint(self.save_dir / "dppo_last.pt", iteration=total_iterations, metrics=history[-1])

        return {
            "best_checkpoint": best_ckpt,
            "best_score": best_score,
            "history": history,
        }
