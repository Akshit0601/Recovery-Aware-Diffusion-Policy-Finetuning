"""
Rollout and Denoising Step Experience Buffer with GAE for DPPO.

Stores trajectories of environment decisions and nested DDPM denoising transitions,
computes Generalized Advantage Estimation (GAE) and discounted return targets,
and generates minibatches of (t, k) pairs for policy optimization.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


class DPPORolloutBuffer:
    """
    Experience buffer storing environment transitions and nested denoising chains.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        # Trajectory storage
        self.episodes: List[Dict[str, Any]] = []
        self.current_episode: Dict[str, Any] = self._create_empty_episode()

    def _create_empty_episode(self) -> Dict[str, Any]:
        return {
            "obs_conds": [],       # s_t ∈ R³⁸
            "rewards": [],         # r_t ∈ {0, 1}
            "values": [],          # V(s_t)
            "dones": [],           # bool
            "denoising_steps": [], # List of lists of (a^k, k, a^{k-1}, mu_old, sigma_k)
            "success": False,
            "disturbed": False,
            "phase": "nominal",
        }

    def start_episode(
        self,
        is_disturbed: bool = False,
        phase: str = "nominal",
    ) -> None:
        """Initialize tracking for a new episode."""
        self.current_episode = self._create_empty_episode()
        self.current_episode["disturbed"] = is_disturbed
        self.current_episode["phase"] = phase

    def add_step(
        self,
        obs_cond: torch.Tensor,
        reward: float,
        value: float,
        done: bool,
        denoising_transitions: List[Dict[str, Any]],
    ) -> None:
        """Add a single environment chunk step and its reverse denoising chain."""
        self.current_episode["obs_conds"].append(obs_cond.detach().cpu())
        self.current_episode["rewards"].append(float(reward))
        self.current_episode["values"].append(float(value))
        self.current_episode["dones"].append(bool(done))

        # Store denoising transitions for this chunk decision step
        step_transitions = []
        for tr in denoising_transitions:
            step_transitions.append({
                "timestep": int(tr["timestep"]),
                "sample": tr["sample"].detach().cpu(),
                "prev_sample": tr["prev_sample"].detach().cpu(),
                "mu": tr["mu"].detach().cpu(),
                "sigma": tr["sigma"].detach().cpu(),
            })
        self.current_episode["denoising_steps"].append(step_transitions)

    def finish_episode(self, success: bool, final_value: float = 0.0) -> None:
        """Mark end of episode, store final bootstrap value, and save to buffer."""
        self.current_episode["success"] = bool(success)
        self.current_episode["final_value"] = float(final_value)
        self.episodes.append(self.current_episode)
        self.current_episode = self._create_empty_episode()

    def compute_gae_and_flatten(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
        """
        Compute GAE advantages and value targets for all stored episodes,
        and flatten into training batches:
        1. List of (t, k) denoising tuples paired with advantages A_t.
        2. List of environment state-target tuples (s_t, R̂_t) for critic training.

        Returns:
            flat_denoising_data: List of dicts for actor updates.
            flat_env_data: List of dicts for critic updates.
            stats: Summary rollout metrics.
        """
        all_advantages = []
        flat_denoising_data = []
        flat_env_data = []

        total_episodes = len(self.episodes)
        total_successes = sum(1 for ep in self.episodes if ep["success"])
        success_rate = (total_successes / total_episodes * 100.0) if total_episodes > 0 else 0.0

        for ep in self.episodes:
            rewards = ep["rewards"]
            values = ep["values"]
            dones = ep["dones"]
            obs_conds = ep["obs_conds"]
            denoising_steps = ep["denoising_steps"]
            T = len(rewards)

            if T == 0:
                continue

            # Compute GAE advantages A_t and returns R̂_t
            advantages = np.zeros(T, dtype=np.float32)
            returns = np.zeros(T, dtype=np.float32)
            last_gae = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = ep["final_value"] if not dones[t] else 0.0
                else:
                    next_val = values[t + 1]

                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (0.0 if dones[t] else 1.0) * last_gae
                advantages[t] = last_gae
                returns[t] = advantages[t] + values[t]

            all_advantages.extend(advantages.tolist())

            # Store env data for critic
            for t in range(T):
                flat_env_data.append({
                    "obs_cond": obs_conds[t],
                    "target": float(returns[t]),
                    "value": float(values[t]),
                })

                # Store denoising data for actor
                adv_t = float(advantages[t])
                for d_step in denoising_steps[t]:
                    flat_denoising_data.append({
                        "obs_cond": obs_conds[t],
                        "timestep": d_step["timestep"],
                        "sample": d_step["sample"],
                        "prev_sample": d_step["prev_sample"],
                        "mu_old": d_step["mu"],
                        "sigma": d_step["sigma"],
                        "advantage": adv_t,
                    })

        # Normalize advantages across batch
        if len(all_advantages) > 0:
            adv_arr = np.array(all_advantages, dtype=np.float32)
            adv_mean = float(np.mean(adv_arr))
            adv_std = float(np.std(adv_arr)) + 1e-8
            for item in flat_denoising_data:
                item["advantage"] = (item["advantage"] - adv_mean) / adv_std

        stats = {
            "total_episodes": total_episodes,
            "success_rate": success_rate,
            "total_env_steps": len(flat_env_data),
            "total_denoising_steps": len(flat_denoising_data),
        }
        return flat_denoising_data, flat_env_data, stats

    def clear(self) -> None:
        """Reset buffer storage."""
        self.episodes = []
        self.current_episode = self._create_empty_episode()
