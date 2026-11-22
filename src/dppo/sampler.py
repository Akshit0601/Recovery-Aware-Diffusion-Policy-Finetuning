"""
DDPM Reverse Chain Sampler with Step Transition Logging for DPPO.

Generates action chunks via the reverse diffusion chain while recording intermediate
Gaussian transitions (s_t, a_t^k, k, a_t^{k-1}, μ_{θ_old}, σ_k) for policy gradient updates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import robomimic.utils.tensor_utils as TensorUtils


class DDPMSampler:
    """
    Stochastic sampler for Conditional DDPM over action chunks.
    """

    def __init__(
        self,
        noise_scheduler: Any,
        obs_encoder: nn.Module,
        noise_pred_net: nn.Module,
        obs_shapes: Dict[str, Any],
        device: torch.device,
        num_timesteps: int = 100,
        prediction_horizon: int = 16,
        action_horizon: int = 8,
        observation_horizon: int = 2,
        action_dim: int = 7,
    ) -> None:
        self.noise_scheduler = noise_scheduler
        self.obs_encoder = obs_encoder
        self.noise_pred_net = noise_pred_net
        self.obs_shapes = obs_shapes
        self.device = device

        self.num_timesteps = num_timesteps
        self.Tp = prediction_horizon
        self.Ta = action_horizon
        self.To = observation_horizon
        self.Da = action_dim

        # Precompute schedule tensors on target device
        self._cache_schedule_constants()

    def _cache_schedule_constants(self) -> None:
        """Precompute DDPM reverse coefficients for fast forward inference."""
        self.noise_scheduler.set_timesteps(self.num_timesteps)
        self.timesteps = [int(t) for t in self.noise_scheduler.timesteps]

        alphas = torch.as_tensor(self.noise_scheduler.alphas, dtype=torch.float32, device=self.device)
        alphas_cumprod = torch.as_tensor(self.noise_scheduler.alphas_cumprod, dtype=torch.float32, device=self.device)
        betas = torch.as_tensor(self.noise_scheduler.betas, dtype=torch.float32, device=self.device)

        T = len(betas)
        self.alpha_prod_t = alphas_cumprod
        self.alpha_prod_t_prev = torch.cat([torch.tensor([1.0], device=self.device), alphas_cumprod[:-1]])
        self.beta_prod_t = 1.0 - self.alpha_prod_t
        self.beta_prod_t_prev = 1.0 - self.alpha_prod_t_prev
        self.betas = betas
        self.alphas = alphas

        # Variance per step: σ_t² = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) * β_t
        var = (self.beta_prod_t_prev / self.beta_prod_t) * self.betas
        var = torch.clamp(var, min=1e-20)
        self.variances = var
        self.sigmas = torch.sqrt(var)

        # Coefficients for pred_original_sample x_0 and current sample x_t
        self.pred_x0_coeff = (torch.sqrt(self.alpha_prod_t_prev) * self.betas) / self.beta_prod_t
        self.current_sample_coeff = (torch.sqrt(self.alphas) * self.beta_prod_t_prev) / self.beta_prod_t

    def encode_observation(
        self,
        obs_dict: Dict[str, Any],
        goal_dict: Optional[Dict[str, Any]] = None,
        encoder: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Encode raw observation dict into flattened observation conditioning vector s_t ∈ R³⁸.
        """
        active_encoder = encoder if encoder is not None else self.obs_encoder

        inputs = {"obs": obs_dict, "goal": goal_dict}
        for k in self.obs_shapes:
            if inputs["obs"][k].ndim - 1 == len(self.obs_shapes[k]):
                inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
            assert inputs["obs"][k].ndim - 2 == len(self.obs_shapes[k])

        obs_features = TensorUtils.time_distributed(
            inputs, active_encoder, inputs_as_kwargs=True
        )
        # obs_features shape is [B, To, D_obs] -> flatten to [B, To * D_obs]
        obs_cond = obs_features.flatten(start_dim=1)
        return obs_cond

    def compute_step_mean(
        self,
        noise_pred_net: nn.Module,
        sample: torch.Tensor,
        t: int,
        obs_cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute DDPM reverse step mean μ_θ(a^t, t, s) and standard deviation σ_t.

        Args:
            noise_pred_net: Noise prediction network ϵ_θ
            sample: Current noisy action sample a^t of shape (B, Tp, Da)
            t: Integer timestep in [0, K-1]
            obs_cond: Observation conditioning vector of shape (B, To*Dobs)

        Returns:
            mu: Reverse Gaussian mean μ_θ of shape (B, Tp, Da)
            sigma: Standard deviation σ_t of shape (B, 1, 1)
        """
        B = sample.shape[0]
        timestep_tensor = torch.full((B,), t, dtype=torch.long, device=sample.device)

        noise_pred = noise_pred_net(
            sample=sample,
            timestep=timestep_tensor,
            global_cond=obs_cond,
        )

        alpha_prod_t = self.alpha_prod_t[t]
        beta_prod_t = self.beta_prod_t[t]

        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        # Clip sample to [-1, 1] as configured in robomimic checkpoint
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        c_orig = self.pred_x0_coeff[t]
        c_curr = self.current_sample_coeff[t]
        mu = c_orig * pred_original_sample + c_curr * sample

        sigma = self.sigmas[t].view(1, 1, 1).expand(B, 1, 1)
        return mu, sigma

    def sample_action_chunk_with_trajectory(
        self,
        obs_dict: Dict[str, Any],
        noise_pred_net: Optional[nn.Module] = None,
        obs_encoder: Optional[nn.Module] = None,
        k_ft_set: Optional[Set[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Run the complete K-step DDPM reverse diffusion chain, recording (s_t, a^k, k, a^{k-1}, μ, σ).

        Args:
            obs_dict: Observation dictionary from environment
            noise_pred_net: Policy network to use (defaults to self.noise_pred_net)
            obs_encoder: Encoder network to use (defaults to self.obs_encoder)
            k_ft_set: Set of timestep integers that receive gradient K_ft (None = all timesteps)

        Returns:
            executed_actions: Action chunk to execute of shape (B, Ta, Da)
            obs_cond: Stacked observation vector s_t of shape (B, To*Dobs)
            transitions: List of recorded denoising step dictionaries
        """
        active_net = noise_pred_net if noise_pred_net is not None else self.noise_pred_net
        active_encoder = obs_encoder if obs_encoder is not None else self.obs_encoder

        obs_cond = self.encode_observation(obs_dict, encoder=active_encoder)
        B = obs_cond.shape[0]

        # Initial random Gaussian sample a^K ~ N(0, I)
        naction = torch.randn((B, self.Tp, self.Da), device=self.device)

        recorded_transitions: List[Dict[str, Any]] = []

        for t in self.timesteps:
            mu, sigma = self.compute_step_mean(active_net, naction, t, obs_cond)

            # Draw a^{k-1} ~ N(μ, σ² I)
            if t > 0:
                noise = torch.randn_like(naction)
                a_prev = mu + sigma * noise
            else:
                a_prev = mu

            # If this step receives fine-tuning gradient
            if k_ft_set is None or t in k_ft_set:
                recorded_transitions.append({
                    "timestep": t,
                    "sample": naction.detach().clone(),
                    "prev_sample": a_prev.detach().clone(),
                    "mu": mu.detach().clone(),
                    "sigma": sigma.detach().clone(),
                })

            naction = a_prev

        # Slice Ta action steps to execute: from (To - 1) to (To - 1 + Ta)
        start = self.To - 1
        end = start + self.Ta
        executed_actions = naction[:, start:end]

        return executed_actions, obs_cond, recorded_transitions
