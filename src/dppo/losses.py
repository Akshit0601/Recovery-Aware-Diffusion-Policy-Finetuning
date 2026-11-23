"""
Loss Functions for DPPO with Analytical KL Divergence Regularization.

Implements Equations (4), (5), (6), and (7) from ECE285_MidTerm.pdf:
- Per-step clipped PPO actor loss L_actor(θ)
- Critic MSE loss L_critic(ϕ)
- Closed-form analytical KL divergence regularization L_KL(θ)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_actor_loss(
    mu_curr: torch.Tensor,
    mu_old: torch.Tensor,
    a_prev: torch.Tensor,
    sigma_k: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute per-step clipped surrogate actor loss (Equation 5).

    ρ_{t,k}(θ) = exp( (||a^{k-1} - μ_{old}||² - ||a^{k-1} - μ_θ||²) / (2 σ_k²) )
    L_actor(θ) = - (1 / |B|) ∑ min( ρ_{t,k} A_t, clip(ρ_{t,k}, 1-ε, 1+ε) A_t )

    Args:
        mu_curr: Current policy mean μ_θ(a^k, k, s) of shape (B, Tp, Da)
        mu_old: Rollout policy mean μ_{θ_old}(a^k, k, s) of shape (B, Tp, Da)
        a_prev: Sampled previous denoising state a^{k-1} of shape (B, Tp, Da)
        sigma_k: Standard deviation or sqrt(variance) for denoising step k of shape (B, 1, 1) or scalar
        advantages: Shared chunk advantage A_t of shape (B,) or (B, 1)
        clip_eps: PPO clipping epsilon ε (default 0.2)

    Returns:
        loss: Scalar actor loss
        metrics: Dictionary of diagnostic metrics (ratio mean, clip fraction, etc.)
    """
    var_k = (sigma_k ** 2).clamp(min=1e-8)

    # Flatten chunk dimensions (Tp, Da) to sum squared Euclidean norms over all action dims
    diff_old = (a_prev - mu_old).view(a_prev.shape[0], -1)
    diff_curr = (a_prev - mu_curr).view(a_prev.shape[0], -1)

    norm_sq_old = torch.sum(diff_old ** 2, dim=-1, keepdim=True)
    norm_sq_curr = torch.sum(diff_curr ** 2, dim=-1, keepdim=True)

    var_flat = var_k.view(a_prev.shape[0], -1)[:, :1]

    # Log ratio: (||a^{k-1} - μ_{old}||² - ||a^{k-1} - μ_θ||²) / (2 σ_k²)
    log_ratio = (norm_sq_old - norm_sq_curr) / (2.0 * var_flat)
    # Clamp log_ratio to prevent numerical overflow
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio).squeeze(-1)  # shape (B,)

    flat_adv = advantages.view(-1)
    surr1 = ratio * flat_adv
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr2 = clipped_ratio * flat_adv

    actor_loss = -torch.mean(torch.min(surr1, surr2))

    # Diagnostic statistics
    with torch.no_grad():
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()
        mean_ratio = torch.mean(ratio).item()
        # Low-variance estimator of KL(π_old || π_θ) per denoising step (Schulman's k3)
        lr_flat = log_ratio.squeeze(-1)
        approx_kl = torch.mean((ratio - 1.0) - lr_flat).item()

    metrics = {
        "actor_loss": actor_loss.item(),
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
        "approx_kl": approx_kl,
    }
    return actor_loss, metrics


def compute_critic_loss(
    values: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute MSE critic value loss (Equation 7):
    L_critic(ϕ) = (1 / |B|) ∑ (V_ϕ(s_t) - R̂_t)²

    Args:
        values: Predicted state values V_ϕ(s_t) of shape (B,) or (B, 1)
        targets: GAE value targets R̂_t of shape (B,) or (B, 1)

    Returns:
        loss: Scalar critic MSE loss
        metrics: Dictionary of diagnostic metrics
    """
    v_flat = values.view(-1)
    t_flat = targets.view(-1)
    loss = F.mse_loss(v_flat, t_flat)

    metrics = {
        "critic_loss": loss.item(),
        "mean_value": v_flat.mean().item(),
        "mean_target": t_flat.mean().item(),
    }
    return loss, metrics


def compute_kl_regularization(
    mu_curr: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma_k: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute analytical closed-form per-step KL divergence (Equation 6):
    KL(π_θ || π_{θ_0})|_{t,k} = ||μ_θ - μ_{θ_0}||² / (2 σ_k²)

    Args:
        mu_curr: Current policy mean μ_θ(a^k, k, s) of shape (B, Tp, Da)
        mu_ref: Pretrained anchor policy mean μ_{θ_0}(a^k, k, s) of shape (B, Tp, Da)
        sigma_k: Denoising step standard deviation of shape (B, 1, 1) or scalar

    Returns:
        loss: Scalar KL penalty term
        metrics: Dictionary of diagnostic metrics
    """
    var_k = (sigma_k ** 2).clamp(min=1e-8)

    diff = (mu_curr - mu_ref).view(mu_curr.shape[0], -1)
    norm_sq = torch.sum(diff ** 2, dim=-1, keepdim=True)

    var_flat = var_k.view(mu_curr.shape[0], -1)[:, :1]
    per_sample_kl = norm_sq / (2.0 * var_flat)
    kl_loss = torch.mean(per_sample_kl)

    metrics = {
        "kl_loss": kl_loss.item(),
    }
    return kl_loss, metrics
