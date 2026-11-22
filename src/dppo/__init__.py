"""
Recovery-Aware DPPO Fine-Tuning Package.
Implements Diffusion Policy Policy Optimization under manipulation disturbances
with KL divergence regularization to a pretrained anchor policy.
"""

from src.dppo.critic import ValueCritic
from src.dppo.sampler import DDPMSampler
from src.dppo.disturbances import DisturbanceManager
from src.dppo.losses import (
    compute_actor_loss,
    compute_critic_loss,
    compute_kl_regularization,
)
from src.dppo.buffer import DPPORolloutBuffer
from src.dppo.trainer import DPPOTrainer

__all__ = [
    "ValueCritic",
    "DDPMSampler",
    "DisturbanceManager",
    "compute_actor_loss",
    "compute_critic_loss",
    "compute_kl_regularization",
    "DPPORolloutBuffer",
    "DPPOTrainer",
]
