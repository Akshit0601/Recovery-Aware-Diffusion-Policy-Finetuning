"""
Unit and Integration Tests for Recovery-Aware DPPO Fine-Tuning Pipeline.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.dppo.critic import ValueCritic
from src.dppo.losses import (
    compute_actor_loss,
    compute_critic_loss,
    compute_kl_regularization,
)
from src.dppo.disturbances import DisturbanceManager, DisturbanceConfig, PHASE_APPROACHING, PHASE_PRE_GRASP
from src.dppo.buffer import DPPORolloutBuffer
from src.dppo.trainer import DPPOTrainer


def test_value_critic():
    """Verify ValueCritic handles 38-dim input and outputs scalar value."""
    critic = ValueCritic(input_dim=38, hidden_dims=(64, 64))
    x = torch.randn(8, 38)
    val = critic(x)
    assert val.shape == (8,)
    assert not torch.isnan(val).any()

    # 1D single state input
    x_single = torch.randn(38)
    val_single = critic(x_single)
    assert val_single.shape == ()


def test_actor_loss_unclipped_and_clipped():
    """Verify clipped PPO actor loss behaves correctly for unclipped and clipped ratios."""
    B, Tp, Da = 4, 16, 7
    mu_curr = torch.randn(B, Tp, Da)
    mu_old = mu_curr.clone()  # identical mean -> ratio should be 1.0
    a_prev = mu_curr + 0.1 * torch.randn(B, Tp, Da)
    sigma_k = torch.ones(B, 1, 1) * 0.1
    adv = torch.tensor([1.0, -1.0, 2.0, -0.5])

    loss, metrics = compute_actor_loss(mu_curr, mu_old, a_prev, sigma_k, adv, clip_eps=0.2)
    assert not torch.isnan(loss)
    assert abs(metrics["mean_ratio"] - 1.0) < 1e-4
    assert metrics["clip_fraction"] == 0.0

    # Perturb mu_curr drastically -> clipping should trigger
    mu_perturbed = mu_curr + 10.0
    loss_clipped, metrics_clipped = compute_actor_loss(mu_perturbed, mu_old, a_prev, sigma_k, adv, clip_eps=0.2)
    assert metrics_clipped["clip_fraction"] > 0.0


def test_critic_loss():
    """Verify MSE critic value loss calculation."""
    values = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 2.0, 3.0])
    loss_zero, _ = compute_critic_loss(values, targets)
    assert loss_zero.item() == 0.0

    targets_diff = torch.tensor([2.0, 3.0, 4.0])
    loss_one, _ = compute_critic_loss(values, targets_diff)
    assert pytest.approx(loss_one.item(), 1e-5) == 1.0


def test_kl_divergence_closed_form():
    """Verify closed-form analytical KL penalty against pretrained anchor."""
    B, Tp, Da = 4, 16, 7
    mu_ref = torch.randn(B, Tp, Da)
    mu_curr = mu_ref.clone()
    sigma_k = torch.ones(B, 1, 1) * 0.2

    # If mu_curr == mu_ref, KL must be exactly 0
    kl_zero, _ = compute_kl_regularization(mu_curr, mu_ref, sigma_k)
    assert kl_zero.item() == 0.0

    # If mu_curr deviates, KL must be strictly positive
    mu_deviant = mu_ref + 0.5
    kl_pos, _ = compute_kl_regularization(mu_deviant, mu_ref, sigma_k)
    assert kl_pos.item() > 0.0


def test_disturbance_manager_stratification():
    """Verify stratified episode planning matches 40% nominal, 60% disturbed and phase ratio."""
    cfg = DisturbanceConfig(
        nominal_fraction=0.4,
        disturbed_fraction=0.6,
        phase_ratio=(0.5, 0.5),
    )
    dm = DisturbanceManager(cfg)
    plan = dm.plan_iteration_episodes(n_episodes=20, mode="train")
    assert len(plan) == 20

    nominal = [p for p in plan if p is None]
    disturbed = [p for p in plan if p is not None]
    assert len(nominal) == 8  # 40% of 20
    assert len(disturbed) == 12  # 60% of 20

    approach = [p for p in disturbed if p["phase"] == PHASE_APPROACHING]
    pre_grasp = [p for p in disturbed if p["phase"] == PHASE_PRE_GRASP]
    assert len(approach) == 6
    assert len(pre_grasp) == 6


def test_rollout_buffer_gae():
    """Verify GAE and advantage normalization in DPPORolloutBuffer."""
    buffer = DPPORolloutBuffer(gamma=0.99, gae_lambda=0.95)
    buffer.start_episode(is_disturbed=False)

    obs_cond = torch.randn(38)
    dummy_denoising = [{
        "timestep": 99,
        "sample": torch.randn(1, 16, 7),
        "prev_sample": torch.randn(1, 16, 7),
        "mu": torch.randn(1, 16, 7),
        "sigma": torch.ones(1, 1, 1) * 0.1,
    }]

    # Step 1: r=0, V=0.5
    buffer.add_step(obs_cond, reward=0.0, value=0.5, done=False, denoising_transitions=dummy_denoising)
    # Step 2: r=1, V=0.8, done=True
    buffer.add_step(obs_cond, reward=1.0, value=0.8, done=True, denoising_transitions=dummy_denoising)
    buffer.finish_episode(success=True)

    flat_denoising, flat_env, stats = buffer.compute_gae_and_flatten()
    assert stats["total_episodes"] == 1
    assert stats["success_rate"] == 100.0
    assert len(flat_env) == 2
    assert len(flat_denoising) == 2

    # Check advantages were computed and normalized
    advs = [x["advantage"] for x in flat_denoising]
    assert not np.isnan(advs).any()


def test_dppo_trainer_integration_smoke():
    """Verify DPPOTrainer loads checkpoint, collects mini rollouts, and updates policy/critic."""
    ckpt_path = Path(__file__).resolve().parent.parent / "robomimic/diffusion_policy_trained_models/test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth"
    if not Path(ckpt_path).exists():
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    mini_cfg = {
        "rollout": {
            "n_episodes": 2,
            "horizon": 16,
            "nominal_fraction": 0.5,
            "disturbed_fraction": 0.5,
            "k_ft_mode": "tail",
            "k_ft_tail_m": 5,
        },
        "rl": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "kl_weight_beta": 0.01,
            "critic_weight_cv": 0.5,
            "epochs": 1,
            "batch_size": 16,
            "actor_lr": 1e-4,
            "critic_lr": 3e-4,
        },
        "weights": {
            "save_dir": "results/test_dppo_checkpoints"
        }
    }

    trainer = DPPOTrainer(
        checkpoint_path=str(ckpt_path),
        config=mini_cfg,
        device="cpu",
    )

    # Collect mini rollouts
    rollout_stats = trainer.collect_rollouts(n_episodes=2, mode="train")
    assert "nominal_success_rate" in rollout_stats

    # Execute mini update
    metrics = trainer.update_policy_and_critic()
    assert "actor_loss" in metrics
    assert "critic_loss" in metrics
    assert "kl_loss" in metrics

    # Verify theta_time (diffusion_step_encoder) remained frozen
    for p in trainer.noise_pred_net.diffusion_step_encoder.parameters():
        assert p.requires_grad is False
        assert p.grad is None
