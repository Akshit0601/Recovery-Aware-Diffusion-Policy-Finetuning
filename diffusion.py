"""
Diffusion Policy implementation for Robomimic + MuJoCo robotic manipulation.

This script implements:
1. DDPM (Denoising Diffusion Probabilistic Model) with cosine beta schedule.
2. Residual MLP denoising backbone (matching standard Diffusion Policy / DPPO).
3. Public pretrained checkpoint loader (DPPO / Robomimic Lift benchmark).
4. Observation and action normalization / un-normalization.
5. Robomimic + MuJoCo environment runner with disturbance injection support.
"""

import os
import math
import json
import logging
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ==============================================================================
# 1. Neural Network Modules for Diffusion
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TwoLayerPreActivationResNetLinear(nn.Module):
    """Residual linear block with pre-activation Mish."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.l2(self.act(self.l1(self.act(x))))


class ResidualMLP(nn.Module):
    """Deep MLP with residual skip connections."""

    def __init__(self, dim_list: List[int]):
        super().__init__()
        hidden_dim = dim_list[1]
        self.layers = nn.ModuleList([nn.Linear(dim_list[0], hidden_dim)])
        for _ in range(1, len(dim_list) - 3, 2):
            self.layers.append(TwoLayerPreActivationResNetLinear(hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, dim_list[-1]))
        self.layers.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DiffusionMLP(nn.Module):
    """
    Diffusion Policy MLP denoising network.
    Predicts noise epsilon given noisy action chunk, diffusion step, and condition state.
    """

    def __init__(
        self,
        action_dim: int = 7,
        horizon_steps: int = 4,
        cond_dim: int = 19,
        time_dim: int = 16,
        mlp_dims: List[int] = [512, 512, 512],
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.cond_dim = cond_dim
        self.time_dim = time_dim

        output_dim = action_dim * horizon_steps
        input_dim = time_dim + output_dim + cond_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.mlp_mean = ResidualMLP([input_dim] + mlp_dims + [output_dim])

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy action chunks, shape (B, horizon_steps, action_dim)
            time: Diffusion timestep indices, shape (B,)
            cond: Dict containing 'state', shape (B, cond_dim)
        Returns:
            Noise prediction, shape (B, horizon_steps, action_dim)
        """
        B, Ta, Da = x.shape
        x_flat = x.view(B, -1)
        state = cond["state"].view(B, -1)
        t_emb = self.time_embedding(time.view(B, 1)).view(B, self.time_dim)
        inp = torch.cat([x_flat, t_emb, state], dim=-1)
        out = self.mlp_mean(inp)
        return out.view(B, Ta, Da)


# ==============================================================================
# 2. Diffusion Sampler (DDPM)
# ==============================================================================

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule for variance betas."""
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=torch.float32)


class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy for robot action generation via iterative denoising.
    """

    def __init__(
        self,
        network: nn.Module,
        denoising_steps: int = 20,
        horizon_steps: int = 4,
        action_dim: int = 7,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.network = network.to(device)
        self.denoising_steps = denoising_steps
        self.horizon_steps = horizon_steps
        self.action_dim = action_dim
        self.device = torch.device(device)

        # Precompute DDPM schedule constants
        betas = cosine_beta_schedule(denoising_steps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))
        self.register_buffer("ddpm_var", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("ddpm_logvar_clipped", torch.log(torch.clamp(self.ddpm_var, min=1e-20)))
        self.register_buffer("ddpm_mu_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("ddpm_mu_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    @torch.no_grad()
    def sample(self, cond_state: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Generate action trajectory chunk given condition observation state.
        Args:
            cond_state: normalized observation tensor, shape (B, cond_dim)
            deterministic: if True, do not add random noise at intermediate steps
        Returns:
            actions: normalized action trajectory chunk, shape (B, horizon_steps, action_dim)
        """
        B = cond_state.shape[0]
        cond = {"state": cond_state.to(self.device)}
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=self.device)

        for t in reversed(range(self.denoising_steps)):
            t_b = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_pred = self.network(x, t_b, cond)

            # Predict x_0
            x_recon = (
                self.sqrt_recip_alphas_cumprod[t] * x
                - self.sqrt_recipm1_alphas_cumprod[t] * eps_pred
            )
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

            # DDPM posterior mean
            mean = (
                self.ddpm_mu_coef1[t] * x_recon
                + self.ddpm_mu_coef2[t] * x
            )

            if deterministic or t == 0:
                x = mean
            else:
                logvar = self.ddpm_logvar_clipped[t]
                std = torch.exp(0.5 * logvar)
                noise = torch.randn_like(x).clamp(-3.0, 3.0)
                x = mean + std * noise

        return x


# ==============================================================================
# 3. Environment & Checkpoint Utilities
# ==============================================================================

class Normalizer:
    """Normalizes and unnormalizes observations and actions."""

    def __init__(self, normalization_path: str):
        data = np.load(normalization_path)
        self.obs_min = data["obs_min"]
        self.obs_max = data["obs_max"]
        self.action_min = data["action_min"]
        self.action_max = data["action_max"]

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return 2.0 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        return (action + 1.0) / 2.0 * (self.action_max - self.action_min) + self.action_min


def create_robomimic_lift_env(render: bool = False):
    """
    Instantiates the standard Robomimic Lift environment backed by robosuite and MuJoCo.
    """
    low_dim_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]
    ObsUtils.initialize_obs_modality_mapping_from_dict({"low_dim": low_dim_keys})

    env_meta = {
        "env_name": "Lift",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "has_renderer": render,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "robots": ["Panda"],
            "reward_shaping": True,
        },
    }

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=render,
        render_offscreen=False,
        use_image_obs=False,
    )
    return env, low_dim_keys


def load_pretrained_diffusion_policy(
    checkpoint_path: str,
    device: str = "cuda:0",
) -> DiffusionPolicy:
    """
    Loads pretrained Diffusion Policy weights into the model.
    """
    net = DiffusionMLP(action_dim=7, horizon_steps=4, cond_dim=19)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["ema"] if "ema" in ckpt else ckpt["model"]

    # Strip prefixes if present
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("network."):
            clean_sd[k[len("network."):]] = v
        elif not k.startswith("critic."):
            clean_sd[k] = v

    net.load_state_dict(clean_sd, strict=True)
    policy = DiffusionPolicy(net, denoising_steps=20, horizon_steps=4, action_dim=7, device=device)
    policy.eval()
    logger.info("Loaded pretrained Diffusion Policy from %s", checkpoint_path)
    return policy


# ==============================================================================
# 4. Rollout and Evaluation Runner
# ==============================================================================

def run_evaluation_episode(
    env,
    policy: DiffusionPolicy,
    normalizer: Normalizer,
    low_dim_keys: List[str],
    max_steps: int = 300,
    disturbance_step: Optional[int] = None,
    disturbance_offset: Optional[np.ndarray] = None,
    device: str = "cuda:0",
) -> Dict[str, any]:
    """
    Runs a single evaluation episode with optional mid-trajectory disturbance injection.

    Args:
        env: Robomimic environment (Robosuite + MuJoCo)
        policy: Pretrained DiffusionPolicy
        normalizer: Observation / Action normalizer
        low_dim_keys: Keys to extract from raw observation
        max_steps: Maximum environment steps
        disturbance_step: Step at which to perturb the object pose
        disturbance_offset: (dx, dy, dz) displacement applied to the cube
        device: Torch compute device
    """
    obs = env.reset()
    done = False
    success = False
    total_reward = 0.0
    step = 0

    while step < max_steps:
        # 1. Inject disturbance if requested (for recovery experiments)
        if disturbance_step is not None and step == disturbance_step and disturbance_offset is not None:
            logger.info("Injecting disturbance at step %d: offset=%s", step, disturbance_offset)
            cube_pos = env.env.sim.data.qpos[:3].copy()
            cube_pos += disturbance_offset
            env.env.sim.data.qpos[:3] = cube_pos
            env.env.sim.forward()
            obs = env.get_observation()

        # 2. Extract and normalize observation
        raw_obs = np.concatenate([obs[k] for k in low_dim_keys], axis=0)
        norm_obs = normalizer.normalize_obs(raw_obs)
        obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # 3. Generate action chunk from diffusion policy
        action_chunk = policy.sample(obs_tensor, deterministic=True).squeeze(0).cpu().numpy()
        action_chunk = normalizer.unnormalize_action(action_chunk)

        # 4. Step environment with the action chunk
        for a in action_chunk:
            obs, reward, done, info = env.step(a)
            total_reward += reward
            step += 1

            if env.is_success()["task"]:
                success = True
                break

        if success or done:
            break

    return {
        "steps": step,
        "success": success,
        "total_reward": total_reward,
    }


def main():
    print("=" * 65)
    print("Pre-trained Diffusion Policy on Robomimic + MuJoCo (Lift Task)")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_path = "/home/akshit/ece285_proj/pretrained/lift/state_8000.pt"
    norm_path = "/home/akshit/ece285_proj/pretrained/lift/normalization.npz"

    if not os.path.exists(ckpt_path) or not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint or normalization file missing. Expected:\n"
            f"  Checkpoint: {ckpt_path}\n"
            f"  Normalization: {norm_path}"
        )

    # 1. Load Policy and Normalizer
    normalizer = Normalizer(norm_path)
    policy = load_pretrained_diffusion_policy(ckpt_path, device=device)

    # 2. Initialize Robomimic + MuJoCo environment
    logger.info("Creating Robomimic environment (Panda robot on MuJoCo)...")
    env, low_dim_keys = create_robomimic_lift_env(render=True)

    # 3. Run Nominal Rollout
    logger.info("Running nominal evaluation episode...")
    nominal_result = run_evaluation_episode(
        env=env,
        policy=policy,
        normalizer=normalizer,
        low_dim_keys=low_dim_keys,
        max_steps=200,
        device=device,
    )
    print("\n--- Nominal Episode Result ---")
    print(f"  Steps taken : {nominal_result['steps']}")
    print(f"  Total reward: {nominal_result['total_reward']:.4f}")
    print(f"  Task success: {nominal_result['success']}")

    # 4. Run Disturbance (Recovery) Rollout
    logger.info("\nRunning recovery evaluation episode (with disturbance injection)...")
    disturbed_result = run_evaluation_episode(
        env=env,
        policy=policy,
        normalizer=normalizer,
        low_dim_keys=low_dim_keys,
        max_steps=200,
        disturbance_step=30,
        disturbance_offset=np.array([0.05, -0.05, 0.0]),
        device=device,
    )
    print("\n--- Disturbed Episode Result ---")
    print(f"  Steps taken : {disturbed_result['steps']}")
    print(f"  Total reward: {disturbed_result['total_reward']:.4f}")
    print(f"  Task success: {disturbed_result['success']}")

    print("\n" + "=" * 65)
    print("✅ Pre-trained Diffusion Policy executed cleanly on Robomimic + MuJoCo!")
    print("=" * 65)


if __name__ == "__main__":
    main()
