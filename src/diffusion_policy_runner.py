#!/usr/bin/env python
"""
Base Runner for Diffusion Policy Inference and Evaluation.

Provides DiffusionPolicyRunner, a modular, extensible object-oriented base class
for loading, configuring, running, and evaluating trained diffusion policies
on Robomimic and Robosuite environments under various configurations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import h5py
import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

DEFAULT_CONFIG_PATH = "config/run_diffusion_policy.yaml"

# Execution Phase Constants
PHASE_APPROACHING: str = "approaching"
PHASE_PRE_GRASP: str = "pre-grasp"
PHASE_GRASPED: str = "grasped"
ALL_PHASES: List[str] = [PHASE_APPROACHING, PHASE_PRE_GRASP, PHASE_GRASPED]

# Configure logging so logging.info calls are visible when this class/module is imported
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    force=True,
)
logging.getLogger("robosuite_logs").propagate = False
logger = logging.getLogger(__name__)

def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update target dictionary with key-values from source dictionary.
    """
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


class DiffusionPolicyRunner:
    """
    Modular Base Runner for Diffusion Policy inference and evaluation.

    This class encapsulates environment creation, checkpoint loading, camera/scene
    configuration, rollout execution, rendering/video logging, and HDF5 recording.

    Subclasses can easily override lifecycle hooks such as:
      - `setup_env()`: Customize or wrap the environment.
      - `get_action()`: Inject perturbations, noise, recovery logic, or safety guards.
      - `step_env()`: Intercept transitions or handle custom simulation dynamics.
      - `on_step_end()`: Monitor intermediate step observations or metrics.
      - `compute_summary()`: Add custom metrics to evaluation statistics.

    Example direct usage:
        runner = DiffusionPolicyRunner(config_path="config/run_diffusion_policy.yaml")
        results = runner.run()

    Example with programmatic overrides:
        runner = DiffusionPolicyRunner(
            config_path="config/run_diffusion_policy.yaml",
            overrides={"rollout": {"n_rollouts": 5}, "rendering": {"render": False}}
        )
        results = runner.run()

    Example with pre-existing policy and environment:
        runner = DiffusionPolicyRunner(policy=my_policy, env=my_env)
        stats, traj = runner.rollout_episode(horizon=400)

    Example subclassing:
        class PerturbedRunner(DiffusionPolicyRunner):
            def get_action(self, obs, step_i=0, policy=None):
                action = super().get_action(obs, step_i=step_i, policy=policy)
                # Apply noise or perturbation during specific steps
                if 20 <= step_i <= 30:
                    action += np.random.normal(0, 0.1, size=action.shape)
                return action
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        policy: Optional[RolloutPolicy] = None,
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
    ) -> None:
        """
        Initialize the runner.

        Args:
            config: Optional config dictionary. If not provided, config is loaded from `config_path`.
            config_path: Path to YAML config file. Defaults to DEFAULT_CONFIG_PATH if config is None.
            overrides: Optional dictionary of configuration overrides to merge recursively.
            policy: Optional pre-loaded RolloutPolicy instance.
            env: Optional pre-constructed EnvBase/EnvWrapper environment instance.
        """
        if config is not None:
            self.cfg: Dict[str, Any] = deepcopy(config)
            self.config_path: Optional[str] = str(config_path) if config_path else None
        elif config_path is not None:
            self.config_path = str(config_path)
            self.cfg = self.load_config_file(self.config_path)
        elif policy is not None or env is not None:
            # Standalone policy/env mode without requiring a YAML checkpoint file
            self.config_path = None
            self.cfg = {}
        else:
            self.config_path = DEFAULT_CONFIG_PATH
            self.cfg = self.load_config_file(self.config_path)

        if overrides:
            deep_update(self.cfg, overrides)

        # Runtime state attributes
        self.device: Optional[torch.device] = None
        self.policy: Optional[RolloutPolicy] = policy
        self.ckpt_dict: Optional[Dict[str, Any]] = None
        self.env: Optional[Union[EnvBase, EnvWrapper]] = env
        self.rollout_horizon: Optional[int] = None

        # Camera & Rendering attributes
        self.camera_names: List[str] = []
        self.video_writer: Optional[imageio.core.format.Writer] = None

        # Dataset logging attributes
        self.data_writer: Optional[h5py.File] = None
        self.data_grp: Optional[h5py.Group] = None
        self.total_samples: int = 0

        # Phase tracking configuration & state
        self.pre_grasp_arm_threshold: float = float(
            self.rollout_cfg.get("pre_grasp_arm_threshold", 0.4)
        )
        self.pre_grasp_gripper_threshold: float = float(
            self.rollout_cfg.get("pre_grasp_gripper_threshold", 0.0)
        )
        self.sticky_phases: bool = bool(
            self.rollout_cfg.get("sticky_phases", True)
        )
        self.current_phase: str = PHASE_APPROACHING

        # Setup flag: ready immediately if policy and env were provided
        self.is_setup: bool = (policy is not None and env is not None)

    # -------------------------------------------------------------------------
    # Configuration Helpers
    # -------------------------------------------------------------------------

    @classmethod
    def resolve_config_path(cls, config_path: Union[str, Path]) -> str:
        """Resolve a configuration path, checking current directory and repo root."""
        path_str = str(config_path)
        if os.path.exists(path_str):
            return os.path.abspath(path_str)

        # Try relative to repo root (one level up from src/)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback = os.path.join(repo_root, path_str)
        if os.path.exists(fallback):
            return os.path.abspath(fallback)

        raise FileNotFoundError(f"Configuration file not found at '{config_path}' or '{fallback}'")

    @classmethod
    def load_config_file(cls, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse a YAML configuration file."""
        resolved = cls.resolve_config_path(config_path)
        with open(resolved, "r") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config format in '{resolved}'. Expected a dictionary.")
        return cfg

    def update_config(self, overrides: Dict[str, Any]) -> None:
        """Recursively update current configuration with overrides."""
        deep_update(self.cfg, overrides)

    # Config section accessors
    @property
    def weights_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("weights", {})

    @property
    def rollout_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("rollout", {})

    @property
    def env_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("env", {})

    @property
    def rendering_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("rendering", {})

    @property
    def camera_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("camera", {})

    @property
    def object_placement_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("object_placement", {})

    @property
    def dataset_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("dataset", {})

    @property
    def system_cfg(self) -> Dict[str, Any]:
        return self.cfg.get("system", {})

    # -------------------------------------------------------------------------
    # Setup Lifecycle Hooks
    # -------------------------------------------------------------------------

    def setup(self, force: bool = False) -> DiffusionPolicyRunner:
        """
        Execute full setup lifecycle: device, seed, policy, horizon, env, camera, and writers.

        Args:
            force: If True, forces re-initialization even if already setup.

        Returns:
            self for fluent method chaining.
        """
        if self.is_setup and not force:
            return self

        self.setup_device()
        self.setup_seed()
        if self.policy is None or force:
            self.load_policy(force=force)
        self.setup_horizon()
        if self.env is None or force:
            self.setup_env(force=force)
        self.setup_object_placement()
        self.setup_camera()
        self.setup_writers()

        self.is_setup = True
        return self

    def setup_device(self) -> torch.device:
        """Determine and configure torch compute device (CUDA / CPU)."""
        cpu = self.system_cfg.get("cpu", False)
        self.device = TorchUtils.get_torch_device(try_to_use_cuda=(not cpu))
        return self.device

    def setup_seed(self) -> Optional[int]:
        """Configure random seeds for NumPy and PyTorch reproducibility."""
        seed = self.rollout_cfg.get("seed", 0)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.system_cfg.get("verbose", False):
                print(f"Random seed set to: {seed}")
        return seed

    def load_policy(self, force: bool = False) -> Tuple[RolloutPolicy, Dict[str, Any]]:
        """
        Load the trained Diffusion Policy and checkpoint dictionary.

        Args:
            force: If True, reloads even if policy is already present.
        """
        if self.policy is not None and not force:
            return self.policy, self.ckpt_dict or {}

        ckpt_path = self.weights_cfg.get("checkpoint_path") or self.cfg.get("agent")
        if not ckpt_path:
            raise ValueError("Missing 'checkpoint_path' under 'weights' in configuration.")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_path}")

        verbose = self.system_cfg.get("verbose", False)
        print(f"\n{'='*70}")
        print("Loading Diffusion Policy Checkpoint:")
        print(f"  {ckpt_path}")
        print(f"{'='*70}")

        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=self.device, verbose=verbose
        )
        self.policy = policy
        self.ckpt_dict = ckpt_dict
        return policy, ckpt_dict

    def setup_horizon(self) -> int:
        """Determine rollout horizon from config override or checkpoint configuration."""
        horizon_override = self.rollout_cfg.get("horizon", None)
        if horizon_override is not None:
            self.rollout_horizon = int(horizon_override)
            print(f"Rollout horizon (override): {self.rollout_horizon}")
        elif self.ckpt_dict is not None:
            ckpt_config, _ = FileUtils.config_from_checkpoint(ckpt_dict=self.ckpt_dict)
            self.rollout_horizon = int(ckpt_config.experiment.rollout.horizon)
            print(f"Rollout horizon (from checkpoint config): {self.rollout_horizon}")
        else:
            self.rollout_horizon = 400
        return self.rollout_horizon

    def setup_env(self, force: bool = False) -> Union[EnvBase, EnvWrapper]:
        """
        Build and configure the evaluation environment from saved checkpoint.

        Args:
            force: If True, recreates environment even if already initialized.
        """
        if self.env is not None and not force:
            return self.env

        if self.ckpt_dict is None:
            raise RuntimeError("Cannot build environment: checkpoint dictionary is not loaded.")

        env_name = self.env_cfg.get("env_name", None)
        render = self.rendering_cfg.get("render", False)
        video_path = self.rendering_cfg.get("video_path", None)
        write_video = video_path is not None
        verbose = self.system_cfg.get("verbose", False)

        if render and write_video:
            raise ValueError("Cannot render on-screen and record video at the same time.")

        env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=self.ckpt_dict,
            env_name=env_name,
            render=render,
            render_offscreen=write_video,
            verbose=verbose,
        )
        self.env = env
        return self.env

    def setup_object_placement(self) -> None:
        """Configure object placement sampler ranges and reference positions."""
        if self.env is None:
            return

        obj_cfg = self.object_placement_cfg
        object_pos = obj_cfg.get("object_pos", None)
        object_x_range = obj_cfg.get("object_x_range", None)
        object_y_range = obj_cfg.get("object_y_range", None)
        object_ref_pos = obj_cfg.get("object_ref_pos", None)

        robosuite_env = (
            self.env.env.env
            if hasattr(self.env, "env") and hasattr(self.env.env, "env")
            else None
        )
        if robosuite_env is not None and hasattr(robosuite_env, "placement_initializer"):
            sampler = robosuite_env.placement_initializer

            if object_pos is not None:
                sampler.x_range = [object_pos[0], object_pos[0]]
                sampler.y_range = [object_pos[1], object_pos[1]]
                print(f"Set fixed object position: X={object_pos[0]}, Y={object_pos[1]}")
            else:
                if object_x_range is not None:
                    sampler.x_range = list(object_x_range)
                    print(f"Set object X sampling range: {sampler.x_range}")
                if object_y_range is not None:
                    sampler.y_range = list(object_y_range)
                    print(f"Set object Y sampling range: {sampler.y_range}")

            if object_ref_pos is not None:
                sampler.reference_pos = np.array(object_ref_pos, dtype=float)
                print(f"Set object reference position: {sampler.reference_pos}")
        else:
            if object_pos or object_x_range or object_y_range:
                print("Warning: placement_initializer not found on environment. Skipping object range modification.")

    def setup_camera(self) -> None:
        """Configure camera views, field of view (FOV), and position."""
        camera_names = self.camera_cfg.get("camera_names", ["agentview"])
        if isinstance(camera_names, str):
            camera_names = [camera_names]
        self.camera_names = list(camera_names)

        camera_dist_scale = self.camera_cfg.get("camera_dist_scale", None)
        camera_fovy = self.camera_cfg.get("camera_fovy", None)
        camera_pos = self.camera_cfg.get("camera_pos", None)

        robosuite_env = (
            self.env.env.env
            if hasattr(self.env, "env") and hasattr(self.env.env, "env")
            else None
        )
        if robosuite_env is not None and hasattr(robosuite_env, "sim"):
            sim = robosuite_env.sim
            for cam_name in self.camera_names:
                try:
                    cam_id = sim.model.camera_name2id(cam_name)
                except Exception:
                    continue

                if camera_fovy is not None:
                    sim.model.cam_fovy[cam_id] = float(camera_fovy)
                    print(f"Adjusted camera '{cam_name}' FOV (fovy) to: {camera_fovy}°")

                if camera_pos is not None:
                    sim.model.cam_pos[cam_id] = np.array(camera_pos, dtype=float)
                    print(f"Adjusted camera '{cam_name}' position to: {sim.model.cam_pos[cam_id]}")
                elif camera_dist_scale is not None and camera_dist_scale != 1.0:
                    target = np.array([0.0, 0.0, 0.8])
                    current_pos = np.array(sim.model.cam_pos[cam_id], dtype=float)
                    new_pos = target + (current_pos - target) * float(camera_dist_scale)
                    sim.model.cam_pos[cam_id] = new_pos
                    print(
                        f"Pushed camera '{cam_name}' {camera_dist_scale}x farther away: "
                        f"pos {current_pos.round(3)} -> {new_pos.round(3)}"
                    )

    def setup_writers(self) -> None:
        """Initialize video and HDF5 dataset recording writers if configured."""
        # Video writer
        video_path = self.rendering_cfg.get("video_path", None)
        if video_path is not None:
            video_dir = os.path.dirname(video_path)
            if video_dir:
                os.makedirs(video_dir, exist_ok=True)
            self.video_writer = imageio.get_writer(video_path, fps=20)
            print(f"Recording video to: {video_path}")
        else:
            self.video_writer = None

        # HDF5 dataset writer
        dataset_path = self.dataset_cfg.get("dataset_path", None)
        if dataset_path is not None:
            data_dir = os.path.dirname(dataset_path)
            if data_dir:
                os.makedirs(data_dir, exist_ok=True)
            self.data_writer = h5py.File(dataset_path, "w")
            self.data_grp = self.data_writer.create_group("data")
            self.total_samples = 0
            print(f"Saving rollout trajectories to: {dataset_path}")
        else:
            self.data_writer = None
            self.data_grp = None
            self.total_samples = 0

    # -------------------------------------------------------------------------
    # Rollout & Inference Lifecycle Hooks
    # -------------------------------------------------------------------------

    def reset_env(
        self,
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
        policy: Optional[RolloutPolicy] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment and policy for a new episode.

        Returns:
            Tuple of (initial_obs, state_dict)
        """
        active_env = env if env is not None else self.env
        active_policy = policy if policy is not None else self.policy
        assert isinstance(active_env, (EnvBase, EnvWrapper)), "Environment not initialized."
        assert isinstance(active_policy, RolloutPolicy), "Policy not initialized."

        active_policy.start_episode()
        obs = active_env.reset()
        state_dict = active_env.get_state() if hasattr(active_env, "get_state") else {}

        # Deterministic action playback reset for robosuite environments
        if hasattr(active_env, "reset_to") and state_dict:
            obs = active_env.reset_to(state_dict)
        return obs, state_dict

    def get_action(
        self,
        obs: Dict[str, Any],
        step_i: int = 0,
        policy: Optional[RolloutPolicy] = None,
    ) -> np.ndarray:
        """
        Query an action from the policy for the given observation.

        Subclasses can override this method to inject noise, action perturbations,
        safety filters, or latency.
        """
        active_policy = policy if policy is not None else self.policy
        assert active_policy is not None, "Policy not initialized."
        return active_policy(ob=obs)

    def step_env(
        self,
        action: np.ndarray,
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Step the environment forward with the chosen action."""
        active_env = env if env is not None else self.env
        assert active_env is not None, "Environment not initialized."
        return active_env.step(action)

    def check_success(
        self,
        obs: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
    ) -> bool:
        """Evaluate task success from environment state."""
        active_env = env if env is not None else self.env
        if active_env is not None and hasattr(active_env, "is_success"):
            res = active_env.is_success()
            if isinstance(res, dict):
                return bool(res.get("task", False))
            return bool(res)
        return False

    def should_terminate(
        self, done: bool, success: bool, step_i: int
    ) -> bool:
        """Determine whether the current episode rollout should terminate early."""
        return done or success

    def on_step_render(
        self,
        step_i: int,
        video_count: int,
        video_skip: int = 5,
        render: bool = False,
        video_writer: Optional[imageio.core.format.Writer] = None,
        camera_names: Optional[List[str]] = None,
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
    ) -> int:
        """Handle on-screen rendering or video frame capture for the current step."""
        active_env = env if env is not None else self.env
        cams = camera_names if camera_names is not None else (self.camera_names if self.camera_names else ["agentview"])

        if render and active_env is not None:
            active_env.render(mode="human", camera_name=cams[0])

        if video_writer is not None and active_env is not None:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in cams:
                    video_img.append(
                        active_env.render(
                            mode="rgb_array", height=512, width=512, camera_name=cam_name
                        )
                    )
                concatenated = np.concatenate(video_img, axis=1)
                video_writer.append_data(concatenated)
            video_count += 1

        return video_count

    def on_episode_start(self, episode_idx: int) -> None:
        """Hook called immediately before starting a rollout episode."""
        pass

    def is_pre_grasp_action(
        self,
        action: np.ndarray,
        arm_threshold: Optional[float] = None,
        gripper_threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if the action corresponds to the pre-grasp phase (policy outputs only gripper actions).

        An action is considered pre-grasp when the arm movement delta is below the stationary
        threshold and the gripper command is active (closing).
        """
        act = np.asarray(action).ravel()
        if len(act) < 2:
            return False
        arm_thresh = arm_threshold if arm_threshold is not None else self.pre_grasp_arm_threshold
        grip_thresh = gripper_threshold if gripper_threshold is not None else self.pre_grasp_gripper_threshold

        arm_action = act[:-1]
        gripper_action = act[-1]
        arm_norm = float(np.linalg.norm(arm_action))

        return bool(gripper_action > grip_thresh and arm_norm <= arm_thresh)

    def detect_phase(
        self,
        action: np.ndarray,
        reward: float = 0.0,
        current_phase: Optional[str] = None,
        obs: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        step_i: int = 0,
    ) -> str:
        """
        Detect the operational execution phase for the current step.

        Phases:
          - 'approaching': Running policy towards the object (arm moving, gripper open).
          - 'pre-grasp': Policy outputs only gripper actions (gripper closing, arm stationary).
          - 'grasped': Reward is 1.0 (object successfully grasped and lifted).

        Args:
            action: Executed or outputted policy action array.
            reward: Step reward returned by the environment.
            current_phase: Phase of the previous step.
            obs: Environment observation dictionary.
            info: Environment info dictionary.
            step_i: Step index in the rollout.

        Returns:
            One of PHASE_APPROACHING, PHASE_PRE_GRASP, or PHASE_GRASPED.
        """
        if reward >= 1.0:
            return PHASE_GRASPED

        if current_phase == PHASE_GRASPED:
            return PHASE_GRASPED

        if self.is_pre_grasp_action(action):
            return PHASE_PRE_GRASP

        if current_phase == PHASE_PRE_GRASP and self.sticky_phases:
            return PHASE_PRE_GRASP

        return PHASE_APPROACHING

    def on_phase_transition(
        self,
        prev_phase: str,
        new_phase: str,
        step_i: int,
        action: np.ndarray,
        reward: float,
    ) -> None:
        """
        Hook called when the rollout transitions from one execution phase to another.

        Subclasses can override this to implement custom logging, perturbation injection,
        or telemetry triggers.
        """
        pass

    def on_step_end(
        self,
        step_i: int,
        obs: Dict[str, Any],
        action: np.ndarray,
        next_obs: Dict[str, Any],
        reward: float,
        done: bool,
        success: bool,
        info: Dict[str, Any],
    ) -> None:
        """Hook called after each environment step."""
        pass

    def on_episode_end(
        self, episode_idx: int, stats: Dict[str, Any], traj: Dict[str, Any]
    ) -> None:
        """Hook called immediately after an episode completes."""
        pass

    def before_rollouts(self) -> None:
        """Hook called before the entire rollout evaluation loop begins."""
        pass

    def after_rollouts(self, rollout_stats: List[Dict[str, Any]]) -> None:
        """Hook called after all rollout episodes have completed."""
        pass

    # -------------------------------------------------------------------------
    # Episode Rollout Execution
    # -------------------------------------------------------------------------

    def rollout_episode(
        self,
        episode_idx: int = 0,
        horizon: Optional[int] = None,
        return_obs: Optional[bool] = None,
        policy: Optional[RolloutPolicy] = None,
        env: Optional[Union[EnvBase, EnvWrapper]] = None,
        render: Optional[bool] = None,
        video_writer: Optional[imageio.core.format.Writer] = None,
        video_skip: Optional[int] = None,
        camera_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a single episode rollout using the loaded diffusion policy.

        Fully covers all capabilities of the legacy rollout function while supporting
        dynamic argument overrides and object-oriented hooks.

        Args:
            episode_idx: Integer index of current episode.
            horizon: Step horizon override (defaults to self.rollout_horizon).
            return_obs: Whether to return observation arrays in traj dict.
            policy: Policy override (defaults to self.policy).
            env: Environment override (defaults to self.env).
            render: Render flag override (defaults to config setting).
            video_writer: Writer instance override (defaults to self.video_writer).
            video_skip: Frame interval override (defaults to config setting).
            camera_names: Camera name(s) override (defaults to self.camera_names).

        Returns:
            Tuple of (stats_dict, trajectory_dict)
        """
        active_policy = policy if policy is not None else self.policy
        active_env = env if env is not None else self.env

        if active_policy is None or active_env is None:
            if not self.is_setup:
                self.setup()
            active_policy = self.policy
            active_env = self.env

        assert isinstance(active_env, (EnvBase, EnvWrapper)), "Environment not initialized."
        assert isinstance(active_policy, RolloutPolicy), "Policy not initialized."

        active_horizon = horizon if horizon is not None else (self.rollout_horizon or 400)
        active_render = render if render is not None else self.rendering_cfg.get("render", False)
        active_video_writer = video_writer if video_writer is not None else self.video_writer
        active_video_skip = video_skip if video_skip is not None else self.rendering_cfg.get("video_skip", 5)
        active_camera_names = (
            camera_names
            if camera_names is not None
            else (self.camera_names if self.camera_names else ["agentview"])
        )

        assert not (active_render and (active_video_writer is not None)), (
            "Cannot render on-screen and record video at the same time."
        )

        write_dataset = self.data_writer is not None
        dataset_obs = self.dataset_cfg.get("dataset_obs", False)
        if return_obs is None:
            return_obs = write_dataset and dataset_obs

        self.on_episode_start(episode_idx)
        obs, state_dict = self.reset_env(env=active_env, policy=active_policy)

        video_count = 0
        total_reward = 0.0
        success = False
        step_i = 0
        current_phase = PHASE_APPROACHING
        self.current_phase = current_phase
        pre_grasp_step: Optional[int] = None
        grasped_step: Optional[int] = None

        traj: Dict[str, Any] = dict(
            actions=[],
            rewards=[],
            dones=[],
            states=[],
            phases=[],
            initial_state_dict=state_dict,
        )
        if return_obs:
            traj.update(dict(obs=[], next_obs=[]))

        exceptions_to_catch = (
            active_env.rollout_exceptions
            if hasattr(active_env, "rollout_exceptions")
            else Exception
        )

        try:
            for step_i in range(active_horizon):
                # Query action
                act = self.get_action(obs=obs, step_i=step_i, policy=active_policy)

                # Step environment
                next_obs, r, done, info = self.step_env(act, env=active_env)
                logging.info(f"the obs looks  like {next_obs}")
                total_reward += float(r)
                success = self.check_success(
                    obs=next_obs, reward=r, done=done, info=info, env=active_env
                )

                # Phase detection & transition
                new_phase = self.detect_phase(
                    action=act,
                    reward=float(r),
                    current_phase=current_phase,
                    obs=next_obs,
                    info=info,
                    step_i=step_i,
                )
                logging.info(f"the current phase for the arm is {new_phase}")
                if new_phase != current_phase:
                    self.on_phase_transition(
                        prev_phase=current_phase,
                        new_phase=new_phase,
                        step_i=step_i,
                        action=act,
                        reward=float(r),
                    )
                    if new_phase == PHASE_PRE_GRASP and pre_grasp_step is None:
                        pre_grasp_step = step_i
                    elif new_phase == PHASE_GRASPED and grasped_step is None:
                        grasped_step = step_i
                    current_phase = new_phase
                    self.current_phase = current_phase

                # Handle rendering & video recording
                video_count = self.on_step_render(
                    step_i=step_i,
                    video_count=video_count,
                    video_skip=active_video_skip,
                    render=active_render,
                    video_writer=active_video_writer,
                    camera_names=active_camera_names,
                    env=active_env,
                )

                # Collect transition data
                traj["actions"].append(act)
                traj["rewards"].append(r)
                traj["dones"].append(done)
                traj["states"].append(state_dict.get("states", []))
                traj["phases"].append(current_phase)
                if return_obs:
                    traj["obs"].append(obs)
                    traj["next_obs"].append(next_obs)

                self.on_step_end(
                    step_i=step_i,
                    obs=obs,
                    action=act,
                    next_obs=next_obs,
                    reward=r,
                    done=done,
                    success=success,
                    info=info,
                )

                # Termination check
                if self.should_terminate(done=done, success=success, step_i=step_i):
                    break

                obs = deepcopy(next_obs)
                state_dict = active_env.get_state() if hasattr(active_env, "get_state") else {}

        except exceptions_to_catch as e:
            print(f"WARNING: rollout exception occurred at episode {episode_idx}: {e}")

        stats = dict(
            Return=round(total_reward, 3),
            Horizon=(step_i + 1),
            Success_Rate=float(success),
            Final_Phase=current_phase,
            Reached_PreGrasp=bool(pre_grasp_step is not None),
            Reached_Grasped=bool(grasped_step is not None),
            PreGrasp_Step=pre_grasp_step,
            Grasped_Step=grasped_step,
        )

        if return_obs:
            traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
            traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

        for k in traj:
            if k == "initial_state_dict":
                continue
            if isinstance(traj[k], dict):
                for kp in traj[k]:
                    traj[k][kp] = np.array(traj[k][kp])
            else:
                traj[k] = np.array(traj[k])

        self.on_episode_end(episode_idx, stats, traj)
        return stats, traj

    @classmethod
    def rollout(
        cls,
        policy: RolloutPolicy,
        env: Union[EnvBase, EnvWrapper],
        horizon: int,
        render: bool = False,
        video_writer: Optional[imageio.core.format.Writer] = None,
        video_skip: int = 5,
        return_obs: bool = False,
        camera_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Execute a single episode rollout using policy and env directly.

        Convenience classmethod delegating directly to rollout_episode on a runner instance,
        avoiding code duplication while providing 100% backwards compatibility.
        """
        runner = cls(policy=policy, env=env)
        return runner.rollout_episode(
            horizon=horizon,
            render=render,
            video_writer=video_writer,
            video_skip=video_skip,
            return_obs=return_obs,
            camera_names=camera_names,
        )

    def record_episode_data(self, episode_idx: int, traj: Dict[str, Any]) -> None:
        """Write an episode trajectory to the HDF5 dataset writer."""
        if self.data_grp is None:
            return

        dataset_obs = self.dataset_cfg.get("dataset_obs", False)
        ep_data_grp = self.data_grp.create_group(f"demo_{episode_idx}")
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

        if dataset_obs and "obs" in traj:
            for k in traj["obs"]:
                ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

        if "phases" in traj and self.data_grp is not None:
            phase_strings = [str(p) for p in traj["phases"]]
            ep_data_grp.create_dataset(
                "phases", data=np.array(phase_strings, dtype=h5py.string_dtype())
            )

        if "model" in traj["initial_state_dict"]:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]

        num_samples = traj["actions"].shape[0]
        ep_data_grp.attrs["num_samples"] = num_samples
        self.total_samples += num_samples

    # -------------------------------------------------------------------------
    # Evaluation Summary & Teardown
    # -------------------------------------------------------------------------

    def compute_summary(self, rollout_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated statistics from rollout results."""
        if not rollout_stats:
            return dict(
                total_rollouts=0,
                success_count=0,
                success_rate=0.0,
                mean_return=0.0,
                mean_horizon=0.0,
                pre_grasp_rate=0.0,
                grasped_rate=0.0,
                mean_pre_grasp_step=None,
                mean_grasped_step=None,
                final_phase_counts={p: 0 for p in ALL_PHASES},
            )

        stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
        successes = int(np.sum(stats_dict["Success_Rate"]))
        total = len(rollout_stats)
        success_rate = (successes / total) * 100.0
        mean_return = float(np.mean(stats_dict["Return"]))
        mean_horizon = float(np.mean(stats_dict["Horizon"]))

        pre_grasp_reached = sum(1 for s in rollout_stats if s.get("Reached_PreGrasp", False))
        grasped_reached = sum(1 for s in rollout_stats if s.get("Reached_Grasped", False))
        pre_grasp_steps = [
            s["PreGrasp_Step"] for s in rollout_stats if s.get("PreGrasp_Step") is not None
        ]
        grasped_steps = [
            s["Grasped_Step"] for s in rollout_stats if s.get("Grasped_Step") is not None
        ]
        final_phases = [s.get("Final_Phase", PHASE_APPROACHING) for s in rollout_stats]
        phase_counts = {p: final_phases.count(p) for p in ALL_PHASES}

        return dict(
            total_rollouts=total,
            success_count=successes,
            success_rate=success_rate,
            mean_return=mean_return,
            mean_horizon=mean_horizon,
            pre_grasp_rate=(pre_grasp_reached / total) * 100.0,
            grasped_rate=(grasped_reached / total) * 100.0,
            mean_pre_grasp_step=float(np.mean(pre_grasp_steps)) if pre_grasp_steps else None,
            mean_grasped_step=float(np.mean(grasped_steps)) if grasped_steps else None,
            final_phase_counts=phase_counts,
        )

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted evaluation summary to stdout."""
        print(f"\n{'='*70}")
        print("                  EVALUATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total Rollouts    : {summary['total_rollouts']}")
        print(f"Success Count     : {summary['success_count']} / {summary['total_rollouts']}")
        print(f"Success Rate      : {summary['success_rate']:.2f}%")
        print(f"Pre-Grasp Rate    : {summary.get('pre_grasp_rate', 0.0):.2f}%")
        print(f"Grasped Rate      : {summary.get('grasped_rate', 0.0):.2f}%")
        print(f"Mean Return       : {summary['mean_return']:.3f}")
        print(f"Mean Horizon (len): {summary['mean_horizon']:.2f} steps")
        if summary.get("mean_pre_grasp_step") is not None:
            print(f"Mean Pre-Grasp Stp: {summary['mean_pre_grasp_step']:.2f}")
        if summary.get("mean_grasped_step") is not None:
            print(f"Mean Grasped Step : {summary['mean_grasped_step']:.2f}")
        print(f"{'='*70}\n")

    def close(self) -> None:
        """Close video writers, HDF5 datasets, and any open resources."""
        if self.video_writer is not None:
            self.video_writer.close()
            video_path = self.rendering_cfg.get("video_path", "")
            print(f"\nVideo saved to: {video_path}")
            self.video_writer = None

        if self.data_writer is not None:
            if self.data_grp is not None:
                self.data_grp.attrs["total"] = self.total_samples
                if self.env is not None and hasattr(self.env, "serialize"):
                    self.data_grp.attrs["env_args"] = json.dumps(self.env.serialize(), indent=4)
            dataset_path = self.dataset_cfg.get("dataset_path", "")
            self.data_writer.close()
            print(f"\nDataset saved to: {dataset_path}")
            self.data_writer = None
            self.data_grp = None

    def __enter__(self) -> DiffusionPolicyRunner:
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # -------------------------------------------------------------------------
    # Main Execution Loop
    # -------------------------------------------------------------------------

    def run(
        self,
        n_rollouts: Optional[int] = None,
        collect_trajectories: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute the full evaluation pipeline across all rollout episodes.

        Args:
            n_rollouts: Number of episodes to run (overrides config if provided).
            collect_trajectories: If True, returns in-memory trajectories for all episodes.

        Returns:
            Dict containing:
              - "summary": Aggregated summary metrics.
              - "rollout_stats": List of per-episode statistics dictionaries.
              - "trajectories": List of episode trajectory dictionaries (if collect_trajectories=True).
        """
        self.setup()

        if n_rollouts is None:
            n_rollouts = self.rollout_cfg.get("n_rollouts", 20)

        self.before_rollouts()

        print(f"\nStarting {n_rollouts} rollouts...\n")

        rollout_stats: List[Dict[str, Any]] = []
        trajectories: List[Dict[str, Any]] = []

        write_dataset = self.data_writer is not None
        pbar = tqdm(range(n_rollouts), desc="Evaluating Diffusion Policy")

        try:
            for i in pbar:
                stats, traj = self.rollout_episode(episode_idx=i)
                rollout_stats.append(stats)

                if collect_trajectories:
                    trajectories.append(traj)

                current_success_count = sum(s["Success_Rate"] for s in rollout_stats)
                current_rate = current_success_count / len(rollout_stats) * 100.0
                pbar.set_postfix(
                    success_rate=f"{current_rate:.1f}%",
                    last_horizon=stats["Horizon"],
                    last_return=stats["Return"],
                )

                if write_dataset:
                    self.record_episode_data(episode_idx=i, traj=traj)

        finally:
            self.close()

        self.after_rollouts(rollout_stats)

        summary = self.compute_summary(rollout_stats)
        self.print_summary(summary)

        result: Dict[str, Any] = {
            "summary": summary,
            "rollout_stats": rollout_stats,
        }
        if collect_trajectories:
            result["trajectories"] = trajectories

        return result

    # -------------------------------------------------------------------------
    # CLI Parser & Entrypoint
    # -------------------------------------------------------------------------

    @classmethod
    def add_cli_args(cls, parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
        """Add standard CLI arguments for running diffusion policy inference."""
        if parser is None:
            parser = argparse.ArgumentParser(
                description="Evaluate trained Diffusion Policy using a YAML configuration file."
            )
        parser.add_argument(
            "--config",
            type=str,
            default=DEFAULT_CONFIG_PATH,
            help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
        )
        parser.add_argument(
            "config_pos",
            nargs="?",
            default=None,
            help="Optional positional path to YAML configuration file",
        )
        return parser

    @classmethod
    def main(cls, args: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main CLI entrypoint."""
        parser = cls.add_cli_args()
        parsed_args = parser.parse_args(args=args)
        config_file = (
            parsed_args.config_pos
            if parsed_args.config_pos is not None
            else parsed_args.config
        )

        runner = cls(config_path=config_file)
        return runner.run()


# Aliases & functional compatibility wrappers
BaseRunner = DiffusionPolicyRunner
BaseDiffusionPolicyRunner = DiffusionPolicyRunner
rollout = DiffusionPolicyRunner.rollout

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PHASE_APPROACHING",
    "PHASE_PRE_GRASP",
    "PHASE_GRASPED",
    "ALL_PHASES",
    "DiffusionPolicyRunner",
    "BaseRunner",
    "BaseDiffusionPolicyRunner",
    "rollout",
    "deep_update",
]


if __name__ == "__main__":
    DiffusionPolicyRunner.main()
