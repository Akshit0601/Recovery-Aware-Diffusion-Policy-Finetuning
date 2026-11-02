#!/usr/bin/env python
"""
Phase-Aware Object Disturbance Evaluator (Config-Free).

Extends DiffusionPolicyRunner to apply physical pose disturbances (translations and yaw rotations)
to the target object during specified execution phases (e.g. approaching, pre-grasp, grasped)
to benchmark policy recovery, robustness, and grasp resilience.

All parameters are defined directly at the top of this file (no argparse required).
Run with:
    python -m src.eval_phase_disturbances
or:
    python src/eval_phase_disturbances.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.diffusion_policy_runner import (
    ALL_PHASES,
    PHASE_APPROACHING,
    PHASE_GRASPED,
    PHASE_PRE_GRASP,
    DiffusionPolicyRunner,
)

# ==============================================================================
# Phase Disturbance Configuration (Edit directly here - No Argparse)
# ==============================================================================
# Target phase(s) during which the object pose disturbance will be injected.
# Options: "approaching", "pre-grasp", "grasped", or a list like ["approaching", "pre-grasp"]
PERTURB_PHASE: Union[str, List[str]] = "pre-grasp"

# Pose Disturbance Displacements (applied to object in simulation):
DELTA_X: float = 0.025          # Linear displacement along X in meters (+2.5 cm)
DELTA_Y: float = 0.020          # Linear displacement along Y in meters (+2.0 cm)
DELTA_Z: float = 0.000          # Linear displacement along Z in meters
DELTA_YAW_DEG: float = 50.0     # Yaw rotation displacement in degrees (+15°)

# Trigger timing within the phase:
PERTURB_ONCE_PER_EPISODE: bool = True   # Apply disturbance only once per rollout episode
PERTURB_STEP_DELAY: int = 0            # Number of steps after entering phase before perturbing (0 = immediately)

# Random Disturbance Mode (Optional override):
# If True, uniform random values within the ranges below are sampled instead of fixed DELTA_*
USE_RANDOM_DISTURBANCE: bool = True
RANDOM_X_RANGE: Tuple[float, float] = (-0.05, 0.05)
RANDOM_Y_RANGE: Tuple[float, float] = (-0.03, 0.03)
RANDOM_YAW_RANGE: Tuple[float, float] = (-30.0, 30.0)

# Initial object spawn placement (None uses default/randomized table sampler):
OBJECT_INIT_X: Optional[float] = None
OBJECT_INIT_Y: Optional[float] = None
OBJECT_INIT_YAW_DEG: Optional[float] = None

# Evaluation & Rollout Settings:
N_ROLLOUTS: int = 200                     # Number of evaluation rollout episodes
HORIZON: int = 400                      # Maximum steps per episode
RENDER: bool = False                    # Set True for real-time graphical visualization
VIDEO_PATH: Optional[str] = None        # e.g. "results/phase_disturbances/video.mp4" (or None)
OUTPUT_DIR: str = "results/phase_disturbances"  # Directory to save metrics and disturbance records
SEED: int = 42

DEFAULT_CHECKPOINT_PATH = (
    "/home/akshit/ece285_proj/robomimic/diffusion_policy_trained_models/"
    "test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth"
)


class PhaseDisturbanceRunner(DiffusionPolicyRunner):
    """
    Evaluator that injects object pose disturbances during specified execution phases.

    Subclasses DiffusionPolicyRunner and intercepts environment steps to manipulate
    the target object's 6-DoF pose in MuJoCo when the active execution phase matches
    the target perturbation criteria.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        perturb_phase: Union[str, List[str]] = PERTURB_PHASE,
        delta_x: float = DELTA_X,
        delta_y: float = DELTA_Y,
        delta_z: float = DELTA_Z,
        delta_yaw_deg: float = DELTA_YAW_DEG,
        perturb_once: bool = PERTURB_ONCE_PER_EPISODE,
        perturb_step_delay: int = PERTURB_STEP_DELAY,
        use_random: bool = USE_RANDOM_DISTURBANCE,
        random_x_range: Tuple[float, float] = RANDOM_X_RANGE,
        random_y_range: Tuple[float, float] = RANDOM_Y_RANGE,
        random_yaw_range: Tuple[float, float] = RANDOM_YAW_RANGE,
        init_x: Optional[float] = OBJECT_INIT_X,
        init_y: Optional[float] = OBJECT_INIT_Y,
        init_yaw_deg: Optional[float] = OBJECT_INIT_YAW_DEG,
        n_rollouts: int = N_ROLLOUTS,
        horizon: int = HORIZON,
        render: bool = RENDER,
        video_path: Optional[str] = VIDEO_PATH,
        output_dir: str = OUTPUT_DIR,
        seed: int = SEED,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[Dict[str, Any]] = None,
        policy: Optional[Any] = None,
        env: Optional[Any] = None,
    ) -> None:
        """Initialize the disturbance runner with specified parameters."""
        self.target_phases = (
            [perturb_phase] if isinstance(perturb_phase, str) else list(perturb_phase)
        )
        self.delta_x = float(delta_x)
        self.delta_y = float(delta_y)
        self.delta_z = float(delta_z)
        self.delta_yaw_deg = float(delta_yaw_deg)
        self.perturb_once = bool(perturb_once)
        self.perturb_step_delay = int(perturb_step_delay)
        self.use_random = bool(use_random)
        self.random_x_range = tuple(random_x_range)
        self.random_y_range = tuple(random_y_range)
        self.random_yaw_range = tuple(random_yaw_range)
        self.init_x = init_x
        self.init_y = init_y
        self.init_yaw_deg = init_yaw_deg
        self.output_dir = output_dir

        # Runtime disturbance tracking state
        self.current_step: int = 0
        self.phase_step_count: int = 0
        self.has_perturbed_in_episode: bool = False
        self.applied_disturbances: List[Dict[str, Any]] = []

        if config is None and config_path is None and policy is None and env is None:
            in_memory_config = {
                "weights": {
                    "checkpoint_path": checkpoint_path,
                },
                "rollout": {
                    "n_rollouts": n_rollouts,
                    "horizon": horizon,
                    "seed": seed,
                },
                "rendering": {
                    "render": render,
                    "video_path": video_path,
                    "video_skip": 5,
                },
                "camera": {
                    "camera_names": ["agentview"],
                },
                "object_placement": {},
                "dataset": {
                    "dataset_path": None,
                    "dataset_obs": False,
                },
            }
            super().__init__(config=in_memory_config, overrides=overrides)
        else:
            super().__init__(
                config=config,
                config_path=config_path,
                overrides=overrides,
                policy=policy,
                env=env,
            )

    # -------------------------------------------------------------------------
    # Object Inspection & Manipulation Helpers
    # -------------------------------------------------------------------------

    def _get_robosuite_env(self) -> Any:
        """Unwrap and retrieve underlying robosuite environment instance."""
        if hasattr(self.env, "env") and hasattr(self.env.env, "env"):
            return self.env.env.env
        if hasattr(self.env, "env"):
            return self.env.env
        return self.env

    def _get_object_joint_name(self) -> str:
        """Determine object joint name in MuJoCo model."""
        robosuite_env = self._get_robosuite_env()
        if (
            hasattr(robosuite_env, "cube")
            and hasattr(robosuite_env.cube, "joints")
            and len(robosuite_env.cube.joints) > 0
        ):
            return robosuite_env.cube.joints[0]

        sim = robosuite_env.sim
        for i in range(sim.model.njnt):
            jname = sim.model.joint_id2name(i)
            if "cube" in jname or "object" in jname:
                return jname
        return "cube_joint0"

    def get_object_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current 3D position and 4D quaternion [qw, qx, qy, qz] of object.

        Returns:
            Tuple of (pos_xyz, quat_wxyz)
        """
        robosuite_env = self._get_robosuite_env()
        sim = robosuite_env.sim
        jname = self._get_object_joint_name()
        qpos = sim.data.get_joint_qpos(jname).copy()
        return qpos[:3], qpos[3:7]

    def set_object_pose(
        self, pos: np.ndarray, quat: Optional[np.ndarray] = None
    ) -> None:
        """
        Directly set object position and quaternion in simulation physics.

        Args:
            pos: 3D position array [X, Y, Z].
            quat: Optional 4D quaternion array [qw, qx, qy, qz].
        """
        robosuite_env = self._get_robosuite_env()
        sim = robosuite_env.sim
        jname = self._get_object_joint_name()
        qpos = sim.data.get_joint_qpos(jname).copy()
        qpos[:3] = np.asarray(pos, dtype=float)
        if quat is not None:
            qpos[3:7] = np.asarray(quat, dtype=float)

        sim.data.set_joint_qpos(jname, qpos)
        sim.data.set_joint_qvel(jname, np.zeros(6))
        sim.forward()

    def apply_object_disturbance(
        self,
        delta_x: Optional[float] = None,
        delta_y: Optional[float] = None,
        delta_z: Optional[float] = None,
        delta_yaw_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Inject a spatial displacement and yaw rotation to the target object.

        Returns:
            Dictionary recording disturbance event details.
        """
        dx = float(delta_x if delta_x is not None else self.delta_x)
        dy = float(delta_y if delta_y is not None else self.delta_y)
        dz = float(delta_z if delta_z is not None else self.delta_z)
        dyaw = float(delta_yaw_deg if delta_yaw_deg is not None else self.delta_yaw_deg)

        if self.use_random:
            dx = float(np.random.uniform(*self.random_x_range))
            dy = float(np.random.uniform(*self.random_y_range))
            dyaw = float(np.random.uniform(*self.random_yaw_range))

        orig_pos, orig_quat = self.get_object_pose()

        # Compute new position
        new_pos = orig_pos.copy()
        new_pos[0] += dx
        new_pos[1] += dy
        new_pos[2] += dz

        # Compute new quaternion rotated around Z axis by dyaw
        new_quat = orig_quat.copy()
        if abs(dyaw) > 1e-4:
            dyaw_rad = np.deg2rad(dyaw)
            q_delta = np.array([
                np.cos(dyaw_rad / 2.0),
                0.0,
                0.0,
                np.sin(dyaw_rad / 2.0),
            ])
            # Quaternion multiplication: q_delta * orig_quat
            qw, qx, qy, qz = q_delta
            w, x, y, z = orig_quat
            new_quat = np.array([
                qw * w - qx * x - qy * y - qz * z,
                qw * x + qx * w + qy * z - qz * y,
                qw * y - qx * z + qy * w + qz * x,
                qw * z + qx * y - qy * x + qz * w,
            ])
            norm = np.linalg.norm(new_quat)
            if norm > 1e-6:
                new_quat = new_quat / norm

        self.set_object_pose(pos=new_pos, quat=new_quat)

        record = {
            "step": self.current_step,
            "phase": self.current_phase,
            "phase_step": self.phase_step_count,
            "orig_pos": orig_pos.tolist(),
            "new_pos": new_pos.tolist(),
            "orig_quat": orig_quat.tolist(),
            "new_quat": new_quat.tolist(),
            "delta_trans": [dx, dy, dz],
            "delta_yaw_deg": dyaw,
        }
        self.applied_disturbances.append(record)
        self.has_perturbed_in_episode = True
        return record

    def _should_disturb(self) -> bool:
        """Determine whether disturbance conditions are met for current step."""
        if self.perturb_once and self.has_perturbed_in_episode:
            return False
        if self.current_phase not in self.target_phases:
            return False
        if self.phase_step_count < self.perturb_step_delay:
            return False
        return True

    # -------------------------------------------------------------------------
    # Lifecycle Hooks
    # -------------------------------------------------------------------------

    def reset_env(
        self, env=None, policy=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Optionally inject initial object spawn placement before reset."""
        active_env = env if env is not None else self.env
        robosuite_env = self._get_robosuite_env()

        if (
            robosuite_env is not None
            and hasattr(robosuite_env, "placement_initializer")
            and (self.init_x is not None or self.init_y is not None or self.init_yaw_deg is not None)
        ):
            sampler = robosuite_env.placement_initializer
            if self.init_x is not None:
                sampler.x_range = [self.init_x, self.init_x]
            if self.init_y is not None:
                sampler.y_range = [self.init_y, self.init_y]
            if self.init_yaw_deg is not None:
                sampler.rotation = float(np.deg2rad(self.init_yaw_deg))
                sampler.rotation_axis = "z"

        return super().reset_env(env=active_env, policy=policy)

    def on_episode_start(self, episode_idx: int) -> None:
        """Reset disturbance tracking state at the beginning of each episode."""
        super().on_episode_start(episode_idx)
        self.current_step = 0
        self.phase_step_count = 0
        self.has_perturbed_in_episode = False
        self.applied_disturbances = []

    def on_phase_transition(
        self,
        prev_phase: str,
        new_phase: str,
        step_i: int,
        action: np.ndarray,
        reward: float,
    ) -> None:
        """Reset phase step counter on phase boundary."""
        super().on_phase_transition(prev_phase, new_phase, step_i, action, reward)
        self.phase_step_count = 0
        logging.info(
            f"[Phase Transition] Step {step_i:2d}: {prev_phase} -> {new_phase}"
        )

    def _refresh_observation(
        self, active_env: Any, action: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Refresh observation from simulation physics, correctly updating FrameStackWrapper history.
        """
        robosuite_env = self._get_robosuite_env()
        raw_obs = robosuite_env._get_observations(force_update=True)

        env_robosuite = active_env
        while hasattr(env_robosuite, "env") and type(env_robosuite).__name__ != "EnvRobosuite":
            env_robosuite = env_robosuite.env

        single_frame_obs = env_robosuite.get_observation(di=raw_obs)

        if hasattr(active_env, "obs_history") and active_env.obs_history is not None:
            active_env.update_obs(single_frame_obs, action=action, reset=False)
            for k in single_frame_obs:
                if k in active_env.obs_history:
                    # Update current step frame in deque
                    active_env.obs_history[k][-1] = single_frame_obs[k][None]
            return active_env._get_stacked_obs_from_history()

        return single_frame_obs

    def step_env(
        self, action: np.ndarray, env=None
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Step simulation and inject object pose disturbance if conditions are met.
        """
        active_env = env if env is not None else self.env
        next_obs, r, done, info = super().step_env(action, env=active_env)

        if self._should_disturb():
            dist_record = self.apply_object_disturbance()
            # Refresh observation with frame stacking so policy perceives updated pose
            next_obs = self._refresh_observation(active_env, action=action)
            logging.info(
                f"[Disturbance Applied] Step {self.current_step} ({self.current_phase}): "
                f"delta=[{dist_record['delta_trans'][0]:+.3f}, {dist_record['delta_trans'][1]:+.3f}, {dist_record['delta_trans'][2]:+.3f}]m, "
                f"dyaw={dist_record['delta_yaw_deg']:+.1f}° "
                f"(pos: {np.round(dist_record['orig_pos'], 3)} -> {np.round(dist_record['new_pos'], 3)})"
            )

        self.current_step += 1
        self.phase_step_count += 1
        return next_obs, r, done, info

    def on_episode_end(
        self, episode_idx: int, stats: Dict[str, Any], traj: Dict[str, Any]
    ) -> None:
        """Append disturbance records to episode statistics and trajectory."""
        super().on_episode_end(episode_idx, stats, traj)
        stats["Num_Disturbances"] = len(self.applied_disturbances)
        stats["Disturbed"] = bool(len(self.applied_disturbances) > 0)
        stats["Disturbances"] = list(self.applied_disturbances)
        traj["disturbances"] = list(self.applied_disturbances)

    def compute_summary(self, rollout_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute aggregated evaluation and recovery summary metrics."""
        summary = super().compute_summary(rollout_stats)
        disturbed_stats = [s for s in rollout_stats if s.get("Disturbed", False)]
        disturbed_count = len(disturbed_stats)
        disturbed_successes = sum(s["Success_Rate"] for s in disturbed_stats)
        recovery_rate = (
            (disturbed_successes / disturbed_count * 100.0)
            if disturbed_count > 0
            else 0.0
        )

        summary["target_phases"] = self.target_phases
        summary["disturbed_episodes"] = disturbed_count
        summary["recovery_rate_under_disturbance"] = recovery_rate
        summary["nominal_episodes"] = len(rollout_stats) - disturbed_count
        return summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Display disturbance evaluation results table."""
        super().print_summary(summary)
        print(f"{'='*70}")
        print("             PHASE DISTURBANCE EVALUATION")
        print(f"{'='*70}")
        print(f"Target Phase(s)   : {', '.join(summary.get('target_phases', []))}")
        print(f"Disturbed Episodes: {summary.get('disturbed_episodes', 0)} / {summary['total_rollouts']}")
        print(f"Recovery Rate     : {summary.get('recovery_rate_under_disturbance', 0.0):.2f}%")
        print(f"{'='*70}\n")

    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """Export evaluation summary and records to JSON."""
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target_file = (
            Path(output_path)
            if output_path
            else out_dir / "phase_disturbance_results.json"
        )

        exportable = {
            "summary": results.get("summary", {}),
            "rollout_stats": results.get("rollout_stats", []),
            "configuration": {
                "target_phases": self.target_phases,
                "delta_x": self.delta_x,
                "delta_y": self.delta_y,
                "delta_z": self.delta_z,
                "delta_yaw_deg": self.delta_yaw_deg,
                "perturb_once": self.perturb_once,
                "perturb_step_delay": self.perturb_step_delay,
                "use_random": self.use_random,
            },
        }

        with open(target_file, "w") as f:
            json.dump(exportable, f, indent=2)

        print(f"Saved evaluation metrics to: {target_file}")
        return str(target_file)


def run_phase_disturbance_eval() -> Dict[str, Any]:
    """Execute phase disturbance evaluation using the parameters defined in headers."""
    runner = PhaseDisturbanceRunner()
    results = runner.run(n_rollouts=N_ROLLOUTS)
    runner.save_results(results)
    return results


if __name__ == "__main__":
    run_phase_disturbance_eval()
