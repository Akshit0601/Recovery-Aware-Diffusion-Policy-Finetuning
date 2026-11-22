"""
Disturbance Manager for DPPO Fine-Tuning and Generalization Evaluation.

Defines training (Ξ_train) and held-out (Ξ_ho) disturbance distributions across
approach and pre-grasp manipulation phases, manages stratified rollout assignments,
and injects physical object perturbations into the MuJoCo simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

PHASE_APPROACHING = "approaching"
PHASE_PRE_GRASP = "pre-grasp"
PHASE_GRASPED = "grasped"


@dataclass
class PhaseDisturbanceSpec:
    x_range: Tuple[float, float] = (-0.04, 0.04)
    y_range: Tuple[float, float] = (-0.04, 0.04)
    yaw_range_deg: Tuple[float, float] = (-45.0, 45.0)
    delay_steps: int = 15


@dataclass
class DisturbanceConfig:
    nominal_fraction: float = 0.4
    disturbed_fraction: float = 0.6
    phase_ratio: Tuple[float, float] = (0.5, 0.5)  # papp : ppre
    train_approach: PhaseDisturbanceSpec = field(
        default_factory=lambda: PhaseDisturbanceSpec((-0.04, 0.04), (-0.04, 0.04), (-45.0, 45.0), 20)
    )
    train_pre_grasp: PhaseDisturbanceSpec = field(
        default_factory=lambda: PhaseDisturbanceSpec((-0.03, 0.03), (-0.02, 0.02), (-20.0, 20.0), 5)
    )
    ho_approach: PhaseDisturbanceSpec = field(
        default_factory=lambda: PhaseDisturbanceSpec((-0.06, 0.06), (-0.06, 0.06), (-90.0, 90.0), 20)
    )
    ho_pre_grasp: PhaseDisturbanceSpec = field(
        default_factory=lambda: PhaseDisturbanceSpec((-0.05, 0.05), (-0.03, 0.03), (-35.0, 35.0), 5)
    )


class DisturbanceManager:
    """
    Manages disturbance sampling and online injection during policy execution.
    """

    def __init__(self, config: Optional[DisturbanceConfig] = None) -> None:
        self.cfg = config or DisturbanceConfig()

        # Tracking state per episode
        self.active_disturbance: Optional[Dict[str, Any]] = None
        self.has_perturbed: bool = False
        self.phase_step_count: int = 0
        self.current_phase: str = PHASE_APPROACHING

    def plan_iteration_episodes(
        self,
        n_episodes: int,
        mode: str = "train",
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Create stratified assignment of episodes for an iteration.

        Args:
            n_episodes: Total number of episodes N_ep
            mode: "train" (samples from Ξ_train) or "held_out" (samples from Ξ_ho)

        Returns:
            List of length n_episodes with disturbance specification dict or None (nominal).
        """
        n_disturbed = int(round(n_episodes * self.cfg.disturbed_fraction))
        n_nominal = n_episodes - n_disturbed

        p_app, p_pre = self.cfg.phase_ratio
        total_p = p_app + p_pre
        n_approach = int(round(n_disturbed * (p_app / total_p)))
        n_pre_grasp = n_disturbed - n_approach

        episode_plan: List[Optional[Dict[str, Any]]] = []

        # Nominal episodes
        for _ in range(n_nominal):
            episode_plan.append(None)

        # Approach disturbed episodes
        for _ in range(n_approach):
            spec = self.cfg.train_approach if mode == "train" else self.cfg.ho_approach
            dx = float(np.random.uniform(*spec.x_range))
            dy = float(np.random.uniform(*spec.y_range))
            dyaw = float(np.random.uniform(*spec.yaw_range_deg))
            episode_plan.append({
                "phase": PHASE_APPROACHING,
                "delta_x": dx,
                "delta_y": dy,
                "delta_z": 0.0,
                "delta_yaw_deg": dyaw,
                "delay_steps": spec.delay_steps,
                "mode": mode,
            })

        # Pre-grasp disturbed episodes
        for _ in range(n_pre_grasp):
            spec = self.cfg.train_pre_grasp if mode == "train" else self.cfg.ho_pre_grasp
            dx = float(np.random.uniform(*spec.x_range))
            dy = float(np.random.uniform(*spec.y_range))
            dyaw = float(np.random.uniform(*spec.yaw_range_deg))
            episode_plan.append({
                "phase": PHASE_PRE_GRASP,
                "delta_x": dx,
                "delta_y": dy,
                "delta_z": 0.0,
                "delta_yaw_deg": dyaw,
                "delay_steps": spec.delay_steps,
                "mode": mode,
            })

        # Shuffle plan for randomized execution order
        indices = np.random.permutation(len(episode_plan))
        return [episode_plan[i] for i in indices]

    def reset_episode(self, disturbance_spec: Optional[Dict[str, Any]]) -> None:
        """Reset state tracking for a new episode."""
        self.active_disturbance = disturbance_spec
        self.has_perturbed = False
        self.phase_step_count = 0
        self.current_phase = PHASE_APPROACHING

    def detect_phase(
        self,
        action: np.ndarray,
        reward: float,
        step_i: int = 0,
        arm_threshold: float = 0.4,
        gripper_threshold: float = 0.0,
    ) -> str:
        """
        Detect current operational phase from step action and reward.
        """
        if reward >= 1.0:
            new_phase = PHASE_GRASPED
        elif self.current_phase == PHASE_GRASPED:
            new_phase = PHASE_GRASPED
        else:
            act = np.asarray(action).ravel()
            arm_norm = float(np.linalg.norm(act[:6])) if len(act) >= 7 else 0.0
            grip_val = float(act[-1]) if len(act) >= 7 else 0.0

            is_pre_grasp = (grip_val > gripper_threshold and arm_norm <= arm_threshold)
            if is_pre_grasp:
                new_phase = PHASE_PRE_GRASP
            elif self.current_phase == PHASE_PRE_GRASP:
                # Sticky pre-grasp unless transitioned
                new_phase = PHASE_PRE_GRASP
            else:
                new_phase = PHASE_APPROACHING

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            self.phase_step_count = 0
        else:
            self.phase_step_count += 1

        return self.current_phase

    def should_perturb(self) -> bool:
        """Check if conditions to inject the active disturbance are met."""
        if self.active_disturbance is None or self.has_perturbed:
            return False

        target_phase = self.active_disturbance["phase"]
        delay = self.active_disturbance.get("delay_steps", 0)

        if self.current_phase == target_phase and self.phase_step_count >= delay:
            return True

        return False

    def inject_disturbance(self, env: Any, action: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Apply physical pose perturbation to target object in simulation and refresh observation.
        """
        if not self.should_perturb():
            return None

        # Unwrap robosuite env
        robosuite_env = env
        while hasattr(robosuite_env, "env") and not hasattr(robosuite_env, "sim"):
            robosuite_env = robosuite_env.env

        sim = robosuite_env.sim
        jname = "cube_joint0"
        for i in range(sim.model.njnt):
            n = sim.model.joint_id2name(i)
            if "cube" in n or "object" in n:
                jname = n
                break

        qpos = sim.data.get_joint_qpos(jname).copy()
        orig_pos = qpos[:3].copy()
        orig_quat = qpos[3:7].copy()

        dx = float(self.active_disturbance["delta_x"])
        dy = float(self.active_disturbance["delta_y"])
        dz = float(self.active_disturbance["delta_z"])
        dyaw = float(self.active_disturbance["delta_yaw_deg"])

        new_pos = orig_pos.copy()
        new_pos[0] += dx
        new_pos[1] += dy
        new_pos[2] += dz

        new_quat = orig_quat.copy()
        if abs(dyaw) > 1e-4:
            dyaw_rad = np.deg2rad(dyaw)
            q_delta = np.array([
                np.cos(dyaw_rad / 2.0),
                0.0,
                0.0,
                np.sin(dyaw_rad / 2.0),
            ])
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

        qpos[:3] = new_pos
        qpos[3:7] = new_quat
        sim.data.set_joint_qpos(jname, qpos)
        sim.data.set_joint_qvel(jname, np.zeros(6))
        sim.forward()

        self.has_perturbed = True

        # Refresh FrameStackWrapper history with new object pose
        if hasattr(env, "obs_history") and env.obs_history is not None:
            raw_obs = robosuite_env._get_observations(force_update=True)
            single_obs = env.env.get_observation(di=raw_obs) if hasattr(env, "env") else raw_obs
            env.update_obs(single_obs, action=action, reset=False)
            for k in single_obs:
                if k in env.obs_history:
                    env.obs_history[k][-1] = single_obs[k][None]

        record = {
            "phase": self.current_phase,
            "orig_pos": orig_pos.tolist(),
            "new_pos": new_pos.tolist(),
            "delta_trans": [dx, dy, dz],
            "delta_yaw_deg": dyaw,
        }
        return record
