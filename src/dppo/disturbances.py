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

    # YAML (split, phase) -> dataclass field
    _SPEC_FIELDS = {
        ("train", "approach"): "train_approach",
        ("train", "pre_grasp"): "train_pre_grasp",
        ("held_out", "approach"): "ho_approach",
        ("held_out", "pre_grasp"): "ho_pre_grasp",
    }

    @classmethod
    def from_config(
        cls,
        rollout_cfg: Optional[Dict[str, Any]] = None,
        dist_cfg: Optional[Dict[str, Any]] = None,
    ) -> "DisturbanceConfig":
        """
        Build from the `rollout:` and `disturbances:` YAML sections.

        Any key present in YAML overrides the dataclass default; missing keys keep
        the default. Unknown splits/phases/keys raise so typos don't get silently
        ignored.
        """
        rollout_cfg = rollout_cfg or {}
        dist_cfg = dist_cfg or {}
        cfg = cls()

        if "nominal_fraction" in rollout_cfg:
            cfg.nominal_fraction = float(rollout_cfg["nominal_fraction"])
        if "disturbed_fraction" in rollout_cfg:
            cfg.disturbed_fraction = float(rollout_cfg["disturbed_fraction"])
        if "phase_ratio" in rollout_cfg:
            pr = rollout_cfg["phase_ratio"]
            if len(pr) != 2:
                raise ValueError(f"rollout.phase_ratio must have 2 entries, got {pr}")
            cfg.phase_ratio = (float(pr[0]), float(pr[1]))
        if abs(cfg.nominal_fraction + cfg.disturbed_fraction - 1.0) > 1e-6:
            raise ValueError(
                f"nominal_fraction ({cfg.nominal_fraction}) + disturbed_fraction "
                f"({cfg.disturbed_fraction}) must sum to 1"
            )

        valid_keys = {"x_range", "y_range", "yaw_range_deg", "delay_steps"}
        for split, phases in dist_cfg.items():
            if split not in ("train", "held_out"):
                raise ValueError(f"Unknown disturbances split '{split}' (expected train/held_out)")
            for phase, overrides in (phases or {}).items():
                key = (split, phase)
                if key not in cls._SPEC_FIELDS:
                    raise ValueError(f"Unknown disturbances phase '{split}.{phase}' (expected approach/pre_grasp)")
                unknown = set(overrides or {}) - valid_keys
                if unknown:
                    raise ValueError(f"Unknown keys in disturbances.{split}.{phase}: {sorted(unknown)}")

                attr = cls._SPEC_FIELDS[key]
                spec: PhaseDisturbanceSpec = getattr(cfg, attr)
                for range_key in ("x_range", "y_range", "yaw_range_deg"):
                    if range_key in overrides:
                        lo, hi = (float(v) for v in overrides[range_key])
                        if lo > hi:
                            raise ValueError(f"disturbances.{split}.{phase}.{range_key}: low > high ({lo}, {hi})")
                        setattr(spec, range_key, (lo, hi))
                if "delay_steps" in overrides:
                    spec.delay_steps = int(overrides["delay_steps"])
        return cfg


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
        Create stratified assignment of episodes for a training iteration.

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
        return self.plan_episodes(n_nominal, n_approach, n_pre_grasp, mode=mode)

    def plan_episodes(
        self,
        n_nominal: int,
        n_approach: int,
        n_pre_grasp: int,
        mode: str = "train",
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Build a shuffled episode plan with explicit counts per regime
        (used directly by evaluation so every regime gets the same sample size).
        """
        if mode not in ("train", "held_out"):
            raise ValueError(f"Unknown disturbance mode '{mode}'")

        episode_plan: List[Optional[Dict[str, Any]]] = [None] * n_nominal

        for phase, n, spec in (
            (PHASE_APPROACHING, n_approach, self.cfg.train_approach if mode == "train" else self.cfg.ho_approach),
            (PHASE_PRE_GRASP, n_pre_grasp, self.cfg.train_pre_grasp if mode == "train" else self.cfg.ho_pre_grasp),
        ):
            for _ in range(n):
                episode_plan.append({
                    "phase": phase,
                    "delta_x": float(np.random.uniform(*spec.x_range)),
                    "delta_y": float(np.random.uniform(*spec.y_range)),
                    "delta_z": 0.0,
                    "delta_yaw_deg": float(np.random.uniform(*spec.yaw_range_deg)),
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
