#!/usr/bin/env python
"""
Run and evaluate the trained Diffusion Policy at a specific object pose (X, Y, Yaw).

Configure the target pose and rendering directly by editing the parameters
defined at the top of this file.
"""

from typing import Any, Dict, Tuple

import numpy as np

from src.diffusion_policy_runner import (
    DEFAULT_CONFIG_PATH,
    DiffusionPolicyRunner,
    rollout,
)

# ==============================================================================
# Target Pose & Rendering Parameters (Edit directly here)
# ==============================================================================
X = 0.0025000000000000022           # Object X coordinate in meters (robot forward/backward)
Y = 0.025           # Object Y coordinate in meters (robot left/right)
YAW = 0.0            # Object Yaw angle in degrees (e.g. 0, 30, 45)
RENDER = True        # Render on-screen in real time (True / False)


class SinglePoseRunner(DiffusionPolicyRunner):
    """
    Runner that evaluates diffusion policy rollouts at a specific fixed (X, Y, Yaw) pose.
    """

    def __init__(
        self,
        x: float = X,
        y: float = Y,
        yaw: float = YAW,
        render: bool = RENDER,
        n_rollouts: int = 1,
        config_path: str = DEFAULT_CONFIG_PATH,
        **kwargs: Any,
    ) -> None:
        self.target_x = float(x)
        self.target_y = float(y)
        self.target_yaw_deg = float(yaw)
        self.target_render = bool(render)

        overrides: Dict[str, Any] = {
            "rendering": {
                "render": self.target_render,
                "video_path": None,
            },
            "rollout": {
                "n_rollouts": int(n_rollouts),
            },
            "object_placement": {
                "object_pos": [self.target_x, self.target_y],
            },
        }

        super().__init__(config_path=config_path, overrides=overrides, **kwargs)

    def reset_env(
        self, env=None, policy=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Inject exact (X, Y) position and Yaw angle into the placement initializer
        prior to environment reset.
        """
        active_env = env if env is not None else self.env
        robosuite_env = (
            active_env.env.env
            if hasattr(active_env, "env") and hasattr(active_env.env, "env")
            else None
        )

        if robosuite_env is not None and hasattr(robosuite_env, "placement_initializer"):
            sampler = robosuite_env.placement_initializer
            sampler.x_range = [self.target_x, self.target_x]
            sampler.y_range = [self.target_y, self.target_y]
            sampler.rotation = float(np.deg2rad(self.target_yaw_deg))
            sampler.rotation_axis = "z"

        return super().reset_env(env=active_env, policy=policy)


def run_pose(
    x: float = X,
    y: float = Y,
    yaw: float = YAW,
    render: bool = RENDER,
    n_rollouts: int = 1,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Dict[str, Any]:
    """Execute evaluation at a specific target pose."""
    print(f"\n{'='*70}")
    print(f"Running Diffusion Policy at Target Pose:")
    print(f"  X      = {x:+.4f} m")
    print(f"  Y      = {y:+.4f} m")
    print(f"  Yaw    = {yaw:+.1f}°")
    print(f"  Render = {render}")
    print(f"{'='*70}\n")

    runner = SinglePoseRunner(
        x=x,
        y=y,
        yaw=yaw,
        render=render,
        n_rollouts=n_rollouts,
        config_path=config_path,
    )
    return runner.run()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    return DiffusionPolicyRunner.load_config_file(config_path)


def run_diffusion_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run diffusion policy evaluation driven by config dictionary."""
    runner = DiffusionPolicyRunner(config=cfg)
    return runner.run()


def main():
    run_pose(
        x=X,
        y=Y,
        yaw=YAW,
        render=RENDER,
        n_rollouts=1,
    )


if __name__ == "__main__":
    main()
