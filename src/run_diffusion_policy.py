#!/usr/bin/env python
"""
Run and evaluate the trained Diffusion Policy on Robomimic / Robosuite Lift environment.

All evaluation, environment, camera, and model checkpoint parameters are loaded
from a YAML configuration file (default: config/run_diffusion_policy.yaml).

Example usage:
    # 1. Run using default config (config/run_diffusion_policy.yaml)
    python src/run_diffusion_policy.py

    # 2. Run with a specific config file
    python src/run_diffusion_policy.py --config config/run_diffusion_policy.yaml

    # 3. Positional config argument
    python src/run_diffusion_policy.py config/custom_eval.yaml
"""

from typing import Any, Dict

from src.diffusion_policy_runner import (
    DEFAULT_CONFIG_PATH,
    DiffusionPolicyRunner,
    rollout,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the YAML configuration file."""
    return DiffusionPolicyRunner.load_config_file(config_path)


def run_diffusion_eval(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run diffusion policy evaluation driven by config dictionary.
    """
    runner = DiffusionPolicyRunner(config=cfg)
    return runner.run()


if __name__ == "__main__":
    DiffusionPolicyRunner.main()
