"""Unit and integration tests for DiffusionPolicyRunner."""

import pytest
import numpy as np
from src.diffusion_policy_runner import (
    ALL_PHASES,
    PHASE_APPROACHING,
    PHASE_GRASPED,
    PHASE_PRE_GRASP,
    BaseRunner,
    DiffusionPolicyRunner,
    deep_update,
    rollout,
)
from src.run_diffusion_policy import load_config, run_diffusion_eval


def test_deep_update():
    base = {"a": 1, "b": {"x": 10, "y": 20}}
    override = {"b": {"y": 99, "z": 100}, "c": 3}
    deep_update(base, override)
    assert base["a"] == 1
    assert base["b"]["x"] == 10
    assert base["b"]["y"] == 99
    assert base["b"]["z"] == 100
    assert base["c"] == 3


def test_runner_initialization_and_overrides():
    runner = DiffusionPolicyRunner(
        config_path="config/run_diffusion_policy.yaml",
        overrides={"rollout": {"n_rollouts": 3, "horizon": 15}},
    )
    assert runner.rollout_cfg["n_rollouts"] == 3
    assert runner.rollout_cfg["horizon"] == 15
    assert runner.rendering_cfg is not None
    assert runner.camera_cfg is not None
    assert runner.object_placement_cfg is not None


def test_runner_subclass_and_execution():
    """Test subclassing and running a minimal headless rollout."""
    custom_hooks_called = {"on_episode_start": 0, "on_step_end": 0, "get_action": 0}

    class CustomExperimentRunner(DiffusionPolicyRunner):
        def on_episode_start(self, episode_idx):
            super().on_episode_start(episode_idx)
            custom_hooks_called["on_episode_start"] += 1

        def get_action(self, obs, step_i=0, policy=None):
            custom_hooks_called["get_action"] += 1
            action = super().get_action(obs, step_i=step_i, policy=policy)
            return action

        def on_step_end(self, step_i, obs, action, next_obs, reward, done, success, info):
            super().on_step_end(step_i, obs, action, next_obs, reward, done, success, info)
            custom_hooks_called["on_step_end"] += 1

    runner = CustomExperimentRunner(
        config_path="config/run_diffusion_policy.yaml",
        overrides={
            "rollout": {"n_rollouts": 1, "horizon": 2},
            "rendering": {"render": False, "video_path": None},
            "dataset": {"dataset_path": None},
        },
    )

    results = runner.run()
    assert "summary" in results
    assert results["summary"]["total_rollouts"] == 1
    assert custom_hooks_called["on_episode_start"] == 1
    assert custom_hooks_called["get_action"] == 2
    assert custom_hooks_called["on_step_end"] == 2


def test_rollout_episode_full_utility_and_delegation():
    """Verify rollout_episode covers bare policy/env execution and delegation."""
    # 1. Setup a runner to obtain policy and env instances
    runner = DiffusionPolicyRunner(
        config_path="config/run_diffusion_policy.yaml",
        overrides={
            "rollout": {"horizon": 2},
            "rendering": {"render": False, "video_path": None},
            "dataset": {"dataset_path": None},
        },
    )
    runner.setup()
    policy = runner.policy
    env = runner.env

    # 2. Test rollout_episode called directly with explicit arguments
    stats1, traj1 = runner.rollout_episode(
        horizon=2,
        policy=policy,
        env=env,
        render=False,
        video_writer=None,
        return_obs=True,
    )
    assert stats1["Horizon"] == 2
    assert "actions" in traj1
    assert "obs" in traj1
    assert len(traj1["actions"]) == 2

    # 3. Test standalone instantiation with policy & env directly (no config file)
    bare_runner = DiffusionPolicyRunner(policy=policy, env=env)
    assert bare_runner.is_setup is True
    stats2, traj2 = bare_runner.rollout_episode(horizon=2, render=False)
    assert stats2["Horizon"] == 2

    # 4. Test rollout() classmethod / module alias delegation
    stats3, traj3 = rollout(
        policy=policy,
        env=env,
        horizon=2,
        render=False,
        video_writer=None,
        return_obs=False,
    )
    assert stats3["Horizon"] == 2
    assert "actions" in traj3


def test_run_diffusion_eval_backwards_compatible():
    cfg = load_config("config/run_diffusion_policy.yaml")
    cfg["rollout"]["n_rollouts"] = 1
    cfg["rollout"]["horizon"] = 2
    cfg["rendering"]["render"] = False
    cfg["rendering"]["video_path"] = None
    cfg["dataset"]["dataset_path"] = None

    results = run_diffusion_eval(cfg)
    assert "summary" in results
    assert results["summary"]["total_rollouts"] == 1


def test_phase_constants_and_detection():
    """Verify phase constants and detection logic."""
    assert PHASE_APPROACHING == "approaching"
    assert PHASE_PRE_GRASP == "pre-grasp"
    assert PHASE_GRASPED == "grasped"
    assert ALL_PHASES == ["approaching", "pre-grasp", "grasped"]

    runner = DiffusionPolicyRunner(
        config_path="config/run_diffusion_policy.yaml",
        overrides={"rollout": {"pre_grasp_arm_threshold": 0.4, "pre_grasp_gripper_threshold": 0.0}},
    )

    # 1. Approaching: arm moving, gripper open (-1.0)
    approach_act = np.array([0.2, 0.1, -0.3, 0.0, 0.0, 0.0, -1.0])
    phase = runner.detect_phase(action=approach_act, reward=0.0)
    assert phase == PHASE_APPROACHING
    assert runner.is_pre_grasp_action(approach_act) is False

    # 2. Pre-grasp: pure gripper action (arm stationary, gripper closing)
    pre_grasp_pure = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    phase = runner.detect_phase(action=pre_grasp_pure, reward=0.0)
    assert phase == PHASE_PRE_GRASP
    assert runner.is_pre_grasp_action(pre_grasp_pure) is True

    # 3. Pre-grasp: realistic small arm movement while gripper closing
    pre_grasp_real = np.array([0.05, 0.02, -0.03, 0.01, 0.0, 0.01, 0.98])
    phase = runner.detect_phase(action=pre_grasp_real, reward=0.0)
    assert phase == PHASE_PRE_GRASP

    # 4. Grasped: reward reaches 1.0
    grasped_act = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    phase = runner.detect_phase(action=grasped_act, reward=1.0)
    assert phase == PHASE_GRASPED


def test_rollout_phase_tracking():
    """Verify phases array, stats tracking, and on_phase_transition hook during rollout."""
    transitions = []

    class PhaseTrackingRunner(DiffusionPolicyRunner):
        def on_phase_transition(self, prev_phase, new_phase, step_i, action, reward):
            transitions.append((prev_phase, new_phase, step_i))

    runner = PhaseTrackingRunner(
        config_path="config/run_diffusion_policy.yaml",
        overrides={
            "rollout": {"n_rollouts": 1, "horizon": 3},
            "rendering": {"render": False, "video_path": None},
            "dataset": {"dataset_path": None},
        },
    )

    results = runner.run()
    stats = results["rollout_stats"][0]

    assert "Final_Phase" in stats
    assert stats["Final_Phase"] in ALL_PHASES
    assert "Reached_PreGrasp" in stats
    assert "Reached_Grasped" in stats
    assert "pre_grasp_rate" in results["summary"]
    assert "grasped_rate" in results["summary"]
    assert "final_phase_counts" in results["summary"]
