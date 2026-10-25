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

import argparse
import json
import os
import sys
from copy import deepcopy

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


def load_config(config_path):
    """Load and validate the YAML configuration file."""
    if not os.path.exists(config_path):
        # Fallback to repo root (one level up from src/)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        fallback = os.path.join(repo_root, config_path)
        if os.path.exists(fallback):
            config_path = fallback
        else:
            print(f"Error: configuration file not found at '{config_path}'")
            sys.exit(1)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid YAML config format in '{config_path}'. Expected a dictionary.")
    return cfg


def rollout(
    policy,
    env,
    horizon,
    render=False,
    video_writer=None,
    video_skip=5,
    return_obs=False,
    camera_names=None,
):
    """
    Execute a single episode rollout using the loaded diffusion policy.
    """
    assert isinstance(env, (EnvBase, EnvWrapper))
    assert isinstance(policy, RolloutPolicy)
    assert not (render and (video_writer is not None))

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()

    # Deterministic action playback reset for robosuite environments
    obs = env.reset_to(state_dict)

    video_count = 0
    total_reward = 0.0
    success = False
    step_i = 0

    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=state_dict)
    if return_obs:
        traj.update(dict(obs=[], next_obs=[]))

    try:
        for step_i in range(horizon):
            # Query action from the diffusion policy
            act = policy(ob=obs)

            # Step the environment
            next_obs, r, done, _ = env.step(act)

            total_reward += float(r)
            success = bool(env.is_success()["task"])

            # On-screen rendering
            if render:
                env.render(mode="human", camera_name=camera_names[0])

            # Video recording
            if video_writer is not None:
                if video_count % video_skip == 0:
                    video_img = []
                    for cam_name in camera_names:
                        video_img.append(
                            env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name)
                        )
                    video_img = np.concatenate(video_img, axis=1)
                    video_writer.append_data(video_img)
                video_count += 1

            # Collect transition
            traj["actions"].append(act)
            traj["rewards"].append(r)
            traj["dones"].append(done)
            traj["states"].append(state_dict["states"])
            if return_obs:
                traj["obs"].append(obs)
                traj["next_obs"].append(next_obs)

            # Break early upon task completion or terminal state
            if done or success:
                break

            obs = deepcopy(next_obs)
            state_dict = env.get_state()

    except env.rollout_exceptions as e:
        print(f"WARNING: rollout exception occurred: {e}")

    stats = dict(
        Return=round(total_reward, 3),
        Horizon=(step_i + 1),
        Success_Rate=float(success),
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

    return stats, traj


def run_diffusion_eval(cfg):
    """
    Run diffusion policy evaluation driven entirely by config dictionary.
    """
    weights_cfg = cfg.get("weights", {})
    rollout_cfg = cfg.get("rollout", {})
    env_cfg = cfg.get("env", {})
    render_cfg = cfg.get("rendering", {})
    camera_cfg = cfg.get("camera", {})
    obj_cfg = cfg.get("object_placement", {})
    dataset_cfg = cfg.get("dataset", {})
    system_cfg = cfg.get("system", {})

    # Weights / Checkpoint
    ckpt_path = weights_cfg.get("checkpoint_path") or cfg.get("agent")
    if not ckpt_path:
        raise ValueError("Missing 'checkpoint_path' under 'weights' in configuration.")
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint file not found at {ckpt_path}")
        sys.exit(1)

    # Rollout parameters
    n_rollouts = rollout_cfg.get("n_rollouts", 20)
    horizon_override = rollout_cfg.get("horizon", None)
    seed = rollout_cfg.get("seed", 0)

    # Environment
    env_name = env_cfg.get("env_name", None)

    # Rendering & Video
    render = render_cfg.get("render", False)
    video_path = render_cfg.get("video_path", None)
    video_skip = render_cfg.get("video_skip", 5)

    write_video = video_path is not None
    if render and write_video:
        raise ValueError("Cannot render on-screen and record video at the same time.")

    # Camera
    camera_names = camera_cfg.get("camera_names", ["agentview"])
    if isinstance(camera_names, str):
        camera_names = [camera_names]
    camera_dist_scale = camera_cfg.get("camera_dist_scale", None)
    camera_fovy = camera_cfg.get("camera_fovy", None)
    camera_pos = camera_cfg.get("camera_pos", None)

    # Object placement
    object_pos = obj_cfg.get("object_pos", None)
    object_x_range = obj_cfg.get("object_x_range", None)
    object_y_range = obj_cfg.get("object_y_range", None)
    object_ref_pos = obj_cfg.get("object_ref_pos", None)

    # Dataset
    dataset_path = dataset_cfg.get("dataset_path", None)
    dataset_obs = dataset_cfg.get("dataset_obs", False)

    # System
    cpu = system_cfg.get("cpu", False)
    verbose = system_cfg.get("verbose", False)

    print(f"\n{'='*70}")
    print(f"Loading Diffusion Policy Checkpoint:")
    print(f"  {ckpt_path}")
    print(f"{'='*70}")

    device = TorchUtils.get_torch_device(try_to_use_cuda=(not cpu))

    # Restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=verbose
    )

    # Determine rollout horizon
    rollout_horizon = horizon_override
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon
        print(f"Rollout horizon (from checkpoint config): {rollout_horizon}")
    else:
        print(f"Rollout horizon (override): {rollout_horizon}")

    # Build environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=env_name,
        render=render,
        render_offscreen=write_video,
        verbose=verbose,
    )

    # Configure object placement sampler
    robosuite_env = env.env.env if hasattr(env, "env") and hasattr(env.env, "env") else None
    if robosuite_env is not None and hasattr(robosuite_env, "placement_initializer"):
        sampler = robosuite_env.placement_initializer

        # If a fixed 2D position (x, y) was requested
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

    # Configure camera position and FOV
    if robosuite_env is not None and hasattr(robosuite_env, "sim"):
        sim = robosuite_env.sim
        for cam_name in camera_names:
            try:
                cam_id = sim.model.camera_name2id(cam_name)
            except Exception:
                continue

            # Modify Field of View (fovy)
            if camera_fovy is not None:
                sim.model.cam_fovy[cam_id] = float(camera_fovy)
                print(f"Adjusted camera '{cam_name}' FOV (fovy) to: {camera_fovy}° (default is 45°)")

            # Move camera farther away
            if camera_pos is not None:
                sim.model.cam_pos[cam_id] = np.array(camera_pos, dtype=float)
                print(f"Adjusted camera '{cam_name}' position to: {sim.model.cam_pos[cam_id]}")
            elif camera_dist_scale is not None and camera_dist_scale != 1.0:
                # Table center reference point
                target = np.array([0.0, 0.0, 0.8])
                current_pos = np.array(sim.model.cam_pos[cam_id], dtype=float)
                new_pos = target + (current_pos - target) * float(camera_dist_scale)
                sim.model.cam_pos[cam_id] = new_pos
                print(
                    f"Pushed camera '{cam_name}' {camera_dist_scale}x farther away: "
                    f"pos {current_pos.round(3)} -> {new_pos.round(3)}"
                )

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed set to: {seed}")

    # Video setup
    video_writer = None
    if write_video:
        video_dir = os.path.dirname(video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)
        print(f"Recording video to: {video_path}")

    # HDF5 dataset setup
    write_dataset = dataset_path is not None
    if write_dataset:
        data_dir = os.path.dirname(dataset_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        data_writer = h5py.File(dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
        print(f"Saving rollout trajectories to: {dataset_path}")

    print(f"\nStarting {n_rollouts} rollouts...\n")

    rollout_stats = []
    pbar = tqdm(range(n_rollouts), desc="Evaluating Diffusion Policy")
    for i in pbar:
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=render,
            video_writer=video_writer,
            video_skip=video_skip,
            return_obs=(write_dataset and dataset_obs),
            camera_names=camera_names,
        )
        rollout_stats.append(stats)

        current_success_count = sum(s["Success_Rate"] for s in rollout_stats)
        current_rate = current_success_count / len(rollout_stats) * 100.0
        pbar.set_postfix(
            success_rate=f"{current_rate:.1f}%",
            last_horizon=stats["Horizon"],
            last_return=stats["Return"],
        )

        if write_dataset:
            ep_data_grp = data_grp.create_group(f"demo_{i}")
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    if video_writer is not None:
        video_writer.close()
        print(f"\nVideo saved to: {video_path}")

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"\nDataset saved to: {dataset_path}")

    # Summary statistics
    stats_dict = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    successes = int(np.sum(stats_dict["Success_Rate"]))
    total = len(rollout_stats)
    success_rate = (successes / total) * 100.0
    mean_return = float(np.mean(stats_dict["Return"]))
    mean_horizon = float(np.mean(stats_dict["Horizon"]))

    print(f"\n{'='*70}")
    print("                  EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Rollouts    : {total}")
    print(f"Success Count     : {successes} / {total}")
    print(f"Success Rate      : {success_rate:.2f}%")
    print(f"Mean Return       : {mean_return:.3f}")
    print(f"Mean Horizon (len): {mean_horizon:.2f} steps")
    print(f"{'='*70}\n")


if __name__ == "__main__":
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

    args = parser.parse_args()
    config_file = args.config_pos if args.config_pos is not None else args.config

    config = load_config(config_file)
    run_diffusion_eval(config)
