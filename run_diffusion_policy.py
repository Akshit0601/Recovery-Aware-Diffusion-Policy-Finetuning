#!/usr/bin/env python
"""
Run and evaluate the trained Diffusion Policy on Robomimic / Robosuite Lift environment.

Default checkpoint:
    /home/akshit/ece285_proj/robomimic/diffusion_policy_trained_models/test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth

Example usage:
    # 1. Quick evaluation (10 rollouts)
    python run_diffusion_policy.py --n_rollouts 10

    # 2. Record video with wider FOV and camera pushed farther away
    python run_diffusion_policy.py --n_rollouts 5 --video_path rollout_wide.mp4 --camera_dist_scale 1.8 --camera_fovy 70

    # 3. Use an alternative built-in wide camera (e.g. frontview or birdview)
    python run_diffusion_policy.py --n_rollouts 5 --video_path rollout_front.mp4 --camera_names frontview

    # 4. On-screen visualization (requires display/GUI)
    python run_diffusion_policy.py --n_rollouts 3 --render --camera_dist_scale 1.5

    # 5. Save rollouts to an HDF5 dataset
    python run_diffusion_policy.py --n_rollouts 10 --dataset_path evaluated_rollouts.hdf5
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
from tqdm import tqdm

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import RolloutPolicy
from robomimic.envs.env_base import EnvBase
from robomimic.envs.wrappers import EnvWrapper

DEFAULT_CKPT = "/home/akshit/ece285_proj/robomimic/diffusion_policy_trained_models/test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth"


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
            # print(f"the rolled out action is {act}")

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


def run_diffusion_eval(args):
    ckpt_path = args.agent
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint file not found at {ckpt_path}")
        sys.exit(1)

    write_video = args.video_path is not None
    if args.render and write_video:
        raise ValueError("Cannot render on-screen and record video at the same time.")

    print(f"\n{'='*70}")
    print(f"Loading Diffusion Policy Checkpoint:")
    print(f"  {ckpt_path}")
    print(f"{'='*70}")

    device = TorchUtils.get_torch_device(try_to_use_cuda=(not args.cpu))

    # Restore policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=args.verbose
    )

    # Determine rollout horizon
    rollout_horizon = args.horizon
    if rollout_horizon is None:
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        rollout_horizon = config.experiment.rollout.horizon

        print(f"Rollout horizon (from config): {rollout_horizon}")
    else:
        print(f"Rollout horizon (override): {rollout_horizon}")

    # Build environment from saved checkpoint
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        env_name=args.env,
        render=args.render,
        render_offscreen=write_video,
        verbose=args.verbose,
    )

    # Configure object placement sampler (Option 2)
    robosuite_env = env.env.env if hasattr(env, "env") and hasattr(env.env, "env") else None
    if robosuite_env is not None and hasattr(robosuite_env, "placement_initializer"):
        sampler = robosuite_env.placement_initializer

        # If a fixed 2D position (x, y) was requested
        if args.object_pos is not None:
            sampler.x_range = [args.object_pos[0], args.object_pos[0]]
            sampler.y_range = [args.object_pos[1], args.object_pos[1]]
            print(f"Set fixed object position: X={args.object_pos[0]}, Y={args.object_pos[1]}")
        else:
            if args.object_x_range is not None:
                sampler.x_range = list(args.object_x_range)
                print(f"Set object X sampling range: {sampler.x_range}")
            if args.object_y_range is not None:
                sampler.y_range = list(args.object_y_range)
                print(f"Set object Y sampling range: {sampler.y_range}")

        if args.object_ref_pos is not None:
            sampler.reference_pos = np.array(args.object_ref_pos, dtype=float)
            print(f"Set object reference position: {sampler.reference_pos}")
    else:
        if args.object_pos or args.object_x_range or args.object_y_range:
            print("Warning: placement_initializer not found on environment. Skipping object range modification.")

    # Configure camera position and FOV (push camera farther away / wider view)
    if robosuite_env is not None and hasattr(robosuite_env, "sim"):
        sim = robosuite_env.sim
        for cam_name in args.camera_names:
            try:
                cam_id = sim.model.camera_name2id(cam_name)
            except Exception:
                continue

            # Modify Field of View (fovy)
            if args.camera_fovy is not None:
                sim.model.cam_fovy[cam_id] = float(args.camera_fovy)
                print(f"Adjusted camera '{cam_name}' FOV (fovy) to: {args.camera_fovy}° (default is 45°)")

            # Move camera farther away
            if args.camera_pos is not None:
                sim.model.cam_pos[cam_id] = np.array(args.camera_pos, dtype=float)
                print(f"Adjusted camera '{cam_name}' position to: {sim.model.cam_pos[cam_id]}")
            elif args.camera_dist_scale is not None and args.camera_dist_scale != 1.0:
                # Table center reference point
                target = np.array([0.0, 0.0, 0.8])
                current_pos = np.array(sim.model.cam_pos[cam_id], dtype=float)
                new_pos = target + (current_pos - target) * float(args.camera_dist_scale)
                sim.model.cam_pos[cam_id] = new_pos
                print(
                    f"Pushed camera '{cam_name}' {args.camera_dist_scale}x farther away: "
                    f"pos {current_pos.round(3)} -> {new_pos.round(3)}"
                )

    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Video setup
    video_writer = None
    if write_video:
        video_dir = os.path.dirname(args.video_path)
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)
        video_writer = imageio.get_writer(args.video_path, fps=20)
        print(f"Recording video to: {args.video_path}")

    # HDF5 dataset setup
    write_dataset = args.dataset_path is not None
    if write_dataset:
        data_dir = os.path.dirname(args.dataset_path)
        if data_dir:
            os.makedirs(data_dir, exist_ok=True)
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
        print(f"Saving rollout trajectories to: {args.dataset_path}")

    print(f"\nStarting {args.n_rollouts} rollouts...\n")

    rollout_stats = []
    pbar = tqdm(range(args.n_rollouts), desc="Evaluating Diffusion Policy")
    for i in pbar:
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=args.render,
            video_writer=video_writer,
            video_skip=args.video_skip,
            return_obs=(write_dataset and args.dataset_obs),
            camera_names=args.camera_names,
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
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset(f"obs/{k}", data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset(f"next_obs/{k}", data=np.array(traj["next_obs"][k]))

            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"]
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
            total_samples += traj["actions"].shape[0]

    if video_writer is not None:
        video_writer.close()
        print(f"\nVideo saved to: {args.video_path}")

    if write_dataset:
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4)
        data_writer.close()
        print(f"\nDataset saved to: {args.dataset_path}")

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
        description="Evaluate trained Diffusion Policy checkpoint on Robosuite environment."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_CKPT,
        help="Path to trained checkpoint .pth file (defaults to your epoch 150 diffusion policy)",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=20,
        help="Number of rollouts to evaluate (default: 20)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Override rollout horizon (default: read from model config, e.g. 400)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Override environment name if different from checkpoint",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render rollouts on-screen (requires display)",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save video of rollouts (.mp4)",
    )
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="Render video every N steps (default: 5)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["agentview"],
        help="Camera name(s) for rendering (choices: agentview, frontview, birdview, sideview; default: agentview)",
    )
    parser.add_argument(
        "--camera_dist_scale",
        type=float,
        default=None,
        help="Multiplier to push the camera farther away from the table center (e.g. 1.5, 2.0; default: 1.0)",
    )
    parser.add_argument(
        "--camera_fovy",
        type=float,
        default=None,
        help="Field of view angle in degrees to widen camera view (e.g. 70 or 80; default: 45)",
    )
    parser.add_argument(
        "--camera_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Explicit camera position [X, Y, Z] in 3D world space",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to save rollout trajectories as an HDF5 dataset (.hdf5)",
    )
    parser.add_argument(
        "--dataset_obs",
        action="store_true",
        help="Include observations in the saved HDF5 dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for evaluation (default: 0)",
    )
    # Object placement modification options (Option 2)
    parser.add_argument(
        "--object_pos",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="Fix object spawn position to specific table coordinates (e.g. --object_pos 0.05 -0.02)",
    )
    parser.add_argument(
        "--object_x_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Set custom X sampling range for object placement (default: -0.03 0.03)",
    )
    parser.add_argument(
        "--object_y_range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Set custom Y sampling range for object placement (default: -0.03 0.03)",
    )
    parser.add_argument(
        "--object_ref_pos",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Override table reference center position [X, Y, Z] (default: 0.0 0.0 0.8)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution instead of CUDA GPU",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose environment and policy loading details",
    )

    args = parser.parse_args()
    run_diffusion_eval(args)
