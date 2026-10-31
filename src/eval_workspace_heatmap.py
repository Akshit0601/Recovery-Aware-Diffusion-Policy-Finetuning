#!/usr/bin/env python
"""
Workspace Grasp & Task Failure Heatmap Evaluator (Config-Free).

Evaluates trained Diffusion Policy across a 2D spatial grid of object spawn poses
and multiple yaw orientations (e.g. 0°, 30°, 45°) directly using DiffusionPolicyRunner's
lifecycle hooks and override methods, without requiring any external YAML config file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.diffusion_policy_runner import DiffusionPolicyRunner

# ==============================================================================
# Evaluation Parameters (Edit directly here)
# ==============================================================================
X_MIN = -0.15               # Minimum X coordinate in meters
X_MAX = 0.15                # Maximum X coordinate in meters
Y_MIN = -0.15               # Minimum Y coordinate in meters
Y_MAX = 0.15                # Maximum Y coordinate in meters
RESOLUTION = 5             # Grid resolution N along each axis (N x N spatial positions)
YAW: Union[float, List[float]] = [0.0, 30.0, 45.0]  # Yaw angle(s) in degrees: e.g. 0.0 or [0.0, 30.0, 45.0]
OUTPUT_DIR = "results/heatmaps"  # Directory to save heatmaps and JSON statistics

DEFAULT_CHECKPOINT_PATH = (
    "/home/akshit/ece285_proj/robomimic/diffusion_policy_trained_models/"
    "test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth"
)


class WorkspaceHeatmapEvaluator(DiffusionPolicyRunner):
    """
    Evaluator that sweeps object poses (X, Y, Yaw) across the robot workspace
    by overriding DiffusionPolicyRunner's reset and placement lifecycle hooks.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
        x_range: Tuple[float, float] = (-0.15, 0.15),
        y_range: Tuple[float, float] = (-0.15, 0.15),
        yaw_degrees: List[float] = (0.0, 30.0, 45.0),
        resolution: int = 5,
        trials_per_pose: int = 1,
        horizon: int = 200,
        output_dir: str = "results/heatmaps",
        cpu: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the evaluator purely in Python without requiring a YAML file.

        Args:
            checkpoint_path: Path to trained .pth model checkpoint.
            x_range: (min_x, max_x) workspace interval in meters.
            y_range: (min_y, max_y) workspace interval in meters.
            yaw_degrees: List of yaw rotation angles in degrees (e.g. [0, 30, 45]).
            resolution: Grid resolution N along each axis (N x N positions).
            trials_per_pose: Number of rollout attempts evaluated at each pose.
            horizon: Maximum simulation steps per rollout episode.
            output_dir: Output folder for heatmaps and JSON statistics.
            cpu: Whether to force CPU execution.
            verbose: Enable verbose logging.
        """
        # Construct configuration purely in-memory (bypassing external YAML file)
        in_memory_config = {
            "weights": {
                "checkpoint_path": checkpoint_path,
            },
            "rollout": {
                "n_rollouts": 1,
                "horizon": horizon,
                "seed": 0,
            },
            "env": {
                "env_name": None,
            },
            "rendering": {
                "render": False,
                "video_path": None,
                "video_skip": 5,
            },
            "camera": {
                "camera_names": ["robot0_eye_in_hand"],
            },
            "object_placement": {},
            "dataset": {
                "dataset_path": None,
                "dataset_obs": False,
            },
            "system": {
                "cpu": cpu,
                "verbose": verbose,
            },
        }

        super().__init__(config=in_memory_config)

        self.x_range = tuple(x_range)
        self.y_range = tuple(y_range)
        self.yaw_degrees = [float(y) for y in yaw_degrees]
        self.resolution = int(resolution)
        self.trials_per_pose = int(trials_per_pose)
        self.horizon = int(horizon)
        self.output_dir = output_dir

        # Current target pose state for hook overrides
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_yaw_deg: float = 0.0

        # Storage for all evaluation results: {yaw_deg: failure_matrix}
        self.results_by_yaw: Dict[float, np.ndarray] = {}
        self.contact_by_yaw: Dict[float, np.ndarray] = {}
        self.detailed_records: List[Dict[str, Any]] = []

    def set_target_pose(self, x: float, y: float, yaw_deg: float) -> None:
        """Set the target object pose for subsequent rollouts."""
        self.current_x = float(x)
        self.current_y = float(y)
        self.current_yaw_deg = float(yaw_deg)

    def reset_env(
        self, env=None, policy=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Override reset_env lifecycle hook to inject custom (X, Y, Yaw) pose
        into the placement sampler before the environment resets.
        """
        active_env = env if env is not None else self.env
        robosuite_env = (
            active_env.env.env
            if hasattr(active_env, "env") and hasattr(active_env.env, "env")
            else None
        )

        if robosuite_env is not None and hasattr(robosuite_env, "placement_initializer"):
            sampler = robosuite_env.placement_initializer
            # Fix X and Y coordinates to current target
            sampler.x_range = [self.current_x, self.current_x]
            sampler.y_range = [self.current_y, self.current_y]
            # Set exact yaw orientation around Z axis
            sampler.rotation = float(np.deg2rad(self.current_yaw_deg))
            sampler.rotation_axis = "z"

        return super().reset_env(env=active_env, policy=policy)

    def evaluate_single_pose(
        self, x: float, y: float, yaw_deg: float
    ) -> Dict[str, Any]:
        """
        Evaluate policy rollouts at a specific (X, Y, Yaw) pose.

        Returns:
            Dict containing pose coordinates, success_rate, failure_rate, contact_rate.
        """
        self.set_target_pose(x=x, y=y, yaw_deg=yaw_deg)

        successes = 0
        contacts = 0
        returns: List[float] = []
        horizons: List[int] = []

        robosuite_env = (
            self.env.env.env
            if hasattr(self.env, "env") and hasattr(self.env.env, "env")
            else None
        )

        for trial_i in range(self.trials_per_pose):
            stats, _ = self.rollout_episode(
                episode_idx=trial_i,
                horizon=self.horizon,
                return_obs=False,
            )

            is_success = bool(stats["Success_Rate"] > 0.0)
            if is_success:
                successes += 1

            returns.append(float(stats["Return"]))
            horizons.append(int(stats["Horizon"]))

            # Check if gripper established physical contact with object
            made_contact = False
            if (
                robosuite_env is not None
                and hasattr(robosuite_env, "check_contact")
                and hasattr(robosuite_env, "cube")
            ):
                gripper = robosuite_env.robots[0].gripper
                made_contact = bool(
                    robosuite_env.check_contact(gripper, robosuite_env.cube)
                )
            if made_contact or is_success:
                contacts += 1

        success_rate = successes / self.trials_per_pose
        failure_rate = 1.0 - success_rate
        contact_rate = contacts / self.trials_per_pose

        return {
            "x": float(x),
            "y": float(y),
            "yaw_deg": float(yaw_deg),
            "trials": self.trials_per_pose,
            "successes": successes,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "contact_rate": contact_rate,
            "mean_return": float(np.mean(returns)),
            "mean_horizon": float(np.mean(horizons)),
        }

    def evaluate_all(self) -> Dict[float, np.ndarray]:
        """
        Run grid evaluation across all (X, Y) positions and all configured Yaw angles.

        Returns:
            Dict mapping each yaw_deg to its 2D failure rate matrix.
        """
        self.setup()

        xs = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        ys = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        total_poses_per_yaw = self.resolution * self.resolution
        total_evaluations = len(self.yaw_degrees) * total_poses_per_yaw

        print(f"\n{'='*75}")
        print("WORKSPACE GRASP FAILURE EVALUATION (CONFIG-FREE)")
        print(f"{'='*75}")
        print(f"Checkpoint        : {self.weights_cfg.get('checkpoint_path')}")
        print(f"X Span            : [{self.x_range[0]:.3f}, {self.x_range[1]:.3f}] m")
        print(f"Y Span            : [{self.y_range[0]:.3f}, {self.y_range[1]:.3f}] m")
        print(f"Yaw Angles        : {self.yaw_degrees}°")
        print(f"Grid Resolution   : {self.resolution} x {self.resolution} ({total_poses_per_yaw} spatial positions)")
        print(f"Trials per Pose   : {self.trials_per_pose}")
        print(f"Rollout Horizon   : {self.horizon} steps")
        print(f"Total Rollouts    : {total_evaluations * self.trials_per_pose}")
        print(f"{'='*75}\n")

        self.results_by_yaw = {}
        self.contact_by_yaw = {}
        self.detailed_records = []

        total_pbar = tqdm(total=total_evaluations, desc="Evaluating (X, Y, Yaw) Poses")

        for yaw in self.yaw_degrees:
            fail_mat = np.zeros((self.resolution, self.resolution), dtype=float)
            cont_mat = np.zeros((self.resolution, self.resolution), dtype=float)

            for i, y in enumerate(reversed(ys)):
                row_idx = self.resolution - 1 - i
                for j, x in enumerate(xs):
                    col_idx = j

                    record = self.evaluate_single_pose(x=x, y=y, yaw_deg=yaw)
                    record["grid_row"] = row_idx
                    record["grid_col"] = col_idx

                    self.detailed_records.append(record)
                    fail_mat[row_idx, col_idx] = record["failure_rate"]
                    cont_mat[row_idx, col_idx] = record["contact_rate"]

                    fail_pct = record["failure_rate"] * 100.0
                    total_pbar.set_postfix(
                        yaw=f"{yaw:g}°",
                        x=f"{x:+.2f}",
                        y=f"{y:+.2f}",
                        fail=f"{fail_pct:.0f}%",
                    )
                    total_pbar.update(1)

            self.results_by_yaw[yaw] = fail_mat
            self.contact_by_yaw[yaw] = cont_mat

        total_pbar.close()
        return self.results_by_yaw

    def plot_heatmaps(
        self,
        comparison_filename: str = "workspace_failure_heatmaps_yaws.png",
    ) -> List[str]:
        """
        Plot and save a multi-panel comparison figure across all yaw orientations,
        as well as individual high-res heatmaps for each yaw.

        Returns:
            List of saved image file paths.
        """
        if not self.results_by_yaw:
            raise RuntimeError("No evaluation results available. Call evaluate_all() first.")

        os.makedirs(self.output_dir, exist_ok=True)
        saved_paths: List[str] = []

        xs = np.linspace(self.x_range[0], self.x_range[1], self.resolution)
        ys = np.linspace(self.y_range[0], self.y_range[1], self.resolution)
        cmap = plt.cm.RdYlGn_r

        # Nominal training distribution box (Lift default: [-0.03, 0.03])
        train_x = [-0.03, 0.03, 0.03, -0.03, -0.03]
        train_y = [-0.03, -0.03, 0.03, 0.03, -0.03]

        n_yaws = len(self.yaw_degrees)
        include_average = n_yaws > 1
        n_panels = n_yaws + (1 if include_average else 0)

        fig, axes = plt.subplots(
            1, n_panels, figsize=(5.5 * n_panels, 5.0), dpi=300, squeeze=False, layout="constrained"
        )
        axes = axes[0]

        # 1. Plot each yaw panel
        for idx, yaw in enumerate(self.yaw_degrees):
            ax = axes[idx]
            fail_mat = self.results_by_yaw[yaw]

            im = ax.imshow(
                fail_mat,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=[self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]],
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )

            # Annotate cell values
            for rec in self.detailed_records:
                if abs(rec["yaw_deg"] - yaw) < 1e-4:
                    val = rec["failure_rate"]
                    text_color = "white" if val > 0.65 or val < 0.25 else "black"
                    ax.text(
                        rec["x"],
                        rec["y"],
                        f"{val*100:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

            ax.plot(train_x, train_y, "b--", linewidth=1.5, label="Train Dist (±0.03m)")
            ax.set_title(f"Yaw = {yaw:g}°", fontsize=13, fontweight="bold", pad=8)
            ax.set_xlabel("X Position (m)", fontsize=11)
            if idx == 0:
                ax.set_ylabel("Y Position (m)", fontsize=11)
            ax.grid(color="gray", linestyle=":", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # 2. Plot overall mean across all yaws
        if include_average:
            ax_avg = axes[-1]
            all_matrices = [self.results_by_yaw[y] for y in self.yaw_degrees]
            mean_mat = np.mean(all_matrices, axis=0)

            im = ax_avg.imshow(
                mean_mat,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=[self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]],
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )

            # Annotate average cell values
            for row in range(self.resolution):
                for col in range(self.resolution):
                    val = mean_mat[row, col]
                    pos_x = xs[col]
                    pos_y = ys[row]
                    text_color = "white" if val > 0.65 or val < 0.25 else "black"
                    ax_avg.text(
                        pos_x,
                        pos_y,
                        f"{val*100:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        fontweight="bold",
                    )

            ax_avg.plot(train_x, train_y, "b--", linewidth=1.5, label="Train Dist (±0.03m)")
            ax_avg.set_title("Mean Across All Yaws", fontsize=13, fontweight="bold", pad=8)
            ax_avg.set_xlabel("X Position (m)", fontsize=11)
            ax_avg.grid(color="gray", linestyle=":", alpha=0.3)
            ax_avg.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # Colorbar
        cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.03)
        cbar.set_label("Failure Rate (1.0 = 100% Fail)", fontsize=11, labelpad=8)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.ax.set_yticklabels(["0% (Pass)", "25%", "50%", "75%", "100% (Fail)"])

        fig.suptitle(
            f"Diffusion Policy Grasp & Task Failure Rates Across Workspace\n"
            f"({self.resolution}x{self.resolution} Grid, {self.trials_per_pose} Trial(s)/Pose)",
            fontsize=15,
            fontweight="bold",
            y=1.03,
        )

        comparison_path = os.path.join(self.output_dir, comparison_filename)
        plt.savefig(comparison_path, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(comparison_path)
        print(f"\nSaved combined comparison heatmap to: {comparison_path}")

        # 3. Save individual figures for each yaw
        for yaw in self.yaw_degrees:
            indiv_path = os.path.join(
                self.output_dir, f"failure_heatmap_yaw_{int(yaw)}deg.png"
            )
            fig_indiv, ax_indiv = plt.subplots(figsize=(7, 6), dpi=300)
            im_indiv = ax_indiv.imshow(
                self.results_by_yaw[yaw],
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                extent=[self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]],
                origin="lower",
                interpolation="nearest",
                aspect="equal",
            )
            cbar_indiv = fig_indiv.colorbar(im_indiv, ax=ax_indiv, fraction=0.046, pad=0.04)
            cbar_indiv.set_label("Failure Rate", fontsize=11)

            for rec in self.detailed_records:
                if abs(rec["yaw_deg"] - yaw) < 1e-4:
                    val = rec["failure_rate"]
                    text_color = "white" if val > 0.65 or val < 0.25 else "black"
                    ax_indiv.text(
                        rec["x"],
                        rec["y"],
                        f"{val*100:.0f}%",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="bold",
                    )

            ax_indiv.plot(train_x, train_y, "b--", linewidth=1.8, label="Train Dist (±0.03m)")
            ax_indiv.set_title(
                f"Grasp & Task Failure Rate (Yaw = {yaw:g}°)",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            ax_indiv.set_xlabel("Object X Position (m)", fontsize=11)
            ax_indiv.set_ylabel("Object Y Position (m)", fontsize=11)
            ax_indiv.grid(color="gray", linestyle=":", alpha=0.3)
            ax_indiv.legend(loc="upper right", fontsize=9)

            plt.tight_layout()
            plt.savefig(indiv_path, bbox_inches="tight")
            plt.close(fig_indiv)
            saved_paths.append(indiv_path)

        return saved_paths

    def save_results_json(self, filename: str = "workspace_failure_results.json") -> str:
        """Export all evaluation results and pose metadata to JSON."""
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, filename)

        summary_by_yaw = {}
        for yaw, mat in self.results_by_yaw.items():
            summary_by_yaw[f"{yaw:g}_deg"] = {
                "mean_failure_rate": float(np.mean(mat)),
                "max_failure_rate": float(np.max(mat)),
                "min_failure_rate": float(np.min(mat)),
            }

        data = {
            "metadata": {
                "checkpoint": self.weights_cfg.get("checkpoint_path"),
                "x_range": list(self.x_range),
                "y_range": list(self.y_range),
                "yaw_degrees": self.yaw_degrees,
                "resolution": self.resolution,
                "trials_per_pose": self.trials_per_pose,
                "horizon": self.horizon,
                "summary_by_yaw": summary_by_yaw,
            },
            "records": self.detailed_records,
        }

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved evaluation metrics to: {out_path}")
        return out_path


def main():
    yaws = [float(YAW)] if isinstance(YAW, (int, float)) else [float(y) for y in YAW]

    evaluator = WorkspaceHeatmapEvaluator(
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        x_range=(X_MIN, X_MAX),
        y_range=(Y_MIN, Y_MAX),
        yaw_degrees=yaws,
        resolution=RESOLUTION,
        trials_per_pose=1,
        horizon=200,
        output_dir=OUTPUT_DIR,
    )

    evaluator.evaluate_all()
    evaluator.plot_heatmaps()
    evaluator.save_results_json()


if __name__ == "__main__":
    main()
