# Diffusion Policy Runner (`DiffusionPolicyRunner`)

A modular, extensible base class for loading, configuring, running, and evaluating trained **Diffusion Policies** on Robomimic / Robosuite environments.

`DiffusionPolicyRunner` provides a decoupled architecture with customizable lifecycle hooks, allowing researchers and developers to easily subclass it for:
- **Perturbation & Recovery Benchmarking**: Injecting action or state disturbances to evaluate policy recovery behavior.
- **Domain Randomization & Sweeps**: Randomizing camera parameters, object spawn coordinates, or physics dynamics across runs.
- **Custom Environment Wrappers**: Wrapping simulation environments with custom observation spaces or safety filters.
- **Evaluation & Trajectory Logging**: Customizing metrics, logging step-by-step telemetries, and exporting HDF5 datasets or video recordings.

---

## Quick Reference

- **Module**: [`src/diffusion_policy_runner.py`](diffusion_policy_runner.py)
- **Import Aliases**:
  ```python
  from src import DiffusionPolicyRunner, BaseRunner, BaseDiffusionPolicyRunner
  ```

---

## Architecture & Lifecycle

The runner divides execution into modular setup, rollout, and teardown stages:

```
[ Initialization: __init__(config, config_path, overrides) ]
                           │
                           ▼
                    [ setup() ]
   ┌───────────────────────┴────────────────────────┐
   │ 1. setup_device()          5. setup_env()      │
   │ 2. setup_seed()            6. setup_object_pos()
   │ 3. load_policy()           7. setup_camera()   │
   │ 4. setup_horizon()         8. setup_writers()  │
   └───────────────────────┬────────────────────────┘
                           │
                           ▼
               [ run() / rollout loop ]
                  before_rollouts()
                           │
       ┌───────────────────┴───────────────────┐
       │ For each episode (0 .. n_rollouts-1): │
       │   on_episode_start(episode_idx)       │
       │   rollout_episode(episode_idx)        │
       │     ├─ reset_env()                    │
       │     └─ For step_i in horizon:         │
       │          ├─ get_action(obs, step_i)   │
       │          ├─ step_env(action)          │
       │          ├─ check_success(...)        │
       │          ├─ on_step_render(...)       │
       │          ├─ on_step_end(...)          │
       │          └─ should_terminate(...)     │
       │   record_episode_data() (HDF5)        │
       │   on_episode_end(stats, traj)         │
       └───────────────────┬───────────────────┘
                           │
                  after_rollouts()
                           │
                           ▼
             [ compute_summary() & print_summary() ]
                           │
                           ▼
                       [ close() ]
     (Closes video writer, finalizes HDF5 datasets)
```

---

## Basic Usage

### 1. Run with a YAML Config File
```python
from src import DiffusionPolicyRunner

runner = DiffusionPolicyRunner(config_path="config/run_diffusion_policy.yaml")
results = runner.run()

print("Success Rate:", results["summary"]["success_rate"])
```

### 2. Run with Programmatic Overrides
You can override any nested configuration parameter without modifying the YAML file:
```python
from src import DiffusionPolicyRunner

runner = DiffusionPolicyRunner(
    config_path="config/run_diffusion_policy.yaml",
    overrides={
        "rollout": {
            "n_rollouts": 5,
            "horizon": 200,
        },
        "rendering": {
            "render": False,
            "video_path": "results/eval_5_episodes.mp4",
        },
        "camera": {
            "camera_names": ["agentview", "frontview"],
        },
    },
)
results = runner.run()
```

### 3. Run with a Pure Python Dictionary
```python
from src import DiffusionPolicyRunner

custom_cfg = {
    "weights": {
        "checkpoint_path": "/path/to/model.pth"
    },
    "rollout": {
        "n_rollouts": 10,
        "seed": 42
    },
    "rendering": {
        "render": False
    }
}

runner = DiffusionPolicyRunner(config=custom_cfg)
results = runner.run()
```

### 4. Direct Execution with Policy & Env (Without a Config File)
If you already instantiated a policy and environment in your code, `DiffusionPolicyRunner` and `rollout_episode` can run them directly without needing a YAML file:
```python
from src import DiffusionPolicyRunner

# 1. Via runner instance
runner = DiffusionPolicyRunner(policy=my_policy, env=my_env)
stats, traj = runner.rollout_episode(horizon=400, render=True)

# 2. Or using dynamic argument overrides on an existing runner
stats, traj = runner.rollout_episode(
    policy=another_policy,
    env=another_env,
    horizon=200,
    camera_names=["frontview"],
)
```

### 5. Context Manager (Safe Resource Teardown)
```python
from src import DiffusionPolicyRunner

with DiffusionPolicyRunner(config_path="config/run_diffusion_policy.yaml") as runner:
    results = runner.run(n_rollouts=5, collect_trajectories=True)
    trajectories = results["trajectories"]
```

### 5. CLI Execution
`DiffusionPolicyRunner` provides command-line support out-of-the-box:
```bash
# Using default config (config/run_diffusion_policy.yaml)
python src/diffusion_policy_runner.py

# Using explicit config file
python src/diffusion_policy_runner.py --config config/custom_eval.yaml
# or
python src/diffusion_policy_runner.py config/custom_eval.yaml
```

---

## Subclassing Guide & Extension Hooks

Subclasses can override specific hooks to tailor behavior without rewriting boilerplate logic.

### 1. Perturbations and Action Interventions (`get_action`)
Override `get_action` to perturb actions, inject noise, evaluate out-of-distribution robustness, or implement recovery behaviors:

```python
import numpy as np
from src import DiffusionPolicyRunner

class PerturbedRunner(DiffusionPolicyRunner):
    def __init__(self, perturbation_step=30, magnitude=0.2, **kwargs):
        super().__init__(**kwargs)
        self.perturbation_step = perturbation_step
        self.magnitude = magnitude

    def get_action(self, obs, step_i=0):
        # Obtain action predicted by the diffusion policy
        action = super().get_action(obs, step_i)

        # Inject sudden delta perturbation to arm EEF
        if step_i == self.perturbation_step:
            action[:3] += np.random.uniform(-self.magnitude, self.magnitude, size=3)
            print(f"[Perturbation] Applied disturbance at step {step_i}")

        return action
```

### 2. Custom Environment Wrapping (`setup_env`)
Override `setup_env` to wrap the environment (e.g., custom Gym wrappers, observation noise, or domain shifts):

```python
from src import DiffusionPolicyRunner

class CustomWrappedRunner(DiffusionPolicyRunner):
    def setup_env(self, force=False):
        # Setup base robomimic environment
        env = super().setup_env(force=force)
        # Apply custom wrapper
        # env = MyCustomObservationWrapper(env)
        return env
```

### 3. Step Telemetry and Intermediate Monitoring (`on_step_end`)
Track custom metrics, distances, or state values at each simulation step:

```python
from src import DiffusionPolicyRunner

class TrackingRunner(DiffusionPolicyRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step_rewards = []

    def on_step_end(self, step_i, obs, action, next_obs, reward, done, success, info):
        self.step_rewards.append(reward)
```

### 4. Phase Monitoring & Transition Interventions (`on_phase_transition`)
`DiffusionPolicyRunner` tracks episode progress through three distinct execution phases:
- `approaching` (`PHASE_APPROACHING`): The policy is running arm motion to approach the object (gripper open).
- `pre-grasp` (`PHASE_PRE_GRASP`): The policy outputs only gripper actions (gripper closing command while arm translation/rotation deltas are stationary/minimal).
- `grasped` (`PHASE_GRASPED`): The environment reward reaches 1.0 (object grasped and lifted).

You can override `on_phase_transition` to react dynamically at phase boundaries (e.g. inject a slip perturbation right as the robot reaches pre-grasp):

```python
from src import DiffusionPolicyRunner, PHASE_PRE_GRASP, PHASE_GRASPED

class SlipPerturbationRunner(DiffusionPolicyRunner):
    def on_phase_transition(self, prev_phase, new_phase, step_i, action, reward):
        super().on_phase_transition(prev_phase, new_phase, step_i, action, reward)
        print(f"[Phase Shift] Step {step_i:2d}: {prev_phase} -> {new_phase}")

        if new_phase == PHASE_PRE_GRASP:
            print(f"Robot entered pre-grasp at step {step_i}!")
```

### 5. Custom Evaluation Metrics (`compute_summary` & `print_summary`)
Augment evaluation output with domain-specific metrics (e.g., recovery time, trajectory jerk, gripper slip):

```python
import numpy as np
from src import DiffusionPolicyRunner

class AdvancedMetricsRunner(DiffusionPolicyRunner):
    def compute_summary(self, rollout_stats):
        summary = super().compute_summary(rollout_stats)
        # Add custom metrics calculated across all rollouts
        returns = [s["Return"] for s in rollout_stats]
        summary["min_return"] = float(np.min(returns)) if returns else 0.0
        summary["max_return"] = float(np.max(returns)) if returns else 0.0
        return summary

    def print_summary(self, summary):
        super().print_summary(summary)
        print(f"Min Return        : {summary['min_return']:.3f}")
        print(f"Max Return        : {summary['max_return']:.3f}")
```

---

## Configuration Reference

The configuration dictionary loaded by `DiffusionPolicyRunner` is divided into logical sections:

| Section | Key | Type | Description |
| :--- | :--- | :--- | :--- |
| `weights` | `checkpoint_path` | `str` | Path to trained `.pth` policy weights |
| `rollout` | `n_rollouts` | `int` | Number of evaluation episodes |
| | `horizon` | `int \| null` | Episode horizon (`null` uses checkpoint's config) |
| | `seed` | `int` | Random seed for NumPy & PyTorch |
| `env` | `env_name` | `str \| null` | Environment name override |
| `rendering` | `render` | `bool` | Real-time on-screen rendering |
| | `video_path` | `str \| null` | Output path for `.mp4` video recording |
| | `video_skip` | `int` | Frame capture interval (every N steps) |
| `camera` | `camera_names` | `list[str]` | Cameras to render (`agentview`, `frontview`, etc.) |
| | `camera_dist_scale` | `float \| null`| Multiplier to scale camera distance from table |
| | `camera_fovy` | `float \| null`| Field of view in degrees |
| | `camera_pos` | `list[float] \| null` | Explicit camera coordinates `[X, Y, Z]` |
| `object_placement` | `object_pos` | `list[float] \| null` | Fixed object spawn `[X, Y]` |
| | `object_x_range` | `list[float] \| null` | X-axis sampling interval `[min, max]` |
| | `object_y_range` | `list[float] \| null` | Y-axis sampling interval `[min, max]` |
| | `object_ref_pos` | `list[float] \| null` | Reference table coordinate `[X, Y, Z]` |
| `dataset` | `dataset_path` | `str \| null` | Path to export rollouts as HDF5 (`.hdf5`) |
| | `dataset_obs` | `bool` | Whether to record observations in HDF5 |
| `system` | `cpu` | `bool` | Force CPU execution instead of CUDA GPU |
| | `verbose` | `bool` | Enable verbose policy / environment loading logs |

---

## Public Method & Hook Reference

### Setup Methods
- `setup(force=False)`: Executes all setup stages.
- `setup_device()`: Configures CUDA / CPU device.
- `setup_seed()`: Seeds NumPy and PyTorch.
- `load_policy(force=False)`: Restores `RolloutPolicy` and checkpoint dict.
- `setup_horizon()`: Determines rollout step horizon.
- `setup_env(force=False)`: Builds Robomimic / Robosuite environment.
- `setup_object_placement()`: Applies object placement randomizer settings.
- `setup_camera()`: Applies camera positioning and FOV settings.
- `setup_writers()`: Initializes video and HDF5 writers.

### Rollout & Step Hooks
- `reset_env()`: Resets policy and env to initial state; returns `(obs, state_dict)`.
- `get_action(obs, step_i=0)`: Queries policy for an action given observation `obs`.
- `step_env(action)`: Steps environment; returns `(next_obs, reward, done, info)`.
- `check_success(obs, reward, done, info)`: Evaluates task success boolean.
- `should_terminate(done, success, step_i)`: Returns whether to stop rollout early.
- `on_step_render(step_i, video_count, video_skip)`: Handles rendering and video capture.
- `on_episode_start(episode_idx)`: Hook executed before starting an episode.
- `on_step_end(step_i, obs, action, next_obs, reward, done, success, info)`: Step hook.
- `on_episode_end(episode_idx, stats, traj)`: Hook executed after episode completion.
- `before_rollouts()`: Hook executed before all rollouts begin.
- `after_rollouts(rollout_stats)`: Hook executed after all rollouts finish.

### Evaluation & Output
- `rollout_episode(episode_idx=0, horizon=None, return_obs=None)`: Runs single episode rollout.
- `run(n_rollouts=None, collect_trajectories=False)`: Runs the entire evaluation loop.
- `record_episode_data(episode_idx, traj)`: Writes trajectory to HDF5.
- `compute_summary(rollout_stats)`: Calculates success rate, returns, and horizons.
- `print_summary(summary)`: Prints formatted summary table.
- `close()`: Closes open file/video handles and writers.
