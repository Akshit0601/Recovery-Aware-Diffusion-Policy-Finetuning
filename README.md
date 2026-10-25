# dp_env: Diffusion Policy on Robosuite Lift

Reproducible environment for evaluating and training Diffusion Policies on the Robomimic / Robosuite Lift task with MuJoCo and PyTorch (CUDA).

---

## Prerequisites

- **Linux** (x86_64) with NVIDIA GPU and CUDA drivers installed.
- **`uv`** package manager (fast, Rust-based Python package manager):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

---

## Quickstart Installation

1. **Clone the repository**:
   ```bash
   git clone <repo_url>
   cd dp_env
   ```

2. **Sync the environment** (single command):
   ```bash
   uv sync
   ```
   *`uv` automatically downloads standalone Python 3.10, creates the `.venv`, and installs all locked dependencies (PyTorch with CUDA 12.4, Robosuite 1.5.1, and Robomimic).*

---

## Configuration & Usage

Evaluation parameters are managed via a centralized YAML configuration file in [`config/run_diffusion_policy.yaml`](config/run_diffusion_policy.yaml).

### 1. Test Simulation Stack
Run a headless MuJoCo + Robosuite sanity test:
```bash
uv run python test/test_robosuite.py
```

### 2. Evaluate Trained Diffusion Policy
Run evaluation rollouts using the default configuration (`config/run_diffusion_policy.yaml`):
```bash
uv run python src/run_diffusion_policy.py
```

You can also pass an explicit configuration file via `--config` or as a positional argument:
```bash
uv run python src/run_diffusion_policy.py --config config/run_diffusion_policy.yaml
# or
uv run python src/run_diffusion_policy.py config/run_diffusion_policy.yaml
```

---

## Configuration Reference (`config/run_diffusion_policy.yaml`)

Edit [`config/run_diffusion_policy.yaml`](config/run_diffusion_policy.yaml) to customize evaluation behavior without passing complex command-line arguments:

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| **`weights`** | `checkpoint_path` | `.../model_epoch_150_...pth` | Path to trained `.pth` checkpoint weights |
| **`rollout`** | `n_rollouts` | `10` | Number of rollout episodes to evaluate |
| | `horizon` | `null` | Max steps per episode (`null` reads from checkpoint config) |
| | `seed` | `0` | Random seed for numpy and PyTorch reproducibility |
| **`env`** | `env_name` | `null` | Environment override (`null` uses checkpoint environment) |
| **`rendering`** | `render` | `false` | Real-time on-screen rendering (requires active display/GUI) |
| | `video_path` | `null` | Filepath to save video (`.mp4`), e.g., `"rollout.mp4"` |
| | `video_skip` | `5` | Record a video frame every N simulation steps |
| **`camera`** | `camera_names` | `["agentview"]` | Camera(s) for rendering (`agentview`, `frontview`, etc.) |
| | `camera_dist_scale` | `null` | Distance multiplier to zoom out/push camera farther away |
| | `camera_fovy` | `null` | Field of view in degrees (e.g., `70` for wide-angle view) |
| | `camera_pos` | `null` | Explicit 3D position `[X, Y, Z]` |
| **`object_placement`** | `object_pos` | `null` | Fix spawn coordinate `[X, Y]` |
| | `object_x_range` | `null` | Custom sampling range `[MIN, MAX]` for X coordinate |
| | `object_y_range` | `null` | Custom sampling range `[MIN, MAX]` for Y coordinate |
| | `object_ref_pos` | `null` | Table center reference coordinate `[X, Y, Z]` |
| **`dataset`** | `dataset_path` | `null` | Path to export rollouts as HDF5 dataset (`.hdf5`) |
| | `dataset_obs` | `false` | Whether to include observation arrays in saved dataset |
| **`system`** | `cpu` | `false` | Force CPU execution instead of CUDA GPU |
| | `verbose` | `false` | Print detailed loading and environment setup logs |


To switch checkpoints, update the `weights.checkpoint_path` field in `config/run_diffusion_policy.yaml`.

---

## (Optional) Activating the Virtual Environment
If you prefer activating the virtual environment directly:
```bash
source .venv/bin/activate
python src/run_diffusion_policy.py
```
