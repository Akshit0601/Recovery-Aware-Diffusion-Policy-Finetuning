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

### Without `uv` (plain Python + pip, e.g. on a server)
Needs Python 3.10. `requirements.txt` is exported from `uv.lock`, so it has the same pinned versions, the CUDA 12.4
PyTorch index and the local editable `./robomimic`:
```bash
python3.10 -m venv .venv            # or: conda create -n dp_env python=3.10 && conda activate dp_env
source .venv/bin/activate
pip install -r requirements.txt
```
Afterwards, every `uv run python ...` command in this README is just `python ...` with the environment activated.

### Server Setup (conda)
All commands are run from the repo root. Install packages with pip, not conda, so the pinned versions in
`requirements.txt` (including `torch==2.6.0+cu124`) are what you get. The server needs `git` and internet access,
because robosuite is installed from a GitHub commit.

1. **Create the env and install dependencies** (Python 3.10 is required):
   ```bash
   conda create -n dp-env python=3.10 -y
   conda activate dp-env
   pip install -r requirements.txt
   ```
   If `dp-env` already exists, just activate it and run `pip install -r requirements.txt`. pip skips packages that
   already match a pin, changes any that are on a different version, and leaves extra packages alone. To preview the
   changes first:
   `pip install -r requirements.txt --dry-run 2>&1 | grep "Would install"`.

2. **Reinstall robomimic in compat mode.** The repo root contains a plain `robomimic/` folder. With the default
   editable install, running from the repo root (or with `PYTHONPATH=.`) makes Python import that folder as an empty
   namespace package instead of the real `robomimic/robomimic/`:
   ```bash
   pip install -e ./robomimic --no-deps --config-settings editable_mode=compat
   ```

3. **Verify:**
   ```bash
   python -c "import robomimic; print(robomimic.__file__)"
   # expected: <repo>/robomimic/robomimic/__init__.py
   python -c "from robomimic.config.base_config import REGISTERED_CONFIGS; print(list(REGISTERED_CONFIGS))"
   # expected: list includes 'diffusion_policy'
   pip check
   ```

4. **Copy the pretrained checkpoint.** It is gitignored; see the rsync command in
   [section 5](#5-dppo-recovery-fine-tuning-server--headless).

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| `Config for algo name diffusion_policy not found` | PyPI `robomimic` (in `site-packages`) is installed instead of the repo copy | `pip uninstall -y robomimic` (repeat until "not installed"), then step 2 |
| `robomimic.__file__` prints `None` | The repo-root `robomimic/` folder shadows the installed package | Step 2 (compat mode) |
| `pip show robomimic` has no `Editable project location` | robomimic was not installed from the repo | Step 2 |
| `Weights only load failed ... numpy.core.multiarray.scalar` | PyTorch >= 2.6 defaults to `torch.load(weights_only=True)` | Already handled in `src/dppo/trainer.py` and `src/diffusion_policy_runner.py`; pull the latest code |

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
uv run python -m src.run_diffusion_policy
```

You can also pass an explicit configuration file via `--config` or as a positional argument:
```bash
uv run python -m src.run_diffusion_policy --config config/run_diffusion_policy.yaml
# or
uv run python -m src.run_diffusion_policy config/run_diffusion_policy.yaml
```

### 3. Programmatic & Subclass Usage (`DiffusionPolicyRunner`)
`src/diffusion_policy_runner.py` provides `DiffusionPolicyRunner` (also aliased as `BaseRunner`), which can be inherited or used directly across different experimental scripts:

```python
from src import DiffusionPolicyRunner

# Run with programmatic configuration overrides
runner = DiffusionPolicyRunner(
    config_path="config/run_diffusion_policy.yaml",
    overrides={"rollout": {"n_rollouts": 5}, "rendering": {"render": False}}
)
results = runner.run()

# Or subclass to customize actions, observations, or perturbations
class PerturbedPolicyRunner(DiffusionPolicyRunner):
    def get_action(self, obs, step_i=0):
        action = super().get_action(obs, step_i)
        # Inject custom action noise or perturbation
        return action
```

### 4. Workspace Grasp Failure Heatmap Evaluation (Config-Free)
Evaluate policy grasping and task failure rates across a 2D spatial grid of object spawn poses and yaw orientations ($0^\circ, 30^\circ, 45^\circ$) directly via runner override methods.

Edit the parameters directly at the top of [`src/eval_workspace_heatmap.py`](src/eval_workspace_heatmap.py):
```python
X_MIN = -0.15               # Minimum X coordinate in meters
X_MAX = 0.15                # Maximum X coordinate in meters
Y_MIN = -0.15               # Minimum Y coordinate in meters
Y_MAX = 0.15                # Maximum Y coordinate in meters
RESOLUTION = 5              # Grid resolution N along each axis (N x N spatial positions)
YAW = [0.0, 30.0, 45.0]     # Yaw angle(s) in degrees
OUTPUT_DIR = "results/heatmaps"  # Destination directory for output files
```

Then run directly:
```bash
uv run python -m src.eval_workspace_heatmap
```




---

### 5. DPPO Recovery Fine-Tuning (server / headless)
Details of the method are in [`fine_tune.md`](fine_tune.md); hyperparameters live in [`config/train_dppo.yaml`](config/train_dppo.yaml).

**One-time setup on the server.** The pretrained checkpoint is gitignored, so copy it over to the same relative path
(`weights.pretrained_checkpoint` is resolved relative to the repo root):
```bash
rsync -av robomimic/diffusion_policy_trained_models/test/20261024164824/models/model_epoch_150_low_dim_v15_success_1.0.pth \
  <server>:<repo>/robomimic/diffusion_policy_trained_models/test/20261024164824/models/
```
Rendering is off and observations are low-dim, so no display / `MUJOCO_GL` setup is needed.

All commands below are run from the repo root with the environment activated (`source .venv/bin/activate`).
With uv instead, replace `python` by `uv run python`.

**Train (foreground):**
```bash
PYTHONPATH=. python -m src.train_dppo --config config/train_dppo.yaml
```

**Train in the background** (survives SSH logout; stdout/stderr go to a timestamped file in `logs/`):
```bash
mkdir -p logs
RUN=dppo_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python -u -m src.train_dppo \
  --config config/train_dppo.yaml > logs/${RUN}.log 2>&1 &
echo $! > logs/${RUN}.pid
tail -f logs/${RUN}.log          # Ctrl-C stops tailing, not training
```
- Checkpoints, `train.log` and per-iteration `metrics.jsonl` are written to `weights.save_dir` (default `results/dppo_checkpoints/`).
- `dppo_last.pt` is overwritten every iteration (with optimizer + RNG state). After a crash, kill or reboot, resume with:
  ```bash
  PYTHONPATH=. nohup python -u -m src.train_dppo --config config/train_dppo.yaml \
    --resume results/dppo_checkpoints/dppo_last.pt > logs/${RUN}_resume.log 2>&1 &
  ```
- Stop a run: `kill $(cat logs/${RUN}.pid)`.
- Things to watch in the log: `ClipFrac` (healthy ≈ 0.1–0.5), `ApproxKL` / `Epochs` (early stopping), and the periodic
  `Evaluation (iter N)` lines versus the `Baseline (pretrained, iter 0)` line.

**Evaluate** (fixed `rl.eval_seed`, so both runs see identical scenarios):
```bash
# pretrained baseline
PYTHONPATH=. python -m src.train_dppo --eval-only --eval-episodes 100
# fine-tuned checkpoint
PYTHONPATH=. python -m src.train_dppo --eval-only --eval-episodes 100 \
  --checkpoint results/dppo_checkpoints/dppo_best_iter_40.pt
```
Keep `rollout.num_envs` the same across runs you compare.

**Tests:**
```bash
PYTHONPATH=. python -m pytest -p no:launch_testing test/test_dppo.py -v
```

**Resources** (measured on an RTX 4050 laptop): ~2 GB GPU for rollouts, ~3.2 GB for updates at `batch_size: 512`
(fits an 8 GB card), ~3 GB RAM for 16 envs. Rollouts take ~3.6 s/episode with `num_envs: 16`; expect roughly
**8–10 h** for the default 100 iterations including evaluations (vs ~28 h with a single env).

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
python -m src.run_diffusion_policy
```
