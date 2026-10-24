# Diffusion Policy Implementation (`diffusion.py`)

This document provides technical documentation, architectural details, and usage guidelines for [`diffusion.py`](file:///home/akshit/ece285_proj/diffusion.py), which implements a Denoising Diffusion Probabilistic Model (DDPM) policy evaluated on the **Robomimic + MuJoCo** robotic manipulation benchmark.

---

## 1. System Overview

[`diffusion.py`](file:///home/akshit/ece285_proj/diffusion.py) integrates generative action chunking with physical simulation:

```
[Robomimic Environment (Robosuite + MuJoCo)]
                    │
           raw observation (19-dim)
                    ▼
          [Normalizer (min-max)]
                    │
         normalized state [-1, 1]
                    ▼
     [DiffusionPolicy (DDPM Sampler)] ◄─── Pretrained Checkpoint (`state_8000.pt`)
                    │
     denoised action chunk (4 x 7-dim)
                    ▼
         [Normalizer (unnormalize)]
                    │
           raw action [-1, 1]
                    ▼
[Robot Controller (OSC_POSE Delta Panda Arm)]
```

### Key Technical Specs

| Component | Specification |
| :--- | :--- |
| **Task** | Robomimic `Lift` (Franka Panda arm picking up a cube) |
| **Simulator** | MuJoCo 3.3.2 via Robosuite 1.5.2 and Robomimic |
| **Observation Dim** | 19 dimensions (`robot0_eef_pos`: 3, `robot0_eef_quat`: 4, `robot0_gripper_qpos`: 2, `object`: 10) |
| **Action Dim** | 7 dimensions (6-DOF end-effector delta pose + 1 gripper actuation) |
| **Action Chunk Horizon ($T_a$)** | 4 steps |
| **Denoising Timesteps ($K$)** | 20 steps (DDPM reverse diffusion) |
| **Backbone Architecture** | Residual MLP with Sinusoidal Positional Time Embeddings |
| **Conditioning Type** | Direct Concatenation Conditioning (early feature fusion) |
| **Pretrained Checkpoint** | Official DPPO public benchmark checkpoint (`state_8000.pt`, 573,516 params) |

---

## 2. Embedding & Conditioning Architecture

Diffusion policies require feeding two distinct signals into the denoising network alongside the noisy action:
1. **Diffusion Timestep ($t \in \{0, \dots, K-1\}$)**: Tells the network how much noise is present and which stage of reverse diffusion is executing.
2. **Environment Observation ($s_t \in \mathbb{R}^{19}$)**: Conditions the generated trajectory on the robot's physical state and object location.

```
┌────────────────────────────────────────────────────────┐
│               Input Assembly (Dimension: 63)           │
├────────────────────────┬───────────────────┬───────────┤
│  Noisy Action Chunk    │   Time Embedding  │ Condition │
│  (Flattened 4 x 7 = 28)│   (Sinusoidal 16) │ (State 19)│
└───────────┬────────────┴─────────┬─────────┴─────┬─────┘
            └──────────────────────┼───────────────┘
                                   ▼
                      [Linear Layer (63 -> 512)]
                                   │
                      [Residual MLP Blocks (512)]
                                   │
                      [Linear Layer (512 -> 28)]
                                   ▼
                       Predicted Noise Epsilon
```

### 2.1 Diffusion Timestep Embedding: Sinusoidal Positional Embeddings

Rather than treating the diffusion step index $t$ as a scalar or a one-hot lookup table, the model uses **Sinusoidal Positional Embeddings** (derived from the Transformer architecture in Vaswani et al., 2017, and popularized in diffusion by DDPM and Diffuser).

#### Mathematical Formulation
Given an embedding dimension $d = 16$ (so half-dimension $d/2 = 8$), a geometric progression of frequencies is defined as:

$$\omega_i = \exp\left( - \frac{\ln(10000)}{d/2 - 1} \cdot i \right), \quad i \in \{0, 1, \dots, d/2 - 1\}$$

For a scalar diffusion step index $t \in [0, K-1]$, the 16-dimensional embedding vector is formed by concatenating sine and cosine projections across all frequencies:

$$\text{PE}(t) = \left[ \sin(t \cdot \omega_0), \dots, \sin(t \cdot \omega_7), \; \cos(t \cdot \omega_0), \dots, \cos(t \cdot \omega_7) \right] \in \mathbb{R}^{16}$$

#### Non-Linear MLP Projection
To allow the network to learn task-specific features from these harmonic frequencies, the sinusoidal vector passes through a 2-layer projection head with **Mish** activations:

$$\text{TimeEmbedding}(t) = W_2 \cdot \text{Mish}(W_1 \cdot \text{PE}(t) + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{32 \times 16}, \; b_1 \in \mathbb{R}^{32}$
- $W_2 \in \mathbb{R}^{16 \times 32}, \; b_2 \in \mathbb{R}^{16}$

#### Why Sinusoidal Embeddings are Used:
- **Continuous and Smooth**: Adjacent diffusion steps produce smoothly varying feature vectors, allowing the network to interpolate between noise levels.
- **Multi-Scale Frequency Resolution**: High-frequency components encode local differences between adjacent steps (e.g., step 1 vs. step 2), while low-frequency components encode global progression (early denoising vs. late denoising).
- **Zero Learned Parameters at Input**: The basis frequencies are fixed and deterministic, preventing overfitting to specific step indices.

---

### 2.2 Observation State Embedding & Representation

The environment observation vector $s_t \in \mathbb{R}^{19}$ captures full task-relevant physical state from MuJoCo:

| Key | Dimension | Physical Meaning |
| :--- | :---: | :--- |
| `robot0_eef_pos` | 3 | Cartesian coordinates $(x, y, z)$ of the Franka Panda end-effector |
| `robot0_eef_quat` | 4 | Unit quaternion orientation $(q_w, q_x, q_y, q_z)$ of the end-effector |
| `robot0_gripper_qpos` | 2 | Position of the two parallel gripper fingers |
| `object` | 10 | Target cube state: 3D position $(x_c, y_c, z_c)$, 4D quaternion orientation, and 3D relative vector from gripper to cube |
| **Total** | **19** | Concatenated observation state vector |

#### Min-Max State Normalization
All raw observation dimensions are normalized to the range $[-1, 1]$ before being input to the network using statistics recorded over expert demonstrations:

$$s_{\text{norm}} = 2 \cdot \left( \frac{s - s_{\min}}{s_{\max} - s_{\min} + 10^{-6}} \right) - 1$$

---

### 2.3 Conditioning Strategy: Direct Concatenation Conditioning

There are three common ways to condition a diffusion policy on observation states:

| Conditioning Method | How it Works | Where It Is Used | Used Here? |
| :--- | :--- | :--- | :---: |
| **Direct Concatenation** | Flattens state and concatenates directly with action chunk and time embedding at the first linear layer. | Low-dim state policies (DPPO, Stanford low-dim DP) | **Yes** |
| **FiLM / Adaptive LayerNorm** | Time and state modulate hidden feature maps via affine scale & shift parameters ($\gamma \cdot h + \beta$). | Diffusion Transformers (DiT), UNet feature blocks | No |
| **Cross-Attention** | State features act as keys/values in multi-head attention queries. | Image diffusion, Stable Diffusion, Octo | No |

**Why Direct Concatenation is Used Here**:
For low-dimensional proprioceptive manipulation ($D_{\text{obs}} = 19$, $D_{\text{act}} = 7$), direct concatenation allows all layers in the residual MLP to jointly model physical cross-couplings between robot joints, end-effector coordinates, and action deltas with minimal computational overhead.

---

### 2.4 Action Chunking Representation

The policy does not predict a single action per timestep. Instead, it predicts a chunk of $T_a = 4$ future actions:

$$A_t = [a_t, a_{t+1}, a_{t+2}, a_{t+3}] \in \mathbb{R}^{4 \times 7}$$

Each individual action $a \in \mathbb{R}^7$ specifies:
- 3 Cartesian position deltas: $(\Delta x, \Delta y, \Delta z)$
- 3 Axis-angle orientation deltas: $(\Delta r_x, \Delta r_y, \Delta r_z)$
- 1 Gripper open/close command: $g \in [-1, 1]$

The entire action chunk is flattened into a 28-dimensional vector during denoising and un-flattened back to $(B, 4, 7)$ at output.

---

## 3. Mathematical Formulation (DDPM)

### Forward (Noising) Schedule
A cosine variance schedule is used:

$$\bar{\alpha}_k = \frac{f(k)}{f(0)}, \quad f(k) = \cos^2\left(\frac{k/K + s}{1 + s} \cdot \frac{\pi}{2}\right)$$

where $s = 0.008$ prevents $\beta_k$ from being too small near $k=0$.

### Reverse Denoising Process
Starting from standard Gaussian noise $x_K \sim \mathcal{N}(0, I)$, the model iteratively denoises the action chunk:

$$x_{k-1} = \mu_\theta(x_k, k, s_t) + \sigma_k \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where the predicted clean action chunk $\hat{x}_0$ is:

$$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_k}} \left( x_k - \sqrt{1 - \bar{\alpha}_k} \cdot \epsilon_\theta(x_k, k, s_t) \right)$$

and the posterior mean $\mu_\theta$ is computed via:

$$\mu_k = \frac{\beta_k \sqrt{\bar{\alpha}_{k-1}}}{1 - \bar{\alpha}_k} \hat{x}_0 + \frac{(1 - \bar{\alpha}_{k-1})\sqrt{\alpha_k}}{1 - \bar{\alpha}_k} x_k$$

---

## 4. Residual MLP Backbone Architecture

The denoising network [`DiffusionMLP`](file:///home/akshit/ece285_proj/diffusion.py#L74) contains **573,516 parameters** arranged as follows:

```text
DiffusionMLP
├── time_embedding: Sequential
│   ├── [0] SinusoidalPosEmb(dim=16)
│   ├── [1] Linear(in_features=16, out_features=32)
│   ├── [2] Mish()
│   └── [3] Linear(in_features=32, out_features=16)
└── mlp_mean: ResidualMLP
    └── layers: ModuleList
        ├── [0] Linear(in_features=63, out_features=512)
        ├── [1] TwoLayerPreActivationResNetLinear (Block 1)
        │   ├── act: Mish()
        │   ├── l1: Linear(in_features=512, out_features=512)
        │   └── l2: Linear(in_features=512, out_features=512)
        ├── [2] TwoLayerPreActivationResNetLinear (Block 2)
        │   ├── act: Mish()
        │   ├── l1: Linear(in_features=512, out_features=512)
        │   └── l2: Linear(in_features=512, out_features=512)
        ├── [3] Linear(in_features=512, out_features=28)
        └── [4] Identity()
```

Each `TwoLayerPreActivationResNetLinear` block computes:
$$\text{Output} = x + W_2 \cdot \text{Mish}(W_1 \cdot \text{Mish}(x))$$

---

## 5. API Reference & Code Map

| Class / Function | Location | Description |
| :--- | :--- | :--- |
| `SinusoidalPosEmb` | [`diffusion.py:28`](file:///home/akshit/ece285_proj/diffusion.py#L28) | Generates multi-frequency harmonic time embeddings |
| `TwoLayerPreActivationResNetLinear` | [`diffusion.py:44`](file:///home/akshit/ece285_proj/diffusion.py#L44) | Pre-activation residual MLP building block |
| `ResidualMLP` | [`diffusion.py:56`](file:///home/akshit/ece285_proj/diffusion.py#L56) | Multi-block residual linear network |
| `DiffusionMLP` | [`diffusion.py:74`](file:///home/akshit/ece285_proj/diffusion.py#L74) | Denoising network taking flattened (Action + Time + State) |
| `DiffusionPolicy` | [`diffusion.py:141`](file:///home/akshit/ece285_proj/diffusion.py#L141) | Full DDPM sampler executing reverse diffusion |
| `Normalizer` | [`diffusion.py:214`](file:///home/akshit/ece285_proj/diffusion.py#L214) | Min-max normalization for observation and action spaces |
| `create_robomimic_lift_env` | [`diffusion.py:231`](file:///home/akshit/ece285_proj/diffusion.py#L231) | Instantiates Robomimic Panda Lift task in MuJoCo |
| `run_evaluation_episode` | [`diffusion.py:284`](file:///home/akshit/ece285_proj/diffusion.py#L284) | Executes rollouts with disturbance injection hooks |

---

## 6. Disturbance Injection for Recovery Experiments

To support the research questions in [`docs/recovery_aware_rl_finetuning_dp_course_project.md`](file:///home/akshit/ece285_proj/docs/recovery_aware_rl_finetuning_dp_course_project.md), `run_evaluation_episode()` includes physical state perturbation hooks:

```python
# In-flight object displacement hook in run_evaluation_episode()
if disturbance_step is not None and step == disturbance_step and disturbance_offset is not None:
    cube_pos = env.env.sim.data.qpos[:3].copy()
    cube_pos += disturbance_offset               # Apply delta [dx, dy, dz]
    env.env.sim.data.qpos[:3] = cube_pos
    env.env.sim.forward()                        # Propagate kinematics & contacts
    obs = env.get_observation()
```

This allows evaluating whether the diffusion policy can recover from out-of-distribution states (e.g., cube knocked $5\text{ cm}$ sideways during reaching).

---

## 7. How to Run

```bash
/home/akshit/ece285_proj/.venv/bin/python /home/akshit/ece285_proj/diffusion.py
```
