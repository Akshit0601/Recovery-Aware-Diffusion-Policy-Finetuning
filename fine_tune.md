# Recovery-Aware RL Fine-Tuning of Diffusion Policies (DPPO)

This document details the recovery-aware reinforcement learning (RL) fine-tuning pipeline for Diffusion Policies under manipulation disturbances, based on **ECE285_MidTerm.pdf** (*Shishodia & Karthik*).

---

## 1. Problem & Motivation

Pretrained behavioral cloning (BC) diffusion policies are expressive but brittle: when external disturbances displace the manipulated object, policies encounter out-of-distribution (OOD) states and often fail by continuing stale action trajectories.

This pipeline fine-tunes a pretrained policy $\pi_{\theta_0}$ using **Diffusion Policy Policy Optimization (DPPO)** with:
- **Online Rollout Disturbances**: Injected phase-conditioned physical object pose shifts ($\Xi_{\text{train}}$) mixed with nominal episodes (40% nominal, 60% disturbed).
- **Nested MDP DPPO**: Treating the 100-step DDPM reverse chain as a nested MDP with per-step clipped surrogate actor loss.
- **Closed-Form KL Anchor**: An analytical per-step Gaussian KL penalty $\text{KL}(\pi_\theta \parallel \pi_{\theta_0})$ to the frozen pretrained weights to prevent catastrophic forgetting on nominal tasks.
- **MLP Critic & GAE**: Value function $V_\phi(s_t)$ predicting discounted returns ($\gamma^{t_{\text{succ}}}$), penalizing multi-chunk recovery delays.
- **Parameter Partitioning & Staging**: Freezing the noise-level time embeddings $\theta_{\text{time}}$ throughout, while training the conditioning pathway $\theta_{\text{cond}}$ with a learning rate decay schedule.

---

## 2. Manipulation Phases & Pre-Grasp Mode

### Operational Execution Phases
- **Approaching ($\|a_{\text{arm}}\|_2 > 0.4$)**: The robot arm moves toward the target cube with open gripper. Disturbances injected around step 20 ($\Delta x, \Delta y \in [-6, 6]$ cm, $\Delta \psi \in [-90^\circ, 90^\circ]$) test trajectory replanning.
- **Pre-Grasp ($\|a_{\text{arm}}\|_2 \le 0.4, a_{\text{grip}} > 0$)**: The arm slows down or hovers over the cube while gripper fingers close. Disturbances injected at phase onset around step 34.2 ($\Delta x \in [-5, 5]$ cm, $\Delta y \in [-3, 3]$ cm, $\Delta \psi \in [-30^\circ, 30^\circ]$) test whether the robot can abort finger closure on an empty space, reopen, and reposition.
- **Grasped ($r_t = 1.0$)**: The object is secured and lifted.

### Bimodal Grasping Modes ($t_{\text{grasp}}$)
Empirical analysis in the paper identified a bimodal distribution of grasp completion step $t_{\text{grasp}}$:
1. **First-Pass Grasp mode** ($t_{\text{grasp}} \le 60$ steps, mean $44.4$ steps): Minor shifts corrected in-flight before the gripper closes.
2. **Multi-Chunk Re-Grasp mode** ($t_{\text{grasp}} > 60$ steps, mean $\approx 99.1$–$109.8$ steps): Triggered in **57.4%** of successful pre-grasp recoveries when the initial grasp misses, forcing a multi-chunk re-grasp sequence.
3. **Discounted Return Incentive**: The return $R = \sum_t \gamma^t r_t = \gamma^{t_{\text{succ}}}$ ($\gamma < 1$) explicitly penalizes this $\sim 100$-step re-grasp delay to train faster post-disturbance recovery.

---

## 3. Mathematical Formulation (DPPO + KL)

### 3.1 Denoising Transition
For step $k \in \{K, \dots, 1\}$ ($K=100$):
$$\pi_\theta(a^{k-1} \mid a^k, k, s) = \mathcal{N}(\mu_\theta(a^k, k, s), \sigma_k^2 I)$$
$$\mu_\theta(a^k, k, s) = \frac{1}{\sqrt{\alpha_k}} \left( a^k - \frac{1 - \alpha_k}{\sqrt{1 - \bar{\alpha}_k}} \epsilon_\theta(a^k, k, s) \right)$$

### 3.2 Per-Step Importance Sampling & Actor Loss
$$\rho_{t,k}(\theta) = \frac{\pi_\theta(a_t^{k-1} \mid a_t^k, k, s_t)}{\pi_{\theta_{\text{old}}}(a_t^{k-1} \mid a_t^k, k, s_t)} = \exp\left( \frac{\|a_t^{k-1} - \mu_{\theta_{\text{old}}}\|^2 - \|a_t^{k-1} - \mu_\theta\|^2}{2\sigma_k^2} \right)$$
$$L_{\text{actor}}(\theta) = -\frac{1}{|B|} \sum_{(t, k) \in B} \min\left( \rho_{t,k} A_t,\, \text{clip}(\rho_{t,k}, 1-\varepsilon, 1+\varepsilon) A_t \right)$$

### 3.3 Critic MSE Loss
$$L_{\text{critic}}(\phi) = \frac{1}{|B|} \sum_{t \in B} \left( V_\phi(s_t) - \hat{R}_t \right)^2$$
where $\hat{R}_t = A_t + V_{\phi_{\text{old}}}(s_t)$ and $A_t$ is computed via GAE:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t), \quad A_t = \sum_{l \ge 0} (\gamma \lambda)^l \delta_{t+l}$$

### 3.4 Closed-Form Analytical KL Divergence
Since $\pi_\theta$ and $\pi_{\theta_0}$ share $\sigma_k$, the per-step Gaussian KL divergence has a closed form:
$$\text{KL}(\pi_\theta \parallel \pi_{\theta_0})|_{t,k} = \frac{\|\mu_\theta - \mu_{\theta_0}\|^2}{2\sigma_k^2} = \frac{(1 - \alpha_k)^2}{2\sigma_k^2 \alpha_k (1 - \bar{\alpha}_k)} \|\epsilon_\theta - \epsilon_{\theta_0}\|^2$$
$$L_{\text{KL}}(\theta) = \frac{1}{|B|} \sum_{(t, k) \in B} \text{KL}|_{t,k}$$

### 3.4.1 Denoising-Step Selection and σ Floor
The true DDPM $\sigma_k$ is tiny near the end of the chain ($\sigma_1 \approx 0.02$, $\sigma_0 \approx 10^{-10}$), so the ratio in §3.2 and the KL in §3.4 explode for small $k$ (a $10^{-3}$ change in $\mu$ gave a KL of ~2800 at $k=0$). Following DPPO:
- $k=0$ is deterministic ($a^0 = \mu_\theta$) and is never part of the objective.
- Only the last $K_{\text{ft}} = 10$ steps ($k = 1..10$) are fine-tuned (`k_ft_mode: tail`); earlier steps are sampled but not trained. This also cuts the update cost ~10×.
- Training rollouts sample with $\max(\sigma_k, 0.1)$, and the same floor is used in $\rho$ and $\text{KL}$ (`min_sampling_std`, `min_logprob_std`). Evaluation uses the true DDPM $\sigma_k$, so its numbers are comparable to the pretrained baseline.

### 3.5 Total Loss
$$L(\theta, \phi) = L_{\text{actor}}(\theta) + c_v L_{\text{critic}}(\phi) + \beta L_{\text{KL}}(\theta)$$

### 3.6 Approximate KL and `target_kl` Early Stopping
Implemented in `compute_actor_loss` (`src/dppo/losses.py`) and `update_policy_and_critic` (`src/dppo/trainer.py`).

1. **Per-step log ratio.** For every stored denoising transition $(t, k)$, using the shared $\sigma_k$ floored at `min_logprob_std` $= 0.1$ and summing over the whole $16 \times 7$ action chunk:
$$\log \rho_{t,k} = \frac{\|a_t^{k-1} - \mu_{\theta_{\text{old}}}\|^2 - \|a_t^{k-1} - \mu_\theta\|^2}{2\sigma_k^2}$$
It is clamped to $[-20, 20]$ only to avoid `exp` overflow.

2. **Estimate from the unclipped ratio.** Schulman's k3 estimator of $\text{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta)$ over a minibatch $B$:
$$\widehat{\text{KL}} = \frac{1}{|B|} \sum_{(t,k) \in B} \left[ (\rho_{t,k} - 1) - \log \rho_{t,k} \right] \ge 0$$
PPO's clip $[1-\varepsilon, 1+\varepsilon]$ is applied only inside $L_{\text{actor}}$ (§3.2). It does not enter this estimate. A clipped sample has zero actor gradient but still adds its full drift to `ApproxKL`, so the KL keeps measuring how far $\pi_\theta$ has moved even when most samples are clipped.

3. **`ClipFrac` is separate:** the fraction of samples with $|\rho_{t,k} - 1| > \varepsilon$. It rises with the KL, but only `ApproxKL` controls early stopping.

4. **Early-stop rule.** Each minibatch's $\widehat{\text{KL}}$ is computed on the forward pass *before* that minibatch's Adam step. After each full epoch the trainer averages that epoch's minibatch values; if the average exceeds `target_kl` ($0.005$), the remaining epochs are skipped (`Early stop after epoch N/10` in the log). The check runs only between epochs, so the epoch that trips it has already gone slightly past $0.005$ by its last minibatches. The critic is not affected and always trains for all `epochs`.

5. **Logged value.** The `ApproxKL` on the `Update:` log line is the mean over *all* epochs of the iteration, including the early ones where it starts near 0. It therefore sits below `target_kl` (≈ 0.003 in the 100-iteration run) even though every iteration stopped on it (at epoch 5–6).

This KL is to the rollout policy $\pi_{\theta_{\text{old}}}$ of the current iteration and resets every iteration. It is different from $L_{\text{KL}}$ (§3.4, `KLLoss` in the log), the closed-form KL to the frozen $\pi_{\theta_0}$, which is a $\beta$-weighted penalty and accumulates over the run (0.003 → 0.27).

---

## 4. Parameter Partitioning Schedule

$\theta_0$ is the **EMA** weight set of the epoch-150 checkpoint (`model_epoch_150_low_dim_v15_success_1.0.pth`, `weights.use_ema: true`). robomimic evaluates with the EMA weights, so that is the policy behind the 100% success; the raw training weights in the same file differ by ~7% (relative norm).

The policy weights $\theta = (\theta_{\text{time}}, \theta_{\text{cond}}, \theta_{\text{rest}})$ are partitioned as follows:
- **$\theta_{\text{time}}$ (`diffusion_step_encoder`)**: **FROZEN** throughout training so the meaning of noise levels remains invariant.
- **$\theta_{\text{cond}}$ (`cond_encoder` in residual blocks)**: Trainable with learning rate decaying from $\eta_0 = 3 \times 10^{-6}$ to $\eta_{\text{min}} = 10^{-6}$ across the initial $N_1 = 100$ iterations (see §4.1).
- **$\theta_{\text{rest}}$ (convolutional backbone)**: Trainable with base learning rate.
- **$\phi$ (critic)**: Trained with a separate Adam optimizer ($\eta_{\text{critic}} = 3 \times 10^{-4}$).

### 4.1 Why $\eta_0 = 3 \times 10^{-6}$ (not $10^{-4}$)
Each $\rho_{t,k}$ is a Gaussian likelihood ratio over the whole action chunk ($16 \times 7 = 112$ dims) at $\sigma = 0.1$, so $\log \rho \approx (a - \mu)\cdot\Delta\mu / \sigma^2$ reacts strongly to small changes in $\mu$. Adam also moves every one of the 65M UNet weights by roughly $\eta$ per step, whatever the gradient size. We measured the clip fraction over 10 epochs on the same real rollouts (48 episodes, ~14 minibatches/epoch):

| $\eta_0$ | $\varepsilon$ | clip fraction, epoch 1 → 10 | approx. KL per step, epoch 10 |
|---|---|---|---|
| $10^{-4}$ | 0.01 | 0.67 → 1.00 (≈0.99 after one step) | — |
| $10^{-5}$ | 0.05 | 0.53 → 0.90 | 0.019 |
| $3\times10^{-6}$ | 0.05 | 0.22 → 0.73 | 0.003 |
| **$3\times10^{-6}$** | **0.1** | **0.07 → 0.54** | **0.007** |
| $10^{-6}$ | 0.05 | 0.05 → 0.42 | 0.001 |

At $10^{-4}$ a single optimizer step pushes almost every sample outside the clip range. Clipped samples have zero actor gradient, so PPO ignores the data and only the KL term moves the policy. $3\times10^{-6}$ with $\varepsilon = 0.1$ is the largest setting we tried that keeps the clip fraction in the usual PPO range for most of an iteration. As a safety net, `target_kl: 0.005` stops an iteration's epochs early once the mean approximate KL to the rollout policy exceeds 0.005 (≈ epoch 6 at these settings). These values come from the first iteration, so watch `ClipFrac` / `ApproxKL` in the logs: lower $\eta_0$ if `ClipFrac` stays above ~0.5, raise it if it stays near 0.

---

## 5. Codebase Architecture

```
src/
├── dppo/
│   ├── __init__.py
│   ├── critic.py          # MLP Critic V_ϕ(s_t)
│   ├── sampler.py         # DDPM reverse stochastic sampler & transition recorder
│   ├── losses.py          # L_actor, L_critic, and analytical L_KL
│   ├── disturbances.py    # Stratified rollout generator & MuJoCo pose injector
│   ├── buffer.py          # Trajectory buffer with GAE and advantage normalizer
│   └── trainer.py         # DPPOTrainer implementing Algorithm 1
├── train_dppo.py          # CLI entrypoint for training and evaluation
config/
└── train_dppo.yaml        # Full hyperparameter configuration
test/
└── test_dppo.py           # Unit tests and end-to-end integration smoke tests
```

---

## 6. Training Setup

| Setting | Value | Note |
|---|---|---|
| Rollouts / iteration | 50 (40% nominal, 30% approach, 30% pre-grasp) | `rollout.n_episodes` |
| Parallel envs | 16 | denoising batched across envs: 251 ms/chunk vs ~980 ms with 1 env |
| Horizon | 400 env steps (50 chunks) | successes finish by ~110 steps |
| Epochs / minibatch | ≤10 (early stop on `target_kl`) / 512 | ~3.2 GB GPU for updates (fits 8 GB) |
| Discount | γ = 0.99 **per chunk** (8 env steps) | ≈ 0.9987 per env step |
| Evaluation | 30 episodes × 5 regimes every 10 iters, fixed `eval_seed` | same scenarios for every checkpoint; pretrained baseline at iter 0; best checkpoint gets a 100/regime final eval |

Expected wall-clock is ~8–10 h for 100 iterations. `dppo_last.pt` is written every iteration and `--resume` continues a killed run (see the README for the `nohup` command).

---

## 7. How to Run

### 1. Run Test Suite
```bash
PYTHONPATH=. uv run pytest -p no:launch_testing test/test_dppo.py -v
```

### 2. Launch Training
```bash
PYTHONPATH=. uv run python -m src.train_dppo --config config/train_dppo.yaml
```

### 3. Launch Evaluation Benchmark on a Checkpoint
`--eval-episodes` is per regime. Run without `--checkpoint` to evaluate the pretrained baseline on the same scenarios.
```bash
PYTHONPATH=. uv run python -m src.train_dppo --eval-only --checkpoint results/dppo_checkpoints/dppo_best_iter_10.pt --eval-episodes 100
```

### 3b. Resume an Interrupted Run
```bash
PYTHONPATH=. uv run python -m src.train_dppo --resume results/dppo_checkpoints/dppo_last.pt
```

### 4. Fast Smoke Run
```bash
PYTHONPATH=. uv run python -m src.train_dppo --iterations 3 --episodes-per-iter 4 --device cpu
```
