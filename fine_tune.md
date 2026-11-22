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

### 3.5 Total Loss
$$L(\theta, \phi) = L_{\text{actor}}(\theta) + c_v L_{\text{critic}}(\phi) + \beta L_{\text{KL}}(\theta)$$

---

## 4. Parameter Partitioning Schedule

The policy weights $\theta = (\theta_{\text{time}}, \theta_{\text{cond}}, \theta_{\text{rest}})$ are partitioned as follows:
- **$\theta_{\text{time}}$ (`diffusion_step_encoder`)**: **FROZEN** throughout training so the meaning of noise levels remains invariant.
- **$\theta_{\text{cond}}$ (`cond_encoder` in residual blocks)**: Trainable with learning rate decaying from $\eta_0 = 10^{-4}$ to $\eta_{\text{min}} = 10^{-5}$ across the initial $N_1 \in [100, 150]$ iterations.
- **$\theta_{\text{rest}}$ (convolutional backbone)**: Trainable with base learning rate.
- **$\phi$ (critic)**: Trained with a separate Adam optimizer ($\eta_{\text{critic}} = 3 \times 10^{-4}$).

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

## 6. How to Run

### 1. Run Test Suite
```bash
PYTHONPATH=. uv run pytest -p no:launch_testing test/test_dppo.py -v
```

### 2. Launch Training
```bash
PYTHONPATH=. uv run python -m src.train_dppo --config config/train_dppo.yaml
```

### 3. Launch Evaluation Benchmark on a Checkpoint
```bash
PYTHONPATH=. uv run python -m src.train_dppo --eval-only --checkpoint results/dppo_checkpoints/dppo_best_iter_10.pt --eval-episodes 20
```

### 4. Fast Smoke Run
```bash
PYTHONPATH=. uv run python -m src.train_dppo --iterations 3 --episodes-per-iter 4 --device cpu
```
