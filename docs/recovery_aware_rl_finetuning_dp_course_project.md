# Recovery-Aware RL Fine-Tuning of Diffusion Policies under Manipulation Disturbances

**Course project proposal — tentative research plan**  
**Course fit:** Deep generative models, especially diffusion models  
**Scope:** Simulation-based robotic manipulation  
**Target duration:** 8 weeks  
**Primary method:** Pretrained Diffusion Policy (DP) fine-tuned with Diffusion Policy Policy Optimization (DPPO)

## Course Alignment

This project is directly relevant to the course's deep generative modeling content, particularly Lectures 10–11 on denoising and score-based diffusion models. A Diffusion Policy is a conditional generative model that generates action chunks through iterative denoising. The project studies how reward-driven fine-tuning changes its generated action-sequence distribution and whether the resulting behavior improves recovery under disturbances.

The simulator is the evaluation environment, while reinforcement learning is the fine-tuning mechanism. The central technical subject is the diffusion policy and its behavior after fine-tuning—not simulator engineering alone. The project also relates conceptually to Lecture 7 on fine-tuning, but DPPO is not inherently a parameter-efficient fine-tuning method. VAE, EM, GAN, or LLM methods do not need to be added unless they become directly useful.

### Submission milestones

- **Proposal (2 pages, NeurIPS format, excluding checklist):** objective and significance, background, research question, proposed gap, and concise experimental plan.
- **Mid-term report (5 pages including revised proposal, excluding references):** method, algorithm, technical formulation, theoretical motivation/assumptions, and preliminary implementation results if available.
- **Final report (8 pages including revised mid-term report, excluding references):** controlled experiments, analysis, conclusions, limitations, and failure cases.


---

## 1. Problem Statement and Motivation

Diffusion Policies (DPs) can represent multimodal action distributions and have shown strong performance in imitation-learning-based robotic manipulation. However, a policy trained mainly on successful demonstrations may behave poorly when execution deviates from the demonstrated trajectories—for example, when an object is displaced, a grasp fails, or an external disturbance occurs during manipulation.

The project investigates **recovery-aware reinforcement-learning fine-tuning of a pretrained Diffusion Policy**. Instead of training a policy from scratch, we will start from a pretrained DP, deliberately induce disturbances in simulation, collect online rollouts, and use reward-driven fine-tuning to improve recovery.

The course-facing technical focus is how fine-tuning reshapes a conditional generative model's action-chunk distribution. Better task success alone does not prove that the model learned a qualitatively new strategy, so action/trajectory analysis will be used where feasible to interpret the results.

### Core research question

> Can disturbance-focused RL fine-tuning improve a pretrained Diffusion Policy's recovery from manipulation disturbances—particularly on disturbance conditions held out from training—while preserving its nominal task performance?

### What problem are we trying to solve?

1. **Failure after distribution shift:** the policy may encounter object poses or execution histories that differ from its demonstrations.
2. **Insufficient recovery behavior:** after a disturbance, the policy may repeat a failed approach, continue an unsuitable action sequence, or fail to reach a successful trajectory.
3. **Generalization beyond trained disturbances:** learning to handle the exact disturbances seen during training is not enough; the updated policy should ideally recover from new combinations of disturbance magnitude, direction, and task phase.
4. **Preserving nominal behavior:** fine-tuning for recovery should not substantially degrade success on undisturbed episodes.

A critical distinction is that an out-of-distribution (OOD) **state** does not necessarily require an OOD **action** or a new **strategy**. The experiments should identify which of these is actually improved.

---

## 2. Related Work

### 2.1 DPPO — Diffusion Policy Policy Optimization

**Reference:** Ren et al., *Diffusion Policy Policy Optimization* (2024).  
Paper: <https://diffusion-ppo.github.io/>  
Official code: <https://github.com/irom-princeton/dppo>

DPPO is an RL fine-tuning framework for diffusion-based policies. It models the denoising process as part of a policy-optimization formulation and uses policy-gradient methods to improve behavior using rewards from environment interaction.

**Relevance to this project**

- Provides a practical way to fine-tune a pretrained DP using simulated rollouts.
- Avoids implementing the diffusion-policy optimization algorithm from scratch.
- Serves as the underlying RL method; the proposed project focuses on the *recovery-training setup and its generalization*, not on claiming a new optimizer.

**Important implication:** DPPO has already demonstrated that RL fine-tuning can improve behavior and learn recovery behavior beyond expert demonstrations. Therefore, “using RL to make a DP recover” is not, by itself, a sufficiently specific novelty claim.

### 2.2 REACH — Explicit Recovery Behavior for Diffusion Policies

**Reference:** Ke et al., *REACH: Explicit Recovery Behavior for Diffusion Policies*, CVPR 2026.  
Paper: <https://openaccess.thecvf.com/content/CVPR2026/html/Ke_REACH_Explicit_Recovery_Behavior_for_Diffusion_Policies_CVPR_2026_paper.html>

REACH addresses undesirable action modes in diffusion policies. It uses an error detector to identify problematic or OOD execution histories and constructs negative action prompts intended to steer the policy away from suboptimal behaviors. Its recovery mechanism uses execution history/backtracking rather than simply assuming that the current action should always be continued.

**Relevance to this project**

- Highlights the importance of detecting when execution has gone wrong.
- Shows that a diffusion policy can be guided away from undesirable behavior using information from failed execution.
- Motivates studying what happens when recovery requires a behavior that the current policy does not reliably produce.

**Potential limitation to investigate**

- Recovery depends on the reliability of the error detector and on whether the history/prompt-based intervention identifies a useful alternative.
- If the detector fails to recognize a failure, or if the available action modes do not contain a suitable recovery behavior, steering away from an undesirable mode may not be enough.
- Repeated backtracking or negative prompting could plausibly fail to resolve a problem when the underlying policy cannot produce a successful alternative. This is a **hypothesis to test**, not a claim that REACH generally enters error loops.

REACH should not be characterized as merely reversing every action. The paper's method is more specific: error detection and negative-prompt-guided steering using execution history.

### 2.3 MoE-DP — Mixture-of-Experts-Enhanced Diffusion Policy

**Reference:** Cheng et al., *MoE-DP: An MoE-Enhanced Diffusion Policy for Robust Long-Horizon Robotic Manipulation with Skill Decomposition and Failure Recovery* (2025 preprint).  
Paper: <https://arxiv.org/abs/2511.05007>  
Project page: <https://moe-dp-website.github.io/MoE-DP-Website/>

MoE-DP inserts a mixture-of-experts module between the visual encoder and diffusion model. A learned routing mechanism activates expert pathways to encourage specialization across task phases and improve robustness in long-horizon manipulation.

**Relevance to this project**

- Addresses robustness and recovery using architectural changes to the policy.
- Provides a relevant comparison for the question of whether specialization improves recovery.
- Motivates asking whether the policy can adapt its behavior through reward-driven learning after disturbances, rather than relying only on its existing learned representation and routing.

**Potential limitation to investigate**

- Expert specialization and routing do not, by themselves, guarantee that the policy will discover a successful response to every unfamiliar disturbance.
- If all experts produce unsuitable responses for a particular failure state, routing among those experts may not be sufficient to escape that failure.
- A policy that continues to generate unsuccessful behavior may repeatedly fail even when the router changes which expert is active.

This should be framed as a **possible failure mode**, not as an established deficiency of MoE-DP. MoE-DP reports improved recovery under its tested conditions, so the project must identify and test a specific condition where additional reward-driven adaptation may help. Its dependence on a diffusion policy alone is not a fair limitation: MoE-DP is explicitly designed as an enhancement to diffusion policies.

---

## 3. Research Gap and Working Hypothesis

### 3.1 Proposed gap

REACH uses error detection and negative prompting to steer away from undesirable action modes; MoE-DP introduces expert specialization and learned routing to improve robust manipulation. These approaches offer different mechanisms for handling failure.

The project will investigate a complementary question:

> When a pretrained diffusion policy fails under a disturbance, can reward-driven fine-tuning on disturbed rollouts improve recovery on *held-out disturbance conditions*, beyond what is achieved by ordinary RL fine-tuning?

The working hypothesis is that deliberately training on recovery episodes will allocate learning signal to failures and post-disturbance behavior, improving generalization to some unseen disturbances. Whether this occurs, and under what conditions, is an empirical question.

A course-aligned formulation is: **Can recovery-focused RL fine-tuning reshape the conditional action-sequence distribution of a pretrained diffusion policy to improve recovery on unseen manipulation disturbances without substantially degrading nominal performance?**

### 3.2 Failure-loop hypothesis

A possible failure loop is:

1. A disturbance invalidates the policy's current action plan.
2. The policy continues to produce an unsuitable action sequence or a recovery mechanism selects an unhelpful alternative.
3. The next observation remains unfavorable, and the policy repeats a similar unsuccessful behavior.
4. The system fails to reach a successful state.

REACH and MoE-DP are not assumed to exhibit this behavior generally. The project can use it as a **diagnostic failure mode** to investigate: does the policy repeatedly produce similar unsuccessful behavior, and can reward-driven fine-tuning reduce that pattern?

### 3.3 Novelty candidate

The most feasible novelty candidate is **recovery-focused RL fine-tuning evaluated on held-out disturbance conditions, with explicit measurement of the recovery–nominal-performance trade-off**.

Potential contributions:

- A controlled comparison of standard DPPO and recovery-focused DPPO.
- Evaluation across disturbance magnitudes, directions, and intervention phases, including combinations excluded from training.
- Measurement of whether recovery gains generalize, rather than only improve performance on familiar disturbances.
- Behavioral analysis of whether successful recovery involves a meaningful change in action sequences or strategy.

This is a candidate contribution, not yet a verified literature gap. Before making a novelty claim, compare the proposed protocol with DPPO, REACH, MoE-DP, and relevant robustness/curriculum-learning papers.

### 3.4 What the project will not claim

- That DPPO is the first method to teach recovery to diffusion policies.
- That REACH or MoE-DP cannot recover from OOD states.
- That a changed action sequence necessarily proves a new strategy.
- That all OOD states require OOD actions.
- That recovery generalization is guaranteed by RL fine-tuning.

---

## 4. Tentative Experimental Pipeline

### Stage 1 — Select a task and establish the baseline

Use a supported simulation benchmark and a pretrained DP. A low-dimensional/state-based Robomimic task such as **Can** is a pragmatic first choice because it reduces perception and environment-engineering overhead.

1. Set up the pretrained policy and evaluation pipeline.
2. Reproduce nominal performance over multiple seeds or episodes.
3. Identify task stages where a disturbance can be introduced reliably.
4. Save the baseline checkpoint and evaluation results.

**Deliverable:** a working pretrained policy and reproducible baseline.

### Stage 2 — Implement controlled disturbances

Inject disturbances during execution, for example:

- **Object displacement:** shift the object laterally or longitudinally.
- **Object orientation:** rotate the object within physically meaningful bounds.
- **Disturbance timing:** perturb during approach, grasping, or transport.
- **Severity:** use small, medium, and large perturbations.

Start with one disturbance type and one task phase. Expand only after the end-to-end pipeline works.

Separate the perturbations into:

- **Training distribution:** disturbance types and parameter ranges used during fine-tuning.
- **Held-out test distribution:** predefined magnitudes, directions, or task-phase combinations excluded from training.

Avoid disturbances that make the task physically impossible; otherwise, failure may reflect task infeasibility rather than poor recovery.

**Deliverable:** repeatable disturbance injection and fixed train/test splits.

### Stage 3 — Fine-tune with DPPO

Compare at least these three conditions:

| Condition | Training | Purpose |
|---|---|---|
| A. Pretrained DP | Demonstrations only | Establish initial robustness |
| B. Standard DPPO | Ordinary task rollouts | Measure the effect of general RL fine-tuning |
| C. Recovery-focused DPPO | Mixture of ordinary and disturbed rollouts | Measure the effect of targeted recovery training |

Use the same task, policy initialization, evaluation conditions, and—where practical—comparable environment-interaction budgets. Keep a mixture of nominal episodes to reduce the risk of forgetting the original task.

For the recovery-focused condition, begin with a simple training mixture. A curriculum or prioritized sampling of failures can be an additional experiment if time permits, rather than a requirement for the minimum viable project.

**Deliverable:** checkpoints for each condition and learning curves.

### Stage 4 — Evaluation

Evaluate each checkpoint on identical episode seeds and disturbance conditions.

**Primary metrics**

1. **Recovery success rate:** fraction of disturbed episodes that eventually complete the task.
2. **Nominal success rate:** fraction of undisturbed episodes that complete the task.
3. **Held-out recovery success:** success on disturbance conditions excluded from training.
4. **Recovery time:** time or environment steps from disturbance to task completion.
5. **Robustness curve:** recovery success as disturbance severity increases.

**Useful secondary analysis**

- Compare training-disturbance performance with held-out performance.
- Inspect action sequences before and after fine-tuning.
- Identify repeated unsuccessful approaches or oscillatory behavior.
- If feasible, compare a simple recovery/failure-loop indicator across policy variants.

Use multiple evaluation episodes and, if compute permits, multiple random seeds. Report uncertainty rather than relying on a single successful rollout.

### Stage 5 — Ablation and interpretation

If the core pipeline is stable, test one additional design choice:

- Does a mixture of normal and disturbed episodes outperform training only on disturbed episodes?
- Does a simple disturbance curriculum improve held-out recovery?
- Does prioritizing failed or near-failed rollouts help under a fixed interaction budget?

Choose one ablation, not all of them, if the two-month deadline is strict.

---

## 5. Tentative Timeline — 8 Weeks

| Week | Work | Exit criterion |
|---|---|---|
| **1** | Install and run the pretrained DP; reproduce nominal evaluation | Baseline runs reliably |
| **2** | Add one disturbance mechanism and verify controlled failures | Disturbance is repeatable; training/test conditions are defined |
| **3** | Run a short DPPO fine-tuning experiment without custom recovery logic | Rollouts, rewards, updates, and checkpoints work |
| **4** | Implement recovery-focused rollout collection and fine-tuning | First recovery-focused checkpoint |
| **5** | Run the main comparison: pretrained DP, standard DPPO, recovery-focused DPPO | Comparable evaluation results |
| **6** | Evaluate held-out disturbance conditions and nominal performance | Main results and robustness curves |
| **7** | Run one ablation; inspect failure cases and action behavior | Evidence supporting or challenging the hypothesis |
| **8** | Consolidate results, plots, limitations, and final report/presentation | Complete project deliverable |

### Two-week feasibility gate

At the end of Week 2, verify that:

- the pretrained policy runs in the selected simulator;
- the disturbance can be injected reliably;
- the task remains solvable after the disturbance; and
- evaluation can distinguish nominal and disturbed episodes.

If these conditions are not met, simplify the environment or disturbance design before adding custom training logic.

---

## 6. Expected Outcomes and Interpretation

The experiment could produce several valid outcomes:

- **Recovery-focused DPPO improves held-out recovery and preserves nominal success:** supports the hypothesis that targeted recovery experience can improve recovery generalization.
- **It improves only on training disturbances:** indicates disturbance-specific adaptation, with limited generalization.
- **Standard DPPO performs just as well:** suggests the benefit may come from RL fine-tuning generally rather than recovery-focused sampling.
- **Recovery improves but nominal performance degrades:** exposes a recovery–retention trade-off.
- **Neither method improves:** may indicate insufficient exploration, a sparse reward, inadequate disturbance coverage, or a limitation of the policy/environment setup.

Negative results are informative if the experiment is controlled and the failure modes are analyzed.

---

## 7. Resources and References

1. **DPPO paper and project page:** <https://diffusion-ppo.github.io/>
2. **Official DPPO implementation:** <https://github.com/irom-princeton/dppo>
3. **REACH, CVPR 2026:** <https://openaccess.thecvf.com/content/CVPR2026/html/Ke_REACH_Explicit_Recovery_Behavior_for_Diffusion_Policies_CVPR_2026_paper.html>
4. **MoE-DP, arXiv preprint:** <https://arxiv.org/abs/2511.05007>
5. **MoE-DP project page:** <https://moe-dp-website.github.io/MoE-DP-Website/>
6. **Diffusion Policy implementation:** <https://github.com/real-stanford/diffusion_policy>

### Final working title

**Recovery-Aware Reinforcement Learning Fine-Tuning of Diffusion Policies under Manipulation Disturbances**

### One-sentence summary

We will fine-tune a pretrained diffusion policy with DPPO using deliberately disturbed simulation rollouts, then test whether recovery-focused training improves generalization to held-out manipulation disturbances without sacrificing nominal task performance.

### Course-project positioning

Present recovery as the application and the diffusion model's generated behavior after RL fine-tuning as the technical subject. To keep the project feasible, reuse an existing pretrained policy, simulator, and DPPO implementation; avoid building a new simulator or claiming a new diffusion optimizer. A rigorous evaluation on one task is preferable to several shallow task demonstrations.
