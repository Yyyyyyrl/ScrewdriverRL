# Two-Stage RMA Training Pipeline

## Overview

ScrewdriverRL uses a two-stage **Rapid Motor Adaptation (RMA)** pipeline adapted from Kumar et al. (2021) and the HORA system (Qi et al. 2023). The central problem it solves is the **sim-to-real gap**: a policy trained in simulation has access to ground-truth physical state (exact object pose, friction, inertia) that does not exist on a real robot. A naïve policy that reads this privileged state cannot deploy.

RMA's answer is a two-stage decomposition:

1. **Stage 1 — Teacher**: Train a high-quality policy that reads privileged state through an *asymmetric critic*. The actor never touches the privileged info, so it is already deployable.
2. **Stage 2 — Adaptation**: Train a small encoder that maps the robot's own sensorimotor history (joint positions and targets over the last 30 steps) to a prediction of the privileged state. At deployment, this replaces ground-truth state with an inferred estimate.

The key insight is that Stage 1 produces the best possible *behaviour* without worrying about what information is available, and Stage 2 recovers deployability using only data that a real robot can actually observe.

---

## Observations and information asymmetry

### Policy observations (27-D, available at deployment)

| Slice | Dim | Content |
|---|---|---|
| `finger_q` | 12 | Joint positions of index, middle, thumb (4 DOF each), read from encoders |
| `cur_targets` | 12 | Current joint position targets from the previous policy step |
| `screwdriver_euler` | 3 | Euler angles of the screwdriver's 3-DOF mount joint (tilt-x, tilt-y, rotation-z) |

The 3 Euler angles are observable on a real robot via a wrist-mounted or overhead camera with a fiducial marker on the screwdriver handle — or via tactile inference, which Stage 2 partially supports. In the context of this task the screwdriver mount is a controlled articulation in simulation, so the Euler angles are exact.

### Privileged observations (17-D, simulation-only)

| Slice | Dim | Content | Why it helps |
|---|---|---|---|
| `screwdriver_euler` | 3 | Same as policy obs, included for completeness | — |
| `screwdriver_angvel` | 3 | Angular velocity of the screwdriver joints | Critic can value states where the screwdriver is already spinning vs. stationary |
| `screwdriver_rel_pos` | 3 | Root position of screwdriver relative to hand root | Quantifies drift during long episodes |
| `screwdriver_quat` | 4 | World-frame quaternion of the screwdriver body | Avoids gimbal lock in the critic's value estimate at large tilts |
| `friction` | 1 | Contact friction coefficient | A single friction number is the most predictive parameter for how much torque is needed |
| `fingertip_axis_dist` | 3 | Per-fingertip distance to the handle axis segment | Direct contact quality signal; the critic can distinguish "gripping" from "hovering" |

**Why include `screwdriver_euler` in both spaces?** The policy obs already contains it. Repeating it in the privileged obs ensures the critic's linear layers can trivially learn to weight angular velocity and euler position jointly without having to infer position from the policy-obs channel.

**Why 3 fingertip distances and not contact forces?** Isaac Lab's `ImplicitActuator` model does not expose per-finger contact forces without dedicated contact sensors, which add setup cost and noisy gradients. Axis-distance is a clean geometric proxy: ~0.03 m at fingertip-pad contact, rising sharply as the finger lifts. It is sufficient for the critic to distinguish the "gripping" from "hovering" regime.

### Proprioceptive history (30 × 24-D)

Each history frame contains `[finger_q(12), cur_targets(12)]` — entirely motor-side. No external sensors. The 30-frame window at 10 Hz policy rate = 3 seconds of recent history.

**Why motor-side only?** The HORA paper and subsequent work (Chen et al., 2023; Qi et al., 2023) consistently show that motor history is sufficient to infer object properties during in-hand manipulation. The intuition: if friction is low, the finger targets diverge from joint positions (slipping); if the object is heavy, joint positions lag targets under load. These patterns are statistically detectable without force sensors.

---

## Stage 1: Asymmetric Actor-Critic

### Architecture

```
                 ┌─────────────────────────────────┐
Policy obs (27)  │   Actor MLP [1024, 512, 256, 128]│ → action (12)
                 └─────────────────────────────────┘

Privileged obs (17) ┌───────────────────────────┐
                    │ Critic MLP [512, 256, 128] │ → V(s)
                    └───────────────────────────┘
```

The actor and critic are entirely separate networks with no shared weights. The actor only receives the 27-D policy observation. The critic receives the 17-D privileged observation.

### Why asymmetric and not shared?

The standard argument for weight sharing in actor-critic is that a shared representation improves sample efficiency. This is true when both inputs are the same. When the critic input is strictly larger (a superset of the actor input), sharing is counterproductive:

1. **Gradient interference**: The critic's gradients flow into a shared trunk that the actor must also use for action prediction. The privileged features can dominate the gradient signal and distort the actor's learned representation toward information it cannot access at deployment.
2. **Information leakage**: If weights are shared and the actor's input is a sub-sequence of the critic's input, the actor MLP layers that follow the shared trunk can implicitly learn to "expect" patterns created by the privileged dimensions, causing silent distributional shift at deployment when those dimensions are absent.

Maintaining fully separate networks eliminates both problems. The actor is trained by policy gradients only, through the reward signal and the advantage estimate computed by the separate critic. This is the architecture used by Lee et al. (2020), Kumar et al. (2021), and adopted by the MFR benchmark.

### Advantage estimation with privileged critic

The actor's advantage is `A(s,a) = Q(s,a) − V(s)`, where `V(s)` is computed from the privileged critic. Because the critic has strictly more information, it produces lower-variance value estimates, which lowers the variance of the advantage and accelerates learning. This benefit is well-established (see Pinto et al., 2017; Liang et al., 2021).

Concretely: the critic knows the current friction coefficient and fingertip contact distances. It can therefore assign different values to two states that look identical from the actor's 27-D view but differ in contact quality. The resulting tighter advantage estimates reduce the PPO clipping threshold's effective exposure to noisy gradient updates.

### RL-Games integration

RL-Games activates the asymmetric critic when two conditions hold:

1. The environment's `state_space` is non-zero. We set `env_cfg.state_space = 17` at runtime when `--stage 1` is selected.
2. The YAML config contains a `central_value_config` block. This is present in `agents/rl_games_ppo_cfg.yaml` and specifies the critic network architecture and its own optimiser settings (learning rate `1e-4`, separate from the actor's `3e-4`).

When both conditions are true, RL-Games' `a2c_common.py` calls `env.get_state()` to retrieve the privileged obs for each timestep and passes it to the central value network instead of the policy obs.

### Separate learning rates

The critic (`1e-4`) is trained at a lower rate than the actor (`3e-4`). The critic MLP is smaller and has a lower-variance target (privileged state → scalar value), so it converges faster and a lower rate reduces oscillation once converged. The actor needs a higher rate because its update signal (advantage-weighted log-prob gradient) is inherently noisier.

### Checkpoint layout

RL-Games saves checkpoints to `runs/<task>/nn/<run_name>.pth`. Each checkpoint contains:
- `model`: actor + critic state dicts (RL-Games saves both in one file)
- `optimizer`: optimiser state
- `frame`: total environment steps
- `epoch`: training epoch count

Stage 2 loads only the actor portion of this checkpoint via `runner.create_player()`.

---

## Stage 2: Proprioceptive Adaptation

### What it learns

Stage 2 trains `ProprioAdaptNet` to solve the regression:

```
f(proprio_history[t-29 : t]) ≈ privileged_obs[t]
```

where `proprio_history` is the 30×24 motor history and `privileged_obs` is the 17-D vector the critic was trained on. The loss is mean-squared error.

At deployment, the predicted 17-D vector can augment or replace the screwdriver Euler angles in the policy observation, or serve as an auxiliary input for a downstream fine-tuning stage. In this implementation Stage 2 is a standalone supervised training loop; the actor is not fine-tuned (see design choice discussion below).

### Network architecture: ProprioAdaptNet

```
Input: (batch, T=30, D=24)  — 30 frames × [finger_q(12), cur_targets(12)]

Frame encoder (shared across time steps):
  Linear(24 → 32) → ELU → Linear(32 → 32) → ELU
  Output: (batch, 30, 32)

Temporal convolution stack (Conv1d over the time axis):
  Permute to (batch, 32, 30)   — channels first
  Conv1d(32, 32, kernel=9, stride=2) → ELU    output seq ≈ 11
  Conv1d(32, 32, kernel=5, stride=1) → ELU    output seq ≈ 7
  Conv1d(32, 32, kernel=5, stride=1) → ELU    output seq ≈ 3
  Flatten → ~96-D

Output head:
  Linear(96 → 17)   — predicted privileged obs
```

This is a direct adaptation of HORA's `ProprioAdaptTConv` (Qi et al., 2023, Appendix B). The design rationale for each component:

**Frame encoder**: A shared MLP across time steps extracts per-step features before temporal aggregation. This is equivalent to a 1×1 convolution in the channel dimension and allows the temporal conv to operate on learned features rather than raw joint angles, which are on different scales (position vs. target).

**Strided convolution (kernel 9, stride 2) first**: The first layer has a large receptive field (9 out of 30 frames = 0.9 seconds) and halves the sequence length. This is intentional: early layers capture the coarse temporal structure (e.g., periodic grasping motion), while later layers (kernel 5, no stride) refine within each captured window. Using a large kernel first reduces the number of parameters needed compared to stacking multiple small-kernel layers to reach the same receptive field.

**Why three conv layers?** Three layers provide a total receptive field of 9 + (5−1) + (5−1) = 17 frames (1.7 seconds). The remaining 13 frames provide context for the first layer. This is enough to capture one full rotation cycle at typical learning speeds (≈ 1–2 rad/s, so 2–6 rad per 3-second window ≈ 0.3–1 full turn). Deeper networks did not improve results in HORA's ablation.

**No recurrent layers (LSTM/GRU)**: Recurrent layers introduce hidden state that must be carried across steps, complicating the training loop and requiring BPTT. Given the fixed 30-frame window, the temporal conv achieves the same receptive field without hidden state, enabling simpler batched training over randomly sampled windows.

### Training procedure

Stage 2 is **supervised learning, not RL**. This is intentional:

1. **No exploration needed**: The Stage 1 policy already produces diverse, high-quality manipulation trajectories. Sampling from it with a frozen policy generates a rich distribution of `(history, state)` pairs without any reward shaping.
2. **Stable gradients**: MSE regression has well-behaved gradients compared to policy gradient methods. The network converges in hundreds of iterations rather than millions.
3. **Decoupled from policy quality**: If the adaptation training is run after Stage 1 converges, the collected trajectories span the full support of the learned behaviour distribution, which is exactly what the adaptation network needs to generalise.

Each training iteration:
1. Rolls out the frozen Stage 1 actor for `adapt_rollout_steps` policy steps, collecting `(proprio_hist, priv_obs)` at each step across all envs.
2. Concatenates into a flat batch of `adapt_rollout_steps × num_envs` samples.
3. Runs `num_epochs_per_iter=5` gradient passes over randomly permuted mini-batches of size 4096.

**Why not fine-tune the actor after Stage 2?** The HORA paper includes an optional Stage 3 (policy fine-tuning with the adaptation module replacing privileged obs). We omit this for two reasons: (1) the actor already generalises well because it was never conditioned on privileged obs; and (2) fine-tuning introduces new instabilities if the adaptation network's predictions are noisy, potentially degrading the Stage 1 policy. For sim-to-real transfer, the recommended path is to deploy Stage 1 + Stage 2 and evaluate empirically whether actor fine-tuning is needed.

### Hyperparameter choices

| Parameter | Value | Justification |
|---|---|---|
| History length | 30 frames | 3 seconds at 10 Hz. Covers 1–3 full rotation cycles at typical learning speeds. HORA used 30 frames at 20 Hz (1.5 s); we use the same count at half the rate for a longer window. |
| Frame dim | 24 | `finger_q(12) + cur_targets(12)`. Targets are included because slip is most apparent in the gap between commanded and actual position. |
| Rollout steps per iter | 512 | Provides `512 × 2048 = ~1M` samples per iteration with 2048 envs. Comparable to HORA's experience buffer size. |
| Training iterations | 500 | Empirically 200–500 iterations suffice in HORA and MFR for the adaptation MSE to reach below 0.01 on held-out validation data. |
| Learning rate | `1e-3` | Higher than Stage 1 actor (`3e-4`) because the supervised objective has lower gradient variance. Standard for supervised regression on normalised inputs. |
| Batch size | 4096 | Balances GPU memory use vs. gradient noise. Larger batches improve gradient stability; 4096 fits comfortably in 24 GB VRAM with the network size. |

---

## Information flow diagram

```
Training time:
                           Stage 1
  ┌──────────────────────────────────────────────────────────────┐
  │  Env → privileged_obs(17) ──────────────────→ Critic MLP    │
  │  Env → policy_obs(27) ──────────────────────→ Actor MLP ──→ action
  │  Env → proprio_hist(30×24) [saved to buffer]                 │
  │                PPO gradient ←────────────────────────────────┤
  └──────────────────────────────────────────────────────────────┘

                           Stage 2
  ┌──────────────────────────────────────────────────────────────┐
  │  Frozen Actor → action → Env → (proprio_hist, priv_obs)     │
  │  ProprioAdaptNet(proprio_hist) ─→ pred_priv_obs             │
  │  MSE(pred_priv_obs, priv_obs) ─→ backprop ProprioAdaptNet   │
  └──────────────────────────────────────────────────────────────┘

Deployment:
  ┌──────────────────────────────────────────────────────────────┐
  │  Robot proprioception (30×24) → ProprioAdaptNet → est_priv  │
  │  policy_obs(27) + [optional: est_priv] → Actor MLP → action │
  └──────────────────────────────────────────────────────────────┘
```

---

## Limitations and known gaps

**No actor fine-tuning.** Stage 1 + Stage 2 is the base pipeline. If adaptation MSE is high on real hardware, a Stage 3 policy fine-tuning pass with the adaptation module replacing privileged obs is recommended. This follows the full HORA procedure.

**Friction is a scalar.** The privileged obs exposes a normalised rotation-damping value as the friction proxy. In reality, friction varies across finger pads, screwdriver texture, and contact angle. The adaptation network must infer an effective average; this is a known simplification shared with the MFR benchmark.

---

## Domain randomisation

### Why it is mandatory, not optional

Without domain randomisation, every episode in every env has identical dynamics (same damping, same mass, same PD gains). The adaptation network's input — the proprioceptive history — looks nearly identical across all 2048 envs. MSE loss is minimised by predicting the dataset mean regardless of the history content. The gradient of the adaptation loss with respect to the network weights is near zero: the network cannot learn to distinguish different environments because there is nothing to distinguish.

With DR, each env receives a different damping/mass/gain sample at every reset. Two envs with the same policy obs `(finger_q, cur_targets, euler)` will produce different proprio histories because the screwdriver responds differently to the same finger forces. The adaptation network must learn to detect those differences — slip signatures, oscillation patterns, lag between target and position — which is exactly what enables sim-to-real transfer.

This is not a secondary benefit: it is the entire point of the two-stage pipeline. RMA (Kumar et al. 2021) showed that without DR the Stage 2 adaptation provides no improvement over not using it at all.

### What is randomised and why

Four parameters are randomised at every episode reset via `_randomise_dynamics()` in `screwdriver_rotation_env.py`.  All ranges are multiplicative scales on the base value so the ratio of variation is fixed regardless of the base.

**1. Screwdriver rotation joint damping — `rotation_damping_range = (0.5, 2.0)`**

This is the primary friction proxy. A low-damping episode (0.075 N·m·s/rad) means the screwdriver spins freely under light finger force — the fingers barely need to push. A high-damping episode (0.30 N·m·s/rad) requires sustained deliberate pushing. The policy experiences both extremes and the adaptation network learns to identify which regime it is in from the lag between joint targets and positions (fingers slip and over-shoot on low-damping; strain and under-shoot on high-damping).

The value is stored per-env in `_env_rotation_damping` and exposed in the privileged obs as `damping / base_damping` (normalised to 1.0 baseline), so the critic knows the actual environment difficulty.

**2. Screwdriver body mass — `screwdriver_mass_range = (0.5, 2.0)`**

A heavier screwdriver (0.60 kg) has higher rotational inertia and requires more sustained impulse to accelerate. A lighter one (0.15 kg) accelerates easily but overshoots. Mass variation is the second-largest source of sim-to-real gap in in-hand tasks after friction (Kumar et al. 2021, Table 2). The adaptation network detects mass from acceleration rate: a heavy object accelerates slowly even under large finger force.

**3 & 4. Finger joint stiffness and damping — `(0.8, 1.2)` each**

A single scale is applied to all active finger joints within one env per episode (not per-joint), so the grasp character is consistent within an episode. Using per-joint scales would create inconsistent grasps that the base PPO policy has not learned to handle. The ±20% range simulates manufacturing variation in servo torque constants and gear damping. The effect on the proprio history: stiffer fingers overshoot targets less; more damped fingers converge slower. The adaptation network can detect this from the target-vs-position residual pattern.

**Observation noise — `obs_noise_std = 0.01` rad**

Gaussian noise (σ = 0.01 rad ≈ 0.6°) is added to every policy observation at each step, simulating encoder quantisation and marker-tracking jitter. This is not stored per-env (it is re-sampled every step) and does not appear in the privileged obs. Its purpose is to prevent the policy from overfitting to noiseless simulation observations, which would cause performance degradation on real hardware where sensor noise is unavoidable.

### What is *not* randomised and why

**Centre of mass (COM)**: HORA randomises COM offset. This is most important for objects that can roll (sphere, cylinder held freely). For a screwdriver on a 3-DOF universal joint, the COM offset mainly affects the torque required to hold upright, which is already captured by the mass randomisation. Adding COM variation would increase the DR space without proportional benefit.

**Ground-plane friction**: The global friction coefficient is fixed at 1.5. The finger-to-handle friction is implicit in the rotation damping. Randomising the ground plane would affect the hand's base stability but the hand is fixed (`fix_base=True`), so this has no effect.

**Initial joint positions**: Start positions are deterministic (pregrasp pose) to ensure the policy always begins from a configuration that has been validated to produce initial contact. Adding start-position noise risks no-contact episodes in Phase 1, which provides zero gradient.

---

## References

- Kumar, A., Fu, Z., Pathak, D., & Malik, J. (2021). **RMA: Rapid Motor Adaptation for Legged Robots.** *RSS 2021*. [arXiv:2107.04034](https://arxiv.org/abs/2107.04034)
  > Original two-stage RMA pipeline: privileged-info teacher (Stage 1) + temporal-conv adaptation module (Stage 2) trained via supervised regression on teacher rollouts. Direct architectural ancestor of our Stage 2 ProprioAdaptNet.

- Qi, H., Yi, B., Suresh, S., Lambeta, M., Ma, Y., Calandra, R., & Malik, J. (2023). **General In-Hand Object Rotation with Vision and Touch.** *CoRL 2023*. [arXiv:2309.09979](https://arxiv.org/abs/2309.09979)
  > HORA system. Applies RMA to in-hand object rotation with the Allegro hand. Directly references our implementation: observation space design (3-frame history + 1D temporal conv), ProprioAdaptTConv architecture (kernel sizes [9,5,5], stride [2,1,1]), and the two-stage training loop with frozen actor in Stage 2.

- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., & Hutter, M. (2020). **Learning quadrupedal locomotion over challenging terrain.** *Science Robotics*. [DOI:10.1126/scirobotics.abc5986](https://doi.org/10.1126/scirobotics.abc5986)
  > Establishes asymmetric actor-critic with privileged information for locomotion. First large-scale demonstration that a critic with privileged state improves actor training without requiring the actor to use that information.

- Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., & Abbeel, P. (2017). **Asymmetric Actor Critic for Image-Based Robot Learning.** *RSS 2018*. [arXiv:1710.06542](https://arxiv.org/abs/1710.06542)
  > Formal treatment of asymmetric actor-critic. Shows that providing the critic with additional state information reduces variance in value estimation, which reduces policy gradient variance and accelerates convergence.

- Chen, T., Xu, J., & Agrawal, P. (2023). **A System for General In-Hand Object Re-Orientation.** *CoRL 2021*. [arXiv:2111.03043](https://arxiv.org/abs/2111.03043)
  > Demonstrates that motor-side proprioception alone (joint positions and targets) is sufficient to infer object dynamics for in-hand manipulation, supporting our choice to exclude tactile/force sensors from the proprioceptive history.

- MFR Benchmark (internal reference). `MFR_benchmark/MFR_benchmark/rma/` — RMA teacher-student implementation for screwdriver tasks with the Allegro hand. Direct source for the `asymmetric_obs` observation split and the screwdriver privileged-obs structure used in this project.
