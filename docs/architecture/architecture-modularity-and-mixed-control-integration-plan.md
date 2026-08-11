# ScrewdriverRL Modularity, Scalability, and Mixed-Control Integration Plan

| Metadata | Value |
|---|---|
| Status | Proposed architecture and implementation plan |
| Last updated | 2026-07-11 |
| Revision | 2 — same-day maintainer-review revision; changes listed in Section 45.1 |
| Reviewed Git revision | 868f101602f87405d6b07696270375b55fcf8579 |
| Reviewed dirty state | Yes: tracked tree was clean before this document/README link; reviewed untracked supplement is identified in Section 40.0 |
| Baseline inventory | Phase-0 deliverable: artifacts/baselines/2026-07-11/review-inventory.json; its digest must be added here when created |
| Primary audience | Repository maintainers, simulation and RL engineers, robot-runtime engineers, safety reviewers, and pipeline integrators |
| Approval status | Pending maintainer and deployment-safety review |
| Implementation tracking | Use the work-item IDs in Section 31 |

## Navigation

This is intentionally a long reference document. Suggested entry points:

- Executive/review context: Sections 3–6.
- Target architecture and contracts: Sections 7–12.
- Runtime, safety, timing, and mixed-control handoff: Sections 13–18.
- Evaluation, training, packaging, APIs, and observability: Sections 19–24.
- Testing, CI, extensibility, and file migration: Sections 25–28.
- Phased implementation, backlog, and acceptance criteria: Sections 29–33.
- Risks, decisions, rollout, evidence, glossary, and document governance: Sections 34–45.

Useful direct links:

- [Immediate blockers](#5-immediate-correctness-and-safety-blockers)
- [Execution environments and verification split](#24-execution-environments-and-verification-split)
- [Capacity model and committed core](#25-capacity-model-committed-core-and-deferred-reference-design)
- [Environment partition of all work items](#execution-environment-partition-part-i--part-ii)
- [Target architecture](#7-target-architecture)
- [Observation and action architecture](#9-observation-and-action-architecture)
- [Policy package architecture](#11-policy-package-architecture)
- [SafetySupervisor](#16-safetysupervisor)
- [Mixed teleoperation-to-RL integration](#18-mixed-teleoperation-to-rl-integration)
- [Phased implementation plan](#29-phased-implementation-plan)
- [Detailed work-item backlog](#31-detailed-work-item-backlog)
- [Top-level acceptance criteria](#33-top-level-acceptance-criteria)
- [First implementation pull requests](#41-first-implementation-pull-requests)

---

## 1. Purpose

This document is the standalone technical plan for evolving ScrewdriverRL from a research-oriented collection of Isaac Lab tasks and a LinkerHand deployment script into a maintainable manipulation platform with:

- independent task and hand-model axes;
- straightforward addition of new robot hands, task variants, grasp profiles, and hardware transports;
- one authoritative observation and action contract shared by training, evaluation, and deployment;
- reproducible, immutable, semantically validated policy artifacts;
- an embeddable runtime rather than a standalone hardware-only application;
- a safe teleoperation-to-RL handoff in which an operator positions an arm and hand, the arm transitions to a deterministic hold controller, and an RL policy takes temporary ownership of fine finger manipulation;
- explicit return of control to teleoperation;
- structured safety, timing, observability, testing, and promotion gates.

This is intentionally both an architecture specification and an implementation plan. A future engineer should be able to use it without access to the original review conversation.

## 2. Scope and non-goals

### 2.1 In scope

- Mounted screwdriver rotation.
- Free-object in-hand rotation/orientation.
- Allegro v4 and LinkerHand L20/G20.
- Isaac Lab simulation and RL-Games training.
- Stage-1 teacher and Stage-2 adaptation training.
- Environment-free policy inference.
- Direct CAN, current ROS1 transport, future ROS2 or in-process integration.
- Policy artifact creation, validation, promotion, deployment, and rollback.
- Arm/hand controller ownership during teleoperation-to-RL transitions.
- Packaging, assets, configuration, tests, CI, and developer workflow.

### 2.2 Not in scope

The first architecture migration must not:

- retune rewards merely because code is being moved;
- change trained policy behavior without an explicit behavior-change work item;
- choose a specific future robot-arm vendor;
- require ROS as the core runtime API;
- make the RL policy responsible for general arm motion;
- treat opening the hand as a universally safe failure response;
- silently convert legacy artifacts into live-deployable artifacts;
- add abstraction layers that have no concrete use in the current two-task, two-hand problem.

### 2.3 Behavior-preserving work versus behavior-changing work

Every implementation change must be labeled as one of:

1. Behavior-preserving extraction
   - Moves existing logic behind a new interface.
   - Must pass trace-parity and golden-output tests.
   - Must not require retraining.

2. Contract correction
   - Fixes a mismatch between training, evaluation, and deployment.
   - May require repackaging or retraining.
   - Must be called out in release notes and artifact compatibility metadata.

3. Behavior change
   - Changes observations, action timing, limits, physics, rewards, reset distribution, hardware mapping, or safety behavior.
   - Requires an explicit experiment, evaluation baseline, and promotion decision.

This distinction is essential. Architectural refactoring and policy retuning must not be combined into one unreviewable change.

### 2.4 Execution environments and verification split

Two machines execute this plan, and they have different capabilities. Every work item is developed and verified against one of three named environments:

- ENV-A — the macOS development machine (this repository's primary editing environment). Available: pure Python, CPU PyTorch, RL-Games (a CPU install already runs tests/test_algo.py), pytest, packaging/build tooling, fake clocks, fake transports, and fake drivers. Not available: Isaac Sim/Lab (no macOS support), CUDA, ROS1, the Linker SDK, and CAN hardware.
- ENV-B — the Ubuntu training rig with a CUDA-capable GPU and Isaac Sim/Lab. Everything simulator- or GPU-dependent verifies here: golden-trace capture, Isaac smoke and parity, Stage-1/Stage-2 training, and the named memory/throughput profiles.
- ENV-C — the Ubuntu hardware station (deployment box with ROS1, the Linker SDK, CAN, and the physical hand; later the arm). It may or may not be the same host as ENV-B; it is tracked separately because it needs no Isaac and adds physical risk.

Rules:

- every work item names its verification environment; the Part I / Part II partition in Section 29 is the per-item authority;
- labels: A means fully implementable and verifiable on ENV-A; A→B and A→C mean implemented and unit-tested on ENV-A against committed fixtures with a named final verification step on the rig or station; B and C mean only meaningfully verifiable there;
- "done" is environment-specific: an item verified only on ENV-A must not claim simulator or hardware behavior, and vice versa;
- fixture workflow: ENV-B records deterministic golden traces and small fixtures; they are committed; ENV-A iterates against them; simulator-touching changes re-verify on ENV-B before merge (TEST-006);
- ENV-A tests must be OS-portable — they run on the macOS machine locally and on Linux CPU runners in CI: no macOS-only shell assumptions inside tests, CPU-only Torch, no CUDA or MPS device requirements;
- batch ENV-B and ENV-C work into planned rig/station sessions and keep a queued list of pending verifications, so ENV-A development never blocks on rig access;
- CI mapping (Section 26): pull-request CPU jobs are ENV-A; the self-hosted Isaac jobs are ENV-B; protected hardware jobs are ENV-C. Test markers (Section 25.1) map the same way.

### 2.5 Capacity model, committed core, and deferred reference design

Reality: this repository has one maintainer working with coding agents. The full backlog (~95 work items, roughly ten XL and thirty L) is a multi-person-year program if executed completely; executing it serially would stall the research this repository exists for. The plan is therefore split into a committed core, which is scheduled work, and triggered reference design, which is design-complete documentation that activates only on a named trigger and must not be built speculatively.

Committed core, in order:

1. Phase 0 — baseline, containment, and packaging foundation, including the FOUND-011 trained-artifact inventory and FOUND-012 throughput baseline.
2. Phase 1 — deployment inference parity; unblocks the only currently intended hardware task.
3. Phase 2, slimmed — the two existing hands and current task variants only; no plugin machinery.
4. Phase 3 — immutable policy artifacts.
5. The Stage-2 memory slice (TRAIN-005/006/007) — may be pulled forward any time after Phase 1; it is nearly independent of the rest and removes a real ~30 GiB obstacle to planned 8,192-environment campaigns.
6. Phase 7 core — calibration, left adapter, strict loading and containment, fake transports, scheduler, safety-supervisor logic, and watchdog basics, as needed for the next hardware session.

Triggered reference design:

| Work | Trigger to activate |
|---|---|
| Phase 4 generic task/hand composition | a second task-hand variant is actually planned, or duplication cost is measured and painful |
| Phase 5 remainder (runners, evaluators, promotion) | the first real promotion decision or hardware campaign needs them |
| Phase 6 distribution/packaging completion | first external consumer or publication |
| Phases 8–9 mixed control and HIL | an arm and teleoperation stack are selected; ADR-016/ADR-017 accepted |
| Phase 10 extensibility proof plus SPEC-011 | a third hand is committed to |
| Trust track (ART-010, TEST-004, signatures/rotation/revocation) | artifacts cross a machine or organization trust boundary |
| TRAIN-008 interleaved Stage-2 | evidence that fixed-corpus training limits adapter quality |

Research-continuity rule: architecture work interleaves with research. No phase may block a planned training campaign; if a campaign interrupts a phase, work lands at the last green resting state and resumes later.

Resting-state invariant: every phase exit is a stable state — all tests green, no dual implementations without a recorded owner and deprecation path. Permanently stopping after any phase must leave the repository strictly better than before that phase started.

---

## 3. Executive summary

The repository has several strong foundations:

- simulator-independent reward primitives;
- a shared mounted-screwdriver base environment;
- an environment-free PyTorch policy implementation;
- strict validation of several Linker calibration fields;
- useful dry-run, ramp, hold, watchdog, and recording behavior;
- explicit curriculum and domain-randomization configuration;
- CPU-oriented tests for reward and deployment utilities.

However, the repository is not yet ready to act as a general task-by-hand platform or a mixed-controller runtime.

The most important conclusions are:

1. The two task families and two hand models are not independent axes.
2. The current Linker in-hand policy artifact cannot run through DeployPolicy because training expects a 96-dimensional, three-frame, scaled proprioceptive input while deployment supplies one raw 32-dimensional frame.
3. The deployment evaluation path bypasses the actual environment-free runtime and can therefore pass while hardware inference fails.
4. Policy artifacts do not identify the hand, semantic joint order, units, control rate, observation codec, calibration compatibility, or complete provenance.
5. Teleoperation-to-RL switching is not bumpless: policy reset returns internal targets to home rather than adopting the target actually applied by the previous controller.
6. Simulation assumes an absolute fixed wrist and provides no explicit task_from_wrist or arm-hold contract.
7. The package build backend is invalid, and a wheel would not contain the YAML and asset resources needed at runtime.
8. Deployment artifacts are mutable task-level files that can be overwritten by smoke runs.
9. Deployment combines policy, mapping, transport, scheduling, signals, logging, and partial safety in one Linker-specific class.
10. Stage-2 data collection scales to roughly 30 GiB transient history memory at current in-hand defaults.

The recommended direction is composition around five central contracts:

- HandSpec
- TaskSpec
- TaskHandProfile
- ProprioCodec, ActorInputAssembler, ConditioningSpec, and ActionTransform
- PolicyPackage, DeploymentBinding, and LaunchConfig

The runtime should then compose:

- PolicySession
- SafetySupervisor
- HandAdapter
- Transport
- HandoffSupervisor
- ArmHoldController

How the plan executes in practice is bounded by Sections 2.4 and 2.5: every work item is partitioned between what is implementable and verifiable on the macOS development machine (Part I) and what requires the Ubuntu CUDA/Isaac rig or hardware station (Part II), and only a committed core is scheduled work — the remainder is triggered reference design.

---

## 4. Current capability matrix

| Capability | Allegro v4 | Linker L20 |
|---|---:|---:|
| Mounted screwdriver rotation | Implemented | Implemented |
| Free-object in-hand rotation | No generic implementation | Implemented as a Linker-specific environment |
| Grasp-cache generation | No generic implementation | Linker-specific |
| Environment-free policy inference | Partially generic | Implemented, but not valid for the current in-hand 96-D policy |
| Hardware joint adapter | Missing | Left-hand-specific |
| Live transport | Missing | CAN and ROS1 |
| Mixed arm/hand handoff | Missing | Missing |
| Teleoperation source (arm/hand command stream) | Missing | Missing |

No teleoperation stack exists in this repository or its current deployment setup; Section 18 is therefore reference design until ADR-016/ADR-017 are accepted.

The target is not necessarily to train all four task-hand combinations immediately. The target is to make the architecture capable of expressing each combination without copying the task environment or deployment loop.

---

## 5. Immediate correctness and safety blockers

### 5.1 In-hand observation contract mismatch

Training behavior:

- A hand frame is 32 values: scaled joint position for 16 independent joints plus 16 current targets.
- The actor receives the latest three frames, producing 96 proprioceptive values.
- The adapter receives a 30-frame history.
- The actor normalizer is therefore 96-dimensional.

Current deployment behavior:

- One raw 32-value joint-position and target frame is constructed.
- No training-side joint scaling is applied.
- No actor-frame stack is created.
- The 32-value vector is passed to a 96-dimensional normalizer and actor.

Required response:

- Do not treat Linker in-hand artifacts as live-deployable until this is fixed.
- Extract simulator proprio/history construction into a shared ProprioCodec and actor-input assembly contract.
- Run the actual environment-free PolicySession inside the simulator deployment gate.
- Add golden trace parity covering every actor input, adapter input, raw action, and integrated target.

### 5.2 Same-size wrong-hand artifacts can reach the Linker mapper

Allegro 4F and Linker both use 16-dimensional action vectors. Anonymous shape validation cannot distinguish them.

Required response:

- Add a semantic joint schema with ordered names and units.
- Bind every artifact to a hand model, side, semantic schema ID, task version, observation codec, action transform, and control period.
- Make live loading fail closed before any connection or ramp.

### 5.3 Teleoperation handoff resets toward home

Current reset semantics set internal current targets to bundle home even if measured joint state is supplied. This is only continuous because the standalone deployer first owns the hand and ramps it to home.

Required response:

- PolicySession must be primed from measured joint state and the post-mapping upstream target accepted at the binding-required acknowledgement level.
- RL should run in shadow mode before ownership transfer.
- The transition controller must report the target actually sent so policy state remains synchronized during blending.

### 5.4 Control period is not serialized

Mounted screwdriver training runs at 10 Hz. In-hand training runs at 20 Hz. Target deltas are integrated per tick.

Required response:

- Store control_dt_ns in the policy manifest.
- Treat frame timing and history offsets as part of the observation contract.
- Reject runtime rate overrides that do not exactly match the package, unless a policy explicitly declares rate invariance and has tests supporting it.

### 5.5 Bundle loading defaults safety-critical fields

Current loading may invent:

- home targets;
- lower and upper limits;
- action delta scale;
- normalizer state.

Required response:

- Manifested live artifacts must contain every required field.
- Unknown, missing, dimensionally inconsistent, or semantically incompatible fields must be errors.
- Legacy permissive loading may exist only inside an offline migration command.

### 5.6 Right-side operation is exposed without a right-hand mapping

The CLI accepts a right side while the conversion constants and functions are explicitly left-hand-only.

Required response:

- Remove or reject right-side live operation until a separately named and tested LinkerL20RightAdapter exists.

### 5.7 Mutable production artifact names

Task-level deploy.pth and deploy_last.pth are overwritten by later Stage-2 runs. This has already caused smoke artifacts to replace useful artifacts.

Required response:

- Never deploy directly from a mutable training output.
- Publish immutable policy packages.
- Promote a content-addressed package through explicit validation gates.

### 5.8 Packaging is not relocatable

Required response:

- Correct the build backend.
- Package or separately version all required YAML, URDF, mesh, manifest, and runtime assets.
- Test a clean wheel installation, not a checkout added to PYTHONPATH.

---

## 6. Design principles

### 6.1 Semantic identity over positional coincidence

Anonymous vectors such as 16 joint values are insufficient. Every boundary must carry or be bound to:

- schema ID;
- ordered semantic names;
- units;
- model and side;
- lower, upper, velocity, and acceleration limits;
- timestamp and validity information where state is involved.

### 6.2 One observation implementation

The same observation semantics must be used by:

- simulation training;
- Stage-2 teacher/student collection;
- play;
- evaluation;
- deployment simulation gate;
- hardware runtime;
- recorded-trace replay.

Duplicated implementations are forbidden unless a parity test proves equivalence at every tick.

### 6.3 One action-integration implementation

Action clipping, delta integration, target bounds, timing phase, and effective-target synchronization must have one authoritative implementation.

### 6.4 Fail closed at hardware boundaries

Convenient defaults are acceptable in experiments. They are not acceptable in live package loading, calibration binding, or hardware mapping.

### 6.5 Pure core, adapter edges

Core contracts, codecs, manifests, policy inference, state machines, and safety logic should not import Isaac, ROS, or vendor SDKs.

Isaac, ROS, CAN, and vendor SDK details belong in adapters.

### 6.6 Explicit ownership

Only one controller may own each actuator resource at a time. Arm and hand ownership are separate.

### 6.7 Effective target is authoritative

The policy must integrate from the highest-trust post-mapping effective target accepted at the binding-required acknowledgement level, not merely the candidate it generated. This matters during blending, clamping, quantization, safety filtering, and transport rejection.

### 6.8 Immutable artifacts and configurations

Resolved experiment configuration, policy packages, calibration, and deployment bindings must be immutable and content-addressed after publication.

### 6.9 Derive and validate

Observation and action dimensions must be derived from semantic specs. Padding or truncating mismatches is not an acceptable recovery strategy.

### 6.10 Structured events, external sinks

Environments and runtimes should emit structured metric and control events. Terminal formatting, JSON, TensorBoard, CSV, and external telemetry are sinks, not task logic.

---

## 7. Target architecture

### 7.1 High-level flow

    HandSpec -----------+
                        |
    TaskSpec -----------+--> ResolvedExperiment --> Isaac Environment --> Training
                        |                                  |
    TaskHandProfile ----+                                  +--> trace data
                        |
    AlgorithmProfile ---+
    CurriculumProfile --+
    DomainRandomization-+
    SimulationProfile --+
    ArmIntegration -----+

    ResolvedExperiment + trained tensors
                    |
                    v
          Immutable PolicyPackage
                    |
                    v
    Teleoperation ----+
    PolicySession ----+--> HandoffSupervisor / LeaseAuthority
    ArmHoldController-+                 |
                                       v
                                CommandArbiter
                                       |
                           Resource TransitionBlend
                                       |
                         Arm/Hand SafetySupervisor
                              /                \
                 ArmCommandGateway       HandDriver
                                         /        \
                                  HandAdapter    Transport

### 7.2 Proposed package layout

The exact names may be adjusted, but ownership boundaries should remain.

    screwdriver_rl/
      contracts/
        identifiers.py
        joints.py
        state.py
        hand.py
        task.py
        experiment.py
        observation.py
        conditioning.py
        action.py
        artifact.py
        deployment.py
        attestation.py
        trust.py
        metrics.py

      hands/
        registry.py
        allegro_v4/
          spec.py
          sim.py
          profiles.py
        linker_l20/
          spec.py
          sim.py
          profiles.py

      task_families/
        registry.py
        screwdriver_rotation/
          spec.py
          object_model.py
          reset.py
          contact.py
          reward.py
          termination.py
          observation.py
          evaluation.py
        inhand_rotation/
          spec.py
          object_model.py
          reset.py
          contact.py
          reward.py
          termination.py
          observation.py
          evaluation.py

      experiments/
        registry.py
        resolver.py
        validation.py
        builtins.py

      sim/
        isaac/
          app.py
          hand_runtime.py
          mount.py
          scene_builder.py
          direct_env.py
          sensors.py
          collision.py
          domain_randomization.py

      training/
        config.py
        runner.py
        rl_games.py
        stage1.py
        stage2.py
        stage2_streaming.py
        checkpoints.py
        artifact_export.py

      runtime/
        policy_session.py
        policy_snapshot.py
        scheduler.py
        safety.py
        ownership.py
        arbitration.py
        handoff.py
        arm_hold.py
        watchdog.py
        telemetry.py
        replay.py

      hardware/
        base.py
        driver.py
        arm_gateway.py
        calibration.py
        linker_l20/
          adapter_left.py
          adapter_right.py
          transport_can.py
          transport_ros1.py
        allegro_v4/
          adapter_right.py
          transport_vendor.py

      evaluation/
        evaluator.py
        gates.py
        report.py

      assets/
        package_resources_or_manifests/

      cli/
        train.py
        eval.py
        play.py
        package_policy.py
        deploy.py
        hand_check.py

The current modules should remain as compatibility wrappers during migration. The goal is not a single disruptive file move.

---

## 8. Core semantic contracts

### 8.1 Identifiers

Use stable, versioned identifiers:

- Hand ID: linker_l20_left_v1, allegro_v4_right_v1.
- Joint schema ID: linker_l20_independent16_rad_v1.
- Task family ID: screwdriver_rotation_v1, inhand_rotation_v1.
- Task-hand profile ID: linker_l20_screwdriver_side_grasp_v3.
- Observation codec ID: inhand_stack3_scaled_q_target_v1.
- Action transform ID: delta_position_per_tick_v1.
- Experiment ID: a resolved combination of the above.
- Policy package ID: content hash.
- Calibration ID: content hash.
- Deployment binding ID: content hash.

IDs must not encode mutable labels such as latest or production.

### 8.2 JointSchema and limit layers

JointSchema describes semantic identity and intrinsic kinematic position bounds. It must not change when a site lowers a commissioning speed cap.

Illustrative pure-Python model:

    @dataclass(frozen=True)
    class JointSpec:
        name: str
        finger: str
        role: str
        control_role: Literal["independent", "derived_follower"]
        position_lower_rad: float
        position_upper_rad: float

    @dataclass(frozen=True)
    class JointSchema:
        schema_id: str
        joints: tuple[JointSpec, ...]
        unit: Literal["rad"]

The cross-layer policy/command JointSchema contains ordered independent joints only. HandSpec separately holds derived follower JointSpecs; the union forms the hand's canonical kinematic namespace. This prevents follower joints from silently inflating policy dimensions while still giving every physical/simulator joint a typed identity.

Required invariants:

- names are unique and ordered;
- every schema joint has control_role independent and uses the canonical semantic namespace;
- intrinsic position limits are finite and lower is strictly less than upper;
- finger and role labels come from typed, versioned vocabularies;
- a schema ID cannot resolve to different content.

Keep four limit layers separate:

1. Intrinsic kinematic position limits in the independent JointSchema and HandSpec's derived JointSpecs.
2. Training/model limits in the resolved policy contract.
3. Calibrated hardware-reachability limits in HandCalibration.
4. Deployment safety caps in DeploymentBinding.

Velocity, acceleration, and jerk limits are controller, hardware, or deployment constraints, not semantic joint identity. Model them in typed ControllerMotionLimits and HardwareMotionLimits records. A field may be explicitly unavailable; do not invent an acceleration limit merely to satisfy a schema. Activation validates that the intersection of all applicable layers is non-empty and that the selected policy mapping strategy can honor it.

### 8.3 CoupledJointSpec

    @dataclass(frozen=True)
    class CoupledJointSpec:
        follower: str
        master: str
        multiplier: float
        offset: float

This is the single source used by simulation, cache metadata, model validation, and hardware documentation.

The coupling graph validator must additionally prove:

- every master and follower exists in the canonical semantic namespace;
- the directed graph is acyclic;
- each follower has exactly one derivation;
- an independent command cannot also be a follower;
- recursively derived follower bounds are finite and compatible with intrinsic bounds;
- multipliers and offsets are finite;
- simulation mimic definitions agree with the resolved graph.

### 8.4 HandSpec

HandSpec owns simulator-neutral kinematic identity and intrinsic capabilities:

    @dataclass(frozen=True)
    class HandSpec:
        hand_id: str
        model: str
        side: Literal["left", "right"]
        independent_joint_schema: JointSchema
        derived_joints: tuple[JointSpec, ...]
        coupled_joints: tuple[CoupledJointSpec, ...]
        fingers: tuple[FingerSpec, ...]
        kinematic_capabilities: frozenset[str]

Intrinsic capabilities may include:

- independent_thumb;
- opposable_thumb;
- independent_abduction;
- coupled_distal_joints.

HandSpec must not contain task reward weights or a task-specific world pose.

Joint-to-finger membership is derived from JointSpec rather than duplicated in finger_joint_names. Simulator link/body names, sensors, and collision filtering belong in HandSimSpec. Hardware telemetry and command capabilities belong in HandDescriptor, so a missing current sensor cannot make an otherwise valid simulated task-hand combination incompatible.

### 8.5 HandSimSpec

Simulation-only information belongs in a separate spec:

    @dataclass(frozen=True)
    class HandSimSpec:
        hand_id: str
        asset_ref: AssetRef
        articulation_factory_id: str
        joint_bindings: tuple[SemanticToSimJointBinding, ...]
        fingertip_body_bindings: tuple[FingerBodyBinding, ...]
        non_tip_bodies: tuple[str, ...]
        actuator_profile_id: str
        collision_profile_id: str
        sim_capabilities: frozenset[str]

This keeps runtime and artifact tooling importable without Isaac.

### 8.6 TaskSpec

TaskSpec owns task semantics:

    @dataclass(frozen=True)
    class TaskSpec:
        task_id: str
        family: str
        version: int
        object_model_family_id: str
        reset_strategy_family_id: str
        contact_model_family_id: str
        reward_model_family_id: str
        termination_model_family_id: str
        default_object_model_ref: ComponentRef
        default_reset_strategy_ref: ComponentRef
        default_contact_model_ref: ComponentRef
        default_reward_model_ref: ComponentRef
        default_termination_model_ref: ComponentRef
        observation_family_id: str
        action_family_id: str
        evaluator_id: str
        required_kinematic_capabilities: frozenset[str]
        required_sim_capabilities: frozenset[str]

TaskSpec must not name Linker or Allegro.

Compatibility is evaluated in layers:

1. TaskSpec against HandSpec kinematic capabilities.
2. TaskHandProfile against HandSimSpec simulator capabilities when constructing simulation.
3. DeploymentBinding against HandDescriptor hardware state/command/telemetry capabilities when constructing a live system.

A task's optional hardware telemetry requirement must not make its simulated experiment disappear; conversely, simulation contact sensors do not prove that equivalent live telemetry exists.

### 8.7 TaskHandProfile

The relationship between a task and a hand is real and should be explicit rather than hidden in inheritance.

    @dataclass(frozen=True)
    class TaskHandProfile:
        profile_id: str
        task_id: str
        hand_id: str
        active_fingers: tuple[str, ...]
        initialization_source_ref: ComponentRef
        task_from_wrist: Pose
        observation_codec_ref: ComponentRef
        conditioning_spec_ref: ComponentRef
        privileged_observation_spec_ref: ComponentRef
        teacher_conditioning_provider_ref: ComponentRef
        action_transform_ref: ComponentRef
        object_model_override_ref: ComponentRef | None
        contact_model_override_ref: ComponentRef | None
        reward_model_override_ref: ComponentRef | None
        physics_profile_ref: ComponentRef
        reset_strategy_override_ref: ComponentRef | None
        termination_model_override_ref: ComponentRef | None
        start_sampling_profile_ref: ComponentRef

Examples:

- Linker side grasp for mounted screwdriver.
- Linker top grasp for mounted screwdriver.
- Allegro 3-finger mounted screwdriver.
- Allegro 4-finger mounted screwdriver.
- Linker palm-up in-hand cage.

Task-dependent pregrasps and the authoritative task_from_wrist transform belong here, not in HandSpec, GraspPreset, or duplicated deployment code. ComponentRef is a typed, schema-versioned discriminated reference; unrestricted Mapping[str, Any] overrides are not allowed in frozen resolved contracts.

initialization_source_ref is a discriminated source: one GraspPreset, a versioned cache/distribution, or a typed generator. The selected ResetStrategy consumes that source. A profile does not pretend that a single nominal grasp represents an in-hand reset distribution.

Concrete strategy resolution uses one precedence tree:

1. TaskSpec declares compatible component families and one default implementation for each task slot.
2. TaskHandProfile may replace only explicitly allowed slots with a component from the same compatible family.
3. ExperimentSpec may apply schema-validated typed field overrides to those selected component records.
4. SimulationProfile supplies backend/solver/stepping behavior, not another competing task-physics definition.
5. The resolver derives dimensions and freezes the exact component IDs/digests.

Unknown slots, family changes, and conflicting ownership fail. Every resolved implementation and typed override enters the experiment fingerprint; deployment-relevant preprocessing/action choices also enter the policy-contract fingerprint.

### 8.8 GraspPreset

    @dataclass(frozen=True)
    class GraspPreset:
        preset_id: str
        hand_id: str
        measured_joint_positions: SemanticJointVector
        commanded_joint_targets: SemanticJointVector
        object_profile_id: str | None
        provenance: Provenance

Measured reset positions and commanded PD targets are distinct because current in-hand resets and temporal histories require both. Mimic values should be derived from coupled-joint specs rather than copied into independent and follower dictionaries.

### 8.9 Training, curriculum, randomization, and simulation profiles

Training axes must be independently selectable rather than hidden in one copied PPO document:

    @dataclass(frozen=True)
    class AlgorithmProfile:
        profile_id: str
        algorithm: str
        network: NetworkSpec
        rollout: RolloutSpec
        optimizer: OptimizerSpec
        checkpointing: CheckpointSpec

    @dataclass(frozen=True)
    class CurriculumProfile:
        profile_id: str
        phases: tuple[CurriculumPhase, ...]

    @dataclass(frozen=True)
    class DomainRandomizationProfile:
        profile_id: str
        distributions: tuple[RandomizationDistribution, ...]

    @dataclass(frozen=True)
    class SimulationProfile:
        profile_id: str
        backend: SimulatorBackendSpec
        solver: SolverSpec
        stepping: SimulationSteppingSpec

    @dataclass(frozen=True)
    class ArmIntegrationProfile:
        profile_id: str
        mode: Literal["fixed", "perturbed_fixed", "compliant_proxy", "full_arm", "recorded_replay"]
        mount_strategy_ref: ComponentRef
        perturbation_or_compliance_ref: ComponentRef | None
        arm_model_ref: ComponentRef | None

A new hand should not require copying a whole PPO YAML when only action and observation dimensions change.

### 8.10 ExperimentSpec and ResolvedExperiment

User-facing configuration should reference stable IDs and overrides:

    experiment:
      task: screwdriver_rotation_v1
      hand: linker_l20_left_v1
      task_hand_profile: linker_l20_screwdriver_side_grasp_v3
      algorithm_profile: rlgames_ppo_screwdriver_v2
      curriculum_profile: screwdriver_curriculum_v2
      domain_randomization_profile: screwdriver_geometry_dr_v1
      simulation_profile: isaac_gpu_contact_v2
      arm_integration_profile: fixed_wrist_exact_v1
      overrides:
        domain_randomization.geometry.enabled: true
    run:
      seed: 42
      num_envs: 2048
      output_root: runs/
      evaluation_scale: standard

Resolution must be pure and idempotent:

    resolved = resolve_experiment(spec, registries)
    validate_resolved_experiment(resolved)

RunSpec is resolved separately and owns seed, environment count, logging/output destinations, evaluator scale, host allocation, and bounded-run options. It may reference an experiment fingerprint but never changes policy tensor dimensions or deployment compatibility.

ResolvedExperiment contains:

- fully derived observation and action dimensions;
- exact asset references and hashes;
- exact semantic joint order;
- final curriculum and domain-randomization values;
- final task-hand transform and initialization/reset source;
- final ArmIntegrationProfile and MountStrategy;
- final network dimensions;
- final control period;
- no unresolved references;
- no mutation required after resolution.

It should be serializable as canonical JSON and hashable.

Do not overload one hash with three different purposes:

- policy_contract_fingerprint covers deployment-relevant semantics: task/hand/profile IDs, ordered joints, codec, deployable ExecutionGraph/conditioning, action transform, network interfaces, control timing, tensor metadata, and relevant asset hashes;
- experiment_fingerprint covers the fully resolved training experiment, including privileged/teacher providers, algorithm, curriculum, domain randomization, ArmIntegrationProfile/MountStrategy, concrete task strategies, and simulation semantics;
- run_id and run manifest cover execution choices such as seed, num_envs, evaluator scale, logging, output destinations, host, and timestamps.

Changing log location or evaluation environment count must not invalidate policy compatibility. Changing a codec phase, joint order, action scale, or control period must.

### 8.11 Resolution lifecycle

Required lifecycle:

1. Parse external configuration.
2. Resolve registry references.
3. Apply explicit overrides.
4. Derive all dependent values.
5. Validate cross-component invariants.
6. Freeze.
7. Construct simulator or training objects.

Do not derive dimensions and assets in class construction before external overrides are applied.

### 8.12 Registry behavior

Registries should contain declarative records, not require handwritten import side effects for each task-hand variant.

Required registry operations:

- register_hand;
- register_task;
- register_task_hand_profile;
- register_algorithm_profile;
- register_curriculum_profile;
- register_domain_randomization_profile;
- register_simulation_profile;
- register_arm_integration_profile;
- register_typed_component;
- resolve_experiment;
- list_compatible_hands(task_id);
- list_compatible_tasks(hand_id);
- validate_all_builtin_experiments.

Gym IDs may be generated from ExperimentSpec records while preserving existing IDs as aliases.

### 8.13 Coordinate-frame and unit contract

All cross-layer spatial data must declare:

- source frame;
- destination/reference frame;
- transform direction;
- position unit;
- angle unit;
- quaternion ordering;
- timestamp and clock domain.

Recommended conventions:

- SI units;
- radians;
- quaternion wxyz inside core contracts;
- pose names use destination_from_source, for example task_from_wrist;
- explicit conversion at ROS/vendor edges;
- no implicit subtraction of world positions as a substitute for a frame transform.

Define a FrameSpec registry for:

- world;
- robot_base;
- arm_wrist;
- hand_palm;
- task_fixture;
- object.

Task and deployment validation must reject unknown or incompatible frame IDs.

### 8.14 Clock-domain contract

Timed state must include local_monotonic_receive_time_ns. It may additionally include:

- source_time_ns when the source provides a real timestamp;
- source_clock_id paired with source_time_ns;
- timestamp_uncertainty_ns where synchronization is estimated.

Drivers must document whether source time is:

- device hardware time;
- ROS time;
- SDK callback time;
- local receive time only.

Safety freshness thresholds must account for clock-conversion uncertainty. Wall-clock/NTP adjustments must never drive the control scheduler.

For the first live implementation, scheduler, arbiter, lease authority, hand/arm gateways, and watchdog must execute in one host monotonic clock domain. generated_time_ns, decision deadlines, valid_until_ns, heartbeats, and token expiry use that domain, and the driver/watchdog evaluates expiry on its own local monotonic clock. A future multi-host design must translate deadlines at the exclusive command gateway using local receive time, authenticated source metadata, measured transit bounds, and clock uncertainty; raw monotonic timestamps from another host are never compared directly.

---

## 9. Observation and action architecture

### 9.1 Why this is the highest-value abstraction

Observation and action contracts connect:

- environment implementation;
- network architecture;
- running normalization;
- Stage-2 history;
- actor frame stacking;
- deployment runtime;
- control frequency;
- handoff state;
- policy artifact compatibility.

The current in-hand failure demonstrates that documenting dimensions is insufficient. Construction must be shared.

### 9.2 Timed policy sample

    @dataclass(frozen=True)
    class TimedHandState:
        schema_id: str
        sequence: int
        local_monotonic_receive_time_ns: int
        source_time_ns: int | None
        source_clock_id: str | None
        timestamp_uncertainty_ns: int | None
        position: Array
        velocity: Array | None
        effort: Array | None
        current: Array | None
        temperature: Array | None
        valid_mask: Array
        faults: frozenset[str]

    @dataclass(frozen=True)
    class EffectiveHandTarget:
        schema_id: str
        control_session_id: str
        source_controller_id: str
        command_id: int
        lease_epoch: int
        local_monotonic_accepted_time_ns: int
        valid_until_ns: int
        acknowledgement_level: AckLevel
        provenance: TargetProvenance
        inferred: bool
        position: Array

    @dataclass(frozen=True)
    class PolicySample:
        hand: TimedHandState
        effective_target: EffectiveHandTarget

Only declared policy inputs belong in PolicySample. Arm state, object sensing, wrench, and safety context must not accidentally leak into a proprioceptive policy.

    @dataclass(frozen=True)
    class TimedArmState:
        sequence: int
        local_monotonic_receive_time_ns: int
        source_time_ns: int | None
        source_clock_id: str | None
        timestamp_uncertainty_ns: int | None
        joint_position: Array
        joint_velocity: Array
        wrist_pose: Pose
        wrist_twist: Twist
        wrist_wrench: Wrench | None
        faults: frozenset[str]

    @dataclass(frozen=True)
    class TimedSafetySignal(Generic[T]):
        source_id: str
        sequence: int
        local_monotonic_receive_time_ns: int
        value: T | None
        channel_health: ChannelHealth
        faults: frozenset[str]

    @dataclass(frozen=True)
    class SafetyContext:
        arm: TimedArmState | None
        task_from_wrist: TimedSafetySignal[Pose]
        object_present: TimedSafetySignal[bool]
        object_dropped: TimedSafetySignal[bool]
        operator_deadman: TimedSafetySignal[bool]
        estop_active: TimedSafetySignal[bool]

    @dataclass(frozen=True)
    class TaskReadinessContext:
        task_state: tuple[TimedSafetySignal[Any], ...]
        task_state_schema_id: str

SafetyContext and TaskReadinessContext are consumed by HandoffSupervisor/SafetySupervisor, not automatically by the actor. Each DeploymentBinding defines maximum age, required channel health, and disconnected/stale behavior for every signal. A lost deadman/E-stop/object publisher never leaves a cached boolean indefinitely valid; fail-safe polarity and response are explicit.

Applied is too strong for most transports. The control trace must distinguish:

1. actor raw action;
2. transformed semantic candidate;
3. transition-blended candidate;
4. safety-filtered semantic command;
5. mapped/native command and its effective semantic round trip;
6. transport-enqueued or bus-sent command;
7. device-accepted command when supported;
8. low-level servo-applied acknowledgement when supported;
9. inferred-applied target when direct acknowledgement is unavailable;
10. later measured state.

AckLevel is an ordered capability enum such as ENQUEUED, SENT_TO_BUS, DEVICE_ACCEPTED, and SERVO_APPLIED. Policy history uses the post-mapping effective semantic setpoint accepted at the highest trustworthy level required by the DeploymentBinding. It never treats a merely proposed candidate as applied. Every command and target carries both a globally unique control_session_id and a lease epoch; epoch alone is insufficient after a process restart.

Initial acknowledgement scope: the current CAN and ROS1 transports can evidence at most ENQUEUED/SENT_TO_BUS. The first implementation therefore distinguishes only candidate, transport-sent, and inferred-applied (next-state reconciliation). DEVICE_ACCEPTED and SERVO_APPLIED remain reserved enum values with no plumbing behind them until a transport actually provides that evidence, and the first DeploymentBindings set their required acknowledgement level accordingly. Raising the level later is a binding change, not a runtime redesign.

If the binding-required acknowledgement is unavailable by its deadline, the runtime does not fabricate an effective target. The tick follows the configured missed-acknowledgement response, invalidates history as required, and records the lower-level evidence that was available.

PolicySnapshotBuilder performs the deterministic temporal join between asynchronous state and acknowledgement streams. It maintains an effective-target timeline keyed by local monotonic acceptance interval and pairs each hand state with the target in force for that state's declared capture phase—not simply the newest acknowledgement. Acknowledgements enter a bounded scheduler queue; they never call PolicySession concurrently. Readiness rejects excessive hand/arm/task-signal skew according to DeploymentBinding.

### 9.3 ProprioCodec and actor-input assembly

    class ProprioCodec(Protocol):
        codec_id: str
        hand_schema_id: str
        control_dt_ns: int

        def reset(self, sample: PolicySample) -> None: ...
        def append_once(self, sample: PolicySample) -> AppendResult: ...
        def actor_proprio(self) -> Tensor: ...
        def adapter_history(self) -> Tensor: ...
        def describe(self) -> ProprioCodecManifest: ...

    class ActorInputAssembler(Protocol):
        assembler_id: str

        def assemble(
            self,
            proprio: Tensor,
            conditioning: ConditioningInput,
        ) -> Tensor: ...

    @dataclass(frozen=True)
    class PrivilegedObservationSpec:
        spec_id: str
        fields: tuple[TensorFieldSpec, ...]
        output_dim: int

    class TeacherConditioningProvider(Protocol):
        provider_id: str

        def produce(
            self,
            privileged: Tensor,
            task_state: TrainingTaskState,
        ) -> ConditioningInput: ...

The proprio codec owns only deployable proprioception and temporal history:

- semantic ordering;
- joint scaling or normalization;
- target representation;
- frame contents;
- history length;
- actor frame count and offsets;
- reset fill behavior;
- capture phase relative to action integration and physics;
- dtype;
- exact shape derivation.

Observation noise is not part of the deployable codec. Training-time joint noise (currently injected inside the in-hand frame construction when domain randomization is enabled) is an input perturbation applied upstream of the codec by the training wrapper and owned by DomainRandomizationProfile. The codec math is identical in training and deployment; only its input differs — noised simulated state versus measured hardware state. Noise configuration enters the experiment fingerprint but never the policy-contract fingerprint, golden parity traces are recorded with noise disabled, and the live runtime has no noise path at all.

The actor-input assembler owns conditioning such as a teacher latent, adapter-predicted latent, or a named legacy predicted tail. This prevents a deployment codec from being confused with privileged training inputs or network-internal latent construction.

All tensor protocols declare whether the leading dimension is batch and preserve it. Scalar PolicySession passes batch size one internally and removes that axis only at its public boundary.

PrivilegedObservationSpec and TeacherConditioningProvider own critic/teacher-only inputs and the Stage-2 supervision target. Their selected IDs, field order, dimensions, units, and implementation digests enter the experiment fingerprint and checkpoint provenance. Golden traces cover privileged tensors and teacher conditioning separately from deployable proprioception; deployment packages may omit teacher-only executable components while retaining provenance.

Sample ingestion is exactly once. A codec keys each appended pair by hand state sequence plus effective-target command ID and control session. prime, preview, activate, and step may all inspect the same sample, but cannot advance history twice; duplicate keys return an explicit DUPLICATE result or fail under strict mode. Reset/fill is a separate declared operation.

Simulation uses a batched functional kernel rather than sharing the scalar live object's mutable state:

    PolicySampleBatch:
      hand_state_by_env
      effective_target_by_env
      state_sequence_by_env
      command_id_by_env
      control_session_or_episode_id_by_env
      reset_mask
      valid_mask
      device
      dtype

    BatchedProprioKernel:
      reset_where(batch, reset_mask)
      append_once(batch)
      actor_proprio_batch()
      adapter_history_batch()

    BatchedActionTransformKernel:
      candidate_batch(raw_action, effective_target, valid_mask)

Every environment has independent sequence/command keys, fill state, action-integration state, and reset semantics. Device/dtype conversions are explicit and shape-preserving. PolicySession is the scalar batch-size-one wrapper over the same proprio/action kernels; vectorized Isaac environments and Stage-2 collection use PolicySampleBatch. Golden traces include asynchronous per-environment resets.

### 9.4 Required built-in codecs

At minimum:

1. Allegro screwdriver legacy codec
   - q plus effective targets as deployable proprioception.
   - any Euler/predicted tail is a separate named ConditioningSpec and ActorInputAssembler.

2. Linker screwdriver latent codec
   - raw q plus effective targets for actor proprioception.
   - exact Stage-2 history phase must be defined and corrected if deployment currently captures old versus new targets differently from training.

3. Linker in-hand stacked codec
   - joint-limit-scaled q;
   - raw effective targets;
   - three actor frames;
   - thirty adapter frames;
   - 20 Hz exact period.

4. Future Allegro in-hand codec
   - derived from the task observation contract and Allegro JointSchema.

### 9.5 History phase

For every codec, explicitly answer:

- Is q sampled before or after the latest command is applied?
- Does the frame contain the previous effective target, newly generated target, or low-level acknowledged target?
- Is history advanced before inference or after inference?
- How are missed ticks represented?
- What happens during shadow priming?
- What happens during blending?
- What happens when safety clips a target?

The preferred runtime rule is:

1. Receive a fresh timestamped measured state.
2. Obtain the post-mapping effective target accepted at the required acknowledgement level for the preceding interval.
3. Append that pair to history.
4. Compute the new candidate.
5. Apply action transform and safety.
6. Send.
7. On acknowledgement or next-state reconciliation, update the effective-target record without appending the same sample twice.

Simulation must mirror the same logical phase.

### 9.6 No silent padding or truncation

The following are errors:

- frame_dim differs from the codec-derived frame dimension;
- actor proprio dimension differs from codec output;
- adapter history length differs from network metadata;
- hand schema differs from codec schema;
- a declared normalizer width differs from its ExecutionGraph input, or normalization mode none is contradicted by a hidden normalizer;
- target vector lacks a semantic joint;
- an unknown joint is present.

### 9.7 ActionTransform protocol

    class ActionTransform(Protocol):
        transform_id: str
        schema_id: str
        control_dt_ns: int

        def candidate(
            self,
            raw_action: Tensor,
            effective_target: EffectiveHandTarget,
        ) -> SemanticJointCandidate: ...

It owns:

- action clipping;
- delta or absolute semantics;
- per-tick versus per-second scaling;
- semantic target limits;
- optional learned or configured synergies;
- expected output units.

SemanticJointCandidate is lease-free and contains schema, source state sequence, generation time, position payload, and candidate diagnostics only. PolicySession cannot mint a lease, control session, final command ID, token, or transport deadline. After CommandArbiter selects the authoritative producer and transition blending finishes, the exclusive command gateway attaches the authorization envelope and final identifiers.

### 9.8 Candidate versus applied command

PolicyResult must distinguish:

- raw action;
- transformed candidate target;
- transition-blended target;
- safety-filtered target;
- native-encoded target and its semantic round trip;
- transport-accepted target;
- device/low-level acknowledged target where available;
- target inferred as applied.

The policy state must synchronize to the highest-confidence effective target required by the deployment contract. It must not assume that a candidate rejected by safety, mapping, or transport was applied.

### 9.9 Trace parity

A golden trace row should contain:

- state sequence and timestamps;
- measured semantic q;
- effective semantic target and acknowledgement provenance;
- encoded frame;
- full adapter history hash or selected snapshots;
- actor proprio;
- predicted latent;
- raw action;
- candidate target;
- filtered target.

For identical traces, simulator-side and runtime-side outputs should match within declared numerical tolerance.

Golden traces are record-then-replay artifacts. Canonical fixtures are recorded once on the CUDA/Isaac rig (ENV-B, Section 2.4) with observation noise and domain randomization disabled and seeds fixed, committed as small versioned files, and replayed deterministically in CPU tests (ENV-A). Regenerating a trace from live simulation is a convenience check, not the contract: GPU physics is not bitwise reproducible across devices and versions, so parity is always judged against the recorded fixture within its declared tolerance.

---

## 10. Simulator architecture

### 10.1 Shared DexterousHandRuntime

Extract hand mechanics currently duplicated across screwdriver and in-hand environments:

- articulation construction;
- joint lookup and exact-name validation;
- independent joint ordering;
- coupled follower resolution and target application;
- target limit construction;
- semantic target writes;
- fingertip and non-tip body lookup;
- self-collision exclusions;
- hand state access;
- proprio codec integration.

HandRuntime does not own articulation-root resets. A typed MountStrategy owns fixed-base, arm-mounted, and future floating-root placement, including task_from_wrist application and reset authority.

Use the neutral attribute name hand, never allegro for a generic articulation.

### 10.2 Task lifecycle composition

A generic direct environment should delegate:

- scene object construction to ObjectModel;
- task reset to ResetStrategy;
- contact data to ContactProvider;
- proprio/history construction to ProprioCodec and actor conditioning to ActorInputAssembler;
- critic/teacher observations to PrivilegedObservationSpec and TeacherConditioningProvider;
- reward to RewardModel;
- termination to TerminationModel;
- metrics to MetricProvider;
- evaluation semantics to TaskEvaluator.

The task environment controls the common DirectRLEnv lifecycle but does not know Linker-specific joint names.

### 10.3 Contact providers

Required providers include:

- distance-and-pad-facing proxy;
- per-fingertip filtered contact force;
- wrong-surface contact force;
- free-object fingertip/non-tip contact.

Contact providers declare required hand and simulator capabilities.

### 10.4 Reward models

Mounted screwdriver reward variants should be selectable independently of hand:

- distance/pad-gated reward;
- force-window role-based reward.

This permits a new hand to use either strategy if it provides the required sensing.

### 10.5 Physics profiles

Separate:

- task object geometry;
- contact materials;
- screwdriver bearing/load parameters;
- hand actuator parameters;
- task-hand-specific physics overrides.

Do not copy a complete ArticulationCfg merely to change contact-sensor activation or damping.

### 10.6 Wrist and task frames

Task geometry and reset poses must be expressed relative to named frames:

- world;
- robot base;
- wrist;
- palm;
- task fixture;
- object.

The fixed-base simulator remains a supported adapter, but its absolute pose should be derived from the authoritative task_from_wrist transform rather than embedded as an unrelated policy constant.

### 10.7 Arm integration simulation modes

Support progressively:

1. Fixed wrist, exact task transform.
2. Fixed wrist with randomized transform error.
3. Compliant wrist proxy with pose/twist disturbances.
4. Full robot-arm articulation with a hold/impedance controller.
5. Recorded teleoperation-state replay into reset and shadow history.

The first mixed-control deployment may use mode 2 or 3 while a full arm model is developed.

ExperimentSpec must select one ArmIntegrationProfile; its MountStrategy and perturbation/compliance/full-arm references are resolved, validated, and fingerprinted. No simulator hand asset supplies a hidden fixed-base default.

### 10.8 Reset contracts

Reset output should be structured:

    ResetResult:
      measured_joint_position
      commanded_joint_target
      task_from_wrist
      object_state
      observation_history_seed
      start_profile_id

Grasp-cache rows should not rely on a hardcoded 39-column anonymous array. They need a manifest describing hand schema, row layout, object prototype, task profile, generator revision, and hashes.

---

## 11. Policy package architecture

### 11.1 Package layout

Recommended immutable directory or archive:

    policy-package/
      manifest.json
      execution-graph.json
      components/
        <component-id>.<declared-tensor-format>
      resolved-policy-contract.json
      training-start-distribution.<declared-data-format>
      conformance/
        inputs.<declared-tensor-format>
        reference-outputs.<declared-tensor-format>
        comparison-spec.json
      SHA256SUMS

The package contract is tensor-format-neutral until the Artifact Tensor Format decision in Section 36.1 is accepted. PyTorch state dictionaries may remain during transition, but untrusted pickle-based artifacts must never be treated as generic safe input.

training-start-distribution is optional only when the manifest explicitly declares that no empirical policy envelope exists; in that case a live DeploymentBinding must not claim empirical-envelope readiness.

ExecutionGraphV1 is a typed acyclic graph of component roles, tensor interfaces, and preprocessing/postprocessing nodes. It declares which nodes are required: actor-only, actor plus adapter, optional critic for offline evaluation, normalization mode none/fixed/running-frozen, and any legacy assembler. Actor, adapter, and normalizer are not globally mandatory filenames. PolicySession loads and executes only declared deployable graph nodes, allowing non-RMA and intentionally unnormalized policies without changing core runtime code.

Validation and promotion records are detached immutable attestations keyed by policy_package_id:

    attestations/
      <policy-package-id>/
        simulation-parity-<attestation-id>.json
        task-evaluation-<attestation-id>.json
        hardware-compatibility-<attestation-id>.json
        promotion-<attestation-id>.json

They are never inserted into or used to mutate the tensor package after hashing.

### 11.2 PolicyManifestV1 required fields

Identity:

- schema_version;
- policy_package_id;
- task family and version;
- registered experiment ID;
- hand model and side;
- joint schema ID and ordered names;
- observation codec ID and version;
- action transform ID and version.

Timing:

- control_dt_ns;
- actor frame count;
- actor frame offsets;
- adapter history length;
- history fill behavior;
- history capture phase.

Model:

- execution-graph schema/version/digest;
- component role, architecture, tensor path/digest, inputs, outputs, dtype, and conditional requirement;
- conditioning/latent dimensions where declared by the graph;
- explicit normalization mode and dimensions/epsilon/state when normalization is enabled;
- required runtime version range.

Action:

- action mode;
- clipping;
- delta scale and whether per-tick or per-second;
- semantic lower and upper target limits;
- maximum policy-command velocity and acceleration.

Training provenance:

- Git revision;
- dirty-state flag plus canonical patch and untracked-source manifest/evidence digests if development policy allows dirty source;
- source Stage-1 checkpoint hash;
- policy-contract fingerprint;
- experiment fingerprint;
- asset and URDF hashes;
- seed;
- dependency and simulator versions;
- training start and completion time;
- Stage-2 iteration and loss;
- training-host metadata needed for reproducibility.

Approved production candidates should be built from clean, reviewed source by default. A development candidate may be dirty only when immutable run evidence retains the canonical tracked patch plus a path/content-digest manifest for untracked source/config files and either the content blobs or a trusted content-addressed reference. A diff hash alone is not reconstructable provenance and cannot satisfy production promotion policy.

Validation:

- package-declared compatibility constraints, not a list of deployment-binding IDs;
- required hand/schema/codec/action/timing/runtime ranges;
- empirical training-start distribution and its provenance;
- explicit split between proprioceptive PolicySample features and supervisor-owned TaskReadinessContext features;
- canonical conformance input/reference-output file hashes and ComparisonSpec digest;
- known model limitations known at creation time.

Task evaluation, site-specific thresholds, hardware binding, and promotion decisions live in detached attestations and DeploymentBinding. This avoids a circular dependency in which a policy lists profiles that themselves point back to the policy.

Define policy_package_id from canonical package content excluding the ID field itself, signatures, and detached attestations. One acceptable scheme is: hash each payload, canonicalize a manifest whose package-id field is omitted, then hash the sorted path-to-hash map plus canonical manifest. SHA256SUMS follows the same exclusion rule. Signatures cover the resulting policy_package_id.

### 11.3 Strict loading sequence

1. Read manifest without loading tensors.
2. Validate schema version.
3. Validate every required field.
4. Validate content hashes.
5. Validate runtime compatibility.
6. Validate requested task, hand, side, and semantic joints.
7. Validate control period and codec availability.
8. Validate limits against deployment hardware.
9. Load data-only tensors.
10. Require exact state-dictionary keys.
11. Verify conformance fixture file hashes, run the packaged inputs, and compare actual tensors to packaged reference outputs under ComparisonSpec.
12. Warm model and measure inference budget.
13. Only then allow a hardware transport to open.

ComparisonSpec names every compared output, reference dtype, allowed runtime dtype conversions, absolute/relative tolerance, NaN/Inf policy, shape rule, and deterministic versus tolerance-based mode. Hashes prove fixture integrity; they are not used as a substitute for numerical comparison across CPU/GPU/runtime implementations.

### 11.4 Artifact creation and promotion

Training produces an unpromoted candidate in a unique run directory.

Promotion stages:

- candidate;
- simulation-parity-passed;
- task-evaluation-passed;
- hardware-compatible;
- HIL-shadow-passed;
- bounded-hardware-passed;
- approved.

Promotion writes a new detached immutable attestation. It does not overwrite the original package or any earlier attestation.

Every AttestationEnvelope contains:

- schema version, attestation ID/type, created time, issuer, and key ID;
- subject policy_package_id and policy-contract fingerprint;
- DeploymentBinding ID when hardware/site-specific;
- exact evaluator/runtime/gate-profile IDs and digests;
- immutable raw evidence/report/trace digests and storage references;
- decision, limitations, expiration/revalidation rule, and approved scope;
- previous-stage attestation ID/digest when advancing a promotion chain;
- detached signature over canonical envelope content.

A later stage validates the full chain and cannot replace evidence from an earlier stage by reusing its label.

Checksums prove internal integrity only. Production authenticity should come from:

- an access-controlled artifact registry;
- or a detached signature from an approved promotion key;
- a configured trust policy in the live runtime.

The runtime should record which trust policy authorized the package.

Committed now versus deferred (Section 2.5): content-addressed packages, immutability, and an access-controlled artifact location are committed from Phase 3. Detached signatures, key rotation, and revocation (ART-010, TEST-004) sit on the deferred trust track (Section 31.11) and activate only when artifacts cross a machine or organization trust boundary. Until then, "trusted" means an exact content digest pinned by the DeploymentBinding and stored in an access-controlled location.

### 11.5 Legacy artifacts

Rules:

- Existing deploy.pth files are legacy-v0.
- Live runtime rejects them by default.
- An offline migrate-policy command may inspect a legacy artifact, require explicit task and hand metadata, derive what is possible, and report anything unverifiable.
- Migrated packages remain unapproved until parity and task gates pass.
- Linker in-hand legacy artifacts require the corrected stacked observation codec and must be tested through the actual runtime.

---

## 12. Deployment binding and launch configuration

Policy content, immutable hardware/site compatibility, and per-launch endpoints/secrets are three separate layers.

DeploymentBinding is content-addressed and contains:

- policy package ID and required detached-attestation policy;
- hardware model, side, permitted serials, and calibration ID;
- arm gateway/controller-manager binding, arm identity constraints, arm safety/watchdog contract, and permitted modes when mixed control is enabled;
- resolved TaskHandProfile ID/hash plus allowed measured task_from_wrist error and wrist pose/twist/wrench envelope; the binding does not duplicate the profile's nominal transform;
- object/tool profile;
- site-specific readiness/start-envelope thresholds;
- shadow-priming strategy and duration;
- transition duration and motion caps;
- training-to-hardware joint-limit mapping strategy;
- required state/command rates and policy period;
- maximum state age, clock uncertainty, and history-gap policy;
- maximum hand/arm/task-signal timestamp skew and temporal-join tolerance;
- minimum acknowledgement level;
- per-stage timing budgets and command-validity duration;
- lease heartbeat, expiration, transfer, and takeover timeouts;
- command/servo mode and any atomic mode-switch procedure;
- required arm/hand watchdog identities, independence levels, arming/heartbeat/expiry limits, and physical behavior after process/host loss;
- missed-state, missed-inference, and missed-command policy;
- hand-back preparation timeout and fallback;
- fault-class-to-response-to-escalation-timeout mapping;
- collision/workspace guard profiles;
- required/optional status, frame IDs, model digests, and freshness limits for every collision/workspace guard;
- timestamp/health/staleness policy for deadman, E-stop, object, task, wrist, and watchdog signals;
- commissioning telemetry-overflow policy;
- bounded trial duration or tick count.

LaunchConfig is deliberately not content-addressed with the policy or binding. It contains local transport endpoints, network interfaces, telemetry destinations, process topology, operator identity, and references to credentials/secrets. Secret values never appear in policy packages, bindings, traces, or generated documentation.

LaunchConfig also declares the exclusive command-gateway topology. The hand gateway must be the sole holder of the CAN/vendor socket or controller command interface; the arm gateway must be the sole controller-manager/vendor command entry. Direct ROS/vendor publishers are disabled, isolated, or ACL-rejected. Cross-process candidates enter through an authenticated local channel, or use non-copyable/verifiable capability material rather than trusting a copied token digest.

This permits the same policy package to be evaluated against multiple calibrated hands and launched through different infrastructure without mutating policy content.

Every live-consumed artifact—DeploymentBinding, HandCalibration, GateProfile, plugin allowlist/package, promotion attestation, and policy package—has a schema version, stable ID, canonical digest, provenance, issuer/key ID, and optional detached signature. The runtime applies a versioned TrustPolicy before opening hardware. TrustPolicy defines allowed issuers/keys, artifact scopes, minimum stages, key-rotation overlap, revocation lists and freshness, offline behavior, and emergency rollback. Tests cover unknown keys, expired/revoked keys, stale revocation data, wrong artifact scope, broken attestation chains, and rotation from old to new keys.

Deferral (Section 2.5): until artifacts cross a machine or organization trust boundary, enforcement stops at exact content digests pinned by the DeploymentBinding plus access-controlled storage. The signature/rotation/revocation machinery above is design-complete reference material on the deferred trust track (Section 31.11); the schema fields exist from the start so that activating enforcement later changes policy data, not artifact formats.

---

## 13. Environment-free runtime

### 13.1 PolicySession

PolicySession is the embeddable policy boundary. It must not own hardware transports, process signals, ROS nodes, or terminal logging.

    class PolicySession:
        def prime(self, sample: PolicySample) -> PolicyReadinessReport: ...
        def preview(self, sample: PolicySample) -> PolicyResult: ...
        def activate(
            self, expected_sample_key: PolicySampleKey
        ) -> PolicyReadinessReport: ...
        def step(self, sample: PolicySample, deadline_ns: int) -> PolicyResult: ...
        def synchronize_effective_target(
            self, target: EffectiveHandTarget
        ) -> None: ...
        def deactivate(self, reason: StopReason) -> PolicySessionSnapshot: ...
        def snapshot(self) -> PolicySessionSnapshot: ...
        def close(self) -> None: ...

Responsibilities:

- own the package-declared deployable ExecutionGraph components, ProprioCodec, ActorInputAssembler, and ActionTransform state;
- validate input schema and timing;
- prime temporal history;
- compute candidate commands;
- measure inference duration;
- expose readiness and diagnostics;
- synchronize to effective targets;
- serialize diagnostic state for replay.

Non-responsibilities:

- no hardware slot conversion;
- no CAN or ROS;
- no process signal handlers;
- no direct file logging;
- no controller ownership arbitration;
- no final safety decision;
- no assumption that every candidate was sent.

synchronize_effective_target only reconciles command/acknowledgement state for the next PolicySample; it never appends a history frame by itself.

deactivate changes session state and returns diagnostic state only. HandoffSupervisor selects a transition response, SafetySupervisor validates it, and HandDriver executes it. PolicySession never manufactures a stop command.

### 13.2 preview semantics

preview is used during RL shadow mode.

It must:

- consume fresh real PolicySample values;
- advance history exactly as active inference would;
- compute candidate commands and diagnostics;
- not claim hardware ownership;
- not imply the candidate was applied;
- continue using the effective target reported by teleoperation or the transition controller.

During HAND_BLEND, preview runs at every policy tick using the latest post-mapping effective blended target. Merely synchronizing a target without advancing the codec and actor at the declared cadence is invalid.

### 13.3 Policy and system readiness

    PolicyReadinessReport:
      ready: bool
      codec_history_complete: bool
      state_fresh: bool
      schema_compatible: bool
      within_policy_start_distribution: bool | None
      inference_budget_ok: bool
      reasons: list[ReadinessReason]
      stable_ticks: int
      required_stable_ticks: int

PolicySession can report only proprioceptive facts derivable from PolicySample and its package. Any empirical start-distribution feature based on object/contact/task state is evaluated from typed TaskReadinessContext by HandoffSupervisor, not smuggled into PolicyReadinessReport. HandoffSupervisor combines policy, task, wrist/task pose, arm hold, transport, operator, E-stop, object/tool, hardware, safety, watchdog, and site-specific readiness into SystemReadinessReport.

Readiness should be actionable. For example:

- joint index_pip tracking error is 0.21 rad, limit is 0.10;
- only 18 of 30 history frames are valid;
- state age is 73 ms, limit is 40 ms;
- effective target is unavailable from the upstream controller.

System-readiness reasons may separately report wrist pose error, arm-hold instability, missing object presence, or an unhealthy teleoperation hand-back path.

### 13.4 PolicyResult

    PolicyResult:
      policy_package_id
      source_state_sequence
      generated_time_ns
      inference_duration_ns
      raw_action
      candidate_command
      conditioning_outputs
      codec_diagnostics
      saturation_flags
      validity
      failure_reason

### 13.5 Runtime support matrix

Runtime must expose a machine-readable support matrix:

- policy package schemas supported;
- observation codecs supported;
- action transforms supported;
- hand schemas supported;
- transports available;
- live-validated combinations versus simulation-only combinations.

Unsupported combinations must be rejected rather than allowed through a best-effort path.

### 13.6 Concurrency and lifecycle

- One active scheduler owns each PolicySession.
- PolicySession is not implicitly thread-safe.
- Shadow and active calls are serialized on policy ticks.
- prime/reset is the only operation allowed to initialize fill frames; preview/step call append_once and cannot advance a repeated sample key.
- activate accepts the already-appended expected sample key, changes lifecycle/readiness state, and never ingests a sample after a same-tick preview.
- concurrent or reentrant calls fail with a typed lifecycle error rather than corrupting temporal history.
- Model tensors are warmed before activation.
- Registry/spec objects are frozen after startup.
- Calibration and adapter state are per instance.
- close is idempotent.
- stop requests are explicit and do not rely on object destruction.
- a session snapshot must record enough codec/action state to replay a transition, but a snapshot does not bypass live reprime requirements after a timing gap.

Handoff lifecycle placement is exact:

- RL_SHADOW and HAND_BLEND call preview only;
- completing alpha=1, receiving the final effective-target evidence, and atomically committing the HAND lease calls activate without appending that tick twice;
- RL_ACTIVE alone calls step;
- hand-back records the final effective RL/blend target, commits HAND transfer, then calls deactivate;
- entering SAFE_HOLD applies the codec-declared freeze-or-invalidate rule and always requires the configured reprime path before a later activation.

---

## 14. Hardware adapters and transports

### 14.1 Separation

HandDriver
      = HandAdapter
      + immutable HandCalibration
      + Transport

HandAdapter is a pure converter: semantic state/command to native representation and back, including a mapping preview and semantic round-trip diagnostics. Transport owns raw native I/O, reader lifecycle, timestamps, and acknowledgement evidence. HandDriver orchestrates adapter, calibration, transport, installed lease authority, and shutdown. Safety selects a SemanticSafeResponse; HandDriver executes that response but must not invent policy or task safety semantics.

### 14.2 HandDescriptor

HardwareIdentity should contain:

- model;
- side;
- serial;
- firmware;
- driver/SDK version;
- native state/command schema IDs.

    HandDescriptor:
      hardware_identity
      semantic_joint_schema
      native_state_schema
      native_command_schema
      state_capabilities
      command_capabilities
      hardware_motion_limits
      requires_keepalive
      state_rate_hz
      command_rate_hz

### 14.3 HandDriver protocol

    class HandDriver(Protocol):
        def identity(self) -> HardwareIdentity: ...
        def descriptor(self) -> HandDescriptor: ...
        def install_lease(self, token: LeaseToken) -> LeaseAck: ...
        def latest_before(self, cutoff_ns: int) -> TimedHandState: ...
        def prepare(
            self, command: AuthorizedSemanticHandCommand
        ) -> PreparedNativeCommand: ...
        def send_prepared(
            self,
            command: PreparedNativeCommand,
            deadline_ns: int,
        ) -> CommandAck: ...
        def execute_safe_response(
            self,
            authority: SafetyAuthorityToken,
            response: SemanticSafeResponse,
            deadline_ns: int,
        ) -> CommandAck: ...
        def shutdown(
            self,
            authority: SafetyAuthorityToken,
            response: SemanticSafeResponse,
            deadline_ns: int,
        ) -> CommandAck: ...
        def close(self) -> None: ...

A transport reader may block internally, but the scheduler consumes a bounded timestamped buffer through latest_before. close releases resources only after explicit shutdown; it must not ambiguously mean stop sending.

PreparedNativeCommand is immutable and binds the authorized semantic command, calibration ID, adapter/mapping digest, native payload, semantic round trip, quantization/clipping diagnostics, preparation time, and validity deadline. Safety validates that exact object; send_prepared verifies its digest/deadline and transmits its native bytes without remapping. This prevents time-of-check/time-of-use divergence.

### 14.4 AuthorizedSemanticHandCommand

    AuthorizedSemanticHandCommand:
      schema_id
      control_session_id
      source_controller_id
      command_mode
      lease_epoch
      lease_token_digest
      command_id
      generated_from_state_sequence
      generated_time_ns
      valid_until_ns
      payload

AuthorizedSemanticHandCommand is a discriminated union of typed position, velocity, torque, or named synergy variants supported by the HandDescriptor. It is created at the exclusive gateway after arbitration/blending from a lease-free candidate. RL ActionTransform currently emits only a SemanticJointCandidate position variant. A transition between modes requires the atomic servo-mode procedure in DeploymentBinding; payload interpretation is never selected by an unchecked string.

Drivers must reject:

- incorrect schema;
- expired command;
- stale lease epoch;
- non-finite values;
- unreachable command;
- command generated from state too old for the deployment binding.

The opaque lease token is issued by LeaseAuthority and installed atomically in the driver. An integer epoch or copyable token digest embedded in an otherwise self-authored command is not proof of authority; physical gateway exclusivity plus authenticated/capability-verified ingress provides enforcement.

### 14.5 CommandAck

    CommandAck:
      control_session_id
      command_id
      accepted
      acknowledgement_level
      local_monotonic_enqueue_time_ns
      local_monotonic_send_time_ns
      device_time_ns
      device_clock_id
      native_command
      effective_semantic_target
      quantization_error
      driver_status
      failure_stage
      failure_reason

Unexpected semantic or native clipping is rejected/faulted in live mode. Routine representational quantization is reported. A commissioning-only profile may explicitly permit bounded clipping for a named test, but clipped_joints is never silently accepted in production. CommandAck reports only evidence the transport/device can actually provide and never upgrades ENQUEUED to SERVO_APPLIED by inference.

Local monotonic enqueue/send times are required. Device time and clock ID are optional but must appear together. failure_stage distinguishes validation, mapping, enqueue, bus send, device rejection, timeout, and safe-response failure.

A preinstalled, narrowly scoped SafetyAuthorityToken or independent emergency channel authorizes safe responses after ordinary owner leases expire. Ordinary producers cannot invoke it. last_safe_target means the latest post-map effective target that passed every configured check and reached the DeploymentBinding's required acknowledgement level.

### 14.6 Required adapters

Initial:

- LinkerL20LeftAdapter.

Future:

- LinkerL20RightAdapter, only after separate tables and tests exist.
- AllegroV4RightAdapter.

Adapters are named by model and side. A generic unchecked side string must not choose a mapping.

### 14.7 Required transports

- Linker direct CAN/SDK.
- Current Linker ROS1 bridge.
- Fake echo transport.
- Recorded replay transport.
- Fault-injection transport.
- Future ROS2 lifecycle adapter.
- Future Allegro vendor transport.

The same PolicySession, SafetySupervisor, scheduler, and handoff logic must run above every transport.

### 14.8 External SDK management

Current deployment depends on local, uncommitted Linker SDK fixes. This is not reproducible.

Required work:

- upstream the fixes or maintain a pinned fork;
- record the exact SDK revision in compatibility and calibration metadata;
- run mapping conformance against that revision;
- refuse unknown incompatible SDK versions for live operation;
- remove sys.path bootstrapping as the long-term package integration mechanism.

### 14.9 ArmCommandGateway

Mixed control requires an arm-side boundary equivalent in authority to HandDriver, even when the actual controller implementation remains external:

    class ArmCommandGateway(Protocol):
        def identity(self) -> ArmHardwareIdentity: ...
        def descriptor(self) -> ArmDescriptor: ...
        def install_lease(self, token: LeaseToken) -> LeaseAck: ...
        def latest_before(self, cutoff_ns: int) -> TimedArmState: ...
        def prepare_mode_switch(self, mode: ArmCommandMode) -> PreparedModeSwitch: ...
        def commit_mode_switch(self, prepared: PreparedModeSwitch) -> ControllerAck: ...
        def send_prepared(self, command: PreparedArmCommand, deadline_ns: int) -> CommandAck: ...
        def execute_safe_response(
            self, authority: SafetyAuthorityToken,
            response: ArmSafeResponse, deadline_ns: int
        ) -> CommandAck: ...
        def watchdog_status(self) -> WatchdogStatus: ...
        def shutdown(
            self, authority: SafetyAuthorityToken,
            response: ArmSafeResponse, deadline_ns: int
        ) -> CommandAck: ...

The gateway may adapt an external robot controller manager rather than implement low-level arm control in this repository, but the integration contract must provide exclusive command access, lease/mode enforcement, timestamped state, acknowledgement, safe response, shutdown, and independent watchdog evidence. ArmHoldController runs above this gateway; it is not itself the ownership boundary.

---

## 15. Calibration architecture

### 15.1 Immutable per-instance calibration

Remove module-global active calibration.

    @dataclass(frozen=True)
    class HandCalibration:
        schema_version: int
        calibration_id: str
        model: str
        side: str
        serial: str
        firmware: str
        driver_version: str
        sdk_revision: str
        semantic_joint_schema_id: str
        native_state_schema_id: str
        native_command_schema_id: str
        state_mappings: tuple[StateJointCalibration, ...]
        command_mappings: tuple[CommandJointCalibration, ...]
        urdf_hash: str
        measured_state_error_rad: tuple[float, ...]
        measured_command_roundtrip_error_rad: tuple[float, ...]
        created_at: str
        tool_version: str
        notes: str

### 15.2 State and command calibration

    StateJointCalibration:
      native_state_channel
      semantic_joint_name
      sign
      native_min
      native_max
      semantic_min_rad
      semantic_max_rad
      offset
      scale
      measured_state_error_rad

    CommandJointCalibration:
      semantic_joint_name
      native_command_channel
      sign
      semantic_min_rad
      semantic_max_rad
      native_min
      native_max
      offset
      scale
      measured_command_error_rad

State-to-semantic and semantic-to-command channels are validated independently; they may differ in slot order, range, direction, and reserved channels. Conformance includes native-state fixtures, native-command fixtures, semantic round trips, and the exact SDK revision. One affine native_slot field is insufficient for the known Linker state-versus-command layout risks.

### 15.3 Binding rules

Live binding requires exact match for:

- model;
- side;
- serial, unless an explicitly authorized fleet calibration is used;
- semantic schema;
- SDK/driver compatibility;
- URDF or hardware-model version.

Warnings are insufficient for mismatches that affect joint meaning or limits.

Production calibration records should use the same trusted-registry or signature model as policy packages. A locally edited calibration file is a development artifact until explicitly approved.

### 15.4 Hardware-reachable limits

The current simulated Linker PIP range extends to approximately 1.57 rad while hardware reaches about 1.08 rad.

Required response:

- choose a named DeploymentBinding mapping strategy: physical_angle_parity or normalized_range_mapping;
- prefer physical-angle parity for production manipulation because it preserves the physical meaning of policy targets;
- when physical-angle parity is selected, make the training model reflect actual hardware range, re-solve pregrasps, and retrain policies whose action distribution used unreachable positions;
- treat normalized-range mapping as a distinct, behavior-changing policy contract that requires simulation parity, task evaluation, and promotion evidence;
- report expected saturation at artifact preflight;
- never silently choose between rescaling and clamping without a named mapping profile.

### 15.5 Mapping diagnostics

Every conversion should report:

- semantic input;
- native output;
- clipped semantic values;
- clipped native values;
- quantization error;
- unmapped/reserved slots;
- expected round-trip error.

This information belongs in per-tick telemetry during commissioning.

---

## 16. SafetySupervisor

### 16.1 Position in the command chain

There is exactly one hand-command chain per tick:

    LeaseAuthority / CommandArbiter
      -> select one lease-authorized producer and its lease-free candidate
      -> transition blending when that controller is the transition owner
      -> exclusive gateway attaches AuthorizedSemanticHandCommand envelope
      -> semantic pre-map safety checks
      -> HandDriver prepares one immutable native command
      -> post-map effective-target and quantization checks
      -> HandDriver sends that exact prepared command
      -> acknowledgement/following-error health update

Teleoperation, transition, RL, and safety are not concurrent command sources into a last-writer-wins driver. CommandArbiter first proves which session/owner/token is authoritative. Safety then validates the single candidate. The arm command path uses the same ownership and safety pattern for teleoperation, arm hold, transition, and safe response; monitoring arm state alone is insufficient.

No controller may bypass arbitration, safety, mapping diagnostics, or lease verification. This is enforced structurally: the command gateways alone own the physical sockets/controller-manager interfaces; direct publishers are disabled/ACL-isolated; candidate ingress is authenticated or capability-verified. HIL must prove that a competing direct publisher or copied token digest cannot actuate either resource.

The software supervisor complements, but does not replace, an independent hardware E-stop and appropriate low-level actuator limits. Safety claims must be limited to what has actually been validated on the final arm, hand, driver, and task setup.

### 16.2 Static preflight checks

Before enabling output:

- validate policy, deployment-binding, trust, calibration, and guard-model hashes;
- validate task, hand, model, side, and ordered joints;
- validate calibration and hardware identity;
- compare policy target limits with hardware reachable limits;
- validate observation codec and model dimensions;
- run finite canonical inference;
- warm inference;
- verify p99 inference fits the deadline budget;
- validate required state capabilities;
- validate transport health;
- verify operator deadman/E-stop path.
- verify independent hand/arm watchdog identity, declared independence level, armed state, heartbeat, expiry deadline, and physical response capability;
- fail if any binding-required collision/workspace model is missing, stale, wrong-frame, or wrong-digest; only explicitly optional guards may be absent.

### 16.3 Per-cycle state checks

- schema ID;
- source sequence monotonicity;
- source and receive timestamp age;
- valid mask;
- finiteness;
- measured joint bounds;
- measured velocity and acceleration;
- driver faults;
- current and temperature where available;
- arm state age and faults;
- wrist pose, twist, and wrench envelope;
- controller heartbeat and lease epoch;
- watchdog heartbeat/armed/expiry health;
- hard hand self-collision, hand/arm collision, workspace, and forbidden-region guards required by the binding.

### 16.4 Per-cycle command checks

Checks are mode-specific. The following position-target checks apply to the current RL and blend path; torque/velocity teleoperation uses separately validated bounds and cannot cross into a position-mode lease without the declared mode switch.

- output shape and schema;
- finiteness;
- position limits;
- velocity and acceleration from the last effective target;
- optional jerk;
- following error;
- repeated saturation;
- command expiration;
- generation-from-state age;
- inference deadline;

Post-map and post-send checks separately cover effective semantic limits, quantization error, unexpected native clipping, acknowledgement level, and following error. They cannot be evaluated truthfully before adapter mapping and transport I/O.

### 16.5 Task-specific supervisory checks

These must not be fed into a proprioceptive policy unless declared, but may gate safety:

- screwdriver fixture/object presence;
- free-object drop detection;
- contact-force or wrench limits;
- object/tool profile match;
- task completion;
- arm-hold loss.

### 16.6 Stop responses

Supported named responses:

- HOLD_LAST_SAFE_TARGET;
- CONTROLLED_HAND_BACK;
- CONTROLLED_RELEASE;
- TORQUE_DISABLE;
- ESTOP.

The deployment binding selects valid responses for each fault class.

CONTROLLED_HAND_BACK is valid only when teleoperation is healthy, synchronized, prepared, and has acknowledged takeover. Otherwise its deterministic fallback is SAFE_HOLD or a higher-severity response.

Opening the hand is not universally safe:

- it may drop a screwdriver;
- it may drop a free object;
- it may create a collision with the environment;
- it may be impossible after communication loss.

### 16.7 Active hold versus stop sending

If hardware requires command keepalives, safe hold must actively resend a validated target.

Stopping sends is valid only if the HandDescriptor explicitly guarantees that the low-level controller safely holds the last target.

An independent low-level/controller-manager watchdog, outside the policy/supervisor process, enforces lease/keepalive expiry after Python hang, SIGKILL, driver-process death, or host communication loss. DeploymentBinding states the response and maximum actuation time. Live commissioning is blocked until process-kill and watchdog-expiry tests prove this path.

Watchdog health is continuously supervised. Failure to arm, identity/independence mismatch, stale heartbeat, unsupported configured response, or expiry uncertainty blocks activation; loss while active triggers the binding's bounded escalation path.

### 16.8 Fault latching

Safety faults should contain:

- stable reason code;
- severity;
- triggering sample and command IDs;
- timestamps;
- measurements and thresholds;
- selected response;
- acknowledgement state.

RL must not automatically resume after recovery. It follows the ownership-consistent recovery path in Section 18.9 and completes shadow priming/readiness validation before any new activation.

When faults are simultaneous, the highest configured severity wins. Safe responses are idempotent. Failure to send or acknowledge a safe response escalates by the deployment binding's bounded timeout; lease expiry transfers authority to safety hold, never implicitly to another ordinary controller.

### 16.9 Exceptions

Any exception in:

- state read;
- observation encoding;
- inference;
- safety;
- mapping;
- send;
- acknowledgement;
- telemetry;

must produce a bounded, tested response. A finally block that only restores signals or closes files is insufficient.

---

## 17. Fixed-rate scheduling and timing

### 17.1 Clock

Use time.monotonic_ns or an injected Clock protocol. Never schedule control using wall-clock time.

### 17.2 Absolute schedule

    scheduled_sample_time = start_time + tick_index * control_dt_ns

Avoid accumulating sleep drift.

Define four distinct times per tick:

- scheduled sample time;
- state-selection cutoff used by latest_before;
- decision/send deadline;
- command validity deadline.

They all use the injected local monotonic clock. Device/source times are converted only for diagnostics and freshness after applying their declared uncertainty.

### 17.3 Per-tick budget

DeploymentBinding should allocate:

- state selection/read;
- observation encoding;
- actor/adapter inference;
- action transformation;
- safety;
- hardware mapping;
- send;
- optional acknowledgement;
- telemetry enqueue.

### 17.4 State freshness

A valid numeric vector is not necessarily fresh.

Timed state must include:

- local monotonic receive timestamp;
- source timestamp/clock/uncertainty when the device or middleware genuinely provides them;
- monotonically changing sequence where supported.

For CAN, the driver or SDK adapter must expose a new-sample indicator rather than treating a cached vector as fresh forever. Before activation, validate HandDescriptor state and command rates against the policy period and DeploymentBinding. State selection uses latest-before semantics; interpolation is forbidden unless a specific codec declares, trains, and tests it.

### 17.5 History gaps

If state spacing exceeds the codec timing tolerance:

- mark history invalid;
- enter safe hold;
- require a full or configured partial reprime;
- after ownership-safe recovery, return through RL_SHADOW (or safety-owned RECOVERY_SHADOW followed by normal handoff) before activation.

Irregularly spaced history must not be silently treated as fixed-rate RMA history. The same rule applies to missed inference or command ticks: holding a previous target changes the temporal contract and is permitted only when the codec and DeploymentBinding explicitly define and test that behavior.

### 17.6 Deadline response

- one isolated miss: apply configured last-safe-target response and record it;
- repeated misses: SAFE_HOLD;
- severe overrun or unresponsive I/O: configured critical response;
- never merely print warnings indefinitely.

### 17.7 Asynchronous I/O and telemetry

- state readers may run independently into a bounded timestamped buffer;
- acknowledgement readers enqueue timestamped evidence into a bounded scheduler-owned queue;
- PolicySnapshotBuilder performs all state/target temporal joins on the scheduler thread;
- telemetry uses a bounded non-blocking queue;
- telemetry drops are counted;
- disk or network logging must not block control;
- model warmup occurs before ownership transfer.

Acknowledgement waiting is capability- and profile-dependent. End-to-end timing evidence must be measured on intended deployment CPU/GPU, transport, telemetry load, and process topology; idle model warmup p99 alone is not a real-time guarantee.

---

## 18. Mixed teleoperation-to-RL integration

Status (revision 2): reference design. No teleoperation stack currently exists in this repository, and ADR-016 (arm ownership) and ADR-017 (teleoperation command representation) are open. The boundary contracts are normative now — ArmCommandGateway (Section 14.9), ArmHoldController (Section 18.10), the lease model (Section 18.2), and RL-owns-fingers-only (Section 18.1, ADR-007). The detailed state machine and transition semantics in Sections 18.3–18.9 are expected to be revised on first contact with the selected controller stack and must be re-baselined when ADR-016/ADR-017 are accepted; do not implement them before that point.

### 18.1 Operating assumption

The current RL policies control finger position targets only.

During RL_ACTIVE:

- the arm is owned by a deterministic hold or impedance controller;
- the RL policy owns the hand only;
- a safety supervisor monitors both;
- teleoperation remains available for prepared hand-back but does not command the same resources concurrently.

### 18.2 Resource leases

Maintain independent leases:

- ARM;
- HAND.

Every lease has:

- globally unique control session ID;
- owner ID;
- monotonically increasing epoch;
- command/servo mode;
- acquisition time;
- expiration/heartbeat rules;
- acknowledgement.

LeaseAuthority is the single authoritative store and exposes compare-and-swap acquire, renew, prepare-transfer, commit-transfer, release, and revoke operations. It issues an opaque LeaseToken bound to resource, control_session_id, owner, epoch, command mode, and expiration. The driver installs/verifies that token atomically. Persisted or randomly unique session identity prevents a restarted supervisor from reusing epoch zero. Authority loss or expiration causes safety ownership/hold; it never grants a normal controller implicitly.

Every command includes its control session, epoch, mode, and lease-token digest. Drivers reject stale or unauthorized commands so delayed messages from a previous owner cannot regain control. Switching from teleoperation torque/velocity mode to RL position mode requires a deployment-binding-defined atomic low-level mode switch and bumpless initialization.

Lease transfer is a linearizable transaction:

1. LeaseAuthority creates a new token in INACTIVE state while the old token remains active.
2. The exclusive resource gateway prepares the inactive token, target/controller state, and any mode switch without accepting its commands.
3. One atomic gateway activation is the linearization point: it activates the new token and revokes the old token, or intentionally enters a no-owner safety-hold interval; it never exposes two active ordinary owners.
4. LeaseAuthority records COMMITTED idempotently from the gateway's authoritative active-token query.
5. On crash/restart at any phase, recovery queries gateway/watchdog state, rejects ambiguous commands, and converges to the single active owner or safety hold before resuming.

Fault tests terminate authority, supervisor, old owner, new owner, and gateway after every phase and prove there is never dual actuation.

### 18.3 Ownership states

| State | Arm owner | Hand owner | Purpose |
|---|---|---|---|
| DISCONNECTED | none | none | No command channel |
| TELEOP_ACTIVE | teleop | teleop | Normal arm and hand teleoperation |
| RL_SHADOW | teleop | teleop | RL history priming and candidate recording only |
| ARM_HOLD_VERIFY | arm-hold | teleop | Bumpless arm transfer and stability verification |
| HAND_BLEND | arm-hold | transition controller | Bounded hand transition |
| RL_ACTIVE | arm-hold | RL | RL finger control |
| HAND_BACK_PREP | arm-hold | RL | Synchronize teleoperation command generators |
| HAND_BACK_BLEND | arm-hold | transition controller | Bounded return of hand ownership |
| SAFE_HOLD | safety/hold | safety/hold | Latched recoverable response |
| RECOVERY_SHADOW | safety/hold | safety/hold | Optional read-only policy reprime while safety retains ownership |
| ESTOP | safety | safety | Critical latched response |

### 18.4 TELEOP_ACTIVE to RL_SHADOW

Entry actions:

- load and validate policy package and deployment binding;
- validate hardware and calibration;
- start exact-rate policy sampling;
- feed measured state and the post-mapping effective teleoperation target selected by PolicySnapshotBuilder;
- compute and record candidates without sending them.

Readiness conditions:

- required history complete;
- state timing valid;
- hand schema valid;
- start envelope valid for a stable window;
- arm/task context within allowed envelope;
- inference budget valid;
- no safety fault.

### 18.5 RL_SHADOW to ARM_HOLD_VERIFY

Preconditions:

- teleoperator explicitly authorizes;
- readiness stable for configured ticks;
- arm hold controller is prepared from the last gateway-acknowledged effective teleoperation arm target;
- controller manager can transfer the ARM lease atomically.

Actions:

- acquire ARM lease with a new epoch;
- acknowledge acquisition;
- keep hand under teleoperation;
- verify wrist tracking, pose, twist, wrench, and health for a stable window.

### 18.6 ARM_HOLD_VERIFY to HAND_BLEND

Preconditions:

- arm hold stable;
- policy history remains valid;
- start envelope remains valid;
- first RL candidate computed from the latest interval-correct effective hand target.

Actions:

- transfer HAND lease to transition controller at a policy tick boundary;
- at every policy tick, advance history using the latest effective blended target and recompute the live policy candidate;
- generate a rate- and acceleration-limited blend from the current effective target toward that current candidate, with alpha=1 equal to the same-tick policy candidate;
- feed every post-mapping effective blend target back into PolicySession exactly once;
- abort to safe hold, or to synchronized teleoperation only if teleoperation is already prepared and acknowledged, when conditions leave the envelope.

### 18.7 HAND_BLEND to RL_ACTIVE

Preconditions:

- blend complete;
- measured following error within limit;
- arm hold remains stable;
- policy and safety healthy.

Actions:

- transfer HAND lease to RL with a new epoch;
- after alpha=1's post-map effective target is recorded, atomically commit the lease and call PolicySession.activate without appending that tick again;
- retain exact codec and scheduler phase;
- begin active task metrics and completion monitoring.

### 18.8 RL_ACTIVE to hand-back

Triggers:

- task completion;
- operator request;
- time/tick bound;
- task-specific noncritical condition;
- planned policy deactivation.

Sequence:

1. Enter HAND_BACK_PREP while RL remains owner.
2. Initialize teleoperation hand targets from the current effective RL targets.
3. Initialize teleoperation arm targets from the held wrist target.
4. Require teleoperation prepared acknowledgement.
5. Transfer HAND to transition controller.
6. Blend if needed.
7. Transfer HAND to teleoperation.
8. Record the final effective hand target and call PolicySession.deactivate after committed HAND transfer.
9. Transfer ARM only after teleoperation acknowledges synchronization to the held pose.

Never implement hand-back by simply stopping RL messages.

HAND_BACK_BLEND uses the same live-endpoint rule: the transition controller recomputes the prepared teleoperation target each tick, blends from the latest effective command, and guarantees that alpha=1 equals that same-tick teleoperation candidate. Continuity is tested immediately before and after both HAND lease transfers.

Every preparation, acknowledgement, blend, lease transfer, and mode switch has a bounded timeout and explicit fallback. If teleoperation fails after HAND transfer but before ARM transfer, retain or reacquire safety/arm-hold ownership and enter SAFE_HOLD. Operator deadman behavior remains explicitly configured during RL_ACTIVE; release may request controlled hand-back or safe hold but can never be ignored accidentally.

### 18.9 Safe transitions

Any state may enter SAFE_HOLD or ESTOP.

Communication recovery does not automatically restore RL. SAFE_HOLD retains safety ownership, so normal recovery is SAFE_HOLD to prepared/acknowledged hand-back, then TELEOP_ACTIVE, then a fresh RL_SHADOW. A separate RECOVERY_SHADOW state is allowed only if safety/hold retains both leases and the policy remains read-only. ESTOP requires manual reset and complete revalidation, normally through DISCONNECTED. No ownership-inconsistent direct SAFE_HOLD to RL_SHADOW transition is legal.

Every non-steady state declares an entry timestamp, deadline, required acknowledgements, timeout reason code, and deterministic timeout transition. Transition side effects are idempotent so supervisor restart/replay cannot double-transfer a lease.

### 18.10 ArmHoldController protocol

    class ArmHoldController(Protocol):
        def prepare(
            self,
            state: TimedArmState,
            applied_target: ArmTarget,
            takeover_deadline_ns: int,
        ) -> ArmHoldReadiness: ...
        def acquire(self, token: LeaseToken) -> ControllerAck: ...
        def status(self) -> ArmHoldStatus: ...
        def heartbeat(self) -> TimestampedControllerHeartbeat: ...
        def prepare_release(
            self, next_owner: ControllerIdentity, deadline_ns: int
        ) -> ControllerAck: ...
        def release(
            self, next_owner: ControllerIdentity, deadline_ns: int
        ) -> ControllerAck: ...

ArmHoldStatus reports the held target, timestamp, heartbeat age, pose/twist/wrench error, controller mode, lease identity, and any takeover/release fault. The arm controller-manager adapter must prove atomic switching rather than relying on two processes racing to publish.

ArmHoldController is constructed over ArmCommandGateway and never publishes around it. Preparation includes gateway identity/watchdog checks; acquire/release use the linearizable lease transaction and atomic mode-switch APIs described above.

### 18.11 Wrist/task start envelope

DeploymentBinding should state:

- authoritative TaskHandProfile ID/hash and tolerance around its task_from_wrist transform, rather than a second nominal transform value;
- position and orientation tolerance;
- maximum wrist linear/angular velocity;
- maximum arm tracking error;
- wrench envelope;
- stable window;
- permitted compliance profile.

### 18.12 Policy start envelope and deterministic system readiness

The empirical policy start envelope is packaged as model evidence and is built from successful, settled training/evaluation starts.

Include:

- joint positions;
- effective targets;
- position tracking error;
- joint velocities;
- recent target velocities;
- actor and adapter history validity;
- proprioceptive features explicitly available in PolicySample.

Use:

- deterministic hard bounds;
- an empirical distance or outlier score based on successful start samples.

A blend handles a small command discontinuity. It must not be used to conceal an out-of-distribution grasp.

Object/contact/task-state portions of an empirical start distribution, when used, live in a separately typed TaskReadinessDistribution evaluated against TaskReadinessContext by HandoffSupervisor. They never expand PolicySample or PolicyReadinessReport implicitly.

Deterministic system readiness is evaluated separately: wrist/task errors, object presence, communication health, E-stop, calibration/hardware identity, arm hold, teleoperation hand-back health, watchdog health, timestamp skew, and safety faults remain explicit guards. DeploymentBinding names a shadow_priming_strategy, such as repeated-reset-fill-compatible or full-live-history, and promotion evidence must cover that strategy because a numerically full history may still be out of the Stage-2 training distribution.

### 18.13 Upstream command representation

The current RMA history assumes position targets.

If teleoperation commands:

- torque;
- velocity;
- synergies;
- Cartesian hand commands;

then an equivalent effective semantic position target may not exist.

Options:

1. Define and validate a conversion into the policy history representation.
2. Expose the actual low-level servo target if available.
3. Retrain the policy and adaptation model using the upstream command representation.

Do not silently substitute measured position for a target without validating the distribution change.

---

## 19. Evaluation architecture

### 19.1 TaskEvaluator protocol

Evaluation must be selected by TaskSpec, not inferred from ad-hoc metric keys.

    class TaskEvaluator(Protocol):
        evaluator_id: str

        def reset(self, context: EvaluationContext) -> None: ...
        def update(self, step: EvaluationStep) -> None: ...
        def finalize(self) -> EvaluationResult: ...

EvaluationStep contains step index, environment/episode IDs, reward, pre-reset observation, action, task metrics, terminal observation, terminated/truncated flags, stable termination-reason codes, final episode extras, and any post-reset next observation as a separately named field. This prevents vector-environment auto-reset from attributing the next episode's state to the episode that just ended. Each TaskSpec owns a physical success definition independent of reward weights.

### 19.2 EvaluationResult

    EvaluationResult:
      schema_version
      evaluation_id
      subject_artifact_type
      subject_artifact_id
      runtime_mode
      policy_contract_fingerprint
      experiment_fingerprint
      resolved_environment_hash
      evaluator_id
      gate_profile_id
      gate_profile_hash
      seed
      num_envs
      num_steps
      completed_episodes
      runtime_versions
      started_at
      completed_at
      raw_episode_evidence_hash
      paired_run_id
      common_metrics
      task_metrics
      warnings
      gate_results
      passed

Common metrics:

- completed episodes;
- success rate;
- termination reason counts;
- state/action validity;
- policy latency;
- finite-output rate;
- saturation rate.

Screwdriver metrics:

- net and total turns;
- forward/reverse velocity;
- fall/tilt rate;
- contact/manipulation fraction;
- coasting diagnostic;
- force or pad-quality metrics;
- task-specific success threshold.

In-hand metrics:

- hold fraction;
- drop rate;
- episode length;
- object height;
- physical angular displacement, direction, and velocity;
- reward components as diagnostics only;
- pose, torque, work, and linear-velocity costs;
- task-specific success threshold.

### 19.3 Machine-readable output

Required outputs:

- canonical JSON;
- JSON schema validation;
- optional human terminal rendering;
- optional structured event stream.

Exit codes:

- 0: evaluation completed and all required gates passed;
- 2: evaluation completed but one or more gates failed;
- 1: runtime, configuration, artifact, or data failure.

External pipelines must never parse colored terminal text to make a promotion decision.

An explicitly ungated evaluation has gate_profile_id set to null, passed set to null, and exits 0 when execution and evidence validation succeed. It must not be mistaken for a promotion pass.

### 19.4 Actual-runtime deployment gate

The deployment gate must instantiate the same PolicySession used on hardware.

Define two actual-runtime levels:

1. Policy parity: PolicySession to semantic HandRuntime.
2. Deployment-binding parity: scheduler, CommandArbiter, SafetySupervisor, immutable prepared-command mapping/round trip, quantization/saturation behavior, fake transport acknowledgements, and effective-target feedback.

Policy-parity simulator loop:

1. Convert simulator hand state into TimedHandState with the resolved semantic schema.
2. provide the post-mapping-equivalent semantic target actually accepted in simulation;
3. call PolicySession.prime or step;
4. pass the already transformed candidate through transition/safety only; PolicySession owns ActionTransform exactly once;
5. apply the safe semantic target through HandRuntime;
6. collect task evaluator metrics.

It must not directly call internal RL-Games actor layers as a substitute.

A third pre-HIL gate replays recorded teleoperation hand/arm state and upstream targets through shadow priming, arm hold, live-endpoint blend, RL activation, and hand-back. Runtime latency promotion is repeated on the intended deployment hardware under representative I/O and telemetry load.

### 19.5 Oracle versus adapter comparison

Run matched seeds and environment settings for:

- oracle teacher latent;
- environment-free adapter runtime.

Report:

- absolute metrics;
- absolute degradation;
- relative degradation;
- confidence intervals;
- pass/fail thresholds;
- number of completed episodes.

GateProfile versions the statistical method and sample unit. Use episode-level samples, not correlated per-step samples. Default success-rate intervals use Wilson intervals; continuous matched oracle/adapter differences use a seeded paired bootstrap. Each metric declares minimum completed episodes, confidence level, aggregation statistic, missing-data rule, and whether pairing is required.

For deterministic promotion, GateProfile also fixes vector-environment episode-seed derivation, environment-to-episode assignment, bootstrap resample count, bootstrap seed derivation, comparator (<, <=, >, or >=), whether the point estimate or named confidence bound drives the decision, numeric precision/rounding, and exact boundary behavior. Evaluation evidence records the realized episode seeds and resample seed. Two implementations given the same evidence must produce the same decision JSON.

### 19.6 Gate profiles

Gate profiles are versioned artifacts selected by task and deployment binding.

Example screwdriver gates:

- minimum success rate;
- maximum fall rate;
- minimum p10 and median net turns;
- maximum free-spin/coasting;
- maximum adapter degradation;
- maximum saturation;
- p99 inference deadline.

Example in-hand gates:

- minimum hold fraction;
- maximum drop rate;
- minimum episode length;
- object-height bounds;
- rotation-performance threshold;
- maximum torque/work;
- maximum adapter degradation.

### 19.7 Evaluation completeness

Fail evaluation when:

- required metrics are missing;
- any required metric is NaN or infinite;
- too few episodes complete;
- an evaluator receives a task it does not support;
- artifact/config identity is inconsistent;
- deployment runtime and simulator period differ.

---

## 20. Training architecture

### 20.1 CLI/library separation

Root train.py, play.py, and eval.py should become thin wrappers.

Reusable APIs:

    train_stage1(
        resolved: ResolvedExperiment,
        options: Stage1Options,
        context: SimulatorContext,
    ) -> TrainingResult

    train_stage2(
        resolved: ResolvedExperiment,
        source: Stage1Checkpoint,
        options: Stage2Options,
        context: SimulatorContext,
    ) -> TrainingResult

    evaluate(
        resolved: ResolvedExperiment,
        policy: PolicyPackage,
        options: EvaluationOptions,
        context: SimulatorContext,
    ) -> EvaluationResult

No library import should:

- parse sys.argv;
- discard unknown arguments;
- start Isaac;
- install global RL-Games registries without an explicit call.

### 20.2 Simulator compatibility adapter

The repeated Isaac Lab/RL-Games import fallback logic should move into one compatibility module with:

- detected versions;
- supported version matrix;
- explicit errors for unsupported combinations;
- one tested adapter per supported branch.

### 20.3 Agent configuration

Use base algorithm templates plus derived dimensions.

Agent configuration resolution:

1. load selected algorithm, curriculum, domain-randomization, and simulation profiles;
2. derive action, actor proprio, privileged, and latent dimensions;
3. apply strict experiment overrides;
4. validate PPO batch/minibatch consistency;
5. freeze and hash;
6. record in run metadata.

Hand-dependent dimensions should not be repeated in YAML.

### 20.4 Public environment export contract

Replace private attribute introspection such as home targets, internal target limits, and private curriculum state with:

    env.policy_contract_snapshot() -> ReadOnlyPolicyContractSnapshot
    env.codec_trace_snapshot() -> ReadOnlyCodecTraceSnapshot
    env.export_policy_samples(selection) -> PolicySampleExportBatch
    env.set_curriculum_phase(phase_id) -> None

Snapshots are immutable copies or read-only views with declared lifetime; training code cannot mutate environment state through them. Raw mutable proprio_history tensors are not a stable public API. Collection obtains versioned sample records through an explicit PolicySampleExport interface.

Stage-2 artifact export failure should be fatal by default. An explicit allow-adapter-only development flag may opt out.

### 20.5 Stage-1 resume

Define exact and warm resume separately.

An exact Stage-1 checkpoint is taken only at a declared PPO-update boundary with no partially consumed rollout. Persist:

- model;
- optimizer;
- scheduler/scaler;
- epoch/update and global environment steps;
- rollout-buffer/cursor state, or an assertion that the boundary has no live rollout;
- complete simulator/environment snapshot needed to continue: articulation/object state, semantic/effective targets, episode progress and IDs, reset/termination state, curriculum/domain-randomization state, per-environment RNG/reset-seed cursors, and simulator/backend RNG/state;
- Python/NumPy/Torch CPU/CUDA and runner RNG state;
- resolved experiment fingerprint;
- source-code and asset provenance.

A backend may claim exact resume only if its snapshot/restore contract passes deterministic equivalence. Otherwise the checkpoint is warm: it restores model/optimizer where valid, starts new environment/episode lineages, records its parent, and cannot claim identical sample or loss continuation. A plain resume must continue the same curriculum rather than requiring manual reconstruction through a CLI counter.

### 20.6 Stage-2 memory problem

Current collection retains every environment history for every rollout step and concatenates afterward.

For history shape 30 by 32:

- 2,048 environments by 512 steps is about 3.75 GiB raw history and at least 7.5 GiB around concatenation peak.
- 8,192 environments by 512 steps is about 15 GiB raw history and at least 30 GiB around concatenation peak.

This excludes simulator, actor, adapter, optimizer, targets, and temporary tensors.

### 20.7 Stage-2 full-corpus bounded-memory collection

The first behavior-preserving migration replaces list/cat accumulation with a chunked or memory-mapped full-corpus writer, followed by training on an immutable frozen dataset snapshot. It preserves the current sample set, rollout-step-major/environment-major ordering, dtype, target calculation, epoch shuffle seed, and number of training passes exactly. It performs no environment subsampling, eviction, reservoir choice, or interleaving. Only bounded staging/prefetch windows reside in RAM/GPU; the complete 8,192-env corpus may occupy roughly 15 GiB plus metadata on disk without a concatenate-sized second copy.

Every record includes source policy-package/checkpoint ID, policy-contract fingerprint, adapter/collector version, codec ID, environment/episode/step identity, sample key, and corpus-order index. Dataset chunks and the ordered manifest are checksummed; restart is idempotent by chunk/sample ID.

After fixed-corpus loss/action parity is demonstrated, a separately versioned interleaved collection-and-training mode may be introduced. Interleaving changes the data distribution and must not be presented as a memory-only refactor.

Behavior-preserving collector requirements:

- chunked/memory-mapped full-corpus storage;
- bounded CPU-pinned staging buffer;
- microbatch prefetch;
- gradient accumulation;
- optional mixed precision;
- explicit host and device memory budgets.

Illustrative later behavior-changing interleaved flow:

    for rollout_step in range(rollout_steps):
        obs = env.step(action)
        selected = sampler.select_envs(obs, samples_per_step)
        exported = env.export_policy_samples(selected)
        ring_buffer.append_async(
            history=exported.adapter_history,
            target=exported.teacher_conditioning,
            metadata=exported.versioned_metadata,
        )

        if trainer.ready_for_update():
            for batch in ring_buffer.prefetch_batches():
                trainer.update(batch)

In that future mode, memory must scale with the configured ring size, not num_envs times rollout_steps.

Ingestion is idempotent by sample ID, and collector restart cannot duplicate records silently.

### 20.8 Later sampling/interleaving requirements

These apply only to the separately versioned behavior-changing mode, not the initial full-corpus migration:

- seed every sampler;
- record selected environment IDs and sample ages when debugging;
- independently shuffle each training epoch;
- prevent on-policy refinement from training indefinitely on stale data;
- expose scale/object/shape/task-profile distribution diagnostics;
- allow stratification where rare geometry or reset modes matter.

### 20.9 Stage-2 memory estimator

Before launching Isaac, print and optionally emit JSON for:

- simulator estimate where available;
- staging/prefetch device and host bytes;
- full-corpus on-disk bytes, free-space requirement, and chunk overhead;
- batch bytes;
- model/optimizer bytes;
- estimated peak;
- configured cap.

Fail before simulator allocation when an explicit cap cannot be met.

Initial benchmark profiles are the current 2,048-env by 512-step and 8,192-env by 512-step in-hand settings with 30-by-32 float32 histories. Proposed default collector budgets, to be accepted or revised by ADR/evidence, are 8 GiB host RSS delta and 1 GiB device allocated-memory delta above the warmed simulator/training baseline; no concatenate-sized second copy is permitted. Measure host current/peak RSS and Torch CUDA memory_allocated/max_memory_allocated from the same baseline immediately before collection. Record on-disk corpus size, chunk/staging capacity, sample count/order hash, dtype, pinning, prefetch depth, and allocator fragmentation. CI may use a proportionally smaller CPU fixture, while the self-hosted Isaac gate runs both named profiles.

Before enabling any changed sampling distribution, train on one frozen reference corpus using old and new loaders and require predefined tolerances for per-batch loss, final adapter outputs on a held-out trace, and downstream deterministic policy actions. The evaluation owner records the numerical tolerance in a versioned Stage2ParityProfile rather than choosing it after seeing results.

### 20.10 Stage-2 resume

Persist:

- adapter;
- optimizer;
- scheduler/scaler;
- iteration and global sample count;
- source actor package/checkpoint identity;
- resolved experiment fingerprint;
- curriculum state;
- Python RNG;
- NumPy RNG;
- Torch CPU/CUDA RNG;
- corpus manifest/chunk digests and completed-chunk cursor;
- epoch shuffle generator/order and batch cursor;
- staging/prefetch state needed at the declared checkpoint boundary.

Large corpus/replay snapshots should be separate from deployable packages.

Define two resume levels:

- exact resume requires the immutable corpus/chunks, manifest/order, cursors, all RNG states, optimizer/scheduler/scaler state, source identities, and checksums; it may claim equivalence;
- warm resume may omit corpus or cursors, creates a new run lineage, records the parent checkpoint, and cannot claim deterministic equivalence.

### 20.11 Resume equivalence tests

Stage-1 and Stage-2 have separate small deterministic tests. Each should:

1. run continuously to iteration N;
2. run to iteration K;
3. save;
4. resume to N;
5. compare weights, optimizer state, environment/sample sequence, curriculum state, and loss trajectory within defined tolerance.

The Stage-1 test includes an exact environment snapshot/restore at a legal PPO boundary and a separate warm-resume lineage assertion. The Stage-2 test covers completed-chunk restart plus an exact optimizer/batch-cursor resume over the same frozen corpus.

Stage-2 acceptance also fixes representative 2,048- and 8,192-environment profiles, default host/device staging caps, required disk capacity/chunk scheme, the allocator metric and sampling point relative to simulator baseline, the corpus/order hash, and fixed-corpus loss/action tolerances. A configurable cap without a reference profile is not an acceptance criterion.

### 20.12 Run directory

    runs/
      experiment-id/
        run-id/
          run.json
          resolved-experiment.json
          environment.json
          stage1/
            checkpoints/
          stage2/
            checkpoints/
            corpora/
          evaluations/
          policy-candidates/

run-id should be unique and contain time, Git SHA, and a collision-resistant suffix.

Corpora and large replay snapshots follow an explicit retention rule recorded in run.json (default proposal, revisable by ADR/evidence: keep the newest two corpora per experiment plus any corpus referenced by a checkpoint lineage). Deleting a corpus demotes dependent checkpoints' exact-resume claim to warm.

---

## 21. Packaging and resources

### 21.1 Build backend

Correct pyproject.toml to a valid backend:

    [build-system]
    requires = ["setuptools>=61"]
    build-backend = "setuptools.build_meta"

### 21.2 Source layout

A src layout is recommended so tests cannot accidentally pass by importing the checkout:

    src/screwdriver_rl/

Perform this migration in the foundation phases, before proliferating new packages. Clean-wheel tests begin immediately and remain required; otherwise later refactors may accidentally depend on checkout-relative imports.

### 21.3 Optional dependencies

Suggested groups:

- train: RL-Games, PyYAML, and ordinary training dependencies;
- sim: Isaac integration dependencies or compatibility metadata;
- deploy: CPU Torch and common runtime dependencies;
- linker: Linker transport dependencies;
- tools: NumPy/SciPy/rendering utilities;
- test: pytest, coverage, build, lint, and type tools;
- dev: aggregate developer tooling.

Isaac Sim/Lab may still require vendor installation instructions, but the supported versions must be machine-readable.

### 21.4 Console entry points

Suggested:

- screwdriver-rl train;
- screwdriver-rl evaluate;
- screwdriver-rl play;
- screwdriver-rl deploy;
- screwdriver-rl list-experiments;
- screwdriver-rl doctor;
- screwdriver-rl policy verify;
- screwdriver-rl policy migrate;
- screwdriver-rl policy promote;
- screwdriver-rl assets sync;
- screwdriver-rl assets verify.

Root scripts remain compatibility wrappers for one migration period.

### 21.5 AssetResolver

Resolution order may support:

1. explicit configuration path;
2. SCREWDRIVER_RL_ASSET_ROOT;
3. installed structural-resource package;
4. synchronized local artifact store.

No core module should assume parents[3] is the repository root.

### 21.6 Structural assets versus generated data

Structural assets:

- URDF;
- meshes;
- small manifests;
- small canonical fixtures.

Generated data:

- large grasp caches;
- renders;
- training outputs;
- experiment artifacts.

Generated data should not live in the normal wheel or ordinary Git history.

Before publishing any wheel or asset package, inventory the project license and every third-party URDF, mesh, texture, model, SDK, and generated derivative. Record source, author, license/SPDX ID, modification status, notice requirement, and redistribution permission. Unknown or incompatible rights block redistribution even when the file is technically packageable. Generate required LICENSES/ and NOTICE material and test that published artifacts include it.

### 21.7 Grasp-cache manifest

Required metadata:

- schema version;
- cache ID;
- task and hand IDs;
- semantic joint schema and order;
- row layout;
- object shape, scale, and prototype;
- generator revision;
- resolved experiment hash;
- URDF/asset hash;
- seed;
- row count;
- dtype and dimensions;
- checksum;
- validation statistics.

### 21.8 Asset store

Choose one:

- DVC;
- Git LFS;
- internal object/artifact store.

Keep small test fixtures in Git. Do not rewrite Git history casually; plan that migration separately.

### 21.9 Offline behavior

Training must not unexpectedly fetch over the network.

When assets are missing:

- list exact required IDs;
- show expected checksums;
- provide an explicit sync command;
- exit before simulator allocation.

### 21.10 Install acceptance

- build wheel and sdist;
- install in an empty virtual environment;
- remove repository root from sys.path;
- import pure runtime;
- list experiments;
- load packaged YAML/templates;
- locate structural assets or report required external asset package;
- run a deploy dry-run.

---

## 22. Reproducible environments

### 22.1 Compatibility manifest

Create environments/compatibility.toml containing supported combinations of:

- Python;
- Isaac Sim;
- Isaac Lab;
- Isaac Lab RL;
- RL-Games;
- Torch;
- CUDA;
- NVIDIA driver;
- Gymnasium;
- PyYAML;
- Linker SDK revision for deployment.

The manifest must also record the three concrete environments from Section 2.4 — the macOS CPU development machine, the Ubuntu CUDA/Isaac training rig, and the Ubuntu deployment/hardware station — so every test and CI job can name which environment it claims to cover.

### 22.2 Locks

Provide:

- a pinned simulator environment or image definition;
- a lightweight CPU deployment lock;
- test/developer lock where practical; it must resolve on both macOS (development) and Linux (CI and the rig).

Pin container images by digest when containers are supported.

### 22.3 doctor command

screwdriver-rl doctor --json should report:

- installed versions;
- compatibility decision;
- GPU/driver;
- simulator availability;
- asset status;
- policy codec support;
- transport/SDK availability;
- hardware permissions where checkable;
- actionable remediation.

### 22.4 Runtime provenance

Every training run, evaluation, and policy package records:

- compatibility-manifest hash;
- detected runtime versions;
- GPU/driver where relevant;
- SDK revision where relevant.

---

## 23. CLI and external-pipeline APIs

### 23.1 Strict arguments

Unknown arguments must fail. parse_known_args followed by discarding unknown values is not acceptable for automation.

### 23.2 Typed option models

Use:

- TrainOptions;
- Stage2Options;
- EvaluationOptions;
- PlayOptions;
- DeployOptions;
- RunContext.

### 23.3 Import behavior

Importing a module must not:

- parse arguments;
- start Isaac;
- initialize ROS;
- mutate global calibration;
- install signal handlers.

### 23.4 Process ownership

Library runtime does not own process signals. CLI wrappers may translate SIGINT/SIGTERM into supervisor stop requests.

### 23.5 Pipeline-facing operations

External pipeline should be able to:

    package = verify_policy_package(path)
    binding = verify_deployment_binding(path, package)
    launch = load_launch_config(path)
    session = create_policy_session(package)
    hand_gateway = create_hand_driver(binding, launch)
    arm_gateway = create_arm_command_gateway(binding, launch)
    lease_authority = create_lease_authority(binding, launch)
    arm_hold = create_arm_hold_controller(arm_gateway, binding)
    supervisor = create_handoff_supervisor(
        session=session,
        hand_gateway=hand_gateway,
        arm_gateway=arm_gateway,
        arm_hold=arm_hold,
        lease_authority=lease_authority,
        safety=safety,
        binding=binding,
    )

No CLI invocation or subprocess should be required for in-process integration.

### 23.6 Plugin discovery

Define Python entry-point groups for external:

- hand specs;
- task-hand profiles;
- hardware adapters;
- transports;
- evaluators.

Plugin loading is opt-in through an allowlisted PluginLoadPlan. Discovery never auto-imports arbitrary installed entry points in a live runtime. Validate plugin API range, package name/version/hash/signature policy, declared capabilities, and duplicate IDs before executing plugin registration. Registries reject duplicates and freeze after resolution; late mutation is an error. The exact plugin package/version/digest and contributed record digests enter experiment and policy-contract fingerprints. Production bindings may disable plugins entirely or name an exact trusted allowlist.

This whole mechanism is deferred to Phase 10 together with ADR-025/ADR-026 (Sections 2.5 and 31.11). Until then the only supported extension path is an in-repo package registered through the ordinary registries.

---

## 24. Observability and replay

### 24.1 Structured events

Define versioned events:

- state received;
- readiness changed;
- ownership transition requested/accepted/rejected;
- policy inference;
- safety decision;
- command mapped;
- command sent/acknowledged;
- fault entered/cleared;
- task metric;
- hand-back.

### 24.2 Per-tick trace

Record:

- wall time for human correlation plus local monotonic scheduled-tick time, state-selection cutoff, decision deadline, and command-validity deadline;
- state source time/clock, local monotonic receive time, uncertainty, and sequence;
- PolicySnapshotBuilder join interval/key plus arm/hand/task timestamp skew;
- control session ID, arm/hand owners, lease epochs/tokens, and the arbiter's authority decision;
- measured semantic q;
- requested, blended, safety-filtered, post-mapping effective, transport-accepted, acknowledged, and inferred-applied targets where available;
- actor proprio hash or optional values;
- declared ExecutionGraph conditioning outputs (for example adapter latent), when present;
- raw action;
- candidate target;
- filtered target;
- prepared-native command, calibration/mapping digest, semantic round trip, and exact transmitted payload digest;
- saturation and quantization;
- transport enqueue/send/device-ack timestamps and acknowledgement level;
- inference/end-to-end timing and deadline-miss classification;
- transition blend weight and endpoint mode;
- safe-response attempt, acknowledgement, and escalation result;
- hand/arm watchdog identity, independence level, armed/heartbeat/expiry state, and physical-response capability;
- exclusive gateway/authenticated-ingress decision and competing-publisher rejection events;
- arm pose/twist/wrench errors;
- readiness guards;
- safety decisions;
- policy, deployment-binding, launch-run, hardware, calibration, and trust-policy IDs.

### 24.3 Asynchronous sinks

Sinks:

- human terminal;
- JSON Lines;
- binary replay trace;
- TensorBoard;
- CSV compatibility;
- external telemetry.

Sinks consume structured events. They do not define metric meaning.

### 24.4 Replay

Replay should support:

- policy-only deterministic replay from PolicySample sequence;
- full supervisor replay with fake clock and driver;
- simulator trace comparison;
- fault injection at specified state/command IDs;
- regression comparison between runtime versions.

### 24.5 Privacy and size

- allow configurable omission of large tensors;
- retain hashes and selected snapshots for parity;
- rotate or bound logs;
- never let recording block control;
- record dropped-event count.

DeploymentBinding chooses the overflow response. Normal production telemetry may drop optional detail and count it; a commissioning profile that requires a complete promotion trace may instead enter a controlled stop when its mandatory event queue would overflow.

---

## 25. Test strategy

### 25.1 Test levels

Use explicit markers:

- unit;
- integration;
- rlgames;
- isaac;
- hardware.

Unexpected absence of a required dependency in its designated job is a failure, not a passing print-and-return test.

Markers map to the environments in Section 2.4: unit, integration, and rlgames verify on ENV-A (the macOS development machine and Linux CPU CI); isaac verifies on ENV-B; hardware verifies on ENV-C.

### 25.2 Pure contract tests

Run without Isaac, ROS, or vendor SDKs:

- every HandSpec validates;
- every TaskSpec validates;
- every TaskHandProfile references compatible specs;
- every ExperimentSpec resolves deterministically;
- observation and action dimensions derive correctly;
- semantic joint orders are unique and complete;
- grasp presets are within limits;
- coupled-joint masters exist;
- artifact schemas validate;
- wrong-hand and wrong-order packages fail;
- calibration binding fails on identity mismatch.

### 25.3 URDF conformance

Parameterize over every validated HandSpec/HandSimSpec pair:

- independent joint names exist;
- mimic/coupled tags agree;
- fingertip and non-tip bodies exist;
- limits agree with spec;
- expected root link exists;
- asset hash agrees with manifest.

Tests should read HandSpec for canonical joints and HandSimSpec for simulator bindings/bodies/collision data, not copy expected sets into each test file.

### 25.4 Codec parity

For each built-in experiment:

- generate or load a deterministic semantic state/target trace;
- run the simulator-side codec;
- run the runtime-side codec;
- compare frame, actor proprio, adapter history, action, and target integration;
- test reset, normal ticks, blend, safety clipping, missed tick, and reprime.

Required explicit in-hand assertions:

- frame is 32-D;
- actor proprio is 96-D;
- adapter history is 30 by 32;
- q is scaled exactly as training;
- control period is 50 ms.

### 25.5 Artifact tests

- missing schema version;
- unsupported version;
- missing required field;
- wrong tensor hash;
- wrong hand/model/side;
- same-size wrong joint order;
- wrong units;
- wrong control period;
- wrong codec;
- missing/wrong declared normalizer, while normalization-mode-none packages load without one;
- missing/extra ExecutionGraph component or incompatible tensor edge;
- altered reference-output fixture or ComparisonSpec;
- exact-mode and dtype-aware absolute/relative tolerance boundary behavior;
- action dimension mismatch;
- legacy migration ambiguity;
- canonical finite inference;
- round-trip package write/read.
- production dirty-source rejection and development canonical patch/untracked-source reconstruction.

### 25.6 Policy parity with RL-Games

A dedicated job with RL-Games installed must prove:

- exported actor produces the same deterministic action as restored RL-Games actor;
- declared normalization node/state is identical, or both paths explicitly use normalization mode none;
- latent-conditioned path is identical;
- both screwdriver and in-hand actor layouts are covered;
- geometry-DR privileged-width variants are covered.

Use real pytest skip markers only for optional local execution. The designated CI job fails if tests skip because RL-Games is absent.

### 25.7 State-machine tests

Use a fake clock and fake controllers.

Cover:

- every legal transition;
- every illegal transition;
- operator cancellation in shadow;
- arm-hold rejection;
- hand blend interruption;
- task completion;
- safe hand-back;
- old lease epoch command rejection;
- no acknowledgement;
- SAFE_HOLD recovery through prepared hand-back and TELEOP_ACTIVE before ordinary RL_SHADOW, or through explicit safety-owned RECOVERY_SHADOW;
- ESTOP manual reset through DISCONNECTED;
- process stop request.

### 25.8 Fault-injection tests

State faults:

- valid but stale repeated state;
- unchanged sequence;
- delayed/out-of-order/future timestamps;
- state/effective-target temporal-join boundary and excessive arm/hand/task skew;
- wrong schema;
- partial valid mask;
- NaN/Inf;
- out-of-range values;
- excessive velocity/acceleration;
- hardware fault/current/temperature.

Control faults:

- inference exception;
- NaN action;
- wrong output shape;
- slow inference;
- safety exception;
- mapping exception;
- blocked send;
- send exception;
- missing/negative acknowledgement;
- excessive following error;
- repeated saturation.

Arm faults:

- wrist drift;
- motion during activation;
- wrench excursion;
- arm-hold heartbeat loss;
- arm lease loss.

System faults:

- logging backpressure;
- clock jump in fake clock;
- source-clock offset/uncertainty change;
- stale controller command;
- supervisor, policy, driver, safety, and arm-hold process SIGKILL/hang;
- independent low-level watchdog expiry;
- lease-authority restart and epoch reuse attempt;
- duplicate/reordered commands and atomic-transfer race;
- transport reconnect exposing cached pre-disconnect state;
- source sequence wrap/reset;
- safe-hold send failure and escalation;
- driver shutdown during active control;
- controller mode-switch failure;
- teleoperation failure midway through hand-back;
- concurrent valid-looking commands from two producers;
- post-mapping clipping after semantic checks pass;
- prepared-native payload mutation/remapping attempt;
- stale/disconnected deadman, E-stop, object, task, or watchdog signal;
- competing publisher attempting to bypass the exclusive hand or arm gateway;
- crash after every lease-transfer prepare/activate/commit phase;
- E-stop;
- deadman release.

Lifecycle and boundary tests additionally cover duplicate PolicySample ingestion, reentrancy rejection, prime/preview/activate/step exactly-once behavior, idempotent close, imports without argument parsing or Isaac launch, strict unknown CLI options, frame/quaternion direction and unit conversions, signature/key revocation for packages/bindings/calibrations/gate profiles, plugin allowlisting/API mismatch, pip check, and optional-dependency import boundaries.

### 25.9 Evaluator tests

- screwdriver metrics finite and complete;
- in-hand metrics finite and complete;
- insufficient episodes fail;
- required NaN fails;
- JSON validates;
- exit codes are correct;
- oracle/adapter comparison uses matched seeds;
- gate thresholds behave at boundaries.
- fixed episode/bootstrap seeds, comparator, confidence-bound choice, and rounding reproduce byte-identical gate decisions.

### 25.10 Training/resume tests

- Stage-1 exact environment snapshot equivalence and warm-lineage distinction;
- full-corpus Stage-2 sample/order hash identical to the legacy collector;
- bounded RAM/GPU independent of full corpus size and no concatenate-sized second copy;
- deterministic epoch shuffle over the frozen corpus;
- chunk completion/restart idempotence;
- CPU-pinned prefetch correctness;
- exact Stage-2 corpus/cursor resume equivalence;
- source actor/config mismatch rejection;
- memory estimate calculation;
- low-memory failure before simulator launch.

Separate TRAIN-008 tests, only when that behavior-changing mode is proposed, cover seeded sampling, FIFO/window/reservoir behavior, ring wraparound, stale-data limits, and controlled task-metric degradation.

### 25.11 Packaging tests

- wheel and sdist build;
- install into empty environment;
- repository root removed from sys.path;
- agent resources load;
- structural assets resolve;
- list-experiments works outside checkout;
- runtime loads without Isaac;
- deploy dry-run works with deploy dependencies only;
- missing asset errors are actionable.

### 25.12 Isaac smoke tests

Parameterize every built-in experiment:

- register;
- construct with a small environment count;
- reset;
- verify observation/action spaces;
- step ten times;
- assert finite observations/reward/actions;
- assert evaluator required metrics are emitted;
- run a short Stage-1 smoke;
- run two bounded Stage-2 updates from a frozen collected snapshot;
- export and load a candidate policy package.

### 25.13 HIL and hardware tests

Manual/protected:

- read-only shadow;
- mapping with motors disabled where possible;
- arm-hold acquisition;
- stale communication safe hold;
- forced policy-process death with independent watchdog evidence;
- E-stop and deadman drills;
- safe hand-back;
- competing-publisher rejection and crash-at-every-lease-transfer-phase drills;
- certified hand-transition trajectory in a mechanically safe dummy setup after arm hold and recovery paths pass;
- bounded screwdriver trial;
- later bounded in-hand trial.

---

## 26. CI strategy

### 26.1 Pull-request CPU jobs

Required:

1. format and lint;
2. type checking;
3. unit tests;
4. integration tests with CPU Torch;
5. RL-Games CPU parity;
6. wheel/sdist build;
7. clean-wheel install/resource test;
8. artifact and deploy dry-run;
9. documentation generation/link checks;
10. unexpected-skip report.

Initial target: complete CPU PR checks within ten minutes.

All PR CPU jobs are ENV-A (Section 2.4): they run on Linux CPU runners and must also pass locally on the macOS development machine. Importing Isaac or requiring CUDA in these jobs is a defect.

### 26.2 Self-hosted Isaac jobs

Nightly full-matrix and manually dispatchable jobs run all experiments. During simulator-composition phases, a targeted Isaac parity job is also a required merge check for changes under task families, simulation, codecs, assets, hand specs, or environment configuration:

- pinned environment;
- registry smoke for all experiments;
- short training/evaluation;
- codec/runtime parity inside simulator;
- memory report;
- policy candidate creation;
- evaluation JSON artifact.

### 26.3 Protected hardware jobs

- require explicit operator;
- require a selected approved policy package and deployment binding;
- default bounded tick count;
- recording mandatory;
- promotion output stored immutably.

### 26.4 Required check behavior

- missing required dependency fails designated job;
- skipped critical test fails;
- generated docs drift fails;
- packaging resource omission fails;
- unknown registry item fails;
- promotion cannot run from an unvalidated candidate.

### 26.5 Coverage

Initial targets:

- core contracts/codecs/artifact/runtime/safety: at least 85 percent line coverage;
- generic simulator code: measured in self-hosted job;
- branch coverage emphasized for safety and state machine.

---

## 27. Adding new capabilities after the refactor

### 27.1 Adding a new hand model

Required steps:

1. Add structural asset and manifest.
2. Define HandSpec:
   - model/side/revision;
   - semantic independent joints;
   - limits;
   - coupled joints;
   - finger membership;
   - intrinsic kinematic capabilities.
3. Define HandSimSpec/articulation factory with simulator joint/body bindings, fingertip/non-tip bodies, collision exclusions, actuator profile, and simulator capabilities.
4. Add URDF conformance tests.
5. Create TaskHandProfile for each supported task:
   - active fingers;
   - typed initialization/reset source;
   - task_from_wrist transform;
   - selected contact/reward/reset profiles;
   - codec parameters.
6. Resolve and run contract tests.
7. Run Isaac smoke and training baseline.
8. For hardware:
   - implement HandAdapter;
   - implement/integrate Transport;
   - create calibration schema/tooling;
   - create hand-specific safety capability/threshold profiles and a DeploymentBinding;
   - validate mapping strategy, acknowledgement level, watchdog behavior, and compatibility attestations;
   - pass fake-driver and HIL suites.

Must not require edits to:

- generic task environment;
- training runner;
- evaluator runner;
- PolicySession;
- SafetySupervisor;
- scheduler;
- handoff state machine.

### 27.2 Adding a new task

Required:

1. TaskSpec.
2. ObjectModel.
3. ResetStrategy.
4. ContactProvider if needed.
5. RewardModel.
6. TerminationModel.
7. ProprioCodec/ActorInputAssembler or reusable typed configuration.
8. ActionTransform or reusable transform.
9. TaskEvaluator and gate profile.
10. TaskHandProfiles for compatible hands.
11. Contract, codec, evaluator, and Isaac tests.

### 27.3 Adding a task-hand combination

Normally requires:

- one TaskHandProfile;
- one typed initialization source, which may wrap one or more GraspPresets or a cache/distribution;
- optional task-specific cache;
- validation and task evaluation.

It should not require a new environment class.

### 27.4 Adding a grasp variant

Examples: Linker side versus top grasp.

Requires:

- named GraspPreset;
- possibly a new typed start/reset sampling profile; the empirical policy start distribution is generated with the policy package;
- ExperimentPreset;
- validation/evaluation.

No copied robot ArticulationCfg.

### 27.5 Adding a new transport

Implement raw Transport behavior and integrate it with the generic HandDriver:

- timestamped native reader and bounded latest-before buffer;
- native command send and evidence-based acknowledgement levels;
- health;
- low-level shutdown/watchdog primitives;
- idempotent resource close after explicit safe shutdown.

Reuse semantic mapping, lease installation, policy, scheduler, safety-response selection, handoff, and telemetry.

### 27.6 Adding a third-party plugin

External packages may register specs/adapters through versioned Python entry points only when an explicit trusted PluginLoadPlan allowlists the exact package/version/digest. Live auto-discovery remains disabled.

Core registry validation applies identically.

Deferred to Phase 10 (Section 2.5); see Section 23.6.

### 27.7 Adding a new arm/controller integration

Required:

1. ArmIntegrationProfile and frame bindings.
2. ArmCommandGateway adapter that exclusively owns the vendor/controller-manager command interface.
3. ArmDescriptor, timestamped state mapping, mode-switch transaction, acknowledgement mapping, and safe-response contract.
4. ArmHoldController implementation over that gateway.
5. Independent watchdog integration and DeploymentBinding thresholds.
6. Fake/replay/fault adapters, controller-transfer crash tests, competing-publisher rejection, and staged HIL evidence.

PolicySession and generic hand/task code remain unchanged.

---

## 28. File-by-file migration map

Unless a table explicitly starts with src/, every unqualified proposed path in this section is relative to the import package root screwdriver_rl/ (and therefore becomes src/screwdriver_rl/ after the source-layout migration). Existing-file paths are relative to the repository root.

### 28.1 New foundational files

| File | Responsibility |
|---|---|
| screwdriver_rl/contracts/identifiers.py | Stable IDs and validation |
| screwdriver_rl/contracts/joints.py | JointSchema and semantic vector utilities |
| screwdriver_rl/contracts/state.py | Timed state, effective target, command, acknowledgement |
| screwdriver_rl/contracts/hand.py | HandSpec and HandSimSpec |
| screwdriver_rl/contracts/task.py | TaskSpec and component references |
| screwdriver_rl/contracts/experiment.py | Profiles, presets, ResolvedExperiment |
| screwdriver_rl/contracts/observation.py | Observation layouts and codec protocols |
| screwdriver_rl/contracts/conditioning.py | PrivilegedObservationSpec, teacher conditioning, and actor-input graph contracts |
| screwdriver_rl/contracts/action.py | Action layouts and transform protocols |
| screwdriver_rl/contracts/artifact.py | PolicyManifest and immutable package-content models |
| screwdriver_rl/contracts/deployment.py | DeploymentBinding, LaunchConfig, timing, lease, and safety response models |
| screwdriver_rl/contracts/attestation.py | Detached evidence/promotion envelopes and chain validation |
| screwdriver_rl/contracts/trust.py | Issuer/key scope, rotation, revocation, and trust-policy models |
| screwdriver_rl/contracts/metrics.py | Metric and evaluation schemas |
| screwdriver_rl/experiments/registry.py | Typed registries and plugins |
| screwdriver_rl/experiments/resolver.py | Override-before-derive resolution |
| screwdriver_rl/experiments/validation.py | Cross-layer validation |
| screwdriver_rl/experiments/builtins.py | Current experiments and legacy IDs |

### 28.2 Hand files

| File | Responsibility |
|---|---|
| screwdriver_rl/hands/allegro_v4/spec.py | Single Allegro kinematic source of truth |
| screwdriver_rl/hands/allegro_v4/sim.py | Isaac articulation factory |
| screwdriver_rl/hands/allegro_v4/profiles.py | Task-hand grasps and tuning |
| screwdriver_rl/hands/linker_l20/spec.py | Single Linker kinematic source of truth |
| screwdriver_rl/hands/linker_l20/sim.py | Isaac articulation factory |
| screwdriver_rl/hands/linker_l20/profiles.py | Linker task-hand profiles |

### 28.3 Generic task files

| File | Responsibility |
|---|---|
| task_families/screwdriver_rotation/spec.py | Generic task definition |
| task_families/screwdriver_rotation/object_model.py | Screwdriver asset/physics factory |
| task_families/screwdriver_rotation/contact.py | Distance and force contact providers |
| task_families/screwdriver_rotation/reward.py | Reward strategies |
| task_families/screwdriver_rotation/reset.py | Reset strategies |
| task_families/screwdriver_rotation/observation.py | Codec configurations |
| task_families/screwdriver_rotation/evaluation.py | Task evaluator |
| task_families/inhand_rotation/spec.py | Generic task definition |
| task_families/inhand_rotation/object_model.py | Mixed object library |
| task_families/inhand_rotation/reset.py | Cache/canonical resets |
| task_families/inhand_rotation/grasp_cache.py | Versioned cache schema |
| task_families/inhand_rotation/reward.py | In-hand reward |
| task_families/inhand_rotation/observation.py | Stacked/scaled codec |
| task_families/inhand_rotation/evaluation.py | In-hand evaluator |

### 28.4 Shared simulator files

| File | Responsibility |
|---|---|
| sim/isaac/hand_runtime.py | Joint/body resolution, coupling, targets, limits |
| sim/isaac/mount.py | Fixed-base/arm-mounted root and reset ownership |
| sim/isaac/scene_builder.py | Common scene assembly |
| sim/isaac/collision.py | Collision exclusions |
| sim/isaac/direct_env.py | Composed task lifecycle |
| sim/isaac/sensors.py | Contact provider helpers |
| sim/isaac/app.py | Explicit simulator lifetime |

### 28.5 Runtime and hardware files

| File | Responsibility |
|---|---|
| runtime/policy_session.py | Exact environment-free inference |
| runtime/policy_snapshot.py | Deterministic state/effective-target temporal join |
| runtime/scheduler.py | Fixed-rate scheduling |
| runtime/safety.py | Static and per-cycle safety |
| runtime/ownership.py | ARM/HAND leases and epochs |
| runtime/arbitration.py | Single-authority command selection for arm and hand |
| runtime/handoff.py | Mixed-control state machine |
| runtime/arm_hold.py | ArmHoldController and controller-manager adapter contracts |
| runtime/watchdog.py | Crash-independent watchdog configuration/health integration |
| runtime/telemetry.py | Structured events and sinks |
| runtime/replay.py | Deterministic trace replay |
| hardware/base.py | Driver/adapter/transport contracts |
| hardware/driver.py | Adapter/calibration/transport/lease orchestration |
| hardware/arm_gateway.py | Exclusive external arm/controller-manager command boundary |
| hardware/calibration.py | Immutable calibration models |
| hardware/linker_l20/adapter_left.py | Semantic Linker left mapping |
| hardware/linker_l20/transport_can.py | Direct CAN/SDK |
| hardware/linker_l20/transport_ros1.py | ROS1 bridge |

### 28.6 Training and evaluator files

| File | Responsibility |
|---|---|
| training/runner.py | Shared training orchestration |
| training/rl_games.py | RL-Games adapter |
| training/stage1.py | Resumable Stage 1 |
| training/stage2.py | Full-corpus chunked collection and resumable Stage 2 |
| training/stage2_streaming.py | Optional later versioned sampled/interleaved mode |
| training/checkpoints.py | Internal checkpoint schemas |
| training/artifact_export.py | Candidate policy package creation |
| evaluation/evaluator.py | Task evaluator runner |
| evaluation/gates.py | Versioned gate profiles |
| evaluation/report.py | JSON and human rendering |

### 28.7 Existing files and transition

| Existing file | Migration |
|---|---|
| tasks/base/screwdriver_rotation_env.py | Rename generic hand attribute; delegate to HandRuntime; later compatibility wrapper |
| tasks/base/screwdriver_rotation_env_cfg.py | Delegate to resolved config; stop owning repo-relative resources |
| tasks/allegro/screwdriver_rotation_env.py | Remove copied maps; bind Allegro profile |
| tasks/allegro/screwdriver_rotation_env_cfg.py | Move hand facts and grasp/profile data |
| tasks/allegro/screwdriver_rotation_4f_env_cfg.py | Convert to profile/preset |
| tasks/linker_l20/screwdriver_rotation_env.py | Move sensors/reward into strategies; remove maps |
| tasks/linker_l20/screwdriver_rotation_env_cfg.py | Split hand, object physics, grasp, reward profile |
| tasks/linker_l20/screwdriver_rotation_top_grasp_env_cfg.py | Convert to GraspPreset |
| tasks/linker_l20/screwdriver_rotation_dr_env_cfg.py | Convert to experiment override/profile |
| tasks/linker_l20/inhand_rotation_env.py | Move task lifecycle to generic in-hand task |
| tasks/linker_l20/inhand_rotation_env_cfg.py | Split object, hand profile, cache, backend config |
| tasks/linker_l20/inhand_grasp_gen_env.py | Make collection a generic workflow |
| deploy/policy.py | Compatibility facade over PolicySession |
| deploy/linker_sdk_map.py | Convert global functions/state to immutable adapter instance |
| deploy/deploy_linker.py | Thin CLI composition or compatibility wrapper |
| train.py/play.py/eval.py | Thin CLI wrappers |
| utils/logging.py | Human sink over structured task metrics |
| agent YAML files | Base templates with derived dimensions |
| linker_calib.json (repository root) | Becomes a versioned per-instance HandCalibration record (Section 15); the root copy is a development fixture until migrated |
| linker_calib_absolute.json (repository root) | Same migration as linker_calib.json |
| pyproject.toml | Backend, resources, dependencies, scripts, plugins |

### 28.8 Tools

All tools should:

- accept experiment/profile IDs rather than hardcoded task constants;
- use registries and AssetResolver;
- operate on semantic joints;
- write manifests and provenance;
- avoid calling private environment rebuild methods;
- expose strict CLI options.

---

## 29. Phased implementation plan

Each phase has a narrow purpose and an explicit exit gate. A phase is not complete merely because code exists.

Gate semantics (revision 2): every work item has exactly one owning phase, and a phase's Backlog gate lists only the items it owns; items required from earlier phases appear as prerequisites. Revision 2 removed the duplicate gate listings present in revision 1 (FOUND-009, PKG-005, HW-001, SPEC-012, TEST-004, HANDOFF-001, and the Phase 8/9 re-lists). An ID appearing in two Backlog gates is a documentation defect.

Each phase also names its verification environment per Section 2.4. The partition below is the per-item authority for what runs on the macOS machine versus the CUDA/Isaac rig.

### Execution-environment partition (Part I / Part II)

Part I is everything implementable and verifiable on the macOS development machine (ENV-A). Part II is everything whose verification requires the Ubuntu CUDA/Isaac rig (ENV-B) or the hardware station (ENV-C). Items labeled A→B or A→C are developed and unit-tested in Part I against committed fixtures and then carry one named Part II verification step; they appear in both lists deliberately, and such an item is not done until both sides are.

Part I — implement and verify on the macOS machine (ENV-A):

- Foundation: FOUND-001, FOUND-002, FOUND-003, FOUND-004, FOUND-005, FOUND-007, FOUND-009, FOUND-010, FOUND-011; the GoldenTraceV1 schema and synthetic fixtures of FOUND-008.
- Contracts and specs: SPEC-001 through SPEC-010, SPEC-012 (contract side), SPEC-013, SPEC-014; TEST-007.
- Codecs, action, session (fixture-verified side): OBS-001, OBS-002, OBS-003, OBS-004, OBS-005, OBS-006, OBS-008, OBS-009; the OBS-010 replay harness; OBS-011 kernels; RUN-001; TEST-003.
- Artifacts: ART-001 through ART-007, ART-009, ART-011, ART-012 (ART-006 and ART-008 each carry a Part II round-trip/evidence step).
- Runtime and safety logic with fakes: RUN-002, RUN-003, RUN-004, RUN-005; SAFE-001 through SAFE-004; HANDOFF-001; the HANDOFF-002/004/006/007 state-machine and blending logic; HW-001; HW-002/HW-003 calibration models and mapping math; HW-006 fake/replay/fault transports; TEST-005 core suite.
- Training and evaluation logic: FOUND-006 import shims; the TRAIN-001/TRAIN-002 API side; TRAIN-005/006/007 collector logic against synthetic CPU fixtures; EVAL-001; EVAL-005.
- Packaging, data, docs: PKG-001 through PKG-006; DATA-001, DATA-003; DOC-001 through DOC-004; TEST-001.

Part II — requires the Ubuntu CUDA/Isaac rig (ENV-B):

- Baseline capture: FOUND-008 golden-trace recording; FOUND-012 throughput and Stage-2 wall-time baseline.
- Live parity: OBS-007; the OBS-010 simulator bridge; final live-trace verification of OBS-002/003/004/005/008 and OBS-011; SPEC-012 exercised in Isaac; TEST-002 full coverage.
- Simulator composition: SIM-001 through SIM-012; TEST-006.
- Training and evaluation executions: TRAIN-003, TRAIN-004, TRAIN-009; TRAIN-005/006/007 memory-profile acceptance on the named 2,048/8,192-environment profiles; TRAIN-008 (deferred); EVAL-002/EVAL-003 live metrics; EVAL-004; EVAL-006; the ART-006 real-run round-trip; ART-008 evaluation evidence; HANDOFF-003/HANDOFF-004 training-data inputs; HANDOFF-008, HANDOFF-009; PKG-007 lock evidence and the PKG-006 doctor-on-rig check; DATA-002 store sync; DATA-004; the HW-008 retraining side.

Part II — requires the hardware station (ENV-C):

- HW-004 and HW-005 live buses; HW-007 SDK conformance against the device; HW-009 and HW-010 live exclusivity; SAFE-002 live validation; SAFE-005 process-kill and watchdog drills; TEST-005 station-side drills; HANDOFF-005 and HANDOFF-007 live behavior; HANDOFF-010 HIL; HW-008 physical verification; Phase 9 stages 4–11.

Practical cadence: keep a queued list of pending ENV-B/ENV-C verifications and burn it down in scheduled rig/station sessions; never block ENV-A development on rig access.

### Phase 0 — Baseline, inventory, and guardrails

Goal:

- capture current behavior before moving ownership boundaries.

Environment: ENV-A except FOUND-008 trace capture and FOUND-012, which form the first ENV-B rig session.

Work:

0. Preserve the review baseline first: commit this plan and the README link (both currently unmanaged by git); write the Section 40.0 supplement manifest and durably back up the untracked grasp caches and reviewed artifacts it hashes — the baseline digest depends on files git does not protect; mark docs/stage2-deployability-plan.md and docs/linker_l20_inhand_gpu_handoff.md as superseded for planning purposes (their technical content remains valid history).
1. Create a machine-readable inventory of:
   - every Gym/task ID;
   - hand and task family;
   - action order;
   - policy observation layout;
   - privileged layout;
   - actor proprio layout;
   - control period;
   - curriculum;
   - domain randomization;
   - grasp/reset profile;
   - evaluator metrics;
   - current deployment status.
2. Record current URDF hashes and joint/body inventories.
3. Implement the minimal JointSchema/SemanticJointVector identity contract needed to label every trace unambiguously.
4. Record the reviewed Git SHA, dirty-tree flag/diff hash, inventory path/hash, environment versions, and exact file list in a machine-readable baseline manifest.
5. Define GoldenTraceV1 schema and synthetic ENV-A fixtures; record the deterministic real-environment traces (observation noise and domain randomization disabled, seeds fixed) during the first ENV-B rig session.
6. Add expected-failing test reproducing the Linker in-hand 96-versus-32 deployment error.
7. Add tests proving missing normalizer and wrong same-size hand are currently accepted; mark as expected failures until corrected.
8. Add a support matrix to user-facing documentation.
9. Mark only Linker-left mounted-screwdriver deployment as currently validated.
10. Add a dedicated live-deployment containment gate that rejects current Linker in-hand, right-side mapping, and policies whose training/reachability limits are unresolved; do not claim physical-limit reconciliation yet.
11. Fix the invalid build backend and perform the src/ package-layout migration.
12. Add the initial compatibility manifest, test/developer dependency lock, packaged agent resources, and clean-wheel CI.
13. Run the trained-artifact value inventory (FOUND-011): enumerate which checkpoints and bundles must survive migration byte-exactly and declare everything else disposable. This decision scopes OBS-004, ART-007, and ADR-020. (The license inventory formerly listed here is owned by Phase 6 as PKG-005.)
14. Establish minimal CPU CI with build and unit tests.
15. During the first ENV-B rig session, record the training-throughput and Stage-2 wall-time baseline (FOUND-012) that the Phase 4 and Phase 5 performance exit criteria compare against.

Backlog gate: FOUND-001, FOUND-002, FOUND-003, FOUND-004, FOUND-005, FOUND-007, FOUND-008, FOUND-009, FOUND-010, FOUND-011, FOUND-012, SPEC-001, HW-001, PKG-001, PKG-003, PKG-004, and TEST-001. PKG-005 is owned by Phase 6. Each implementation PR must name the individual IDs for its numbered steps; this phase-level list is not permission to bundle them all.

Primary files:

- pyproject.toml;
- README.md;
- docs/DEPLOY.md;
- tests/test_deploy_policy_bundle.py;
- tests/test_algo.py;
- new inventory/contract tests.

Exit criteria:

- current IDs and tensor contracts are captured;
- all Python files named in the reviewed working-tree inventory parse;
- minimal CI runs;
- clean wheel installs and packaged resources load outside the checkout;
- unsafe unsupported combinations fail clearly;
- the plan, README link, and Section 40.0 baseline supplement inputs are committed or durably preserved;
- trained-artifact preservation targets are enumerated (FOUND-011);
- the two ENV-B items (FOUND-008 capture, FOUND-012) are complete or explicitly queued for the first rig session; queuing them does not block Phase 1 ENV-A work;
- no policy behavior has changed.

Rollback:

- documentation/test-only changes are reversible;
- CLI containment may have an explicit development override but no live default.

Retraining:

- none.

### Phase 1 — Correct deployment inference parity

Goal:

- make actual environment-free inference match training for every currently intended deployable task.

Work:

1. Implement shared scalar/batched PolicySample, ProprioCodec/ActionTransform kernels, ActorInputAssembler, ConditioningSpec, PrivilegedObservationSpec, and TeacherConditioningProvider protocols.
2. Implement:
   - Linker screwdriver raw one-frame codec;
   - Linker in-hand scaled three-frame actor codec with 30-frame adapter history;
   - Allegro legacy codec as needed for artifact compatibility.
3. Implement shared ActionTransform with explicit timing and effective-target state.
4. Define exact history capture phase.
5. Decide how to handle the current screwdriver training/runtime target-phase discrepancy:
   - option A: preserve a named legacy codec for existing weights;
   - option B: correct and retrain.
6. Make DeployPolicy or its replacement derive actor input from the codec.
7. Make eval deployment mode call the actual environment-free PolicySession.
8. Enforce 10 Hz versus 20 Hz in the resolved runtime contract; artifact serialization follows in Phase 3.
9. Require exact declared normalization state; support explicit normalization mode none without inventing a normalizer.
10. Remove silent frame padding/truncation.
11. Add 100-tick trace parity for screwdriver and in-hand.

Environment: ENV-A against synthetic and recorded fixtures; OBS-007 and the OBS-010 simulator bridge, plus final live-trace parity, are ENV-B.

Backlog gate: OBS-001 through OBS-010, RUN-001, and TEST-003. Prerequisite: SPEC-001 (Phase 0). OBS-004 is conditional on the FOUND-011 inventory finding an Allegro checkpoint worth preserving.

Primary files:

- new contracts/observation.py;
- new contracts/action.py;
- new runtime/policy_session.py;
- deploy/policy.py;
- base screwdriver environment;
- inhand_rotation_env.py;
- eval.py;
- play.py;
- tests/test_codec_parity.py;
- tests/test_inhand_deploy_contract.py.

Exit criteria:

- in-hand PolicySession consumes 96 actor-proprio values and a 30-by-32 adapter history;
- scaling matches simulation exactly;
- simulator and runtime actions/targets match golden traces;
- eval uses PolicySession;
- missing declared normalizer is fatal and undeclared normalization is rejected;
- wrong dimensions are fatal.

Rollback:

- existing environment class paths remain;
- a named LegacyScrewdriverCodecV0 may support exact old weights offline.

Retraining:

- not required if the codec exactly reconstructs training behavior;
- required if history phase or preprocessing is intentionally changed.

### Phase 2 — Domain specs and strict experiment resolution

Goal:

- create one simulator-neutral source of truth for hand, task, profile, and dimensions.

Work:

1. Extend the Phase-0 JointSchema into typed HandSpec records for Allegro v4 right and Linker L20 left.
2. Extract all joint names/coupling/intrinsic limits into HandSpec and simulator bodies/collision data into HandSimSpec.
3. Implement TaskSpec for mounted screwdriver and in-hand rotation.
4. Implement TaskHandProfiles, typed initialization sources, privileged/teacher contracts, and ArmIntegrationProfiles for all current variants.
5. Implement typed initialization sources, using GraspPreset for nominal grasps and a versioned cache/distribution for in-hand resets:
   - Allegro 3F;
   - Allegro 4F;
   - Linker side;
   - Linker top;
   - Linker in-hand palm-up cache/distribution.
6. Implement ExperimentSpec for every current Gym ID.
7. Implement resolver with strict overrides-before-derive.
8. Derive observation/action/network dimensions.
9. Implement registry validation.
10. Make current environments consume specs through compatibility accessors without changing behavior.
11. Defer external plugin loading (SPEC-011) to Phase 10; this phase adds no plugin machinery.
12. Add a fixture third-hand conformance package, registered through the ordinary in-repo registries, to challenge the abstractions early.

Environment: ENV-A; the no-behavior-change claim is re-verified by the ENV-B Isaac smoke.

Backlog gate: SPEC-002 through SPEC-010 and SPEC-012, plus TEST-007. SPEC-011 is owned by Phase 10 with ADR-025/ADR-026. Full simulator/runtime frame/clock TEST-002 waits for Phases 4 and 7; TEST-004 sits on the deferred trust track (Section 31.11).

Primary files:

- contracts and experiments packages;
- hands/allegro_v4;
- hands/linker_l20;
- current configs as wrappers;
- registry tests.

Exit criteria:

- every built-in experiment resolves without Isaac;
- identical resolved inputs yield identical policy-contract and experiment fingerprints, while run-only choices do not change the policy-contract fingerprint;
- unknown override fails;
- each current Gym ID maps to exactly one resolved experiment; multiple legacy IDs may be explicit aliases of the same ExperimentPreset;
- no duplicated Linker coupling table remains outside compatibility-generated views;
- dimensions are derived.

Rollback:

- compatibility classes can still expose old constants generated from HandSpec.

Retraining:

- none; this phase is behavior-preserving extraction.

### Phase 3 — Versioned immutable policy artifacts

Goal:

- eliminate semantic ambiguity and mutable deployment files.

Work:

1. Implement PolicyManifestV1.
2. Separate JSON metadata, ExecutionGraphV1, conditional component tensors, and numerical conformance fixtures.
3. Require hand/model/side/joints/units/codec/action/timing.
4. Add complete provenance.
5. Add strict loader sequence and canonical inference against reference tensors plus ComparisonSpec.
6. Add unique run and candidate directories.
7. Add policy verify command.
8. Add offline legacy migration.
9. Migrate the exporter to the Phase-2 public read-only policy-contract/sample-export interface and remove private introspection.
10. Make Stage 2 round-trip-load the candidate before reporting success.
11. Make export failure fatal by default.
12. Define detached attestation schemas, but defer promotion decisions until Phase 5 evaluators/gates exist.

Environment: ENV-A; the ART-006 round-trip from a real Stage-2 run is ENV-B.

Backlog gate: ART-001 through ART-007, ART-009, ART-011, and ART-012. Prerequisites: FOUND-009 (Phase 0) and SPEC-012 (Phase 2). ART-008 promotion is excluded until Phase 5 evaluators/gates exist; ART-010 sits on the deferred trust track.

Primary files:

- contracts/artifact.py;
- training/artifact_export.py;
- runtime policy loader;
- cli/policy commands;
- train/proprio_adapt integration;
- artifact tests.

Exit criteria:

- same-size wrong-hand artifact rejected before connection;
- missing limits/home or any ExecutionGraph-declared component rejected;
- smoke run cannot overwrite a candidate package;
- modified tensor fails checksum;
- every candidate has source checkpoint and experiment hash;
- legacy artifact requires explicit migration.

Rollback:

- training may continue to save internal legacy checkpoints;
- live runtime accepts only manifested package.

Retraining:

- none for representable legacy weights;
- repackaging and validation required.

### Phase 4 — Shared hand runtime and generic task composition

Goal:

- separate hand mechanics from task semantics and remove task-environment forks.

Work:

1. Rename generic articulation from allegro to hand.
2. Extract HandRuntime:
   - joint/body resolution;
   - semantic ordering;
   - coupling;
   - limit tensors;
   - target writes;
   - collision exclusions;
3. Extract MountStrategy for fixed-base and arm-mounted root/reset ownership.
4. Extract common scene logic.
5. Create ContactProvider strategies.
6. Extract screwdriver RewardModel variants.
7. Extract reset and termination strategies.
8. Convert Linker screwdriver overrides into selected strategies.
9. Convert in-hand environment to a generic task family.
10. Convert grasp generation into a generic workflow.
11. Retain old classes as aliases/wrappers.
12. Construct the Phase-2 fixture third hand against a generic task as an abstraction smoke; reserve real Allegro in-hand support for the explicit Phase-10 work items.

Environment: ENV-B for all verification; kernels and unit scaffolds may be drafted on ENV-A.

Backlog gate: SIM-001 through SIM-010 and TEST-006. SIM-011 is owned by Phase 8 and may begin only after trace parity for the extracted runtime.

Primary files:

- sim/isaac/hand_runtime.py;
- task_families directories;
- current task modules;
- tools.

Exit criteria:

- generic task code contains no hand-specific names or fixed DOF constants;
- no generic self.allegro remains;
- current experiments pass trace parity;
- adding a fixture hand/profile requires no generic environment edit;
- Linker in-hand and screwdriver use the same HandSpec and HandRuntime;
- training throughput for both tasks stays within the recorded budget of the FOUND-012 baseline (proposed default: at least 90 percent of baseline env-steps/sec, accepted or revised by ADR/evidence).

Rollback:

- old class import paths instantiate the new generic environment with the same profile.

Retraining:

- none for behavior-preserving strategy extraction;
- any numerical drift beyond tolerance blocks merge.

### Phase 5 — Training, evaluation, and Stage-2 scalability

Goal:

- make training/evaluation callable, reproducible, resumable, and task-generic.

Work:

1. Move CLI parsing to cli package.
2. Add explicit simulator context.
3. Centralize Isaac/RL-Games compatibility.
4. Build agent config from resolved dimensions.
5. Add public environment export/phase/history interfaces.
6. Implement TaskEvaluator for both task families.
7. Add canonical EvaluationResult JSON and exit codes.
8. Implement matched oracle/adapter gates.
9. Replace Stage-2 list/cat collection with a full-corpus chunked/memory-mapped frozen snapshot that preserves sample/order semantics; keep interleaved sampling as a separate later work item.
10. Add memory estimator and caps.
11. Add exact-versus-warm Stage-1 and Stage-2 resume state.
12. Add separate Stage-1 environment-snapshot and Stage-2 corpus/cursor equivalence tests.
13. Add structured run directory.
14. Implement detached evaluation/promotion attestations and promotion commands now that gate profiles exist.

Environment: mixed — schemas, gate logic, and collector logic against synthetic CPU fixtures on ENV-A; training runs, live evaluators, and the named memory/throughput profiles on ENV-B.

Backlog gate: FOUND-006, TRAIN-001 through TRAIN-007, TRAIN-009, EVAL-001 through EVAL-005, and ART-008. ART-010 and TEST-004 sit on the deferred trust track (Section 31.11). TRAIN-008 interleaved streaming is a behavior-changing post-Phase-5 track and is not required for this phase. The TRAIN-005/006/007 slice may be pulled forward independently of the rest of this phase when a large-environment campaign needs it (Section 2.5).

Primary files:

- training package;
- evaluation package;
- root wrappers;
- algos/proprio_adapt.py;
- agent configs;
- utils/logging.py.

Exit criteria:

- imports do not launch Isaac;
- unknown CLI options fail;
- both tasks emit finite complete JSON;
- Stage-2 memory bounded independently of env-count times rollout-length;
- representative 2,048- and 8,192-environment profiles meet the fixed host/device caps, full-corpus/order rules, and allocator measurements defined in Sections 20.9–20.11;
- interrupted/resumed CPU run matches uninterrupted run;
- Stage-2 collection wall time and training throughput stay within the recorded FOUND-012 budget;
- deploy candidate export uses public contract.

Rollback:

- root commands remain wrappers with existing flags.

Retraining:

- no mandatory retraining for runner extraction;
- Stage-2 algorithm sampling changes require a controlled loss/action comparison before adoption.

### Phase 6 — Packaging, resources, and data lifecycle

Goal:

- make repository components relocatable and reproducible.

Work:

1. Add simulator/deployment release locks and remaining dependency groups/scripts after the foundation src migration.
2. Implement AssetResolver.
3. Complete agent/resource packaging begun in Phase 0.
4. Decide structural asset package strategy subject to the license inventory.
5. Externalize large grasp caches.
6. Add cache manifests and deterministic seeds.
7. Complete compatibility evidence for the simulator/deployment locks.
8. Add the separately tracked doctor command.
9. Expand clean-wheel CI across optional-dependency boundaries.
10. Generate docs/task table from registry.

Environment: ENV-A except rig-side lock/doctor evidence (ENV-B).

Backlog gate: PKG-002, PKG-005, PKG-006, PKG-007, DATA-001 through DATA-003, and DOC-001. PKG-005 is owned here (moved out of Phase 0). FOUND-005 and the initial PKG-003 test lock were completed in Phase 0.

Exit criteria:

- clean wheel works outside checkout;
- deploy-only environment dry-runs;
- all resources resolve;
- missing assets fail before Isaac;
- caches cannot cross hand/URDF/schema;
- compatibility decision is machine-readable.

Rollback:

- SCREWDRIVER_RL_ASSET_ROOT remains supported during migration.

Retraining:

- none.

### Phase 7 — Deployment decomposition and safety

Goal:

- replace LinkerDeployer monolith with embeddable, testable layers.

Work:

1. Implement immutable calibration.
2. Implement LinkerL20LeftAdapter.
3. Extract CAN and ROS1 transports.
4. Implement fake/replay/fault transports.
5. Implement timed state and acknowledgements.
6. Implement fixed-rate scheduler.
7. Implement SafetySupervisor.
8. Implement structured telemetry.
9. Replace process-signal ownership with stop requests.
10. Enforce support matrix and left-side mapping.
11. Pin corrected SDK.
12. Reconcile training and hardware limits.
13. Implement LeaseAuthority/CommandArbiter integration and opaque driver lease installation.
14. Implement exclusive HandDriver and ArmCommandGateway ownership, authenticated candidate ingress, prepared-native command binding, latest-before buffering, acknowledgement evidence, and explicit shutdown.
15. Implement the crash-independent arm/hand watchdogs and process-kill tests.
16. Implement pre-map/prepared-native/post-map/send/ack safety phases and arm command arbitration.

Environment: logic with fakes on ENV-A; live buses, SDK conformance, and process-kill drills on ENV-C.

Backlog gate: RUN-002 through RUN-005, HW-002 through HW-010, SAFE-001 through SAFE-005, HANDOFF-001, TEST-002, and TEST-005. Prerequisite: HW-001 (Phase 0). TEST-004 sits on the deferred trust track.

Exit criteria:

- two adapter instances with different calibrations do not interact;
- valid-but-stale state trips safety;
- every exception path produces bounded response;
- right hand rejected unless right adapter exists;
- no module-global calibration;
- mapping reports clipping/quantization;
- safety/fault suite passes.

Rollback:

- old deploy CLI may compose new layers behind existing flags.

Retraining:

- required after changing simulated hardware limits or action reachable set;
- not required for pure adapter/transport extraction.

### Phase 8 — Mixed controller and arm-hold integration

Goal:

- implement and validate the complete teleoperation-to-RL-to-teleoperation lifecycle.

Work:

1. Integrate authoritative ARM/HAND leases with control-session IDs, opaque tokens, CAS transfer, heartbeats, and command modes.
2. Implement and crash-test the linearizable inactive-prepare/atomic-activate/commit/recovery transaction.
3. Implement handoff state machine and explicit PolicySession lifecycle placement.
4. Define integration with teleoperation controller targets.
5. Implement RL shadow priming and PolicySnapshotBuilder temporal joins.
6. Implement proprioceptive and task-context start-readiness builders/evaluators.
7. Integrate ArmHoldController over ArmCommandGateway.
8. Implement live-endpoint transition blending synchronized to post-mapping effective targets.
9. Implement hand-back preparation and acknowledgement.
10. Add wrist/task frames and deployment tolerances.
11. Add recorded teleoperation replay.
12. Add arm/wrist perturbation simulation.
13. Run full fault matrix.
14. Define deadman behavior during RL_ACTIVE and every hand-back timeout/fallback.

Environment: state machine and blending with fakes on ENV-A; simulator integration on ENV-B; live drills on ENV-C. This phase must not start before ADR-016 and ADR-017 are accepted (Section 18 status note).

Backlog gate: HANDOFF-002 through HANDOFF-009, HANDOFF-011, and SIM-011. Prerequisites: HANDOFF-001, RUN-001, RUN-004, RUN-005, HW-010, SAFE-002, SAFE-005, SIM-010, and TEST-005 (Phases 1, 4, and 7).

Exit criteria:

- old-epoch commands rejected;
- activation from arbitrary invalid posture rejected;
- first blend and RL targets meet discontinuity/rate limits;
- arm remains within pose/twist/wrench envelope;
- hand-back synchronizes both controllers;
- every fault returns to safe hold or E-stop;
- recovery never resumes active directly: it returns through hand-back/TELEOP_ACTIVE/RL_SHADOW or explicit safety-owned RECOVERY_SHADOW.

Rollback:

- operator can remain in TELEOP_ACTIVE;
- mixed integration is gated behind an explicit deployment binding.

Retraining:

- likely required or at least fine-tuning required after adding realistic handoff histories, wrist perturbations, hardware limits, and compliance.

### Phase 9 — Simulator and hardware rollout

Goal:

- promote the new runtime through progressively higher-risk validation.

Stages:

1. CPU replay.
2. Isaac parity.
3. Isaac mixed-control flow.
4. Live read-only shadow.
5. Arm-hold acquisition/verification with the hand still teleoperated.
6. Independent-watchdog process-loss and competing-publisher rejection drills in a certified safe setup.
7. E-stop, communication-loss, controller-loss, safe-hold-failure, crash-at-transfer-phase, and repeated hand-back drills.
8. Certified commissioning trajectory or mechanically safe dummy-object blend under reduced power; never activate an object-dependent manipulation policy in free air merely as a blend test.
9. Bounded screwdriver trial.
10. Bounded in-hand trial after codec, initialization/reset source, policy, and object-presence validation.
11. Approved policy/deployment-binding promotion attestation and operational runbook publication.

Environment: stages 1–3 on ENV-B; stages 4–11 on ENV-C.

Backlog gate: HANDOFF-010 and DOC-003. Prerequisites: ART-008, SAFE-005, HW-008, and every required evaluation/gate/attestation ID selected by the release record.

Exit criteria:

- every stage stores immutable trace and decision;
- abort criteria defined before each run;
- policy/calibration/hardware identity reconstructable;
- promotion references exact validation evidence.

### Phase 10 — Extensibility proof

Goal:

- prove that the architecture, not merely current code, supports extension.

Proof options:

- make Allegro in-hand genuinely runnable in simulation with a validated grasp/reset source and short task evaluation;
- implement SPEC-011 trusted plugin loading (ADR-025/ADR-026) and optionally graduate the Phase-2 fixture hand into a packaged external conformance example;
- publish the adding-hand guide and complete ADR/experiment-history cleanup from the proven implementation.

Environment: contracts on ENV-A; simulator proof on ENV-B.

Backlog gate: SPEC-011, SPEC-013, SPEC-014, OBS-011, DATA-004, SIM-012, EVAL-006, DOC-002, and DOC-004. Prerequisite: TEST-006 (Phase 4).

Exit criteria:

- no generic task/runtime modifications;
- only spec, profile, sim adapter/assets, and optional hardware adapter added;
- registry, codec, evaluator, and smoke tests pass.

---

## 30. Compatibility and deprecation policy

For at least one migration release:

- preserve all current Gym IDs;
- preserve old environment/config import paths;
- preserve root CLI commands and major flags;
- preserve SCREWDRIVER_RL_ASSET_ROOT;
- preserve existing metric keys where practical;
- preserve known legacy checkpoint loading for matching versioned experiments;
- issue one-time deprecation warnings for direct legacy imports.

Never:

- silently reinterpret an artifact as another hand;
- infer a 16-D hand from dimension alone;
- treat a changed observation/history phase as compatible;
- map right-hand operation through left-hand tables;
- deploy an unmanifested legacy artifact without explicit migration.

### 30.1 Legacy codec strategy

Possible named codecs:

- AllegroScrewdriverLegacyV0;
- LinkerScrewdriverLegacyHistoryPhaseV0;
- LinkerInhandStack3ScaledV1.

Existing weights may run without retraining only when the old contract is reconstructed exactly.

Which legacy codecs are actually built is decided by the FOUND-011 trained-artifact inventory, not by this list. As of the reviewed snapshot, both tasks' deploy.pth files are iteration-1 smoke artifacts (docs/DEPLOY.md), the in-hand policy is mid-retuning and will be retrained under the corrected codec regardless, and no Allegro hardware path exists — so the only plausible byte-exact preservation target is a converged Linker screwdriver latent policy. If FOUND-011 confirms no production-worthy legacy artifact exists, AllegroScrewdriverLegacyV0 and LinkerScrewdriverLegacyHistoryPhaseV0 are dropped, OBS-004 shrinks to sim-eval-only or is deleted, and ADR-020 defaults to corrected-phase-plus-retrain.

### 30.2 Legacy task IDs

Gym aliases point to ExperimentPreset IDs. They contain no duplicated configuration.

### 30.3 Removal criteria

Remove compatibility wrappers only after:

- current production policies are migrated or explicitly archived;
- downstream scripts use new APIs;
- two release notes have announced removal;
- no supported task depends on legacy wrapper behavior.

---

## 31. Detailed work-item backlog

Sizes are relative:

- S: focused change with narrow surface;
- M: cross-module change with several tests;
- L: architectural extraction or runtime feature;
- XL: multi-stage integration requiring simulator/hardware validation.

Every ID has exactly one owning phase (its Backlog gate in Section 29) and one verification environment (the Part I / Part II partition in Section 29; Section 2.4). Deferred items are collected in Section 31.11.

### 31.1 Foundation and containment

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| FOUND-001 | P0 | S | Correct pyproject build backend | none | wheel backend loads |
| FOUND-002 | P0 | S | Add explicit deployment support matrix | none | unsupported in-hand/right live runs reject |
| FOUND-003 | P0 | M | Add current-contract inventory | none | every Gym ID has machine-readable contract |
| FOUND-004 | P0 | M | Add minimal CPU CI | FOUND-001 | build and unit jobs required |
| FOUND-005 | P0 | M | Add initial compatibility manifest | none | supported Python/test/runtime ranges validate as data |
| FOUND-006 | P1 | M | Centralize Isaac/RL-Games imports | FOUND-005 | no duplicated fallback blocks |
| FOUND-007 | P0 | S | Persist reviewed baseline manifest | FOUND-003 | Git/diff/inventory/environment hashes reconstruct review |
| FOUND-008 | P0 | M | Define GoldenTraceV1 schema and fixtures | FOUND-003, SPEC-001 | deterministic traces validate before refactor |
| FOUND-009 | P0 | M | Canonical JSON and content-digest utility | FOUND-001 | cross-platform golden vectors produce identical digests |
| FOUND-010 | P0 | S | Block unresolved live-deployment combinations | FOUND-002 | in-hand/right-side/unresolved-limit live runs fail without claiming reconciliation |
| FOUND-011 | P0 | S | Trained-artifact value inventory | FOUND-003 | checkpoints worth byte-exact preservation are enumerated, all others declared disposable, and the OBS-004/ART-007/ADR-020 scope decision is recorded |
| FOUND-012 | P0 | S | Training-throughput and Stage-2 wall-time baseline (ENV-B) | FOUND-003 | env-steps/sec and collection wall time recorded on the rig for both tasks at reference settings |

### 31.2 Observation and action parity

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| OBS-001 | P0 | L | Define scalar/batched PolicySample, proprio/action kernels, actor input, and privileged/teacher protocols | SPEC-001, FOUND-003 | vector reset/device/dtype and scalar-wrapper tests pass |
| OBS-002 | P0 | M | Linker screwdriver codec | OBS-001 | golden trace matches |
| OBS-003 | P0 | M | Linker in-hand scaled/stacked codec | OBS-001 | 32/96/30x32 contract passes |
| OBS-004 | P1 | M | Allegro legacy codec (conditional on FOUND-011) | OBS-001, FOUND-011 | legacy trace matches, or the item is closed as not needed |
| OBS-005 | P0 | M | Define shared ActionTransform | OBS-001 | sim/runtime target parity |
| OBS-006 | P0 | S | Remove padding/truncation | OBS-002, OBS-003 | mismatch is fatal |
| OBS-007 | P0 | M | Make eval use PolicySession | OBS-002, OBS-003, RUN-001 | no direct actor bypass |
| OBS-008 | P0 | M | Define history phase and legacy policy | OBS-002 | decision recorded and tested |
| OBS-009 | P0 | S | Enforce runtime control period | OBS-001 | 10/20 Hz runtime contracts reject mismatch |
| OBS-010 | P1 | M | Golden trace replay and simulator bridge | FOUND-008, OBS-001 | scalar/batched proprio/action/teacher traces parameterized across built-ins |
| OBS-011 | P1 | M | Allegro in-hand batched/scalar codec | OBS-001, SPEC-002 | reset/trace/dimension parity passes |

### 31.3 Domain specs and experiment resolution

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| SPEC-001 | P0 | M | JointSchema and semantic vector | FOUND-003 | name/order/unit validation |
| SPEC-002 | P1 | M | Allegro HandSpec | SPEC-001 | URDF conformance passes |
| SPEC-003 | P1 | M | Linker HandSpec | SPEC-001 | URDF conformance passes |
| SPEC-004 | P1 | M | Screwdriver TaskSpec | SPEC-001 | no hand-specific names |
| SPEC-005 | P1 | M | In-hand TaskSpec | SPEC-001 | no Linker dependency |
| SPEC-006 | P1 | L | Current TaskHandProfiles | SPEC-002, SPEC-003, SPEC-004, SPEC-005 | all variants represented |
| SPEC-007 | P1 | M | GraspPreset model/migration | SPEC-002, SPEC-003 | no duplicated pose source |
| SPEC-008 | P1 | M | Experiment registry | SPEC-004, SPEC-005, SPEC-006 | IDs unique and listable |
| SPEC-009 | P1 | L | Strict resolver | SPEC-008 | override-before-derive |
| SPEC-010 | P1 | M | Registry-driven Gym aliases | SPEC-009 | all old IDs work |
| SPEC-011 | P1 | M | Opt-in trusted plugin discovery and fixture (owned by Phase 10) | SPEC-008, FOUND-009, ADR-025/026 | allowlist/version/digest checks and frozen registry pass |
| SPEC-012 | P1 | M | Read-only policy-contract and versioned sample-export API | SPEC-009, OBS-005 | current environments export candidates/samples without private introspection |
| SPEC-013 | P1 | M | Allegro in-hand TaskHandProfile and initialization contract | SPEC-002, SPEC-005 | profile resolves without generic edits |
| SPEC-014 | P1 | M | Allegro in-hand ExperimentPreset | SPEC-009, SPEC-013, OBS-011, DATA-004 | experiment resolves and fingerprints |

### 31.4 Policy artifacts

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| ART-001 | P0 | L | PolicyManifestV1 and serialized timing contract | SPEC-001, OBS-001, OBS-009 | schema includes all semantics |
| ART-002 | P0 | M | Strict manifest loader | ART-001 | missing fields fail |
| ART-003 | P0 | M | Exact ExecutionGraph/component loading | ART-002 | missing declared component fails; normalization-none remains valid |
| ART-004 | P0 | M | Content hashes and package-ID scheme | ART-001, FOUND-009 | mutation detected without self-reference |
| ART-005 | P0 | M | Unique run/candidate layout | none | smoke cannot overwrite |
| ART-006 | P1 | L | Candidate exporter | ART-001, SPEC-009, SPEC-012 | round-trip verified through public contract |
| ART-007 | P1 | M | Legacy migration command | ART-002, OBS-002, OBS-003, OBS-004 | ambiguity rejected |
| ART-008 | P1 | M | Evaluation/promotion attestation production | ART-004, ART-012, EVAL-005 | immutable evidence-linked stages |
| ART-009 | P1 | M | Production dirty-source/evidence policy | ART-006 | clean default or canonical patch/untracked source evidence reconstructs candidate |
| ART-010 | P3 | M | Cross-artifact trust/rotation/revocation policy (deferred trust track, Section 31.11) | ART-008, ART-011 | approved source/signature verified |
| ART-011 | P0 | L | DeploymentBinding and LaunchConfig schemas | ART-001, SPEC-003 | immutable binding and secret-free launch separation validate |
| ART-012 | P1 | M | Detached attestation envelope and chain schema | ART-004 | package/evidence/gate/previous-stage digests are non-self-referential |

### 31.5 Simulator composition

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| SIM-001 | P1 | M | Neutral hand attribute migration | SPEC-002, SPEC-003 | no generic self.allegro |
| SIM-002 | P1 | L | HandRuntime extraction | SIM-001 | both tasks share runtime |
| SIM-003 | P1 | M | Collision filtering extraction | SIM-002 | parity passes |
| SIM-004 | P1 | L | ContactProvider strategies | SIM-002 | distance/force selectable |
| SIM-005 | P1 | L | RewardModel strategies | SIM-004 | reward trace parity |
| SIM-006 | P1 | M | ResetStrategy extraction | SIM-002 | reset trace parity |
| SIM-007 | P1 | L | Generic screwdriver task | SIM-004, SIM-005, SIM-006 | old classes thin |
| SIM-008 | P1 | XL | Generic in-hand task | SIM-002, OBS-003 | no Linker-specific lifecycle |
| SIM-009 | P1 | L | Generic grasp collection | SIM-008 | experiment-driven tool |
| SIM-010 | P1 | L | Named wrist/task frames | SPEC-006 | fixed adapter derives pose |
| SIM-011 | P1 | L | Wrist perturbation/compliance mode | SIM-010 | handoff envelope testable |
| SIM-012 | P1 | L | Runnable Allegro in-hand extensibility proof | SPEC-014, OBS-011, DATA-004, SIM-008 | construct/reset/step and short evaluation pass |

### 31.6 Training and evaluation

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| TRAIN-001 | P1 | L | Reusable runner APIs | SPEC-009 | import-safe callable API |
| TRAIN-002 | P1 | M | Derived agent config | SPEC-009 | YAML has no hand widths |
| TRAIN-003 | P1 | M | Public env export contract | SIM-007, SIM-008 | no private introspection |
| TRAIN-004 | P1 | L | Exact-versus-warm Stage-1 checkpointing | TRAIN-001, TRAIN-003 | environment/runner state and lineage semantics persist |
| TRAIN-005 | P1 | XL | Full-corpus chunked Stage-2 collector | TRAIN-001, TRAIN-003 | identical corpus/order with bounded RAM/GPU and no second copy |
| TRAIN-006 | P1 | L | Complete Stage-2 resume | TRAIN-005 | equivalence test |
| TRAIN-007 | P1 | M | Memory estimator/caps | TRAIN-005 | prelaunch failure |
| TRAIN-008 | P2 | XL | Versioned interleaved/sampled Stage-2 mode | TRAIN-005, TRAIN-006, EVAL-004 | separately approved behavior change meets parity gates |
| TRAIN-009 | P1 | L | Stage-1 exact/warm resume equivalence suite | TRAIN-004 | exact path matches and warm path starts new lineage |
| EVAL-001 | P1 | M | EvaluationResult schema | SPEC-004, SPEC-005 | JSON validates |
| EVAL-002 | P1 | M | Screwdriver evaluator | EVAL-001 | complete finite report |
| EVAL-003 | P1 | M | In-hand evaluator | EVAL-001 | complete finite report |
| EVAL-004 | P1 | M | Oracle/adapter comparison | OBS-007, EVAL-001 | matched-seed degradation |
| EVAL-005 | P1 | M | Deterministic gate profiles/exit codes | EVAL-002, EVAL-003, EVAL-004 | seed/bootstrap/comparator/boundary fields reproduce machine decision |
| EVAL-006 | P1 | M | Allegro in-hand short evaluator gate | EVAL-003, SIM-012 | physical metrics finite and threshold decision reproducible |

### 31.7 Runtime, hardware, and safety

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| RUN-001 | P0 | L | PolicySession | OBS-001, OBS-005 | prime/preview/step works |
| RUN-002 | P1 | M | Timed state/command contracts | SPEC-001 | freshness/lease fields |
| RUN-003 | P1 | L | Fixed-rate scheduler | RUN-002, ART-011 | fake-clock timing/deadline tests |
| RUN-004 | P1 | L | Structured telemetry/replay | RUN-002 | nonblocking trace |
| RUN-005 | P0 | L | PolicySnapshotBuilder temporal join | RUN-002, HW-009 | state pairs with interval-correct effective target without concurrent mutation |
| HW-001 | P0 | M | Remove right-side false support | none | right rejected |
| HW-002 | P1 | L | Immutable separate state/command calibration | SPEC-003 | per-instance channel maps and evidence do not alias |
| HW-003 | P1 | L | Pure Linker left adapter and mapping preview | HW-002, SPEC-003 | semantic/native round-trip and clipping diagnostics validate |
| HW-004 | P1 | M | CAN transport extraction | RUN-002 | timestamped state |
| HW-005 | P1 | M | ROS1 transport extraction | RUN-002 | timestamped state |
| HW-006 | P1 | M | Fake/replay/fault transports | RUN-002 | fault suite support |
| HW-007 | P1 | M | Pin corrected SDK | none | revision reproducible |
| HW-008 | P0 | L | Reconcile physical joint limits | SPEC-003 | sim/hardware align |
| HW-009 | P0 | L | Exclusive HandDriver, prepared-native send, and explicit shutdown | HW-003, HW-004, HW-005, HW-006, HANDOFF-001 | only gateway owns I/O; prepared bytes/lease/ack/safe shutdown pass |
| HW-010 | P0 | XL | ArmCommandGateway/controller-manager adapter | RUN-002, HANDOFF-001 | exclusive arm state/command/ack/mode/safety/watchdog contract passes |
| SAFE-001 | P0 | L | SafetySupervisor static checks | ART-002, ART-011, HW-003 | fail-closed preflight |
| SAFE-002 | P0 | XL | Per-cycle staged safety checks | RUN-003, HW-004, HW-005, HW-006, HW-009, HW-010 | pre-map/prepared/post-map/send/ack arm/hand fault matrix passes |
| SAFE-003 | P1 | M | Named stop responses | SAFE-002 | deployment-binding-defined behavior |
| SAFE-004 | P1 | M | Fault latching/reporting | SAFE-002 | explicit recovery |
| SAFE-005 | P0 | L | Independent arm/hand watchdog health and response | HW-004, HW-005, HW-006, HW-010, SAFE-003 | armed health plus process-kill expiry produce bounded verified response |

### 31.8 Mixed-controller integration

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| HANDOFF-001 | P0 | L | LeaseAuthority, exclusive arbiter, and linearizable transfer | RUN-002 | session/capability/CAS/atomic activation/crash recovery pass |
| HANDOFF-002 | P0 | XL | Handoff state machine | HANDOFF-001, SAFE-002 | all transitions tested |
| HANDOFF-003 | P0 | L | RL shadow priming | RUN-001, HANDOFF-002 | exact history from teleop |
| HANDOFF-004 | P0 | L | Start-envelope model | SPEC-006, EVAL-001 | actionable readiness |
| HANDOFF-005 | P0 | XL | ArmHoldController integration | HANDOFF-002, HANDOFF-011, HW-010, SIM-010 | stable exclusive ARM lease |
| HANDOFF-006 | P0 | L | Effective-target synchronized live-endpoint blend | HANDOFF-002, RUN-001, OBS-005 | bounded transition |
| HANDOFF-007 | P0 | L | Safe hand-back | HANDOFF-002 | teleop acknowledges sync |
| HANDOFF-008 | P1 | L | Recorded teleoperation replay | RUN-004 | sim/fake-driver coverage |
| HANDOFF-009 | P1 | XL | Full simulator integration | SIM-011, RUN-005, HANDOFF-003, HANDOFF-004, HANDOFF-005, HANDOFF-006, HANDOFF-007, HANDOFF-008 | end-to-end flow |
| HANDOFF-010 | P0 | XL | HIL rollout | HANDOFF-009, SAFE-005, HW-008, ART-008, TEST-005 | bounded evidence including exclusivity/process-loss/transfer-crash drills and physical-limit compatibility |
| HANDOFF-011 | P0 | M | ArmHoldController and controller-manager adapter contract | HANDOFF-001, SIM-010 | takeover/release/mode/heartbeat semantics tested |

### 31.9 Packaging, data, and documentation

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| PKG-001 | P0 | M | Package agent resources | FOUND-001 | wheel loads YAML |
| PKG-002 | P1 | L | AssetResolver | FOUND-001 | no repo-relative path |
| PKG-003 | P0 | M | Initial test/developer extras and lock | FOUND-005 | clean CPU test setup reproduces |
| PKG-004 | P0 | L | src layout | PKG-001 | tests use installed wheel |
| PKG-005 | P0 | M | License and redistribution inventory | FOUND-001 | unknown-rights assets block publication and notices ship |
| PKG-006 | P1 | M | doctor command | FOUND-005, PKG-007 | machine-readable installed compatibility decision |
| PKG-007 | P1 | L | Simulator/deployment release locks | FOUND-005, PKG-003 | clean supported sim/deploy setups reproduce |
| DATA-001 | P1 | M | Cache manifest | SPEC-001 | compatibility validated |
| DATA-002 | P2 | L | External cache store | DATA-001 | ordinary Git reduced |
| DATA-003 | P1 | S | Seed all generators | DATA-001 | provenance records seed |
| DATA-004 | P1 | L | Validated Allegro in-hand initialization/reset source | DATA-001, SPEC-013 | schema/asset/provenance checks and reset statistics pass |
| DOC-001 | P1 | M | Generate task catalog | SPEC-008 | README matches registry |
| DOC-002 | P1 | M | Adding-hand guide | SPEC-011 | fixture proof |
| DOC-003 | P1 | M | Artifact/runtime runbooks | ART-008, HANDOFF-002 | operational lifecycle documented |
| DOC-004 | P2 | M | ADR and experiment-history cleanup | DOC-001, DOC-003, ART-008 | rationale and released decisions centralized |

### 31.10 Verification infrastructure

| ID | Priority | Size | Work item | Dependencies | Done when |
|---|---:|---:|---|---|---|
| TEST-001 | P0 | M | Import/CLI/optional-dependency boundary suite | FOUND-001, PKG-003, PKG-004 | import is inert, unknown args fail, pip check passes |
| TEST-002 | P0 | M | Frame, unit, clock, and quaternion conformance | SPEC-001, SIM-010, RUN-002 | conversions and uncertainty rules pass |
| TEST-003 | P0 | M | Session exactly-once/lifecycle suite | RUN-001, OBS-001 | duplicates/reentrancy/close behavior pass |
| TEST-004 | P3 | L | Trust, signature, rotation, and revocation suite (deferred trust track, Section 31.11) | ART-010, PKG-005 | package/binding/calibration/gate/plugin trust failures are rejected |
| TEST-005 | P0 | XL | Exclusive-gateway, lease-phase crash, process-death, and watchdog suite | HANDOFF-001, HW-009, HW-010, SAFE-005 | bypass/copy/race/restart/SIGKILL/fallback evidence passes |
| TEST-006 | P1 | L | Required path-filtered Isaac parity CI | SIM-002, FOUND-004 | affected simulator changes cannot merge without parity |
| TEST-007 | P1 | M | Pure frame/unit/quaternion contract tests | SPEC-001, SPEC-006 | simulator-free direction/unit golden vectors pass |

### 31.11 Deferred reference-design track

These items are design-complete but unscheduled. Do not build them speculatively; activate them on the named trigger (Section 2.5) and re-review the design at activation time.

| Deferred item | Where specified | Trigger to activate |
|---|---|---|
| ART-010 cross-artifact trust/rotation/revocation | Sections 11.4, 12 | artifacts cross a machine or organization trust boundary |
| TEST-004 trust/signature suite | Section 25.8 | with ART-010 |
| AckLevel plumbing beyond SENT_TO_BUS plus inferred-applied | Section 9.2 | a transport that can evidence device/servo acknowledgements |
| Multi-host clock-domain deadline translation | Section 8.14 | control components split across hosts |
| External trusted plugin loading (SPEC-011 beyond the in-repo fixture) | Sections 23.6, 27.6, 36.13 | a real third-party hand/task package (Phase 10) |
| TRAIN-008 interleaved Stage-2 | Sections 20.7–20.8 | evidence that fixed-corpus training limits adapter quality |
| Full arm simulation modes 4–5 | Section 10.7 | arm vendor selected (ADR-016) |

---

## 32. Parallel workstreams and ownership

Reality check (revision 2): this repository currently has one maintainer working with coding agents. The workstreams below are sequencing lanes and review boundaries, not staffed teams; run them serially or at most two at a time, bounded by review bandwidth, and treat the labels as the order in which ownership would be delegated if collaborators join.

Suggested teams or ownership areas:

### Workstream A — Contracts and simulator architecture

Owns:

- HandSpec/TaskSpec/profiles;
- resolver;
- codecs;
- HandRuntime;
- generic tasks.

### Workstream B — Training, evaluation, and artifacts

Owns:

- runner APIs;
- Stage-2 full-corpus chunked frozen-snapshot collection, versioned optional streaming, and resume;
- evaluators;
- policy package exporter;
- promotion gates.

### Workstream C — Runtime, hardware, and safety

Owns:

- PolicySession;
- scheduler;
- adapters/transports/calibration;
- HandDriver, CommandArbiter, LeaseAuthority integration, and independent watchdog;
- SafetySupervisor;
- telemetry/replay.

### Workstream D — Mixed-controller integration

Owns:

- ownership leases;
- teleoperation interface;
- arm hold;
- state machine;
- start envelope;
- HIL rollout.

### Workstream E — Platform

Owns:

- packaging;
- assets/data;
- environment locks;
- CI;
- generated documentation.

### Dependency guidance

- A and E can begin immediately.
- B begins after minimal spec/codec contracts stabilize.
- C begins PolicySession immediately but hardware binding depends on artifact and joint schemas.
- D depends on PolicySession, timed driver, safety, and wrist/task frame contracts.
- HIL begins only after simulator/fault gates.

Avoid parallel edits to the current large environment files until HandSpec and codec interfaces are reviewed. Otherwise merge conflicts will obscure behavioral changes.

---

## 33. Top-level acceptance criteria

The modernization is complete only when all of the following hold.

### 33.1 Maintainability

- Generic task code contains no Allegro or Linker class names.
- Generic runtime contains no Linker slot assumptions.
- No generic module uses fixed 16, 32, 39, 51, 96, or 105 dimensions without deriving them.
- No generic articulation is named allegro.
- Joint/coupling metadata has one source per hand.
- Task-specific initialization/reset sources have one typed owner per task-hand profile.
- Current environment classes are thin compatibility wrappers or removed after deprecation.

### 33.2 Extensibility

- A third simulated hand requires HandSpec, sim asset/factory, profiles, and tests only.
- A new task-hand combination requires a profile and typed initialization/cache data, not an environment fork.
- A new transport implements edge interfaces only.
- A third-party fixture plugin passes registry and smoke tests without editing core registration.

### 33.3 Training/deployment parity

- Every supported experiment has a shared codec and action transform.
- Simulator and environment-free traces match within declared tolerance.
- Batched vectorized kernels and scalar PolicySession share semantics, including per-environment resets.
- Privileged/teacher conditioning contracts are typed, fingerprinted, and separately traced.
- Linker in-hand uses scaled q, 96 actor proprio, 30-by-32 adapter history, and 20 Hz runtime.
- Declared ExecutionGraph normalization/components load exactly; normalization mode none is explicit.
- Evaluation uses actual PolicySession.

### 33.4 Artifact safety

- Wrong same-size hand rejected.
- Wrong side rejected.
- Wrong joint order/unit rejected.
- Wrong control period rejected.
- Missing home/bounds or any ExecutionGraph-declared preprocessing component rejected.
- Artifact and tensor hashes verified.
- Production package immutable and traceable.
- Validation/promotion attestations are detached, immutable, hash-chained to exact evidence, and signed once the deferred trust track is activated; until then they are content-addressed and access-controlled.
- Smoke runs cannot replace approved packages.

### 33.5 Handoff

- Policy primes from measured q and prior effective target.
- Shadow mode fills exact history without sending.
- ARM and HAND have separate authoritative session-bound leases, opaque tokens, modes, and epochs.
- First transition target satisfies configured discontinuity/rate/acceleration.
- Policy tracks interval-correct effective blend targets from PolicySnapshotBuilder.
- Wrist/task envelope remains valid.
- Teleoperation hand-back is synchronized and acknowledged.
- State recovery requires reprime.

### 33.6 Safety

- Stale but valid state detected.
- Every exception path tested.
- Following error, limits, rate, temperature/fault, deadline, lease, and wrist guards implemented as supported.
- Stop response is task/deployment-binding-specific.
- Faults latch and emit structured reasons.
- No unsupported right-hand mapping.
- Independent watchdog reaches the declared safe response after supervisor/driver process loss.
- Command arbitration proves only one authorized arm and hand candidate can enter each safety chain.
- Exclusive arm/hand gateways reject direct competing publishers and send only safety-validated PreparedNativeCommands.
- Stale/disconnected safety signals and missing binding-required collision guards fail closed.

### 33.7 Scalability and reproducibility

- Stage-2 preserves the full corpus/order while RAM/GPU are bounded by staging windows and disk chunks.
- Stage-1 and Stage-2 exact resume paths produce equivalence; warm paths create explicit new lineages.
- Clean wheel installs outside checkout.
- Required resources resolve.
- Environments are pinned and doctor reports compatibility.
- Generated caches have manifests and checksums.
- Training throughput and Stage-2 collection wall time remain within the recorded budgets relative to the FOUND-012 baseline.

### 33.8 Testing

- CPU CI required.
- RL-Games parity required.
- Every experiment has Isaac smoke coverage.
- Critical skips cannot report pass.
- Handoff/fault matrix covered with fake clock/driver.
- Hardware promotion references immutable evidence.

---

## 34. Risk register

| Risk | Impact | Likelihood | Early signal | Mitigation |
|---|---|---:|---|---|
| Behavior changes during extraction | Policy regressions attributed to architecture | High | golden trace changes | phase-0 traces, small PRs, no simultaneous retuning |
| Observation history phase correction changes distribution | Existing adapter no longer valid | High | action/latent trace divergence | named legacy codec or explicit retraining |
| Legacy artifacts lack enough metadata | Unsafe or impossible migration | High | ambiguous 16-D package | reject ambiguity; require manual declaration and full validation |
| Isaac config objects resist late composition | Resolver difficult to map cleanly | Medium | callable config entry points unsupported | keep simulator-neutral resolver and thin static entry functions |
| Python runtime misses deadlines | unsafe latency/jitter | Medium | p99 budget or repeated overruns fail | warmup, fixed scheduler, bounded telemetry, consider isolated process/native loop |
| Vendor state lacks true timestamps | stale data undetected | High | cached state appears fresh | driver sequence/receive metadata, SDK patch, conservative freshness |
| Hardware limits differ from training | systematic saturation and sim-real gap | High | clipping diagnostics | correct URDF, re-solve grasp, retrain |
| Arm hold differs from fixed-base assumption | object/tool instability | High | wrist drift/force excursions | deployment envelope, compliance randomization, staged rollout |
| External SDK fixes are lost | incorrect mapping after reinstall | High | mapping conformance fails | pinned fork/upstream, version check |
| Large cache migration disrupts contributors | broken clones/branches | Medium | missing assets and LFS confusion | separate planned migration, explicit sync/verify, keep fixtures |
| Over-abstraction slows research iteration | research stalls; architecture bypassed or abandoned | High | new ad-hoc task forks; phases overrun during experiments | committed core and triggers (Section 2.5), resting-state invariant, keep contracts concrete, fast local workflow |
| Migration abandoned midway on solo capacity | dual implementations and half-moved code | Medium | phase overruns during research deadlines | every phase exit is a stable resting state; tier-1 core first; phase-sized PRs |
| Plan document drifts from implementation | misleading source of direction | High | merged PRs not reflected here | document frozen as reference at revision 2; live tracking in the work-item tracker; normative changes land as ADRs |
| Safety thresholds poorly tuned | nuisance holds or insufficient protection | Medium | excessive false positives or near misses | shadow telemetry, staged calibration, deployment-binding versioning |
| Two controllers publish concurrently | unpredictable commands | High | lease/ack mismatch | exclusive gateways, authenticated ingress, linearizable leases, competing-publisher HIL test |
| State/target streams join at the wrong interval | temporal policy distribution shift | High | trace mismatch around delayed acknowledgements | scheduler-owned PolicySnapshotBuilder and golden skew/delay traces |
| Log/trace volume affects control | missed deadlines/disk pressure | Medium | queue drops, latency correlation | bounded async queue, rotation, configurable payload |
| CI cannot run Isaac frequently | simulator regressions reach main | Medium | nightly failures after merges | required contract tests, scheduled self-hosted smoke, manual gate before release |

### 34.1 Risk handling rule

Any behavior-changing phase must name:

- policy populations affected;
- whether retraining is required;
- rollback package/runtime version;
- simulator validation;
- HIL validation;
- promotion owner.

---

## 35. Architectural decision records to create

Create ADR files as decisions are accepted.

| ADR | Decision |
|---|---|
| ADR-001 | Semantic joint schemas are mandatory at all cross-layer boundaries |
| ADR-002 | Hand/task/profile composition replaces task-by-hand environment forks |
| ADR-003 | ProprioCodec, ActorInputAssembler, ConditioningSpec, and ActionTransform are shared by simulation and runtime |
| ADR-004 | Policy packages are immutable, manifested, and fail closed |
| ADR-005 | Highest-trust post-mapping effective target, not generated candidate, is policy integration state |
| ADR-006 | ARM and HAND use authoritative session-bound ownership leases with opaque tokens |
| ADR-007 | RL owns fingers only; deterministic controller holds the arm |
| ADR-008 | Command arbitration selects one owner before transition, safety, mapping, and send |
| ADR-009 | Structural assets and generated caches have separate lifecycles |
| ADR-010 | CLIs are adapters over import-safe library APIs |
| ADR-011 | Stage-2 first preserves the full corpus/order with bounded staging memory; any sampled/interleaved mode is separately versioned and resumable |
| ADR-012 | Task evaluation is registry-owned and machine-readable |
| ADR-027 | Joint identity, training limits, calibrated reachability, and deployment caps are separate layers |
| ADR-028 | Model packages and later validation/promotion attestations are separate immutable artifacts |
| ADR-029 | HandAdapter is pure conversion, Transport is raw I/O, and HandDriver orchestrates both |
| ADR-030 | A crash-independent low-level watchdog is mandatory for live control |
| ADR-031 | TaskHandProfile owns the canonical task_from_wrist transform |
| ADR-032 | Task success is a physical evaluator contract independent of reward weights |
| ADR-033 | Scalar live policy and batched simulator kernels share one typed temporal contract |
| ADR-034 | Arm and hand command gateways are exclusive physical ownership boundaries |
| ADR-035 | Post-map safety validates an immutable PreparedNativeCommand sent without remapping |
| ADR-036 | PolicySnapshotBuilder performs scheduler-owned state/effective-target temporal joins |
| ADR-037 | Task defaults, task-hand overrides, experiment overrides, and simulator backend follow one fixed precedence tree |

Each ADR should contain:

- context;
- decision;
- alternatives;
- consequences;
- compatibility effect;
- validation requirement;
- date and owners.

---

## 36. Open decisions

These choices should be resolved before the dependent phase, not guessed inside implementation.

The decision deadline is a merge gate, not a target date. The named owner is a role until a maintainer assigns a person; in the current single-maintainer reality every role resolves to the maintainer, and the labels mark which hat is worn and where delegation would occur first. Accepted decisions require a merged ADR plus the evidence stated below.

| Decision | ADR | Owner role | Status | Decision deadline | Blocked work IDs | Acceptance evidence |
|---|---|---|---|---|---|---|
| Artifact tensor format | ADR-013 | ML platform lead | Open | Before ART-006 starts | ART-006, ART-007 | Round-trip, corruption, untrusted-input, and clean-runtime tests |
| Asset distribution | ADR-014 | Packaging/data lead | Open | Before PKG-002 or DATA-002 starts | PKG-002, DATA-002 | Wheel-size/license/offline-resolution prototype |
| Pipeline middleware boundary | ADR-015 | Runtime integration lead | Open | Before HW-005 replacement or ROS2 work | HW-005, HANDOFF-009 | Latency/lifecycle/reconnect integration spike |
| Arm ownership implementation | ADR-016 | Robot-controls lead | Open | Before HW-010 or HANDOFF-011 starts | HW-010, HANDOFF-005, HANDOFF-011 | Atomic controller-manager takeover/release test |
| Teleoperation command representation | ADR-017 | Teleoperation + RL leads | Open | Before HANDOFF-003 starts | HANDOFF-003, HANDOFF-006 | Recorded upstream command/state trace and distribution comparison |
| Start-envelope method | ADR-018 | RL evaluation lead | Open | Before HANDOFF-004 starts | HANDOFF-004 | False-accept/reject analysis on held-out successful/failed starts |
| Real-time execution boundary | ADR-019 | Runtime/safety lead | Open | Before RUN-003 starts; revalidate before HIL | RUN-003, SAFE-005, HANDOFF-010 | Initial process-boundary decision plus deployment-hardware latency/process-loss evidence |
| Legacy screwdriver history phase | ADR-020 | Policy owner | Open | Before OBS-008 completes | OBS-002, OBS-008, ART-007 | Legacy trace parity or retraining decision |
| Allegro hardware deployment scope | ADR-021 | Allegro hardware owner | Open | Before Allegro DeploymentBinding | HW adapter/profile work not yet scheduled | Hardware/driver/calibration inventory |
| Supported Python/runtime versions | ADR-022 | Platform lead | Open | Before FOUND-005 starts | FOUND-005, PKG-003, PKG-007 | Compatibility matrix and clean-install CI |
| Lock generation/update tooling | ADR-023 | Platform lead | Open | Before PKG-003 starts | PKG-003, PKG-007 | Reproducible lock regeneration on supported hosts |
| Canonical JSON and package-ID algorithm | ADR-024 | Artifact/security lead | Open | Before FOUND-009 completes | FOUND-009, ART-004, SPEC-011 | Cross-language golden digest vectors and self-reference analysis |
| Runtime/plugin API versioning | ADR-025 | Architecture lead | Open | Before SPEC-011 completes (Phase 10) | SPEC-011, DOC-002 | Compatibility/upgrade fixture across two API versions |
| Trusted plugin loading policy | ADR-026 | Runtime/security lead | Open | Before SPEC-011 starts (Phase 10) | SPEC-011, ART-010, TEST-004 | Live-disabled default plus allowlist/signature/revocation/adversarial duplicate-ID tests |

### 36.1 Artifact tensor format

Recommended default:

- JSON manifest/ExecutionGraph plus safetensors for whichever tensor components the graph declares.

Alternative:

- strict data-only torch format during transition.

Decision required before ART-006.

### 36.2 Asset distribution

Options:

- assets inside the Python wheel;
- separate versioned asset wheel/package;
- external artifact store with manifest.

Recommended:

- package small structural configs/manifests;
- separate versioned structural assets if wheel size is excessive;
- external generated caches.

Decision required before PKG-002/DATA-002.

### 36.3 ROS1, ROS2, or in-process first integration

Recommended:

- middleware-neutral core;
- retain ROS1 adapter for current Linker setup;
- add ROS2 lifecycle wrapper only when the arm/teleoperation system interface is known;
- support in-process embedding for low-latency pipelines.

### 36.4 Arm ownership implementation

Need to select:

- controller manager;
- lease authority;
- arm hold/impedance implementation;
- acknowledgements;
- E-stop/deadman integration.

Decision required before HW-010 or HANDOFF-011, and therefore before HANDOFF-005.

### 36.5 Upstream teleoperation command representation

Determine whether teleoperation can provide:

- actual low-level hand position targets;
- only measured positions;
- torque/velocity/synergy commands.

This determines whether current RMA histories can be primed faithfully.

Decision required before HANDOFF-003.

### 36.6 Start-envelope method

Recommended:

- deterministic hard safety bounds;
- simple empirical standardized-distance or nearest-neighbor score first;
- avoid a learned readiness classifier until data justifies it.

### 36.7 Real-time execution boundary

Determine whether Python scheduling meets the final required deadline.

Start with measured Python runtime. Move scheduler/driver to a real-time or native process only if p99 and fault testing show it is necessary.

### 36.8 Legacy screwdriver history phase

Choose:

- preserve exact legacy phase for existing policies;
- correct phase and retrain;
- temporarily support both named codecs.

Recommendation:

- support named legacy codec while new policies use corrected documented phase.

### 36.9 Allegro hardware deployment

Clarify:

- exact Allegro hardware revision;
- vendor driver and transport;
- tactile sensor usage;
- side;
- calibrated limits.

No generic runtime changes should be required once decided.

### 36.10 Supported Python/runtime versions and lock tooling

Select the minimum/maximum Python versions separately for pure runtime, training, and Isaac integration. Select one lock generator, document how platform markers and vendor packages are handled, and require a review bot/CI job to regenerate and diff locks. A hand-edited lock is not authoritative.

### 36.11 Canonical JSON and digest algorithm

Specify Unicode normalization, object-key ordering, number representation, forbidden NaN/Infinity values, path normalization, manifest exclusions, and package-tree hashing. Publish golden vectors consumable outside Python. This decision also defines how IDs are derived without hashing themselves.

### 36.12 Runtime and plugin API versioning

Version the pure protocol surface independently of package release version. Define compatible ranges, deprecation windows, capability negotiation, and whether an adapter shim may bridge one previous API version. Unknown major versions fail before plugin import or hardware connection.

### 36.13 Trusted plugin loading

Decide whether production permits third-party code at all. If permitted, define the exact allowlist source, signature chain, package digest pin, sandbox/process boundary, revocation update path, audit logging, and incident-disable mechanism. Python entry-point discovery alone is never trust.

---

## 37. Implementation conventions

### 37.1 Types and immutability

- frozen dataclasses for specs, manifests, calibration, and profiles;
- recursively freeze nested collections: canonical sorted tuples or a reviewed immutable-map type, never a mutable dict hidden inside a frozen dataclass;
- mutable state isolated in runtime/session objects;
- explicit units in names or schema;
- monotonic nanosecond timestamps for control;
- no Mapping[str, Any] at safety-critical live boundaries after resolution.

### 37.2 Errors

Define typed errors:

- SpecValidationError;
- ExperimentResolutionError;
- ArtifactValidationError;
- CodecMismatchError;
- HardwareIdentityError;
- CalibrationMismatchError;
- StateFreshnessError;
- SafetyViolation;
- OwnershipError;
- DeadlineError.

CLI may render them; library callers receive structured errors.

### 37.3 Logging

- no print in core contracts;
- no terminal logger constructed inside environment;
- events contain stable IDs;
- warnings do not substitute for failed invariants.

### 37.4 Configuration

- strict unknown-key rejection;
- override before derivation;
- canonical serialization;
- no private post-construction rebuild required;
- no mutable class-level config as cross-task state.

### 37.5 Tests

- test public behavior;
- avoid source-string assertions except packaging/lint-specific tests;
- parameterize registries;
- fake clocks rather than sleeping;
- fake transports rather than process-global state;
- record parity tolerances.

### 37.6 Git and review

Each major PR should state:

- work-item IDs;
- behavior-preserving or behavior-changing;
- artifacts/checkpoints tested;
- trace/evaluation comparison;
- retraining requirement;
- rollback.

Avoid combining:

- file reorganization;
- observation changes;
- reward retuning;
- hardware mapping changes;

in the same PR.

---

## 38. Operational rollout checklist

### 38.1 Before any live command

- policy package verified;
- deployment binding, detached attestations, trust policy, and launch configuration verified;
- hardware identity verified;
- calibration verified;
- SDK/driver version verified;
- arm/hand gateways exclusively own their physical command interfaces and reject a competing publisher;
- policy/hardware limits compared;
- canonical inference finite;
- runtime warmed;
- E-stop/deadman tested;
- all required timestamped safety signals and collision/workspace models are healthy, fresh, correct-frame, and correct-digest;
- exclusion zone established and collision clearance inspected;
- physical E-stop verified by the operator at the final station;
- independent arm/hand watchdog identities, armed/heartbeat/expiry state, and lease rejection verified after forced main-process termination;
- reduced power/torque/current mode selected for commissioning;
- tool tether, catch fixture, containment, or mechanically safe dummy setup installed as applicable;
- recovery owner and second observer assigned where the site risk assessment requires them;
- bounded tick count selected for commissioning;
- trace recording active;
- operator and abort criteria named.

### 38.2 Shadow mode

- state source sequence advances;
- source age below threshold;
- no invalid masks;
- policy period exact;
- history complete;
- start envelope stable;
- policy candidate finite and bounded;
- p99 inference within budget;
- no repeated saturation;
- arm/task envelope valid.

### 38.3 Arm hold

- initialized from current teleoperation target;
- ARM lease acknowledged;
- wrist pose/twist stable;
- wrench acceptable;
- teleoperation retains HAND;
- abort path exercised.

### 38.4 Hand blend

- live policy endpoint is recomputed every tick from the latest effective blended target;
- HAND lease transferred to transition controller;
- rate/acceleration within deployment binding;
- post-map safety approved the exact PreparedNativeCommand payload that was transmitted;
- measured following error within limit;
- PolicySession advances exactly once from post-mapping effective targets;
- arm hold remains valid.

### 38.5 RL active

- HAND lease held by RL;
- task metrics healthy;
- safety and driver heartbeats healthy;
- state freshness maintained;
- no control deadline trend;
- run remains within time/tick bound.

### 38.6 Hand-back

- teleoperation target generator synchronized;
- prepared acknowledgement received;
- bounded blend if needed;
- HAND ownership transferred first;
- ARM transferred only after pose synchronization;
- recorded ownership epochs confirm no overlap.

### 38.7 After run

- safety/fault summary reviewed;
- trace checksum stored;
- evaluator report generated;
- hardware/calibration/policy identities recorded;
- promotion or rejection decision recorded;
- no mutable production pointer changed without explicit approval.

---

## 39. Initial safety threshold process

This plan intentionally does not invent final numeric hardware thresholds.

Threshold workflow:

1. Start from manufacturer limits and conservative software caps.
2. Collect read-only/shadow traces.
3. Measure nominal distributions:
   - state age;
   - policy latency;
   - following error;
   - joint velocities/accelerations;
   - wrist error/wrench;
   - current/temperature;
   - quantization and saturation.
4. Define conservative commissioning thresholds.
5. Version them in DeploymentBinding.
6. Test deliberate threshold crossings.
7. Expand only through reviewed evidence.

Every threshold includes:

- value and unit;
- source/rationale;
- severity;
- debounce/persistence;
- stop response;
- last validation date;
- owner.

---

## 40. Baseline evidence from the 2026-07-11 review

This section records why the plan exists. Line numbers refer to the identified reviewed snapshot and may move later.

### 40.0 Reviewed snapshot identity

- Git revision: 868f101602f87405d6b07696270375b55fcf8579.
- Pre-document tracked patch: empty; SHA-256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855.
- Dirty-tree flag: true because user-owned untracked work was present and included where relevant to the review.
- Untracked-supplement aggregate SHA-256: 588519107ab1856842ac230cda3708afb5ff302acc32169856b0d1bcac1858dd.
- Aggregate definition: SHA-256 over lexicographically sorted lines of the form content-sha256, two spaces, repository-relative path, newline, for every non-ignored untracked file present at review finalization except this plan itself.
- Supplement scope: artifacts/, the nine linker_l20_palmup_grasp_50k cache files then present under assets/grasp_cache/, docs/linker_l20_inhand_gpu_handoff.md, and the untracked generation/optimization/rendering scripts under tools/.
- This plan and its README link are review outputs, not review inputs.

Exact macOS/zsh reproduction command used for the supplement digest:

    git ls-files --others --exclude-standard -z |
      while IFS= read -r -d '' f; do
        if [ "$f" != "docs/architecture-modularity-and-mixed-control-integration-plan.md" ]; then
          h=$(shasum -a 256 "$f" | cut -d ' ' -f 1)
          printf '%s  %s\n' "$h" "$f"
        fi
      done | LC_ALL=C sort | shasum -a 256

At final document QA this command saw 73 untracked paths including this plan, hashed the other 72, and reproduced 588519107ab1856842ac230cda3708afb5ff302acc32169856b0d1bcac1858dd. Sorting is by the complete digest/path line, not by path alone.

Phase 0 must materialize artifacts/baselines/2026-07-11/review-inventory.json with the exact path-level manifest, environment versions, and its own digest, then replace the pending metadata-table value above. Until then, the Git revision plus aggregate digest identifies content but is less convenient than the required full manifest.

### 40.1 In-hand deployment mismatch

- in-hand actor proprio_dim is 96 in screwdriver_rl/tasks/linker_l20/agents/rl_games_inhand_ppo_cfg.yaml around lines 14–19;
- simulator scales q and constructs frames in screwdriver_rl/tasks/linker_l20/inhand_rotation_env.py around lines 470–477;
- simulator stacks three frames around lines 275–286;
- deployment constructs one raw frame in screwdriver_rl/deploy/policy.py around lines 209–214 and 239–247;
- eval bypasses PolicySession in eval.py around lines 440–456;
- existing synthetic bundle tests derive proprio_dim as two times action DOFs.

Targeted PyTorch reproduction produced a 32-versus-96 normalizer mismatch.

### 40.2 Bundle semantic weakness

- train.py around lines 320–333 exports task and dimensions but not hand ID, side, semantic joint names, units, codec, or period;
- deploy/policy.py around lines 161–195 tolerates missing normalizer and defaults bounds/home/action scale;
- deploy/deploy_linker.py around lines 80–100 accepts a bundle before applying Linker mapping;
- Allegro 4F and Linker both use 16 actions.

Targeted reproduction confirmed a normalized actor loads without its saved normalizer.

### 40.3 Handoff and fixed wrist

- DeployPolicy reset around lines 217–222 returns target state to home;
- Linker deploy startup around lines 182–201 ramps to home before activation;
- Allegro, Linker screwdriver, and Linker in-hand simulations use fixed bases and absolute poses;
- reset writes hand root poses directly;
- no runtime contract contains wrist/task transform or arm hold.

### 40.4 Hand/task coupling

- Linker in-hand is a direct 766-line concrete environment;
- Linker kinematic maps and collision exclusions are copied between screwdriver and in-hand environments;
- generic screwdriver base is over 1,200 lines;
- Linker screwdriver overrides scene, curriculum, reward, contact, and privileged observation;
- base code still uses the name allegro for every hand.

### 40.5 Configuration

- dimensions are repeated across Python and YAML;
- geometry DR warns that post-construction overrides miss post-init derivation;
- spaces/assets/pregrasp tables are mutated during post-init;
- simulator and deployment pad/truncate mismatched history.

### 40.6 Packaging

- pyproject.toml declared nonexistent setuptools.backends.legacy:build;
- direct wheel inspection contained Python only, without agent YAML/assets;
- assets are resolved relative to repository parents;
- project dependencies omit required training/test components.

### 40.7 Artifacts and reproducibility

- Stage 2 writes task-level fixed filenames;
- docs/DEPLOY.md records smoke runs overwriting useful deployment bundles;
- manifest lacks source/config/environment/asset hashes;
- Stage-1 curriculum state requires manual reconstruction;
- Stage-2 checkpoints omit optimizer/RNG/replay and cannot resume exactly.

### 40.8 Runtime and hardware

- calibration is module-global;
- right side accepted while mapping is left-only;
- CAN valid-state checking lacks source freshness;
- deadline overruns only warn;
- live safety lacks following error, actuator health, wrist guards, and explicit E-stop/deadman integration;
- documentation records uncommitted SDK patches;
- simulated PIP reach exceeds hardware reach.

### 40.9 Tests and evaluation

- critical RL-Games tests catch all import exceptions, print skip, return, and may be reported as passed;
- many tests inspect AST/source text;
- eval metrics are screwdriver-specific while in-hand emits a separate schema;
- no CI or test lock was present;
- available local interpreters lacked pytest/Isaac for the complete suite;
- all reviewed Python files parsed successfully.

### 40.10 Stage-2 memory

- current collector stores every history and then concatenates;
- in-hand defaults imply about 15 GiB raw history and at least 30 GiB around concatenation peak.

---

## 41. First implementation pull requests

Recommended first sequence. Every PR below is implementable and verifiable on ENV-A (the macOS machine) except the two flagged rig-session items; batch ENV-B verification into scheduled rig sessions rather than blocking each PR on rig access.

### PR 0 — Preserve the review baseline

- commit this plan and the README link (both currently unmanaged by git and at risk of loss);
- write the Section 40.0 untracked-supplement manifest and durably back up the grasp-cache files and reviewed artifacts it hashes;
- FOUND-011 trained-artifact value inventory;
- mark docs/stage2-deployability-plan.md and docs/linker_l20_inhand_gpu_handoff.md as superseded for planning purposes.

Documentation and inventory only; no code change.

### PR 1 — Reproducible baseline and containment

- FOUND-002, FOUND-003, FOUND-007, and FOUND-008;
- SPEC-001 minimal semantic identity types required by GoldenTraceV1;
- expected-failing in-hand deployment test;
- HW-001 right-side rejection and FOUND-010 live containment for unresolved physical limits.

No policy behavior change. FOUND-008 real-trace capture and the FOUND-012 baseline are the first ENV-B rig session and may land as a follow-up commit from the rig.

### PR 2 — Build/install foundation

- FOUND-001, FOUND-004, FOUND-005, and FOUND-009;
- PKG-001, PKG-003, and PKG-004;
- TEST-001;
- src/ migration, packaged agent resources, clean-wheel install, and dependency boundary checks. (License inventory PKG-005 is owned by Phase 6.)

No policy behavior change; keep compatibility import wrappers in the same PR.

### PR 3 — Semantic codec and action contracts

- OBS-001;
- OBS-005;
- pure models;
- current Linker/Allegro joint orders represented;
- no environment integration yet.

No policy behavior change.

### PR 4 — HandSpec sources

- SPEC-002 and SPEC-003;
- intrinsic-versus-sim/hardware limit separation;
- URDF conformance;
- compatibility-generated current maps.

Behavior-preserving.

### PR 5 — Correct runtime codecs and PolicySession

- OBS-002;
- OBS-003;
- OBS-004;
- OBS-008;
- OBS-009;
- RUN-001 minimal integration;
- TEST-003 exactly-once/lifecycle coverage;
- actual 96-D actor path;
- strict declared normalization mode/state;
- 100-tick screwdriver and in-hand synthetic parity.

Contract correction; no live approval until simulator gate.

### PR 6 — Actual-runtime simulator gate

- OBS-007;
- OBS-010;
- simulator PolicySession bridge;
- Linker in-hand and screwdriver parity;
- machine-readable result stub.

Contract correction.

### PR 7 — Artifact manifest and strict loader

- ART-001 through ART-005;
- wrong-hand/side/order tests;
- control period;
- immutable candidate directory;
- ExecutionGraph/component and reference-output ComparisonSpec validation.

No reward/physics change.

These early PRs remove the most dangerous ambiguity while keeping the later generic-environment extraction reviewable.

---

## 42. Definition of done for an individual work item

A work item is complete when:

- code and public types exist;
- tests cover success, mismatch, and relevant fault paths;
- documentation and examples are updated;
- structured errors are exposed;
- no unreviewed behavior change occurred;
- performance/memory effect is measured where relevant;
- compatibility effect is recorded;
- retraining requirement is recorded;
- generated artifacts are schema-validated;
- tests ran in the item's designated verification environment (Section 2.4), and any claim about another environment is explicitly recorded as unverified;
- a machine-checkable verification command is recorded with the tracker entry;
- the linked phase exit criterion remains satisfied.

---

## 43. Glossary

Effective target:

- the post-mapping semantic setpoint accepted at the acknowledgement level required by DeploymentBinding, with explicit confidence/provenance; it is used as policy integration state.

Candidate target:

- a lease-free policy/controller output before authorization, safety, blending, mapping, and transport.

PreparedNativeCommand:

- immutable authorization/calibration/mapping-bound native payload plus semantic round trip; post-map safety validates the exact object later transmitted.

Codec:

- deterministic stateful transformation between semantic runtime state and network tensor layout.

DeploymentBinding:

- immutable, content-addressed binding of a policy package to hardware, calibration, task frame, timing, transition, and safety settings.

LaunchConfig:

- per-launch local endpoints, telemetry destinations, process topology, and secret references; it is separate from content-addressed compatibility artifacts.

ExperimentSpec:

- named, unresolved selection of task, hand, task-hand profile, algorithm, curriculum, domain-randomization, simulation profiles, and typed semantic overrides; run settings are a separate RunSpec.

HandAdapter:

- pure semantic-to-native/native-to-semantic mapping for one hardware model and side; it performs no I/O and chooses no safety response.

HandDriver:

- exclusive hand command gateway/orchestrator for one adapter, immutable calibration, transport, installed lease authority, prepared-command transmission, and explicit safe shutdown.

ArmCommandGateway:

- exclusive vendor/controller-manager boundary providing arm identity, state, lease/mode enforcement, acknowledgement, safe response, shutdown, and watchdog evidence.

HandSpec:

- simulator-neutral intrinsic facts about a hand.

Transport:

- raw native I/O, timestamp, reader-buffer, health, and acknowledgement evidence implementation; it does not interpret policy semantics.

Lease token:

- opaque authority proof bound to control session, resource, owner, epoch, mode, and expiry; the epoch rejects delayed previous-owner messages within the session.

PolicyPackage:

- immutable manifested ExecutionGraph, declared tensor components, preprocessing, creation-time contract/provenance, and numerical conformance fixtures; later evaluation/promotion evidence is detached.

PolicySession:

- environment-free stateful policy inference API.

PolicySnapshotBuilder:

- scheduler-owned deterministic temporal join that pairs measured state with the effective target in force for the codec's capture interval.

ResolvedExperiment:

- fully derived, validated, immutable experiment contract.

RL_SHADOW:

- state where real inputs prime and evaluate RL but RL commands are not applied.

SafetySupervisor:

- deterministic component downstream of all controllers that validates state and candidate commands and selects a bounded response.

TaskHandProfile:

- typed intersection between a generic task and a hand, including initialization source, active fingers, task_from_wrist, preprocessing/conditioning, and concrete strategy selections.

TaskSpec:

- hand-independent manipulation semantics.

Wrist/task envelope:

- allowed wrist transform, motion, wrench, and stability conditions for policy activation and continued operation.

---

## 44. Final architecture outcome

When this plan is complete:

- mounted screwdriver and in-hand rotation are reusable task families;
- Allegro, Linker, and future hands are reusable hand models;
- task-hand tuning is explicit profile data;
- observation and action behavior is identical in simulation and deployment;
- scalar live and batched simulator/training kernels share reset, history, action, and conditioning semantics;
- every policy knows exactly which semantic joints, hand, side, task, control rate, codec, and assets it belongs to;
- policy packages and calibrations are immutable and traceable;
- hardware adapters are per-instance and transport-independent;
- deployment is callable as a library;
- teleoperation, arm hold, transition, RL, safety, and hand-back have explicit ownership;
- exclusive arm/hand gateways, linearizable lease transfer, prepared-native safety, and crash-independent watchdogs enforce that ownership;
- failure paths are bounded and tested;
- adding a hand no longer requires copying task environments or deployment loops;
- external pipelines can discover, validate, embed, evaluate, and control the runtime through stable APIs.

---

## 45. Document maintenance

This document is the architectural source of direction until replaced by accepted ADRs and implemented contracts.

Maintenance rules:

- update Last updated when normative content changes;
- record approval state and reviewers;
- link completed work-item IDs to commits/PRs in the project tracker;
- do not erase baseline evidence after code moves; annotate it as historical;
- move accepted decisions into ADRs and link them here;
- keep phase exit criteria aligned with CI and promotion gates;
- update the current capability/support matrix after each validated deployment;
- distinguish proposed, implemented, simulation-validated, HIL-validated, and live-approved features;
- require a safety reviewer for changes to Sections 14–18, 33.5–33.6, 38, or 39;
- require an artifact/training reviewer for changes to observation codecs, action transforms, timing, or compatibility claims;
- as of revision 2 this document is frozen as the reference architecture record: operative day-to-day tracking lives in the work-item tracker, and normative design changes land as ADRs that are back-annotated here.

Interpretation:

- must and required indicate a release/integration requirement;
- should and recommended indicate the default design unless an ADR records an alternative;
- illustrative data models describe ownership and semantics; exact Python syntax may change without changing the contract;
- numeric safety thresholds remain deployment-binding data and require hardware evidence.

### 45.1 Revision log

- Revision 1 (2026-07-11): initial plan produced by the architecture review.
- Revision 2 (2026-07-11): maintainer-review revision.
  - Added the execution-environment split (Section 2.4) and the Part I / Part II work partition with per-phase environment lines (Section 29), reflecting that development happens on a macOS machine without Isaac/CUDA while training, parity, and hardware work run on an Ubuntu CUDA/Isaac rig and hardware station.
  - Added the capacity model with committed core, activation triggers, research-continuity rule, and resting-state invariant (Section 2.5); reframed workstreams for solo capacity (Section 32).
  - Added FOUND-011 trained-artifact value inventory and FOUND-012 throughput/wall-time baseline (Phase 0, Section 31.1); made OBS-004 conditional on FOUND-011 and re-scoped the legacy-codec strategy (Section 30.1); added throughput exit criteria to Phases 4–5 and Section 33.7.
  - Created the deferred reference-design track (Section 31.11): ART-010 and TEST-004 re-tiered to P3 with the trust machinery in Sections 11.4 and 12 marked deferred behind a trust-boundary trigger; SPEC-011 external plugin loading moved to Phase 10 (Sections 23.6, 27.6); AckLevel plumbing limited to what current transports can evidence (Section 9.2).
  - Reclassified Section 18's state machine as reference design pending ADR-016/ADR-017 and added the missing teleoperation row to the Section 4 capability matrix.
  - Defined single-owner gate semantics and removed the revision-1 duplicate gate listings (FOUND-009, PKG-005, HW-001, SPEC-012, TEST-004, HANDOFF-001, Phase 8/9 re-lists); PKG-005 license inventory moved from Phase 0 to Phase 6; SIM-011 ownership moved to Phase 8.
  - Specified observation-noise ownership outside the deployable codec (Section 9.3) and made golden traces record-then-replay artifacts captured on the rig and replayed on CPU (Sections 9.9, Phase 0).
  - Added PR 0 baseline preservation and PR environment notes (Section 41); Stage-2 corpus retention (Section 20.12); calibration-file migration rows (Section 28.7); environment rows in the compatibility manifest and locks (Section 22); test-marker/CI environment mapping (Sections 25.1, 26.1); verification-environment and verification-command bullets in the definition of done (Section 42); risk-register updates including promoting research-velocity risk to High (Section 34).
