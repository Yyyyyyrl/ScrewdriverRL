# Dexterous Manipulation Architecture Revision V2.1

| Metadata | Value |
|---|---|
| Status | Proposed architecture and implementation plan |
| Date | 2026-07-16 |
| Revision | V2.1 — post-review amendments: Vive-source correction, continuous-shadow fast toggle, single-repo packaging, thin critical path |
| Replaces | No existing document; this is a new integration-specific revision |
| Builds on | architecture-modularity-and-mixed-control-integration-plan.md |
| Training repository | dex-forge, renamed from ScrewdriverRL |
| Runtime repository | dex-manipulation, working name pending repository creation |
| Teleoperation migration source | dex_teleop |
| Initial operator inputs | Manus glove for the hand and Vive tracker for the arm (the Vive tracker source is new code; current dex_teleop arm input is OpenXR hand tracking — see 4.2) |
| Initial hand hardware | Physically verified left LinkerHand L20/G20 mapping from dex-forge |
| Arm hardware | Not yet selected; all arm integration remains behind a capability-driven gateway |
| Initial readiness evidence | Operator confirmation plus machine-verifiable state freshness and limits |
| Intended readers | RL, simulation, teleoperation, robot-runtime, controls, and safety engineers |

## 1. Executive decision

The integrated system will be split into two durable products rather than turning
the existing training repository into a hardware-control monolith.

1. dex-forge owns simulation, task definitions, policy training, evaluation,
   policy packaging, and promotion.
2. dex-manipulation owns operator input, teleoperation retargeting,
   policy selection, policy execution, hardware gateways, mixed-control
   handoff, safety supervision, and observability.
3. dex_teleop is a migration source. Its Manus, Vive, retargeting, LinkerHand,
   and future-useful input code will be extracted behind runtime contracts.
   Its actuator-owning experiment scripts will not become the final runtime.
4. The runtime repository is one self-contained installable distribution. The
   contracts live in an internal dex_contracts subpackage; no separately
   published contracts wheel is planned. dex-forge and the runtime exchange
   golden-trace fixture files for parity testing instead of sharing a
   package. The runtime must never depend on Isaac Lab or import dex-forge
   training code, and a bootstrap script must install all pinned dependencies
   on a fresh machine.
5. Teleoperation and RL will never publish native actuator messages directly.
   Both produce semantic command candidates. One exclusive gateway per
   physical resource performs arbitration, safety checks, calibration, native
   encoding, transport, acknowledgement, and watchdog handling.
6. During RL execution, the arm is held by a deterministic controller and the
   policy controls the finger joints only.
7. Multiple policies may be installed. A local operator surface (terminal
   status first, richer UI later) lists compatible policies, performs
   preflight validation, presents readiness evidence, and requests handoff.
   Mode switching is triggered by a tap-to-toggle OperatorSwitchSource bound
   to a configurable key; the existing PCsensor USB foot switch is later
   programmed to emit the same key. Neither is an independent safety device;
   the machine-verifiable gates remain mandatory.
8. Operator confirmation is the initial task-readiness input. The readiness
   interface is provider-based so camera, tactile, force/torque, or object-pose
   providers can be added without changing the handoff state machine.
9. Switching latency is a first-class requirement. Once a policy is armed,
   its shadow session runs continuously during teleoperation, and the Manus
   retargeter keeps running during RL execution, so both handoff and
   hand-back reduce to machine-gate checks plus the blend window.

The architecture supports mounted screwdriver rotation and free-object in-hand
rotation from the beginning. References to doing one task first and the other
second mean only the recommended hardware commissioning order:

1. validate the complete handoff on the mounted screwdriver because the object
   fixture and task frame are constrained;
2. commission free-object rotation after the same runtime has passed its
   fixed-fixture safety and continuity gates.

This ordering is a risk-control recommendation, not a long-term hierarchy and
not a reason to hard-code task-specific behavior into the runtime.

## 2. Goals

The system must allow an operator to:

1. use a Manus glove to form and adjust a dexterous-hand posture;
2. use a Vive tracker to move the robot arm toward the task;
3. see available policies and their compatibility in a UI;
4. select a policy without loading unsafe or incompatible artifacts;
5. request a transition while teleoperation remains active;
6. let the runtime prime and evaluate the policy in shadow mode;
7. transfer the arm to a verified hold controller;
8. blend finger control from teleoperation to RL without a target jump;
9. let the selected RL policy execute precise finger manipulation;
10. abort or request hand-back at any time;
11. return both resources to teleoperation without a target jump;
12. obtain a complete event and command trace for diagnosis and replay.

The engineering platform must also:

- add new policies without modifying the supervisor;
- add a future robot arm through one adapter and capability declaration;
- add a camera or other readiness sensor through one provider interface;
- add a teleoperation source without giving it hardware access;
- reproduce training-time policy inputs exactly in deployment;
- reject artifacts with the wrong task, hand, side, joint order, codec, rate,
  calibration, or runtime API version;
- keep the hardware runtime installable without Isaac Lab;
- make all unsafe ownership races structurally impossible rather than merely
  discouraged by convention.

## 3. Non-goals

The first integration does not:

- give a finger-only RL policy control of general arm motion;
- promise arbitrary hot-swapping between active RL policies;
- treat operator confirmation as proof that physical geometry is correct;
- make ROS 2 the core in-process API;
- require a camera before the first laboratory handoff;
- implement an arm-specific adapter before the next arm is selected;
- preserve direct actuator access in migrated experiment scripts;
- automatically open the hand on every fault;
- silently reinterpret a legacy checkpoint as a deployable policy package;
- rename every existing Python import at the same time as the repository;
- create a distributed lease service for a single-host laboratory runtime
  unless a concrete deployment topology requires one.

## 4. Current-system findings that constrain the design

### 4.1 dex-forge training and deployment

The training repository already has several useful foundations:

- mounted screwdriver and free-object LinkerHand tasks;
- Stage-1 privileged training and Stage-2 adaptation;
- an environment-free PyTorch deployment implementation;
- a physically verified left-hand semantic-to-SDK mapping;
- direct CAN and ROS transport prototypes;
- dry-run, ramp, watchdog, logging, and calibration utilities;
- task-specific control rates and action integration;
- mapping round-trip, monotonicity, and calibration-overlay tests
  (tests/test_linker_sdk_map.py) that seed the section 19.2 suite.

The current code also has blockers that must be corrected before handoff:

- in-hand training feeds the actor three 32-value frames, while the deployment
  implementation feeds only one raw 32-value frame;
- in-hand training scales measured positions and deployment does not;
- policy reset always seeds current targets to home instead of adopting the
  target applied by teleoperation;
- policy packages do not fully identify the hand, side, semantic joint order,
  units, observation codec, control period, calibration, task frame, or
  complete provenance;
- the live deployment class owns policy, mapping, transport, scheduling,
  signals, logging, and safety in one object;
- simulation fixes the hand base and does not yet export a task-from-wrist
  contract for an actual robot arm;
- mounted screwdriver and free-object starts are narrow task-specific
  distributions, not arbitrary postures that merely look plausible;
- the deployment CLI --hz flag defaults to 10 Hz and would silently misrun a
  20 Hz in-hand package: the concrete instance of the silent rate-override
  hazard banned in section 6.3;
- the hand generation is selected only by a --hand-joint flag (default G20)
  while only L20 native tables are vendored; the calibration artifact must
  record hand_joint explicitly.

These are policy-runtime contract corrections. They are not optional cleanup.

### 4.2 dex_teleop

The selected operator setup is composed from one existing path and one new
one:

- Manus glove data and dex-retargeting for hand posture (existing);
- Vive tracker data for the arm (new code; see the correction below).

Correction in this revision: dex_teleop contains no Vive tracker reader. Its
current arm input is OpenXR headset hand tracking served by the WiVRn runtime
(vr_utils/vr_hand_reader.py), from which a wrist pose drives Hitbot-oriented
incremental control. Only the wrist-delta arithmetic in main_new.py is design
evidence for the ArmTargetGenerator; the ViveArmSource itself is greenfield,
and its acquisition stack (SteamVR/openvr, OpenXR with the tracker extension,
or libsurvive) is a new open decision in section 23.

The current repository is valuable as working experimental code, but it is not
yet a supervisory runtime:

- the hand controller opens CAN directly;
- the arm controller opens its robot socket directly;
- different scripts duplicate retargeting and hardware loops;
- samples do not carry a shared monotonic timestamp, sequence, deadline, or
  deadman state;
- the arm and hand have independent queues and may consume different operator
  frames;
- queue overflow can silently drop data;
- input loss has no explicit task-aware safe response;
- process failure is not centrally supervised;
- shutdown uses process termination rather than an acknowledged safe sequence;
- the arm endpoint and transforms are hard-coded;
- the arm command is event-driven and incremental rather than a supervised,
  bounded absolute target;
- there is no policy registry, policy selector, readiness model, arm hold,
  hand blending, ownership transfer, or hand-back protocol;
- there is no top-level installable Python distribution or contract test suite.

### 4.3 Physically significant hand-mapping conflict

The current teleoperation code and dex-forge disagree on the thumb meanings of
SDK slots 5 and 10. The dex-forge mapping has been physically verified and is
therefore authoritative for the initial left LinkerHand deployment:

| Semantic joint | SDK slot |
|---|---:|
| index_mcp_roll | 6 |
| index_mcp_pitch | 1 |
| index_pip | 16 |
| middle_mcp_roll | 7 |
| middle_mcp_pitch | 2 |
| middle_pip | 17 |
| ring_mcp_roll | 8 |
| ring_mcp_pitch | 3 |
| ring_pip | 18 |
| pinky_mcp_roll | 9 |
| pinky_mcp_pitch | 4 |
| pinky_pip | 19 |
| thumb_cmc_yaw | 10 |
| thumb_cmc_roll | 5 |
| thumb_cmc_pitch | 0 |
| thumb_mcp | 15 |

The runtime must not carry a second mapping in its Manus retargeter or hand
controller. Retargeting emits a named semantic vector. The exclusive hand
adapter alone converts that vector to the SDK representation using the
physically verified calibration.

Two further migration hazards were verified in dex_teleop: the swapped thumb
table is duplicated in both qpos_to_cmd.py and robot_hand_controller.py, and
both copies also apply a hard-coded -10 degree thumb_cmc_roll bias before
sending. That bias is an undocumented behavior tweak; it must be either
promoted into the versioned TeleopProfile or explicitly dropped, otherwise
recorded-input parity tests will fail for no visible reason.

### 4.4 Robot-model duplication

The two LinkerHand URDF files are not byte-identical. The dex_teleop copy adds
five fixed fingertip reference frames used by retargeting. The articulated
portion is otherwise unchanged in the current textual diff.

The canonical runtime/training asset should be the superset model with
fingertip frames. Its model digest, semantic-joint schema, mimic-joint rules,
mesh digests, and coordinate conventions must be versioned. Retargeting,
training, policy packaging, and runtime preflight must all reference that
identity.

## 5. Repository and package boundaries

### 5.1 dex-forge

dex-forge is the policy forge. It owns:

- Isaac Lab task definitions;
- robot and task simulation assets;
- reward, reset, curriculum, and domain-randomization logic;
- grasp-cache generation and validation;
- Stage-1 and Stage-2 training;
- evaluation and regression baselines;
- deployment-codec parity tests;
- policy-package export;
- artifact promotion, provenance, and compatibility declaration;
- simulated wrist perturbation and handoff-distribution generation.

dex-forge must not own:

- live Manus or Vive acquisition;
- the production UI;
- the live arm socket;
- the live hand CAN socket;
- resource ownership transfer;
- production watchdogs;
- task-aware hardware safe responses.

The repository rename from ScrewdriverRL to dex-forge is a product/repository
rename. The existing screwdriver_rl Python package should remain importable
during the first architecture phase. Renaming it to dex_forge is a separate,
reviewable migration with a compatibility window; it must not be mixed with
the GitHub and local-folder rename.

Recommended future tree:

    dex-forge/
      pyproject.toml
      src/
        dex_forge/
          tasks/
          hands/
          observations/
          actions/
          training/
          evaluation/
          artifacts/
      assets/
      configs/
      tests/
      tools/
      docs/

### 5.2 dex-manipulation

dex-manipulation is a neutral consumer of policy artifacts. It owns:

- pure protocol and schema definitions;
- policy package validation and inference;
- the policy registry and local cache;
- Manus and Vive source adapters;
- dex-retargeting integration;
- hand and arm semantic command candidates;
- the multi-rate scheduler;
- handoff and hand-back state machines;
- readiness aggregation;
- arm and hand safety supervision;
- exclusive hardware gateways;
- operator UI and physical-input integration;
- event logging, trace recording, and replay;
- fake gateways and hardware-in-the-loop test harnesses.

Decided workspace (one source tree, one installable distribution):

    dex-manipulation/
      pyproject.toml
      bootstrap.sh              # fresh-machine dependency install
      src/
        dex_contracts/
        dex_runtime/
        dex_teleop_adapters/
        dex_hardware_linker/
        dex_ui/                 # deferred; terminal status ships first
      configs/
      tests/
      docs/

The logical boundaries between subpackages are mandatory and are enforced by
an import-linter contract in CI, not by physical package separation. Separate
wheels may be split out later if a concrete deployment topology requires
them. Downloading the repository to a fresh machine and running the bootstrap
script must be sufficient to run the runtime.

### 5.3 dex_teleop migration status

dex_teleop should remain unchanged as a known-working reference until parity
tests prove the new adapters. Code is migrated in small, behavior-labeled
steps:

| Current area | Runtime destination | Migration rule |
|---|---|---|
| Manus ROS 2 subscription | teleop Manus source adapter | Preserve layout and side validation; add timestamps and health |
| OpenXR wrist reader (vr_hand_reader.py) | design evidence for ArmTargetGenerator only | No Vive reader exists; ViveArmSource is new code (see 4.2) |
| retargeting adapters/configs | teleop retargeting package | Emit named semantic radians only |
| RobotHandController | Linker adapter and gateway | Replace duplicate mapping with verified dex-forge mapping |
| ArmController | future arm adapter evidence only | Do not treat Hitbot API as the future generic contract |
| main_new multiprocessing | supervisor integration tests | Do not migrate direct actuator ownership |
| LinkerHand SDK copy | pinned hardware dependency | One source, version, and digest |

After equivalent recorded-input output and live low-power tests pass, old
direct-actuation scripts are marked experimental, then deprecated. They must
not remain an undocumented bypass around the exclusive gateways.

### 5.4 Dependency direction

Allowed dependencies (dex_contracts is an internal subpackage; the rules are
enforced by import-linter in CI):

    dex_runtime -> dex_contracts
    teleop adapters -> dex_contracts
    hardware adapters -> dex_contracts
    UI -> dex_runtime public API
    dex-forge <-> runtime: golden-trace fixture files only, no shared package

Forbidden dependencies:

    dex-runtime -> Isaac Lab
    dex-runtime -> dex-forge source tree
    dex-forge -> live hardware gateway
    teleop source -> hand or arm transport
    policy implementation -> hand or arm transport
    UI -> vendor SDK

## 6. Logical runtime architecture

### 6.1 Data flow

    Manus ROS 2 source
            |
            v
    timestamped human-hand sample
            |
            v
    dex-retargeting
            |
            v
    TeleopHandCandidate ------------------------+
                                                  \
    measured HandState -> PolicySession -----------> CommandArbiter
                              |                    /
                              v                   /
                       PolicyHandCandidate ------+
                              |
                              v
                    transition/blend controller
                              |
                              v
                       SafetySupervisor
                              |
                              v
                    exclusive Linker gateway
                              |
                              v
                         CAN / hand

    Vive source
        |
        v
    timestamped tracker sample
        |
        v
    bounded absolute arm target generator
        |
        +----------------------+
                               v
                         HandoffSupervisor
                               |
                   +-----------+-----------+
                   |                       |
                   v                       v
            teleop arm candidate       ArmHoldController
                   |                       |
                   +-----------+-----------+
                               v
                        arm safety checks
                               |
                               v
                     exclusive ArmGateway
                               |
                               v
                          future arm

### 6.2 Process topology

Logical ownership matters more than the exact initial process count. The
decided first topology is a single Python process (thin critical path): each
gateway runs on its own dedicated thread, sources run on worker threads, and
blocking vendor calls never run on the policy scheduler thread. The
multi-process split below is a later hardening step, not a prerequisite for
first handoff:

- supervisor and policy process;
- Manus source process or ROS 2 node;
- Vive source process;
- exclusive Linker gateway process;
- exclusive arm gateway process once hardware is selected;
- UI process;
- independent hardware/controller watchdog path.

The Manus and Vive workers may fail or restart without acquiring actuator
access. Each gateway is the only process that opens its physical transport.
The gateway accepts one local authenticated command channel plus a narrowly
scoped safety channel.

The interfaces and ownership rules are identical in both topologies, so the
process split can happen later without changing any contract.

### 6.3 Clock and scheduling model

Every runtime host uses monotonic time for:

- source generation time;
- local receive time;
- state acquisition time;
- decision deadline;
- command validity;
- heartbeat age;
- ownership epoch;
- transition deadline;
- acknowledgement latency.

Wall-clock time is recorded for human correlation only.

The runtime is explicitly multi-rate:

| Loop | Rate source |
|---|---|
| Manus acquisition | source capability and ROS publisher |
| Vive acquisition | source capability |
| retargeting | new valid Manus samples or bounded worker rate |
| mounted screwdriver policy | package-declared 10 Hz |
| free-object policy | package-declared 20 Hz |
| hand gateway | hardware-safe rate, holding/interpolating semantic targets |
| arm teleoperation | selected arm capability |
| arm hold | selected arm servo/hold capability |
| UI | non-real-time status rate |

No CLI default may override the policy package control period silently.
Overruns are measured against absolute deadlines rather than accumulated
relative sleeps.

## 7. Core contracts

The contracts package contains immutable, serializable value types. Hardware
objects, Torch modules, ROS nodes, and vendor SDK instances are excluded.

### 7.1 Identity types

Every state, candidate, command, acknowledgement, and trace carries enough
identity to prevent same-shape misuse:

- runtime protocol version;
- control session ID;
- source ID;
- resource ID;
- hand model and side;
- semantic joint schema ID;
- task ID and version where applicable;
- policy package ID where applicable;
- calibration ID;
- control epoch;
- monotonic sequence number.

### 7.2 SemanticJointSchema

SemanticJointSchema declares:

- ordered joint names;
- units;
- position limits;
- optional velocity and effort limits;
- continuous versus bounded joints;
- mimic relationships;
- hand model;
- hand side;
- schema version and digest.

All cross-component hand vectors use this schema. Positional arrays without a
schema ID are invalid at process and artifact boundaries.

The initial Linker schema uses the physically verified 16-joint order:

    index_mcp_roll
    index_mcp_pitch
    index_pip
    middle_mcp_roll
    middle_mcp_pitch
    middle_pip
    ring_mcp_roll
    ring_mcp_pitch
    ring_pip
    pinky_mcp_roll
    pinky_mcp_pitch
    pinky_pip
    thumb_cmc_yaw
    thumb_cmc_roll
    thumb_cmc_pitch
    thumb_mcp

### 7.3 TimestampedSample

All source and hardware samples include:

- generated_time_ns when supplied by the source;
- received_time_ns in the local runtime clock;
- sequence;
- source health;
- validity mask;
- coordinate-frame ID;
- units;
- optional source-specific diagnostics.

generated_time_ns from another clock domain is never compared directly with a
local monotonic deadline without a declared clock translation and uncertainty.

### 7.4 HandState

HandState contains:

- measured semantic joint position;
- optional velocity and effort;
- state sequence and acquisition time;
- raw-native state reference for diagnostics;
- state quality and missing-joint mask;
- hardware fault and temperature fields when available;
- last acknowledged effective semantic target;
- acknowledgement capability level.

The runtime distinguishes:

- candidate proposed;
- command authorized;
- command encoded;
- command sent to bus;
- device accepted, if evidence exists;
- servo applied, only if evidence exists.

The initial Linker transport may only prove sent-to-bus. It must not label that
as servo-applied.

### 7.5 Hand candidates and authorized commands

TeleopHandCandidate and PolicyHandCandidate are lease-free proposals. They
contain:

- schema and source identity;
- semantic position target in radians;
- generation and validity times;
- source-state sequence;
- candidate diagnostics;
- optional confidence.

Only the CommandArbiter and transition controller select a candidate. The
exclusive gateway attaches:

- control session;
- owner;
- control epoch;
- final command ID;
- command mode;
- deadline;
- safety decision;
- calibration and mapping IDs.

### 7.6 ArmState and ArmTarget

ArmState is capability-driven and may include:

- measured joint positions and velocities;
- TCP pose and twist;
- controller mode;
- following error;
- wrench;
- joint, drive, and communication faults;
- state timestamp and heartbeat age.

ArmTarget is explicit about its representation:

- absolute joint position;
- absolute TCP pose in a named robot-base frame;
- bounded Cartesian twist;
- or another adapter-declared mode.

Unchecked strings do not select modes. The first generic teleoperation
contract should use a bounded absolute TCP or joint target. Vive deltas update
an internal absolute target; they are not sent directly to the arm gateway.

### 7.7 TaskReadinessEvidence

TaskReadinessEvidence is emitted by providers. Common fields are:

- provider ID and version;
- task and hand identity;
- generation time and validity;
- result: pass, fail, unknown, or operator-confirmed;
- quantitative measurements;
- reason codes;
- confidence when meaningful;
- evidence references for trace replay.

Initial providers:

- OperatorConfirmationProvider;
- HandStateFreshnessProvider;
- ArmStateFreshnessProvider;
- WristMotionProvider;
- JointEnvelopeProvider;
- GatewayHealthProvider;
- PolicyCompatibilityProvider.

Future providers:

- CameraObjectPoseProvider;
- ObjectPresenceProvider;
- TactileContactProvider;
- DropDetectorProvider;
- ForceTorqueProvider;
- FixturePoseProvider.

The HandoffSupervisor consumes the aggregate interface and does not import any
specific sensor library.

## 8. Canonical hand mapping and calibration

### 8.1 Source of truth

The physically verified mapping in dex-forge is the initial authority. Before
runtime extraction it must be converted from mutable module globals into an
immutable calibration artifact with:

- hand model and side;
- hand generation as sent to the SDK (hand_joint G20 or L20);
- serial number or fleet applicability;
- semantic schema digest;
- semantic-to-native slot table;
- semantic soft limits;
- native arc/range limits;
- direction and sign;
- offsets;
- inactive slots and fill behavior;
- calibration procedure version;
- calibration author/date;
- physical verification evidence;
- artifact digest.

### 8.2 Pure mapping layers

Mapping is divided into pure stages:

1. schema validation;
2. semantic safety clamp;
3. semantic-to-calibrated-native preview;
4. native quantization;
5. native-to-semantic diagnostic inverse;
6. round-trip error measurement.

The transport receives prepared native bytes or values. It does not repeat
calibration logic.

### 8.3 Teleoperation rule

dex-retargeting output remains in its own named URDF joint order until a pure
name-based projection produces the canonical semantic vector. It must not use
its current qpos-to-command slot map. The same semantic target may then pass
through the exact adapter used for RL. The hard-coded -10 degree
thumb_cmc_roll bias found in dex_teleop is a TeleopProfile decision, not a
mapping property (section 4.3).

### 8.4 Robot-model rule

One canonical URDF includes the required fingertip frames. CI verifies:

- model digest;
- expected actuated and mimic joints;
- semantic schema coverage;
- fingertip link presence;
- limits used by retargeting;
- limits used by training;
- mesh resolution;
- no duplicate asset with a different digest in released packages.

## 9. Teleoperation architecture

### 9.1 ManusHandSource

ManusHandSource owns ROS 2 subscription and source validation only. It:

- validates message layout and glove side;
- assigns source sequence and local receive time;
- preserves source time if present;
- reports tracking quality and stale intervals;
- emits a normalized human-hand sample;
- never imports the Linker SDK;
- never performs actuator mapping;
- never holds a hand ownership token.

### 9.2 ManusRetargeter

ManusRetargeter:

- consumes normalized Manus samples;
- applies the selected dex-retargeting configuration;
- resolves retargeting joint names by name;
- emits a canonical 16-joint semantic target;
- reports solver time, convergence, clipping, and missing points;
- supports recorded-input deterministic replay;
- is testable without ROS and without hardware.

Retargeting filters are part of a versioned TeleopProfile. Their state is reset
explicitly when a session starts or tracking recovers.

### 9.3 ViveArmSource

ViveArmSource is new code: dex_teleop has no Vive reader to migrate (section
4.2), and the acquisition stack is an open decision (section 23).
Behaviorally, ViveArmSource:

- emits timestamped tracker pose in a named Vive frame;
- reports tracking quality and origin validity;
- never imports a robot vendor SDK;
- never performs robot IK;
- never sends a robot command.

### 9.4 ArmTargetGenerator

ArmTargetGenerator begins with a deliberate clutch/anchor operation:

1. read a fresh measured arm pose;
2. capture the current Vive pose;
3. create a teleoperation anchor;
4. transform subsequent tracker displacement through a calibrated transform;
5. integrate it into an absolute bounded target;
6. apply workspace, orientation, rate, and singularity-related constraints;
7. emit an ArmCandidate with a deadline.

Tracking recovery requires a new anchor. It must not apply the displacement
accumulated while tracking was lost.

### 9.5 Independent source health

Manus and Vive health are independent:

- loss of Manus does not make a Vive sample invalid;
- loss of Vive does not make a Manus sample invalid;
- the supervisor decides the task-aware response for each resource;
- a shared operator session may correlate their data without forcing them
  through one failure-prone queue operation.

## 10. Policy architecture

### 10.1 PolicyPackage

A deployable PolicyPackage is immutable and content-addressed. At minimum it
contains:

- package format and protocol versions;
- package ID and digest;
- task ID and task version;
- hand model, side, and schema digest;
- required calibration compatibility;
- policy control period;
- actor and adapter tensor files;
- exact network architecture;
- ProprioCodec specification;
- actor-input assembler specification;
- action transform;
- action limits and integration semantics;
- history length and reset semantics;
- expected state fields and acknowledgement level;
- task-from-wrist and gravity-relative posture requirements;
- training commit and dirty-state declaration;
- resolved training configuration digest;
- URDF and asset digests;
- evaluation results and promotion status;
- supported runtime API range.

The runtime rejects missing safety-critical fields. It does not invent defaults
for hand side, schema, codec, control rate, calibration, or target reset.

### 10.2 ProprioCodec

Each task family owns an explicit codec.

Mounted screwdriver codec:

- measured 16-joint semantic position;
- effective 16-joint current target;
- declared scaling;
- declared legacy history behavior;
- 10 Hz control period.

Free-object codec:

- measured position scaled exactly as in training;
- effective current target;
- 32-value frame;
- 30-frame adapter history;
- latest three frames flattened to 96 actor values;
- 20 Hz control period.

Training, evaluation, package export, replay, and live execution call the same
pure codec implementation or prove golden-trace equivalence.

### 10.3 PolicySession lifecycle

PolicySession exposes explicit lifecycle operations:

- load and validate;
- reset from measured position and effective target;
- observe;
- preview without ownership;
- synchronize to an effective target;
- activate at a specific tick and control epoch;
- step while active;
- deactivate;
- close.

reset must not replace the effective teleoperation target with home.

During shadow and blend:

- history is built from fresh measured state and the effective target actually
  selected by the runtime;
- a private unexecuted policy target does not masquerade as applied history;
- preview runs at the package-declared cadence;
- blend completion and activation do not append the transfer tick twice.

### 10.4 Policy registry and selection

The runtime scans one or more configured immutable package stores. It builds a
PolicyDescriptor index without loading all Torch weights. The UI displays:

- human-readable policy name;
- task and hand;
- package ID;
- promotion status;
- required rate and runtime version;
- calibration compatibility;
- latest evaluation summary;
- readiness requirements;
- load/preflight status.

Selection is two-stage:

1. select and preflight a package while teleoperation or safe hold owns the
   resources; a successfully preflighted package becomes the armed policy,
   and its shadow session starts immediately and runs continuously
   (section 14.3);
2. explicitly request activation through the handoff state machine; with a
   policy armed, the switch tap is the activation request.

Selecting a row in the UI never actuates hardware.

### 10.5 Multiple active-policy rules

Only one policy may own the hand. RL-to-RL hot swap is initially prohibited.
To change policies:

1. request hand-back or safe hold;
2. return through a stable non-RL ownership state;
3. preflight the new package;
4. prime it in shadow;
5. perform the normal arm-hold and hand-blend protocol.

A future direct RL-to-RL transition may be added only with explicit
compatibility, state-transfer, history, action-mode, and live-endpoint blend
contracts.

## 11. Arm abstraction for unknown future hardware

### 11.1 Capability discovery

Because the next arm is not selected, the architecture defines behavior rather
than vendor calls. ArmCapabilities declares:

- supported command modes;
- state fields and rates;
- minimum and maximum command rates;
- whether atomic mode switching exists;
- whether controller-side hold exists;
- watchdog and keepalive support;
- acknowledgement levels;
- joint and Cartesian limits;
- fault and E-stop visibility;
- shutdown semantics;
- controller and adapter identity.

An adapter cannot claim a capability without a conformance test.

### 11.2 ArmGateway interface

The exclusive ArmGateway provides:

- connect and identity verification;
- latest timestamped state;
- prepare ownership transfer;
- install owner/control epoch;
- enter a declared command mode;
- send authorized target;
- acknowledge target at its supported level;
- enter safe hold;
- report watchdog status;
- release ownership through a prepared transition;
- close after an explicit safe response.

The gateway is the only process that opens the vendor connection.

### 11.3 ArmHoldController

ArmHoldController is separate from ArmGateway. It:

- captures the last accepted absolute teleoperation target;
- compares it with fresh measured state;
- acquires the gateway through an atomic or verified takeover;
- commands the selected arm's appropriate hold mode;
- runs at the required rate;
- reports pose/twist/following error;
- enforces bounded takeover and release times;
- remains active throughout hand blend and RL;
- does not release until teleoperation has prepared a new anchor.

The eventual arm is not eligible for live RL handoff until process-kill,
connection-loss, missed-deadline, following-error, and mode-transfer tests
prove a bounded safe response.

### 11.4 Development before arm selection

Work that can proceed now:

- contracts and capability model;
- fake arm gateway;
- deterministic simulated hold;
- Vive source and target generator;
- ownership and handoff state tests;
- UI and readiness flows;
- trace and replay.

Work that must wait:

- vendor SDK adapter;
- exact command rate;
- real workspace and joint limits;
- controller-side mode transfer;
- real hold gains;
- independent watchdog integration;
- HIL acceptance thresholds.

No Hitbot-specific behavior should be promoted into the generic contract merely
because the current prototype uses Hitbot.

## 12. Task readiness

### 12.1 Initial operator-confirmation workflow

For the first laboratory version, the operator:

1. selects a policy;
2. uses Manus and Vive teleoperation to reach the task posture;
3. observes live hand, arm, source-health, and policy-preflight status;
4. checks the task-specific checklist;
5. confirms object/fixture readiness;
6. arms the policy, which starts continuous shadow;
7. taps the switch key or pedal to request the transition;
8. the runtime completes the transition only after shadow and hold checks
   pass; otherwise it reports the failed gate and stays in teleoperation.

Operator confirmation is recorded with timestamp, operator/session identity,
policy package, and displayed evidence. It has a bounded validity period and is
invalidated by material movement, tracking loss, gateway restart, policy
change, or readiness-provider failure.

### 12.2 Machine-verifiable checks remain mandatory

Even without a camera, operator confirmation does not override:

- stale Manus or Vive input;
- stale arm or hand state;
- mismatched hand or policy identity;
- joint-limit or target-rate violation;
- policy codec/rate mismatch;
- arm motion above the handoff threshold;
- gateway or watchdog fault;
- missing effective-target evidence;
- expired transition deadline.

### 12.3 Provider aggregation

ReadinessPolicy declares required, optional, and advisory providers. Example:

    mounted screwdriver:
      required:
        operator confirmation
        fixture profile selected
        wrist transform within envelope
        low arm twist
        hand posture/contact proxy within envelope
        fresh states
      optional:
        camera fixture pose
        fingertip contact

    free-object rotation:
      required:
        operator confirmation
        gravity-relative wrist orientation
        hand posture and target-gap envelope
        low arm twist
        fresh states
      optional:
        object pose
        tactile contact
        drop detector

Adding a camera later requires:

1. a provider package;
2. configuration and calibration;
3. provider-specific tests;
4. a ReadinessPolicy update.

It does not require changes to PolicySession, CommandArbiter, the handoff state
machine, or hardware gateways.

## 13. Operator UI and physical controls

### 13.1 UI responsibilities

The local UI provides:

- connected-source and gateway health;
- Manus and Vive freshness;
- selected arm/hand identities;
- current owners and control epoch;
- available policies and compatibility;
- package preflight failures;
- task-specific readiness checklist;
- policy shadow metrics;
- arm-hold verification;
- hand-blend progress;
- active-policy status;
- abort, hand-back, and safe-hold requests;
- trace-recording status;
- clear fault reason and recovery steps.

The UI sends requests. It does not publish actuator commands.

### 13.2 Input roles

Recommended roles:

- keyboard/mouse: policy browsing, configuration, confirmation, and
  commissioning controls;
- switch key: one configurable key that toggles teleop-to-RL and
  RL-to-teleop through the state machine (tap, not hold);
- foot pedal: the connected PCsensor FootSwitch (USB 3553:b001) programmed to
  emit the same switch key. Recommended binding: F13, which no ordinary
  keyboard emits; read from
  /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd via evdev with an
  exclusive grab and debounce;
- independent robot E-stop: emergency actuation removal or controller-defined
  stop outside the Python/UI path.

The switch input is modeled as a timestamped OperatorSwitchSource emitting
press and release events, so a maintained-deadman mode can be added later as
task configuration without changing the state machine. A switch tap is a
request, not a command: every machine-verifiable gate in section 12.2 still
applies, and a tap during a fault or unmet gate produces a visible rejection,
never actuation.

Opening the hand is not a universal fault response.

### 13.3 UI failure

Loss of the UI must not remove gateway safety. The supervisor and watchdogs
continue enforcing deadlines. Depending on mode and deadman configuration, UI
loss causes a bounded safe hold or controlled hand-back request.

## 14. Ownership and mixed-control state machine

### 14.1 Resources

ARM and HAND have independent owners. Initial owner classes:

- none/disconnected;
- safety;
- teleoperation;
- arm hold;
- transition controller;
- selected RL policy.

Each ownership state carries:

- control session ID;
- resource;
- owner;
- control epoch;
- command mode;
- start and expiry time;
- gateway acknowledgement;
- watchdog state.

For the initial single-host runtime, structural exclusivity plus an
unforgeable local channel and gateway-enforced epoch is sufficient. If
arbitrary ROS, network, or multi-host producers later reach a gateway, the
protocol must add authenticated capability tokens and replay protection.

### 14.2 States

| State | ARM owner | HAND owner | Purpose |
|---|---|---|---|
| DISCONNECTED | none | none | No actuation |
| SAFE_HOLD | safety | safety or task-safe hand hold | Fault containment |
| TELEOP_ACTIVE | teleop | teleop | Manus/Vive operation |
| POLICY_PREFLIGHT | teleop | teleop | Artifact load and compatibility |
| RL_SHADOW | teleop | teleop | Exact-rate observation and preview; continuous while a policy is armed |
| ARM_HOLD_PREPARE | teleop transitioning to hold | teleop | Capture target and prepare mode |
| ARM_HOLD_VERIFY | arm hold | teleop | Verify stable deterministic hold |
| HAND_BLEND | arm hold | transition controller | Bumpless transfer to RL |
| RL_ACTIVE | arm hold | selected policy | Fine finger manipulation |
| HAND_BACK_PREPARE | arm hold | selected policy | Prepare fresh teleop target |
| HAND_BACK_BLEND | arm hold | transition controller | Bumpless return of hand |
| ARM_TELEOP_REANCHOR | arm hold | teleop | Re-anchor Vive to held/measured arm pose |
| ESTOP | external safety | external safety | Manual-reset emergency state |

Until real arm hardware exists, the ARM states are exercised against the fake
arm gateway: hold prepare and verify complete trivially, and the first
physical milestone is the hand-only cycle TELEOP_ACTIVE, HAND_BLEND,
RL_ACTIVE, HAND_BACK_BLEND, TELEOP_ACTIVE running through the same state
machine.

### 14.3 Arming: TELEOP_ACTIVE to RL_SHADOW

Entry requires:

- selected policy descriptor;
- successful package load and digest validation;
- compatible hand, side, schema, calibration, runtime, and rate;
- fresh hand state and effective teleop target;
- fresh Manus and Vive status;
- non-faulted gateways;
- current operator confirmation;
- task-specific readiness evidence.

On entry:

- reset PolicySession from measured q and effective teleop target;
- start exact-rate codec updates;
- record candidates but never authorize them;
- keep both resources under teleoperation;
- show shadow duration, history readiness, action magnitude, predicted target,
  and compatibility diagnostics in the UI.

The full required history must be populated with real effective teleoperation
targets. Repeating one frame may be used only if the package explicitly
declares it compatible.

Shadow is not a blocking ceremony: it starts when the policy is armed and
runs continuously alongside teleoperation until the policy is disarmed or a
gate invalidates it. A switch tap therefore finds a warm, history-complete
session, and transition latency is dominated by arm-hold verification and
the blend window alone.

### 14.4 RL_SHADOW to ARM_HOLD_VERIFY

The transition is requested by a switch tap (section 13.2) or a UI request.
Entry requires:

- enough consecutive valid policy ticks;
- no deadline or codec failure;
- policy candidate within configured handoff bounds;
- low arm motion;
- unexpired operator confirmation;
- unexpired readiness.

Transfer:

1. capture last accepted absolute arm target and measured state;
2. prepare the arm gateway and hold controller;
3. atomically switch the arm command mode/owner if supported;
4. command hold;
5. verify heartbeat, state freshness, pose/twist error, and following error;
6. leave hand ownership with teleoperation.

Any timeout returns to TELEOP_ACTIVE if ownership remains safe, otherwise to
SAFE_HOLD.

### 14.5 ARM_HOLD_VERIFY to HAND_BLEND

Entry requires stable hold for a configured dwell window. The transition
controller then acquires HAND at a policy tick boundary.

At every blend tick:

1. obtain the latest valid teleoperation semantic target;
2. synchronize policy history to the latest effective target;
3. compute a current policy preview;
4. compute time- and limit-bounded alpha;
5. blend from the latest effective command toward the current policy preview;
6. run safety, mapping preview, and round-trip diagnostics;
7. send through the exclusive hand gateway;
8. record the effective semantic target accepted at the required evidence
   level;
9. feed that effective target into the next policy observation.

The endpoint is live, not frozen at transition start.

### 14.6 HAND_BLEND to RL_ACTIVE

Completion requires:

- alpha reached one;
- final effective target evidence exists;
- no discontinuity or state-age violation;
- arm hold remains valid;
- operator confirmation remains valid;
- policy session activation and HAND ownership commit occur on the same
  declared tick without double-advancing history.

### 14.7 RL_ACTIVE

During RL_ACTIVE:

- the selected policy alone proposes finger commands;
- the arm hold controller alone commands the arm;
- a switch tap or UI request initiates hand-back; abort is always available;
- Manus and Vive keep running (mandatory, so hand-back is a prepared fast
  path), but they do not own hardware;
- policy, state, source, arm-hold, and watchdog deadlines remain enforced;
- a different UI policy selection is staged only; it cannot take ownership.

### 14.8 Hand-back

Hand-back is symmetrical:

1. validate a fresh Manus target;
2. prepare the retargeter and filter state;
3. blend from the latest effective RL command toward the same-tick teleop
   target;
4. transfer HAND to teleoperation;
5. establish a new Vive/arm anchor using the held and measured arm state;
6. verify the first teleoperation arm target is continuous;
7. transfer ARM from hold to teleoperation;
8. return to TELEOP_ACTIVE.

If Manus is unavailable, remain in task-aware hand hold while the arm remains
held. If Vive is unavailable after the hand returns, keep arm hold until a new
anchor is prepared.

## 15. Safety architecture

### 15.1 Layers

Safety is layered:

1. source validation;
2. semantic command validation;
3. state freshness and sequence checks;
4. task-specific limits;
5. mapping preview and round-trip diagnostics;
6. gateway authorization and deadline checks;
7. hardware/controller watchdog;
8. independent E-stop.

No higher layer substitutes for a missing lower layer.

### 15.2 Hand checks

At minimum:

- schema and calibration match;
- finite values;
- semantic position limits;
- per-tick and per-second target delta;
- target-to-measured following error;
- state freshness;
- command deadline;
- mapping saturation;
- semantic-native-semantic round-trip error;
- hardware fault/temperature when available;
- task-aware minimum grasp constraints during object manipulation.

### 15.3 Arm checks

Once the arm is selected:

- joint position, speed, acceleration, and effort;
- Cartesian workspace and orientation;
- singularity/proximity metrics if available;
- TCP speed and acceleration;
- following error;
- controller mode;
- state and acknowledgement freshness;
- communication and drive faults;
- collision or wrench limits when supported;
- hold pose/twist error;
- watchdog and E-stop status.

### 15.4 Fault responses

Safe response is task and hardware specific:

| Fault | Initial response |
|---|---|
| Manus stale during TELEOP_ACTIVE | Hold last safe hand target; require recovery confirmation |
| Vive stale during TELEOP_ACTIVE | Arm safe hold; hand may remain teleop if configured |
| Policy deadline missed | Keep arm hold; hold last safe hand target; leave RL ownership |
| Hand state stale | Hold or controller-safe response; do not continue policy integration |
| Arm state stale | Controller/watchdog hold or stop; block handoff |
| Arm hold error | Stop hand transition; task-aware hand hold; escalate |
| Mapping/calibration mismatch | Refuse connection/activation |
| UI lost | Follow configured safe response; gateways remain safe |
| Supervisor killed | Independent gateways/watchdogs enforce expiry |
| Gateway killed | Controller/hardware watchdog response |
| E-stop | Immediate external response and manual reset |

All transition timeouts and fault responses are idempotent.

## 16. Training and sim-to-real changes

### 16.1 Explicit task-from-wrist

Each TaskHandProfile must define:

- task frame;
- wrist/hand-base frame;
- desired task-from-wrist transform;
- allowed position and orientation envelope;
- allowed wrist twist at activation;
- gravity-relative orientation;
- object/fixture assumptions;
- expected contact or target-gap conditions.

The current fixed absolute hand poses become derived profile data rather than
implicit world coordinates.

### 16.2 Wrist perturbation and compliance

Training/evaluation should introduce:

- bounded wrist position and orientation perturbation;
- gravity-relative orientation variation within the real start envelope;
- held-arm compliance or small pose disturbances;
- latency and target-hold effects;
- hardware position limits and mapping quantization;
- measured-versus-target lag;
- state noise and missed samples within accepted bounds.

These variations are behavior changes and require explicit experiments.

### 16.3 Handoff-start distribution

Recorded teleoperation sessions should produce a dataset containing:

- measured semantic q;
- effective teleoperation targets;
- target-to-measured gap;
- Manus retargeter diagnostics;
- wrist/task transform;
- source and state timing;
- operator confirmation;
- readiness outcomes;
- success/failure labels where available.

This data is used to:

- evaluate existing policy start coverage;
- tune start envelopes;
- replay codec histories;
- augment reset and history distributions;
- fine-tune or retrain policies when necessary.

### 16.4 Mounted screwdriver

The mounted task is recommended for first physical commissioning because:

- the fixture constrains the object pose;
- task-from-wrist can be measured against a stable reference;
- a dropped free object is not an additional failure mode;
- the current actor input is simpler than the in-hand 96-value actor path.

This does not remove the need for:

- exact 10 Hz timing;
- current-target adoption;
- mapping validation;
- arm-hold verification;
- contact/posture envelope;
- task-frame calibration.

### 16.5 Free-object rotation

Free-object commissioning additionally requires:

- corrected 96-value actor assembly and scaling;
- validated 30-frame history;
- a start envelope including gravity-relative wrist orientation;
- target-to-measured grasp squeeze evidence;
- an operator workflow for object-present and object-not-dropped confirmation;
- conservative safe-hold behavior;
- preferably a later tactile or visual drop detector.

## 17. Configuration and packaging

### 17.1 Runtime installation profiles

Recommended dependency groups:

- core: contracts, runtime, policy validation, CPU inference;
- ui: local UI and status transport;
- manus: ROS 2 message/source adapter;
- vive: tracker adapter;
- retargeting: pinned dex-retargeting;
- linker: pinned Linker SDK and CAN dependencies;
- arm-vendor-name: future arm adapter;
- dev: tests, lint, typing, trace tools;
- hil: protected hardware test utilities.

All groups install from the one repository via the bootstrap script. The
pinned Linker SDK must include the G20 CAN driver, and dex-retargeting is
pinned by version and digest.

Isaac Lab is not a runtime dependency.

### 17.2 Configuration

Hard-coded IP addresses, CAN channels, sides, transforms, rates, and torque
settings move into validated deployment configuration. A DeploymentBinding
selects:

- hardware identities;
- transports;
- calibration artifacts;
- source profiles;
- policy stores;
- scheduler rates;
- safety limits;
- readiness policy;
- deadman behavior;
- logging destination;
- safe responses.

Unknown or missing safety-critical fields fail before hardware connection.

### 17.3 Entrypoints

Illustrative entrypoints:

    dex-runtime preflight CONFIG
    dex-runtime run CONFIG
    dex-runtime list-policies CONFIG
    dex-runtime replay TRACE
    dex-runtime verify-package PACKAGE
    dex-runtime calibrate-linker CONFIG
    dex-runtime ui CONFIG

Training and export remain dex-forge commands.

## 18. Observability and replay

### 18.1 Event log

Every transition and rejection is structured:

- timestamp;
- session;
- state and requested transition;
- owner and control epoch;
- policy package;
- readiness snapshot;
- reason code;
- deadline;
- gateway acknowledgements;
- safe response;
- operator action.

### 18.2 Control trace

The trace records, at bounded rates:

- Manus and Vive sample metadata;
- retargeted hand candidate;
- arm target candidate;
- measured hand and arm state;
- policy codec input;
- policy latent/output;
- teleop and policy candidates;
- arbitration result;
- blend alpha;
- safety decision;
- native mapping preview;
- effective semantic target;
- acknowledgement evidence;
- scheduler lateness;
- fault and UI events.

Large raw images or high-rate vendor packets may be referenced externally by
digest rather than embedded.

### 18.3 Replay modes

- source replay: repeat Manus/Vive samples through retargeting;
- policy replay: repeat measured/effective-target histories;
- supervisor replay: repeat candidates and states through the state machine;
- mapping replay: verify semantic/native golden vectors;
- fault replay: inject recorded deadline and connection failures.

Replay never connects to live hardware unless a separate protected HIL command
is explicitly used.

## 19. Testing strategy

### 19.1 Pure contract tests

- schema serialization and version rejection;
- identity mismatch;
- timestamp and deadline behavior with fake clocks;
- readiness aggregation;
- ownership epoch transitions;
- idempotent timeouts;
- policy descriptor filtering.

### 19.2 Mapping tests

- physically verified semantic/native golden vectors;
- thumb slot 5/10 regression;
- semantic-native-semantic round trip;
- saturation and quantization;
- calibration digest mismatch;
- retargeter name projection;
- proof that teleop and RL use the same adapter.

### 19.3 Policy parity tests

- simulator codec versus runtime codec;
- actor/adapter tensor shape;
- normalization buffers;
- reset from measured q and effective target;
- exact 10 Hz and 20 Hz trace replay;
- preview/activate no-double-step behavior;
- package rejection for wrong hand, rate, codec, or calibration.

### 19.4 Teleoperation tests

- recorded Manus deterministic retargeting;
- Vive anchoring and re-anchoring;
- independent source loss;
- queue/latest-buffer overflow behavior;
- tracking recovery without accumulated jump;
- filter reset;
- no actuator imports in source packages.

### 19.5 Supervisor and fault tests

- every state transition and timeout;
- process death before and after each ownership commit;
- stale source/state;
- policy overrun;
- mapping rejection;
- arm-hold error;
- UI loss;
- deadman release;
- prepared hand-back with missing Manus or Vive;
- competing command rejection;
- supervisor restart and stale epoch rejection.

### 19.6 Hardware-in-the-loop ladder

1. identity and read-only state;
2. mapping preview without sending;
3. low-power named-joint wiggle;
4. teleoperation through exclusive hand gateway;
5. policy shadow;
6. fake/simulated arm hold;
7. selected real arm read-only state;
8. real arm teleoperation with limits;
9. real arm hold and process-kill test;
10. hand blend without object;
11. hand-back without object;
12. mounted screwdriver at conservative limits;
13. free-object hold and abort;
14. free-object rotation.

## 20. Implementation program

### Adopted thin critical path

The full phase list below remains the target architecture, but the decided
delivery order optimizes time-to-first-hardware-handoff:

- M0 — mapping freeze: immutable calibration artifact from the verified
  dex-forge mapping, thumb slot 5/10 golden regression, canonical URDF
  decision (from Phases 0 and 2).
- M1 — teleop through the gateway: runtime repo skeleton with the
  dex_contracts subpackage, exclusive Linker gateway on its own thread,
  ManusHandSource and ManusRetargeter with name-based projection (the
  qpos_to_cmd slot map is deleted; the -10 degree thumb bias is explicitly
  kept or dropped in the TeleopProfile), live Manus teleoperation of the real
  hand (from Phases 1, 2, 4, 6).
- M2 — hand-only switching demo: ProprioCodec and effective-target reset
  corrections, PolicySession, continuous shadow, tap-to-toggle blend and
  hand-back against the fake arm gateway; first physical
  teleop-to-RL-to-teleop cycle on the real hand (from Phases 3, 5, 7).
- M3 — operability: JSONL event and trace logging, terminal status display,
  pedal bound to the switch key (from Phase 5 and section 18).

Deferred until after M3 or until hardware exists: the dex_ui package, the
multi-process split, the replay harness (traces are recorded from M2; the
replay tooling comes later), authenticated capability tokens, and all
arm-vendor work (Phase 8 and the arm gates are unchanged).

### Phase 0: Preserve and re-baseline

Deliverables:

- repository rename recorded separately from Python package rename;
- current dirty-state and artifact inventory;
- source commit and config digests;
- dex_teleop migration inventory;
- authoritative Linker mapping/calibration frozen;
- canonical URDF decision;
- runtime repository name accepted.

Exit criteria:

- no user changes lost;
- all remotes and branch tracking work after rename;
- no ambiguity about the mapping source of truth.

### Phase 1: Create runtime skeleton and contracts

Deliverables:

- new neutral repository;
- dex-contracts package;
- state, candidate, identity, readiness, package, and capability schemas;
- fake clocks and fake gateways;
- CI and package build;
- dependency rules.

Exit criteria:

- contracts install without Isaac, ROS, CAN, or vendor SDK;
- forbidden imports are enforced.

### Phase 2: Extract canonical hand assets and mapping

Deliverables:

- superset Linker URDF with fingertip frames;
- immutable calibration artifact;
- pure Linker adapter;
- mapping golden tests;
- pinned SDK boundary;
- fake transport.

Exit criteria:

- the same semantic vector maps identically for teleop and RL;
- physical thumb regression passes.

### Phase 3: Correct policy runtime and packages

Deliverables:

- shared ProprioCodecs;
- mounted and in-hand trace fixtures;
- corrected in-hand 96-value actor input;
- effective-target reset;
- PolicySession lifecycle;
- immutable package exporter and validator;
- package registry.

Exit criteria:

- simulator and runtime outputs match within declared tolerance;
- incompatible artifacts fail before gateway connection.

### Phase 4: Extract Manus and Vive adapters

Deliverables:

- ManusHandSource;
- ManusRetargeter;
- ViveArmSource;
- ArmTargetGenerator;
- recorded-input fixtures;
- independent source-health model.

Exit criteria:

- output parity with selected dex_teleop recordings;
- source packages have no actuator access.

### Phase 5: Supervisor, UI, and manual readiness

Deliverables:

- state machine;
- owner/control-epoch enforcement;
- policy-selection UI;
- OperatorConfirmationProvider;
- machine readiness providers;
- keyboard commissioning controls;
- optional foot-pedal interface;
- structured events and traces.

Exit criteria:

- full workflow runs with fake hand and fake arm;
- selecting a policy cannot actuate.

### Phase 6: Exclusive Linker gateway

Deliverables:

- sole CAN owner;
- prepared command path;
- state freshness;
- acknowledgements at truthful capability level;
- command and process watchdog behavior;
- explicit safe shutdown;
- migrated teleoperation through the gateway.

Exit criteria:

- legacy scripts cannot compete during the production launch;
- process-kill tests reach the declared safe response;
- teleoperation behavior remains acceptable.

### Phase 7: Generic arm handoff with fakes

Deliverables:

- ArmCapabilities;
- fake ArmGateway;
- ArmHoldController;
- Vive re-anchor;
- ARM ownership transition tests;
- arm-loss fault matrix.

Exit criteria:

- complete shadow, hold, blend, active, and hand-back cycle passes without a
  vendor arm.

### Phase 8: Selected arm adapter

Trigger: the next arm and controller are selected.

Deliverables:

- vendor adapter;
- measured limits and rates;
- mode/hold implementation;
- controller watchdog and E-stop integration;
- HIL conformance evidence.

Exit criteria:

- every claimed capability passes;
- process-kill and connection-loss responses are bounded;
- no live hand blend is allowed before this gate.

### Phase 9: Training-domain alignment

Deliverables:

- task-from-wrist profiles;
- wrist perturbation/compliance evaluation;
- teleoperation handoff recordings;
- start-envelope analysis;
- reset/history augmentation where required;
- promoted compatible policies.

Exit criteria:

- held-out handoff starts meet task-specific coverage and safety criteria.

### Phase 10: Physical task commissioning

Recommended order:

1. mounted screwdriver;
2. free-object rotation.

Both use the same runtime. Task-specific differences live in PolicyPackage,
TaskHandProfile, ReadinessPolicy, and SafetyProfile.

### Phase 11: Optional perception

Trigger: camera or other sensor hardware becomes available.

Deliverables:

- provider adapter;
- calibration;
- timestamp and uncertainty model;
- provider tests;
- readiness-policy update;
- UI visualization.

Exit criteria:

- provider failure degrades to explicit unknown/fail behavior;
- no changes are required in the ownership state machine or command gateways.

## 21. Release gates

No live mixed-control release is permitted until:

- exactly one hand gateway and one arm gateway can actuate;
- the verified Linker mapping is the only production mapping;
- policy package identity and codec validation are strict;
- mounted and in-hand policy parity tests pass for any released package;
- teleoperation-to-policy target discontinuity stays within a declared bound;
- policy-to-teleoperation return stays within a declared bound;
- arm takeover, hold, and release stay within pose/twist/following-error bounds;
- stale input and stale state produce bounded safe responses;
- supervisor and gateway process-kill tests pass;
- the independent E-stop path is documented and exercised;
- the UI displays the actual owner, policy, readiness, and fault state;
- a complete trace can reconstruct every ownership and command decision;
- rollback to teleoperation-only operation is documented and tested.

## 22. Naming and migration policy

### 22.1 Repository names

- ScrewdriverRL becomes dex-forge.
- The local checkout becomes dex-forge.
- The neutral runtime working name is dex-manipulation.
- dex_teleop keeps its name while serving as the migration reference.

### 22.2 Git compatibility

The repository rename procedure must:

1. rename the GitHub repository;
2. update origin to the new canonical SSH URL rather than relying only on a
   GitHub redirect;
3. preserve all branches, tags, issues, releases, actions, and repository ID;
4. rename the local folder without touching the worktree content;
5. verify status, branch upstream, fetch, and remote resolution;
6. avoid staging, committing, stashing, rebasing, or cleaning unrelated dirty
   files.

### 22.3 Python package names

The existing screwdriver_rl import remains initially. A future dex_forge
package migration should:

- introduce dex_forge modules;
- keep a compatibility import layer for a declared period;
- update entrypoints and artifact provenance;
- avoid changing policy behavior;
- provide deprecation warnings;
- remove the old import only after downstream users and artifacts migrate.

Repository branding and Python API compatibility are separate changes.

## 23. Open decisions

The following remain open:

1. Final GitHub name for the neutral runtime repository. This document uses
   dex-manipulation as a working name.
2. Selected arm vendor, controller, and safety capabilities.
3. Vive tracker acquisition stack for ViveArmSource: SteamVR/openvr, OpenXR
   with the tracker extension, or libsurvive. No migration source exists
   (section 4.2).
4. Final switch-key binding (F13 recommended for the pedal) and whether a
   maintained-deadman mode is ever enabled per task.
5. Final start-envelope thresholds for each task.
6. Whether camera perception is added before or after the first free-object
   release.
7. Final policy package tensor format and signing/trust requirements.
8. Whether the recommended mounted-then-free-object commissioning order is
   accepted; the architecture itself does not depend on the order.

## 24. Decisions recorded in this revision

| Decision | Status |
|---|---|
| Manus glove is the initial hand input | Accepted |
| Vive tracker is the initial arm input | Accepted |
| Current Hitbot code does not define the future arm contract | Accepted |
| Arm integration remains capability-driven until hardware selection | Accepted |
| Operator UI is required because multiple policies may be installed | Accepted |
| Keyboard/UI is allowed for commissioning | Accepted |
| Switching is tap-to-toggle on a configurable key; the PCsensor foot switch binds to the same key | Accepted (2026-07-16 review) |
| Arm remains under deterministic hold during RL execution | Confirmed (2026-07-16 review) |
| Armed policies run continuous shadow for fast switching | Accepted (2026-07-16 review) |
| Runtime is one self-contained repository; no published contracts wheel | Accepted (2026-07-16 review) |
| Thin critical path M0-M3 precedes the full phase program | Accepted (2026-07-16 review) |
| First physical milestone is hand-only switching with a fake arm gateway | Accepted (2026-07-16 review) |
| dex-forge Linker mapping is physically verified and authoritative | Accepted |
| Initial task readiness uses operator confirmation | Accepted |
| Camera and other sensors use optional readiness providers | Accepted |
| Integrated live runtime belongs in a neutral repository | Accepted |
| ScrewdriverRL repository is renamed dex-forge | Accepted |
| Existing draft is not overwritten | Accepted |

## 25. Definition of success

The architecture is successful when an operator can:

1. start one validated runtime configuration;
2. see healthy Manus, Vive, hand, and arm connections;
3. teleoperate arm and hand through exclusive gateways;
4. select one compatible promoted policy in the UI;
5. confirm task readiness;
6. observe successful policy shadow and arm-hold verification;
7. transfer the hand to RL without a visible or measured command jump;
8. execute the precise task while the arm remains stably held;
9. request hand-back or release the deadman;
10. regain teleoperation without a jump;
11. recover from source, policy, UI, or process faults through a bounded,
    explainable safe response;
12. replay the session and account for every selected command.

The engineering implementation is successful when adding a camera, a new arm,
or a new policy requires a new provider/adapter/package and configuration, not
a rewrite of the supervisor or handoff state machine.
