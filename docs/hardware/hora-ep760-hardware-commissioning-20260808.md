# HORA epoch-760 hardware commissioning — 2026-08-08

## Release inputs

- Policy: `runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth`
- Policy SHA-256: `3079f57c71b8de3a39911daf65eaf28597c4ff6c884da0f5ccf0e392532235c0`
- Calibration: `linker_calib_deploy.json`
- Calibration SHA-256: `b728cd58f15408193080a0bd35dc6ae281adff9a96d6115ded012498b6f7ecb7`
- Hand: LinkerHand G20 left, serial `LHT20-010-415-L-B-1-D`
- SDK root: `/home/user/linkerhand-ros-sdk`, commit `2aa379cd11562d953f8b449561107b58c120676e`
- Required patched G20 CAN source SHA-256: `513be964dee481773ad4d346559e59912b3b70c79c920528783648631f9e10b9`
- Required SDK setting SHA-256: `029cb8ea5e96542f118dd68135aa0811ec176998f375f86b9734b982563dda2c`

This policy was trained directly in the promoted G20 OG-local-q coordinate
contract. Do not pass a legacy `--policy-coordinate-adapter`.

## Current readiness

- Simulator acceptance: PASS (see `TRAINING_REPORT_20260807.md`).
- Deploy/SDK contract tests: PASS.
- Exact bundle offline load and 10 Hz execution: PASS.
- Offline echo behavior: rail-lock at policy step 24. Echo has no screwdriver
  contact dynamics, so this is not a hardware rejection; it requires bounded
  hardware gates and forbids an unbounded first live run.
- Live CAN verification: BLOCKED until `can0` is up and the hand is powered.
  At preparation time `can0` was `DOWN`, CAN state `STOPPED`, RX/TX packets 0.

## Operator prerequisites

1. Physical emergency stop is reachable and tested independently of Python.
2. The 64 mm fixture, wrist frame, palm-down orientation, and screwdriver axis
   match the reviewed task frame.
3. One operator watches the hand and tool; another can stop the process or power.
4. No ROS LinkerHand node or second CAN consumer is running.
5. Start with the tool installed. Free-air policy behavior is out of distribution.
6. Do not use `--release` during initial policy gates; default HOLD avoids a
   blind open or dropping the tool after a fault.

Run every command below from `/home/user/dex-forge`.

## Gate 0 — artifact and CAN identity

```bash
sha256sum \
  runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth \
  linker_calib_deploy.json \
  /home/user/linkerhand-ros-sdk/linker_hand_sdk_ros/scripts/LinkerHand/core/can/linker_hand_g20_can.py

/home/user/linkerhand-ros-sdk/find_can.sh
ip -details -statistics link show can0
```

Required: `can0` is UP/ERROR-ACTIVE at 1 Mbit/s, counters are not climbing with
bus errors, and the three hashes exactly match the release inputs above.

## Gate 1 — query-only CAN preflight (no motion command)

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth \
  --side left --transport can --hand-joint G20 --can can0 \
  --sdk-root /home/user/linkerhand-ros-sdk \
  --expected-serial LHT20-010-415-L-B-1-D \
  --calib linker_calib_deploy.json --device cpu \
  --no-send --max-ticks 20 \
  --record records/hora_ep760_hardware_commissioning_20260808/01_nosend_20ticks.csv
```

Required: identity and firmware are non-empty, touch type is reported, every
position/fault query receives five fresh finger frames, all unmasked fault bytes
are zero, exit code is 0, held reads are 0, and no control overrun is reported.
The 20-tick bound is intentional: fixed measured state in no-send exaggerates
closed-loop target drift and may eventually rail-lock.

## Gate 2 — startup-only posture (moves hand, never enables policy)

Only run after visually confirming the fixture/task frame.

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth \
  --side left --transport can --hand-joint G20 --can can0 \
  --sdk-root /home/user/linkerhand-ros-sdk \
  --expected-serial LHT20-010-415-L-B-1-D \
  --calib linker_calib_deploy.json --device cpu \
  --startup-only --speed 40 --torque 80 \
  --ramp-s 5 --contact-ramp-s 8 \
  --record records/hora_ep760_hardware_commissioning_20260808/02_startup_only.csv
```

Required: smooth collision-free approach, correct thumb yaw/roll directions,
all five fingertips plausibly contact the fixture, no rail-lock/fault, no tool
drop, and the final posture visually matches the reviewed simulator posture.
Because startup-only rows do not contain measured q/state, repeat Gate 1 for five
ticks afterward with a new output filename to capture post-startup readback.

## Gate 3 — first closed-loop policy pulse (moves hand)

Only after Gates 0–2 pass and the operator explicitly confirms the task frame.

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m screwdriver_rl.deploy.deploy_linker \
  --checkpoint runs/Isaac-LinkerL20-Screwdriver-Rotation-Topdown-Hora/full_stage1_20260807_env12288/stage2_nn/deploy.pth \
  --side left --transport can --hand-joint G20 --can can0 \
  --sdk-root /home/user/linkerhand-ros-sdk \
  --expected-serial LHT20-010-415-L-B-1-D \
  --calib linker_calib_deploy.json --device cpu \
  --task-frame-confirmed --speed 40 --torque 80 \
  --ramp-s 5 --contact-ramp-s 8 --max-ticks 3 \
  --record records/hora_ep760_hardware_commissioning_20260808/03_live_3ticks.csv
```

Analyze immediately:

```bash
PYTHONPATH=/home/user/dex-forge \
/home/user/miniconda3/envs/env_isaaclab/bin/python -m tools.analyze_deploy_record \
  records/hora_ep760_hardware_commissioning_20260808/03_live_3ticks.csv \
  --calib linker_calib_deploy.json --phase policy --last 0
```

Stop and do not extend the run on any wrong-direction motion, unexpected
object-side contact, non-zero fault, missing fresh frame, raw command/state error
above 12 counts, four-joint rail-lock, tool drop, sudden jump, or abnormal heat.
If the 3-tick evidence passes, increase only in reviewed stages: 10, 20, then 50
ticks, with a new CSV and synchronized camera video for every stage.

## Recovery

After a watchdog or fault, HOLD is the default. Do not blind-open. Remove load or
support the tool, verify fresh state and zero faults, then use `--release-only`
with `--speed 40 --torque 80 --ramp-s 5` if an open-pose handoff is safe.
