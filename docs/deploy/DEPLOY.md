# Deploying on the physical LinkerHand L20/G20 (left)

End-to-end guide from a trained Stage-2 checkpoint to the real hand. Design
details live in `docs/3-deployment.md`; this is the operational runbook.

The hand connects to a **deploy box** (any Linux machine with a USB-CAN
adapter) that does not need Isaac, rl_games, or a GPU — `DeployPolicy` is
plain PyTorch. Two transports:

| transport | needs | status |
|---|---|---|
| `can` (default) | LinkerHand SDK checkout + `python-can` + CAN up | primary, recommended |
| `ros` | ROS1 Noetic + the SDK's `linker_hand.launch` node | secondary, code-complete but not hardware-tested |

> The user's hand reports as **G20** (`--hand-joint G20`, the default).
> G20 and L20 are physically identical; the SDK just uses different CAN classes.

## 1. Deploy-box setup

### Option A — direct CAN, no ROS (recommended)

Any Ubuntu with Python **3.10+**:

```bash
git clone <ScrewdriverRL>            # or copy the repo
git clone <linkerhand-ros-sdk>
python3 -m venv ~/venvs/deploy && source ~/venvs/deploy/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel is enough
pip install python-can pyyaml
export PYTHONPATH=/path/to/ScrewdriverRL          # pip install -e . also works on ≥3.10
export LINKERHAND_SDK_ROOT=/path/to/linkerhand-ros-sdk
```

Copy from the training box: only `runs/<task>/<run>/stage2_nn/deploy.pth`
(self-contained: actor + adapter + normaliser + joint bounds + pregrasp).

> **This box (2026-07):** the training machine doubles as the deploy box — a
> PEAK PCAN-USB adapter is attached (`can0`). Use the `env_isaaclab` conda env
> (already has torch + `screwdriver_rl` editable; `pip install python-can` was
> the only missing dep), and the SDK checkout at
> `/home/user/linkerhand-ros-sdk` (also the built-in default of
> `hw_utils.bootstrap_sdk`, so `$LINKERHAND_SDK_ROOT` is optional here).

### Option B — ROS1 Noetic box (Python 3.8)

The deploy modules are 3.8-compatible (future-annotations, torch-only imports),
but the package metadata requires ≥3.10 — so **use `PYTHONPATH`, not
`pip install -e .`**. Install torch + `pip install -r <sdk>/requirements.txt`,
build/launch the SDK node
(`roslaunch linker_hand_sdk_ros linker_hand.launch`), then run `deploy_linker
--transport ros`. Speed/torque are applied via `/cb_hand_setting_cmd`.

### SDK configuration (both options)

Edit `<sdk>/linker_hand_sdk_ros/scripts/LinkerHand/config/setting.yaml`:

- `PASSWORD`: the machine's sudo password — the SDK uses it to auto-`ip link set
  can0 up` from Python;
- `LINKER_HAND: LEFT_HAND: EXISTS: True`, `JOINT: G20` (the `JOINT` field only
  matters for the ROS node; the CAN path takes `--hand-joint` directly).

### CAN bring-up

```bash
sudo apt install can-utils
bash <sdk>/find_can.sh                       # discover adapter + hand
# or manually:
sudo ip link set can0 up type can bitrate 1000000
candump can0 -n 5                            # sanity: frames flowing
```

Left hand answers on CAN id `0x28`, right on `0x27`. `LinkerHandApi` exits
immediately if the interface is down.

## 2. Producing `deploy.pth` (training box)

```bash
# Stage 2 (adaptation) from the best Stage-1 checkpoint — writes stage2_nn/deploy.pth:
python train.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0 --stage 2 \
    --checkpoint runs/Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0/<run>/nn/<best>.pth

# Sim gate — the exact deploy-time inference (predicted latent, no privileged obs).
# Compare against the oracle run; a modest NetTurns/fall-rate gap is the go signal:
python eval.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0 \
    --checkpoint runs/<task>/<run>/nn/<best>.pth \
    --deploy_eval --adapter_checkpoint runs/<task>/<run>/stage2_nn/deploy.pth --num_envs 256
```

> ⚠️ **`deploy.pth` is overwritten by every Stage-2 run for the task, including
> short smoke runs** (`stage2_nn/` is task-level, not per-run, and the best-loss
> tracking resets each run — a 1-iteration debug run leaves a garbage bundle
> behind). Before deploying, always check the bundle's provenance:
>
> ```python
> b = torch.load("stage2_nn/deploy.pth", map_location="cpu")
> print(b["iter"], b["loss"])   # a real run shows the final iter (e.g. 150), not 1
> ```
>
> `deploy_last.pth` (written alongside) holds the final-iteration bundle of the
> same run and is the fallback when `deploy.pth` was clobbered — e.g. as of
> 2026-07-06 the base task's trained bundle is
> `runs/Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0/stage2_nn/deploy_last.pth`
> (iter 150, Jun 30); both tasks' `deploy.pth` are iter-1 smoke artifacts.

> ⚠️ **Run the sim gate in the env the policy was trained in.** Env-cfg edits
> after training (e.g. the 2026-07-02 palm move + posture re-solve, commits
> `1ff78b2`/`5811398`) silently invalidate older checkpoints: the Jun-30 policy
> scores NetTurns +2.35 / 0% falls at its training commit (`bb5e2c0`) but
> ~0 turns / 98% falls at HEAD — in eval, play.py, *and* resumed training.
> When gating a checkpoint older than the last env change, run `eval.py` from a
> `git worktree` at the training-time commit. To find that env: the bundle's
> `config.home_targets` fingerprint the training posture (compare against
> `pregrasp_positions` across commits, full precision). Deploying such a bundle
> is still fine — it carries its own pregrasp/bounds — but a policy for the
> *current* cfg requires retraining.

## 3. Offline dry-run (any machine, no hardware)

```bash
python -m screwdriver_rl.deploy.deploy_linker --checkpoint <stage2_nn/deploy.pth> \
    --hand-joint L20 --calib linker_calib_deploy.json --dry-run \
    --max-ticks 50 --ramp-s 1 --contact-ramp-s 1 --record /tmp/dry.csv
```

Runs the full session (safe approach → contact ramp → policy loop → shutdown) against a perfect-tracking
echo simulator and records the CSV. Commands must stay inside 0..255 with slots
11–14 at 0.

## 4. First-power-on checklist (deploy box, strict order)

Keep fingers clear of the hand whenever it is powered. Start with conservative
`--speed 40 --torque 80` and no screwdriver mounted.

```bash
alias hc='python -m screwdriver_rl.deploy.hand_check'

hc info                       # 1. serial/version/state sanity; wrong-side check
hc echo                       # 2. flex a finger BY HAND: values must track it
                              #    (proves get_state() measures, not echoes)
hc ramp --ramp-s 5            # 3. slow open ⇄ pregrasp cycle
hc wiggle --out linker_calib.json
                              # 4. guided per-joint direction test (~2 min).
                              #    Answers y/n per joint; writes the calibration
                              #    overlay. Abduction + thumb signs are exactly
                              #    what this pins down.
hc pose --calib linker_calib.json
                              # 5. hold pregrasp; visually compare against the
                              #    sim render (render_posture.py). Top-down
                              #    bundles require the 1.08-rad PIP URDF/calibration
                              #    contract. thumb_cmc_roll may still use fraction
                              #    mapping because its SDK and URDF zeros differ.
hc roundtrip --calib linker_calib.json
                              # 6. command↔measure per-joint error table (exit 1
                              #    over --tol 0.15 rad)
```

All subcommands accept `--dry-run` to rehearse without hardware.

## 5. Live run

```bash
# 7. Pre-flight: policy runs on live state, nothing is sent. The history is
# seeded from the measured, uncommanded state (never from a fictitious home ACK):
python -m screwdriver_rl.deploy.deploy_linker --checkpoint <deploy.pth> \
    --hand-joint L20 --calib linker_calib_deploy.json \
    --no-send --max-ticks 50 --record preflight.csv

# 8. Top-down first live run: mount the exact fixture/tool, keep it bounded,
# use conservative drive settings and slower-than-default startup ramps:
python -m screwdriver_rl.deploy.deploy_linker --checkpoint <deploy.pth> \
    --hand-joint L20 --calib linker_calib_deploy.json --speed 40 --torque 80 \
    --ramp-s 5 --contact-ramp-s 8 --max-ticks 100 --record run1.csv

# 9. Only after reviewing the bounded CSV and emergency-stop rehearsal:
python -m screwdriver_rl.deploy.deploy_linker --checkpoint <deploy.pth> \
    --hand-joint L20 --calib linker_calib_deploy.json --record run2.csv
```

Session behaviour:

- **Startup**: reads state, ramps current→the bundle's collision-safe reset over
  `--ramp-s`, then reset→contact home over `--contact-ramp-s` (both default 3 s),
  re-reads, and seeds history from measured joints plus the calibration-quantized
  acknowledged home. Legacy bundles without a separate reset keep one ramp.
- **Loop**: uses the control rate declared by the bundle ProprioCodec (10 Hz for
  mounted Linker policies); a conflicting `--hz` override is rejected. Targets are
  hard-clamped to the bundle's `home ± 0.35 rad` motion window, and each tick
  moves ≤ 0.05 rad per joint.
- **Watchdog**: `--stale-limit` (default 10) consecutive invalid state reads →
  hold last command, exit code 2.
- **Ctrl+C = HOLD** (stop sending; the servo keeps the last target — safest
  while gripping a tool). `--release` instead ramps to the open pose, on clean
  stops only, never after a watchdog trip.
- `--record` CSV: `tick, phase, t_wall, t_mono, q0..15, tgt0..15, act0..15,
  cmd0..19, state0..19` — plot `q` vs `tgt` to check tracking; persistent lag
  means `--speed/--torque` are too low.

## 6. Bring-up findings (2026-07-06, hand LHT20-010-415-L-B-1-D)

The full checklist was executed on the real hand. Outcomes and fixes:

- **Two G20 SDK bugs found and patched** in the SDK checkout
  (`core/can/linker_hand_g20_can.py`; ⚠️ **uncommitted — commit/upstream them**,
  a fresh SDK checkout resurrects both):
  1. `joint_state_to_cmd_state` scattered ring/little "reserved" values into
     slots 15/16, permanently zeroing the thumb/index **tip state**;
  2. `cmd_range_to_joint_range` had the thumb row as `[10, 5, ...]` — thumb
     **rotation/side-bend commands swapped** vs the state layout (verified on
     hardware via `wiggle`).
- **Calibration**: all 16 joints verified, **no flips needed**
  (`linker_calib.json`). Roundtrip worst error 0.008 rad (tol 0.15).
- **Aperture distortion measured**: at gait targets the fraction map leaves the
  fingertips 0.20–0.36 rad straighter than sim and the thumb tip 0.13 rad more
  closed. `linker_calib_absolute.json` adds `lo`/`hi` overrides that make the
  bend/abduction axes **absolute-angle** instead; A/B the two overlays with
  `hc pose --calib <overlay> --bundle <deploy_last.pth>` against the sim render
  and deploy with the better-matching one.
- **Hardware range limit**: the real tip bend saturates at the SDK's 1.08 rad
  while the training URDF allows 1.57 (policy window reaches ≈1.41). Under the
  absolute overlay such commands clamp (the truth); under the fraction map they
  silently rescale. **Recommendation for the next training run: set the URDF
  pip limits to the hardware's 1.08 rad** (and re-solve the pregrasp) so sim
  and hand share one geometry.
- **Bundle deployed**: `deploy_last.pth` (Jun 30, iter 150). Sim gate at its
  training commit `bb5e2c0`: oracle +2.35 NetTurns / 0% falls / 94.7% success;
  deploy-eval +1.89 / 0% falls / 63.3% — PASS. Live runs (100 + 600 ticks,
  10 Hz): zero stale reads, zero overruns; gait fires with 2–4 s cycles.
  Free-air behavior is out-of-distribution for the adapter — judge only with a
  tool mounted (base variant: handle Ø40 × 100 mm, ~300 g, Ø10 mm shaft).

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `LinkerHandApi` exits at startup | CAN down — `find_can.sh`, `ip link`, re-plug USB-CAN |
| state reads are `[-1]*20` | CAN RX cache not populated / parse failure — reseat adapter, check bitrate 1 Mbps |
| watchdog trips immediately | wrong `--can` channel, or hand on the other id (left `0x28` / right `0x27`) |
| a joint moves the wrong way | rerun `hand_check wiggle` for that joint; check the overlay is passed via `--calib` |
| thumb behaves inverted with an old calibration | tables predating the fix used the SDK's stale `range_to_arc` module (slots 0/10 inverted) — re-run wiggle |
| real grasp visibly more open than sim | expected fraction-mapping distortion (URDF pip 1.57 vs SDK tip 1.08 rad) — verify with `hc pose`, tune pregrasp if needed |
| SDK arc topics empty for G20 | known SDK gap (`mapping.py` has no G20 branch) — irrelevant, we convert 0..255 ourselves |
| q lags tgt in the CSV | raise `--speed`/`--torque` gradually |
