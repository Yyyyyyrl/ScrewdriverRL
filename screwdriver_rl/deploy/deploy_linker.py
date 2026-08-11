"""Live LinkerHand L20/G20 deployment node for a Stage-2 ``deploy.pth`` bundle.

ScrewdriverRL analogue of HORA's ``deploy_ros2.py``.  Each control tick uses
the rate declared by the policy bundle (20 Hz for free-object in-hand tasks,
10 Hz for the mounted screwdriver tasks):

    read joint state ─▶ build finger_q ─▶ DeployPolicy.act(finger_q)
        ─▶ 16 rad targets ─▶ joints16_to_sdk_range ─▶ 20× 0..255
        ─▶ LinkerHandApi.finger_move  (or publish /cb_<side>_hand_control_cmd)

Session structure (shared by every transport):

    startup   read state → smooth ramp current→collision-safe reset → slow
              contact ramp reset→home → settle → re-read → policy.reset(measured
              finger_q, acknowledged home).  Legacy non-topdown bundles whose
              reset equals home retain the original single-ramp path.
    loop      bundle-declared rate; invalid/stale state → hold last command; ``stale_limit``
              consecutive bad reads trip the watchdog (exit code 2).
    shutdown  default HOLD (stop sending; the position servo keeps the last
              target — safest while gripping a tool).  ``--release`` ramps to
              the open pose, on clean stops only, never after a watchdog trip.

Transports:
  * ``can``  — talk to the hand directly via ``LinkerHandApi`` (primary path;
    needs the SDK checkout — ``--sdk-root``/$LINKERHAND_SDK_ROOT — and a CAN
    interface, no ROS).
  * ``ros``  — ROS1/rospy against the SDK's ``linker_hand.launch`` node:
    subscribe ``/cb_<side>_hand_state`` (0..255), publish
    ``/cb_<side>_hand_control_cmd``, speed/torque via ``/cb_hand_setting_cmd``.
    Functional but not hardware-tested; the CAN path is the reference.

Offline modes:
  * ``--dry-run``   — no SDK/ROS import at all: a perfect-tracking echo
    simulator stands in for the hand, and the full startup/loop/shutdown path
    runs on any machine (CI-safe).
  * ``--no-send``   — on-hardware pre-flight: real state is read and the policy
    runs, but nothing is sent to the hand; speed/torque setup is skipped.

⚠️ Run the ``hand_check`` sequence (info → echo → ramp → wiggle → pose →
roundtrip) before the first live run, and pass its calibration overlay via
``--calib``.  Start every live session with ``--max-ticks`` bounded and
``--record`` on.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import time
from typing import Callable, Mapping, Sequence

from screwdriver_rl.deploy.policy import DeployPolicy
from screwdriver_rl.deploy.policy_coordinate_adapter import (
    LIVE_PROMOTED_STATUS,
    PolicyCoordinateAdapter,
    load_policy_coordinate_adapter,
)
from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from screwdriver_rl.deploy import hw_utils


class HardwareSafetyError(RuntimeError):
    """A fail-closed hardware gate stopped the session."""


class LinkerDeployer:
    def __init__(
        self,
        ckpt: "str | Mapping",
        side: str = "left",
        hz: "float | None" = None,
        device: str = "cpu",
        dry_run: bool = False,
        hand_joint: str = "G20",
        can_channel: str = "can0",
        sdk_root: "str | None" = None,
        speed: str = "120",
        torque: str = "150",
        ramp_s: float = 3.0,
        contact_ramp_s: float = 3.0,
        ramp_hz: float = 20.0,
        stale_limit: int = 10,
        record: "str | None" = None,
        max_ticks: int = 0,
        release: bool = False,
        calib: "str | Mapping | None" = None,
        no_send: bool = False,
        startup_only: bool = False,
        release_only: bool = False,
        contact_timeout_s: float = 3.0,
        rail_lock_ticks: int = 20,
        rail_lock_joints: int = 4,
        fault_limit: int = 2,
        fault_poll_ticks: int = 10,
        allowed_fault_mask: int = 0,
        expected_serial: "str | None" = None,
        task_frame_confirmed: bool = False,
        policy_coordinate_adapter: "str | None" = None,
        candidate_adapter_gate: bool = False,
    ) -> None:
        self.policy = DeployPolicy(ckpt, device=device)
        self.side = side
        package_hz = 1_000_000_000.0 / float(
            self.policy.codec.spec.control_period_ns
        )
        if hz is None:
            self.hz = package_hz
        else:
            self.hz = float(hz)
            if abs(self.hz - package_hz) > 1.0e-9:
                raise ValueError(
                    f"requested {self.hz:g} Hz conflicts with package-declared "
                    f"{package_hz:g} Hz control rate"
                )
        self.dry_run = dry_run
        self.hand_joint = hand_joint
        self.can_channel = can_channel
        self.sdk_root = sdk_root
        self.speed5 = hw_utils.parse_five(speed, "speed")
        self.torque5 = hw_utils.parse_five(torque, "torque")
        self.ramp_s = float(ramp_s)
        self.contact_ramp_s = float(contact_ramp_s)
        self.ramp_hz = float(ramp_hz)
        self.stale_limit = int(stale_limit)
        self.record = record
        self.max_ticks = int(max_ticks)
        self.release = release
        self.no_send = no_send
        self.startup_only = bool(startup_only)
        self.release_only = bool(release_only)
        if self.startup_only and self.release_only:
            raise ValueError("startup_only and release_only are mutually exclusive")
        self.contact_timeout_s = float(contact_timeout_s)
        self.rail_lock_ticks = int(rail_lock_ticks)
        self.rail_lock_joints = int(rail_lock_joints)
        self.fault_limit = int(fault_limit)
        self.fault_poll_ticks = int(fault_poll_ticks)
        self.allowed_fault_mask = int(allowed_fault_mask)
        self.expected_serial = expected_serial
        self.task_frame_confirmed = bool(task_frame_confirmed)
        self.candidate_adapter_gate = bool(candidate_adapter_gate)
        if self.contact_timeout_s <= 0.0:
            raise ValueError("contact_timeout_s must be positive")
        if self.rail_lock_ticks < 0 or not 0 <= self.rail_lock_joints <= 16:
            raise ValueError("rail_lock_ticks must be >= 0 and rail_lock_joints in 0..16")
        if self.fault_limit <= 0 or self.fault_poll_ticks <= 0:
            raise ValueError("fault_limit and fault_poll_ticks must be positive")
        if not 0 <= self.allowed_fault_mask <= 255:
            raise ValueError("allowed_fault_mask must be in 0..255")

        sdkmap.apply_calibration(calib)
        specs = sdkmap.active_joints()
        self.policy_coordinate_adapter: "PolicyCoordinateAdapter | None" = None
        if policy_coordinate_adapter is not None:
            if not isinstance(ckpt, str):
                raise ValueError(
                    "policy coordinate adapter requires a file-backed checkpoint"
                )
            if not isinstance(calib, str):
                raise ValueError(
                    "policy coordinate adapter requires a file-backed calibration overlay"
                )
            self.policy_coordinate_adapter = load_policy_coordinate_adapter(
                self.policy,
                policy_coordinate_adapter,
                ckpt,
                calib,
            )
            self.policy_coordinate_adapter.verify_active_hardware_limits(
                [joint.name for joint in specs],
                [joint.lo for joint in specs],
                [joint.hi for joint in specs],
            )
            self.policy = self.policy_coordinate_adapter
            print(
                "[deploy] policy coordinate adapter: "
                f"{self.policy.adapter_id} ({self.policy.status})",
                flush=True,
            )
        print("[deploy] joint map: "
              + ", ".join(f"{js.name}→s{js.slot}{'(flip)' if js.flip else ''}" for js in specs),
              flush=True)

        self.last_tick: dict = {}
        self._last_cmd: "list[int] | None" = None
        self._n_emitted = 0
        self._n_held = 0
        self._n_overruns = 0
        self._ticks = 0
        self._stop = False
        self._csv = None
        self._csv_file = None
        self._rail_lock_run = 0

    @property
    def _is_topdown(self) -> bool:
        return "Topdown" in str(self.policy.cfg.get("task", ""))

    @property
    def _is_free_inhand(self) -> bool:
        return "Inhand-Rotation" in str(self.policy.cfg.get("task", ""))

    @property
    def _requires_fixed_wrist_gate(self) -> bool:
        return self._is_topdown or self._is_free_inhand

    def _validate_topdown_mapping(self) -> None:
        """Require measured pitch and PIP LUTs for fixed-wrist task policies."""
        if not self._requires_fixed_wrist_gate:
            return
        specs = {js.name: js for js in sdkmap.active_joints()}
        signed_limits = {js.name: js for js in sdkmap.DEFAULT_JOINTS}
        bad = []
        for finger in ("index", "middle", "ring", "pinky"):
            pitch_name = f"{finger}_mcp_pitch"
            pitch = specs[pitch_name]
            if abs(pitch.lo) > 1.0e-9:
                bad.append(f"{pitch_name}=lo-{pitch.lo:g}")
            elif pitch.physical_raw_knots is None:
                bad.append(f"{pitch_name}=missing-physical-lut")

            pip_name = f"{finger}_pip"
            pip = specs[pip_name]
            # Until 2026-08-03 this compared against a hardcoded 1.08 rad, which
            # was the SDK tip arc rather than anything measured on this hand.
            # The four PIPs are now measured separately (1.539..1.749) and signed
            # into the semantic schema, so the contract is "agrees with the signed
            # calibration", which cannot go stale the next time it is remeasured.
            signed = signed_limits[pip_name]
            if abs(pip.lo - signed.lo) > 1.0e-9 or abs(pip.hi - signed.hi) > 1.0e-9:
                bad.append(
                    f"{pip_name}=[{pip.lo:g},{pip.hi:g}]!=signed[{signed.lo:g},{signed.hi:g}]"
                )
            elif pip.physical_raw_knots is None:
                bad.append(f"{pip_name}=missing-physical-lut")
        if bad:
            raise HardwareSafetyError(
                "fixed-wrist task pitch/PIP mapping is not the required measured "
                f"physical-LUT contract ({', '.join(bad)}). Calibrate every "
                "finger separately before live full-hand policy deployment."
            )

    def _validate_policy_coordinate_adapter_status(self, transport: str = "can") -> None:
        """Candidate adapters may run dry/no-send/startup gates, not live policy."""

        adapter = self.policy_coordinate_adapter
        if adapter is None or adapter.status == LIVE_PROMOTED_STATUS:
            return
        if self.dry_run or self.no_send or self.startup_only or self.release_only:
            return
        if self.candidate_adapter_gate:
            if transport != "can":
                raise HardwareSafetyError("candidate adapter gate requires CAN transport")
            if not 0 < self.max_ticks <= 100:
                raise HardwareSafetyError(
                    "candidate adapter gate requires --max-ticks in 1..100"
                )
            if not self.record:
                raise HardwareSafetyError("candidate adapter gate requires --record CSV")
            if not self.expected_serial:
                raise HardwareSafetyError("candidate adapter gate requires exact serial binding")
            if not self.task_frame_confirmed:
                raise HardwareSafetyError(
                    "candidate adapter gate requires --task-frame-confirmed"
                )
            return
        raise HardwareSafetyError(
            f"policy coordinate adapter {adapter.adapter_id!r} has status "
            f"{adapter.status!r}, not {LIVE_PROMOTED_STATUS!r}; complete and record "
            "the bounded startup/contact/dynamic live gates before enabling policy output"
        )

    # -- recording ---------------------------------------------------------- #
    def _open_record(self) -> None:
        if not self.record:
            return
        self._csv_file = open(self.record, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(
            ["tick", "phase", "t_wall", "t_mono"]
            + [f"q{i}" for i in range(16)]
            + [f"tgt{i}" for i in range(16)]
            + [f"act{i}" for i in range(16)]
            + [f"cmd{i}" for i in range(20)]
            + [f"state{i}" for i in range(20)]
        )

    def _record_row(self, phase: str, q=None, tgt=None, act=None, cmd=None, state=None) -> None:
        if self._csv is None:
            return
        def _pad(vals, n):
            return list(vals) if vals is not None else [""] * n
        self._csv.writerow(
            [self._ticks, phase, time.time(), time.monotonic()]
            + _pad(q, 16) + _pad(tgt, 16) + _pad(act, 16) + _pad(cmd, 20) + _pad(state, 20)
        )
        self._csv_file.flush()

    # -- shared step: SDK 0..255 state -> command --------------------------- #
    def _step(self, state_range20) -> list[int]:
        finger_q = sdkmap.sdk_range_to_joints16(list(state_range20))
        targets_t, action_t = self.policy.act(finger_q, return_action=True)
        targets = targets_t[0].tolist()
        action = action_t[0].tolist()
        cmd = sdkmap.joints16_to_sdk_range(targets)
        self.last_tick = {
            "finger_q": finger_q, "targets": targets, "action": action,
            "cmd": cmd, "state": list(state_range20),
        }
        self._n_emitted += 1
        if self.dry_run and self._n_emitted <= 5:
            print(f"[deploy] tick {self._n_emitted}: cmd={cmd}", flush=True)
        return cmd

    def _rail_locked(self, targets: "list[float]") -> bool:
        """True once enough joint targets have sat pinned at their limits.

        The integrated targets clamp to ``[finger_lower, finger_upper]``.  When a
        policy pushes a saturated action into a rail the target stops moving, so
        the measured joints stop moving, so the proprio history the adapter reads
        stops changing — the latent and the action freeze with it and the policy
        cannot recover.  Observed on hardware 2026-07-30: every joint pinned
        within 1.5 s, then 45 ticks of zero motion.
        """
        if self.rail_lock_ticks <= 0:
            return False
        lo = self.policy.finger_lower[0].tolist()
        hi = self.policy.finger_upper[0].tolist()
        pinned = sum(
            1 for value, low, high in zip(targets, lo, hi)
            if value <= low + 1.0e-6 or value >= high - 1.0e-6
        )
        if pinned >= self.rail_lock_joints:
            self._rail_lock_run += 1
        else:
            self._rail_lock_run = 0
        return self._rail_lock_run >= self.rail_lock_ticks

    # -- session helpers (transport-agnostic, fake-testable) ---------------- #
    def _read_valid(self, read_fn, timeout_s: float = 5.0) -> "list[float] | None":
        """Poll ``read_fn`` until it yields a valid 20-slot state or timeout."""
        deadline = time.monotonic() + timeout_s
        while True:
            state = hw_utils.validate_state20(read_fn())
            if state is not None:
                return state
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    @staticmethod
    def _numeric_vector(values, size: int, *, nonnegative: bool = True) -> "list[float] | None":
        try:
            seq = list(values)
        except (TypeError, ValueError):
            return None
        if len(seq) != size:
            return None
        out = []
        for value in seq:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return None
            number = float(value)
            if not math.isfinite(number) or (nonnegative and number < 0.0):
                return None
            out.append(number)
        return out

    def _fault_snapshot(self, fault_fn: "Callable | None") -> "list[int] | None":
        if fault_fn is None:
            return None
        try:
            values = self._numeric_vector(fault_fn(), 20)
        except Exception as exc:
            print(f"[deploy] WARNING: fault read failed: {exc}", flush=True)
            return None
        if values is None or any(value > 255.0 for value in values):
            return None
        return [int(value) for value in values]

    def _active_faults(self, values: Sequence[int]) -> list[tuple[int, int]]:
        inverse_mask = 0xFF ^ self.allowed_fault_mask
        return [
            (slot, int(value))
            for slot, value in enumerate(values)
            if int(value) & inverse_mask
        ]

    def _require_fault_clear(self, fault_fn: "Callable | None", where: str) -> None:
        if fault_fn is None:
            return
        values = self._fault_snapshot(fault_fn)
        if values is None:
            raise HardwareSafetyError(f"fault telemetry unavailable {where}")
        active = self._active_faults(values)
        if active:
            detail = ", ".join(f"slot{slot}={value}" for slot, value in active)
            raise HardwareSafetyError(f"non-zero G20 fault {where}: {detail}")

    def _contact_snapshot(self, contact_fn: "Callable | None") -> "list[float] | None":
        if contact_fn is None:
            return None
        try:
            return self._numeric_vector(contact_fn(), 5)
        except Exception as exc:
            print(f"[deploy] WARNING: tactile read failed: {exc}", flush=True)
            return None

    # NOTE: the tactile contact gate was removed 2026-07-30.  On hand
    # LHT20-010-415 three of the five 72-cell tactile pads report a
    # constant 0 (no signal), so a "N fingers in contact" precondition can
    # never be satisfied regardless of grasp quality — it only ever blocked
    # good runs.  Tactile mass is still printed as telemetry by
    # _contact_snapshot().  Restore the gate only once the pads are fixed.

    def _ramp(self, send_fn, start16, end16, duration_s: float, phase: str = "ramp") -> None:
        """Smoothstep joint-space ramp; final command is exactly end16's command."""
        period = 1.0 / self.ramp_hz
        for frame in hw_utils.ramp_frames(start16, end16, duration_s, self.ramp_hz):
            t0 = time.monotonic()
            cmd = sdkmap.joints16_to_sdk_range(frame)
            send_fn(cmd)
            self._last_cmd = cmd
            self._record_row(phase, tgt=frame, cmd=cmd)
            if self._stop:  # allow Ctrl+C to interrupt a ramp
                return
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)

    def _startup(self, read_fn, send_fn, fault_fn: "Callable | None" = None) -> None:
        state = self._read_valid(read_fn, timeout_s=5.0)
        if state is None:
            raise RuntimeError(
                "no valid joint state from the hand within 5 s — check the CAN link "
                "(`ip link` / find_can.sh), power, --can channel, and hand side."
            )
        q0 = sdkmap.sdk_range_to_joints16(state)
        reset_target = self.policy.startup_reset_targets[0].tolist()
        home = self.policy.home_targets[0].tolist()
        effective_home = sdkmap.sdk_range_to_joints16(
            sdkmap.joints16_to_sdk_range(home)
        )
        staged = max(abs(a - b) for a, b in zip(reset_target, home)) > 1.0e-6
        if staged:
            print(
                f"[deploy] approaching collision-safe reset posture over "
                f"{self.ramp_s:.1f} s "
                f"({'not sending' if self.no_send else 'live'})",
                flush=True,
            )
            self._ramp(send_fn, q0, reset_target, self.ramp_s, phase="approach")
            time.sleep(0.2)
            self._require_fault_clear(fault_fn, "after collision-safe approach")
            reset_state = self._read_valid(read_fn, timeout_s=1.0)
            reset_measured = (
                sdkmap.sdk_range_to_joints16(reset_state)
                if reset_state is not None
                else reset_target
            )
            print(
                f"[deploy] closing reset posture to contact home over "
                f"{self.contact_ramp_s:.1f} s",
                flush=True,
            )
            self._ramp(
                send_fn,
                reset_measured,
                home,
                self.contact_ramp_s,
                phase="contact-ramp",
            )
        else:
            print(
                f"[deploy] ramping to pregrasp over {self.ramp_s:.1f} s "
                f"({'not sending' if self.no_send else 'live'})",
                flush=True,
            )
            self._ramp(send_fn, q0, home, self.ramp_s)
        time.sleep(0.3)  # let the servos settle before seeding the policy
        settled = self._read_valid(read_fn, timeout_s=1.0)
        if settled is None:
            print("[deploy] WARNING: no state after ramp; seeding history from pregrasp", flush=True)
            q_meas = home
        else:
            q_meas = sdkmap.sdk_range_to_joints16(settled)
        effective_target = q_meas if self.no_send else effective_home
        self.policy.reset(q_meas, effective_target)

    def _loop(self, read_fn, send_fn, fault_fn: "Callable | None" = None) -> int:
        period = 1.0 / self.hz
        bad = 0
        bad_fault = 0
        while not self._stop and (self.max_ticks == 0 or self._ticks < self.max_ticks):
            t0 = time.monotonic()
            if fault_fn is not None and self._ticks % self.fault_poll_ticks == 0:
                faults = self._fault_snapshot(fault_fn)
                active = None if faults is None else self._active_faults(faults)
                if active is None or active:
                    bad_fault += 1
                    if active:
                        detail = ", ".join(f"slot{slot}={value}" for slot, value in active)
                    else:
                        detail = "telemetry unavailable"
                    print(
                        f"[deploy] WARNING: fault sample {bad_fault}/{self.fault_limit}: {detail}",
                        flush=True,
                    )
                    if bad_fault >= self.fault_limit:
                        print(
                            "[deploy] FAULT WATCHDOG: holding last command and stopping.",
                            flush=True,
                        )
                        return 4
                else:
                    bad_fault = 0
            state = hw_utils.validate_state20(read_fn())
            if state is None:
                bad += 1
                self._n_held += 1
                if self._last_cmd is not None:
                    send_fn(self._last_cmd)
                self._record_row("hold", cmd=self._last_cmd)
                if bad >= self.stale_limit:
                    print(f"[deploy] WATCHDOG: {bad} consecutive bad/stale state reads — "
                          "holding last command and stopping.", flush=True)
                    return 2
            else:
                bad = 0
                cmd = self._step(state)
                send_fn(cmd)
                self._last_cmd = cmd
                t = self.last_tick
                self._record_row("policy", q=t["finger_q"], tgt=t["targets"],
                                 act=t["action"], cmd=cmd, state=state)
                if self._rail_locked(t["targets"]):
                    print(
                        f"[deploy] RAIL LOCK: >= {self.rail_lock_joints} joint targets "
                        f"pinned at their limits for {self.rail_lock_ticks} ticks — the "
                        "policy is saturated and cannot recover. Holding and stopping.",
                        flush=True,
                    )
                    return 5
            self._ticks += 1
            dt = period - (time.monotonic() - t0)
            if dt > 0:
                time.sleep(dt)
            else:
                self._n_overruns += 1
                if self._n_overruns in (1, 25) or self._n_overruns % 100 == 0:
                    print(f"[deploy] WARNING: control tick overran the {period*1000:.0f} ms "
                          f"period ({self._n_overruns}× so far)", flush=True)
        return 0

    def _shutdown(self, read_fn, send_fn, clean: bool) -> None:
        if self.release and clean:
            state = hw_utils.validate_state20(read_fn())
            start = (sdkmap.sdk_range_to_joints16(state) if state is not None
                     else self.policy.cur_targets[0].tolist())
            print("[deploy] releasing: ramping to open pose", flush=True)
            self._stop = False  # allow the release ramp to run to completion
            self._ramp(send_fn, start, sdkmap.open_pose_16(), self.ramp_s, phase="release")
        else:
            # HOLD: stop sending; the position servo keeps the last target.
            # After a watchdog trip the state is untrusted — never blind-open.
            print("[deploy] holding last position (use --release for open-pose handoff)", flush=True)
        print(f"[deploy] session: {self._ticks} ticks, {self._n_emitted} policy steps, "
              f"{self._n_held} held, {self._n_overruns} overruns"
              + (f", log → {self.record}" if self.record else ""), flush=True)

    def _run_session(
        self,
        read_fn,
        send_fn,
        fault_fn: "Callable | None" = None,
        contact_fn: "Callable | None" = None,
    ) -> int:
        prev = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev[sig] = signal.signal(sig, lambda *_: setattr(self, "_stop", True))
            except ValueError:  # not in main thread
                pass
        self._open_record()
        code = 1
        try:
            self._require_fault_clear(fault_fn, "before startup")
            if self.release_only:
                state = self._read_valid(read_fn, timeout_s=2.0)
                if state is None:
                    raise HardwareSafetyError("no valid state for release-only")
                start = sdkmap.sdk_range_to_joints16(state)
                print(
                    f"[deploy] RELEASE-ONLY: current pose to open over {self.ramp_s:.1f} s; "
                    "policy and contact-home are bypassed",
                    flush=True,
                )
                self._ramp(
                    send_fn,
                    start,
                    sdkmap.open_pose_16(),
                    self.ramp_s,
                    phase="release-only",
                )
                code = 0
            else:
                self._startup(read_fn, send_fn, fault_fn=fault_fn)
                self._require_fault_clear(fault_fn, "after contact-home ramp")
            if self.release_only:
                pass
            elif self.startup_only:
                values = self._contact_snapshot(contact_fn)
                if values is not None:
                    print(
                        "[deploy] STARTUP-ONLY tactile mass: "
                        + ", ".join(f"{value:.0f}" for value in values),
                        flush=True,
                    )
                print("[deploy] STARTUP-ONLY complete; policy was not enabled.", flush=True)
                code = 0
            else:
                code = self._loop(read_fn, send_fn, fault_fn=fault_fn)
        except HardwareSafetyError as exc:
            print(f"[deploy] SAFETY GATE: {exc}", flush=True)
            code = 4
        finally:
            if self._last_cmd is not None:
                self._shutdown(read_fn, send_fn, clean=(code == 0))
            if self._csv_file is not None:
                self._csv_file.close()
            for sig, h in prev.items():
                signal.signal(sig, h)
        return code

    # -- offline dry-run (no SDK, no ROS, no hardware) ----------------------- #
    def _run_dry(self) -> int:
        echo = [float(v) for v in sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())]

        def read_fn():
            return list(echo)

        def send_fn(cmd):
            echo[:] = [float(c) for c in cmd]  # perfect-tracking hand model

        print(f"[deploy] DRY RUN: echo simulator, {self.hz} Hz, "
              f"max_ticks={self.max_ticks or '∞'}", flush=True)
        return self._run_session(read_fn, send_fn)

    # -- CAN transport (direct SDK) ------------------------------------------ #
    def _validate_can_identity(self, api) -> tuple[str, list[int], int]:
        if self._requires_fixed_wrist_gate and self.hand_joint != "G20":
            raise HardwareSafetyError(
                "this fixed-wrist deployment is validated for the G20 transport only; "
                f"got --hand-joint {self.hand_joint}"
            )
        serial = ""
        version: list[int] = []
        for _ in range(5):
            serial = str(api.get_serial_number()).strip().strip("\x00")
            try:
                version = [int(value) for value in api.get_embedded_version()]
            except (TypeError, ValueError):
                version = []
            if serial not in ("", "-1") and version and any(version):
                break
            time.sleep(0.05)
        if serial in ("", "-1"):
            raise HardwareSafetyError("G20 serial number is unavailable")
        if not version or not any(version):
            raise HardwareSafetyError("G20 embedded version is unavailable")
        expected_side_token = "-L-" if self.side == "left" else "-R-"
        if expected_side_token not in serial:
            raise HardwareSafetyError(
                f"serial {serial!r} does not match requested {self.side} hand"
            )
        if self.expected_serial and serial != self.expected_serial:
            raise HardwareSafetyError(
                f"serial mismatch: expected {self.expected_serial!r}, got {serial!r}"
            )
        try:
            touch_type = int(api.get_touch_type())
        except (TypeError, ValueError):
            touch_type = -1
        if not self.no_send and self.startup_only:
            if touch_type != 2:
                raise HardwareSafetyError(
                    f"G20 matrix tactile type 2 is required, got touch type {touch_type}"
                )
        print(
            f"[deploy] identity: serial={serial} embedded={version} touch_type={touch_type}",
            flush=True,
        )
        return serial, version, touch_type

    @staticmethod
    def _g20_fault_reader(api):
        hand = api.hand
        hand.get_thumb_fault()
        hand.get_index_fault()
        hand.get_middle_fault()
        hand.get_ring_fault()
        hand.get_little_fault()
        time.sleep(0.03)
        return hand.joint_state_to_cmd_state(
            [hand.x59, hand.x5A, hand.x5B, hand.x5C, hand.x5D]
        )

    @staticmethod
    def _g20_contact_reader(api):
        matrices = api.get_matrix_touch()
        time.sleep(0.02)
        masses = []
        for matrix in matrices:
            flat = [float(value) for row in matrix for value in row]
            if len(flat) != 72 or any(not math.isfinite(value) or value < 0.0 for value in flat):
                return None
            masses.append(sum(flat))
        return masses

    def _run_can(self) -> int:
        hw_utils.bootstrap_sdk(self.sdk_root)
        from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

        api = LinkerHandApi(hand_type=self.side, hand_joint=self.hand_joint, can=self.can_channel)
        try:
            self._validate_can_identity(api)
            if self._requires_fixed_wrist_gate and not self.no_send and not self.startup_only and not self.release_only:
                if not self.task_frame_confirmed:
                    raise HardwareSafetyError(
                        "fixed-wrist task-frame alignment has not been confirmed; "
                        "run --startup-only first, align and load the 64-mm cube, then add "
                        "--task-frame-confirmed"
                    )
            if self.no_send:
                print("[deploy] NO-SEND pre-flight: reading state, computing commands, "
                      "sending nothing (speed/torque setup skipped)", flush=True)
                send_fn = lambda cmd: None  # noqa: E731
            else:
                api.set_speed(speed=self.speed5)
                api.set_torque(torque=self.torque5)
                send_fn = lambda cmd: api.finger_move(pose=cmd)  # noqa: E731
            print(f"[deploy] CAN: LinkerHandApi {self.side} {self.hand_joint} on "
                  f"{self.can_channel} @ {self.hz} Hz  speed={self.speed5} torque={self.torque5}",
                  flush=True)
            fault_fn = lambda: self._g20_fault_reader(api)
            contact_fn = lambda: self._g20_contact_reader(api)
            return self._run_session(
                api.get_state,
                send_fn,
                fault_fn=fault_fn,
                contact_fn=contact_fn,
            )
        finally:
            close = getattr(api.hand, "close_can_interface", None)
            if callable(close):
                close()

    # -- ROS1 transport (SDK linker_hand.launch node) ------------------------ #
    def _run_ros(self) -> int:
        if self._requires_fixed_wrist_gate and not self.no_send:
            raise HardwareSafetyError(
                "live fixed-wrist task ROS transport has no identity/fault/tactile gates; use --transport can"
            )
        import rospy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        # disable_signals: keep our own SIGINT handling (hold-on-Ctrl+C).
        rospy.init_node("screwdriver_rl_deploy", anonymous=True, disable_signals=True)
        cache = {"state": None, "t": 0.0}

        def on_state(msg: "JointState") -> None:
            if len(msg.position) >= 20:
                cache["state"] = list(msg.position[:20])
                cache["t"] = time.monotonic()

        state_topic = f"/cb_{self.side}_hand_state"
        cmd_topic = f"/cb_{self.side}_hand_control_cmd"
        rospy.Subscriber(state_topic, JointState, on_state, queue_size=10)
        pub = rospy.Publisher(cmd_topic, JointState, queue_size=10)
        setting_pub = rospy.Publisher("/cb_hand_setting_cmd", String, queue_size=2)

        stale_s = 2.0 / self.hz  # state older than 2 control periods counts as missing

        def read_fn():
            if cache["state"] is None or time.monotonic() - cache["t"] > stale_s:
                return None
            return list(cache["state"])

        def _publish(cmd):
            msg = JointState()
            msg.header.stamp = rospy.Time.now()
            msg.position = [float(c) for c in cmd]
            msg.velocity = [0.0] * 20
            msg.effort = [0.0] * 20
            pub.publish(msg)

        send_fn = (lambda cmd: None) if self.no_send else _publish
        if not self.no_send:
            time.sleep(0.5)  # let publishers register with the SDK node
            for cmd_name, key, vals in (("set_speed", "speed", self.speed5),
                                        ("set_max_torque_limits", "torque", self.torque5)):
                setting_pub.publish(String(data=json.dumps(
                    {"setting_cmd": cmd_name, "params": {"hand_type": self.side, key: vals}})))
        print(f"[deploy] ROS: sub {state_topic} → pub {cmd_topic} @ {self.hz} Hz "
              f"(no_send={self.no_send})", flush=True)
        return self._run_session(read_fn, send_fn)

    def run(self, transport: str = "can") -> int:
        try:
            self._validate_policy_coordinate_adapter_status(transport)
            if self.dry_run:
                return self._run_dry()
            self._validate_topdown_mapping()
            if transport == "can":
                return self._run_can()
            if transport == "ros":
                return self._run_ros()
            raise ValueError(f"unknown transport {transport!r} (expected 'can' or 'ros')")
        except HardwareSafetyError as exc:
            print(f"[deploy] SAFETY GATE: {exc}", flush=True)
            return 4


def main() -> None:
    p = argparse.ArgumentParser(
        description="LinkerHand L20/G20 deployment for a Stage-2 deploy.pth.")
    p.add_argument("--checkpoint", required=True, help="Path to stage2_nn/deploy.pth")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--transport", default="can", choices=["can", "ros"])
    p.add_argument("--hand-joint", default="G20", choices=["G20", "L20"],
                   help="SDK model name; fixed-wrist live deployment requires G20")
    p.add_argument(
        "--expected-serial",
        default="LHT20-010-415-L-B-1-D",
        help="exact G20 serial bound to this deployment (empty string disables exact match)",
    )
    p.add_argument("--can", dest="can_channel", default="can0")
    p.add_argument("--sdk-root", default=None,
                   help="linkerhand-ros-sdk checkout (default: $LINKERHAND_SDK_ROOT)")
    p.add_argument(
        "--hz",
        type=float,
        default=None,
        help="control rate override; must match the package ProprioCodec rate",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--speed", default="120", help="finger speed 0..255, one int or 5 csv")
    p.add_argument("--torque", default="150", help="max torque 0..255, one int or 5 csv")
    p.add_argument("--ramp-s", type=float, default=3.0,
                   help="current pose to collision-safe reset duration")
    p.add_argument("--contact-ramp-s", type=float, default=3.0,
                   help="collision-safe reset to contact-home duration")
    p.add_argument("--stale-limit", type=int, default=10,
                   help="consecutive bad state reads before the watchdog stops (exit 2)")
    p.add_argument("--fault-limit", type=int, default=2,
                   help="consecutive bad/nonzero fault samples before stop (exit 4)")
    p.add_argument("--fault-poll-ticks", type=int, default=10,
                   help="poll G20 faults every N policy ticks")
    p.add_argument(
        "--allowed-fault-mask",
        type=lambda value: int(value, 0),
        default=0,
        help="explicitly allowed G20 fault bits (default 0; e.g. 0x40 only after vendor approval)",
    )
    p.add_argument("--contact-timeout-s", type=float, default=3.0,
                   help="tactile telemetry read timeout")
    p.add_argument("--rail-lock-ticks", type=int, default=20,
                   help="abort if >= --rail-lock-joints joint targets sit pinned at their "
                        "limits for this many consecutive ticks (0 disables). Once targets "
                        "clamp at the rail the proprio history stops changing, so the latent "
                        "and the action freeze too — the policy cannot recover on its own.")
    p.add_argument("--rail-lock-joints", type=int, default=4,
                   help="how many pinned joints count as a rail lock-up")
    p.add_argument("--record", default=None, metavar="CSV",
                   help="record per-tick q/targets/action/cmd/state to a CSV file")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="stop after N control ticks (0 = run until Ctrl+C)")
    p.add_argument("--release", action="store_true",
                   help="on clean stop, ramp to the open pose instead of holding")
    p.add_argument("--calib", default=None, metavar="JSON",
                   help="calibration overlay from `hand_check wiggle`")
    p.add_argument(
        "--policy-coordinate-adapter",
        default=None,
        metavar="JSON",
        help=(
            "content-addressed legacy-policy coordinate bridge bound to the exact "
            "checkpoint and --calib overlay"
        ),
    )
    p.add_argument(
        "--candidate-adapter-gate",
        action="store_true",
        help=(
            "allow a non-promoted adapter only for a recorded, serial-bound CAN "
            "gate with --task-frame-confirmed and --max-ticks 1..100"
        ),
    )
    p.add_argument("--dry-run", action="store_true",
                   help="offline echo simulation; no SDK/ROS/hardware required")
    p.add_argument("--no-send", action="store_true",
                   help="on-hardware pre-flight: read state + run policy, send nothing")
    p.add_argument("--startup-only", action="store_true",
                   help="ramp reset→contact-home and hold; never enable the policy")
    p.add_argument("--release-only", action="store_true",
                   help="after a clear fault check, ramp current pose directly to open; no policy/home")
    p.add_argument(
        "--task-frame-confirmed",
        action="store_true",
        help="operator confirms the 64-mm top-down fixture matches the reviewed task frame",
    )
    args = p.parse_args()
    code = LinkerDeployer(
        ckpt=args.checkpoint, side=args.side, hz=args.hz, device=args.device,
        dry_run=args.dry_run, hand_joint=args.hand_joint, can_channel=args.can_channel,
        sdk_root=args.sdk_root, speed=args.speed, torque=args.torque,
        ramp_s=args.ramp_s, contact_ramp_s=args.contact_ramp_s,
        stale_limit=args.stale_limit, record=args.record,
        max_ticks=args.max_ticks, release=args.release, calib=args.calib,
        no_send=args.no_send, startup_only=args.startup_only,
        release_only=args.release_only,
        contact_timeout_s=args.contact_timeout_s,
        rail_lock_ticks=args.rail_lock_ticks,
        rail_lock_joints=args.rail_lock_joints,
        fault_limit=args.fault_limit, fault_poll_ticks=args.fault_poll_ticks,
        allowed_fault_mask=args.allowed_fault_mask,
        expected_serial=(args.expected_serial or None),
        task_frame_confirmed=args.task_frame_confirmed,
        policy_coordinate_adapter=args.policy_coordinate_adapter,
        candidate_adapter_gate=args.candidate_adapter_gate,
    ).run(transport=args.transport)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
