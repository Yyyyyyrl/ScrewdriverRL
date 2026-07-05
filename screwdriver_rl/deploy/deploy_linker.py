"""Live LinkerHand L20/G20 deployment node for a Stage-2 ``deploy.pth`` bundle.

ScrewdriverRL analogue of HORA's ``deploy_ros2.py``.  Each control tick (10 Hz,
matching the training ``policy_dt`` of 0.1 s):

    read joint state ─▶ build finger_q ─▶ DeployPolicy.act(finger_q)
        ─▶ 16 rad targets ─▶ joints16_to_sdk_range ─▶ 20× 0..255
        ─▶ LinkerHandApi.finger_move  (or publish /cb_<side>_hand_control_cmd)

Session structure (shared by every transport):

    startup   read state → smooth ramp current→pregrasp (smoothstep, joint
              space) → settle → re-read → policy.reset(measured finger_q).
              The ramp's final command equals the command for zero action, so
              the first policy tick is continuous by construction.
    loop      10 Hz; invalid/stale state → hold last command; ``stale_limit``
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
    runs, but nothing is ever sent to the hand (speed/torque setup included).

⚠️ Run the ``hand_check`` sequence (info → echo → ramp → wiggle → pose →
roundtrip) before the first live run, and pass its calibration overlay via
``--calib``.  Start every live session with ``--max-ticks`` bounded and
``--record`` on.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from typing import Mapping

from screwdriver_rl.deploy.policy import DeployPolicy
from screwdriver_rl.deploy import linker_sdk_map as sdkmap
from screwdriver_rl.deploy import hw_utils


class LinkerDeployer:
    def __init__(
        self,
        ckpt: "str | Mapping",
        side: str = "left",
        hz: float = 10.0,
        device: str = "cpu",
        dry_run: bool = False,
        hand_joint: str = "G20",
        can_channel: str = "can0",
        sdk_root: "str | None" = None,
        speed: str = "120",
        torque: str = "150",
        ramp_s: float = 3.0,
        ramp_hz: float = 20.0,
        stale_limit: int = 10,
        record: "str | None" = None,
        max_ticks: int = 0,
        release: bool = False,
        calib: "str | Mapping | None" = None,
        no_send: bool = False,
    ) -> None:
        self.policy = DeployPolicy(ckpt, device=device)
        self.side = side
        self.hz = float(hz)
        self.dry_run = dry_run
        self.hand_joint = hand_joint
        self.can_channel = can_channel
        self.sdk_root = sdk_root
        self.speed5 = hw_utils.parse_five(speed, "speed")
        self.torque5 = hw_utils.parse_five(torque, "torque")
        self.ramp_s = float(ramp_s)
        self.ramp_hz = float(ramp_hz)
        self.stale_limit = int(stale_limit)
        self.record = record
        self.max_ticks = int(max_ticks)
        self.release = release
        self.no_send = no_send

        sdkmap.apply_calibration(calib)
        specs = sdkmap.active_joints()
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

    def _startup(self, read_fn, send_fn) -> None:
        state = self._read_valid(read_fn, timeout_s=5.0)
        if state is None:
            raise RuntimeError(
                "no valid joint state from the hand within 5 s — check the CAN link "
                "(`ip link` / find_can.sh), power, --can channel, and hand side."
            )
        q0 = sdkmap.sdk_range_to_joints16(state)
        home = self.policy.home_targets[0].tolist()
        print(f"[deploy] ramping to pregrasp over {self.ramp_s:.1f} s "
              f"({'not sending' if self.no_send else 'live'})", flush=True)
        self._ramp(send_fn, q0, home, self.ramp_s)
        time.sleep(0.3)  # let the servos settle before seeding the policy
        settled = self._read_valid(read_fn, timeout_s=1.0)
        if settled is None:
            print("[deploy] WARNING: no state after ramp; seeding history from pregrasp", flush=True)
            q_meas = home
        else:
            q_meas = sdkmap.sdk_range_to_joints16(settled)
        self.policy.reset(q_meas)

    def _loop(self, read_fn, send_fn) -> int:
        period = 1.0 / self.hz
        bad = 0
        while not self._stop and (self.max_ticks == 0 or self._ticks < self.max_ticks):
            t0 = time.monotonic()
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

    def _run_session(self, read_fn, send_fn) -> int:
        prev = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                prev[sig] = signal.signal(sig, lambda *_: setattr(self, "_stop", True))
            except ValueError:  # not in main thread
                pass
        self._open_record()
        code = 1
        try:
            self._startup(read_fn, send_fn)
            code = self._loop(read_fn, send_fn)
            self._shutdown(read_fn, send_fn, clean=(code == 0))
        finally:
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
    def _run_can(self) -> int:
        hw_utils.bootstrap_sdk(self.sdk_root)
        from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

        api = LinkerHandApi(hand_type=self.side, hand_joint=self.hand_joint, can=self.can_channel)
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
        return self._run_session(api.get_state, send_fn)

    # -- ROS1 transport (SDK linker_hand.launch node) ------------------------ #
    def _run_ros(self) -> int:
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
        if self.dry_run:
            return self._run_dry()
        if transport == "can":
            return self._run_can()
        if transport == "ros":
            return self._run_ros()
        raise ValueError(f"unknown transport {transport!r} (expected 'can' or 'ros')")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LinkerHand L20/G20 deployment for a Stage-2 deploy.pth.")
    p.add_argument("--checkpoint", required=True, help="Path to stage2_nn/deploy.pth")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--transport", default="can", choices=["can", "ros"])
    p.add_argument("--hand-joint", default="G20", choices=["G20", "L20"],
                   help="SDK model name; G20/L20 hands are physically identical")
    p.add_argument("--can", dest="can_channel", default="can0")
    p.add_argument("--sdk-root", default=None,
                   help="linkerhand-ros-sdk checkout (default: $LINKERHAND_SDK_ROOT)")
    p.add_argument("--hz", type=float, default=10.0,
                   help="control rate; keep at the training policy_dt rate (10 Hz)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--speed", default="120", help="finger speed 0..255, one int or 5 csv")
    p.add_argument("--torque", default="150", help="max torque 0..255, one int or 5 csv")
    p.add_argument("--ramp-s", type=float, default=3.0, help="startup ramp duration")
    p.add_argument("--stale-limit", type=int, default=10,
                   help="consecutive bad state reads before the watchdog stops (exit 2)")
    p.add_argument("--record", default=None, metavar="CSV",
                   help="record per-tick q/targets/action/cmd/state to a CSV file")
    p.add_argument("--max-ticks", type=int, default=0,
                   help="stop after N control ticks (0 = run until Ctrl+C)")
    p.add_argument("--release", action="store_true",
                   help="on clean stop, ramp to the open pose instead of holding")
    p.add_argument("--calib", default=None, metavar="JSON",
                   help="calibration overlay from `hand_check wiggle`")
    p.add_argument("--dry-run", action="store_true",
                   help="offline echo simulation; no SDK/ROS/hardware required")
    p.add_argument("--no-send", action="store_true",
                   help="on-hardware pre-flight: read state + run policy, send nothing")
    args = p.parse_args()
    code = LinkerDeployer(
        ckpt=args.checkpoint, side=args.side, hz=args.hz, device=args.device,
        dry_run=args.dry_run, hand_joint=args.hand_joint, can_channel=args.can_channel,
        sdk_root=args.sdk_root, speed=args.speed, torque=args.torque,
        ramp_s=args.ramp_s, stale_limit=args.stale_limit, record=args.record,
        max_ticks=args.max_ticks, release=args.release, calib=args.calib,
        no_send=args.no_send,
    ).run(transport=args.transport)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
