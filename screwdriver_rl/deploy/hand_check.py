"""First-power-on & calibration utility for the LinkerHand L20/G20 (left).

Run the subcommands in order on a freshly connected hand, *before* the first
policy deployment (``deploy_linker``):

    python -m screwdriver_rl.deploy.hand_check info
    python -m screwdriver_rl.deploy.hand_check echo          # flex a finger by
        # hand: if the printed state tracks it, get_state() is a real
        # measurement, not an echo of the last command
    python -m screwdriver_rl.deploy.hand_check ramp          # open ⇄ pregrasp
    python -m screwdriver_rl.deploy.hand_check wiggle --out linker_calib.json
        # guided per-joint direction test → writes the calibration overlay
    python -m screwdriver_rl.deploy.hand_check pose --calib linker_calib.json
        # hold pregrasp for visual parity vs the sim render (render_posture.py)
    python -m screwdriver_rl.deploy.hand_check roundtrip --calib linker_calib.json

All subcommands accept ``--dry-run`` (an echo simulator stands in for the hand;
``wiggle`` auto-answers "yes") so the tool itself can be smoke-tested anywhere.

Pose source precedence: ``--pose-file`` (JSON list of 16 radians) >
``--bundle deploy.pth`` (its ``config.home_targets``) > the vendored
``PREGRASP_16``.  Imports stay SDK/torch-free until actually needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from screwdriver_rl.deploy import hw_utils
from screwdriver_rl.deploy import linker_sdk_map as sdkmap

RAMP_HZ = 20.0

# What each joint should visibly do when driven toward its upper bound.  The
# flexion chains are unambiguous; abduction and thumb signs are exactly what
# `wiggle` exists to pin down.
EXPECTED_MOTION = {
    "index_mcp_roll": "INDEX fingertip swings SIDEWAYS (abduction — sign unknown a priori)",
    "index_mcp_pitch": "INDEX base knuckle curls toward the palm",
    "index_pip": "INDEX middle+tip segments curl in (tip follows mechanically)",
    "middle_mcp_roll": "MIDDLE fingertip swings SIDEWAYS (abduction — sign unknown a priori)",
    "middle_mcp_pitch": "MIDDLE base knuckle curls toward the palm",
    "middle_pip": "MIDDLE middle+tip segments curl in",
    "ring_mcp_roll": "RING fingertip swings SIDEWAYS (abduction — sign unknown a priori)",
    "ring_mcp_pitch": "RING base knuckle curls toward the palm",
    "ring_pip": "RING middle+tip segments curl in",
    "pinky_mcp_roll": "PINKY fingertip swings SIDEWAYS (abduction — sign unknown a priori)",
    "pinky_mcp_pitch": "PINKY base knuckle curls toward the palm",
    "pinky_pip": "PINKY middle+tip segments curl in",
    "thumb_cmc_yaw": "THUMB rotates ACROSS the palm toward opposition (slot 10 rotation)",
    "thumb_cmc_roll": "THUMB swings sideways away from the palm plane (slot 5 side-bend)",
    "thumb_cmc_pitch": "THUMB root flexes toward the fingers (slot 0 root-flex)",
    "thumb_mcp": "THUMB distal segments curl in (tip follows mechanically)",
}


class FakeApi:
    """Perfect-tracking echo simulator matching the LinkerHandApi surface."""

    def __init__(self) -> None:
        self._state = [float(v) for v in sdkmap.joints16_to_sdk_range(sdkmap.open_pose_16())]

    def get_state(self):
        return list(self._state)

    def finger_move(self, pose=[]):  # noqa: B006 - mirror the SDK signature
        self._state = [float(v) for v in pose]

    def set_speed(self, speed):
        pass

    def set_torque(self, torque):
        pass

    def get_serial_number(self):
        return "FAKE-DRY-RUN"

    def get_embedded_version(self):
        return [0]

    def get_fault(self):
        return [0] * 5

    def get_touch_type(self):
        return -1


def _connect(args):
    sdkmap.apply_calibration(args.calib)
    if args.dry_run:
        print("[hand_check] DRY RUN: echo simulator (no hardware)", flush=True)
        return FakeApi()
    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_type=args.side, hand_joint=args.hand_joint, can=args.can)
    api.set_speed(speed=hw_utils.parse_five(args.speed, "speed"))
    api.set_torque(torque=hw_utils.parse_five(args.torque, "torque"))
    return api


def _read_state(api, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while True:
        state = hw_utils.validate_state20(api.get_state())
        if state is not None:
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError("no valid joint state from the hand — check CAN link/power/--can")
        time.sleep(0.05)


def _resolve_pose(args) -> list[float]:
    if getattr(args, "pose_file", None):
        with open(args.pose_file) as f:
            pose = json.load(f)
        if not (isinstance(pose, list) and len(pose) == 16):
            raise ValueError(f"--pose-file must hold a JSON list of 16 radians, got {pose!r}")
        return [float(v) for v in pose]
    if getattr(args, "bundle", None):
        import torch  # lazy: only needed for bundle poses

        cfg = torch.load(args.bundle, map_location="cpu").get("config", {})
        home = cfg.get("home_targets")
        if home is None:
            raise ValueError(f"{args.bundle} has no config.home_targets")
        return [float(v) for v in home]
    if getattr(args, "pose", "pregrasp") == "open":
        return sdkmap.open_pose_16()
    return list(sdkmap.PREGRASP_16)


def _ramp_to(api, target16, duration_s: float, start16=None) -> None:
    if start16 is None:
        start16 = sdkmap.sdk_range_to_joints16(_read_state(api))
    period = 1.0 / RAMP_HZ
    for frame in hw_utils.ramp_frames(start16, target16, duration_s, RAMP_HZ):
        api.finger_move(pose=sdkmap.joints16_to_sdk_range(frame))
        time.sleep(period)


def _fmt16(vals, fmt="{:+.3f}") -> str:
    return " ".join(fmt.format(v) for v in vals)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #

def cmd_info(args) -> int:
    api = _connect(args)
    for label, fn in (("serial", "get_serial_number"), ("embedded version", "get_embedded_version"),
                      ("touch type", "get_touch_type"), ("faults", "get_fault")):
        try:
            print(f"[info] {label}: {getattr(api, fn)()}")
        except Exception as exc:  # per-field: any one query may be unsupported
            print(f"[info] {label}: n/a ({exc})")
    state = _read_state(api)
    print(f"[info] state (0..255): {[int(v) for v in state]}")
    q = sdkmap.sdk_range_to_joints16(state)
    for js, v in zip(sdkmap.active_joints(), q):
        print(f"[info]   {js.name:18s} slot {js.slot:2d}  {v:+.3f} rad")
    return 0


def cmd_echo(args) -> int:
    api = _connect(args)
    print("[echo] streaming state — flex a finger BY HAND: if the numbers track, "
          "get_state() is a real measurement. Ctrl+C to stop.")
    prev = None
    t_end = time.monotonic() + args.seconds if args.seconds > 0 else None
    try:
        while t_end is None or time.monotonic() < t_end:
            state = hw_utils.validate_state20(api.get_state())
            if state is None:
                print("[echo] (invalid state read)")
            else:
                if prev is not None:
                    moved = [i for i, (a, b) in enumerate(zip(prev, state)) if abs(a - b) > 2]
                    tag = f"  Δ slots {moved}" if moved else ""
                else:
                    tag = ""
                print(f"[echo] {[int(v) for v in state]}{tag}")
                prev = state
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_ramp(args) -> int:
    api = _connect(args)
    pregrasp = _resolve_pose(args)
    open16 = sdkmap.open_pose_16()
    for c in range(args.cycles):
        print(f"[ramp] cycle {c + 1}/{args.cycles}: open → pregrasp")
        _ramp_to(api, open16, args.ramp_s)
        time.sleep(0.5)
        _ramp_to(api, pregrasp, args.ramp_s, start16=open16)
        time.sleep(args.hold_s)
        print("[ramp] pregrasp → open")
        _ramp_to(api, open16, args.ramp_s, start16=pregrasp)
    return 0


def cmd_wiggle(args) -> int:
    api = _connect(args)
    base = _resolve_pose(args)
    table = {js.name: js for js in sdkmap.active_joints()}
    names = list(table) if args.joints == "all" else [n.strip() for n in args.joints.split(",")]
    for n in names:
        if n not in table:
            raise ValueError(f"unknown joint {n!r} (expected one of {list(table)})")

    print(f"[wiggle] ramping to base pose; then testing {len(names)} joints "
          f"with ±{args.delta:.2f} rad")
    _ramp_to(api, base, 3.0)
    flips: dict[str, bool] = {}
    for name in names:
        js = table[name]
        i = [j.name for j in sdkmap.active_joints()].index(name)
        # Pick the test direction with room inside [lo, hi].
        up = base[i] + args.delta <= js.hi
        target = list(base)
        target[i] = base[i] + args.delta if up else base[i] - args.delta
        toward_hi = up  # moving toward the upper bound?
        while True:
            desc = EXPECTED_MOTION[name]
            arrow = "toward its FLEXED/upper end" if toward_hi else "back toward its OPEN/lower end"
            print(f"\n[wiggle] {name} (slot {js.slot}{', flip' if js.flip else ''}): "
                  f"moving {arrow}. Expected: {desc}")
            _ramp_to(api, target, 0.6, start16=base)
            time.sleep(args.settle_s)
            _ramp_to(api, base, 0.6, start16=target)
            if args.dry_run:
                ans = "y"
            else:
                ans = input("      moved as expected? [y]es [n]o-opposite [r]epeat [s]kip [q]uit: ").strip().lower()
            if ans == "r":
                continue
            if ans == "y":
                flips[name] = js.flip           # current setting verified
            elif ans == "n":
                flips[name] = not js.flip       # direction inverted → flip it
            elif ans == "s":
                pass
            elif ans == "q":
                print("[wiggle] aborted; nothing written")
                return 1
            break

    overlay = {
        "version": 1,
        "note": f"hand_check wiggle {time.strftime('%Y-%m-%d %H:%M')} "
                f"{args.hand_joint} {args.side}" + (" DRY-RUN" if args.dry_run else ""),
        "joints": {n: {"flip": f} for n, f in flips.items()},
    }
    sdkmap.build_joint_table(overlay)  # validate before writing
    with open(args.out, "w") as f:
        json.dump(overlay, f, indent=2)
    print(f"\n[wiggle] wrote {args.out}")
    for n, f in flips.items():
        mark = " (FLIPPED vs default)" if f != dict((j.name, j.flip) for j in sdkmap.DEFAULT_JOINTS)[n] else ""
        print(f"[wiggle]   {n:18s} flip={f}{mark}")
    return 0


def cmd_pose(args) -> int:
    api = _connect(args)
    pose = _resolve_pose(args)
    print(f"[pose] ramping to pose over {args.ramp_s:.1f} s; "
          f"{'holding until Ctrl+C' if args.hold_s <= 0 else f'holding {args.hold_s:.1f} s'}")
    print(f"[pose] target (rad): {_fmt16(pose)}")
    _ramp_to(api, pose, args.ramp_s)
    try:
        if args.hold_s <= 0:
            while True:
                time.sleep(0.5)
        time.sleep(args.hold_s)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_roundtrip(args) -> int:
    api = _connect(args)
    pose = _resolve_pose(args)
    _ramp_to(api, pose, 3.0)
    time.sleep(args.settle_s)
    acc = [0.0] * 20
    for _ in range(args.samples):
        state = _read_state(api)
        acc = [a + s for a, s in zip(acc, state)]
        time.sleep(0.05)
    mean_state = [a / args.samples for a in acc]
    meas = sdkmap.sdk_range_to_joints16(mean_state)
    cmd = sdkmap.joints16_to_sdk_range(pose)
    print(f"\n[roundtrip] per-joint tracking after {args.settle_s:.1f} s settle "
          f"({args.samples} samples):")
    print(f"{'joint':18s} {'slot':>4s} {'target':>8s} {'meas':>8s} {'err':>8s} "
          f"{'cmd':>4s} {'state':>6s}")
    worst, fails = 0.0, []
    for js, tgt, m in zip(sdkmap.active_joints(), pose, meas):
        err = m - tgt
        worst = max(worst, abs(err))
        flag = "  <-- OVER TOL" if abs(err) > args.tol else ""
        if flag:
            fails.append(js.name)
        print(f"{js.name:18s} {js.slot:4d} {tgt:8.3f} {m:8.3f} {err:+8.3f} "
              f"{cmd[js.slot]:4d} {mean_state[js.slot]:6.1f}{flag}")
    print(f"[roundtrip] worst |err| = {worst:.3f} rad (tol {args.tol})")
    if fails:
        print(f"[roundtrip] FAIL: {fails}")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--sdk-root", default=None)
    shared.add_argument("--hand-joint", default="G20", choices=["G20", "L20"])
    shared.add_argument("--side", default="left", choices=["left", "right"])
    shared.add_argument("--can", default="can0")
    shared.add_argument("--calib", default=None, metavar="JSON")
    shared.add_argument("--speed", default="120")
    shared.add_argument("--torque", default="150")
    shared.add_argument("--dry-run", action="store_true")
    pose_src = argparse.ArgumentParser(add_help=False)
    pose_src.add_argument("--pose-file", default=None, metavar="JSON16")
    pose_src.add_argument("--bundle", default=None, metavar="DEPLOY_PTH")

    p = argparse.ArgumentParser(prog="hand_check",
                                description="LinkerHand L20/G20 first-power-on & calibration checks")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", parents=[shared])
    sp = sub.add_parser("echo", parents=[shared])
    sp.add_argument("--hz", type=float, default=5.0)
    sp.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl+C")
    sp = sub.add_parser("ramp", parents=[shared, pose_src])
    sp.add_argument("--ramp-s", type=float, default=3.0)
    sp.add_argument("--hold-s", type=float, default=2.0)
    sp.add_argument("--cycles", type=int, default=1)
    sp = sub.add_parser("wiggle", parents=[shared, pose_src])
    sp.add_argument("--delta", type=float, default=0.15)
    sp.add_argument("--joints", default="all", help="'all' or csv of joint names")
    sp.add_argument("--settle-s", type=float, default=0.8)
    sp.add_argument("--out", default="linker_calib.json")
    sp = sub.add_parser("pose", parents=[shared, pose_src])
    sp.add_argument("--pose", default="pregrasp", choices=["pregrasp", "open"])
    sp.add_argument("--ramp-s", type=float, default=3.0)
    sp.add_argument("--hold-s", type=float, default=0.0, help="0 = hold until Ctrl+C")
    sp = sub.add_parser("roundtrip", parents=[shared, pose_src])
    sp.add_argument("--pose", default="pregrasp", choices=["pregrasp", "open"])
    sp.add_argument("--settle-s", type=float, default=1.5)
    sp.add_argument("--samples", type=int, default=5)
    sp.add_argument("--tol", type=float, default=0.15)

    args = p.parse_args(argv)
    fn = {"info": cmd_info, "echo": cmd_echo, "ramp": cmd_ramp,
          "wiggle": cmd_wiggle, "pose": cmd_pose, "roundtrip": cmd_roundtrip}[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
