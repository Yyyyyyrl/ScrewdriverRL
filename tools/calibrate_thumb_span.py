"""Empirically calibrate a thumb joint against a measurable fingertip span.

Why
---
The vendored SDK arc table is authoritative for the four fingers but the thumb
row is flagged provisional, and hardware disagrees with it: driving the sim
startup pose lands the thumb ~15 mm closer to the fingers than sim
(84-85 mm vs 99.5 mm thumb-tip to middle-tip), even though the hand can reach
the sim pose when back-driven by hand.  Direction and routing are verified, so
the error is in the *value* mapping.

This sweeps ONE thumb joint across a range of commanded values, holding each so
an operator can measure the thumb-tip to middle-tip distance with a caliper.
Feed the measurements back to pick the calibration that reproduces the sim span.

Usage (hardware, screwdriver removed)::

    python tools/calibrate_thumb_span.py --sdk-root /home/user/linkerhand-ros-sdk \\
        --bundle <deploy.pth> --joint thumb_cmc_roll --values 1.085,1.15,1.22 \\
        --hold-s 6

Target: the sim value of the swept joint at the startup pose, and the sim span
it produces, are printed at startup so you know what you are aiming at.
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdk-root", default=None)
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--hand-joint", default="G20", choices=["G20", "L20"])
    p.add_argument("--can", default="can0")
    p.add_argument("--bundle", required=True)
    p.add_argument("--calib", default="linker_calib_deploy.json")
    p.add_argument("--joint", default="thumb_cmc_roll")
    p.add_argument("--values", default=None,
                   help="Comma-separated commanded values (training-URDF rad). "
                        "Default: sweep the joint's calibrated lo..hi in 5 steps.")
    p.add_argument("--hold-s", type=float, default=6.0)
    p.add_argument("--ramp-s", type=float, default=2.0)
    p.add_argument("--speed", default="80")
    p.add_argument("--torque", default="120")
    args = p.parse_args()

    import torch
    from screwdriver_rl.deploy import hw_utils, linker_sdk_map as sdkmap

    sdkmap.apply_calibration(args.calib)
    joints = sdkmap.active_joints()
    names = [j.name for j in joints]
    if args.joint not in names:
        raise SystemExit(f"unknown joint {args.joint!r}; expected one of {names}")
    ji = names.index(args.joint)
    js = joints[ji]

    cfg = torch.load(args.bundle, map_location="cpu")["config"]
    start = [float(v) for v in cfg["startup_reset_targets"]]

    if args.values:
        values = [float(v) for v in args.values.split(",") if v.strip()]
    else:
        values = [js.lo + i * (js.hi - js.lo) / 4.0 for i in range(5)]

    print(f"joint      : {args.joint} (slot {js.slot}, calibrated lo/hi = {js.lo}..{js.hi})")
    print(f"sim startup: {start[ji]:.4f} rad   <- this is what the policy asks for")
    print(f"TARGET     : thumb-tip to middle-tip = 99.5 mm in sim at the startup pose")
    print(f"sweep      : {values}\n")

    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_type=args.side, hand_joint=args.hand_joint, can=args.can)
    api.set_speed(speed=hw_utils.parse_five(args.speed, "speed"))
    try:
        api.set_torque(torque=hw_utils.parse_five(args.torque, "torque"))
    except Exception:
        pass  # not supported on every hand_joint

    def send(vec16: list[float]) -> list[int]:
        cmd = sdkmap.joints16_to_sdk_range(vec16)
        api.finger_move(pose=cmd)
        return cmd

    print("ramping to the sim startup pose …")
    send(start)
    time.sleep(args.ramp_s + 2.0)

    for v in values:
        pose = list(start)
        pose[ji] = v
        cmd = send(pose)
        print(f"\n--- {args.joint} = {v:.4f} rad   (slot {js.slot} -> range {cmd[js.slot]})")
        print(f"    MEASURE thumb-tip to middle-tip now (target 99.5 mm). "
              f"holding {args.hold_s:.0f}s …")
        time.sleep(args.hold_s)

    print("\nrestoring the startup pose.")
    send(start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
