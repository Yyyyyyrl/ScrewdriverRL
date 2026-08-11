"""Empirically map the G20 thumb's per-finger CAN slots.

Why this exists
---------------
``linker_sdk_map`` builds its 20-vector in the **L20** slot layout (0-4 root-flex,
5-9 abduction, 10 thumb rotation, 15-19 tip), which the L20 CAN driver slices *by
joint type*.  The **G20** driver instead regroups the same 20-vector *by finger*
and sends 6 values per finger::

    '拇指': [5, 10, 0, 11, 12, 15]   # -> THUMB_POS frame

Slots 11-14 are "reserved" in the L20 layout, so we send 0 there — meaning two of
the six values the G20 thumb frame receives are always 0.  If those correspond to
real thumb actuators, the thumb is pinned to a pose the policy never asked for.

This probe moves ONE raw slot at a time (including 11/12) and holds, so an
operator can see which physical thumb joint each slot actually drives.

Usage (hardware required)::

    python tools/probe_g20_thumb_slots.py --sdk-root /home/user/linkerhand-ros-sdk \
        --bundle <deploy.pth> --slots 0,5,10,11,12,15

Safety: starts from the bundle's startup pose, moves one slot by --delta range
units, holds --hold-s, then always restores the baseline before the next slot.
Run with the screwdriver REMOVED.
"""

from __future__ import annotations

import argparse
import time

from screwdriver_rl.deploy import hw_utils, linker_sdk_map as sdkmap


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdk-root", default=None, help="linkerhand-ros-sdk checkout (pin it explicitly)")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--hand-joint", default="G20", choices=["G20", "L20"])
    p.add_argument("--can", default="can0")
    p.add_argument("--bundle", required=True, help="deploy.pth — baseline is its startup_reset_targets")
    p.add_argument("--calib", default="linker_calib_deploy.json")
    p.add_argument("--slots", default="0,5,10,11,12,15",
                   help="Raw SDK slots to probe, comma separated.")
    p.add_argument("--delta", type=int, default=70, help="Range-unit step (0..255 scale).")
    p.add_argument("--hold-s", type=float, default=2.5)
    p.add_argument("--speed", default="80")
    p.add_argument("--torque", default="120")
    args = p.parse_args()

    import torch

    sdkmap.apply_calibration(args.calib)
    cfg = torch.load(args.bundle, map_location="cpu")["config"]
    baseline = sdkmap.joints16_to_sdk_range([float(v) for v in cfg["startup_reset_targets"]])
    slots = [int(s) for s in args.slots.split(",") if s.strip()]

    slot_owner = {js.slot: js.name for js in sdkmap.active_joints()}
    print("baseline (startup) 20-slot command:")
    print("  " + " ".join(f"{i}:{v}" for i, v in enumerate(baseline)))
    print("\nG20 thumb frame draws slots [5, 10, 0, 11, 12, 15]; "
          "slots 11-14 are 0 in the L20 layout we build.\n")

    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    api = LinkerHandApi(hand_type=args.side, hand_joint=args.hand_joint, can=args.can)
    api.set_speed(speed=hw_utils.parse_five(args.speed, "speed"))
    api.set_torque(torque=hw_utils.parse_five(args.torque, "torque"))

    def send(vec: list[int]) -> None:
        api.finger_move(pose=[int(max(0, min(255, v))) for v in vec])

    print("moving to baseline …")
    send(baseline)
    time.sleep(3.0)

    for slot in slots:
        owner = slot_owner.get(slot, "RESERVED in our layout (we always send 0)")
        base_v = baseline[slot]
        # Step toward whichever side has room, so we never clip to a no-op.
        target_v = base_v - args.delta if base_v > 127 else base_v + args.delta
        probe = list(baseline)
        probe[slot] = target_v
        print(f"\n--- slot {slot}  ({owner})")
        print(f"    {base_v} -> {target_v}.  WATCH THE THUMB: which joint moves, and how?")
        send(probe)
        time.sleep(args.hold_s)
        send(baseline)
        time.sleep(1.5)

    print("\nrestoring baseline and exiting (hand holds the startup pose).")
    send(baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
