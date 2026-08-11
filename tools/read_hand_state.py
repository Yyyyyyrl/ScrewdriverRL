"""Read the hand's 20-slot state WITHOUT sending anything to it.

Purpose: back-driven calibration.  Power the hand down, pose it by hand to match
a reference (e.g. the sim startup render), power it back up, and run this before
any position command is issued — the encoders then report the pose you set.
That single read constrains all 16 joints at once, which is what the fingertip
-distance approach could not do.

Unlike ``hand_check echo`` this never calls ``set_speed``/``set_torque``/
``finger_move``, so it cannot stiffen or nudge the hand.

    python tools/read_hand_state.py --sdk-root /home/user/linkerhand-ros-sdk --samples 5

Output: raw 0..255 per slot, plus the same vector decoded into the 16 semantic
joints (via the current calibration overlay) for comparison against sim targets.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sdk-root", default=None)
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--hand-joint", default="G20", choices=["G20", "L20"])
    p.add_argument("--can", default="can0")
    p.add_argument("--calib", default="linker_calib_deploy.json")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--json-out", default=None, help="Write the median state here.")
    p.add_argument("--compare-bundle", default=None,
                   help="deploy.pth — also print its startup targets next to the reading.")
    args = p.parse_args()

    from screwdriver_rl.deploy import hw_utils, linker_sdk_map as sdkmap

    sdkmap.apply_calibration(args.calib)
    hw_utils.bootstrap_sdk(args.sdk_root)
    from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

    # NOTE: constructing the API opens CAN but sends no motion command.
    api = LinkerHandApi(hand_type=args.side, hand_joint=args.hand_joint, can=args.can)

    reads: list[list[int]] = []
    for _ in range(args.samples):
        st = hw_utils.validate_state20(api.get_state())
        if st is not None:
            reads.append([int(v) for v in st])
        time.sleep(1.0 / max(args.hz, 0.1))
    if not reads:
        print("no valid state read — CAN up? hand powered?")
        return 1

    med = [sorted(c)[len(c) // 2] for c in zip(*reads)]
    print(f"samples: {len(reads)}   (median shown)")
    print("raw 0..255 per slot:")
    print("  " + " ".join(f"{i}:{v}" for i, v in enumerate(med)))

    joints = sdkmap.active_joints()
    decoded = sdkmap.sdk_range_to_joints16(med) if hasattr(sdkmap, "sdk_range_to_joints16") else None
    print("\nper joint (slot -> raw):")
    ref = None
    if args.compare_bundle:
        import torch
        ref = [float(v) for v in torch.load(args.compare_bundle, map_location="cpu")["config"]["startup_reset_targets"]]
    for i, js in enumerate(joints):
        line = f"  {js.name:16s} slot {js.slot:2d}  raw {med[js.slot]:3d}"
        if decoded is not None:
            line += f"  decoded {decoded[i]:+.4f} rad"
        if ref is not None:
            line += f"   sim startup {ref[i]:+.4f} rad"
        print(line)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"state20": med, "samples": len(reads)}, fh, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
