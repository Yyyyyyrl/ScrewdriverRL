#!/usr/bin/env python3
"""Run a tightly bounded top-down policy diagnostic with no tool present.

The operator must have confirmed that the hand is running in free air.  This
wrapper does not validate the screwdriver task frame and must never be used
with a tool, fixture, or person inside the hand's swept volume.
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--expected-serial", required=True)
    parser.add_argument("--can", default="can0")
    parser.add_argument("--max-ticks", type=int, default=5)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_ticks <= 5:
        parser.error("--max-ticks must be in 1..5")

    from screwdriver_rl.deploy.deploy_linker import LinkerDeployer

    print(
        "[free-air] OPERATOR CONTEXT: no screwdriver/tool/fixture is present. "
        "This is a motion-and-SDK diagnostic only; task-frame validation is "
        "explicitly NOT claimed.",
        flush=True,
    )
    deployer = LinkerDeployer(
        ckpt=args.checkpoint,
        side="left",
        device="cpu",
        hand_joint="G20",
        can_channel=args.can,
        sdk_root=args.sdk_root,
        speed="40",
        torque="80",
        ramp_s=5.0,
        contact_ramp_s=8.0,
        stale_limit=1,
        record=args.record,
        max_ticks=args.max_ticks,
        release=False,
        calib=args.calib,
        no_send=False,
        startup_only=False,
        release_only=False,
        rail_lock_ticks=3,
        rail_lock_joints=4,
        fault_limit=1,
        fault_poll_ticks=1,
        allowed_fault_mask=0,
        expected_serial=args.expected_serial,
        # The formal task-frame gate has no "not applicable" state.  This
        # wrapper supplies True only after enforcing the no-tool, <=5-tick
        # diagnostic envelope above; it does not claim fixture validation.
        task_frame_confirmed=True,
    )
    return deployer.run(transport="can")


if __name__ == "__main__":
    raise SystemExit(main())
