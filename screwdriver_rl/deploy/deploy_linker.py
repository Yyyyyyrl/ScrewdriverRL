"""Live LinkerHand L20/G20 deployment node for a Stage-2 ``deploy.pth`` bundle.

ScrewdriverRL analogue of HORA's ``deploy_ros2.py``.  Each control tick (10 Hz,
matching the training ``policy_dt`` of 0.1 s):

    read joint state ─▶ build finger_q ─▶ DeployPolicy.act(finger_q)
        ─▶ 16 rad targets ─▶ joints16_to_sdk_range ─▶ 20× 0..255
        ─▶ publish /cb_<side>_hand_control_cmd  (or LinkerHandApi.finger_move)

Two transports:
  * ``ros``  — subscribe ``/cb_<side>_hand_state`` (0..255) and publish
    ``/cb_<side>_hand_control_cmd`` (``sensor_msgs/JointState``).  Requires the
    LinkerHand ROS node to be running (ROS1 / rospy, matching the SDK).
  * ``can``  — talk to the hand directly via ``LinkerHandApi`` (``get_state`` /
    ``finger_move``).  Requires the SDK + a CAN interface.

ROS and SDK imports are deferred into :meth:`run`, so importing this module (and
the rest of :mod:`screwdriver_rl.deploy`) never requires ROS or the SDK.

⚠️ This node cannot be exercised without the physical hand; it is validated
structurally and through the sim gate (``eval.py --deploy_eval``).  The thumb
slot mapping in ``linker_sdk_map`` is provisional — verify on hardware first, and
start with ``--dry-run``.
"""

from __future__ import annotations

import argparse
import time

from screwdriver_rl.deploy.policy import DeployPolicy
from screwdriver_rl.deploy import linker_sdk_map as sdkmap


class LinkerDeployer:
    def __init__(self, ckpt: str, side: str = "left", hz: float = 10.0,
                 device: str = "cpu", dry_run: bool = False) -> None:
        self.policy = DeployPolicy(ckpt, device=device)
        self.side = side
        self.hz = float(hz)
        self.dry_run = dry_run
        self._latest_range20: list[float] | None = None
        self._n_emitted = 0

    # -- shared step: SDK 0..255 state -> command -------------------------- #
    def _step(self, state_range20) -> list[int]:
        finger_q = sdkmap.sdk_range_to_joints16(list(state_range20))
        targets = self.policy.act(finger_q)[0].tolist()  # 16 rad
        cmd = sdkmap.joints16_to_sdk_range(targets)        # 20x 0..255
        self._n_emitted += 1
        if self.dry_run and self._n_emitted <= 5:
            print(f"[deploy] tick {self._n_emitted}: cmd={cmd}", flush=True)
        return cmd

    # -- ROS1 transport ---------------------------------------------------- #
    def _run_ros(self) -> None:
        import rospy
        from sensor_msgs.msg import JointState

        rospy.init_node("screwdriver_rl_deploy", anonymous=True)
        pub = rospy.Publisher(f"/cb_{self.side}_hand_control_cmd", JointState, queue_size=10)
        state_topic = f"/cb_{self.side}_hand_state"

        def on_state(msg: "JointState") -> None:
            if len(msg.position) >= 20:
                self._latest_range20 = list(msg.position[:20])

        rospy.Subscriber(state_topic, JointState, on_state, queue_size=10)
        print(f"[deploy] ROS: sub {state_topic} -> pub /cb_{self.side}_hand_control_cmd "
              f"@ {self.hz} Hz (dry_run={self.dry_run})", flush=True)

        # Seed cur_targets/history once the first state arrives.
        rate = rospy.Rate(self.hz)
        seeded = False
        while not rospy.is_shutdown():
            if self._latest_range20 is not None:
                if not seeded:
                    self.policy.reset(sdkmap.sdk_range_to_joints16(self._latest_range20))
                    seeded = True
                cmd = self._step(self._latest_range20)
                if not self.dry_run:
                    msg = JointState()
                    msg.header.stamp = rospy.Time.now()
                    msg.position = [float(c) for c in cmd]
                    msg.velocity = [0.0] * 20
                    msg.effort = [0.0] * 20
                    pub.publish(msg)
            rate.sleep()

    # -- CAN transport (direct SDK) ---------------------------------------- #
    def _run_can(self) -> None:
        from LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

        api = LinkerHandApi(hand_type=self.side, hand_joint="L20")
        self.policy.reset(sdkmap.sdk_range_to_joints16(list(api.get_state())))
        print(f"[deploy] CAN: LinkerHandApi {self.side} L20 @ {self.hz} Hz "
              f"(dry_run={self.dry_run})", flush=True)
        period = 1.0 / self.hz
        while True:
            t0 = time.time()
            cmd = self._step(list(api.get_state()))
            if not self.dry_run:
                api.finger_move(pose=cmd)
            dt = period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

    def run(self, transport: str = "ros") -> None:
        if transport == "ros":
            self._run_ros()
        elif transport == "can":
            self._run_can()
        else:
            raise ValueError(f"unknown transport {transport!r} (expected 'ros' or 'can')")


def main() -> None:
    p = argparse.ArgumentParser(description="LinkerHand L20 deployment for a Stage-2 deploy.pth.")
    p.add_argument("--checkpoint", required=True, help="Path to stage2_nn/deploy.pth")
    p.add_argument("--side", default="left", choices=["left", "right"])
    p.add_argument("--transport", default="ros", choices=["ros", "can"])
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and print commands but do not send them to the hand.")
    args = p.parse_args()
    LinkerDeployer(
        ckpt=args.checkpoint, side=args.side, hz=args.hz,
        device=args.device, dry_run=args.dry_run,
    ).run(transport=args.transport)


if __name__ == "__main__":
    main()
