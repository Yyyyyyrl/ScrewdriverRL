#!/usr/bin/env python3
"""Render the LinkerHand G20 URDF at given joint poses from a D435-matched camera.

Built for the sim->SDK->hardware visual A/B check.  The hand is spawned alone at
the world origin with a fixed base, so the camera pose given on the command line
is directly the camera's pose **in the hand base frame** -- which is what the
extrinsic solve produces and what makes the render comparable to the photograph.

The camera uses the D435 colour stream's measured intrinsics rather than a
nominal FOV.  That stream reports zero distortion coefficients, so a pinhole
model is exact and no undistortion step is needed on either side.

Poses are semantic 16-vectors in the repo's canonical joint order.  Mimic joints
are *not* passed in: they are resolved from the URDF's own mimic tags, so a
render can never disagree with the URDF about the coupling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"

SEMANTIC_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="JSON: {name: [16 semantic rad]}")
    parser.add_argument("--intrinsics", required=True, help="JSON from the D435 colour profile")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=HAND_URDF,
        help=(
            "URDF to render; pass the isolated diagnostic asset explicitly for "
            "unlocked mimic-follower registration"
        ),
    )
    parser.add_argument(
        "--camera-pos", nargs=3, type=float,
        help="camera position in the hand base frame (metres)",
    )
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument(
        "--camera-quat", nargs=4, type=float,
        help="camera orientation as w x y z in the hand base frame (ROS optical convention)",
    )
    orientation.add_argument(
        "--look-at", nargs=3, type=float,
        help="point the camera at this hand-base-frame position instead of giving a quaternion",
    )
    parser.add_argument(
        "--cameras",
        help=(
            "JSON list of {name, pos, look_at|quat} rendered in one process. "
            "URDF->USD conversion dominates start-up, so sweeping viewpoints in a "
            "single run is far cheaper than one process per viewpoint."
        ),
    )
    parser.add_argument(
        "--up", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        help="world up hint used to roll the camera when --look-at is given",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dump-fk", action="store_true", help="also write per-link FK poses")
    parser.add_argument(
        "--physics-mode",
        choices=("kinematic", "filtered", "unfiltered"),
        default="kinematic",
        help=(
            "kinematic writes the requested joint state and renders without a physics "
            "step; filtered settles with the training environment's self-collision "
            "filters; unfiltered is a diagnostic reproduction of the old renderer"
        ),
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=60,
        help="physics steps used by filtered/unfiltered modes",
    )
    parser.add_argument(
        "--background-gray",
        type=float,
        default=1.0,
        help=(
            "dome/background gray level in [0,1]; use about 0.55 for a "
            "clear white-hand silhouette"
        ),
    )
    parser.add_argument(
        "--focus-finger",
        choices=("none", "index", "middle", "ring", "pinky", "thumb"),
        default="none",
        help=(
            "visually hide every non-target finger while retaining the palm; "
            "this changes only USD visibility, never joints or physics"
        ),
    )
    parser.add_argument("--headless", action="store_true", default=True)
    return parser.parse_args()


def _look_at_quat_ros(position, target, up) -> list[float]:
    """Quaternion (w x y z) aiming a ROS-optical camera from ``position`` at ``target``.

    ROS optical frame is +Z forward, +X right, +Y *down*, which is why the second
    axis comes out of ``cross(forward, right)`` rather than the usual up vector.
    """
    import numpy as np

    position = np.asarray(position, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - position
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("--look-at coincides with --camera-pos")
    forward /= norm
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("--up is parallel to the view direction; pass a different --up")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    rot = np.column_stack((right, down, forward))
    trace = np.trace(rot)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        i = int(np.argmax(np.diag(rot)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
        q = [0.0, 0.0, 0.0]
        w = (rot[k, j] - rot[j, k]) / s
        q[i] = 0.25 * s
        q[j] = (rot[j, i] + rot[i, j]) / s
        q[k] = (rot[k, i] + rot[i, k]) / s
        x, y, z = q
    quat = np.array([w, x, y, z], dtype=float)
    return (quat / np.linalg.norm(quat)).tolist()


def main() -> int:
    args = parse_args()
    hand_urdf = args.urdf.resolve()
    if not hand_urdf.is_file():
        raise FileNotFoundError(hand_urdf)
    if not 0.0 <= args.background_gray <= 1.0:
        raise ValueError("--background-gray must be in [0,1]")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses: dict[str, object] = json.loads(Path(args.poses).read_text())
    intr = json.loads(Path(args.intrinsics).read_text())
    resolved_poses: dict[str, dict[str, object]] = {}
    for name, spec in poses.items():
        if isinstance(spec, list):
            semantic = spec
            overrides: dict[str, float] = {}
        elif isinstance(spec, dict):
            semantic = spec.get("semantic")
            overrides = {
                str(joint): float(value)
                for joint, value in spec.get("joint_overrides", {}).items()
            }
        else:
            raise ValueError(
                f"pose {name!r} must be a 16-vector or a pose specification"
            )
        if not isinstance(semantic, list) or len(semantic) != 16:
            length = len(semantic) if isinstance(semantic, list) else "non-list"
            raise ValueError(
                f"pose {name!r} has semantic length {length}, expected 16"
            )
        resolved_poses[name] = {
            "semantic": [float(value) for value in semantic],
            "joint_overrides": overrides,
        }

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=args.headless, enable_cameras=True).app

    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    )

    # Flat, shadow-light illumination: the goal is a readable silhouette and
    # joint angles, not a photometric match to the room.  A dome fills the
    # underside so downward-facing links do not go to black.
    background = (args.background_gray,) * 3
    dome = sim_utils.DomeLightCfg(intensity=1200.0, color=background)
    dome.func("/World/dome", dome)
    key = sim_utils.DistantLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0))
    key.func("/World/key", key, orientation=(0.86, 0.28, 0.42, 0.06))

    physics_enabled = args.physics_mode != "kinematic"
    hand_cfg = ArticulationCfg(
        prim_path="/World/LinkerHand",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(hand_urdf),
            fix_base=True,
            merge_fixed_joints=False,
            replace_cylinders_with_capsules=False,
            make_instanceable=False,
            collider_type="convex_hull",
            self_collision=physics_enabled,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=physics_enabled,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        # Poses are written straight into the joint state, so the drive only has
        # to hold them still; it is never asked to track a trajectory here.
        actuators={
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"], stiffness=1000.0, damping=100.0
            )
        },
    )
    hand = Articulation(hand_cfg)

    camera_cfg = CameraCfg(
        prim_path="/World/d435",
        update_period=0.0,
        height=int(intr["height"]),
        width=int(intr["width"]),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=[
                intr["fx"], 0.0, intr["ppx"],
                0.0, intr["fy"], intr["ppy"],
                0.0, 0.0, 1.0,
            ],
            width=int(intr["width"]),
            height=int(intr["height"]),
            focal_length=24.0,
            clipping_range=(0.01, 10.0),
        ),
    )
    camera = Camera(camera_cfg)

    sim.reset()

    filtered_pair_count = 0
    if args.physics_mode == "filtered":
        import omni.usd
        from pxr import Sdf, UsdPhysics
        from screwdriver_rl.tasks.linker_l20.screwdriver_rotation_env import (
            LinkerL20ScrewdriverRotationEnv,
        )

        stage = omni.usd.get_context().get_stage()
        for link_a, link_b in (
            LinkerL20ScrewdriverRotationEnv.SELF_COLLISION_FILTER_PAIRS
        ):
            a_path = f"/World/LinkerHand/{link_a}"
            b_path = f"/World/LinkerHand/{link_b}"
            prim_a = stage.GetPrimAtPath(a_path)
            prim_b = stage.GetPrimAtPath(b_path)
            if not prim_a.IsValid() or not prim_b.IsValid():
                raise RuntimeError(
                    f"missing self-collision filter prim: {a_path} or {b_path}"
                )
            api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
            api.CreateFilteredPairsRel().AddTarget(Sdf.Path(b_path))
            filtered_pair_count += 1
        print(
            f"[render] applied {filtered_pair_count} training self-collision filters",
            flush=True,
        )

    if args.cameras:
        views = json.loads(Path(args.cameras).read_text())
    else:
        if args.camera_pos is None:
            raise ValueError("give either --cameras or --camera-pos with an orientation")
        views = [{
            "name": "cam",
            "pos": list(args.camera_pos),
            **({"quat": list(args.camera_quat)} if args.camera_quat is not None
               else {"look_at": list(args.look_at)}),
        }]
    for view in views:
        view["quat"] = (
            list(view["quat"]) if "quat" in view
            else _look_at_quat_ros(view["pos"], view["look_at"], view.get("up", args.up))
        )

    def aim(view: dict) -> None:
        camera.set_world_poses(
            positions=torch.tensor([view["pos"]], device=sim.device),
            orientations=torch.tensor([view["quat"]], device=sim.device),
            convention="ros",
        )

    joint_names = list(hand.joint_names)
    index_of = {name: joint_names.index(name) for name in joint_names}
    missing = [name for name in SEMANTIC_ORDER if name not in index_of]
    if missing:
        raise RuntimeError(f"articulation is missing semantic joints: {missing}")
    unknown_overrides = sorted(
        {
            joint_name
            for spec in resolved_poses.values()
            for joint_name in spec["joint_overrides"]
            if joint_name not in index_of
        }
    )
    if unknown_overrides:
        raise RuntimeError(
            f"pose joint_overrides reference missing joints: {unknown_overrides}"
        )

    hidden_body_names: list[str] = []
    if args.focus_finger != "none":
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        keep_prefix = f"{args.focus_finger}_"
        for body_name in hand.body_names:
            if body_name == "hand_base_link" or body_name.startswith(keep_prefix):
                continue
            prim_path = f"/World/LinkerHand/{body_name}"
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(f"missing body prim for visual focus: {prim_path}")
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden_body_names.append(body_name)
        print(
            f"[render] focus_finger={args.focus_finger}; visually hid "
            f"{len(hidden_body_names)} non-target bodies",
            flush=True,
        )

    # Mimic coupling read straight out of the URDF so a render cannot disagree
    # with the signed asset about the multipliers.
    import xml.etree.ElementTree as ET

    mimic: dict[str, tuple[str, float, float]] = {}
    for joint in ET.parse(hand_urdf).getroot().findall("joint"):
        tag = joint.find("mimic")
        if tag is not None:
            mimic[str(joint.get("name"))] = (
                str(tag.get("joint")),
                float(tag.get("multiplier", "1")),
                float(tag.get("offset", "0")),
            )

    import imageio.v3 as iio

    manifest: dict[str, dict] = {}
    for name, spec in resolved_poses.items():
        semantic = spec["semantic"]
        overrides = spec["joint_overrides"]
        q = hand.data.default_joint_pos.clone()
        applied = dict(zip(SEMANTIC_ORDER, semantic))
        for joint_name, value in applied.items():
            q[0, index_of[joint_name]] = float(value)
        for follower, (source, mult, offset) in mimic.items():
            if follower in index_of:
                q[0, index_of[follower]] = applied[source] * mult + offset
        # Explicit follower overrides are applied after URDF mimic resolution.
        # This is used only with an isolated diagnostic URDF whose mimic tag was
        # removed, allowing local-q calibration without mixing parent-q error
        # into the follower measurement.
        for joint_name, value in overrides.items():
            q[0, index_of[joint_name]] = float(value)
        zeros = torch.zeros_like(q)
        hand.write_joint_state_to_sim(q, zeros)
        hand.set_joint_position_target(q)
        hand.reset()
        if physics_enabled:
            for _ in range(args.settle_steps):
                sim.step(render=False)
        hand.update(dt=0.0)
        actual_q = hand.data.joint_pos[0].detach().cpu()
        requested_q = q[0].detach().cpu()
        drift = actual_q - requested_q

        images: dict[str, str] = {}
        for view in views:
            aim(view)
            for _ in range(3):
                sim.render()
                camera.update(dt=0.0)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.dtype != np.uint8:
                rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            suffix = "_sim" if len(views) == 1 else f"_{view['name']}_sim"
            path = out_dir / f"{name}{suffix}.png"
            iio.imwrite(path, rgb[..., :3])
            images[view["name"]] = path.name

        entry = {
            "semantic": list(map(float, semantic)),
            "joint_overrides": {
                joint: float(value) for joint, value in overrides.items()
            },
            "images": images,
            "requested_joint_pos": {
                joint: float(requested_q[i]) for i, joint in enumerate(joint_names)
            },
            "rendered_joint_pos": {
                joint: float(actual_q[i]) for i, joint in enumerate(joint_names)
            },
            "joint_drift_rad": {
                joint: float(drift[i]) for i, joint in enumerate(joint_names)
            },
            "max_abs_joint_drift_rad": float(torch.max(torch.abs(drift))),
        }
        if args.dump_fk:
            body_names = list(hand.body_names)
            positions = hand.data.body_pos_w[0].detach().cpu().numpy()
            quats = hand.data.body_quat_w[0].detach().cpu().numpy()
            entry["fk"] = {
                body: {"pos": positions[i].tolist(), "quat_wxyz": quats[i].tolist()}
                for i, body in enumerate(body_names)
            }
        manifest[name] = entry
        print(f"[render] {name} -> {len(images)} view(s)", flush=True)

    (out_dir / "render_manifest.json").write_text(
        json.dumps(
            {
                "urdf": str(hand_urdf),
                "urdf_sha256": hashlib.sha256(hand_urdf.read_bytes()).hexdigest(),
                "intrinsics": intr,
                "cameras_hand_base": views,
                "physics_mode": args.physics_mode,
                "settle_steps": args.settle_steps if physics_enabled else 0,
                "training_self_collision_filter_count": filtered_pair_count,
                "background_gray": args.background_gray,
                "focus_finger": args.focus_finger,
                "hidden_body_names": hidden_body_names,
                "semantic_order": list(SEMANTIC_ORDER),
                "poses": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[render] wrote {out_dir/'render_manifest.json'}", flush=True)
    # ``app.close()`` leaves Omniverse worker threads alive, so the interpreter
    # never reaches the interpreter exit and the process keeps its ~3 GB of VRAM.
    # Six of these left running in parallel is enough to exhaust a 24 GB card, so
    # tear the process down hard once the images are on disk.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
