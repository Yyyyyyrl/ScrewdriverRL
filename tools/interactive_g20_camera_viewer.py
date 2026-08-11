#!/usr/bin/env python3
"""Open an interactive Isaac viewport and continuously save its camera pose."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


SEMANTIC_ORDER = (
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
    "thumb_cmc_yaw",
    "thumb_cmc_roll",
    "thumb_cmc_pitch",
    "thumb_mcp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--pose-name", required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    poses = json.loads(args.poses.read_text(encoding="utf-8"))
    semantic = poses[args.pose_name]
    if len(semantic) != len(SEMANTIC_ORDER):
        raise ValueError(f"{args.pose_name} does not contain 16 semantic joints")
    intr = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    initial_view = json.loads(args.camera.read_text(encoding="utf-8"))[0]

    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=False, enable_cameras=True).app

    import imageio.v3 as iio
    import numpy as np
    import torch
    import xml.etree.ElementTree as ET
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import Camera, CameraCfg
    from isaacsim.core.prims import XFormPrim
    from isaaclab.utils.math import (
        convert_camera_frame_orientation_convention,
        quat_apply,
    )
    from pxr import Gf, Usd, UsdGeom

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0")
    )
    dome = sim_utils.DomeLightCfg(
        intensity=650.0, color=(0.82, 0.86, 0.92)
    )
    dome.func("/World/dome", dome)
    key = sim_utils.DistantLightCfg(
        intensity=900.0, color=(1.0, 0.95, 0.88)
    )
    key.func("/World/key", key, orientation=(0.86, 0.28, 0.42, 0.06))

    hand = Articulation(
        ArticulationCfg(
            prim_path="/World/LinkerHand",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(args.urdf.resolve()),
                fix_base=True,
                merge_fixed_joints=False,
                replace_cylinders_with_capsules=False,
                make_instanceable=False,
                collider_type="convex_hull",
                self_collision=False,
                activate_contact_sensors=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=None, damping=None
                    )
                ),
            ),
            actuators={
                "all": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness=1000.0, damping=100.0
                )
            },
        )
    )
    camera = Camera(
        CameraCfg(
            prim_path="/World/d435",
            update_period=0.0,
            height=int(intr["height"]),
            width=int(intr["width"]),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=[
                    intr["fx"],
                    0.0,
                    intr["ppx"],
                    0.0,
                    intr["fy"],
                    intr["ppy"],
                    0.0,
                    0.0,
                    1.0,
                ],
                width=int(intr["width"]),
                height=int(intr["height"]),
                focal_length=24.0,
                clipping_range=(0.01, 10.0),
            ),
        )
    )
    sim.reset()

    joint_names = list(hand.joint_names)
    index_of = {name: joint_names.index(name) for name in joint_names}
    applied = dict(zip(SEMANTIC_ORDER, semantic))
    q = hand.data.default_joint_pos.clone()
    for joint_name, value in applied.items():
        q[0, index_of[joint_name]] = float(value)
    for joint in ET.parse(args.urdf).getroot().findall("joint"):
        mimic = joint.find("mimic")
        follower = str(joint.get("name"))
        if mimic is not None and follower in index_of:
            source = str(mimic.get("joint"))
            multiplier = float(mimic.get("multiplier", "1"))
            offset = float(mimic.get("offset", "0"))
            q[0, index_of[follower]] = applied[source] * multiplier + offset
    hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    hand.set_joint_position_target(q)
    hand.reset()

    camera.set_world_poses(
        positions=torch.tensor([initial_view["pos"]], device=sim.device),
        orientations=torch.tensor([initial_view["quat"]], device=sim.device),
        convention="ros",
    )
    for _ in range(4):
        sim.render()
        camera.update(dt=0.0)

    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    # Use the freely navigable Perspective camera, initialize its navigation
    # controller from the D435 optical direction, then restore the *exact*
    # OpenGL camera transform (including roll) with Kit's USD transform command.
    nav_path = "/OmniverseKit_Persp"
    viewport.camera_path = nav_path
    nav_camera = XFormPrim(nav_path)
    initial_pos = torch.tensor([initial_view["pos"]], device=sim.device)
    initial_quat_ros = torch.tensor([initial_view["quat"]], device=sim.device)
    initial_quat_opengl = convert_camera_frame_orientation_convention(
        initial_quat_ros, origin="ros", target="opengl"
    )
    for _ in range(20):
        sim.render()
    forward_world = quat_apply(
        initial_quat_ros,
        torch.tensor([[0.0, 0.0, 1.0]], device=sim.device),
    )
    initial_target = initial_pos + 0.35 * forward_world
    sim.set_camera_view(
        eye=initial_pos[0].detach().cpu().tolist(),
        target=initial_target[0].detach().cpu().tolist(),
        camera_prim_path=nav_path,
    )
    sim.render()

    import omni.kit.commands

    nav_prim = viewport.stage.GetPrimAtPath(nav_path)
    usd_camera = UsdGeom.Camera(nav_prim)
    parent_world = usd_camera.ComputeParentToWorldTransform(
        Usd.TimeCode.Default()
    )
    old_local = UsdGeom.Xformable(nav_prim).GetLocalTransformation(
        Usd.TimeCode.Default()
    )
    pos_values = initial_pos[0].detach().cpu().tolist()
    quat_values = initial_quat_opengl[0].detach().cpu().tolist()
    exact_world = Gf.Matrix4d(1.0)
    exact_world.SetRotate(
        Gf.Quatd(
            float(quat_values[0]),
            Gf.Vec3d(*[float(value) for value in quat_values[1:]]),
        )
    )
    exact_world.SetTranslateOnly(
        Gf.Vec3d(*[float(value) for value in pos_values])
    )
    exact_local = exact_world * parent_world.GetInverse()
    omni.kit.commands.create(
        "TransformPrimCommand",
        path=nav_path,
        new_transform_matrix=exact_local,
        old_transform_matrix=old_local,
        time_code=Usd.TimeCode.Default(),
        usd_context_name=viewport.usd_context_name,
    ).do()
    nav_prim.GetAttribute("omni:kit:centerOfInterest").Set(
        Gf.Vec3d(0.0, 0.0, -0.35)
    )
    sim.render()
    pose_path = args.out_dir / "live_camera_ros.json"
    preview_path = args.out_dir / "live_camera_preview.png"
    session_path = args.out_dir / "interactive_session.json"
    session_path.write_text(
        json.dumps(
            {
                "urdf": str(args.urdf.resolve()),
                "poses": str(args.poses.resolve()),
                "pose_name": args.pose_name,
                "intrinsics": str(args.intrinsics.resolve()),
                "initial_camera": initial_view,
                "camera_pose_autosave": str(pose_path.resolve()),
                "preview_autosave": str(preview_path.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("[interactive-camera] Isaac viewport is ready.", flush=True)
    print(
        "[interactive-camera] Navigate the exact-roll Perspective view; "
        "camera pose and "
        f"preview autosave under {args.out_dir}.",
        flush=True,
    )

    frame = 0
    last_save = 0.0
    while app.is_running():
        pos_tensor, quat_opengl = nav_camera.get_world_poses()
        quat_ros_tensor = convert_camera_frame_orientation_convention(
            quat_opengl, origin="opengl", target="ros"
        )
        camera.set_world_poses(
            positions=pos_tensor,
            orientations=quat_ros_tensor,
            convention="ros",
        )
        sim.render()
        camera.update(dt=0.0)
        now = time.monotonic()
        if now - last_save >= 0.25:
            pos = pos_tensor[0].detach().cpu().tolist()
            quat = quat_ros_tensor[0].detach().cpu().tolist()
            pose_path.write_text(
                json.dumps(
                    [
                        {
                            "name": "interactive_saved",
                            "pos": [float(value) for value in pos],
                            "quat": [float(value) for value in quat],
                        }
                    ],
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if frame % 8 == 0:
                rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
                iio.imwrite(preview_path, rgb[..., :3])
            last_save = now
        frame += 1

    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
