#!/usr/bin/env python3
"""Static full-mesh and convex-hull validation for the Linker L20 top-down grasp.

The validator consumes the JSON emitted by ``fit_linker_l20_screwdriver_topdown``
and checks the real hand collision STLs against the independent 64 mm screwdriver
asset.  It is deliberately stricter than the fitting objective and writes a
machine-readable summary.  Isaac physics/contact validation remains a separate,
required gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from screwdriver_rl.utils.linker_topdown_geometry import (  # noqa: E402
    FINGERTIP_LINKS,
    NON_DISTAL_LINKS,
    PALM_NORMAL_LOCAL,
    SCREWDRIVER_ROOT_POS_W,
    TIP_MARKER_LINKS,
    TOPDOWN_CAP_THICKNESS_M,
    TOPDOWN_HANDLE_LENGTH_M,
    TOPDOWN_HANDLE_RADIUS_M,
    UrdfGeometry,
    convex_hulls_intersect,
    iter_unfiltered_self_pairs,
    joint_limit_margins,
    mesh_pair_metric,
    rotate_vector_wxyz,
)
from screwdriver_rl.utils.linker_topdown_grasp_wrench import (  # noqa: E402
    grasp_wrench_gate,
)


HAND_URDF = REPO_ROOT / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
ORIGINAL_SCREWDRIVER_URDF = REPO_ROOT / "assets/screwdriver/screwdriver_isaaclab.urdf"
TOPDOWN_SCREWDRIVER_URDF = REPO_ROOT / "assets/screwdriver/screwdriver_64mm_handle.urdf"
ORIGINAL_SCREWDRIVER_SHA256 = "aa6b2a38f82845e7c0665d2d480114dead56f4457930f90fc765a3d937b6a949"

FINGERS = ("index", "middle", "ring", "pinky", "thumb")
MIN_JOINT_MARGIN_RAD = 0.10
# Isaac stores articulation root quaternions in float32; an exact analytic -Z
# orientation round-trips with about 1e-7 directional error.
MAX_PALM_NORMAL_ERROR = 1.0e-6
MAX_CONTACT_DISTANCE_M = 0.0010
#: Fingertips that must bear on the handle.  Four, not five, because a
#: five-pad grasp is not reachable on this hand: the middle finger can only put
#: its pad on the lateral wall from a configuration that is kinematically stiff
#: along the reset placement randomisation, and it then absorbs that +/-8 mm
#: offset as force -- 26.1 N peak over 256 replicas against an 8 N ceiling.  Its
#: alternative placements are a back contact on the cap top face or no contact,
#: and only the last is acceptable.
MIN_CONTACT_FINGERS = 4
#: A fingertip that is not bearing on the handle has to be clearly clear.  This
#: separates "retracted by design" from "missed the handle by a hair", which the
#: contact count alone cannot distinguish and which would otherwise let a posture
#: silently shed a finger.
MIN_DELIBERATE_CLEARANCE_M = 0.0020
MAX_CONTACT_PENETRATION_M = 0.0010
MIN_NON_DISTAL_CLEARANCE_M = 0.0015
MIN_SELF_CLEARANCE_M = 0.00025
CONVEX_CONTACT_EXPANSION_M = 0.00075


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_link(root: ET.Element, name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == name:
            return link
    raise KeyError(name)


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise KeyError(name)


def _cylinder_radii(link: ET.Element) -> dict[str, float]:
    out: dict[str, float] = {}
    for role in ("visual", "collision"):
        cylinder = link.find(f"{role}/geometry/cylinder")
        if cylinder is None:
            raise AssertionError(f"{link.get('name')} missing {role} cylinder")
        out[role] = float(cylinder.get("radius"))
    return out


def _topology(root: ET.Element) -> dict:
    return {
        "links": sorted(link.get("name") for link in root.findall("link")),
        "joints": sorted(
            (
                joint.get("name"),
                joint.get("type"),
                joint.find("parent").get("link"),
                joint.find("child").get("link"),
            )
            for joint in root.findall("joint")
        ),
    }


def _element_signature(element: ET.Element) -> tuple:
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        tuple(_element_signature(child) for child in element),
    )


def _asset_checks() -> dict:
    original_root = ET.parse(ORIGINAL_SCREWDRIVER_URDF).getroot()
    topdown_root = ET.parse(TOPDOWN_SCREWDRIVER_URDF).getroot()
    original_hash = _sha256(ORIGINAL_SCREWDRIVER_URDF)

    body_radii = _cylinder_radii(_find_link(topdown_root, "screwdriver_body"))
    cap_radii = _cylinder_radii(_find_link(topdown_root, "screwdriver_cap"))

    # Remove only the four confirmed radius attributes and robot name, then the
    # complete XML structure/values must match the source.  Comments are not
    # represented by ElementTree's default parser.
    original_copy = ET.fromstring(ET.tostring(original_root))
    topdown_copy = ET.fromstring(ET.tostring(topdown_root))
    original_copy.set("name", "canonical")
    topdown_copy.set("name", "canonical")
    for root in (original_copy, topdown_copy):
        for link_name in ("screwdriver_body", "screwdriver_cap"):
            link = _find_link(root, link_name)
            for role in ("visual", "collision"):
                link.find(f"{role}/geometry/cylinder").set("radius", "canonical")

    unchanged_except_radii = _element_signature(original_copy) == _element_signature(topdown_copy)
    result = {
        "original_sha256": original_hash,
        "expected_original_sha256": ORIGINAL_SCREWDRIVER_SHA256,
        "original_unchanged": original_hash == ORIGINAL_SCREWDRIVER_SHA256,
        "topology_matches_original": _topology(original_root) == _topology(topdown_root),
        "unchanged_except_robot_name_and_body_cap_radii": unchanged_except_radii,
        "body_radii_m": body_radii,
        "cap_radii_m": cap_radii,
        "body_visual_collision_64mm": all(
            math.isclose(value, TOPDOWN_HANDLE_RADIUS_M, abs_tol=1.0e-12)
            for value in body_radii.values()
        ),
        "cap_visual_collision_64mm": all(
            math.isclose(value, TOPDOWN_HANDLE_RADIUS_M, abs_tol=1.0e-12)
            for value in cap_radii.values()
        ),
    }
    result["pass"] = all(
        (
            result["original_unchanged"],
            result["topology_matches_original"],
            result["unchanged_except_robot_name_and_body_cap_radii"],
            result["body_visual_collision_64mm"],
            result["cap_visual_collision_64mm"],
        )
    )
    return result


def _aabb_distance(first, second) -> float:
    a_min, a_max = first.bounds
    b_min, b_max = second.bounds
    gap = np.maximum(np.maximum(a_min - b_max, b_min - a_max), 0.0)
    return float(np.linalg.norm(gap))


def _metric_dict(metric) -> dict:
    return {
        "first": metric.first,
        "second": metric.second,
        "surface_distance_m": metric.surface_distance_m,
        "penetration_m": metric.penetration_m,
        "intersects": metric.intersects,
    }


def _contact_checks(hand_meshes, screwdriver_meshes) -> dict:
    """Each fingertip must contact the handle; which part of it is role-neutral.

    This used to require the index on ``screwdriver_cap`` specifically.  The task
    itself retired that role: the top-down config sets
    ``role_neutral_fingertip_contact = True``, under which the env zeroes
    ``index_cap_reward`` outright.  Keeping the validator's stricter role would
    reject a posture the task is indifferent to -- and would reject a *better*
    one, since an index on the body adds a fifth tangential drive contact
    instead of pressing the cap axially, which contributes no torque about the
    rotation axis.  Each finger is therefore scored against whichever handle part
    it is closest to, and the recorded ``target`` says which that was.
    """
    candidate_targets = ("screwdriver_body", "screwdriver_cap")
    rows: dict[str, dict] = {}
    passed = True
    for finger, distal in zip(FINGERS, FINGERTIP_LINKS):
        metrics = {
            target: mesh_pair_metric(
                distal,
                hand_meshes[distal],
                target,
                screwdriver_meshes[target],
                sample_points=4000,
            )
            for target in candidate_targets
        }
        # Which part the finger BEARS on is the nearest one.
        target = min(metrics, key=lambda name: metrics[name].surface_distance_m)
        metric = metrics[target]
        # But the penetration limit applies to EVERY part, not just that one.
        # Scoring only the nearest hid a gross interpenetration: a fingertip
        # buried 4.94 mm in screwdriver_body was reported against
        # screwdriver_cap, where it was only 0.50 mm deep, and passed.  Once a
        # tip intersects both parts their surface distances are both exactly
        # zero, so "nearest" stops discriminating and the tie silently selects
        # whichever part the finger happens to be shallower in.
        deepest = max(m.penetration_m for m in metrics.values())
        convex_contact = convex_hulls_intersect(
            hand_meshes[distal],
            screwdriver_meshes[target],
            expansion_m=CONVEX_CONTACT_EXPANSION_M,
        )
        contacting = (
            metric.surface_distance_m <= MAX_CONTACT_DISTANCE_M
            and convex_contact
        )
        # A finger either bears on the handle within the penetration limit, or
        # stands clearly off it.  Hovering in between is rejected: that is how a
        # posture loses a finger without anyone noticing.
        if contacting:
            ok = deepest <= MAX_CONTACT_PENETRATION_M
        else:
            ok = metric.surface_distance_m >= MIN_DELIBERATE_CLEARANCE_M
        row = _metric_dict(metric)
        row.update({
            "target": target,
            "convex_contact": convex_contact,
            "contacting": contacting,
            "deepest_penetration_m": deepest,
            "penetration_by_target_m": {
                name: m.penetration_m for name, m in metrics.items()
            },
            "pass": ok,
        })
        rows[finger] = row
        passed &= ok

    contacting_count = sum(1 for row in rows.values() if row["contacting"])
    passed &= contacting_count >= MIN_CONTACT_FINGERS
    return {
        "fingers": rows,
        "contacting_count": contacting_count,
        "minimum_contacting": MIN_CONTACT_FINGERS,
        "pass": passed,
    }


def _non_distal_checks(
    hand_meshes,
    screwdriver_meshes,
    *,
    min_clearance_m: float = MIN_NON_DISTAL_CLEARANCE_M,
) -> dict:
    rows: dict[str, dict] = {}
    passed = True
    for link in NON_DISTAL_LINKS:
        link_rows: list[dict] = []
        for target in ("screwdriver_body", "screwdriver_cap", "screwdriver_stick"):
            aabb = _aabb_distance(hand_meshes[link], screwdriver_meshes[target])
            if aabb > 0.010:
                link_rows.append(
                    {
                        "target": target,
                        "aabb_distance_m": aabb,
                        "surface_distance_m": aabb,
                        "penetration_m": 0.0,
                        "convex_intersection": False,
                    }
                )
                continue
            metric = mesh_pair_metric(
                link,
                hand_meshes[link],
                target,
                screwdriver_meshes[target],
                sample_points=3000,
            )
            convex_intersection = convex_hulls_intersect(
                hand_meshes[link], screwdriver_meshes[target]
            )
            row = _metric_dict(metric)
            row.update({"target": target, "aabb_distance_m": aabb, "convex_intersection": convex_intersection})
            link_rows.append(row)
        minimum = min(float(row["surface_distance_m"]) for row in link_rows)
        penetration = max(float(row["penetration_m"]) for row in link_rows)
        convex_hit = any(bool(row["convex_intersection"]) for row in link_rows)
        ok = minimum >= min_clearance_m and penetration == 0.0 and not convex_hit
        rows[link] = {
            "minimum_surface_distance_m": minimum,
            "maximum_penetration_m": penetration,
            "any_convex_intersection": convex_hit,
            "targets": link_rows,
            "pass": ok,
        }
        passed &= ok
    return {
        "links": rows,
        "minimum_required_surface_distance_m": float(min_clearance_m),
        "pass": passed,
    }


def _self_collision_checks(hand_model: UrdfGeometry, hand_meshes) -> dict:
    failures: list[dict] = []
    checked: list[dict] = []
    minimum = math.inf
    for first, second in iter_unfiltered_self_pairs(hand_model, hand_meshes):
        aabb = _aabb_distance(hand_meshes[first], hand_meshes[second])
        minimum = min(minimum, aabb)
        if aabb > 0.004:
            continue
        metric = mesh_pair_metric(
            first,
            hand_meshes[first],
            second,
            hand_meshes[second],
            sample_points=1800,
        )
        convex_intersection = convex_hulls_intersect(hand_meshes[first], hand_meshes[second])
        ok = (
            metric.surface_distance_m >= MIN_SELF_CLEARANCE_M
            and metric.penetration_m == 0.0
            and not convex_intersection
        )
        row = _metric_dict(metric)
        row.update({"aabb_distance_m": aabb, "convex_intersection": convex_intersection, "pass": ok})
        checked.append(row)
        minimum = min(minimum, metric.surface_distance_m)
        if not ok:
            failures.append(row)
    return {
        "near_pairs_checked": checked,
        "failures": failures,
        "minimum_unfiltered_pair_clearance_m": minimum,
        "pass": not failures,
    }


def _joint_and_pose_checks(hand_model: UrdfGeometry, posture: dict) -> dict:
    joints = posture["joint_positions_independent"]
    margins = joint_limit_margins(hand_model, joints)
    minimum_margin = min(margins.values())
    quat = posture["root_quat_wxyz"]
    palm_normal = rotate_vector_wxyz(quat, PALM_NORMAL_LOCAL)
    normal_error = float(np.linalg.norm(palm_normal - np.asarray((0.0, 0.0, -1.0))))
    root_z = float(posture["root_pos_w"][2])
    handle_top = float(SCREWDRIVER_ROOT_POS_W[2] + 0.100 + 0.100 + 0.001)
    result = {
        "joint_limit_margins_rad": margins,
        "minimum_joint_limit_margin_rad": minimum_margin,
        "minimum_required_joint_limit_margin_rad": MIN_JOINT_MARGIN_RAD,
        "palm_normal_w": [float(v) for v in palm_normal],
        "palm_normal_error": normal_error,
        "root_z_m": root_z,
        "handle_top_z_m": handle_top,
        "hand_root_above_handle": root_z > handle_top,
    }
    result["pass"] = (
        minimum_margin >= MIN_JOINT_MARGIN_RAD - 1.0e-9
        and normal_error <= MAX_PALM_NORMAL_ERROR
        and result["hand_root_above_handle"]
    )
    return result


def _grasp_wrench_checks(hand_meshes: dict, contact_rows: dict) -> dict:
    """Reject postures that touch the handle but cannot apply torque to it.

    The contact point per finger is the collision vertex closest to the handle's
    lateral surface, which is where a tangential drive force would actually act.

    Only fingers contacting the *body* are held to the distribution
    requirements.  A finger on the cap presses along the rotation axis and
    cannot contribute torque about it, so requiring it to sit inside the lateral
    height band would be meaningless.  That is decided from the measured nearest
    target rather than by finger name, so the check follows whatever role the
    posture actually adopts -- the task is role-neutral
    (``role_neutral_fingertip_contact = True``) and does not fix the index to the
    cap.
    """
    axis_xy = np.asarray(SCREWDRIVER_ROOT_POS_W[:2], dtype=np.float64)
    body_base_z = float(SCREWDRIVER_ROOT_POS_W[2]) + 0.100
    body_top_z = body_base_z + TOPDOWN_HANDLE_LENGTH_M
    # The lateral surface spans body AND cap: they share a radius, so the side
    # wall is continuous and the nearest lateral point may lie on either.
    lateral_top_z = body_top_z + TOPDOWN_CAP_THICKNESS_M

    contacts: dict[str, list[float]] = {}
    for finger, link in zip(FINGERS, FINGERTIP_LINKS):
        vertices = np.asarray(hand_meshes[link].vertices, dtype=np.float64)
        radial = np.linalg.norm(vertices[:, :2] - axis_xy, axis=1)
        distance = (
            np.abs(radial - TOPDOWN_HANDLE_RADIUS_M)
            + np.maximum(0.0, body_base_z - vertices[:, 2])
            + np.maximum(0.0, vertices[:, 2] - lateral_top_z)
        )
        contacts[finger] = [float(v) for v in vertices[int(np.argmin(distance))]]

    # A finger drives the handle only if its contact PATCH lies on the 100 mm
    # body side wall, where a tangential force produces torque about the axis.
    #
    # Two weaker versions of this test were tried and both passed grasps that
    # cannot drive.  Classifying by link name excluded genuine side-wall
    # contacts sitting near the top edge.  Classifying by contact radius alone
    # then admitted contacts on the flat cap top: the outer rim of that face is
    # at exactly the handle radius, so an edge vertex satisfies a radius test
    # while the bulk of the patch presses axially.  Measured on one such
    # posture, index/middle/ring had 115/582/494 vertices on the top face and
    # 0/0/42 on the side wall -- three fingers pressing the lid, reported as
    # three lateral drivers.
    #
    # So count vertices actually near the side wall.  For reference, the
    # configuration with measured fall-rate and net-turn evidence has 561/378/240
    # side-wall vertices on index/ring/thumb: three real drivers.
    patch_tolerance_m = 0.0005
    minimum_patch_vertices = 20
    driving = []
    side_wall_vertices: dict[str, int] = {}
    for finger, link in zip(FINGERS, FINGERTIP_LINKS):
        vertices = np.asarray(hand_meshes[link].vertices, dtype=np.float64)
        radial = np.linalg.norm(vertices[:, :2] - axis_xy, axis=1)
        wall_distance = (
            np.abs(radial - TOPDOWN_HANDLE_RADIUS_M)
            + np.maximum(0.0, body_base_z - vertices[:, 2])
            + np.maximum(0.0, vertices[:, 2] - body_top_z)
        )
        count = int((wall_distance < patch_tolerance_m).sum())
        side_wall_vertices[finger] = count
        if count >= minimum_patch_vertices:
            driving.append(finger)
    if len(driving) < 3:
        return {
            "pass": False,
            "reason": (
                "fewer than three fingers bear on the 100 mm body side wall; "
                "contacts on the cap top face press axially and cannot drive"
            ),
            "evaluated_fingers": driving,
            "side_wall_vertices": side_wall_vertices,
        }
    result = grasp_wrench_gate(
        contacts, axis_xy, body_base_z, body_top_z, active_fingers=driving
    )
    result["side_wall_vertices"] = side_wall_vertices
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posture-json",
        type=Path,
        default=REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/posture_fit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/linker_l20_screwdriver_topdown/validation_summary.json",
    )
    args = parser.parse_args()

    posture = json.loads(args.posture_json.read_text())
    hand_model = UrdfGeometry(HAND_URDF)
    screwdriver_model = UrdfGeometry(TOPDOWN_SCREWDRIVER_URDF)
    hand_meshes = hand_model.collision_meshes_world(
        posture["joint_positions_independent"],
        posture["root_pos_w"],
        posture["root_quat_wxyz"],
    )
    screwdriver_joint_positions = posture.get("screwdriver_joint_positions", {})
    screwdriver_meshes = screwdriver_model.collision_meshes_world(
        screwdriver_joint_positions,
        SCREWDRIVER_ROOT_POS_W,
        (1.0, 0.0, 0.0, 0.0),
    )

    summary = {
        "posture_json": str(args.posture_json),
        "thresholds": {
            "max_contact_distance_m": MAX_CONTACT_DISTANCE_M,
            "max_contact_penetration_m": MAX_CONTACT_PENETRATION_M,
            "min_non_distal_clearance_m": MIN_NON_DISTAL_CLEARANCE_M,
            "min_self_clearance_m": MIN_SELF_CLEARANCE_M,
            "min_joint_margin_rad": MIN_JOINT_MARGIN_RAD,
        },
        "asset": _asset_checks(),
        "pose_and_joint_limits": _joint_and_pose_checks(hand_model, posture),
        "fingertip_contacts": (contacts := _contact_checks(hand_meshes, screwdriver_meshes)),
        "non_distal_clearance": _non_distal_checks(hand_meshes, screwdriver_meshes),
        "unfiltered_self_collision": _self_collision_checks(hand_model, hand_meshes),
        "grasp_wrench": _grasp_wrench_checks(hand_meshes, contacts["fingers"]),
    }
    summary["static_validation_pass"] = all(
        summary[key]["pass"]
        for key in (
            "asset",
            "pose_and_joint_limits",
            "fingertip_contacts",
            "non_distal_clearance",
            "unfiltered_self_collision",
            "grasp_wrench",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.output}", flush=True)
    raise SystemExit(0 if summary["static_validation_pass"] else 1)


if __name__ == "__main__":
    main()
