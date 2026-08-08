"""Pure-Python geometry helpers for the Linker L20 top-down screwdriver grasp.

The module deliberately has no Isaac Lab dependency.  It parses the real URDFs,
performs URDF forward kinematics, loads the collision STL meshes with ``trimesh``
and provides deterministic mesh/convex-hull checks used by both the fitting tool
and CPU-only regression tests.

Isaac Lab still remains the final authority for cooked-collider/contact behavior;
these helpers are the static validation layer that runs before physics startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import trimesh

try:  # Optional exact triangle-query accelerator used by the validation CLI.
    import open3d as o3d
except ImportError:  # pragma: no cover - trimesh remains the portable fallback
    o3d = None


TOPDOWN_HANDLE_RADIUS_M = 0.032
TOPDOWN_HANDLE_LENGTH_M = 0.100
TOPDOWN_CAP_THICKNESS_M = 0.001
SCREWDRIVER_ROOT_POS_W = np.asarray((-0.009, 0.0, 1.205), dtype=np.float64)

# The L20 hand-base frame uses local +X as the outward palm normal and local +Z
# as the wrist-to-fingers longitudinal axis.  This fixed base rotation maps +X
# exactly to world -Z and +Z to world -Y.  A world-Z yaw may be left-multiplied
# without changing the palm-normal constraint.
PALM_DOWN_BASE_QUAT_WXYZ = np.asarray((0.5, 0.5, 0.5, -0.5), dtype=np.float64)
PALM_NORMAL_LOCAL = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
HAND_LONG_AXIS_LOCAL = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

FINGERTIP_LINKS = (
    "index_distal",
    "middle_distal",
    "ring_distal",
    "pinky_distal",
    "thumb_distal",
)
TIP_MARKER_LINKS = (
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
    "thumb_tip",
)
NON_DISTAL_LINKS = (
    "hand_base_link",
    "index_metacarpals",
    "index_proximal",
    "index_middle",
    "middle_metacarpals",
    "middle_proximal",
    "middle_middle",
    "ring_metacarpals",
    "ring_proximal",
    "ring_middle",
    "pinky_metacarpals",
    "pinky_proximal",
    "pinky_middle",
    "thumb_metacarpals_base2",
    "thumb_metacarpals_base1",
    "thumb_metacarpals",
    "thumb_proximal",
)

# Must stay equal to LinkerL20ScrewdriverRotationEnv.SELF_COLLISION_FILTER_PAIRS.
# These are importer convex-hull artifacts close to rigid/adjacent palm geometry;
# every other non-adjacent pair remains collision-active and must be clear.
SELF_COLLISION_FILTER_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("hand_base_link", "index_proximal"),
        ("hand_base_link", "middle_proximal"),
        ("hand_base_link", "ring_proximal"),
        ("hand_base_link", "pinky_proximal"),
        ("hand_base_link", "thumb_metacarpals_base1"),
        ("hand_base_link", "thumb_metacarpals"),
        ("hand_base_link", "thumb_proximal"),
        ("index_metacarpals", "middle_metacarpals"),
        ("middle_metacarpals", "ring_metacarpals"),
        ("ring_metacarpals", "pinky_metacarpals"),
        ("thumb_metacarpals_base2", "index_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_metacarpals"),
        ("thumb_metacarpals_base2", "thumb_proximal"),
        ("thumb_metacarpals_base2", "thumb_distal"),
        ("thumb_metacarpals_base1", "thumb_proximal"),
        ("thumb_metacarpals_base1", "thumb_distal"),
        ("thumb_metacarpals", "thumb_distal"),
    )
)


def _vec(text: str | None, default: Iterable[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if text is None:
        return np.asarray(tuple(default), dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm == 0.0 or angle == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.asarray(
        (
            (c + x * x * C, x * y * C - z * s, x * z * C + y * s),
            (y * x * C + z * s, c + y * y * C, y * z * C - x * s),
            (z * x * C - y * s, z * y * C + x * s, c + z * z * C),
        ),
        dtype=np.float64,
    )


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw rotation matrix."""

    roll, pitch, yaw = (float(v) for v in rpy)
    return (
        _axis_angle_matrix(np.asarray((0.0, 0.0, 1.0)), yaw)
        @ _axis_angle_matrix(np.asarray((0.0, 1.0, 0.0)), pitch)
        @ _axis_angle_matrix(np.asarray((1.0, 0.0, 0.0)), roll)
    )


def transform(xyz: Iterable[float] = (0.0, 0.0, 0.0), rotation: np.ndarray | None = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64)
    out[:3, 3] = np.asarray(tuple(xyz), dtype=np.float64)
    return out


def quat_normalize_wxyz(quat: Iterable[float]) -> np.ndarray:
    q = np.asarray(tuple(quat), dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        raise ValueError("zero quaternion")
    return q / norm


def quat_multiply_wxyz(lhs: Iterable[float], rhs: Iterable[float]) -> np.ndarray:
    aw, ax, ay, az = quat_normalize_wxyz(lhs)
    bw, bx, by, bz = quat_normalize_wxyz(rhs)
    return quat_normalize_wxyz(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quat_to_matrix_wxyz(quat: Iterable[float]) -> np.ndarray:
    w, x, y, z = quat_normalize_wxyz(quat)
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def yaw_quaternion_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.asarray((math.cos(half), 0.0, 0.0, math.sin(half)), dtype=np.float64)


def palm_down_quaternion_wxyz(yaw: float = 0.0) -> np.ndarray:
    return quat_multiply_wxyz(yaw_quaternion_wxyz(yaw), PALM_DOWN_BASE_QUAT_WXYZ)


def rotate_vector_wxyz(quat: Iterable[float], vector: Iterable[float]) -> np.ndarray:
    return quat_to_matrix_wxyz(quat) @ np.asarray(tuple(vector), dtype=np.float64)


@dataclass(frozen=True)
class Mimic:
    source: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    mimic: Mimic | None


class UrdfGeometry:
    """Small URDF FK/collision-mesh loader supporting this repository's assets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.root = ET.parse(self.path).getroot()
        self.links = {link.get("name"): link for link in self.root.findall("link")}
        self.joints: dict[str, Joint] = {}
        self.children: dict[str, list[Joint]] = {name: [] for name in self.links}
        child_links: set[str] = set()

        for element in self.root.findall("joint"):
            origin_elem = element.find("origin")
            xyz = _vec(None if origin_elem is None else origin_elem.get("xyz"))
            rpy = _vec(None if origin_elem is None else origin_elem.get("rpy"))
            parent = element.find("parent").get("link")
            child = element.find("child").get("link")
            axis_elem = element.find("axis")
            axis = _vec(None if axis_elem is None else axis_elem.get("xyz"), (1.0, 0.0, 0.0))
            limit = element.find("limit")
            mimic_elem = element.find("mimic")
            mimic = None
            if mimic_elem is not None:
                mimic = Mimic(
                    source=mimic_elem.get("joint"),
                    multiplier=float(mimic_elem.get("multiplier", "1")),
                    offset=float(mimic_elem.get("offset", "0")),
                )
            joint = Joint(
                name=element.get("name"),
                kind=element.get("type"),
                parent=parent,
                child=child,
                origin=transform(xyz, _rpy_matrix(rpy)),
                axis=axis,
                lower=None if limit is None or limit.get("lower") is None else float(limit.get("lower")),
                upper=None if limit is None or limit.get("upper") is None else float(limit.get("upper")),
                mimic=mimic,
            )
            self.joints[joint.name] = joint
            self.children.setdefault(parent, []).append(joint)
            child_links.add(child)

        roots = set(self.links).difference(child_links)
        if len(roots) != 1:
            raise ValueError(f"expected one URDF root link in {self.path}, got {sorted(roots)}")
        self.root_link = next(iter(roots))
        self._collision_mesh_cache: dict[str, trimesh.Trimesh | None] = {}

    @property
    def independent_joint_names(self) -> tuple[str, ...]:
        return tuple(
            joint.name
            for joint in self.joints.values()
            if joint.kind in {"revolute", "prismatic", "continuous"} and joint.mimic is None
        )

    def joint_limits(self, margin: float = 0.0) -> dict[str, tuple[float, float]]:
        limits: dict[str, tuple[float, float]] = {}
        for name in self.independent_joint_names:
            joint = self.joints[name]
            if joint.lower is None or joint.upper is None:
                continue
            lo = joint.lower + margin
            hi = joint.upper - margin
            if lo > hi:
                raise ValueError(f"joint {name} has no room for margin {margin}")
            limits[name] = (lo, hi)
        return limits

    def expanded_positions(self, independent: Mapping[str, float]) -> dict[str, float]:
        q = {
            name: float(independent.get(name, 0.0))
            for name in self.independent_joint_names
        }
        unresolved = {joint.name for joint in self.joints.values() if joint.mimic is not None}
        while unresolved:
            progress = False
            for name in tuple(unresolved):
                mimic = self.joints[name].mimic
                if mimic.source in q:
                    q[name] = q[mimic.source] * mimic.multiplier + mimic.offset
                    unresolved.remove(name)
                    progress = True
            if not progress:
                raise ValueError(f"unresolved mimic chain: {sorted(unresolved)}")
        return q

    def forward_kinematics(
        self,
        joint_positions: Mapping[str, float] | None = None,
        root_pos: Iterable[float] = (0.0, 0.0, 0.0),
        root_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> dict[str, np.ndarray]:
        q = self.expanded_positions(joint_positions or {})
        root_tf = transform(root_pos, quat_to_matrix_wxyz(root_quat_wxyz))
        link_tf: dict[str, np.ndarray] = {self.root_link: root_tf}
        queue = [self.root_link]
        while queue:
            parent = queue.pop(0)
            for joint in self.children.get(parent, ()):
                motion = np.eye(4, dtype=np.float64)
                value = float(q.get(joint.name, 0.0))
                if joint.kind in {"revolute", "continuous"}:
                    motion[:3, :3] = _axis_angle_matrix(joint.axis, value)
                elif joint.kind == "prismatic":
                    motion[:3, 3] = joint.axis * value
                link_tf[joint.child] = link_tf[parent] @ joint.origin @ motion
                queue.append(joint.child)
        if set(link_tf) != set(self.links):
            missing = set(self.links).difference(link_tf)
            raise ValueError(f"FK did not reach links: {sorted(missing)}")
        return link_tf

    def collision_mesh_local(self, link_name: str) -> trimesh.Trimesh | None:
        if link_name in self._collision_mesh_cache:
            cached = self._collision_mesh_cache[link_name]
            return None if cached is None else cached.copy()

        link = self.links[link_name]
        pieces: list[trimesh.Trimesh] = []
        for collision in link.findall("collision"):
            origin_elem = collision.find("origin")
            xyz = _vec(None if origin_elem is None else origin_elem.get("xyz"))
            rpy = _vec(None if origin_elem is None else origin_elem.get("rpy"))
            local_tf = transform(xyz, _rpy_matrix(rpy))
            geometry = collision.find("geometry")
            mesh_elem = geometry.find("mesh")
            cyl_elem = geometry.find("cylinder")
            box_elem = geometry.find("box")
            sphere_elem = geometry.find("sphere")
            if mesh_elem is not None:
                mesh_path = (self.path.parent / mesh_elem.get("filename")).resolve()
                # ``process=True`` welds exact duplicate STL vertices while
                # preserving every triangle.  Linker collision STLs repeat each
                # shared vertex per face (up to 128k records versus 21k unique
                # vertices), so welding is essential for tractable full-mesh
                # proximity checks and is not mesh simplification.
                loaded = trimesh.load_mesh(mesh_path, process=True)
                if isinstance(loaded, trimesh.Scene):
                    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
                piece = loaded.copy()
                scale = _vec(mesh_elem.get("scale"), (1.0, 1.0, 1.0))
                piece.apply_scale(scale)
            elif cyl_elem is not None:
                piece = trimesh.creation.cylinder(
                    radius=float(cyl_elem.get("radius")),
                    height=float(cyl_elem.get("length")),
                    sections=256,
                )
            elif box_elem is not None:
                piece = trimesh.creation.box(extents=_vec(box_elem.get("size")))
            elif sphere_elem is not None:
                piece = trimesh.creation.icosphere(
                    subdivisions=4, radius=float(sphere_elem.get("radius"))
                )
            else:
                raise ValueError(f"unsupported collision geometry on {link_name}")
            piece.apply_transform(local_tf)
            pieces.append(piece)

        combined = None if not pieces else trimesh.util.concatenate(pieces)
        self._collision_mesh_cache[link_name] = None if combined is None else combined.copy()
        return None if combined is None else combined.copy()

    def collision_meshes_world(
        self,
        joint_positions: Mapping[str, float] | None = None,
        root_pos: Iterable[float] = (0.0, 0.0, 0.0),
        root_quat_wxyz: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
        links: Iterable[str] | None = None,
        convex_hull: bool = False,
    ) -> dict[str, trimesh.Trimesh]:
        fk = self.forward_kinematics(joint_positions, root_pos, root_quat_wxyz)
        names = tuple(self.links) if links is None else tuple(links)
        out: dict[str, trimesh.Trimesh] = {}
        for name in names:
            mesh = self.collision_mesh_local(name)
            if mesh is None:
                continue
            if convex_hull:
                mesh = mesh.convex_hull
            mesh.apply_transform(fk[name])
            out[name] = mesh
        return out

    def adjacent_link_pairs(self) -> frozenset[frozenset[str]]:
        return frozenset(frozenset((joint.parent, joint.child)) for joint in self.joints.values())


def _iter_mesh_point_chunks(
    mesh: trimesh.Trimesh,
    max_surface_points: int = 2500,
    chunk_size: int = 2048,
) -> Iterable[np.ndarray]:
    """Yield every mesh vertex plus deterministic surface samples in chunks.

    The Linker STLs contain up to ~129k vertices.  Chunking preserves the
    full-vertex check while preventing trimesh proximity queries from allocating
    enormous all-pairs intermediates.
    """

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    for start in range(0, len(vertices), chunk_size):
        yield vertices[start : start + chunk_size]
    if len(mesh.faces) and max_surface_points > 0:
        count = min(max_surface_points, max(256, len(mesh.faces)))
        sampled, _ = trimesh.sample.sample_surface(mesh, count, seed=0)
        for start in range(0, len(sampled), chunk_size):
            yield sampled[start : start + chunk_size]


@dataclass(frozen=True)
class MeshPairMetric:
    first: str
    second: str
    surface_distance_m: float
    penetration_m: float
    intersects: bool


def mesh_pair_metric(
    first_name: str,
    first: trimesh.Trimesh,
    second_name: str,
    second: trimesh.Trimesh,
    sample_points: int = 2500,
    contact_tolerance_m: float = 1.0e-5,
) -> MeshPairMetric:
    """Deterministic bidirectional full-mesh proximity/intersection estimate.

    Vertices plus seeded surface samples are queried against each watertight STL.
    This catches both vertex-in-volume and broad face crossings in the repository's
    dense hand meshes.  The independent convex-hull feasibility check below is the
    conservative backstop for face/edge cases with no contained source vertex.
    """

    if len(first.vertices) == 0 or len(second.vertices) == 0:
        return MeshPairMetric(first_name, second_name, math.inf, 0.0, False)
    distance = math.inf
    penetration = 0.0
    for source, target in ((first, second), (second, first)):
        ray_scene = None
        if o3d is not None:
            legacy = o3d.geometry.TriangleMesh()
            legacy.vertices = o3d.utility.Vector3dVector(
                np.asarray(target.vertices, dtype=np.float64)
            )
            legacy.triangles = o3d.utility.Vector3iVector(
                np.asarray(target.faces, dtype=np.int32)
            )
            ray_scene = o3d.t.geometry.RaycastingScene()
            ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
        for points in _iter_mesh_point_chunks(source, sample_points):
            if ray_scene is None:
                _, distances, _ = trimesh.proximity.closest_point(target, points)
                signed = (
                    trimesh.proximity.signed_distance(target, points)
                    if target.is_watertight
                    else np.empty(0, dtype=np.float64)
                )
                inside_depth = float(max(0.0, np.max(signed))) if len(signed) else 0.0
            else:
                query = o3d.core.Tensor(
                    np.ascontiguousarray(points, dtype=np.float32)
                )
                distances = ray_scene.compute_distance(query).numpy()
                if target.is_watertight:
                    signed = ray_scene.compute_signed_distance(
                        query, nsamples=5
                    ).numpy()
                    # Open3D is negative inside; trimesh's fallback is positive.
                    inside_depth = float(max(0.0, -np.min(signed)))
                else:
                    inside_depth = 0.0
            if len(distances):
                distance = min(distance, float(np.min(distances)))
            penetration = max(penetration, inside_depth)
    # Ray-parity can classify points inside a disconnected mesh component even
    # when the two links are provably disjoint.  Non-overlapping convex hulls
    # are an exact separating certificate, so they take precedence over a
    # signed-distance containment result.
    if penetration > contact_tolerance_m and not convex_hulls_intersect(first, second):
        penetration = 0.0
    intersects = penetration > contact_tolerance_m
    return MeshPairMetric(first_name, second_name, distance, penetration, intersects)


def _hull_halfspaces(mesh: trimesh.Trimesh, expansion_m: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    hull = ConvexHull(np.asarray(mesh.vertices, dtype=np.float64))
    # scipy equations are normal*x + offset <= 0 inside.  Qhull returns unit
    # normals, but normalise explicitly before applying a metric expansion.
    normals = hull.equations[:, :3]
    offsets = hull.equations[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    normals = normals / norms[:, None]
    offsets = offsets / norms
    # Expanded hull: normal*x + offset <= expansion.
    return normals, np.full(len(offsets), float(expansion_m)) - offsets


def convex_hulls_intersect(
    first: trimesh.Trimesh,
    second: trimesh.Trimesh,
    expansion_m: float = 0.0,
) -> bool:
    """Exact convex-polytope intersection feasibility via half-space LP."""

    a1, b1 = _hull_halfspaces(first.convex_hull, expansion_m)
    a2, b2 = _hull_halfspaces(second.convex_hull, expansion_m)
    result = linprog(
        c=np.zeros(3, dtype=np.float64),
        A_ub=np.vstack((a1, a2)),
        b_ub=np.concatenate((b1, b2)),
        bounds=((None, None), (None, None), (None, None)),
        method="highs",
    )
    return bool(result.success)


def iter_unfiltered_self_pairs(
    model: UrdfGeometry,
    mesh_names: Iterable[str],
) -> Iterable[tuple[str, str]]:
    names = sorted(mesh_names)
    adjacent = model.adjacent_link_pairs()
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            pair = frozenset((first, second))
            if pair in adjacent or pair in SELF_COLLISION_FILTER_PAIRS:
                continue
            yield first, second


def joint_limit_margins(
    model: UrdfGeometry,
    independent: Mapping[str, float],
) -> dict[str, float]:
    expanded = model.expanded_positions(independent)
    margins: dict[str, float] = {}
    for name, value in expanded.items():
        joint = model.joints[name]
        if joint.lower is None or joint.upper is None:
            continue
        margins[name] = min(value - joint.lower, joint.upper - value)
    return margins


def cylindrical_surface_clearance(
    points_w: np.ndarray,
    axis_xy: Iterable[float],
    radius_m: float,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Signed distance to a capped, world-Z cylinder (negative inside)."""

    points = np.asarray(points_w, dtype=np.float64)
    center_xy = np.asarray(tuple(axis_xy), dtype=np.float64)
    radial = np.linalg.norm(points[:, :2] - center_xy[None, :], axis=1) - radius_m
    dz = np.maximum(z_min - points[:, 2], points[:, 2] - z_max)
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(dz, 0.0))
    inside = np.minimum(np.maximum(radial, dz), 0.0)
    return outside + inside


def posture_frame(
    model: UrdfGeometry,
    root_pos: Iterable[float],
    root_quat_wxyz: Iterable[float],
    joint_positions: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Compact FK output useful to fitting objectives and JSON summaries."""

    fk = model.forward_kinematics(joint_positions, root_pos, root_quat_wxyz)
    return {
        "palm_normal_w": rotate_vector_wxyz(root_quat_wxyz, PALM_NORMAL_LOCAL),
        "long_axis_w": rotate_vector_wxyz(root_quat_wxyz, HAND_LONG_AXIS_LOCAL),
        "root_pos_w": np.asarray(tuple(root_pos), dtype=np.float64),
        "tip_points_w": np.vstack([fk[name][:3, 3] for name in TIP_MARKER_LINKS]),
    }

