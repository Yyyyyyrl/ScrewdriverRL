"""Numeric validation of LinkerL20 in-hand grasp caches (no simulator needed).

Checks per cache file: shape (N, 23), finite values, unit object quaternions,
all 16 joint values within URDF limits, object height above the fall/reset
threshold, and prints distribution stats for eyeballing.

Usage:
    python tools/validate_inhand_grasp_cache.py --all
    python tools/validate_inhand_grasp_cache.py --scale 0.8
"""

from __future__ import annotations

import argparse
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "inhand_rotation_env_cfg.py"
_URDF = _ROOT / "assets" / "linker_hand_l20" / "linkerhand_l20_left.urdf"


def _load_cfg_constants() -> dict:
    """Extract the needed cfg constants via AST (importing the cfg module would
    pull in isaaclab, which needs the Isaac Sim runtime)."""
    tree = ast.parse(_CFG.read_text())

    def literal(name: str, body=tree.body):
        for node in body:
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                target, value = node.targets[0].id, node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target, value = node.target.id, node.value
            if target == name:
                return ast.literal_eval(value)
        raise KeyError(name)

    def fn(name: str):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        mod = ast.Module(body=[node], type_ignores=[])
        ast.fix_missing_locations(mod)
        ns: dict = {}
        exec(compile(mod, filename=f"<{name}>", mode="exec"), ns)
        return ns[name]

    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LinkerL20InhandRotationEnvCfg"
    )
    tag = fn("_scale_cache_tag")
    filename = fn("grasp_cache_filename")
    filename.__globals__["_scale_cache_tag"] = tag
    filename.__globals__["_SHAPE_CACHE_TAGS"] = literal("_SHAPE_CACHE_TAGS")
    n_protos = {
        "cylinder": len(literal("MIX_CYLINDER_LENGTHS")),
        "cuboid": len(literal("MIX_CUBOID_SIZES")),
        "sphere": len(literal("MIX_SPHERE_RADII")),
    }
    return {
        "scales": literal("HORA_CYLINDER_SCALES"),
        "shapes": literal("INHAND_CACHE_SHAPES"),
        "n_protos": n_protos,
        "reset_height_threshold": literal("reset_height_threshold", cls.body),
        "grasp_cache_filename": filename,
    }

FINGER_JOINT_ORDER = (
    "index_mcp_roll", "index_mcp_pitch", "index_pip",
    "middle_mcp_roll", "middle_mcp_pitch", "middle_pip",
    "ring_mcp_roll", "ring_mcp_pitch", "ring_pip",
    "pinky_mcp_roll", "pinky_mcp_pitch", "pinky_pip",
    "thumb_cmc_yaw", "thumb_cmc_roll", "thumb_cmc_pitch", "thumb_mcp",
)


def _urdf_limits() -> tuple[np.ndarray, np.ndarray]:
    joints = {j.get("name"): j for j in ET.parse(_URDF).getroot().findall("joint")}
    lo, hi = [], []
    for name in FINGER_JOINT_ORDER:
        limit = joints[name].find("limit")
        lo.append(float(limit.get("lower")))
        hi.append(float(limit.get("upper")))
    return np.asarray(lo), np.asarray(hi)


def validate(path: Path, lo: np.ndarray, hi: np.ndarray, reset_z: float) -> list[str]:
    errors: list[str] = []
    arr = np.load(path)
    print(f"\n=== {path.name} ===")
    if arr.ndim != 2 or arr.shape[1] != 39:
        return [f"bad shape {arr.shape}, expected (N, 39) = [q(16), targets(16), obj pose(7)]"]
    n = arr.shape[0]
    print(f"  rows: {n}")
    if n < 10000:
        errors.append(f"only {n} rows (< 10000)")
    if not np.isfinite(arr).all():
        errors.append("non-finite values present")

    for label, block in (("q", arr[:, :16]), ("target", arr[:, 16:32])):
        below = (block < lo - 1e-4).any(axis=0)
        above = (block > hi + 1e-4).any(axis=0)
        for j, name in enumerate(FINGER_JOINT_ORDER):
            if below[j] or above[j]:
                errors.append(
                    f"{label}:{name} outside URDF limits [{lo[j]:.3f}, {hi[j]:.3f}]: "
                    f"cache range [{block[:, j].min():.3f}, {block[:, j].max():.3f}]"
                )
    q = arr[:, :16]
    squeeze = np.abs(arr[:, 16:32] - q).mean()
    print(f"  mean |target - q| (grip squeeze): {squeeze:.4f} rad")

    pos = arr[:, 32:35]
    quat = arr[:, 35:39]
    quat_norm = np.linalg.norm(quat, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1e-3):
        errors.append(
            f"object quats not unit: |q| in [{quat_norm.min():.4f}, {quat_norm.max():.4f}]"
        )

    z = pos[:, 2]
    pcts = np.percentile(z, [0, 5, 50, 95, 100])
    print(
        "  obj z   min/p5/med/p95/max: "
        + " / ".join(f"{v:.4f}" for v in pcts)
        + f"   (reset threshold {reset_z:.3f})"
    )
    if z.min() < reset_z:
        errors.append(f"{int((z < reset_z).sum())} rows below reset threshold {reset_z}")
    print(
        f"  obj xy  x in [{pos[:, 0].min():+.4f}, {pos[:, 0].max():+.4f}], "
        f"y in [{pos[:, 1].min():+.4f}, {pos[:, 1].max():+.4f}]"
    )
    print(
        "  joint mean/std (16): "
        + " ".join(f"{m:+.2f}" for m in q.mean(axis=0))
        + "  /  "
        + " ".join(f"{s:.2f}" for s in q.std(axis=0))
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Validate all scale x shape caches."
    )
    parser.add_argument("--scale", type=float, action="append", default=None)
    parser.add_argument(
        "--shape",
        type=str,
        action="append",
        default=None,
        choices=("cylinder", "cuboid", "sphere"),
    )
    parser.add_argument("--dir", type=str, default=None, help="Cache directory override.")
    args = parser.parse_args()

    constants = _load_cfg_constants()
    grasp_cache_filename = constants["grasp_cache_filename"]
    cache_dir = Path(args.dir) if args.dir else _ROOT / "assets" / "grasp_cache"
    reset_z = float(constants["reset_height_threshold"])
    scales = list(constants["scales"]) if (args.all or not args.scale) else args.scale
    shapes = list(constants["shapes"]) if (args.all or not args.shape) else args.shape

    lo, hi = _urdf_limits()
    failures: dict[str, list[str]] = {}
    n_files = 0
    for shape in shapes:
        for scale in scales:
            for proto in range(constants["n_protos"][shape]):
                n_files += 1
                path = cache_dir / grasp_cache_filename(scale, shape=shape, proto=proto)
                if not path.exists():
                    failures[path.name] = ["file missing"]
                    print(f"\n=== {path.name} ===\n  MISSING")
                    continue
                errors = validate(path, lo, hi, reset_z)
                if errors:
                    failures[path.name] = errors

    print()
    if failures:
        for name, errors in failures.items():
            for err in errors:
                print(f"[FAIL] {name}: {err}")
        return 1
    print(f"[OK] {n_files} cache file(s) passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
