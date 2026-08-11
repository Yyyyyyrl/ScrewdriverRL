"""Versioned provenance contract for free-object grasp caches.

The numeric ``.npy`` rows are not self-describing.  In particular, an old cache
can remain finite and inside the current joint limits while having been produced
for another hand mesh, object size, posture namespace, or semantic mapping.
This module binds every cache bank to a versioned orientation/posture ID plus
the model/object identities and verifies file digests before Isaac replays a
row.  Any root pose or canonical-seed change must bump that ID/cache namespace.

It intentionally imports neither Isaac Lab nor torch so cache manifests can be
checked in unit tests and on deployment/documentation machines.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


CACHE_MANIFEST_SCHEMA = "dex-forge-inhand-grasp-cache"
CACHE_MANIFEST_VERSION = 2
CACHE_ROW_WIDTH = 39


class GraspCacheManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(cache_dir: Path, cache_name: str) -> Path:
    return cache_dir / f"{cache_name}_manifest.json"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def model_identity(repo_root: Path) -> dict[str, str]:
    model_path = repo_root / "assets/linker_hand_l20/model_manifest_v1.json"
    urdf_path = repo_root / "assets/linker_hand_l20/linkerhand_l20_left.urdf"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    actual_urdf_sha256 = sha256_file(urdf_path)
    recorded_urdf_sha256 = str(model["urdf"]["sha256"])
    if actual_urdf_sha256 != recorded_urdf_sha256:
        raise GraspCacheManifestError(
            "hand model manifest does not match the current URDF: "
            f"recorded={recorded_urdf_sha256}, actual={actual_urdf_sha256}"
        )
    return {
        "model_id": str(model["model_id"]),
        "model_digest": str(model["model_digest"]),
        "semantic_schema_id": str(model["semantic_schema_id"]),
        "semantic_schema_digest": str(model["semantic_schema_digest"]),
        "urdf_sha256": actual_urdf_sha256,
    }


def new_manifest(
    *,
    cache_name: str,
    orientation: str,
    object_scales: Iterable[float],
    cube_edge_m: float,
    repo_root: Path,
    certification: Mapping | None = None,
) -> dict:
    manifest = {
        "schema": CACHE_MANIFEST_SCHEMA,
        "schema_version": CACHE_MANIFEST_VERSION,
        "cache_name": str(cache_name),
        "orientation": str(orientation),
        "row_layout": "q16,target16,object_position3,object_quaternion_wxyz4",
        "row_width": CACHE_ROW_WIDTH,
        "object": {
            "kind": "cuboid",
            "nominal_size_m": [float(cube_edge_m)] * 3,
            "scales": [float(value) for value in object_scales],
            "prototype_count": 1,
        },
        "hand": model_identity(repo_root),
        "entries": {},
    }
    if certification is not None:
        manifest["certification"] = dict(certification)
    return manifest


def write_manifest(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical_json(dict(value)) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_manifest_for_update(path: Path, expected_base: Mapping) -> dict:
    """Load an accumulating manifest without allowing its identity to drift."""
    if not path.exists():
        return dict(expected_base)
    value = json.loads(path.read_text(encoding="utf-8"))
    for key, expected_value in expected_base.items():
        if key == "entries":
            continue
        if value.get(key) != expected_value:
            raise GraspCacheManifestError(
                f"cannot append to incompatible grasp-cache manifest {path}: "
                f"{key}={value.get(key)!r}, expected {expected_value!r}. "
                "Use a new cache namespace for a changed hand, object, or posture."
            )
    entries = value.get("entries")
    if not isinstance(entries, dict):
        raise GraspCacheManifestError(
            f"cannot append to {path}: entries must be a JSON object"
        )
    return value


def record_cache_entry(
    manifest: dict,
    *,
    cache_path: Path,
    scale: float,
    shape: str,
    prototype: int,
    rows: int,
) -> None:
    entries = manifest.setdefault("entries", {})
    entries[cache_path.name] = {
        "sha256": sha256_file(cache_path),
        "rows": int(rows),
        "scale": float(scale),
        "shape": str(shape),
        "prototype": int(prototype),
    }


def load_and_validate_manifest(
    *,
    path: Path,
    expected: Mapping,
    required_files: Iterable[Path],
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required grasp-cache manifest is missing: {path}. Regenerate the "
            "cache with tools/gen_inhand_grasp_cache.py."
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "schema",
        "schema_version",
        "cache_name",
        "orientation",
        "row_width",
        "object",
        "hand",
        "entries",
    ):
        if key not in value:
            raise GraspCacheManifestError(f"grasp-cache manifest missing {key!r}")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise GraspCacheManifestError(
                f"grasp-cache manifest {key}={value.get(key)!r}, expected "
                f"{expected_value!r}"
            )
    entries = value["entries"]
    if not isinstance(entries, Mapping):
        raise GraspCacheManifestError("grasp-cache manifest entries must be an object")
    for cache_path in required_files:
        entry = entries.get(cache_path.name)
        if not isinstance(entry, Mapping):
            raise GraspCacheManifestError(
                f"grasp-cache manifest has no entry for {cache_path.name}"
            )
        if not cache_path.is_file():
            raise FileNotFoundError(f"Required grasp cache is missing: {cache_path}")
        actual = sha256_file(cache_path)
        if entry.get("sha256") != actual:
            raise GraspCacheManifestError(
                f"grasp-cache digest mismatch for {cache_path.name}: "
                f"manifest={entry.get('sha256')!r}, actual={actual}"
            )
        rows = entry.get("rows")
        if not isinstance(rows, int) or rows < 1:
            raise GraspCacheManifestError(
                f"grasp-cache manifest has invalid row count for {cache_path.name}: "
                f"{rows!r}"
            )
    return value
