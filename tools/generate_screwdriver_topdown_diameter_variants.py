#!/usr/bin/env python3
"""Generate the fixed-length diameter bank for the Linker L20 top-down task.

The nominal 64 mm asset remains the source of truth.  Variants change only the
visual/collision cylinder radii of ``screwdriver_body`` and ``screwdriver_cap``;
all lengths, joints, masses and inertia tensors remain byte-for-byte equivalent
at the parsed XML level.  Runtime assignment is cyclic/stratified so variant
identity never depends on an inertial signature.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "assets" / "screwdriver"
SOURCE_URDF = ASSET_ROOT / "screwdriver_64mm_handle.urdf"
OUT_DIR = ASSET_ROOT / "topdown_variants"
MANIFEST = OUT_DIR / "manifest.json"

DIAMETERS_M = (0.060, 0.064, 0.068)
BASE_RADIUS_M = 0.032
HANDLE_LENGTH_M = 0.100
BODY_AND_CAP = ("screwdriver_body", "screwdriver_cap")


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


def _cylinder(link: ET.Element, role: str) -> ET.Element:
    cylinder = link.find(f"{role}/geometry/cylinder")
    if cylinder is None:
        raise ValueError(f"{link.get('name')} has no {role} cylinder")
    return cylinder


def _canonical_signature(root: ET.Element) -> tuple:
    clone = copy.deepcopy(root)
    clone.set("name", "canonical")
    for link_name in BODY_AND_CAP:
        link = _find_link(clone, link_name)
        for role in ("visual", "collision"):
            _cylinder(link, role).set("radius", "canonical")

    def visit(element: ET.Element) -> tuple:
        return (
            element.tag,
            tuple(sorted(element.attrib.items())),
            tuple(visit(child) for child in element),
        )

    return visit(clone)


def _inertial_signature(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            result[link.get("name")] = ET.tostring(inertial, encoding="unicode")
    return result


def _source_properties(root: ET.Element) -> tuple[float, float]:
    body = _find_link(root, "screwdriver_body")
    radius = float(_cylinder(body, "collision").get("radius"))
    length = float(_cylinder(body, "collision").get("length"))
    if not math.isclose(radius, BASE_RADIUS_M, abs_tol=1.0e-12):
        raise ValueError(f"nominal asset radius is {radius}, expected {BASE_RADIUS_M}")
    if not math.isclose(length, HANDLE_LENGTH_M, abs_tol=1.0e-12):
        raise ValueError(f"nominal asset length is {length}, expected {HANDLE_LENGTH_M}")
    mass = float(body.find("inertial/mass").get("value"))
    izz = float(body.find("inertial/inertia").get("izz"))
    return mass, izz


def _build_variant(source: ET.Element, radius: float) -> ET.Element:
    root = copy.deepcopy(source)
    for link_name in BODY_AND_CAP:
        link = _find_link(root, link_name)
        for role in ("visual", "collision"):
            _cylinder(link, role).set("radius", f"{radius:.8g}")
    return root


def _validate_variant(
    source: ET.Element,
    variant: ET.Element,
    radius: float,
) -> None:
    if _canonical_signature(variant) != _canonical_signature(source):
        raise AssertionError("variant differs from nominal asset outside body/cap radii")
    if _inertial_signature(variant) != _inertial_signature(source):
        raise AssertionError("variant changed a mass or inertia value")
    for link_name in BODY_AND_CAP:
        link = _find_link(variant, link_name)
        for role in ("visual", "collision"):
            actual = float(_cylinder(link, role).get("radius"))
            if not math.isclose(actual, radius, abs_tol=1.0e-12):
                raise AssertionError(
                    f"{link_name} {role} radius {actual} != requested {radius}"
                )


def main() -> None:
    source_tree = ET.parse(SOURCE_URDF)
    source = source_tree.getroot()
    body_mass, body_izz = _source_properties(source)
    source_hash = _sha256(SOURCE_URDF)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants: list[dict] = []
    for index, diameter in enumerate(DIAMETERS_M):
        radius = 0.5 * diameter
        root = _build_variant(source, radius)
        _validate_variant(source, root, radius)
        filename = f"screwdriver_topdown_d{round(diameter * 1000):03d}.urdf"
        path = OUT_DIR / filename
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        variants.append(
            {
                "file": f"topdown_variants/{filename}",
                "name": filename,
                "index": index,
                "diameter": diameter,
                "radius": radius,
                "length": HANDLE_LENGTH_M,
                "mass": body_mass,
                "izz": body_izz,
                "diameter_scale": radius / BASE_RADIUS_M,
                "length_scale": 1.0,
                "diameter_bucket": index,
                "length_bucket": 0,
                "bucket": index,
            }
        )

    manifest = {
        "schema_version": 1,
        "source_asset": "screwdriver_64mm_handle.urdf",
        "source_sha256": source_hash,
        "assignment_mode": "cyclic",
        "preserve_source_mass_inertia": True,
        "base": {
            "diameter": 2.0 * BASE_RADIUS_M,
            "radius": BASE_RADIUS_M,
            "length": HANDLE_LENGTH_M,
            "mass": body_mass,
            "izz": body_izz,
        },
        "num_diameter_buckets": len(DIAMETERS_M),
        "num_length_buckets": 1,
        "num_buckets": len(DIAMETERS_M),
        "variants": variants,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(variants)} fixed-inertia diameter variants to {OUT_DIR}")
    for row in variants:
        print(
            f"  {row['name']}: diameter={row['diameter'] * 1000:.0f} mm, "
            f"length={row['length'] * 1000:.0f} mm, mass={row['mass']:.6g}, "
            f"izz={row['izz']:.6g}"
        )


if __name__ == "__main__":
    main()
