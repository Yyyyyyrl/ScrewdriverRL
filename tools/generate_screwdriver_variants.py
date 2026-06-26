"""Generate screwdriver geometry variants for domain randomization.

Emits a small grid of ``(diameter x length)`` URDF variants of the base
screwdriver into ``assets/screwdriver/variants/`` plus a ``manifest.json`` that
the env uses to map each spawned handle (identified by its measured
``(mass, izz)`` signature) back to its geometry buckets and scales.

Every variant is **topologically identical** to the base — same link/joint
names and parent/child structure — because the env resolves bodies/joints by
name and the actuator ``joint_names_expr`` regexes depend on those names.  Only
the handle (``screwdriver_body``) and its cap (``screwdriver_cap``) change
size; ``screwdriver_stick`` is kept fixed so the handle's *base* height (and
therefore the mount/pregrasp geometry) is invariant — only the cap end moves
with length.

Run (no Isaac Lab needed)::

    python tools/generate_screwdriver_variants.py

The script self-checks each variant and fails loudly if any invariant is
violated (see ``_validate``).
"""

from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

# --- Paths -----------------------------------------------------------------
_ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "screwdriver"
BASE_URDF = _ASSET_ROOT / "screwdriver_isaaclab.urdf"
OUT_DIR = _ASSET_ROOT / "variants"
MANIFEST = OUT_DIR / "manifest.json"

# --- Base reference geometry (must match screwdriver_isaaclab.urdf) ---------
R0 = 0.02       # base handle/cap radius [m]
L0 = 0.10       # base handle length [m]
BODY_MASS0 = 0.3    # base handle mass [kg]
CAP_MASS0 = 0.005   # base cap mass [kg]
CAP_LEN = 0.001     # cap thickness [m] (kept fixed)

# --- Variation grid (modest; see plan section 3a + Risks) -------------------
# Diameter is +/-15% of base; length is +/-10% of base.  Each cell is one
# variant AND one (diameter_bucket, length_bucket) pregrasp bucket.
DIAMETERS = [0.017, 0.020, 0.023]   # radius [m]; bucket index = position
LENGTHS = [0.090, 0.110]            # handle length [m]; bucket index = position

# Minimum separation between any two variants in the (mass, izz) signature so
# the env can identify geometry unambiguously (relative, see _validate).
SIGNATURE_REL_TOL = 0.02


def _cyl_inertia(mass: float, radius: float, length: float) -> tuple[float, float]:
    """Solid-cylinder (ixx==iyy, izz) about its centre, axis = z."""
    izz = 0.5 * mass * radius**2
    ixx = (1.0 / 12.0) * mass * (3.0 * radius**2 + length**2)
    return ixx, izz


def _body_props(radius: float, length: float) -> dict:
    """Physically-consistent handle mass/inertia at constant density."""
    mass = BODY_MASS0 * (radius / R0) ** 2 * (length / L0)
    ixx, izz = _cyl_inertia(mass, radius, length)
    return {"mass": mass, "ixx": ixx, "iyy": ixx, "izz": izz}


def _cap_props(radius: float) -> dict:
    """Cap mass/inertia (length fixed, only radius tracks the handle)."""
    mass = CAP_MASS0 * (radius / R0) ** 2
    ixx, izz = _cyl_inertia(mass, radius, CAP_LEN)
    return {"mass": mass, "ixx": ixx, "iyy": ixx, "izz": izz}


def _set_origin_z(elem: ET.Element, z: float) -> None:
    origin = elem.find("origin")
    xyz = origin.get("xyz").split()
    origin.set("xyz", f"{xyz[0]} {xyz[1]} {z:.8g}")


def _set_cylinder(elem: ET.Element, radius: float, length: float) -> None:
    cyl = elem.find("geometry/cylinder")
    cyl.set("radius", f"{radius:.8g}")
    cyl.set("length", f"{length:.8g}")


def _set_inertial(link: ET.Element, props: dict, origin_z: float) -> None:
    inertial = link.find("inertial")
    _set_origin_z(inertial, origin_z)
    inertial.find("mass").set("value", f"{props['mass']:.8g}")
    inertia = inertial.find("inertia")
    inertia.set("ixx", f"{props['ixx']:.8g}")
    inertia.set("iyy", f"{props['iyy']:.8g}")
    inertia.set("izz", f"{props['izz']:.8g}")


def _find_link(root: ET.Element, name: str) -> ET.Element:
    for link in root.findall("link"):
        if link.get("name") == name:
            return link
    raise KeyError(f"link {name!r} not found")


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    for joint in root.findall("joint"):
        if joint.get("name") == name:
            return joint
    raise KeyError(f"joint {name!r} not found")


def _topology(root: ET.Element) -> tuple:
    """A hashable signature of the URDF's link/joint structure (names only)."""
    links = tuple(sorted(l.get("name") for l in root.findall("link")))
    joints = tuple(
        sorted(
            (
                j.get("name"),
                j.get("type"),
                j.find("parent").get("link"),
                j.find("child").get("link"),
            )
            for j in root.findall("joint")
        )
    )
    return links, joints


def _build_variant(base_root: ET.Element, radius: float, length: float) -> ET.Element:
    root = copy.deepcopy(base_root)

    body = _find_link(root, "screwdriver_body")
    body_props = _body_props(radius, length)
    _set_inertial(body, body_props, origin_z=length / 2.0)
    _set_cylinder(body.find("visual"), radius, length)
    _set_origin_z(body.find("visual"), length / 2.0)
    _set_cylinder(body.find("collision"), radius, length)
    _set_origin_z(body.find("collision"), length / 2.0)

    # The cap joint sits at the top of the handle body.
    _set_origin_z(_find_joint(root, "screwdriver_body_cap_joint"), length)

    cap = _find_link(root, "screwdriver_cap")
    _set_inertial(cap, _cap_props(radius), origin_z=CAP_LEN / 2.0)
    _set_cylinder(cap.find("visual"), radius, CAP_LEN)
    _set_cylinder(cap.find("collision"), radius, CAP_LEN)

    return root


def _validate(base_root: ET.Element, variants: list[dict]) -> None:
    base_topo = _topology(base_root)
    for v in variants:
        root = ET.parse(v["file"]).getroot()

        # (a) topology / names identical to base
        if _topology(root) != base_topo:
            raise AssertionError(f"{v['file']}: topology/name drift vs base")

        # (b) dimensions match the requested grid cell
        body = _find_link(root, "screwdriver_body")
        cyl = body.find("collision/geometry/cylinder")
        assert math.isclose(float(cyl.get("radius")), v["radius"], rel_tol=1e-6)
        assert math.isclose(float(cyl.get("length")), v["length"], rel_tol=1e-6)
        cap_joint = _find_joint(root, "screwdriver_body_cap_joint")
        cap_z = float(cap_joint.find("origin").get("xyz").split()[2])
        assert math.isclose(cap_z, v["length"], rel_tol=1e-6)

        # (c) inertials satisfy the constant-density formulas
        ref = _body_props(v["radius"], v["length"])
        assert math.isclose(v["mass"], ref["mass"], rel_tol=1e-6)
        assert math.isclose(v["izz"], ref["izz"], rel_tol=1e-6)
        inertia = body.find("inertial/inertia")
        assert math.isclose(float(inertia.get("izz")), ref["izz"], rel_tol=1e-6)

    # (d) all (mass, izz) signatures mutually separated for unambiguous ID
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            dm = abs(a["mass"] - b["mass"]) / max(a["mass"], b["mass"])
            dz = abs(a["izz"] - b["izz"]) / max(a["izz"], b["izz"])
            if dm < SIGNATURE_REL_TOL and dz < SIGNATURE_REL_TOL:
                raise AssertionError(
                    f"variants {a['file'].name} and {b['file'].name} have "
                    f"near-identical (mass, izz) signature "
                    f"(dm={dm:.4f}, dz={dz:.4f} < {SIGNATURE_REL_TOL}); "
                    "identification would be ambiguous"
                )


def main() -> None:
    base_tree = ET.parse(BASE_URDF)
    base_root = base_tree.getroot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants: list[dict] = []
    idx = 0
    for di, radius in enumerate(DIAMETERS):
        for li, length in enumerate(LENGTHS):
            root = _build_variant(base_root, radius, length)
            out = OUT_DIR / f"screwdriver_v{idx:02d}.urdf"
            ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)

            props = _body_props(radius, length)
            variants.append(
                {
                    "file": out,
                    "name": out.name,
                    "index": idx,
                    "radius": radius,
                    "length": length,
                    "mass": props["mass"],
                    "izz": props["izz"],
                    "diameter_scale": radius / R0,
                    "length_scale": length / L0,
                    "diameter_bucket": di,
                    "length_bucket": li,
                    "bucket": di * len(LENGTHS) + li,
                }
            )
            idx += 1

    _validate(base_root, variants)

    manifest = {
        "base": {"radius": R0, "length": L0, "mass": BODY_MASS0},
        "num_diameter_buckets": len(DIAMETERS),
        "num_length_buckets": len(LENGTHS),
        "num_buckets": len(DIAMETERS) * len(LENGTHS),
        "signature_rel_tol": SIGNATURE_REL_TOL,
        "variants": [
            {k: (str(v.relative_to(_ASSET_ROOT)) if k == "file" else v) for k, v in e.items()}
            for e in variants
        ],
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(variants)} variants + manifest to {OUT_DIR}")
    for e in variants:
        print(
            f"  {e['name']}: r={e['radius']:.3f} L={e['length']:.3f} "
            f"m={e['mass']:.5f} izz={e['izz']:.3e} "
            f"bucket=({e['diameter_bucket']},{e['length_bucket']})"
        )


if __name__ == "__main__":
    main()
