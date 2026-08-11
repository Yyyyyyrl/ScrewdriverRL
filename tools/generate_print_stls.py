#!/usr/bin/env python3
"""Generate STL files for 3D-printing the training objects at sim-accurate sizes.

Dimensions are copied verbatim from:
  - screwdriver_rl/tasks/linker_l20/inhand_rotation_env_cfg.py
      HORA_CYLINDER_RADIUS, MIX_CYLINDER_LENGTHS, MIX_CUBOID_SIZES, MIX_SPHERE_RADII
  - assets/screwdriver/variants/manifest.json (handle radius/length per variant)
  - assets/screwdriver/variants/screwdriver_v*.urdf (stick r=0.005 L=0.1, cap thickness 0.001)

The env cfg module imports Isaac Lab at module level, so the constants are
re-declared here instead of imported. STL units are millimeters; every mesh
sits with its print face on z=0, oriented as it should be printed.

Usage: python tools/generate_print_stls.py [outdir]   (default: print/stl)
"""

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent

# --- inhand_rotation_env_cfg.py constants (meters) ---
HORA_CYLINDER_RADIUS = 0.04
MIX_CYLINDER_LENGTHS = (0.064, 0.080, 0.096)
MIX_CUBOID_SIZES = ((0.064, 0.064, 0.080), (0.072, 0.072, 0.064))
MIX_SPHERE_RADII = (0.034, 0.038, 0.042, 0.046)
# Printed subset of HORA_CYLINDER_SCALES covering the DR range 0.70-0.86
PRINT_SCALES = (0.70, 0.78, 0.86)

# --- screwdriver URDF constants (meters) ---
STICK_RADIUS = 0.005
STICK_LENGTH = 0.100
CAP_THICKNESS = 0.001  # cap radius equals handle radius in every variant

# hollow handle: sim density (~2.25 g/cm3) exceeds PLA, so the handle gets a
# coaxial blind pocket (opening at the cap face, printed facing the bed) to be
# filled with steel shot / sand up to the sim mass, then closed with the plug
WALL = 0.003            # handle wall & roof thickness [m]
PLUG_CLEARANCE = 0.0003  # radial clearance of the plug disc [m]

M2MM = 1000.0


def on_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
    return mesh


def make_cylinder(radius_m: float, height_m: float) -> trimesh.Trimesh:
    return on_bed(trimesh.creation.cylinder(
        radius=radius_m * M2MM, height=height_m * M2MM, sections=256))


def make_cuboid(size_m) -> trimesh.Trimesh:
    return on_bed(trimesh.creation.box(extents=[s * M2MM for s in size_m]))


def make_sphere(radius_m: float) -> trimesh.Trimesh:
    return on_bed(trimesh.creation.icosphere(subdivisions=5, radius=radius_m * M2MM))


def make_screwdriver(handle_radius_m: float, handle_length_m: float) -> trimesh.Trimesh:
    """One-piece hollow screwdriver, printed cap-face-down: cap+handle cylinder
    with the stick on top and a coaxial blind ballast pocket opening at the cap
    face. The pocket roof is a 45-degree cone so it prints without supports.
    Built as a surface of revolution so the mesh is watertight."""
    rh = handle_radius_m * M2MM
    lh = (handle_length_m + CAP_THICKNESS) * M2MM  # cap merged into handle
    rs = STICK_RADIUS * M2MM
    ls = STICK_LENGTH * M2MM
    rp = rh - WALL * M2MM                  # pocket radius
    z_cyl = lh - WALL * M2MM - rp          # cylindrical pocket ends here
    z_apex = z_cyl + rp                    # 45-degree cone apex (= lh - wall)
    profile = np.array([
        [0.0, z_apex],
        [rp, z_cyl],
        [rp, 0.0],
        [rh, 0.0],
        [rh, lh],
        [rs, lh],
        [rs, lh + ls],
        [0.0, lh + ls],
    ])
    return on_bed(trimesh.creation.revolve(profile, sections=256))


def make_plug(handle_radius_m: float) -> trimesh.Trimesh:
    """Flush disc glued into the pocket mouth after adding ballast."""
    rp = (handle_radius_m - WALL - PLUG_CLEARANCE) * M2MM
    return on_bed(trimesh.creation.cylinder(
        radius=rp, height=WALL * M2MM, sections=256))


def pocket_volume_cm3(handle_radius_m: float, handle_length_m: float) -> float:
    rp = (handle_radius_m - WALL) * 100  # cm
    lh = (handle_length_m + CAP_THICKNESS) * 100
    z_cyl = lh - WALL * 100 - rp
    return float(np.pi * rp * rp * z_cyl + np.pi * rp ** 3 / 3)


def main() -> None:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "print" / "stl"
    outdir.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, trimesh.Trimesh] = {}

    for scale in PRINT_SCALES:
        tag = f"s{int(round(scale * 100)):03d}"
        for i, length in enumerate(MIX_CYLINDER_LENGTHS):
            meshes[f"cyl{i}_{tag}"] = make_cylinder(
                HORA_CYLINDER_RADIUS * scale, length * scale)
        for i, size in enumerate(MIX_CUBOID_SIZES):
            meshes[f"cub{i}_{tag}"] = make_cuboid([s * scale for s in size])
        for i, radius in enumerate(MIX_SPHERE_RADII):
            meshes[f"sph{i}_{tag}"] = make_sphere(radius * scale)

    manifest = json.loads((REPO / "assets/screwdriver/variants/manifest.json").read_text())
    base = manifest["base"]  # nominal geometry used when randomize_geometry=False
    screwdrivers = [("base", base["radius"], base["length"], base["mass"])]
    screwdrivers += [(f"v{v['index']:02d}", v["radius"], v["length"], v["mass"])
                     for v in manifest["variants"]]
    for tag, radius, length, mass in screwdrivers:
        name = f"screwdriver_{tag}_d{round(radius * 2 * M2MM)}_l{round(length * M2MM)}"
        meshes[name] = make_screwdriver(radius, length)
        meshes[f"plug_{tag}_d{round((radius - WALL) * 2 * M2MM)}"] = make_plug(radius)
        vol = pocket_volume_cm3(radius, length)
        print(f"{name}: sim mass {mass * 1000:.0f} g, pocket {vol:.0f} cm3 "
              f"(~{vol * 4.8:.0f} g steel shot / ~{vol * 1.6:.0f} g sand)")

    for name, mesh in meshes.items():
        assert mesh.is_watertight, f"{name} is not watertight"
        mesh.export(outdir / f"{name}.stl")
        b = mesh.bounds
        size = b[1] - b[0]
        print(f"{name:32s} {size[0]:7.2f} x {size[1]:7.2f} x {size[2]:7.2f} mm")
    print(f"\n{len(meshes)} STLs written to {outdir}")


if __name__ == "__main__":
    main()
