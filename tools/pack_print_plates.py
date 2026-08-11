#!/usr/bin/env python3
"""Pack the print STLs onto X2D plates (256x256) and emit one positioned,
named 3MF per plate plus a labeled top-view layout map PNG.

Plate grouping: a "starter" plate (nominal screwdriver + one representative
per in-hand object family, for quick deployment checks) and the rest split
by object type so each print job stays short. The starter screwdriver gets
per-object solid-infill overrides injected as Bambu model_settings.config;
the rest of its plate slices at the light in-hand profile.

Cell size per object = XY footprint + reserved margin:
  - spheres get +0.39*D+1 mm (tree-support ground spread measured from trial
    slices: +18.2 mm on a 47.6 sphere, +30.3 mm on a 79.1 sphere = 0.383*D)
  - everything else +0
A 5 mm gap is added between cells and a 4 mm margin at the plate border.
"""

import json
from pathlib import Path

import trimesh
from PIL import Image, ImageDraw

STL = Path("/home/user/ScrewdriverRL/print/stl")
OUT = Path("/home/user/ScrewdriverRL/print")
PLATE = 256.0
MARGIN = 4.0
GAP = 3.0
SPHERE_PAD_RATIO = 0.39

SCALES = ("s070", "s078", "s086")

# starter plate: nominal screwdriver (+ its ballast plug) + mid-scale
# representative per family
STARTER = ("screwdriver_base_d40_l100", "plug_base_d34",
           "cyl1_s078", "cub0_s078", "sph2_s078")

# per-object overrides injected for the starter screwdriver; the handle is a
# hollow ballast shell now, so no infill override — just walls and a brim for
# the 201 mm column
SCREWDRIVER_OVERRIDES = {
    "wall_loops": "3",
    "brim_type": "outer_only",
    "brim_width": "8",
}


def footprint(name: str, mesh: trimesh.Trimesh) -> float:
    b = mesh.bounds
    fp = max(b[1][0] - b[0][0], b[1][1] - b[0][1])
    if name.startswith("sph"):
        fp += SPHERE_PAD_RATIO * fp + 1.0
    if name.startswith("screwdriver"):
        fp += 16.0  # 8 mm brim on each side
    if name.startswith("plug"):
        fp += 2.0
    return fp


def _pack_one(items: list[tuple[str, float]], bin_w: float):
    """Try to pack all items into one bin_w x bin_w bin. Returns placements
    [(name, x_center, y_center)] or None."""
    import numpy as np
    from trimesh.path import packing

    # inflate each square by GAP and pack with spacing=0: neighbors end up
    # >= GAP apart (trimesh's own `spacing` pads both sides, wasting 2x)
    extents = np.array([[s + GAP, s + GAP] for _, s in items])
    bounds, inserted = packing.rectangles(
        extents, size=[bin_w + GAP, bin_w + GAP], spacing=0, rotate=False,
        iterations=250)
    if not inserted.all():
        return None
    order = [i for i, ins in enumerate(inserted) if ins]
    return [(items[i][0], (bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2)
            for i, bb in zip(order, bounds)]


def shelf_pack(items: list[tuple[str, float]], bin_w: float,
               max_bins: int | None = None) -> list[list[tuple[str, float, float]]]:
    """Balanced multi-bin packing: snake-deal items (largest first) across n
    bins, verify each bin packs, repair failures by random inter-bin swaps."""
    import random

    rng = random.Random(0)
    n_min = max(1, -(-int(sum(s * s for _, s in items)) // int(0.83 * bin_w * bin_w)))
    for n_bins in range(n_min, len(items) + 1):
        ordered = sorted(items, key=lambda t: -t[1])
        assign = [[] for _ in range(n_bins)]
        for i, it in enumerate(ordered):
            k = i % (2 * n_bins)
            assign[k if k < n_bins else 2 * n_bins - 1 - k].append(it)
        for attempt in range(1200):
            packed = [_pack_one(b, bin_w) for b in assign]
            if all(p is not None for p in packed):
                print(f"packed into {n_bins} bins after {attempt} repair attempts")
                return packed
            bad = [i for i, p in enumerate(packed) if p is None]
            src = rng.choice(bad)
            dst = rng.randrange(n_bins)
            if dst == src or not assign[src]:
                continue
            i = rng.randrange(len(assign[src]))
            if rng.random() < 0.5 and assign[dst]:
                j = rng.randrange(len(assign[dst]))
                if assign[dst][j][1] < assign[src][i][1]:
                    assign[src][i], assign[dst][j] = assign[dst][j], assign[src][i]
            else:
                assign[dst].append(assign[src].pop(i))
        print(f"{n_bins} bins failed, trying {n_bins + 1}")
    raise RuntimeError("packing failed")


def build_plate(plate_name: str, placements: list[tuple[str, float, float]]) -> dict:
    scene = trimesh.Scene()
    info = []
    for name, cx, cy in placements:
        mesh = trimesh.load(STL / f"{name}.stl")
        b = mesh.bounds
        mcx, mcy = (b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2
        x, y = MARGIN + cx, MARGIN + cy
        mesh.apply_translation([x - mcx, y - mcy, 0])
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        size = mesh.bounds[1] - mesh.bounds[0]
        info.append({"name": name, "x": round(x, 1), "y": round(y, 1),
                     "w": round(size[0], 2), "d": round(size[1], 2), "h": round(size[2], 2)})
    path = OUT / f"{plate_name}.3mf"
    scene.export(path)
    sb = scene.bounds
    assert sb[0][0] >= 0 and sb[0][1] >= 0 and sb[1][0] <= PLATE and sb[1][1] <= PLATE, \
        f"{plate_name} exceeds plate: {sb}"
    print(f"{plate_name}: {len(placements)} objects, bounds x[{sb[0][0]:.0f},{sb[1][0]:.0f}] "
          f"y[{sb[0][1]:.0f},{sb[1][1]:.0f}] zmax {sb[1][2]:.0f}")
    return {"plate": plate_name, "objects": info}


def draw_map(plates: list[dict]) -> None:
    px = 3  # pixels per mm
    for p in plates:
        img = Image.new("RGB", (int(PLATE * px), int(PLATE * px)), "white")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, PLATE * px - 1, PLATE * px - 1], outline="black", width=3)
        for o in p["objects"]:
            x, y, w, dd = o["x"] * px, (PLATE - o["y"]) * px, o["w"] * px, o["d"] * px
            name = o["name"]
            color = ("#4c9be8" if name.startswith("cyl") else
                     "#e8a04c" if name.startswith("cub") else
                     "#7cc47c" if name.startswith("sph") else "#c47c7c")
            if name.startswith(("cyl", "sph", "screw", "plug")):
                d.ellipse([x - w / 2, y - dd / 2, x + w / 2, y + dd / 2],
                          outline=color, fill=None, width=4)
            else:
                d.rectangle([x - w / 2, y - dd / 2, x + w / 2, y + dd / 2],
                            outline=color, fill=None, width=4)
            label = name.replace(".stl", "").replace("screwdriver_", "")
            d.text((x - len(label) * 3.2, y - 6), label, fill="black")
        img.save(OUT / f"{p['plate']}_layout.png")


def inject_object_settings(path: Path, overrides_by_name: dict[str, dict[str, str]]) -> None:
    """Add Bambu per-object settings to a plain 3MF as model_settings.config."""
    import re
    import zipfile

    with zipfile.ZipFile(path) as z:
        model = z.read("3D/3dmodel.model").decode()
    ids = {name: oid for oid, name in
           re.findall(r'<object id="(\d+)" name="([^"]+)"', model)}
    blocks = []
    for name, overrides in overrides_by_name.items():
        metas = "\n".join(f'    <metadata key="{k}" value="{v}"/>'
                          for k, v in overrides.items())
        blocks.append(f'  <object id="{ids[name]}">\n'
                      f'    <metadata key="name" value="{name}"/>\n{metas}\n  </object>')
    cfg = ('<?xml version="1.0" encoding="UTF-8"?>\n<config>\n'
           + "\n".join(blocks) + "\n</config>\n")
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("Metadata/model_settings.config", cfg)
    print(f"injected per-object settings into {path.name}: {list(overrides_by_name)}")


def main() -> None:
    fp = {f.stem: footprint(f.stem, trimesh.load(f)) for f in sorted(STL.glob("*.stl"))}
    groups = {
        "starter_plate": [n for n in STARTER],
        "cylinders_plate": [n for n in fp if n.startswith("cyl") and n not in STARTER],
        "cuboids_plate": [n for n in fp if n.startswith("cub") and n not in STARTER],
        "spheres_plate": [n for n in fp if n.startswith("sph") and n not in STARTER],
        "screwdrivers_plate": [n for n in fp if n.startswith(("screwdriver", "plug"))
                               and n not in STARTER],
    }
    plates = []
    for gname, names in groups.items():
        bins = shelf_pack([(n, fp[n]) for n in names], PLATE - 2 * MARGIN)
        for i, b in enumerate(bins):
            pname = gname if len(bins) == 1 else f"{gname}{i + 1}"
            plates.append(build_plate(pname, b))
            # wide flat discs need no brim (the plate-level brim would spill
            # past their reserved cells); starter screwdriver gets its own
            overrides = {n: {"brim_type": "no_brim"}
                         for n, _, _ in bins[i] if n.startswith("plug")}
            if STARTER[0] in [n for n, _, _ in bins[i]]:
                overrides[STARTER[0]] = SCREWDRIVER_OVERRIDES
            if overrides:
                inject_object_settings(OUT / f"{pname}.3mf", overrides)
    draw_map(plates)
    (OUT / "layout.json").write_text(json.dumps(plates, indent=1))
    print(f"total plates: {len(plates)}")


if __name__ == "__main__":
    main()
