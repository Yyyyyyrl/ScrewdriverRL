"""Tests for screwdriver geometry variants + per-env identification. No Isaac Sim.

Covers the offline generator self-checks, the manifest/URDF invariants, the
``(mass, izz)`` → variant identification (round-trip + ambiguity guard), and the
per-(diameter,length)-bucket pregrasp seeding.

Run:  python tests/test_variants.py   (or  python -m pytest tests/ -q)
"""

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screwdriver_rl.utils.variants import (  # noqa: E402
    identify_variants,
    load_variant_table,
    seed_pregrasp_buckets,
)


def _load_generator():
    path = ROOT / "tools" / "generate_screwdriver_variants.py"
    spec = importlib.util.spec_from_file_location("genvar", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()
MANIFEST = ROOT / "assets" / "screwdriver" / "variants" / "manifest.json"

_LINKER_PREGRASP = {
    "index":  (0.112665, 0.073330, 1.073807),
    "middle": (-0.024834, 0.138236, 1.057659),
    "ring":   (0.053014, 0.399275, 0.909139),
    "pinky":  (0.123575, 0.694853, 0.638524),
    "thumb":  (1.065296, 0.601400, 0.040001, 0.574150),
}
_FLEX_WEIGHTS = {
    "index":  (0.0, 0.5, 0.5),
    "middle": (0.0, 0.5, 0.5),
    "ring":   (0.0, 0.5, 0.5),
    "pinky":  (0.0, 0.5, 0.5),
    "thumb":  (0.0, 0.5, 0.5, 0.0),
}


def test_generator_runs_and_self_checks_pass():
    """main() regenerates variants and runs every self-check (topology, dims,
    inertia formulas, mutual signature separation); it raises on any failure."""
    GEN.main()  # must not raise
    assert MANIFEST.exists()


def test_inertia_formulas():
    p = GEN._body_props(GEN.R0, GEN.L0)
    assert abs(p["mass"] - GEN.BODY_MASS0) < 1e-9
    # izz = 1/2 m r^2 at base
    assert abs(p["izz"] - 0.5 * GEN.BODY_MASS0 * GEN.R0**2) < 1e-12
    # mass scales as r^2 * L
    p2 = GEN._body_props(2 * GEN.R0, GEN.L0)
    assert abs(p2["mass"] - 4 * GEN.BODY_MASS0) < 1e-9


def test_variants_preserve_topology_and_dims():
    table = load_variant_table(MANIFEST)
    base_root = ET.parse(GEN.BASE_URDF).getroot()
    base_topo = GEN._topology(base_root)
    sd_root = MANIFEST.parent.parent  # .../assets/screwdriver
    for i, f in enumerate(table.files):
        root = ET.parse(sd_root / f).getroot()
        assert GEN._topology(root) == base_topo, f"{f}: topology drift"
        cyl = GEN._find_link(root, "screwdriver_body").find(
            "collision/geometry/cylinder"
        )
        assert abs(float(cyl.get("radius")) - float(table.radius[i])) < 1e-6
        assert abs(float(cyl.get("length")) - float(table.length[i])) < 1e-6


def test_signatures_mutually_separated():
    table = load_variant_table(MANIFEST)
    n = table.num_variants
    for i in range(n):
        for j in range(i + 1, n):
            dm = abs(table.mass[i] - table.mass[j]) / max(table.mass[i], table.mass[j])
            dz = abs(table.izz[i] - table.izz[j]) / max(table.izz[i], table.izz[j])
            assert max(dm, dz) >= table.rel_tol


def test_identify_roundtrip_exact():
    table = load_variant_table(MANIFEST)
    order = torch.arange(2 * table.num_variants) % table.num_variants
    vidx, bidx, gscale = identify_variants(table.mass[order], table.izz[order], table)
    assert torch.equal(vidx, order)
    assert torch.equal(bidx, table.bucket[order])
    # geometry scale channels match the manifest
    assert torch.allclose(gscale[:, 0], table.diameter_scale[order])
    assert torch.allclose(gscale[:, 1], table.length_scale[order])


def test_identify_tolerant_to_small_noise():
    table = load_variant_table(MANIFEST)
    torch.manual_seed(0)
    order = torch.arange(table.num_variants)
    # 0.1% jitter — well inside the separation tolerance
    m = table.mass[order] * (1 + 0.001 * torch.randn(table.num_variants))
    z = table.izz[order] * (1 + 0.001 * torch.randn(table.num_variants))
    vidx, _, _ = identify_variants(m, z, table)
    assert torch.equal(vidx, order)


def test_identify_rejects_unknown_signature():
    table = load_variant_table(MANIFEST)
    try:
        identify_variants(torch.tensor([999.0]), torch.tensor([999.0]), table)
    except AssertionError:
        return
    raise AssertionError("expected identify_variants to reject an unknown signature")


def test_seed_pregrasp_buckets():
    import json

    manifest = json.loads(MANIFEST.read_text())
    buckets = seed_pregrasp_buckets(_LINKER_PREGRASP, manifest, _FLEX_WEIGHTS)
    assert len(buckets) == manifest["num_buckets"]
    # every finger present with the right joint count
    for b in buckets:
        for finger, vals in _LINKER_PREGRASP.items():
            assert len(b[finger]) == len(vals)
    # thinner handle (smaller radius) → MORE index flexion than a thicker one
    by_bucket = {v["bucket"]: v for v in manifest["variants"]}
    thin = min(by_bucket, key=lambda k: by_bucket[k]["radius"])
    thick = max(by_bucket, key=lambda k: by_bucket[k]["radius"])
    assert buckets[thin]["index"][1] > buckets[thick]["index"][1]
    # abduction joint (weight 0) is unchanged from base
    assert abs(buckets[thin]["index"][0] - _LINKER_PREGRASP["index"][0]) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
