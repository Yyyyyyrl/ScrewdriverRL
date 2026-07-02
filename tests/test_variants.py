"""Tests for screwdriver geometry variants + per-env identification. No Isaac Sim.

Covers the offline generator self-checks, the manifest/URDF invariants, the
``(mass, izz)`` → variant identification (round-trip + ambiguity guard), and the
per-(diameter,length)-bucket pregrasp seeding.

The pregrasp/flex-weight fixtures are read **live** from the task source via ``ast``
(no ``isaaclab`` import needed) so these tests validate whatever posture is currently
in the cfg — they catch a joint-limit-violating posture or a pregrasp/init_state
desync as the posture is tuned.

Run:  python tests/test_variants.py   (or  python -m pytest tests/ -q)
"""

import ast
import importlib.util
import json
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
HAND_URDF = ROOT / "assets" / "linker_hand_l20" / "linkerhand_l20_left.urdf"
CFG_PATH = ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "screwdriver_rotation_env_cfg.py"
ENV_PATH = ROOT / "screwdriver_rl" / "tasks" / "linker_l20" / "screwdriver_rotation_env.py"

# Reset target-clamp constants (mirror cfg defaults: base ``joint_target_margin``,
# LinkerL20 ``joint_motion_range``).  Soft limits == URDF hard limits (no
# ``soft_joint_pos_limit_factor`` override), so the reset clamp window is
# ``[lo+MARGIN, hi-MARGIN]`` intersected with ``home ± MOTION``.
JOINT_TARGET_MARGIN = 0.02
JOINT_MOTION_RANGE = 0.35


# ---------------------------------------------------------------------------
# Live source extraction (via ast — no isaaclab / torch on the task modules)
# ---------------------------------------------------------------------------

def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_assign_value(tree: ast.Module, name: str) -> ast.expr:
    """Value node of the first top-level-or-nested ``name = ...`` / ``name: T = ...``."""
    for node in ast.walk(tree):
        targets = (
            node.targets if isinstance(node, ast.Assign)
            else [node.target] if isinstance(node, ast.AnnAssign) and node.target
            else []
        )
        for t in targets:
            if isinstance(t, ast.Name) and t.id == name:
                return node.value
    raise KeyError(f"{name} not found")


def _literal(path: Path, name: str):
    return ast.literal_eval(_find_assign_value(_module_ast(path), name))


def _field_default_dict(path: Path, name: str) -> dict:
    """Extract the dict from ``name: T = field(default_factory=lambda: {...})``."""
    val = _find_assign_value(_module_ast(path), name)
    assert isinstance(val, ast.Call), f"{name} is not a field(...) call"
    for kw in val.keywords:
        if kw.arg == "default_factory":
            assert isinstance(kw.value, ast.Lambda)
            return ast.literal_eval(kw.value.body)
    raise KeyError(f"{name} has no default_factory")


def _hand_init_joint_pos() -> dict:
    """The hand's ``init_state`` ``joint_pos={...}`` dict from the cfg."""
    for node in ast.walk(_module_ast(CFG_PATH)):
        if isinstance(node, ast.keyword) and node.arg == "joint_pos" and isinstance(node.value, ast.Dict):
            d = ast.literal_eval(node.value)
            if "index_pip" in d:  # the hand's (not the screwdriver's) joint_pos
                return d
    raise KeyError("hand init_state joint_pos not found")


def _hand_joint_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(HAND_URDF).getroot()
    out: dict[str, tuple[float, float]] = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is not None and lim.get("lower") is not None and lim.get("upper") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


# Live posture / weights / hand layout, parsed from the task source.
_LINKER_PREGRASP = _field_default_dict(CFG_PATH, "pregrasp_positions")
_FLEX_WEIGHTS = _literal(CFG_PATH, "_FLEX_WEIGHTS")
_FINGER_JOINT_NAMES = _literal(ENV_PATH, "FINGER_JOINT_NAMES")
_COUPLED_JOINTS = _literal(ENV_PATH, "COUPLED_JOINTS")


def _apply_couplers(independent: dict[str, float]) -> dict[str, float]:
    """Add the mimic followers (``follower = master*mult + offset``)."""
    out = dict(independent)
    for follower, (master, mult, off) in _COUPLED_JOINTS.items():
        out[follower] = out[master] * mult + off
    return out


# ---------------------------------------------------------------------------
# Generator + manifest + identification (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pregrasp seeding (validated against the LIVE posture)
# ---------------------------------------------------------------------------

def test_seed_pregrasp_buckets():
    manifest = json.loads(MANIFEST.read_text())
    buckets = seed_pregrasp_buckets(_LINKER_PREGRASP, manifest, _FLEX_WEIGHTS)
    assert len(buckets) == manifest["num_buckets"]
    # every finger present with the right joint count
    for b in buckets:
        for finger, vals in _LINKER_PREGRASP.items():
            assert len(b[finger]) == len(vals)

    by_bucket = {v["bucket"]: v for v in manifest["variants"]}
    thin = min(by_bucket, key=lambda k: by_bucket[k]["radius"])
    thick = max(by_bucket, key=lambda k: by_bucket[k]["radius"])
    assert by_bucket[thin]["radius"] < by_bucket[thick]["radius"]

    # Each weighted joint flexes MORE on the thinner handle; zero-weight joints are
    # identical to the base posture across all buckets.  Design-agnostic: works
    # whichever joints the flex weights are assigned to.
    saw_weighted = False
    for finger, weights in _FLEX_WEIGHTS.items():
        for i, w in enumerate(weights):
            if w > 0:
                assert buckets[thin][finger][i] > buckets[thick][finger][i]
                saw_weighted = True
            else:
                assert abs(buckets[thin][finger][i] - _LINKER_PREGRASP[finger][i]) < 1e-9
                assert abs(buckets[thick][finger][i] - _LINKER_PREGRASP[finger][i]) < 1e-9
    assert saw_weighted, "expected at least one weighted flexion joint"


def test_seeded_buckets_within_joint_limits():
    """Every seeded bucket posture (independent joints AND coupled followers) must
    sit inside the URDF hard limits *and* the reset target-clamp window, else the
    over-flex is silently clamped / the pregrasp target is yanked at episode start.

    This is the tripwire the earlier ``thumb_cmc_roll=1.180`` posture would have
    tripped (its thin bucket reached 1.2175 vs the 1.22 limit / 1.20 clamp ceiling).
    """
    manifest = json.loads(MANIFEST.read_text())
    buckets = seed_pregrasp_buckets(_LINKER_PREGRASP, manifest, _FLEX_WEIGHTS)
    limits = _hand_joint_limits()
    eps = 1e-6
    for b_idx, posture in enumerate(buckets):
        independent = {
            name: v
            for finger, names in _FINGER_JOINT_NAMES.items()
            for name, v in zip(names, posture[finger])
        }
        for name, v in _apply_couplers(independent).items():
            lo, hi = limits[name]
            assert lo - eps <= v <= hi + eps, (
                f"bucket {b_idx}: {name}={v:.4f} exceeds URDF limit [{lo}, {hi}]"
            )
            up = min(hi - JOINT_TARGET_MARGIN, v + JOINT_MOTION_RANGE)
            dn = max(lo + JOINT_TARGET_MARGIN, v - JOINT_MOTION_RANGE)
            assert dn - eps <= v <= up + eps, (
                f"bucket {b_idx}: {name}={v:.4f} would be clamped at reset "
                f"(window [{dn:.4f}, {up:.4f}]); needs >= {JOINT_TARGET_MARGIN} rad "
                f"of margin to its URDF limit"
            )


def test_pregrasp_matches_init_state():
    """The pregrasp table and the hand ``init_state.joint_pos`` must agree: the
    independent joints are equal and the followers equal ``master*mult+offset`` —
    the comment-only sync the posture-tuning keeps by hand."""
    jp = _hand_init_joint_pos()
    for finger, names in _FINGER_JOINT_NAMES.items():
        for name, v in zip(names, _LINKER_PREGRASP[finger]):
            assert abs(jp[name] - v) < 1e-6, (
                f"init_state.{name}={jp[name]} != pregrasp {v}"
            )
    for follower, (master, mult, off) in _COUPLED_JOINTS.items():
        expected = jp[master] * mult + off
        assert abs(jp[follower] - expected) < 1e-4, (
            f"init_state.{follower}={jp[follower]} != {master}*{mult}+{off}={expected:.6f}"
        )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
