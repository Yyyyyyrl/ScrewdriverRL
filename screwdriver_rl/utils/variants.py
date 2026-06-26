"""Screwdriver geometry-variant manifest loading + per-env identification.

Pure-``torch`` helpers (no Isaac Lab dependency) so they can be unit-tested
offline.  The manifest is produced by ``tools/generate_screwdriver_variants.py``.

The env spawns a fixed-per-env geometry variant (PhysX cannot rescale a cooked
collider at runtime), then recovers *which* variant landed in each env from the
handle's un-randomised ``(default_mass, default_izz)`` signature — a 2-D key that
uniquely pins ``(radius, length)``.  A 1-D mass key would alias ``(r, L)`` cells
(mass ∝ r²·L), so both channels are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Base reference geometry (mirrors tools/generate_screwdriver_variants.py and
# the base screwdriver URDF).  Used to turn a diameter scale into a radius.
BASE_RADIUS = 0.02


class VariantTable:
    """Manifest rows as parallel tensors for vectorised lookup."""

    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        vs = manifest["variants"]
        self.files: list[str] = [v["file"] for v in vs]
        self.mass = torch.tensor([v["mass"] for v in vs], dtype=torch.float32)
        self.izz = torch.tensor([v["izz"] for v in vs], dtype=torch.float32)
        self.bucket = torch.tensor([v["bucket"] for v in vs], dtype=torch.long)
        self.diameter_scale = torch.tensor(
            [v["diameter_scale"] for v in vs], dtype=torch.float32
        )
        self.length_scale = torch.tensor(
            [v["length_scale"] for v in vs], dtype=torch.float32
        )
        self.radius = torch.tensor([v["radius"] for v in vs], dtype=torch.float32)
        self.length = torch.tensor([v["length"] for v in vs], dtype=torch.float32)
        self.num_buckets: int = int(manifest["num_buckets"])
        self.rel_tol: float = float(manifest.get("signature_rel_tol", 0.02))

    @property
    def num_variants(self) -> int:
        return int(self.mass.shape[0])


def load_variant_table(path: str | Path) -> VariantTable:
    with open(path) as f:
        return VariantTable(json.load(f))


def identify_variants(
    masses: torch.Tensor,
    izzs: torch.Tensor,
    table: VariantTable,
    rel_tol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match each env's measured ``(mass, izz)`` to exactly one variant.

    Args:
        masses: ``(N,)`` measured per-env handle mass.
        izzs:   ``(N,)`` measured per-env handle z-inertia.
        table:  the parsed variant manifest.
        rel_tol: optional override of the manifest's relative tolerance.

    Returns:
        ``(variant_idx (N,), bucket_idx (N,), geom_scale (N, 2))`` where
        ``geom_scale = [diameter_scale, length_scale]``.

    Raises:
        AssertionError: if any env's signature is not within ``rel_tol`` of its
        nearest variant (i.e. the spawned handle is not a known variant).
    """
    masses = masses.to(torch.float32).flatten()
    izzs = izzs.to(torch.float32).flatten()
    tol = table.rel_tol if rel_tol is None else float(rel_tol)
    device = masses.device

    m_ref = table.mass.to(device)            # (V,)
    z_ref = table.izz.to(device)             # (V,)
    dm = (masses[:, None] - m_ref[None, :]).abs() / m_ref[None, :]   # (N, V)
    dz = (izzs[:, None] - z_ref[None, :]).abs() / z_ref[None, :]     # (N, V)
    # Both channels must match → score is the worse of the two; nearest wins.
    score = torch.maximum(dm, dz)
    best = torch.argmin(score, dim=1)                                # (N,)
    best_score = score.gather(1, best[:, None]).squeeze(1)           # (N,)

    if bool((best_score > tol).any()):
        n_bad = int((best_score > tol).sum())
        worst = float(best_score.max())
        raise AssertionError(
            f"{n_bad} env(s) did not match any geometry variant within "
            f"rel_tol={tol} (worst rel-dist {worst:.4f}). The spawned handle's "
            "(mass, izz) signature does not correspond to a known variant — "
            "check the MultiAssetSpawner / manifest."
        )

    bucket_idx = table.bucket.to(device)[best]
    geom_scale = torch.stack(
        [table.diameter_scale.to(device)[best], table.length_scale.to(device)[best]],
        dim=-1,
    )
    return best, bucket_idx, geom_scale


# ---------------------------------------------------------------------------
# Per-(diameter,length)-bucket pregrasp seeding (see plan §3d)
# ---------------------------------------------------------------------------
# The LinkerL20 index finger stabilises the cap, so handle *length* is NOT
# grasp-neutral → postures are bucketed over BOTH diameter and length.  Each
# bucket's posture is seeded analytically from the validated base grasp; the
# 32-step close-to-contact settle absorbs the residual, and the seed should be
# refined in-sim with tools/render_task_configs.py + the contact-band gate.
#
# Diameter → radial reach: a thinner handle (r < BASE_RADIUS) needs MORE flexion
# to keep the fingertip on the surface.  delta_flex = (BASE_RADIUS - r)/RADIAL_ARM,
# split over the per-finger flexion joints by ``flex_weights``.  Length is left at
# gain 0 by default (the cap shift is <=0.01 m, absorbed by the settle) — TUNE.
RADIAL_ARM = 0.04  # effective finger lever [m]; TUNE with the render tool
LENGTH_GAIN = 0.0  # per-(length-deviation) index adjustment; TUNE


def seed_pregrasp_buckets(
    base_pregrasp: dict[str, tuple[float, ...]],
    manifest: dict,
    flex_weights: dict[str, tuple[float, ...]],
    *,
    base_radius: float = BASE_RADIUS,
    radial_arm: float = RADIAL_ARM,
    length_gain: float = LENGTH_GAIN,
) -> list[dict[str, tuple[float, ...]]]:
    """Build one pregrasp dict per variant bucket from the base grasp.

    Buckets are indexed by the manifest's combined ``bucket`` id; in the default
    grid each variant is its own bucket.  ``flex_weights`` gives, per finger, the
    fraction of the radial flexion delta applied to each joint of its pregrasp
    tuple (abduction joint typically 0, flexion joints share the rest).
    """
    buckets: list[dict | None] = [None] * int(manifest["num_buckets"])
    base_len = float(manifest["base"]["length"])
    for v in manifest["variants"]:
        d_flex = (base_radius - float(v["radius"])) / radial_arm
        l_dev = float(v["length"]) - base_len
        posture: dict[str, tuple[float, ...]] = {}
        for finger, vals in base_pregrasp.items():
            w = flex_weights[finger]
            row = [p + d_flex * w[i] for i, p in enumerate(vals)]
            if finger == "index":  # only the cap stabiliser tracks length
                row = [p + length_gain * l_dev * w[i] for i, p in enumerate(row)]
            posture[finger] = tuple(row)
        buckets[int(v["bucket"])] = posture
    return [b if b is not None else dict(base_pregrasp) for b in buckets]
