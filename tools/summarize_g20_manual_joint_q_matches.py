#!/usr/bin/env python3
"""Summarize human same-view physical-raw to URDF-local-q matches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def direction_curve(
    rows: list[dict[str, Any]], directions: set[str]
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for item in rows:
        if item["direction"] in directions:
            grouped[int(item["stable_readback_raw"])].append(
                float(item["q_urdf_rad"])
            )
    raw = np.asarray(sorted(grouped), dtype=float)
    q = np.asarray([np.mean(grouped[int(value)]) for value in raw])
    return raw, q


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matches_payload = json.loads(args.matches.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    all_rows = list(matches_payload["matches"])
    matched = [row for row in all_rows if row["status"] == "matched"]
    if len(all_rows) != int(matches_payload["sample_count"]):
        raise RuntimeError("not every sample has been reviewed")
    if len(matched) != len(all_rows):
        raise RuntimeError("summary requires every reviewed sample to be matched")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in matched:
        grouped[int(item["stable_readback_raw"])].append(item)
    unique: list[dict[str, Any]] = []
    for raw in sorted(grouped):
        rows = grouped[raw]
        q_values = np.asarray([float(row["q_urdf_rad"]) for row in rows])
        unique.append(
            {
                "stable_readback_raw": raw,
                "q_urdf_local_rad": float(np.mean(q_values)),
                "q_urdf_local_deg": float(math.degrees(np.mean(q_values))),
                "sample_count": len(rows),
                "q_min_rad": float(np.min(q_values)),
                "q_max_rad": float(np.max(q_values)),
                "directions": sorted(
                    {str(row["direction"]) for row in rows}
                ),
                "sample_ids": [str(row["sample_id"]) for row in rows],
            }
        )

    raw = np.asarray(
        [row["stable_readback_raw"] for row in unique], dtype=float
    )
    q = np.asarray([row["q_urdf_local_rad"] for row in unique], dtype=float)
    slope, intercept = np.polyfit(raw, q, 1)
    monotonic_direction = "increasing" if slope >= 0.0 else "decreasing"
    monotonic_violations = [
        {
            "raw_a": int(raw[index]),
            "q_a": float(q[index]),
            "raw_b": int(raw[index + 1]),
            "q_b": float(q[index + 1]),
        }
        for index in range(len(raw) - 1)
        if (
            q[index + 1] < q[index] - 1.0e-9
            if monotonic_direction == "increasing"
            else q[index + 1] > q[index] + 1.0e-9
        )
    ]

    affine_pred = slope * raw + intercept
    affine_residual = q - affine_pred

    forward_raw, forward_q = direction_curve(
        matched, {"forward", "forward_reference"}
    )
    reverse_raw, reverse_q = direction_curve(matched, {"reverse"})
    if len(forward_raw) and len(reverse_raw):
        overlap_low = max(float(forward_raw.min()), float(reverse_raw.min()))
        overlap_high = min(float(forward_raw.max()), float(reverse_raw.max()))
        evaluation_raw = np.asarray(
            sorted(
                {
                    float(value)
                    for value in np.concatenate((forward_raw, reverse_raw))
                    if overlap_low <= value <= overlap_high
                }
            )
        )
        forward_interp = np.interp(evaluation_raw, forward_raw, forward_q)
        reverse_interp = np.interp(evaluation_raw, reverse_raw, reverse_q)
        hysteresis = forward_interp - reverse_interp
    else:
        overlap_low = None
        overlap_high = None
        evaluation_raw = np.asarray([], dtype=float)
        hysteresis = np.asarray([], dtype=float)

    production_limit = dataset.get(
        "production_urdf_limit_rad", [0.0, 1.22]
    )
    production_lower = float(production_limit[0])
    production_upper = float(production_limit[1])
    candidate_step = float(
        dataset["q_step_rad"]
        if "q_step_rad" in dataset
        else dataset["q_grid_rad"]["step"]
    )
    outside_rows = [
        item for item in matched
        if float(item["q_urdf_rad"]) > production_upper + 1.0e-9
        or float(item["q_urdf_rad"]) < production_lower - 1.0e-9
    ]
    outside_unique = [
        item for item in unique
        if float(item["q_urdf_local_rad"]) > production_upper + 1.0e-9
        or float(item["q_urdf_local_rad"]) < production_lower - 1.0e-9
    ]

    full_csv = args.out_dir / "manual_matches_all_samples.csv"
    full_fields = [
        "sample_id",
        "direction",
        "command_raw",
        "stable_readback_raw",
        "q_urdf_rad",
        "q_urdf_deg",
        "status",
        "real_color",
    ]
    with full_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=full_fields)
        writer.writeheader()
        for item in matched:
            writer.writerow({field: item.get(field) for field in full_fields})

    lut_csv = args.out_dir / "stable_raw_to_urdf_local_q_lut.csv"
    lut_fields = [
        "stable_readback_raw",
        "q_urdf_local_rad",
        "q_urdf_local_deg",
        "sample_count",
        "q_min_rad",
        "q_max_rad",
        "directions",
        "sample_ids",
    ]
    with lut_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=lut_fields)
        writer.writeheader()
        for item in unique:
            output = dict(item)
            output["directions"] = "|".join(output["directions"])
            output["sample_ids"] = "|".join(output["sample_ids"])
            writer.writerow(output)

    figure_path = args.out_dir / "stable_raw_vs_urdf_local_q.png"
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=160)
    ax.axhspan(
        production_lower,
        production_upper,
        color="#DCE2EA",
        alpha=0.65,
        label="original OG URDF declared range",
    )
    ax.axhline(
        production_upper,
        color="#424A55",
        linewidth=1.5,
        linestyle="--",
        label=f"OG upper = {production_upper:.2f} rad",
    )
    ax.plot(
        raw,
        q,
        color="#2E6F9E",
        linewidth=2.2,
        marker="o",
        markersize=4.5,
        label="unique stable-raw LUT (duplicate mean)",
    )
    ax.scatter(
        forward_raw,
        forward_q,
        s=54,
        marker="o",
        facecolors="white",
        edgecolors="#2E6F9E",
        linewidths=1.5,
        label="forward / reference photos",
        zorder=4,
    )
    ax.scatter(
        reverse_raw,
        reverse_q,
        s=52,
        marker="^",
        color="#D4812A",
        edgecolors="#7B4A1F",
        linewidths=0.8,
        label="reverse photos",
        zorder=4,
    )
    ax.plot(
        raw,
        affine_pred,
        color="#69727E",
        linewidth=1.4,
        linestyle=":",
        label="affine diagnostic",
    )
    endpoints = (unique[0], unique[-1])
    for point in endpoints:
        is_right_endpoint = int(point["stable_readback_raw"]) < 50
        ax.annotate(
            (
                f"raw {point['stable_readback_raw']}  "
                f"q={point['q_urdf_local_rad']:.2f}"
            ),
            (
                point["stable_readback_raw"],
                point["q_urdf_local_rad"],
            ),
            xytext=(-92, 12) if is_right_endpoint else (10, 12),
            textcoords="offset points",
            fontsize=9,
            color="#252A31",
        )
    ax.set_title(
        f"{dataset['joint']}: physical stable raw vs OG URDF local q",
        loc="left",
        pad=38,
        fontsize=15,
        fontweight="bold",
        color="#252A31",
    )
    ax.text(
        0.0,
        1.015,
        (
            f"{len(matched)} same-view manual matches; "
            f"{len(unique)} unique stable raw values; "
            f"candidate step {candidate_step:.2f} rad"
        ),
        transform=ax.transAxes,
        fontsize=10,
        color="#5F6873",
    )
    ax.set_xlabel("Physical SDK stable readback raw")
    ax.set_ylabel("OG URDF local joint q (rad)")
    ax.set_xlim(255, 0)
    plot_lower = min(production_lower, float(q.min())) - 0.08
    plot_upper = max(production_upper, float(q.max())) + 0.08
    ax.set_ylim(plot_lower, plot_upper)
    ax.grid(axis="both", color="#D9DEE5", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "candidate_only_not_promoted",
        "method": "human_same_view_photo_to_og_urdf_local_q",
        "joint": dataset["joint"],
        "provenance": {
            "matches": str(args.matches.resolve()),
            "matches_sha256": sha256(args.matches),
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": sha256(args.dataset),
            "camera": dataset["camera"],
            "camera_sha256": dataset["camera_sha256"],
            "render_urdf": dataset["urdf"],
            "render_urdf_sha256": dataset["urdf_sha256"],
            "production_og_urdf": dataset.get("production_og_urdf", dataset.get("production_urdf")),
            "production_og_urdf_sha256": dataset.get(
                "production_og_urdf_sha256",
                dataset.get("production_urdf_sha256"),
            ),
        },
        "review": {
            "sample_count": len(all_rows),
            "matched_count": len(matched),
            "unique_stable_raw_count": len(unique),
            "candidate_step_rad": candidate_step,
            "quantization_half_step_rad": candidate_step / 2.0,
            "monotonic_direction_from_affine_slope": monotonic_direction,
            "monotonic_violation_count": len(monotonic_violations),
            "monotonic_violations": monotonic_violations,
        },
        "observed_absolute_urdf_local_q_range": {
            "min_raw": int(raw.min()),
            "max_raw": int(raw.max()),
            "q_at_min_raw_rad": float(q[0]),
            "q_at_max_raw_rad": float(q[-1]),
            "q_min_rad": float(q.min()),
            "q_max_rad": float(q.max()),
            "span_rad": float(q.max() - q.min()),
            "span_deg": float(math.degrees(q.max() - q.min())),
        },
        "original_og_limit_assessment": {
            "declared_limit_rad": [production_lower, production_upper],
            "photo_count_outside": len(outside_rows),
            "unique_raw_count_outside": len(outside_unique),
            "lower_shortfall_to_observed_endpoint_rad": float(
                max(0.0, production_lower - q.min())
            ),
            "upper_shortfall_to_observed_endpoint_rad": float(
                max(0.0, q.max() - production_upper)
            ),
            "physical_semantic_zero_at_max_raw": int(raw.max()),
            "physical_semantic_zero_at_max_raw_local_q_rad": float(q[-1]),
        },
        "affine_diagnostic": {
            "formula": "q_urdf_local = slope_rad_per_raw * raw + intercept_rad",
            "slope_rad_per_raw": float(slope),
            "intercept_rad": float(intercept),
            "rmse_rad": float(
                np.sqrt(np.mean(affine_residual * affine_residual))
            ),
            "max_abs_rad": float(np.max(np.abs(affine_residual))),
        },
        "direction_hysteresis_interpolated": {
            "overlap_raw": (
                [overlap_low, overlap_high]
                if overlap_low is not None
                else None
            ),
            "evaluation_count": int(len(evaluation_raw)),
            "forward_minus_reverse_mean_rad": (
                float(np.mean(hysteresis)) if len(hysteresis) else None
            ),
            "rmse_rad": (
                float(np.sqrt(np.mean(hysteresis * hysteresis)))
                if len(hysteresis)
                else None
            ),
            "max_abs_rad": (
                float(np.max(np.abs(hysteresis)))
                if len(hysteresis)
                else None
            ),
        },
        "physical_lut_urdf_local_q": {
            "raw": [int(value) for value in raw],
            "q_urdf_local_rad": [float(value) for value in q],
            "duplicate_policy": "mean of same stable raw manual matches",
        },
        "artifacts": {
            "all_samples_csv": str(full_csv.resolve()),
            "unique_lut_csv": str(lut_csv.resolve()),
            "audit_figure": str(figure_path.resolve()),
        },
        "promotion_blockers": (
            [
                (
                    f"Original OG {dataset['joint']} limit "
                    f"[{production_lower},{production_upper}] does not cover "
                    "the observed absolute local-q interval."
                )
            ]
            if outside_rows
            else []
        )
        + [
            (
                "All 21 joints must complete the same visual-registration "
                "review before any runtime configuration is changed."
            ),
            (
                "Manual matching values carry at least half a candidate-grid "
                "step of quantization uncertainty."
            ),
        ],
    }
    summary_path = args.out_dir / "manual_registration_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "matched": len(matched),
                "unique_raw": len(unique),
                "q_range_rad": [float(q.min()), float(q.max())],
                "monotonic_violations": len(monotonic_violations),
                "outside_original_og_limit_photos": len(outside_rows),
                "affine_rmse_rad": payload["affine_diagnostic"]["rmse_rad"],
                "hysteresis_max_abs_rad": payload[
                    "direction_hysteresis_interpolated"
                ]["max_abs_rad"],
                "figure": str(figure_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
