#!/usr/bin/env python3
"""Build the canonical technical-report artifact for the G20 mapping validation."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "records/g20_og_local_q_candidate_validation_20260805"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def build_key_visual(ablation_dir: Path, output: Path) -> None:
    pose_names = ["diag_thumb_opposition", "random_11", "diag_flex_wave"]
    column_titles = ["REAL D435", "CANDIDATE MIMIC", "OG MIMIC", "OVERLAY"]
    row_labels = ["Thumb opposition", "Random 11", "Flex wave"]
    panel_width, panel_height = 410, 330
    left, top, gap = 170, 64, 10
    canvas = np.full(
        (top + len(pose_names) * (panel_height + gap) + 30,
         left + len(column_titles) * (panel_width + gap) + 10, 3),
        246,
        dtype=np.uint8,
    )
    for col, title in enumerate(column_titles):
        x = left + col * (panel_width + gap)
        cv2.putText(
            canvas, title, (x + 8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
            (35, 40, 48), 2, cv2.LINE_AA,
        )
    for row, (name, row_label) in enumerate(zip(pose_names, row_labels)):
        image = cv2.imread(str(ablation_dir / f"{name}_calibration_ablation.png"))
        if image is None or image.shape[1] < 5120:
            raise RuntimeError(f"invalid ablation panel for {name}")
        y = top + row * (panel_height + gap)
        cv2.putText(
            canvas, row_label, (12, y + panel_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 40, 48), 2, cv2.LINE_AA,
        )
        for col in range(4):
            crop = image[70:570, 480 + col * 1280:1000 + col * 1280]
            crop = cv2.resize(crop, (panel_width, panel_height))
            x = left + col * (panel_width + gap)
            canvas[y:y + panel_height, x:x + panel_width] = crop
            cv2.rectangle(canvas, (x, y), (x + panel_width, y + panel_height), (175, 179, 186), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]):
        raise RuntimeError(f"failed to write {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RECORD / "report")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hardware_path = RECORD / "hardware/exp300_15poses_candidate_v1/summary.json"
    static_path = RECORD / "static_validation.json"
    og_metrics_path = RECORD / "analysis/exp300_15poses_settled_candidate/metrics.json"
    candidate_metrics_path = RECORD / "analysis/exp300_15poses_settled_candidate_mimic_affine/metrics.json"
    local_metrics_path = RECORD / "analysis/mimic_affine_vs_og_change_band_15poses/metrics.json"
    provenance_path = ROOT / "assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805_provenance.json"
    intrinsics_path = RECORD / "camera/current_d435_814412070035_intrinsics.json"
    camera_path = RECORD / "camera/camera_user_approved_current_hand_20260805.json"
    return_path = RECORD / "hardware/return_to_initial_candidate_pose/summary.json"

    hardware = load(hardware_path)
    static = load(static_path)
    og_metrics = load(og_metrics_path)
    candidate_metrics = load(candidate_metrics_path)
    local_metrics = load(local_metrics_path)
    provenance = load(provenance_path)
    intrinsics = load(intrinsics_path)

    all_joint_rows = [item for pose in hardware["poses"].values() for item in pose["per_joint"]]
    errors = np.array([abs(item["error_rad"]) for item in all_joint_rows], dtype=float)
    by_joint: list[dict] = []
    static_joint = {item["joint"]: item for item in static["active_joints"]}
    for name in hardware["joint_order16"]:
        rows = [item for item in all_joint_rows if item["name"] == name]
        values = np.array([abs(item["error_rad"]) for item in rows], dtype=float)
        meta = static_joint[name]
        by_joint.append(
            {
                "joint": name,
                "slot": int(meta["slot"]),
                "direction": meta["q_direction_with_increasing_raw"],
                "sample_count": len(rows),
                "mean_abs_error_rad": float(values.mean()),
                "p95_abs_error_rad": float(np.percentile(values, 95)),
                "max_abs_error_rad": float(values.max()),
                "raw_domain_low": float(meta["raw_domain"][0]),
                "raw_domain_high": float(meta["raw_domain"][1]),
                "q_domain_low_rad": float(meta["q_domain_rad"][0]),
                "q_domain_high_rad": float(meta["q_domain_rad"][1]),
            }
        )
    by_joint.sort(key=lambda row: row["max_abs_error_rad"], reverse=True)

    og_pose = {row["pose"]: row for row in og_metrics["poses"]}
    candidate_pose = {row["pose"]: row for row in candidate_metrics["poses"]}
    local_pose = {row["pose"]: row for row in local_metrics["poses"]}
    pose_rows: list[dict] = []
    for name, pose in hardware["poses"].items():
        pose_errors = np.array([abs(item["error_rad"]) for item in pose["per_joint"]])
        pose_rows.append(
            {
                "pose": name,
                "pose_type": "diagnostic" if name.startswith("diag_") else "random",
                "mean_abs_joint_error_rad": float(pose_errors.mean()),
                "worst_abs_joint_error_rad": float(pose_errors.max()),
                "fault_free": not any(pose["faults20_after"]),
                "max_temperature_c": int(max(pose["temperature20_after"])),
                "og_contour_mean_px": float(og_pose[name]["symmetric_contour_mean_px"]),
                "candidate_contour_mean_px": float(candidate_pose[name]["symmetric_contour_mean_px"]),
                "candidate_minus_og_contour_px": float(
                    candidate_pose[name]["symmetric_contour_mean_px"]
                    - og_pose[name]["symmetric_contour_mean_px"]
                ),
                "og_iou": float(og_pose[name]["silhouette_iou"]),
                "candidate_iou": float(candidate_pose[name]["silhouette_iou"]),
                "localized_candidate_minus_og_px": float(
                    local_pose[name]["candidate_minus_og_symmetric_px"]
                ),
            }
        )

    contour_rows = []
    for row in pose_rows:
        for variant, field in (
            ("OG mimic", "og_contour_mean_px"),
            ("Candidate mimic", "candidate_contour_mean_px"),
        ):
            contour_rows.append(
                {
                    "pose": row["pose"],
                    "pose_type": row["pose_type"],
                    "variant": variant,
                    "contour_mean_px": row[field],
                    "worst_abs_joint_error_rad": row["worst_abs_joint_error_rad"],
                    "max_temperature_c": row["max_temperature_c"],
                }
            )

    follower_rows = []
    follower_tidy = []
    followers = {item["joint"]: item for item in provenance["follower_mimic_candidates"]}
    for item in static["follower_mimic_diagnostics"]:
        fit = followers[item["joint"]]
        row = {
            "joint": item["joint"],
            "source_joint": item["source"],
            "sample_count": int(fit["sample_count"]),
            "candidate_multiplier": float(fit["affine_fit"]["multiplier"]),
            "candidate_offset_rad": float(fit["affine_fit"]["offset"]),
            "candidate_rmse_rad": float(item["candidate_affine_rmse_rad"]),
            "og_multiplier": float(item["og_mimic_multiplier"]),
            "og_rmse_rad": float(item["og_mimic_rmse_rad"]),
            "archive_rmse_reduction_percent": 100.0 * float(item["rmse_reduction_fraction"]),
            "independent_visual_decision": "inconclusive; do not promote",
        }
        follower_rows.append(row)
        for variant, value in (
            ("OG mimic", row["og_rmse_rad"]),
            ("Candidate affine", row["candidate_rmse_rad"]),
        ):
            follower_tidy.append(
                {
                    "joint": row["joint"],
                    "source_joint": row["source_joint"],
                    "variant": variant,
                    "rmse_rad": value,
                    "sample_count": row["sample_count"],
                    "archive_rmse_reduction_percent": row["archive_rmse_reduction_percent"],
                }
            )

    active = set(hardware["joint_order16"])
    follower = set(followers)
    calibration_rows = []
    for joint, item in provenance["joint_archives"].items():
        calibration_rows.append(
            {
                "joint": joint,
                "runtime_role": "active SDK" if joint in active else "mechanical follower",
                "source_joint": followers[joint]["source"] if joint in follower else "",
                "unique_stable_raw_count": int(item["unique_stable_raw_count"]),
                "observed_q_low_rad": float(item["observed_q_range_rad"][0]),
                "observed_q_high_rad": float(item["observed_q_range_rad"][1]),
                "outside_og_limit_count": int(item["unique_raw_count_outside_production_limit"]),
                "archive_affine_rmse_rad": float(item["affine_diagnostic_rmse_rad"]),
                "archive_status": "visual review pass; candidate only",
            }
        )

    headline = [{
        "pose_count": len(hardware["poses"]),
        "joint_observation_count": len(all_joint_rows),
        "p95_abs_error_rad": float(np.percentile(errors, 95)),
        "max_abs_error_rad": float(errors.max()),
        "fault_free_pose_count": sum(not any(p["faults20_after"]) for p in hardware["poses"].values()),
        "max_temperature_c": max(max(p["temperature20_after"]) for p in hardware["poses"].values()),
        "active_joint_count": len(active),
        "follower_joint_count": len(follower),
        "localized_candidate_better_count": int(local_metrics["candidate_better_pose_count"]),
        "localized_candidate_change_percent": float(local_metrics["candidate_improvement_percent"]),
    }]

    key_visual_path = args.out_dir / "key_visual_real_candidate_og.jpg"
    build_key_visual(RECORD / "comparisons/mimic_affine_vs_og_15poses", key_visual_path)
    encoded_visual = base64.b64encode(key_visual_path.read_bytes()).decode("ascii")

    datasets = {
        "headline": headline,
        "joint_errors": by_joint,
        "pose_metrics": pose_rows,
        "contour_comparison": contour_rows,
        "follower_rmse": follower_tidy,
        "follower_detail": follower_rows,
        "calibration_21": calibration_rows,
    }
    row_dir = args.out_dir / "report_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    for dataset_id, rows in datasets.items():
        (row_dir / f"{dataset_id}.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

    def dataset_source(dataset_id: str, label: str, definitions: list[str]) -> dict:
        path = relative(row_dir / f"{dataset_id}.json")
        return {
            "id": f"{dataset_id}_rows",
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_json_auto('{path}')",
                "description": f"Read the reviewed, materialized {dataset_id} rows used by this report.",
                "tables_used": [path],
                "filters": ["Materialized from the 2026-08-05 validation record; no row sampling."],
                "metric_definitions": definitions,
                "executed_at": datetime.now(timezone.utc).isoformat(),
            },
        }

    sources = [
        dataset_source("headline", "Report headline metrics", ["P95 and maximum are computed over 240 absolute settled active-joint q errors.", "Fault-free pose count requires all 20 post-pose fault slots to equal zero."]),
        dataset_source("joint_errors", "Active-joint error rows", ["Mean, P95, and maximum absolute q error are computed independently for each active joint across 15 poses."]),
        dataset_source("pose_metrics", "Per-pose validation rows", ["Worst absolute q error is the maximum across 16 active joints in one settled pose.", "Contour distance is the symmetric mean edge distance within ROI [480,90,1000,550]."]),
        dataset_source("contour_comparison", "OG and candidate contour rows", ["Each pose contributes one OG-mimic and one candidate-mimic symmetric contour distance in pixels."]),
        dataset_source("follower_rmse", "Follower RMSE comparison rows", ["RMSE is calculated in OG URDF local radians against independently matched follower archive samples."]),
        dataset_source("follower_detail", "Follower fit detail rows", ["Archive RMSE reduction percent equals (OG RMSE - candidate RMSE) / OG RMSE × 100."]),
        dataset_source("calibration_21", "Complete 21-joint archive rows", ["Observed q range and stable raw sample count come from each archived manual-registration summary."]),
        {"id": "hardware_summary", "label": "15-pose hardware validation summary", "path": relative(hardware_path)},
        {"id": "static_validation", "label": "Candidate static and numeric validation", "path": relative(static_path)},
        {"id": "og_visual", "label": "OG-mimic same-view visual metrics", "path": relative(og_metrics_path)},
        {"id": "candidate_visual", "label": "Candidate-mimic same-view visual metrics", "path": relative(candidate_metrics_path)},
        {"id": "localized_ablation", "label": "Follower mimic localized change-band analysis", "path": relative(local_metrics_path)},
        {"id": "calibration_provenance", "label": "21-joint calibration provenance", "path": relative(provenance_path)},
        {"id": "camera_intrinsics", "label": "Current D435 color intrinsics", "path": relative(intrinsics_path)},
        {"id": "camera_extrinsics", "label": "User-approved Isaac camera pose", "path": relative(camera_path)},
        {"id": "return_summary", "label": "Safe return-to-initial-pose summary", "path": relative(return_path)},
        {"id": "visual_ablation", "label": "Real / candidate mimic / OG mimic A/B panels", "path": relative(RECORD / "comparisons/mimic_affine_vs_og_15poses/manifest.json")},
    ]
    source_refs = [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources]

    title = "G20 sim→SDK→真机 21关节映射验证（2026-08-05）"
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Technical validation of the isolated raw-to-OG-URDF-q candidate mapping and follower mimic ablation.",
        "generatedAt": generated,
        "cards": [
            {"id": "poses_card", "dataset": "headline", "sourceId": "headline_rows", "description": "Three diagnostics plus twelve deterministic random poses.", "metrics": [{"label": "Hardware poses", "field": "pose_count", "format": "number"}]},
            {"id": "p95_card", "dataset": "headline", "sourceId": "headline_rows", "description": "P95 across 240 active-joint settled readbacks.", "metrics": [{"label": "P95 |q error|, rad", "field": "p95_abs_error_rad", "format": "number"}]},
            {"id": "max_card", "dataset": "headline", "sourceId": "headline_rows", "description": "Worst active-joint settled error across the full pose set.", "metrics": [{"label": "Max |q error|, rad", "field": "max_abs_error_rad", "format": "number"}]},
            {"id": "fault_card", "dataset": "headline", "sourceId": "headline_rows", "description": "Every post-pose 20-slot fault vector was zero.", "metrics": [{"label": "Fault-free poses", "field": "fault_free_pose_count", "format": "number"}]},
            {"id": "temp_card", "dataset": "headline", "sourceId": "headline_rows", "description": "Maximum motor temperature observed after a pose.", "metrics": [{"label": "Peak temperature, °C", "field": "max_temperature_c", "format": "number"}]},
        ],
        "charts": [
            {
                "id": "joint_error_chart",
                "title": "Maximum settled readback error by active joint",
                "subtitle": "15 poses per joint; absolute error in OG URDF local radians.",
                "type": "bar",
                "dataset": "joint_errors",
                "sourceId": "joint_errors_rows",
                "encodings": {
                    "x": {"field": "joint", "type": "nominal", "label": "Active joint"},
                    "y": {"field": "max_abs_error_rad", "type": "quantitative", "label": "Max absolute error (rad)", "format": "number"},
                },
            },
            {
                "id": "pose_error_chart",
                "title": "Worst settled readback error by validation pose",
                "subtitle": "Three diagnostics and twelve deterministic random poses; maximum across 16 active joints.",
                "type": "bar",
                "dataset": "pose_metrics",
                "sourceId": "pose_metrics_rows",
                "encodings": {
                    "x": {"field": "pose", "type": "nominal", "label": "Pose"},
                    "y": {"field": "worst_abs_joint_error_rad", "type": "quantitative", "label": "Worst absolute error (rad)", "format": "number"},
                },
            },
            {
                "id": "follower_rmse_chart",
                "title": "Follower calibration RMSE against archived visual matches",
                "subtitle": "Five mechanical followers; lower is better, measured in OG URDF local radians.",
                "type": "bar",
                "dataset": "follower_rmse",
                "sourceId": "follower_rmse_rows",
                "encodings": {
                    "x": {"field": "joint", "type": "nominal", "label": "Follower joint"},
                    "y": {"field": "rmse_rad", "type": "quantitative", "label": "RMSE (rad)", "format": "number"},
                    "color": {"field": "variant", "type": "nominal", "label": "Mimic model"},
                },
            },
            {
                "id": "mimic_contour_chart",
                "title": "Same-view silhouette distance by pose and mimic model",
                "subtitle": "Indicative whole-hand contour metric; real and URDF meshes differ, so small deltas are not metrology.",
                "type": "bar",
                "dataset": "contour_comparison",
                "sourceId": "contour_comparison_rows",
                "encodings": {
                    "x": {"field": "pose", "type": "nominal", "label": "Pose"},
                    "y": {"field": "contour_mean_px", "type": "quantitative", "label": "Symmetric contour distance (px)", "format": "number"},
                    "color": {"field": "variant", "type": "nominal", "label": "Mimic model"},
                },
            },
        ],
        "tables": [
            {
                "id": "joint_error_table",
                "title": "Active-joint readback detail",
                "subtitle": "Mean, P95, and maximum absolute error across 15 settled poses.",
                "dataset": "joint_errors",
                "sourceId": "joint_errors_rows",
                "defaultSort": {"field": "max_abs_error_rad", "direction": "desc"},
                "columns": [
                    {"field": "joint", "label": "Joint", "type": "text"},
                    {"field": "slot", "label": "SDK slot", "format": "number"},
                    {"field": "direction", "label": "q vs raw", "type": "text"},
                    {"field": "mean_abs_error_rad", "label": "Mean |error| rad", "format": "number"},
                    {"field": "p95_abs_error_rad", "label": "P95 |error| rad", "format": "number"},
                    {"field": "max_abs_error_rad", "label": "Max |error| rad", "format": "number"},
                ],
            },
            {
                "id": "pose_table",
                "title": "Per-pose hardware and visual detail",
                "subtitle": "All 15 poses, including fault, temperature, readback, and OG/candidate silhouette comparisons.",
                "dataset": "pose_metrics",
                "sourceId": "pose_metrics_rows",
                "defaultSort": {"field": "worst_abs_joint_error_rad", "direction": "desc"},
                "columns": [
                    {"field": "pose", "label": "Pose", "type": "text"},
                    {"field": "pose_type", "label": "Type", "type": "text"},
                    {"field": "worst_abs_joint_error_rad", "label": "Worst |error| rad", "format": "number"},
                    {"field": "fault_free", "label": "Fault-free", "type": "text"},
                    {"field": "max_temperature_c", "label": "Max °C", "format": "number"},
                    {"field": "og_contour_mean_px", "label": "OG contour px", "format": "number"},
                    {"field": "candidate_contour_mean_px", "label": "Candidate contour px", "format": "number"},
                    {"field": "candidate_minus_og_contour_px", "label": "Candidate − OG px", "format": "number", "movement": True},
                ],
            },
            {
                "id": "follower_table",
                "title": "Follower mimic evidence and decision",
                "subtitle": "Archive fit improves, but independent 15-pose same-view regression does not show stable superiority.",
                "dataset": "follower_detail",
                "sourceId": "follower_detail_rows",
                "defaultSort": {"field": "archive_rmse_reduction_percent", "direction": "desc"},
                "columns": [
                    {"field": "joint", "label": "Follower", "type": "text"},
                    {"field": "source_joint", "label": "Parent", "type": "text"},
                    {"field": "candidate_multiplier", "label": "Candidate multiplier", "format": "number"},
                    {"field": "candidate_offset_rad", "label": "Offset rad", "format": "number"},
                    {"field": "candidate_rmse_rad", "label": "Candidate RMSE", "format": "number"},
                    {"field": "og_rmse_rad", "label": "OG RMSE", "format": "number"},
                    {"field": "archive_rmse_reduction_percent", "label": "Archive reduction %", "format": "number"},
                    {"field": "independent_visual_decision", "label": "Independent decision", "type": "text"},
                ],
            },
            {
                "id": "calibration_table",
                "title": "Complete 21-joint calibration archive",
                "subtitle": "Sixteen active SDK joints plus five mechanical followers; every archive passed manual review as candidate-only.",
                "dataset": "calibration_21",
                "sourceId": "calibration_21_rows",
                "defaultSort": {"field": "joint", "direction": "asc"},
                "columns": [
                    {"field": "joint", "label": "Joint", "type": "text"},
                    {"field": "runtime_role", "label": "Runtime role", "type": "text"},
                    {"field": "source_joint", "label": "Follower parent", "type": "text"},
                    {"field": "unique_stable_raw_count", "label": "Stable raw samples", "format": "number"},
                    {"field": "observed_q_low_rad", "label": "Observed q low", "format": "number"},
                    {"field": "observed_q_high_rad", "label": "Observed q high", "format": "number"},
                    {"field": "outside_og_limit_count", "label": "Outside OG limit", "format": "number"},
                    {"field": "archive_status", "label": "Archive status", "type": "text"},
                ],
            },
        ],
        "sources": source_refs,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {"id": "technical_summary", "type": "markdown", "body": "## Technical summary\n\n**The isolated 16-active-joint raw→OG-URDF-q mapping passes static validation and the approved 15-pose sim→SDK→real still-image regression.** Across 240 settled joint observations, P95 absolute readback error is **0.0110 rad**, the maximum is **0.03295 rad**, all 15 poses are fault-free, and peak temperature is **52 °C**.\n\n**The full 21-joint package is not yet cleared for production or training-asset promotion.** The five mechanical follower fits improve their original per-joint archive RMSE, but their independent same-view ablation is mixed: candidate is better in 8/15 poses and the localized mean is 0.6% worse than OG, which is noise-level rather than stable superiority. Keep the follower fits as provenance only; keep the production OG URDF and default runtime mapping unchanged."},
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["poses_card", "p95_card", "max_card", "fault_card", "temp_card"]},
            {"id": "active_result", "type": "markdown", "sourceId": "hardware_summary", "body": "## Sixteen active SDK joints pass the hardware gate\n\nThe candidate LUT was exercised through the real SDK on three diagnostic and twelve deterministic random poses. Every command used a rate-limited ramp and settled readback; all 20-slot fault vectors remained zero. The largest single error occurred at thumb CMC roll in `random_05` (0.03295 rad, three raw counts), while the remaining distribution stays compact."},
            {"id": "joint_error_chart_block", "type": "chart", "chartId": "joint_error_chart", "layout": "full"},
            {"id": "joint_error_table_block", "type": "table", "tableId": "joint_error_table", "layout": "full"},
            {"id": "pose_result", "type": "markdown", "sourceId": "hardware_summary", "body": "## All fifteen sampled poses completed safely\n\nThe pose set deliberately mixes isolated diagnostics with broad random configurations. The chart shows the worst of 16 active-joint errors per pose; the table preserves exact fault, temperature, and visual comparison fields. After the run, the hand returned to its initial candidate pose with no fault and a worst residual of 0.020 rad."},
            {"id": "pose_error_chart_block", "type": "chart", "chartId": "pose_error_chart", "layout": "full"},
            {"id": "pose_table_block", "type": "table", "tableId": "pose_table", "layout": "full"},
            {"id": "visual_result", "type": "markdown", "body": "## Same-view A/B confirms the active-chain direction and amplitude\n\nThe user-approved Isaac camera uses the current D435 intrinsics and a manually aligned extrinsic pose. Direct inspection of all 15 real/Isaac/overlay panels found stable palm registration, matching four-finger motion direction and broadly matching amplitude, with no thumb self-collision drift or chain flip. The representative panels below compare real hardware, candidate follower mimic, OG follower mimic, and the two simulated contours together."},
            {"id": "visual_image", "type": "html", "sourceId": "visual_ablation", "body": f"<figure><img src=\"data:image/jpeg;base64,{encoded_visual}\" alt=\"Three representative real, candidate mimic, OG mimic, and overlay comparisons\" style=\"width:100%;height:auto;display:block\"><figcaption>Representative same-view ROI comparisons. In overlay panels, green is candidate mimic and magenta is OG mimic. Full 15-pose sheets remain in the validation record.</figcaption></figure>"},
            {"id": "follower_result", "type": "markdown", "body": "## Five follower fits remain candidate-only\n\nThe archive-matched affine fits reduce RMSE versus OG by 24.4%–64.4%, which confirms that the manual local-q registrations contain signal. However, the independent whole-hand still-image regression does not rank the candidate consistently: overall contour mean changes from 19.497 px (OG) to 19.581 px (candidate), and the follower-focused change band is only 8/15 better with a 0.058 px mean disadvantage. These differences are below the reliability of the mesh/depth silhouette method, so the correct decision is **inconclusive—not promoted**, not “candidate is worse.”"},
            {"id": "follower_rmse_chart_block", "type": "chart", "chartId": "follower_rmse_chart", "layout": "full"},
            {"id": "mimic_contour_chart_block", "type": "chart", "chartId": "mimic_contour_chart", "layout": "full"},
            {"id": "follower_table_block", "type": "table", "tableId": "follower_table", "layout": "full"},
            {"id": "scope_definitions", "type": "markdown", "body": f"## Scope, data, and metric definitions\n\n- **Coordinate contract:** stable physical SDK raw `[0,255]` → `linker_hand_l20_OG` URDF local revolute q in radians.\n- **Active population:** 16 SDK-addressable joints × 15 poses = 240 settled observations.\n- **Followers:** index/middle/ring/pinky DIP and thumb IP are mechanical mimic joints without SDK slots.\n- **Camera:** RealSense D435 serial `{intrinsics['serial']}`, 1280×720, exposure 300, gain 32, white balance 4600; Isaac uses the approved same-view pose and current color intrinsics.\n- **Readback error:** absolute difference between target candidate q and stable SDK raw mapped back through the same candidate LUT.\n- **Visual metric:** symmetric depth-mask/Isaac-contour distance in a fixed hand ROI; it is diagnostic, not calibrated metrology, because real and URDF meshes differ."},
            {"id": "methodology", "type": "markdown", "body": "## Methodology\n\n1. Build an isolated 16-joint LUT overlay from the 21 manually reviewed raw↔OG-local-q archives; retain five follower fits only in provenance.\n2. Validate schema, slot uniqueness, monotonicity in both q directions, exact integer raw round-trip, quantized q round-trip, and restoration of the default mapping.\n3. Align the Isaac camera to the current D435 view; fix exposure at 300.\n4. Command 15 bounded poses through the SDK using rate-limited ramps, record stable raw/fault/temperature values and synchronized color/depth stills, then return safely to the initial pose.\n5. Render settled readback q in Isaac with physics disabled for kinematic comparison, first with OG mimic and then an isolated candidate-mimic URDF.\n6. Compose full-image and ROI A/B panels, inspect them directly, compute whole-hand silhouette diagnostics, then repeat in the pixel band changed by the two mimic models."},
            {"id": "calibration_archive", "type": "markdown", "sourceId": "calibration_provenance", "body": "## The complete 21-joint archive is intact\n\nAll 21 joint archives remain marked `visual_review_pass_candidate_only_not_promoted`. The table below exposes sample counts, observed local-q ranges, old-limit excursions, and active/follower role without implying that old URDF limits should automatically expand."},
            {"id": "calibration_table_block", "type": "table", "tableId": "calibration_table", "layout": "full"},
            {"id": "limitations", "type": "markdown", "body": "## Limitations, uncertainty, and robustness checks\n\n- The approved still-image run validates direction, amplitude, gross coupling, and the absence of visible thumb instability; it is not a force/contact or dynamic-latency test.\n- No synchronized continuous-motion A/B video was recorded because the repository has no pre-existing reviewed control-and-record path for this hand. This omission does not invalidate the static LUT result, but it remains a production gate for dynamic behavior.\n- The current camera is D435 serial 814412070035, not the earlier calibration camera. Intrinsics and exposure were re-read and the extrinsic view was manually re-gated.\n- Silhouette scores are sensitive to real-vs-URDF mesh differences, depth segmentation, and residual camera translation; sub-pixel and small single-pixel differences must not be overinterpreted.\n- Candidate follower fits are supported by per-joint archive matching but not independently confirmed as superior in the whole-hand regression.\n- The production OG URDF SHA256 remains `697fe08490c957e4c9fa595ac0750cc80512b1f28dd3f53382db0d24ed202b2f`; default runtime mapping was restored after validation."},
            {"id": "next_steps", "type": "markdown", "body": "## Recommended next steps\n\n1. **Keep the 16-active-joint overlay isolated** and use it for further validation; do not switch the default mapping yet.\n2. **Do not promote the candidate follower mimic values.** Keep OG mimic for production until a follower-specific side-view or marker-based independent test resolves the mixed result.\n3. **Add one reviewed synchronized continuous trajectory** with bounded flex/extend and thumb opposition, then compare Isaac and D435 videos for direction, hysteresis, lag, coupling, and self-collision.\n4. If the dynamic gate passes, promote the active LUT with an explicit digest, rollback path, and a training-asset decision separate from any URDF limit expansion.\n5. Treat URDF limits, follower mimic parameters, and runtime raw↔q mapping as three independent promotion decisions."},
            {"id": "further_questions", "type": "markdown", "body": "## Further questions\n\n- Can a follower-specific side camera or small fiducial markers distinguish the candidate affine fit from OG mimic beyond silhouette noise?\n- Does continuous motion reveal hysteresis or load-dependent follower behavior that static free-air poses cannot?\n- Should training preserve OG geometric limits while using the active LUT only for deployment conversion, or should any limit changes be validated as a separate asset experiment?"},
        ],
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
        "package_info": {
            "report_status": "active_mapping_pass_followers_inconclusive_not_promoted",
            "camera_serial": intrinsics["serial"],
            "candidate_overlay_sha256": static["candidate_sha256"],
            "production_og_urdf_sha256": static["production_og_urdf_sha256"],
        },
    }
    artifact_path = args.out_dir / "artifact.json"
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    chart_map = """# Chart map\n\n| Section | Analytical question | Family | Fields | Supported claim |\n|---|---|---|---|---|\n| Active joints | Which active joint has the largest settled error? | comparison / bar | joint, max_abs_error_rad | Active readback remains compact; thumb roll is worst |\n| Pose safety | Which pose produces the largest active-joint residual? | comparison / bar | pose, worst_abs_joint_error_rad | All 15 poses complete; random_05 is worst |\n| Follower archive | Do candidate fits match archived local-q samples better than OG? | grouped comparison / bar | joint, variant, rmse_rad | Candidate archive RMSE is lower for all five followers |\n| Independent visual | Does candidate mimic beat OG in the same-view 15-pose regression? | grouped comparison / bar | pose, variant, contour_mean_px | Differences are mixed and noise-level |\n\nPalette policy: hard two-root cap for candidate-vs-OG comparisons; single-root preferred for active-joint and pose bars. Native artifact rendering owns final colors; series labels and exact tables provide non-color distinction.\n"""
    (args.out_dir / "chart_map.md").write_text(chart_map, encoding="utf-8")
    print(json.dumps({"artifact": str(artifact_path), "key_visual": str(key_visual_path), "datasets": {k: len(v) for k, v in artifact["snapshot"]["datasets"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
