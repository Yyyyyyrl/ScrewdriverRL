#!/usr/bin/env python3
"""Create a presentation-ready summary of the G20 visual/URDF-q audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = args.root

    active_specs = [
        ("Index MCP roll", "index_mcp_roll/archive_blue_marker_roll_2d/archive_roll_registration_2d.json", "old_vs_reconstructed", "pass"),
        ("Index MCP pitch", "index_mcp_pitch/archive_blue_marker_fit_v2_centroid_axis/archive_urdf_local_q_fit_v2.json", "old_projected_comparison", "pass"),
        ("Index PIP", "index_pip/archive_blue_marker_chain_v1/archive_chain_registration_unwrapped.json", "old_vs_reconstructed", "pass"),
        ("Middle MCP roll", "middle_mcp_roll/archive_blue_marker_roll_2d/archive_roll_registration_2d.json", "old_vs_reconstructed", "pass"),
        ("Middle MCP pitch", "middle_pitch/archive_blue_marker_chain_v1/archive_chain_registration.json", "old_vs_reconstructed", "pass"),
        ("Middle PIP", "middle_pip/archive_blue_marker_chain_v1/archive_chain_registration_unwrapped.json", "old_vs_reconstructed", "pass"),
        ("Ring MCP roll", "ring_mcp_roll/archive_blue_marker_roll_2d/archive_roll_registration_2d.json", "old_vs_reconstructed", "pass"),
        ("Ring MCP pitch", "ring_pitch/archive_blue_marker_chain_v1/archive_chain_registration.json", "old_vs_reconstructed", "pass"),
        ("Ring PIP", "ring_pip/archive_blue_marker_chain_v1/archive_chain_registration_unwrapped.json", "old_vs_reconstructed", "pass"),
        ("Pinky MCP roll", "pinky_mcp_roll/archive_blue_marker_roll_2d/archive_roll_registration_2d.json", "old_vs_reconstructed", "pass"),
        ("Pinky MCP pitch", "pinky_pitch/archive_blue_marker_chain_v1/archive_chain_registration.json", "old_vs_reconstructed", "depth_diag"),
        ("Pinky PIP", "pinky_pip/archive_blue_marker_chain_v1/archive_chain_registration_unwrapped.json", "old_vs_reconstructed", "pass"),
        ("Thumb CMC yaw", "thumb_cmc_yaw/archive_formal_palm_l_v1/visual_vs_projected_urdf_q.json", None, "pass"),
        ("Thumb CMC pitch", "thumb_cmc_pitch/archive_blue_marker_chain_v1/archive_chain_registration.json", "old_vs_reconstructed", "pass"),
        ("Thumb MCP", "thumb_mcp/archive_blue_marker_chain_v1/archive_chain_registration.json", "old_vs_reconstructed", "pass"),
    ]
    active = []
    for label, relative, metric_key, status in active_specs:
        payload = _load(root / relative)
        metric = payload if metric_key is None else payload[metric_key]
        mean_key = (
            "mean_old_minus_q_rad"
            if "mean_old_minus_q_rad" in metric
            else "mean_rad"
        )
        count = metric.get("count", len(payload.get("rows", [])))
        active.append(
            {
                "joint": label,
                "status": status,
                "count": int(count),
                "mean_rad": float(metric[mean_key]),
                "rmse_rad": float(metric["rmse_rad"]),
                "max_abs_rad": float(metric["max_abs_rad"]),
            }
        )

    mimic_specs = [
        ("Index DIP", "index_pip/archive_blue_marker_chain_v1/index_dip_mimic_comparison.json"),
        ("Middle DIP", "middle_pip/archive_blue_marker_chain_v1/middle_dip_mimic_comparison.json"),
        ("Ring DIP", "ring_pip/archive_blue_marker_chain_v1/ring_dip_mimic_comparison.json"),
        ("Pinky DIP", "pinky_pip/archive_blue_marker_chain_v1/pinky_dip_mimic_comparison.json"),
        ("Thumb IP", "thumb_mcp/archive_blue_marker_chain_v1/thumb_ip_mimic_comparison.json"),
    ]
    mimic = []
    for label, relative in mimic_specs:
        metrics = _load(root / relative)["metrics"]
        og = float(metrics["OG"]["rmse_rad"])
        calibrated = float(metrics["calibrated"]["rmse_rad"])
        mimic.append(
            {
                "joint": label,
                "count": int(metrics["calibrated"]["count"]),
                "og_rmse_rad": og,
                "calibrated_rmse_rad": calibrated,
                "reduction_fraction": 1.0 - calibrated / og,
            }
        )

    thumb_roll = _load(
        root
        / "thumb_cmc_roll/archive_formal_white_palm_v1/"
        "full_sweep_visual_azimuth_vs_fitted_axis.json"
    )
    blocker = thumb_roll["old_visual_q_vs_axis_rotation"]
    summary = {
        "schema_version": 1,
        "active_verified": 15,
        "active_total": 16,
        "passive_mimic_improved": 5,
        "passive_mimic_total": 5,
        "live_heldout_started": False,
        "active": active,
        "mimic": mimic,
        "thumb_cmc_roll_blocker": {
            "reason": "visual semantic zero is not registered to URDF local q=0",
            "formal_frame_count": int(thumb_roll["count"]),
            **blocker,
        },
    }
    (args.out_dir / "g20_urdf_local_q_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=(0.72, 3.8, 2.3),
        width_ratios=(1.35, 1.0),
        hspace=0.35,
        wspace=0.28,
    )
    fig.suptitle(
        "G20 Visual Measurement → URDF Local-q Audit",
        x=0.055,
        y=0.975,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#172033",
    )
    fig.text(
        0.055,
        0.938,
        "Archived calibration photographs, camera-matched Isaac renders, and explicit coordinate-basis checks",
        fontsize=11,
        color="#59647A",
    )

    card_ax = fig.add_subplot(grid[0, :])
    card_ax.axis("off")
    cards = [
        ("15 / 16", "active joints verified", "#1967D2", "#EAF2FF"),
        ("5 / 5", "mimic joints improved", "#0B8043", "#E8F5EE"),
        ("1 blocker", "thumb CMC roll basis", "#C5221F", "#FDECEA"),
        ("not started", "live held-out gate", "#7A4D00", "#FFF4D6"),
    ]
    for index, (value, label, color, background) in enumerate(cards):
        x = 0.01 + index * 0.247
        card = FancyBboxPatch(
            (x, 0.08),
            0.225,
            0.78,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#D7DCE5",
            facecolor=background,
            transform=card_ax.transAxes,
        )
        card_ax.add_patch(card)
        card_ax.text(
            x + 0.018,
            0.57,
            value,
            transform=card_ax.transAxes,
            fontsize=20,
            fontweight="bold",
            color=color,
        )
        card_ax.text(
            x + 0.018,
            0.25,
            label,
            transform=card_ax.transAxes,
            fontsize=10,
            color="#4F596D",
        )

    active_ax = fig.add_subplot(grid[1:, 0])
    labels = [row["joint"] for row in active][::-1]
    values_deg = [
        math.degrees(row["rmse_rad"]) for row in active
    ][::-1]
    statuses = [row["status"] for row in active][::-1]
    colors = [
        "#6A75C4" if status == "depth_diag" else "#2F6FDB"
        for status in statuses
    ]
    bars = active_ax.barh(
        np.arange(len(labels)),
        values_deg,
        color=colors,
        height=0.67,
        edgecolor="white",
    )
    for bar, value, status in zip(bars, values_deg, statuses):
        active_ax.text(
            bar.get_width() + 0.04,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.2f}°",
            va="center",
            fontsize=8.5,
            color="#2B3345",
        )
        if status == "depth_diag":
            bar.set_hatch("///")
    active_ax.set_yticks(np.arange(len(labels)), labels)
    active_ax.set_xlabel("RMSE of archived measurement vs reconstructed/projected URDF local q (degrees)")
    active_ax.set_title(
        "15 active joints with a consistent visual ↔ URDF-q basis",
        loc="left",
        fontweight="bold",
    )
    active_ax.grid(axis="x", color="#E7EAF0", linewidth=0.8)
    active_ax.set_axisbelow(True)
    active_ax.spines[["top", "right", "left"]].set_visible(False)
    active_ax.legend(
        handles=[
            Line2D([0], [0], color="#2F6FDB", lw=8, label="primary / accepted"),
            Line2D([0], [0], color="#6A75C4", lw=8, label="depth diagnostic; same-view overlay is primary"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )

    mimic_ax = fig.add_subplot(grid[1, 1])
    y = np.arange(len(mimic))
    og_deg = [math.degrees(row["og_rmse_rad"]) for row in mimic]
    current_deg = [
        math.degrees(row["calibrated_rmse_rad"]) for row in mimic
    ]
    height = 0.34
    mimic_ax.barh(
        y + height / 2,
        og_deg,
        height=height,
        color="#F2994A",
        label="OG multiplier",
    )
    mimic_ax.barh(
        y - height / 2,
        current_deg,
        height=height,
        color="#18A36B",
        label="measured multiplier",
    )
    for index, row in enumerate(mimic):
        mimic_ax.text(
            max(og_deg[index], current_deg[index]) + 0.25,
            index,
            f"−{100.0 * row['reduction_fraction']:.0f}%",
            va="center",
            fontsize=9,
            color="#0B6A45",
            fontweight="bold",
        )
    mimic_ax.set_yticks(y, [row["joint"] for row in mimic])
    mimic_ax.invert_yaxis()
    mimic_ax.set_xlabel("Passive-joint RMSE (degrees)")
    mimic_ax.set_title(
        "All 5 mimic joints improve vs OG",
        loc="left",
        fontweight="bold",
    )
    mimic_ax.grid(axis="x", color="#E7EAF0", linewidth=0.8)
    mimic_ax.set_axisbelow(True)
    mimic_ax.spines[["top", "right", "left"]].set_visible(False)
    mimic_ax.legend(frameon=False, loc="lower right", fontsize=9)

    blocker_ax = fig.add_subplot(grid[2, 1])
    blocker_ax.axis("off")
    box = FancyBboxPatch(
        (0.0, 0.02),
        1.0,
        0.95,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.2,
        edgecolor="#E7A09E",
        facecolor="#FFF4F3",
        transform=blocker_ax.transAxes,
    )
    blocker_ax.add_patch(box)
    blocker_ax.text(
        0.045,
        0.79,
        "BLOCKER · thumb CMC roll",
        transform=blocker_ax.transAxes,
        fontsize=14,
        color="#B3261E",
        fontweight="bold",
    )
    blocker_ax.text(
        0.045,
        0.58,
        "White-marker session is stable, but its semantic 0 rad was created by\n"
        "shifting raw248 to zero; it was never registered to URDF local q=0.",
        transform=blocker_ax.transAxes,
        fontsize=10,
        color="#394155",
        linespacing=1.35,
    )
    blocker_ax.text(
        0.045,
        0.30,
        f"34-frame fixed-axis audit  RMSE {math.degrees(blocker['rmse_rad']):.1f}°  ·  "
        f"max {math.degrees(blocker['max_abs_rad']):.1f}°",
        transform=blocker_ax.transAxes,
        fontsize=11,
        color="#B3261E",
        fontweight="bold",
    )
    blocker_ax.text(
        0.045,
        0.12,
        "Gate: re-register physical raw, visual angle, and URDF local q before live validation.",
        transform=blocker_ax.transAxes,
        fontsize=9.5,
        color="#5D6474",
    )

    fig.text(
        0.055,
        0.018,
        "Interpretation: residuals are method-specific archive diagnostics, not deployment LUT accuracy. "
        "Same-view silhouette checks remain primary where depth is view-axis sensitive.",
        fontsize=8.5,
        color="#687084",
    )
    png = args.out_dir / "g20_visual_urdf_local_q_audit_summary.png"
    pdf = args.out_dir / "g20_visual_urdf_local_q_audit_summary.pdf"
    fig.savefig(png, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {png}")
    print(f"wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
