#!/usr/bin/env python3
"""Plot archived visual-angle measurements against reconstructed URDF local q."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_json", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-pdf", type=Path)
    args = parser.parse_args()

    fit = json.loads(args.fit_json.read_text())
    rows = sorted(
        (row for row in fit["rows"] if row["direction"] == "forward"),
        key=lambda row: row["stable_readback_raw"],
        reverse=True,
    )
    raw = np.asarray([row["stable_readback_raw"] for row in rows], dtype=float)
    q_urdf = np.asarray([row["q_urdf_local_rad"] for row in rows], dtype=float)
    q_old = np.asarray([row["old_projected_rad"] for row in rows], dtype=float)
    residual = q_old - q_urdf
    forward_rmse = float(np.sqrt(np.mean(residual**2)))
    forward_mean = float(np.mean(residual))
    max_i = int(np.argmax(np.abs(residual)))
    full = fit["old_projected_comparison"]

    blue, orange, charcoal, grid = "#1768AC", "#D97706", "#30343B", "#D9DEE5"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, (ax_top, ax_res) = plt.subplots(
        2,
        1,
        figsize=(10.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.10},
    )
    fig.suptitle(
        "Index MCP pitch — archived blue-marker measurement vs URDF local q",
        x=0.09,
        y=0.975,
        ha="left",
        fontweight="bold",
        fontsize=15,
    )
    fig.text(
        0.09,
        0.935,
        "Same archived RGB-D sequence; camera/view held fixed; flexion increases left → right",
        color=charcoal,
        fontsize=9,
    )

    ax_top.plot(
        raw,
        q_old,
        color=blue,
        linewidth=2.0,
        marker="o",
        markersize=4.6,
        label="Archived projected visual angle",
    )
    ax_top.plot(
        raw,
        q_urdf,
        color=orange,
        linewidth=2.0,
        linestyle="--",
        marker="s",
        markersize=4.2,
        label="Reconstructed URDF local q",
    )
    ax_top.set_ylabel("Joint angle (rad)")
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.grid(axis="y", color=grid, linewidth=0.8)
    ax_top.spines[["top", "right"]].set_visible(False)

    ax_res.axhline(0.0, color=charcoal, linewidth=1.0)
    ax_res.plot(
        raw,
        residual,
        color=blue,
        linewidth=1.7,
        marker="o",
        markersize=4.4,
    )
    ax_res.scatter(
        raw[max_i],
        residual[max_i],
        s=60,
        facecolor=orange,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax_res.annotate(
        f"max |Δ| {abs(residual[max_i]):.4f} rad "
        f"({np.degrees(abs(residual[max_i])):.2f}°)",
        (raw[max_i], residual[max_i]),
        xytext=(-190, 18),
        textcoords="offset points",
        color=charcoal,
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": charcoal, "lw": 0.8},
    )
    ax_res.set_ylabel("Residual (rad)")
    ax_res.set_xlabel("Stable SDK raw readback (255 → 0)")
    ax_res.grid(axis="y", color=grid, linewidth=0.8)
    ax_res.spines[["top", "right"]].set_visible(False)
    ax_res.set_xlim(255, 0)

    fig.text(
        0.09,
        0.022,
        (
            f"Forward sweep n={len(rows)}: mean Δ={forward_mean:+.4f} rad; "
            f"RMSE={forward_rmse:.4f} rad ({np.degrees(forward_rmse):.2f}°).  "
            f"All comparable observations n={full['count']}: "
            f"RMSE={full['rmse_rad']:.4f} rad; max |Δ|={full['max_abs_rad']:.4f} rad.\n"
            "Source: archived blue-marker RGB-D sweep. "
            "Diagnostic comparison only — not yet a deployment LUT."
        ),
        color=charcoal,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.09, right=0.97, top=0.89, bottom=0.16)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=180, facecolor="white")
    if args.output_pdf:
        fig.savefig(args.output_pdf, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
