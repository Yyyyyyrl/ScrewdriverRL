#!/usr/bin/env python3
"""Visualize G20 index calibration results against the original affine map.

Offline-only: reads recorded JSON artifacts and writes PNG/SVG figures.  It
does not import the LinkerHand SDK, open CAN, or command hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


BLUE = "#2F6BFF"
ORANGE = "#E07A2D"
GOLD = "#C89B3C"
INK = "#222832"
MID = "#697386"
GRID = "#D9DEE7"
PALE_BLUE = "#DDE8FF"
PALE_ORANGE = "#FBE5D5"


def _label_bars(axis, bars, fmt: str, *, offset: float = 3.0) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            fmt.format(value),
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thin-json", type=Path, required=True)
    parser.add_argument("--wide-json", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()

    thin = json.loads(args.thin_json.read_text(encoding="utf-8"))
    wide = json.loads(args.wide_json.read_text(encoding="utf-8"))

    cjk_font_path = Path(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
    )
    if not cjk_font_path.is_file():
        raise FileNotFoundError(f"required CJK font not found: {cjk_font_path}")
    font_manager.fontManager.addfont(str(cjk_font_path))
    cjk_font_name = font_manager.FontProperties(
        fname=str(cjk_font_path)
    ).get_name()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [cjk_font_name, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": MID,
            "axes.labelcolor": INK,
            "xtick.color": MID,
            "ytick.color": MID,
            "text.color": INK,
            "svg.fonttype": "path",
        }
    )

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    figure.patch.set_facecolor("#FFFFFF")
    figure.subplots_adjust(
        left=0.075,
        right=0.975,
        top=0.865,
        bottom=0.105,
        wspace=0.24,
        hspace=0.34,
    )

    # A. Full physical curve against the original affine interpretation.
    axis = axes[0, 0]
    old_raw = np.linspace(0.0, 255.0, 256)
    old_q = (255.0 - old_raw) / 255.0 * 1.08
    midpoint = wide["directional_piecewise_lut"]["midpoint"]
    anchors = thin["anchors"]
    thin_raw = np.asarray([row["command_raw"] for row in anchors], dtype=float)
    thin_q = np.asarray([row["pip_rad"] for row in anchors], dtype=float)

    axis.plot(
        old_raw,
        old_q,
        color=ORANGE,
        linewidth=2.3,
        linestyle="--",
        label="原 affine：0..255 → 0..1.08 rad",
    )
    axis.plot(
        midpoint["raw"],
        midpoint["pip_rad"],
        color=BLUE,
        linewidth=2.7,
        marker="o",
        markersize=4.5,
        label="完整扫程实测中点曲线",
    )
    axis.scatter(
        thin_raw,
        thin_q,
        s=68,
        color=GOLD,
        edgecolor=INK,
        linewidth=0.8,
        zorder=5,
        label="细胶带验证锚点",
    )
    axis.axhline(1.08, color=MID, linewidth=1.0, linestyle=":")
    axis.annotate(
        "Isaac 上限 1.08 rad\n原映射 raw=0；实测约 raw=101",
        xy=(101, 1.08),
        xytext=(82, 1.37),
        arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.0},
        fontsize=9.5,
        ha="center",
    )
    axis.set_xlim(262, -7)
    axis.set_ylim(-0.03, 1.68)
    axis.set_xlabel("SDK raw（255=伸直；向右越小越弯）")
    axis.set_ylabel("PIP 物理弯曲角（rad）")
    axis.set_title("A  PIP：物理曲线与原 affine 映射", loc="left", fontweight="bold")
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    axis.legend(loc="upper left", frameon=False, fontsize=9)

    # B. Command raw for representative Isaac targets.
    axis = axes[0, 1]
    mappings = thin["sim_target_mapping"]
    labels = [f"{row['sim_pip_rad']:.3f} rad" for row in mappings]
    calibrated = np.asarray(
        [row["candidate_command_raw"] for row in mappings], dtype=float
    )
    original = np.asarray(
        [row["current_affine_command_raw"] for row in mappings], dtype=float
    )
    x = np.arange(len(labels))
    width = 0.34
    original_bars = axis.bar(
        x - width / 2,
        original,
        width,
        color=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.5,
        label="原 affine command",
    )
    calibrated_bars = axis.bar(
        x + width / 2,
        calibrated,
        width,
        color=PALE_BLUE,
        edgecolor=BLUE,
        linewidth=1.5,
        label="标定后候选 command",
    )
    _label_bars(axis, original_bars, "{:.0f}")
    _label_bars(axis, calibrated_bars, "{:.0f}")
    for index, delta in enumerate(calibrated - original):
        axis.text(
            index,
            max(original[index], calibrated[index]) + 27,
            f"差 {delta:+.0f} raw",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color=INK,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 255)
    axis.set_ylabel("SDK command raw")
    axis.set_title("B  同一 Isaac 目标的新旧 command", loc="left", fontweight="bold")
    axis.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    axis.legend(loc="upper left", frameon=False, fontsize=9)

    # C. Readback error at the measured non-zero anchors.
    axis = axes[1, 0]
    nonzero = [row for row in anchors if row["pip_rad"] > 0.0]
    error = np.asarray(
        [row["current_affine_readback_error_rad"] for row in nonzero],
        dtype=float,
    )
    readback_labels = [
        f"raw {row['stable_readback_raw']:.0f}\n物理 {row['pip_rad']:.3f} rad"
        for row in nonzero
    ]
    y = np.arange(len(nonzero))
    bars = axis.barh(
        y,
        error,
        height=0.56,
        color=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.5,
    )
    axis.axvline(0.0, color=INK, linewidth=1.1)
    for bar, value in zip(bars, error):
        axis.text(
            value - 0.012,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.3f} rad",
            ha="right",
            va="center",
            fontsize=10,
            color=INK,
        )
    axis.set_yticks(y, readback_labels)
    axis.invert_yaxis()
    axis.set_xlim(-0.5, 0.03)
    axis.set_xlabel("原软件回读 − 物理角（rad）")
    axis.set_title("C  原回读系统性低估 PIP 物理角", loc="left", fontweight="bold")
    axis.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.75)

    # D. Physical DIP versus the current Isaac mimic.
    axis = axes[1, 1]
    physical_dip = np.asarray([row["dip_rad"] for row in nonzero], dtype=float)
    mimic_dip = np.asarray(
        [row["isaac_mimic_dip_rad"] for row in nonzero], dtype=float
    )
    pip_labels = [f"PIP {row['pip_rad']:.3f}" for row in nonzero]
    x = np.arange(len(nonzero))
    actual_bars = axis.bar(
        x - width / 2,
        physical_dip,
        width,
        color=PALE_BLUE,
        edgecolor=BLUE,
        linewidth=1.5,
        label="实测 DIP",
    )
    mimic_bars = axis.bar(
        x + width / 2,
        mimic_dip,
        width,
        color=PALE_ORANGE,
        edgecolor=ORANGE,
        linewidth=1.5,
        label="Isaac mimic（0.8917×PIP）",
    )
    _label_bars(axis, actual_bars, "{:.3f}", offset=3)
    _label_bars(axis, mimic_bars, "{:.3f}", offset=3)
    for index, delta in enumerate(mimic_dip - physical_dip):
        axis.text(
            index,
            max(physical_dip[index], mimic_dip[index]) + 0.11,
            f"Isaac 多弯 {delta:.3f} rad",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    axis.set_xticks(x, pip_labels)
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("DIP 弯曲角（rad）")
    axis.set_title("D  DIP：实测耦合与 Isaac mimic", loc="left", fontweight="bold")
    axis.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.75)
    axis.legend(loc="upper left", frameon=False, fontsize=9)

    figure.suptitle(
        "G20 食指标定结果 vs 原映射误差",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.075,
        0.915,
        "PIP command/readback 与物理 rad 对齐；DIP 为机械耦合实测。候选标定尚未写入生产配置。",
        ha="left",
        fontsize=11,
        color=MID,
    )
    figure.text(
        0.075,
        0.035,
        "来源：细胶带验证 candidate JSON + 宽胶带完整扫程 LUT。"
        " 误差定义见各面板；raw 越小代表越弯。",
        ha="left",
        fontsize=9,
        color=MID,
    )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = args.out_prefix.with_suffix(".png")
    svg = args.out_prefix.with_suffix(".svg")
    figure.savefig(png, dpi=180, facecolor=figure.get_facecolor())
    figure.savefig(svg, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(
        json.dumps(
            {
                "png": str(png.resolve()),
                "svg": str(svg.resolve()),
                "thin_json": str(args.thin_json.resolve()),
                "wide_json": str(args.wide_json.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
