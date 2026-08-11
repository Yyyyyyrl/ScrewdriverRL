#!/usr/bin/env python3
"""Build 16:9 G20 calibration charts for an informal presentation.

Charts 01-03 intentionally use deterministic, exaggerated illustrative data and
carry a visible NOT MEASURED label. Chart 04 uses verified repository metrics.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables/g20_calibration_presentation_charts_20260809"
PROVENANCE = (
    ROOT
    / "assets/calibrations/linker_g20_left_lht20_010_415_og_local_q_visual_candidate_20260805_provenance.json"
)
STATIC = ROOT / "records/g20_og_local_q_candidate_validation_20260805/static_validation.json"

INK = "#17212B"
MUTED = "#64748B"
GRID = "#D9E0E8"
PAPER = "#F8FAFC"
ORANGE = "#E36A2E"
ORANGE_LIGHT = "#F6C4A9"
BLUE = "#2563A6"
BLUE_LIGHT = "#AFCBE8"
GOLD = "#C79224"
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

FOLLOWERS = {"index_dip", "middle_dip", "ring_dip", "pinky_dip", "thumb_ip"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family(name: str) -> str:
    if name.endswith("mcp_roll"):
        return "MCP roll"
    if name.endswith("mcp_pitch"):
        return "MCP pitch"
    if name.endswith("_pip") or name.endswith("_dip"):
        return "PIP / DIP"
    if name == "thumb_cmc_yaw":
        return "Thumb yaw"
    if name == "thumb_cmc_roll":
        return "Thumb roll"
    if name == "thumb_cmc_pitch":
        return "Thumb pitch"
    return "Thumb distal"


def short_label(name: str) -> str:
    return (
        name.replace("middle", "mid")
        .replace("thumb_cmc", "thumb")
        .replace("_mcp_", " mcp-")
        .replace("_", " ")
    )


def build_rows() -> list[dict]:
    provenance = read_json(PROVENANCE)
    rng = np.random.default_rng(20260809)
    offsets = {
        "MCP roll": 2.5,
        "MCP pitch": 0.5,
        "PIP / DIP": 1.8,
        "Thumb yaw": 6.2,
        "Thumb roll": 8.0,
        "Thumb pitch": 4.4,
        "Thumb distal": 3.4,
    }
    rows: list[dict] = []
    for name, source in provenance["joint_archives"].items():
        group = family(name)
        rmse_deg = math.degrees(float(source["affine_diagnostic_rmse_rad"]))
        outside = int(source["unique_raw_count_outside_production_limit"])
        before = 4.8 + 2.15 * rmse_deg + 0.34 * outside + offsets[group]
        before += float(rng.uniform(-0.75, 0.75))
        before = float(np.clip(before, 5.0, 19.5))
        after = 0.25 + 0.105 * rmse_deg + 0.014 * outside
        if group.startswith("Thumb"):
            after += 0.20
        after += float(rng.uniform(0.03, 0.17))
        after = float(np.clip(after, 0.28, 1.15))
        rows.append(
            {
                "joint": name,
                "display_joint": short_label(name),
                "family": group,
                "role": "follower" if name in FOLLOWERS else "active",
                "unique_stable_raw_count": int(source["unique_stable_raw_count"]),
                "outside_original_limit_count": outside,
                "actual_affine_diagnostic_rmse_deg": rmse_deg,
                "illustrative_before_mismatch_deg": before,
                "illustrative_after_mismatch_deg": after,
                "illustrative_improvement_percent": 100.0 * (1.0 - after / before),
            }
        )
    return rows


def apply_style() -> None:
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 24,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "svg.fonttype": "none",
        }
    )


def badge(fig: plt.Figure, text: str, color: str) -> None:
    fig.text(
        0.955,
        0.955,
        text,
        ha="right",
        va="top",
        fontsize=11,
        weight="bold",
        color="white",
        zorder=100,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": color},
    )


def footer(fig: plt.Figure, text: str) -> None:
    fig.text(0.04, 0.025, text, ha="left", va="bottom", fontsize=9.5, color=MUTED)


def new_figure() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=160)
    fig.subplots_adjust(left=0.19, right=0.955, top=0.84, bottom=0.12)
    return fig, ax


def chart_joint_dumbbell(rows: list[dict]) -> plt.Figure:
    ordered = sorted(rows, key=lambda row: row["illustrative_before_mismatch_deg"])
    labels = [row["display_joint"] for row in ordered]
    before = np.array([row["illustrative_before_mismatch_deg"] for row in ordered])
    after = np.array([row["illustrative_after_mismatch_deg"] for row in ordered])
    y = np.arange(len(ordered))
    fig, ax = new_figure()
    ax.hlines(y, after, before, color="#B8C2CC", linewidth=2.2, zorder=1)
    ax.scatter(before, y, s=88, color=ORANGE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.scatter(after, y, s=70, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 21)
    ax.set_xlabel("示意性关节空间偏差（deg）")
    ax.set_title("21 个关节：校准前后偏差（夸张演示版）", loc="left", pad=18)
    ax.text(
        0,
        1.015,
        "橙色 = 校准前；蓝色 = 校准后。视觉幅度按真实关节复杂度构造，但数值不是实测。",
        transform=ax.transAxes,
        fontsize=12.5,
        color=MUTED,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for index in np.argsort(before)[-5:]:
        ax.text(before[index] + 0.25, y[index], f"{before[index]:.1f}°", va="center", fontsize=10, color=ORANGE)
    ax.text(16.1, -1.35, "● 校准前", color=ORANGE, fontsize=11, weight="bold")
    ax.text(18.25, -1.35, "● 校准后", color=BLUE, fontsize=11, weight="bold")
    badge(fig, "夸张示意 · NOT MEASURED", ORANGE)
    footer(fig, "固定种子 20260809；仅供非正式 presentation。正式结果以 2026-08-05 calibration report 为准。")
    return fig


def synthetic_samples(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260809)
    before_samples: list[float] = []
    after_samples: list[float] = []
    for row in rows:
        before = row["illustrative_before_mismatch_deg"]
        after = row["illustrative_after_mismatch_deg"]
        before_samples.extend(np.clip(rng.normal(before, max(1.0, 0.24 * before), 20), 0.4, 24))
        after_samples.extend(np.clip(rng.normal(after, max(0.12, 0.34 * after), 20), 0.03, 2.2))
    return np.array(before_samples), np.array(after_samples)


def chart_distribution(rows: list[dict]) -> plt.Figure:
    before, after = synthetic_samples(rows)
    fig, ax = new_figure()
    bins = np.linspace(0, 24, 25)
    weights_before = np.full(before.shape, 100.0 / before.size)
    weights_after = np.full(after.shape, 100.0 / after.size)
    ax.hist(before, bins=bins, weights=weights_before, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.3, label="校准前")
    ax.hist(after, bins=bins, weights=weights_after, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.3, label="校准后")
    before_median, before_p95 = np.median(before), np.percentile(before, 95)
    after_median, after_p95 = np.median(after), np.percentile(after, 95)
    ax.axvline(before_median, color=ORANGE, linewidth=2.2, linestyle="--")
    ax.axvline(after_median, color=BLUE, linewidth=2.2, linestyle="--")
    ax.set_xlim(0, 24)
    ax.set_xlabel("示意性关节空间偏差（deg）")
    ax.set_ylabel("样本占比（%）")
    ax.set_title("偏差分布：从宽而长尾，压缩到接近零点", loc="left", pad=18)
    ax.text(
        0,
        1.015,
        "21 joints × 20 synthetic poses；强调视觉反差，不代表相机或编码器实测分布。",
        transform=ax.transAxes,
        fontsize=12.5,
        color=MUTED,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    ax.text(0.985, 0.70, f"校准前\nmedian {before_median:.1f}°\nP95 {before_p95:.1f}°", transform=ax.transAxes, ha="right", va="top", color=ORANGE, fontsize=15, weight="bold")
    ax.text(0.985, 0.48, f"校准后\nmedian {after_median:.2f}°\nP95 {after_p95:.2f}°", transform=ax.transAxes, ha="right", va="top", color=BLUE, fontsize=15, weight="bold")
    badge(fig, "夸张示意 · NOT MEASURED", ORANGE)
    footer(fig, "分布由固定种子生成；只用于讲故事，不可引用为 calibration accuracy。")
    return fig


def chart_family(rows: list[dict]) -> plt.Figure:
    names = ["MCP roll", "MCP pitch", "PIP / DIP", "Thumb yaw", "Thumb roll", "Thumb pitch", "Thumb distal"]
    before = []
    after = []
    for name in names:
        subset = [row for row in rows if row["family"] == name]
        before.append(float(np.mean([row["illustrative_before_mismatch_deg"] for row in subset])))
        after.append(float(np.mean([row["illustrative_after_mismatch_deg"] for row in subset])))
    y = np.arange(len(names))
    height = 0.34
    fig, ax = new_figure()
    ax.barh(y + height / 2, before, height, color=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=1.3, label="校准前")
    ax.barh(y - height / 2, after, height, color=BLUE, edgecolor=BLUE, linewidth=1.0, label="校准后")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 21)
    ax.set_xlabel("示意性平均偏差（deg）")
    ax.set_title("按关节家族汇总的校准改善（夸张演示版）", loc="left", pad=18)
    ax.text(0, 1.015, "拇指链被有意强化，以对应旧映射中最直观的空间轨迹错位。", transform=ax.transAxes, fontsize=12.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=12)
    for i, (left, right) in enumerate(zip(before, after)):
        improvement = 100.0 * (1.0 - right / left)
        ax.text(left + 0.25, i + height / 2, f"{left:.1f}°", va="center", fontsize=10.5, color=ORANGE)
        ax.text(max(1.2, right + 0.22), i - height / 2, f"{right:.2f}°  ({improvement:.0f}%↓)", va="center", fontsize=10.5, color=BLUE, weight="bold")
    badge(fig, "夸张示意 · NOT MEASURED", ORANGE)
    footer(fig, "聚合口径：各家族内 synthetic joint mismatch 的算术平均。")
    return fig


def chart_verified() -> plt.Figure:
    static = read_json(STATIC)
    active = static["active_joints"]
    values = [math.degrees(row["q_roundtrip_after_uint8_command"]["max_abs_q_error_rad"]) for row in active]
    labels = [short_label(row["joint"]) for row in active]
    order = np.argsort(values)
    values = np.array(values)[order]
    labels = np.array(labels)[order]

    fig = plt.figure(figsize=(12, 6.75), dpi=160, facecolor=PAPER)
    grid = fig.add_gridspec(2, 4, height_ratios=[0.9, 3.3], left=0.13, right=0.96, top=0.84, bottom=0.11, hspace=0.35, wspace=0.18)
    kpis = [
        ("21 / 21", "关节人工视觉配准完成"),
        ("0", "单调性违规"),
        ("15 / 15", "静态整手姿态通过"),
        ("0.010 rad", "最大 q→raw→q 量化误差"),
    ]
    for index, (value, label) in enumerate(kpis):
        ax = fig.add_subplot(grid[0, index])
        ax.set_facecolor("white")
        ax.text(0.06, 0.64, value, transform=ax.transAxes, fontsize=25, weight="bold", color=BLUE, va="center")
        ax.text(0.06, 0.24, label, transform=ax.transAxes, fontsize=10.5, color=MUTED, va="center")
        ax.axhline(0.97, color=BLUE, linewidth=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

    ax = fig.add_subplot(grid[1, :])
    y = np.arange(len(values))
    ax.barh(y, values, color=BLUE_LIGHT, edgecolor=BLUE, linewidth=1.1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 0.66)
    ax.set_xlabel("每关节最大 q→raw→q 误差（deg）")
    ax.set_title("16 个主动关节的量化回环误差", loc="left", pad=20, fontsize=19)
    ax.text(0, 1.01, "真实验证数据；uint8 command 量化 + piecewise-LUT 逆映射，不等同于视觉轮廓误差。", transform=ax.transAxes, fontsize=11.5, color=MUTED)
    ax.axvline(math.degrees(0.01), color=INK, linewidth=1.5, linestyle="--")
    ax.text(math.degrees(0.01) - 0.006, len(values) - 0.2, "0.01 rad gate", ha="right", va="top", fontsize=10, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, value in enumerate(values):
        ax.text(value + 0.008, i, f"{value:.2f}°", va="center", fontsize=9.5, color=BLUE)

    fig.suptitle("G20 OG-local-q 校准：已验证结果", x=0.055, y=0.96, ha="left", fontsize=27, weight="bold", color=INK)
    badge(fig, "VERIFIED REPOSITORY DATA", BLUE)
    footer(fig, "Source: provenance + static_validation.json；hardware static tracking: mean 0.00449 rad, P95 0.01101 rad, max 0.03295 rad across 240 observations.")
    return fig


def write_data(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "chart_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before, after = synthetic_samples(rows)
    summary = {
        "schema_version": 1,
        "status": "presentation_only_exaggerated_not_measured",
        "seed": 20260809,
        "source_provenance": str(PROVENANCE.relative_to(ROOT)),
        "source_static_validation": str(STATIC.relative_to(ROOT)),
        "synthetic_distribution": {
            "sample_count_per_state": int(before.size),
            "before_median_deg": float(np.median(before)),
            "before_p95_deg": float(np.percentile(before, 95)),
            "after_median_deg": float(np.median(after)),
            "after_p95_deg": float(np.percentile(after, 95)),
        },
    }
    (OUT / "chart_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    apply_style()
    rows = build_rows()
    write_data(rows)
    charts = [
        ("01_joint_mismatch_before_after", chart_joint_dumbbell(rows)),
        ("02_error_distribution", chart_distribution(rows)),
        ("03_joint_family_improvement", chart_family(rows)),
        ("04_verified_calibration_results", chart_verified()),
    ]
    with PdfPages(OUT / "g20_calibration_presentation_charts.pdf") as pdf:
        for name, fig in charts:
            fig.savefig(OUT / f"{name}.png", dpi=160, facecolor=fig.get_facecolor())
            fig.savefig(OUT / f"{name}.svg", facecolor=fig.get_facecolor())
            pdf.savefig(fig, facecolor=fig.get_facecolor())
            plt.close(fig)
    print(f"wrote {len(charts)} charts to {OUT}")


if __name__ == "__main__":
    main()
