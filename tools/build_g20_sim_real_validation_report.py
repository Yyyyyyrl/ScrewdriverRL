#!/usr/bin/env python3
"""Build a presentation-ready HTML/Markdown report for G20 visual validation."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path


POSE_LABELS = {
    "diag_flex_wave": "诊断：长指递增屈曲",
    "diag_roll_fan": "诊断：MCP roll 扇形",
    "diag_thumb_opposition": "诊断：拇指对掌",
    "random_00": "随机姿态 00",
    "random_01": "随机姿态 01",
    "random_02": "随机姿态 02",
    "random_03": "随机姿态 03",
    "random_04": "随机姿态 04",
}


def deg(rad: float) -> float:
    return math.degrees(rad)


def rel(report_dir: Path, target: Path) -> str:
    return "../" + str(target.relative_to(report_dir.parent))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.record_root.resolve()
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((root / "analysis/metrics.json").read_text())
    hardware = json.loads((root / "stills/real_seed20260803/summary.json").read_text())
    camera = json.loads((root / "camera/fingertip_yaw_50_up20mm.json").read_text())
    camera = camera[0] if isinstance(camera, list) else camera
    thumb_manifest = json.loads(
        (root / "data/thumb_roll_visual_zero_candidate_manifest.json").read_text()
    )

    joint_stats: dict[str, list[float]] = defaultdict(list)
    for record in hardware["poses"].values():
        for row in record["per_joint"]:
            joint_stats[row["name"]].append(abs(float(row["error_rad"])))
    joint_rows = sorted(
        [
            {
                "joint": name,
                "mean": sum(values) / len(values),
                "max": max(values),
            }
            for name, values in joint_stats.items()
        ],
        key=lambda row: row["max"],
        reverse=True,
    )

    e = metrics["joint_readback_abs_error_rad"]
    image = metrics["image_metric_scope"]
    visual_pass = False
    verdict = "NOT READY — 视觉几何门未通过"
    pose_cards = []
    visual_bars = []
    pose_table_rows = []
    max_contour = max(row["symmetric_contour_mean_px"] for row in metrics["poses"])
    for row in metrics["poses"]:
        name = row["pose"]
        ab_path = root / f"comparisons/static_seed20260803/{name}_ab.png"
        overlay_path = root / f"analysis/{name}_depth_silhouette_overlay.png"
        pose_cards.append(
            f"""
            <article class="pose-card">
              <div class="pose-head">
                <div><span class="eyebrow">{html.escape(name)}</span>
                <h3>{POSE_LABELS[name]}</h3></div>
                <span class="metric-pill">{row['symmetric_contour_mean_px']:.1f} px</span>
              </div>
              <a href="{rel(report_dir, ab_path)}"><img src="{rel(report_dir, ab_path)}"
                 alt="{name} real Isaac overlay"></a>
              <a href="{rel(report_dir, overlay_path)}"><img class="depth"
                 src="{rel(report_dir, overlay_path)}"
                 alt="{name} depth silhouette overlay"></a>
              <div class="mini-grid">
                <span>IoU <b>{row['silhouette_iou']:.3f}</b></span>
                <span>轮廓均值 <b>{row['symmetric_contour_mean_px']:.1f}px</b></span>
                <span>质心差 <b>{row['centroid_delta_px']:.1f}px</b></span>
                <span>SDK 最差 <b>{deg(row['worst_abs_joint_error_rad']):.2f}°</b></span>
              </div>
            </article>
            """
        )
        width = 100.0 * row["symmetric_contour_mean_px"] / max_contour
        visual_bars.append(
            f"""
            <div class="bar-row"><span>{html.escape(name)}</span>
              <div class="bar-track"><i style="width:{width:.1f}%"></i></div>
              <b>{row['symmetric_contour_mean_px']:.1f}px</b>
            </div>
            """
        )
        pose_table_rows.append(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{deg(row['worst_abs_joint_error_rad']):.2f}°</td>"
            f"<td>{row['silhouette_iou']:.3f}</td>"
            f"<td>{row['symmetric_contour_mean_px']:.1f}px</td>"
            f"<td>{row['centroid_delta_px']:.1f}px</td><td>FAIL</td></tr>"
        )

    joint_table_rows = []
    for row in joint_rows:
        joint_table_rows.append(
            f"<tr><td>{html.escape(row['joint'])}</td>"
            f"<td>{deg(row['mean']):.3f}°</td><td>{deg(row['max']):.3f}°</td></tr>"
        )

    camera_position = camera["pos"]
    camera_look_at = camera["look_at"]
    html_report = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G20 Sim → SDK → 真机视觉验证</title>
<style>
:root {{ --ink:#e9eef7; --muted:#9aa8bd; --panel:#111a2a; --panel2:#172237;
  --line:#2b3a53; --cyan:#3bd8e6; --orange:#ffb454; --red:#ff667a;
  --green:#56d69a; --bg:#08101d; }}
* {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; }}
body {{ margin:0; background:radial-gradient(circle at 80% -10%,#173154 0,#08101d 42%);
  color:var(--ink); font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }}
a {{ color:inherit; }} .wrap {{ width:min(1500px,94vw); margin:auto; }}
header {{ padding:76px 0 44px; border-bottom:1px solid var(--line); }}
.kicker,.eyebrow {{ color:var(--cyan); font-size:12px; font-weight:800; letter-spacing:.12em;
  text-transform:uppercase; }} h1 {{ font-size:clamp(42px,6vw,82px); letter-spacing:-.055em;
  line-height:.96; margin:12px 0 22px; max-width:1050px; }}
.lead {{ color:#c4cfde; font-size:20px; max-width:1000px; }}
.verdict {{ margin-top:28px; display:inline-flex; gap:12px; align-items:center; padding:12px 18px;
  border:1px solid #704052; border-radius:999px; background:#321723; color:#ffdbe1; font-weight:800; }}
.dot {{ width:10px; height:10px; border-radius:50%; background:var(--red); box-shadow:0 0 18px var(--red); }}
section {{ padding:58px 0; border-bottom:1px solid var(--line); }}
h2 {{ margin:0 0 10px; font-size:34px; letter-spacing:-.025em; }}
h3 {{ margin:3px 0 0; font-size:20px; }} .section-note {{ margin:0 0 28px; color:var(--muted); max-width:940px; }}
.kpis {{ display:grid; grid-template-columns:repeat(5,1fr); gap:14px; margin-top:34px; }}
.kpi,.gate,.pose-card,.callout {{ background:linear-gradient(145deg,var(--panel2),var(--panel));
  border:1px solid var(--line); border-radius:18px; }}
.kpi {{ padding:22px; }} .kpi b {{ font-size:30px; display:block; letter-spacing:-.04em; }}
.kpi span {{ color:var(--muted); font-size:13px; }} .good b {{ color:var(--green); }} .bad b {{ color:var(--red); }}
.gates {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }}
.gate {{ padding:22px; }} .gate .status {{ font-weight:900; margin-bottom:9px; }}
.pass {{ color:var(--green); }} .fail {{ color:var(--red); }} .gate p {{ color:var(--muted); margin:0; }}
.two {{ display:grid; grid-template-columns:1.1fr .9fr; gap:20px; align-items:start; }}
.callout {{ padding:26px; }} .callout.warning {{ border-color:#714252; background:#271521; }}
.callout h3 {{ margin-bottom:10px; }} .callout ul {{ padding-left:20px; margin-bottom:0; }}
.bars {{ padding:22px; background:var(--panel); border:1px solid var(--line); border-radius:18px; }}
.bar-row {{ display:grid; grid-template-columns:150px 1fr 68px; gap:12px; align-items:center; margin:12px 0; }}
.bar-track {{ height:10px; border-radius:6px; background:#26344a; overflow:hidden; }}
.bar-track i {{ height:100%; display:block; background:linear-gradient(90deg,var(--orange),var(--red)); }}
.pose-grid {{ display:grid; grid-template-columns:repeat(2,1fr); gap:20px; }}
.pose-card {{ padding:16px; overflow:hidden; }} .pose-head {{ display:flex; justify-content:space-between; align-items:center; margin:4px 4px 14px; }}
.metric-pill {{ background:#35202b; color:#ffd1d8; border:1px solid #704052; border-radius:999px; padding:5px 10px; font-weight:800; }}
.pose-card img {{ width:100%; display:block; border-radius:11px; background:#e9e9e9; }}
.pose-card img.depth {{ margin-top:10px; }}
.mini-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:12px; }}
.mini-grid span {{ background:#0c1422; border-radius:9px; padding:8px; color:var(--muted); font-size:12px; }}
.mini-grid b {{ color:var(--ink); display:block; font-size:14px; }}
table {{ width:100%; border-collapse:collapse; background:var(--panel); border-radius:14px; overflow:hidden; }}
th,td {{ padding:11px 13px; text-align:left; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.07em; }}
td:last-child {{ font-weight:800; color:var(--red); }}
.evidence {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
.evidence figure {{ margin:0; background:var(--panel); padding:14px; border:1px solid var(--line); border-radius:16px; }}
.evidence img {{ width:100%; border-radius:10px; display:block; }}
figcaption {{ color:var(--muted); font-size:13px; margin-top:10px; }}
code {{ background:#101929; border:1px solid var(--line); padding:2px 6px; border-radius:5px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; font-size:12px; color:#b6c3d5; }}
footer {{ padding:40px 0 70px; color:var(--muted); }}
@media(max-width:1000px) {{ .kpis,.gates {{ grid-template-columns:repeat(2,1fr); }}
  .two,.pose-grid,.evidence {{ grid-template-columns:1fr; }} }}
@media(max-width:620px) {{ .kpis,.gates,.mini-grid {{ grid-template-columns:1fr; }} h1 {{ font-size:44px; }} }}
</style>
</head>
<body>
<header><div class="wrap">
  <div class="kicker">Dex-Forge · Visual Validation · 2026-08-03</div>
  <h1>G20 Sim → SDK → 真机<br>视觉映射验证</h1>
  <p class="lead">相机视角由 D435 实时轮廓叠加手工锁定；8 个姿态均用真机稳态读回值在 Isaac 中重渲染。结论不是“命令有没有到”，而是“同一语义关节状态是否产生同一几何姿态”。</p>
  <div class="verdict"><span class="dot"></span>{verdict}</div>
  <div class="kpis">
    <div class="kpi good"><b>8 / 8</b><span>姿态无故障完成</span></div>
    <div class="kpi good"><b>{deg(e['mean']):.2f}°</b><span>SDK 平均绝对读回误差</span></div>
    <div class="kpi good"><b>{deg(e['p95']):.2f}°</b><span>SDK P95 读回误差</span></div>
    <div class="kpi bad"><b>{image['mean_symmetric_contour_distance_px']:.1f}px</b><span>平均对称轮廓距离</span></div>
    <div class="kpi bad"><b>0.25–0.37</b><span>深度 / Isaac 轮廓 IoU</span></div>
  </div>
</div></header>

<section><div class="wrap">
  <h2>一页结论</h2>
  <p class="section-note">控制链条跟踪准确，但物理几何与 URDF 几何没有在验证姿态上重合。现在不应进入训练部署验收。</p>
  <div class="gates">
    <div class="gate"><div class="status pass">PASS · 相机注册</div><p>最终视角由操作者在实时 D435 画面上对齐 Isaac 固定轮廓，并确认掌部与四指锚点。</p></div>
    <div class="gate"><div class="status pass">PASS · Isaac 稳定性</div><p>正式 17 组训练自碰撞过滤；8 个姿态 settle 后最大漂移 0 rad。</p></div>
    <div class="gate"><div class="status pass">PASS · SDK 跟踪与安全</div><p>128 个关节观测，最大误差 {deg(e['max']):.2f}°；无故障；最高温度 {metrics['max_temperature_c']}°C。</p></div>
    <div class="gate"><div class="status fail">FAIL · 视觉几何</div><p>所有 8 个姿态轮廓均未达到可信重合；系统性偏差不能由单帧噪声解释。</p></div>
  </div>
</div></section>

<section><div class="wrap">
  <h2>为什么判定失败</h2>
  <div class="two">
    <div class="bars">
      <span class="eyebrow">Mean symmetric contour distance</span>
      {''.join(visual_bars)}
    </div>
    <div class="callout warning">
      <h3>观察到的系统性模式</h3>
      <ul>
        <li><b>拇指：</b>在手工锚点处，CMC roll 加 +20° 后可重合；但 yaw / pitch / MCP 共同变化后，真实拇指与 Isaac 拇指重新分离，说明单一 roll 零偏修正不足以覆盖整条拇指链。</li>
        <li><b>四根长指：</b>多个姿态中真机屈曲显著深于 Isaac，优先复核 MCP pitch 与 PIP 的绝对零位和比例，而不是 SDK 到达误差。</li>
        <li><b>不是自碰撞乱飘：</b>本轮所有姿态在过滤物理中漂移为 0；此前 Claude 的 4 个姿态在过滤/不过滤各 120 step 也均为 0。</li>
        <li><b>不是相机单独造成：</b>掌座锚点在多帧中保持相对稳定，而末端随关节变化产生的误差显著增大。</li>
      </ul>
    </div>
  </div>
</div></section>

<section><div class="wrap">
  <h2>8 个姿态的 A/B 证据</h2>
  <p class="section-note">每个卡片第一张为 REAL / ISAAC（使用真机稳态读回）/ 红色 Isaac 轮廓；第二张用 D435 深度提取真实前景，绿色为真实深度轮廓、红色和蓝色为 Isaac 轮廓/区域。白框是固定指标 ROI。</p>
  <div class="pose-grid">{''.join(pose_cards)}</div>
</div></section>

<section><div class="wrap">
  <h2>量化结果</h2>
  <p class="section-note">图像指标用于比较同一固定相机下的相对重合，不等价于经过标定的三维测量；真实外壳与 URDF mesh 差异会降低绝对 IoU，但不能解释随关节姿态变化出现的方向性分离。</p>
  <table><thead><tr><th>姿态</th><th>最差 SDK 误差</th><th>轮廓 IoU</th><th>轮廓均值</th><th>质心差</th><th>视觉门</th></tr></thead>
  <tbody>{''.join(pose_table_rows)}</tbody></table>
  <h3 style="margin-top:30px">关节读回误差（8 姿态聚合）</h3>
  <table><thead><tr><th>关节</th><th>平均绝对误差</th><th>最大绝对误差</th></tr></thead>
  <tbody>{''.join(joint_table_rows)}</tbody></table>
</div></section>

<section><div class="wrap">
  <h2>已完成的拇指 roll 修正</h2>
  <div class="evidence">
    <figure><a href="{rel(report_dir, root / 'camera/live_thumb_roll_plus20_confirmation/aligned_20260803_175830_overlay.png')}">
      <img src="{rel(report_dir, root / 'camera/live_thumb_roll_plus20_confirmation/aligned_20260803_175830_overlay.png')}" alt="thumb roll plus 20 confirmation"></a>
      <figcaption>第一物理点：+20° Isaac 轮廓由操作者确认重合。</figcaption></figure>
    <figure><a href="{rel(report_dir, root / 'camera/thumb_roll_second_point_offset_sweep_ab/camera_candidate_ab_contact.png')}">
      <img src="{rel(report_dir, root / 'camera/thumb_roll_second_point_offset_sweep_ab/camera_candidate_ab_contact.png')}" alt="thumb roll second point sweep"></a>
      <figcaption>第二物理点：+15 / +20 / +25 / +30° 扫描中 +20° 仍为最佳。由此支持固定零偏，而不是 roll 斜率错误。</figcaption></figure>
  </div>
  <div class="callout" style="margin-top:20px">
    <b>候选覆盖：</b>原始 raw knot 全部保留，仅对 <code>thumb_cmc_roll</code> 的语义 rad 加
    <code>+{deg(thumb_manifest['correction']['offset_rad']):.1f}°</code>。原标定文件未覆盖；候选文件为
    <span class="mono">linker_calib_phase_c_thumb_yaw_roll_visual_zero_candidate_20260803.json</span>。
  </div>
</div></section>

<section><div class="wrap">
  <h2>建议的修复与复验顺序</h2>
  <div class="two">
    <div class="callout">
      <span class="eyebrow">Stop condition</span>
      <h3>暂缓训练部署验收</h3>
      <p>当前结果足以否定“sim → SDK → 真机几何已完全精准”。继续录连续动作只会重复静态门已经显示的偏差，并增加不必要的真机运动。</p>
    </div>
    <div class="callout">
      <span class="eyebrow">Next pass</span>
      <h3>分离校准，再跑同一验证集</h3>
      <ol>
        <li>锁住其余关节，在两个以上绝对 CAD 锚点复核 thumb CMC yaw、CMC pitch、MCP。</li>
        <li>对四指 MCP pitch 与 PIP 各取低/中/高三点，用同一 D435/Isaac 轮廓方式复核零位和比例。</li>
        <li>保持本次相机外参和 seed=20260803 的 8 姿态不变，复跑以便前后直接比较。</li>
        <li>静态视觉门通过后，再录一段同步连续 A/B 视频作为动态滞后和方向性的最终检查。</li>
      </ol>
    </div>
  </div>
</div></section>

<section><div class="wrap">
  <h2>复现信息</h2>
  <table><tbody>
    <tr><th>真机</th><td>G20 left · LHT20-010-415-L-B-1-D · CAN can0 @ 1 Mbps</td></tr>
    <tr><th>相机</th><td>D435 143322073091 · 1280×720@30 · exposure 166 · gain 32 · WB 4600</td></tr>
    <tr><th>Isaac camera position</th><td class="mono">{html.escape(str(camera_position))}</td></tr>
    <tr><th>Isaac camera look_at</th><td class="mono">{html.escape(str(camera_look_at))}</td></tr>
    <tr><th>姿态</th><td>3 个诊断姿态 + 5 个随机姿态 · seed=20260803</td></tr>
    <tr><th>渲染</th><td>真机稳态读回语义值 · filtered physics · 17 training collision filters · 120 settle steps</td></tr>
    <tr><th>图像修正</th><td>按实测 D435 fx/fy/ppx/ppy 对 Isaac 图执行确定性后投影修正</td></tr>
  </tbody></table>
</div></section>

<footer><div class="wrap">
  <b>Verdict:</b> {verdict}. 原始图像、深度、关节读回、渲染清单、CSV 和 JSON 指标均保存在本记录目录；报告不删除或覆盖 Claude 的原始记录。
</div></footer>
</body></html>"""
    (report_dir / "index.html").write_text(html_report)

    markdown = f"""# G20 Sim → SDK → 真机视觉验证

**结论：{verdict}。**

- 8/8 姿态无故障，最高温度 {metrics['max_temperature_c']}°C。
- 128 个关节观测的平均绝对读回误差 {deg(e['mean']):.2f}°，P95 {deg(e['p95']):.2f}°，最大 {deg(e['max']):.2f}°。
- Isaac 使用 17 组训练自碰撞过滤；8 个姿态 settle 后最大关节漂移 0 rad。
- D435 深度轮廓与 Isaac 轮廓：平均对称距离 {image['mean_symmetric_contour_distance_px']:.1f}px，IoU 0.25–0.37。
- 视觉偏差集中在拇指链以及部分长指屈曲。SDK 跟踪通过不等于物理几何通过。

完整证据和 presentation 版图表见 [HTML report](index.html)。

## 决策

暂缓训练部署验收。先分离复核 thumb CMC yaw / pitch / MCP，以及四指 MCP pitch / PIP 的绝对零位与比例；修正后保持相机和 seed=20260803 验证集不变重跑。静态门通过后再录同步连续 A/B 视频。
"""
    (report_dir / "REPORT.md").write_text(markdown)
    print(f"wrote {report_dir / 'index.html'}")
    print(f"wrote {report_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
