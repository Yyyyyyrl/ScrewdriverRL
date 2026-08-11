#!/usr/bin/env python3
"""Time-align D435 and Isaac trajectory frames into a side-by-side A/B MP4."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import json
from pathlib import Path

import cv2
import numpy as np

from tools.compose_g20_matched_pose_ab import intrinsic_correct


def nearest_index(values: list[int], target: int) -> int:
    index = bisect_left(values, target)
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    return index if values[index] - target < target - values[index - 1] else index - 1


def label(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 62), (20, 23, 28), -1)
    cv2.putText(
        result, title, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
        (248, 249, 251), 2, cv2.LINE_AA,
    )
    cv2.putText(
        result, subtitle, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
        (196, 202, 211), 1, cv2.LINE_AA,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-video", type=Path, required=True)
    parser.add_argument("--real-meta", type=Path, required=True)
    parser.add_argument("--trajectory-log", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--out-video", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--panel-height", type=int, default=540)
    args = parser.parse_args()

    camera = json.loads(args.real_meta.read_text(encoding="utf-8"))
    trajectory = json.loads(args.trajectory_log.read_text(encoding="utf-8"))
    intrinsics = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    render_manifest = json.loads(
        (args.sim_dir / "render_manifest.json").read_text(encoding="utf-8")
    )
    commands = trajectory["commands"]
    camera_times = [int(frame["wall_time_ns"]) for frame in camera["frames"]]
    command_times = [int(command["wall_time_ns"]) for command in commands]
    if command_times[0] < camera_times[0] or command_times[-1] > camera_times[-1]:
        raise RuntimeError(
            "camera does not cover the full command window: "
            f"camera={camera_times[0]}..{camera_times[-1]}, "
            f"commands={command_times[0]}..{command_times[-1]}"
        )
    if len(render_manifest["poses"]) != len(commands):
        raise RuntimeError("Isaac manifest frame count does not match trajectory")
    max_drift = max(
        float(row["max_abs_joint_drift_rad"])
        for row in render_manifest["poses"].values()
    )
    if max_drift > 1e-7:
        raise RuntimeError(f"Isaac kinematic drift is {max_drift}")

    real_indices = [nearest_index(camera_times, value) for value in command_times]
    cap = cv2.VideoCapture(str(args.real_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {args.real_video}")
    args.out_video.parent.mkdir(parents=True, exist_ok=True)
    args.out_meta.parent.mkdir(parents=True, exist_ok=True)
    fps = float(trajectory["rate_hz"])
    writer = cv2.VideoWriter(
        str(args.out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (args.panel_width * 2, args.panel_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer {args.out_video}")

    current_index = -1
    current_frame = None
    alignment_rows = []
    try:
        for index, (command, real_index) in enumerate(zip(commands, real_indices)):
            while current_index < real_index:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"real video ended before frame {real_index}")
                current_index += 1
                current_frame = frame
            if current_frame is None:
                raise RuntimeError("real video supplied no frames")

            sim_path = args.sim_dir / f"frame_{index:04d}_sim.png"
            sim = cv2.imread(str(sim_path), cv2.IMREAD_COLOR)
            if sim is None:
                raise FileNotFoundError(sim_path)
            sim, _ = intrinsic_correct(sim, intrinsics)
            real_panel = cv2.resize(
                current_frame, (args.panel_width, args.panel_height),
                interpolation=cv2.INTER_AREA,
            )
            sim_panel = cv2.resize(
                sim, (args.panel_width, args.panel_height),
                interpolation=cv2.INTER_AREA,
            )
            elapsed = (command_times[index] - command_times[0]) / 1e9
            phase = str(command["phase"])
            real_panel = label(real_panel, "REAL D435", f"t={elapsed:05.2f}s | {phase}")
            sim_panel = label(sim_panel, "ISAAC TARGET | OG MIMIC", f"t={elapsed:05.2f}s | {phase}")
            writer.write(np.hstack([real_panel, sim_panel]))
            alignment_rows.append(
                {
                    "output_frame": index,
                    "command_wall_time_ns": command_times[index],
                    "real_frame_index": real_index,
                    "real_wall_time_ns": camera_times[real_index],
                    "alignment_error_ms": (
                        camera_times[real_index] - command_times[index]
                    ) / 1e6,
                    "phase": phase,
                }
            )
    finally:
        writer.release()
        cap.release()

    check = cv2.VideoCapture(str(args.out_video))
    frame_count = int(round(check.get(cv2.CAP_PROP_FRAME_COUNT)))
    check_fps = float(check.get(cv2.CAP_PROP_FPS))
    width = int(round(check.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(check.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ok, _ = check.read()
    check.release()
    if not ok or frame_count != len(commands):
        raise RuntimeError(
            f"output verification failed: readable={ok}, "
            f"frames={frame_count}/{len(commands)}"
        )
    errors = np.array([abs(row["alignment_error_ms"]) for row in alignment_rows])
    payload = {
        "schema_version": 1,
        "trajectory": trajectory["name"],
        "real_video": str(args.real_video),
        "sim_dir": str(args.sim_dir),
        "out_video": str(args.out_video),
        "frame_count": frame_count,
        "fps": check_fps,
        "duration_s": frame_count / check_fps,
        "width": width,
        "height": height,
        "isaac_max_joint_drift_rad": max_drift,
        "alignment_abs_error_ms": {
            "mean": float(errors.mean()),
            "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max()),
        },
        "frames": alignment_rows,
    }
    args.out_meta.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[compose] wrote {args.out_video}: {frame_count} frames, "
        f"{check_fps:.2f} fps, alignment p95={np.percentile(errors,95):.2f}ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
