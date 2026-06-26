"""Render static multi-view PNGs for the configured screwdriver tasks.

This script instantiates the real Gym/Isaac Lab task configs from
``screwdriver_rl.tasks``, resets one environment into its configured initial
posture, then captures the viewer camera from several angles.

Examples:

    python tools/render_task_configs.py --headless
    python tools/render_task_configs.py --task Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0 --headless
    python tools/render_task_configs.py --views iso,front,top,detail:35:18 --render_width 1600 --render_height 1200 --headless
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import traceback
from pathlib import Path
from typing import Iterable

from isaaclab.app import AppLauncher


DEFAULT_VIEW_PRESETS: dict[str, tuple[float, float]] = {
    # name: (azimuth_deg, elevation_deg), Z-up world coordinates.
    "iso": (45.0, 30.0),
    "front": (-90.0, 12.0),
    "right": (0.0, 12.0),
    "back": (90.0, 12.0),
    "left": (180.0, 12.0),
    "top": (45.0, 82.0),
}

DEFAULT_TASK_IDS: tuple[str, ...] = (
    "Isaac-Allegro-Screwdriver-Rotation-Direct-v0",
    "Isaac-Allegro-4F-Screwdriver-Rotation-Direct-v0",
    "Isaac-LinkerL20-Screwdriver-Rotation-Direct-v0",
    "Isaac-LinkerL20-Screwdriver-Rotation-Top-Grasp-Direct-v0",
)


parser = argparse.ArgumentParser(
    description="Render multi-view PNGs of the ScrewdriverRL task initial configurations."
)
parser.add_argument(
    "--task",
    action="append",
    default=None,
    help=(
        "Task id to render. May be passed multiple times or as a comma-separated "
        "list. Defaults to all registered screwdriver rotation tasks."
    ),
)
parser.add_argument(
    "--list_tasks",
    action="store_true",
    help="List discovered screwdriver task ids and exit.",
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/task_config_renders"),
    help="Directory where PNGs and render_summary.json are written.",
)
parser.add_argument("--render_width", "--width", dest="render_width", type=int, default=1280, help="PNG width in pixels.")
parser.add_argument("--render_height", "--height", dest="render_height", type=int, default=720, help="PNG height in pixels.")
parser.add_argument("--seed", type=int, default=0, help="Reset seed for deterministic renders.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to instantiate. Env 0 is rendered.")
parser.add_argument(
    "--views",
    type=str,
    default="iso,front,right,back,left,top",
    help=(
        "Comma-separated view names. Built-ins: "
        f"{', '.join(DEFAULT_VIEW_PRESETS)}. Custom format: name:azimuth:elevation."
    ),
)
parser.add_argument(
    "--distance",
    type=float,
    default=None,
    help="Camera distance in meters. Defaults to a value derived from the task body extents.",
)
parser.add_argument(
    "--radius_scale",
    type=float,
    default=2.8,
    help="Camera distance multiplier used when --distance is not set.",
)
parser.add_argument(
    "--min_distance",
    type=float,
    default=0.35,
    help="Minimum auto camera distance in meters.",
)
parser.add_argument(
    "--warmup_frames",
    type=int,
    default=8,
    help="Render frames after each camera move before saving the PNG.",
)
parser.add_argument(
    "--domain_rand",
    action="store_true",
    help="Leave task domain randomization enabled. Default disables it for repeatable config renders.",
)
parser.add_argument(
    "--random_start",
    action="store_true",
    help="Leave screwdriver start angle randomization enabled. Default fixes the configured reset angle.",
)
parser.add_argument(
    "--no_contact_sheet",
    action="store_true",
    help="Only write individual view PNGs, not the per-task contact_sheet.png.",
)
parser.add_argument(
    "--show_viewport",
    action="store_true",
    help="Open the Isaac Sim viewport instead of the default headless offscreen render.",
)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

if args.list_tasks:
    raw_tasks = args.task or ["all"]
    listed: list[str] = []
    for item in raw_tasks:
        for token in item.split(","):
            task_id = token.strip()
            if not task_id:
                continue
            if task_id.lower() == "all":
                listed.extend(DEFAULT_TASK_IDS)
            else:
                listed.append(task_id)
    for task_id in dict.fromkeys(listed):
        print(task_id, flush=True)
    raise SystemExit(0)

# Offscreen PNG generation needs camera rendering enabled even in headless mode.
args.enable_cameras = True
if not args.show_viewport:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np

import screwdriver_rl.tasks  # noqa: F401

try:
    from isaaclab_tasks.utils import parse_env_cfg
except ImportError:
    try:
        from omni.isaac.lab_tasks.utils import parse_env_cfg
    except ImportError:
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _discover_screwdriver_tasks() -> list[str]:
    registry = getattr(gym, "registry", None)
    if registry is None:
        registry = gym.envs.registry
    registered = {
        spec.id
        for spec in registry.values()
        if "Screwdriver-Rotation" in spec.id
    }
    known = [task_id for task_id in DEFAULT_TASK_IDS if not registered or task_id in registered]
    extras = sorted(registered.difference(DEFAULT_TASK_IDS))
    return known + extras


def _expand_task_args(raw_tasks: Iterable[str] | None) -> list[str]:
    discovered = _discover_screwdriver_tasks()
    if not raw_tasks:
        return discovered

    expanded: list[str] = []
    for item in raw_tasks:
        for token in item.split(","):
            task_id = token.strip()
            if not task_id:
                continue
            if task_id.lower() == "all":
                expanded.extend(discovered)
            else:
                expanded.append(task_id)

    # Preserve user order while removing duplicates.
    deduped = list(dict.fromkeys(expanded))
    unknown = [task_id for task_id in deduped if task_id not in discovered]
    if unknown:
        available = "\n  ".join(discovered)
        raise ValueError(
            "Unknown task id(s): "
            + ", ".join(unknown)
            + "\nDiscovered screwdriver tasks:\n  "
            + available
        )
    return deduped


def _parse_views(raw: str) -> list[tuple[str, float, float]]:
    views: list[tuple[str, float, float]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if token in DEFAULT_VIEW_PRESETS:
            azimuth, elevation = DEFAULT_VIEW_PRESETS[token]
            views.append((token, azimuth, elevation))
            continue

        parts = token.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Unknown view {token!r}. Use one of {sorted(DEFAULT_VIEW_PRESETS)} "
                "or custom format name:azimuth:elevation."
            )
        name, azimuth_s, elevation_s = parts
        views.append((name.strip(), float(azimuth_s), float(elevation_s)))

    if not views:
        raise ValueError("At least one view is required.")
    return views


def _camera_eye(
    target: np.ndarray,
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> tuple[float, float, float]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    horizontal = distance * math.cos(elevation)
    eye = np.array(
        [
            target[0] + horizontal * math.cos(azimuth),
            target[1] + horizontal * math.sin(azimuth),
            target[2] + distance * math.sin(elevation),
        ],
        dtype=np.float64,
    )
    return tuple(float(x) for x in eye)


def _compute_focus(base_env, env_idx: int = 0) -> tuple[np.ndarray, float]:
    points = []
    for attr_name in ("allegro", "screwdriver"):
        asset = getattr(base_env, attr_name, None)
        if asset is None:
            continue
        body_pos = asset.data.body_state_w[env_idx, :, :3].detach().cpu().numpy()
        points.append(body_pos)

    if not points:
        return np.array([0.0, 0.0, 1.25], dtype=np.float64), 0.2

    all_points = np.concatenate(points, axis=0)
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = float(max(np.linalg.norm(upper - lower) * 0.5, 0.15))
    return center.astype(np.float64), radius


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] > 3:
        image = image[:, :, :3]

    try:
        from PIL import Image

        Image.fromarray(image).save(path)
        return
    except ImportError:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(path, image)
        return
    except ImportError:
        pass

    import matplotlib.image as mpimg

    mpimg.imsave(path, image)


def _make_contact_sheet(
    frames: list[np.ndarray],
    labels: list[str],
    columns: int = 3,
) -> np.ndarray:
    if not frames:
        raise ValueError("Cannot build a contact sheet with no frames.")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        rows = math.ceil(len(frames) / columns)
        height, width = frames[0].shape[:2]
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        tiles = frames + [blank] * (rows * columns - len(frames))
        row_images = [
            np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1)
            for row in range(rows)
        ]
        return np.concatenate(row_images, axis=0)

    tile_h, tile_w = frames[0].shape[:2]
    label_h = 28
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for index, (frame, label) in enumerate(zip(frames, labels)):
        row = index // columns
        col = index % columns
        x = col * tile_w
        y = row * (tile_h + label_h)
        draw.rectangle((x, y, x + tile_w, y + label_h), fill=(24, 24, 24))
        draw.text((x + 10, y + 8), label, fill=(235, 235, 235), font=font)
        sheet.paste(Image.fromarray(frame[:, :, :3]), (x, y + label_h))

    return np.asarray(sheet)


def _capture_rgb(env, warmup_frames: int) -> np.ndarray:
    frame = None
    for _ in range(max(0, warmup_frames)):
        frame = env.render()

    for _ in range(8):
        frame = env.render()
        if frame is None:
            continue
        image = np.asarray(frame)
        if image.size > 0 and image.max() > 0:
            return image[:, :, :3].copy()

    if frame is None:
        raise RuntimeError("env.render() returned None. Make sure render_mode='rgb_array' is active.")
    return np.asarray(frame)[:, :, :3].copy()


def _configure_env(task_id: str):
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.viewer.resolution = (args.render_width, args.render_height)
    env_cfg.viewer.cam_prim_path = "/OmniverseKit_Persp"

    if not args.domain_rand and hasattr(env_cfg, "domain_rand"):
        env_cfg.domain_rand.enabled = False
    if not args.random_start and hasattr(env_cfg, "randomize_obj_start"):
        env_cfg.randomize_obj_start = False
    return env_cfg


def _render_task(task_id: str, views: list[tuple[str, float, float]]) -> dict:
    task_dir = args.output / _slugify(task_id)
    env_cfg = _configure_env(task_id)
    env = None
    result = {
        "task": task_id,
        "output_dir": str(task_dir),
        "views": [],
    }

    try:
        print(f"[render] Creating env: {task_id}", flush=True)
        env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
        base_env = env.unwrapped
        env.reset(seed=args.seed)

        target, radius = _compute_focus(base_env, env_idx=0)
        distance = args.distance
        if distance is None:
            distance = max(args.min_distance, radius * args.radius_scale)

        frames: list[np.ndarray] = []
        labels: list[str] = []
        camera_path = base_env.cfg.viewer.cam_prim_path
        target_tuple = tuple(float(x) for x in target)

        for view_name, azimuth, elevation in views:
            eye = _camera_eye(target, distance, azimuth, elevation)
            base_env.sim.set_camera_view(eye, target_tuple, camera_prim_path=camera_path)
            frame = _capture_rgb(env, args.warmup_frames)

            png_path = task_dir / f"{_slugify(view_name)}.png"
            _save_png(png_path, frame)
            print(f"[render] Wrote {png_path}", flush=True)

            frames.append(frame)
            labels.append(f"{view_name}  az={azimuth:g}  el={elevation:g}")
            result["views"].append(
                {
                    "name": view_name,
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "eye": list(eye),
                    "target": list(target_tuple),
                    "file": str(png_path),
                }
            )

        if not args.no_contact_sheet:
            sheet = _make_contact_sheet(frames, labels)
            sheet_path = task_dir / "contact_sheet.png"
            _save_png(sheet_path, sheet)
            result["contact_sheet"] = str(sheet_path)
            print(f"[render] Wrote {sheet_path}", flush=True)

        result["camera_distance"] = distance
        result["focus_radius"] = radius
        return result
    finally:
        if env is not None:
            env.close()
        gc.collect()


def main() -> None:
    tasks = _expand_task_args(args.task)
    if args.list_tasks:
        for task_id in tasks:
            print(task_id)
        return

    views = _parse_views(args.views)
    args.output.mkdir(parents=True, exist_ok=True)

    summary = {
        "output": str(args.output),
        "resolution": [args.render_width, args.render_height],
        "tasks": [],
    }
    for task_id in tasks:
        summary["tasks"].append(_render_task(task_id, views))

    summary_path = args.output / "render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[render] Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
