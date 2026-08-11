#!/usr/bin/env python3
"""Interactively match formal D435 photos to same-view OG-URDF local-q renders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


WINDOW = "G20 physical raw -> OG URDF local q | manual matcher"
INK = (35, 40, 48)
MUTED = (110, 120, 132)
PANEL_BG = (242, 244, 247)
HEADER_BG = (24, 30, 40)
BLUE = (195, 116, 43)
GOLD = (35, 176, 225)
ORANGE = (36, 92, 235)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=(260, 70, 1080, 650),
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument("--start-sample", type=str)
    parser.add_argument("--full-screen", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
    border: tuple[int, int, int] = (205, 210, 217),
) -> np.ndarray:
    result = np.full(
        (image.shape[0] + 58, image.shape[1], 3),
        PANEL_BG,
        dtype=np.uint8,
    )
    result[58:] = image
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 57), HEADER_BG, -1)
    cv2.rectangle(
        result,
        (0, 0),
        (result.shape[1] - 1, result.shape[0] - 1),
        border,
        2,
    )
    cv2.putText(
        result, title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX,
        0.63, (247, 249, 252), 2, cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            result, subtitle, (14, 49), cv2.FONT_HERSHEY_SIMPLEX,
            0.43, (190, 207, 228), 1, cv2.LINE_AA,
        )
    return result


def fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), PANEL_BG, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def estimate_background_bgr(image: np.ndarray) -> tuple[int, int, int]:
    patch = max(8, min(image.shape[:2]) // 24)
    corners = np.concatenate(
        (
            image[:patch, :patch].reshape(-1, 3),
            image[:patch, -patch:].reshape(-1, 3),
            image[-patch:, :patch].reshape(-1, 3),
            image[-patch:, -patch:].reshape(-1, 3),
        ),
        axis=0,
    )
    median = np.median(corners, axis=0)
    return tuple(int(round(value)) for value in median)


def intrinsic_correct(sim: np.ndarray, intr: dict[str, Any]) -> np.ndarray:
    height, width = sim.shape[:2]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    ppx = float(intr["ppx"])
    ppy = float(intr["ppy"])
    focal = (fx + fy) / 2.0
    matrix = np.asarray(
        [
            [fx / focal, 0.0, ppx - (fx / focal) * width / 2.0],
            [0.0, fy / focal, ppy - (fy / focal) * height / 2.0],
        ],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        sim,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=estimate_background_bgr(sim),
    )


def sim_mask(sim: np.ndarray) -> np.ndarray:
    background = np.empty_like(sim)
    background[:] = estimate_background_bgr(sim)
    delta = np.max(
        np.abs(sim.astype(np.int16) - background.astype(np.int16)), axis=2
    )
    mask = (delta > 8).astype(np.uint8) * 255
    return cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )


def overlay(real: np.ndarray, sim: np.ndarray) -> np.ndarray:
    result = real.copy()
    mask = sim_mask(sim)
    edges = cv2.dilate(
        cv2.Canny(mask, 50, 150), np.ones((3, 3), np.uint8)
    )
    tint = result.copy()
    tint[mask > 0] = BLUE
    result = cv2.addWeighted(result, 0.86, tint, 0.14, 0.0)
    result[edges > 0] = ORANGE
    return result


class Matcher:
    def __init__(
        self,
        dataset_path: Path,
        render_manifest_path: Path,
        out_dir: Path,
        roi: tuple[int, int, int, int],
    ) -> None:
        self.dataset_path = dataset_path.resolve()
        self.render_manifest_path = render_manifest_path.resolve()
        self.out_dir = out_dir.resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "accepted_panels").mkdir(exist_ok=True)
        self.dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        self.render = json.loads(
            render_manifest_path.read_text(encoding="utf-8")
        )
        self.intr = json.loads(
            Path(self.dataset["intrinsics"]).read_text(encoding="utf-8")
        )
        self.samples = list(self.dataset["samples"])
        self.candidate_mode = self.dataset.get("candidate_mode", "global")
        if self.candidate_mode not in ("global", "per_sample"):
            raise ValueError(
                f"unsupported candidate_mode: {self.candidate_mode!r}"
            )
        self.global_candidates = list(self.dataset.get("candidates", []))
        self.roi = tuple(roi)
        self.render_dir = render_manifest_path.resolve().parent
        self.real_cache: dict[int, np.ndarray] = {}
        self.sim_cache: dict[tuple[int, int], np.ndarray] = {}
        self.matches: dict[str, dict[str, Any]] = {}
        self.progress_path = self.out_dir / "manual_matches.json"
        self.csv_path = self.out_dir / "manual_matches.csv"
        if self.progress_path.exists():
            old = json.loads(self.progress_path.read_text(encoding="utf-8"))
            if old.get("dataset_sha256") != sha256(self.dataset_path):
                raise RuntimeError(
                    "existing progress belongs to a different dataset"
                )
            self.matches = {
                item["sample_id"]: item for item in old.get("matches", [])
            }
        self.sample_index = self.first_unreviewed()
        self.candidates = self.candidates_for_sample(self.sample_index)
        self.candidate_index = self.initial_candidate(self.sample_index)
        self.last_canvas: np.ndarray | None = None
        self.thumb_indices: list[int] = []
        self.validate()

    def candidates_for_sample(self, sample_index: int) -> list[dict[str, Any]]:
        if self.candidate_mode == "per_sample":
            return list(self.samples[sample_index].get("candidates", []))
        return self.global_candidates

    def validate(self) -> None:
        missing: list[str] = []
        for sample in self.samples:
            if not Path(sample["real_color"]).is_file():
                missing.append(sample["real_color"])
        candidate_counts: set[int] = set()
        candidate_sets = (
            [self.candidates_for_sample(index) for index in range(len(self.samples))]
            if self.candidate_mode == "per_sample"
            else [self.global_candidates]
        )
        checked_names: set[str] = set()
        for candidates in candidate_sets:
            candidate_counts.add(len(candidates))
            for candidate in candidates:
                if candidate["name"] in checked_names:
                    continue
                checked_names.add(candidate["name"])
                entry = self.render.get("poses", {}).get(candidate["name"])
                if entry is None:
                    missing.append(f"manifest pose:{candidate['name']}")
                    continue
                image_name = next(iter(entry["images"].values()))
                if not (self.render_dir / image_name).is_file():
                    missing.append(str(self.render_dir / image_name))
        if not candidate_counts or 0 in candidate_counts:
            raise ValueError("every sample must have at least one candidate")
        if len(candidate_counts) != 1:
            raise ValueError(
                "all per-sample candidate grids must have the same length"
            )
        if missing:
            raise FileNotFoundError(
                "manual matcher inputs missing:\n" + "\n".join(missing)
            )

    def first_unreviewed(self) -> int:
        for index, sample in enumerate(self.samples):
            if sample["sample_id"] not in self.matches:
                return index
        return 0

    def initial_candidate(self, sample_index: int) -> int:
        saved = self.matches.get(self.samples[sample_index]["sample_id"])
        if saved and saved.get("candidate_index") is not None:
            return int(saved["candidate_index"])
        # The queue is ordered from high raw/low q toward low raw/high q.
        for index in range(sample_index - 1, -1, -1):
            previous = self.matches.get(self.samples[index]["sample_id"])
            if previous and previous.get("candidate_index") is not None:
                return int(previous["candidate_index"])
        sample = self.samples[sample_index]
        parent_q = sample.get("parent_q_urdf_rad")
        mimic = self.dataset.get("production_mimic")
        if parent_q is not None and mimic is not None:
            predicted = (
                float(parent_q) * float(mimic.get("multiplier", 1.0))
                + float(mimic.get("offset", 0.0))
            )
            return min(
                range(len(self.candidates)),
                key=lambda index: abs(
                    float(self.candidates[index]["q_urdf_rad"]) - predicted
                ),
            )
        production_limit = self.dataset.get(
            "production_urdf_limit_rad", self.dataset.get("urdf_limit_rad")
        )
        if production_limit is not None:
            semantic_zero = float(production_limit[0])
            return min(
                range(len(self.candidates)),
                key=lambda index: abs(
                    float(self.candidates[index]["q_urdf_rad"])
                    - semantic_zero
                ),
            )
        return 0

    def real(self, index: int) -> np.ndarray:
        if index not in self.real_cache:
            image = cv2.imread(
                self.samples[index]["real_color"], cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(
                    f"failed to read {self.samples[index]['real_color']}"
                )
            self.real_cache[index] = image
        return self.real_cache[index]

    def sim(self, index: int) -> np.ndarray:
        cache_key = (
            (self.sample_index, index)
            if self.candidate_mode == "per_sample"
            else (-1, index)
        )
        if cache_key not in self.sim_cache:
            candidate = self.candidates[index]
            entry = self.render["poses"][candidate["name"]]
            image_name = next(iter(entry["images"].values()))
            image = cv2.imread(
                str(self.render_dir / image_name), cv2.IMREAD_COLOR
            )
            if image is None:
                raise RuntimeError(f"failed to read {image_name}")
            self.sim_cache[cache_key] = intrinsic_correct(image, self.intr)
        return self.sim_cache[cache_key]

    def crop(self, image: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.roi
        h, w = image.shape[:2]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(f"ROI {self.roi} is outside image {w}x{h}")
        return image[y1:y2, x1:x2]

    def reviewed_count(self) -> int:
        return sum(
            sample["sample_id"] in self.matches for sample in self.samples
        )

    def set_candidate(self, index: int) -> None:
        self.candidate_index = max(
            0, min(len(self.candidates) - 1, int(index))
        )
        cv2.setTrackbarPos(
            "URDF q candidate", WINDOW, self.candidate_index
        )

    def move_sample(self, delta: int) -> None:
        self.sample_index = (
            self.sample_index + delta
        ) % len(self.samples)
        self.candidates = self.candidates_for_sample(self.sample_index)
        self.set_candidate(self.initial_candidate(self.sample_index))

    def save_progress(self) -> None:
        ordered = [
            self.matches[sample["sample_id"]]
            for sample in self.samples
            if sample["sample_id"] in self.matches
        ]
        payload = {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "method": "human_same_view_photo_to_og_urdf_local_q",
            "dataset": str(self.dataset_path),
            "dataset_sha256": sha256(self.dataset_path),
            "render_manifest": str(self.render_manifest_path),
            "render_manifest_sha256": sha256(self.render_manifest_path),
            "camera": self.dataset["camera"],
            "camera_sha256": self.dataset["camera_sha256"],
            "urdf": self.dataset["urdf"],
            "urdf_sha256": self.dataset["urdf_sha256"],
            "reviewed_count": len(ordered),
            "sample_count": len(self.samples),
            "matches": ordered,
        }
        self.progress_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        fields = [
            "sample_id",
            "direction",
            "command_raw",
            "stable_readback_raw",
            "status",
            "candidate_index",
            "q_urdf_rad",
            "q_urdf_deg",
            "real_color",
            "reviewed_utc",
        ]
        with self.csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in ordered:
                writer.writerow({field: item.get(field) for field in fields})

    def record(self, status: str) -> None:
        sample = self.samples[self.sample_index]
        candidate = self.candidates[self.candidate_index]
        use_q = status != "skipped"
        self.matches[sample["sample_id"]] = {
            "sample_id": sample["sample_id"],
            "direction": sample["direction"],
            "command_raw": sample["command_raw"],
            "stable_readback_raw": sample["stable_readback_raw"],
            "status": status,
            "candidate_index": self.candidate_index if use_q else None,
            "candidate_name": candidate["name"] if use_q else None,
            "q_urdf_rad": candidate["q_urdf_rad"] if use_q else None,
            "q_urdf_deg": candidate["q_urdf_deg"] if use_q else None,
            "parent_joint": self.dataset.get("parent_joint"),
            "parent_q_urdf_rad": sample.get("parent_q_urdf_rad"),
            "real_color": sample["real_color"],
            "reviewed_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.save_progress()
        if self.last_canvas is not None:
            cv2.imwrite(
                str(
                    self.out_dir
                    / "accepted_panels"
                    / f"{sample['sample_id']}_{status}.png"
                ),
                self.last_canvas,
            )
        self.move_sample(1)

    def render_canvas(self) -> np.ndarray:
        sample = self.samples[self.sample_index]
        candidate = self.candidates[self.candidate_index]
        production_limit = self.dataset.get(
            "production_urdf_limit_rad", self.dataset["urdf_limit_rad"]
        )
        outside_og_limit = not (
            float(production_limit[0]) - 1.0e-9
            <= float(candidate["q_urdf_rad"])
            <= float(production_limit[1]) + 1.0e-9
        )
        real_full = self.real(self.sample_index)
        sim_full = self.sim(self.candidate_index)
        real_roi = self.crop(real_full)
        sim_roi = self.crop(sim_full)
        overlay_roi = self.crop(overlay(real_full, sim_full))

        canvas = np.full((930, 1900, 3), (248, 249, 251), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (1899, 68), HEADER_BG, -1)
        header = (
            f"{self.dataset['joint']} | sample {self.sample_index + 1}/"
            f"{len(self.samples)} | stable raw {sample['stable_readback_raw']:03d}"
            f" | command {sample['command_raw']:03d} | {sample['direction']}"
        )
        cv2.putText(
            canvas, header, (22, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.72, (248, 249, 252), 2, cv2.LINE_AA,
        )
        status = self.matches.get(sample["sample_id"], {}).get(
            "status", "UNREVIEWED"
        )
        detail = (
            f"candidate {self.candidate_index + 1}/{len(self.candidates)}  "
            f"q={candidate['q_urdf_rad']:.4f} rad  "
            f"({candidate['q_urdf_deg']:.2f} deg) | "
            f"reviewed {self.reviewed_count()}/{len(self.samples)} | {status}"
        )
        focus_finger = self.render.get("focus_finger", "none")
        if focus_finger != "none":
            detail += f" | visual focus={focus_finger}"
        if sample.get("parent_q_urdf_rad") is not None:
            detail += (
                f" | fixed {sample.get('parent_joint', 'parent')} "
                f"q={float(sample['parent_q_urdf_rad']):.4f} rad"
            )
        if outside_og_limit:
            detail += (
                " | OUTSIDE OG LIMIT "
                f"[{float(production_limit[0]):.2f}, "
                f"{float(production_limit[1]):.2f}] rad"
            )
        cv2.putText(
            canvas, detail, (22, 56), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (190, 210, 235), 1, cv2.LINE_AA,
        )

        panel_width, panel_height = 612, 470
        body_height = panel_height - 58
        panels = [
            label(
                fit(real_roi, panel_width, body_height),
                "FORMAL D435 PHOTO",
                sample["sample_id"],
            ),
            label(
                fit(sim_roi, panel_width, body_height),
                (
                    "UNLOCKED FOLLOWER DIAGNOSTIC RENDER"
                    if self.candidate_mode == "per_sample"
                    else "FOCUSED OG URDF RENDER"
                    if focus_finger != "none"
                    else "OG URDF RENDER"
                ),
                (
                    f"local q={candidate['q_urdf_rad']:.4f} rad"
                    + (
                        " | DIAGNOSTIC EXTENSION"
                        if outside_og_limit else ""
                    )
                ),
                border=ORANGE if outside_og_limit else GOLD,
            ),
            label(
                fit(overlay_roi, panel_width, body_height),
                "SAME-PIXEL OVERLAY",
                "orange=edge, blue=open fill",
            ),
        ]
        for index, panel in enumerate(panels):
            x = 20 + index * 626
            canvas[80 : 80 + panel_height, x : x + panel_width] = panel

        center = self.candidate_index
        start = max(0, min(center - 5, len(self.candidates) - 11))
        self.thumb_indices = list(
            range(start, min(start + 11, len(self.candidates)))
        )
        thumb_y, thumb_w, thumb_h = 580, 164, 196
        for slot, candidate_index in enumerate(self.thumb_indices):
            q = self.candidates[candidate_index]
            image = fit(self.crop(self.sim(candidate_index)), thumb_w, 145)
            tile = np.full((thumb_h, thumb_w, 3), PANEL_BG, dtype=np.uint8)
            tile[:145] = image
            selected = candidate_index == self.candidate_index
            border = GOLD if selected else (195, 201, 209)
            thickness = 4 if selected else 1
            cv2.rectangle(
                tile, (0, 0), (thumb_w - 1, thumb_h - 1),
                border, thickness,
            )
            cv2.putText(
                tile, f"{q['q_urdf_rad']:.3f} rad", (12, 168),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, INK, 1, cv2.LINE_AA,
            )
            cv2.putText(
                tile, f"{q['q_urdf_deg']:.1f} deg", (12, 188),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, MUTED, 1, cv2.LINE_AA,
            )
            x = 20 + slot * 170
            canvas[thumb_y : thumb_y + thumb_h, x : x + thumb_w] = tile

        cv2.putText(
            canvas,
            "A/D or arrows: q +/-1   J/L: q +/-5   click thumbnail/trackbar",
            (24, 824), cv2.FONT_HERSHEY_SIMPLEX, 0.54, INK, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "[/]: previous/next photo   ENTER: match+next   U: uncertain+next",
            (24, 853), cv2.FONT_HERSHEY_SIMPLEX, 0.54, INK, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "K: skip+next   R: clear current result   Q/ESC: save and quit",
            (24, 882), cv2.FONT_HERSHEY_SIMPLEX, 0.54, INK, 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Old visual angle is intentionally hidden; select only by same-view shape.",
            (24, 914), cv2.FONT_HERSHEY_SIMPLEX, 0.50, ORANGE, 1, cv2.LINE_AA,
        )
        self.last_canvas = canvas
        return canvas

    def on_mouse(self, event: int, x: int, y: int, *_: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or not (580 <= y < 776):
            return
        slot = (x - 20) // 170
        if 0 <= slot < len(self.thumb_indices):
            self.set_candidate(self.thumb_indices[slot])

    def run(self, full_screen: bool = False) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 1900, 1020)
        if full_screen:
            cv2.setWindowProperty(
                WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        cv2.createTrackbar(
            "URDF q candidate",
            WINDOW,
            self.candidate_index,
            len(self.candidates) - 1,
            lambda value: setattr(self, "candidate_index", value),
        )
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render_canvas())
            key = cv2.waitKeyEx(30)
            if key < 0:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            low = key & 0xFF
            if low in (ord("q"), 27):
                break
            if low in (ord("a"), ord(",")) or key == 2424832:
                self.set_candidate(self.candidate_index - 1)
            elif low in (ord("d"), ord(".")) or key == 2555904:
                self.set_candidate(self.candidate_index + 1)
            elif low == ord("j"):
                self.set_candidate(self.candidate_index - 5)
            elif low == ord("l"):
                self.set_candidate(self.candidate_index + 5)
            elif low == ord("["):
                self.move_sample(-1)
            elif low == ord("]"):
                self.move_sample(1)
            elif low in (13, 10):
                self.record("matched")
            elif low == ord("u"):
                self.record("uncertain")
            elif low == ord("k"):
                self.record("skipped")
            elif low == ord("r"):
                sample_id = self.samples[self.sample_index]["sample_id"]
                self.matches.pop(sample_id, None)
                self.save_progress()
        self.save_progress()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    matcher = Matcher(
        args.dataset,
        args.render_manifest,
        args.out_dir,
        tuple(args.roi),
    )
    if args.start_sample:
        matches = [
            index for index, sample in enumerate(matcher.samples)
            if sample["sample_id"] == args.start_sample
        ]
        if not matches:
            raise KeyError(f"unknown --start-sample {args.start_sample}")
        matcher.sample_index = matches[0]
        matcher.candidates = matcher.candidates_for_sample(
            matcher.sample_index
        )
        matcher.candidate_index = matcher.initial_candidate(
            matcher.sample_index
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "joint": matcher.dataset["joint"],
                    "samples": len(matcher.samples),
                    "candidate_mode": matcher.candidate_mode,
                    "candidates_per_sample": len(matcher.candidates),
                    "reviewed": matcher.reviewed_count(),
                    "roi": matcher.roi,
                    "progress": str(matcher.progress_path),
                },
                indent=2,
            )
        )
        return 0
    matcher.run(full_screen=args.full_screen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
