"""Multi-person tracker using OC-SORT for robust ID-persistent tracking.

Phase 1 of multi-person capture: detect all people in a video and return
per-person bounding box tracks with interpolation and smoothing.

Falls back to YOLO's built-in tracker if OC-SORT is not installed.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from ultralytics import YOLO
from hmr4d import PROJ_ROOT
from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth

try:
    from ocsort.ocsort import OCSort
    OCSORT_AVAILABLE = True
except ImportError:
    OCSORT_AVAILABLE = False


# Distinct colors for track visualization (BGR for OpenCV)
_TRACK_COLORS = [
    (230, 159, 0),    # orange
    (86, 180, 233),   # sky blue
    (0, 158, 115),    # bluish green
    (240, 228, 66),   # yellow
    (0, 114, 178),    # blue
    (213, 94, 0),     # vermilion
    (204, 121, 167),  # reddish purple
    (110, 206, 206),  # teal
    (150, 75, 200),   # purple
    (0, 0, 0),        # black
]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return _TRACK_COLORS[track_id % len(_TRACK_COLORS)]


@dataclass
class TrackResult:
    track_id: int
    bbx_xyxy: torch.Tensor   # (N_frames, 4) interpolated, smoothed, full video length
    frame_mask: torch.Tensor  # (N_frames,) bool — which frames have real detections

    @property
    def num_real_detections(self) -> int:
        return int(self.frame_mask.sum().item())

    @property
    def mean_area(self) -> float:
        valid = self.bbx_xyxy[self.frame_mask]
        wh = valid[:, 2:] - valid[:, :2]
        return float((wh[:, 0] * wh[:, 1]).mean().item())


@dataclass
class AllTracksResult:
    tracks: list[TrackResult]
    num_frames: int
    video_path: str

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, idx: int) -> TrackResult:
        return self.tracks[idx]

    def __iter__(self):
        return iter(self.tracks)


class PersonTracker:
    """Multi-person tracker with persistent IDs.

    Uses OC-SORT when available, falls back to YOLO's built-in BoT-SORT tracker.
    """

    def __init__(
        self,
        yolo_model: str | Path | None = None,
        conf_threshold: float = 0.5,
        min_track_duration: int = 15,
        smooth_window: int = 5,
        smooth_passes: int = 2,
        use_ocsort: bool = True,
    ) -> None:
        if yolo_model is None:
            yolo_model = PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt"
        self.yolo = YOLO(str(yolo_model))
        self.conf_threshold = conf_threshold
        self.min_track_duration = min_track_duration
        self.smooth_window = smooth_window
        self.smooth_passes = smooth_passes
        self.use_ocsort = use_ocsort and OCSORT_AVAILABLE
        self.tracker_backend = "ocsort" if self.use_ocsort else "yolo"

        if use_ocsort and not OCSORT_AVAILABLE:
            print(
                "[PersonTracker] OC-SORT not installed — falling back to YOLO tracker. "
                "Install with: pip install ocsort"
            )

    def cache_metadata(self) -> dict:
        """Describe how this tracker instance produced cached tracks."""
        return {
            "tracking_version": 2,
            "tracker_backend": self.tracker_backend,
            "conf_threshold": float(self.conf_threshold),
            "smooth_window": int(self.smooth_window),
            "smooth_passes": int(self.smooth_passes),
            "min_track_duration": int(self.min_track_duration),
        }

    def _track_with_ocsort(self, video_path: str) -> list[list[dict]]:
        tracker = OCSort(
            det_thresh=self.conf_threshold,
            max_age=30,
            min_hits=3,
            iou_threshold=0.3,
            delta_t=3,
            asso_func="iou",
            inertia=0.2,
            use_byte=False,
        )
        num_frames = get_video_lwh(video_path)[0]
        track_history: list[list[dict]] = []

        results = self.yolo.predict(
            video_path,
            device="cuda",
            conf=self.conf_threshold,
            classes=0,
            verbose=False,
            stream=True,
        )

        for result in tqdm(results, total=num_frames, desc="OC-SORT Tracking"):
            boxes = result.boxes
            if len(boxes) > 0:
                # OC-SORT expects torch tensor [x1,y1,x2,y2,score,class]
                dets = torch.cat([boxes.xyxy, boxes.conf.unsqueeze(1), boxes.cls.unsqueeze(1)], dim=1).cpu()
            else:
                dets = torch.empty((0, 6))

            online_targets = tracker.update(dets, None)

            frame_detections = []
            if online_targets is not None and len(online_targets) > 0:
                for t in online_targets:
                    frame_detections.append(
                        {"id": int(t[4]), "bbx_xyxy": t[:4].astype(np.float32),
                         "conf": float(t[5]) if len(t) > 5 else 1.0}
                    )
            track_history.append(frame_detections)

        return track_history

    def _track_with_yolo(self, video_path: str) -> list[list[dict]]:
        cfg = {
            "device": "cuda",
            "conf": self.conf_threshold,
            "classes": 0,
            "verbose": False,
            "stream": True,
        }
        num_frames = get_video_lwh(video_path)[0]
        results = self.yolo.track(video_path, **cfg)

        track_history: list[list[dict]] = []
        for result in tqdm(results, total=num_frames, desc="YOLO Tracking"):
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
                bbx_xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().tolist()
                frame_detections = [
                    {"id": track_ids[i], "bbx_xyxy": bbx_xyxy[i], "conf": confs[i]}
                    for i in range(len(track_ids))
                ]
            else:
                frame_detections = []
            track_history.append(frame_detections)

        return track_history

    def _raw_track(self, video_path: str) -> list[list[dict]]:
        if self.use_ocsort:
            return self._track_with_ocsort(video_path)
        return self._track_with_yolo(video_path)

    @staticmethod
    def _parse_track_history(
        track_history: list[list[dict]],
    ) -> tuple[dict[int, list[int]], dict[int, np.ndarray], dict[int, list[float]]]:
        id_to_frame_ids: dict[int, list[int]] = defaultdict(list)
        id_to_bbx_lists: dict[int, list[np.ndarray]] = defaultdict(list)
        id_to_conf_lists: dict[int, list[float]] = defaultdict(list)

        for frame_id, frame in enumerate(track_history):
            for det in frame:
                tid = det["id"]
                id_to_frame_ids[tid].append(frame_id)
                id_to_bbx_lists[tid].append(det["bbx_xyxy"])
                id_to_conf_lists[tid].append(det.get("conf", 1.0))

        id_to_bbx_xyxys = {tid: np.array(bboxes) for tid, bboxes in id_to_bbx_lists.items()}
        return id_to_frame_ids, id_to_bbx_xyxys, id_to_conf_lists

    @staticmethod
    def _sort_ids_by_area(
        id_to_bbx_xyxys: dict[int, np.ndarray],
        video_w: int,
        video_h: int,
    ) -> list[int]:
        id_area_sum = {}
        for tid, bboxes in id_to_bbx_xyxys.items():
            wh = bboxes[:, 2:] - bboxes[:, :2]
            id_area_sum[tid] = float((wh[:, 0] * wh[:, 1] / video_w / video_h).sum())
        return sorted(id_area_sum, key=id_area_sum.get, reverse=True)

    def _process_track(
        self,
        frame_ids: list[int],
        bbx_xyxys: np.ndarray,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Interpolate gaps and smooth a single track's bounding boxes.

        Returns (bbx_xyxy, frame_mask) or None if too short.
        """
        if len(frame_ids) < self.min_track_duration:
            return None

        frame_ids_t = torch.tensor(frame_ids, dtype=torch.long)
        bbx_t = torch.tensor(bbx_xyxys, dtype=torch.float32)

        mask = frame_id_to_mask(frame_ids_t, num_frames)
        bbx_full = rearrange_by_mask(bbx_t, mask)
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)
        bbx_full = linear_interpolate_frame_ids(bbx_full, missing_frame_id_list)

        # Clamp leading/trailing zeros to nearest valid bbox
        nonzero = bbx_full.sum(1) != 0
        if nonzero.any() and not nonzero.all():
            first_valid = nonzero.nonzero(as_tuple=True)[0][0].item()
            last_valid = nonzero.nonzero(as_tuple=True)[0][-1].item()
            if first_valid > 0:
                bbx_full[:first_valid] = bbx_full[first_valid]
            if last_valid < num_frames - 1:
                bbx_full[last_valid + 1:] = bbx_full[last_valid]

        for _ in range(self.smooth_passes):
            bbx_full = moving_average_smooth(bbx_full, window_size=self.smooth_window, dim=0)

        return bbx_full, mask

    def detect_and_track_all(
        self,
        video_path: str | Path,
        min_track_frames: int | None = None,
    ) -> list[dict]:
        """Detect and track all people. Returns list of track dicts sorted by area.

        Each dict has:
            - track_id: int
            - bbx_xyxy: Tensor (F, 4) — interpolated & smoothed
            - detection_mask: Tensor (F,) bool
        """
        video_path = str(video_path)
        if min_track_frames is not None:
            orig = self.min_track_duration
            self.min_track_duration = min_track_frames

        num_frames, video_w, video_h = get_video_lwh(video_path)
        track_history = self._raw_track(video_path)
        id_to_frame_ids, id_to_bbx_xyxys, id_to_confs = self._parse_track_history(track_history)
        sorted_ids = self._sort_ids_by_area(id_to_bbx_xyxys, video_w, video_h)

        tracks = []
        for tid in sorted_ids:
            result = self._process_track(
                id_to_frame_ids[tid], id_to_bbx_xyxys[tid], num_frames
            )
            if result is None:
                continue
            bbx_xyxy, frame_mask = result

            # Build per-frame detection confidence (0.0 for interpolated frames)
            det_conf = torch.zeros(num_frames, dtype=torch.float32)
            for fi, c in zip(id_to_frame_ids[tid], id_to_confs[tid]):
                det_conf[fi] = c

            tracks.append({
                "track_id": tid,
                "bbx_xyxy": bbx_xyxy.float(),
                "detection_mask": frame_mask,
                "detection_conf": det_conf,
            })

        if min_track_frames is not None:
            self.min_track_duration = orig

        return tracks


def merge_tracks(
    track_a: dict,
    track_b: dict,
    num_frames: int,
    smooth_window: int = 5,
    smooth_passes: int = 2,
) -> dict:
    """Merge two track dicts into one, combining their real detections.

    For each frame, picks the bbox from whichever track has a real detection.
    If both have detections at the same frame, uses the one with higher
    detection confidence. Gaps between detections are interpolated and the
    result is smoothed.

    Args:
        track_a: primary track dict (its track_id is kept)
        track_b: track to merge in
        num_frames: total video frame count
        smooth_window: moving average window for smoothing
        smooth_passes: number of smoothing passes

    Returns:
        New merged track dict with keys: track_id, bbx_xyxy, detection_mask,
        detection_conf, merged_from (list of source track_ids).
    """
    mask_a = track_a["detection_mask"]
    mask_b = track_b["detection_mask"]
    conf_a = track_a.get("detection_conf", torch.zeros(num_frames))
    conf_b = track_b.get("detection_conf", torch.zeros(num_frames))
    boxes_a = track_a["bbx_xyxy"]
    boxes_b = track_b["bbx_xyxy"]

    # Ensure tensors
    if not isinstance(mask_a, torch.Tensor):
        mask_a = torch.tensor(mask_a, dtype=torch.bool)
    if not isinstance(mask_b, torch.Tensor):
        mask_b = torch.tensor(mask_b, dtype=torch.bool)
    if not isinstance(boxes_a, torch.Tensor):
        boxes_a = torch.tensor(boxes_a, dtype=torch.float32)
    if not isinstance(boxes_b, torch.Tensor):
        boxes_b = torch.tensor(boxes_b, dtype=torch.float32)
    if not isinstance(conf_a, torch.Tensor):
        conf_a = torch.tensor(conf_a, dtype=torch.float32)
    if not isinstance(conf_b, torch.Tensor):
        conf_b = torch.tensor(conf_b, dtype=torch.float32)

    n = min(num_frames, len(mask_a), len(mask_b))
    merged_mask = torch.zeros(num_frames, dtype=torch.bool)
    merged_conf = torch.zeros(num_frames, dtype=torch.float32)
    merged_boxes = torch.zeros(num_frames, 4, dtype=torch.float32)

    for f in range(n):
        has_a = mask_a[f].item()
        has_b = mask_b[f].item()
        if has_a and has_b:
            # Both have real detections — pick higher confidence
            if conf_a[f] >= conf_b[f]:
                merged_boxes[f] = boxes_a[f]
                merged_conf[f] = conf_a[f]
            else:
                merged_boxes[f] = boxes_b[f]
                merged_conf[f] = conf_b[f]
            merged_mask[f] = True
        elif has_a:
            merged_boxes[f] = boxes_a[f]
            merged_conf[f] = conf_a[f]
            merged_mask[f] = True
        elif has_b:
            merged_boxes[f] = boxes_b[f]
            merged_conf[f] = conf_b[f]
            merged_mask[f] = True

    # Interpolate gaps between real detections
    if merged_mask.any():
        missing = get_frame_id_list_from_mask(~merged_mask)
        merged_boxes = linear_interpolate_frame_ids(merged_boxes, missing)

        # Clamp leading/trailing to nearest valid bbox
        nonzero = merged_boxes.sum(1) != 0
        if nonzero.any() and not nonzero.all():
            first = nonzero.nonzero(as_tuple=True)[0][0].item()
            last = nonzero.nonzero(as_tuple=True)[0][-1].item()
            if first > 0:
                merged_boxes[:first] = merged_boxes[first]
            if last < num_frames - 1:
                merged_boxes[last + 1:] = merged_boxes[last]

        # Smooth
        for _ in range(smooth_passes):
            merged_boxes = moving_average_smooth(
                merged_boxes, window_size=smooth_window, dim=0
            )

    merged_from = [track_a["track_id"]]
    if track_b["track_id"] not in merged_from:
        merged_from.append(track_b["track_id"])
    # Carry forward any prior merge history
    for t in (track_a, track_b):
        for tid in t.get("merged_from", []):
            if tid not in merged_from:
                merged_from.append(tid)

    return {
        "track_id": track_a["track_id"],
        "bbx_xyxy": merged_boxes.float(),
        "detection_mask": merged_mask,
        "detection_conf": merged_conf,
        "merged_from": merged_from,
    }


def describe_track(track: dict, fps: float = 30.0) -> str:
    """One-line description of a track for UI display."""
    tid = track["track_id"]
    mask = track["detection_mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    mask = np.asarray(mask, dtype=bool)
    n_dets = int(mask.sum())
    if n_dets == 0:
        return f"Track {tid} (no detections)"
    det_frames = np.where(mask)[0]
    first, last = int(det_frames[0]), int(det_frames[-1])
    return f"Track {tid} (frames {first}-{last}, {n_dets} dets)"


def render_track_visualization(
    video_path: str | Path,
    all_tracks: list[dict],
    output_path: str | Path,
    fps: int | None = None,
    bbox_thickness: int = 2,
    font_scale: float = 0.7,
):
    """Overlay colored bounding boxes with track IDs on the video."""
    video_path = str(video_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    if fps is None:
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(total), desc="Rendering track visualization"):
        ret, frame = cap.read()
        if not ret:
            break

        for track in all_tracks:
            tid = track["track_id"]
            bbox = track["bbx_xyxy"][frame_idx].numpy()
            color = _color_for_id(tid)

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thickness)

            label = f"ID {tid}"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            label_y = y1 - 6 if y1 - lh - 6 > 0 else y2 + lh + 6
            cv2.rectangle(
                frame,
                (x1, label_y - lh - baseline),
                (x1 + lw, label_y + baseline),
                color,
                cv2.FILLED,
            )
            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
            cv2.putText(
                frame, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA,
            )

            # Green dot = real detection, red = interpolated
            is_real = track["detection_mask"][frame_idx].item()
            dot_color = (0, 200, 0) if is_real else (0, 0, 200)
            cv2.circle(frame, (x2 - 8, y1 + 8), 5, dot_color, cv2.FILLED)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[PersonTracker] Visualization saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-person tracking")
    parser.add_argument("video", type=str, help="Path to input video")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--min-duration", type=int, default=15)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--no-ocsort", action="store_true")
    args = parser.parse_args()

    tracker = PersonTracker(
        conf_threshold=args.conf,
        min_track_duration=args.min_duration,
        use_ocsort=not args.no_ocsort,
    )
    result = tracker.detect_and_track_all(args.video)

    print(f"\nFound {len(result)} tracks:")
    for t in result:
        n_real = int(t['detection_mask'].sum().item())
        print(f"  Track {t['track_id']}: {n_real} real detections")

    output = args.output or str(Path(args.video).with_stem(Path(args.video).stem + "_tracks"))
    render_track_visualization(args.video, result, output)
