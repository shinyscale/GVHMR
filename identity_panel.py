"""Interactive Identity Inspector panel for multi-person capture.

Provides frame-by-frame identity verification, confidence visualization,
direct two-click bbox editing, keyframe management, and per-person
incremental reprocessing. All panel logic lives here to keep gvhmr_gui.py
manageable.

Exports:
    build_identity_panel() — constructs Gradio components, returns component dict
    init_panel_state() — populates state from pipeline results
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

from identity_confidence import TrackConfidence, confidence_to_array, compute_bbox_iou
from identity_tracking import IdentityTrack, IdentityKeyframe, merge_identity_tracks, split_identity_track
from identity_reid import ShapeReIdentifier
from pose_correction import CorrectionTrack
from pose_correction_panel import (
    build_pose_correction_panel,
    init_pose_session,
)

matplotlib.use("Agg")

# ── Module-level session data (heavy, not serialized through Gradio) ──

_SESSION_DATA: dict[str, Any] = {}
# Keys: video_path, identity_tracks, all_bboxes, original_bboxes,
#        bbox_corrections, dirty_persons, confidences, num_frames,
#        fps, frame_cache, output_dir, person_dirs, all_tracks,
#        pipeline_params, img_width, img_height

_FRAME_CACHE_SIZE = 50
_PERSON_META_FILENAME = "person_meta.json"

# Person colors for bbox overlay (up to 8 people)
_PERSON_COLORS = [
    (66, 133, 244),   # blue
    (234, 67, 53),    # red
    (52, 168, 83),    # green
    (251, 188, 4),    # yellow
    (171, 71, 188),   # purple
    (0, 172, 193),    # cyan
    (255, 112, 67),   # deep orange
    (158, 157, 36),   # lime
]


class _LRUFrameCache:
    """Simple LRU cache for decoded video frames."""

    def __init__(self, maxsize: int = _FRAME_CACHE_SIZE):
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, frame_idx: int) -> np.ndarray | None:
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]
        return None

    def put(self, frame_idx: int, frame: np.ndarray) -> None:
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[frame_idx] = frame

    def clear(self) -> None:
        self._cache.clear()


def _extract_frame(video_path: str, frame_idx: int, session_id: str) -> np.ndarray | None:
    """Extract a single frame from video, using cache."""
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return None

    cache = sd.get("frame_cache")
    if cache is None:
        cache = _LRUFrameCache()
        sd["frame_cache"] = cache

    cached = cache.get(frame_idx)
    if cached is not None:
        return cached

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cache.put(frame_idx, frame_rgb)
    return frame_rgb


def _confidence_color(value: float) -> str:
    """Return CSS color for confidence value."""
    if value >= 0.8:
        return "#34a853"  # green
    elif value >= 0.4:
        return "#fbbc04"  # yellow
    return "#ea4335"  # red


def _render_frame_with_bboxes(
    frame: np.ndarray,
    bboxes: dict[int, np.ndarray],
    confidences: dict[int, TrackConfidence],
    selected_person: int,
    bbox_edit_state: dict | None = None,
) -> np.ndarray:
    """Draw bboxes, corner handles, confidence badges, and labels via cv2.

    Returns a single np.ndarray (RGB) for gr.Image.
    """
    img = frame.copy()

    for pid, bbox in sorted(bboxes.items()):
        if bbox is None:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color = _PERSON_COLORS[pid % len(_PERSON_COLORS)]
        thickness = 3 if pid == selected_person else 2

        # Bbox rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Corner handles: TL=0, TR=1, BL=2, BR=3
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for corner_idx, (cx, cy) in enumerate(corners):
            is_active = (
                bbox_edit_state is not None
                and bbox_edit_state.get("pid") == pid
                and bbox_edit_state.get("corner") == corner_idx
            )
            if is_active:
                cv2.circle(img, (cx, cy), 12, (0, 255, 255), -1)  # filled yellow
            else:
                cv2.circle(img, (cx, cy), 12, color, 2)  # hollow outline
                # Semi-transparent hit-zone ring
                overlay = img.copy()
                cv2.circle(overlay, (cx, cy), 12, color, -1)
                cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Confidence badge + label
        conf = confidences.get(pid)
        overall = conf.overall if conf else 0.0
        marker = "\u25b6 " if pid == selected_person else ""
        label = f"{marker}Person {pid} ({overall:.2f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img


def _render_confidence_plot(
    overall_array: np.ndarray,
    current_frame: int,
    keyframe_frames: list[int],
    num_frames: int,
    issues: list[dict] | None = None,
) -> plt.Figure:
    """Render confidence heatmap timeline as matplotlib figure.

    Optional issue markers drawn above the confidence heatmap.
    """
    fig, ax = plt.subplots(figsize=(14, 1.2))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    if num_frames > 0 and len(overall_array) > 0:
        colors = plt.cm.RdYlGn(overall_array)
        ax.bar(range(len(overall_array)), np.ones(len(overall_array)),
               width=1.0, color=colors, edgecolor="none")

    # Keyframe diamonds
    for kf in keyframe_frames:
        if 0 <= kf < num_frames:
            ax.plot(kf, 1.08, "D", color="#60a5fa", markersize=5, zorder=5)

    # Issue markers (Phase 1.4)
    if issues:
        for issue in issues:
            f = issue.get("frame", 0)
            if f < 0 or f >= num_frames:
                continue
            itype = issue.get("type", "")
            if itype == "potential_swap":
                ax.plot(f, 1.13, "v", color="#ea4335", markersize=6, zorder=6)
            elif itype == "shape_drift":
                ax.plot(f, 1.13, "s", color="#ff9800", markersize=5, zorder=6)
            elif itype == "track_gap":
                span = issue.get("span")
                if span:
                    ax.axvspan(span[0], span[1], ymin=0.85, ymax=0.95,
                               color="#ff5722", alpha=0.4, zorder=4)

    # Playhead
    ax.axvline(current_frame, color="white", lw=1.5, zorder=10)

    ax.set_xlim(0, max(num_frames, 1))
    ax.set_ylim(0, 1.25)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    return fig


def _get_bboxes_at_frame(session_id: str, frame_idx: int) -> dict[int, np.ndarray]:
    """Get per-person bboxes at a given frame."""
    sd = _SESSION_DATA.get(session_id, {})
    all_bboxes = sd.get("all_bboxes", {})
    result = {}
    for pid, bbox_array in all_bboxes.items():
        if bbox_array is not None and frame_idx < len(bbox_array):
            result[pid] = bbox_array[frame_idx]
    return result


def _get_confidences_at_frame(session_id: str, frame_idx: int) -> dict[int, TrackConfidence]:
    """Get per-person confidence at a given frame."""
    sd = _SESSION_DATA.get(session_id, {})
    all_confs = sd.get("confidences", {})
    result = {}
    for pid, conf_list in all_confs.items():
        if conf_list is not None and frame_idx < len(conf_list):
            result[pid] = conf_list[frame_idx]
    return result


def _save_bbox_corrections(session_id: str, pid: int) -> None:
    """Persist bbox corrections for a person to JSON."""
    sd = _SESSION_DATA.get(session_id, {})
    corrections = sd.get("bbox_corrections", {}).get(pid, {})
    person_dir = sd.get("pid_to_dir", {}).get(pid)
    if person_dir is not None:
        out_path = Path(person_dir) / "bbox_corrections.json"
        data = {
            "person_id": pid,
            "corrections": {str(f): bbox.tolist() for f, bbox in corrections.items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(out_path), "w") as f:
            import json
            json.dump(data, f, indent=2)


def _load_bbox_corrections(person_dir: str) -> dict[int, np.ndarray]:
    """Load bbox corrections from JSON. Returns {frame_idx: np.array([x1,y1,x2,y2])}."""
    import json
    corr_path = Path(person_dir) / "bbox_corrections.json"
    if not corr_path.exists():
        return {}
    try:
        with open(str(corr_path)) as f:
            data = json.load(f)
        return {
            int(frame): np.array(bbox, dtype=np.float32)
            for frame, bbox in data.get("corrections", {}).items()
        }
    except Exception:
        return {}


def _load_person_binding(person_dir: str) -> dict | None:
    meta_path = Path(person_dir) / _PERSON_META_FILENAME
    if not meta_path.exists():
        return None
    try:
        import json
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def _get_identity_track(session_id: str, person_id: int) -> IdentityTrack | None:
    """Get identity track for a person."""
    sd = _SESSION_DATA.get(session_id, {})
    tracks = sd.get("identity_tracks", [])
    for t in tracks:
        if t.person_id == person_id:
            return t
    return None


def _save_identity_track(session_id: str, person_id: int) -> None:
    """Save identity track to JSON."""
    sd = _SESSION_DATA.get(session_id, {})
    pid_to_dir = sd.get("pid_to_dir", {})
    track = _get_identity_track(session_id, person_id)
    if track is None:
        return
    person_dir = pid_to_dir.get(person_id)
    if person_dir is not None:
        track.save_json(Path(person_dir) / "identity_track.json")


def _keyframe_dataframe(session_id: str, person_id: int) -> list[list]:
    """Build keyframe DataFrame rows for a person."""
    track = _get_identity_track(session_id, person_id)
    if track is None:
        return []
    sd = _SESSION_DATA.get(session_id, {})
    fps = sd.get("fps", 30.0)
    rows = []
    for kf in track.keyframes:
        time_str = f"{kf.frame_index / fps:.2f}s"
        conf = kf.confidence.overall if kf.confidence else 0.0
        verified = "✓" if kf.verified else "○"
        rows.append([kf.frame_index, time_str, f"{conf:.2f}", verified])
    return rows


def _person_choices(session_id: str) -> list[str]:
    """Get person dropdown choices."""
    sd = _SESSION_DATA.get(session_id, {})
    tracks = sd.get("identity_tracks", [])
    return [f"Person {t.person_id}" for t in tracks]


# ── Phase 1: Confidence-Guided Review ──


def compute_review_issues(session_id: str) -> list[dict]:
    """Scan all persons for identity problems. Returns sorted issue list.

    Each issue: {
        "type": "low_confidence" | "potential_swap" | "track_gap" | "shape_drift",
        "frame": int,
        "person_id": int,
        "severity": float,  # 0-1, higher = worse
        "description": str,
        "span": (int, int) | None,
    }
    """
    sd = _SESSION_DATA.get(session_id, {})
    identity_tracks = sd.get("identity_tracks", [])
    all_confs = sd.get("confidences", {})
    all_bboxes = sd.get("all_bboxes", {})
    num_frames = sd.get("num_frames", 0)
    issues: list[dict] = []

    for id_track in identity_tracks:
        pid = id_track.person_id
        confs = all_confs.get(pid, [])
        if not confs:
            continue

        # 1. Low-confidence spans
        spans = id_track.low_confidence_spans(confs, threshold=0.4)
        for start, end in spans:
            span_confs = [confs[f].overall for f in range(start, min(end + 1, len(confs)))]
            mean_conf = np.mean(span_confs) if span_confs else 0.0
            mid = (start + end) // 2
            issues.append({
                "type": "low_confidence",
                "frame": mid,
                "person_id": pid,
                "severity": 1.0 - mean_conf,
                "description": f"Person {pid}: low confidence frames {start}-{end} (mean {mean_conf:.2f})",
                "span": (start, end),
            })

        # 3. Track gaps: contiguous False spans > 15 frames in detection_mask
        track_data = None
        pid_to_index = sd.get("pid_to_index", {})
        idx = pid_to_index.get(pid)
        all_tracks = sd.get("all_tracks", [])
        if idx is not None and idx < len(all_tracks):
            track_data = all_tracks[idx]

        if track_data is not None:
            import torch
            det_mask = track_data.get("detection_mask")
            if det_mask is not None:
                if isinstance(det_mask, torch.Tensor):
                    det_mask = det_mask.numpy()
                in_gap = False
                gap_start = 0
                for f in range(len(det_mask)):
                    if not det_mask[f]:
                        if not in_gap:
                            gap_start = f
                            in_gap = True
                    else:
                        if in_gap and f - gap_start > 15:
                            mid = (gap_start + f - 1) // 2
                            issues.append({
                                "type": "track_gap",
                                "frame": mid,
                                "person_id": pid,
                                "severity": min(1.0, (f - gap_start) / 60.0),
                                "description": f"Person {pid}: lost tracking frames {gap_start}-{f - 1} ({f - gap_start} frames)",
                                "span": (gap_start, f - 1),
                            })
                        in_gap = False
                if in_gap and len(det_mask) - gap_start > 15:
                    mid = (gap_start + len(det_mask) - 1) // 2
                    issues.append({
                        "type": "track_gap",
                        "frame": mid,
                        "person_id": pid,
                        "severity": min(1.0, (len(det_mask) - gap_start) / 60.0),
                        "description": f"Person {pid}: lost tracking frames {gap_start}-{len(det_mask) - 1}",
                        "span": (gap_start, len(det_mask) - 1),
                    })

    # Build ShapeReIdentifier from tracks (used by swaps + drift checks)
    reid = ShapeReIdentifier(identity_tracks)

    # 2. Potential swaps: check pairs at frames with bbox overlap > 0.3
    pids = [t.person_id for t in identity_tracks]
    if len(pids) >= 2:
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                pid_a, pid_b = pids[i], pids[j]
                bboxes_a = all_bboxes.get(pid_a)
                bboxes_b = all_bboxes.get(pid_b)
                if bboxes_a is None or bboxes_b is None:
                    continue

                n = min(len(bboxes_a), len(bboxes_b), num_frames)
                checked_swap = False
                for f in range(n):
                    iou = compute_bbox_iou(bboxes_a[f], bboxes_b[f])
                    if iou > 0.3 and not checked_swap:
                        # Check for swap at this overlap point
                        per_person_betas = {}
                        for check_pid in [pid_a, pid_b]:
                            track = _get_identity_track(session_id, check_pid)
                            if track and track.established_betas is not None:
                                per_person_betas[check_pid] = track.established_betas
                        swaps = reid.detect_swap(f, per_person_betas)
                        if swaps:
                            issues.append({
                                "type": "potential_swap",
                                "frame": f,
                                "person_id": pid_a,
                                "severity": 0.9,
                                "description": f"Potential swap between Person {pid_a} and Person {pid_b} at frame {f}",
                                "span": None,
                            })
                            checked_swap = True

    # 4. Shape drift
    for id_track in identity_tracks:
        pid = id_track.person_id
        if id_track.established_betas is None:
            continue
        # Get per-frame betas from SMPL params if available
        pid_to_index = sd.get("pid_to_index", {})
        idx = pid_to_index.get(pid)
        person_dirs = sd.get("person_dirs", [])
        if idx is not None and idx < len(person_dirs):
            pdir = Path(person_dirs[idx])
            pt_files = list(pdir.rglob("hmr4d_results.pt"))
            if pt_files:
                try:
                    from smplx_to_bvh import extract_gvhmr_params
                    params = extract_gvhmr_params(str(pt_files[0]))
                    per_frame_betas = params["betas"]
                    drift_frames = reid.verify_track_consistency(pid, per_frame_betas)
                    for f in drift_frames:
                        issues.append({
                            "type": "shape_drift",
                            "frame": f,
                            "person_id": pid,
                            "severity": 0.6,
                            "description": f"Person {pid}: shape drift at frame {f}",
                            "span": None,
                        })
                except Exception:
                    pass

    # Sort by severity descending
    issues.sort(key=lambda x: x["severity"], reverse=True)
    return issues


# ── Phase 2: Visual ReID Gallery ──


def _extract_thumbnails(
    video_path: str,
    bboxes: np.ndarray,
    keyframes: list[IdentityKeyframe],
    confidences: list[TrackConfidence],
    max_thumbnails: int = 3,
    target_size: tuple[int, int] = (128, 192),
    padding_ratio: float = 0.2,
) -> list[tuple[int, np.ndarray]]:
    """Extract cropped thumbnails at highest-confidence keyframe locations.
    Returns [(frame_idx, rgb_array), ...] sorted by frame order.
    """
    # Find best keyframe frames by confidence
    kf_confs = []
    for kf in keyframes:
        if kf.confidence:
            kf_confs.append((kf.frame_index, kf.confidence.overall))
    if not kf_confs:
        # Fall back to evenly spaced frames from high-confidence regions
        overall = confidence_to_array(confidences) if confidences else np.zeros(0)
        if len(overall) > 0:
            top_frames = np.argsort(overall)[-max_thumbnails:]
            kf_confs = [(int(f), float(overall[f])) for f in top_frames]
        else:
            return []

    kf_confs.sort(key=lambda x: x[1], reverse=True)
    selected = kf_confs[:max_thumbnails]

    thumbnails = []
    cap = cv2.VideoCapture(video_path)
    tw, th = target_size

    for frame_idx, _ in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame_idx >= len(bboxes):
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = bboxes[frame_idx]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Add padding
        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_rgb.shape[1], x2 + pad_x)
        y2 = min(frame_rgb.shape[0], y2 + pad_y)

        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (tw, th))
        thumbnails.append((frame_idx, crop))

    cap.release()
    thumbnails.sort(key=lambda x: x[0])
    return thumbnails


def _build_gallery_items(session_id: str, selected_person: int) -> list[tuple[np.ndarray, str]]:
    """Build gallery items. Selected person's thumbnails first (highlighted in caption).
    Then other persons' thumbnails for reference.
    """
    sd = _SESSION_DATA.get(session_id, {})
    all_thumbs = sd.get("thumbnails", {})
    identity_tracks = sd.get("identity_tracks", [])
    items = []

    # Selected person first
    thumbs = all_thumbs.get(selected_person, [])
    for frame_idx, img in thumbs:
        items.append((img, f">> Person {selected_person} (f{frame_idx})"))

    # Other persons
    for track in identity_tracks:
        pid = track.person_id
        if pid == selected_person:
            continue
        other_thumbs = all_thumbs.get(pid, [])
        for frame_idx, img in other_thumbs[:2]:  # limit to 2 per other person
            items.append((img, f"Person {pid} (f{frame_idx})"))

    return items


# ── Public API ──


def init_panel_state(
    video_path: str,
    identity_tracks: list[IdentityTrack],
    all_tracks: list[dict],
    person_dirs: list,
    fps: float = 30.0,
    output_dir: str = "",
    pipeline_params: dict | None = None,
) -> dict:
    """Initialize session data from pipeline results.

    Args:
        pipeline_params: dict with keys "static_cam", "use_dpvo" for reprocessing config.

    Returns a lightweight gr.State dict.
    """
    import torch

    session_id = str(uuid.uuid4())

    # Extract per-person bbox arrays and confidence lists
    all_bboxes = {}
    all_confidences = {}

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    str_person_dirs = [str(p) for p in person_dirs]
    pid_to_dir = {}
    pid_to_index = {}
    for i, track in enumerate(all_tracks):
        pid = track.get("track_id", i)
        if i < len(str_person_dirs):
            pid_to_dir[pid] = str_person_dirs[i]
        pid_to_index[pid] = i
    for pdir in str_person_dirs:
        binding = _load_person_binding(pdir)
        if binding is None:
            continue
        pid = binding.get("track_id")
        source_index = binding.get("source_index")
        if pid is not None:
            pid_to_dir[int(pid)] = pdir
        if pid is not None and source_index is not None:
            pid_to_index[int(pid)] = int(source_index)

    for i, track in enumerate(all_tracks):
        pid = track.get("track_id", i)
        boxes = track.get("bbx_xyxy")
        if boxes is not None:
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.numpy()
            all_bboxes[pid] = np.array(boxes, dtype=np.float32)

    # Keep a copy of original bboxes before corrections
    original_bboxes = {pid: arr.copy() for pid, arr in all_bboxes.items()}

    # Load and apply bbox corrections per person
    bbox_corrections: dict[int, dict[int, np.ndarray]] = {}
    for pdir in str_person_dirs:
        binding = _load_person_binding(pdir)
        pid = int(binding["track_id"]) if binding and "track_id" in binding else None
        if pid is None:
            continue
        corr = _load_bbox_corrections(pdir)
        if corr:
            bbox_corrections[pid] = corr
            # Apply corrections on top of all_bboxes
            if pid in all_bboxes:
                for frame_idx, bbox in corr.items():
                    if frame_idx < len(all_bboxes[pid]):
                        all_bboxes[pid][frame_idx] = bbox

    # Load confidence from identity tracks or recompute
    for id_track in identity_tracks:
        pid = id_track.person_id
        # Try loading confidence CSV
        person_dir = pid_to_dir.get(pid)
        if person_dir is not None:
            csv_path = Path(person_dir) / "confidence.csv"
            if csv_path.exists():
                confs = _load_confidence_csv(str(csv_path), num_frames)
                all_confidences[pid] = confs
                continue
        # Fallback: synthesize from keyframes
        confs = []
        for f in range(num_frames):
            kf = id_track.get_nearest(f) if id_track.keyframes else None
            if kf and kf.confidence:
                confs.append(kf.confidence)
            else:
                confs.append(TrackConfidence())
        all_confidences[pid] = confs

    _SESSION_DATA[session_id] = {
        "video_path": video_path,
        "identity_tracks": identity_tracks,
        "all_bboxes": all_bboxes,
        "original_bboxes": original_bboxes,
        "bbox_corrections": bbox_corrections,
        "dirty_persons": set(),
        "confidences": all_confidences,
        "num_frames": num_frames,
        "fps": fps,
        "frame_cache": _LRUFrameCache(),
        "output_dir": output_dir,
        "person_dirs": str_person_dirs,
        "all_tracks": all_tracks,
        "pipeline_params": pipeline_params or {},
        "img_width": img_width,
        "img_height": img_height,
        "pid_to_dir": pid_to_dir,
        "pid_to_index": pid_to_index,
        "operation_log": [],
        "archived_persons": {},
        "thumbnails": {},
        "review_issues": [],
        "review_index": -1,
    }

    # Extract thumbnails for gallery (Phase 2)
    sd = _SESSION_DATA[session_id]
    for id_track in identity_tracks:
        pid = id_track.person_id
        bboxes = all_bboxes.get(pid)
        confs = all_confidences.get(pid, [])
        if bboxes is not None:
            try:
                thumbs = _extract_thumbnails(video_path, bboxes, id_track.keyframes, confs)
                sd["thumbnails"][pid] = thumbs
            except Exception:
                sd["thumbnails"][pid] = []

    # Compute review issues (Phase 1)
    try:
        sd["review_issues"] = compute_review_issues(session_id)
    except Exception:
        sd["review_issues"] = []

    # Load per-person SMPL params and correction tracks for pose correction
    smplx_params = {}
    correction_tracks = {}
    str_person_dirs = [str(p) for p in person_dirs]

    for pid, pdir in pid_to_dir.items():
        pdir_path = Path(pdir)

        # Load GVHMR params (hmr4d_results.pt)
        pt_files = list(pdir_path.rglob("hmr4d_results.pt"))
        if pt_files:
            try:
                from smplx_to_bvh import extract_gvhmr_params
                params = extract_gvhmr_params(str(pt_files[0]))
                smplx_params[pid] = params
            except Exception as e:
                print(f"[identity_panel] Failed to load params for person {pid}: {e}")

        # Load existing correction tracks
        corr_path = pdir_path / "pose_corrections.json"
        if corr_path.exists():
            try:
                correction_tracks[pid] = CorrectionTrack.load_json(corr_path)
            except Exception:
                correction_tracks[pid] = CorrectionTrack(person_id=pid)
        else:
            correction_tracks[pid] = CorrectionTrack(person_id=pid)

    # Initialize pose correction session
    init_pose_session(
        session_id=session_id,
        smplx_params=smplx_params,
        correction_tracks=correction_tracks,
        person_dirs=str_person_dirs,
        video_path=video_path,
        num_frames=num_frames,
        fps=fps,
        pid_to_dir=pid_to_dir,
    )

    return {
        "session_id": session_id,
        "current_frame": 0,
        "selected_person": identity_tracks[0].person_id if identity_tracks else 0,
    }


def _load_confidence_csv(csv_path: str, num_frames: int) -> list[TrackConfidence]:
    """Load per-frame confidence from CSV sidecar."""
    confs = []
    try:
        with open(csv_path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 8:
                    confs.append(TrackConfidence(
                        detection_score=float(parts[3]),
                        visible_keypoints=float(parts[4]),
                        bbox_overlap=float(parts[5]),
                        shape_consistency=float(parts[6]),
                        motion_consistency=float(parts[7]),
                    ))
    except Exception:
        pass
    # Pad to num_frames if needed
    while len(confs) < num_frames:
        confs.append(TrackConfidence())
    return confs[:num_frames]


def _reprocess_btn_label(session_id: str) -> str:
    """Dynamic label for the reprocess button showing dirty count."""
    sd = _SESSION_DATA.get(session_id, {})
    n = len(sd.get("dirty_persons", set()))
    return f"Reprocess All ({n} dirty)"


# ── Callback Functions ──


def update_frame_display(frame_idx, state):
    """Main callback: update frame preview, confidence, timeline on frame change."""
    if state is None:
        return None, "", "", "", "", "", "", None, []

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return None, "", "", "", "", "", "", None, []

    frame_idx = int(frame_idx)
    selected_person = state.get("selected_person", 0)
    video_path = sd["video_path"]
    num_frames = sd["num_frames"]
    fps = sd["fps"]

    # Extract frame
    frame = _extract_frame(video_path, frame_idx, session_id)
    if frame is None:
        return None, "", "", "", "", "", "", None, []

    # Get bboxes and confidences at this frame
    bboxes = _get_bboxes_at_frame(session_id, frame_idx)
    frame_confs = _get_confidences_at_frame(session_id, frame_idx)

    # Render frame with bboxes drawn directly
    bbox_edit_state = state.get("bbox_edit_state") if state else None
    rendered_frame = _render_frame_with_bboxes(
        frame, bboxes, frame_confs, selected_person, bbox_edit_state
    )

    # Confidence breakdown for selected person
    conf = frame_confs.get(selected_person, TrackConfidence())
    det_str = f"{conf.detection_score:.2f}"
    vis_str = f"{conf.visible_keypoints:.2f}"
    ovl_str = f"{conf.bbox_overlap:.2f}"
    shp_str = f"{conf.shape_consistency:.2f}"
    mot_str = f"{conf.motion_consistency:.2f}"
    overall_str = f"{conf.overall:.2f}"

    # Confidence timeline
    all_confs = sd["confidences"].get(selected_person, [])
    overall_array = confidence_to_array(all_confs) if all_confs else np.zeros(num_frames)
    track = _get_identity_track(session_id, selected_person)
    kf_frames = [kf.frame_index for kf in track.keyframes] if track else []
    # Filter issues for selected person
    all_issues = sd.get("review_issues", [])
    person_issues = [i for i in all_issues if i.get("person_id") == selected_person]
    fig = _render_confidence_plot(overall_array, frame_idx, kf_frames, num_frames, issues=person_issues)

    # Keyframe DataFrame
    kf_data = _keyframe_dataframe(session_id, selected_person)

    # Frame info text
    time_str = f"{frame_idx / fps:.2f}s" if fps > 0 else ""
    frame_label = f"Frame {frame_idx}/{num_frames - 1}  |  {time_str}"

    # Update state
    state["current_frame"] = frame_idx

    return (
        rendered_frame,  # gr.Image (np.ndarray)
        det_str, vis_str, ovl_str, shp_str, mot_str, overall_str,
        fig,       # confidence plot
        kf_data,   # keyframe dataframe
    )


def on_person_change(person_str, frame_idx, state):
    """Callback when person dropdown changes. Returns state + display_outputs + gallery."""
    if not person_str or state is None:
        return state, *update_frame_display(frame_idx, state), []

    # Parse "Person N" -> N
    try:
        pid = int(person_str.split()[-1])
    except (ValueError, IndexError):
        pid = 0

    state["selected_person"] = pid
    session_id = state.get("session_id", "")
    gallery = _build_gallery_items(session_id, pid)
    return (state, *update_frame_display(frame_idx, state), gallery)


def on_verify(frame_idx, state):
    """Mark keyframe at current frame as verified (or create one).

    Also marks the person dirty so reprocessing picks up verified identity data.
    """
    if state is None:
        return state, gr.update(), *update_frame_display(0, state)

    session_id = state["session_id"]
    pid = state["selected_person"]
    frame_idx = int(frame_idx)
    track = _get_identity_track(session_id, pid)
    if track is None:
        return state, gr.update(), *update_frame_display(frame_idx, state)

    # Find existing keyframe at this frame
    existing = [kf for kf in track.keyframes if kf.frame_index == frame_idx]
    if existing:
        existing[0].verified = True
    else:
        # Create new verified keyframe from current bbox/confidence
        bboxes = _get_bboxes_at_frame(session_id, frame_idx)
        confs = _get_confidences_at_frame(session_id, frame_idx)
        bbox = bboxes.get(pid, np.zeros(4))
        conf = confs.get(pid, TrackConfidence())
        sd = _SESSION_DATA[session_id]
        track.add_keyframe(
            frame_index=frame_idx,
            bbox=bbox,
            betas=track.established_betas if track.established_betas is not None else np.zeros(10),
            confidence=conf,
            verified=True,
            timestamp=frame_idx / sd.get("fps", 30.0),
        )

    sd = _SESSION_DATA.get(session_id, {})
    sd.setdefault("dirty_persons", set()).add(pid)
    _save_identity_track(session_id, pid)
    return (state, gr.update(value=_reprocess_btn_label(session_id)),
            *update_frame_display(frame_idx, state))


def on_add_keyframe(frame_idx, state):
    """Add a keyframe at current frame and mark person dirty for reprocessing."""
    if state is None:
        return state, gr.update(), *update_frame_display(0, state)

    session_id = state["session_id"]
    pid = state["selected_person"]
    frame_idx = int(frame_idx)
    track = _get_identity_track(session_id, pid)
    if track is None:
        return state, gr.update(), *update_frame_display(frame_idx, state)

    bboxes = _get_bboxes_at_frame(session_id, frame_idx)
    confs = _get_confidences_at_frame(session_id, frame_idx)
    bbox = bboxes.get(pid, np.zeros(4))
    conf = confs.get(pid, TrackConfidence())
    sd = _SESSION_DATA[session_id]

    track.add_keyframe(
        frame_index=frame_idx,
        bbox=bbox,
        betas=track.established_betas if track.established_betas is not None else np.zeros(10),
        confidence=conf,
        verified=False,
        timestamp=frame_idx / sd.get("fps", 30.0),
    )
    sd.setdefault("dirty_persons", set()).add(pid)
    _save_identity_track(session_id, pid)
    return (state, gr.update(value=_reprocess_btn_label(session_id)),
            *update_frame_display(frame_idx, state))


def on_remove_keyframe(frame_idx, state):
    """Remove keyframe at current frame."""
    if state is None:
        return state, *update_frame_display(0, state)

    session_id = state["session_id"]
    pid = state["selected_person"]
    frame_idx = int(frame_idx)
    track = _get_identity_track(session_id, pid)
    if track is None:
        return state, *update_frame_display(frame_idx, state)

    track.remove_keyframe(frame_idx)
    _save_identity_track(session_id, pid)
    return (state, *update_frame_display(frame_idx, state))


def on_swap_ids(frame_idx, person_a_str, person_b_str, state):
    """Swap person_id assignments between two people from current frame onward."""
    if state is None:
        return state, gr.update(), *update_frame_display(0, state)

    session_id = state["session_id"]
    frame_idx = int(frame_idx)

    try:
        pid_a = int(person_a_str.split()[-1])
        pid_b = int(person_b_str.split()[-1])
    except (ValueError, IndexError):
        return state, gr.update(), *update_frame_display(frame_idx, state)

    if pid_a == pid_b:
        return state, gr.update(), *update_frame_display(frame_idx, state)

    track_a = _get_identity_track(session_id, pid_a)
    track_b = _get_identity_track(session_id, pid_b)
    if track_a is None or track_b is None:
        return state, gr.update(), *update_frame_display(frame_idx, state)

    sd = _SESSION_DATA[session_id]

    # Swap bboxes from frame_idx onward
    bboxes_a = sd["all_bboxes"].get(pid_a)
    bboxes_b = sd["all_bboxes"].get(pid_b)
    if bboxes_a is not None and bboxes_b is not None:
        n = min(len(bboxes_a), len(bboxes_b))
        for f in range(frame_idx, n):
            bboxes_a[f], bboxes_b[f] = bboxes_b[f].copy(), bboxes_a[f].copy()

    # Swap confidences from frame_idx onward
    confs_a = sd["confidences"].get(pid_a, [])
    confs_b = sd["confidences"].get(pid_b, [])
    n = min(len(confs_a), len(confs_b))
    for f in range(frame_idx, n):
        confs_a[f], confs_b[f] = confs_b[f], confs_a[f]

    # Add verified keyframes at the swap point
    for track, pid in [(track_a, pid_a), (track_b, pid_b)]:
        bboxes = _get_bboxes_at_frame(session_id, frame_idx)
        confs = _get_confidences_at_frame(session_id, frame_idx)
        track.add_keyframe(
            frame_index=frame_idx,
            bbox=bboxes.get(pid, np.zeros(4)),
            betas=track.established_betas if track.established_betas is not None else np.zeros(10),
            confidence=confs.get(pid, TrackConfidence()),
            verified=True,
            timestamp=frame_idx / sd.get("fps", 30.0),
            metadata={"swap_with": pid_b if pid == pid_a else pid_a},
        )

    # Re-establish identity betas for both tracks (Phase 4.1)
    track_a.establish_identity()
    track_b.establish_identity()

    # Log swap for undo (Phase 4.1)
    sd.setdefault("operation_log", []).append({
        "operation": "swap",
        "frame": frame_idx,
        "pid_a": pid_a,
        "pid_b": pid_b,
    })

    # Mark both persons as dirty — SMPL params need reprocessing to match swapped bboxes
    sd.setdefault("dirty_persons", set()).update({pid_a, pid_b})

    _save_identity_track(session_id, pid_a)
    _save_identity_track(session_id, pid_b)
    return (state, gr.update(value=_reprocess_btn_label(session_id)),
            *update_frame_display(frame_idx, state))


def on_prev_keyframe(frame_idx, state):
    """Navigate to previous keyframe for selected person."""
    if state is None:
        return int(frame_idx)
    session_id = state["session_id"]
    pid = state["selected_person"]
    track = _get_identity_track(session_id, pid)
    if track is None:
        return int(frame_idx)

    frame_idx = int(frame_idx)
    prev_frames = [kf.frame_index for kf in track.keyframes if kf.frame_index < frame_idx]
    return max(prev_frames) if prev_frames else frame_idx


def on_next_keyframe(frame_idx, state):
    """Navigate to next keyframe for selected person."""
    if state is None:
        return int(frame_idx)
    session_id = state["session_id"]
    pid = state["selected_person"]
    track = _get_identity_track(session_id, pid)
    if track is None:
        return int(frame_idx)

    frame_idx = int(frame_idx)
    next_frames = [kf.frame_index for kf in track.keyframes if kf.frame_index > frame_idx]
    return min(next_frames) if next_frames else frame_idx


def on_df_select(evt: gr.SelectData, state):
    """Navigate to frame when keyframe DataFrame row is clicked."""
    if state is None or evt is None:
        return 0
    session_id = state["session_id"]
    pid = state["selected_person"]
    track = _get_identity_track(session_id, pid)
    if track is None:
        return 0

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < len(track.keyframes):
        return track.keyframes[row_idx].frame_index
    return 0


# ── Bbox Editing Callbacks ──


def on_frame_click(evt: gr.SelectData, frame_idx, state):
    """Two-click bbox corner editing: first click selects corner, second moves it."""
    if state is None or evt is None:
        return None, "", state, gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return None, "", state, gr.update()

    frame_idx = int(frame_idx)
    click_x, click_y = evt.index[0], evt.index[1]
    selected_person = state.get("selected_person", 0)
    bbox_edit_state = state.get("bbox_edit_state")

    if bbox_edit_state is None:
        # First click: find nearest corner handle across all visible bboxes
        bboxes = _get_bboxes_at_frame(session_id, frame_idx)
        best_dist = 40.0  # threshold in pixels
        best_pid = None
        best_corner = None

        for pid, bbox in bboxes.items():
            if bbox is None:
                continue
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for ci, (cx, cy) in enumerate(corners):
                dist = np.sqrt((click_x - cx) ** 2 + (click_y - cy) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid
                    best_corner = ci

        if best_pid is not None:
            state["bbox_edit_state"] = {"pid": best_pid, "corner": best_corner, "frame": frame_idx}
            status = f"Corner selected for Person {best_pid} \u2014 click new position"
        else:
            status = ""
    else:
        # Second click: move corner to new position
        pid = bbox_edit_state["pid"]
        corner = bbox_edit_state["corner"]
        edit_frame = bbox_edit_state["frame"]

        if pid in sd["all_bboxes"] and edit_frame < len(sd["all_bboxes"][pid]):
            bbox = sd["all_bboxes"][pid][edit_frame].copy()
            img_w = sd.get("img_width", 1920)
            img_h = sd.get("img_height", 1080)

            new_x = max(0, min(float(click_x), img_w - 1))
            new_y = max(0, min(float(click_y), img_h - 1))

            # Update corner: TL=0, TR=1, BL=2, BR=3
            if corner == 0:
                bbox[0], bbox[1] = new_x, new_y
            elif corner == 1:
                bbox[2], bbox[1] = new_x, new_y
            elif corner == 2:
                bbox[0], bbox[3] = new_x, new_y
            elif corner == 3:
                bbox[2], bbox[3] = new_x, new_y

            # Ensure x1 < x2, y1 < y2
            x1, y1, x2, y2 = bbox
            bbox = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                            dtype=np.float32)

            # Enforce minimum bbox size (20px)
            if bbox[2] - bbox[0] < 20:
                bbox[2] = bbox[0] + 20
            if bbox[3] - bbox[1] < 20:
                bbox[3] = bbox[1] + 20

            # Update session bboxes
            sd["all_bboxes"][pid][edit_frame] = bbox

            # Store correction
            pid_corrections = sd.setdefault("bbox_corrections", {}).setdefault(pid, {})
            pid_corrections[edit_frame] = bbox.copy()

            # Mark person as dirty
            sd.setdefault("dirty_persons", set()).add(pid)

            _save_bbox_corrections(session_id, pid)

            n_edited = len(pid_corrections)
            status = f"{n_edited} frames edited for Person {pid}"
        else:
            status = ""

        state["bbox_edit_state"] = None

    # Re-render frame
    frame = _extract_frame(sd["video_path"], frame_idx, session_id)
    if frame is not None:
        bboxes = _get_bboxes_at_frame(session_id, frame_idx)
        confs = _get_confidences_at_frame(session_id, frame_idx)
        img = _render_frame_with_bboxes(frame, bboxes, confs, selected_person,
                                        state.get("bbox_edit_state"))
    else:
        img = None

    return img, status, state, gr.update(value=_reprocess_btn_label(session_id))


def on_cancel_bbox_edit(state):
    """Cancel an in-progress bbox corner edit."""
    if state is not None:
        state["bbox_edit_state"] = None
    return "", state


def on_interpolate_bboxes(state):
    """Interpolate bbox corrections between edited keyframes for selected person."""
    if state is None:
        return "", state, gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session data", state, gr.update()

    pid = state.get("selected_person", 0)
    corrections = sd.get("bbox_corrections", {}).get(pid, {})
    original = sd.get("original_bboxes", {}).get(pid)

    if not corrections or original is None:
        return "No corrections to interpolate", state, gr.update()

    from multi_person_split import interpolate_bbox_corrections
    updated = interpolate_bbox_corrections(original, corrections)

    # Apply interpolated bboxes
    sd["all_bboxes"][pid] = updated

    # Update correction dict with all interpolated frames
    sorted_frames = sorted(corrections.keys())
    if len(sorted_frames) >= 2:
        first_f = sorted_frames[0]
        last_f = sorted_frames[-1]
        new_corrections = sd["bbox_corrections"].setdefault(pid, {})
        for f in range(first_f, last_f + 1):
            if f not in new_corrections and f < len(original):
                # Only store if different from original
                if not np.allclose(updated[f], original[f], atol=0.5):
                    new_corrections[f] = updated[f].copy()

    sd.setdefault("dirty_persons", set()).add(pid)
    _save_bbox_corrections(session_id, pid)
    n_total = len(sd.get("bbox_corrections", {}).get(pid, {}))
    return (f"Interpolated: {n_total} frames corrected for Person {pid}", state,
            gr.update(value=_reprocess_btn_label(session_id)))


def on_apply_keyframes(state):
    """Apply identity keyframe bboxes as corrections and mark all persons dirty.

    Takes the bboxes stored in each person's identity keyframes, writes them
    as bbox corrections, interpolates between them, and marks each person
    dirty for reprocessing. This is the bridge between "I keyframed the right
    person at these frames" and "re-run the solve with corrected assignments".
    """
    if state is None:
        return "", state, gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session data", state, gr.update()

    from multi_person_split import interpolate_bbox_corrections

    status_parts = []
    identity_tracks = sd.get("identity_tracks", [])

    for id_track in identity_tracks:
        pid = id_track.person_id
        original = sd.get("original_bboxes", {}).get(pid)
        if original is None:
            continue

        # Extract bbox corrections from identity keyframes
        kf_corrections = {}
        for kf in id_track.keyframes:
            f = kf.frame_index
            if f < len(original) and kf.bbox is not None:
                kf_corrections[f] = np.array(kf.bbox, dtype=np.float32)

        if not kf_corrections:
            continue

        # Store as bbox corrections
        sd.setdefault("bbox_corrections", {}).setdefault(pid, {}).update(kf_corrections)

        # Interpolate between keyframes
        all_corrections = sd["bbox_corrections"][pid]
        updated = interpolate_bbox_corrections(original, all_corrections)
        sd["all_bboxes"][pid] = updated

        # Expand interpolated frames into corrections
        sorted_frames = sorted(all_corrections.keys())
        if len(sorted_frames) >= 2:
            first_f = sorted_frames[0]
            last_f = sorted_frames[-1]
            for f in range(first_f, last_f + 1):
                if f not in all_corrections and f < len(original):
                    if not np.allclose(updated[f], original[f], atol=0.5):
                        all_corrections[f] = updated[f].copy()

        sd.setdefault("dirty_persons", set()).add(pid)
        _save_bbox_corrections(session_id, pid)
        n = len(all_corrections)
        status_parts.append(f"Person {pid}: {n} frames from {len(kf_corrections)} keyframes")

    if not status_parts:
        return "No keyframes with bboxes found", state, gr.update()

    status = "Applied keyframes:\n" + "\n".join(status_parts)
    return status, state, gr.update(value=_reprocess_btn_label(session_id))


# ── Phase 1 Callbacks: Confidence-Guided Review ──


def on_refresh_issues(state):
    """Rescan all persons for identity problems."""
    if state is None:
        return "No session"
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session data"

    issues = compute_review_issues(session_id)
    sd["review_issues"] = issues
    sd["review_index"] = -1

    if not issues:
        return "No issues found"

    # Summarize by type
    counts: dict[str, int] = {}
    for iss in issues:
        counts[iss["type"]] = counts.get(iss["type"], 0) + 1
    parts = []
    for t, c in counts.items():
        label = t.replace("_", " ")
        parts.append(f"{c} {label}")
    return f"{len(issues)} issues: {', '.join(parts)}"


def on_next_issue(frame_idx, state):
    """Navigate to next review issue."""
    if state is None:
        return int(frame_idx), "No session", gr.update()
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return int(frame_idx), "No session data", gr.update()

    issues = sd.get("review_issues", [])
    if not issues:
        return int(frame_idx), "No issues — run Scan first", gr.update()

    idx = sd.get("review_index", -1) + 1
    if idx >= len(issues):
        idx = 0
    sd["review_index"] = idx

    issue = issues[idx]
    state["selected_person"] = issue["person_id"]
    choices = _person_choices(session_id)
    person_str = f"Person {issue['person_id']}"

    summary = f"Issue {idx + 1}/{len(issues)}: {issue['description']}"
    return (
        issue["frame"],
        summary,
        gr.update(value=person_str if person_str in choices else choices[0] if choices else None),
    )


def on_prev_issue(frame_idx, state):
    """Navigate to previous review issue."""
    if state is None:
        return int(frame_idx), "No session", gr.update()
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return int(frame_idx), "No session data", gr.update()

    issues = sd.get("review_issues", [])
    if not issues:
        return int(frame_idx), "No issues — run Scan first", gr.update()

    idx = sd.get("review_index", 0) - 1
    if idx < 0:
        idx = len(issues) - 1
    sd["review_index"] = idx

    issue = issues[idx]
    state["selected_person"] = issue["person_id"]
    choices = _person_choices(session_id)
    person_str = f"Person {issue['person_id']}"

    summary = f"Issue {idx + 1}/{len(issues)}: {issue['description']}"
    return (
        issue["frame"],
        summary,
        gr.update(value=person_str if person_str in choices else choices[0] if choices else None),
    )


# ── Phase 3 Callbacks: Track Merge ──


def on_merge_tracks(person_a_str, merge_target_str, state, progress=gr.Progress()):
    """Merge merge_target into selected person (primary)."""
    empty_display = update_frame_display(0, state)

    if state is None or not merge_target_str:
        return (state, "Select a merge target", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return (state, "No session data", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    try:
        pid_a = int(person_a_str.split()[-1])
        pid_b = int(merge_target_str.split()[-1])
    except (ValueError, IndexError):
        return (state, "Invalid person selection", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    if pid_a == pid_b:
        return (state, "Cannot merge person with itself", gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), *empty_display)

    pid_to_index = sd.get("pid_to_index", {})
    idx_a = pid_to_index.get(pid_a)
    idx_b = pid_to_index.get(pid_b)
    if idx_a is None or idx_b is None:
        return (state, "Person not found", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    from multi_person_split import merge_persons

    progress(0.1, desc=f"Merging Person {pid_b} into Person {pid_a}...")

    try:
        merge_record = merge_persons(
            video_path=sd["video_path"],
            primary_index=idx_a,
            secondary_index=idx_b,
            person_dirs=sd["person_dirs"],
            all_tracks=sd["all_tracks"],
            slam_path=str(Path(sd["output_dir"]) / "shared_slam.pt"),
            masks_dir=str(Path(sd["output_dir"]) / "masks"),
            output_dir=sd["output_dir"],
            static_cam=sd.get("pipeline_params", {}).get("static_cam", False),
            use_dpvo=sd.get("pipeline_params", {}).get("use_dpvo", False),
            progress_callback=lambda f, m: progress(0.1 + f * 0.8, desc=m),
        )
    except Exception as e:
        return (state, f"Merge failed: {e}", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    # Update session data: remove secondary
    # Remove secondary from identity_tracks
    sd["identity_tracks"] = [t for t in sd["identity_tracks"] if t.person_id != pid_b]

    # Update primary's identity track and confidences
    if merge_record.get("merged_identity_track"):
        for i, t in enumerate(sd["identity_tracks"]):
            if t.person_id == pid_a:
                sd["identity_tracks"][i] = merge_record["merged_identity_track"]
                break
    if merge_record.get("merged_confidences"):
        sd["confidences"][pid_a] = merge_record["merged_confidences"]

    # Remove secondary from all data
    sd["all_bboxes"].pop(pid_b, None)
    sd["original_bboxes"].pop(pid_b, None)
    sd["confidences"].pop(pid_b, None)
    sd["thumbnails"].pop(pid_b, None)
    sd["pid_to_dir"].pop(pid_b, None)
    sd["pid_to_index"].pop(pid_b, None)

    # Rebuild pid_to_index (indices may have shifted)
    for i, track in enumerate(sd["all_tracks"]):
        pid = track.get("track_id", i)
        sd["pid_to_index"][pid] = i

    # Log for undo
    sd["operation_log"].append(merge_record)

    # Re-extract thumbnails for merged person
    bboxes = sd["all_bboxes"].get(pid_a)
    confs = sd["confidences"].get(pid_a, [])
    merged_track = _get_identity_track(session_id, pid_a)
    if bboxes is not None and merged_track:
        try:
            sd["thumbnails"][pid_a] = _extract_thumbnails(
                sd["video_path"], bboxes, merged_track.keyframes, confs,
            )
        except Exception:
            pass

    # Rescan issues
    try:
        sd["review_issues"] = compute_review_issues(session_id)
        sd["review_index"] = -1
    except Exception:
        pass

    progress(1.0, desc="Merge complete")

    # Update UI
    choices = _person_choices(session_id)
    frame_idx = state.get("current_frame", 0)
    gallery = _build_gallery_items(session_id, pid_a)
    issue_summary = f"{len(sd.get('review_issues', []))} issues" if sd.get("review_issues") else "No issues"

    display = update_frame_display(frame_idx, state)
    return (
        state,
        f"Merged Person {pid_b} into Person {pid_a}",
        gr.update(value=_reprocess_btn_label(session_id)),
        gr.update(choices=choices, value=f"Person {pid_a}" if f"Person {pid_a}" in choices else None),
        gr.update(choices=choices),
        gr.update(choices=choices),
        gallery,
        issue_summary,
        *display,
    )


def on_undo_last(state):
    """Single-level undo. Handles merge and swap operations."""
    if state is None:
        return state, "No session"

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return state, "No session data"

    log = sd.get("operation_log", [])
    if not log:
        return state, "Nothing to undo"

    record = log.pop()
    op_type = record.get("operation")

    if op_type == "swap":
        # Swap is its own inverse — just re-run the swap at the same frame
        pid_a = record["pid_a"]
        pid_b = record["pid_b"]
        frame = record["frame"]
        on_swap_ids(frame, f"Person {pid_a}", f"Person {pid_b}", state)
        # Remove the new swap entry from the log (undo shouldn't stack)
        if sd["operation_log"] and sd["operation_log"][-1].get("operation") == "swap":
            sd["operation_log"].pop()
        return state, f"Undid swap of Person {pid_a} and Person {pid_b}"

    elif op_type == "merge":
        # Restore archived secondary dir
        import shutil
        archived_dir = record.get("archived_dir")
        secondary_index = record.get("secondary_index")
        primary_index = record.get("primary_index")
        secondary_tid = record.get("secondary_track_id")

        if archived_dir and Path(archived_dir).exists():
            # Restore secondary dir
            original_dir = archived_dir.replace(".archived", "")
            if not Path(original_dir).exists():
                Path(archived_dir).rename(original_dir)

        # Restore primary bboxes
        primary_bboxes = record.get("primary_original_bboxes")
        all_tracks = sd.get("all_tracks", [])
        if primary_bboxes is not None and primary_index < len(all_tracks):
            import torch
            all_tracks[primary_index]["bbx_xyxy"] = torch.from_numpy(primary_bboxes)

        return state, f"Undid merge (Person {secondary_tid} restored). Re-run pipeline to complete."

    return state, f"Unknown operation type: {op_type}"


# ── Phase 5 Callbacks: Track Split ──


def on_split_track(frame_idx, state, progress=gr.Progress()):
    """Split selected person at current frame into two tracks."""
    empty_display = update_frame_display(0, state)

    if state is None:
        return (state, "No session", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return (state, "No session data", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    pid = state.get("selected_person", 0)
    frame_idx = int(frame_idx)
    pid_to_index = sd.get("pid_to_index", {})
    idx = pid_to_index.get(pid)
    if idx is None:
        return (state, "Person not found", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    from multi_person_split import split_person

    progress(0.1, desc=f"Splitting Person {pid} at frame {frame_idx}...")

    try:
        split_record = split_person(
            video_path=sd["video_path"],
            person_index=idx,
            split_frame=frame_idx,
            person_dirs=sd["person_dirs"],
            all_tracks=sd["all_tracks"],
            slam_path=str(Path(sd["output_dir"]) / "shared_slam.pt"),
            masks_dir=str(Path(sd["output_dir"]) / "masks"),
            output_dir=sd["output_dir"],
            static_cam=sd.get("pipeline_params", {}).get("static_cam", False),
            use_dpvo=sd.get("pipeline_params", {}).get("use_dpvo", False),
            progress_callback=lambda f, m: progress(0.1 + f * 0.8, desc=m),
        )
    except Exception as e:
        return (state, f"Split failed: {e}", gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), *empty_display)

    new_pid = split_record["new_person_id"]
    new_dir = split_record["new_person_dir"]

    # Add new person to session data
    sd["person_dirs"].append(new_dir)
    new_track = sd["all_tracks"][-1]  # split_person appended it
    sd["pid_to_dir"][new_pid] = new_dir
    sd["pid_to_index"][new_pid] = len(sd["all_tracks"]) - 1

    # Create identity track for new person
    new_bboxes = new_track["bbx_xyxy"]
    if hasattr(new_bboxes, "numpy"):
        new_bboxes = new_bboxes.numpy()
    sd["all_bboxes"][new_pid] = new_bboxes
    sd["original_bboxes"][new_pid] = new_bboxes.copy()

    # Update original person's bboxes
    orig_track = sd["all_tracks"][idx]
    orig_bboxes = orig_track["bbx_xyxy"]
    if hasattr(orig_bboxes, "numpy"):
        orig_bboxes = orig_bboxes.numpy()
    sd["all_bboxes"][pid] = orig_bboxes
    sd["original_bboxes"][pid] = orig_bboxes.copy()

    # Create identity tracks for both
    result_a = split_record.get("result_a", {})
    result_b = split_record.get("result_b", {})

    if result_a.get("identity_track"):
        for i, t in enumerate(sd["identity_tracks"]):
            if t.person_id == pid:
                sd["identity_tracks"][i] = result_a["identity_track"]
                break
    if result_a.get("confidences"):
        sd["confidences"][pid] = result_a["confidences"]

    new_id_track = result_b.get("identity_track", IdentityTrack(person_id=new_pid))
    sd["identity_tracks"].append(new_id_track)
    if result_b.get("confidences"):
        sd["confidences"][new_pid] = result_b["confidences"]

    # Extract thumbnails for both
    for p in [pid, new_pid]:
        bx = sd["all_bboxes"].get(p)
        co = sd["confidences"].get(p, [])
        tr = _get_identity_track(session_id, p)
        if bx is not None and tr:
            try:
                sd["thumbnails"][p] = _extract_thumbnails(sd["video_path"], bx, tr.keyframes, co)
            except Exception:
                pass

    sd["operation_log"].append(split_record)

    # Rescan issues
    try:
        sd["review_issues"] = compute_review_issues(session_id)
        sd["review_index"] = -1
    except Exception:
        pass

    progress(1.0, desc="Split complete")

    choices = _person_choices(session_id)
    gallery = _build_gallery_items(session_id, pid)
    issue_summary = f"{len(sd.get('review_issues', []))} issues" if sd.get("review_issues") else "No issues"
    display = update_frame_display(frame_idx, state)

    return (
        state,
        f"Split Person {pid} at frame {frame_idx} — new Person {new_pid}",
        gr.update(value=_reprocess_btn_label(session_id)),
        gr.update(choices=choices, value=f"Person {pid}"),
        gr.update(choices=choices),
        gr.update(choices=choices),
        gallery,
        issue_summary,
        *display,
    )


def on_reprocess_all_dirty(state, progress=gr.Progress(track_tqdm=False),
                           regenerate_preview: bool = False):
    """Reprocess all persons with pending bbox edits."""
    _extra = (gr.update(),) if regenerate_preview else ()

    if state is None:
        return state, "No session", gr.update(), *_extra, *update_frame_display(0, state)

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return state, "No session data", gr.update(), *_extra, *update_frame_display(0, state)

    dirty = sd.get("dirty_persons", set())
    if not dirty:
        return (state, "No dirty persons",
                gr.update(value=_reprocess_btn_label(session_id)),
                *_extra,
                *update_frame_display(state.get("current_frame", 0), state))

    from multi_person_split import reprocess_person

    all_tracks = sd.get("all_tracks", [])
    slam_path = str(Path(sd["output_dir"]) / "shared_slam.pt")
    masks_dir = str(Path(sd["output_dir"]) / "masks")
    pp = sd.get("pipeline_params", {})
    pid_to_dir = sd.get("pid_to_dir", {})
    pid_to_index = sd.get("pid_to_index", {})

    dirty_list = sorted(dirty)
    total = len(dirty_list)
    status_lines = []

    for idx, pid in enumerate(dirty_list):
        progress((idx) / total, desc=f"Reprocessing person {pid} ({idx + 1}/{total})...")

        person_dir = pid_to_dir.get(pid)
        p_index = pid_to_index.get(pid)
        if person_dir is None or p_index is None:
            status_lines.append(f"Person {pid}: skipped (invalid index)")
            continue

        # Build updated bboxes array
        updated_bboxes = sd["all_bboxes"].get(pid)
        if updated_bboxes is None:
            status_lines.append(f"Person {pid}: skipped (no bboxes)")
            continue

        try:
            result = reprocess_person(
                video_path=sd["video_path"],
                person_index=p_index,
                person_dir=person_dir,
                updated_bboxes=updated_bboxes,
                all_tracks=all_tracks,
                slam_path=slam_path,
                masks_dir=masks_dir,
                static_cam=pp.get("static_cam", False),
                use_dpvo=pp.get("use_dpvo", False),
            )

            # Refresh session data for this person
            if result and result.get("pt_path"):
                from smplx_to_bvh import extract_gvhmr_params
                from pose_correction import CorrectionTrack
                from pose_correction_panel import _POSE_SESSION

                params = extract_gvhmr_params(result["pt_path"])

                # Update confidences
                if result.get("confidences"):
                    sd["confidences"][pid] = result["confidences"]

                # Update identity track
                if result.get("identity_track"):
                    for i, t in enumerate(sd["identity_tracks"]):
                        if t.person_id == pid:
                            sd["identity_tracks"][i] = result["identity_track"]
                            break

                # Refresh pose session
                pose_sd = _POSE_SESSION.get(session_id)
                if pose_sd is not None:
                    pose_sd["smplx_params"][pid] = params
                    # Preserve existing corrections
                    if pid not in pose_sd["correction_tracks"]:
                        pose_sd["correction_tracks"][pid] = CorrectionTrack(person_id=pid)

                status_lines.append(f"Person {pid}: reprocessed OK")
            else:
                status_lines.append(f"Person {pid}: reprocess failed")
        except Exception as e:
            status_lines.append(f"Person {pid}: error \u2014 {e}")

    dirty.clear()

    # Regenerate scene preview with updated results
    scene_preview_update = gr.update()
    if regenerate_preview:
        try:
            from multi_person_split import render_multi_person_incam
            assembly_dir = Path(sd["output_dir"]) / "assembly"
            assembly_dir.mkdir(parents=True, exist_ok=True)
            scene_path = str(assembly_dir / "scene_preview.mp4")
            Path(scene_path).unlink(missing_ok=True)

            # Ensure person_dirs matches all_tracks order
            person_dirs_ordered = []
            for t in all_tracks:
                tid = t.get("track_id", t.get("id"))
                d = pid_to_dir.get(tid)
                if d:
                    person_dirs_ordered.append(d)

            progress(0.9, desc="Regenerating scene preview...")
            render_multi_person_incam(
                video_path=sd["video_path"],
                person_dirs=person_dirs_ordered,
                all_tracks=all_tracks,
                output_path=scene_path,
                fps=sd.get("fps", 30.0),
            )
            scene_preview_update = gr.update(value=scene_path)
        except Exception as e:
            status_lines.append(f"Scene preview failed: {e}")
            scene_preview_update = gr.update()

    progress(1.0, desc="Reprocessing complete")

    frame_idx = state.get("current_frame", 0)
    status = "\n".join(status_lines)
    return (state, status, gr.update(value=_reprocess_btn_label(session_id)),
            *((scene_preview_update,) if regenerate_preview else ()),
            *update_frame_display(frame_idx, state))


# ── Transport helpers ──

def _nav_first():
    return 0

def _nav_last(state):
    if state is None:
        return 0
    sd = _SESSION_DATA.get(state.get("session_id", ""), {})
    return max(0, sd.get("num_frames", 1) - 1)

def _nav_back10(frame_idx):
    return max(0, int(frame_idx) - 10)

def _nav_fwd10(frame_idx, state):
    if state is None:
        return int(frame_idx)
    sd = _SESSION_DATA.get(state.get("session_id", ""), {})
    return min(sd.get("num_frames", 1) - 1, int(frame_idx) + 10)

def _nav_back1(frame_idx):
    return max(0, int(frame_idx) - 1)

def _nav_fwd1(frame_idx, state):
    if state is None:
        return int(frame_idx)
    sd = _SESSION_DATA.get(state.get("session_id", ""), {})
    return min(sd.get("num_frames", 1) - 1, int(frame_idx) + 1)


# ── Build Panel ──


def build_identity_panel(scene_preview_video=None) -> dict[str, Any]:
    """Construct the Identity Inspector Gradio components.

    Args:
        scene_preview_video: optional gr.Video component to update after reprocessing.

    Returns dict of component references for wiring in gvhmr_gui.py.
    """
    panel_state = gr.State(value=None)

    with gr.Row():
        person_dropdown = gr.Dropdown(
            label="Person",
            choices=[],
            interactive=True,
            scale=1,
        )
        verify_btn = gr.Button("Verify", size="sm", scale=0, min_width=80)
        add_kf_btn = gr.Button("Add Keyframe", size="sm", scale=0, min_width=110)
        remove_kf_btn = gr.Button("Remove Keyframe", size="sm", scale=0, min_width=130)
        swap_btn = gr.Button("Swap IDs", size="sm", scale=0, min_width=90)
        swap_target = gr.Dropdown(
            label="Swap with",
            choices=[],
            interactive=True,
            scale=1,
            visible=True,
        )

    # Phase 3: Merge / Undo / Split row
    with gr.Row():
        merge_target = gr.Dropdown(
            label="Merge into selected", choices=[], interactive=True, scale=1,
        )
        merge_btn = gr.Button("Merge Tracks", size="sm", scale=0, min_width=120,
                              variant="secondary")
        split_btn = gr.Button("Split Here", size="sm", scale=0, min_width=100)
        undo_btn = gr.Button("Undo Last", size="sm", scale=0, min_width=100)
        merge_status = gr.Textbox(label="", value="", interactive=False, scale=2)

    # Phase 2: Gallery
    reid_gallery = gr.Gallery(
        label="Person Reference",
        columns=6,
        rows=1,
        height=160,
        object_fit="contain",
        show_label=True,
    )

    # CSS to prevent native browser image drag (otherwise misclicks
    # drag a thumbnail of the video frame around instead of editing)
    gr.HTML(
        "<style>"
        "#bbox_frame_preview img { "
        "  -webkit-user-drag: none; "
        "  user-select: none; "
        "  pointer-events: auto; "
        "}"
        "</style>"
    )

    # Frame preview (full width)
    frame_image = gr.Image(
        label="Frame Preview (click bbox corner to edit)",
        interactive=False,
        type="numpy",
        elem_id="bbox_frame_preview",
    )

    # Timeline controls right under the frame
    with gr.Row():
        first_btn = gr.Button("|<", size="sm", scale=0, min_width=40)
        back10_btn = gr.Button("<<", size="sm", scale=0, min_width=40)
        back1_btn = gr.Button("<", size="sm", scale=0, min_width=40)
        frame_label = gr.Textbox(
            label="Frame",
            value="Frame 0/0",
            interactive=False,
            scale=1,
        )
        fwd1_btn = gr.Button(">", size="sm", scale=0, min_width=40)
        fwd10_btn = gr.Button(">>", size="sm", scale=0, min_width=40)
        last_btn = gr.Button(">|", size="sm", scale=0, min_width=40)
        prev_kf_btn = gr.Button("◇<", size="sm", scale=0, min_width=50)
        next_kf_btn = gr.Button("◇>", size="sm", scale=0, min_width=50)

    frame_slider = gr.Slider(
        minimum=0,
        maximum=1,
        step=1,
        value=0,
        label="Frame",
        interactive=True,
    )

    confidence_plot = gr.Plot(label="Confidence Timeline")

    # Phase 1: Review navigation row
    with gr.Row():
        review_summary = gr.Textbox(
            label="Issues", value="Run Scan to detect problems",
            interactive=False, scale=2,
        )
        prev_issue_btn = gr.Button("< Prev Issue", size="sm", scale=0, min_width=110)
        next_issue_btn = gr.Button("Next Issue >", size="sm", scale=0, min_width=110)
        refresh_issues_btn = gr.Button("Scan", size="sm", scale=0, min_width=70)

    # Keyframes + confidence breakdown side by side
    with gr.Row():
        with gr.Column(scale=2):
            kf_dataframe = gr.DataFrame(
                headers=["Frame", "Time", "Confidence", "Verified"],
                datatype=["number", "str", "str", "str"],
                interactive=False,
                row_count=(5, "dynamic"),
            )

        with gr.Column(scale=1):
            gr.Markdown("**Confidence Breakdown**")
            with gr.Row():
                conf_detection = gr.Textbox(label="Detection", interactive=False, scale=1)
                conf_visibility = gr.Textbox(label="Visibility", interactive=False, scale=1)
            with gr.Row():
                conf_overlap = gr.Textbox(label="Overlap", interactive=False, scale=1)
                conf_shape = gr.Textbox(label="Shape", interactive=False, scale=1)
            with gr.Row():
                conf_motion = gr.Textbox(label="Motion", interactive=False, scale=1)
                conf_overall = gr.Textbox(label="Overall", interactive=False, scale=1)

    # Bbox corrections
    with gr.Row():
        bbox_edit_status = gr.Textbox(
            label="Bbox Edit Status", value="", interactive=False, scale=2,
        )
        cancel_edit_btn = gr.Button("Cancel Edit", size="sm", scale=0, min_width=100)
        interpolate_btn = gr.Button("Interpolate Bboxes", size="sm", scale=0, min_width=140)
        apply_kf_btn = gr.Button(
            "Apply Keyframes", size="sm", scale=0, min_width=140,
        )
        reprocess_btn = gr.Button(
            "Reprocess All (0 dirty)", variant="primary", size="sm", scale=0, min_width=180,
        )
    reprocess_status = gr.Textbox(
        label="Reprocess Status", value="", interactive=False, lines=2,
    )

    # ── Pose Corrector sub-panel (embedded below timeline) ──
    pose_panel = build_pose_correction_panel(frame_slider, panel_state)

    # ── All display outputs (used by update_frame_display) ──
    display_outputs = [
        frame_image,
        conf_detection, conf_visibility, conf_overlap,
        conf_shape, conf_motion, conf_overall,
        confidence_plot,
        kf_dataframe,
    ]

    # ── Wire callbacks ──

    # Frame slider → update display
    frame_slider.change(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
    )

    # Person dropdown → update display + gallery
    person_dropdown.change(
        fn=on_person_change,
        inputs=[person_dropdown, frame_slider, panel_state],
        outputs=[panel_state] + display_outputs + [reid_gallery],
    )

    # Transport buttons → update slider (which triggers display update)
    first_btn.click(fn=_nav_first, inputs=[], outputs=[frame_slider])
    last_btn.click(fn=_nav_last, inputs=[panel_state], outputs=[frame_slider])
    back10_btn.click(fn=_nav_back10, inputs=[frame_slider], outputs=[frame_slider])
    fwd10_btn.click(fn=_nav_fwd10, inputs=[frame_slider, panel_state], outputs=[frame_slider])
    back1_btn.click(fn=_nav_back1, inputs=[frame_slider], outputs=[frame_slider])
    fwd1_btn.click(fn=_nav_fwd1, inputs=[frame_slider, panel_state], outputs=[frame_slider])

    # Keyframe navigation
    prev_kf_btn.click(
        fn=on_prev_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider],
    )
    next_kf_btn.click(
        fn=on_next_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider],
    )

    # Identity operations
    verify_btn.click(
        fn=on_verify,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state, reprocess_btn] + display_outputs,
    )
    add_kf_btn.click(
        fn=on_add_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state, reprocess_btn] + display_outputs,
    )
    remove_kf_btn.click(
        fn=on_remove_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state] + display_outputs,
    )
    swap_btn.click(
        fn=on_swap_ids,
        inputs=[frame_slider, person_dropdown, swap_target, panel_state],
        outputs=[panel_state, reprocess_btn] + display_outputs,
    )

    # Keyframe DataFrame click → navigate
    kf_dataframe.select(
        fn=on_df_select,
        inputs=[panel_state],
        outputs=[frame_slider],
    )

    # Bbox click editing
    frame_image.select(
        fn=on_frame_click,
        inputs=[frame_slider, panel_state],
        outputs=[frame_image, bbox_edit_status, panel_state, reprocess_btn],
    )

    cancel_edit_btn.click(
        fn=on_cancel_bbox_edit,
        inputs=[panel_state],
        outputs=[bbox_edit_status, panel_state],
    )

    interpolate_btn.click(
        fn=on_interpolate_bboxes,
        inputs=[panel_state],
        outputs=[bbox_edit_status, panel_state, reprocess_btn],
    )

    apply_kf_btn.click(
        fn=on_apply_keyframes,
        inputs=[panel_state],
        outputs=[bbox_edit_status, panel_state, reprocess_btn],
    )

    # Phase 1: Review navigation callbacks
    refresh_issues_btn.click(
        fn=on_refresh_issues,
        inputs=[panel_state],
        outputs=[review_summary],
    )
    next_issue_btn.click(
        fn=on_next_issue,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider, review_summary, person_dropdown],
    )
    prev_issue_btn.click(
        fn=on_prev_issue,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider, review_summary, person_dropdown],
    )

    # Phase 2: Gallery updates on person change (piggyback via existing callback)
    # Gallery is updated separately via merge/split/reprocess outputs

    # Phase 3: Merge callbacks
    _merge_split_outputs = [
        panel_state, merge_status, reprocess_btn,
        person_dropdown, swap_target, merge_target,
        reid_gallery, review_summary,
    ] + display_outputs

    merge_btn.click(
        fn=on_merge_tracks,
        inputs=[person_dropdown, merge_target, panel_state],
        outputs=_merge_split_outputs,
    )

    # Phase 3: Undo callback
    undo_btn.click(
        fn=on_undo_last,
        inputs=[panel_state],
        outputs=[panel_state, merge_status],
    )

    # Phase 5: Split callback
    split_btn.click(
        fn=on_split_track,
        inputs=[frame_slider, panel_state],
        outputs=_merge_split_outputs,
    )

    _reprocess_outputs = [panel_state, reprocess_status, reprocess_btn]
    if scene_preview_video is not None:
        _reprocess_outputs.append(scene_preview_video)
    _reprocess_outputs += display_outputs

    reprocess_btn.click(
        fn=lambda state, prog=gr.Progress(track_tqdm=False): on_reprocess_all_dirty(
            state, progress=prog, regenerate_preview=scene_preview_video is not None,
        ),
        inputs=[panel_state],
        outputs=_reprocess_outputs,
    )

    return {
        "state": panel_state,
        "person_dropdown": person_dropdown,
        "swap_target": swap_target,
        "frame_slider": frame_slider,
        "frame_image": frame_image,
        "confidence_plot": confidence_plot,
        "kf_dataframe": kf_dataframe,
        "frame_label": frame_label,
        "bbox_edit_status": bbox_edit_status,
        "reprocess_btn": reprocess_btn,
        "reprocess_status": reprocess_status,
        "conf_detection": conf_detection,
        "conf_visibility": conf_visibility,
        "conf_overlap": conf_overlap,
        "conf_shape": conf_shape,
        "conf_motion": conf_motion,
        "conf_overall": conf_overall,
        "pose_panel": pose_panel,
        # Phase 1: Review
        "review_summary": review_summary,
        # Phase 2: Gallery
        "reid_gallery": reid_gallery,
        # Phase 3: Merge / Split / Undo
        "merge_target": merge_target,
        "merge_status": merge_status,
    }


def populate_panel(
    state_dict: dict,
) -> tuple:
    """Generate initial values for panel components after init_panel_state.

    Returns tuple of Gradio updates for:
    (state, person_dropdown, swap_target, frame_slider, frame_label,
     + 9 display_outputs, review_summary, reid_gallery, merge_target)
    """
    if state_dict is None:
        return (None,) * 17

    session_id = state_dict.get("session_id", "")
    sd = _SESSION_DATA.get(session_id, {})
    num_frames = sd.get("num_frames", 1)
    choices = _person_choices(session_id)
    selected_pid = state_dict.get("selected_person", 0)

    # Initial display at frame 0
    display = update_frame_display(0, state_dict)

    # Review summary (Phase 1)
    issues = sd.get("review_issues", [])
    if issues:
        counts: dict[str, int] = {}
        for iss in issues:
            counts[iss["type"]] = counts.get(iss["type"], 0) + 1
        parts = [f"{c} {t.replace('_', ' ')}" for t, c in counts.items()]
        review_text = f"{len(issues)} issues: {', '.join(parts)}"
    else:
        review_text = "No issues found"

    # Gallery (Phase 2)
    gallery = _build_gallery_items(session_id, selected_pid)

    return (
        state_dict,                                     # state
        gr.update(choices=choices, value=choices[0] if choices else None),  # person_dropdown
        gr.update(choices=choices, value=choices[1] if len(choices) > 1 else None),  # swap_target
        gr.update(maximum=max(num_frames - 1, 1), value=0),  # frame_slider
        f"Frame 0/{num_frames - 1}" if num_frames > 0 else "Frame 0/0",  # frame_label
        *display,                                       # 9 display outputs
        review_text,                                    # review_summary
        gallery,                                        # reid_gallery
        gr.update(choices=choices),                     # merge_target
    )
