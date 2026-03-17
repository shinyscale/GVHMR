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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

from identity_confidence import TrackConfidence, confidence_to_array
from identity_tracking import IdentityTrack, IdentityKeyframe
from pose_correction import CorrectionTrack
from pose_correction_panel import (
    build_pose_correction_panel,
    init_pose_session,
    on_pose_frame_change,
)

matplotlib.use("Agg")

# ── Module-level session data (heavy, not serialized through Gradio) ──

_SESSION_DATA: dict[str, Any] = {}
# Keys: video_path, identity_tracks, all_bboxes, original_bboxes,
#        bbox_corrections, dirty_persons, confidences, num_frames,
#        fps, frame_cache, output_dir, person_dirs, all_tracks,
#        pipeline_params, img_width, img_height

_FRAME_CACHE_SIZE = 200
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


def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=8):
    """Draw a dashed line on img from pt1 to pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    dist = max(1, int((dx**2 + dy**2) ** 0.5))
    for i in range(0, dist, dash_len * 2):
        s = i / dist
        e = min((i + dash_len) / dist, 1.0)
        sx, sy = int(x1 + dx * s), int(y1 + dy * s)
        ex, ey = int(x1 + dx * e), int(y1 + dy * e)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


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


_READAHEAD = 15  # cache this many frames ahead after each seek


def _extract_frame(video_path: str, frame_idx: int, session_id: str) -> np.ndarray | None:
    """Extract a single frame from video with read-ahead caching.

    After the expensive H.264 seek, reads the next _READAHEAD frames too
    (sequential reads are nearly free). This makes forward stepping instant.
    """
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

    num_frames = sd.get("num_frames", 0)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    result = None
    for i in range(_READAHEAD + 1):
        f = frame_idx + i
        if f >= num_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cache.put(f, rgb)
        if i == 0:
            result = rgb

    cap.release()
    return result


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
    inactive_tracks: list[dict] | None = None,
    frame_idx: int = 0,
    show_all_tracks: bool = False,
) -> np.ndarray:
    """Draw bboxes, corner handles, confidence badges, and labels via cv2.

    Returns a single np.ndarray (RGB) for gr.Image.
    """
    img = frame.copy()

    # Draw inactive tracks as dashed semi-transparent overlays
    if show_all_tracks and inactive_tracks:
        for track in inactive_tracks:
            tid = track["track_id"]
            mask = track.get("detection_mask")
            boxes = track.get("bbx_xyxy")
            if boxes is None or frame_idx >= len(boxes):
                continue
            bbox = boxes[frame_idx]
            if hasattr(bbox, "numpy"):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue
            color = (128, 128, 128)
            has_det = mask[frame_idx].item() if mask is not None and frame_idx < len(mask) else False
            # Dashed rectangle via short line segments
            for start, end in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                               ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
                _draw_dashed_line(img, start, end, color, thickness=2, dash_len=8)
            # Detection indicator dot
            dot_color = (0, 180, 0) if has_det else (0, 0, 180)
            cv2.circle(img, (x2 - 6, y1 + 6), 4, dot_color, -1)
            # Label
            label = f"Track {tid} (inactive)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, 0.4, 1)
            cv2.rectangle(img, (x1, y2), (x1 + tw + 4, y2 + th + baseline + 4), (80, 80, 80), -1)
            cv2.putText(img, label, (x1 + 2, y2 + th + 2), font, 0.4,
                        (200, 200, 200), 1, cv2.LINE_AA)

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
        label = f"{marker}ID {pid} ({overall:.2f})"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2), font, font_scale,
                    (255, 255, 255), 1, cv2.LINE_AA)

    return img


_CONFIDENCE_TIMELINE_HEIGHT = 48   # px
_CONFIDENCE_TIMELINE_MIN_WIDTH = 1600  # px — render wide, Gradio scales to fit


def _render_confidence_timeline(
    overall_array: np.ndarray,
    current_frame: int,
    keyframe_frames: list[int],
    num_frames: int,
    has_real_data: bool = True,
    height: int = _CONFIDENCE_TIMELINE_HEIGHT,
) -> np.ndarray:
    """Render confidence heatmap timeline as an RGB numpy image.

    Width = max(num_frames, _CONFIDENCE_TIMELINE_MIN_WIDTH) so the image is
    always wide enough for Gradio to stretch to full column width.
    Returns np.ndarray (H, W, 3) suitable for gr.Image(type="numpy").
    """
    width = max(num_frames, _CONFIDENCE_TIMELINE_MIN_WIDTH, 1)
    img = np.full((height, width, 3), 26, dtype=np.uint8)  # #1a1a2e background

    if num_frames <= 0:
        return img

    diamond_h = 12  # top region for keyframe diamonds
    bar_h = height - diamond_h

    if not has_real_data:
        # Gray bar with "No confidence data" text
        img[diamond_h:, :] = 77  # gray
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "No confidence data"
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        tx = (width - tw) // 2
        ty = diamond_h + (bar_h + th) // 2
        cv2.putText(img, text, (tx, ty), font, 0.5, (136, 136, 136), 1, cv2.LINE_AA)
    elif len(overall_array) > 0:
        # RdYlGn colormap — each pixel column maps to a frame
        cmap = plt.cm.RdYlGn
        colors_rgba = cmap(overall_array)  # (N, 4) float 0-1
        colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)  # (N, 3)
        for x in range(width):
            f = min(int(x * num_frames / width), len(colors_rgb) - 1)
            img[diamond_h:, x] = colors_rgb[f]

    # Keyframe diamonds (downward-pointing triangles at top)
    for kf in keyframe_frames:
        if 0 <= kf < num_frames:
            kx = int(kf * width / num_frames)
            sz = 5
            pts = np.array([[kx, diamond_h - 1], [kx - sz, 1], [kx + sz, 1]], dtype=np.int32)
            cv2.fillPoly(img, [pts], (96, 165, 250))  # #60a5fa

    # Playhead — white vertical line
    px = int(current_frame * width / num_frames)
    px = max(0, min(px, width - 1))
    cv2.line(img, (px, 0), (px, height - 1), (255, 255, 255), 2)

    return img


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
    return [f"ID {t.person_id}" for t in tracks]


# ── Public API ──


def init_panel_state(
    video_path: str,
    identity_tracks: list[IdentityTrack],
    all_tracks: list[dict],
    person_dirs: list,
    fps: float = 30.0,
    output_dir: str = "",
    pipeline_params: dict | None = None,
    inactive_tracks: list[dict] | None = None,
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
        "inactive_tracks": inactive_tracks or [],
        "crossing_spans": {},  # {person_id: [(start, end), ...]}
    }

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
    return f"Reprocess All ({n} IDs dirty)"


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
    show_all = state.get("show_all_tracks", False) if state else False
    rendered_frame = _render_frame_with_bboxes(
        frame, bboxes, frame_confs, selected_person, bbox_edit_state,
        inactive_tracks=sd.get("inactive_tracks"),
        frame_idx=frame_idx,
        show_all_tracks=show_all,
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
    # Check if any frame has real detection data (not all stubs)
    has_real_data = any(c.detection_score > 0 for c in all_confs) if all_confs else False
    track = _get_identity_track(session_id, selected_person)
    kf_frames = [kf.frame_index for kf in track.keyframes] if track else []
    fig = _render_confidence_timeline(overall_array, frame_idx, kf_frames, num_frames,
                                      has_real_data=has_real_data)

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
    """Callback when person dropdown changes."""
    if not person_str or state is None:
        return state, *update_frame_display(frame_idx, state)

    # Parse "ID N" -> N
    try:
        pid = int(person_str.split()[-1])
    except (ValueError, IndexError):
        pid = 0

    state["selected_person"] = pid
    return (state, *update_frame_display(frame_idx, state))


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
            status = f"Corner selected for ID {best_pid} \u2014 click new position"
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
            status = f"{n_edited} frames edited for ID {pid}"
        else:
            status = ""

        state["bbox_edit_state"] = None

    # Re-render frame
    frame = _extract_frame(sd["video_path"], frame_idx, session_id)
    if frame is not None:
        bboxes = _get_bboxes_at_frame(session_id, frame_idx)
        confs = _get_confidences_at_frame(session_id, frame_idx)
        show_all = state.get("show_all_tracks", False) if state else False
        img = _render_frame_with_bboxes(frame, bboxes, confs, selected_person,
                                        state.get("bbox_edit_state"),
                                        inactive_tracks=sd.get("inactive_tracks"),
                                        frame_idx=frame_idx,
                                        show_all_tracks=show_all)
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
    return (f"Interpolated: {n_total} frames corrected for ID {pid}", state,
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
        status_parts.append(f"ID {pid}: {n} frames from {len(kf_corrections)} keyframes")

    if not status_parts:
        return "No keyframes with bboxes found", state, gr.update()

    status = "Applied keyframes:\n" + "\n".join(status_parts)
    return status, state, gr.update(value=_reprocess_btn_label(session_id))


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
        progress((idx) / total, desc=f"Reprocessing ID {pid} ({idx + 1}/{total})...")

        person_dir = pid_to_dir.get(pid)
        p_index = pid_to_index.get(pid)
        if person_dir is None or p_index is None:
            status_lines.append(f"ID {pid}: skipped (invalid index)")
            continue

        # Build updated bboxes array
        updated_bboxes = sd["all_bboxes"].get(pid)
        if updated_bboxes is None:
            status_lines.append(f"ID {pid}: skipped (no bboxes)")
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

                status_lines.append(f"ID {pid}: reprocessed OK")
            else:
                status_lines.append(f"ID {pid}: reprocess failed")
        except Exception as e:
            status_lines.append(f"ID {pid}: error \u2014 {e}")

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


# ── ReID Gallery ──


def _extract_person_thumbnails(
    session_id: str,
    person_id: int,
    max_thumbs: int = 4,
    thumb_size: int = 128,
) -> list[np.ndarray]:
    """Extract thumbnail crops of a person from their highest-confidence frames."""
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return []

    confs = sd.get("confidences", {}).get(person_id, [])
    bboxes = sd.get("all_bboxes", {}).get(person_id)
    if not confs or bboxes is None:
        return []

    # Find top-confidence frames
    scored = [(f, c.overall if hasattr(c, "overall") else 0.0) for f, c in enumerate(confs)]
    scored.sort(key=lambda x: -x[1])

    # Pick well-spaced frames from top candidates
    num_frames = sd.get("num_frames", len(confs))
    min_spacing = max(10, num_frames // (max_thumbs * 3))
    selected = []
    for frame_idx, score in scored:
        if score < 0.3:
            break
        if all(abs(frame_idx - s) >= min_spacing for s in selected):
            selected.append(frame_idx)
        if len(selected) >= max_thumbs:
            break

    if not selected:
        return []
    selected.sort()

    thumbnails = []
    for f in selected:
        frame = _extract_frame(sd["video_path"], f, session_id)
        if frame is None:
            continue
        bbox = bboxes[f] if f < len(bboxes) else None
        if bbox is None:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            continue
        crop = frame[y1:y2, x1:x2]
        # Resize to uniform height, preserving aspect ratio
        crop_h, crop_w = crop.shape[:2]
        scale = thumb_size / crop_h
        new_w = max(1, int(crop_w * scale))
        thumb = cv2.resize(crop, (new_w, thumb_size), interpolation=cv2.INTER_AREA)
        thumbnails.append(thumb)

    return thumbnails


def _build_gallery_image(
    session_id: str,
    selected_person: int,
    thumb_size: int = 128,
) -> np.ndarray | None:
    """Build a horizontal gallery strip showing thumbnails for each active person.

    Selected person shown first with a colored border.
    """
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return None

    all_tracks = sd.get("all_tracks", [])
    if not all_tracks:
        return None

    strips = []
    person_order = [selected_person] + [
        t["track_id"] for t in all_tracks if t["track_id"] != selected_person
    ]

    for pid in person_order:
        thumbs = _extract_person_thumbnails(session_id, pid, max_thumbs=3, thumb_size=thumb_size)
        if not thumbs:
            continue

        color = _PERSON_COLORS[pid % len(_PERSON_COLORS)]
        border = 3 if pid == selected_person else 1

        # Add label bar on top
        max_w = max(t.shape[1] for t in thumbs)
        label_h = 20
        label_bar = np.zeros((label_h, max_w, 3), dtype=np.uint8)
        label_bar[:] = color
        label = f"ID {pid}"
        cv2.putText(label_bar, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Stack thumbs horizontally
        row_h = thumb_size
        row = np.zeros((row_h, 0, 3), dtype=np.uint8)
        for thumb in thumbs:
            # Pad to same height if needed
            if thumb.shape[0] != row_h:
                pad = np.zeros((row_h, thumb.shape[1], 3), dtype=np.uint8)
                pad[:thumb.shape[0]] = thumb
                thumb = pad
            row = np.concatenate([row, thumb], axis=1)

        # Pad label bar to match row width
        if label_bar.shape[1] < row.shape[1]:
            pad = np.zeros((label_h, row.shape[1] - label_bar.shape[1], 3), dtype=np.uint8)
            pad[:] = color
            label_bar = np.concatenate([label_bar, pad], axis=1)
        elif label_bar.shape[1] > row.shape[1]:
            label_bar = label_bar[:, :row.shape[1]]

        person_strip = np.concatenate([label_bar, row], axis=0)

        # Add border for selected person
        if pid == selected_person:
            cv2.rectangle(person_strip, (0, 0),
                         (person_strip.shape[1] - 1, person_strip.shape[0] - 1),
                         color, border)

        strips.append(person_strip)

    if not strips:
        return None

    # Stack strips vertically, padding to same width
    max_w = max(s.shape[1] for s in strips)
    padded = []
    for s in strips:
        if s.shape[1] < max_w:
            pad = np.zeros((s.shape[0], max_w - s.shape[1], 3), dtype=np.uint8)
            s = np.concatenate([s, pad], axis=1)
        padded.append(s)

    # Add 2px gap between strips
    gap = np.zeros((2, max_w, 3), dtype=np.uint8)
    result_parts = []
    for i, s in enumerate(padded):
        if i > 0:
            result_parts.append(gap)
        result_parts.append(s)

    return np.concatenate(result_parts, axis=0)


# ── Review Scanner ──


@dataclass
class ReviewIssue:
    """A flagged issue found by the review scanner."""
    frame: int
    person_id: int
    issue_type: str  # "low_confidence", "potential_swap", "shape_drift", "track_gap"
    description: str
    severity: float  # 0-1


def compute_review_issues(
    confidences: dict[int, list],
    all_tracks: list[dict],
    num_frames: int,
    low_conf_threshold: float = 0.4,
    gap_threshold: int = 30,
) -> list[ReviewIssue]:
    """Scan all persons for issues that need human review.

    Returns issues sorted by frame number.
    """
    from dataclasses import dataclass as _dc  # already imported at module level
    issues = []

    for pid, confs in confidences.items():
        if not confs:
            continue

        # Low-confidence spans
        span_start = None
        for f in range(len(confs)):
            overall = confs[f].overall if hasattr(confs[f], "overall") else 0.0
            if overall < low_conf_threshold:
                if span_start is None:
                    span_start = f
            else:
                if span_start is not None and f - span_start >= 5:
                    mid = (span_start + f) // 2
                    issues.append(ReviewIssue(
                        frame=mid, person_id=pid,
                        issue_type="low_confidence",
                        description=f"Low confidence span frames {span_start}-{f-1} ({f - span_start} frames)",
                        severity=0.7,
                    ))
                span_start = None
        if span_start is not None and len(confs) - span_start >= 5:
            mid = (span_start + len(confs)) // 2
            issues.append(ReviewIssue(
                frame=mid, person_id=pid,
                issue_type="low_confidence",
                description=f"Low confidence span frames {span_start}-{len(confs)-1}",
                severity=0.7,
            ))

        # High overlap (potential swap)
        for f in range(len(confs)):
            overlap = confs[f].bbox_overlap if hasattr(confs[f], "bbox_overlap") else 0.0
            if overlap > 0.5:
                # Only flag first frame of each overlap burst
                if f == 0 or (confs[f-1].bbox_overlap if hasattr(confs[f-1], "bbox_overlap") else 0.0) <= 0.5:
                    issues.append(ReviewIssue(
                        frame=f, person_id=pid,
                        issue_type="potential_swap",
                        description=f"High bbox overlap ({overlap:.2f}) — possible identity swap",
                        severity=0.9,
                    ))

        # Shape drift
        for f in range(len(confs)):
            shape = confs[f].shape_consistency if hasattr(confs[f], "shape_consistency") else 0.0
            if shape > 0.6:
                if f == 0 or (confs[f-1].shape_consistency if hasattr(confs[f-1], "shape_consistency") else 0.0) <= 0.6:
                    issues.append(ReviewIssue(
                        frame=f, person_id=pid,
                        issue_type="shape_drift",
                        description=f"Shape inconsistency ({shape:.2f}) — body shape changed significantly",
                        severity=0.6,
                    ))

    # Track gaps (detection mask gaps)
    for track in all_tracks:
        pid = track.get("track_id")
        mask = track.get("detection_mask")
        if mask is None:
            continue
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        mask = np.asarray(mask, dtype=bool)
        gap_start = None
        for f in range(len(mask)):
            if not mask[f]:
                if gap_start is None:
                    gap_start = f
            else:
                if gap_start is not None and f - gap_start >= gap_threshold:
                    mid = (gap_start + f) // 2
                    issues.append(ReviewIssue(
                        frame=mid, person_id=pid,
                        issue_type="track_gap",
                        description=f"Detection gap frames {gap_start}-{f-1} ({f - gap_start} frames)",
                        severity=0.5,
                    ))
                gap_start = None

    issues.sort(key=lambda i: i.frame)
    return issues


def on_scan_issues(state):
    """Run the review scanner and return results."""
    if state is None:
        return "No session", gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session", gr.update()

    issues = compute_review_issues(
        sd.get("confidences", {}),
        sd.get("all_tracks", []),
        sd.get("num_frames", 0),
    )
    state["review_issues"] = issues
    state["review_issue_idx"] = 0

    if not issues:
        return "No issues found.", gr.update(value=[])

    rows = [[i.frame, f"ID {i.person_id}", i.issue_type, i.description]
            for i in issues]
    return f"Found {len(issues)} issues. Use Next/Prev to navigate.", gr.update(value=rows)


def on_next_issue(state):
    """Navigate to the next review issue."""
    if state is None:
        return 0, ""
    issues = state.get("review_issues", [])
    if not issues:
        return state.get("current_frame", 0), "No issues"
    idx = state.get("review_issue_idx", 0)
    idx = min(idx + 1, len(issues) - 1) if idx < len(issues) - 1 else 0
    state["review_issue_idx"] = idx
    issue = issues[idx]
    return issue.frame, f"[{idx+1}/{len(issues)}] ID {issue.person_id}: {issue.description}"


def on_prev_issue(state):
    """Navigate to the previous review issue."""
    if state is None:
        return 0, ""
    issues = state.get("review_issues", [])
    if not issues:
        return state.get("current_frame", 0), "No issues"
    idx = state.get("review_issue_idx", 0)
    idx = max(idx - 1, 0) if idx > 0 else len(issues) - 1
    state["review_issue_idx"] = idx
    issue = issues[idx]
    return issue.frame, f"[{idx+1}/{len(issues)}] ID {issue.person_id}: {issue.description}"


# ── Track Split ──


def on_split_track(frame_idx, state):
    """Split the selected person's track at the current frame.

    Everything from frame_idx onward becomes a new inactive track.
    The active track keeps frames before frame_idx.
    """
    if state is None:
        return state, "No session", gr.update(), gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return state, "No session", gr.update(), gr.update()

    selected_person = state.get("selected_person", 0)
    frame_idx = int(frame_idx)
    num_frames = sd.get("num_frames", 0)

    if frame_idx <= 0 or frame_idx >= num_frames - 1:
        return state, f"Cannot split at frame {frame_idx} (must be between 1 and {num_frames - 2})", gr.update(), gr.update()

    # Find active track
    active_tracks = sd.get("all_tracks", [])
    target_track = None
    target_idx = None
    for i, t in enumerate(active_tracks):
        if t.get("track_id") == selected_person:
            target_track = t
            target_idx = i
            break
    if target_track is None:
        return state, f"ID {selected_person} not found", gr.update(), gr.update()

    boxes = target_track["bbx_xyxy"]
    mask = target_track["detection_mask"]
    conf = target_track.get("detection_conf", torch.zeros(num_frames))
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.clone()
    if isinstance(mask, torch.Tensor):
        mask = mask.clone()
    if isinstance(conf, torch.Tensor):
        conf = conf.clone()

    # New track gets frames from frame_idx onward
    new_tid = max(t["track_id"] for t in active_tracks + sd.get("inactive_tracks", [])) + 1
    new_boxes = torch.zeros_like(boxes)
    new_mask = torch.zeros_like(mask)
    new_conf = torch.zeros_like(conf)
    new_boxes[frame_idx:] = boxes[frame_idx:]
    new_mask[frame_idx:] = mask[frame_idx:]
    new_conf[frame_idx:] = conf[frame_idx:]

    # Truncate original track
    boxes[frame_idx:] = boxes[frame_idx - 1].unsqueeze(0).expand(num_frames - frame_idx, -1)
    mask[frame_idx:] = False
    conf[frame_idx:] = 0.0

    target_track["bbx_xyxy"] = boxes
    target_track["detection_mask"] = mask
    target_track["detection_conf"] = conf

    new_track = {
        "track_id": new_tid,
        "bbx_xyxy": new_boxes,
        "detection_mask": new_mask,
        "detection_conf": new_conf,
    }
    sd.setdefault("inactive_tracks", []).append(new_track)

    # Update bboxes in session
    sd["all_bboxes"][selected_person] = boxes.numpy().astype(np.float32)
    sd["original_bboxes"][selected_person] = boxes.numpy().astype(np.float32).copy()
    sd["dirty_persons"].add(selected_person)

    from person_tracker import describe_track
    inactive = sd.get("inactive_tracks", [])
    merge_choices = [describe_track(t) for t in inactive]

    status = f"Split ID {selected_person} at frame {frame_idx}. New {describe_track(new_track)} available for merge."
    return state, status, gr.update(choices=merge_choices, value=None), gr.update(value=_reprocess_btn_label(session_id))


# ── Track Merge Callbacks ──


def on_toggle_show_all_tracks(show_all, state):
    """Toggle inactive track visibility in frame preview."""
    if state is None:
        return state
    state["show_all_tracks"] = bool(show_all)
    return state


def on_merge_track(merge_source_label, frame_idx, state):
    """Merge an inactive track fragment into the currently selected active person."""
    if state is None or not merge_source_label:
        return state, "No track selected", gr.update()

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return state, "No session", gr.update()

    # Parse track ID from dropdown label (e.g., "Track 4 (frames 400-899, 499 dets)")
    import re
    m = re.match(r"Track\s+(\d+)", merge_source_label)
    if not m:
        return state, f"Cannot parse track ID from: {merge_source_label}", gr.update()
    merge_tid = int(m.group(1))

    # Find the inactive track
    inactive = sd.get("inactive_tracks", [])
    source_track = None
    source_idx = None
    for i, t in enumerate(inactive):
        if t["track_id"] == merge_tid:
            source_track = t
            source_idx = i
            break
    if source_track is None:
        return state, f"Track {merge_tid} not found in inactive tracks", gr.update()

    # Find active person's track
    selected_person = state.get("selected_person", 0)
    active_tracks = sd.get("all_tracks", [])
    target_track = None
    target_idx = None
    for i, t in enumerate(active_tracks):
        if t.get("track_id") == selected_person:
            target_track = t
            target_idx = i
            break
    if target_track is None:
        return state, f"Active person {selected_person} not found", gr.update()

    from person_tracker import merge_tracks, describe_track
    merged = merge_tracks(target_track, source_track, sd["num_frames"])

    # Update active track in place
    active_tracks[target_idx] = merged
    sd["all_bboxes"][selected_person] = merged["bbx_xyxy"].numpy().astype(np.float32)
    sd["original_bboxes"][selected_person] = merged["bbx_xyxy"].numpy().astype(np.float32).copy()

    # Remove from inactive list
    inactive.pop(source_idx)

    # Mark dirty
    sd["dirty_persons"].add(selected_person)

    # Rebuild merge dropdown choices
    choices = [describe_track(t) for t in inactive]

    status = f"Merged Track {merge_tid} into ID {selected_person}. Marked dirty."
    return state, status, gr.update(choices=choices, value=None)


# ── Crossing Span Callbacks ──

_CROSSING_SPAN_PENDING: dict[str, int | None] = {}  # session_id → pending start frame


def on_crossing_start(frame_idx, state):
    """Mark the start of a crossing span at the current frame."""
    if state is None:
        return "No session"
    session_id = state.get("session_id", "")
    _CROSSING_SPAN_PENDING[session_id] = int(frame_idx)
    return f"Crossing start marked at frame {int(frame_idx)}. Now click 'Mark Crossing End'."


def on_crossing_end(frame_idx, state):
    """Mark the end of a crossing span and save it."""
    if state is None:
        return "No session", gr.update()
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session", gr.update()

    start = _CROSSING_SPAN_PENDING.pop(session_id, None)
    if start is None:
        return "Click 'Mark Crossing Start' first.", gr.update()

    end = int(frame_idx)
    if end <= start:
        return f"End frame ({end}) must be after start frame ({start}).", gr.update()

    selected_person = state.get("selected_person", 0)
    spans = sd.setdefault("crossing_spans", {})
    person_spans = spans.setdefault(selected_person, [])
    person_spans.append((start, end))
    person_spans.sort()

    # Persist to disk
    pid_to_dir = sd.get("pid_to_dir", {})
    person_dir = pid_to_dir.get(selected_person)
    if person_dir:
        import json
        spans_path = Path(person_dir) / "crossing_spans.json"
        spans_path.write_text(json.dumps(person_spans))

    # Build DataFrame
    rows = _build_crossing_spans_df(sd.get("crossing_spans", {}))
    status = f"Crossing span added: frames {start}-{end} ({end - start} frames) for ID {selected_person}"
    return status, gr.update(value=rows)


def _build_crossing_spans_df(crossing_spans: dict) -> list[list]:
    """Build DataFrame rows from crossing spans dict."""
    rows = []
    for pid, spans in sorted(crossing_spans.items()):
        for start, end in spans:
            rows.append([f"ID {pid}", start, end, end - start])
    return rows


def on_remove_crossing_span(state):
    """Remove the last crossing span for the selected person."""
    if state is None:
        return "No session", gr.update()
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session", gr.update()

    selected_person = state.get("selected_person", 0)
    spans = sd.get("crossing_spans", {})
    person_spans = spans.get(selected_person, [])
    if not person_spans:
        return f"No crossing spans for ID {selected_person}", gr.update()

    removed = person_spans.pop()

    # Persist to disk
    pid_to_dir = sd.get("pid_to_dir", {})
    person_dir = pid_to_dir.get(selected_person)
    if person_dir:
        import json
        spans_path = Path(person_dir) / "crossing_spans.json"
        spans_path.write_text(json.dumps(person_spans))

    rows = _build_crossing_spans_df(spans)
    return (f"Removed crossing span {removed[0]}-{removed[1]} for ID {selected_person}",
            gr.update(value=rows))


def on_crossing_df_select(evt: gr.SelectData, state):
    """Store selected crossing span row index for targeted removal."""
    if state is None or evt is None:
        return state
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    state["_crossing_selected_row"] = row_idx
    return state


def on_remove_selected_crossing(state):
    """Remove the selected crossing span (by DataFrame row click)."""
    if state is None:
        return "No session", gr.update()
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return "No session", gr.update()

    row_idx = state.get("_crossing_selected_row")
    spans = sd.get("crossing_spans", {})

    if row_idx is None:
        # No row selected — remove last span for selected person
        return on_remove_crossing_span(state)

    # Map row index back to (pid, span_index) — rows are sorted by pid then span order
    flat_idx = 0
    target_pid = None
    target_span_idx = None
    for pid in sorted(spans.keys()):
        for si, span in enumerate(spans[pid]):
            if flat_idx == row_idx:
                target_pid = pid
                target_span_idx = si
                break
            flat_idx += 1
        if target_pid is not None:
            break

    if target_pid is None:
        return "Invalid selection", gr.update()

    removed = spans[target_pid].pop(target_span_idx)
    state.pop("_crossing_selected_row", None)

    # Persist
    pid_to_dir = sd.get("pid_to_dir", {})
    person_dir = pid_to_dir.get(target_pid)
    if person_dir:
        import json
        spans_path = Path(person_dir) / "crossing_spans.json"
        spans_path.write_text(json.dumps(spans[target_pid]))

    rows = _build_crossing_spans_df(spans)
    return (f"Removed crossing span {removed[0]}-{removed[1]} for ID {target_pid}",
            gr.update(value=rows))


def on_confidence_click(evt: gr.SelectData, state):
    """Click on confidence timeline image to seek to that frame.

    Image is rendered at width = max(num_frames, min_width), so we map
    x-coordinate back to frame index.
    """
    if state is None or evt is None:
        return 0
    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return 0
    num_frames = sd.get("num_frames", 1)
    img_width = max(num_frames, _CONFIDENCE_TIMELINE_MIN_WIDTH, 1)
    idx = evt.index
    click_x = idx[0] if isinstance(idx, (list, tuple)) else idx
    frame = int(click_x * num_frames / img_width)
    return max(0, min(frame, num_frames - 1))


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
        swap_btn = gr.Button("Swap IDs", size="sm", scale=0, min_width=90)
        swap_target = gr.Dropdown(
            label="Swap with",
            choices=[],
            interactive=True,
            scale=1,
            visible=True,
        )

    # ── ReID Gallery ──
    reid_gallery = gr.Image(
        label="Person Identity Gallery (high-confidence crops)",
        interactive=False,
        type="numpy",
    )

    # ── Track Merge & Crossing Span Controls ──
    with gr.Accordion("Track Merge & Crossings", open=False):
        with gr.Row():
            show_all_tracks = gr.Checkbox(
                label="Show All Tracks", value=False,
                info="Overlay inactive/discarded track fragments on frame preview",
            )
        gr.Markdown("**Split / Merge Tracks**")
        with gr.Row():
            split_btn = gr.Button("Split Track at Frame", size="sm", scale=0, min_width=160)
            split_status = gr.Textbox(
                label="", value="", interactive=False, scale=3,
            )
        with gr.Row():
            merge_source = gr.Dropdown(
                label="Inactive Track",
                choices=[],
                interactive=True,
                scale=2,
            )
            merge_btn = gr.Button("Merge", size="sm", scale=0, min_width=80)
            merge_status = gr.Textbox(
                label="", value="", interactive=False, scale=2,
            )
        gr.Markdown("**Crossing Spans** (SLERP bridge through occlusion)")
        with gr.Row():
            crossing_start_btn = gr.Button("Mark Crossing Start", size="sm", scale=0, min_width=150)
            crossing_end_btn = gr.Button("Mark Crossing End", size="sm", scale=0, min_width=150)
            remove_crossing_btn = gr.Button("Remove Selected", size="sm", scale=0, min_width=130)
            crossing_status = gr.Textbox(
                label="", value="", interactive=False, scale=2,
            )
        crossing_spans_df = gr.DataFrame(
            headers=["Person", "Start", "End", "Duration"],
            datatype=["str", "number", "number", "number"],
            interactive=False,
            row_count=(3, "dynamic"),
        )
        gr.Markdown("**Review Scanner** (find problem frames automatically)")
        with gr.Row():
            scan_btn = gr.Button("Scan for Issues", size="sm", scale=0, min_width=140)
            prev_issue_btn = gr.Button("< Prev Issue", size="sm", scale=0, min_width=110)
            next_issue_btn = gr.Button("Next Issue >", size="sm", scale=0, min_width=110)
            review_summary = gr.Textbox(
                label="", value="", interactive=False, scale=3,
            )
        review_issues_df = gr.DataFrame(
            headers=["Frame", "Person", "Type", "Description"],
            datatype=["number", "str", "str", "str"],
            interactive=False,
            row_count=(5, "dynamic"),
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
        play_btn = gr.Button("Play", size="sm", scale=0, min_width=60,
                             elem_id="play_pause_btn")
        frame_label = gr.Textbox(
            label="Frame",
            value="Frame 0/0",
            interactive=False,
            scale=1,
        )
        fwd1_btn = gr.Button(">", size="sm", scale=0, min_width=40)
        fwd10_btn = gr.Button(">>", size="sm", scale=0, min_width=40)
        last_btn = gr.Button(">|", size="sm", scale=0, min_width=40)

    frame_slider = gr.Slider(
        minimum=0,
        maximum=1,
        step=1,
        value=0,
        label="Frame",
        interactive=True,
        elem_id="frame_slider",
    )

    confidence_plot = gr.Image(
        label="Confidence Timeline (click to seek)",
        interactive=False,
        type="numpy",
        elem_id="confidence_timeline",
    )

    # CSS: make confidence timeline fill width, hide chrome
    gr.HTML(
        "<style>"
        "#confidence_timeline { padding: 0; }"
        "#confidence_timeline img { width: 100% !important; height: auto !important; }"
        "#confidence_timeline .icon-buttons, "
        "#confidence_timeline .download-btn, "
        "#confidence_timeline button.share, "
        "#confidence_timeline .fullscreen-btn "
        "{ display: none !important; }"
        "</style>"
    )

    # Keyframe buttons — placed right below the timelines where keyframes appear
    with gr.Row():
        verify_btn = gr.Button("Verify", size="sm", scale=0, min_width=80)
        add_kf_btn = gr.Button("Add Keyframe", size="sm", scale=0, min_width=110)
        remove_kf_btn = gr.Button("Remove Keyframe", size="sm", scale=0, min_width=130)
        prev_kf_btn2 = gr.Button("< Prev KF", size="sm", scale=0, min_width=90)
        next_kf_btn2 = gr.Button("Next KF >", size="sm", scale=0, min_width=90)

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
            "Reprocess All (0 IDs dirty)", variant="primary", size="sm", scale=0, min_width=180,
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

    # Frame slider → update display (hidden progress prevents frame blanking)
    frame_slider.change(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
    )

    # Play/Pause — JS in HTML, button click toggles interval on slider
    gr.HTML("""<script>
    document.addEventListener('click', function(e) {
        // Match click on the play button or anything inside it
        const wrapper = e.target.closest('#play_pause_btn');
        if (!wrapper) return;
        // Find the slider range input
        const slider = document.querySelector('#frame_slider input[type=range]');
        if (!slider) {
            console.warn('[Play] Could not find #frame_slider input[type=range]');
            return;
        }
        // Find the visible button text element
        const btnEl = wrapper.querySelector('button') || wrapper;
        if (window._playInterval) {
            clearInterval(window._playInterval);
            window._playInterval = null;
            btnEl.textContent = 'Play';
        } else {
            btnEl.textContent = 'Pause';
            window._playInterval = setInterval(() => {
                let val = parseInt(slider.value) + 1;
                let max = parseInt(slider.max);
                if (val > max) {
                    clearInterval(window._playInterval);
                    window._playInterval = null;
                    btnEl.textContent = 'Play';
                    return;
                }
                // Set value via native setter to trigger Gradio's change detection
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLInputElement.prototype, 'value').set;
                nativeSetter.call(slider, val);
                slider.dispatchEvent(new Event('input', {bubbles: true}));
                slider.dispatchEvent(new Event('change', {bubbles: true}));
            }, 100);
        }
    });
    </script>""")

    # Person dropdown → update display
    def _refresh_gallery(state):
        if state is None:
            return None
        sid = state.get("session_id", "")
        pid = state.get("selected_person", 0)
        return _build_gallery_image(sid, pid)

    person_dropdown.change(
        fn=on_person_change,
        inputs=[person_dropdown, frame_slider, panel_state],
        outputs=[panel_state] + display_outputs,
    ).then(
        fn=_refresh_gallery,
        inputs=[panel_state],
        outputs=[reid_gallery],
    ).then(
        fn=on_pose_frame_change,
        inputs=[frame_slider, panel_state],
        outputs=[
            pose_panel["skeleton_image"],
            pose_panel["corrections_df"],
            pose_panel["euler_x"],
            pose_panel["euler_y"],
            pose_panel["euler_z"],
        ],
    ).then(
        fn=lambda p: gr.update(value=p),
        inputs=[person_dropdown],
        outputs=[pose_panel["pc_person_dropdown"]],
    )

    # Transport buttons → update slider (which triggers display update)
    first_btn.click(fn=_nav_first, inputs=[], outputs=[frame_slider])
    last_btn.click(fn=_nav_last, inputs=[panel_state], outputs=[frame_slider])
    back10_btn.click(fn=_nav_back10, inputs=[frame_slider], outputs=[frame_slider])
    fwd10_btn.click(fn=_nav_fwd10, inputs=[frame_slider, panel_state], outputs=[frame_slider])
    back1_btn.click(fn=_nav_back1, inputs=[frame_slider], outputs=[frame_slider])
    fwd1_btn.click(fn=_nav_fwd1, inputs=[frame_slider, panel_state], outputs=[frame_slider])

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

    # Track merge & crossing span callbacks
    show_all_tracks.change(
        fn=on_toggle_show_all_tracks,
        inputs=[show_all_tracks, panel_state],
        outputs=[panel_state],
    ).then(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
    )

    split_btn.click(
        fn=on_split_track,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state, split_status, merge_source, reprocess_btn],
    ).then(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
    )

    merge_btn.click(
        fn=on_merge_track,
        inputs=[merge_source, frame_slider, panel_state],
        outputs=[panel_state, merge_status, merge_source],
    ).then(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
    )

    crossing_start_btn.click(
        fn=on_crossing_start,
        inputs=[frame_slider, panel_state],
        outputs=[crossing_status],
    )
    crossing_end_btn.click(
        fn=on_crossing_end,
        inputs=[frame_slider, panel_state],
        outputs=[crossing_status, crossing_spans_df],
    )
    crossing_spans_df.select(
        fn=on_crossing_df_select,
        inputs=[panel_state],
        outputs=[panel_state],
    )
    remove_crossing_btn.click(
        fn=on_remove_selected_crossing,
        inputs=[panel_state],
        outputs=[crossing_status, crossing_spans_df],
    )

    # Confidence timeline click → seek to frame
    confidence_plot.select(
        fn=on_confidence_click,
        inputs=[panel_state],
        outputs=[frame_slider],
    )

    # Keyframe buttons (below timeline)
    prev_kf_btn2.click(
        fn=on_prev_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider],
    )
    next_kf_btn2.click(
        fn=on_next_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider],
    )

    # Review scanner
    scan_btn.click(
        fn=on_scan_issues,
        inputs=[panel_state],
        outputs=[review_summary, review_issues_df],
    )
    next_issue_btn.click(
        fn=on_next_issue,
        inputs=[panel_state],
        outputs=[frame_slider, review_summary],
    )
    prev_issue_btn.click(
        fn=on_prev_issue,
        inputs=[panel_state],
        outputs=[frame_slider, review_summary],
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
    ).then(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
    )

    apply_kf_btn.click(
        fn=on_apply_keyframes,
        inputs=[panel_state],
        outputs=[bbox_edit_status, panel_state, reprocess_btn],
    ).then(
        fn=update_frame_display,
        inputs=[frame_slider, panel_state],
        outputs=display_outputs,
        show_progress="hidden",
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
        "merge_target": merge_source,
        "crossing_spans_df": crossing_spans_df,
        "review_summary": review_summary,
        "review_issues_df": review_issues_df,
        "reid_gallery": reid_gallery,
    }


def populate_panel(
    state_dict: dict,
) -> tuple:
    """Generate initial values for panel components after init_panel_state.

    Returns tuple of Gradio updates for:
    (state, person_dropdown, swap_target, frame_slider, frame_label,
     + 9 display_outputs, merge_target, crossing_spans_df)
    """
    if state_dict is None:
        return (None,) * 17

    session_id = state_dict.get("session_id", "")
    sd = _SESSION_DATA.get(session_id, {})
    num_frames = sd.get("num_frames", 1)
    choices = _person_choices(session_id)

    # Initial display at frame 0
    display = update_frame_display(0, state_dict)

    # Merge dropdown choices from inactive tracks
    from person_tracker import describe_track
    inactive = sd.get("inactive_tracks", [])
    merge_choices = [describe_track(t) for t in inactive]

    # Load crossing spans from disk
    crossing_spans = sd.get("crossing_spans", {})
    for pid, pdir in sd.get("pid_to_dir", {}).items():
        import json
        spans_path = Path(pdir) / "crossing_spans.json"
        if spans_path.exists():
            try:
                loaded = json.loads(spans_path.read_text())
                crossing_spans[pid] = [tuple(s) for s in loaded]
            except Exception:
                pass
    sd["crossing_spans"] = crossing_spans
    spans_rows = _build_crossing_spans_df(crossing_spans)

    return (
        state_dict,                                     # state
        gr.update(choices=choices, value=choices[0] if choices else None),  # person_dropdown
        gr.update(choices=choices, value=choices[1] if len(choices) > 1 else None),  # swap_target
        gr.update(maximum=max(num_frames - 1, 1), value=0),  # frame_slider
        f"Frame 0/{num_frames - 1}" if num_frames > 0 else "Frame 0/0",  # frame_label
        *display,                                       # 9 display outputs
        gr.update(choices=merge_choices, value=None),   # merge_target
        gr.update(value=spans_rows),                    # crossing_spans_df
        _build_gallery_image(session_id, selected_person=sd.get("all_tracks", [{}])[0].get("track_id", 0) if sd.get("all_tracks") else 0),  # reid_gallery
    )
