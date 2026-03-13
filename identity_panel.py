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

from identity_confidence import TrackConfidence, confidence_to_array
from identity_tracking import IdentityTrack, IdentityKeyframe
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
) -> plt.Figure:
    """Render confidence heatmap timeline as matplotlib figure."""
    fig, ax = plt.subplots(figsize=(14, 1.0))
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

    # Playhead
    ax.axvline(current_frame, color="white", lw=1.5, zorder=10)

    ax.set_xlim(0, max(num_frames, 1))
    ax.set_ylim(0, 1.2)
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
    person_dirs = sd.get("person_dirs", [])
    track = _get_identity_track(session_id, person_id)
    if track is None:
        return
    for i, t in enumerate(sd.get("identity_tracks", [])):
        if t.person_id == person_id and i < len(person_dirs):
            track.save_json(Path(person_dirs[i]) / "identity_track.json")
            break


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
    for i, pdir in enumerate(str_person_dirs):
        pid = all_tracks[i].get("track_id", i) if i < len(all_tracks) else i
        corr = _load_bbox_corrections(pdir)
        if corr:
            bbox_corrections[pid] = corr
            # Apply corrections on top of all_bboxes
            if pid in all_bboxes:
                for frame_idx, bbox in corr.items():
                    if frame_idx < len(all_bboxes[pid]):
                        all_bboxes[pid][frame_idx] = bbox

    # Load confidence from identity tracks or recompute
    for i, id_track in enumerate(identity_tracks):
        pid = id_track.person_id
        # Try loading confidence CSV
        if i < len(person_dirs):
            csv_path = Path(person_dirs[i]) / "confidence.csv"
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

    # Build pid → directory and pid → index mappings (track_id may != array index)
    pid_to_dir = {}
    pid_to_index = {}
    for i, track in enumerate(all_tracks):
        pid = track.get("track_id", i)
        if i < len(str_person_dirs):
            pid_to_dir[pid] = str_person_dirs[i]
        pid_to_index[pid] = i

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
    }

    # Load per-person SMPL params and correction tracks for pose correction
    smplx_params = {}
    correction_tracks = {}
    str_person_dirs = [str(p) for p in person_dirs]

    for i, pdir in enumerate(str_person_dirs):
        pid = identity_tracks[i].person_id if i < len(identity_tracks) else i
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
    fig = _render_confidence_plot(overall_array, frame_idx, kf_frames, num_frames)

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

    # Parse "Person N" -> N
    try:
        pid = int(person_str.split()[-1])
    except (ValueError, IndexError):
        pid = 0

    state["selected_person"] = pid
    return (state, *update_frame_display(frame_idx, state))


def on_verify(frame_idx, state):
    """Mark keyframe at current frame as verified (or create one)."""
    if state is None:
        return state, *update_frame_display(0, state)

    session_id = state["session_id"]
    pid = state["selected_person"]
    frame_idx = int(frame_idx)
    track = _get_identity_track(session_id, pid)
    if track is None:
        return state, *update_frame_display(frame_idx, state)

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

    _save_identity_track(session_id, pid)
    return (state, *update_frame_display(frame_idx, state))


def on_add_keyframe(frame_idx, state):
    """Add an unverified keyframe at current frame."""
    if state is None:
        return state, *update_frame_display(0, state)

    session_id = state["session_id"]
    pid = state["selected_person"]
    frame_idx = int(frame_idx)
    track = _get_identity_track(session_id, pid)
    if track is None:
        return state, *update_frame_display(frame_idx, state)

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
    _save_identity_track(session_id, pid)
    return (state, *update_frame_display(frame_idx, state))


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


def on_reprocess_all_dirty(state, progress=gr.Progress(track_tqdm=False)):
    """Reprocess all persons with pending bbox edits."""
    if state is None:
        return state, "No session", gr.update(), *update_frame_display(0, state)

    session_id = state.get("session_id", "")
    sd = _SESSION_DATA.get(session_id)
    if sd is None:
        return state, "No session data", gr.update(), *update_frame_display(0, state)

    dirty = sd.get("dirty_persons", set())
    if not dirty:
        return (state, "No dirty persons",
                gr.update(value=_reprocess_btn_label(session_id)),
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
    progress(1.0, desc="Reprocessing complete")

    frame_idx = state.get("current_frame", 0)
    status = "\n".join(status_lines)
    return (state, status, gr.update(value=_reprocess_btn_label(session_id)),
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


def build_identity_panel() -> dict[str, Any]:
    """Construct the Identity Inspector Gradio components.

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

    # Person dropdown → update display
    person_dropdown.change(
        fn=on_person_change,
        inputs=[person_dropdown, frame_slider, panel_state],
        outputs=[panel_state] + display_outputs,
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
        outputs=[panel_state] + display_outputs,
    )
    add_kf_btn.click(
        fn=on_add_keyframe,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state] + display_outputs,
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

    reprocess_btn.click(
        fn=on_reprocess_all_dirty,
        inputs=[panel_state],
        outputs=[panel_state, reprocess_status, reprocess_btn] + display_outputs,
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
    }


def populate_panel(
    state_dict: dict,
) -> tuple:
    """Generate initial values for panel components after init_panel_state.

    Returns tuple of Gradio updates for:
    (state, person_dropdown, swap_target, frame_slider, frame_label, + display_outputs)
    """
    if state_dict is None:
        return (None,) * 14

    session_id = state_dict.get("session_id", "")
    sd = _SESSION_DATA.get(session_id, {})
    num_frames = sd.get("num_frames", 1)
    choices = _person_choices(session_id)

    # Initial display at frame 0
    display = update_frame_display(0, state_dict)

    return (
        state_dict,                                     # state
        gr.update(choices=choices, value=choices[0] if choices else None),  # person_dropdown
        gr.update(choices=choices, value=choices[1] if len(choices) > 1 else None),  # swap_target
        gr.update(maximum=max(num_frames - 1, 1), value=0),  # frame_slider
        f"Frame 0/{num_frames - 1}" if num_frames > 0 else "Frame 0/0",  # frame_label
        *display,                                       # 9 display outputs
    )
