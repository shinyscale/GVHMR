"""Pose Corrector panel — Gradio UI for keyframe-based skeleton correction.

Embedded within the Identity Inspector accordion. Provides skeleton preview
with click-to-select joint interaction, euler rotation sliders, quick-fix
actions, corrections table, and re-export to BVH/FBX.

Exports:
    build_pose_correction_panel() — constructs Gradio components
    get_pose_session_data() / set_pose_session_data() — module-level state
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gradio as gr

from pose_correction import (
    BODY_JOINT_NAMES,
    CorrectionTrack,
    PoseCorrection,
    apply_corrections,
    axis_angle_to_euler_deg,
    euler_deg_to_axis_angle,
    find_nearest_joint,
    flip_global_orient,
    mirror_lr_pose,
    copy_pose_from_frame,
)
from visualize_skeleton import render_skeleton_frame

# ── Module-level session state (shared with identity_panel) ──

_POSE_SESSION: dict[str, Any] = {}
# Keys per session_id:
#   smplx_params: dict[int, dict]       — per person_id SMPL params
#   correction_tracks: dict[int, CorrectionTrack]
#   camera_K: np.ndarray | None
#   person_dirs: list[str]
#   video_path: str
#   num_frames: int
#   fps: float


def get_pose_session_data(session_id: str) -> dict | None:
    return _POSE_SESSION.get(session_id)


def set_pose_session_data(session_id: str, key: str, value: Any) -> None:
    if session_id not in _POSE_SESSION:
        _POSE_SESSION[session_id] = {}
    _POSE_SESSION[session_id][key] = value


def init_pose_session(
    session_id: str,
    smplx_params: dict[int, dict],
    correction_tracks: dict[int, CorrectionTrack],
    person_dirs: list[str],
    video_path: str,
    num_frames: int,
    fps: float,
    pid_to_dir: dict[int, str] | None = None,
) -> None:
    """Initialize pose correction session data."""
    _POSE_SESSION[session_id] = {
        "smplx_params": smplx_params,
        "correction_tracks": correction_tracks,
        "person_dirs": person_dirs,
        "video_path": video_path,
        "num_frames": num_frames,
        "fps": fps,
        "pid_to_dir": pid_to_dir or {},
    }


# ── Frame extraction (reuse identity_panel's LRU cache) ──


def _extract_frame_cached(video_path: str, frame_idx: int, session_id: str) -> np.ndarray | None:
    """Extract frame using identity_panel's LRU cache (lazy import to avoid circular)."""
    from identity_panel import _extract_frame
    return _extract_frame(video_path, frame_idx, session_id)


def _pc_person_choices(session_id: str) -> list[str]:
    """Return person dropdown choices from the pose session."""
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return []
    return [f"ID {pid}" for pid in sorted(sd["smplx_params"].keys())]


def get_pc_person_choices_update(state: dict | None) -> "gr.update":
    """Return a gr.update for pc_person_dropdown after session load."""
    if state is None:
        return gr.update()
    session_id = state.get("session_id", "")
    choices = _pc_person_choices(session_id)
    selected = state.get("selected_person", 0)
    value = f"ID {selected}" if f"ID {selected}" in choices else (choices[0] if choices else None)
    return gr.update(choices=choices, value=value)


def _camera_space_params(params: dict) -> dict:
    """Remap camera-space arrays to default keys for rendering/editing.

    GVHMR params store world-space in default keys and camera-space
    in suffixed keys. The pose corrector works in camera space (what
    the user sees on the video), so we swap before FK + projection.
    """
    if "global_orient_cam" not in params:
        return params
    p = dict(params)
    p["global_orient"] = params["global_orient_cam"]
    p["body_pose"] = params["body_pose_cam"]
    p["transl"] = params["transl_cam"]
    return p


# ── Core rendering callback ──


def _render_skeleton_preview(
    session_id: str,
    person_id: int,
    frame_idx: int,
    selected_joint: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Render skeleton overlay for the correction panel.

    Returns (annotated_image_rgb, joints_2d) or (None, None).
    """
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, None

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, None

    params = _camera_space_params(params)

    video_path = sd["video_path"]
    frame = _extract_frame_cached(video_path, frame_idx, session_id)
    if frame is None:
        return None, None

    ct = sd["correction_tracks"].get(person_id)

    img, joints_2d = render_skeleton_frame(
        params=params,
        frame_idx=frame_idx,
        video_frame=frame,
        selected_joint=selected_joint,
        corrections=ct,
    )
    return img, joints_2d


def _get_joint_euler(
    session_id: str,
    person_id: int,
    frame_idx: int,
    joint_idx: int,
) -> tuple[float, float, float]:
    """Get current euler XYZ degrees for a joint (with corrections applied)."""
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return 0.0, 0.0, 0.0

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return 0.0, 0.0, 0.0

    params = _camera_space_params(params)

    ct = sd["correction_tracks"].get(person_id)
    if ct is not None:
        params = apply_corrections(params, ct)

    n = params["num_frames"]
    f = max(0, min(frame_idx, n - 1))

    if joint_idx == 0:
        aa = params["global_orient"][f]
    elif 1 <= joint_idx <= 21:
        aa = params["body_pose"][f, joint_idx - 1]
    else:
        return 0.0, 0.0, 0.0

    euler = axis_angle_to_euler_deg(aa)
    return float(euler[0]), float(euler[1]), float(euler[2])


def _save_correction_track(session_id: str, person_id: int) -> None:
    """Persist correction track to JSON."""
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return
    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        return
    person_dir = sd.get("pid_to_dir", {}).get(person_id)
    if person_dir is not None:
        ct.save_json(Path(person_dir) / "pose_corrections.json")


def _corrections_dataframe(session_id: str, person_id: int) -> list[list]:
    """Build corrections DataFrame rows."""
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return []
    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        return []

    rows = []
    for corr in ct.corrections:
        joints_str = "root" if corr.correction_type == "global_orient" else ""
        if corr.correction_type == "copy_from_frame":
            joints_str = f"all (from {corr.source_frame})"
        elif corr.body_pose:
            joint_names = []
            for j_idx in sorted(corr.body_pose.keys()):
                if j_idx + 1 < len(BODY_JOINT_NAMES):
                    joint_names.append(BODY_JOINT_NAMES[j_idx + 1])
            joints_str = ", ".join(joint_names[:3])
            if len(joint_names) > 3:
                joints_str += f" +{len(joint_names) - 3}"
        rows.append([corr.frame_index, corr.correction_type, joints_str])
    return rows


# ── Gradio Callbacks ──


def on_pc_person_change(person_str, frame_idx, state):
    """Handle person selection within pose corrector."""
    if state is None or not person_str:
        return state, None, [], 0.0, 0.0, 0.0
    try:
        pid = int(person_str.split()[-1])
    except (ValueError, IndexError):
        return state, None, [], 0.0, 0.0, 0.0
    state = dict(state)
    state["selected_person"] = pid
    img, df_data, ex, ey, ez = on_pose_frame_change(frame_idx, state)
    return state, img, df_data, ex, ey, ez


def on_skeleton_click(evt: gr.SelectData, frame_idx, state):
    """Handle click on skeleton image to select nearest joint."""
    if state is None or evt is None:
        return (
            gr.update(),  # joint_dropdown
            0.0, 0.0, 0.0,  # sliders
            None,  # skeleton_image
            state,
        )

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    # Get joints_2d for hit testing
    _, joints_2d = _render_skeleton_preview(session_id, person_id, frame_idx, None)
    if joints_2d is None:
        return gr.update(), 0.0, 0.0, 0.0, None, state

    click_x, click_y = evt.index[0], evt.index[1]
    joint_idx = find_nearest_joint(click_x, click_y, joints_2d, threshold=30)

    if joint_idx is None:
        return gr.update(), 0.0, 0.0, 0.0, None, state

    # Update selection
    state["pc_selected_joint"] = joint_idx
    joint_name = BODY_JOINT_NAMES[joint_idx] if joint_idx < len(BODY_JOINT_NAMES) else f"Joint {joint_idx}"

    # Get euler angles
    ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, joint_idx)

    # Re-render with selection
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)

    return (
        gr.update(value=joint_name),  # joint_dropdown
        ex, ey, ez,                    # sliders
        img,                           # skeleton_image
        state,
    )


def on_joint_dropdown_change(joint_name, frame_idx, state):
    """Handle joint dropdown selection change."""
    if state is None or not joint_name:
        return 0.0, 0.0, 0.0, None, state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    # Find joint index from name
    joint_idx = None
    for i, name in enumerate(BODY_JOINT_NAMES):
        if name == joint_name:
            joint_idx = i
            break
    if joint_idx is None:
        return 0.0, 0.0, 0.0, None, state

    state["pc_selected_joint"] = joint_idx
    ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, joint_idx)
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)

    return ex, ey, ez, img, state


def on_euler_change(euler_x, euler_y, euler_z, frame_idx, state):
    """Handle euler slider changes — update working correction and re-render."""
    if state is None:
        return None, []

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    joint_idx = state.get("pc_selected_joint")
    frame_idx = int(frame_idx)

    if joint_idx is None:
        return None, []

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, []

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, []

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    # Build the new axis-angle from euler
    new_aa = euler_deg_to_axis_angle(np.array([euler_x, euler_y, euler_z], dtype=np.float32))

    # Get or create correction at this frame
    corr = ct.get_correction(frame_idx)
    if corr is None:
        # Create new correction
        if joint_idx == 0:
            ct.add_correction(
                frame_index=frame_idx,
                correction_type="global_orient",
                global_orient=new_aa,
            )
        else:
            ct.add_correction(
                frame_index=frame_idx,
                correction_type="joint",
                body_pose={joint_idx - 1: new_aa},
            )
    else:
        # Update existing correction
        if joint_idx == 0:
            corr.global_orient = new_aa
            if corr.correction_type not in ("copy_from_frame", "full_pose"):
                corr.correction_type = "global_orient"
        else:
            if corr.body_pose is None:
                corr.body_pose = {}
            corr.body_pose[joint_idx - 1] = new_aa
            if corr.correction_type == "global_orient":
                corr.correction_type = "joint"

    # Re-render
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return img, df_data


def on_apply_correction(frame_idx, state):
    """Save current correction to track and persist to JSON."""
    if state is None:
        return state, []

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    _save_correction_track(session_id, person_id)
    df_data = _corrections_dataframe(session_id, person_id)
    return state, df_data


def on_reset_joint(frame_idx, state):
    """Remove selected joint's rotation from the correction at current frame."""
    if state is None:
        return 0.0, 0.0, 0.0, None, [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    joint_idx = state.get("pc_selected_joint")
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None or joint_idx is None:
        return 0.0, 0.0, 0.0, None, [], state

    ct = sd["correction_tracks"].get(person_id)
    if ct is not None:
        corr = ct.get_correction(frame_idx)
        if corr is not None:
            if joint_idx == 0:
                corr.global_orient = None
            elif corr.body_pose and (joint_idx - 1) in corr.body_pose:
                del corr.body_pose[joint_idx - 1]

    ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, joint_idx)
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return ex, ey, ez, img, df_data, state


def on_reset_all(frame_idx, state):
    """Remove all corrections at current frame."""
    if state is None:
        return None, [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is not None:
        ct = sd["correction_tracks"].get(person_id)
        if ct is not None:
            ct.remove_correction(frame_idx)

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return img, df_data, state


def on_flip_whole_body(frame_idx, state):
    """Flip global orient 180° around Y (yaw) — fixes wrong-way facing."""
    if state is None:
        return None, [], 0.0, 0.0, 0.0, state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, [], 0.0, 0.0, 0.0, state

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, [], 0.0, 0.0, 0.0, state

    params = _camera_space_params(params)
    new_go = flip_global_orient(params, frame_idx, axis="yaw")

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    ct.add_correction(
        frame_index=frame_idx,
        correction_type="global_orient",
        global_orient=new_go,
    )

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)

    # Update euler sliders if root is selected
    ex, ey, ez = 0.0, 0.0, 0.0
    if joint_idx == 0:
        ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, 0)

    return img, df_data, ex, ey, ez, state


def on_invert_upright(frame_idx, state):
    """Flip global orient 180° around X (pitch) — fixes upside-down."""
    if state is None:
        return None, [], 0.0, 0.0, 0.0, state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, [], 0.0, 0.0, 0.0, state

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, [], 0.0, 0.0, 0.0, state

    params = _camera_space_params(params)
    new_go = flip_global_orient(params, frame_idx, axis="pitch")

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    ct.add_correction(
        frame_index=frame_idx,
        correction_type="global_orient",
        global_orient=new_go,
    )

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)

    ex, ey, ez = 0.0, 0.0, 0.0
    if joint_idx == 0:
        ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, 0)

    return img, df_data, ex, ey, ez, state


def on_mirror_lr(frame_idx, state):
    """Swap left/right joint rotations."""
    if state is None:
        return None, [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, [], state

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, [], state

    params = _camera_space_params(params)
    mirrored = mirror_lr_pose(params, frame_idx)

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    ct.add_correction(
        frame_index=frame_idx,
        correction_type="joint",
        body_pose=mirrored,
    )

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return img, df_data, state


def on_copy_from_frame(src_frame_num, frame_idx, state):
    """Copy full pose from source frame to current frame."""
    if state is None:
        return None, [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, [], state

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None, [], state

    params = _camera_space_params(params)

    try:
        src_frame = int(src_frame_num)
    except (TypeError, ValueError):
        return None, [], state

    corr = copy_pose_from_frame(params, src_frame, frame_idx)
    corr.person_id = person_id

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    ct.add_correction(
        frame_index=frame_idx,
        correction_type="copy_from_frame",
        global_orient=corr.global_orient,
        body_pose=corr.body_pose,
        transl=corr.transl,
        source_frame=src_frame,
    )

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return img, df_data, state


def on_delete_correction(evt: gr.SelectData, frame_idx, state):
    """Delete a correction from the table."""
    if state is None or evt is None:
        return None, [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None, [], state

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        return None, [], state

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < len(ct.corrections):
        corr_frame = ct.corrections[row_idx].frame_index
        ct.remove_correction(corr_frame)

    joint_idx = state.get("pc_selected_joint")
    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)
    return img, df_data, state


def on_auto_detect_span(frame_idx, state):
    """Auto-detect low-confidence span around current frame."""
    if state is None:
        return 0, 0

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)

    # Import identity tracking to use low_confidence_spans
    from identity_panel import _SESSION_DATA, _get_identity_track
    from identity_confidence import TrackConfidence

    id_sd = _SESSION_DATA.get(session_id, {})
    track = _get_identity_track(session_id, person_id)
    confs = id_sd.get("confidences", {}).get(person_id, [])

    if track is None or not confs:
        return max(0, frame_idx - 15), frame_idx + 15

    spans = track.low_confidence_spans(confs, threshold=0.4)
    for start, end in spans:
        if start <= frame_idx <= end:
            return start, end

    # No span found — return a small window
    return max(0, frame_idx - 15), frame_idx + 15


def on_reexport_bvh(frame_idx, state, progress=gr.Progress(track_tqdm=False)):
    """Re-export corrected BVH for the selected person."""
    if state is None:
        return None

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return None

    params = sd["smplx_params"].get(person_id)
    if params is None:
        return None

    ct = sd["correction_tracks"].get(person_id)
    fps = sd.get("fps", 30.0)

    # Save corrections first
    _save_correction_track(session_id, person_id)

    # Import BVH converter
    from smplx_to_bvh import convert_params_to_bvh

    person_dir = sd.get("pid_to_dir", {}).get(person_id)
    if person_dir is not None:
        out_path = str(Path(person_dir) / "corrected_body.bvh")
    else:
        return None

    # Build reference_params for "carried" space overrides
    ref_params = None
    if ct is not None and ct.space_overrides:
        ref_pids = {o.reference_person for o in ct.space_overrides
                    if o.space == "carried" and o.reference_person is not None}
        if ref_pids:
            ref_params = {}
            for rpid in ref_pids:
                rp = sd["smplx_params"].get(rpid)
                if rp is not None:
                    ref_params[rpid] = rp

    progress(0.1, desc="Applying corrections...")
    result = convert_params_to_bvh(
        params,
        out_path,
        fps=fps,
        skip_world_grounding=True,
        corrections=ct,
        reference_params=ref_params,
    )
    progress(1.0, desc="BVH export complete")
    return result


def on_reexport_fbx(frame_idx, state, progress=gr.Progress(track_tqdm=False)):
    """Re-export corrected BVH + FBX for the selected person."""
    bvh_path = on_reexport_bvh(frame_idx, state, progress)
    if bvh_path is None:
        return None, None

    try:
        from bvh_to_fbx import convert_bvh_to_fbx
        fbx_path = str(Path(bvh_path).with_suffix(".fbx"))
        convert_bvh_to_fbx(bvh_path, fbx_path)
        return bvh_path, fbx_path
    except Exception:
        return bvh_path, None


def on_pose_frame_change(frame_idx, state):
    """Update skeleton preview and euler sliders when frame changes."""
    if state is None:
        return None, [], 0.0, 0.0, 0.0

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)
    frame_idx = int(frame_idx)
    joint_idx = state.get("pc_selected_joint")

    img, _ = _render_skeleton_preview(session_id, person_id, frame_idx, joint_idx)
    df_data = _corrections_dataframe(session_id, person_id)

    ex, ey, ez = 0.0, 0.0, 0.0
    if joint_idx is not None:
        ex, ey, ez = _get_joint_euler(session_id, person_id, frame_idx, joint_idx)

    return img, df_data, ex, ey, ez


def _space_overrides_dataframe(session_id: str, person_id: int) -> list[list]:
    """Build space overrides DataFrame rows."""
    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return []
    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        return []

    rows = []
    for o in ct.space_overrides:
        ref_str = f"Person {o.reference_person}" if o.reference_person is not None else "-"
        rows.append([o.frame_start, o.frame_end, o.space, ref_str, o.y_offset])
    return rows


def on_add_space_override(space_str, start_frame, end_frame, carrier_str,
                          y_offset, frame_idx, state):
    """Add a frame-space override for the selected person."""
    if state is None:
        return [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return [], state

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        ct = CorrectionTrack(person_id=person_id)
        sd["correction_tracks"][person_id] = ct

    space_map = {"World": "world", "Camera": "camera", "Carried": "carried"}
    space = space_map.get(space_str, "world")

    ref_person = None
    if carrier_str and carrier_str != "-":
        try:
            ref_person = int(carrier_str.split()[-1])
        except (ValueError, IndexError):
            pass

    try:
        f_start = int(start_frame)
        f_end = int(end_frame)
    except (TypeError, ValueError):
        return _space_overrides_dataframe(session_id, person_id), state

    if f_end < f_start:
        f_start, f_end = f_end, f_start

    from pose_correction import FrameSpaceOverride
    override = FrameSpaceOverride(
        frame_start=f_start,
        frame_end=f_end,
        space=space,
        reference_person=ref_person,
        y_offset=float(y_offset) if y_offset else 0.4,
    )
    ct.add_space_override(override)
    _save_correction_track(session_id, person_id)

    return _space_overrides_dataframe(session_id, person_id), state


def on_delete_space_override(evt: gr.SelectData, frame_idx, state):
    """Delete a space override from the table."""
    if state is None or evt is None:
        return [], state

    session_id = state.get("session_id", "")
    person_id = state.get("selected_person", 0)

    sd = _POSE_SESSION.get(session_id)
    if sd is None:
        return [], state

    ct = sd["correction_tracks"].get(person_id)
    if ct is None:
        return [], state

    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if row_idx < len(ct.space_overrides):
        o = ct.space_overrides[row_idx]
        ct.remove_space_override(o.frame_start, o.frame_end)
        _save_correction_track(session_id, person_id)

    return _space_overrides_dataframe(session_id, person_id), state


# ── Build Panel ──


def build_pose_correction_panel(
    frame_slider: gr.Slider,
    panel_state: gr.State,
) -> dict[str, Any]:
    """Construct the Pose Corrector Gradio components.

    Args:
        frame_slider: Shared frame slider from identity panel.
        panel_state: Shared gr.State from identity panel.

    Returns dict of component references.
    """
    with gr.Accordion("Pose Corrector", open=False) as pose_accordion:

        # Frame navigation (local to pose corrector)
        with gr.Row():
            pc_person_dropdown = gr.Dropdown(
                label="Person",
                choices=[],
                interactive=True,
                scale=1,
            )

        with gr.Row():
            pc_prev_btn = gr.Button("< Prev", size="sm", scale=0, min_width=70)
            pc_frame_num = gr.Number(
                label="Frame", precision=0, value=0, scale=1, min_width=80,
            )
            pc_next_btn = gr.Button("Next >", size="sm", scale=0, min_width=70)

        with gr.Row():
            # Left: Skeleton preview
            with gr.Column(scale=2):
                skeleton_image = gr.Image(
                    label="Skeleton Preview (click joint to select)",
                    interactive=False,
                    type="numpy",
                )

            # Right: Joint controls
            with gr.Column(scale=1):
                joint_dropdown = gr.Dropdown(
                    label="Selected Joint",
                    choices=list(BODY_JOINT_NAMES),
                    value="Pelvis",
                    interactive=True,
                )

                euler_x = gr.Slider(
                    minimum=-180, maximum=180, step=0.5, value=0,
                    label="X rotation (pitch)",
                    interactive=True,
                )
                euler_y = gr.Slider(
                    minimum=-180, maximum=180, step=0.5, value=0,
                    label="Y rotation (yaw)",
                    interactive=True,
                )
                euler_z = gr.Slider(
                    minimum=-180, maximum=180, step=0.5, value=0,
                    label="Z rotation (roll)",
                    interactive=True,
                )

                with gr.Row():
                    reset_joint_btn = gr.Button("Reset Joint", size="sm", min_width=90)
                    reset_all_btn = gr.Button("Reset All", size="sm", min_width=80)
                with gr.Row():
                    mirror_btn = gr.Button("Mirror L/R", size="sm", min_width=90)
                    flip180_btn = gr.Button("Flip 180° (Yaw)", size="sm", min_width=120)

        # Quick actions row
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    copy_frame_input = gr.Number(label="Copy from frame", precision=0)
                    copy_frame_btn = gr.Button("Copy", size="sm", min_width=60)
                with gr.Row():
                    flip_body_btn = gr.Button("Flip Whole Body", size="sm")
                    invert_btn = gr.Button("Invert Upright", size="sm")

            with gr.Column(scale=1):
                gr.Markdown("**Correction Span**")
                with gr.Row():
                    span_start = gr.Number(label="Start", precision=0)
                    span_end = gr.Number(label="End", precision=0)
                auto_span_btn = gr.Button("Auto-detect from conf dip", size="sm")

        # Corrections table
        corrections_df = gr.DataFrame(
            headers=["Frame", "Type", "Joints"],
            datatype=["number", "str", "str"],
            interactive=False,
            row_count=(4, "dynamic"),
            label="Corrections",
        )

        # ── Frame Space Override ──
        gr.Markdown("**Frame Space**")
        with gr.Row():
            space_dropdown = gr.Dropdown(
                label="Space",
                choices=["World", "Camera", "Carried"],
                value="World",
                interactive=True,
                scale=1,
            )
            space_start = gr.Number(label="From frame", precision=0, scale=1)
            space_end = gr.Number(label="To frame", precision=0, scale=1)
        with gr.Row():
            space_carrier = gr.Dropdown(
                label="Carrier Person",
                choices=[],
                interactive=True,
                scale=1,
            )
            space_y_offset = gr.Number(label="Y offset (m)", value=0.4, precision=2, scale=1)
            add_space_btn = gr.Button("Add Override", size="sm", min_width=110, scale=0)

        space_overrides_df = gr.DataFrame(
            headers=["Start", "End", "Space", "Ref Person", "Y Offset"],
            datatype=["number", "number", "str", "str", "number"],
            interactive=False,
            row_count=(3, "dynamic"),
            label="Space Overrides",
        )

        # Action buttons
        with gr.Row():
            apply_btn = gr.Button("Apply & Save", variant="primary", size="sm")
            reexport_bvh_btn = gr.Button("Re-export BVH", size="sm")
            reexport_fbx_btn = gr.Button("Re-export BVH+FBX", size="sm")

        with gr.Row():
            corrected_bvh_file = gr.File(label="Corrected BVH", visible=True)
            corrected_fbx_file = gr.File(label="Corrected FBX", visible=True)

    # ── Wire callbacks ──

    # Person dropdown within pose corrector
    pc_person_dropdown.change(
        fn=on_pc_person_change,
        inputs=[pc_person_dropdown, frame_slider, panel_state],
        outputs=[panel_state, skeleton_image, corrections_df, euler_x, euler_y, euler_z],
    )

    # Click on skeleton image to select joint
    skeleton_image.select(
        fn=on_skeleton_click,
        inputs=[frame_slider, panel_state],
        outputs=[joint_dropdown, euler_x, euler_y, euler_z, skeleton_image, panel_state],
    )

    # Joint dropdown change
    joint_dropdown.change(
        fn=on_joint_dropdown_change,
        inputs=[joint_dropdown, frame_slider, panel_state],
        outputs=[euler_x, euler_y, euler_z, skeleton_image, panel_state],
    )

    # Euler slider changes (debounced by Gradio)
    for slider in [euler_x, euler_y, euler_z]:
        slider.release(
            fn=on_euler_change,
            inputs=[euler_x, euler_y, euler_z, frame_slider, panel_state],
            outputs=[skeleton_image, corrections_df],
        )

    # Apply & Save
    apply_btn.click(
        fn=on_apply_correction,
        inputs=[frame_slider, panel_state],
        outputs=[panel_state, corrections_df],
    )

    # Reset joint
    reset_joint_btn.click(
        fn=on_reset_joint,
        inputs=[frame_slider, panel_state],
        outputs=[euler_x, euler_y, euler_z, skeleton_image, corrections_df, panel_state],
    )

    # Reset all
    reset_all_btn.click(
        fn=on_reset_all,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, panel_state],
    )

    # Flip 180° (yaw)
    flip180_btn.click(
        fn=on_flip_whole_body,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, euler_x, euler_y, euler_z, panel_state],
    )

    # Flip whole body (same as flip180 — alias)
    flip_body_btn.click(
        fn=on_flip_whole_body,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, euler_x, euler_y, euler_z, panel_state],
    )

    # Invert upright (pitch)
    invert_btn.click(
        fn=on_invert_upright,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, euler_x, euler_y, euler_z, panel_state],
    )

    # Mirror L/R
    mirror_btn.click(
        fn=on_mirror_lr,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, panel_state],
    )

    # Copy from frame
    copy_frame_btn.click(
        fn=on_copy_from_frame,
        inputs=[copy_frame_input, frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, panel_state],
    )

    # Auto-detect span
    auto_span_btn.click(
        fn=on_auto_detect_span,
        inputs=[frame_slider, panel_state],
        outputs=[span_start, span_end],
    )

    # Delete correction on table click
    corrections_df.select(
        fn=on_delete_correction,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, panel_state],
    )

    # Space override: add
    add_space_btn.click(
        fn=on_add_space_override,
        inputs=[space_dropdown, space_start, space_end, space_carrier,
                space_y_offset, frame_slider, panel_state],
        outputs=[space_overrides_df, panel_state],
    )

    # Space override: delete on table click
    space_overrides_df.select(
        fn=on_delete_space_override,
        inputs=[frame_slider, panel_state],
        outputs=[space_overrides_df, panel_state],
    )

    # Re-export BVH
    reexport_bvh_btn.click(
        fn=on_reexport_bvh,
        inputs=[frame_slider, panel_state],
        outputs=[corrected_bvh_file],
    )

    # Re-export FBX
    reexport_fbx_btn.click(
        fn=on_reexport_fbx,
        inputs=[frame_slider, panel_state],
        outputs=[corrected_bvh_file, corrected_fbx_file],
    )

    # Update skeleton + euler sliders when frame slider changes
    frame_slider.change(
        fn=on_pose_frame_change,
        inputs=[frame_slider, panel_state],
        outputs=[skeleton_image, corrections_df, euler_x, euler_y, euler_z],
    )

    # Sync shared frame_slider → local frame number display
    frame_slider.change(
        fn=lambda f: int(f),
        inputs=[frame_slider],
        outputs=[pc_frame_num],
    )

    # Local frame nav → update shared frame_slider
    pc_prev_btn.click(
        fn=lambda f: max(0, int(f) - 1),
        inputs=[frame_slider],
        outputs=[frame_slider],
    )
    pc_next_btn.click(
        fn=lambda f, s: min(
            _POSE_SESSION.get((s or {}).get("session_id", ""), {}).get("num_frames", 1) - 1,
            int(f) + 1,
        ),
        inputs=[frame_slider, panel_state],
        outputs=[frame_slider],
    )
    pc_frame_num.submit(
        fn=lambda f, s: max(0, min(
            int(f or 0),
            _POSE_SESSION.get((s or {}).get("session_id", ""), {}).get("num_frames", 1) - 1,
        )),
        inputs=[pc_frame_num, panel_state],
        outputs=[frame_slider],
    )

    return {
        "accordion": pose_accordion,
        "pc_person_dropdown": pc_person_dropdown,
        "skeleton_image": skeleton_image,
        "joint_dropdown": joint_dropdown,
        "euler_x": euler_x,
        "euler_y": euler_y,
        "euler_z": euler_z,
        "corrections_df": corrections_df,
        "corrected_bvh_file": corrected_bvh_file,
        "corrected_fbx_file": corrected_fbx_file,
        "span_start": span_start,
        "span_end": span_end,
        "space_dropdown": space_dropdown,
        "space_overrides_df": space_overrides_df,
    }
