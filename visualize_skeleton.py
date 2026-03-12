"""Skeleton overlay visualization for SMPL-X perfcap output.

Draws a colored 2D skeleton on top of the original video frames using
forward kinematics from the SMPL-X parameters. Uses the actual SMPL-X model
when available for accurate shape-dependent joint positions, falling back to
approximate FK with model-derived default offsets.
"""

from pathlib import Path

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

from smplx_to_bvh import (
    JOINT_NAMES,
    JOINT_PARENTS,
    DEFAULT_OFFSETS,
    extract_smplx_params,
    activate_coordinate_space,
    _root_translation_origin,
    _camera_to_world_orient,
    _correct_tilt,
    _smooth_rotations,
    _add_hand_mean_pose,
    _compute_root_from_contacts,
    _normalize_world_space_motion,
)
from hmr4d.utils.video_io_utils import get_stream_writer

# Try to import smplx for model-based FK
try:
    import smplx as smplx_lib
    import torch
    _HAS_SMPLX = True
except ImportError:
    _HAS_SMPLX = False

# ── Bone connections for drawing ──

BONE_CONNECTIONS = [
    # Spine / torso
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
]

# Add hand bones
for wrist, start_idx in [(20, 22), (21, 37)]:
    for finger_base in range(start_idx, start_idx + 15, 3):
        BONE_CONNECTIONS.append((wrist, finger_base))
        BONE_CONNECTIONS.append((finger_base, finger_base + 1))
        BONE_CONNECTIONS.append((finger_base + 1, finger_base + 2))

# ── Hand joint/bone subsets for zoomed overlay ──

_LEFT_HAND_JOINT_INDICES = [20] + list(range(22, 37))   # wrist + 15 finger joints
_RIGHT_HAND_JOINT_INDICES = [21] + list(range(37, 52))
_LEFT_HAND_JOINTS_SET = set(_LEFT_HAND_JOINT_INDICES)
_RIGHT_HAND_JOINTS_SET = set(_RIGHT_HAND_JOINT_INDICES)
_LEFT_HAND_BONES = [(j1, j2) for j1, j2 in BONE_CONNECTIONS
                     if j1 in _LEFT_HAND_JOINTS_SET or j2 in _LEFT_HAND_JOINTS_SET]
_RIGHT_HAND_BONES = [(j1, j2) for j1, j2 in BONE_CONNECTIONS
                      if j1 in _RIGHT_HAND_JOINTS_SET or j2 in _RIGHT_HAND_JOINTS_SET]
_LEFT_FINGERTIPS = [24, 27, 30, 33, 36]
_RIGHT_FINGERTIPS = [39, 42, 45, 48, 51]

# ── Colors (BGR) ──

_SPINE_COLOR = (255, 255, 255)
_LEFT_COLOR = (255, 128, 0)    # orange-blue
_RIGHT_COLOR = (0, 128, 255)   # orange-red
_HAND_L_COLOR = (255, 200, 0)  # cyan-ish
_HAND_R_COLOR = (0, 200, 255)  # yellow-ish

# SMPLest-X camera parameters (from smplest_x_h config)
_INTERNAL_FOCAL = 5000.0
_INPUT_BODY_H = 256
_INPUT_BODY_W = 192
_BBOX_RATIO = 1.25
_TARGET_ASPECT = _INPUT_BODY_W / _INPUT_BODY_H  # 192/256 = 0.75

# Search paths for SMPL-X model files
_SMPLX_MODEL_DIRS = [
    Path("F:/GVHMR/GVHMR/inputs/checkpoints/body_models"),
    Path("F:/SMPLest-X/human_models/human_model_files"),
    Path("/mnt/f/SMPLest-X/human_models/human_model_files"),
    Path.home() / "human_model_files",
    Path.home() / "models",
]

# SMPL-X joint indices → our joint indices mapping
# Our 0-21 = SMPLX 0-21 (body), skip jaw(22)/leye(23)/reye(24)
# Our 22-36 = SMPLX 25-39 (left hand)
# Our 37-51 = SMPLX 40-54 (right hand)
_SMPLX_TO_OURS = list(range(22)) + list(range(25, 40)) + list(range(40, 55))


def _has_smplx_neutral_model(model_dir: Path) -> bool:
    """Return True if the directory contains the neutral SMPL-X model."""
    try:
        return (model_dir / "smplx" / "SMPLX_NEUTRAL.npz").is_file()
    except OSError:
        return False


def _find_smplx_model_dir() -> Path | None:
    """Find directory containing SMPL-X model files."""
    for d in _SMPLX_MODEL_DIRS:
        if _has_smplx_neutral_model(d):
            return d
    return None


def _smplx_fk_status() -> tuple[Path | None, str | None]:
    """Report whether model-based SMPL-X FK is available."""
    if not _HAS_SMPLX:
        return None, "smplx package not installed in this Python environment"
    model_dir = _find_smplx_model_dir()
    if model_dir is None:
        return None, "no readable SMPL-X model directory found"
    return model_dir, None


def _process_bbox(bbox_xyxy):
    """Replicate SMPLest-X's process_bbox to get the actual crop region.

    SMPLest-X expands the YOLO bbox by bbox_ratio and adjusts aspect ratio
    to match the model's input shape (256x192 = 4:3). The cam_trans is
    computed relative to this processed bbox, not the raw YOLO bbox.

    Returns (x_topleft, y_topleft, width, height).
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    # Expand by ratio
    w *= _BBOX_RATIO
    h *= _BBOX_RATIO

    # Adjust aspect ratio to match model input (W:H = 3:4)
    if w / max(h, 1e-6) > _TARGET_ASPECT:
        h = w / _TARGET_ASPECT
    else:
        w = h * _TARGET_ASPECT

    return cx - w / 2, cy - h / 2, w, h


def _bone_color(j1, j2):
    """Color based on body region."""
    if j1 >= 22 or j2 >= 22:
        if j1 <= 36 or (j1 == 20 and j2 >= 22 and j2 <= 36):
            return _HAND_L_COLOR
        return _HAND_R_COLOR
    n1 = JOINT_NAMES[j1] if j1 < len(JOINT_NAMES) else ""
    n2 = JOINT_NAMES[j2] if j2 < len(JOINT_NAMES) else ""
    if "L_" in n1 or "L_" in n2:
        return _LEFT_COLOR
    if "R_" in n1 or "R_" in n2:
        return _RIGHT_COLOR
    return _SPINE_COLOR


def _select_smplx_device():
    if not _HAS_SMPLX:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _smplx_batch_size(device: str) -> int:
    if device != "cuda" or not torch.cuda.is_available():
        return 64
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_gb >= 80:
        return 512
    if total_gb >= 40:
        return 256
    return 128


def compute_all_joints_smplx(params: dict, model_dir: Path) -> np.ndarray | None:
    """Compute 3D joint positions for all frames using the actual SMPL-X model.

    Returns (N_frames, 52, 3) array in camera space, or None on failure.
    """
    if not _HAS_SMPLX:
        return None

    try:
        n_frames = params["num_frames"]
        device = _select_smplx_device()

        model = smplx_lib.create(
            str(model_dir),
            model_type="smplx",
            gender="neutral",
            use_face_contour=False,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
            ext="npz",
        ).to(device)

        # Prepare tensors — use torch directly to avoid numpy/torch conversion issues
        global_orient = torch.as_tensor(params["global_orient"], dtype=torch.float32, device=device)
        body_pose = torch.as_tensor(
            params["body_pose"].reshape(n_frames, -1), dtype=torch.float32, device=device
        )
        lhand = torch.as_tensor(
            params["left_hand_pose"].reshape(n_frames, -1), dtype=torch.float32, device=device
        )
        rhand = torch.as_tensor(
            params["right_hand_pose"].reshape(n_frames, -1), dtype=torch.float32, device=device
        )
        transl = torch.as_tensor(params["transl"], dtype=torch.float32, device=device)

        # Use mean betas (per-frame estimation has noise), properly tiled
        if "betas" in params:
            mean_b = torch.as_tensor(
                params["betas"], dtype=torch.float32, device=device
            ).mean(dim=0, keepdim=True)
            betas = mean_b.expand(n_frames, -1).contiguous()
        else:
            betas = torch.zeros(n_frames, 10, device=device)

        batch_size = _smplx_batch_size(device)
        all_joints = []
        for start in range(0, n_frames, batch_size):
            end = min(start + batch_size, n_frames)
            bs = end - start
            with torch.no_grad():
                output = model(
                    global_orient=global_orient[start:end],
                    body_pose=body_pose[start:end],
                    left_hand_pose=lhand[start:end],
                    right_hand_pose=rhand[start:end],
                    jaw_pose=torch.zeros(bs, 3, device=device),
                    leye_pose=torch.zeros(bs, 3, device=device),
                    reye_pose=torch.zeros(bs, 3, device=device),
                    expression=torch.zeros(bs, 10, device=device),
                    betas=betas[start:end],
                    transl=transl[start:end],
                    return_verts=False,
                )
            # Extract our 52 joints from model's 127
            joints = output.joints[:, _SMPLX_TO_OURS, :].detach().cpu().numpy()
            all_joints.append(joints)

        return np.concatenate(all_joints, axis=0)  # (N, 52, 3)

    except Exception as e:
        print(f"[skeleton] SMPL-X model FK failed ({e}), falling back to manual FK")
        return None


def forward_kinematics(params: dict, frame: int) -> np.ndarray:
    """Compute 3D joint positions in camera space for one frame (fallback).

    Uses model-derived rest-pose offsets (not shape-dependent).
    Returns (N_joints, 3) array.
    """
    n_joints = len(JOINT_NAMES)
    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    # Offsets in meters
    offsets = np.zeros((n_joints, 3))
    for i, name in enumerate(JOINT_NAMES):
        offsets[i] = DEFAULT_OFFSETS.get(name, [0, 0, 0])

    # Root
    accumulated_R[0] = Rotation.from_rotvec(params["global_orient"][frame]).as_matrix()
    positions[0] = params["transl"][frame]

    # Children (joint indices are already in parent-first order: 0,1,2,...,51)
    for j in range(1, n_joints):
        parent = JOINT_PARENTS[j]

        # Local rotation axis-angle
        if 1 <= j <= 21:
            rot_aa = params["body_pose"][frame, j - 1]
        elif 22 <= j <= 36:
            rot_aa = params["left_hand_pose"][frame, j - 22]
        elif 37 <= j <= 51:
            rot_aa = params["right_hand_pose"][frame, j - 37]
        else:
            rot_aa = np.zeros(3)

        R_local = Rotation.from_rotvec(rot_aa).as_matrix()
        accumulated_R[j] = accumulated_R[parent] @ R_local
        positions[j] = positions[parent] + accumulated_R[parent] @ offsets[j]

    return positions


def project_to_2d(
    positions_3d: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Perspective projection from camera-space 3D to 2D pixel coords.

    Returns (N, 2) array of (u, v) image coordinates.
    """
    z = positions_3d[:, 2:3]
    z = np.maximum(z, 0.01)  # avoid divide-by-zero

    u = fx * (positions_3d[:, 0:1] / z) + cx
    v = fy * (positions_3d[:, 1:2] / z) + cy

    return np.hstack([u, v])


def _get_projection_intrinsics(
    params: dict,
    frame_idx: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float, str]:
    """Pick the right projection model for the current frame."""
    if "K_fullimg" in params:
        k_frame = np.asarray(params["K_fullimg"][frame_idx])
        fx = float(k_frame[0, 0])
        fy = float(k_frame[1, 1])
        cx = float(k_frame[0, 2])
        cy = float(k_frame[1, 2])
        return fx, fy, cx, cy, "full-image K"

    if "bbox" in params:
        bbox_xyxy = params["bbox"][frame_idx]
        bx, by, bw, bh = _process_bbox(bbox_xyxy)
    else:
        bx, by, bw, bh = 0, 0, img_w, img_h

    fx = _INTERNAL_FOCAL / _INPUT_BODY_W * bw
    fy = _INTERNAL_FOCAL / _INPUT_BODY_H * bh
    cx = bx + bw / 2
    cy = by + bh / 2
    return fx, fy, cx, cy, "smplestx bbox camera"


def render_skeleton_video(
    pt_path: str = None,
    video_path: str = "",
    output_path: str = "",
    fps: float = 30.0,
    progress_callback=None,
    params: dict = None,
    coordinate_space: str = "camera",
    render_label: str = "Skeleton",
    prefer_nvenc: bool = True,
) -> str:
    """Render skeleton overlay on video.

    Args:
        pt_path: SMPL-X .pt file from SMPLest-X. Ignored if params is provided.
        video_path: Original input video.
        output_path: Output overlay video path.
        fps: Frame rate.
        progress_callback: Optional callable(frac, msg) for GUI progress.
        params: Pre-processed SMPL-X params dict. If provided, skips extraction.

    Returns:
        Path to output video.
    """
    if params is None:
        params = extract_smplx_params(pt_path)
    params = activate_coordinate_space(params, coordinate_space, strict=True)
    if params.get("coordinate_space") != coordinate_space:
        raise ValueError(
            f"render_skeleton_video requires {coordinate_space}-space params, "
            f"got {params.get('coordinate_space')!r}"
        )
    if "_lh_mean" not in params:
        params = _add_hand_mean_pose(params)
    n_frames = params["num_frames"]
    n_joints = len(JOINT_NAMES)
    has_bbox = "bbox" in params

    # Try model-based FK for accurate joint positions
    model_dir, smplx_fk_error = _smplx_fk_status()
    all_joints_3d = None
    if model_dir is not None:
        all_joints_3d = compute_all_joints_smplx(params, model_dir)
        if all_joints_3d is not None:
            print(f"[skeleton] Using SMPL-X model FK ({model_dir})")
    if all_joints_3d is None:
        if smplx_fk_error is not None:
            print(f"[skeleton] SMPL-X model FK unavailable: {smplx_fk_error}")
        print("[skeleton] Using manual FK with default offsets")
    projection_mode = params.get("camera_model", "smplestx_crop")
    print(f"[skeleton] Projection mode: {projection_mode}")

    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = get_stream_writer(
        output_path,
        width=img_w,
        height=img_h,
        fps=fps,
        crf=18,
        prefer_nvenc=prefer_nvenc,
    )

    for frame_idx in range(n_frames):
        ret, img = cap.read()
        if not ret:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        else:
            # Ensure writable + contiguous (OpenCV 4.13 returns read-only buffers)
            img = (img * 0.6).astype(np.uint8).copy()

        # Get 3D joint positions in camera space
        if all_joints_3d is not None:
            joints_3d = all_joints_3d[frame_idx]
        else:
            joints_3d = forward_kinematics(params, frame_idx)

        fx, fy, cx, cy, intrinsics_label = _get_projection_intrinsics(
            params,
            frame_idx,
            img_w,
            img_h,
        )

        # Project to 2D
        joints_2d = project_to_2d(joints_3d, fx, fy, cx, cy)

        # Draw bones
        for j1, j2 in BONE_CONNECTIONS:
            if j1 >= n_joints or j2 >= n_joints:
                continue
            p1 = (int(joints_2d[j1, 0]), int(joints_2d[j1, 1]))
            p2 = (int(joints_2d[j2, 0]), int(joints_2d[j2, 1]))
            # Skip wildly off-screen points
            margin = max(img_w, img_h)
            if not (-margin < p1[0] < 2 * margin and -margin < p1[1] < 2 * margin):
                continue
            if not (-margin < p2[0] < 2 * margin and -margin < p2[1] < 2 * margin):
                continue
            color = _bone_color(j1, j2)
            thickness = 1 if (j1 >= 22 or j2 >= 22) else 2
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

        # Draw body joint dots (skip hand joints for clarity)
        for j in range(min(22, n_joints)):
            p = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
            if 0 <= p[0] < img_w and 0 <= p[1] < img_h:
                cv2.circle(img, p, 4, (0, 255, 0), -1, cv2.LINE_AA)

        # Draw YOLO bbox if available
        if has_bbox:
            bbox_xyxy = params["bbox"][frame_idx]
            x1, y1 = int(bbox_xyxy[0]), int(bbox_xyxy[1])
            x2, y2 = int(bbox_xyxy[2]), int(bbox_xyxy[3])
            for pt_a, pt_b in [
                ((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1)),
            ]:
                cv2.line(img, pt_a, pt_b, (0, 255, 255), 1)

        # Frame number
        cv2.putText(
            img, f"{render_label} F{frame_idx + 1}/{n_frames}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )
        if frame_idx == 0:
            cv2.putText(
                img,
                intrinsics_label,
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (160, 200, 255),
                1,
            )

        writer.write_frame(img)

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / n_frames, f"Rendering {render_label} {frame_idx + 1}/{n_frames}")

    cap.release()
    writer.close()

    return str(output_path)


def _world_to_screen(joints_3d, cam_forward_axis, width, height, scale, center_y):
    """Project world-space joints to screen coords for a given view.

    Args:
        joints_3d: (N_joints, 3) world-space positions.
        cam_forward_axis: Which world axis the camera looks along.
            'z' = front view (camera on +Z, looking -Z): screen X = world X, screen Y = world Y
            'x' = side view (camera on +X, looking -X): screen X = world Z, screen Y = world Y
        width, height: Frame dimensions.
        scale: Pixels per meter.
        center_y: World Y coordinate mapped to vertical center of frame.

    Returns:
        (N_joints, 2) screen coordinates.
    """
    pts_2d = np.zeros((len(joints_3d), 2))
    if cam_forward_axis == 'z':
        # Front view: X right, Y up
        pts_2d[:, 0] = width / 2 + joints_3d[:, 0] * scale
        pts_2d[:, 1] = height / 2 - (joints_3d[:, 1] - center_y) * scale
    else:
        # Side view: Z right, Y up
        pts_2d[:, 0] = width / 2 - joints_3d[:, 2] * scale
        pts_2d[:, 1] = height / 2 - (joints_3d[:, 1] - center_y) * scale
    return pts_2d


def render_world_views(
    pt_path: str = None,
    output_path: str = "",
    fps: float = 30.0,
    width: int = 640,
    height: int = 480,
    pitch_adjust_deg: float = 0.0,
    progress_callback=None,
    params: dict = None,
    skip_world_grounding: bool = False,
    prefer_nvenc: bool = True,
) -> str:
    """Render world-space front and side view skeleton video.

    Produces a side-by-side video (2*width x height) showing the skeleton
    from the front (+Z) and side (+X) after tilt correction and foot grounding.

    Args:
        pt_path: SMPL-X .pt file path. Ignored if params is provided.
        output_path: Output video path.
        fps: Frame rate.
        width: Width of each panel.
        height: Height of each panel.
        pitch_adjust_deg: Manual pitch adjustment on top of auto tilt correction.
        progress_callback: Optional callable(frac, msg).
        params: Pre-processed SMPL-X params dict. If provided, skips extraction
                and (optionally) world-grounding steps.
        skip_world_grounding: If True and params is provided, preserve the incoming
                world-space Y and only normalize X/Z plus heading.

    Returns:
        Path to the output video.
    """
    if params is None:
        # Load and process params identically to BVH pipeline
        params = extract_smplx_params(pt_path)
        params = _add_hand_mean_pose(params)
        params = _camera_to_world_orient(params)
        params = _correct_tilt(params, pitch_adjust_deg=pitch_adjust_deg)
        params = _smooth_rotations(params, window=5)
        params["transl"] = _compute_root_from_contacts(params)
        params["translation_origin"] = "pelvis"
    elif not skip_world_grounding:
        # params provided but still need world grounding
        params = _add_hand_mean_pose(params)
        params = _camera_to_world_orient(params)
        params = _correct_tilt(params, pitch_adjust_deg=pitch_adjust_deg)
        params = _smooth_rotations(params, window=5)
        params["transl"] = _compute_root_from_contacts(params)
        params["translation_origin"] = "pelvis"
    else:
        params = _add_hand_mean_pose(params)
        params = _normalize_world_space_motion(params)

    n_frames = params["num_frames"]
    n_joints = len(JOINT_NAMES)

    model_dir, smplx_fk_error = _smplx_fk_status()
    all_joints = None
    if model_dir is not None:
        all_joints = compute_all_joints_smplx(params, model_dir)
        if all_joints is not None:
            print(f"[world] Using SMPL-X model FK ({model_dir})")
    if all_joints is None:
        if smplx_fk_error is not None:
            print(f"[world] SMPL-X model FK unavailable: {smplx_fk_error}")
        print("[world] Using manual FK with default offsets")
        all_joints = np.zeros((n_frames, n_joints, 3))
        for f in range(n_frames):
            all_joints[f] = forward_kinematics(params, f)

    # Compute scale from bounding box of all joints across all frames
    all_flat = all_joints[:, :22, :].reshape(-1, 3)  # body joints only for framing
    y_min, y_max = all_flat[:, 1].min(), all_flat[:, 1].max()
    x_range = all_flat[:, 0].max() - all_flat[:, 0].min()
    z_range = all_flat[:, 2].max() - all_flat[:, 2].min()
    body_height = y_max - y_min
    max_span = max(body_height, x_range, z_range, 0.5)

    # Scale so skeleton fills ~75% of frame height
    scale = (height * 0.75) / max_span
    center_y = (y_min + y_max) / 2

    # Ground line Y in screen coords
    ground_screen_y = int(height / 2 - (0.0 - center_y) * scale)

    panel_w = width
    total_w = panel_w * 2
    bg_color = (40, 40, 40)  # dark gray
    ground_color = (80, 80, 80)
    writer = get_stream_writer(
        output_path,
        width=total_w,
        height=height,
        fps=fps,
        crf=18,
        prefer_nvenc=prefer_nvenc,
    )

    for f in range(n_frames):
        frame_img = np.full((height, total_w, 3), bg_color, dtype=np.uint8)

        joints_3d = all_joints[f]

        for panel_idx, (axis, label, x_off) in enumerate([
            ('z', 'FRONT', 0),
            ('x', 'SIDE', panel_w),
        ]):
            pts = _world_to_screen(joints_3d, axis, panel_w, height, scale, center_y)
            pts[:, 0] += x_off  # offset for panel

            # Ground line
            cv2.line(frame_img, (x_off, ground_screen_y),
                     (x_off + panel_w - 1, ground_screen_y), ground_color, 1)

            # Bones
            for j1, j2 in BONE_CONNECTIONS:
                if j1 >= n_joints or j2 >= n_joints:
                    continue
                p1 = (int(pts[j1, 0]), int(pts[j1, 1]))
                p2 = (int(pts[j2, 0]), int(pts[j2, 1]))
                color = _bone_color(j1, j2)
                thickness = 1 if (j1 >= 22 or j2 >= 22) else 2
                cv2.line(frame_img, p1, p2, color, thickness, cv2.LINE_AA)

            # Body joint dots
            for j in range(min(22, n_joints)):
                p = (int(pts[j, 0]), int(pts[j, 1]))
                cv2.circle(frame_img, p, 3, (0, 255, 0), -1, cv2.LINE_AA)

            # Panel label
            cv2.putText(frame_img, label, (x_off + 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Divider line between panels
        cv2.line(frame_img, (panel_w, 0), (panel_w, height - 1), (100, 100, 100), 1)

        # Frame counter
        cv2.putText(frame_img, f"F{f + 1}/{n_frames}", (total_w - 130, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        writer.write_frame(frame_img)

        if progress_callback and f % 30 == 0:
            progress_callback(f / n_frames, f"World view {f + 1}/{n_frames}")

    writer.close()

    return str(output_path)


def render_hand_overlay_video(
    pt_path: str = None,
    video_path: str = "",
    output_path: str = "",
    fps: float = 30.0,
    progress_callback=None,
    params: dict = None,
    coordinate_space: str = "camera",
    output_size: int = 512,
    render_label: str = "Hands",
    prefer_nvenc: bool = True,
) -> str:
    """Render zoomed hand overlay video (left | right side-by-side).

    Produces a 2*output_size x output_size video showing each hand zoomed in
    with skeleton drawn on top of the cropped video frame.

    Args:
        pt_path: SMPL-X .pt file. Ignored if params is provided.
        video_path: Original input video.
        output_path: Output overlay video path.
        fps: Frame rate.
        progress_callback: Optional callable(frac, msg).
        params: Pre-processed SMPL-X params dict.
        coordinate_space: Coordinate space (default "camera").
        output_size: Size of each hand panel (output is 2*output_size x output_size).
        render_label: Label prefix for frame counter.
        prefer_nvenc: Use NVENC if available.

    Returns:
        Path to output video.
    """
    if params is None:
        params = extract_smplx_params(pt_path)
    params = activate_coordinate_space(params, coordinate_space, strict=True)
    if params.get("coordinate_space") != coordinate_space:
        raise ValueError(
            f"render_hand_overlay_video requires {coordinate_space}-space params, "
            f"got {params.get('coordinate_space')!r}"
        )
    if "_lh_mean" not in params:
        params = _add_hand_mean_pose(params)
    n_frames = params["num_frames"]
    n_joints = len(JOINT_NAMES)

    # FK
    model_dir, smplx_fk_error = _smplx_fk_status()
    all_joints_3d = None
    if model_dir is not None:
        all_joints_3d = compute_all_joints_smplx(params, model_dir)
        if all_joints_3d is not None:
            print(f"[hands] Using SMPL-X model FK ({model_dir})")
    if all_joints_3d is None:
        if smplx_fk_error is not None:
            print(f"[hands] SMPL-X model FK unavailable: {smplx_fk_error}")
        print("[hands] Using manual FK with default offsets")

    cap = cv2.VideoCapture(video_path)
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_w = output_size * 2
    total_h = output_size

    writer = get_stream_writer(
        output_path,
        width=total_w,
        height=total_h,
        fps=fps,
        crf=18,
        prefer_nvenc=prefer_nvenc,
    )

    # EMA state for bbox smoothing (cx, cy, size) per hand
    bbox_ema = [None, None]  # [left, right]
    ema_alpha = 0.7

    hand_sides = [
        (_LEFT_HAND_JOINT_INDICES, _LEFT_HAND_BONES, _HAND_L_COLOR, _LEFT_FINGERTIPS, "L", 20),
        (_RIGHT_HAND_JOINT_INDICES, _RIGHT_HAND_BONES, _HAND_R_COLOR, _RIGHT_FINGERTIPS, "R", 21),
    ]

    for frame_idx in range(n_frames):
        ret, img = cap.read()
        if not ret:
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Dim once for all crops
        dimmed = (img * 0.6).astype(np.uint8)

        # 3D joints
        if all_joints_3d is not None:
            joints_3d = all_joints_3d[frame_idx]
        else:
            joints_3d = forward_kinematics(params, frame_idx)

        # Project to 2D
        fx, fy, cx, cy, _ = _get_projection_intrinsics(params, frame_idx, img_w, img_h)
        joints_2d = project_to_2d(joints_3d, fx, fy, cx, cy)

        out_frame = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        for side_idx, (joint_indices, bones, color, fingertips, label, wrist_idx) in enumerate(hand_sides):
            panel_x_off = side_idx * output_size
            hand_pts = joints_2d[joint_indices]  # (16, 2)

            # Check if hand joints are reasonably on-screen
            margin = max(img_w, img_h)
            in_frame = (
                (hand_pts[:, 0] > -margin) & (hand_pts[:, 0] < 2 * margin) &
                (hand_pts[:, 1] > -margin) & (hand_pts[:, 1] < 2 * margin)
            )
            if not in_frame.any():
                cv2.putText(
                    out_frame, f"{label} (off-screen)",
                    (panel_x_off + 10, output_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1,
                )
                continue

            # Bbox from valid hand joints — include wrist with extra padding
            # so the full hand, wrist, and a bit of forearm are always visible.
            valid_pts = hand_pts[in_frame]
            wrist_pt = joints_2d[wrist_idx]  # always include wrist
            all_pts = np.vstack([valid_pts, wrist_pt[np.newaxis]])
            x_min, y_min = all_pts.min(axis=0)
            x_max, y_max = all_pts.max(axis=0)
            bbox_cx = (x_min + x_max) / 2
            bbox_cy = (y_min + y_max) / 2
            bbox_size = max(x_max - x_min, y_max - y_min)

            # 150% margin — keeps full wrist and spread fingers in frame
            bbox_size = max(bbox_size * 2.5, 80)

            # EMA smoothing
            if bbox_ema[side_idx] is None:
                bbox_ema[side_idx] = (bbox_cx, bbox_cy, bbox_size)
            else:
                prev = bbox_ema[side_idx]
                bbox_ema[side_idx] = (
                    ema_alpha * bbox_cx + (1 - ema_alpha) * prev[0],
                    ema_alpha * bbox_cy + (1 - ema_alpha) * prev[1],
                    ema_alpha * bbox_size + (1 - ema_alpha) * prev[2],
                )
            s_cx, s_cy, s_size = bbox_ema[side_idx]

            # Square crop region clipped to frame
            half = s_size / 2
            crop_x1 = int(max(0, s_cx - half))
            crop_y1 = int(max(0, s_cy - half))
            crop_x2 = int(min(img_w, s_cx + half))
            crop_y2 = int(min(img_h, s_cy + half))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            # Crop and resize with letterbox
            crop = dimmed[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_h_actual, crop_w_actual = crop.shape[:2]
            scale_factor = output_size / max(crop_h_actual, crop_w_actual, 1)
            new_w = int(crop_w_actual * scale_factor)
            new_h = int(crop_h_actual * scale_factor)
            resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            panel = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            pad_x = (output_size - new_w) // 2
            pad_y = (output_size - new_h) // 2
            panel[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

            # Map full-frame joint coords → panel coords
            def _to_panel(pt, _cx1=crop_x1, _cy1=crop_y1, _sf=scale_factor, _px=pad_x, _py=pad_y):
                return (int((pt[0] - _cx1) * _sf + _px),
                        int((pt[1] - _cy1) * _sf + _py))

            # Draw bones
            for j1, j2 in bones:
                if j1 >= n_joints or j2 >= n_joints:
                    continue
                p1 = _to_panel(joints_2d[j1])
                p2 = _to_panel(joints_2d[j2])
                cv2.line(panel, p1, p2, color, 2, cv2.LINE_AA)

            # Draw fingertip dots
            for ft in fingertips:
                if ft < n_joints:
                    p = _to_panel(joints_2d[ft])
                    if 0 <= p[0] < output_size and 0 <= p[1] < output_size:
                        cv2.circle(panel, p, 5, (0, 255, 0), -1, cv2.LINE_AA)

            # Wrist dot
            wp = _to_panel(joints_2d[wrist_idx])
            if 0 <= wp[0] < output_size and 0 <= wp[1] < output_size:
                cv2.circle(panel, wp, 5, (255, 255, 255), -1, cv2.LINE_AA)

            # Panel label
            cv2.putText(panel, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            out_frame[:, panel_x_off:panel_x_off + output_size] = panel

        # Divider between panels
        cv2.line(out_frame, (output_size, 0), (output_size, total_h - 1), (100, 100, 100), 1)

        # Frame counter
        cv2.putText(
            out_frame, f"{render_label} F{frame_idx + 1}/{n_frames}",
            (total_w - 220, total_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )

        writer.write_frame(out_frame)

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / n_frames, f"Rendering {render_label} {frame_idx + 1}/{n_frames}")

    cap.release()
    writer.close()

    return str(output_path)


def render_skeleton_frame(
    params: dict,
    frame_idx: int,
    video_frame: np.ndarray,
    selected_joint: int | None = None,
    corrections=None,
    dim_factor: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Render skeleton overlay on a single video frame for interactive use.

    Args:
        params: SMPL-X params dict (camera-space).
        frame_idx: Frame index to render.
        video_frame: RGB video frame (H, W, 3).
        selected_joint: If set, highlight this joint index with a yellow ring.
        corrections: Optional CorrectionTrack — applied before FK.
        dim_factor: Darken the background frame (0-1).

    Returns:
        (annotated_image, joints_2d) where annotated_image is RGB (H, W, 3)
        and joints_2d is (N_joints, 2) pixel coordinates.
    """
    if corrections is not None:
        from pose_correction import apply_corrections
        params = apply_corrections(params, corrections)

    n_joints = len(JOINT_NAMES)
    img_h, img_w = video_frame.shape[:2]

    # Dim background
    img = (video_frame.astype(np.float32) * dim_factor).astype(np.uint8).copy()

    # FK
    joints_3d = forward_kinematics(params, frame_idx)

    # Project
    fx, fy, cx, cy, _ = _get_projection_intrinsics(params, frame_idx, img_w, img_h)
    joints_2d = project_to_2d(joints_3d, fx, fy, cx, cy)

    # Draw bones (BGR for cv2)
    margin = max(img_w, img_h)
    for j1, j2 in BONE_CONNECTIONS:
        if j1 >= n_joints or j2 >= n_joints:
            continue
        # Skip hand bones for clarity in interactive mode
        if j1 >= 22 or j2 >= 22:
            continue
        p1 = (int(joints_2d[j1, 0]), int(joints_2d[j1, 1]))
        p2 = (int(joints_2d[j2, 0]), int(joints_2d[j2, 1]))
        if not (-margin < p1[0] < 2 * margin and -margin < p1[1] < 2 * margin):
            continue
        if not (-margin < p2[0] < 2 * margin and -margin < p2[1] < 2 * margin):
            continue
        color = _bone_color(j1, j2)
        # Convert BGR bone color to RGB for the output
        color_rgb = (color[2], color[1], color[0])
        cv2.line(img, p1, p2, color_rgb, 2, cv2.LINE_AA)

    # Draw body joint dots
    for j in range(min(22, n_joints)):
        p = (int(joints_2d[j, 0]), int(joints_2d[j, 1]))
        if not (0 <= p[0] < img_w and 0 <= p[1] < img_h):
            continue

        # Color by region
        name = JOINT_NAMES[j] if j < len(JOINT_NAMES) else ""
        if "L_" in name:
            dot_color = (255, 128, 0)  # orange (RGB)
        elif "R_" in name:
            dot_color = (0, 128, 255)  # blue (RGB)
        else:
            dot_color = (0, 255, 0)    # green (RGB)

        radius = 6
        cv2.circle(img, p, radius, dot_color, -1, cv2.LINE_AA)

        # Selected joint highlight
        if j == selected_joint:
            cv2.circle(img, p, 10, (255, 255, 0), 2, cv2.LINE_AA)  # yellow ring
            cv2.circle(img, p, 14, (255, 255, 0), 1, cv2.LINE_AA)  # outer ring
            # Label
            label = JOINT_NAMES[j] if j < len(JOINT_NAMES) else f"J{j}"
            cv2.putText(
                img, label,
                (p[0] + 16, p[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA,
            )

    return img, joints_2d


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualize_skeleton.py <smplx_output.pt> <video_path> [output.mp4] [fps]")
        sys.exit(1)

    pt = sys.argv[1]
    video = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else str(Path(pt).with_suffix("")) + "_skeleton.mp4"
    fps = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0

    print(f"Rendering skeleton overlay...")
    result = render_skeleton_video(pt, video, out, fps=fps)
    print(f"Written: {result}")
