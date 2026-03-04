"""SMPL-X to BVH converter — body + hands skeleton animation."""

import struct
from pathlib import Path

import numpy as np
import torch

# Attempt to import smplx for forward kinematics; fall back to manual computation
try:
    import smplx as smplx_lib
    HAS_SMPLX = True
except ImportError:
    HAS_SMPLX = False

from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d


# ── SMPL-X Joint Hierarchy ──
# 22 body joints + 15 left hand + 15 right hand = 52 joints
# We skip jaw (22), left eye (23), right eye (24) — driven by ARKit blendshapes

JOINT_NAMES = [
    # Body (0-21)
    "Pelvis",           # 0 - root
    "L_Hip",            # 1
    "R_Hip",            # 2
    "Spine1",           # 3
    "L_Knee",           # 4
    "R_Knee",           # 5
    "Spine2",           # 6
    "L_Ankle",          # 7
    "R_Ankle",          # 8
    "Spine3",           # 9
    "L_Foot",           # 10
    "R_Foot",           # 11
    "Neck",             # 12
    "L_Collar",         # 13
    "R_Collar",         # 14
    "Head",             # 15
    "L_Shoulder",       # 16
    "R_Shoulder",       # 17
    "L_Elbow",          # 18
    "R_Elbow",          # 19
    "L_Wrist",          # 20
    "R_Wrist",          # 21
    # Left Hand (22-36) — SMPL-X indices 25-39
    "L_Index1",         # 22
    "L_Index2",         # 23
    "L_Index3",         # 24
    "L_Middle1",        # 25
    "L_Middle2",        # 26
    "L_Middle3",        # 27
    "L_Pinky1",         # 28
    "L_Pinky2",         # 29
    "L_Pinky3",         # 30
    "L_Ring1",          # 31
    "L_Ring2",          # 32
    "L_Ring3",          # 33
    "L_Thumb1",         # 34
    "L_Thumb2",         # 35
    "L_Thumb3",         # 36
    # Right Hand (37-51) — SMPL-X indices 40-54
    "R_Index1",         # 37
    "R_Index2",         # 38
    "R_Index3",         # 39
    "R_Middle1",        # 40
    "R_Middle2",        # 41
    "R_Middle3",        # 42
    "R_Pinky1",         # 43
    "R_Pinky2",         # 44
    "R_Pinky3",         # 45
    "R_Ring1",          # 46
    "R_Ring2",          # 47
    "R_Ring3",          # 48
    "R_Thumb1",         # 49
    "R_Thumb2",         # 50
    "R_Thumb3",         # 51
]

# Parent indices in the BVH hierarchy (-1 = root)
JOINT_PARENTS = [
    -1,  # 0  Pelvis (root)
    0,   # 1  L_Hip -> Pelvis
    0,   # 2  R_Hip -> Pelvis
    0,   # 3  Spine1 -> Pelvis
    1,   # 4  L_Knee -> L_Hip
    2,   # 5  R_Knee -> R_Hip
    3,   # 6  Spine2 -> Spine1
    4,   # 7  L_Ankle -> L_Knee
    5,   # 8  R_Ankle -> R_Knee
    6,   # 9  Spine3 -> Spine2
    7,   # 10 L_Foot -> L_Ankle
    8,   # 11 R_Foot -> R_Ankle
    9,   # 12 Neck -> Spine3
    9,   # 13 L_Collar -> Spine3
    9,   # 14 R_Collar -> Spine3
    12,  # 15 Head -> Neck
    13,  # 16 L_Shoulder -> L_Collar
    14,  # 17 R_Shoulder -> R_Collar
    16,  # 18 L_Elbow -> L_Shoulder
    17,  # 19 R_Elbow -> R_Shoulder
    18,  # 20 L_Wrist -> L_Elbow
    19,  # 21 R_Wrist -> R_Elbow
    # Left hand -> L_Wrist
    20, 22, 23,  # L_Index 1,2,3
    20, 25, 26,  # L_Middle 1,2,3
    20, 28, 29,  # L_Pinky 1,2,3
    20, 31, 32,  # L_Ring 1,2,3
    20, 34, 35,  # L_Thumb 1,2,3
    # Right hand -> R_Wrist
    21, 37, 38,  # R_Index 1,2,3
    21, 40, 41,  # R_Middle 1,2,3
    21, 43, 44,  # R_Pinky 1,2,3
    21, 46, 47,  # R_Ring 1,2,3
    21, 49, 50,  # R_Thumb 1,2,3
]

# Default T-pose offsets (meters) — from SMPL-X NEUTRAL model (zero betas)
# These are parent-relative offsets in the rest pose. Pelvis offset is the
# absolute position in the model's canonical frame (≈35cm below origin).
DEFAULT_OFFSETS = {
    # Body
    "Pelvis": [0.003, -0.351, 0.012],
    "L_Hip": [0.058, -0.093, -0.026],
    "R_Hip": [-0.063, -0.104, -0.021],
    "Spine1": [-0.003, 0.110, -0.028],
    "L_Knee": [0.055, -0.379, -0.009],
    "R_Knee": [-0.044, -0.362, -0.017],
    "Spine2": [0.009, 0.132, -0.006],
    "L_Ankle": [-0.043, -0.403, -0.032],
    "R_Ankle": [0.015, -0.411, -0.020],
    "Spine3": [-0.011, 0.052, 0.028],
    "L_Foot": [0.047, -0.058, 0.118],
    "R_Foot": [-0.039, -0.058, 0.119],
    "Neck": [-0.012, 0.165, -0.032],
    "L_Collar": [0.046, 0.085, -0.007],
    "R_Collar": [-0.048, 0.084, -0.013],
    "Head": [0.025, 0.160, 0.021],
    "L_Shoulder": [0.119, 0.058, -0.015],
    "R_Shoulder": [-0.103, 0.054, -0.013],
    "L_Elbow": [0.254, -0.072, -0.042],
    "R_Elbow": [-0.271, -0.036, -0.026],
    "L_Wrist": [0.252, 0.023, -0.002],
    "R_Wrist": [-0.249, -0.005, -0.015],
    # Left hand
    "L_Index1": [0.102, -0.009, 0.019],
    "L_Index2": [0.032, 0.002, 0.003],
    "L_Index3": [0.023, -0.002, 0.000],
    "L_Middle1": [0.109, -0.006, -0.004],
    "L_Middle2": [0.031, 0.001, -0.004],
    "L_Middle3": [0.024, -0.002, -0.004],
    "L_Pinky1": [0.084, -0.015, -0.044],
    "L_Pinky2": [0.015, -0.001, -0.012],
    "L_Pinky3": [0.016, -0.002, -0.011],
    "L_Ring1": [0.097, -0.009, -0.027],
    "L_Ring2": [0.028, 0.001, -0.005],
    "L_Ring3": [0.023, -0.001, -0.007],
    "L_Thumb1": [0.041, -0.018, 0.026],
    "L_Thumb2": [0.017, 0.001, 0.025],
    "L_Thumb3": [0.021, -0.005, 0.016],
    # Right hand
    "R_Index1": [-0.100, -0.012, 0.020],
    "R_Index2": [-0.032, 0.002, 0.003],
    "R_Index3": [-0.023, -0.002, 0.000],
    "R_Middle1": [-0.107, -0.009, -0.004],
    "R_Middle2": [-0.031, 0.001, -0.004],
    "R_Middle3": [-0.024, -0.002, -0.004],
    "R_Pinky1": [-0.082, -0.018, -0.044],
    "R_Pinky2": [-0.015, -0.001, -0.012],
    "R_Pinky3": [-0.016, -0.002, -0.011],
    "R_Ring1": [-0.095, -0.012, -0.027],
    "R_Ring2": [-0.028, 0.001, -0.005],
    "R_Ring3": [-0.023, -0.001, -0.007],
    "R_Thumb1": [-0.039, -0.021, 0.026],
    "R_Thumb2": [-0.017, 0.001, 0.025],
    "R_Thumb3": [-0.021, -0.005, 0.016],
}


def axis_angle_to_euler_zxy(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotations to ZXY Euler angles (BVH convention).

    Args:
        axis_angle: Array of shape (..., 3) axis-angle rotations.

    Returns:
        Array of same leading shape + (3,) with ZXY Euler angles in degrees.
    """
    original_shape = axis_angle.shape
    flat = axis_angle.reshape(-1, 3)

    # Handle zero rotations
    angles = np.linalg.norm(flat, axis=1, keepdims=True)
    # Avoid division by zero
    safe_angles = np.where(angles > 1e-8, angles, np.ones_like(angles))
    axes = flat / safe_angles

    rotations = Rotation.from_rotvec(flat)
    euler = rotations.as_euler("ZXY", degrees=True)

    result_shape = original_shape[:-1] + (3,)
    return euler.reshape(result_shape)


def extract_smplx_params(pt_path: str) -> dict:
    """Extract SMPL-X parameters from SMPLest-X output .pt file.

    Returns dict with:
        - global_orient: (N, 3) global orientation axis-angle
        - body_pose: (N, 21*3) or (N, 21, 3) body joint rotations
        - left_hand_pose: (N, 15*3) or (N, 15, 3) left hand rotations
        - right_hand_pose: (N, 15*3) or (N, 15, 3) right hand rotations
        - transl: (N, 3) root translation
        - num_frames: int
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # Handle different output formats from SMPLest-X
    if isinstance(data, dict):
        params = data
    elif isinstance(data, list):
        # List of per-frame results — stack them
        params = {}
        keys = data[0].keys() if data else []
        for k in keys:
            vals = [d[k] for d in data if k in d]
            if isinstance(vals[0], (torch.Tensor, np.ndarray)):
                params[k] = torch.stack([torch.as_tensor(v) for v in vals])
    else:
        raise ValueError(f"Unexpected .pt format: {type(data)}")

    result = {}

    # Global orientation
    for key in ["global_orient", "root_orient", "smplx_root_pose"]:
        if key in params:
            result["global_orient"] = np.array(params[key]).reshape(-1, 3)
            break
    if "global_orient" not in result:
        raise KeyError(f"No global orientation found. Keys: {list(params.keys())}")

    n_frames = len(result["global_orient"])

    # Body pose (21 joints)
    for key in ["body_pose", "smplx_body_pose"]:
        if key in params:
            result["body_pose"] = np.array(params[key]).reshape(n_frames, 21, 3)
            break
    if "body_pose" not in result:
        result["body_pose"] = np.zeros((n_frames, 21, 3))

    # Hand poses (15 joints each)
    for key in ["left_hand_pose", "smplx_lhand_pose", "lhand_pose"]:
        if key in params:
            result["left_hand_pose"] = np.array(params[key]).reshape(n_frames, 15, 3)
            break
    if "left_hand_pose" not in result:
        result["left_hand_pose"] = np.zeros((n_frames, 15, 3))

    for key in ["right_hand_pose", "smplx_rhand_pose", "rhand_pose"]:
        if key in params:
            result["right_hand_pose"] = np.array(params[key]).reshape(n_frames, 15, 3)
            break
    if "right_hand_pose" not in result:
        result["right_hand_pose"] = np.zeros((n_frames, 15, 3))

    # Translation
    for key in ["transl", "trans", "smplx_transl", "root_transl", "cam_trans"]:
        if key in params:
            result["transl"] = np.array(params[key]).reshape(n_frames, 3)
            break
    if "transl" not in result:
        result["transl"] = np.zeros((n_frames, 3))

    # Shape (beta) parameters — needed for shape-aware FK
    for key in ["betas", "smplx_shape", "shape"]:
        if key in params:
            result["betas"] = np.array(params[key]).reshape(n_frames, -1)
            break

    # Bounding boxes (for visualization)
    for key in ["bbox", "bboxes", "person_bbox", "bb_xyxy"]:
        if key in params:
            bbox_data = np.array(params[key])
            if bbox_data.ndim == 1:
                bbox_data = bbox_data.reshape(1, -1)
            if bbox_data.shape[-1] >= 4:
                result["bbox"] = bbox_data[:n_frames, :4].reshape(n_frames, 4)
            break

    result["num_frames"] = n_frames
    return result


def extract_gvhmr_params(pt_path: str) -> dict:
    """Load GVHMR hmr4d_results.pt → SMPL-X-compatible params dict.

    GVHMR's smpl_params_global contains world-frame, Y-up, meters data.
    body_pose is (N, 63) flattened — reshape to (N, 21, 3).
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    g = data["smpl_params_global"]
    n = g["body_pose"].shape[0]
    return {
        "global_orient": np.array(g["global_orient"]).reshape(n, 3),
        "body_pose": np.array(g["body_pose"]).reshape(n, 21, 3),
        "left_hand_pose": np.zeros((n, 15, 3)),
        "right_hand_pose": np.zeros((n, 15, 3)),
        "transl": np.array(g["transl"]).reshape(n, 3),
        "betas": np.array(g.get("betas", torch.zeros(n, 10))).reshape(n, -1),
        "num_frames": n,
    }


def merge_gvhmr_smplestx_params(gvhmr: dict, smplestx: dict) -> dict:
    """Merge GVHMR body/translation with SMPLest-X hand poses.

    Takes body_pose, global_orient, transl, betas from GVHMR.
    Takes left_hand_pose, right_hand_pose from SMPLest-X.
    Truncates both to min(N_gvhmr, N_smplestx) frames.
    """
    n_gvhmr = gvhmr["num_frames"]
    n_smplestx = smplestx["num_frames"]
    n = min(n_gvhmr, n_smplestx)

    diff = abs(n_gvhmr - n_smplestx)
    max_n = max(n_gvhmr, n_smplestx)
    if diff > 5 and (max_n == 0 or diff / max_n > 0.02):
        print(f"[merge] WARNING: Frame count mismatch — GVHMR={n_gvhmr}, SMPLest-X={n_smplestx} (diff={diff})")
    elif diff > 0:
        print(f"[merge] Truncating to {n} frames (GVHMR={n_gvhmr}, SMPLest-X={n_smplestx})")

    return {
        "global_orient": gvhmr["global_orient"][:n],
        "body_pose": gvhmr["body_pose"][:n],
        "left_hand_pose": smplestx["left_hand_pose"][:n],
        "right_hand_pose": smplestx["right_hand_pose"][:n],
        "transl": gvhmr["transl"][:n],
        "betas": gvhmr["betas"][:n] if "betas" in gvhmr else np.zeros((n, 10)),
        "num_frames": n,
    }


def _compute_ground_offset_cm() -> float:
    """Compute the Y offset (cm) to add to the ROOT so feet rest at Y=0.

    Walks both leg chains in the rest pose (zero rotations) and returns
    the offset that places the lowest foot at Y=0.
    """
    left_chain = ["L_Hip", "L_Knee", "L_Ankle", "L_Foot"]
    right_chain = ["R_Hip", "R_Knee", "R_Ankle", "R_Foot"]

    left_y = sum(DEFAULT_OFFSETS[j][1] for j in left_chain) * 100  # m → cm
    right_y = sum(DEFAULT_OFFSETS[j][1] for j in right_chain) * 100

    # Ground level = most negative foot Y. Offset = -ground_y so feet sit at 0.
    ground_y = min(left_y, right_y)
    return -ground_y


# Precompute pelvis height above ground (cm) for root translation
PELVIS_HEIGHT_CM = _compute_ground_offset_cm()


def _build_hierarchy_string(joint_idx: int, depth: int, offsets: dict,
                            pelvis_height_cm: float) -> str:
    """Recursively build BVH HIERARCHY section."""
    indent = "\t" * depth
    name = JOINT_NAMES[joint_idx]
    offset = offsets.get(name, [0.0, 0.0, 0.0])
    # Scale from meters to centimeters for BVH
    ox, oy, oz = [v * 100 for v in offset]

    # ROOT: place pelvis at correct height with feet at Y=0, centered at origin
    if joint_idx == 0:
        ox, oy, oz = 0.0, pelvis_height_cm, 0.0

    children = [i for i, p in enumerate(JOINT_PARENTS) if p == joint_idx]

    if joint_idx == 0:
        lines = [f"{indent}ROOT {name}"]
    else:
        lines = [f"{indent}JOINT {name}"]

    lines.append(f"{indent}{{")
    lines.append(f"{indent}\tOFFSET {ox:.4f} {oy:.4f} {oz:.4f}")

    if joint_idx == 0:
        lines.append(f"{indent}\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")

    if not children:
        if name == "Head" and "_Head_EndSite" in offsets:
            # Use precomputed EndSite from _remap_offsets_for_bvh (~14cm)
            es = offsets["_Head_EndSite"]
            ex, ey, ez = [v * 100 for v in es]
        else:
            # End site — extend last bone by 50% of its own offset for proper bone length
            ex, ey, ez = ox * 0.5, oy * 0.5, oz * 0.5
        lines.append(f"{indent}\tEnd Site")
        lines.append(f"{indent}\t{{")
        lines.append(f"{indent}\t\tOFFSET {ex:.4f} {ey:.4f} {ez:.4f}")
        lines.append(f"{indent}\t}}")
    else:
        for child in children:
            lines.extend(
                _build_hierarchy_string(child, depth + 1, offsets,
                                        pelvis_height_cm).split("\n"))

    lines.append(f"{indent}}}")
    return "\n".join(lines)


def _get_traversal_order(joint_idx=0):
    """Get depth-first traversal order of the joint hierarchy.

    This MUST match the order joints appear in the BVH HIERARCHY section,
    since BVH MOTION data is written in hierarchy traversal order.
    """
    order = [joint_idx]
    children = [i for i, p in enumerate(JOINT_PARENTS) if p == joint_idx]
    for child in children:
        order.extend(_get_traversal_order(child))
    return order


# Precompute at module level
TRAVERSAL_ORDER = _get_traversal_order(0)


def _get_joint_rotation(params: dict, frame: int, joint_idx: int) -> np.ndarray:
    """Get axis-angle rotation for a specific joint at a specific frame."""
    if joint_idx == 0:
        return params["global_orient"][frame]
    elif 1 <= joint_idx <= 21:
        return params["body_pose"][frame, joint_idx - 1]
    elif 22 <= joint_idx <= 36:
        return params["left_hand_pose"][frame, joint_idx - 22]
    elif 37 <= joint_idx <= 51:
        return params["right_hand_pose"][frame, joint_idx - 37]
    return np.zeros(3)


def _add_hand_mean_pose(params: dict) -> dict:
    """Add the SMPL-X hand mean pose to the raw hand parameters.

    SMPLest-X outputs hand pose parameters as inputs to the SMPL-X model.
    Internally, SMPL-X adds the hand mean pose: actual_rot = raw_param + mean.
    The mean represents the natural ~30° curl of relaxed fingers. Without it,
    fingers in the BVH appear to bend backward from the flat T-pose rest.
    """
    try:
        import smplx as _smplx

        # Search for model files
        model_dirs = [
            Path("/mnt/f/SMPLest-X/human_models/human_model_files"),
            Path.home() / "human_model_files",
        ]
        model_dir = None
        for d in model_dirs:
            if (d / "smplx" / "SMPLX_NEUTRAL.npz").is_file():
                model_dir = d
                break

        if model_dir is None:
            return params

        model = _smplx.create(
            str(model_dir), model_type="smplx", gender="neutral",
            use_pca=False, flat_hand_mean=False, ext="npz",
        )
        lh_mean = model.left_hand_mean.detach().numpy().reshape(15, 3)
        rh_mean = model.right_hand_mean.detach().numpy().reshape(15, 3)

        params = dict(params)
        params["left_hand_pose"] = params["left_hand_pose"] + lh_mean[np.newaxis, :, :]
        params["right_hand_pose"] = params["right_hand_pose"] + rh_mean[np.newaxis, :, :]
        return params

    except Exception:
        return params


def _camera_to_world_orient(params: dict) -> dict:
    """Transform root orientation from camera space to world space (180° X flip).

    Does NOT compute translation — that happens after rotation smoothing
    so FK-based foot contact detection uses clean rotations.
    """
    params = dict(params)
    n = len(params["global_orient"])

    R_flip = Rotation.from_euler("X", 180, degrees=True)
    root_aa = params["global_orient"]
    new_root = np.zeros_like(root_aa)
    for i in range(n):
        R_cam = Rotation.from_rotvec(root_aa[i])
        R_world = R_flip * R_cam
        new_root[i] = R_world.as_rotvec()
    params["global_orient"] = new_root

    return params


def _correct_tilt(params: dict, pitch_adjust_deg: float = 0.0) -> dict:
    """Auto-correct systematic camera-angle tilt, with optional manual adjustment.

    Estimates the average forward lean of the pelvis and removes it,
    then applies any manual pitch adjustment on top.
    """
    params = dict(params)
    n = len(params["global_orient"])

    # Extract pitch angle per frame from the pelvis forward direction
    pitches = np.zeros(n)
    for i in range(n):
        R = Rotation.from_rotvec(params["global_orient"][i]).as_matrix()
        # Local Z axis rotated by root rotation = pelvis forward direction
        forward = R[:, 2]  # third column
        # Project onto YZ plane (sagittal) to isolate pitch
        yz_len = np.sqrt(forward[1] ** 2 + forward[2] ** 2)
        if yz_len > 1e-8:
            pitches[i] = np.arctan2(forward[1], forward[2])

    mean_pitch = np.mean(pitches)
    correction_deg = -np.degrees(mean_pitch) + pitch_adjust_deg
    print(f"[tilt] Detected mean pitch: {np.degrees(mean_pitch):.1f}°, "
          f"correction: {correction_deg:.1f}° (manual adjust: {pitch_adjust_deg:.1f}°)")

    R_correction = Rotation.from_euler("X", correction_deg, degrees=True)
    new_orient = np.zeros_like(params["global_orient"])
    for i in range(n):
        R_orig = Rotation.from_rotvec(params["global_orient"][i])
        R_corrected = R_correction * R_orig
        new_orient[i] = R_corrected.as_rotvec()

    params["global_orient"] = new_orient
    return params


def _remap_offsets_for_bvh(offsets: dict) -> dict:
    """Remap Spine3/Neck/Head offsets for DCC tool (Cascadeur/Mixamo) compatibility.

    SMPL-X places joints at anatomical landmarks that differ from game rigs:
      SMPL-X Neck (idx 12) = base of skull (C1 atlas)
      SMPL-X Head (idx 15) = top of skull

    Game rigs (Cascadeur/Mixamo) expect:
      Neck joint = base of neck (C7, collar height)
      Head joint = base of skull
      Head EndSite = top of skull

    In BVH, the "Neck bone" length = Head offset (Neck→Head distance), and
    the "Head bone" length = EndSite offset. So we redistribute into 3 segments:
      Spine3→Neck:  ~8.5cm  (upper spine, collar height)
      Neck→Head:    ~10cm   (visible neck in Cascadeur)
      Head→EndSite: ~14cm   (head sphere size — set in _build_hierarchy_string)

    Total Spine3→skull-top distance is preserved.
    """
    offsets = dict(offsets)  # shallow copy

    neck = np.array(offsets["Neck"])       # Spine3→Neck (~16.5cm Y)
    head = np.array(offsets["Head"])       # Neck→Head  (~16.0cm Y)
    l_collar = np.array(offsets["L_Collar"])
    r_collar = np.array(offsets["R_Collar"])

    # Total Y distance from Spine3 to top of skull
    total_y = neck[1] + head[1]  # ~0.325m

    # Segment 1: Spine3→Neck at collar height
    collar_mid_y = (l_collar[1] + r_collar[1]) / 2.0  # ~0.085m
    t_neck = collar_mid_y / neck[1] if abs(neck[1]) > 1e-6 else 0.5
    new_neck = neck * t_neck  # ~[x, 0.085, z]

    # Segment 2: Neck→Head = reasonable neck length (~10cm)
    # Scale XZ proportionally along the original Neck→Head direction
    target_neck_bone_y = 0.10  # 10cm — typical game-rig neck length
    remaining_y = total_y - collar_mid_y  # ~0.240m
    t_head = target_neck_bone_y / remaining_y if abs(remaining_y) > 1e-6 else 0.4
    # Blend the direction from original neck residual and head offset
    residual = neck * (1.0 - t_neck)
    combined = residual + head  # full remaining vector (~0.240m Y)
    new_head = combined * t_head  # ~[x, 0.10, z]

    offsets["Neck"] = new_neck.tolist()
    offsets["Head"] = new_head.tolist()

    # Stash the remaining distance for Head EndSite (used in _build_hierarchy_string)
    # EndSite Y = total - collar - neck_bone ≈ 0.325 - 0.085 - 0.10 = 0.14m
    endsite = combined * (1.0 - t_head)
    offsets["_Head_EndSite"] = endsite.tolist()

    return offsets


def _fk_chain_local(params: dict, frame: int, chain: list) -> np.ndarray:
    """Compute end-effector position in root's local frame via FK.

    Args:
        params: SMPL-X parameters (with body_pose rotations).
        frame: Frame index.
        chain: List of joint indices from root's child down to the end-effector.

    Returns:
        (3,) position of the last joint in the chain, in root local coords (meters).
    """
    pos = np.zeros(3)
    R = np.eye(3)

    for jidx in chain:
        name = JOINT_NAMES[jidx]
        offset = np.array(DEFAULT_OFFSETS.get(name, [0.0, 0.0, 0.0]))
        pos = pos + R @ offset
        rot_aa = _get_joint_rotation(params, frame, jidx)
        R = R @ Rotation.from_rotvec(rot_aa).as_matrix()

    return pos


def _propagate_root_xz(l_offset, r_offset, fulcrum_left, fulcrum_right, n,
                        initial_y):
    """Forward-propagate root XZ from fulcrum foot selections.

    For each frame, the selected fulcrum foot's world XZ is locked at the
    position it had when it first became the fulcrum. Root XZ is then:
        root_xz = locked_foot_xz - foot_offset_xz

    Returns (N, 3) root positions.
    """
    root = np.zeros((n, 3))
    root[:, 1] = initial_y

    l_locked = np.array([l_offset[0, 0], l_offset[0, 2]])
    r_locked = np.array([r_offset[0, 0], r_offset[0, 2]])
    prev_fl = fulcrum_left[0]
    prev_fr = fulcrum_right[0]

    for f in range(n):
        # Lock foot position when it first becomes fulcrum
        if fulcrum_left[f] and not prev_fl:
            l_locked = np.array([
                root[max(f - 1, 0), 0] + l_offset[f, 0],
                root[max(f - 1, 0), 2] + l_offset[f, 2],
            ])
        if fulcrum_right[f] and not prev_fr:
            r_locked = np.array([
                root[max(f - 1, 0), 0] + r_offset[f, 0],
                root[max(f - 1, 0), 2] + r_offset[f, 2],
            ])

        if fulcrum_left[f] and not fulcrum_right[f]:
            root[f, 0] = l_locked[0] - l_offset[f, 0]
            root[f, 2] = l_locked[1] - l_offset[f, 2]
        elif fulcrum_right[f] and not fulcrum_left[f]:
            root[f, 0] = r_locked[0] - r_offset[f, 0]
            root[f, 2] = r_locked[1] - r_offset[f, 2]
        elif fulcrum_left[f] and fulcrum_right[f]:
            lx = l_locked[0] - l_offset[f, 0]
            lz = l_locked[1] - l_offset[f, 2]
            rx = r_locked[0] - r_offset[f, 0]
            rz = r_locked[1] - r_offset[f, 2]
            root[f, 0] = (lx + rx) / 2
            root[f, 2] = (lz + rz) / 2
        else:
            if f > 0:
                root[f, 0] = root[f - 1, 0]
                root[f, 2] = root[f - 1, 2]

        prev_fl = fulcrum_left[f]
        prev_fr = fulcrum_right[f]

    return root


def _compute_root_from_contacts(params: dict) -> np.ndarray:
    """Derive root translation from foot-ground contact detection (2-pass).

    Pass 1: Rough root estimate using simple both-feet averaging.
    Pass 2: Compute foot world velocities from pass 1, then select the
             more STATIONARY foot as the fulcrum. This properly detects
             pivot points where weight transfers between feet.

    Returns (N, 3) root translation in world coords (meters), feet at Y=0.
    """
    n = params["num_frames"]
    pelvis_y = PELVIS_HEIGHT_CM / 100.0

    left_chain = [1, 4, 7, 10]    # L_Hip → L_Knee → L_Ankle → L_Foot
    right_chain = [2, 5, 8, 11]   # R_Hip → R_Knee → R_Ankle → R_Foot

    # FK — foot positions in root local frame
    l_local = np.array([_fk_chain_local(params, f, left_chain) for f in range(n)])
    r_local = np.array([_fk_chain_local(params, f, right_chain) for f in range(n)])

    # Apply root rotation → world-frame offsets from root
    l_offset = np.zeros((n, 3))
    r_offset = np.zeros((n, 3))
    for f in range(n):
        R = Rotation.from_rotvec(params["global_orient"][f]).as_matrix()
        l_offset[f] = R @ l_local[f]
        r_offset[f] = R @ r_local[f]

    # Contact detection (foot near ground)
    l_wy = pelvis_y + l_offset[:, 1]
    r_wy = pelvis_y + r_offset[:, 1]
    min_foot_y = min(l_wy.min(), r_wy.min())
    adjusted_pelvis_y = pelvis_y - min_foot_y
    l_wy -= min_foot_y
    r_wy -= min_foot_y

    contact_thresh = 0.03  # 3cm (tighter now that tilt correction improves FK)
    l_contact = l_wy < contact_thresh
    r_contact = r_wy < contact_thresh

    # ── PASS 1: rough root estimate (average both feet in double contact) ──
    root1 = _propagate_root_xz(l_offset, r_offset, l_contact, r_contact, n,
                                adjusted_pelvis_y)

    # ── PASS 2: velocity-based fulcrum selection ──
    # Compute foot world XZ positions from pass 1
    l_world_xz = root1[:, [0, 2]] + l_offset[:, [0, 2]]
    r_world_xz = root1[:, [0, 2]] + r_offset[:, [0, 2]]

    # Frame-to-frame XZ speed (meters/frame)
    l_speed = np.zeros(n)
    r_speed = np.zeros(n)
    l_speed[1:] = np.linalg.norm(np.diff(l_world_xz, axis=0), axis=1)
    r_speed[1:] = np.linalg.norm(np.diff(r_world_xz, axis=0), axis=1)

    # Smooth velocities to get stable fulcrum assignment
    if n >= 7:
        win_v = min(11, n - 1 if n % 2 == 0 else n)
        if win_v % 2 == 0:
            win_v -= 1
        win_v = max(win_v, 5)
        l_speed = savgol_filter(l_speed, win_v, 2)
        r_speed = savgol_filter(r_speed, win_v, 2)

    # Select fulcrum: the MORE STATIONARY foot (lower world speed)
    # Add hysteresis to prevent rapid switching
    hysteresis = 0.002  # 2mm/frame
    fulcrum_l = np.zeros(n, dtype=bool)
    fulcrum_r = np.zeros(n, dtype=bool)
    current = 'left' if l_speed[0] <= r_speed[0] else 'right'

    for f in range(n):
        if not l_contact[f] and not r_contact[f]:
            # Airborne — keep current
            if current == 'left':
                fulcrum_l[f] = True
            else:
                fulcrum_r[f] = True
        elif l_contact[f] and not r_contact[f]:
            fulcrum_l[f] = True
            current = 'left'
        elif r_contact[f] and not l_contact[f]:
            fulcrum_r[f] = True
            current = 'right'
        else:
            # Both in contact — pick more stationary, with hysteresis
            if current == 'left':
                if r_speed[f] + hysteresis < l_speed[f]:
                    current = 'right'
            else:
                if l_speed[f] + hysteresis < r_speed[f]:
                    current = 'left'
            if current == 'left':
                fulcrum_l[f] = True
            else:
                fulcrum_r[f] = True

    # Refined root from single-fulcrum selection
    root = _propagate_root_xz(l_offset, r_offset, fulcrum_l, fulcrum_r, n,
                               adjusted_pelvis_y)

    # Dynamic pelvis Y: contact foot at Y=0
    for f in range(n):
        if fulcrum_l[f]:
            root[f, 1] = -l_offset[f, 1]
        elif fulcrum_r[f]:
            root[f, 1] = -r_offset[f, 1]

    # Light smoothing on X and Z only — Y preserves ground contact
    if n >= 7:
        win = min(7, n - 1 if n % 2 == 0 else n)
        if win % 2 == 0:
            win -= 1
        win = max(win, 5)
        for ax in [0, 2]:  # X and Z only
            root[:, ax] = savgol_filter(root[:, ax], win, 2)

    # Mean-center XZ (keep Y as-is for ground contact)
    root[:, 0] -= np.mean(root[:, 0])
    root[:, 2] -= np.mean(root[:, 2])

    return root


def _smooth_rotations(params: dict, window: int = 5, keys: list | None = None) -> dict:
    """Temporal smoothing of joint rotations via quaternion-space filtering.

    SMPLest-X is a per-frame estimator with no temporal coherence. Each frame
    gets an independent rotation estimate, causing ~0.5°/frame jitter on every
    joint. Worse, the root orientation uses axis-angle near π, where the
    representation is discontinuous — a 179° CW rotation and 181° CCW rotation
    are nearly identical poses but completely different vectors, causing
    apparent 360°/frame jumps.

    Fix: convert to quaternions, enforce continuity (negate q when dot product
    with previous frame is negative), smooth each quaternion component with a
    Savitzky-Golay filter, renormalize, convert back to axis-angle.

    Args:
        params: SMPL-X parameter dict.
        window: Savitzky-Golay filter window size.
        keys: Which param keys to smooth. Default: all four rotation groups.
    """
    params = dict(params)  # shallow copy

    if keys is None:
        keys = ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose"]

    for key in keys:
        aa = params[key]
        if aa.ndim == 2:
            # (N, 3) — single joint (global_orient)
            aa_3d = aa[:, np.newaxis, :]  # (N, 1, 3)
        else:
            aa_3d = aa  # (N, J, 3)

        n_frames, n_joints, _ = aa_3d.shape
        if n_frames < window:
            continue

        smoothed = np.zeros_like(aa_3d)
        for j in range(n_joints):
            joint_aa = aa_3d[:, j, :]  # (N, 3)

            # Axis-angle → quaternion (scipy uses [x, y, z, w])
            quats = Rotation.from_rotvec(joint_aa).as_quat()  # (N, 4)

            # Enforce quaternion continuity — flip sign when dot product is negative
            for i in range(1, n_frames):
                if np.dot(quats[i], quats[i - 1]) < 0:
                    quats[i] = -quats[i]

            # Savitzky-Golay filter on each quaternion component
            poly_order = min(3, window - 2)
            for c in range(4):
                quats[:, c] = savgol_filter(quats[:, c], window, poly_order)

            # Renormalize
            norms = np.linalg.norm(quats, axis=1, keepdims=True)
            quats = quats / np.maximum(norms, 1e-8)

            # Quaternion → axis-angle
            smoothed[:, j, :] = Rotation.from_quat(quats).as_rotvec()

        if aa.ndim == 2:
            params[key] = smoothed[:, 0, :]
        else:
            params[key] = smoothed

    return params


def convert_params_to_bvh(
    params: dict,
    output_path: str,
    fps: float = 30.0,
    *,
    skip_world_grounding: bool = False,
    pitch_adjust_deg: float = 0.0,
    smooth_body: bool = True,
    smooth_hands: bool = True,
) -> str:
    """Convert SMPL-X params dict to BVH file.

    Args:
        params: SMPL-X parameter dict (from extract_smplx_params, extract_gvhmr_params,
                or merge_gvhmr_smplestx_params).
        output_path: Path for output .bvh file.
        fps: Frames per second.
        skip_world_grounding: If True, use params["transl"] directly instead of
            camera→world flip + tilt correction + foot contact root computation.
            Use for GVHMR data which is already world-grounded.
        pitch_adjust_deg: Manual pitch adjustment (only used when not skipping grounding).
        smooth_body: Whether to smooth body/global_orient rotations.
        smooth_hands: Whether to smooth hand rotations.

    Returns:
        Path to the written BVH file.
    """
    # Add SMPL-X hand mean pose — safe even with zero hands (adds natural curl)
    params = _add_hand_mean_pose(params)

    if not skip_world_grounding:
        # SMPLest-X path: camera→world transform + tilt correction + contact root
        params = _camera_to_world_orient(params)
        params = _correct_tilt(params, pitch_adjust_deg=pitch_adjust_deg)

    # Build smooth keys list based on flags
    smooth_keys = []
    if smooth_body:
        smooth_keys.extend(["global_orient", "body_pose"])
    if smooth_hands:
        smooth_keys.extend(["left_hand_pose", "right_hand_pose"])
    if smooth_keys:
        params = _smooth_rotations(params, window=5, keys=smooth_keys)

    if not skip_world_grounding:
        # Derive root translation from foot-ground contacts (uses smoothed rotations)
        params["transl"] = _compute_root_from_contacts(params)
    else:
        # GVHMR data is already world-grounded — re-center XZ to start at origin
        params = dict(params)
        transl = params["transl"].copy()
        transl[:, 0] -= transl[0, 0]  # X starts at 0
        transl[:, 2] -= transl[0, 2]  # Z starts at 0
        params["transl"] = transl

    n_frames = params["num_frames"]
    frame_time = 1.0 / fps

    # Build hierarchy — use remapped offsets for DCC-compatible proportions
    bvh_offsets = _remap_offsets_for_bvh(dict(DEFAULT_OFFSETS))
    pelvis_height_cm = _compute_ground_offset_cm()
    hierarchy = _build_hierarchy_string(0, 0, bvh_offsets, pelvis_height_cm)

    # Build motion data
    # CRITICAL: joint rotation order in MOTION must exactly match the depth-first
    # traversal order of the HIERARCHY section. SMPL-X stores body_pose in joint
    # index order (1,2,3,...,21) but the hierarchy tree order is different
    # (e.g., 1→4→7→10, then 2→5→8→11, then 3→6→9→...).

    motion_lines = []

    for frame in range(n_frames):
        values = []

        # Root: 6 channels (Xpos Ypos Zpos Zrot Xrot Yrot)
        tx, ty, tz = params["transl"][frame] * 100  # meters -> centimeters
        values.extend([tx, ty, tz])

        # All joints in hierarchy traversal order (root first, then depth-first)
        for joint_idx in TRAVERSAL_ORDER:
            rot = _get_joint_rotation(params, frame, joint_idx)
            euler = axis_angle_to_euler_zxy(rot[np.newaxis])[0]
            values.extend(euler.tolist())

        motion_lines.append(" ".join(f"{v:.4f}" for v in values))

    # Assemble BVH
    bvh_content = f"""HIERARCHY
{hierarchy}
MOTION
Frames: {n_frames}
Frame Time: {frame_time:.6f}
"""
    bvh_content += "\n".join(motion_lines) + "\n"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(bvh_content)

    return output_path


def convert_smplx_to_bvh(
    pt_path: str,
    output_path: str,
    fps: float = 30.0,
    pitch_adjust_deg: float = 0.0,
) -> str:
    """Convert SMPLest-X output to BVH file.

    Thin wrapper: extracts params then delegates to convert_params_to_bvh().

    Args:
        pt_path: Path to SMPLest-X .pt output file.
        output_path: Path for output .bvh file.
        fps: Frames per second.
        pitch_adjust_deg: Manual pitch adjustment on top of auto tilt correction.

    Returns:
        Path to the written BVH file.
    """
    params = extract_smplx_params(pt_path)
    return convert_params_to_bvh(
        params, output_path, fps=fps,
        pitch_adjust_deg=pitch_adjust_deg,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python smplx_to_bvh.py <smplestx_output.pt> [output.bvh] [fps]")
        sys.exit(1)

    pt_path = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else str(Path(pt_path).with_suffix(".bvh"))
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    result = convert_smplx_to_bvh(pt_path, output, fps=fps)
    print(f"BVH written to: {result}")
