"""HaMeR hand mesh recovery — extract MANO hand poses from video.

Integrates geopavlakos/hamer (CVPR 2024) to produce per-frame MANO
parameters for left and right hands, compatible with SMPL-X hand_pose.

Requires: HaMeR cloned at third_party/hamer/ with model weights downloaded
via fetch_demo_data.sh, and MANO_RIGHT.pkl in _DATA/data/mano/.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def _load_smplx_hand_mean():
    """Load SMPL-X hand mean pose (15x3 axis-angle per hand).

    Returns (left_mean, right_mean) numpy arrays, or (None, None) if unavailable.
    """
    try:
        import smplx as _smplx
        search_dirs = [
            Path("F:/GVHMR/GVHMR/inputs/checkpoints/body_models"),
            Path("F:/SMPLest-X/human_models/human_model_files"),
            Path("/mnt/f/SMPLest-X/human_models/human_model_files"),
            Path.home() / "human_model_files",
        ]
        for d in search_dirs:
            npz = d / "smplx" / "SMPLX_NEUTRAL.npz"
            if npz.exists():
                model = _smplx.create(
                    str(d), model_type="smplx", gender="neutral",
                    use_pca=False, flat_hand_mean=False, ext="npz",
                )
                lh = model.left_hand_mean.detach().numpy().reshape(15, 3)
                rh = model.right_hand_mean.detach().numpy().reshape(15, 3)
                return lh, rh
    except Exception:
        pass
    return None, None


# ── HaMeR Setup ──

HAMER_DIR = Path(__file__).parent / "third_party" / "hamer"


def _ensure_hamer_on_path():
    """Add HaMeR to sys.path if not already present."""
    hamer_str = str(HAMER_DIR)
    if hamer_str not in sys.path:
        sys.path.insert(0, hamer_str)


def _load_hamer_model(device: str = "cuda"):
    """Load the HaMeR model using the official load_hamer() function.

    The HaMeR codebase uses relative paths (./_DATA/...), so we temporarily
    change CWD to the HaMeR directory during loading.
    """
    _ensure_hamer_on_path()

    checkpoint = HAMER_DIR / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"HaMeR checkpoint not found at {checkpoint}. "
            f"Run: cd {HAMER_DIR} && bash fetch_demo_data.sh"
        )

    old_cwd = os.getcwd()
    os.chdir(str(HAMER_DIR))
    try:
        from hamer.models import load_hamer
        model, model_cfg = load_hamer(str(checkpoint))
    finally:
        os.chdir(old_cwd)

    model = model.to(device)
    model.eval()
    return model, model_cfg


def _preprocess_hand(
    frame_bgr: np.ndarray,
    bbox: np.ndarray,
    is_right: int,
    model_cfg,
    rescale_factor: float = 2.0,
) -> torch.Tensor:
    """Preprocess a hand crop for HaMeR, replicating ViTDetDataset logic.

    Steps: bbox → center/scale → aspect-ratio expansion → anti-alias blur →
    affine crop (with flip for left hands) → BGR→RGB → normalize.

    Returns:
        (3, IMAGE_SIZE, IMAGE_SIZE) float tensor, ImageNet-normalized.
    """
    from hamer.datasets.utils import (
        expand_to_aspect_ratio,
        generate_image_patch_cv2,
        convert_cvimg_to_tensor,
    )
    from skimage.filters import gaussian

    img_size = model_cfg.MODEL.IMAGE_SIZE  # 256
    BBOX_SHAPE = model_cfg.MODEL.get("BBOX_SHAPE", None)  # [192, 256]

    center = (bbox[2:4] + bbox[0:2]) / 2.0
    scale = rescale_factor * (bbox[2:4] - bbox[0:2]) / 200.0
    bbox_size = expand_to_aspect_ratio(
        scale * 200, target_aspect_ratio=BBOX_SHAPE
    ).max()

    flip = is_right == 0  # Left hands get mirrored so model always sees "right hand"

    cvimg = frame_bgr.copy()
    downsampling_factor = (bbox_size * 1.0) / img_size / 2.0
    if downsampling_factor > 1.1:
        cvimg = gaussian(
            cvimg,
            sigma=(downsampling_factor - 1) / 2,
            channel_axis=2,
            preserve_range=True,
        )

    img_patch, _ = generate_image_patch_cv2(
        cvimg,
        center[0],
        center[1],
        bbox_size,
        bbox_size,
        img_size,
        img_size,
        flip,
        1.0,
        0,
        border_mode=cv2.BORDER_CONSTANT,
    )

    img_patch = img_patch[:, :, ::-1]  # BGR → RGB
    img_tensor = convert_cvimg_to_tensor(img_patch)  # HWC uint8 → CHW float

    mean = 255.0 * np.array(model_cfg.MODEL.IMAGE_MEAN)
    std = 255.0 * np.array(model_cfg.MODEL.IMAGE_STD)
    for c in range(3):
        img_tensor[c] = (img_tensor[c] - mean[c]) / std[c]

    return torch.from_numpy(img_tensor.copy())


def _make_hand_bbox(
    wrist_x: float,
    wrist_y: float,
    frame_w: int,
    frame_h: int,
    elbow_x: float | None = None,
    elbow_y: float | None = None,
    hand_size_frac: float = 0.22,
) -> np.ndarray:
    """Create an [x1, y1, x2, y2] bounding box for a hand.

    Centers the box slightly past the wrist (toward the fingers) using the
    elbow→wrist direction vector.  The full wrist and spread fingers should
    always be visible in the crop — HaMeR's rescale_factor=2.0 further
    expands this during preprocessing.
    """
    hand_size = min(frame_w, frame_h) * hand_size_frac
    half = hand_size / 2

    # Offset center from wrist toward fingers (~25% of hand_size along
    # the forearm direction).  Modest offset keeps the wrist well inside
    # the crop while still biasing toward fingers.
    cx, cy = wrist_x, wrist_y
    if elbow_x is not None and elbow_y is not None:
        dx = wrist_x - elbow_x
        dy = wrist_y - elbow_y
        length = max((dx**2 + dy**2) ** 0.5, 1e-6)
        offset = hand_size * 0.25
        cx += dx / length * offset
        cy += dy / length * offset

    return np.array(
        [
            max(0, cx - half),
            max(0, cy - half),
            min(frame_w, cx + half),
            min(frame_h, cy + half),
        ],
        dtype=np.float32,
    )


def run_hamer(
    video_path: str,
    vitpose_path: str | None = None,
    person_bboxes: np.ndarray | None = None,
    device: str = "cuda",
    batch_size: int = 128,
    model=None,
    model_cfg=None,
) -> dict:
    """Run HaMeR on a video to extract MANO hand parameters.

    Uses ViTPose wrist keypoints (9=left_wrist, 10=right_wrist) to create
    hand bounding boxes, then runs HaMeR with proper preprocessing.

    Returns:
        Dict with left/right_hand_pose (F, 15, 3) and confidence (F,).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    zeros_result = {
        "left_hand_pose": np.zeros((n_frames, 15, 3)),
        "right_hand_pose": np.zeros((n_frames, 15, 3)),
        "left_confidence": np.zeros(n_frames),
        "right_confidence": np.zeros(n_frames),
    }

    # Load ViTPose for wrist keypoints
    vitpose = None
    if vitpose_path and Path(vitpose_path).is_file():
        vp = torch.load(vitpose_path, map_location="cpu", weights_only=False)
        if isinstance(vp, dict):
            for key in ["vitpose", "keypoints", "poses"]:
                if key in vp:
                    vp = vp[key]
                    break
        vitpose = np.array(vp)  # (F, 17, 3)

    # Load HaMeR model (skip if caller passed a pre-loaded model)
    _model_loaded_here = model is None
    if _model_loaded_here:
        try:
            model, model_cfg = _load_hamer_model(device)
        except Exception as e:
            print(f"[HaMeR] Failed to load model: {e}")
            cap.release()
            return zeros_result

    # COCO-17 indices
    LEFT_ELBOW_IDX = 7
    RIGHT_ELBOW_IDX = 8
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10
    WRIST_CONF_THRESH = 0.3

    # Results arrays
    left_poses = np.zeros((n_frames, 15, 3))
    right_poses = np.zeros((n_frames, 15, 3))
    left_conf = np.zeros(n_frames)
    right_conf = np.zeros(n_frames)

    # Accumulate preprocessed crops for batched inference
    crop_tensors = []  # preprocessed image tensors
    crop_meta = []  # (frame_idx, is_right)

    def _flush_batch():
        """Run model on accumulated crops and store results."""
        if not crop_tensors:
            return
        batch_img = torch.stack(crop_tensors).to(device)
        with torch.no_grad():
            output = model({"img": batch_img})

        hp_rotmat = output["pred_mano_params"]["hand_pose"].cpu().numpy()
        B, J = hp_rotmat.shape[:2]  # (B, 15, 3, 3)
        hp_aa = Rotation.from_matrix(hp_rotmat.reshape(-1, 3, 3)).as_rotvec()
        hp_aa = hp_aa.reshape(B, J, 3)

        for i, (fidx, is_r) in enumerate(crop_meta):
            aa = hp_aa[i]
            if is_r == 0:
                # Left hand was flipped — mirror Y/Z axis rotations back.
                # Horizontal flip preserves X-axis rotation (curl) but
                # negates Y (twist) and Z (spread) components.
                aa[:, 1:] *= -1
                left_poses[fidx] = aa
                left_conf[fidx] = 0.8
            else:
                right_poses[fidx] = aa
                right_conf[fidx] = 0.8

        crop_tensors.clear()
        crop_meta.clear()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n_frames:
            break

        # Determine wrist positions
        wrists = []  # [(wx, wy, is_right, elbow_x, elbow_y), ...]

        if vitpose is not None and frame_idx < len(vitpose):
            lw = vitpose[frame_idx, LEFT_WRIST_IDX]
            rw = vitpose[frame_idx, RIGHT_WRIST_IDX]
            le = vitpose[frame_idx, LEFT_ELBOW_IDX]
            re = vitpose[frame_idx, RIGHT_ELBOW_IDX]
            if lw[2] > WRIST_CONF_THRESH:
                ex, ey = (le[0], le[1]) if le[2] > 0.3 else (None, None)
                wrists.append((lw[0], lw[1], 0, ex, ey))
            if rw[2] > WRIST_CONF_THRESH:
                ex, ey = (re[0], re[1]) if re[2] > 0.3 else (None, None)
                wrists.append((rw[0], rw[1], 1, ex, ey))
        elif person_bboxes is not None and frame_idx < len(person_bboxes):
            bbox = person_bboxes[frame_idx]
            bx1, by1, bx2, by2 = bbox
            if max(bbox) <= 1.0:
                bx1, bx2 = bx1 * frame_w, bx2 * frame_w
                by1, by2 = by1 * frame_h, by2 * frame_h
            wrists.append((bx2 - (bx2 - bx1) * 0.15, by1 + (by2 - by1) * 0.70, 0, None, None))
            wrists.append((bx1 + (bx2 - bx1) * 0.15, by1 + (by2 - by1) * 0.70, 1, None, None))

        for wx, wy, is_r, ex, ey in wrists:
            bbox = _make_hand_bbox(wx, wy, frame_w, frame_h, ex, ey)
            if bbox[2] - bbox[0] < 10 or bbox[3] - bbox[1] < 10:
                continue
            try:
                tensor = _preprocess_hand(frame, bbox, is_r, model_cfg)
                crop_tensors.append(tensor)
                crop_meta.append((frame_idx, is_r))
            except Exception as e:
                print(f"[HaMeR] Preprocessing error frame {frame_idx}: {e}")

        if len(crop_tensors) >= batch_size:
            _flush_batch()

        frame_idx += 1

    cap.release()
    _flush_batch()

    # HaMeR's MANO head outputs absolute rotations (mean pose baked in via IEF).
    # Our BVH pipeline expects offsets-from-mean (SMPLest-X convention), with
    # _add_hand_mean_pose() adding the mean later.  Subtract it here so all
    # hand data enters the pipeline in the same space.
    lh_mean, rh_mean = _load_smplx_hand_mean()
    if lh_mean is not None:
        for f in range(n_frames):
            if left_conf[f] > 0:
                left_poses[f] -= lh_mean
            if right_conf[f] > 0:
                right_poses[f] -= rh_mean

    return {
        "left_hand_pose": left_poses,
        "right_hand_pose": right_poses,
        "left_confidence": left_conf,
        "right_confidence": right_conf,
    }


def merge_gvhmr_hamer_params(
    gvhmr_params: dict,
    hamer_params: dict,
    confidence_threshold: float = 0.5,
) -> dict:
    """Merge GVHMR body with HaMeR hand poses based on confidence.

    For each frame and each hand independently: if HaMeR confidence exceeds
    the threshold, use HaMeR hand pose; otherwise fall back to the existing
    hand pose from gvhmr_params (which may be zeros or SMPLest-X data).
    """
    result = dict(gvhmr_params)
    n = result["num_frames"]

    n_hamer = len(hamer_params["left_hand_pose"])
    n_use = min(n, n_hamer)

    left_hp = result["left_hand_pose"].copy()
    right_hp = result["right_hand_pose"].copy()

    left_conf = hamer_params["left_confidence"][:n_use]
    right_conf = hamer_params["right_confidence"][:n_use]

    for f in range(n_use):
        if left_conf[f] >= confidence_threshold:
            left_hp[f] = hamer_params["left_hand_pose"][f]

    for f in range(n_use):
        if right_conf[f] >= confidence_threshold:
            right_hp[f] = hamer_params["right_hand_pose"][f]

    result["left_hand_pose"] = left_hp
    result["right_hand_pose"] = right_hp
    result["hand_source"] = "hamer"

    return result
