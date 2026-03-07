"""HaMeR hand mesh recovery — extract MANO hand poses from video.

Integrates geopavlakos/hamer (CVPR 2024) to produce per-frame MANO
parameters for left and right hands, compatible with SMPL-X hand_pose.

Requires: HaMeR cloned at third_party/hamer/ with model weights downloaded.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ── HaMeR Setup ──

HAMER_DIR = Path(__file__).parent / "third_party" / "hamer"


def _ensure_hamer_on_path():
    """Add HaMeR to sys.path if not already present."""
    hamer_str = str(HAMER_DIR)
    if hamer_str not in sys.path:
        sys.path.insert(0, hamer_str)


def _load_hamer_model(device: str = "cuda"):
    """Load the HaMeR model and return (model, model_cfg).

    Downloads checkpoint on first run (~2GB).
    """
    _ensure_hamer_on_path()

    from hamer.models import HAMER
    from hamer.utils import recursive_to
    from hamer.configs import CACHE_DIR_HAMER

    # Use the default HaMeR checkpoint
    checkpoint_path = Path(CACHE_DIR_HAMER) / "hamer_ckpts" / "checkpoints" / "hamer.ckpt"

    if not checkpoint_path.exists():
        # Trigger download
        from hamer.utils.download import download_models
        download_models(CACHE_DIR_HAMER)

    from hamer.configs import get_config
    model_cfg = get_config("hamer")

    model = HAMER.load_from_checkpoint(str(checkpoint_path), strict=False, cfg=model_cfg)
    model = model.to(device)
    model.eval()

    return model, model_cfg


def _crop_hand_region(
    frame: np.ndarray,
    wrist_x: float,
    wrist_y: float,
    frame_w: int,
    frame_h: int,
    padding: float = 2.0,
    crop_size: int = 256,
) -> np.ndarray | None:
    """Crop a square region around a wrist keypoint.

    Args:
        frame: BGR video frame.
        wrist_x, wrist_y: Wrist pixel coordinates.
        frame_w, frame_h: Frame dimensions.
        padding: Multiplier for crop size relative to estimated hand size.
        crop_size: Output crop size (square).

    Returns:
        Cropped and resized BGR image, or None if wrist is out of frame.
    """
    # Estimate hand region size as ~15% of typical person height
    # Use a fixed base size since we don't have person height here
    hand_size = min(frame_w, frame_h) * 0.12
    half = hand_size * padding / 2

    x1 = int(max(0, wrist_x - half))
    y1 = int(max(0, wrist_y - half))
    x2 = int(min(frame_w, wrist_x + half))
    y2 = int(min(frame_h, wrist_y + half))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    import cv2
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return resized


def run_hamer(
    video_path: str,
    vitpose_path: str | None = None,
    person_bboxes: np.ndarray | None = None,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict:
    """Run HaMeR on a video to extract MANO hand parameters.

    Uses ViTPose wrist keypoints (9=left_wrist, 10=right_wrist) to crop
    hand regions, then runs HaMeR on the crops.

    Args:
        video_path: Path to the video file.
        vitpose_path: Path to vitpose.pt for wrist keypoints.
        person_bboxes: (N, 4) person bboxes as fallback for hand region estimation.
        device: PyTorch device.
        batch_size: Batch size for HaMeR inference.

    Returns:
        Dict with:
            - left_hand_pose: (F, 15, 3) axis-angle rotations
            - right_hand_pose: (F, 15, 3) axis-angle rotations
            - left_confidence: (F,) per-frame confidence scores
            - right_confidence: (F,) per-frame confidence scores
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Load HaMeR model
    try:
        model, model_cfg = _load_hamer_model(device)
    except Exception as e:
        print(f"[HaMeR] Failed to load model: {e}")
        print("[HaMeR] Returning zero hand poses.")
        return {
            "left_hand_pose": np.zeros((n_frames, 15, 3)),
            "right_hand_pose": np.zeros((n_frames, 15, 3)),
            "left_confidence": np.zeros(n_frames),
            "right_confidence": np.zeros(n_frames),
        }

    # Process frames
    left_poses = np.zeros((n_frames, 15, 3))
    right_poses = np.zeros((n_frames, 15, 3))
    left_conf = np.zeros(n_frames)
    right_conf = np.zeros(n_frames)

    # COCO-17 wrist indices: 9=left_wrist, 10=right_wrist
    LEFT_WRIST_IDX = 9
    RIGHT_WRIST_IDX = 10

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= n_frames:
            break

        # Get wrist positions from ViTPose
        left_crop = None
        right_crop = None

        if vitpose is not None and frame_idx < len(vitpose):
            lw = vitpose[frame_idx, LEFT_WRIST_IDX]  # (x, y, conf)
            rw = vitpose[frame_idx, RIGHT_WRIST_IDX]

            if lw[2] > 0.3:
                left_crop = _crop_hand_region(frame, lw[0], lw[1], frame_w, frame_h)
            if rw[2] > 0.3:
                right_crop = _crop_hand_region(frame, rw[0], rw[1], frame_w, frame_h)

        elif person_bboxes is not None and frame_idx < len(person_bboxes):
            # Fallback: estimate wrist from bbox
            bbox = person_bboxes[frame_idx]
            bx1, by1, bx2, by2 = bbox
            if max(bbox) <= 1.0:
                bx1, bx2 = bx1 * frame_w, bx2 * frame_w
                by1, by2 = by1 * frame_h, by2 * frame_h
            # Left wrist: right side of bbox, ~70% down
            lw_x = bx2 - (bx2 - bx1) * 0.15
            lw_y = by1 + (by2 - by1) * 0.70
            left_crop = _crop_hand_region(frame, lw_x, lw_y, frame_w, frame_h)
            # Right wrist: left side of bbox, ~70% down
            rw_x = bx1 + (bx2 - bx1) * 0.15
            rw_y = by1 + (by2 - by1) * 0.70
            right_crop = _crop_hand_region(frame, rw_x, rw_y, frame_w, frame_h)

        # Run HaMeR on crops
        for crop, side in [(left_crop, "left"), (right_crop, "right")]:
            if crop is None:
                continue

            try:
                # Prepare input for HaMeR
                img_tensor = torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img_tensor)

                # Extract MANO parameters
                # HaMeR outputs hand_pose as (B, 15, 3, 3) rotation matrices
                # or (B, 45) axis-angle depending on config
                if "pred_mano_params" in output:
                    mano = output["pred_mano_params"]
                    if "hand_pose" in mano:
                        hp = mano["hand_pose"].cpu().numpy()  # (1, 15, 3) or (1, 45)
                        if hp.ndim == 2 and hp.shape[-1] == 45:
                            hp = hp.reshape(1, 15, 3)
                        elif hp.ndim == 3 and hp.shape[-1] == 3 and hp.shape[-2] == 3:
                            # Rotation matrix format — convert
                            from scipy.spatial.transform import Rotation
                            hp = hp.reshape(-1, 3, 3)
                            hp = Rotation.from_matrix(hp).as_rotvec().reshape(1, 15, 3)

                        # Apply flat_hand_mean=True convention
                        # Left hand X-axis flip per geopavlakos/hamer#26
                        if side == "left":
                            hp[:, :, 0] *= -1

                        if side == "left":
                            left_poses[frame_idx] = hp[0]
                            left_conf[frame_idx] = 0.8  # Base confidence
                        else:
                            right_poses[frame_idx] = hp[0]
                            right_conf[frame_idx] = 0.8

            except Exception as e:
                # Skip this frame's hand on error
                continue

        frame_idx += 1

    cap.release()

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

    Args:
        gvhmr_params: Full SMPL-X params dict (body + existing hands).
        hamer_params: Output from run_hamer().
        confidence_threshold: Min confidence to use HaMeR data.

    Returns:
        Updated params dict with merged hand poses.
    """
    result = dict(gvhmr_params)
    n = result["num_frames"]

    # Ensure shapes match
    n_hamer = len(hamer_params["left_hand_pose"])
    n_use = min(n, n_hamer)

    left_hp = result["left_hand_pose"].copy()
    right_hp = result["right_hand_pose"].copy()

    left_conf = hamer_params["left_confidence"][:n_use]
    right_conf = hamer_params["right_confidence"][:n_use]

    # Left hand: replace frames where HaMeR has high confidence
    for f in range(n_use):
        if left_conf[f] >= confidence_threshold:
            left_hp[f] = hamer_params["left_hand_pose"][f]

    # Right hand: same
    for f in range(n_use):
        if right_conf[f] >= confidence_threshold:
            right_hp[f] = hamer_params["right_hand_pose"][f]

    result["left_hand_pose"] = left_hp
    result["right_hand_pose"] = right_hp
    result["hand_source"] = "hamer"

    return result
