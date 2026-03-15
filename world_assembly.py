"""World-space assembly for multi-person capture.

Recovers relative positions from YOLO bbox foot positions + camera K,
applies offsets to per-person SMPL-X translations, and exports scene-level results.
"""

import json
import numpy as np
import torch
from pathlib import Path


def _track_detection_mask(track) -> np.ndarray:
    """Return a track's real-detection mask as a numpy bool array."""
    mask = track.get("detection_mask")
    if mask is None:
        boxes = track["bbx_xyxy"]
        length = len(boxes)
        return np.ones(length, dtype=bool)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    return np.asarray(mask, dtype=bool)


def choose_reference_frame(
    all_tracks,
    preferred_frame=0,
):
    """Pick a frame with the strongest simultaneous real detections."""
    if not all_tracks:
        return 0, {"strategy": "empty"}

    masks = [_track_detection_mask(track) for track in all_tracks]
    common_len = min(len(mask) for mask in masks)
    if common_len <= 0:
        return 0, {"strategy": "empty_masks"}

    preferred_frame = max(0, min(int(preferred_frame), common_len - 1))
    stacked = np.stack([mask[:common_len] for mask in masks], axis=0)
    counts = stacked.sum(axis=0)

    if stacked[:, preferred_frame].all():
        return preferred_frame, {
            "strategy": "preferred_all_real",
            "num_real_tracks": int(counts[preferred_frame]),
        }

    best_count = int(counts.max())
    if best_count <= 0:
        return preferred_frame, {
            "strategy": "no_real_detections",
            "num_real_tracks": 0,
        }

    candidates = np.flatnonzero(counts == best_count)
    later = candidates[candidates >= preferred_frame]
    chosen = int(later[0] if len(later) else candidates[0])
    return chosen, {
        "strategy": "max_real_tracks",
        "num_real_tracks": best_count,
    }


def bbox_bottom_center(bbox_xyxy):
    """Get the bottom-center point of a bbox (approximate foot position).

    Args:
        bbox_xyxy: (4,) array [x1, y1, x2, y2]

    Returns:
        (2,) array [x, y] of bottom-center point
    """
    x1, y1, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) / 2.0, y2])


def estimate_depth_from_bbox_height(bbox_xyxy, K, assumed_height_m=1.7):
    """Estimate person depth from bbox height using camera intrinsics.

    Assumes average human height of 1.7m. Taller bbox = closer.

    Args:
        bbox_xyxy: (4,) array [x1, y1, x2, y2]
        K: (3, 3) camera intrinsic matrix
        assumed_height_m: assumed real-world height in meters

    Returns:
        Estimated depth in meters
    """
    x1, y1, x2, y2 = bbox_xyxy
    bbox_height_px = y2 - y1
    if bbox_height_px < 1:
        return 5.0  # fallback far distance
    fy = K[1, 1] if isinstance(K, np.ndarray) else float(K[1, 1])
    depth = (assumed_height_m * fy) / bbox_height_px
    return float(depth)


def compute_person_offsets(
    all_tracks,
    camera_K,
    reference_frame=0,
    assumed_height_m=1.7,
):
    """Compute XZ world-space offsets for each person relative to person 0.

    Uses bbox bottom-center (feet position) projected through camera intrinsics
    to estimate relative ground-plane positions.

    Args:
        all_tracks: list of dicts with 'track_id' and 'bbx_xyxy' (N_frames, 4) tensors
        camera_K: (3, 3) camera intrinsic matrix (numpy or tensor)
        reference_frame: frame index for computing offsets
        assumed_height_m: assumed person height for depth estimation

    Returns:
        dict mapping track_id -> (3,) offset array [dx, 0, dz]
    """
    if isinstance(camera_K, torch.Tensor):
        camera_K = camera_K.cpu().numpy()

    if len(all_tracks) == 0:
        return {}, {
            "chosen_reference_frame": None,
            "reference_info": {"strategy": "empty"},
            "per_track": {},
        }

    offsets = {}
    offset_metadata = {}
    chosen_frame, reference_info = choose_reference_frame(all_tracks, preferred_frame=reference_frame)

    ref_track = all_tracks[0]
    ref_boxes = ref_track["bbx_xyxy"]
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    ref_mask = _track_detection_mask(ref_track)

    if chosen_frame >= len(ref_boxes):
        chosen_frame = len(ref_boxes) - 1
    if not ref_mask[min(chosen_frame, len(ref_mask) - 1)] and ref_mask.any():
        chosen_frame = int(np.flatnonzero(ref_mask)[0])

    ref_bbox = ref_boxes[chosen_frame]
    ref_feet = bbox_bottom_center(ref_bbox)
    ref_depth = estimate_depth_from_bbox_height(ref_bbox, camera_K, assumed_height_m)

    fx = camera_K[0, 0]

    for track in all_tracks:
        tid = track["track_id"]
        boxes = track["bbx_xyxy"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        mask = _track_detection_mask(track)

        if len(boxes) == 0:
            offsets[tid] = np.zeros(3, dtype=np.float32)
            offset_metadata[tid] = {
                "reference_frame": None,
                "approximate": True,
                "reason": "empty_track",
            }
            continue

        track_frame = min(chosen_frame, len(boxes) - 1)
        approximate = False
        if track_frame >= len(mask):
            track_frame = len(mask) - 1
        if track_frame < 0 or not mask[track_frame]:
            real_frames = np.flatnonzero(mask)
            if len(real_frames) == 0:
                offsets[tid] = np.zeros(3, dtype=np.float32)
                offset_metadata[tid] = {
                    "reference_frame": None,
                    "approximate": True,
                    "reason": "no_real_detection",
                }
                continue
            nearest_idx = int(np.argmin(np.abs(real_frames - chosen_frame)))
            track_frame = int(real_frames[nearest_idx])
            approximate = True

        bbox = boxes[track_frame]
        feet = bbox_bottom_center(bbox)
        depth = estimate_depth_from_bbox_height(bbox, camera_K, assumed_height_m)

        # Horizontal offset: pixel displacement -> meters using depth + focal length
        dx_pixels = feet[0] - ref_feet[0]
        dx_meters = dx_pixels * depth / fx

        # Depth offset: difference in estimated depth (approximate Z offset)
        dz_meters = depth - ref_depth

        offsets[tid] = np.array([dx_meters, 0.0, dz_meters])
        offset_metadata[tid] = {
            "reference_frame": int(track_frame),
            "approximate": approximate,
        }

    return offsets, {
        "chosen_reference_frame": int(chosen_frame),
        "reference_info": reference_info,
        "per_track": offset_metadata,
    }


def apply_offsets_to_smplx(smplx_params_path, offset, output_path=None):
    """Apply a world-space offset to SMPL-X translation parameters.

    Args:
        smplx_params_path: path to .pt file with SMPL-X params
        offset: (3,) array [dx, dy, dz] to add to translations
        output_path: if provided, save modified params here

    Returns:
        Modified params dict
    """
    params = torch.load(smplx_params_path, map_location="cpu", weights_only=False)

    offset_tensor = torch.tensor(offset, dtype=torch.float32)

    # Apply offset to translation
    if "transl" in params:
        params["transl"] = params["transl"] + offset_tensor
    if "transl_world" in params:
        params["transl_world"] = params["transl_world"] + offset_tensor

    if output_path:
        torch.save(params, output_path)

    return params


def assemble_scene(
    person_dirs,
    all_tracks,
    camera_K,
    output_dir,
    reference_frame=0,
):
    """Assemble all per-person results into a shared world coordinate system.

    Args:
        person_dirs: list of Path objects for each person's output directory
        all_tracks: list of track dicts from detection
        camera_K: camera intrinsic matrix
        output_dir: directory for assembled outputs
        reference_frame: frame to use for offset computation

    Returns:
        dict with assembly results
    """
    output_dir = Path(output_dir)
    assembly_dir = output_dir / "assembly"
    assembly_dir.mkdir(parents=True, exist_ok=True)

    # Compute offsets
    offsets, offset_info = compute_person_offsets(all_tracks, camera_K, reference_frame)

    # Save offsets
    offsets_json = {
        str(tid): offset.tolist()
        for tid, offset in offsets.items()
    }
    with open(assembly_dir / "person_offsets.json", "w") as f:
        json.dump({
            "offsets": offsets_json,
            "metadata": offset_info,
        }, f, indent=2)

    # Apply offsets to each person's SMPL-X params
    assembled_params = {}
    for i, person_dir in enumerate(person_dirs):
        person_dir = Path(person_dir)
        if i >= len(all_tracks):
            break

        track = all_tracks[i]
        tid = track["track_id"]
        offset = offsets.get(tid, np.zeros(3))

        # Find the hybrid SMPL-X .pt file
        pt_files = list(person_dir.glob("*_hybrid_smplx.pt"))
        if not pt_files:
            pt_files = list(person_dir.glob("*.pt"))
        if not pt_files:
            continue

        pt_path = pt_files[0]
        output_pt = person_dir / f"{pt_path.stem}_world_offset.pt"
        params = apply_offsets_to_smplx(str(pt_path), offset, str(output_pt))
        assembled_params[tid] = params

    # Save assembled params (all people in one file)
    if assembled_params:
        torch.save(assembled_params, assembly_dir / "assembled_smplx.pt")

    return {
        "offsets": offsets,
        "assembly_dir": str(assembly_dir),
        "num_persons": len(assembled_params),
    }


def save_session_manifest(
    output_dir,
    video_path,
    num_persons,
    all_tracks,
    offsets,
    person_dirs,
    slam_path=None,
    person_bindings=None,
):
    """Save a session manifest JSON with all metadata."""
    manifest = {
        "video_path": str(video_path),
        "num_persons": num_persons,
        "slam_path": str(slam_path) if slam_path else None,
        "tracks": [
            {
                "track_id": t["track_id"],
                "num_detected_frames": int(t["detection_mask"].sum().item()),
            }
            for t in all_tracks
        ],
        "offsets": {
            str(tid): offset.tolist()
            for tid, offset in offsets.items()
        },
        "person_dirs": [str(d) for d in person_dirs],
    }
    if person_bindings is not None:
        manifest["person_bindings"] = person_bindings

    manifest_path = Path(output_dir) / "session_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return str(manifest_path)
