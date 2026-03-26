"""Per-frame, per-person confidence scoring for multi-person tracking.

Combines detection confidence, keypoint visibility, bbox overlap, shape
consistency, and motion consistency into a single overall score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class TrackConfidence:
    """Per-frame confidence for a single person."""
    detection_score: float = 0.0       # YOLO confidence (0-1)
    visible_keypoints: float = 0.0     # Fraction of ViTPose joints with conf > 0.5
    bbox_overlap: float = 0.0          # Max IoU with other people's bboxes (0 = no overlap)
    shape_consistency: float = 0.0     # Beta distance from established identity (0 = match)
    motion_consistency: float = 0.0    # Deviation from predicted position (0 = on track)

    @property
    def overall(self) -> float:
        """Weighted combination. Returns 0-1 where 1 = fully confident."""
        weights = [0.15, 0.30, 0.20, 0.20, 0.15]
        raw = [
            self.detection_score,
            self.visible_keypoints,
            1.0 - self.bbox_overlap,
            1.0 - min(self.shape_consistency, 1.0),
            1.0 - min(self.motion_consistency, 1.0),
        ]
        return max(0.0, min(1.0, sum(w * v for w, v in zip(weights, raw))))

    def to_dict(self) -> dict:
        return {
            "detection_score": self.detection_score,
            "visible_keypoints": self.visible_keypoints,
            "bbox_overlap": self.bbox_overlap,
            "shape_consistency": self.shape_consistency,
            "motion_consistency": self.motion_consistency,
            "overall": self.overall,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrackConfidence:
        return cls(
            detection_score=d.get("detection_score", 0.0),
            visible_keypoints=d.get("visible_keypoints", 0.0),
            bbox_overlap=d.get("bbox_overlap", 0.0),
            shape_consistency=d.get("shape_consistency", 0.0),
            motion_consistency=d.get("motion_consistency", 0.0),
        )


# ── Confidence computation functions ──


def compute_detection_confidence(
    track: dict,
    num_frames: int,
) -> np.ndarray:
    """Extract per-frame YOLO detection confidence from a track dict.

    Returns (num_frames,) float array. 0.0 for frames without detection.
    """
    if "detection_conf" in track:
        conf = track["detection_conf"]
        if isinstance(conf, torch.Tensor):
            conf = conf.numpy().astype(np.float32)
        else:
            conf = np.array(conf, dtype=np.float32)
        # Guard against stale cache with all-zero conf despite detections
        if conf.sum() > 0:
            return conf

    # Fallback: use detection_mask as binary confidence
    mask = track["detection_mask"]
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
    return mask.astype(np.float32)


def compute_keypoint_visibility(
    vitpose: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute fraction of visible joints per frame from ViTPose output.

    Args:
        vitpose: (F, J, 3) where [:, :, 2] is per-joint confidence
        threshold: minimum joint confidence to count as visible

    Returns:
        (F,) float array — fraction of joints visible per frame
    """
    if vitpose.ndim != 3 or vitpose.shape[2] < 3:
        return np.ones(len(vitpose), dtype=np.float32)

    joint_conf = vitpose[:, :, 2]  # (F, J)
    visible = (joint_conf > threshold).sum(axis=1)  # (F,)
    total_joints = joint_conf.shape[1]
    return (visible / total_joints).astype(np.float32)


def compute_bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two xyxy bboxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_bbox_overlap(
    track_idx: int,
    all_tracks: list[dict],
    num_frames: int,
    ignored_track_ids: set[int] | None = None,
) -> np.ndarray:
    """Compute max IoU with other people's bboxes per frame.

    Returns (num_frames,) float array.
    """
    overlap = np.zeros(num_frames, dtype=np.float32)
    target_boxes = all_tracks[track_idx]["bbx_xyxy"]
    if isinstance(target_boxes, torch.Tensor):
        target_boxes = target_boxes.numpy()
    target_mask = all_tracks[track_idx].get("detection_mask")
    if isinstance(target_mask, torch.Tensor):
        target_mask = target_mask.numpy()
    if target_mask is None:
        target_mask = np.ones(min(num_frames, len(target_boxes)), dtype=bool)

    for i, other_track in enumerate(all_tracks):
        if i == track_idx:
            continue
        other_tid = int(other_track.get("track_id", i))
        if ignored_track_ids and other_tid in ignored_track_ids:
            continue
        other_boxes = other_track["bbx_xyxy"]
        if isinstance(other_boxes, torch.Tensor):
            other_boxes = other_boxes.numpy()
        other_mask = other_track.get("detection_mask")
        if isinstance(other_mask, torch.Tensor):
            other_mask = other_mask.numpy()
        if other_mask is None:
            other_mask = np.ones(min(num_frames, len(other_boxes)), dtype=bool)

        n = min(num_frames, len(target_boxes), len(other_boxes), len(target_mask), len(other_mask))
        for f in range(n):
            # Ignore synthetic/interpolated boxes when measuring crossing risk.
            if not target_mask[f] or not other_mask[f]:
                continue
            iou = compute_bbox_iou(target_boxes[f], other_boxes[f])
            overlap[f] = max(overlap[f], iou)

    return overlap


def smooth_signal(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Light temporal smoothing for noisy per-frame confidence terms."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0 or window <= 1:
        return arr
    radius = max(0, window // 2)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - radius)
        hi = min(len(arr), i + radius + 1)
        out[i] = float(arr[lo:hi].mean())
    return out


def compute_anchor_support(
    num_frames: int,
    anchor_frames: set[int] | None = None,
    radius: int = 24,
) -> np.ndarray:
    """Dense 0-1 support signal from sparse manual/verified anchor frames."""
    support = np.zeros(num_frames, dtype=np.float32)
    if not anchor_frames or num_frames <= 0:
        return support
    radius = max(1, int(radius))
    anchors = sorted(f for f in anchor_frames if 0 <= f < num_frames)
    if not anchors:
        return support
    for f in range(num_frames):
        dist = min(abs(f - anchor) for anchor in anchors)
        support[f] = max(0.0, 1.0 - float(dist) / float(radius))
    return smooth_signal(support, window=5)


def compute_motion_consistency(
    track: dict,
    num_frames: int,
) -> np.ndarray:
    """Compute motion consistency from bbox center displacement.

    Measures deviation from constant-velocity prediction.
    Returns (num_frames,) float array — 0.0 = consistent, higher = erratic.
    """
    boxes = track["bbx_xyxy"]
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()

    centers = np.stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2,
    ], axis=1)  # (F, 2)

    consistency = np.zeros(num_frames, dtype=np.float32)
    if num_frames < 3:
        return consistency

    # Velocity at each frame
    velocity = np.diff(centers, axis=0)  # (F-1, 2)

    # Predicted position = prev_center + prev_velocity
    for f in range(2, min(num_frames, len(centers))):
        predicted = centers[f - 1] + velocity[f - 2]
        actual = centers[f]
        # Normalize by bbox diagonal
        diag = math.sqrt(
            (boxes[f, 2] - boxes[f, 0]) ** 2 + (boxes[f, 3] - boxes[f, 1]) ** 2
        )
        if diag > 0:
            consistency[f] = float(np.linalg.norm(actual - predicted) / diag)

    return np.clip(consistency, 0.0, 1.0)


def compute_shape_consistency(
    per_frame_betas: np.ndarray,
    established_betas: np.ndarray | None,
    stable_reference_frames: int = 0,
) -> np.ndarray:
    """Compute per-frame shape consistency from SMPL betas.

    Args:
        per_frame_betas: (F, 10) SMPL betas per frame
        established_betas: (10,) mean betas for this person's identity

    Returns:
        (F,) float array — L2 distance normalized to ~[0, 1] range
    """
    if established_betas is None:
        return np.zeros(len(per_frame_betas), dtype=np.float32)

    distances = np.linalg.norm(per_frame_betas - established_betas[None], axis=1)
    # Typical beta distances are noisy frame-to-frame; only trust this term
    # strongly once identity has a stable reference set.
    reference_scale = min(1.0, max(0.0, stable_reference_frames / 5.0))
    return np.clip((distances / 7.5) * reference_scale, 0.0, 1.0).astype(np.float32)


def compute_all_confidences(
    track_idx: int,
    all_tracks: list[dict],
    num_frames: int,
    vitpose: np.ndarray | None = None,
    per_frame_betas: np.ndarray | None = None,
    established_betas: np.ndarray | None = None,
    overlap_signal: np.ndarray | None = None,
    verified_frames: set[int] | None = None,
    anchor_frames: set[int] | None = None,
    stable_reference_frames: int = 0,
    ignored_track_ids: set[int] | None = None,
) -> list[TrackConfidence]:
    """Compute full per-frame confidence for a tracked person.

    Returns list of TrackConfidence, one per frame.
    """
    track = all_tracks[track_idx]

    det_conf = compute_detection_confidence(track, num_frames)
    anchor_support = compute_anchor_support(num_frames, anchor_frames)
    overlap = compute_bbox_overlap(
        track_idx,
        all_tracks,
        num_frames,
        ignored_track_ids=ignored_track_ids,
    )
    if overlap_signal is not None:
        overlap_signal = np.asarray(overlap_signal, dtype=np.float32)
        if len(overlap_signal) < num_frames:
            overlap_signal = np.pad(overlap_signal, (0, num_frames - len(overlap_signal)))
        else:
            overlap_signal = overlap_signal[:num_frames]
        # Prefer finer mask-derived overlap where available.
        overlap = np.clip(overlap_signal, 0.0, 1.0)
    overlap = smooth_signal(overlap, window=7)

    motion = compute_motion_consistency(track, num_frames)
    # Legitimate crossings and mirrored choreography inflate constant-velocity
    # residuals, so damp motion penalty during heavy overlap.
    motion *= (1.0 - 0.7 * np.clip(overlap, 0.0, 1.0))
    motion *= (1.0 - 0.5 * anchor_support)
    motion = smooth_signal(motion, window=5)

    if vitpose is not None:
        kp_vis = compute_keypoint_visibility(vitpose)
        # Pad/truncate to num_frames
        if len(kp_vis) < num_frames:
            kp_vis = np.pad(kp_vis, (0, num_frames - len(kp_vis)))
        else:
            kp_vis = kp_vis[:num_frames]
    else:
        kp_vis = det_conf.copy()  # Approximate from detection conf
    # Manual/verified anchors are strong identity evidence even when the 2D
    # keypoint detector is noisy, so raise the visibility floor near anchors.
    kp_floor = np.clip(0.45 * det_conf + 0.55 * anchor_support, 0.0, 1.0)
    kp_vis = np.maximum(kp_vis, kp_floor)

    if per_frame_betas is not None:
        shape_dist = compute_shape_consistency(
            per_frame_betas,
            established_betas,
            stable_reference_frames=stable_reference_frames,
        )
        if len(shape_dist) < num_frames:
            shape_dist = np.pad(shape_dist, (0, num_frames - len(shape_dist)))
        else:
            shape_dist = shape_dist[:num_frames]
    else:
        shape_dist = np.zeros(num_frames, dtype=np.float32)
    shape_dist *= (1.0 - 0.35 * anchor_support)
    shape_dist = smooth_signal(shape_dist, window=9)

    confidences = []
    verified_frames = verified_frames or set()
    for f in range(num_frames):
        overlap_term = float(overlap[f])
        motion_term = float(motion[f])
        shape_term = float(shape_dist[f])
        kp_term = float(kp_vis[f])
        if anchor_support[f] > 0.0:
            overlap_term *= (1.0 - 0.5 * float(anchor_support[f]))
            motion_term *= (1.0 - 0.35 * float(anchor_support[f]))
            shape_term *= (1.0 - 0.25 * float(anchor_support[f]))
        if f in verified_frames:
            overlap_term *= 0.25
            motion_term *= 0.25
            shape_term *= 0.5
            kp_term = max(kp_term, 0.9)
        confidences.append(TrackConfidence(
            detection_score=float(det_conf[f]),
            visible_keypoints=kp_term,
            bbox_overlap=overlap_term,
            shape_consistency=shape_term,
            motion_consistency=motion_term,
        ))

    return confidences


def confidence_to_array(confidences: list[TrackConfidence]) -> np.ndarray:
    """Extract overall confidence as a numpy array."""
    return np.array([c.overall for c in confidences], dtype=np.float32)


def export_confidence_csv(
    confidences: list[TrackConfidence],
    person_id: int,
    output_path: str,
    bridged_frames: set[int] | None = None,
) -> None:
    """Export per-frame confidence as CSV sidecar."""
    bridged = bridged_frames or set()
    with open(output_path, "w") as f:
        f.write("frame,person_id,overall,detection,visible_kp,bbox_overlap,shape_dist,motion_dist,bridged\n")
        for i, c in enumerate(confidences):
            f.write(
                f"{i},{person_id},{c.overall:.3f},{c.detection_score:.3f},"
                f"{c.visible_keypoints:.3f},{c.bbox_overlap:.3f},"
                f"{c.shape_consistency:.3f},{c.motion_consistency:.3f},"
                f"{'true' if i in bridged else 'false'}\n"
            )
