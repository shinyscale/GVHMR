"""Anchor-driven bbox track regeneration for reprocess reruns.

This module rebuilds a dense target track from sparse user anchors by choosing
between the already-tracked people in ``all_tracks``. It is intentionally
separate from ``reprocess_person`` so both GVHMR and GEM-X reruns can consume
the same regenerated bbox stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _track_boxes(track: dict, num_frames: int) -> np.ndarray:
    boxes = track["bbx_xyxy"]
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
    boxes = np.asarray(boxes, dtype=np.float32)
    if len(boxes) >= num_frames:
        return boxes[:num_frames].copy()
    if len(boxes) == 0:
        return np.zeros((num_frames, 4), dtype=np.float32)
    pad = np.repeat(boxes[-1:], num_frames - len(boxes), axis=0)
    return np.concatenate([boxes, pad], axis=0)


def _track_mask(track: dict, num_frames: int) -> np.ndarray:
    mask = track.get("detection_mask")
    if mask is None:
        return np.ones(num_frames, dtype=bool)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    if len(mask) >= num_frames:
        return mask[:num_frames].copy()
    pad = np.zeros(num_frames - len(mask), dtype=bool)
    return np.concatenate([mask, pad], axis=0)


def _track_conf(track: dict, num_frames: int) -> np.ndarray:
    conf = track.get("detection_conf")
    if conf is None:
        return _track_mask(track, num_frames).astype(np.float32)
    if isinstance(conf, torch.Tensor):
        conf = conf.detach().cpu().numpy()
    conf = np.asarray(conf, dtype=np.float32)
    if len(conf) >= num_frames:
        return conf[:num_frames].copy()
    pad = np.zeros(num_frames - len(conf), dtype=np.float32)
    return np.concatenate([conf, pad], axis=0)


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _box_center(box: np.ndarray) -> np.ndarray:
    return np.array(
        [(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5],
        dtype=np.float32,
    )


def _box_diag(box: np.ndarray) -> float:
    w = max(1.0, float(box[2] - box[0]))
    h = max(1.0, float(box[3] - box[1]))
    return float(np.hypot(w, h))


def _continuity_cost(box_a: np.ndarray, box_b: np.ndarray) -> float:
    center_delta = np.linalg.norm(_box_center(box_a) - _box_center(box_b))
    size_a = np.array([box_a[2] - box_a[0], box_a[3] - box_a[1]], dtype=np.float32)
    size_b = np.array([box_b[2] - box_b[0], box_b[3] - box_b[1]], dtype=np.float32)
    size_delta = np.linalg.norm(size_a - size_b)
    norm = max(_box_diag(box_a), _box_diag(box_b), 1.0)
    return float(np.clip((center_delta + 0.5 * size_delta) / norm, 0.0, 3.0))


def _interpolate_anchor_prior(
    original_bboxes: np.ndarray,
    anchor_boxes: dict[int, np.ndarray],
) -> np.ndarray:
    result = np.asarray(original_bboxes, dtype=np.float32).copy()
    if not anchor_boxes:
        return result

    frames = sorted(f for f in anchor_boxes if 0 <= f < len(result))
    if not frames:
        return result

    deltas = {f: anchor_boxes[f] - result[f] for f in frames}
    first = frames[0]
    last = frames[-1]
    for f in range(0, first):
        result[f] = original_bboxes[f] + deltas[first]
    for f in frames:
        result[f] = anchor_boxes[f]
    for left, right in zip(frames, frames[1:]):
        span = max(1, right - left)
        for f in range(left + 1, right):
            t = (f - left) / span
            result[f] = original_bboxes[f] + (1.0 - t) * deltas[left] + t * deltas[right]
    for f in range(last + 1, len(result)):
        result[f] = original_bboxes[f] + deltas[last]
    return result


@dataclass
class RegeneratedTrack:
    boxes: np.ndarray
    detection_mask: np.ndarray
    detection_conf: np.ndarray
    source_indices: np.ndarray
    source_track_ids: np.ndarray
    anchor_frames: list[int]

    def as_track_dict(self, base_track: dict) -> dict:
        merged_from = list(base_track.get("merged_from", []))
        for tid in np.unique(self.source_track_ids):
            if tid < 0:
                continue
            if int(tid) not in merged_from:
                merged_from.append(int(tid))
        return {
            "track_id": int(base_track.get("track_id", 0)),
            "bbx_xyxy": torch.from_numpy(self.boxes).float(),
            "detection_mask": torch.from_numpy(self.detection_mask.astype(bool)),
            "detection_conf": torch.from_numpy(self.detection_conf.astype(np.float32)),
            "merged_from": merged_from,
            "regenerated_from_anchors": True,
            "source_track_ids": torch.from_numpy(self.source_track_ids.astype(np.int64)),
            "anchor_frames": list(self.anchor_frames),
        }


def regenerate_dense_track(
    all_tracks: list[dict],
    target_index: int,
    original_bboxes: np.ndarray,
    manual_bbox_keyframes: dict[int, np.ndarray] | None = None,
    verified_identity_keyframes: list[dict] | None = None,
) -> RegeneratedTrack:
    """Regenerate a dense bbox track from anchors and existing tracked people.

    The result differs from plain interpolation because the final boxes come
    from the best-matching tracked-person trajectory at each frame, not from
    interpolated anchor deltas. Sparse manual/verified anchors are only used as
    hard constraints and guidance for the dynamic-programming source selection.
    """
    original = np.asarray(original_bboxes, dtype=np.float32)
    num_frames = len(original)
    if num_frames == 0:
        return RegeneratedTrack(
            boxes=np.zeros((0, 4), dtype=np.float32),
            detection_mask=np.zeros((0,), dtype=bool),
            detection_conf=np.zeros((0,), dtype=np.float32),
            source_indices=np.zeros((0,), dtype=np.int32),
            source_track_ids=np.zeros((0,), dtype=np.int64),
            anchor_frames=[],
        )

    anchor_boxes: dict[int, np.ndarray] = {}
    for frame_idx, bbox in (manual_bbox_keyframes or {}).items():
        if 0 <= int(frame_idx) < num_frames:
            anchor_boxes[int(frame_idx)] = np.asarray(bbox, dtype=np.float32)
    for entry in verified_identity_keyframes or []:
        frame_idx = int(entry.get("frame", -1))
        bbox = np.asarray(entry.get("bbox"), dtype=np.float32)
        if 0 <= frame_idx < num_frames and bbox.shape == (4,):
            anchor_boxes.setdefault(frame_idx, bbox)

    source_tracks = []
    for idx, track in enumerate(all_tracks):
        boxes = _track_boxes(track, num_frames)
        mask = _track_mask(track, num_frames)
        conf = _track_conf(track, num_frames)
        source_tracks.append(
            {
                "index": idx,
                "track_id": int(track.get("track_id", idx)),
                "boxes": boxes,
                "mask": mask,
                "conf": conf,
            }
        )

    anchor_prior = _interpolate_anchor_prior(original, anchor_boxes)
    anchor_frames = sorted(anchor_boxes.keys())

    if not anchor_frames:
        chosen = source_tracks[target_index]
        return RegeneratedTrack(
            boxes=chosen["boxes"].copy(),
            detection_mask=chosen["mask"].copy(),
            detection_conf=chosen["conf"].copy(),
            source_indices=np.full(num_frames, chosen["index"], dtype=np.int32),
            source_track_ids=np.full(num_frames, chosen["track_id"], dtype=np.int64),
            anchor_frames=[],
        )

    per_source_anchor_cost = []
    for source in source_tracks:
        if not anchor_frames:
            per_source_anchor_cost.append(0.0)
            continue
        ious = [
            _bbox_iou(source["boxes"][f], anchor_boxes[f])
            for f in anchor_frames
        ]
        per_source_anchor_cost.append(1.0 - float(np.mean(ious)))

    num_sources = len(source_tracks)
    dp = np.full((num_frames, num_sources), np.inf, dtype=np.float32)
    backptr = np.full((num_frames, num_sources), -1, dtype=np.int32)

    def emission_cost(frame_idx: int, source_idx: int) -> float:
        source = source_tracks[source_idx]
        box = source["boxes"][frame_idx]
        conf = float(source["conf"][frame_idx])
        has_det = bool(source["mask"][frame_idx])

        if frame_idx in anchor_boxes:
            anchor = anchor_boxes[frame_idx]
            anchor_term = (1.0 - _bbox_iou(box, anchor)) * 6.0
        else:
            anchor_term = (1.0 - _bbox_iou(box, anchor_prior[frame_idx])) * 1.5

        det_penalty = 0.0 if has_det else 0.45
        conf_penalty = 0.25 * (1.0 - conf)
        bias_penalty = 0.35 * per_source_anchor_cost[source_idx]
        return float(anchor_term + det_penalty + conf_penalty + bias_penalty)

    for s in range(num_sources):
        dp[0, s] = emission_cost(0, s)
        if s != target_index:
            dp[0, s] += 0.05

    for f in range(1, num_frames):
        for s in range(num_sources):
            emit = emission_cost(f, s)
            curr_box = source_tracks[s]["boxes"][f]
            best_cost = np.inf
            best_prev = -1
            for prev in range(num_sources):
                prev_box = source_tracks[prev]["boxes"][f - 1]
                switch_penalty = 0.35 if prev != s else 0.0
                continuity_penalty = 0.6 * _continuity_cost(prev_box, curr_box)
                cost = dp[f - 1, prev] + emit + switch_penalty + continuity_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev
            dp[f, s] = best_cost
            backptr[f, s] = best_prev

    final_state = int(np.argmin(dp[-1]))
    source_indices = np.zeros(num_frames, dtype=np.int32)
    source_indices[-1] = final_state
    for f in range(num_frames - 1, 0, -1):
        source_indices[f - 1] = backptr[f, source_indices[f]]

    boxes = np.zeros_like(original)
    detection_mask = np.zeros(num_frames, dtype=bool)
    detection_conf = np.zeros(num_frames, dtype=np.float32)
    source_track_ids = np.full(num_frames, -1, dtype=np.int64)
    for f, source_idx in enumerate(source_indices):
        source = source_tracks[int(source_idx)]
        boxes[f] = source["boxes"][f]
        detection_mask[f] = source["mask"][f]
        detection_conf[f] = source["conf"][f]
        source_track_ids[f] = source["track_id"]

    for frame_idx, bbox in anchor_boxes.items():
        boxes[frame_idx] = bbox
        detection_mask[frame_idx] = True
        detection_conf[frame_idx] = 1.0

    return RegeneratedTrack(
        boxes=boxes,
        detection_mask=detection_mask,
        detection_conf=detection_conf,
        source_indices=source_indices,
        source_track_ids=source_track_ids,
        anchor_frames=anchor_frames,
    )
