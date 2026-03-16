"""Identity keyframes and tracks for multi-person body capture.

Ported from facepipe's Keyframe/KeyframeTrack pattern. Keyframes are
verified identity assignments at specific frames. IdentityTrack manages
an ordered collection of keyframes for a single person.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from identity_confidence import TrackConfidence, confidence_to_array


def _np_to_list(arr: np.ndarray | None) -> list | None:
    if arr is None:
        return None
    return arr.tolist()


def _list_to_np(lst: list | None, dtype=np.float32) -> np.ndarray | None:
    if lst is None:
        return None
    return np.array(lst, dtype=dtype)


@dataclass
class IdentityKeyframe:
    """A verified identity assignment at a specific frame."""
    id: str
    frame_index: int
    timestamp: float
    person_id: int
    bbox: np.ndarray                    # (4,) xyxy
    betas: np.ndarray                   # (10,) SMPL shape parameters
    body_pose: np.ndarray | None = None  # (J*3,) joint rotations
    confidence: TrackConfidence | None = None
    verified: bool = False
    thumbnail: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "person_id": self.person_id,
            "bbox": _np_to_list(self.bbox),
            "betas": _np_to_list(self.betas),
            "body_pose": _np_to_list(self.body_pose),
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "verified": self.verified,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> IdentityKeyframe:
        conf = None
        if d.get("confidence"):
            conf = TrackConfidence.from_dict(d["confidence"])
        return cls(
            id=d["id"],
            frame_index=d["frame_index"],
            timestamp=d.get("timestamp", 0.0),
            person_id=d["person_id"],
            bbox=_list_to_np(d["bbox"]),
            betas=_list_to_np(d["betas"]),
            body_pose=_list_to_np(d.get("body_pose")),
            confidence=conf,
            verified=d.get("verified", False),
            metadata=d.get("metadata", {}),
        )


@dataclass
class IdentityTrack:
    """Ordered keyframes for a single person across a video."""
    person_id: int
    performer_name: str = ""
    keyframes: list[IdentityKeyframe] = field(default_factory=list)
    established_betas: np.ndarray | None = None  # Mean betas from high-conf frames

    def add_keyframe(
        self,
        frame_index: int,
        bbox: np.ndarray,
        betas: np.ndarray,
        body_pose: np.ndarray | None = None,
        confidence: TrackConfidence | None = None,
        verified: bool = False,
        timestamp: float = 0.0,
        metadata: dict | None = None,
    ) -> IdentityKeyframe:
        """Add a keyframe at the given frame index. Replaces if one already exists."""
        # Remove existing keyframe at this frame
        self.keyframes = [kf for kf in self.keyframes if kf.frame_index != frame_index]

        kf = IdentityKeyframe(
            id=str(uuid.uuid4()),
            frame_index=frame_index,
            timestamp=timestamp,
            person_id=self.person_id,
            bbox=bbox,
            betas=betas,
            body_pose=body_pose,
            confidence=confidence,
            verified=verified,
            metadata=metadata or {},
        )
        self.keyframes.append(kf)
        self.keyframes.sort(key=lambda k: k.frame_index)
        return kf

    def remove_keyframe(self, frame_index: int) -> bool:
        """Remove keyframe at frame_index. Returns True if found."""
        before = len(self.keyframes)
        self.keyframes = [kf for kf in self.keyframes if kf.frame_index != frame_index]
        return len(self.keyframes) < before

    def get_nearest(self, frame_index: int) -> IdentityKeyframe | None:
        """Return the keyframe nearest to frame_index."""
        if not self.keyframes:
            return None
        return min(self.keyframes, key=lambda kf: abs(kf.frame_index - frame_index))

    def get_surrounding(self, frame_index: int) -> tuple[IdentityKeyframe | None, IdentityKeyframe | None]:
        """Return (previous_kf, next_kf) surrounding frame_index."""
        prev_kf = None
        next_kf = None
        for kf in self.keyframes:
            if kf.frame_index <= frame_index:
                prev_kf = kf
            if kf.frame_index >= frame_index and next_kf is None:
                next_kf = kf
        return prev_kf, next_kf

    def high_confidence_frames(self, threshold: float = 0.8) -> list[int]:
        """Return frame indices of keyframes with confidence >= threshold."""
        return [
            kf.frame_index for kf in self.keyframes
            if kf.confidence and kf.confidence.overall >= threshold
        ]

    def low_confidence_spans(
        self,
        confidences: list[TrackConfidence],
        threshold: float = 0.4,
    ) -> list[tuple[int, int]]:
        """Find contiguous spans where confidence < threshold.

        Returns list of (start_frame, end_frame) inclusive tuples.
        """
        overall = confidence_to_array(confidences)
        spans = []
        in_span = False
        start = 0

        for i, conf in enumerate(overall):
            if conf < threshold:
                if not in_span:
                    start = i
                    in_span = True
            else:
                if in_span:
                    spans.append((start, i - 1))
                    in_span = False

        if in_span:
            spans.append((start, len(overall) - 1))

        return spans

    def establish_identity(self, threshold: float = 0.8) -> np.ndarray | None:
        """Compute mean betas from high-confidence keyframes.

        Updates self.established_betas and returns it.
        """
        high_conf_kfs = [
            kf for kf in self.keyframes
            if kf.confidence and kf.confidence.overall >= threshold
            and kf.betas is not None
        ]
        if not high_conf_kfs:
            # Fall back to all keyframes with betas
            high_conf_kfs = [kf for kf in self.keyframes if kf.betas is not None]

        if not high_conf_kfs:
            return None

        all_betas = np.stack([kf.betas for kf in high_conf_kfs])
        self.established_betas = all_betas.mean(axis=0)
        return self.established_betas

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "performer_name": self.performer_name,
            "keyframes": [kf.to_dict() for kf in self.keyframes],
            "established_betas": _np_to_list(self.established_betas),
        }

    @classmethod
    def from_dict(cls, d: dict) -> IdentityTrack:
        return cls(
            person_id=d["person_id"],
            performer_name=d.get("performer_name", ""),
            keyframes=[IdentityKeyframe.from_dict(kf) for kf in d.get("keyframes", [])],
            established_betas=_list_to_np(d.get("established_betas")),
        )

    def save_json(self, path: str | Path) -> None:
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> IdentityTrack:
        with open(str(path)) as f:
            return cls.from_dict(json.load(f))


def merge_identity_tracks(
    primary: IdentityTrack,
    secondary: IdentityTrack,
    primary_bboxes: np.ndarray,
    secondary_bboxes: np.ndarray,
    primary_mask: np.ndarray,
    secondary_mask: np.ndarray,
    primary_confidences: list[TrackConfidence],
    secondary_confidences: list[TrackConfidence],
    num_frames: int,
) -> tuple[IdentityTrack, np.ndarray, np.ndarray, list[TrackConfidence]]:
    """Merge secondary track into primary.

    Frame-by-frame logic:
    - Both have real detection: keep higher-confidence bbox
    - Only one has detection: use that bbox
    - Neither (gap): linearly interpolate between last known and next known

    Returns: (merged_track, merged_bboxes, merged_mask, merged_confidences)
    """
    merged_bboxes = np.zeros((num_frames, 4), dtype=np.float32)
    merged_mask = np.zeros(num_frames, dtype=bool)
    merged_confidences: list[TrackConfidence] = []

    for f in range(num_frames):
        p_det = f < len(primary_mask) and primary_mask[f]
        s_det = f < len(secondary_mask) and secondary_mask[f]

        p_conf = primary_confidences[f].overall if f < len(primary_confidences) else 0.0
        s_conf = secondary_confidences[f].overall if f < len(secondary_confidences) else 0.0

        if p_det and s_det:
            # Both detected — keep higher confidence
            if p_conf >= s_conf:
                merged_bboxes[f] = primary_bboxes[f]
                merged_confidences.append(primary_confidences[f])
            else:
                merged_bboxes[f] = secondary_bboxes[f]
                merged_confidences.append(secondary_confidences[f])
            merged_mask[f] = True
        elif p_det:
            merged_bboxes[f] = primary_bboxes[f]
            merged_confidences.append(primary_confidences[f] if f < len(primary_confidences) else TrackConfidence())
            merged_mask[f] = True
        elif s_det:
            merged_bboxes[f] = secondary_bboxes[f]
            merged_confidences.append(secondary_confidences[f] if f < len(secondary_confidences) else TrackConfidence())
            merged_mask[f] = True
        else:
            merged_confidences.append(TrackConfidence())

    # Fill gaps via linear interpolation between detected frames
    detected_frames = np.where(merged_mask)[0]
    if len(detected_frames) >= 2:
        for i in range(len(detected_frames) - 1):
            f_a = detected_frames[i]
            f_b = detected_frames[i + 1]
            if f_b - f_a <= 1:
                continue
            for f in range(f_a + 1, f_b):
                t = (f - f_a) / (f_b - f_a)
                merged_bboxes[f] = (1 - t) * merged_bboxes[f_a] + t * merged_bboxes[f_b]
                merged_mask[f] = True

    # Build merged track — combine keyframes from both tracks
    merged_track = IdentityTrack(
        person_id=primary.person_id,
        performer_name=primary.performer_name or secondary.performer_name,
    )
    all_kfs = list(primary.keyframes) + list(secondary.keyframes)
    seen_frames: set[int] = set()
    for kf in sorted(all_kfs, key=lambda k: k.frame_index):
        if kf.frame_index not in seen_frames:
            merged_track.keyframes.append(IdentityKeyframe(
                id=str(uuid.uuid4()),
                frame_index=kf.frame_index,
                timestamp=kf.timestamp,
                person_id=primary.person_id,
                bbox=merged_bboxes[kf.frame_index] if kf.frame_index < num_frames else kf.bbox,
                betas=kf.betas,
                body_pose=kf.body_pose,
                confidence=kf.confidence,
                verified=kf.verified,
                metadata=kf.metadata,
            ))
            seen_frames.add(kf.frame_index)

    merged_track.establish_identity()
    return merged_track, merged_bboxes, merged_mask, merged_confidences


def split_identity_track(
    track: IdentityTrack,
    split_frame: int,
    new_person_id: int,
) -> tuple[IdentityTrack, IdentityTrack]:
    """Split track at split_frame. Before-track keeps original person_id.
    After-track gets new_person_id. Both re-establish betas.

    Returns (track_before, track_after).
    """
    track_before = IdentityTrack(
        person_id=track.person_id,
        performer_name=track.performer_name,
    )
    track_after = IdentityTrack(
        person_id=new_person_id,
        performer_name="",
    )

    for kf in track.keyframes:
        new_kf = IdentityKeyframe(
            id=str(uuid.uuid4()),
            frame_index=kf.frame_index,
            timestamp=kf.timestamp,
            person_id=track.person_id if kf.frame_index < split_frame else new_person_id,
            bbox=kf.bbox,
            betas=kf.betas,
            body_pose=kf.body_pose,
            confidence=kf.confidence,
            verified=kf.verified,
            metadata=kf.metadata,
        )
        if kf.frame_index < split_frame:
            track_before.keyframes.append(new_kf)
        else:
            track_after.keyframes.append(new_kf)

    track_before.establish_identity()
    track_after.establish_identity()
    return track_before, track_after


def auto_generate_keyframes(
    track: IdentityTrack,
    confidences: list[TrackConfidence],
    per_frame_betas: np.ndarray,
    per_frame_bboxes: np.ndarray,
    per_frame_poses: np.ndarray | None = None,
    min_spacing: int = 30,
    threshold: float = 0.7,
    fps: float = 30.0,
) -> list[IdentityKeyframe]:
    """Auto-generate keyframes at confidence peaks.

    Finds local maxima of overall confidence, spaced at least min_spacing
    frames apart, above threshold. All auto-generated keyframes are
    marked verified=False.

    Returns the list of newly created keyframes.
    """
    overall = confidence_to_array(confidences)
    num_frames = len(overall)

    # Find local maxima above threshold
    candidates = []
    for i in range(1, num_frames - 1):
        if overall[i] >= threshold and overall[i] >= overall[i - 1] and overall[i] >= overall[i + 1]:
            candidates.append((i, overall[i]))

    # Also consider first and last frames if above threshold
    if num_frames > 0 and overall[0] >= threshold:
        candidates.insert(0, (0, overall[0]))
    if num_frames > 1 and overall[-1] >= threshold:
        candidates.append((num_frames - 1, overall[-1]))

    # Sort by confidence descending, then greedily select with min_spacing
    candidates.sort(key=lambda x: x[1], reverse=True)

    selected_frames = []
    for frame_idx, conf in candidates:
        if all(abs(frame_idx - s) >= min_spacing for s in selected_frames):
            selected_frames.append(frame_idx)

    selected_frames.sort()

    # Create keyframes
    new_keyframes = []
    for frame_idx in selected_frames:
        pose = per_frame_poses[frame_idx] if per_frame_poses is not None else None
        kf = track.add_keyframe(
            frame_index=frame_idx,
            bbox=per_frame_bboxes[frame_idx],
            betas=per_frame_betas[frame_idx],
            body_pose=pose,
            confidence=confidences[frame_idx],
            verified=False,
            timestamp=frame_idx / fps,
        )
        new_keyframes.append(kf)

    return new_keyframes
