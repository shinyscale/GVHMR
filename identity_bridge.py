"""Occlusion bridging with cosine-falloff interpolation.

Ported from facepipe's InterpolationEngine. Interpolates body pose
through low-confidence spans using SLERP for rotations and linear
interpolation for translations.

Key adaptation: facepipe interpolates 52D blendshape weights (linear).
Body tracking interpolates axis-angle rotations (requires SLERP) and
3D translations (linear).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from identity_confidence import TrackConfidence, confidence_to_array
from identity_tracking import IdentityTrack


def crossing_spans_from_signal(
    overlap: np.ndarray,
    threshold: float = 0.15,
    dilate: int = 5,
    merge_gap: int = 5,
) -> list[tuple[int, int]]:
    """Detect crossing windows from an overlap-like per-frame signal."""
    if overlap.size == 0:
        return []

    overlap = np.asarray(overlap, dtype=np.float32)
    N = len(overlap)

    # Find contiguous spans where overlap >= threshold
    above = overlap >= threshold
    if not above.any():
        return []

    spans = []
    in_span = False
    start = 0
    for i in range(N):
        if above[i] and not in_span:
            start = i
            in_span = True
        elif not above[i] and in_span:
            spans.append((start, i - 1))
            in_span = False
    if in_span:
        spans.append((start, N - 1))

    # Dilate each span
    spans = [(max(0, s - dilate), min(N - 1, e + dilate)) for s, e in spans]

    # Merge spans separated by <= merge_gap
    if len(spans) <= 1:
        return spans

    merged = [spans[0]]
    for s, e in spans[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e + merge_gap + 1:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))

    return merged


def crossing_spans_from_overlap(
    confidences: list[TrackConfidence],
    iou_threshold: float = 0.15,
    dilate: int = 5,
    merge_gap: int = 5,
) -> list[tuple[int, int]]:
    """Detect crossing windows from bbox overlap IoU.

    Uses raw bbox_overlap from TrackConfidence instead of the weighted overall
    score, which dilutes overlap signal and fragments crossings into tiny spans.
    """
    if not confidences:
        return []

    overlap = np.array([c.bbox_overlap for c in confidences], dtype=np.float32)
    return crossing_spans_from_signal(
        overlap,
        threshold=iou_threshold,
        dilate=dilate,
        merge_gap=merge_gap,
    )


class OcclusionBridge:
    """Interpolates body pose through low-confidence spans.

    Uses cosine-falloff weighting from surrounding keyframes,
    with SLERP for rotations and linear interpolation for translations.
    """

    def __init__(
        self,
        track: IdentityTrack,
        confidence_threshold: float = 0.4,
        influence_radius: int = 15,
    ):
        self.track = track
        self.confidence_threshold = confidence_threshold
        self.influence_radius = influence_radius

    def _alpha(self, distance: float | np.ndarray) -> float | np.ndarray:
        """Cosine falloff: 1.0 at center, 0.0 at radius edge."""
        r = self.influence_radius
        t = np.clip(distance / r, 0.0, 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * t))

    def _crossfade_boundaries(self, original, output, start, end):
        """Blend bridged data with original at span boundaries using cosine falloff.

        For 3D arrays (N,J,3 rotations): SLERP per joint.
        For 2D arrays (N,3 translations): linear blend.
        Modifies output in-place.
        """
        span_len = end - start + 1
        fade = min(self.influence_radius, span_len // 2)
        if fade <= 0:
            return

        use_slerp = original.ndim == 3

        for boundary, direction in [(start, 1), (end, -1)]:
            for i in range(fade):
                f = boundary + i * direction
                # Scale so full cosine curve fits within fade frames
                d = (fade - i) / fade * self.influence_radius
                blend = self._alpha(d)  # 0 at boundary, ~1 at interior

                if use_slerp:
                    for j in range(original.shape[1]):
                        r_orig = Rotation.from_rotvec(original[f, j])
                        r_brdg = Rotation.from_rotvec(output[f, j])
                        slerp_fn = Slerp([0.0, 1.0],
                                         Rotation.concatenate([r_orig, r_brdg]))
                        output[f, j] = slerp_fn([blend]).as_rotvec()[0]
                else:
                    output[f] = blend * output[f] + (1.0 - blend) * original[f]

    def bridge_translations(
        self,
        translations: np.ndarray,
        confidences: list[TrackConfidence],
    ) -> tuple[np.ndarray, set[int]]:
        """Bridge low-confidence translation spans with linear interpolation.

        Args:
            translations: (N, 3) per-frame translations
            confidences: per-frame confidence scores

        Returns:
            (bridged_translations, bridged_frame_indices)
        """
        overall = confidence_to_array(confidences)
        num_frames = len(translations)
        output = translations.copy()
        bridged_frames = set()

        spans = self.track.low_confidence_spans(confidences, self.confidence_threshold)

        for start, end in spans:
            # Find boundary keyframes
            prev_kf, next_kf = self.track.get_surrounding(start)

            if prev_kf is None and next_kf is None:
                continue

            prev_frame = prev_kf.frame_index if prev_kf and prev_kf.frame_index < start else max(0, start - 1)
            next_frame = next_kf.frame_index if next_kf and next_kf.frame_index > end else min(num_frames - 1, end + 1)

            for f in range(start, end + 1):
                if next_frame > prev_frame:
                    t = (f - prev_frame) / (next_frame - prev_frame)
                else:
                    t = 0.0

                # Linear interpolation of translation
                output[f] = (1 - t) * translations[prev_frame] + t * translations[next_frame]
                bridged_frames.add(f)

            self._crossfade_boundaries(translations, output, start, end)

        return output, bridged_frames

    def bridge_rotations(
        self,
        rotations: np.ndarray,
        confidences: list[TrackConfidence],
    ) -> tuple[np.ndarray, set[int]]:
        """Bridge low-confidence rotation spans with SLERP.

        Args:
            rotations: (N, J, 3) axis-angle rotations per joint per frame
            confidences: per-frame confidence scores

        Returns:
            (bridged_rotations, bridged_frame_indices)
        """
        overall = confidence_to_array(confidences)
        num_frames, num_joints = rotations.shape[0], rotations.shape[1]
        output = rotations.copy()
        bridged_frames = set()

        spans = self.track.low_confidence_spans(confidences, self.confidence_threshold)

        for start, end in spans:
            prev_kf, next_kf = self.track.get_surrounding(start)

            if prev_kf is None and next_kf is None:
                continue

            prev_frame = prev_kf.frame_index if prev_kf and prev_kf.frame_index < start else max(0, start - 1)
            next_frame = next_kf.frame_index if next_kf and next_kf.frame_index > end else min(num_frames - 1, end + 1)

            span_len = end - start + 1

            # Per-joint SLERP
            for j in range(num_joints):
                aa_prev = rotations[prev_frame, j]
                aa_next = rotations[next_frame, j]

                r_prev = Rotation.from_rotvec(aa_prev)
                r_next = Rotation.from_rotvec(aa_next)

                # Create Slerp interpolator
                key_times = [0.0, 1.0]
                key_rots = Rotation.concatenate([r_prev, r_next])
                slerp = Slerp(key_times, key_rots)

                for f in range(start, end + 1):
                    if next_frame > prev_frame:
                        t = (f - prev_frame) / (next_frame - prev_frame)
                    else:
                        t = 0.0
                    t = max(0.0, min(1.0, t))

                    interp_rot = slerp([t])
                    output[f, j] = interp_rot.as_rotvec()[0]
                    bridged_frames.add(f)

            self._crossfade_boundaries(rotations, output, start, end)

        return output, bridged_frames

    def bridge_poses(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        confidences: list[TrackConfidence],
    ) -> dict:
        """Bridge all pose components through low-confidence spans.

        Args:
            body_pose: (N, 21, 3) or (N, 63) axis-angle body joint rotations
            global_orient: (N, 3) axis-angle root orientation
            transl: (N, 3) root translation
            confidences: per-frame confidence scores

        Returns:
            dict with 'body_pose', 'global_orient', 'transl', 'bridged_frames'
        """
        num_frames = len(body_pose)

        # Reshape body_pose to (N, 21, 3) if flat
        if body_pose.ndim == 2 and body_pose.shape[1] == 63:
            body_pose = body_pose.reshape(num_frames, 21, 3)
        elif body_pose.ndim == 2:
            num_joints = body_pose.shape[1] // 3
            body_pose = body_pose.reshape(num_frames, num_joints, 3)

        # Reshape global_orient to (N, 1, 3)
        global_orient_3d = global_orient.reshape(num_frames, 1, 3)

        # Bridge rotations (body + global orient together)
        all_rots = np.concatenate([global_orient_3d, body_pose], axis=1)  # (N, 22, 3)
        bridged_rots, bridged_frames = self.bridge_rotations(all_rots, confidences)

        # Bridge translations
        bridged_transl, transl_bridged = self.bridge_translations(transl, confidences)
        bridged_frames.update(transl_bridged)

        return {
            "body_pose": bridged_rots[:, 1:].reshape(num_frames, -1, 3),
            "global_orient": bridged_rots[:, 0],
            "transl": bridged_transl,
            "bridged_frames": bridged_frames,
        }

    def bridge_with_spans(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        spans: list[tuple[int, int]],
        overall_confidence: np.ndarray | None = None,
        min_confidence: float = 0.85,
    ) -> dict:
        """Bridge explicit crossing spans using direct frame anchors.

        Unlike bridge_poses() which computes spans from confidence scores and
        looks up keyframes, this takes pre-computed spans (from IoU overlap)
        and uses the immediately adjacent clean frames as SLERP anchors.

        Args:
            body_pose: (N, 21, 3) or (N, 63) axis-angle body joint rotations
            global_orient: (N, 3) axis-angle root orientation
            transl: (N, 3) root translation
            spans: list of (start, end) inclusive crossing spans
            overall_confidence: per-frame confidence scores; if provided,
                spans where min confidence >= min_confidence are skipped
            min_confidence: threshold above which GVHMR output is trusted

        Returns:
            dict with 'body_pose', 'global_orient', 'transl', 'bridged_frames'
        """
        num_frames = len(body_pose)

        # Reshape body_pose to (N, 21, 3) if flat
        if body_pose.ndim == 2 and body_pose.shape[1] == 63:
            body_pose = body_pose.reshape(num_frames, 21, 3)
        elif body_pose.ndim == 2:
            num_joints = body_pose.shape[1] // 3
            body_pose = body_pose.reshape(num_frames, num_joints, 3)

        # Concat global orient + body pose for joint SLERP
        global_orient_3d = global_orient.reshape(num_frames, 1, 3)
        all_rots = np.concatenate([global_orient_3d, body_pose], axis=1)  # (N, 22, 3)
        rot_output = all_rots.copy()
        transl_output = transl.copy()
        bridged_frames = set()

        for start, end in spans:
            prev_frame = max(0, start - 1)
            next_frame = min(num_frames - 1, end + 1)

            # Guard: if span covers entire clip, no clean anchors exist
            if prev_frame == start and next_frame == end:
                continue

            # Tail-span guard: no clean exit anchor when span reaches end of sequence
            if next_frame <= end:
                print(f"[OcclusionBridge] Skip tail span ({start}-{end}): no clean exit anchor")
                continue

            # Confidence gate: skip spans where GVHMR output is reliable
            if overall_confidence is not None:
                span_min = overall_confidence[start:end + 1].min()
                if span_min >= min_confidence:
                    print(f"[OcclusionBridge] Skip high-conf span ({start}-{end}): "
                          f"min {span_min:.3f} >= {min_confidence}")
                    continue

            num_joints = all_rots.shape[1]

            # Per-joint SLERP for rotations
            for j in range(num_joints):
                r_prev = Rotation.from_rotvec(all_rots[prev_frame, j])
                r_next = Rotation.from_rotvec(all_rots[next_frame, j])
                slerp = Slerp([0.0, 1.0], Rotation.concatenate([r_prev, r_next]))

                for f in range(start, end + 1):
                    if next_frame > prev_frame:
                        t = (f - prev_frame) / (next_frame - prev_frame)
                    else:
                        t = 0.0
                    t = max(0.0, min(1.0, t))
                    rot_output[f, j] = slerp([t]).as_rotvec()[0]
                    bridged_frames.add(f)

            # Linear interpolation for translations
            for f in range(start, end + 1):
                if next_frame > prev_frame:
                    t = (f - prev_frame) / (next_frame - prev_frame)
                else:
                    t = 0.0
                transl_output[f] = (1 - t) * transl[prev_frame] + t * transl[next_frame]

            # No crossfade for crossing spans — unlike confidence-based spans where
            # boundary frames are "somewhat OK", crossing span boundaries contain
            # corrupted HMR output. The SLERP from prev_frame→next_frame already
            # provides smooth transitions (t≈0 at start, t≈1 at end).

        return {
            "body_pose": rot_output[:, 1:].reshape(num_frames, -1, 3),
            "global_orient": rot_output[:, 0],
            "transl": transl_output,
            "bridged_frames": bridged_frames,
        }

    def bake(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        confidences: list[TrackConfidence],
    ) -> dict:
        """Vectorized bake of all frames. Alias for bridge_poses.

        Returns dict with bridged pose components and set of bridged frame indices.
        """
        return self.bridge_poses(body_pose, global_orient, transl, confidences)


def classify_occlusion(
    span_start: int,
    span_end: int,
    fps: float = 30.0,
) -> str:
    """Classify an occlusion span by duration.

    Returns one of: 'brief', 'partial', 'full', 'extended'
    """
    duration_frames = span_end - span_start + 1
    duration_seconds = duration_frames / fps

    if duration_seconds < 0.5:
        return "brief"
    elif duration_seconds < 1.5:
        return "partial"
    elif duration_seconds < 3.0:
        return "full"
    else:
        return "extended"


def summarize_bridging(
    confidences: list[TrackConfidence],
    bridged_frames: set[int],
    fps: float = 30.0,
) -> dict:
    """Generate a summary of bridging results."""
    overall = confidence_to_array(confidences)
    num_frames = len(overall)

    # Count frames by confidence tier
    high = int((overall >= 0.8).sum())
    medium = int(((overall >= 0.4) & (overall < 0.8)).sum())
    low = int((overall < 0.4).sum())

    return {
        "total_frames": num_frames,
        "high_confidence_frames": high,
        "medium_confidence_frames": medium,
        "low_confidence_frames": low,
        "bridged_frames": len(bridged_frames),
        "bridged_duration_seconds": len(bridged_frames) / fps,
        "confidence_mean": float(overall.mean()),
    }
