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
