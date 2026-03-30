"""Shape-based re-identification using SMPL betas.

Matches detected people to known identities via L2 distance in the
10-dimensional SMPL beta (body shape) space. Detects identity swaps
by checking if current assignments better match swapped tracks.
"""

from __future__ import annotations

import numpy as np

from identity_tracking import IdentityTrack


class ShapeReIdentifier:
    """Matches detected people to known identities via body shape."""

    def __init__(self, tracks: list[IdentityTrack]):
        self.known_betas: dict[int, np.ndarray] = {}
        for track in tracks:
            if track.established_betas is not None:
                self.known_betas[track.person_id] = track.established_betas

    @staticmethod
    def _confidence_threshold(dim: int) -> float:
        """Scale the L2 distance→confidence denominator for shape vector dimensionality.

        SMPL betas are 10-dim (threshold ~5.0); GEM-X identity_coeffs are
        45-dim (higher L2 norms expected). Scale by sqrt(dim/10).
        """
        return 5.0 * (dim / 10) ** 0.5

    def match(self, detected_betas: np.ndarray) -> tuple[int, float]:
        """Match detected betas to known identity.

        Returns (person_id, confidence) where confidence is 0-1.
        Works with any shape vector dimensionality (10 for SMPL, 45 for GEM-X).
        """
        if not self.known_betas:
            return -1, 0.0

        distances = {
            pid: float(np.linalg.norm(detected_betas - known))
            for pid, known in self.known_betas.items()
        }
        best_id = min(distances, key=distances.get)
        threshold = self._confidence_threshold(len(detected_betas.ravel()))
        confidence = max(0.0, 1.0 - distances[best_id] / threshold)
        return best_id, confidence

    def match_all(
        self,
        detected_betas_per_person: dict[int, np.ndarray],
    ) -> dict[int, tuple[int, float]]:
        """Match multiple detected persons to known identities.

        Uses Hungarian-style greedy assignment to avoid duplicate matches.

        Returns {detected_track_id: (matched_person_id, confidence)}
        """
        if not self.known_betas or not detected_betas_per_person:
            return {}

        # Build cost matrix
        det_ids = list(detected_betas_per_person.keys())
        known_ids = list(self.known_betas.keys())

        costs = np.zeros((len(det_ids), len(known_ids)), dtype=np.float32)
        for i, did in enumerate(det_ids):
            for j, kid in enumerate(known_ids):
                costs[i, j] = float(np.linalg.norm(
                    detected_betas_per_person[did] - self.known_betas[kid]
                ))

        # Greedy assignment (sufficient for small N)
        assignments = {}
        used_known = set()

        while len(assignments) < min(len(det_ids), len(known_ids)):
            best_cost = float("inf")
            best_i, best_j = -1, -1

            for i, did in enumerate(det_ids):
                if did in assignments:
                    continue
                for j, kid in enumerate(known_ids):
                    if kid in used_known:
                        continue
                    if costs[i, j] < best_cost:
                        best_cost = costs[i, j]
                        best_i, best_j = i, j

            if best_i < 0:
                break

            dim = len(detected_betas_per_person[det_ids[best_i]].ravel())
            threshold = self._confidence_threshold(dim)
            confidence = max(0.0, 1.0 - best_cost / threshold)
            assignments[det_ids[best_i]] = (known_ids[best_j], confidence)
            used_known.add(known_ids[best_j])

        return assignments

    def detect_swap(
        self,
        frame_idx: int,
        per_person_betas: dict[int, np.ndarray],
    ) -> list[tuple[int, int]]:
        """Detect if track IDs have swapped by checking beta distances.

        Args:
            frame_idx: current frame (for logging)
            per_person_betas: {track_id: (10,) betas at this frame}

        Returns:
            list of (track_id_a, track_id_b) pairs that should swap
        """
        if len(per_person_betas) < 2 or len(self.known_betas) < 2:
            return []

        track_ids = list(per_person_betas.keys())
        swaps = []

        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                tid_a, tid_b = track_ids[i], track_ids[j]
                betas_a = per_person_betas[tid_a]
                betas_b = per_person_betas[tid_b]

                if tid_a not in self.known_betas or tid_b not in self.known_betas:
                    continue

                known_a = self.known_betas[tid_a]
                known_b = self.known_betas[tid_b]

                # Current assignment distances
                dist_aa = float(np.linalg.norm(betas_a - known_a))
                dist_bb = float(np.linalg.norm(betas_b - known_b))
                current_cost = dist_aa + dist_bb

                # Swapped assignment distances
                dist_ab = float(np.linalg.norm(betas_a - known_b))
                dist_ba = float(np.linalg.norm(betas_b - known_a))
                swapped_cost = dist_ab + dist_ba

                # Swap if it significantly improves the match
                if swapped_cost < current_cost * 0.7:
                    swaps.append((tid_a, tid_b))

        return swaps

    def verify_track_consistency(
        self,
        track_id: int,
        per_frame_betas: np.ndarray,
        window: int = 30,
        threshold: float = 2.0,
    ) -> list[int]:
        """Find frames where betas drift suggests identity swap.

        Args:
            track_id: which track to check
            per_frame_betas: (F, 10) betas for this track
            window: frames to look at for running mean
            threshold: L2 distance threshold for flagging

        Returns:
            list of frame indices where drift is detected
        """
        if track_id not in self.known_betas:
            return []

        known = self.known_betas[track_id]
        distances = np.linalg.norm(per_frame_betas - known[None], axis=1)

        flagged = []
        for i in range(len(distances)):
            # Compare against local window mean
            start = max(0, i - window)
            end = min(len(distances), i + window)
            local_mean = distances[start:end].mean()

            if distances[i] > threshold and distances[i] > local_mean * 2.0:
                flagged.append(i)

        return flagged
