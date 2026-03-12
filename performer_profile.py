"""Persistent performer profiles for identity across sessions.

Reuses facepipe's PoseLibrary pattern for body: a performer's SMPL betas
(body shape) don't change between sessions, so they act as a biometric
fingerprint for re-identification.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class PerformerProfile:
    """Persistent identity for a performer across sessions."""
    performer_id: str
    name: str
    betas_mean: np.ndarray       # (10,) mean SMPL shape from all sessions
    betas_std: np.ndarray        # (10,) std dev for matching confidence
    height_cm: float | None = None
    sessions: list[str] = field(default_factory=list)
    total_keyframes: int = 0
    created: str = ""
    updated: str = ""

    def __post_init__(self):
        if not self.created:
            self.created = datetime.now().isoformat()
        if not self.updated:
            self.updated = self.created

    def update_betas(self, new_betas: np.ndarray, session_id: str = "") -> None:
        """Running mean/std update from new session data.

        Args:
            new_betas: (K, 10) betas from K high-confidence frames in new session
        """
        if new_betas.ndim == 1:
            new_betas = new_betas[None]  # (1, 10)

        n_old = self.total_keyframes
        n_new = len(new_betas)
        n_total = n_old + n_new

        new_mean = new_betas.mean(axis=0)
        new_std = new_betas.std(axis=0) if n_new > 1 else np.zeros(10, dtype=np.float32)

        if n_old == 0:
            self.betas_mean = new_mean
            self.betas_std = new_std
        else:
            # Welford-style running mean update
            old_mean = self.betas_mean
            self.betas_mean = (old_mean * n_old + new_mean * n_new) / n_total

            # Combined variance
            old_var = self.betas_std ** 2
            new_var = new_std ** 2
            combined_var = (
                (n_old * (old_var + (old_mean - self.betas_mean) ** 2) +
                 n_new * (new_var + (new_mean - self.betas_mean) ** 2))
                / n_total
            )
            self.betas_std = np.sqrt(combined_var)

        self.total_keyframes = n_total
        if session_id and session_id not in self.sessions:
            self.sessions.append(session_id)
        self.updated = datetime.now().isoformat()

    def match_confidence(self, candidate_betas: np.ndarray) -> float:
        """Mahalanobis-like distance normalized to 0-1 confidence.

        Uses per-dimension std to weight the distance. Dimensions with
        high variance (less discriminative) contribute less.
        """
        if candidate_betas.ndim > 1:
            candidate_betas = candidate_betas.mean(axis=0)

        diff = candidate_betas - self.betas_mean

        # Use Mahalanobis-like weighting where std > 0
        std_safe = np.maximum(self.betas_std, 0.1)  # Floor to avoid div/0
        weighted_diff = diff / std_safe
        distance = float(np.linalg.norm(weighted_diff))

        # Normalize: typical weighted distances are 0-10
        confidence = max(0.0, 1.0 - distance / 10.0)
        return confidence

    def estimate_height(self) -> float | None:
        """Estimate height in cm from SMPL betas.

        Uses the first beta component which correlates strongly with height.
        This is a rough approximation.
        """
        if self.betas_mean is None:
            return None
        # Beta[0] roughly correlates with height: 0 ≈ 170cm, +1 ≈ +5cm
        base_height = 170.0
        height = base_height + self.betas_mean[0] * 5.0
        self.height_cm = float(np.clip(height, 120.0, 220.0))
        return self.height_cm

    def to_dict(self) -> dict:
        return {
            "performer_id": self.performer_id,
            "name": self.name,
            "betas_mean": self.betas_mean.tolist(),
            "betas_std": self.betas_std.tolist(),
            "height_cm": self.height_cm,
            "sessions": self.sessions,
            "total_keyframes": self.total_keyframes,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PerformerProfile:
        return cls(
            performer_id=d["performer_id"],
            name=d["name"],
            betas_mean=np.array(d["betas_mean"], dtype=np.float32),
            betas_std=np.array(d["betas_std"], dtype=np.float32),
            height_cm=d.get("height_cm"),
            sessions=d.get("sessions", []),
            total_keyframes=d.get("total_keyframes", 0),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
        )

    def save_json(self, path: str | Path) -> None:
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> PerformerProfile:
        with open(str(path)) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def create(cls, name: str, initial_betas: np.ndarray) -> PerformerProfile:
        """Create a new profile from initial session betas."""
        if initial_betas.ndim == 1:
            initial_betas = initial_betas[None]

        profile = cls(
            performer_id=str(uuid.uuid4()),
            name=name,
            betas_mean=initial_betas.mean(axis=0),
            betas_std=initial_betas.std(axis=0) if len(initial_betas) > 1 else np.zeros(10, dtype=np.float32),
            total_keyframes=len(initial_betas),
        )
        profile.estimate_height()
        return profile


class ProfileLibrary:
    """Collection of performer profiles with matching."""

    def __init__(self, profiles_dir: str | Path | None = None):
        self.profiles: list[PerformerProfile] = []
        self.profiles_dir = Path(profiles_dir) if profiles_dir else None
        if self.profiles_dir and self.profiles_dir.exists():
            self._load_all()

    def _load_all(self):
        if not self.profiles_dir:
            return
        for p in sorted(self.profiles_dir.glob("*.json")):
            try:
                self.profiles.append(PerformerProfile.load_json(p))
            except (json.JSONDecodeError, KeyError):
                pass

    def add(self, profile: PerformerProfile) -> None:
        self.profiles.append(profile)
        if self.profiles_dir:
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            profile.save_json(self.profiles_dir / f"{profile.performer_id}.json")

    def find_match(
        self,
        candidate_betas: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[PerformerProfile | None, float]:
        """Find best matching profile for candidate betas.

        Returns (profile, confidence) or (None, 0.0) if no match above threshold.
        """
        best_profile = None
        best_conf = 0.0

        for profile in self.profiles:
            conf = profile.match_confidence(candidate_betas)
            if conf > best_conf:
                best_conf = conf
                best_profile = profile

        if best_conf < threshold:
            return None, 0.0

        return best_profile, best_conf

    def match_all(
        self,
        per_person_betas: dict[int, np.ndarray],
        threshold: float = 0.5,
    ) -> dict[int, tuple[PerformerProfile, float]]:
        """Match multiple detected persons to known profiles.

        Returns {person_id: (profile, confidence)} for matches above threshold.
        """
        matches = {}
        used_profiles = set()

        # Sort by confidence descending for greedy assignment
        candidates = []
        for pid, betas in per_person_betas.items():
            for profile in self.profiles:
                if profile.performer_id in used_profiles:
                    continue
                conf = profile.match_confidence(betas)
                candidates.append((pid, profile, conf))

        candidates.sort(key=lambda x: x[2], reverse=True)

        for pid, profile, conf in candidates:
            if pid in matches or profile.performer_id in used_profiles:
                continue
            if conf >= threshold:
                matches[pid] = (profile, conf)
                used_profiles.add(profile.performer_id)

        return matches

    def save_all(self) -> None:
        if not self.profiles_dir:
            return
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        for profile in self.profiles:
            profile.save_json(self.profiles_dir / f"{profile.performer_id}.json")
