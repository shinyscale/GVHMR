# Multi-Person Identity Tracking with Keyframe Verification

**Status:** Draft
**Date:** 2026-03-11
**Depends on:** `multi_person_inpainting.md` (SAM2 + inpainting pipeline)
**Inspired by:** facepipe keyframe/interpolation/corrector architecture

## Problem

Multi-person body capture breaks during occlusion. When one performer lifts or
crosses in front of another, three failure modes occur:

1. **Identity swap** — Person A's skeleton jumps onto Person B
2. **Track loss** — Occluded person vanishes, re-detected as new ID after
3. **Pose hallucination** — Model guesses wrong pose for the occluded person

Current GVHMR processes only the single largest person (`tracker.get_one_track()`).
The multi-person inpainting spec handles the "clean video per person" problem but
doesn't address identity persistence, confidence scoring, or correction through
occlusion.

## Design Principles

1. **Skeleton shape is biometric** — SMPL betas (10D body shape) don't change
   between frames. Limb ratios, height, shoulder width are fingerprints.
2. **Keyframes are truth anchors** — verified identity at known-good frames
   constrains what happens in uncertain frames.
3. **Confidence should be explicit** — the system must know when it's uncertain
   and say so, rather than silently guessing wrong.
4. **Same architecture as facepipe** — keyframe model, interpolation engine,
   and corrector pattern translate directly to body tracking.

## Architecture

### Tier 1: Identity Establishment (ROM equivalent)

Before processing a multi-person video, establish identity profiles:

```
Option A: First-clear-frame auto-detection
  - Find first frame where all N people are clearly separated (no bbox overlap)
  - Extract SMPL betas for each person → identity fingerprint
  - Auto-assign track IDs: Person 0, Person 1, ...

Option B: User-guided identification
  - User clicks on each person in a reference frame
  - Assigns name/role labels ("Dancer A", "Dancer B")
  - System extracts betas + appearance embedding for each

Option C: Known performer profiles
  - Load saved PerformerProfile from previous sessions
  - Match detected people to known profiles via beta distance
```

### Tier 2: Confidence-Scored Tracking

Per-person, per-frame confidence score computed from multiple signals:

```python
@dataclass
class TrackConfidence:
    detection_score: float      # YOLO confidence (0-1)
    visible_keypoints: float    # Fraction of ViTPose joints with conf > 0.5
    bbox_overlap: float         # IoU with other people's bboxes (0 = no overlap)
    shape_consistency: float    # Beta distance from established identity (0 = match)
    motion_consistency: float   # Deviation from predicted position (0 = on track)

    @property
    def overall(self) -> float:
        """Weighted combination. Returns 0-1 where 1 = fully confident."""
        weights = [0.15, 0.30, 0.20, 0.20, 0.15]
        raw = [
            self.detection_score,
            self.visible_keypoints,
            1.0 - self.bbox_overlap,      # Less overlap = more confident
            1.0 - self.shape_consistency,  # Closer to identity = more confident
            1.0 - self.motion_consistency, # Closer to prediction = more confident
        ]
        return sum(w * v for w, v in zip(weights, raw))
```

**Confidence thresholds:**
- `>= 0.8` — High confidence. Use output directly.
- `0.4 - 0.8` — Medium confidence. Flag for review, use with caution.
- `< 0.4` — Low confidence. Likely occluded or swapped. Bridge with interpolation.

### Tier 3: Identity Keyframes

Adapted from facepipe's `Keyframe` and `KeyframeTrack`:

```python
@dataclass
class IdentityKeyframe:
    """A verified identity assignment at a specific frame."""
    id: str                         # UUID
    frame_index: int
    timestamp: float
    person_id: int                  # Which tracked person (0, 1, ...)
    bbox: np.ndarray                # (4,) xyxy bounding box
    betas: np.ndarray               # (10,) SMPL shape parameters
    body_pose: np.ndarray           # (J*3,) joint rotations at this frame
    confidence: TrackConfidence     # Computed confidence at this frame
    verified: bool = False          # True if user-verified (vs auto-generated)
    thumbnail: bytes | None = None  # Cropped frame JPEG
    metadata: dict = field(default_factory=dict)


@dataclass
class IdentityTrack:
    """Ordered keyframes for a single person across a video."""
    person_id: int
    performer_name: str = ""
    keyframes: list[IdentityKeyframe] = field(default_factory=list)
    established_betas: np.ndarray | None = None  # Mean betas from high-conf frames

    def add_keyframe(self, frame_index: int, ...) -> IdentityKeyframe
    def remove_keyframe(self, frame_index: int) -> bool
    def get_nearest(self, frame_index: int) -> IdentityKeyframe | None
    def high_confidence_frames(self, threshold: float = 0.8) -> list[int]
    def low_confidence_spans(self, threshold: float = 0.4) -> list[tuple[int, int]]

    def save_json(self, path: str | Path) -> None
    @classmethod
    def load_json(cls, path: str | Path) -> IdentityTrack
```

**Auto-keyframe generation:**
- After initial tracking, auto-create keyframes at confidence peaks
  (local maxima of `overall` confidence, spaced >= 30 frames apart)
- Mark as `verified=False` — user can confirm or correct in the GUI

### Tier 4: Occlusion Bridging

When confidence drops below threshold, bridge the gap using the same
cosine-falloff interpolation as facepipe:

```python
class OcclusionBridge:
    """Interpolates body pose through low-confidence spans."""

    def __init__(self, track: IdentityTrack, full_sequence: np.ndarray,
                 confidence_threshold: float = 0.4,
                 influence_radius: int = 15):
        ...

    def _alpha(self, distance: float) -> float:
        """Cosine falloff: 0.5 * (1 + cos(π * distance / radius))"""
        t = min(distance / self.influence_radius, 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    def compute_frame(self, frame_idx: int, person_id: int) -> np.ndarray:
        """Returns interpolated body pose for a single frame.

        If frame is high-confidence: return original tracked pose.
        If frame is low-confidence: blend between nearest verified
        keyframes using cosine falloff.
        """
        ...

    def bake(self, person_id: int) -> np.ndarray:
        """Returns (N, J*3) fully interpolated sequence.
        Vectorized distance matrix approach, same as facepipe.
        """
        ...
```

**Bridging strategy for different occlusion types:**

| Scenario | Duration | Strategy |
|----------|----------|----------|
| Brief crossing (< 15 frames) | < 0.5s | Cosine interpolation between boundary keyframes |
| Partial occlusion (some joints visible) | Variable | Blend: visible joints from tracker, occluded joints from interpolation |
| Full occlusion (lift, behind) | 0.5-3s | Interpolation + motion prediction from pre-occlusion velocity |
| Extended occlusion (> 3s) | > 3s | Hold last pose, flag for manual review |

### Tier 5: Shape-Based Re-Identification

When a person reappears after occlusion, match them to the correct identity
using SMPL betas:

```python
class ShapeReIdentifier:
    """Matches detected people to known identities via body shape."""

    def __init__(self, profiles: list[IdentityTrack]):
        # Extract mean betas from each track's high-confidence frames
        self.known_betas = {
            track.person_id: track.established_betas
            for track in profiles
        }

    def match(self, detected_betas: np.ndarray) -> tuple[int, float]:
        """Returns (person_id, confidence) for best match.

        Uses L2 distance in beta space. Confidence = 1 - normalized_distance.
        """
        distances = {
            pid: np.linalg.norm(detected_betas - known)
            for pid, known in self.known_betas.items()
        }
        best_id = min(distances, key=distances.get)
        # Normalize: typical beta distances are 0-5
        confidence = max(0, 1.0 - distances[best_id] / 5.0)
        return best_id, confidence

    def detect_swap(self, frame_idx: int,
                    assignments: dict[int, np.ndarray]) -> list[tuple[int, int]]:
        """Detect if track IDs have swapped by checking beta distances.

        Returns list of (track_id_a, track_id_b) pairs that should swap.
        """
        ...
```

### Tier 6: Performer Profiles (Persistent)

Reuses facepipe's PoseLibrary pattern for body:

```python
@dataclass
class PerformerProfile:
    """Persistent identity for a performer across sessions."""
    performer_id: str               # UUID
    name: str                       # "Jane Doe"
    betas_mean: np.ndarray          # (10,) mean SMPL shape from all sessions
    betas_std: np.ndarray           # (10,) std dev (for matching confidence)
    height_cm: float | None = None  # Estimated from SMPL
    sessions: list[str] = field(default_factory=list)  # Session IDs
    total_keyframes: int = 0        # Accumulated across all sessions
    created: str = ""
    updated: str = ""

    def update_betas(self, new_betas: np.ndarray) -> None:
        """Running mean/std update from new session data."""
        ...

    def match_confidence(self, candidate_betas: np.ndarray) -> float:
        """Mahalanobis-like distance normalized to 0-1 confidence."""
        ...

    def save_json(self, path: str | Path) -> None
    @classmethod
    def load_json(cls, path: str | Path) -> PerformerProfile
```

## Integration with Existing Pipeline

### Modified Pipeline Flow

```
Current (single person):
  Video → YOLO(largest) → ViTPose → GVHMR → BVH/FBX

Proposed (multi-person):
  Video → YOLO(all tracks) → Identity Establishment
        → Per-person: ViTPose → GVHMR → confidence scoring
        → Shape re-ID at confidence drops
        → Occlusion bridging for low-confidence spans
        → User keyframe verification (GUI)
        → Per-person BVH/FBX export with identity metadata
```

### File Changes

| File | Change |
|------|--------|
| `hmr4d/utils/preproc/tracker.py` | Expose `track()` results, add `get_all_tracks()` |
| `tools/demo/demo.py` | Loop over tracks instead of `get_one_track()` |
| `smplx_to_bvh.py` | Add `person_id` to output metadata, multi-person export |
| `gvhmr_gui.py` | Person selector, keyframe panel, confidence timeline |

### New Files

```
src/tracking/
├── confidence.py       # TrackConfidence scoring
├── identity.py         # IdentityKeyframe, IdentityTrack
├── bridge.py           # OcclusionBridge (cosine interpolation)
├── reid.py             # ShapeReIdentifier
└── profile.py          # PerformerProfile (persistent)
```

## GUI: Identity & Keyframe Panel

Add to gvhmr_gui.py (new tab or panel within existing tabs):

```
┌─────────────────────────────────────────────────────┐
│ Multi-Person Tracking                               │
├─────────────────────────────────────────────────────┤
│ Persons: [Person 0 ▼] [Person 1 ▼]  [+ Add]        │
│                                                     │
│ Timeline: ═══●══════●═══░░░░░░░●══════●═══          │
│           kf12    kf45   occluded   kf78  kf102      │
│           ✓        ✓    (bridged)    ✓     ○         │
│                                                     │
│ Confidence: ▁▂▃▅▇▇▇▅▃▁▁▁▁▁▃▅▇▇▇▇▅▃▂▁              │
│             ────────────────────────────              │
│             Frame 1              Frame N             │
│                                                     │
│ Current Frame: 45                                   │
│ Person 0: conf=0.92 ✓  bbox=[120,50,340,480]        │
│ Person 1: conf=0.31 ⚠  bbox=[180,60,290,420]        │
│                                                     │
│ [Verify Identity] [Add Keyframe] [Swap IDs]          │
│ [Auto-Keyframe All] [Bridge Occlusions]              │
│                                                     │
│ Performer Profiles:                                 │
│   Jane Doe (3 sessions, 47 keyframes)  [Load]       │
│   John Smith (1 session, 12 keyframes) [Load]       │
│   [+ New Profile]                                   │
└─────────────────────────────────────────────────────┘
```

### Confidence Timeline Visualization

The confidence bar shows per-frame confidence for the selected person:
- Green (▇) = high confidence (>= 0.8)
- Yellow (▅) = medium (0.4-0.8)
- Red (▁) = low (< 0.4), likely occluded
- Gray (░) = bridged by interpolation

Keyframe markers (●) on the timeline show verified (✓) and unverified (○) anchors.

## Export

### Per-Person BVH with Identity

```bvh
HIERARCHY
ROOT Pelvis
{
  ; person_id: 0
  ; performer: Jane Doe
  ; confidence_mean: 0.87
  ; bridged_frames: 45-62, 178-195
  ...
}
MOTION
Frames: 300
Frame Time: 0.0333
...
```

### Multi-Person FBX

Single FBX file with multiple armatures:
- `Person_0_JaneDoe` — skeleton + animation
- `Person_1_JohnSmith` — skeleton + animation
- Shared timeline, individual tracks

### Confidence Channel Export

Optional sidecar CSV with per-frame confidence:

```csv
frame,person_id,overall,detection,visible_kp,bbox_overlap,shape_dist,motion_dist,bridged
0,0,0.95,0.98,1.0,0.0,0.02,0.01,false
0,1,0.91,0.95,0.94,0.0,0.03,0.02,false
...
45,1,0.28,0.40,0.35,0.65,0.08,0.15,true
```

## Implementation Order

1. **`confidence.py`** — TrackConfidence dataclass + scoring from YOLO/ViTPose signals
2. **`identity.py`** — IdentityKeyframe, IdentityTrack (port from facepipe Keyframe/KeyframeTrack)
3. **`reid.py`** — ShapeReIdentifier using SMPL betas L2 distance
4. **`bridge.py`** — OcclusionBridge with cosine falloff (port from facepipe InterpolationEngine)
5. **`profile.py`** — PerformerProfile persistence
6. **`tracker.py` changes** — `get_all_tracks()`, confidence integration
7. **`demo.py` changes** — Multi-person loop with re-ID
8. **GUI** — Identity panel, confidence timeline, keyframe management
9. **Export** — Multi-person BVH/FBX with identity metadata

## Relationship to Facepipe

This system shares the same conceptual architecture:

| Concept | Facepipe | Bodypipe |
|---------|----------|----------|
| Data unit | (52,) blendshape weights | (J*3,) joint rotations + (10,) betas |
| Keyframe | (raw, corrected) blendshapes | (person_id, bbox, betas, pose) |
| Influence radius | Frames (default 15) | Frames (default 15) |
| Falloff | Cosine | Cosine |
| Corrector | Ridge: raw→corrected blendshapes | OcclusionBridge: interpolated pose |
| Performer ID | PoseLibrary (ROM calibration) | PerformerProfile (betas fingerprint) |
| Actor/Character | ActorProfile + CharacterProfile | PerformerProfile + Role binding |
| Persistence | JSON (keyframe track, pose library) | JSON (identity track, performer profile) |

The key difference: facepipe corrects **what** the capture says (blendshape values),
bodypipe corrects **who** the capture is tracking (identity through occlusion).
Both use keyframes as truth anchors and cosine-falloff interpolation for bridging.

## Connection to Scripty Capture Mode

This maps to ShootDay's entity model:

- `PerformerProfile` → `SD_Talent` (biometric: face geometry, body shape)
- Role binding → `SD_Role` (talent + character, with body/face flags)
- Session → `SD_Take` within `SD_ShotSetup`
- Multi-person tracking → multiple `SD_Role` entries per `SD_ShotSetup`

The naming convention `{Episode}_{Scene}_{Moment}_{ShotSetup}_{Take}` can encode
person identity in the export filenames:
`EP1_S01A_LiftSequence_TwoDancers_T03_Person0_JaneDoe.bvh`
