"""Keyframe-based pose correction for multi-person capture.

Data model, SLERP interpolation, FK helpers, and quick-fix utilities
for correcting bad GVHMR poses (flipped orientations, impossible poses
during lifts). Corrections at keyframes are interpolated through bad
spans, then re-exported to BVH/FBX.

Follows the same pattern as IdentityTrack / IdentityKeyframe in
identity_tracking.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from smplx_to_bvh import JOINT_NAMES, JOINT_PARENTS


# ── Frame Space Override (per-person, per-frame-range) ──


@dataclass
class FrameSpaceOverride:
    """Override the coordinate space used for a range of frames.

    Spaces:
        "world"   — Normal GVHMR world-space output (default, works when grounded).
        "camera"  — Camera-space body pose + root orient; root translation interpolated
                     from nearest grounded frames. Skips all world grounding.
        "carried" — Camera-space body pose; root position derived from reference_person's
                     world root + y_offset. For lifts where the carried person's position
                     is relative to the carrier.
    """
    frame_start: int
    frame_end: int
    space: str = "world"  # "world" | "camera" | "carried"
    reference_person: int | None = None  # for "carried" mode
    y_offset: float = 0.4  # meters above reference person's root

    def to_dict(self) -> dict:
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "space": self.space,
            "reference_person": self.reference_person,
            "y_offset": self.y_offset,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FrameSpaceOverride:
        return cls(
            frame_start=d["frame_start"],
            frame_end=d["frame_end"],
            space=d.get("space", "world"),
            reference_person=d.get("reference_person"),
            y_offset=d.get("y_offset", 0.4),
        )


# ── Left/Right joint swap pairs (body joints 1-21) ──
# SMPL-X body joints are laid out with L/R pairs at adjacent indices.
_LR_SWAP_PAIRS = [
    (1, 2),    # L_Hip <-> R_Hip
    (4, 5),    # L_Knee <-> R_Knee
    (7, 8),    # L_Ankle <-> R_Ankle
    (10, 11),  # L_Foot <-> R_Foot
    (13, 14),  # L_Collar <-> R_Collar
    (16, 17),  # L_Shoulder <-> R_Shoulder
    (18, 19),  # L_Elbow <-> R_Elbow
    (20, 21),  # L_Wrist <-> R_Wrist
]

# Body joint names for dropdown (indices 0-21)
BODY_JOINT_NAMES = JOINT_NAMES[:22]


def _np_to_list(arr: np.ndarray | None) -> list | None:
    if arr is None:
        return None
    return arr.tolist()


def _list_to_np(lst: list | None, dtype=np.float32) -> np.ndarray | None:
    if lst is None:
        return None
    return np.array(lst, dtype=dtype)


def _dict_to_sparse(d: dict | None) -> dict[int, np.ndarray] | None:
    """Convert JSON dict (str keys) to sparse joint dict (int keys, ndarray values)."""
    if d is None:
        return None
    return {int(k): np.array(v, dtype=np.float32) for k, v in d.items()}


def _sparse_to_dict(d: dict[int, np.ndarray] | None) -> dict | None:
    """Convert sparse joint dict to JSON-serializable dict."""
    if d is None:
        return None
    return {str(k): v.tolist() for k, v in d.items()}


# ── Data Model ──


@dataclass
class PoseCorrection:
    """A pose correction at a specific frame for a specific person."""
    frame_index: int
    person_id: int
    correction_type: str  # "global_orient" | "joint" | "full_pose" | "copy_from_frame"
    global_orient: np.ndarray | None = None   # (3,) axis-angle
    body_pose: dict[int, np.ndarray] | None = None  # {joint_idx: (3,) axis-angle}, sparse
    transl: np.ndarray | None = None          # (3,)
    source_frame: int | None = None           # for copy_from_frame type

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "person_id": self.person_id,
            "correction_type": self.correction_type,
            "global_orient": _np_to_list(self.global_orient),
            "body_pose": _sparse_to_dict(self.body_pose),
            "transl": _np_to_list(self.transl),
            "source_frame": self.source_frame,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoseCorrection:
        return cls(
            frame_index=d["frame_index"],
            person_id=d["person_id"],
            correction_type=d["correction_type"],
            global_orient=_list_to_np(d.get("global_orient")),
            body_pose=_dict_to_sparse(d.get("body_pose")),
            transl=_list_to_np(d.get("transl")),
            source_frame=d.get("source_frame"),
        )


@dataclass
class CorrectionTrack:
    """Ordered corrections for a single person across a video."""
    person_id: int
    corrections: list[PoseCorrection] = field(default_factory=list)
    space_overrides: list[FrameSpaceOverride] = field(default_factory=list)

    def add_correction(
        self,
        frame_index: int,
        correction_type: str = "joint",
        global_orient: np.ndarray | None = None,
        body_pose: dict[int, np.ndarray] | None = None,
        transl: np.ndarray | None = None,
        source_frame: int | None = None,
    ) -> PoseCorrection:
        """Add or replace a correction at the given frame."""
        self.corrections = [c for c in self.corrections if c.frame_index != frame_index]
        corr = PoseCorrection(
            frame_index=frame_index,
            person_id=self.person_id,
            correction_type=correction_type,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            source_frame=source_frame,
        )
        self.corrections.append(corr)
        self.corrections.sort(key=lambda c: c.frame_index)
        return corr

    def remove_correction(self, frame_index: int) -> bool:
        """Remove correction at frame_index. Returns True if found."""
        before = len(self.corrections)
        self.corrections = [c for c in self.corrections if c.frame_index != frame_index]
        return len(self.corrections) < before

    def get_correction(self, frame_index: int) -> PoseCorrection | None:
        """Get correction at exact frame, or None."""
        for c in self.corrections:
            if c.frame_index == frame_index:
                return c
        return None

    def get_surrounding(self, frame_index: int) -> tuple[PoseCorrection | None, PoseCorrection | None]:
        """Return (prev_correction, next_correction) surrounding frame_index."""
        prev = None
        nxt = None
        for c in self.corrections:
            if c.frame_index <= frame_index:
                prev = c
            if c.frame_index >= frame_index and nxt is None:
                nxt = c
        return prev, nxt

    def add_space_override(self, override: FrameSpaceOverride) -> None:
        """Add or replace a frame-space override (merges overlapping ranges)."""
        # Remove any existing overrides that overlap
        self.space_overrides = [
            o for o in self.space_overrides
            if o.frame_end < override.frame_start or o.frame_start > override.frame_end
        ]
        self.space_overrides.append(override)
        self.space_overrides.sort(key=lambda o: o.frame_start)

    def remove_space_override(self, frame_start: int, frame_end: int) -> bool:
        """Remove override covering the given range. Returns True if found."""
        before = len(self.space_overrides)
        self.space_overrides = [
            o for o in self.space_overrides
            if not (o.frame_start == frame_start and o.frame_end == frame_end)
        ]
        return len(self.space_overrides) < before

    def get_space_at_frame(self, frame: int) -> FrameSpaceOverride | None:
        """Return the space override active at a given frame, or None (=world)."""
        for o in self.space_overrides:
            if o.frame_start <= frame <= o.frame_end:
                return o
        return None

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "corrections": [c.to_dict() for c in self.corrections],
            "space_overrides": [o.to_dict() for o in self.space_overrides],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorrectionTrack:
        return cls(
            person_id=d["person_id"],
            corrections=[PoseCorrection.from_dict(c) for c in d.get("corrections", [])],
            space_overrides=[FrameSpaceOverride.from_dict(o) for o in d.get("space_overrides", [])],
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(str(path), "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> CorrectionTrack:
        with open(str(path)) as f:
            return cls.from_dict(json.load(f))


# ── SLERP Interpolation ──


def _slerp_axis_angle(aa_a: np.ndarray, aa_b: np.ndarray, t: float) -> np.ndarray:
    """SLERP between two axis-angle rotations at parameter t in [0, 1]."""
    r_a = Rotation.from_rotvec(aa_a)
    r_b = Rotation.from_rotvec(aa_b)
    key_rots = Rotation.concatenate([r_a, r_b])
    slerp = Slerp([0.0, 1.0], key_rots)
    return slerp([max(0.0, min(1.0, t))]).as_rotvec()[0]


def _build_full_correction_pose(
    params: dict,
    correction: PoseCorrection,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a complete (global_orient, body_pose_21x3, transl) from a correction.

    For fields not specified in the correction, uses original params at that frame.
    """
    f = correction.frame_index
    n_frames = params["num_frames"]
    f = max(0, min(f, n_frames - 1))

    # Start from original
    go = params["global_orient"][f].copy()  # (3,)
    bp = params["body_pose"][f].copy()      # (21, 3)
    tr = params["transl"][f].copy()         # (3,)

    if correction.correction_type == "copy_from_frame" and correction.source_frame is not None:
        sf = max(0, min(correction.source_frame, n_frames - 1))
        go = params["global_orient"][sf].copy()
        bp = params["body_pose"][sf].copy()
        tr = params["transl"][sf].copy()

    # Override with explicit correction values
    if correction.global_orient is not None:
        go = correction.global_orient.copy()
    if correction.body_pose is not None:
        for j_idx, aa in correction.body_pose.items():
            if 0 <= j_idx < bp.shape[0]:
                bp[j_idx] = aa.copy()
    if correction.transl is not None:
        tr = correction.transl.copy()

    return go, bp, tr


def apply_corrections(params: dict, correction_track: CorrectionTrack) -> dict:
    """Apply corrections with SLERP interpolation between correction keyframes.

    For each pair of adjacent corrections, SLERP-interpolate per-joint
    rotations and linear-interpolate translations through the span.
    Outside correction spans, preserves original params.

    Returns a new params dict with corrections baked in.
    """
    if not correction_track.corrections:
        return params

    params = dict(params)
    params["global_orient"] = params["global_orient"].copy()
    params["body_pose"] = params["body_pose"].copy()
    params["transl"] = params["transl"].copy()

    n_frames = params["num_frames"]
    corrections = sorted(correction_track.corrections, key=lambda c: c.frame_index)

    # Build full corrected poses at each correction keyframe
    corrected_poses = {}
    for corr in corrections:
        go, bp, tr = _build_full_correction_pose(params, corr)
        corrected_poses[corr.frame_index] = (go, bp, tr)

    # Apply exact corrections at keyframes
    for f, (go, bp, tr) in corrected_poses.items():
        if 0 <= f < n_frames:
            params["global_orient"][f] = go
            params["body_pose"][f] = bp
            params["transl"][f] = tr

    # Interpolate between adjacent correction pairs
    for i in range(len(corrections) - 1):
        f_start = corrections[i].frame_index
        f_end = corrections[i + 1].frame_index

        if f_end - f_start <= 1:
            continue

        go_a, bp_a, tr_a = corrected_poses[f_start]
        go_b, bp_b, tr_b = corrected_poses[f_end]

        for f in range(f_start + 1, f_end):
            t = (f - f_start) / (f_end - f_start)

            # SLERP global orient
            params["global_orient"][f] = _slerp_axis_angle(go_a, go_b, t)

            # SLERP each body joint
            n_joints = bp_a.shape[0]
            for j in range(n_joints):
                params["body_pose"][f, j] = _slerp_axis_angle(bp_a[j], bp_b[j], t)

            # Linear interpolate translation
            params["transl"][f] = (1 - t) * tr_a + t * tr_b

    return params


# ── Single-Frame FK / Projection Helpers ──


def compute_skeleton_frame(
    params: dict,
    frame_idx: int,
    img_w: int,
    img_h: int,
    corrections: CorrectionTrack | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 3D and 2D joint positions for a single frame.

    If corrections are provided, applies them first. Uses forward_kinematics()
    and project_to_2d() from visualize_skeleton.

    Returns (joints_3d (N,3), joints_2d (N,2)).
    """
    from visualize_skeleton import forward_kinematics, project_to_2d, _get_projection_intrinsics

    if corrections is not None:
        params = apply_corrections(params, corrections)

    joints_3d = forward_kinematics(params, frame_idx)
    fx, fy, cx, cy, _ = _get_projection_intrinsics(params, frame_idx, img_w, img_h)
    joints_2d = project_to_2d(joints_3d, fx, fy, cx, cy)

    return joints_3d, joints_2d


def find_nearest_joint(
    click_x: float,
    click_y: float,
    joints_2d: np.ndarray,
    threshold: float = 20.0,
    max_joint: int = 22,
) -> int | None:
    """Hit-test: returns body joint index nearest to click, within threshold pixels.

    Only considers body joints (0..max_joint-1) by default, not hand joints.
    """
    body_joints = joints_2d[:max_joint]
    dists = np.sqrt((body_joints[:, 0] - click_x) ** 2 + (body_joints[:, 1] - click_y) ** 2)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= threshold:
        return min_idx
    return None


# ── Quick-Fix Helpers ──


def flip_global_orient(params: dict, frame: int, axis: str = "yaw") -> np.ndarray:
    """Apply 180-degree rotation to global_orient at a frame.

    Args:
        axis: "yaw" (Y-axis), "pitch" (X-axis), or "roll" (Z-axis)

    Returns the new global_orient (3,) axis-angle.
    """
    go = params["global_orient"][frame].copy()
    R_orig = Rotation.from_rotvec(go)

    axis_map = {"pitch": [1, 0, 0], "yaw": [0, 1, 0], "roll": [0, 0, 1]}
    flip_axis = axis_map.get(axis, [0, 1, 0])
    R_flip = Rotation.from_rotvec(np.array(flip_axis, dtype=np.float64) * np.pi)

    R_new = R_flip * R_orig
    return R_new.as_rotvec().astype(np.float32)


def mirror_lr_pose(params: dict, frame: int) -> dict[int, np.ndarray]:
    """Swap left/right body joint rotations at a frame.

    Returns sparse body_pose dict suitable for PoseCorrection.body_pose.
    """
    bp = params["body_pose"][frame].copy()  # (21, 3)
    mirrored = {}

    for l_idx, r_idx in _LR_SWAP_PAIRS:
        # body_pose indices are joint_idx - 1 (joint 0 is root/global_orient)
        l_bp = l_idx - 1
        r_bp = r_idx - 1
        if 0 <= l_bp < bp.shape[0] and 0 <= r_bp < bp.shape[0]:
            mirrored[l_bp] = bp[r_bp].copy()
            mirrored[r_bp] = bp[l_bp].copy()

    return mirrored


def copy_pose_from_frame(
    params: dict,
    src_frame: int,
    dst_frame: int,
) -> PoseCorrection:
    """Create a full-body copy correction from src_frame to dst_frame."""
    n = params["num_frames"]
    sf = max(0, min(src_frame, n - 1))

    go = params["global_orient"][sf].copy()
    bp = params["body_pose"][sf].copy()  # (21, 3)
    tr = params["transl"][sf].copy()

    # Convert full body_pose to sparse dict
    sparse_bp = {j: bp[j].copy() for j in range(bp.shape[0])}

    return PoseCorrection(
        frame_index=dst_frame,
        person_id=0,  # caller should set
        correction_type="copy_from_frame",
        global_orient=go,
        body_pose=sparse_bp,
        transl=tr,
        source_frame=src_frame,
    )


def axis_angle_to_euler_deg(aa: np.ndarray) -> np.ndarray:
    """Convert (3,) axis-angle to (3,) XYZ euler degrees."""
    return Rotation.from_rotvec(aa).as_euler("XYZ", degrees=True).astype(np.float32)


def euler_deg_to_axis_angle(euler_deg: np.ndarray) -> np.ndarray:
    """Convert (3,) XYZ euler degrees to (3,) axis-angle."""
    return Rotation.from_euler("XYZ", euler_deg, degrees=True).as_rotvec().astype(np.float32)


# ── Foot-Slide Correction ──


def compute_foot_slide_offsets(
    params: dict,
    foot_joint_idx: int,
    anchors: list[tuple[int, np.ndarray]],
    frame_start: int,
    frame_end: int,
    blend_frames: int = 5,
    forward_kinematics_fn=None,
    propagate: bool = False,
) -> dict[int, np.ndarray]:
    """Compute per-frame root translation offsets to pin a foot at anchor positions.

    Target foot positions are linearly interpolated between anchors (held
    constant outside anchor range). At every frame, FK is run to find the
    actual foot position, and the offset ``target - actual`` is returned.
    At range boundaries, offsets are faded to zero with a cosine ease over
    ``blend_frames``.

    Parameters
    ----------
    params : dict
        SMPL-X params with ``transl`` (or ``transl_world``), ``global_orient``,
        ``body_pose``, etc.
    foot_joint_idx : int
        Joint index for the foot (10 = L_Foot, 11 = R_Foot).
    anchors : list of (frame_idx, target_foot_pos (3,))
        User-specified anchor frames and desired foot positions.
    frame_start, frame_end : int
        Inclusive frame range to correct.
    blend_frames : int
        Cosine ease-in/out width at range boundaries (0 = hard cut).
    forward_kinematics_fn : callable(params, frame) -> (J, 3), optional
        FK function.  If *None*, imports from ``views.mesh_viewport``.

    Returns
    -------
    offsets : dict[int, np.ndarray]
        ``{frame_idx: (3,) offset}`` for every frame in ``[frame_start, frame_end]``.
    """
    if not anchors:
        return {}

    if forward_kinematics_fn is None:
        from views.mesh_viewport import forward_kinematics
        forward_kinematics_fn = forward_kinematics

    anchors = sorted(anchors, key=lambda a: a[0])
    anchor_frames = [a[0] for a in anchors]
    anchor_targets = [np.asarray(a[1], dtype=np.float64) for a in anchors]

    # --- interpolate TARGET POSITIONS for every frame in range ---
    raw_offsets: dict[int, np.ndarray] = {}

    for f in range(frame_start, frame_end + 1):
        # Determine the target foot position at this frame
        if f <= anchor_frames[0]:
            target = anchor_targets[0]
        elif f >= anchor_frames[-1]:
            target = anchor_targets[-1]
        else:
            # Between two anchors — lerp target positions
            for i in range(len(anchor_frames) - 1):
                if anchor_frames[i] <= f <= anchor_frames[i + 1]:
                    t = (f - anchor_frames[i]) / max(1, anchor_frames[i + 1] - anchor_frames[i])
                    target = (1.0 - t) * anchor_targets[i] + t * anchor_targets[i + 1]
                    break

        # FK at this frame to get actual foot position, then compute offset
        joints = forward_kinematics_fn(params, f)
        actual_foot = joints[foot_joint_idx]
        raw_offsets[f] = target - actual_foot

    # --- cosine blend at range boundaries ---
    if blend_frames > 0:
        for f in range(frame_start, min(frame_start + blend_frames, frame_end + 1)):
            # ease-in: 0 at frame_start, 1 at frame_start + blend_frames
            progress = (f - frame_start) / blend_frames
            weight = 0.5 * (1.0 - np.cos(np.pi * progress))
            raw_offsets[f] = raw_offsets[f] * weight

        if not propagate:
            for f in range(max(frame_end - blend_frames + 1, frame_start), frame_end + 1):
                # ease-out: 1 at frame_end - blend_frames, 0 at frame_end
                progress = (frame_end - f) / blend_frames
                weight = 0.5 * (1.0 - np.cos(np.pi * progress))
                raw_offsets[f] = raw_offsets[f] * weight

    return {f: off.astype(np.float32) for f, off in raw_offsets.items()}


# ── Drift Correction ──

def compute_drift_offsets(
    transl: np.ndarray,       # (N, 3) full translation array
    frame_start: int,
    frame_end: int,
    xz_only: bool = True,
    blend_frames: int = 5,
) -> dict[int, np.ndarray]:
    """Remove linear positional drift from a frame range.

    Computes net drift between frame_start and frame_end, then subtracts
    a proportional share at each frame so the character ends where it
    started while preserving all frame-to-frame micro-movements.

    Returns dict[int, (3,) float32] of per-frame offsets, same format as
    compute_foot_slide_offsets.
    """
    if frame_start >= frame_end:
        return {}

    drift = transl[frame_end] - transl[frame_start]
    if xz_only:
        drift = drift.copy()
        drift[1] = 0.0  # zero Y component

    span = frame_end - frame_start
    raw_offsets: dict[int, np.ndarray] = {}
    for f in range(frame_start, frame_end + 1):
        t = (f - frame_start) / span
        raw_offsets[f] = -t * drift

    # cosine ease-in at start boundary
    if blend_frames > 0:
        for f in range(frame_start, min(frame_start + blend_frames, frame_end + 1)):
            progress = (f - frame_start) / blend_frames
            weight = 0.5 * (1.0 - np.cos(np.pi * progress))
            raw_offsets[f] = raw_offsets[f] * weight

    return {f: off.astype(np.float32) for f, off in raw_offsets.items()}
