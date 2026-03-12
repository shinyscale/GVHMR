# Keyframe Editor Code Review

**Commit:** `f4395b89` — "Add human-in-the-loop capture cleaning pipeline"
**Scope:** 8 new files (3,073 additions), 5 modified files (584 additions, 9 deletions)
**Reviewer:** Claude (2026-03-12)

---

## Architecture Summary

Three-tier interactive cleaning system for multi-person mocap, embedded into the existing Gradio GUI:

| Tier | Purpose | Mechanism |
|------|---------|-----------|
| **1 — Bbox Correction** | Fix detection failures | Two-click corner editing, delta interpolation between edited keyframes |
| **2 — Per-Person Reprocessing** | Re-run GVHMR on corrected detections | Selective re-isolation reusing cached SLAM and SAM2 masks |
| **3 — Frame-Space Overrides** | Handle lifts, aerial poses | Per-person per-frame-range coordinate space (world/camera/carried) |

On top of these sits a **pose correction** system: per-joint rotation editing with SLERP interpolation between correction keyframes.

---

## High Priority Issues (Correctness / Data Loss)

### 1. Person ID vs directory index mismatch

**File:** `pose_correction_panel.py`, `_save_correction_track()` (~line 173-177)

`person_id` is matched to `person_dir` by array index (`for i, pdir in enumerate(person_dirs): if i == person_id`). If person IDs become non-contiguous after track filtering or deletion, corrections save to the wrong person's directory.

**Fix:** Match by person directory name (e.g., `person_00`, `person_01`) parsed from the directory path, not by enumeration index.

### 2. Reprocess deletes data before re-running

**File:** `multi_person_split.py`, `reprocess_person()`

`unlink()` on the isolated video and `shutil.rmtree()` on the demo directory happen before the GVHMR re-run. If reprocessing crashes (OOM, CUDA error, etc.), the old data is permanently lost.

**Fix:** Rename to `.bak` before reprocessing, delete `.bak` only after successful completion:
```python
old_video = person_dir / "isolated_video.mp4"
old_demo = person_dir / "demo"
if old_video.exists():
    old_video.rename(old_video.with_suffix(".mp4.bak"))
if old_demo.exists():
    old_demo.rename(person_dir / "demo.bak")
# ... run reprocess ...
# on success:
(person_dir / "isolated_video.mp4.bak").unlink(missing_ok=True)
shutil.rmtree(person_dir / "demo.bak", ignore_errors=True)
```

### 3. ID swap doesn't swap SMPL params

**File:** `identity_panel.py`, `on_swap_ids()`

Swaps bboxes and confidences from `frame_idx` onward, but does NOT swap the SMPL params or correction tracks. After a swap, the skeleton overlay shows the wrong person's body on the swapped bounding box until the user reprocesses.

**Fix options:**
- Also swap the SMPL params in session state (quick but creates a disconnected state from disk)
- Auto-mark both persons as dirty after swap and show a warning: "Reprocess required after ID swap"
- Prevent opening the pose corrector for swapped-but-not-reprocessed persons

---

## Medium Priority Issues (UX / Robustness)

### 4. Video reopened on every frame render

**File:** `pose_correction_panel.py`, `_extract_frame_direct()`

Opens and closes `cv2.VideoCapture` on every call — triggered on every slider change and click. The identity panel has an LRU frame cache (50 frames) but the pose panel doesn't use it.

**Fix:** Add a frame cache to `_POSE_SESSION`, or better, share the identity panel's cache via the session_id linkage.

### 5. Dead code in OcclusionBridge

**File:** `identity_bridge.py`

`_alpha()` (cosine falloff) and the `influence_radius` parameter are defined but never referenced by any bridge method. Vestigial from the facepipe port.

**Fix:** Remove `_alpha()` and `influence_radius`, or implement falloff-weighted blending at span boundaries (see issue #6).

### 6. Abrupt transitions at span boundaries

**File:** `identity_bridge.py`, `bridge_poses()`

The transition from original data to bridged data is instantaneous at span edges. Frame 99 has original data, frame 100 has fully interpolated data. This creates a visible pop.

**Fix:** Add a crossfade region (3-5 frames) at each span boundary:
```python
crossfade = 5
for i in range(crossfade):
    alpha = (i + 1) / (crossfade + 1)
    bridged[start + i] = slerp(original[start + i], bridged[start + i], alpha)
    bridged[end - i] = slerp(original[end - i], bridged[end - i], alpha)
```

### 7. Re-export hardcodes skip_world_grounding=True

**File:** `pose_correction_panel.py`, `on_reexport_bvh()`

Always passes `skip_world_grounding=True` to `convert_params_to_bvh()`. If a user only made joint corrections (no space overrides) and wants world-grounded output, they can't get it through this path.

**Fix:** Add a checkbox: "Skip world grounding (required for camera/carried space overrides)". Auto-check it when any space override exists.

### 8. Unbounded session data growth

**File:** `identity_panel.py`, `_SESSION_DATA`

Module-level dict grows with each session — decoded video frames in the LRU cache, full bbox arrays, SMPL params. No cleanup when a Gradio session ends.

**Fix:** Add a TTL or max-sessions limit. Gradio's `gr.State` has session lifecycle hooks — register cleanup on session end, or use a bounded dict that evicts oldest sessions.

### 9. Reprocess button label is static

**File:** `identity_panel.py`

Button text says "Reprocess All (0 dirty)" but the dirty count is never dynamically updated after persons become dirty or reprocessing completes.

**Fix:** Return updated button label from callbacks that modify dirty state.

### 10. classify_occlusion missing "partial" case

**File:** `identity_bridge.py`, `classify_occlusion()`

Docstring says returns `'brief', 'partial', 'full', 'extended'` but the code jumps from brief (<0.5s) directly to full (0.5-3s). The "partial" case is missing.

**Fix:** Either add the partial case (e.g., 0.5-1.0s) or update the docstring to match the code.

---

## Low Priority Issues (Improvements)

### 11. No undo/redo for pose corrections

Interactive joint editing without undo is painful — accidentally nudging a slider creates a correction that requires explicit "Reset Joint" to undo. Even a simple stack would help:
```python
@dataclass
class CorrectionUndoEntry:
    frame: int
    person_id: int
    old_correction: PoseCorrection | None
    new_correction: PoseCorrection | None
```

### 12. Euler slider creates correction on every release

**File:** `pose_correction_panel.py`, `on_euler_change()`

A correction is created at the current frame even for tiny accidental slider movements. Consider a deadzone threshold (e.g., ignore changes < 1 degree from original) or only create corrections on explicit "Apply" button press.

### 13. Hardcoded normalization constants

**File:** `identity_confidence.py`

Shape distance divided by 5.0, motion distance clipped to [0,1]. These are empirically tuned but undocumented. Different HMR models or video types may need different values.

**Suggestion:** Move to a config dict at the top of the file with comments explaining calibration context.

### 14. Greedy assignment instead of Hungarian

**Files:** `identity_reid.py` `match_all()`, `performer_profile.py` `match_all()`

Greedy (pick global minimum, assign, repeat) works for 2-3 persons but produces suboptimal assignments for 4+.

**Suggestion:** Add `scipy.optimize.linear_sum_assignment` as an option for N >= 4.

### 15. Camera-space X/Z drift during space overrides

**File:** `smplx_to_bvh.py`, `_apply_space_overrides()`

In camera mode, only Y translation is interpolated from world frames. X and Z come from camera-space translation. For handheld/moving cameras, the character slides horizontally during camera-space spans.

**Suggestion:** Document this limitation prominently, or optionally interpolate X/Z from world frames too.

### 16. Carried mode fallback is static

**File:** `smplx_to_bvh.py`, `_apply_space_overrides()`

When reference person params are unavailable, the carried person floats at a fixed height (`y_before + y_offset`) regardless of carrier motion.

**Suggestion:** Log a clear warning when the fallback activates.

### 17. No unit tests

Given the complexity of the interpolation logic (SLERP, bbox delta interpolation, space overrides, confidence scoring), unit tests would catch regressions early. Suggested test cases:

- `test_slerp_axis_angle()` — known rotation pairs, identity, 180-degree edge case
- `test_apply_corrections()` — single keyframe, multi-keyframe, out-of-range frames
- `test_interpolate_bbox_corrections()` — single correction, adjacent, boundary, empty
- `test_compute_all_confidences()` — synthetic track with known occlusion patterns
- `test_bridge_poses()` — verify SLERP output at midpoint, boundary frames

---

## What's Working Really Well

- **Corrections applied before smoothing** in `smplx_to_bvh.py` — exactly right. Corrections become authoritative data that flows through the rest of the pipeline naturally.
- **Delta-based bbox interpolation** — preserving original tracking jitter instead of creating artificial smoothness is the correct choice.
- **Click-to-select joints** with 30px hit test — intuitive within Gradio's constraints.
- **Auto-detect span** bridging identity confidence into pose correction — nice cross-system integration.
- **Performer profiles with Welford running mean** — cross-session identity persistence for studio workflows is forward-thinking.
- **Incremental reprocessing reusing shared SLAM** — critical for iteration speed. Nobody wants to wait for SLAM to re-run when they only changed one person's bbox.
- **Sparse body_pose storage** (only edited joints) is memory-efficient and serialization-friendly.
- **"Carried" space mode** deriving root position from a reference person's world root + Y offset is a clever solution for dance lifts.
- **Detection confidence propagation** through `person_tracker.py` is clean and backward-compatible.

---

## Fix Status

| # | Issue | Status | Commit |
|---|-------|--------|--------|
| 1 | Person ID vs directory index mismatch | **FIXED** | `pid_to_dir`/`pid_to_index` maps in both panels |
| 2 | Reprocess deletes data before re-running | **FIXED** | `.bak` rename/restore pattern |
| 3 | ID swap doesn't mark dirty | **FIXED** | Both persons marked dirty after swap |
| 4 | Video reopened on every frame render | **FIXED** | Pose panel shares identity panel's LRU cache |
| 5 | Dead code in OcclusionBridge | **FIXED** | Revived by #6 — `_alpha()` and `influence_radius` used in crossfade |
| 6 | Abrupt transitions at span boundaries | **FIXED** | `_crossfade_boundaries()` with cosine falloff at span edges |
| 7 | Re-export hardcodes skip_world_grounding | Open | |
| 8 | Unbounded session data growth | Open | |
| 9 | Reprocess button label is static | **FIXED** | Dynamic label from `_reprocess_btn_label()` |
| 10 | classify_occlusion missing "partial" case | **FIXED** | Added "partial" (0.5-1.5s) between brief and full |
| 11-17 | Low priority improvements | Open | |

---

## Suggested Priority Order (remaining)

1. **Span boundary crossfade** (#6) — visual quality
2. **Skip world grounding checkbox** (#7) — workflow flexibility
3. **Remove dead code** (#5, #10) — cleanup
4. **Session cleanup** (#8) — memory management
5. **Undo/redo** (#11) — quality of life
6. **Unit tests** (#17) — regression safety net
