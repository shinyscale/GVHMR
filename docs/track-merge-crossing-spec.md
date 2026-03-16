# Track Merging & Crossing Annotation for Multi-Person Occlusion

## Problem Statement

Two people hugging with an orbiting camera produces 5+ fragmented OC-SORT tracks for 2 actual people. After `max_persons=2` caps to the 2 largest by cumulative area, each surviving track may only cover part of the video. The discarded fragments contain real detections for the same people during different time ranges — detections that are currently lost.

The existing Identity Inspector has bbox editing, interpolation, ID swapping, and reprocessing — all functional. But two capabilities are missing:

1. **Track merging** — recovering discarded fragments and stitching them into an active track
2. **Crossing span annotation** — marking time ranges where occlusion makes HMR output unreliable, so the system can SLERP-bridge through them

## Existing Infrastructure

| Component | File | Status |
|-----------|------|--------|
| OC-SORT/YOLO tracking | `person_tracker.py` | Working |
| Track cap (max_persons) | `multi_person_split.py:597-600` | Working |
| SAM2 segmentation | `sam2_segmenter.py` | Working |
| ProPainter inpainting | `propainter_inpaint.py` | Working (just optimized 150x) |
| GVHMR solve | `tools/demo/demo.py` | Working (Hydra comma fix) |
| Identity Inspector UI | `identity_panel.py` | Working (bbox edit, keyframes, swap, reprocess) |
| Occlusion bridging (SLERP) | `identity_bridge.py` | Working (`bridge_with_spans()`) |
| Bbox interpolation | `multi_person_split.py` | Working (`interpolate_bbox_corrections()`) |
| Incremental reprocess | `multi_person_split.py` | Working (`reprocess_person()`) |
| Confidence scoring | `identity_confidence.py` | Working (5-component per-frame) |

---

## Approach A: Track Merge + Crossing Spans (UI-driven)

### Concept
Keep the tracker as-is. Add UI tools to the Identity Inspector that let the user manually merge fragments and mark crossing spans. Reprocessing handles the rest.

### Changes

**1. Preserve inactive tracks**
- At the `max_persons` cap, store discarded tracks in `MultiPersonResult.inactive_tracks` instead of dropping them
- Pass them through to the Identity Inspector session data
- Cache already stores the full track list — just stop truncating in-memory

**2. `merge_tracks()` function** (`person_tracker.py`)
- Pure function: takes two track dicts, returns one merged dict
- Per-frame: pick bbox from whichever track has a real detection; if both, use higher confidence
- Re-interpolate gaps, re-smooth with existing `_process_track()` logic

**3. Merge UI** (`identity_panel.py`)
- Dropdown listing inactive tracks with frame range + detection count
- "Merge" button combines selected fragment into the active person's track
- "Show All Tracks" checkbox draws inactive tracks as dashed overlays on frame preview

**4. Crossing span annotation** (`identity_panel.py`)
- "Mark Crossing Start" / "Mark Crossing End" buttons
- Spans DataFrame with delete per row
- Persisted in `person_dir/crossing_spans.json`

**5. Apply spans during reprocess** (`multi_person_split.py`)
- Load manual spans, merge with auto-detected spans from `crossing_spans_from_overlap()`
- Pass to existing `OcclusionBridge.bridge_with_spans()` for SLERP bridging

### Workflow
1. Run pipeline → 2 active tracks, 3 inactive fragments
2. Open Inspector → "Show All Tracks" to see all 5 overlaid
3. Merge relevant fragments into active tracks
4. Mark hug span as crossing
5. Interpolate bboxes through the gap
6. Reprocess → clean output

### Pros
- Builds entirely on existing infrastructure
- User has full control over what gets merged and where crossings are
- No changes to tracker or detection pipeline
- Low risk of breaking existing workflows

### Cons
- Requires manual intervention for every difficult shot
- User must visually identify which fragments belong to which person
- Doesn't improve the tracker itself — same fragmentation next time

---

## Approach B: Smarter Tracker + Auto-Merge (Algorithmic)

### Concept
Improve OC-SORT parameters and add post-hoc track merging heuristics to reduce fragmentation automatically, with manual tools as fallback.

### Changes

**1. Tune OC-SORT parameters** (`person_tracker.py`)
- Increase `max_age` from 30 to 90 frames (hold IDs longer through occlusion)
- Increase `delta_t` from 3 to 10 (longer velocity estimation window)
- Lower `min_hits` from 3 to 2 (faster track initialization)
- These help with orbiting camera where people leave/re-enter frame

**2. Post-hoc fragment merging** (`person_tracker.py`)
- After `detect_and_track_all()`, analyze all tracks for merge candidates:
  - Two tracks that never overlap temporally (or overlap < 10%)
  - Whose bbox positions at boundary frames are close (IoU > 0.3)
  - Auto-merge them with the same `merge_tracks()` function
- Add `auto_merge_fragments(tracks, iou_threshold=0.3, max_temporal_overlap=0.1)` function

**3. Max-persons cap operates on merged tracks**
- Run auto-merge before the cap, so merged tracks are larger and more likely to survive

**4. Manual tools as fallback** (same as Approach A steps 3-5)
- Keep the merge UI and crossing span annotation for cases auto-merge can't handle

### Workflow
1. Run pipeline → tracker auto-merges fragments → 2-3 tracks instead of 5
2. Cap to 2 → likely gets complete tracks
3. If still fragmented, use manual merge tools
4. Mark crossing spans if needed
5. Reprocess

### Pros
- Reduces manual work for most shots
- Better default behavior without user intervention
- Tuned OC-SORT params help across all future videos

### Cons
- Auto-merge heuristics can make wrong decisions (merge two different people)
- OC-SORT parameter changes may affect other video types (static camera, many people)
- More complex to test and validate
- Still needs manual fallback for hard cases

---

## Approach C: Hybrid — Auto-Merge with Confirmation UI

### Concept
Run auto-merge as in Approach B, but present merge candidates to the user for confirmation rather than applying them silently. Combines the automation of B with the safety of A.

### Changes

**1. Auto-merge candidate detection** (same as Approach B step 2)
- Detect merge candidates but don't apply them
- Store as `merge_suggestions: list[tuple[track_a_id, track_b_id, reason]]`

**2. Merge confirmation UI** (`identity_panel.py`)
- Before the main Inspector, show a "Track Merge Suggestions" panel
- Each suggestion shows: "Track 0 (frames 0-250) ↔ Track 4 (frames 400-899) — likely same person (IoU 0.45 at boundary)"
- Accept / Reject buttons per suggestion
- Accepted merges applied before pipeline continues

**3. Same crossing span and manual merge tools as Approach A**
- For anything the auto-detection missed

**4. OC-SORT tuning** (same as Approach B step 1)
- Conservative parameter changes that reduce fragmentation without breaking existing behavior

### Workflow
1. Run pipeline → auto-detect merge candidates → show confirmation UI
2. User accepts/rejects merges → merged tracks used for pipeline
3. If needed, manual merge for missed fragments
4. Mark crossing spans
5. Reprocess

### Pros
- Best of both worlds: automation catches obvious merges, user validates
- Safe — no silent wrong merges
- Scales: auto-detection handles easy cases, user handles hard ones

### Cons
- Most complex to implement (auto-detection + confirmation UI + manual fallback)
- Extra UI step before pipeline runs
- Auto-detection heuristics still needed (same complexity as B)

---

## Key Files

| File | Role |
|------|------|
| `person_tracker.py` | Track detection, `merge_tracks()`, auto-merge heuristics |
| `multi_person_split.py` | Pipeline orchestration, `MultiPersonResult`, `reprocess_person()`, crossing span application |
| `identity_panel.py` | All UI — merge dropdown, Show All Tracks, crossing spans, confirmation panel |
| `identity_bridge.py` | `bridge_with_spans()` — already handles SLERP bridging (no changes needed) |
| `identity_tracking.py` | `IdentityTrack` — add `crossing_spans` field for persistence |

## Reusable Existing Code

- `PersonTracker._process_track()` (`person_tracker.py:237`) — gap interpolation + smoothing
- `OcclusionBridge.bridge_with_spans()` (`identity_bridge.py`) — SLERP through marked spans
- `crossing_spans_from_overlap()` (`identity_bridge.py`) — auto-detect overlap-based crossings
- `interpolate_bbox_corrections()` (`multi_person_split.py`) — linear delta interpolation
- `_render_frame_with_bboxes()` (`identity_panel.py`) — bbox overlay rendering to extend
