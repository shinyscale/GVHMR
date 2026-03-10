# Multi-Person Mocap via Segmentation + Inpainting: Design Spec

## Problem

The current pipeline (GVHMR + SMPLest-X) is single-person only. YOLO detects all people but `Tracker.get_one_track()` and SMPLest-X both select only the largest person. When two people overlap (e.g. arm around shoulder), the occluded limbs of the secondary person are lost, and the primary person's pose estimation is confused by the other person's visible limbs.

**Goal**: Use generative AI (segmentation + inpainting) to isolate each person into their own "clean" video, then process each through the existing single-person pipeline.

---

## Proposed Pipeline

```
Input Video (multi-person)
    │
    ▼
┌─────────────────────────┐
│ 1. YOLO + OC-SORT       │  Detect all people, track identities robustly
│    Detection & Tracking  │  Pose-aware association handles crossings
└──────────┬──────────────┘
           │ per-person bboxes + track IDs
           ▼
┌─────────────────────────┐
│ 2. Camera Motion (SLAM) │  Run SimpleVO/DPVO ONCE on original video
│    (shared across all)   │  Shared by all per-person pipeline runs
└──────────┬──────────────┘
           │ cam_angvel (saved to disk)
           ▼
┌─────────────────────────┐
│ 3. SAM2 Video           │  Segment each person with precise masks
│    Segmentation          │  Track identity across frames (memory module)
│                          │  Auto-prompted from YOLO bboxes
└──────────┬──────────────┘
           │ per-person binary masks (N_people × N_frames × H × W)
           ▼
┌─────────────────────────┐
│ 4. Overlap Check        │  Per-frame: do any masks overlap?
│    + Smart Isolation     │  NO overlap → crop only (fast path)
│                          │  YES overlap → inpaint others away
└──────────┬──────────────┘
           │ N clean single-person videos
           ▼
┌─────────────────────────┐
│ 5. Per-Person Pipeline   │  Run GVHMR + SMPLest-X + Face on each
│    (with shared SLAM)    │  Uses --slam-override for shared camera motion
└──────────┬──────────────┘
           │ N sets of SMPL-X params (each in own frame)
           ▼
┌─────────────────────────┐
│ 6. World-Space Assembly  │  Recover relative positioning from original
│                          │  YOLO detections + depth estimates
│                          │  Place all people in shared coordinate system
└─────────────────────────┘
```

---

## Step 1: Person Detection & Tracking

**Current code**: `hmr4d/utils/preproc/tracker.py`
- `Tracker.track()` returns full multi-person tracking history with YOLO track IDs
- `Tracker.get_one_track()` selects only the largest — this is the only bottleneck
- `sort_track_length()` already ranks all tracks by area + duration

**Changes needed**:
1. Expose `track()` results (all people) via `get_all_tracks()` instead of only `get_one_track()`
2. Replace raw YOLO tracking with **OC-SORT** for more robust identity persistence

### Why OC-SORT over raw YOLO tracking

YOLO's built-in tracker assigns IDs based on IoU overlap between frames. This fails when:
- Two people cross paths (IDs swap)
- A person is briefly occluded (ID lost, new ID assigned on reappearance)
- Fast motion causes large bbox displacement between frames

[OC-SORT](https://github.com/noahcao/OC_SORT) (CVPR 2023) adds:
- **Observation-centric re-identification**: recovers lost tracks using motion model + appearance
- **Pose-aware association**: can incorporate ViTPose keypoint similarity (not just bbox IoU)
- **Virtual trajectory**: maintains predicted trajectory during occlusion for seamless re-association

OC-SORT is a drop-in replacement — same input (per-frame detections) and output (track IDs + bboxes) as the current tracker.

**Alternative**: ByteTrack (simpler, also effective). OC-SORT is preferred for the occlusion recovery.

---

## Step 2: Shared Camera Motion Estimation

**Critical**: Camera motion (SimpleVO/DPVO) estimates ego-motion from the full scene. It must run on the **original unmodified video**, not on inpainted per-person videos where scene content has changed.

```python
def compute_shared_slam(video_path: str, output_path: str):
    """Run SLAM once on the original video. Save results for all per-person runs."""
    # Existing SLAM code from preprocess.py
    cam_angvel = run_dpvo(video_path)  # or SimpleVO
    torch.save(cam_angvel, output_path)
```

**Changes to existing pipeline**:
- Add `--slam-override <path>` flag to `demo.py` that loads pre-computed camera motion instead of running SLAM
- When `--slam-override` is provided, skip the SLAM step entirely and load from disk
- This saves N-1 redundant SLAM runs (one per extra person)

**Why this matters**: If SLAM runs on an inpainted video where a person has been removed, the inpainted pixels (which may shimmer or have temporal artifacts) could confuse the visual odometry. The original video gives the cleanest camera motion estimate.

---

## Step 3: SAM2 Video Segmentation

**Model**: [SAM2](https://github.com/facebookresearch/sam2) (Meta, 2024)
- Extends Segment Anything to video with a per-session memory module
- Tracks objects across frames through occlusions via temporal attention
- 30-44 FPS depending on model size (base+: 80.8M params, 35 FPS)
- **VRAM**: ~4-8 GB depending on resolution and model size

**Prompting strategy**: Use YOLO/OC-SORT bboxes as initial prompts for SAM2.
- Frame 0: provide each person's bbox → SAM2 generates precise mask
- Subsequent frames: SAM2 propagates masks automatically via memory
- Re-prompt on keyframes if tracking drifts (every ~30 frames, or when OC-SORT confidence drops)

**Output**: Per-person binary mask sequence `(N_frames × H × W)` per tracked person, saved as compressed numpy arrays or PNG sequences.

**Installation**: Direct Python package — `pip install segment-anything-2`. No ComfyUI needed.

---

## Step 4: Smart Isolation (Crop vs Inpaint)

Not all frames require inpainting. When people don't overlap, a simple crop is faster and produces cleaner input for the pose estimator.

### Per-Frame Decision Logic

```python
def should_inpaint(masks: dict[int, np.ndarray], target_person: int, frame: int) -> bool:
    """Check if any other person's mask overlaps the target person's bbox region."""
    target_mask = masks[target_person][frame]
    target_bbox = mask_to_bbox(target_mask, padding=20)  # slight padding

    for person_id, person_masks in masks.items():
        if person_id == target_person:
            continue
        other_mask = person_masks[frame]
        # Check if the other person's mask falls within target's bbox region
        other_in_bbox = other_mask[target_bbox.y1:target_bbox.y2, target_bbox.x1:target_bbox.x2]
        if other_in_bbox.any():
            return True
    return False
```

### Two isolation modes:

**Crop mode** (no overlap): Extract the target person's bbox region from the original video. This is what the existing single-person pipeline already does — fast, no quality loss.

**Inpaint mode** (overlap detected): Mask out all other people using their SAM2 masks, inpaint with ProPainter to fill the gaps, then crop the target person's region.

### Hybrid output

For a typical two-person scene, many frames will use crop mode. Inpainting only activates for frames where people are close enough to interfere. The output video seamlessly blends both modes since the target person's pixels are untouched in both cases.

---

## Step 5: Video Inpainting (When Needed)

### ProPainter (Recommended)

[ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023)
- **Approach**: Recurrent flow completion + dual-domain propagation + sparse video transformer
- **Temporal consistency**: Propagates through both image and feature domains — prevents drift
- **Speed**: Fast (non-diffusion). Processes video in chunks
- **VRAM**: ~8-12 GB for 720p
- **Resolution**: Process at 720p for speed. ViTPose/GVHMR don't need 4K input.
- **Best for**: Removing people where the background is fairly static

### DiffuEraser (Quality Upgrade Path)

[DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) (2025)
- **Approach**: Diffusion UNet + BrushNet branch + AnimateDiff-style temporal attention
- **Temporal consistency**: Superior to ProPainter
- **Speed**: Medium-slow (diffusion inference per chunk)
- **VRAM**: ~16-24 GB for 720p
- **Best for**: Complex scenes, moving backgrounds

### Recommendation

**ProPainter** for the initial implementation — it's fast, proven, and the quality is good enough for the purpose (we just need clean-enough input for ViTPose/GVHMR, not pixel-perfect reconstruction). DiffuEraser as upgrade path if ProPainter's quality isn't sufficient for edge cases.

---

## Step 6: World-Space Assembly

Each person is processed independently through GVHMR, which sets translation origin at each person's starting position. Without correction, all people would appear at the same world-space origin.

### Recovering Relative Positioning

Use the original YOLO detections to compute each person's offset in the scene:

```python
def compute_person_offsets(
    all_tracks: dict[int, list[BBox]],
    camera_K: np.ndarray,
    reference_frame: int = 0,
) -> dict[int, np.ndarray]:
    """Compute XZ world-space offsets for each person relative to person 0.

    Uses bbox bottom-center (feet position) projected through camera intrinsics
    to estimate relative ground-plane positions.
    """
    offsets = {}
    ref_person = list(all_tracks.keys())[0]
    ref_feet = bbox_bottom_center(all_tracks[ref_person][reference_frame])

    for person_id, track in all_tracks.items():
        feet = bbox_bottom_center(track[reference_frame])
        # Horizontal offset from pixel displacement + assumed ground plane
        dx_pixels = feet[0] - ref_feet[0]
        # Convert pixel offset to meters using focal length and estimated depth
        # Depth estimated from bbox height (taller bbox = closer)
        depth = estimate_depth_from_bbox_height(track[reference_frame], camera_K)
        dx_meters = dx_pixels * depth / camera_K[0, 0]
        offsets[person_id] = np.array([dx_meters, 0.0, 0.0])

    return offsets
```

**Limitations**: Monocular depth estimation from bbox height is approximate (~20-30cm accuracy). For most use cases (placing people in correct relative positions for a scene), this is sufficient. Multi-camera capture (future) eliminates this limitation via triangulation.

### Applying Offsets

After GVHMR produces per-person SMPL-X params:
```python
for person_id, smplx_params in per_person_results.items():
    smplx_params["transl"] += person_offsets[person_id]
```

---

## The Arm-Around-Shoulder Problem (Specific)

This is the hardest case because:
1. Person A's arm overlaps Person B's torso
2. When we mask out Person B, we also lose part of Person A's arm
3. The inpainter needs to hallucinate the rest of Person A's arm

**Why it works anyway**:
- The inpainter sees Person A's shoulder, upper arm, and where the arm disappears behind the mask. Flow-based and diffusion models understand human anatomy and will extend the arm plausibly
- The result doesn't need to be *photographically correct* — it needs to be good enough for ViTPose to detect the arm's approximate position
- ViTPose is already robust to partial occlusion; even a rough inpainted arm gives it enough signal

**When it might struggle**:
- If the arm is almost entirely hidden (elbow to hand all behind the other person)
- If both people are wearing similar clothing (mask boundary ambiguity)
- Fast motion during heavy overlap (SAM2 mask tracking may drift)

---

## Integration: Python Preprocessing Script

Run SAM2 + ProPainter directly in Python as a preprocessing stage. No ComfyUI dependency.

```python
# New file: multi_person_split.py

def split_multi_person_video(
    video_path: str,
    output_dir: str,
    min_track_duration: int = 30,  # minimum frames to consider a person
) -> MultiPersonResult:
    """Full multi-person isolation pipeline.

    Returns:
        MultiPersonResult with per-person video paths, masks, offsets,
        and shared SLAM path.
    """
    # 1. Detect & track all people (OC-SORT)
    tracks = detect_and_track_all(video_path)
    tracks = filter_short_tracks(tracks, min_track_duration)

    # 2. Compute shared camera motion (once)
    slam_path = output_dir / "shared_slam.pt"
    compute_shared_slam(video_path, slam_path)

    # 3. Segment all people (SAM2)
    masks = sam2_segment_all(video_path, tracks)

    # 4. Per-person: smart isolate (crop or inpaint)
    person_videos = []
    for person_id in tracks:
        needs_inpaint = compute_overlap_frames(masks, person_id)
        if any(needs_inpaint):
            other_masks = combine_masks_except(masks, person_id)
            video = inpaint_and_crop(video_path, other_masks, tracks[person_id], needs_inpaint)
        else:
            video = crop_only(video_path, tracks[person_id])
        person_videos.append(video)

    # 5. Compute relative positioning offsets
    offsets = compute_person_offsets(tracks, camera_K)

    return MultiPersonResult(
        person_videos=person_videos,
        slam_path=slam_path,
        offsets=offsets,
        masks=masks,
        tracks=tracks,
    )
```

### GUI Integration

Add a "Multi-Person" checkbox/toggle in `gvhmr_gui.py`:
- When enabled, runs `split_multi_person_video()` as preprocessing
- Shows detected people with track IDs for user verification
- Processes each person through the existing pipeline with `--slam-override`
- Assembles results with computed offsets
- Displays all people's skeletons in the 3D viewer

---

## Output Structure

```
outputs/multi_person/<session_name>/
  original_video.mp4                  # Input video (symlink or copy)
  shared_slam.pt                      # Camera motion from original video
  detection/
    all_tracks.pt                     # OC-SORT tracking results (all people)
    track_visualization.mp4           # Video with bbox overlays + IDs
  masks/
    person_0_masks.npz                # SAM2 masks (compressed)
    person_1_masks.npz
    overlap_map.npz                   # Per-frame overlap flags
  person_0/
    isolated_video.mp4                # Crop or inpainted video
    isolation_mode.json               # Per-frame: "crop" or "inpaint"
    demo/                             # Full GVHMR output
    <stem>_hybrid_smplx.pt            # SMPL-X params
    <stem>_body_hands.bvh             # BVH animation
    <stem>_body_hands.fbx             # FBX animation
    <stem>_arkit_blendshapes.csv      # Face capture (if applicable)
  person_1/
    ...
  assembly/
    person_offsets.json               # Relative XZ positions
    assembled_smplx.pt                # All people in shared world frame
    scene_preview.mp4                 # All skeletons overlaid on original video
  session_manifest.json               # Metadata: people count, processing modes, paths
```

---

## New Files

| File | Purpose |
|------|---------|
| `multi_person_split.py` | Orchestration: detection, segmentation, isolation, offset computation |
| `sam2_segmenter.py` | SAM2 wrapper: bbox-prompted segmentation, mask propagation, re-prompting |
| `propainter_inpaint.py` | ProPainter wrapper: mask-based video inpainting with chunked processing |
| `person_tracker.py` | OC-SORT wrapper: robust multi-person tracking with pose-aware association |
| `world_assembly.py` | Offset computation, per-person SMPL-X transform, scene-level export |

## Changes to Existing Files

| File | Change | Why |
|------|--------|-----|
| `hmr4d/utils/preproc/tracker.py` | Add `get_all_tracks()` method | Expose all tracked people, not just largest |
| `tools/demo/demo.py` | Add `--slam-override <path>` flag | Load pre-computed camera motion instead of re-running SLAM |
| `gvhmr_gui.py` | Add "Multi-Person" toggle + per-person result display | GUI integration |

---

## Dependencies

| Package | Purpose | VRAM | Install |
|---------|---------|------|---------|
| `segment-anything-2` | Person segmentation | ~4-8 GB | `pip install segment-anything-2` |
| `ProPainter` | Video inpainting | ~8-12 GB | Clone repo, install deps |
| `ocsort` | Robust multi-person tracking | CPU | `pip install ocsort` or clone repo |

**Peak VRAM**: ~12 GB (one model loaded at a time). Models are loaded sequentially: OC-SORT (CPU) → SAM2 (GPU) → ProPainter (GPU) → GVHMR (GPU). No concurrent GPU models needed.

---

## Implementation Phases

### Phase 1: Detection & Tracking
- Integrate OC-SORT or expose `get_all_tracks()` with existing YOLO tracker
- Track visualization overlay
- **Deliverable**: Can detect and persistently track N people across a video

### Phase 2: Segmentation
- SAM2 integration with bbox prompting from tracks
- Mask quality verification (overlay on original video)
- Per-frame overlap computation
- **Deliverable**: Clean per-person masks for a multi-person video

### Phase 3: Isolation (Crop + Inpaint)
- Crop-only fast path for non-overlapping frames
- ProPainter integration for overlapping frames
- Hybrid video assembly (seamless blend of crop and inpaint frames)
- **Deliverable**: Clean single-person videos suitable for GVHMR input

### Phase 4: Pipeline Integration
- `--slam-override` flag in demo.py
- Shared SLAM computation
- Per-person pipeline execution
- World-space assembly with offset recovery
- Scene preview visualization
- **Deliverable**: End-to-end multi-person mocap from a single video

### Phase 5: GUI Integration
- Multi-Person toggle in gvhmr_gui.py
- Person selection / track review UI
- Per-person + assembled result display
- **Deliverable**: One-click multi-person capture in the GUI

---

## Tradeoffs & Challenges

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Temporal consistency of inpainting | HIGH | Use ProPainter (flow-based) — NOT per-frame SD |
| Identity persistence during crossings | HIGH | OC-SORT with pose-aware association; SAM2 memory module |
| SLAM quality on inpainted video | HIGH | Run SLAM once on original video, share via --slam-override |
| Relative positioning accuracy | MEDIUM | Bbox-based depth estimation (~20-30cm). Acceptable for most scenes. |
| Anatomical plausibility of hallucinated limbs | MEDIUM | ViTPose is forgiving of imperfect inpainting input |
| Processing time (inpainting is slow) | MEDIUM | Crop-only fast path skips inpainting for non-overlapping frames |
| SAM2 mask quality at occlusion boundaries | MEDIUM | Re-prompt from OC-SORT every 30 frames. Dilate masks slightly |
| VRAM requirements | LOW | Sequential model loading. Peak ~12 GB. Fits on any modern GPU |

---

## What This Does NOT Solve

- **Identical twins / similar appearance**: SAM2 + OC-SORT may swap identities
- **Fully hidden person**: If someone is completely behind another, no signal to segment
- **Very tight embraces**: Where body boundaries are ambiguous even to humans
- **Real-time processing**: This is a batch offline workflow
- **Precise depth ordering**: Monocular depth from bbox height is approximate

---

## Recommended First Experiment

Before building the full pipeline:
1. Record a test video with two people (side-by-side + arm-around-shoulder moments)
2. Run SAM2 in Python to segment both people → verify mask quality
3. Run ProPainter on the masked video → verify inpainting quality
4. Feed the inpainted single-person video through the existing GVHMR pipeline
5. Compare pose quality vs. running the original multi-person video (where GVHMR picks one person)
6. Verify that ViTPose detects the inpainted arm plausibly

This validates the core hypothesis end-to-end before building the integration.
