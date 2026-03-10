# Multi-Person + Multi-Camera Mocap: Combined Design Spec

## Problem

Multi-person and multi-camera are individually useful, but their real power is in combination. A typical production mocap scenario: 2-4 actors performing a scene, captured from 2-3 angles. Each feature solves different problems:

- **Multi-person** solves: inter-person occlusion, isolating individual performances
- **Multi-camera** solves: self-occlusion, depth ambiguity, limited viewpoints

Combined, they address the full range of occlusion scenarios that make single-camera single-person mocap unreliable for production use.

**Goal**: Capture N people from M cameras, producing one fused SMPL-X motion sequence per person, with all people positioned correctly in a shared world coordinate system.

---

## Combined Pipeline

```
M Camera Videos (each showing N people)
    │
    ▼
┌─────────────────────────────┐
│ 1. Calibrate Camera Rig     │  Checkerboard → K per camera + extrinsics (R, t)
│    (once per session)        │  Audio sync → frame offsets
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 2. Sync + Trim Videos       │  Align all cameras to common timeline
│    (per camera)              │  Resample to common FPS
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. Detect All People        │  YOLO tracking on each camera independently
│    (per camera)              │  → track IDs + bboxes per camera
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. Cross-Camera Person      │  Match person identities across cameras
│    Identity Matching         │  Person "Alice" in cam1 = person "Alice" in cam2
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 5. Isolate Per-Person       │  For each camera × each person:
│    Videos                    │  Segment (SAM2) + Inpaint (ProPainter)
│    (per camera × person)     │  OR crop-only if people don't overlap
└──────────────┬──────────────┘
               │ M × N clean single-person videos
               ▼
┌─────────────────────────────┐
│ 6. Run Single-Person        │  Existing pipeline: GVHMR + hands + face
│    Pipeline                  │  Unchanged code, per video
│    (per camera × person)     │
└──────────────┬──────────────┘
               │ M × N sets of SMPL-X params (each in own camera's world frame)
               ▼
┌─────────────────────────────┐
│ 7. Transform to Shared      │  Use calibration extrinsics to bring
│    World Frame               │  all cameras' results into reference frame
│    (per camera × person)     │
└──────────────┬──────────────┘
               │ M × N aligned SMPL-X params (all in shared frame)
               ▼
┌─────────────────────────────┐
│ 8. Multi-View Fusion        │  Per person: fuse M camera results
│    (per person)              │  → 1 fused SMPL-X sequence per person
└──────────────┬──────────────┘
               │ N fused SMPL-X sequences
               ▼
┌─────────────────────────────┐
│ 9. Scene Assembly            │  All N people in shared world frame
│                              │  Export: per-person BVH/FBX + scene FBX
└─────────────────────────────┘
```

---

## Step 4: Cross-Camera Person Identity Matching (New Problem)

In single-camera multi-person, YOLO assigns track IDs within that camera. But `person_0` in camera 1 might be `person_1` in camera 2 (different viewing angles → different area rankings). We need to match identities across cameras.

### Method A: Geometric Matching (Recommended)

Use calibrated camera geometry to project each person's 3D position and match by proximity:

1. For each camera, take the person's bbox center at frame 0
2. Project to a 3D ray using the camera's intrinsics K
3. For each pair of cameras, intersect rays (or find closest approach) using extrinsics
4. Match people by spatial proximity of their 3D positions

```python
def match_people_across_cameras(
    per_camera_tracks: dict[int, list[PersonTrack]],
    camera_rig: CameraRig,
) -> dict[str, dict[int, int]]:
    """Match person identities across cameras.

    Returns: {"person_A": {cam0: track_3, cam1: track_1, cam2: track_0}, ...}
    """
    # For each camera pair, triangulate person positions from bbox centers
    # Use Hungarian algorithm to find optimal assignment minimizing 3D distance
    # Propagate assignments transitively across all cameras
```

This works because:
- Calibration gives us exact camera geometry
- People are spatially separated (if they overlap, multi-person isolation handles that)
- Bbox centers project to reasonable 3D estimates at the person's depth

### Method B: Appearance Matching (Fallback)

If geometric matching is ambiguous (e.g., two people very close together from one angle):
- Extract appearance features (torso color histogram, clothing texture) per person per camera
- Match by appearance similarity
- More complex, but handles degenerate geometric configurations

### Method C: Manual Assignment (Simplest)

User clicks on each person in each camera view and names them. Works for 2-3 people / 2-3 cameras. Reliable but tedious.

**Recommendation**: Method A with Method C as override. Most multi-person multi-camera setups have clear spatial separation that geometric matching handles easily. Provide a GUI to manually reassign if needed.

---

## Processing Order Optimization

The naive approach processes `M × N` videos independently. With 3 cameras × 3 people = 9 pipeline runs, each taking ~5 minutes, that's 45 minutes. We can optimize:

### Shared Camera Motion

Camera motion (SimpleVO/DPVO) is scene-level, not person-level. For each camera, compute it once on the original video:

```
Per camera (M total):
  1. SimpleVO on full original video → cam_angvel       [COMPUTE ONCE]
  2. Per person (N total):
     a. Isolate person video (inpaint/crop)              [PER PERSON]
     b. YOLO + ViTPose + ViT features on isolated video  [PER PERSON]
     c. GVHMR with shared cam_angvel + per-person data    [PER PERSON]
```

This requires `demo.py` to accept pre-computed SLAM results (the `--slam-override` flag from the multi-person spec). Saves re-running SimpleVO N times per camera.

### Parallel Per-Camera Processing

Cameras are independent until fusion. If VRAM allows, process 2 cameras in parallel:
- Camera 1 processing on GPU (8-10 GB)
- Camera 2 processing on GPU (8-10 GB)
- 96 GB VRAM can easily handle 2-3 concurrent pipelines if models are shared

### Pipeline: Batch SAM2 Across Cameras

SAM2 video segmentation can process multiple cameras' worth of masks in sequence while the model is loaded, rather than loading/unloading between pipeline stages:

```
Load SAM2 model once
  → Segment all people in camera 0
  → Segment all people in camera 1
  → Segment all people in camera 2
Unload SAM2

Load ProPainter once
  → Inpaint all person-videos for camera 0
  → Inpaint all person-videos for camera 1
  → ...
Unload ProPainter

Load GVHMR + ViTPose + ViT once
  → Process all person-videos across all cameras
Unload
```

This minimizes model loading overhead and VRAM churn.

---

## Scene Assembly

After fusion, we have N people each with one fused SMPL-X sequence, all in the shared world frame.

### Relative Positioning

People's positions in the shared world frame should match their actual spatial arrangement. This happens naturally because:
1. GVHMR estimates world-space translation per person
2. Multi-camera fusion averages translations across views
3. Calibration ensures all results are in the same coordinate system

The main source of error: GVHMR sets each person's translation origin at their starting position. The absolute XZ offset between people comes from the camera geometry + bbox projection. This should be accurate to within ~10-20cm for typical capture setups.

### Interaction Verification

For scenes with physical interaction (handshake, dance), verify spatial consistency:
- Check that hand-to-hand distance at contact moments is physically plausible (< 5cm)
- Flag frames where people overlap (interpenetration) — indicates alignment error
- Optionally, add a contact optimization pass that adjusts relative positions to satisfy contact constraints

### Export Formats

**Per-person** (same as single-person pipeline):
- `person_alice_body_hands.bvh` — individual animation
- `person_alice_body_hands.fbx` — individual FBX
- `person_alice_arkit_blendshapes.csv` — facial animation

**Scene-level** (new):
- `scene.fbx` — single FBX with multiple armatures, correctly positioned
  - Blender: import normally, each person is a separate Armature object
  - UE5: import as Level Sequence with multiple Skeletal Meshes
- `scene_manifest.json` — metadata linking person IDs to output files:
  ```json
  {
    "people": [
      {
        "id": "alice",
        "cameras_used": [0, 1, 2],
        "bvh": "person_alice_body_hands.bvh",
        "fbx": "person_alice_body_hands.fbx",
        "face_csv": "person_alice_arkit_blendshapes.csv",
        "fused_smplx": "person_alice_fused_smplx.pt"
      },
      {
        "id": "bob",
        ...
      }
    ],
    "cameras": [
      {"id": 0, "video": "cam0_original.mp4"},
      ...
    ],
    "calibration": "calibration/camera_rig.json",
    "world_frame": "camera_0_gravity_aligned",
    "fps": 30
  }
  ```

---

## Output Directory Structure

```
outputs/multicam_multiperson/<session_name>/
  calibration/
    camera_rig.json
    sync_offsets.json
  camera_0/
    original_synced.mp4
    all_tracks.pt                          # YOLO tracking (all people)
    slam.pt                                # SimpleVO camera motion (computed once)
    person_alice/
      isolated_video.mp4                   # Inpainted or cropped
      demo/                                # GVHMR outputs
      hybrid_smplx.pt                      # Camera-0 SMPL-X for Alice
      body_hands.bvh
    person_bob/
      ...
  camera_1/
    ...
  identity_matching/
    cross_camera_assignments.json          # Person matching across cameras
    matching_visualization.png             # Shows matched identities
  fused/
    person_alice/
      aligned_cam0.pt                      # Cam 0 result in shared frame
      aligned_cam1.pt                      # Cam 1 result in shared frame
      fused_smplx.pt                       # Multi-view fused
      fused_body_hands.bvh
      fused_body_hands.fbx
      fused_arkit_blendshapes.csv          # Best-camera face (or averaged)
      confidence_per_camera.png
    person_bob/
      ...
  scene/
    scene.fbx                              # All people, one FBX
    scene_manifest.json
    scene_preview.mp4                      # All skeletons overlaid on ref camera
    interaction_report.json                # Contact/overlap analysis
```

---

## Face Capture in Multi-Camera

Face blendshapes are extracted from the person's face crop. In multi-camera mode:

**Best-camera selection** (recommended for face):
- For each frame, pick the camera where the face is most frontal and largest
- Frontal-ness: compute angle between face normal (from MediaPipe landmarks) and camera viewing direction
- The face pipeline produces 52 blendshapes — these don't fuse well via averaging because blendshape values are nonlinear (e.g., averaging half-open and closed eye doesn't give correct intermediate)

**Per-camera face confidence**:
```python
def face_confidence(face_landmarks, camera_direction):
    """Higher confidence when face is more frontal to camera."""
    nose = face_landmarks[1]       # nose tip
    forehead = face_landmarks[10]  # forehead
    chin = face_landmarks[152]     # chin
    face_normal = cross(forehead - chin, left_ear - right_ear)
    frontality = dot(normalize(face_normal), camera_direction)
    return max(0, frontality)  # 1.0 = perfectly frontal, 0 = profile
```

---

## Failure Modes & Mitigations

| Scenario | What Breaks | Mitigation |
|----------|------------|------------|
| Person visible in only 1 camera | Fusion falls back to single camera | Confidence weighting gives 100% to the visible camera. No degradation vs. single-cam. |
| Person identity swap across cameras | Wrong person's motions get fused together | Geometric matching + manual override UI. Alert user if confidence is low. |
| Calibration drift (camera bumped) | World frame misalignment | Detect via inconsistency between cameras (fused translation has high variance). Re-calibrate. |
| Different lighting per camera | Different ViTPose / GVHMR quality | Usually not significant. GVHMR is robust to exposure/lighting. |
| Person enters/leaves frame at different times per camera | Temporal gaps in some cameras | Per-frame confidence handles this — absent frames get zero weight. |
| Tight interaction (embrace, fight) | Inpainting fails, fusion disagrees | Degrade gracefully: use highest-confidence single camera, warn user. |
| Sync off by 1+ frames | Body parts jitter after fusion | Audio sync should prevent this. Add post-fusion temporal smoothing as safety net. |

---

## Implementation Strategy

### Build Order

The features share infrastructure but can be developed independently:

```
              Multi-Person (worktree A)
             /
main ───────
             \
              Multi-Camera (worktree B)

Then merge both → Combined (worktree C or main)
```

### Phase 1: Multi-Person (independent)
From `spec/multi_person_inpainting.md` + plan:
1. `get_all_tracks()` in tracker
2. `multi_person.py` — crop mode first, then inpaint mode
3. Per-person pipeline execution
4. Output organization + GUI

### Phase 2: Multi-Camera (independent)
From `spec/multi_camera.md`:
1. Camera calibration + audio sync
2. Per-camera independent processing
3. World frame alignment
4. Parameter fusion + export

### Phase 3: Combined
From this spec:
1. Cross-camera person identity matching
2. Processing order optimization (shared camera motion, batched models)
3. Per-person multi-view fusion
4. Scene assembly + combined export
5. Face best-camera selection

### Phase 4: Quality Polish
- Interaction verification (interpenetration detection)
- Contact optimization
- Temporal smoothing on fused results
- Comprehensive visualization

---

## Dependencies Summary

All features share the same dependency set:

| Dependency | Used By | VRAM | Notes |
|------------|---------|------|-------|
| SAM2 | Multi-person segmentation | ~4-8 GB | `segment-anything-2`, facebookresearch |
| ProPainter | Multi-person inpainting | ~8-12 GB | `sczhou/ProPainter`, ICCV 2023 |
| OpenCV | Camera calibration | CPU | Already in environment |
| SciPy | Audio sync (cross-correlation) | CPU | Already in environment |
| GVHMR | Core pipeline | ~8-10 GB | Already in environment |
| SMPLest-X | Hand capture | ~6-8 GB | Separate conda env |
| MediaPipe | Face capture | ~1 GB | Already in environment |

Peak VRAM usage: ~12 GB (one model at a time). 96 GB Blackwell is massively over-provisioned. Could run 2-3 cameras in parallel if desired.

---

## What This Does NOT Solve

- **Real-time multi-person multi-camera**: This is batch offline processing. Real-time would require streaming inference architecture.
- **More than ~5 people**: Inpainting cost scales quadratically (N people × (N-1) inpaint passes per camera). Practical limit ~5 people.
- **More than ~4 cameras**: Diminishing returns on fusion quality. Calibration complexity grows. Designed for 2-4.
- **Moving cameras**: Calibration assumes static cameras. A camera operator walking with a camera would need per-frame extrinsic estimation — a research problem.
- **Identical clothing / twins**: Person identity matching relies on spatial separation. Identical appearance + close proximity = ambiguous matching.
- **Full-body contact (wrestling, grappling)**: Segmentation can't reliably separate tightly intertwined bodies. Degrade to best single-view estimate.
