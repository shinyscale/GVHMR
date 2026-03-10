# Multi-Camera Mocap Capture: Design Spec

## Problem

The current pipeline uses a single monocular camera. GVHMR estimates camera motion (SimpleVO/DPVO) and rolls out the person's local translation velocity into a world-space trajectory. This works remarkably well for one camera, but has inherent limitations:

- **Depth ambiguity**: Monocular 2D→3D lifting has fundamental scale/depth uncertainty
- **Self-occlusion**: Limbs hidden from the single viewpoint produce noisy or frozen poses
- **Camera motion estimation errors**: SimpleVO/DPVO drift accumulates over long sequences
- **Limited capture volume**: Person must stay in frame of one camera

**Goal**: Support synchronized multi-camera capture where each camera's results are fused into a single, more accurate SMPL-X motion sequence. No changes to the GVHMR model itself — fusion happens at the output level.

---

## Architecture: Independent Processing + Calibrated Fusion

```
Camera 1 Video ──┐
Camera 2 Video ──┼── [Calibrate] ── Shared World Frame Definition
Camera N Video ──┘         │
      │                    │
      ▼ (per camera)       │
  Full Single-Camera       │
  Pipeline (unchanged)     │
      │                    │
      ▼                    ▼
  Per-Camera           Transform Each
  SMPL-X Params  ───►  to Shared Frame  ───►  Multi-View Fusion  ───►  BVH/FBX
```

The key insight: GVHMR already produces world-space SMPL-X params per camera. We just need to:
1. Know how the cameras relate to each other (calibration)
2. Transform each camera's world frame into a shared frame
3. Fuse the aligned results

---

## The World Frame Problem (Critical)

This is the core technical challenge. Each camera's GVHMR output lives in a **different world frame** because:

1. SimpleVO/DPVO sets frame-0 camera pose as identity independently per camera
2. GVHMR's `get_smpl_params_w_Rt_v2()` (gvhmr_pipeline.py:322) rolls out trajectory relative to this per-camera origin
3. The `any->ay` gravity alignment (line 381) further rotates each camera's output independently

**Concretely, in `get_smpl_params_w_Rt_v2()`:**
```python
# Camera angular velocity → cumulative rotation from frame 0
R_t_to_tp1 = rotation_6d_to_matrix(cam_angvel)  # per-frame rotation
R_t_to_0 = cumulative_product(R_t_to_tp1)       # accumulated to frame 0

# World orientation = camera trajectory × body orientation
global_orient = R_t_to_0 @ R_gv

# World translation = rollout of local velocity through global orientation
transl = rollout_local_transl_vel(local_transl_vel, global_orient)

# Gravity alignment — rotates so Y is truly vertical
global_orient, transl, _ = get_tgtcoord_rootparam(global_orient, transl, tsf="any->ay")
```

Each camera starts with `R_t_to_0[frame_0] = I`, meaning "camera 1's viewing direction at frame 0 = identity". Camera 2 points a different direction, so its identity means something different.

**Solution**: Given calibration extrinsics `(R_c1_to_c2, t_c1_to_c2)`:
- Camera 1's world frame W1 at frame 0: `R_W1_to_c1 = I` (SimpleVO convention)
- Camera 2's world frame W2 at frame 0: `R_W2_to_c2 = I`
- From calibration: `R_c2_to_c1` relates the physical cameras
- Therefore: `R_W2_to_W1 = R_c2_to_c1` (since W2=c2 at frame 0 and W1=c1 at frame 0)
- Apply: `global_orient_in_W1 = R_W2_to_W1 @ global_orient_in_W2`
- Apply: `transl_in_W1 = R_W2_to_W1 @ transl_in_W2 + t_c2_to_c1`

The gravity alignment step (`any->ay`) complicates this because it applies an additional rotation per camera that depends on the estimated gravity direction. After alignment, both cameras should have Y-up, but the yaw (rotation around Y) may differ. The yaw offset can be computed from the calibration extrinsics projected onto the horizontal plane.

---

## Step 1: Camera Calibration

### Method: Checkerboard (OpenCV)

Standard and reliable. Record a checkerboard visible to all cameras simultaneously.

```python
# Per camera: intrinsics
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size)

# Per camera pair: extrinsics
ret, K1, d1, K2, d2, R, t, E, F = cv2.stereoCalibrate(
    obj_points, img_points_1, img_points_2, K1, d1, K2, d2, img_size
)
# R, t: rotation and translation from camera 1 to camera 2
```

**Output**: `CameraRig` object containing:
- Per-camera: intrinsic matrix K (3x3), distortion coefficients
- Per-pair: extrinsic R (3x3) and t (3x1)
- All relative to a reference camera (camera 0)

**Stored as**: `camera_rig.json`
```json
{
  "reference_camera": 0,
  "cameras": [
    {
      "id": 0,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist": [k1, k2, p1, p2, k3],
      "R_to_ref": [[1,0,0],[0,1,0],[0,0,1]],
      "t_to_ref": [0, 0, 0]
    },
    {
      "id": 1,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "dist": [k1, k2, p1, p2, k3],
      "R_to_ref": [[...]],
      "t_to_ref": [tx, ty, tz]
    }
  ]
}
```

### Future: COLMAP Auto-Calibration

For users without a checkerboard. Use COLMAP's structure-from-motion to reconstruct camera poses from overlapping views of the capture space. More automated but less deterministic.

---

## Step 2: Temporal Synchronization

Multi-camera requires frame-level sync. Cameras may start recording at different times and may have slightly different frame rates.

### Method: Audio Cross-Correlation

1. Extract audio from each video (`ffmpeg -i video.mp4 -vn audio.wav`)
2. Cross-correlate waveforms to find time offset:
   ```python
   correlation = scipy.signal.correlate(audio_1, audio_2, mode='full')
   offset_samples = correlation.argmax() - len(audio_2) + 1
   offset_seconds = offset_samples / sample_rate
   offset_frames = offset_seconds * fps
   ```
3. A clap or sharp sound makes this very precise (sub-frame accuracy)

### Method: Visual Flash Detection (Fallback)

If audio is unavailable, detect a flash/clapper visible to all cameras. Find the frame with maximum brightness change per camera, compute offsets.

### Applying Sync

- Resample all cameras to a common FPS (use existing `preprocess.py` FPS resampling)
- Trim each camera's video to the shared time window
- Store offsets in `sync_offsets.json`:
  ```json
  {
    "reference_camera": 0,
    "common_fps": 30,
    "offsets_frames": [0, 3, -2],
    "trim_start_frame": [12, 15, 10],
    "trim_end_frame": [450, 453, 448]
  }
  ```

---

## Step 3: Per-Camera Processing

Run the existing single-camera pipeline on each camera's video independently. Two modifications needed:

### 3a. Use Calibrated Intrinsics

Currently, `demo.py:load_data_dict()` estimates K from frame dimensions:
```python
K_fullimg = estimate_K(width, height).repeat(length, 1, 1)  # Assumes 53° diagonal FOV
```

For calibrated cameras, use the actual K from calibration:
```python
# New --intrinsics flag:
K_fullimg = load_calibrated_K(intrinsics_path).repeat(length, 1, 1)
```

This is a small but important accuracy improvement — estimated focal length can be off by 10-20%.

### 3b. Processing

Each camera is processed as a completely independent run:
```bash
# Camera 0
python tools/demo/demo.py --video cam0_synced.mp4 --intrinsics rig/cam0_K.json --output_dir outputs/multicam/session/cam0/

# Camera 1
python tools/demo/demo.py --video cam1_synced.mp4 --intrinsics rig/cam1_K.json --output_dir outputs/multicam/session/cam1/
```

Followed by SMPLest-X/HaMeR hand merge and face capture per camera, same as single-camera.

---

## Step 4: World Frame Alignment

Transform each camera's GVHMR output into the reference camera's world frame.

```python
def transform_to_reference_frame(smplx_params, R_cam_to_ref, t_cam_to_ref):
    """Transform SMPL-X params from one camera's world frame to reference frame.

    After GVHMR + gravity alignment, each camera's output has:
    - Y axis = up (gravity aligned)
    - XZ plane = ground plane
    - But yaw (rotation around Y) differs per camera

    The calibration extrinsics tell us the full 3D relationship,
    but after gravity alignment we only need to apply the yaw component.
    """
    # Extract yaw-only rotation from full R_cam_to_ref
    # (gravity alignment already handled pitch/roll)
    yaw = extract_yaw_around_y(R_cam_to_ref)
    R_yaw = axis_angle_to_matrix(torch.tensor([0, yaw, 0]))

    # Rotate global orientation
    global_orient_mat = axis_angle_to_matrix(smplx_params["global_orient"])
    global_orient_aligned = matrix_to_axis_angle(R_yaw @ global_orient_mat)

    # Rotate and translate position
    transl_aligned = (R_yaw @ smplx_params["transl"].T).T + t_cam_to_ref_horizontal

    return {**smplx_params, "global_orient": global_orient_aligned, "transl": transl_aligned}
```

---

## Step 5: Multi-View Fusion

### Strategy A: Parameter-Space Weighted Average (Recommended Start)

For each frame, combine the aligned per-camera SMPL-X params:

```python
def fuse_params(per_camera_params, per_camera_confidence):
    """Weighted average of SMPL-X params across cameras.

    Confidence per camera derived from:
    - Mean ViTPose keypoint confidence for that frame
    - Whether the person was detected (vs interpolated bbox)
    """
    # Translations: weighted arithmetic mean
    weights = normalize(per_camera_confidence)  # (N_cameras,)
    fused_transl = sum(w * p["transl"] for w, p in zip(weights, per_camera_params))

    # Rotations: quaternion weighted average (SLERP for 2 cameras)
    fused_orient = quaternion_weighted_average(
        [p["global_orient"] for p in per_camera_params], weights
    )

    # Body pose: weighted average of axis-angle (works for small differences)
    fused_body = sum(w * p["body_pose"] for w, p in zip(weights, per_camera_params))

    # Hands: take from camera with highest wrist keypoint confidence
    best_left = argmax([conf_left_wrist for conf in per_camera_confidence])
    best_right = argmax([conf_right_wrist for conf in per_camera_confidence])
    fused_left_hand = per_camera_params[best_left]["left_hand_pose"]
    fused_right_hand = per_camera_params[best_right]["right_hand_pose"]

    return fused
```

### Strategy B: Per-Joint 3D Fusion (Quality Upgrade)

More principled — works at the joint level rather than parameter level:

1. Forward kinematics: convert each camera's SMPL-X params to 3D joint positions (52 joints)
2. Per joint, per frame: weight by visibility (ViTPose confidence for that joint in that camera)
3. Weighted average of 3D positions
4. Inverse kinematics to recover SMPL-X pose from fused 3D joints

This handles the case where joint A is well-visible in camera 1 but occluded in camera 2, while joint B is the opposite.

### Strategy C: Reprojection Optimization (Future)

Optimize SMPL-X params to minimize reprojection error across all cameras simultaneously:
```
argmin_{pose, transl} Σ_cameras ||project(FK(pose, transl), K_cam, R_cam, t_cam) - kp2d_cam||²
```

Most principled but requires differentiable SMPL-X FK and an optimizer loop. Significantly slower.

---

## Confidence Estimation

Each camera's per-frame confidence drives the fusion weights:

```python
def compute_per_camera_confidence(vitpose_path, bbx_path):
    """Compute per-frame confidence for a camera's GVHMR output."""
    kp2d = torch.load(vitpose_path)  # (F, 17, 3) — last dim is confidence
    bbx = torch.load(bbx_path)

    # Mean keypoint confidence (0-1)
    kp_conf = kp2d[:, :, 2].mean(dim=1)  # (F,)

    # Bbox area (larger = more pixels = more reliable)
    bbx_area = (bbx["bbx_xyxy"][:, 2] - bbx["bbx_xyxy"][:, 0]) * \
               (bbx["bbx_xyxy"][:, 3] - bbx["bbx_xyxy"][:, 1])
    area_conf = bbx_area / bbx_area.max()

    # Combined
    confidence = 0.7 * kp_conf + 0.3 * area_conf
    return confidence
```

---

## Output Structure

```
outputs/multicam/<session_name>/
  calibration/
    camera_rig.json              # Intrinsics K + extrinsics (R, t) per camera
    sync_offsets.json             # Frame alignment offsets
    calibration_board.mp4        # (optional) recording used for calibration
  camera_0/
    synced_video.mp4             # Trimmed + resampled video
    demo/                        # Full GVHMR output (preprocess/, hmr4d_results.pt, etc.)
    <stem>_hybrid_smplx.pt       # Per-camera SMPL-X (world space, camera's own frame)
    <stem>_body_hands.bvh        # Per-camera BVH
  camera_1/
    ...
  fused/
    aligned_cam0.pt              # Camera 0 params transformed to shared frame
    aligned_cam1.pt              # Camera 1 params transformed to shared frame
    fused_smplx.pt               # Fused SMPL-X params
    fused_body_hands.bvh         # Fused animation
    fused_body_hands.fbx         # Fused FBX for DCC import
    confidence_plot.png          # Per-camera confidence over time
    comparison.mp4               # Side-by-side: each camera's skeleton + fused
  session_manifest.json          # Metadata, camera count, sync info, output paths
```

---

## New Files

| File | Purpose |
|------|---------|
| `camera_calibration.py` | `CameraRig` class, `calibrate_from_checkerboard()`, `calibrate_from_colmap()`, save/load |
| `multi_camera.py` | Orchestration: `MultiCameraSession`, sync, per-camera dispatch, result collection |
| `multi_camera_fusion.py` | `transform_to_reference_frame()`, `fuse_params()`, confidence computation |
| `sync_audio.py` | Audio extraction + cross-correlation for temporal sync |

## Changes to Existing Files

| File | Change | Why |
|------|--------|-----|
| `tools/demo/demo.py` | Add `--intrinsics <path>` flag | Use calibrated K instead of estimated |
| `preprocess.py` | Add `trim_to_range(start_frame, end_frame)` | Apply sync offsets |
| `gvhmr_gui.py` | New "Multi-Camera" tab | Camera rig config, multi-video input, sync, per-camera + fused results |

---

## Implementation Phases

### Phase 1: Calibration + Sync Infrastructure
- `camera_calibration.py` with checkerboard calibration using OpenCV
- `sync_audio.py` with audio cross-correlation
- `CameraRig` JSON format and load/save
- Basic GUI tab for camera rig management
- **Deliverable**: Can calibrate a 2-camera rig and verify reprojection error

### Phase 2: Independent Multi-Camera Processing
- `multi_camera.py` orchestration — process each camera via subprocess
- `--intrinsics` flag in `demo.py`
- Sync-aware video trimming in `preprocess.py`
- Collect per-camera results into session directory
- **Deliverable**: Can process 2 cameras independently, view results side-by-side

### Phase 3: Fusion (Parameter Averaging)
- `multi_camera_fusion.py` — world frame alignment + weighted parameter averaging
- Confidence estimation from ViTPose keypoints
- Fused BVH/FBX export
- Comparison visualization
- **Deliverable**: Fused result that is measurably smoother than single camera

### Phase 4: Advanced Fusion (Future)
- Per-joint 3D fusion with FK/IK
- Reprojection optimization
- COLMAP auto-calibration
- Support for 3+ cameras

---

## Hardware Considerations

- **VRAM**: Each camera's GVHMR run uses ~8-10 GB. Process sequentially (not in parallel) to avoid contention. 96 GB is more than enough.
- **Storage**: Each camera generates ~500MB of intermediate data (features, keypoints, SLAM, results). For a 2-camera 60-second session at 30fps: ~1GB per camera + originals.
- **Processing time**: Roughly linear in number of cameras. 2 cameras ≈ 2× processing time. Fusion step itself is fast (< 1 second).

---

## Tradeoffs & Challenges

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| World frame alignment accuracy | HIGH | Careful calibration + yaw-only correction after gravity alignment |
| Gravity alignment disagreement between cameras | MEDIUM | Average gravity vectors, or use reference camera's alignment for all |
| Temporal sync precision | MEDIUM | Audio cross-correlation gives sub-frame accuracy; clap at start |
| Calibration drift (cameras moved between calibration and capture) | HIGH | Calibrate immediately before capture, or use rigid camera rig |
| Different image quality / exposure per camera | LOW | GVHMR is fairly robust to exposure differences |
| Person not visible in all cameras simultaneously | LOW | Confidence weighting handles this — zero-confidence frames get zero weight |

---

## What This Does NOT Solve

- **Real-time multi-camera capture**: This is batch/offline processing
- **Dense multi-view reconstruction**: We don't produce mesh/point cloud — just skeletal motion
- **Camera arrays (10+ cameras)**: Designed for 2-4 cameras. Larger arrays would benefit from a different architecture
- **Moving cameras**: Calibration assumes fixed/static cameras. Moving multi-camera is a much harder problem (each camera's SimpleVO would need to be fused at the camera level, not the body level)
