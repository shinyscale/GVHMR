# GenAI-Assisted Multi-Person Mocap: Design Exploration

## Problem

The current pipeline (GVHMR + SMPLest-X) is single-person only. YOLO detects all people but `Tracker.get_one_track()` and SMPLest-X both select only the largest person. When two people overlap (e.g. arm around shoulder), the occluded limbs of the secondary person are lost, and the primary person's pose estimation is confused by the other person's visible limbs.

**Goal**: Use generative AI (segmentation + inpainting) to isolate each person into their own "clean" video, then process each through the existing single-person pipeline.

---

## Proposed Pipeline

```
Input Video (multi-person)
    |
    v
+---------------------+
| 1. YOLO Detection   |  Already in pipeline -- detects all people per frame
|    (all people)      |  Returns bboxes + track IDs across frames
+---------+-----------+
          | per-person bboxes
          v
+---------------------+
| 2. SAM2 Video       |  Segment each person with precise masks
|    Segmentation      |  Track identity across frames (memory module)
|                      |  Auto-prompted from YOLO bboxes
+---------+-----------+
          | per-person binary masks (N_people x N_frames)
          v
+---------------------+
| 3. Video Inpainting |  For each person: mask OUT the others,
|    (per person)      |  inpaint the removed regions to complete
|                      |  the target person's body plausibly
+---------+-----------+
          | N clean single-person videos
          v
+---------------------+
| 4. Existing Pipeline|  Run GVHMR + SMPLest-X + Face on each
|    (per person)      |  No changes needed to existing code
+---------------------+
```

---

## Step 1: Person Detection (Already Exists)

**Current code**: `hmr4d/utils/preproc/tracker.py`
- `Tracker.track()` returns full multi-person tracking history with YOLO track IDs
- `Tracker.get_one_track()` selects only the largest -- this is the only bottleneck
- `sort_track_length()` already ranks all tracks by area + duration

**What's needed**: Expose `track()` results (all people) instead of only `get_one_track()`. Trivial change.

---

## Step 2: SAM2 Video Segmentation

**Model**: [SAM2](https://github.com/facebookresearch/sam2) (Meta, 2024)
- Extends Segment Anything to video with a per-session memory module
- Tracks objects across frames through occlusions via temporal attention
- 30-44 FPS depending on model size (base+: 80.8M params, 35 FPS)

**Prompting strategy**: Use YOLO bboxes as initial prompts for SAM2.
- Frame 0: provide each person's YOLO bbox -> SAM2 generates precise mask
- Subsequent frames: SAM2 propagates masks automatically via memory
- Re-prompt on keyframes if tracking drifts (every ~30 frames)

**ComfyUI nodes**:
- [ComfyUI-segment-anything-2 (Kijai)](https://github.com/kijai/ComfyUI-segment-anything-2) -- most mature
- [ComfyUI-SAMURAI](https://github.com/takemetosiberia/ComfyUI-SAMURAI--SAM2-) -- motion-aware variant

**Output**: Per-person binary mask sequence (N_frames x H x W) per tracked person.

---

## Step 3: Video Inpainting

This is the core GenAI step. For each person, we need to:
1. Take the original video
2. Mask out all OTHER people (using their SAM2 masks)
3. Inpaint the masked regions so the target person appears alone

### Option A: ProPainter (Recommended for Speed)

[ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023)
- **Approach**: Recurrent flow completion + dual-domain propagation + sparse video transformer
- **Temporal consistency**: Propagates through both image and feature domains -- prevents drift
- **Speed**: Fast (non-diffusion). Processes video in chunks
- **VRAM**: Medium (~8-12 GB for 720p)
- **ComfyUI**: [ComfyUI_ProPainter_Nodes](https://github.com/daniabib/ComfyUI_ProPainter_Nodes)
- **Best for**: Removing people where the background is fairly static

### Option B: DiffuEraser (Best Quality)

[DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) (2025)
- **Approach**: Diffusion UNet + BrushNet branch + AnimateDiff-style temporal attention
- **Temporal consistency**: Superior to ProPainter -- diffusion naturally smooths
- **Speed**: Medium-slow (diffusion inference per chunk)
- **VRAM**: High (~16-24 GB for 720p)
- **ComfyUI**: [ComfyUI_DiffuEraser](https://github.com/smthemex/ComfyUI_DiffuEraser)
- **Best for**: Complex scenes, moving backgrounds, higher quality needs

### Option C: SD Inpainting + AnimateDiff (Most Flexible)

- Per-frame SD inpainting with AnimateDiff motion modules for temporal coherence
- **Speed**: Slow (full diffusion per frame)
- **VRAM**: High
- Most control (prompts, ControlNet conditioning) but overkill for person removal
- **Best for**: When you need artistic control over what fills the gap

### Recommendation

**ProPainter** for the initial implementation -- it's fast, proven, and the quality is good enough for the purpose (we just need clean-enough input for ViTPose/GVHMR, not pixel-perfect reconstruction).

**DiffuEraser** as upgrade path if ProPainter's quality isn't sufficient for edge cases.

---

## The Arm-Around-Shoulder Problem (Specific)

This is the hardest case because:
1. Person A's arm overlaps Person B's torso
2. When we mask out Person B, we also lose part of Person A's arm
3. The inpainter needs to hallucinate the rest of Person A's arm

**Why it works anyway**:
- The inpainter sees Person A's shoulder, upper arm, and where the arm disappears behind the mask. Diffusion/flow-based models understand human anatomy and will extend the arm plausibly
- The result doesn't need to be *photographically correct* -- it needs to be good enough for ViTPose to detect the arm's approximate position
- ViTPose is already robust to partial occlusion; even a rough inpainted arm gives it enough signal

**When it might struggle**:
- If the arm is almost entirely hidden (elbow to hand all behind the other person)
- If both people are wearing similar clothing (mask boundary ambiguity)
- Fast motion during heavy overlap (SAM2 mask tracking may drift)

---

## Integration Options

### Option 1: Separate ComfyUI Preprocessing Step

```
User workflow:
  1. Load video into ComfyUI
  2. Run "Multi-Person Split" workflow -> outputs person_1.mp4, person_2.mp4, ...
  3. Feed each video into existing GVHMR pipeline (gvhmr_gui.py)
```

**Pros**: No changes to existing pipeline code. Full ComfyUI flexibility. Easy to iterate on the workflow.
**Cons**: Manual step. User needs ComfyUI installed and configured.

### Option 2: Python Preprocessing Script (No ComfyUI)

Run SAM2 + ProPainter directly in Python as a preprocessing stage:
```python
# New file: multi_person_split.py
def split_multi_person_video(video_path: str, output_dir: str) -> list[str]:
    """Detect, segment, and inpaint to produce one clean video per person."""
    tracks = yolo_detect_all(video_path)
    masks = sam2_segment_all(video_path, tracks)
    clean_videos = []
    for person_id, person_masks in masks.items():
        other_masks = combine_masks_except(masks, person_id)
        clean_video = propainter_inpaint(video_path, other_masks)
        clean_videos.append(clean_video)
    return clean_videos
```

**Pros**: Fully automated. Could integrate into gvhmr_gui.py as a checkbox.
**Cons**: Need to install SAM2 + ProPainter in the GVHMR conda env. More dependencies.

### Option 3: Hybrid -- ComfyUI API

Call ComfyUI's API from gvhmr_gui.py to execute a saved workflow:
```python
# gvhmr_gui.py -- new preprocessing stage
comfyui_result = requests.post("http://localhost:8188/prompt", json=workflow)
```

**Pros**: Leverages ComfyUI's model management, GPU scheduling, and node ecosystem.
**Cons**: Requires ComfyUI running as a service. Brittle coupling.

---

## Tradeoffs & Challenges

| Challenge | Severity | Mitigation |
|-----------|----------|------------|
| Temporal consistency of inpainting | HIGH | Use ProPainter (flow-based) or DiffuEraser (temporal attention) -- NOT per-frame SD |
| Anatomical plausibility of hallucinated limbs | MEDIUM | Diffusion models understand anatomy. ViTPose is forgiving of imperfect input |
| Processing time (inpainting is slow) | MEDIUM | ProPainter is fast enough. Process at 720p, not full res |
| SAM2 mask quality at occlusion boundaries | MEDIUM | Use YOLO re-prompting every 30 frames. Dilate masks slightly |
| VRAM requirements | LOW-MED | ProPainter ~8-12 GB. DiffuEraser ~16-24 GB. Fits on most production GPUs |
| Mask assignment ambiguity (whose arm is it?) | LOW | SAM2 tracks identity. YOLO track IDs persist. Edge cases exist but are rare |

---

## What This Does NOT Solve

- **Identical twins / similar appearance**: SAM2 may swap identities
- **Fully hidden person**: If someone is completely behind another, no signal to segment
- **Very tight embraces**: Where body boundaries are ambiguous even to humans
- **Real-time processing**: This is a batch offline workflow

---

## Recommended First Experiment

Before building anything:
1. Install [ComfyUI_ProPainter_Nodes](https://github.com/daniabib/ComfyUI_ProPainter_Nodes) + [ComfyUI-segment-anything-2](https://github.com/kijai/ComfyUI-segment-anything-2)
2. Take a test video with two people (arm-around-shoulder scenario)
3. Manually segment Person B with SAM2 in ComfyUI
4. Inpaint Person B's region with ProPainter
5. Run the resulting clean video through the existing GVHMR pipeline
6. Compare pose quality vs. running the original multi-person video

This validates the approach end-to-end before any code integration work.
