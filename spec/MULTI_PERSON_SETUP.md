# GVHMR Multi-Person Crossing: Powerhouse Setup Guide

> For Claude Code on Powerhouse (`shinyscale@100.125.206.12`). Base single-person env already works.

## What This Is

Multi-person body capture fails when two characters cross in video. Commit `1e50a45` (Mar 14) added a complete multi-person pipeline (~8K lines, 10 modules) using **segmentation + inpainting isolation** — isolate each person into their own clean video via SAM2 masks + ProPainter inpainting, then run single-person GVHMR on each.

That same commit fixed 3 critical bugs that were the direct cause of crossing failures:
1. **Mask loading bug** — `reprocess_person()` only loaded the target person's mask, never removed the other person
2. **ProPainter wrapper broken** — called `InpaintGenerator` directly instead of the full 4-stage pipeline
3. **DPVO never built** — silently fell back to SimpleVO (which is fine, we skip DPVO intentionally)

**The pipeline has never been run end-to-end on GPU.** Code is complete but untested.

---

## Step 1: Pull Latest & Verify Fix Commit

```bash
cd ~/GVHMR
git pull origin main
git log --oneline -5  # MUST see 1e50a45 "Fix multi-person crossing"
```

Initialize the ProPainter submodule (DPVO submodule is intentionally skipped):

```bash
cd ~/GVHMR
git submodule update --init third-party/ProPainter
```

**Verify the 3 fixes are present:**
- `propainter_inpaint.py` should have a full 4-stage pipeline in `inpaint_video()`: RAFT flow (line ~208-232), RecurrentFlowCompleteNet (line ~234-258), image propagation (line ~263-294), InpaintGenerator (line ~296-338) — NOT a simple `self._model(frames, masks)` call
- `multi_person_split.py` `reprocess_person()` (line ~763) should load ALL people's masks in a loop over `all_tracks` (lines ~806-812), not just the target person
- These are non-negotiable — without them, crossing will still fail

---

## Step 2: Install Multi-Person Dependencies

The base env (PyTorch, pytorch3d, ultralytics, etc.) already works. Three packages enable the crossing fix.

### 2a. SAM2 — precise per-person segmentation masks

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

> **Note:** The PyPI package `segment-anything-2` may not exist or may be stale.
> Installing from the repo directly is the reliable path.

Download the base_plus checkpoint (~300MB):
```bash
mkdir -p ~/GVHMR/inputs/checkpoints/sam2
wget -P ~/GVHMR/inputs/checkpoints/sam2/ \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

- Without SAM2: falls back to bbox-only blackout (won't cleanly isolate overlapping people)
- `sam2_segmenter.py` (line 36-40) searches these paths in order:
  1. `inputs/checkpoints/sam2/`
  2. `~/.cache/sam2/`
  3. `/mnt/f/models/sam2/`
- Config mapping is in `_MODEL_CONFIGS` (line 28-33): `"base_plus"` → `("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt")`
- Verify: `python -c "from sam2.build_sam import build_sam2_video_predictor; print('SAM2 OK')"`

### 2b. ProPainter — temporal video inpainting

ProPainter is already a git submodule at `third-party/ProPainter`. After `git submodule update --init third-party/ProPainter`:

```bash
cd ~/GVHMR/third-party/ProPainter
pip install -r requirements.txt
```

Download model weights:
```bash
mkdir -p ~/GVHMR/third-party/ProPainter/weights
wget -P ~/GVHMR/third-party/ProPainter/weights/ https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget -P ~/GVHMR/third-party/ProPainter/weights/ https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
wget -P ~/GVHMR/third-party/ProPainter/weights/ https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
```

- `propainter_inpaint.py` (line 22-36) searches these paths in order:
  1. `/mnt/f/ProPainter`
  2. `third-party/ProPainter` (relative to the script — this is the submodule)
  3. `~/ProPainter`
- It checks for `(path / "model").is_dir() or (path / "inference_propainter.py").exists()` to validate
- **Import conflict warning**: The wrapper uses `sys.path.insert(0, pp_path)` (line 101-102). ProPainter's `core/` directory can shadow other packages. If you see `from core.utils import to_tensors` fail, check `sys.path` ordering.
- Weights are auto-downloaded by `load_file_from_url()` at runtime if not present, but pre-downloading is more reliable.
- Verify: `python -c "import sys; sys.path.insert(0, 'third-party/ProPainter'); from model.propainter import InpaintGenerator; print('ProPainter OK')"`

### 2c. OC-SORT — robust tracking through crossings

```bash
pip install ocsort
```

- Without OC-SORT: uses YOLO's BoT-SORT which **swaps IDs at crossings** — this is a major contributor to the crossing problem
- `person_tracker.py` (line 30-34) checks: `from ocsort.ocsort import OCSort` with ImportError fallback
- Verify: `python -c "from ocsort.ocsort import OCSort; print('OC-SORT OK')"`

### 2d. DPVO — skip intentionally

SimpleVO fallback works fine. DPVO's custom CUDA kernels are risky on Blackwell (sm_120). The code handles this gracefully — `compute_shared_slam()` (line 205-215 in multi_person_split.py) falls back to SimpleVO when DPVO isn't available.

---

## Step 2.5: Clear Cached Results From Previous Runs

If you've already run the pipeline with bbox-blackout mode, cached intermediate
files will prevent the inpainting path from being used. The pipeline skips any
step whose output already exists.

**Delete these before re-running with `use_inpainting=True`:**

```bash
# Per-person isolated videos (will be re-created with inpainting)
rm -f outputs/multi_person/*/person_*/isolated_video.mp4
rm -rf outputs/multi_person/*/person_*/isolation_mode.json

# Per-person GVHMR results (must re-solve on clean isolated video)
rm -rf outputs/multi_person/*/person_*/demo/

# Per-person BVH/FBX (will be regenerated from new solve)
rm -f outputs/multi_person/*/person_*/body.bvh
rm -f outputs/multi_person/*/person_*/body.fbx

# Bridging/identity artifacts
rm -f outputs/multi_person/*/person_*/identity_track.json
rm -f outputs/multi_person/*/person_*/confidence.csv

# Scene preview (will be re-rendered)
rm -rf outputs/multi_person/*/assembly/
rm -f outputs/multi_person/*/session_manifest.json
```

**Usually keep these** (expensive to recompute, input-only, still valid):
- `detection/all_tracks.pt` — OC-SORT tracking cache. Keep it only if it was produced by the current tracker stack and IDs already look stable in `track_visualization.mp4`; otherwise delete it and re-track.
- `detection/track_visualization.mp4` — same
- `shared_slam.pt` — camera motion (computed on original video, not affected)
- `masks/` — SAM2 masks (if they exist, they're already correct)

---

## Step 3: Run End-to-End Test

### 3a. Quick sanity check — verify single-person still works

```bash
cd ~/GVHMR
python tools/demo/demo.py --video=docs/example_video/tennis.mp4 --static_cam
```

This should produce output in `outputs/demo/tennis/` with incam overlay video.

### 3b. Run multi-person pipeline on crossing test video

You need a test video with two people crossing. If you don't have one, any video with 2+ people works for basic validation.

**Via Python:**
```python
import sys
sys.path.insert(0, '.')
from multi_person_split import split_multi_person_video

result = split_multi_person_video(
    video_path="<PATH_TO_CROSSING_VIDEO>",
    output_dir="outputs/multi_person/crossing_test",
    min_track_duration=30,
    static_cam=False,    # set True if camera is tripod-mounted
    use_dpvo=False,      # always False on Powerhouse
    use_inpainting=True,  # REQUIRED for crossings — without this, bbox blackout is used
    max_persons=2,
)
print(f"Persons detected: {result.num_persons}")
for i, path in enumerate(result.person_video_paths):
    print(f"Person {i}: {path}")
```

**Via GUI (alternative):**
```bash
cd ~/GVHMR
python gvhmr_gui.py  # Gradio on :7860
# Multi-Person tab → set video path → CHECK "SAM2 + ProPainter inpainting" → Run
```

> **The checkbox defaults to off.** If you don't check it, the pipeline uses bbox-blackout
> isolation which is the exact path that fails during crossings.

The pipeline runs 7 steps:
1. **Detection & Tracking** — OC-SORT via YOLO detections (person_tracker.py)
2. **Shared SLAM** — Camera motion computed once on original video (SimpleVO)
3. **SAM2 Segmentation** — Per-person pixel masks (sam2_segmenter.py)
4. **Per-person Isolation** — Smart crop+inpaint where overlap detected (propainter_inpaint.py)
5. **Per-person GVHMR** — Run standard pipeline on each isolated video (demo.py with --slam_override)
6. **Identity Verification** — Confidence scoring, IoU-based crossing bridge (SLERP interpolation as safety net)
7. **World Assembly** — Compute relative offsets, render scene preview

---

## Step 4: Verification Checklist

| Check | Where | What to look for |
|-------|-------|-----------------|
| Tracking IDs stable | `detection/track_visualization.mp4` | Both people keep same color bbox through the crossing |
| Masks exist | `masks/person_*_masks.npz` | Both people have mask sequences |
| Overlap detected | `masks/overlap_map.npz` | Shows True for crossing frames |
| Inpainting activated | `person_*/isolation_mode.json` | Shows `"inpaint"` during crossing frames, `"crop"` elsewhere |
| Clean isolation | `person_*/isolated_video.mp4` | Other person removed cleanly in overlap frames |
| Per-person skeleton correct | `person_*/demo/` results | Skeleton tracks the right body, doesn't jump |
| World assembly | `assembly/person_offsets.json` | `offsets` should be non-zero where expected; `metadata` should show a sensible reference frame and avoid obviously synthetic early-frame anchors |
| Scene preview | `assembly/scene_preview.mp4` | Both skeletons visible and correctly positioned |

**The acid test**: At the exact crossing frames, does each person's skeleton stay on their own body?

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `isolation_mode` is all `"crop"`, never `"inpaint"` | SAM2 not found or masks failed | Check `segment-anything-2` installed, checkpoint at `inputs/checkpoints/sam2/sam2_hiera_base_plus.pt` |
| Inpainting produces black/garbage frames | ProPainter not found or weights missing | Verify `third-party/ProPainter` exists with `weights/` dir, confirm commit 1e50a45 present |
| IDs swap at the crossing point | OC-SORT not installed | `pip install ocsort`, verify `OCSORT_AVAILABLE` in person_tracker.py |
| OOM during SAM2/ProPainter | Video too long or high-res | Trim to 10-15s, or reduce resolution. 96GB VRAM should handle most cases |
| `from core.utils import to_tensors` import error | ProPainter's `core/` shadows another package | Check `sys.path` ordering |
| ProPainter version regex crash | PyTorch nightly version string | Already patched in 1e50a45 — verify it's present |
| `No module named 'model.modules.flow_comp_raft'` | ProPainter submodule not initialized | `git submodule update --init third-party/ProPainter` |
| DPVO import warning at startup | Expected — DPVO intentionally not built | Ignore, SimpleVO fallback is used automatically |
| Pipeline runs but `isolation_mode.json` missing | `use_inpainting=False` (the default) | Pass `use_inpainting=True` or check the GUI checkbox |
| Re-run with inpainting still uses old results | Cached `isolated_video.mp4` / `demo/` from prior run | Delete cached outputs per Step 2.5, then re-run |
| `NameError: name 'cfg' is not defined` in SLAM | DPVO partially importable (config loads, model doesn't) | Expected on Blackwell — availability check now tests full import chain, falls back to SimpleVO |

---

## Expected Output Structure

```
outputs/multi_person/crossing_test/
  shared_slam.pt                     # Camera motion (computed once)
  detection/
    all_tracks.pt                    # OC-SORT tracking results
    track_visualization.mp4          # Bbox overlay video with IDs
  masks/
    person_0_masks.npz               # SAM2 masks
    person_1_masks.npz
    overlap_map.npz                  # Per-frame overlap flags
  person_0/
    isolated_video.mp4               # Clean single-person video
    isolation_mode.json              # Per-frame: "crop" or "inpaint"
    identity_track.json              # Keyframe-based identity
    confidence.csv                   # Per-frame confidence scores
    body.bvh                         # BVH animation export
    demo/                            # Full GVHMR output (hmr4d_results.pt, etc)
  person_1/
    ...                              # Same structure
  assembly/
    person_offsets.json              # Relative XZ positions + reference-frame metadata
    assembled_smplx.pt               # All people in shared world frame
    scene_preview.mp4                # Combined overlay video
  session_manifest.json              # Full session metadata
```

---

## Key Files Reference

| File | Lines | Role |
|------|-------|------|
| `multi_person_split.py` | 1188 | **Orchestrator** — 7-step pipeline, `split_multi_person_video()` entry point |
| `person_tracker.py` | 407 | OC-SORT / YOLO tracking with interpolation & smoothing |
| `sam2_segmenter.py` | 277 | SAM2 video segmentation with bbox re-prompting |
| `propainter_inpaint.py` | 472 | 4-stage ProPainter inpainting (the critical fix) |
| `world_assembly.py` | 229 | World-space offset computation & scene assembly |
| `identity_confidence.py` | 282 | Per-frame 5-component confidence scoring |
| `identity_tracking.py` | 286 | Keyframe-based identity persistence |
| `identity_bridge.py` | 430 | IoU-based crossing detection + SLERP interpolation (safety net after inpainting) |
| `identity_reid.py` | 174 | SMPL betas shape-based re-identification |
| `tools/demo/demo.py` | 360 | Single-person pipeline with `--slam_override` support |
