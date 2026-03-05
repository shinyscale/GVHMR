"""Smoke tests for the motion capture pipeline.

Standalone script — no pytest needed. Run: python smoke_test.py
Tests truncate to 5 frames for speed. Missing data → SKIP (not FAIL).
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np

# ── Test registry ──

_tests = []


def test(fn):
    _tests.append(fn)
    return fn


# ── Test data paths ──

_VIDEO = "outputs/demo/Move One Test 02 Source 1_fixed/0_input_video.mp4"
_SMPLESTX_PT = "outputs/perfcap/Move One Test 02 Source 1_fixed/smplestx/Move One Test 02 Source 1_fixed_smplx.pt"
_HYBRID_PT = "outputs/perfcap/Move One Test 02 Source 1_fixed/Move One Test 02 Source 1_fixed_hybrid_smplx.pt"


def _require(*paths):
    """Skip test if any required file is missing."""
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(f"SKIP: {p} not found")


# ── Tests ──


@test
def test_face_crop_extraction():
    _require(_VIDEO, _SMPLESTX_PT)
    import torch
    from face_capture import extract_face_crops

    data = torch.load(_SMPLESTX_PT, map_location="cpu", weights_only=False)
    bboxes = np.array(data["bbox"])[:5]

    crops = extract_face_crops(_VIDEO, bboxes)
    assert len(crops) >= 1, "No crops returned"
    assert crops[0].shape == (256, 256, 3), f"Wrong crop shape: {crops[0].shape}"


@test
def test_face_blendshapes():
    _require(_VIDEO, _SMPLESTX_PT)
    import torch
    from face_capture import extract_face_crops, run_face_blendshapes

    data = torch.load(_SMPLESTX_PT, map_location="cpu", weights_only=False)
    bboxes = np.array(data["bbox"])[:5]

    crops = extract_face_crops(_VIDEO, bboxes)
    bs = run_face_blendshapes(crops[:5])
    assert len(bs) >= 1, "No blendshapes returned"
    assert len(bs[0]) >= 50, f"Expected ~52 channels, got {len(bs[0])}"


@test
def test_smooth_blendshapes():
    from face_capture import smooth_blendshapes, ARKIT_BLENDSHAPES

    # Synthetic data: 20 frames
    data = [{name: float(np.random.rand()) for name in ARKIT_BLENDSHAPES} for _ in range(20)]
    smoothed = smooth_blendshapes(data, window=5)
    assert len(smoothed) == 20
    for frame in smoothed:
        for v in frame.values():
            assert 0.0 <= v <= 1.0, f"Value out of range: {v}"


@test
def test_eq_blendshapes():
    from face_capture import eq_blendshapes, ARKIT_BLENDSHAPES

    data = [{name: float(np.random.rand()) for name in ARKIT_BLENDSHAPES} for _ in range(20)]
    result = eq_blendshapes(data, preset="subtle")
    assert len(result) == 20


@test
def test_csv_export():
    from face_capture import export_blendshapes_csv, ARKIT_BLENDSHAPES

    data = [{name: 0.5 for name in ARKIT_BLENDSHAPES} for _ in range(10)]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    try:
        export_blendshapes_csv(data, path, fps=30.0)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 11, f"Expected 11 lines (header + 10), got {len(lines)}"
        header = lines[0].strip().split(",")
        assert "Timecode" in header
        assert "EyeBlinkLeft" in header
    finally:
        os.unlink(path)


@test
def test_extract_smplx_params():
    _require(_SMPLESTX_PT)
    from smplx_to_bvh import extract_smplx_params

    params = extract_smplx_params(_SMPLESTX_PT)
    n = params["num_frames"]
    assert n > 0, "No frames"
    assert params["global_orient"].shape == (n, 3)
    assert params["body_pose"].shape == (n, 21, 3)
    assert params["left_hand_pose"].shape == (n, 15, 3)
    assert params["right_hand_pose"].shape == (n, 15, 3)
    assert "bbox" in params, "bbox missing from extracted params"
    assert params["bbox"].shape == (n, 4)


@test
def test_bvh_conversion():
    _require(_SMPLESTX_PT)
    from smplx_to_bvh import extract_smplx_params, convert_params_to_bvh

    params = extract_smplx_params(_SMPLESTX_PT)
    # Truncate to 5 frames
    for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]:
        params[k] = params[k][:5]
    if "betas" in params:
        params["betas"] = params["betas"][:5]
    if "bbox" in params:
        params["bbox"] = params["bbox"][:5]
    params["num_frames"] = 5

    with tempfile.NamedTemporaryFile(suffix=".bvh", delete=False) as f:
        path = f.name

    try:
        convert_params_to_bvh(params, path, fps=30.0)
        content = open(path).read()
        assert "HIERARCHY" in content, "Missing HIERARCHY"
        assert "MOTION" in content, "Missing MOTION"
        assert "Frames: 5" in content, "Wrong frame count"
    finally:
        os.unlink(path)


@test
def test_skeleton_overlay():
    _require(_SMPLESTX_PT, _VIDEO)
    from smplx_to_bvh import extract_smplx_params
    from visualize_skeleton import render_skeleton_video

    params = extract_smplx_params(_SMPLESTX_PT)
    for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]:
        params[k] = params[k][:5]
    if "betas" in params:
        params["betas"] = params["betas"][:5]
    if "bbox" in params:
        params["bbox"] = params["bbox"][:5]
    params["num_frames"] = 5

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    try:
        render_skeleton_video(video_path=_VIDEO, output_path=path, fps=30.0, params=params)
        assert os.path.getsize(path) > 0, "Skeleton video is empty"
    finally:
        os.unlink(path)


@test
def test_world_views():
    _require(_SMPLESTX_PT)
    from smplx_to_bvh import extract_smplx_params
    from visualize_skeleton import render_world_views

    params = extract_smplx_params(_SMPLESTX_PT)
    for k in ["global_orient", "body_pose", "left_hand_pose", "right_hand_pose", "transl"]:
        params[k] = params[k][:5]
    if "betas" in params:
        params["betas"] = params["betas"][:5]
    if "bbox" in params:
        params["bbox"] = params["bbox"][:5]
    params["num_frames"] = 5

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    try:
        render_world_views(output_path=path, fps=30.0, params=params)
        assert os.path.getsize(path) > 0, "World view video is empty"
    finally:
        os.unlink(path)


@test
def test_face_mesh_video():
    _require(_VIDEO, _SMPLESTX_PT)
    import torch
    from face_capture import render_face_mesh_video

    data = torch.load(_SMPLESTX_PT, map_location="cpu", weights_only=False)
    bboxes = np.array(data["bbox"])[:5]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    try:
        render_face_mesh_video(_VIDEO, bboxes, path, fps=30.0)
        assert os.path.getsize(path) > 0, "Face mesh video is empty"
    finally:
        os.unlink(path)


@test
def test_merge_preserves_bbox():
    _require(_SMPLESTX_PT)
    from smplx_to_bvh import extract_smplx_params, merge_gvhmr_smplestx_params

    smplestx = extract_smplx_params(_SMPLESTX_PT)
    assert "bbox" in smplestx, "SMPLest-X params missing bbox"

    # Fake GVHMR params
    n = smplestx["num_frames"]
    gvhmr = {
        "global_orient": np.zeros((n, 3)),
        "body_pose": np.zeros((n, 21, 3)),
        "left_hand_pose": np.zeros((n, 15, 3)),
        "right_hand_pose": np.zeros((n, 15, 3)),
        "transl": np.zeros((n, 3)),
        "betas": np.zeros((n, 10)),
        "num_frames": n,
    }

    merged = merge_gvhmr_smplestx_params(gvhmr, smplestx)
    assert "bbox" in merged, "Merged params missing bbox"
    assert merged["bbox"].shape[0] == n


# ── Runner ──

if __name__ == "__main__":
    passed = 0
    failed = 0
    skipped = 0

    for test_fn in _tests:
        name = test_fn.__name__
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except FileNotFoundError as e:
            print(f"  SKIP  {name} — {e}")
            skipped += 1
        except Exception as e:
            print(f"  FAIL  {name} — {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(1 if failed else 0)
