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


@test
def test_vitpose_face_crops_synthetic():
    """Test ViTPose-based face crop extraction with synthetic data."""
    import tempfile
    import cv2
    import torch
    from face_capture import extract_face_crops_from_keypoints

    # Create a synthetic video (10 frames, 640x480, black)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        video_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        vitpose_path = f.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, 30, (640, 480))
        for _ in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Draw a white circle where the "face" is
            cv2.circle(frame, (320, 120), 40, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()

        # Create synthetic ViTPose data: (10, 17, 3) — face at ~(320, 120)
        vitpose = np.zeros((10, 17, 3))
        # nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4
        vitpose[:, 0] = [320, 120, 0.9]  # nose
        vitpose[:, 1] = [305, 110, 0.9]  # left eye
        vitpose[:, 2] = [335, 110, 0.9]  # right eye
        vitpose[:, 3] = [290, 120, 0.8]  # left ear
        vitpose[:, 4] = [350, 120, 0.8]  # right ear
        torch.save(torch.tensor(vitpose, dtype=torch.float32), vitpose_path)

        crops = extract_face_crops_from_keypoints(video_path, vitpose_path)
        assert len(crops) == 10, f"Expected 10 crops, got {len(crops)}"
        assert crops[0].shape == (384, 384, 3), f"Wrong crop shape: {crops[0].shape}"
        # Crop should contain some non-zero pixels (the white circle)
        assert crops[0].max() > 0, "Crop is all black — face not captured"
    finally:
        os.unlink(video_path)
        os.unlink(vitpose_path)


@test
def test_merge_gvhmr_hamer_params():
    """Test HaMeR hand merge with synthetic data."""
    from hamer_inference import merge_gvhmr_hamer_params

    n = 20
    gvhmr_params = {
        "global_orient": np.zeros((n, 3)),
        "body_pose": np.zeros((n, 21, 3)),
        "left_hand_pose": np.ones((n, 15, 3)) * 0.1,  # SMPLest-X baseline
        "right_hand_pose": np.ones((n, 15, 3)) * 0.1,
        "transl": np.zeros((n, 3)),
        "betas": np.zeros((n, 10)),
        "num_frames": n,
    }

    # HaMeR: confident on first 10 frames only
    hamer_params = {
        "left_hand_pose": np.ones((n, 15, 3)) * 0.5,
        "right_hand_pose": np.ones((n, 15, 3)) * 0.5,
        "left_confidence": np.concatenate([np.ones(10) * 0.9, np.zeros(10)]),
        "right_confidence": np.concatenate([np.ones(10) * 0.9, np.zeros(10)]),
    }

    merged = merge_gvhmr_hamer_params(gvhmr_params, hamer_params, confidence_threshold=0.5)

    # First 10 frames should use HaMeR (0.5)
    assert np.allclose(merged["left_hand_pose"][0], 0.5), "Frame 0 should use HaMeR"
    assert np.allclose(merged["left_hand_pose"][9], 0.5), "Frame 9 should use HaMeR"
    # Last 10 frames should keep SMPLest-X baseline (0.1)
    assert np.allclose(merged["left_hand_pose"][10], 0.1), "Frame 10 should keep SMPLest-X"
    assert np.allclose(merged["left_hand_pose"][19], 0.1), "Frame 19 should keep SMPLest-X"
    assert merged["hand_source"] == "hamer"


@test
def test_multi_signal_contact_detection():
    """Test enhanced foot contact detection with synthetic trajectory data."""
    from smplx_to_bvh import _compute_root_from_contacts, PELVIS_HEIGHT_CM

    n = 30
    # Create synthetic params where a person is standing, then walking
    params = {
        "global_orient": np.zeros((n, 3)),
        "body_pose": np.zeros((n, 21, 3)),
        "left_hand_pose": np.zeros((n, 15, 3)),
        "right_hand_pose": np.zeros((n, 15, 3)),
        "transl": np.zeros((n, 3)),
        "num_frames": n,
    }
    # Set a slight hip rotation so feet separate in FK
    # L_Hip (body_pose index 0) and R_Hip (body_pose index 1)
    # Small rotation to spread legs naturally
    params["body_pose"][:, 0, 2] = 0.1   # L_Hip slight outward
    params["body_pose"][:, 1, 2] = -0.1  # R_Hip slight outward

    root = _compute_root_from_contacts(params, fps=30.0)

    assert root.shape == (n, 3), f"Wrong root shape: {root.shape}"
    # Root Y should be positive (above ground)
    assert root[:, 1].min() > -0.1, "Root Y dropped below ground"
    # XZ should be approximately centered
    assert abs(root[:, 0].mean()) < 0.1, "Root X not centered"
    assert abs(root[:, 2].mean()) < 0.1, "Root Z not centered"


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
