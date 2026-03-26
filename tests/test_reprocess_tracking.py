"""Tests for anchor-driven bbox track regeneration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reprocess_tracking import regenerate_dense_track
from identity_confidence import compute_all_confidences
from multi_person_split import (
    _invalidate_reprocess_outputs,
    _export_person_bvh,
    _merge_duplicate_tracks,
    _duplicate_track_ids_for_target,
    _stitch_fragmented_tracks,
)


def _track(track_id: int, boxes, mask=None, conf=None):
    boxes_arr = torch.tensor(boxes, dtype=torch.float32)
    n = len(boxes_arr)
    if mask is None:
        mask = [True] * n
    if conf is None:
        conf = [1.0 if m else 0.0 for m in mask]
    return {
        "track_id": track_id,
        "bbx_xyxy": boxes_arr,
        "detection_mask": torch.tensor(mask, dtype=torch.bool),
        "detection_conf": torch.tensor(conf, dtype=torch.float32),
    }


def test_regenerate_dense_track_hard_applies_manual_and_verified_anchors():
    original = np.array(
        [
            [0.0, 0.0, 20.0, 40.0],
            [5.0, 0.0, 25.0, 40.0],
            [10.0, 0.0, 30.0, 40.0],
            [15.0, 0.0, 35.0, 40.0],
        ],
        dtype=np.float32,
    )
    other = np.array(
        [
            [40.0, 0.0, 60.0, 40.0],
            [45.0, 0.0, 65.0, 40.0],
            [50.0, 0.0, 70.0, 40.0],
            [55.0, 0.0, 75.0, 40.0],
        ],
        dtype=np.float32,
    )
    tracks = [_track(10, original), _track(20, other)]

    result = regenerate_dense_track(
        all_tracks=tracks,
        target_index=0,
        original_bboxes=original,
        manual_bbox_keyframes={1: np.array([100.0, 10.0, 140.0, 90.0], dtype=np.float32)},
        verified_identity_keyframes=[
            {"frame": 3, "bbox": [160.0, 20.0, 200.0, 100.0], "verified": True}
        ],
    )

    np.testing.assert_allclose(result.boxes[1], [100.0, 10.0, 140.0, 90.0], atol=1e-6)
    np.testing.assert_allclose(result.boxes[3], [160.0, 20.0, 200.0, 100.0], atol=1e-6)
    assert result.detection_mask[1]
    assert result.detection_mask[3]
    assert result.detection_conf[1] == 1.0
    assert result.detection_conf[3] == 1.0


def test_regenerate_dense_track_prefers_consistent_source_over_linear_interp():
    original = np.array(
        [
            [0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 20.0, 40.0],
            [0.0, 0.0, 20.0, 40.0],
        ],
        dtype=np.float32,
    )
    alternate = np.array(
        [
            [0.0, 0.0, 20.0, 40.0],
            [20.0, 0.0, 40.0, 40.0],
            [120.0, 0.0, 140.0, 40.0],
            [140.0, 0.0, 160.0, 40.0],
            [160.0, 0.0, 180.0, 40.0],
        ],
        dtype=np.float32,
    )
    tracks = [_track(10, original), _track(20, alternate)]

    result = regenerate_dense_track(
        all_tracks=tracks,
        target_index=0,
        original_bboxes=original,
        manual_bbox_keyframes={
            1: alternate[1].copy(),
            3: alternate[3].copy(),
        },
        verified_identity_keyframes=[],
    )

    interpolated_midpoint = np.array([80.0, 0.0, 100.0, 40.0], dtype=np.float32)
    np.testing.assert_allclose(result.boxes[1], alternate[1], atol=1e-6)
    np.testing.assert_allclose(result.boxes[3], alternate[3], atol=1e-6)
    np.testing.assert_allclose(result.boxes[2], alternate[2], atol=1e-6)
    assert not np.allclose(result.boxes[2], interpolated_midpoint)
    assert int(result.source_track_ids[2]) == 20


def test_invalidate_reprocess_outputs_removes_stale_motion_artifacts(tmp_path):
    person_dir = tmp_path / "person_0"
    (person_dir / "demo" / "isolated_video" / "preprocess").mkdir(parents=True)
    (person_dir / "preprocess").mkdir(parents=True)
    stale_files = [
        person_dir / "body.bvh",
        person_dir / "confidence.csv",
        person_dir / "hpe_results.pt",
        person_dir / "demo" / "isolated_video" / "hmr4d_results.pt",
    ]
    for path in stale_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale")

    _invalidate_reprocess_outputs(person_dir, estimation_backend="gemx")

    for path in stale_files:
        assert not path.exists()
    assert not (person_dir / "preprocess").exists()
    assert not (person_dir / "demo" / "isolated_video" / "preprocess").exists()


def test_export_person_bvh_uses_gvhmr_results(tmp_path, monkeypatch):
    person_dir = tmp_path / "person_0"
    result_dir = person_dir / "demo" / "isolated_video"
    result_dir.mkdir(parents=True)
    pt_path = result_dir / "hmr4d_results.pt"
    pt_path.write_text("placeholder")

    captured = {}

    def fake_extract(path):
        captured["extract_path"] = path
        return {"dummy": True}

    def fake_convert(params, output_path, skip_world_grounding=True):
        captured["convert_params"] = params
        captured["output_path"] = output_path
        captured["skip_world_grounding"] = skip_world_grounding
        Path(output_path).write_text("BVH")

    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "smplx_to_bvh",
        types.SimpleNamespace(
            convert_params_to_bvh=fake_convert,
            extract_gvhmr_params=fake_extract,
        ),
    )

    bvh_path = _export_person_bvh(person_dir)

    assert bvh_path == person_dir / "body.bvh"
    assert captured["extract_path"] == str(pt_path)
    assert captured["output_path"] == str(person_dir / "body.bvh")
    assert captured["skip_world_grounding"] is True
    assert bvh_path.read_text() == "BVH"


def test_compute_all_confidences_prefers_mask_overlap_and_verified_frames():
    target = _track(
        10,
        [
            [0.0, 0.0, 20.0, 40.0],
            [10.0, 0.0, 30.0, 40.0],
            [20.0, 0.0, 40.0, 40.0],
            [30.0, 0.0, 50.0, 40.0],
        ],
        conf=[0.9, 0.9, 0.9, 0.9],
    )
    other = _track(
        20,
        [
            [5.0, 0.0, 25.0, 40.0],
            [15.0, 0.0, 35.0, 40.0],
            [25.0, 0.0, 45.0, 40.0],
            [35.0, 0.0, 55.0, 40.0],
        ],
        conf=[0.9, 0.9, 0.9, 0.9],
    )
    vitpose = np.ones((4, 17, 3), dtype=np.float32)
    vitpose[..., 2] = 1.0
    betas = np.zeros((4, 10), dtype=np.float32)

    baseline = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, other],
        num_frames=4,
        vitpose=vitpose,
        per_frame_betas=betas,
        established_betas=np.ones(10, dtype=np.float32),
        stable_reference_frames=1,
    )
    robust = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, other],
        num_frames=4,
        vitpose=vitpose,
        per_frame_betas=betas,
        established_betas=np.ones(10, dtype=np.float32),
        overlap_signal=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        verified_frames={1},
        stable_reference_frames=6,
    )

    assert robust[1].overall > baseline[1].overall
    assert robust[1].bbox_overlap < baseline[1].bbox_overlap


def test_compute_all_confidences_manual_anchor_frames_raise_support_without_verify():
    target = _track(
        10,
        [
            [0.0, 0.0, 20.0, 40.0],
            [10.0, 0.0, 30.0, 40.0],
            [20.0, 0.0, 40.0, 40.0],
            [30.0, 0.0, 50.0, 40.0],
        ],
        conf=[0.8, 0.8, 0.8, 0.8],
    )
    other = _track(
        20,
        [
            [5.0, 0.0, 25.0, 40.0],
            [15.0, 0.0, 35.0, 40.0],
            [25.0, 0.0, 45.0, 40.0],
            [35.0, 0.0, 55.0, 40.0],
        ],
        conf=[0.8, 0.8, 0.8, 0.8],
    )
    vitpose = np.zeros((4, 17, 3), dtype=np.float32)
    vitpose[..., 2] = 0.2
    betas = np.zeros((4, 10), dtype=np.float32)

    baseline = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, other],
        num_frames=4,
        vitpose=vitpose,
        per_frame_betas=betas,
        established_betas=np.ones(10, dtype=np.float32),
        overlap_signal=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        stable_reference_frames=1,
    )
    anchored = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, other],
        num_frames=4,
        vitpose=vitpose,
        per_frame_betas=betas,
        established_betas=np.ones(10, dtype=np.float32),
        overlap_signal=np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32),
        anchor_frames={1, 3},
        stable_reference_frames=2,
    )

    assert anchored[1].overall > baseline[1].overall
    assert anchored[1].visible_keypoints > baseline[1].visible_keypoints


def test_compute_all_confidences_ignores_same_body_duplicate_track():
    repeated = [[100.0, 0.0, 140.0, 80.0]] * 40
    target = _track(
        6,
        repeated,
        conf=[0.95] * 40,
    )
    duplicate = _track(
        1,
        repeated,
        conf=[0.95] * 40,
    )
    other = _track(
        2,
        [[0.0, 0.0, 40.0, 80.0]] * 40,
        conf=[0.95] * 40,
    )
    vitpose = np.ones((40, 17, 3), dtype=np.float32)
    vitpose[..., 2] = 1.0

    poisoned = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, duplicate, other],
        num_frames=40,
        vitpose=vitpose,
    )
    cleaned = compute_all_confidences(
        track_idx=0,
        all_tracks=[target, duplicate, other],
        num_frames=40,
        vitpose=vitpose,
        ignored_track_ids={1},
    )

    assert poisoned[0].bbox_overlap > 0.9
    assert cleaned[0].bbox_overlap < 0.1
    assert cleaned[0].overall > poisoned[0].overall


def test_duplicate_track_ids_for_target_detects_same_body_fragment():
    repeated = [[100.0, 0.0, 140.0, 80.0]] * 40
    target = _track(
        6,
        repeated,
    )
    duplicate = _track(
        1,
        repeated,
    )
    other = _track(
        2,
        [[0.0, 0.0, 40.0, 80.0]] * 40,
    )

    ignored = _duplicate_track_ids_for_target([target, duplicate, other], 0)

    assert ignored == {1}


def test_merge_duplicate_tracks_collapses_same_body_fragment():
    repeated = [
        [100.0 + i, 0.0, 140.0 + i, 80.0]
        for i in range(40)
    ]
    primary = _track(
        6,
        repeated,
        mask=[True] * 40,
        conf=[0.9] * 40,
    )
    fragment = _track(
        1,
        repeated,
        mask=[True] * 32 + [False] * 8,
        conf=[0.95] * 32 + [0.0] * 8,
    )
    other = _track(
        2,
        [[0.0, 0.0, 40.0, 80.0]] * 40,
    )

    merged, pairs = _merge_duplicate_tracks([primary, fragment, other])

    assert len(merged) == 2
    assert any(pair[0] == 6 and pair[1] == 1 for pair in pairs)
    surviving = next(track for track in merged if int(track["track_id"]) == 6)
    merged_from = surviving.get("merged_from", [])
    assert 1 in merged_from


def test_stitch_fragmented_tracks_merges_sequential_handoff():
    early_boxes = [[1000.0 + i, 200.0, 1400.0 + i, 960.0] for i in range(40)]
    late_boxes = [[760.0 - i, 198.0, 1080.0 - i, 958.0] for i in range(40)]
    persistent_boxes = [[500.0 + i, 180.0, 900.0 + i, 940.0] for i in range(90)]

    early = _track(
        1,
        early_boxes + [early_boxes[-1]] * 50,
        mask=[True] * 40 + [False] * 50,
        conf=[0.94] * 40 + [0.0] * 50,
    )
    late = _track(
        6,
        [late_boxes[0]] * 45 + late_boxes,
        mask=[False] * 45 + [True] * 40,
        conf=[0.0] * 45 + [0.93] * 40,
    )
    persistent = _track(
        2,
        persistent_boxes,
        mask=[True] * 90,
        conf=[0.95] * 90,
    )

    stitched, pairs = _stitch_fragmented_tracks([persistent, late, early])

    assert len(stitched) == 2
    assert any({pair[0], pair[1]} == {1, 6} for pair in pairs)
    surviving = next(track for track in stitched if int(track["track_id"]) in {1, 6})
    merged_from = set(surviving.get("merged_from", []))
    assert {1, 6}.issubset(merged_from | {int(surviving["track_id"])})
