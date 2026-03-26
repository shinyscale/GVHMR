"""Multi-person capture orchestration.

Ties together detection, segmentation, isolation, shared SLAM computation,
per-person pipeline execution, and world-space assembly.
"""

import json
import subprocess
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
from reprocess_tracking import regenerate_dense_track


def _log_vram(label: str):
    """Print GPU memory usage at a pipeline checkpoint."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[VRAM] {label}: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved")


GVHMR_DIR = Path(__file__).resolve().parent
DEMO_SCRIPT = GVHMR_DIR / "tools" / "demo" / "demo.py"
PERSON_META_FILENAME = "person_meta.json"
TRACKING_VERSION = 4


def _transform_crop_verts_to_fullframe(verts, K_crop, K_full, crop_x1, crop_y1):
    """Transform 3D vertices from cropped camera space to full-frame camera space.

    When a person's video was cropped, GVHMR estimates K from the crop dimensions
    and solves incam params in that space.  To render on the original full-frame
    video we need to re-express the vertices so they project correctly with K_full.
    """
    X, Y, Z = verts[..., 0], verts[..., 1], verts[..., 2]

    fx_c, cx_c = K_crop[0, 0], K_crop[0, 2]
    fy_c, cy_c = K_crop[1, 1], K_crop[1, 2]
    fx_f, cx_f = K_full[0, 0], K_full[0, 2]
    fy_f, cy_f = K_full[1, 1], K_full[1, 2]

    # 2D in full frame: u = fx_c * X/Z + cx_c + crop_x1
    # Want:             u = fx_f * X'/Z + cx_f
    # => X' = (fx_c * X + (cx_c + crop_x1 - cx_f) * Z) / fx_f
    X_new = (fx_c * X + (cx_c + crop_x1 - cx_f) * Z) / fx_f
    Y_new = (fy_c * Y + (cy_c + crop_y1 - cy_f) * Z) / fy_f

    return torch.stack([X_new, Y_new, Z], dim=-1)


def render_multi_person_incam(
    video_path: str,
    person_dirs: list,
    all_tracks: list,
    output_path: str,
    fps: float = 30.0,
    person_colors: list | None = None,
    progress_callback=None,
):
    """Render all tracked people's mesh overlays on the original video.

    For each frame: reads original image, then composites each person's
    SMPL mesh on top using render_mesh() sequentially.

    Handles cropped isolation: if a person's isolated video was smaller than the
    original, their incam vertices are transformed back to full-frame camera space.

    Args:
        video_path: path to original multi-person video
        person_dirs: list of per-person output directories (matches all_tracks order)
        all_tracks: list of track dicts with 'bbx_xyxy' tensors
        output_path: where to write scene_preview.mp4
        fps: output video fps
        person_colors: list of [R, G, B] per person (0-1). Auto-assigned if None.
        progress_callback: fn(frac, msg)
    """
    from hmr4d.utils.video_io_utils import get_video_reader, get_writer
    from hmr4d.utils.vis.renderer import Renderer
    from hmr4d.utils.smplx_utils import make_smplx
    from hmr4d.utils.net_utils import to_cuda

    # Default distinct colors per person
    DEFAULT_COLORS = [
        [0.53, 0.81, 0.92],  # light blue
        [0.98, 0.50, 0.45],  # salmon
        [0.56, 0.93, 0.56],  # light green
        [0.93, 0.79, 0.47],  # gold
        [0.80, 0.60, 0.87],  # lavender
    ]

    # Original video dimensions and K
    length, width, height = get_video_lwh(video_path)
    K_full = estimate_K(width, height)

    # Load SMPL model + conversion matrix once
    smplx_model = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(
        "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        weights_only=False,
    ).cuda()
    faces_smpl = make_smplx("smpl").faces

    # Load per-person results
    CROP_PADDING = 40  # must match _crop_person_video default
    person_data = []
    for i, pdir in enumerate(person_dirs):
        pdir = Path(pdir)
        pt_files = list(pdir.rglob("hmr4d_results.pt"))
        if not pt_files:
            continue
        pred = torch.load(pt_files[0], map_location="cpu", weights_only=False)
        if "smpl_params_incam" not in pred:
            continue

        # Apply pose corrections if they exist for this person
        corr_path = pdir / "pose_corrections.json"
        if corr_path.exists():
            try:
                from pose_correction import CorrectionTrack, apply_corrections
                ct = CorrectionTrack.load_json(str(corr_path))
                if ct.corrections:
                    ic = pred["smpl_params_incam"]
                    n = ic["body_pose"].shape[0]
                    corr_params = {
                        "global_orient": np.array(ic["global_orient"]).reshape(n, 3),
                        "body_pose": np.array(ic["body_pose"]).reshape(n, 21, 3),
                        "transl": np.array(ic["transl"]).reshape(n, 3),
                        "num_frames": n,
                    }
                    corrected = apply_corrections(corr_params, ct)
                    ic["global_orient"] = torch.from_numpy(
                        corrected["global_orient"].reshape(n, 3)).float()
                    ic["body_pose"] = torch.from_numpy(
                        corrected["body_pose"].reshape(n, -1)).float()
                    ic["transl"] = torch.from_numpy(
                        corrected["transl"].reshape(n, 3)).float()
            except Exception as e:
                print(f"[scene_preview] Failed to apply corrections for person {i}: {e}")

        # Compute SMPL vertices in camera space
        smplx_out = smplx_model(**to_cuda(pred["smpl_params_incam"]))
        verts = torch.stack([
            torch.matmul(smplx2smpl, v) for v in smplx_out.vertices
        ])  # (L, 6890, 3)

        K_person = pred["K_fullimg"][0]  # (3, 3)

        # Check if this person's K differs from the original video's K
        # (means their isolated video was cropped, not full-frame inpainted)
        if not torch.allclose(K_person, K_full, atol=1.0):
            # Recover crop offset from track bboxes (mirrors _crop_person_video)
            if i < len(all_tracks):
                bboxes = all_tracks[i]["bbx_xyxy"]
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                crop_x1 = max(0, int(bboxes[:, 0].min()) - CROP_PADDING)
                crop_y1 = max(0, int(bboxes[:, 1].min()) - CROP_PADDING)
                verts = _transform_crop_verts_to_fullframe(
                    verts, K_person, K_full, crop_x1, crop_y1,
                )

        color = (person_colors[i] if person_colors and i < len(person_colors)
                 else DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        person_data.append({"verts": verts, "color": color, "index": i})

    if not person_data:
        return

    # Setup video I/O
    reader = get_video_reader(video_path)
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K_full)
    writer = get_writer(output_path, fps=fps, crf=23)

    for frame_idx, img in enumerate(reader):
        # Composite each person onto this frame
        for pd in person_data:
            if frame_idx < len(pd["verts"]):
                img = renderer.render_mesh(
                    pd["verts"][frame_idx].cuda(),
                    img,
                    pd["color"],
                )

        writer.write_frame(img)

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx / length, f"Rendering frame {frame_idx}/{length}")

    writer.close()
    reader.close()


@dataclass
class MultiPersonResult:
    """Results from the multi-person splitting pipeline."""
    person_video_paths: list[str] = field(default_factory=list)
    person_dirs: list[Path] = field(default_factory=list)
    slam_path: str = ""
    offsets: dict = field(default_factory=dict)
    all_tracks: list = field(default_factory=list)
    inactive_tracks: list = field(default_factory=list)
    masks: dict = field(default_factory=dict)
    num_persons: int = 0
    output_dir: str = ""
    identity_tracks: list = field(default_factory=list)  # IdentityTrack per person
    bridging_summaries: list = field(default_factory=list)


def _track_id(track: dict, fallback: int | None = None) -> int:
    value = track.get("track_id", fallback)
    if value is None:
        raise ValueError("Track is missing track_id")
    return int(value)


def _track_numpy_bboxes(track: dict) -> np.ndarray:
    bboxes = track["bbx_xyxy"]
    if isinstance(bboxes, torch.Tensor):
        return bboxes.cpu().numpy()
    return np.asarray(bboxes)


def _track_detection_mask(track: dict, num_frames: int | None = None) -> np.ndarray:
    mask = track.get("detection_mask")
    if mask is None:
        length = num_frames if num_frames is not None else len(track["bbx_xyxy"])
        return np.ones(length, dtype=bool)
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = np.asarray(mask, dtype=bool)
    if num_frames is not None:
        return mask[:num_frames]
    return mask


def _track_detection_conf(track: dict, num_frames: int | None = None) -> np.ndarray:
    conf = track.get("detection_conf")
    if conf is None:
        length = num_frames if num_frames is not None else len(track["bbx_xyxy"])
        return np.zeros(length, dtype=np.float32)
    if isinstance(conf, torch.Tensor):
        conf = conf.cpu().numpy()
    conf = np.asarray(conf, dtype=np.float32)
    if num_frames is not None:
        return conf[:num_frames]
    return conf


def _track_area_sum(track: dict) -> float:
    boxes = _track_numpy_bboxes(track)
    mask = _track_detection_mask(track, len(boxes))
    if len(boxes) == 0 or not mask.any():
        return 0.0
    valid = boxes[mask]
    wh = np.maximum(valid[:, 2:] - valid[:, :2], 0.0)
    return float((wh[:, 0] * wh[:, 1]).sum())


def _track_real_frame_ids(track: dict) -> np.ndarray:
    mask = _track_detection_mask(track, len(track["bbx_xyxy"]))
    return np.flatnonzero(mask)


def _track_quality_key(track: dict) -> tuple[float, float, float]:
    mask = _track_detection_mask(track, len(track["bbx_xyxy"]))
    conf = _track_detection_conf(track, len(mask))
    mean_conf = float(conf[mask].mean()) if mask.any() else 0.0
    return (float(mask.sum()), _track_area_sum(track), mean_conf)


def _duplicate_track_stats(
    track_a: dict,
    track_b: dict,
    *,
    high_iou_threshold: float = 0.9,
) -> dict:
    boxes_a = _track_numpy_bboxes(track_a)
    boxes_b = _track_numpy_bboxes(track_b)
    mask_a = _track_detection_mask(track_a, len(boxes_a))
    mask_b = _track_detection_mask(track_b, len(boxes_b))
    n = min(len(boxes_a), len(boxes_b), len(mask_a), len(mask_b))
    if n <= 0:
        return {"overlap_frames": 0, "high_iou_frames": 0, "mean_iou": 0.0}

    ious = []
    for f in range(n):
        if not mask_a[f] or not mask_b[f]:
            continue
        ious.append(_bbox_iou(boxes_a[f], boxes_b[f]))
    if not ious:
        return {"overlap_frames": 0, "high_iou_frames": 0, "mean_iou": 0.0}

    ious_arr = np.asarray(ious, dtype=np.float32)
    return {
        "overlap_frames": int(len(ious_arr)),
        "high_iou_frames": int((ious_arr >= high_iou_threshold).sum()),
        "mean_iou": float(ious_arr.mean()),
    }


def _is_duplicate_track_pair(
    track_a: dict,
    track_b: dict,
    *,
    min_overlap_frames: int = 30,
    high_iou_threshold: float = 0.9,
    min_high_iou_fraction: float = 0.8,
    min_mean_iou: float = 0.85,
) -> tuple[bool, dict]:
    stats = _duplicate_track_stats(
        track_a,
        track_b,
        high_iou_threshold=high_iou_threshold,
    )
    overlap_frames = stats["overlap_frames"]
    if overlap_frames < min_overlap_frames:
        return False, stats
    high_fraction = stats["high_iou_frames"] / float(overlap_frames)
    is_dup = high_fraction >= min_high_iou_fraction and stats["mean_iou"] >= min_mean_iou
    return is_dup, stats


def _duplicate_track_ids_for_target(
    all_tracks: list[dict],
    target_index: int,
) -> set[int]:
    if not (0 <= target_index < len(all_tracks)):
        return set()
    target = all_tracks[target_index]
    ignored: set[int] = set()
    for idx, other in enumerate(all_tracks):
        if idx == target_index:
            continue
        is_dup, _ = _is_duplicate_track_pair(target, other)
        if is_dup:
            ignored.add(_track_id(other, idx))
    return ignored


def _merge_duplicate_tracks(
    all_tracks: list[dict],
) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """Merge near-identical duplicate tracks so one dancer keeps one ID."""
    if len(all_tracks) < 2:
        return all_tracks, []

    from person_tracker import merge_tracks

    num_frames = len(_track_numpy_bboxes(all_tracks[0]))
    ordered_indices = sorted(
        range(len(all_tracks)),
        key=lambda idx: _track_quality_key(all_tracks[idx]),
        reverse=True,
    )
    consumed: set[int] = set()
    merged_tracks: list[dict] = []
    merged_pairs: list[tuple[int, int, int]] = []

    for idx in ordered_indices:
        if idx in consumed:
            continue
        primary = all_tracks[idx]
        for jdx in ordered_indices:
            if jdx == idx or jdx in consumed:
                continue
            secondary = all_tracks[jdx]
            is_dup, stats = _is_duplicate_track_pair(primary, secondary)
            if not is_dup:
                continue
            primary = merge_tracks(primary, secondary, num_frames)
            consumed.add(jdx)
            merged_pairs.append(
                (
                    _track_id(primary, idx),
                    _track_id(secondary, jdx),
                    int(stats["high_iou_frames"]),
                )
            )
        merged_tracks.append(primary)

    merged_tracks.sort(key=_track_quality_key, reverse=True)
    return merged_tracks, merged_pairs


def _fragment_stitch_stats(
    earlier_track: dict,
    later_track: dict,
) -> dict:
    earlier_ids = _track_real_frame_ids(earlier_track)
    later_ids = _track_real_frame_ids(later_track)
    if len(earlier_ids) == 0 or len(later_ids) == 0:
        return {
            "gap_frames": None,
            "temporal_overlap_frames": 0,
            "combined_coverage_ratio": 0.0,
            "boundary_y_delta_ratio": float("inf"),
            "boundary_height_ratio": float("inf"),
            "boundary_width_ratio": float("inf"),
        }

    earlier_boxes = _track_numpy_bboxes(earlier_track)
    later_boxes = _track_numpy_bboxes(later_track)
    num_frames = max(len(earlier_boxes), len(later_boxes))

    earlier_set = set(int(f) for f in earlier_ids.tolist())
    later_set = set(int(f) for f in later_ids.tolist())
    temporal_overlap_frames = len(earlier_set & later_set)
    gap_frames = int(later_ids[0] - earlier_ids[-1] - 1)
    combined_coverage_ratio = len(earlier_set | later_set) / float(max(1, num_frames))

    e_window = earlier_boxes[earlier_ids[-10:]]
    l_window = later_boxes[later_ids[:10]]
    e_center = np.stack(
        [(e_window[:, 0] + e_window[:, 2]) / 2, (e_window[:, 1] + e_window[:, 3]) / 2],
        axis=1,
    )
    l_center = np.stack(
        [(l_window[:, 0] + l_window[:, 2]) / 2, (l_window[:, 1] + l_window[:, 3]) / 2],
        axis=1,
    )
    e_size = np.maximum(e_window[:, 2:] - e_window[:, :2], 1.0)
    l_size = np.maximum(l_window[:, 2:] - l_window[:, :2], 1.0)
    mean_height = float((e_size[:, 1].mean() + l_size[:, 1].mean()) / 2.0)
    boundary_y_delta_ratio = float(abs(l_center[:, 1].mean() - e_center[:, 1].mean()) / max(mean_height, 1.0))
    boundary_height_ratio = float(max(e_size[:, 1].mean(), l_size[:, 1].mean()) / max(min(e_size[:, 1].mean(), l_size[:, 1].mean()), 1.0))
    boundary_width_ratio = float(max(e_size[:, 0].mean(), l_size[:, 0].mean()) / max(min(e_size[:, 0].mean(), l_size[:, 0].mean()), 1.0))

    return {
        "gap_frames": gap_frames,
        "temporal_overlap_frames": temporal_overlap_frames,
        "combined_coverage_ratio": combined_coverage_ratio,
        "boundary_y_delta_ratio": boundary_y_delta_ratio,
        "boundary_height_ratio": boundary_height_ratio,
        "boundary_width_ratio": boundary_width_ratio,
    }


def _is_fragment_stitch_candidate(
    earlier_track: dict,
    later_track: dict,
    *,
    max_gap_frames: int = 45,
    max_temporal_overlap_frames: int = 15,
    min_combined_coverage_ratio: float = 0.8,
    max_individual_coverage_ratio: float = 0.75,
    max_boundary_y_delta_ratio: float = 0.2,
    max_boundary_height_ratio: float = 1.8,
    max_boundary_width_ratio: float = 1.8,
) -> tuple[bool, dict]:
    stats = _fragment_stitch_stats(earlier_track, later_track)
    if stats["gap_frames"] is None:
        return False, stats

    num_frames = max(len(earlier_track["bbx_xyxy"]), len(later_track["bbx_xyxy"]))
    earlier_cov = len(_track_real_frame_ids(earlier_track)) / float(max(1, num_frames))
    later_cov = len(_track_real_frame_ids(later_track)) / float(max(1, num_frames))

    is_candidate = (
        0 <= stats["gap_frames"] <= max_gap_frames
        and stats["temporal_overlap_frames"] <= max_temporal_overlap_frames
        and stats["combined_coverage_ratio"] >= min_combined_coverage_ratio
        and earlier_cov <= max_individual_coverage_ratio
        and later_cov <= max_individual_coverage_ratio
        and stats["boundary_y_delta_ratio"] <= max_boundary_y_delta_ratio
        and stats["boundary_height_ratio"] <= max_boundary_height_ratio
        and stats["boundary_width_ratio"] <= max_boundary_width_ratio
    )
    return is_candidate, stats


def _stitch_fragmented_tracks(
    all_tracks: list[dict],
) -> tuple[list[dict], list[tuple[int, int, int]]]:
    """Stitch sequential fragments that likely belong to the same dancer."""
    if len(all_tracks) < 2:
        return all_tracks, []

    from person_tracker import merge_tracks

    num_frames = len(_track_numpy_bboxes(all_tracks[0]))
    ordered_indices = sorted(
        range(len(all_tracks)),
        key=lambda idx: _track_real_frame_ids(all_tracks[idx])[0]
        if len(_track_real_frame_ids(all_tracks[idx])) > 0 else num_frames,
    )
    consumed: set[int] = set()
    stitched_tracks: list[dict] = []
    stitched_pairs: list[tuple[int, int, int]] = []

    for pos, idx in enumerate(ordered_indices):
        if idx in consumed:
            continue
        current = all_tracks[idx]
        current_ids = _track_real_frame_ids(current)
        if len(current_ids) == 0:
            stitched_tracks.append(current)
            continue

        while True:
            current_last = _track_real_frame_ids(current)
            next_match = None
            next_stats = None
            for jdx in ordered_indices:
                if jdx == idx or jdx in consumed:
                    continue
                candidate = all_tracks[jdx]
                candidate_ids = _track_real_frame_ids(candidate)
                if len(candidate_ids) == 0 or candidate_ids[0] <= current_last[-1]:
                    continue
                ok, stats = _is_fragment_stitch_candidate(current, candidate)
                if not ok:
                    continue
                if next_match is None or stats["gap_frames"] < next_stats["gap_frames"]:
                    next_match = jdx
                    next_stats = stats
            if next_match is None:
                break
            candidate = all_tracks[next_match]
            primary, secondary = (
                (current, candidate)
                if _track_quality_key(current) >= _track_quality_key(candidate)
                else (candidate, current)
            )
            current = merge_tracks(primary, secondary, num_frames)
            consumed.add(next_match)
            stitched_pairs.append(
                (
                    _track_id(primary, idx),
                    _track_id(secondary, next_match),
                    int(next_stats["gap_frames"]),
                )
            )
        stitched_tracks.append(current)

    stitched_tracks.sort(key=_track_quality_key, reverse=True)
    return stitched_tracks, stitched_pairs


def _person_meta_path(person_dir: str | Path) -> Path:
    return Path(person_dir) / PERSON_META_FILENAME


def _load_person_meta(person_dir: str | Path) -> dict | None:
    meta_path = _person_meta_path(person_dir)
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def _save_person_meta(person_dir: str | Path, meta: dict) -> Path:
    meta_path = _person_meta_path(person_dir)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _make_person_meta(
    track: dict,
    source_index: int,
    isolation_result: dict | None = None,
    bbox_override_path: str | None = None,
    estimation_backend: str | None = None,
) -> dict:
    meta = {
        "binding_version": 1,
        "track_id": _track_id(track, source_index),
        "source_index": int(source_index),
        "num_detected_frames": int(_track_detection_mask(track).sum()),
    }
    if isolation_result is not None:
        meta["crop_bbox"] = isolation_result.get("crop_bbox")
        meta["isolation_debug"] = isolation_result.get("debug", {})
        meta["num_inpainted"] = isolation_result.get("num_inpainted", 0)
        meta["num_crop_only"] = isolation_result.get("num_crop_only", 0)
        meta["warnings"] = isolation_result.get("warnings", [])
    if bbox_override_path:
        meta["bbox_override_path"] = str(bbox_override_path)
    if estimation_backend:
        meta["estimation_backend"] = str(estimation_backend)
    return meta


def _person_meta_matches_track(meta: dict | None, track: dict, source_index: int) -> bool:
    if not meta:
        return False
    return (
        int(meta.get("binding_version", 0)) >= 1
        and int(meta.get("track_id", -1)) == _track_id(track, source_index)
        and int(meta.get("source_index", -1)) == int(source_index)
    )


def _build_bbox_override_payload(
    bboxes_fullframe: np.ndarray,
    crop_bbox: list[int] | tuple[int, int, int, int] | None,
    output_size: tuple[int, int] | None = None,
) -> dict:
    """Convert full-frame boxes into isolated-video coordinates for demo.py."""
    bboxes = np.asarray(bboxes_fullframe, dtype=np.float32).copy()
    if crop_bbox is not None:
        x1, y1, x2, y2 = [float(v) for v in crop_bbox]
        bboxes[:, [0, 2]] -= x1
        bboxes[:, [1, 3]] -= y1
        if output_size is None:
            output_size = (max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1))))

    if output_size is not None:
        out_w, out_h = output_size
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, max(out_w - 1, 1))
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, max(out_h - 1, 1))

    bbx_xyxy = torch.from_numpy(bboxes).float()
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()
    return {
        "bbx_xyxy": bbx_xyxy,
        "bbx_xys": bbx_xys,
    }


def _save_bbox_override(
    person_dir: str | Path,
    track: dict,
    crop_bbox: list[int] | tuple[int, int, int, int] | None,
    output_size: tuple[int, int] | None = None,
) -> Path:
    override_path = Path(person_dir) / "bbox_override.pt"
    payload = _build_bbox_override_payload(
        _track_numpy_bboxes(track),
        crop_bbox=crop_bbox,
        output_size=output_size,
    )
    torch.save(payload, override_path)
    return override_path


def _invalidate_reprocess_outputs(person_dir: Path, estimation_backend: str) -> None:
    """Remove cached artifacts that can make incremental reruns stale."""
    demo_dir = person_dir / "demo"
    preprocess_dirs = [
        demo_dir / "isolated_video" / "preprocess",
        person_dir / "preprocess",
    ]
    artifact_paths = [
        person_dir / "body.bvh",
        person_dir / "confidence.csv",
        person_dir / "hpe_results.pt",
        person_dir / "0_kp2d77_overlay.mp4",
        person_dir / "isolated_video_1_incam.mp4",
        person_dir / "isolated_video_2_global.mp4",
        person_dir / "isolated_video_3_incam_global_horiz.mp4",
        person_dir / "demo" / "isolated_video" / "hmr4d_results.pt",
        person_dir / "demo" / "isolated_video" / "1_incam.mp4",
        person_dir / "demo" / "isolated_video" / "2_global.mp4",
        person_dir / "demo" / "isolated_video" / "isolated_video_3_incam_global_horiz.mp4",
    ]

    for path in artifact_paths:
        path.unlink(missing_ok=True)
    for preprocess_dir in preprocess_dirs:
        if preprocess_dir.is_dir():
            shutil.rmtree(str(preprocess_dir), ignore_errors=True)


def _export_person_bvh(person_dir: Path) -> Path | None:
    """Regenerate per-person BVH from the freshest solver output."""
    bvh_path = person_dir / "body.bvh"
    # Try GEM-X/SOMA output first.
    hpe_files = list(person_dir.rglob("hpe_results.pt"))
    if hpe_files:
        data = torch.load(str(hpe_files[0]), map_location="cpu", weights_only=False)
        bp = data.get("body_params_incam") or data.get("body_params_global") or {}
        body_pose = bp.get("body_pose")
        if body_pose is not None and body_pose.shape[-1] // 3 > 21:
            n_frames = body_pose.shape[0]
            body_pose_3 = np.array(body_pose).reshape(n_frames, -1, 3)
            go_3 = np.array(bp["global_orient"]).reshape(n_frames, 1, 3)
            poses = np.concatenate([go_3, body_pose_3[:, :76]], axis=1).astype(np.float32)
            soma_params = {
                "poses": poses,
                "transl": np.array(bp["transl"]).astype(np.float32),
            }
            if "identity_coeffs" in bp:
                soma_params["identity_coeffs"] = np.array(bp["identity_coeffs"]).astype(np.float32)
            if "scale_params" in bp:
                soma_params["scale_params"] = np.array(bp["scale_params"]).astype(np.float32)

            bodypipe_dir = GVHMR_DIR.parent / "bodypipe"
            if str(bodypipe_dir) not in sys.path:
                sys.path.insert(0, str(bodypipe_dir))
            from workers.soma_bvh_export import convert_soma_to_bvh

            convert_soma_to_bvh(soma_params, bvh_path)
            return bvh_path

    pt_files = list(person_dir.rglob("hmr4d_results.pt"))
    if not pt_files:
        return None
    from smplx_to_bvh import convert_params_to_bvh, extract_gvhmr_params as _extract_gvhmr

    params = _extract_gvhmr(str(pt_files[0]))
    convert_params_to_bvh(params, str(bvh_path), skip_world_grounding=True)
    return bvh_path


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _collapsed_track_pairs(
    all_tracks: list[dict],
    iou_threshold: float = 0.65,
    min_overlap_frames: int = 10,
) -> list[tuple[int, int, int]]:
    """Detect suspicious track collapse where two tracks follow the same body."""
    suspicious = []
    for i in range(len(all_tracks)):
        boxes_i = _track_numpy_bboxes(all_tracks[i])
        mask_i = _track_detection_mask(all_tracks[i], len(boxes_i))
        tid_i = _track_id(all_tracks[i], i)
        for j in range(i + 1, len(all_tracks)):
            boxes_j = _track_numpy_bboxes(all_tracks[j])
            mask_j = _track_detection_mask(all_tracks[j], len(boxes_j))
            tid_j = _track_id(all_tracks[j], j)
            n = min(len(boxes_i), len(boxes_j), len(mask_i), len(mask_j))
            overlap_frames = 0
            for f in range(n):
                if not mask_i[f] or not mask_j[f]:
                    continue
                iou = _bbox_iou(boxes_i[f], boxes_j[f])
                if iou >= iou_threshold:
                    overlap_frames += 1
            if overlap_frames >= min_overlap_frames:
                suspicious.append((tid_i, tid_j, overlap_frames))
    return suspicious


def compute_shared_slam(
    video_path,
    output_path,
    static_cam=False,
    use_dpvo=False,
    f_mm=None,
):
    """Run SLAM once on the original video. Save results for all per-person runs.

    Must run on the ORIGINAL unmodified video, not inpainted versions.
    """
    output_path = Path(output_path)
    if output_path.exists():
        return str(output_path)

    # Free any lingering GPU memory from prior pipeline stages (YOLO, SAM2, etc.)
    torch.cuda.empty_cache()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if static_cam:
        length = get_video_lwh(video_path)[0]
        # For static cam, save identity rotations
        R_identity = np.eye(4)[None].repeat(length, axis=0).astype(np.float32)
        torch.save(R_identity, str(output_path))
        return str(output_path)

    # Run SLAM via the existing pipeline utilities
    from hmr4d.utils.preproc import SimpleVO
    from hmr4d.utils.geo.hmr_cam import estimate_K, convert_K_to_K4

    # Check DPVO availability — must test the full import chain that slam.py needs,
    # not just dpvo.config. slam.py has a bare `except: pass` that silently leaves
    # cfg undefined if dpvo.dpvo.DPVO fails to import (e.g. build mismatch).
    dpvo_available = False
    if use_dpvo:
        try:
            from dpvo.utils import Timer as _dpvo_timer
            from dpvo.dpvo import DPVO as _dpvo_cls
            from dpvo.config import cfg as _dpvo_cfg
            dpvo_available = True
        except Exception as e:
            print(f"[SLAM] DPVO not available ({e}) — falling back to SimpleVO.")

    if not use_dpvo or not dpvo_available:
        simple_vo = SimpleVO(str(video_path), scale=0.5, step=8, method="sift", f_mm=f_mm)
        vo_results = simple_vo.compute()
        torch.save(vo_results, str(output_path))
    else:
        from hmr4d.utils.preproc.slam import SLAMModel
        from tqdm import tqdm

        length, width, height = get_video_lwh(video_path)
        K_fullimg = estimate_K(width, height)
        intrinsics = convert_K_to_K4(K_fullimg)
        # Scale buffer to video length — DPVO pre-allocates patches_ tensor
        # proportional to buffer size (~0.87 MB per slot). 4000 = 3.5 GB.
        dpvo_buffer = min(max(length + 100, 512), 4000)
        _log_vram("before DPVO init")
        slam = SLAMModel(str(video_path), width, height, intrinsics, buffer=dpvo_buffer, resize=0.5)
        _log_vram(f"after DPVO init (buffer={dpvo_buffer}, {width}x{height} @ 0.5x)")
        bar = tqdm(total=length, desc="DPVO (shared)")
        while True:
            ret = slam.track()
            if ret:
                bar.update()
            else:
                break
        slam_results = slam.process()
        del slam
        torch.cuda.empty_cache()
        torch.save(slam_results, str(output_path))

    return str(output_path)


def run_person_pipeline(
    video_path,
    person_id,
    output_dir,
    slam_override_path=None,
    bbx_override_path=None,
    static_cam=False,
    use_dpvo=False,
    f_mm=None,
    skip_render=False,
    render_incam_only=False,
    force_preprocess=False,
    force_infer=False,
):
    """Run the existing single-person GVHMR pipeline on an isolated person video.

    Args:
        video_path: path to the isolated single-person video
        person_id: person track ID (for logging)
        output_dir: root output directory for this person
        slam_override_path: path to shared SLAM results
        bbx_override_path: path to pre-approved bounding boxes for this person
        static_cam: whether camera is static
        use_dpvo: whether to use DPVO (ignored if slam_override provided)
        f_mm: focal length override
        skip_render: skip all video rendering (solve only)
        render_incam_only: render in-camera overlay only (skip global)

    Returns:
        (pt_path, log_lines) tuple
    """
    output_dir = Path(output_dir)

    # Use GVHMR's own venv Python if available (bodypipe's sys.executable
    # doesn't have torch when running with separate venvs on DGX Spark).
    gvhmr_python = GVHMR_DIR / ".venv" / "bin" / "python"
    python_exe = str(gvhmr_python) if gvhmr_python.exists() else sys.executable

    cmd = [
        python_exe, str(DEMO_SCRIPT),
        f"--video={video_path}",
        f"--output_root={output_dir / 'demo'}",
    ]
    if static_cam:
        cmd.append("--static_cam")
    if use_dpvo and not slam_override_path:
        cmd.append("--use_dpvo")
    if f_mm:
        cmd.append(f"--f_mm={f_mm}")
    if slam_override_path:
        cmd.append(f"--slam_override={slam_override_path}")
    if bbx_override_path:
        cmd.append(f"--bbx_override={bbx_override_path}")
    if skip_render:
        cmd.append("--skip_render")
    elif render_incam_only:
        cmd.append("--render_incam_only")
    if force_preprocess:
        cmd.append("--force_preprocess")
    if force_infer:
        cmd.append("--force_infer")

    log_lines = [f"[Person {person_id}] Running GVHMR: {' '.join(cmd)}", ""]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(GVHMR_DIR),
        bufsize=1,
    )

    for line in proc.stdout:
        log_lines.append(line.rstrip())

    proc.wait()

    if proc.returncode != 0:
        log_lines.append(f"\n[Person {person_id}] ERROR: GVHMR exited with code {proc.returncode}")
        return None, log_lines

    # Find the output .pt file
    video_stem = Path(video_path).stem
    demo_out = output_dir / "demo" / video_stem
    if demo_out.is_dir():
        pt_files = list(demo_out.rglob("hmr4d_results.pt"))
        if pt_files:
            return str(pt_files[0]), log_lines

    log_lines.append(f"\n[Person {person_id}] ERROR: hmr4d_results.pt not found")
    return None, log_lines


def run_gemx_pipeline(
    video_path,
    person_id,
    output_dir,
    static_cam=False,
    skip_render=True,
    bbx_override_path=None,
    force_preprocess=False,
    force_infer=False,
):
    """Run GEM-X (SOMA) estimation on a single isolated video.

    Parallel to run_person_pipeline() but uses GEM-X's demo_soma.py
    to produce SOMA-77 params with identity_coeffs and scale_params.

    Returns:
        (pt_path, log_lines) tuple — path to hpe_results.pt or None
    """
    output_dir = Path(output_dir)
    gemx_root = GVHMR_DIR.parent / "GEM-X"
    demo_script = gemx_root / "scripts" / "demo" / "demo_soma.py"

    if not demo_script.is_file():
        return None, [f"[Person {person_id}] ERROR: GEM-X not found at {gemx_root}"]

    # Use GEM-X venv Python
    gemx_python = gemx_root / ".venv" / "bin" / "python"
    python_exe = str(gemx_python) if gemx_python.exists() else sys.executable

    # GEM-X output goes into person_dir directly (output_root)
    cmd = [
        python_exe, str(demo_script),
        f"--video={video_path}",
        f"--output_root={output_dir}",
    ]
    if bbx_override_path:
        cmd.append(f"--bbx_override={bbx_override_path}")
    if static_cam:
        cmd.append("--static_cam")
    if force_preprocess:
        cmd.append("--force_preprocess")
    if force_infer:
        cmd.append("--force_infer")

    log_lines = [f"[Person {person_id}] Running GEM-X: {' '.join(cmd)}", ""]

    # GEM-X needs its own PYTHONPATH for submodule imports
    import os
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra_paths = [str(gemx_root)]
    # Add GEM-X third_party dirs
    tp = gemx_root / "third_party"
    if tp.is_dir():
        for d in tp.iterdir():
            if d.is_dir() and (d / "__init__.py").exists() or (d / "setup.py").exists():
                extra_paths.append(str(d))
    env["PYTHONPATH"] = ":".join(extra_paths + ([existing] if existing else []))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(gemx_root),
        env=env,
        bufsize=1,
    )

    for line in proc.stdout:
        log_lines.append(line.rstrip())

    proc.wait()

    if proc.returncode != 0:
        log_lines.append(f"\n[Person {person_id}] ERROR: GEM-X exited with code {proc.returncode}")
        return None, log_lines

    # Find hpe_results.pt in output
    for pt_file in sorted(output_dir.rglob("hpe_results.pt")):
        return str(pt_file), log_lines

    log_lines.append(f"\n[Person {person_id}] ERROR: hpe_results.pt not found")
    return None, log_lines


def split_multi_person_video(
    video_path,
    output_dir,
    min_track_duration=30,
    static_cam=False,
    use_dpvo=False,
    f_mm=None,
    max_persons=0,
    render_overlays=False,
    use_inpainting=False,
    progress_callback=None,
    estimation_backend="gvhmr",
):
    """Full multi-person isolation pipeline.

    Steps:
        1. Detect & track all people (OC-SORT / YOLO fallback)
        2. Compute shared camera motion (SLAM) once on original video
        3. (If use_inpainting) Segment all people (SAM2) + inpaint overlaps
           (If not) Bbox crop with blackout — fast path, usually sufficient
        4. Per-person: run existing GVHMR pipeline with shared SLAM
        5. Compute relative offsets and assemble world-space results

    Args:
        video_path: path to input multi-person video
        output_dir: base output directory
        min_track_duration: minimum frames to consider a track
        static_cam: whether camera is stationary
        use_dpvo: use DPVO for camera estimation
        f_mm: focal length in mm (None for auto)
        max_persons: limit to N largest people (0 = all)
        render_overlays: render per-person mesh overlays
        use_inpainting: use SAM2 + ProPainter for pixel-accurate isolation
            (slow but better for heavy occlusion). When False, uses fast
            bbox-crop-and-blackout which is sufficient for most crossings.
        progress_callback: fn(frac, msg) for progress updates

    Returns:
        MultiPersonResult
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = MultiPersonResult(output_dir=str(output_dir))
    log_lines = []
    _log_vram("pipeline start")

    def _progress(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)
        log_lines.append(f"[{frac:.0%}] {msg}")

    # ── Step 1: Detection & Tracking ──
    _log_vram("before tracking")
    _progress(0.05, "Detecting and tracking all people...")

    detection_dir = output_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = detection_dir / "all_tracks.pt"

    all_tracks = None
    all_tracks_data = None
    cache_reason = None
    current_tracker_backend = "unknown"
    if tracks_path.exists():
        tracks_data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
        cached_tracks = tracks_data["tracks"] if isinstance(tracks_data, dict) and "tracks" in tracks_data else tracks_data
        cache_meta = tracks_data.get("metadata", {}) if isinstance(tracks_data, dict) else {}

        from person_tracker import OCSORT_AVAILABLE
        current_tracker_backend = "ocsort" if OCSORT_AVAILABLE else "yolo"
        cached_backend = cache_meta.get("tracker_backend")
        cached_version = int(cache_meta.get("tracking_version", 0))

        if cached_version < TRACKING_VERSION:
            cache_reason = "cached tracks predate current duplicate/fragment merge fixes"
        elif cached_backend == "yolo" and current_tracker_backend == "ocsort":
            cache_reason = "cached tracks were generated without OC-SORT"
        else:
            all_tracks = cached_tracks
            _progress(0.10, f"Loaded {len(all_tracks)} cached tracks ({cached_backend or 'legacy cache'})")

    tracks_cache_dirty = False
    if all_tracks is None:
        from person_tracker import PersonTracker, render_track_visualization

        if cache_reason:
            _progress(0.08, f"Ignoring cached tracks: {cache_reason}")
        tracker = PersonTracker()
        current_tracker_backend = tracker.tracker_backend
        all_tracks = tracker.detect_and_track_all(
            video_path, min_track_frames=min_track_duration
        )
        torch.save(
            {"tracks": all_tracks, "metadata": tracker.cache_metadata()},
            str(tracks_path),
        )
        tracks_cache_dirty = True
        del tracker
        torch.cuda.empty_cache()

        # Render track visualization
        viz_path = detection_dir / "track_visualization.mp4"
        render_track_visualization(video_path, all_tracks, str(viz_path))

        _progress(0.10, f"Detected {len(all_tracks)} people ({current_tracker_backend})")

    merged_tracks, merged_pairs = _merge_duplicate_tracks(all_tracks)
    if len(merged_tracks) != len(all_tracks):
        summary = ", ".join(f"{a}<-{b} ({frames} hi-IoU frames)" for a, b, frames in merged_pairs)
        _progress(0.10, f"Merged duplicate track fragments: {summary}")
        all_tracks = merged_tracks
        tracks_cache_dirty = True

    stitched_tracks, stitched_pairs = _stitch_fragmented_tracks(all_tracks)
    if len(stitched_tracks) != len(all_tracks):
        summary = ", ".join(f"{a}<-{b} (gap {gap}f)" for a, b, gap in stitched_pairs)
        _progress(0.10, f"Stitched sequential track fragments: {summary}")
        all_tracks = stitched_tracks
        tracks_cache_dirty = True

    if tracks_cache_dirty:
        if all_tracks_data is None and tracks_path.exists():
            all_tracks_data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
        cache_metadata = (
            all_tracks_data.get("metadata", {})
            if isinstance(all_tracks_data, dict)
            else {}
        )
        cache_metadata = dict(cache_metadata)
        cache_metadata["tracking_version"] = max(TRACKING_VERSION, int(cache_metadata.get("tracking_version", 0)))
        cache_metadata["duplicate_merge_count"] = len(merged_pairs)
        cache_metadata["fragment_stitch_count"] = len(stitched_pairs)
        torch.save({"tracks": all_tracks, "metadata": cache_metadata}, str(tracks_path))

    # Cap to max_persons largest tracks (already sorted by area)
    inactive_tracks = []
    if max_persons > 0 and len(all_tracks) > max_persons:
        _progress(0.10, f"Capping from {len(all_tracks)} tracks to {max_persons} largest")
        inactive_tracks = all_tracks[max_persons:]
        all_tracks = all_tracks[:max_persons]

    result.all_tracks = all_tracks
    result.inactive_tracks = inactive_tracks
    result.num_persons = len(all_tracks)

    collapsed_pairs = _collapsed_track_pairs(all_tracks)
    if collapsed_pairs:
        summary = ", ".join(
            f"{a}<->{b} ({frames} frames)" for a, b, frames in collapsed_pairs
        )
        _progress(0.11, f"WARNING: Suspicious track overlap detected: {summary}")

    if len(all_tracks) == 0:
        _progress(1.0, "No people detected in video.")
        return result

    if len(all_tracks) == 1:
        _progress(0.10, "Only 1 person detected — using standard single-person pipeline.")
        # Fall through to single-person processing below

    # ── Step 2: Shared SLAM ──
    _log_vram("before SLAM")
    _progress(0.12, "Computing shared camera motion (SLAM)...")

    slam_path = output_dir / "shared_slam.pt"
    compute_shared_slam(video_path, slam_path, static_cam, use_dpvo, f_mm)
    result.slam_path = str(slam_path)

    _progress(0.20, "SLAM complete")

    _log_vram("after SLAM")

    # ── Step 3: Segmentation (only if >1 person AND inpainting enabled) ──
    masks = {}
    if len(all_tracks) > 1 and not use_inpainting:
        _progress(0.22, "Using fast bbox-crop isolation (unsafe for crossings)")
        _progress(0.35, "Skipped segmentation — bbox blackout mode may swap/erase overlapping people")
    elif len(all_tracks) > 1:
        _progress(0.22, "Segmenting all people with SAM2...")

        masks_dir = output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached masks
        cached = all(
            (masks_dir / f"person_{t['track_id']}_masks.npz").exists()
            for t in all_tracks
        )

        if cached:
            for t in all_tracks:
                tid = t["track_id"]
                mask_data = np.load(str(masks_dir / f"person_{tid}_masks.npz"))
                masks[tid] = mask_data["masks"]
            _progress(0.35, f"Loaded cached masks for {len(masks)} people")
        else:
            try:
                from sam2_segmenter import SAM2Segmenter

                segmenter = SAM2Segmenter()
                masks = segmenter.segment_all_persons(video_path, all_tracks)
                del segmenter
                torch.cuda.empty_cache()

                # Save masks
                for tid, mask_array in masks.items():
                    np.savez_compressed(
                        str(masks_dir / f"person_{tid}_masks.npz"),
                        masks=mask_array,
                    )

                # Compute and save overlap map
                overlap = _compute_overlap_map(masks, all_tracks)
                np.savez_compressed(str(masks_dir / "overlap_map.npz"), overlap=overlap)

                _progress(0.35, f"Segmented {len(masks)} people")
            except Exception as e:
                if use_inpainting:
                    raise RuntimeError(
                        "SAM2 inpainting was requested, but segmentation could not start. "
                        f"Underlying error: {e!r}. "
                        "Install/configure SAM2 instead of falling back to bbox blackout."
                    ) from e
                _progress(0.35, f"SAM2 not available ({e}). Using bbox-only isolation (no inpainting).")
                masks = {}

        result.masks = masks

    _log_vram("after segmentation")

    # ── Step 4: Per-person isolation ──
    _progress(0.37, "Isolating per-person videos...")

    person_dirs = []
    person_video_paths = []
    person_meta_by_tid = {}
    _shared_inpainter = None  # Shared across persons, freed after loop

    for i, track in enumerate(all_tracks):
        tid = track["track_id"]
        person_dir = output_dir / f"person_{i}"
        person_dir.mkdir(parents=True, exist_ok=True)
        person_dirs.append(person_dir)

        isolated_video = person_dir / "isolated_video.mp4"
        isolation_mode_path = person_dir / "isolation_mode.json"
        meta = _load_person_meta(person_dir)
        can_reuse_isolation = isolated_video.exists()
        if can_reuse_isolation and not _person_meta_matches_track(meta, track, i):
            can_reuse_isolation = False
            _progress(
                0.37 + (i + 1) / len(all_tracks) * 0.13,
                f"Person {i} cache belongs to a different track; regenerating",
            )
        if can_reuse_isolation and use_inpainting and len(all_tracks) > 1:
            can_reuse_isolation = isolation_mode_path.exists()
            if not can_reuse_isolation:
                _progress(
                    0.37 + (i + 1) / len(all_tracks) * 0.13,
                    f"Person {i} isolation cache missing mode log; regenerating",
                )

        if can_reuse_isolation:
            crop_bbox = meta.get("crop_bbox") if meta else None
            bbox_override_path = _save_bbox_override(person_dir, track, crop_bbox=crop_bbox)
            if meta:
                meta["bbox_override_path"] = str(bbox_override_path)
                _save_person_meta(person_dir, meta)
                person_meta_by_tid[tid] = meta
            person_video_paths.append(str(isolated_video))
            _progress(
                0.37 + (i + 1) / len(all_tracks) * 0.13,
                f"Person {i} already isolated",
            )
            continue

        isolation_result = None
        if len(all_tracks) == 1:
            # Single person — just symlink/copy the original video
            shutil.copy2(video_path, str(isolated_video))
            person_video_paths.append(str(isolated_video))
            isolation_result = {
                "crop_bbox": None,
                "debug": {"crop_area_ratio": 1.0, "trusted_crop_frames": int(_track_detection_mask(track).sum())},
                "num_inpainted": 0,
                "num_crop_only": len(_track_numpy_bboxes(track)),
                "warnings": [],
            }
        elif masks:
            # Multi-person with masks — smart isolation
            try:
                from propainter_inpaint import isolate_person

                if _shared_inpainter is None:
                    from propainter_inpaint import ProPainterInpainter
                    _shared_inpainter = ProPainterInpainter()

                isolation_result = isolate_person(
                    video_path=video_path,
                    target_person_id=tid,
                    target_track_bboxes=track["bbx_xyxy"].cpu().numpy(),
                    target_detection_mask=_track_detection_mask(track, len(track["bbx_xyxy"])),
                    all_masks=masks,
                    output_path=str(isolated_video),
                    inpainter=_shared_inpainter,
                )

                # Save isolation mode log
                with open(person_dir / "isolation_mode.json", "w") as f:
                    json.dump(isolation_result.get("isolation_modes", []), f)

                with open(person_dir / "isolation_debug.json", "w") as f:
                    json.dump(isolation_result.get("debug", {}), f, indent=2)

                person_video_paths.append(str(isolated_video))
            except Exception as e:
                if use_inpainting:
                    raise RuntimeError(
                        "SAM2/ProPainter inpainting was requested, but isolation fell back before producing "
                        "an inpainted clip. Fix ProPainter/dependencies instead of using bbox blackout."
                    ) from e
                # No ProPainter — crop with blackout of other people
                other_bb = [
                    all_tracks[j]["bbx_xyxy"].cpu().numpy()
                    for j in range(len(all_tracks)) if j != i
                ]
                isolation_result = _crop_person_video(
                    video_path, track["bbx_xyxy"].cpu().numpy(),
                    str(isolated_video), other_bboxes=other_bb,
                    detection_mask=_track_detection_mask(track, len(track["bbx_xyxy"])),
                )
                person_video_paths.append(str(isolated_video))
        else:
            # No masks available — crop with blackout of other people
            other_bb = [
                all_tracks[j]["bbx_xyxy"].cpu().numpy()
                for j in range(len(all_tracks)) if j != i
            ]
            isolation_result = _crop_person_video(
                video_path, track["bbx_xyxy"].cpu().numpy(),
                str(isolated_video), other_bboxes=other_bb,
                detection_mask=_track_detection_mask(track, len(track["bbx_xyxy"])),
            )
            person_video_paths.append(str(isolated_video))

        crop_bbox = isolation_result.get("crop_bbox") if isolation_result else None
        bbox_override_path = _save_bbox_override(person_dir, track, crop_bbox=crop_bbox)
        meta = _make_person_meta(
            track,
            source_index=i,
            isolation_result=isolation_result,
            bbox_override_path=str(bbox_override_path),
            estimation_backend=estimation_backend,
        )
        _save_person_meta(person_dir, meta)
        person_meta_by_tid[tid] = meta
        if isolation_result is not None:
            with open(person_dir / "isolation_debug.json", "w") as f:
                json.dump(isolation_result.get("debug", {}), f, indent=2)

        debug = (isolation_result or {}).get("debug", {})
        crop_ratio = float(debug.get("crop_area_ratio", 0.0))
        if crop_ratio >= 0.85:
            _progress(
                0.37 + (i + 1) / len(all_tracks) * 0.13,
                f"WARNING: Person {i} crop covers {crop_ratio:.0%} of frame; isolation may be contaminated",
            )
        if "low_target_coverage" in (isolation_result or {}).get("warnings", []):
            _progress(
                0.37 + (i + 1) / len(all_tracks) * 0.13,
                f"WARNING: Person {i} target mask coverage is very low; GVHMR may still lock onto the wrong performer",
            )

        _progress(
            0.37 + (i + 1) / len(all_tracks) * 0.13,
            f"Person {i} isolated",
        )

    # Free SAM2 masks from RAM (saved to disk, can reload if needed)
    masks.clear()

    # Free ProPainter GPU memory before GVHMR runs
    if _shared_inpainter is not None:
        _shared_inpainter.cleanup()
        del _shared_inpainter
        _shared_inpainter = None
        torch.cuda.empty_cache()

    result.person_video_paths = person_video_paths
    result.person_dirs = person_dirs

    _log_vram("after isolation")

    # ── Step 5: Per-person pipeline execution ──
    _progress(0.50, "Running per-person GVHMR pipelines...")

    for i, (person_video, person_dir) in enumerate(zip(person_video_paths, person_dirs)):
        track = all_tracks[i]
        tid = track["track_id"]
        meta = _load_person_meta(person_dir)
        crop_bbox = meta.get("crop_bbox") if meta else None
        bbox_override_path = _save_bbox_override(person_dir, track, crop_bbox=crop_bbox)
        if meta is not None:
            meta["bbox_override_path"] = str(bbox_override_path)
            _save_person_meta(person_dir, meta)
            person_meta_by_tid[tid] = meta

        # Check if already processed (either GVHMR or GEM-X output)
        existing_pt = list(person_dir.rglob("hmr4d_results.pt")) + list(person_dir.rglob("hpe_results.pt"))
        isolated_mtime = Path(person_video).stat().st_mtime if Path(person_video).exists() else 0.0
        newest_pt_mtime = max((pt.stat().st_mtime for pt in existing_pt), default=0.0)
        if existing_pt and newest_pt_mtime >= isolated_mtime:
            _progress(
                0.50 + (i + 1) / len(all_tracks) * 0.30,
                f"Person {i} already processed",
            )
            continue
        if existing_pt:
            _progress(
                0.50 + i / len(all_tracks) * 0.30,
                f"Person {i} isolation changed; re-running estimation",
            )

        backend_label = "GEM-X" if estimation_backend == "gemx" else "GVHMR"
        _progress(
            0.50 + i / len(all_tracks) * 0.30,
            f"Running {backend_label} for person {i}...",
        )

        if estimation_backend == "gemx":
            pt_path, person_log = run_gemx_pipeline(
                video_path=person_video,
                person_id=tid,
                output_dir=person_dir,
                static_cam=static_cam,
                bbx_override_path=str(bbox_override_path),
            )
        else:
            pt_path, person_log = run_person_pipeline(
                video_path=person_video,
                person_id=tid,
                output_dir=person_dir,
                slam_override_path=str(slam_path) if not static_cam else None,
                bbx_override_path=str(bbox_override_path),
                static_cam=static_cam,
                use_dpvo=use_dpvo,
                f_mm=f_mm,
                skip_render=not render_overlays,
                render_incam_only=render_overlays,
            )

        if pt_path is None:
            log_lines.extend(person_log)
            _progress(
                0.50 + (i + 1) / len(all_tracks) * 0.30,
                f"Person {i} {backend_label} FAILED",
            )

    # ── Step 5.5: Identity verification & occlusion bridging ──
    _progress(0.82, "Running identity verification...")

    try:
        from identity_confidence import compute_all_confidences, export_confidence_csv
        from identity_tracking import IdentityTrack, auto_generate_keyframes
        from identity_reid import ShapeReIdentifier
        from identity_bridge import (
            OcclusionBridge,
            crossing_spans_from_overlap,
            crossing_spans_from_signal,
            summarize_bridging,
        )
        from smplx_to_bvh import extract_gvhmr_params

        identity_tracks = []
        bridging_summaries = []

        for i, person_dir in enumerate(person_dirs):
            if i >= len(all_tracks):
                break

            track = all_tracks[i]
            tid = track["track_id"]

            # Load GVHMR results for this person
            pt_files = list(person_dir.rglob("hmr4d_results.pt"))
            if not pt_files:
                continue

            params = extract_gvhmr_params(str(pt_files[0]))
            per_frame_betas = params["betas"]  # (N, 10)
            num_person_frames = params["num_frames"]

            # Load per-person ViTPose if available
            vitpose_files = list(person_dir.rglob("vitpose.pt"))
            vitpose = None
            if vitpose_files:
                vp = torch.load(str(vitpose_files[0]), map_location="cpu", weights_only=False)
                if isinstance(vp, torch.Tensor):
                    vitpose = vp.numpy()
                elif isinstance(vp, np.ndarray):
                    vitpose = vp

            # Compute confidence scores
            ignored_track_ids = _duplicate_track_ids_for_target(all_tracks, i)
            mask_overlap = _compute_track_mask_overlap(
                target_person_id=tid,
                all_masks=masks,
                num_frames=num_person_frames,
                ignored_track_ids=ignored_track_ids,
            )
            initial_confidences = compute_all_confidences(
                track_idx=i,
                all_tracks=all_tracks,
                num_frames=num_person_frames,
                vitpose=vitpose,
                per_frame_betas=per_frame_betas,
                overlap_signal=mask_overlap,
                ignored_track_ids=ignored_track_ids,
            )

            # Create identity track and auto-generate keyframes
            id_track = IdentityTrack(person_id=tid)
            bboxes = track["bbx_xyxy"].numpy() if hasattr(track["bbx_xyxy"], "numpy") else track["bbx_xyxy"]
            bboxes = bboxes[:num_person_frames]

            auto_generate_keyframes(
                track=id_track,
                confidences=initial_confidences,
                per_frame_betas=per_frame_betas,
                per_frame_bboxes=bboxes,
                per_frame_poses=params.get("body_pose", params.get("body_pose_world")),
            )
            id_track.establish_identity()
            verified_frames = {
                kf.frame_index for kf in id_track.keyframes
                if getattr(kf, "verified", False)
            }
            confidences = compute_all_confidences(
                track_idx=i,
                all_tracks=all_tracks,
                num_frames=num_person_frames,
                vitpose=vitpose,
                per_frame_betas=per_frame_betas,
                established_betas=id_track.established_betas,
                overlap_signal=mask_overlap,
                verified_frames=verified_frames,
                stable_reference_frames=len(id_track.keyframes),
                ignored_track_ids=ignored_track_ids,
            )
            for kf in id_track.keyframes:
                if 0 <= kf.frame_index < len(confidences):
                    kf.confidence = confidences[kf.frame_index]

            # Bridge low-confidence spans
            bridge = OcclusionBridge(id_track)
            body_pose = params.get("body_pose_world", params.get("body_pose"))
            global_orient = params.get("global_orient_world", params.get("global_orient"))
            transl = params.get("transl_world", params.get("transl"))

            if body_pose is not None and global_orient is not None and transl is not None:
                if mask_overlap is not None:
                    crossing_spans = crossing_spans_from_signal(mask_overlap, threshold=0.10)
                else:
                    crossing_spans = crossing_spans_from_overlap(confidences)

                # Merge manual crossing spans from user annotation
                manual_spans_path = person_dir / "crossing_spans.json"
                if manual_spans_path.exists():
                    try:
                        manual = json.loads(manual_spans_path.read_text())
                        manual_tuples = [tuple(s) for s in manual]
                        crossing_spans = _merge_crossing_spans(crossing_spans + manual_tuples)
                    except Exception:
                        pass

                if crossing_spans:
                    total_crossing = sum(e - s + 1 for s, e in crossing_spans)
                    _progress(0.82 + i / max(len(person_dirs), 1) * 0.03,
                              f"Person {i}: {len(crossing_spans)} crossing spans detected, "
                              f"bridging {total_crossing} frames")
                    bridge_result = bridge.bridge_with_spans(
                        body_pose, global_orient, transl, crossing_spans)
                else:
                    bridge_result = bridge.bake(body_pose, global_orient, transl, confidences)
                bridged_frames = bridge_result["bridged_frames"]

                # Determine if we need to modify pt_data
                verified_kfs = [kf for kf in id_track.keyframes
                                if kf.bbox is not None and kf.verified]
                need_position_constraints = bool(verified_kfs)
                need_save = bool(bridged_frames) or need_position_constraints

                if need_save:
                    pt_data = torch.load(str(pt_files[0]), map_location="cpu", weights_only=False)

                    if bridged_frames:
                        g = pt_data["smpl_params_global"]

                        bridged_bp = bridge_result["body_pose"]     # (N, 21, 3)
                        bridged_go = bridge_result["global_orient"]  # (N, 3)
                        bridged_tr = bridge_result["transl"]         # (N, 3)

                        g["body_pose"] = torch.from_numpy(bridged_bp.reshape(num_person_frames, -1)).float()
                        g["global_orient"] = torch.from_numpy(bridged_go.reshape(num_person_frames, -1)).float()
                        g["transl"] = torch.from_numpy(bridged_tr).float()

                        # Also bridge smpl_params_incam so scene preview shows corrected poses
                        if crossing_spans and "smpl_params_incam" in pt_data:
                            ic = pt_data["smpl_params_incam"]
                            ic_bp = np.array(ic["body_pose"]).reshape(num_person_frames, -1, 3)
                            ic_go = np.array(ic["global_orient"]).reshape(num_person_frames, 3)
                            ic_tr = np.array(ic["transl"]).reshape(num_person_frames, 3)
                            ic_result = bridge.bridge_with_spans(ic_bp, ic_go, ic_tr, crossing_spans)
                            ic["body_pose"] = torch.from_numpy(
                                ic_result["body_pose"].reshape(num_person_frames, -1)).float()
                            ic["global_orient"] = torch.from_numpy(
                                ic_result["global_orient"].reshape(num_person_frames, -1)).float()
                            ic["transl"] = torch.from_numpy(ic_result["transl"]).float()

                    # Apply identity keyframe position constraints
                    if need_position_constraints:
                        from world_assembly import apply_identity_position_constraints
                        K_fullimg = pt_data.get("K_fullimg")
                        if K_fullimg is not None:
                            K_np = np.array(K_fullimg[0])
                        else:
                            _, w, h = get_video_lwh(video_path)
                            K_np = estimate_K(w, h).numpy()
                        apply_identity_position_constraints(
                            pt_data, verified_kfs, K_np, num_person_frames)

                    torch.save(pt_data, str(pt_files[0]))

                    parts = []
                    if bridged_frames:
                        parts.append(f"{len(bridged_frames)} frames bridged")
                    if need_position_constraints:
                        parts.append(f"{len(verified_kfs)} position constraints applied")
                    _progress(0.82 + i / max(len(person_dirs), 1) * 0.03,
                              f"Person {i}: {', '.join(parts)}, saved")

                summary = summarize_bridging(confidences, bridged_frames)
                bridging_summaries.append(summary)

                _progress(0.82 + i / max(len(person_dirs), 1) * 0.03,
                          f"Person {i}: {summary['bridged_frames']} frames bridged, "
                          f"mean conf {summary['confidence_mean']:.2f}")
            else:
                bridged_frames = set()
                bridging_summaries.append({})

            # Save identity track and confidence CSV
            id_track.save_json(person_dir / "identity_track.json")
            export_confidence_csv(
                confidences, tid,
                str(person_dir / "confidence.csv"),
                bridged_frames=bridged_frames,
            )

            identity_tracks.append(id_track)

        result.identity_tracks = identity_tracks
        result.bridging_summaries = bridging_summaries

        # Shape re-identification (detect swaps)
        if len(identity_tracks) >= 2:
            reid = ShapeReIdentifier(identity_tracks)
            per_person_tail_betas = {
                t.person_id: t.keyframes[-1].betas
                for t in identity_tracks
                if t.keyframes and t.keyframes[-1].betas is not None
            }
            swaps = reid.detect_swap(0, per_person_tail_betas)
            if swaps:
                _progress(0.84, f"WARNING: Possible identity swaps detected: {swaps}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _progress(0.84, f"Identity verification skipped: {e}")

    # ── Step 5.7: Generate per-person BVH ──
    _progress(0.85, "Generating BVH files...")

    for i, person_dir in enumerate(person_dirs):
        try:
            bvh_path = _export_person_bvh(person_dir)
            if bvh_path is None:
                continue
            _progress(0.85 + (i + 1) / len(person_dirs) * 0.05,
                      f"Person {i} BVH exported")
        except Exception as e:
            _progress(0.85 + (i + 1) / len(person_dirs) * 0.05,
                      f"Person {i} BVH failed: {e}")

    # ── Step 6: World-space assembly ──
    _progress(0.90, "Assembling world-space results...")

    from world_assembly import assemble_scene, save_session_manifest

    # Get camera K
    _, width, height = get_video_lwh(video_path)
    camera_K = estimate_K(width, height).numpy()

    assembly_result = assemble_scene(
        person_dirs=person_dirs,
        all_tracks=all_tracks,
        camera_K=camera_K,
        output_dir=output_dir,
    )
    result.offsets = assembly_result["offsets"]

    # Save manifest
    save_session_manifest(
        output_dir=output_dir,
        video_path=video_path,
        num_persons=result.num_persons,
        all_tracks=all_tracks,
        offsets=result.offsets,
        person_dirs=person_dirs,
        slam_path=str(slam_path),
        person_bindings=[
            {
                "person_dir": str(person_dir),
                **(_load_person_meta(person_dir) or {
                    "track_id": _track_id(all_tracks[i], i),
                    "source_index": i,
                }),
            }
            for i, person_dir in enumerate(person_dirs)
            if i < len(all_tracks)
        ],
    )

    # ── Step 7: Render combined incam overlay ──
    _progress(0.92, "Rendering scene preview...")
    try:
        assembly_dir = output_dir / "assembly"
        assembly_dir.mkdir(parents=True, exist_ok=True)
        scene_preview_path = str(assembly_dir / "scene_preview.mp4")
        render_multi_person_incam(
            video_path=video_path,
            person_dirs=person_dirs,
            all_tracks=all_tracks,
            output_path=scene_preview_path,
            fps=30.0,
            progress_callback=lambda f, m: _progress(0.92 + f * 0.07, m),
        )
    except Exception as e:
        _progress(0.99, f"Scene preview failed: {e}")

    _progress(1.0, f"Multi-person pipeline complete. {result.num_persons} people processed.")

    return result


def reprocess_person(
    video_path: str,
    person_index: int,
    person_dir: str,
    original_bboxes: np.ndarray,
    all_tracks: list[dict],
    slam_path: str,
    masks_dir: str,
    static_cam: bool = False,
    use_dpvo: bool = False,
    progress_callback=None,
    estimation_backend: str = "gvhmr",
    manual_bbox_keyframes: dict[int, np.ndarray] | None = None,
    verified_identity_keyframes: list[dict] | None = None,
) -> dict:
    """Re-run isolation + estimation for a single person after bbox corrections.

    Caches shared SLAM and SAM2 masks. Only re-runs isolation + pipeline.
    Supports both GVHMR and GEM-X backends via estimation_backend param.

    Returns dict with keys: identity_track, confidences, pt_path (or empty on failure).
    """
    person_dir = Path(person_dir)
    person_dir.mkdir(parents=True, exist_ok=True)

    track = all_tracks[person_index]
    tid = track.get("track_id", person_index)
    regenerated = regenerate_dense_track(
        all_tracks=all_tracks,
        target_index=person_index,
        original_bboxes=np.asarray(original_bboxes, dtype=np.float32),
        manual_bbox_keyframes=manual_bbox_keyframes,
        verified_identity_keyframes=verified_identity_keyframes,
    )
    track = regenerated.as_track_dict(track)
    all_tracks[person_index] = track
    updated_bboxes = regenerated.boxes

    print(f"[reprocess_person] START person_index={person_index}, tid={tid}, "
          f"backend={estimation_backend}, bboxes={updated_bboxes.shape}, "
          f"anchors={len(regenerated.anchor_frames)}")

    # 1. Update track bboxes in-place
    track["bbx_xyxy"] = torch.from_numpy(updated_bboxes)

    # 2. Back up stale outputs (restore on failure, delete on success)
    isolated_video = person_dir / "isolated_video.mp4"
    demo_dir = person_dir / "demo"
    bak_video = person_dir / "isolated_video.mp4.bak"
    bak_demo = person_dir / "demo.bak"
    if isolated_video.exists():
        isolated_video.rename(bak_video)
    if demo_dir.exists():
        demo_dir.rename(bak_demo)
    _invalidate_reprocess_outputs(person_dir, estimation_backend)

    # 3. Re-isolate person
    if progress_callback:
        progress_callback(0.1, f"Re-isolating person {person_index}...")

    # Load ALL people's cached SAM2 masks so isolate_person can black out others
    masks = {}
    masks_dir_path = Path(masks_dir)
    for t in all_tracks:
        other_tid = t.get("track_id", t.get("id"))
        mask_file = masks_dir_path / f"person_{other_tid}_masks.npz"
        if mask_file.exists():
            mask_data = np.load(str(mask_file))
            masks[other_tid] = mask_data["masks"]

    # Collect other people's bboxes for blackout during crop isolation
    other_bb = [
        all_tracks[j]["bbx_xyxy"].cpu().numpy()
        if isinstance(all_tracks[j]["bbx_xyxy"], torch.Tensor)
        else all_tracks[j]["bbx_xyxy"]
        for j in range(len(all_tracks)) if j != person_index
    ]
    isolation_result = None

    if len(all_tracks) == 1:
        shutil.copy2(video_path, str(isolated_video))
        isolation_result = {
            "crop_bbox": None,
            "debug": {"crop_area_ratio": 1.0},
            "num_inpainted": 0,
            "num_crop_only": len(updated_bboxes),
            "warnings": [],
        }
    elif masks:
        try:
            from propainter_inpaint import isolate_person

            isolation_result = isolate_person(
                video_path=video_path,
                target_person_id=tid,
                target_track_bboxes=updated_bboxes,
                target_detection_mask=_track_detection_mask(track, len(updated_bboxes)),
                all_masks=masks,
                output_path=str(isolated_video),
            )
        except ImportError:
            isolation_result = _crop_person_video(
                video_path, updated_bboxes, str(isolated_video),
                other_bboxes=other_bb, detection_mask=_track_detection_mask(track, len(updated_bboxes)),
            )
    else:
        isolation_result = _crop_person_video(
            video_path, updated_bboxes, str(isolated_video),
            other_bboxes=other_bb, detection_mask=_track_detection_mask(track, len(updated_bboxes)),
        )

    if isolation_result is not None:
        method = "inpaint" if masks else "crop"
        print(f"[reprocess_person] Isolation complete (method={method}), "
              f"video exists={isolated_video.exists()}")
    else:
        print(f"[reprocess_person] Isolation returned None")

    if not isolated_video.exists():
        print(f"[reprocess_person] FAIL: isolated video not created, restoring backups")
        # Restore backups — isolation failed
        if bak_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists():
            bak_demo.rename(demo_dir)
        return {}

    # 4. Re-run estimation pipeline
    backend_label = "GEM-X" if estimation_backend == "gemx" else "GVHMR"
    if progress_callback:
        progress_callback(0.4, f"Running {backend_label} for person {person_index}...")

    bbox_override_path = _save_bbox_override(
        person_dir, track, crop_bbox=(isolation_result or {}).get("crop_bbox"),
    )
    _save_person_meta(
        person_dir,
        _make_person_meta(
            track,
            source_index=person_index,
            isolation_result=isolation_result,
            bbox_override_path=str(bbox_override_path),
                estimation_backend=estimation_backend,
        ),
    )
    if isolation_result is not None:
        with open(person_dir / "isolation_debug.json", "w") as f:
            json.dump(isolation_result.get("debug", {}), f, indent=2)

    backend_label = "GEM-X" if estimation_backend == "gemx" else "GVHMR"
    print(f"[reprocess_person] Running {backend_label} pipeline for person {person_index}...")

    try:
        if estimation_backend == "gemx":
            pt_path, _ = run_gemx_pipeline(
                video_path=str(isolated_video),
                person_id=tid,
                output_dir=person_dir,
                static_cam=static_cam,
                skip_render=True,
                bbx_override_path=str(bbox_override_path),
                force_preprocess=True,
                force_infer=True,
            )
        else:
            pt_path, _ = run_person_pipeline(
                video_path=str(isolated_video),
                person_id=tid,
                output_dir=person_dir,
                slam_override_path=slam_path if not static_cam else None,
                bbx_override_path=str(bbox_override_path),
                static_cam=static_cam,
                use_dpvo=use_dpvo,
                skip_render=True,
                force_preprocess=True,
                force_infer=True,
            )
    except Exception as e:
        print(f"[reprocess_person] Pipeline crashed for person {person_index}: {e}")
        import traceback
        traceback.print_exc()
        # Restore backups
        if bak_video.exists() and not isolated_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists() and not demo_dir.exists():
            bak_demo.rename(demo_dir)
        return {}

    if pt_path is None:
        print(f"[reprocess_person] FAIL: {backend_label} returned no pt_path, restoring backups")
        # Pipeline failed — restore backups
        if bak_video.exists() and not isolated_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists() and not demo_dir.exists():
            bak_demo.rename(demo_dir)
        return {}

    # Pipeline succeeded — clean up backups
    print(f"[reprocess_person] {backend_label} succeeded, pt_path={pt_path}")
    bak_video.unlink(missing_ok=True)
    shutil.rmtree(str(bak_demo), ignore_errors=True)

    # 5. Re-compute confidence & identity
    if progress_callback:
        progress_callback(0.8, f"Verifying person {person_index}...")

    result = {"pt_path": pt_path}

    try:
        from identity_confidence import compute_all_confidences, export_confidence_csv
        from identity_tracking import IdentityTrack, auto_generate_keyframes
        from identity_bridge import (
            OcclusionBridge,
            summarize_bridging,
            crossing_spans_from_overlap,
            crossing_spans_from_signal,
        )
        from smplx_to_bvh import extract_gvhmr_params

        params = extract_gvhmr_params(pt_path)
        per_frame_betas = params["betas"]
        num_person_frames = params["num_frames"]

        # Load per-person ViTPose if available
        vitpose_files = list(person_dir.rglob("vitpose.pt"))
        vitpose = None
        if vitpose_files:
            vp = torch.load(str(vitpose_files[0]), map_location="cpu", weights_only=False)
            if isinstance(vp, torch.Tensor):
                vitpose = vp.numpy()
            elif isinstance(vp, np.ndarray):
                vitpose = vp

        anchor_frames = set(manual_bbox_keyframes.keys())
        anchor_frames.update(
            int(entry.get("frame", -1))
            for entry in (verified_identity_keyframes or [])
            if int(entry.get("frame", -1)) >= 0
        )

        ignored_track_ids = _duplicate_track_ids_for_target(all_tracks, person_index)
        mask_overlap = _compute_track_mask_overlap(
            target_person_id=tid,
            all_masks=masks,
            num_frames=num_person_frames,
            ignored_track_ids=ignored_track_ids,
        )
        initial_confidences = compute_all_confidences(
            track_idx=person_index,
            all_tracks=all_tracks,
            num_frames=num_person_frames,
            vitpose=vitpose,
            per_frame_betas=per_frame_betas,
            overlap_signal=mask_overlap,
            anchor_frames=anchor_frames,
            ignored_track_ids=ignored_track_ids,
        )

        id_track = IdentityTrack(person_id=tid)
        bboxes = updated_bboxes[:num_person_frames]
        auto_generate_keyframes(
            track=id_track,
            confidences=initial_confidences,
            per_frame_betas=per_frame_betas,
            per_frame_bboxes=bboxes,
            per_frame_poses=params.get("body_pose", params.get("body_pose_world")),
        )
        manual_verified = verified_identity_keyframes or []
        poses = params.get("body_pose", params.get("body_pose_world"))
        for entry in manual_verified:
            frame_idx = int(entry.get("frame", -1))
            if frame_idx < 0 or frame_idx >= num_person_frames:
                continue
            bbox = np.asarray(entry.get("bbox"), dtype=np.float32)
            if bbox.shape != (4,):
                continue
            pose = poses[frame_idx] if poses is not None else None
            conf = initial_confidences[frame_idx] if frame_idx < len(initial_confidences) else None
            id_track.add_keyframe(
                frame_index=frame_idx,
                bbox=bbox,
                betas=np.asarray(per_frame_betas[frame_idx], dtype=np.float32),
                body_pose=pose,
                confidence=conf,
                verified=bool(entry.get("verified", True)),
                timestamp=frame_idx / 30.0,
                metadata={"source": "bodypipe_reprocess"},
            )
        id_track.establish_identity()
        verified_frames = {
            kf.frame_index for kf in id_track.keyframes
            if getattr(kf, "verified", False)
        }
        confidence_anchor_frames = set(anchor_frames) | set(verified_frames)
        confidences = compute_all_confidences(
            track_idx=person_index,
            all_tracks=all_tracks,
            num_frames=num_person_frames,
            vitpose=vitpose,
            per_frame_betas=per_frame_betas,
            established_betas=id_track.established_betas,
            overlap_signal=mask_overlap,
            verified_frames=verified_frames,
            anchor_frames=confidence_anchor_frames,
            stable_reference_frames=max(len(id_track.keyframes), len(confidence_anchor_frames)),
            ignored_track_ids=ignored_track_ids,
        )
        for kf in id_track.keyframes:
            if 0 <= kf.frame_index < len(confidences):
                kf.confidence = confidences[kf.frame_index]

        # Bridge low-confidence spans
        bridge = OcclusionBridge(id_track)
        body_pose = params.get("body_pose_world", params.get("body_pose"))
        global_orient = params.get("global_orient_world", params.get("global_orient"))
        transl = params.get("transl_world", params.get("transl"))

        bridged_frames = set()
        if body_pose is not None and global_orient is not None and transl is not None:
            if mask_overlap is not None:
                crossing_spans = crossing_spans_from_signal(mask_overlap, threshold=0.10)
            else:
                crossing_spans = crossing_spans_from_overlap(confidences)

            # Merge manual crossing spans from user annotation
            manual_spans_path = person_dir / "crossing_spans.json"
            if manual_spans_path.exists():
                try:
                    manual = json.loads(manual_spans_path.read_text())
                    manual_tuples = [tuple(s) for s in manual]
                    crossing_spans = _merge_crossing_spans(crossing_spans + manual_tuples)
                    print(f"[reprocess_person] Merged {len(manual_tuples)} manual crossing spans "
                          f"for person {person_index} → {len(crossing_spans)} total spans")
                except Exception as e:
                    print(f"[reprocess_person] Failed to load manual crossing spans: {e}")

            if crossing_spans:
                bridge_result = bridge.bridge_with_spans(
                    body_pose, global_orient, transl, crossing_spans)
            else:
                bridge_result = bridge.bake(body_pose, global_orient, transl, confidences)
            bridged_frames = bridge_result["bridged_frames"]

            # Determine if we need to modify pt_data
            verified_kfs = [kf for kf in id_track.keyframes
                            if kf.bbox is not None and kf.verified]
            need_position_constraints = bool(verified_kfs)
            need_save = bool(bridged_frames) or need_position_constraints

            if need_save:
                pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)

                if bridged_frames:
                    g = pt_data["smpl_params_global"]
                    g["body_pose"] = torch.from_numpy(
                        bridge_result["body_pose"].reshape(num_person_frames, -1)).float()
                    g["global_orient"] = torch.from_numpy(
                        bridge_result["global_orient"].reshape(num_person_frames, -1)).float()
                    g["transl"] = torch.from_numpy(bridge_result["transl"]).float()

                    if crossing_spans and "smpl_params_incam" in pt_data:
                        ic = pt_data["smpl_params_incam"]
                        ic_bp = np.array(ic["body_pose"]).reshape(num_person_frames, -1, 3)
                        ic_go = np.array(ic["global_orient"]).reshape(num_person_frames, 3)
                        ic_tr = np.array(ic["transl"]).reshape(num_person_frames, 3)
                        ic_result = bridge.bridge_with_spans(ic_bp, ic_go, ic_tr, crossing_spans)
                        ic["body_pose"] = torch.from_numpy(
                            ic_result["body_pose"].reshape(num_person_frames, -1)).float()
                        ic["global_orient"] = torch.from_numpy(
                            ic_result["global_orient"].reshape(num_person_frames, -1)).float()
                        ic["transl"] = torch.from_numpy(ic_result["transl"]).float()

                # Apply identity keyframe position constraints
                if need_position_constraints:
                    from world_assembly import apply_identity_position_constraints
                    K_fullimg = pt_data.get("K_fullimg")
                    if K_fullimg is not None:
                        K_np = np.array(K_fullimg[0])
                    else:
                        _, w, h = get_video_lwh(video_path)
                        K_np = estimate_K(w, h).numpy()
                    apply_identity_position_constraints(
                        pt_data, verified_kfs, K_np, num_person_frames)

                torch.save(pt_data, pt_path)

        # Save
        id_track.save_json(person_dir / "identity_track.json")
        export_confidence_csv(
            confidences, tid,
            str(person_dir / "confidence.csv"),
            bridged_frames=bridged_frames,
        )

        result["identity_track"] = id_track
        result["confidences"] = confidences
        result["regenerated_track"] = track
        bvh_path = _export_person_bvh(person_dir)
        if bvh_path is not None:
            result["bvh_path"] = str(bvh_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[reprocess_person] Identity verification failed for person {person_index}: {e}")

    if progress_callback:
        progress_callback(1.0, f"Person {person_index} reprocess complete")

    return result


def interpolate_bbox_corrections(
    original_bboxes: np.ndarray,
    corrections: dict[int, np.ndarray],
) -> np.ndarray:
    """Interpolate bbox corrections between edited keyframes.

    Between two corrected frames, linearly interpolates the delta from original.
    Before first / after last correction: constant extrapolation of nearest delta.

    Args:
        original_bboxes: (N, 4) original bbox array.
        corrections: sparse {frame_idx: np.array([x1,y1,x2,y2])} of overrides.

    Returns:
        (N, 4) array with interpolated corrections applied.
    """
    result = original_bboxes.copy()
    if not corrections:
        return result

    sorted_frames = sorted(corrections.keys())

    # Compute deltas at correction frames
    deltas = {}
    for f in sorted_frames:
        if f < len(original_bboxes):
            deltas[f] = corrections[f] - original_bboxes[f]

    if not deltas:
        return result

    sorted_frames = [f for f in sorted_frames if f in deltas]
    if not sorted_frames:
        return result

    # Before first correction: constant extrapolation
    first_f = sorted_frames[0]
    first_delta = deltas[first_f]
    for f in range(first_f):
        result[f] = original_bboxes[f] + first_delta

    # At correction frames: apply directly
    for f in sorted_frames:
        result[f] = corrections[f]

    # Between corrections: linear interpolation of delta
    for i in range(len(sorted_frames) - 1):
        f_a = sorted_frames[i]
        f_b = sorted_frames[i + 1]
        delta_a = deltas[f_a]
        delta_b = deltas[f_b]
        span = f_b - f_a
        for f in range(f_a + 1, f_b):
            t = (f - f_a) / span
            result[f] = original_bboxes[f] + (1 - t) * delta_a + t * delta_b

    # After last correction: constant extrapolation
    last_f = sorted_frames[-1]
    last_delta = deltas[last_f]
    for f in range(last_f + 1, len(original_bboxes)):
        result[f] = original_bboxes[f] + last_delta

    return result


def _compute_overlap_map(masks, all_tracks):
    """Compute per-frame boolean: does any pair of masks overlap?"""
    track_ids = [t["track_id"] for t in all_tracks]
    if len(track_ids) < 2 or not masks:
        return np.array([])

    first_tid = track_ids[0]
    num_frames = masks[first_tid].shape[0]
    overlap = np.zeros(num_frames, dtype=bool)

    for f in range(num_frames):
        for i, tid_a in enumerate(track_ids):
            if tid_a not in masks:
                continue
            for tid_b in track_ids[i + 1:]:
                if tid_b not in masks:
                    continue
                if np.logical_and(masks[tid_a][f], masks[tid_b][f]).any():
                    overlap[f] = True
                    break
            if overlap[f]:
                break

    return overlap


def _merge_crossing_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent crossing spans into non-overlapping sorted list."""
    if not spans:
        return []
    sorted_spans = sorted(spans, key=lambda s: s[0])
    merged = [sorted_spans[0]]
    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:  # overlapping or adjacent
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _compute_track_mask_overlap(
    target_person_id,
    all_masks,
    num_frames,
    ignored_track_ids: set[int] | None = None,
):
    """Measure overlap of one person's mask against all others per frame.

    Returns a per-frame fraction based on intersection over the smaller mask area.
    This is more sensitive to front-occlusion than bbox IoU and is only used when
    real SAM2 masks are available.
    """
    if not all_masks or target_person_id not in all_masks:
        return None

    target_masks = all_masks[target_person_id]
    n = min(num_frames, len(target_masks))
    if n <= 0:
        return None

    overlap = np.zeros(n, dtype=np.float32)
    for other_tid, other_masks in all_masks.items():
        if other_tid == target_person_id:
            continue
        if ignored_track_ids and int(other_tid) in ignored_track_ids:
            continue
        m = min(n, len(other_masks))
        for f in range(m):
            target = target_masks[f]
            other = other_masks[f]
            inter = np.logical_and(target, other).sum()
            if inter <= 0:
                continue
            denom = min(target.sum(), other.sum())
            if denom > 0:
                overlap[f] = max(overlap[f], float(inter / denom))

    return overlap


def _compute_stable_crop_bbox(
    bboxes: np.ndarray,
    img_w: int,
    img_h: int,
    padding: int = 40,
    detection_mask: np.ndarray | None = None,
) -> tuple[list[int], dict]:
    bboxes = np.asarray(bboxes, dtype=np.float32)
    trusted = None
    if detection_mask is not None:
        detection_mask = np.asarray(detection_mask, dtype=bool)[:len(bboxes)]
        if detection_mask.any():
            trusted = detection_mask

    crop_source = bboxes[trusted] if trusted is not None else bboxes
    x1_min = max(0, int(crop_source[:, 0].min()) - padding)
    y1_min = max(0, int(crop_source[:, 1].min()) - padding)
    x2_max = min(img_w, int(crop_source[:, 2].max()) + padding)
    y2_max = min(img_h, int(crop_source[:, 3].max()) + padding)
    crop_w = max(2, x2_max - x1_min)
    crop_h = max(2, y2_max - y1_min)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    crop_bbox = [x1_min, y1_min, x1_min + crop_w, y1_min + crop_h]
    debug = {
        "crop_area_ratio": float((crop_w * crop_h) / max(img_w * img_h, 1)),
        "trusted_crop_frames": int(trusted.sum()) if trusted is not None else int(len(bboxes)),
        "crop_from_trusted_frames": bool(trusted is not None),
    }
    return crop_bbox, debug


def _crop_person_video(video_path, bboxes, output_path, padding=40,
                       other_bboxes=None, detection_mask=None):
    """Crop-and-mask isolation: extract person's bbox region, black out others.

    Args:
        video_path: input video
        bboxes: (N_frames, 4) array of xyxy bboxes for target person
        output_path: output video path
        padding: extra pixels around bbox
        other_bboxes: list of (N_frames, 4) arrays for OTHER people to mask out
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_bbox, debug = _compute_stable_crop_bbox(
        bboxes, img_w=img_w, img_h=img_h, padding=padding, detection_mask=detection_mask,
    )
    x1_min, y1_min, x2_max, y2_max = crop_bbox
    crop_w = x2_max - x1_min
    crop_h = y2_max - y1_min

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    for f in range(min(total, len(bboxes))):
        ret, frame = cap.read()
        if not ret:
            break

        # Black out other people's bboxes before cropping
        if other_bboxes:
            for ob in other_bboxes:
                if f < len(ob):
                    ox1 = max(0, int(ob[f, 0]))
                    oy1 = max(0, int(ob[f, 1]))
                    ox2 = min(img_w, int(ob[f, 2]))
                    oy2 = min(img_h, int(ob[f, 3]))
                    frame[oy1:oy2, ox1:ox2] = 0

        crop = frame[y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]
        writer.write(crop)

    cap.release()
    writer.release()
    return {
        "crop_bbox": crop_bbox,
        "debug": debug,
        "num_inpainted": 0,
        "num_crop_only": min(total, len(bboxes)),
        "warnings": ["crop_near_full_frame"] if debug["crop_area_ratio"] >= 0.85 else [],
    }
