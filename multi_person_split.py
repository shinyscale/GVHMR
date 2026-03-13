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
from hmr4d.utils.geo.hmr_cam import estimate_K


GVHMR_DIR = Path(__file__).resolve().parent
DEMO_SCRIPT = GVHMR_DIR / "tools" / "demo" / "demo.py"


@dataclass
class MultiPersonResult:
    """Results from the multi-person splitting pipeline."""
    person_video_paths: list[str] = field(default_factory=list)
    person_dirs: list[Path] = field(default_factory=list)
    slam_path: str = ""
    offsets: dict = field(default_factory=dict)
    all_tracks: list = field(default_factory=list)
    masks: dict = field(default_factory=dict)
    num_persons: int = 0
    output_dir: str = ""
    identity_tracks: list = field(default_factory=list)  # IdentityTrack per person
    bridging_summaries: list = field(default_factory=list)


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

    # Check DPVO availability
    dpvo_available = False
    if use_dpvo:
        try:
            from dpvo.config import cfg as _dpvo_cfg
            dpvo_available = True
        except ImportError:
            print("[SLAM] DPVO not installed — falling back to SimpleVO.")

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
        slam = SLAMModel(str(video_path), width, height, intrinsics, buffer=4000, resize=0.5)
        bar = tqdm(total=length, desc="DPVO (shared)")
        while True:
            ret = slam.track()
            if ret:
                bar.update()
            else:
                break
        slam_results = slam.process()
        torch.save(slam_results, str(output_path))

    return str(output_path)


def run_person_pipeline(
    video_path,
    person_id,
    output_dir,
    slam_override_path=None,
    static_cam=False,
    use_dpvo=False,
    f_mm=None,
):
    """Run the existing single-person GVHMR pipeline on an isolated person video.

    Args:
        video_path: path to the isolated single-person video
        person_id: person track ID (for logging)
        output_dir: root output directory for this person
        slam_override_path: path to shared SLAM results
        static_cam: whether camera is static
        use_dpvo: whether to use DPVO (ignored if slam_override provided)
        f_mm: focal length override

    Returns:
        (pt_path, log_lines) tuple
    """
    output_dir = Path(output_dir)

    cmd = [
        sys.executable, str(DEMO_SCRIPT),
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


def split_multi_person_video(
    video_path,
    output_dir,
    min_track_duration=30,
    static_cam=False,
    use_dpvo=False,
    f_mm=None,
    max_persons=0,
    progress_callback=None,
):
    """Full multi-person isolation pipeline.

    Steps:
        1. Detect & track all people (OC-SORT / YOLO fallback)
        2. Compute shared camera motion (SLAM) once on original video
        3. Segment all people (SAM2)
        4. Per-person: smart isolate (crop or inpaint)
        5. Per-person: run existing GVHMR pipeline with shared SLAM
        6. Compute relative offsets and assemble world-space results

    Args:
        video_path: path to input multi-person video
        output_dir: base output directory
        min_track_duration: minimum frames to consider a track
        static_cam: whether camera is stationary
        use_dpvo: use DPVO for camera estimation
        f_mm: focal length in mm (None for auto)
        max_persons: limit to N largest people (0 = all)
        progress_callback: fn(frac, msg) for progress updates

    Returns:
        MultiPersonResult
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = MultiPersonResult(output_dir=str(output_dir))
    log_lines = []

    def _progress(frac, msg):
        if progress_callback:
            progress_callback(frac, msg)
        log_lines.append(f"[{frac:.0%}] {msg}")

    # ── Step 1: Detection & Tracking ──
    _progress(0.05, "Detecting and tracking all people...")

    detection_dir = output_dir / "detection"
    detection_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = detection_dir / "all_tracks.pt"

    if tracks_path.exists():
        tracks_data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
        all_tracks = tracks_data["tracks"]
        _progress(0.10, f"Loaded {len(all_tracks)} cached tracks")
    else:
        from person_tracker import PersonTracker, render_track_visualization

        tracker = PersonTracker()
        all_tracks = tracker.detect_and_track_all(
            video_path, min_track_frames=min_track_duration
        )
        torch.save({"tracks": all_tracks}, str(tracks_path))
        del tracker

        # Render track visualization
        viz_path = detection_dir / "track_visualization.mp4"
        render_track_visualization(video_path, all_tracks, str(viz_path))

        _progress(0.10, f"Detected {len(all_tracks)} people")

    # Cap to max_persons largest tracks (already sorted by area)
    if max_persons > 0 and len(all_tracks) > max_persons:
        _progress(0.10, f"Capping from {len(all_tracks)} tracks to {max_persons} largest")
        all_tracks = all_tracks[:max_persons]

    result.all_tracks = all_tracks
    result.num_persons = len(all_tracks)

    if len(all_tracks) == 0:
        _progress(1.0, "No people detected in video.")
        return result

    if len(all_tracks) == 1:
        _progress(0.10, "Only 1 person detected — using standard single-person pipeline.")
        # Fall through to single-person processing below

    # ── Step 2: Shared SLAM ──
    _progress(0.12, "Computing shared camera motion (SLAM)...")

    slam_path = output_dir / "shared_slam.pt"
    compute_shared_slam(video_path, slam_path, static_cam, use_dpvo, f_mm)
    result.slam_path = str(slam_path)

    _progress(0.20, "SLAM complete")

    # ── Step 3: Segmentation (only if >1 person) ──
    masks = {}
    if len(all_tracks) > 1:
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
            except ImportError as e:
                _progress(0.35, f"SAM2 not available ({e}). Using bbox-only isolation (no inpainting).")
                masks = {}

        result.masks = masks

    # ── Step 4: Per-person isolation ──
    _progress(0.37, "Isolating per-person videos...")

    person_dirs = []
    person_video_paths = []

    for i, track in enumerate(all_tracks):
        tid = track["track_id"]
        person_dir = output_dir / f"person_{i}"
        person_dir.mkdir(parents=True, exist_ok=True)
        person_dirs.append(person_dir)

        isolated_video = person_dir / "isolated_video.mp4"

        if isolated_video.exists():
            person_video_paths.append(str(isolated_video))
            _progress(
                0.37 + (i + 1) / len(all_tracks) * 0.13,
                f"Person {i} already isolated",
            )
            continue

        if len(all_tracks) == 1:
            # Single person — just symlink/copy the original video
            shutil.copy2(video_path, str(isolated_video))
            person_video_paths.append(str(isolated_video))
        elif masks:
            # Multi-person with masks — smart isolation
            try:
                from propainter_inpaint import isolate_person

                isolation_result = isolate_person(
                    video_path=video_path,
                    target_person_id=tid,
                    target_track_bboxes=track["bbx_xyxy"].cpu().numpy(),
                    all_masks=masks,
                    output_path=str(isolated_video),
                )

                # Save isolation mode log
                with open(person_dir / "isolation_mode.json", "w") as f:
                    json.dump(isolation_result.get("isolation_modes", []), f)

                person_video_paths.append(str(isolated_video))
            except ImportError:
                # No ProPainter — crop only
                _crop_person_video(
                    video_path, track["bbx_xyxy"].cpu().numpy(),
                    str(isolated_video),
                )
                person_video_paths.append(str(isolated_video))
        else:
            # No masks available — crop only
            _crop_person_video(
                video_path, track["bbx_xyxy"].cpu().numpy(),
                str(isolated_video),
            )
            person_video_paths.append(str(isolated_video))

        _progress(
            0.37 + (i + 1) / len(all_tracks) * 0.13,
            f"Person {i} isolated",
        )

    result.person_video_paths = person_video_paths
    result.person_dirs = person_dirs

    # ── Step 5: Per-person pipeline execution ──
    _progress(0.50, "Running per-person GVHMR pipelines...")

    for i, (person_video, person_dir) in enumerate(zip(person_video_paths, person_dirs)):
        track = all_tracks[i]
        tid = track["track_id"]

        # Check if already processed
        existing_pt = list(person_dir.rglob("hmr4d_results.pt"))
        if existing_pt:
            _progress(
                0.50 + (i + 1) / len(all_tracks) * 0.30,
                f"Person {i} already processed",
            )
            continue

        _progress(
            0.50 + i / len(all_tracks) * 0.30,
            f"Running GVHMR for person {i}...",
        )

        pt_path, person_log = run_person_pipeline(
            video_path=person_video,
            person_id=tid,
            output_dir=person_dir,
            slam_override_path=str(slam_path) if not static_cam else None,
            static_cam=static_cam,
            use_dpvo=use_dpvo,
            f_mm=f_mm,
        )

        if pt_path is None:
            log_lines.extend(person_log)
            _progress(
                0.50 + (i + 1) / len(all_tracks) * 0.30,
                f"Person {i} GVHMR FAILED",
            )

    # ── Step 5.5: Identity verification & occlusion bridging ──
    _progress(0.82, "Running identity verification...")

    try:
        from identity_confidence import compute_all_confidences, confidence_to_array, export_confidence_csv
        from identity_tracking import IdentityTrack, auto_generate_keyframes
        from identity_reid import ShapeReIdentifier
        from identity_bridge import OcclusionBridge, summarize_bridging
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
            confidences = compute_all_confidences(
                track_idx=i,
                all_tracks=all_tracks,
                num_frames=num_person_frames,
                vitpose=vitpose,
                per_frame_betas=per_frame_betas,
            )

            # Create identity track and auto-generate keyframes
            id_track = IdentityTrack(person_id=tid)
            bboxes = track["bbx_xyxy"].numpy() if hasattr(track["bbx_xyxy"], "numpy") else track["bbx_xyxy"]
            bboxes = bboxes[:num_person_frames]

            auto_generate_keyframes(
                track=id_track,
                confidences=confidences,
                per_frame_betas=per_frame_betas,
                per_frame_bboxes=bboxes,
                per_frame_poses=params.get("body_pose", params.get("body_pose_world")),
            )
            id_track.establish_identity()

            # Bridge low-confidence spans
            bridge = OcclusionBridge(id_track)
            body_pose = params.get("body_pose_world", params.get("body_pose"))
            global_orient = params.get("global_orient_world", params.get("global_orient"))
            transl = params.get("transl_world", params.get("transl"))

            if body_pose is not None and global_orient is not None and transl is not None:
                bridge_result = bridge.bake(body_pose, global_orient, transl, confidences)
                bridged_frames = bridge_result["bridged_frames"]
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
            per_person_mean_betas = {
                t.person_id: t.established_betas
                for t in identity_tracks
                if t.established_betas is not None
            }
            swaps = reid.detect_swap(0, per_person_mean_betas)
            if swaps:
                _progress(0.84, f"WARNING: Possible identity swaps detected: {swaps}")

    except Exception as e:
        _progress(0.84, f"Identity verification skipped: {e}")

    # ── Step 6: World-space assembly ──
    _progress(0.85, "Assembling world-space results...")

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
    )

    _progress(1.0, f"Multi-person pipeline complete. {result.num_persons} people processed.")

    return result


def reprocess_person(
    video_path: str,
    person_index: int,
    person_dir: str,
    updated_bboxes: np.ndarray,
    all_tracks: list[dict],
    slam_path: str,
    masks_dir: str,
    static_cam: bool = False,
    use_dpvo: bool = False,
    progress_callback=None,
) -> dict:
    """Re-run isolation + GVHMR for a single person after bbox corrections.

    Caches shared SLAM and SAM2 masks. Only re-runs isolation + pipeline.

    Returns dict with keys: identity_track, confidences, pt_path (or empty on failure).
    """
    person_dir = Path(person_dir)
    person_dir.mkdir(parents=True, exist_ok=True)

    track = all_tracks[person_index]
    tid = track.get("track_id", person_index)

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

    # 3. Re-isolate person
    if progress_callback:
        progress_callback(0.1, f"Re-isolating person {person_index}...")

    masks = {}
    masks_dir_path = Path(masks_dir)
    mask_file = masks_dir_path / f"person_{tid}_masks.npz"
    if mask_file.exists():
        mask_data = np.load(str(mask_file))
        masks[tid] = mask_data["masks"]

    if len(all_tracks) == 1:
        shutil.copy2(video_path, str(isolated_video))
    elif masks:
        try:
            from propainter_inpaint import isolate_person

            isolate_person(
                video_path=video_path,
                target_person_id=tid,
                target_track_bboxes=updated_bboxes,
                all_masks=masks,
                output_path=str(isolated_video),
            )
        except ImportError:
            _crop_person_video(video_path, updated_bboxes, str(isolated_video))
    else:
        _crop_person_video(video_path, updated_bboxes, str(isolated_video))

    if not isolated_video.exists():
        # Restore backups — isolation failed
        if bak_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists():
            bak_demo.rename(demo_dir)
        return {}

    # 4. Re-run GVHMR pipeline
    if progress_callback:
        progress_callback(0.4, f"Running GVHMR for person {person_index}...")

    try:
        pt_path, _ = run_person_pipeline(
            video_path=str(isolated_video),
            person_id=tid,
            output_dir=person_dir,
            slam_override_path=slam_path if not static_cam else None,
            static_cam=static_cam,
            use_dpvo=use_dpvo,
        )
    except Exception as e:
        print(f"[reprocess_person] Pipeline crashed for person {person_index}: {e}")
        # Restore backups
        if bak_video.exists() and not isolated_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists() and not demo_dir.exists():
            bak_demo.rename(demo_dir)
        return {}

    if pt_path is None:
        # Pipeline failed — restore backups
        if bak_video.exists() and not isolated_video.exists():
            bak_video.rename(isolated_video)
        if bak_demo.exists() and not demo_dir.exists():
            bak_demo.rename(demo_dir)
        return {}

    # Pipeline succeeded — clean up backups
    bak_video.unlink(missing_ok=True)
    shutil.rmtree(str(bak_demo), ignore_errors=True)

    # 5. Re-compute confidence & identity
    if progress_callback:
        progress_callback(0.8, f"Verifying person {person_index}...")

    result = {"pt_path": pt_path}

    try:
        from identity_confidence import compute_all_confidences, export_confidence_csv
        from identity_tracking import IdentityTrack, auto_generate_keyframes
        from identity_bridge import OcclusionBridge, summarize_bridging
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

        confidences = compute_all_confidences(
            track_idx=person_index,
            all_tracks=all_tracks,
            num_frames=num_person_frames,
            vitpose=vitpose,
            per_frame_betas=per_frame_betas,
        )

        id_track = IdentityTrack(person_id=tid)
        bboxes = updated_bboxes[:num_person_frames]

        auto_generate_keyframes(
            track=id_track,
            confidences=confidences,
            per_frame_betas=per_frame_betas,
            per_frame_bboxes=bboxes,
            per_frame_poses=params.get("body_pose", params.get("body_pose_world")),
        )
        id_track.establish_identity()

        # Bridge low-confidence spans
        bridge = OcclusionBridge(id_track)
        body_pose = params.get("body_pose_world", params.get("body_pose"))
        global_orient = params.get("global_orient_world", params.get("global_orient"))
        transl = params.get("transl_world", params.get("transl"))

        bridged_frames = set()
        if body_pose is not None and global_orient is not None and transl is not None:
            bridge_result = bridge.bake(body_pose, global_orient, transl, confidences)
            bridged_frames = bridge_result["bridged_frames"]

        # Save
        id_track.save_json(person_dir / "identity_track.json")
        export_confidence_csv(
            confidences, tid,
            str(person_dir / "confidence.csv"),
            bridged_frames=bridged_frames,
        )

        result["identity_track"] = id_track
        result["confidences"] = confidences

    except Exception as e:
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


def _crop_person_video(video_path, bboxes, output_path, padding=40):
    """Simple crop-only isolation: extract person's bbox region from each frame.

    Args:
        video_path: input video
        bboxes: (N_frames, 4) array of xyxy bboxes
        output_path: output video path
        padding: extra pixels around bbox
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute stable crop region (max bbox + padding)
    x1_min = max(0, int(bboxes[:, 0].min()) - padding)
    y1_min = max(0, int(bboxes[:, 1].min()) - padding)
    x2_max = min(img_w, int(bboxes[:, 2].max()) + padding)
    y2_max = min(img_h, int(bboxes[:, 3].max()) + padding)

    crop_w = x2_max - x1_min
    crop_h = y2_max - y1_min

    # Make dimensions even (required by most codecs)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

    for f in range(min(total, len(bboxes))):
        ret, frame = cap.read()
        if not ret:
            break
        crop = frame[y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]
        writer.write(crop)

    cap.release()
    writer.release()
