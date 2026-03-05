"""Motion Capture Studio — GVHMR Body + Full Performance Capture GUI."""

import subprocess
import re
import shutil
import json
from pathlib import Path

import numpy as np
import torch
import gradio as gr

from preprocess import preprocess_video
from face_capture import run_face_pipeline, render_face_mesh_video
from smplx_to_bvh import (
    convert_smplx_to_bvh,
    convert_params_to_bvh,
    extract_gvhmr_params,
    extract_smplx_params,
    merge_gvhmr_smplestx_params,
)
from bvh_to_fbx import convert_bvh_to_fbx
from visualize_skeleton import render_skeleton_video, render_world_views

GVHMR_DIR = Path("/mnt/f/GVHMR/GVHMR")
DEMO_SCRIPT = GVHMR_DIR / "tools" / "demo" / "demo.py"
SMPLESTX_DIR = Path("/mnt/f/SMPLest-X")
SMPLESTX_ENV = "smplestx"

# Pipeline stages for GVHMR tab
STAGE_PATTERNS = [
    (re.compile(r"preprocess|loading video|reading video", re.I), "Preprocessing", 0.05),
    (re.compile(r"yolo|tracking|detection", re.I), "YOLO Tracking", 0.15),
    (re.compile(r"vitpose|pose estimation|2d pose", re.I), "ViTPose", 0.30),
    (re.compile(r"hmr2|hmr4d_feature|feature extraction", re.I), "HMR2 Features", 0.45),
    (re.compile(r"dpvo|simple_vo|camera estimation|slam", re.I), "Camera Estimation", 0.60),
    (re.compile(r"gvhmr|predicting|prediction|diffusion", re.I), "GVHMR Prediction", 0.80),
    (re.compile(r"render|saving|visualization", re.I), "Rendering", 0.95),
]


# ── Shared Utilities ──

def resolve_video_path(video_upload, video_path_text: str) -> str:
    """Resolve video input from upload or text path."""
    if video_path_text and video_path_text.strip():
        video_path = video_path_text.strip()
        if not Path(video_path).is_file():
            raise gr.Error(f"File not found: {video_path}")
        return video_path
    elif video_upload is not None:
        return video_upload
    else:
        raise gr.Error("Please upload a video or enter a file path.")


def find_output_dir(video_path: str) -> Path | None:
    """Find the GVHMR output directory for a processed video."""
    video_name = Path(video_path).stem
    candidates = [
        GVHMR_DIR / "outputs" / "demo" / video_name,
        GVHMR_DIR / "outputs" / video_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    demo_out = GVHMR_DIR / "outputs" / "demo"
    if demo_out.is_dir():
        dirs = sorted(demo_out.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        if dirs:
            return dirs[0]
    return None


def find_file(output_dir: Path, pattern: str) -> str | None:
    """Find a file matching a glob pattern in the output dir."""
    matches = list(output_dir.rglob(pattern))
    return str(matches[0]) if matches else None


def get_video_fps(video_path: str) -> float:
    """Get video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                rate = streams[0].get("r_frame_rate", "30/1")
                if "/" in rate:
                    num, den = rate.split("/")
                    return float(num) / float(den)
                return float(rate)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, ZeroDivisionError):
        pass
    return 30.0


# ── Tab 1: GVHMR Body Pipeline ──

def run_gvhmr(
    video_upload,
    video_path_text: str,
    static_cam: bool,
    use_dpvo: bool,
    focal_length: str,
    progress=gr.Progress(track_tqdm=False),
):
    """Run GVHMR demo.py on the input video (body-only, world-grounded)."""

    video_path = resolve_video_path(video_upload, video_path_text)

    # Portrait video fix
    progress(0.01, desc="Checking video rotation...")
    video_path, preprocess_msg = preprocess_video(video_path)
    log_lines = [preprocess_msg, ""]

    # Build command
    cmd = ["python", str(DEMO_SCRIPT), f"--video={video_path}"]
    if static_cam:
        cmd.append("--static_cam")
    if use_dpvo:
        cmd.append("--use_dpvo")
    if focal_length and focal_length.strip():
        try:
            fl = float(focal_length.strip())
            cmd.append(f"--f_mm={int(fl)}")
        except ValueError:
            raise gr.Error(f"Invalid focal length: {focal_length}")

    progress(0.02, desc="Starting GVHMR pipeline...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(GVHMR_DIR),
        bufsize=1,
    )

    for line in proc.stdout:
        line = line.rstrip()
        log_lines.append(line)
        for pattern, stage_name, frac in STAGE_PATTERNS:
            if pattern.search(line):
                progress(frac, desc=stage_name)
                break

    proc.wait()
    full_log = "\n".join(log_lines)

    if proc.returncode != 0:
        return None, None, None, None, full_log

    progress(1.0, desc="Complete!")

    output_dir = find_output_dir(video_path)
    if output_dir is None:
        return None, None, None, None, full_log + "\n\nERROR: Output directory not found."

    side_by_side = find_file(output_dir, "*side_by_side*.*")
    incam = find_file(output_dir, "*incam*.*")
    global_view = find_file(output_dir, "*global*.*")
    pt_file = find_file(output_dir, "hmr4d_results.pt")

    if not side_by_side and not incam and not global_view:
        mp4s = sorted(output_dir.rglob("*.mp4"))
        if len(mp4s) >= 3:
            side_by_side, incam, global_view = str(mp4s[0]), str(mp4s[1]), str(mp4s[2])
        elif len(mp4s) == 2:
            side_by_side, incam = str(mp4s[0]), str(mp4s[1])
        elif len(mp4s) == 1:
            side_by_side = str(mp4s[0])

    return side_by_side, incam, global_view, pt_file, full_log


# ── Tab 2: Full Performance Capture Pipeline ──

def _run_smplestx_subprocess(video_path: str, fps: float, output_dir: Path) -> tuple[str | None, list[str]]:
    """Run SMPLest-X inference as a subprocess in its conda env.

    Returns (pt_output_path, log_lines).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", SMPLESTX_ENV,
        "python", str(GVHMR_DIR / "smplestx_inference.py"),
        "--video", str(video_path),
        "--fps", str(int(fps)),
        "--output_dir", str(output_dir),
    ]

    log_lines = [f"Running SMPLest-X: {' '.join(cmd)}", ""]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(SMPLESTX_DIR),
        bufsize=1,
    )

    for line in proc.stdout:
        log_lines.append(line.rstrip())

    proc.wait()

    if proc.returncode != 0:
        log_lines.append(f"\nERROR: SMPLest-X exited with code {proc.returncode}")
        return None, log_lines

    # Find the output .pt file
    pt_files = sorted(output_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Also check for .npz
    npz_files = sorted(output_dir.rglob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)

    pt_path = str(pt_files[0]) if pt_files else (str(npz_files[0]) if npz_files else None)

    if pt_path is None:
        log_lines.append("\nERROR: SMPLest-X completed but no output file found.")

    return pt_path, log_lines


def _extract_bboxes_from_smplestx(output_dir: Path) -> np.ndarray | None:
    """Extract person bounding boxes from SMPLest-X output.

    SMPLest-X uses YOLOv8 internally. We look for bbox data in its outputs.
    """
    import torch

    # Check common output patterns
    for pattern in ["*.pt", "*.npz"]:
        for f in sorted(output_dir.rglob(pattern)):
            try:
                if f.suffix == ".pt":
                    data = torch.load(str(f), map_location="cpu", weights_only=False)
                else:
                    data = dict(np.load(str(f), allow_pickle=True))

                # Look for bbox keys
                for key in ["person_bbox", "bboxes", "bbox", "bb_xyxy", "pred_bboxes"]:
                    if key in data:
                        return np.array(data[key]).reshape(-1, 4)
            except Exception:
                continue

    return None


def _fallback_face_crops_from_video(video_path: str) -> np.ndarray:
    """Generate approximate face bboxes using upper portion of frame.

    Fallback when SMPLest-X bbox data isn't available.
    Uses a simple heuristic: assume person is roughly centered.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Approximate: full-width, top 60% of frame as "person"
    bboxes = np.zeros((n_frames, 4))
    bboxes[:, 0] = w * 0.2   # x1
    bboxes[:, 1] = 0          # y1
    bboxes[:, 2] = w * 0.8   # x2
    bboxes[:, 3] = h * 0.6   # y2
    return bboxes


def _run_gvhmr_subprocess(video_path: str, static_cam: bool, use_dpvo: bool) -> tuple[str | None, list[str]]:
    """Run GVHMR demo.py as a subprocess in the current conda env.

    Returns (pt_output_path, log_lines).
    """
    cmd = ["python", str(DEMO_SCRIPT), f"--video={video_path}"]
    if static_cam:
        cmd.append("--static_cam")
    if use_dpvo:
        cmd.append("--use_dpvo")

    log_lines = [f"Running GVHMR: {' '.join(cmd)}", ""]

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
        log_lines.append(f"\nERROR: GVHMR exited with code {proc.returncode}")
        return None, log_lines

    # Find the output .pt file
    output_dir = find_output_dir(video_path)
    if output_dir is None:
        log_lines.append("\nERROR: GVHMR output directory not found.")
        return None, log_lines

    pt_file = find_file(output_dir, "hmr4d_results.pt")
    if pt_file is None:
        log_lines.append("\nERROR: hmr4d_results.pt not found in output.")
        return None, log_lines

    return pt_file, log_lines


def _save_merged_pt(params: dict, output_path: str) -> str:
    """Save merged SMPL-X params dict as a .pt file for re-export."""
    save_dict = {
        "global_orient": torch.tensor(params["global_orient"], dtype=torch.float32),
        "body_pose": torch.tensor(params["body_pose"].reshape(params["num_frames"], -1), dtype=torch.float32),
        "left_hand_pose": torch.tensor(params["left_hand_pose"].reshape(params["num_frames"], -1), dtype=torch.float32),
        "right_hand_pose": torch.tensor(params["right_hand_pose"].reshape(params["num_frames"], -1), dtype=torch.float32),
        "transl": torch.tensor(params["transl"], dtype=torch.float32),
    }
    if "betas" in params:
        save_dict["betas"] = torch.tensor(params["betas"], dtype=torch.float32)
    if "bbox" in params:
        save_dict["bbox"] = torch.tensor(params["bbox"], dtype=torch.float32)
    torch.save(save_dict, output_path)
    return output_path


def run_full_pipeline(
    video_upload,
    video_path_text: str,
    target_fps: str,
    fbx_naming: str,
    pitch_adjust: float,
    pipeline_mode: str,
    hybrid_static_cam: bool,
    hybrid_use_dpvo: bool,
    progress=gr.Progress(track_tqdm=False),
):
    """Run the full performance capture pipeline: body+hands+face."""

    video_path = resolve_video_path(video_upload, video_path_text)
    log_lines = []
    is_hybrid = "Hybrid" in pipeline_mode

    # Parse FPS
    try:
        fps = float(target_fps.strip()) if target_fps and target_fps.strip() else 30.0
    except ValueError:
        fps = 30.0

    # ── Stage 1: Preprocess (portrait fix) ──
    progress(0.02, desc="Preprocessing video...")
    video_path, preprocess_msg = preprocess_video(video_path)
    log_lines.append(f"[Preprocess] {preprocess_msg}")

    # Detect actual FPS if not specified
    actual_fps = get_video_fps(video_path)
    if fps <= 0:
        fps = actual_fps
    log_lines.append(f"[Info] Video FPS: {actual_fps:.2f}, Target FPS: {fps:.2f}")
    log_lines.append(f"[Info] Pipeline mode: {pipeline_mode}")

    # Set up output directory
    video_stem = Path(video_path).stem
    output_base = GVHMR_DIR / "outputs" / "perfcap" / video_stem
    output_base.mkdir(parents=True, exist_ok=True)

    # Track merged params for hybrid pipeline
    merged_params = None
    pt_path = None  # final .pt for downloads

    if is_hybrid:
        # ── Hybrid Stage 2a: Run GVHMR (body + world trajectory) ──
        progress(0.05, desc="Running GVHMR (body + world trajectory)...")
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append("[Stage 2a] GVHMR — World-Grounded Body Capture")
        log_lines.append("=" * 60)

        gvhmr_pt_path, gvhmr_log = _run_gvhmr_subprocess(
            video_path, hybrid_static_cam, hybrid_use_dpvo,
        )
        log_lines.extend(gvhmr_log)

        if gvhmr_pt_path is None:
            full_log = "\n".join(log_lines)
            return None, None, None, None, None, None, None, full_log

        log_lines.append(f"\n[GVHMR] Output: {gvhmr_pt_path}")
        progress(0.25, desc="GVHMR complete. Running SMPLest-X for hands...")

        # ── Hybrid Stage 2b: Run SMPLest-X (hands only) ──
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append("[Stage 2b] SMPLest-X — Hand Capture")
        log_lines.append("=" * 60)

        smplestx_output_dir = output_base / "smplestx"
        smplestx_pt_path, smplestx_log = _run_smplestx_subprocess(video_path, fps, smplestx_output_dir)
        log_lines.extend(smplestx_log)

        if smplestx_pt_path is None:
            full_log = "\n".join(log_lines)
            return None, None, None, None, None, None, None, full_log

        log_lines.append(f"\n[SMPLest-X] Output: {smplestx_pt_path}")
        progress(0.45, desc="SMPLest-X complete. Merging...")

        # ── Hybrid Stage 2c: Merge GVHMR body + SMPLest-X hands ──
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append("[Stage 2c] Merging — GVHMR Body + SMPLest-X Hands")
        log_lines.append("=" * 60)

        try:
            gvhmr_params = extract_gvhmr_params(gvhmr_pt_path)
            smplestx_params = extract_smplx_params(smplestx_pt_path)
            merged_params = merge_gvhmr_smplestx_params(gvhmr_params, smplestx_params)

            log_lines.append(f"[Merge] Body from GVHMR: {gvhmr_params['num_frames']} frames")
            log_lines.append(f"[Merge] Hands from SMPLest-X: {smplestx_params['num_frames']} frames")
            log_lines.append(f"[Merge] Merged: {merged_params['num_frames']} frames")

            # Save merged .pt for re-export
            merged_pt_path = str(output_base / f"{video_stem}_hybrid_smplx.pt")
            _save_merged_pt(merged_params, merged_pt_path)
            pt_path = merged_pt_path
            log_lines.append(f"[Merge] Saved: {merged_pt_path}")
        except Exception as e:
            log_lines.append(f"[Merge] ERROR: {e}")
            full_log = "\n".join(log_lines)
            return None, None, None, None, None, None, None, full_log

        progress(0.48, desc="Merge complete.")

        # Get bboxes for face capture from SMPLest-X output
        bboxes = _extract_bboxes_from_smplestx(smplestx_output_dir)

    else:
        # ── SMPLest-X Only: Stage 2 ──
        progress(0.05, desc="Running SMPLest-X (body + hands)...")
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append("[Stage 2] SMPLest-X — Body + Hand Capture")
        log_lines.append("=" * 60)

        smplestx_output_dir = output_base / "smplestx"
        pt_path, smplestx_log = _run_smplestx_subprocess(video_path, fps, smplestx_output_dir)
        log_lines.extend(smplestx_log)

        if pt_path is None:
            full_log = "\n".join(log_lines)
            return None, None, None, None, None, None, None, full_log

        log_lines.append(f"\n[SMPLest-X] Output: {pt_path}")
        progress(0.45, desc="SMPLest-X complete.")

        # Get bboxes for face capture
        bboxes = _extract_bboxes_from_smplestx(smplestx_output_dir)

    # ── Stage 3: Face Capture (same for both modes) ──
    progress(0.50, desc="Extracting face captures...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 3] Face Capture — MediaPipe ARKit Blendshapes")
    log_lines.append("=" * 60)

    if bboxes is not None:
        log_lines.append(f"[Face] Using {len(bboxes)} person bboxes from SMPLest-X output.")
    else:
        log_lines.append("[Face] WARNING: No person bboxes found. Using heuristic fallback.")
        bboxes = _fallback_face_crops_from_video(video_path)

    face_csv_path = str(output_base / f"{video_stem}_arkit_blendshapes.csv")

    def face_progress(frac, msg):
        overall = 0.50 + frac * 0.20
        progress(overall, desc=msg)
        log_lines.append(f"[Face] {msg}")

    try:
        run_face_pipeline(video_path, bboxes, face_csv_path, fps=fps, progress_callback=face_progress)
        log_lines.append(f"[Face] ARKit CSV: {face_csv_path}")
    except Exception as e:
        log_lines.append(f"[Face] ERROR: {e}")
        face_csv_path = None

    # ── Stage 3.5: Face Mesh Visualization ──
    progress(0.72, desc="Rendering face mesh overlay...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 3.5] Face Mesh Visualization")
    log_lines.append("=" * 60)

    face_mesh_video = str(output_base / f"{video_stem}_face_mesh.mp4")
    try:
        def face_mesh_progress(frac, msg):
            progress(0.72 + frac * 0.03, desc=msg)

        render_face_mesh_video(
            video_path, bboxes, face_mesh_video, fps=fps,
            progress_callback=face_mesh_progress,
        )
        log_lines.append(f"[FaceMesh] Video: {face_mesh_video}")
    except Exception as e:
        log_lines.append(f"[FaceMesh] ERROR: {e}")
        face_mesh_video = None

    # ── Stage 4: BVH Conversion ──
    progress(0.75, desc="Converting to BVH...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 4] BVH Conversion")
    log_lines.append("=" * 60)

    bvh_path = str(output_base / f"{video_stem}_body_hands.bvh")

    try:
        if is_hybrid and merged_params is not None:
            # Hybrid: use merged params directly, skip world grounding (GVHMR is already grounded),
            # only smooth hands (GVHMR body is temporally coherent)
            convert_params_to_bvh(
                merged_params, bvh_path, fps=fps,
                skip_world_grounding=True,
                smooth_body=False,
                smooth_hands=True,
            )
        else:
            # SMPLest-X only: full processing pipeline
            convert_smplx_to_bvh(pt_path, bvh_path, fps=fps, pitch_adjust_deg=pitch_adjust)
        log_lines.append(f"[BVH] Written: {bvh_path}")
    except Exception as e:
        log_lines.append(f"[BVH] ERROR: {e}")
        bvh_path = None

    # ── Stage 4.5: World View (Front + Side) ──
    world_view_video = None
    progress(0.80, desc="Rendering world view (front + side)...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 4.5] World View Rendering (Front + Side)")
    log_lines.append("=" * 60)

    world_view_video = str(output_base / f"{video_stem}_world_view.mp4")
    try:
        def wv_progress(frac, msg):
            progress(0.80 + frac * 0.07, desc=msg)

        if is_hybrid and merged_params is not None:
            render_world_views(
                output_path=world_view_video, fps=fps,
                params=merged_params,
                skip_world_grounding=True,
                progress_callback=wv_progress,
            )
        else:
            render_world_views(
                pt_path, world_view_video, fps=fps,
                pitch_adjust_deg=pitch_adjust,
                progress_callback=wv_progress,
            )
        log_lines.append(f"[WorldView] Video: {world_view_video}")
    except Exception as e:
        log_lines.append(f"[WorldView] ERROR: {e}")
        world_view_video = None

    # ── Stage 5: BVH → FBX (for Cascadeur / DCC tools) ──
    fbx_path = None
    if bvh_path:
        naming_key = "ue5" if "ue5" in fbx_naming.lower() else "mixamo"

        progress(0.90, desc=f"Converting BVH to FBX ({naming_key})...")
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append(f"[Stage 5] BVH → FBX Conversion (Blender, {naming_key} naming)")
        log_lines.append("=" * 60)

        fbx_path = str(Path(bvh_path).with_suffix(".fbx"))
        fbx_log = convert_bvh_to_fbx(bvh_path, fbx_path, fps=fps, naming=naming_key)
        log_lines.append(fbx_log)

        if "ERROR" in fbx_log:
            fbx_path = None

    # ── Stage 6: Skeleton Visualization ──
    skeleton_video = None
    progress(0.93, desc="Rendering skeleton overlay...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 6] Skeleton Overlay Visualization")
    log_lines.append("=" * 60)

    skeleton_video = str(output_base / f"{video_stem}_skeleton.mp4")
    try:
        def viz_progress(frac, msg):
            progress(0.93 + frac * 0.06, desc=msg)

        if is_hybrid and merged_params is not None:
            # Use SMPLest-X params for camera-space overlay (has bbox + cam_trans)
            render_skeleton_video(
                pt_path=smplestx_pt_path,
                video_path=video_path,
                output_path=skeleton_video,
                fps=fps,
                progress_callback=viz_progress,
            )
            log_lines.append(f"[Viz] Skeleton video: {skeleton_video}")
        else:
            render_skeleton_video(pt_path, video_path, skeleton_video, fps=fps, progress_callback=viz_progress)
            log_lines.append(f"[Viz] Skeleton video: {skeleton_video}")
    except Exception as e:
        log_lines.append(f"[Viz] ERROR: {e}")
        skeleton_video = None

    # ── Done ──
    progress(1.0, desc="Pipeline complete!")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append(f"[Done] {'Hybrid' if is_hybrid else 'SMPLest-X Only'} pipeline complete.")
    log_lines.append(f"  BVH:       {bvh_path or 'FAILED'}")
    log_lines.append(f"  FBX:       {fbx_path or 'SKIPPED'}")
    log_lines.append(f"  Face CSV:  {face_csv_path or 'FAILED'}")
    log_lines.append(f"  SMPL-X:    {pt_path}")
    log_lines.append(f"  Skeleton:  {skeleton_video or 'SKIPPED'}")
    log_lines.append(f"  FaceMesh:  {face_mesh_video or 'SKIPPED'}")
    log_lines.append(f"  WorldView: {world_view_video or 'SKIPPED'}")
    log_lines.append("=" * 60)

    full_log = "\n".join(log_lines)
    return skeleton_video, face_mesh_video, world_view_video, bvh_path, fbx_path, face_csv_path, pt_path, full_log


# ── Gradio UI ──

with gr.Blocks(title="Motion Capture Studio") as app:
    gr.Markdown("# Motion Capture Studio")

    with gr.Tabs():
        # ── Tab 1: GVHMR Body ──
        with gr.TabItem("GVHMR Body"):
            gr.Markdown("World-grounded body motion capture (SMPL, 24 joints). Upload a video and run the pipeline.")

            with gr.Row():
                with gr.Column(scale=1):
                    gvhmr_video_upload = gr.Video(label="Upload Video", sources=["upload"])
                    gvhmr_video_path = gr.Textbox(
                        label="Or enter WSL2 file path",
                        placeholder="/mnt/f/Videos/take1.mp4",
                    )

                with gr.Column(scale=1):
                    static_cam = gr.Checkbox(label="Static Camera", value=False)
                    use_dpvo = gr.Checkbox(label="Use DPVO (better camera, slower)", value=False)
                    focal_length = gr.Textbox(
                        label="Focal Length (mm, leave blank for auto)",
                        placeholder="auto",
                    )
                    gvhmr_run_btn = gr.Button("Run GVHMR", variant="primary", size="lg")

            gr.Markdown("### Results")
            with gr.Row():
                vid_sbs = gr.Video(label="Side-by-Side")
                vid_incam = gr.Video(label="In-Camera Overlay")
                vid_global = gr.Video(label="Global 3D View")

            gvhmr_pt_download = gr.File(label="Download hmr4d_results.pt")

            with gr.Accordion("Pipeline Log", open=False):
                gvhmr_log = gr.Textbox(
                    label="Log Output", lines=15, max_lines=50,
                    interactive=False,
                )

            gvhmr_run_btn.click(
                fn=run_gvhmr,
                inputs=[gvhmr_video_upload, gvhmr_video_path, static_cam, use_dpvo, focal_length],
                outputs=[vid_sbs, vid_incam, vid_global, gvhmr_pt_download, gvhmr_log],
            )

        # ── Tab 2: Full Performance Capture ──
        with gr.TabItem("Full Performance Capture"):
            gr.Markdown(
                "**Body + Hands + Face** from monocular video.\n\n"
                "**Hybrid mode** (recommended): GVHMR for world-grounded body + SMPLest-X for hands.\n"
                "**SMPLest-X Only**: Original pipeline using SMPLest-X for everything.\n\n"
                "Output is ready for Cascadeur / Unreal Engine 5 MetaHuman import."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    perf_video_upload = gr.Video(label="Upload Video", sources=["upload"])
                    perf_video_path = gr.Textbox(
                        label="Or enter WSL2 file path",
                        placeholder="/mnt/f/Videos/performance_take.mp4",
                    )

                with gr.Column(scale=1):
                    pipeline_mode = gr.Radio(
                        label="Body Tracking",
                        choices=["Hybrid: GVHMR Body + SMPLest-X Hands", "SMPLest-X Only"],
                        value="Hybrid: GVHMR Body + SMPLest-X Hands",
                    )
                    with gr.Row():
                        hybrid_static_cam = gr.Checkbox(
                            label="Static Camera", value=False,
                            info="GVHMR option: check if camera is fixed/tripod",
                        )
                        hybrid_use_dpvo = gr.Checkbox(
                            label="Use DPVO", value=False,
                            info="GVHMR option: better camera estimation, slower",
                        )
                    perf_fps = gr.Textbox(
                        label="Target FPS",
                        value="30",
                        placeholder="30",
                    )
                    fbx_naming = gr.Dropdown(
                        label="FBX Bone Naming",
                        choices=["Mixamo (Cascadeur)", "UE5 Mannequin"],
                        value="Mixamo (Cascadeur)",
                    )
                    pitch_adjust = gr.Slider(
                        label="Pitch Adjust (°)",
                        minimum=-30, maximum=30, step=0.5, value=0,
                        info="Fine-tune on top of auto tilt correction. 0 = auto only. (SMPLest-X Only mode)",
                    )
                    perf_run_btn = gr.Button(
                        "Run Full Pipeline",
                        variant="primary",
                        size="lg",
                    )

            gr.Markdown("### Preview")
            with gr.Row():
                skeleton_preview = gr.Video(label="Skeleton Overlay")
                face_mesh_preview = gr.Video(label="Face Mesh")
                world_view_preview = gr.Video(label="World View (Front + Side)")

            gr.Markdown("### Downloads")
            with gr.Row():
                bvh_download = gr.File(label="Download BVH (body + hands)")
                fbx_download = gr.File(label="Download FBX (Cascadeur / DCC)")
                face_csv_download = gr.File(label="Download ARKit Face CSV")
                smplx_pt_download = gr.File(label="Download SMPL-X .pt")

            with gr.Accordion("Pipeline Log", open=False):
                perf_log = gr.Textbox(
                    label="Log Output", lines=20, max_lines=80,
                    interactive=False,
                )

            # Show/hide GVHMR options based on pipeline mode
            def _toggle_gvhmr_opts(mode):
                visible = "Hybrid" in mode
                return gr.update(visible=visible), gr.update(visible=visible)

            pipeline_mode.change(
                fn=_toggle_gvhmr_opts,
                inputs=[pipeline_mode],
                outputs=[hybrid_static_cam, hybrid_use_dpvo],
            )

            perf_run_btn.click(
                fn=run_full_pipeline,
                inputs=[
                    perf_video_upload, perf_video_path, perf_fps, fbx_naming, pitch_adjust,
                    pipeline_mode, hybrid_static_cam, hybrid_use_dpvo,
                ],
                outputs=[skeleton_preview, face_mesh_preview, world_view_preview, bvh_download, fbx_download, face_csv_download, smplx_pt_download, perf_log],
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
