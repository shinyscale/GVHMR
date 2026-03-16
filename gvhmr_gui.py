"""bodypipe — Body + Hand + Face Motion Capture GUI."""

import os
import subprocess
import re
import shutil
import json
from pathlib import Path

import numpy as np
import torch
import gradio as gr

from preprocess import preprocess_video, get_video_fps
from face_capture import run_face_pipeline, render_face_mesh_video, extract_face_crops_from_keypoints
from smplx_to_bvh import (
    convert_smplx_to_bvh,
    convert_params_to_bvh,
    extract_gvhmr_params,
    extract_smplx_params,
    merge_gvhmr_smplestx_params,
)
from hamer_inference import run_hamer, merge_gvhmr_hamer_params
from bvh_to_fbx import convert_bvh_to_fbx
from visualize_skeleton import render_skeleton_video, render_world_views, render_hand_overlay_video
from multi_person_split import split_multi_person_video, MultiPersonResult
from identity_panel import build_identity_panel, init_panel_state, populate_panel

GVHMR_DIR = Path(__file__).resolve().parent
DEMO_SCRIPT = GVHMR_DIR / "tools" / "demo" / "demo.py"
SMPLESTX_DIR = Path("/mnt/f/SMPLest-X")
SMPLESTX_ENV = "smplestx"
SMPLESTX_PYTHON = Path("/home/shinyscale/miniconda3/envs") / SMPLESTX_ENV / "bin" / "python"

# ── Dark orange theme matching facepipe's visual style ──

BODYPIPE_THEME = gr.themes.Default(
    primary_hue="orange",
    neutral_hue="slate",
)

BODYPIPE_CSS = """
.gradio-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
#bodypipe-header {
    background: linear-gradient(135deg, #7C2D12 0%, #B45309 50%, #D97706 100%);
    padding: 14px 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    width: 100vw;
    margin-top: calc(-1 * var(--layout-gap, 16px));
    margin-bottom: 16px;
}
"""

BODYPIPE_HEADER = """
<div id="bodypipe-header">
    <h1 style="color:white; margin:0; font-size:24px; font-weight:700; font-family:Roboto,Inter,system-ui,sans-serif;">bodypipe</h1>
    <span style="color:rgba(255,255,255,0.70); font-size:14px; font-family:Roboto,Inter,system-ui,sans-serif;">Markerless Performance Capture Suite</span>
</div>
"""

BODYPIPE_JS = """
() => {
    document.body.classList.add('dark');
}
"""

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


def save_solve_config(output_dir: Path, tab: str, **params) -> Path:
    """Save UI parameters alongside solve results so they can be restored."""
    config = {"tab": tab, **params}
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "solve_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def load_solve_config(video_path: str, tab: str) -> dict | None:
    """Load saved UI parameters for a previously-solved video, if they exist."""
    stem = Path(video_path).stem
    tab_dirs = {
        "gvhmr": "demo",
        "perfcap": "perfcap",
        "multi_person": "multi_person",
    }
    subdir = tab_dirs.get(tab)
    if not subdir:
        return None
    config_path = GVHMR_DIR / "outputs" / subdir / stem / "solve_config.json"
    if config_path.is_file():
        with open(config_path) as f:
            return json.load(f)
    return None


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

    # Save solve config for session restore
    stem = Path(video_path).stem
    save_solve_config(
        GVHMR_DIR / "outputs" / "demo" / stem,
        tab="gvhmr",
        static_cam=bool(static_cam),
        use_dpvo=bool(use_dpvo),
        focal_length=focal_length or "",
    )

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
        str(SMPLESTX_PYTHON),
        str(GVHMR_DIR / "smplestx_inference.py"),
        "--video", str(video_path),
        "--fps", str(fps),
        "--output_dir", str(output_dir),
        "--no_render",
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
        "coordinate_space": params.get("coordinate_space", "world"),
        "camera_model": params.get("camera_model", "world_space"),
        "translation_origin": params.get("translation_origin", "model_origin"),
        "source": params.get("source", "hybrid"),
    }
    if "betas" in params:
        save_dict["betas"] = torch.tensor(params["betas"], dtype=torch.float32)
    if "bbox" in params:
        save_dict["bbox"] = torch.tensor(params["bbox"], dtype=torch.float32)
    for key in [
        "transl_cam",
        "transl_world",
        "global_orient_cam",
        "global_orient_world",
        "body_pose_cam",
        "body_pose_world",
        "K_fullimg",
    ]:
        if key in params:
            value = params[key]
            if key.startswith("body_pose"):
                value = value.reshape(params["num_frames"], -1)
            save_dict[key] = torch.tensor(value, dtype=torch.float32)
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
    use_vitpose_face: bool = True,
    hand_source: str = "SMPLest-X (default)",
    body_smooth_preset: str = "Moderate (default)",
    progress=gr.Progress(track_tqdm=False),
):
    """Run the full performance capture pipeline: body+hands+face."""

    source_video_path = resolve_video_path(video_upload, video_path_text)
    log_lines = []
    is_hybrid = "Hybrid" in pipeline_mode

    # Parse FPS
    try:
        fps = float(target_fps.strip()) if target_fps and target_fps.strip() else 30.0
    except ValueError:
        fps = 30.0

    source_fps = get_video_fps(source_video_path)
    source_stem = Path(source_video_path).stem

    # Set up output directory early so shared preprocessing artifacts land with the run.
    output_base = GVHMR_DIR / "outputs" / "perfcap" / source_stem
    output_base.mkdir(parents=True, exist_ok=True)

    # Save solve config for session restore
    save_solve_config(
        output_base,
        tab="perfcap",
        target_fps=target_fps or "30",
        fbx_naming=fbx_naming,
        pitch_adjust=float(pitch_adjust),
        pipeline_mode=pipeline_mode,
        static_cam=bool(hybrid_static_cam),
        use_dpvo=bool(hybrid_use_dpvo),
        use_vitpose_face=bool(use_vitpose_face),
        hand_source=hand_source,
        body_smooth_preset=body_smooth_preset,
    )

    # ── Stage 1: Preprocess (portrait fix) ──
    progress(0.02, desc="Preprocessing video...")
    video_path, preprocess_msg = preprocess_video(
        source_video_path,
        output_dir=str(output_base / "preprocess"),
        target_fps=fps if fps > 0 else None,
    )
    log_lines.append(f"[Preprocess] {preprocess_msg}")

    # Detect actual FPS after shared preprocessing / resampling.
    actual_fps = get_video_fps(video_path)
    if fps <= 0:
        fps = actual_fps
    log_lines.append(f"[Info] Source video FPS: {source_fps:.2f}")
    log_lines.append(f"[Info] Analysis video FPS: {actual_fps:.2f}")
    log_lines.append(f"[Info] Target FPS: {fps:.2f}")
    log_lines.append(f"[Info] Pipeline mode: {pipeline_mode}")
    if abs(actual_fps - fps) > 0.5:
        log_lines.append("[Info] WARNING: Shared preprocessing did not fully match the requested FPS.")
    video_stem = source_stem

    # Track merged params for hybrid pipeline
    merged_camera_params = None
    merged_world_params = None
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
            return None, None, None, None, None, None, None, None, full_log

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
            return None, None, None, None, None, None, None, None, full_log

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
            merged_world_params = merge_gvhmr_smplestx_params(
                gvhmr_params,
                smplestx_params,
                coordinate_space="world",
            )
            merged_camera_params = merge_gvhmr_smplestx_params(
                gvhmr_params,
                smplestx_params,
                coordinate_space="camera",
            )

            log_lines.append(f"[Merge] Body from GVHMR: {gvhmr_params['num_frames']} frames")
            log_lines.append(f"[Merge] Hands from SMPLest-X: {smplestx_params['num_frames']} frames")
            log_lines.append(f"[Merge] Hybrid world bundle: {merged_world_params['num_frames']} frames")
            log_lines.append(f"[Merge] Hybrid camera bundle: {merged_camera_params['num_frames']} frames")

            # Save merged .pt for re-export
            merged_pt_path = str(output_base / f"{video_stem}_hybrid_smplx.pt")
            _save_merged_pt(merged_world_params, merged_pt_path)
            pt_path = merged_pt_path
            log_lines.append(f"[Merge] Saved: {merged_pt_path}")
        except Exception as e:
            log_lines.append(f"[Merge] ERROR: {e}")
            full_log = "\n".join(log_lines)
            return None, None, None, None, None, None, None, None, full_log

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
            return None, None, None, None, None, None, None, None, full_log

        log_lines.append(f"\n[SMPLest-X] Output: {pt_path}")
        progress(0.45, desc="SMPLest-X complete.")

        # Get bboxes for face capture
        bboxes = _extract_bboxes_from_smplestx(smplestx_output_dir)

    # ── Stage 2d (optional): HaMeR Hand Capture ──
    if "HaMeR" in hand_source and is_hybrid and merged_world_params is not None:
        progress(0.48, desc="Running HaMeR hand capture...")
        log_lines.append("")
        log_lines.append("=" * 60)
        log_lines.append("[Stage 2d] HaMeR — Dedicated Hand Mesh Recovery")
        log_lines.append("=" * 60)

        try:
            # Find vitpose.pt for wrist keypoints
            hamer_vitpose = None
            gvhmr_out = find_output_dir(video_path)
            if gvhmr_out:
                vp = find_file(gvhmr_out, "vitpose.pt")
                if vp:
                    hamer_vitpose = vp

            hamer_result = run_hamer(
                video_path,
                vitpose_path=hamer_vitpose,
                person_bboxes=bboxes,
                device="cuda",
            )
            merged_world_params = merge_gvhmr_hamer_params(merged_world_params, hamer_result)
            merged_camera_params = merge_gvhmr_hamer_params(merged_camera_params, hamer_result)
            log_lines.append(f"[HaMeR] Left hand: {(hamer_result['left_confidence'] > 0.5).sum()} confident frames")
            log_lines.append(f"[HaMeR] Right hand: {(hamer_result['right_confidence'] > 0.5).sum()} confident frames")

            # Re-save merged .pt with HaMeR hands
            merged_pt_path = str(output_base / f"{video_stem}_hybrid_smplx.pt")
            _save_merged_pt(merged_world_params, merged_pt_path)
            pt_path = merged_pt_path
        except Exception as e:
            log_lines.append(f"[HaMeR] ERROR: {e} — falling back to SMPLest-X hands")

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

    # Find vitpose.pt from GVHMR preprocess directory
    vitpose_path = None
    if use_vitpose_face:
        gvhmr_output_dir = find_output_dir(video_path)
        if gvhmr_output_dir:
            vp = find_file(gvhmr_output_dir, "vitpose.pt")
            if vp:
                vitpose_path = vp
                log_lines.append(f"[Face] Found ViTPose: {vitpose_path}")
        # Also check the preprocess subdirectory pattern
        if not vitpose_path:
            preprocess_dir = GVHMR_DIR / "outputs" / "demo" / video_stem / "preprocess"
            vp_candidate = preprocess_dir / "vitpose.pt"
            if vp_candidate.is_file():
                vitpose_path = str(vp_candidate)
                log_lines.append(f"[Face] Found ViTPose: {vitpose_path}")
        if not vitpose_path:
            log_lines.append("[Face] ViTPose data not found, falling back to bbox crops.")

    def face_progress(frac, msg):
        overall = 0.50 + frac * 0.20
        progress(overall, desc=msg)
        log_lines.append(f"[Face] {msg}")

    try:
        run_face_pipeline(
            video_path, bboxes, face_csv_path, fps=fps,
            progress_callback=face_progress,
            vitpose_path=vitpose_path,
            use_vitpose_crops=use_vitpose_face and vitpose_path is not None,
        )
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
            vitpose_path=vitpose_path,
            use_vitpose_crops=use_vitpose_face and vitpose_path is not None,
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

    # Parse smoothing preset
    _preset_map = {"Light": "light", "Moderate (default)": "moderate", "Heavy": "heavy"}
    smooth_key = _preset_map.get(body_smooth_preset, "moderate")

    try:
        if is_hybrid and merged_world_params is not None:
            # Hybrid: use merged params directly, skip world grounding (GVHMR is already grounded),
            # only smooth hands (GVHMR body is temporally coherent)
            convert_params_to_bvh(
                merged_world_params, bvh_path, fps=fps,
                skip_world_grounding=True,
                smooth_body=False,
                smooth_hands=True,
                body_smooth_preset=smooth_key,
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

        if is_hybrid and merged_world_params is not None:
            render_world_views(
                output_path=world_view_video, fps=fps,
                params=merged_world_params,
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
    progress(0.90, desc="Rendering skeleton overlay...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 6] Skeleton Overlay Visualization")
    log_lines.append("=" * 60)

    skeleton_video = str(output_base / f"{video_stem}_skeleton.mp4")
    try:
        def viz_progress(frac, msg):
            progress(0.90 + frac * 0.04, desc=msg)

        if is_hybrid and merged_camera_params is not None:
            log_lines.append("[Viz] Rendering Hybrid-camera overlay (GVHMR body/root + SMPLest-X hands).")
            render_skeleton_video(
                video_path=video_path,
                output_path=skeleton_video,
                fps=fps,
                progress_callback=viz_progress,
                params=merged_camera_params,
                coordinate_space="camera",
                render_label="Hybrid-camera",
            )
            log_lines.append(f"[Viz] Skeleton video: {skeleton_video}")
        else:
            render_skeleton_video(
                pt_path,
                video_path,
                skeleton_video,
                fps=fps,
                progress_callback=viz_progress,
                coordinate_space="camera",
                render_label="SMPLest-X camera",
            )
            log_lines.append(f"[Viz] Skeleton video: {skeleton_video}")
    except Exception as e:
        log_lines.append(f"[Viz] ERROR: {e}")
        skeleton_video = None

    # ── Stage 6b: Hand Overlay Visualization ──
    hand_overlay_video = None
    progress(0.94, desc="Rendering hand overlay...")
    log_lines.append("")
    log_lines.append("=" * 60)
    log_lines.append("[Stage 6b] Hand Overlay Visualization")
    log_lines.append("=" * 60)

    hand_overlay_video = str(output_base / f"{video_stem}_hand_overlay.mp4")
    try:
        def hand_progress(frac, msg):
            progress(0.94 + frac * 0.05, desc=msg)

        if is_hybrid and merged_camera_params is not None:
            render_hand_overlay_video(
                video_path=video_path,
                output_path=hand_overlay_video,
                fps=fps,
                progress_callback=hand_progress,
                params=merged_camera_params,
                coordinate_space="camera",
                render_label="Hands (Hybrid)",
            )
        else:
            render_hand_overlay_video(
                pt_path=pt_path,
                video_path=video_path,
                output_path=hand_overlay_video,
                fps=fps,
                progress_callback=hand_progress,
                coordinate_space="camera",
                render_label="Hands (SMPLest-X)",
            )
        log_lines.append(f"[Hands] Hand overlay video: {hand_overlay_video}")
    except Exception as e:
        log_lines.append(f"[Hands] ERROR: {e}")
        hand_overlay_video = None

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
    log_lines.append(f"  HandOvl:   {hand_overlay_video or 'SKIPPED'}")
    log_lines.append("=" * 60)

    full_log = "\n".join(log_lines)
    return skeleton_video, face_mesh_video, world_view_video, hand_overlay_video, bvh_path, fbx_path, face_csv_path, pt_path, full_log


# ── Multi-Person Pipeline ──

def run_multi_person_pipeline(
    video_upload,
    video_path_text: str,
    target_fps: str,
    fbx_naming: str,
    multi_static_cam: bool,
    multi_use_dpvo: bool,
    max_persons: int,
    render_overlays: bool = False,
    use_inpainting: bool = False,
    progress=gr.Progress(track_tqdm=False),
):
    """Run the multi-person capture pipeline."""

    source_video_path = resolve_video_path(video_upload, video_path_text)
    log_lines = []

    # Parse FPS
    try:
        fps = float(target_fps.strip()) if target_fps and target_fps.strip() else 30.0
    except ValueError:
        fps = 30.0

    source_stem = Path(source_video_path).stem

    # Save solve config for session restore
    save_solve_config(
        GVHMR_DIR / "outputs" / "multi_person" / source_stem,
        tab="multi_person",
        target_fps=target_fps or "30",
        fbx_naming=fbx_naming,
        static_cam=bool(multi_static_cam),
        use_dpvo=bool(multi_use_dpvo),
        max_persons=int(max_persons) if max_persons else 0,
        use_inpainting=bool(use_inpainting),
    )

    # Preprocess
    progress(0.02, desc="Preprocessing video...")
    video_path, preprocess_msg = preprocess_video(
        source_video_path,
        output_dir=str(GVHMR_DIR / "outputs" / "multi_person" / source_stem / "preprocess"),
        target_fps=fps if fps > 0 else None,
    )
    log_lines.append(f"[Preprocess] {preprocess_msg}")

    output_dir = GVHMR_DIR / "outputs" / "multi_person" / source_stem

    def mp_progress(frac, msg):
        progress(0.05 + frac * 0.85, desc=msg)
        log_lines.append(f"[MultiPerson] {msg}")

    try:
        result = split_multi_person_video(
            video_path=video_path,
            output_dir=str(output_dir),
            min_track_duration=30,
            static_cam=multi_static_cam,
            use_dpvo=multi_use_dpvo,
            max_persons=int(max_persons) if max_persons else 0,
            render_overlays=render_overlays,
            use_inpainting=use_inpainting,
            progress_callback=mp_progress,
        )
    except Exception as e:
        import traceback
        log_lines.append(f"[MultiPerson] ERROR: {e}")
        log_lines.append(traceback.format_exc())
        full_log = "\n".join(log_lines)
        return None, None, 0, None, None, None, None, None, full_log

    # Collect outputs
    track_viz = None
    detection_dir = output_dir / "detection"
    track_viz_path = detection_dir / "track_visualization.mp4"
    if track_viz_path.exists():
        track_viz = str(track_viz_path)

    scene_preview = None
    assembly_dir = output_dir / "assembly"
    scene_preview_path = assembly_dir / "scene_preview.mp4"
    if scene_preview_path.exists():
        scene_preview = str(scene_preview_path)

    # Collect per-person BVH files and convert to FBX
    bvh_files = []
    fbx_files = []
    naming_key = "ue5" if "ue5" in fbx_naming.lower() else "mixamo"
    for i, person_dir in enumerate(result.person_dirs):
        person_dir = Path(person_dir)
        bvh = list(person_dir.rglob("*.bvh"))
        if bvh:
            bvh_path = str(bvh[0])
            bvh_files.append(bvh_path)
            fbx_path = str(Path(bvh_path).with_suffix(".fbx"))
            if not Path(fbx_path).exists():
                progress(0.92 + i * 0.02, desc=f"Converting person {i} BVH to FBX...")
                fbx_log = convert_bvh_to_fbx(bvh_path, fbx_path, fps=fps, naming=naming_key)
                log_lines.append(f"[FBX] Person {i}: {fbx_log}")
                if "ERROR" in fbx_log:
                    continue
            fbx_files.append(fbx_path)
        else:
            fbx = list(person_dir.rglob("*.fbx"))
            if fbx:
                fbx_files.append(str(fbx[0]))

    manifest_path = output_dir / "session_manifest.json"
    manifest = str(manifest_path) if manifest_path.exists() else None

    # Collect confidence CSVs
    confidence_csv_files = []
    for i, person_dir in enumerate(result.person_dirs):
        person_dir = Path(person_dir)
        csv_path = person_dir / "confidence.csv"
        if csv_path.exists():
            confidence_csv_files.append(str(csv_path))

    # Initialize identity panel state
    panel_state_dict = None
    if result.identity_tracks:
        try:
            panel_state_dict = init_panel_state(
                video_path=video_path,
                identity_tracks=result.identity_tracks,
                all_tracks=result.all_tracks,
                person_dirs=result.person_dirs,
                fps=fps if fps > 0 else 30.0,
                output_dir=str(output_dir),
                pipeline_params={
                    "static_cam": multi_static_cam,
                    "use_dpvo": multi_use_dpvo,
                },
                inactive_tracks=result.inactive_tracks,
            )
        except Exception as e:
            log_lines.append(f"[Identity Panel] Init warning: {e}")

    progress(1.0, desc=f"Complete! {result.num_persons} people processed.")
    log_lines.append(f"\n[Done] {result.num_persons} people processed.")
    log_lines.append(f"  Output: {output_dir}")
    full_log = "\n".join(log_lines)

    return (
        track_viz,
        scene_preview,
        result.num_persons,
        bvh_files if bvh_files else None,
        fbx_files if fbx_files else None,
        manifest,
        panel_state_dict,
        confidence_csv_files if confidence_csv_files else None,
        full_log,
    )


# ── Gradio UI ──

with gr.Blocks(
    title="bodypipe",
    theme=BODYPIPE_THEME,
    css=BODYPIPE_CSS,
    js=BODYPIPE_JS,
) as app:
    gr.HTML(BODYPIPE_HEADER)

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
                    static_cam.change(
                        fn=lambda s: gr.update(interactive=not s, value=False if s else gr.update()),
                        inputs=[static_cam], outputs=[use_dpvo],
                    )
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

            # Restore saved params when a video path is entered
            def _restore_gvhmr_config(video_path_text):
                if not video_path_text or not video_path_text.strip():
                    return gr.update(), gr.update(), gr.update()
                cfg = load_solve_config(video_path_text.strip(), "gvhmr")
                if cfg is None:
                    return gr.update(), gr.update(), gr.update()
                return (
                    gr.update(value=cfg.get("static_cam", False)),
                    gr.update(value=cfg.get("use_dpvo", False)),
                    gr.update(value=cfg.get("focal_length", "")),
                )

            gvhmr_video_path.change(
                fn=_restore_gvhmr_config,
                inputs=[gvhmr_video_path],
                outputs=[static_cam, use_dpvo, focal_length],
            )
            gvhmr_video_upload.upload(
                fn=lambda v: _restore_gvhmr_config(v),
                inputs=[gvhmr_video_upload],
                outputs=[static_cam, use_dpvo, focal_length],
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
                    hybrid_static_cam.change(
                        fn=lambda s: gr.update(interactive=not s, value=False if s else gr.update()),
                        inputs=[hybrid_static_cam], outputs=[hybrid_use_dpvo],
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
                    use_vitpose_face = gr.Checkbox(
                        label="Use ViTPose face crops",
                        value=True,
                        info="Use ViTPose keypoints for tight face crops (better for full-body shots)",
                    )
                    hand_source = gr.Radio(
                        label="Hand Source",
                        choices=["SMPLest-X (default)", "HaMeR"],
                        value="SMPLest-X (default)",
                        info="HaMeR produces better finger articulation but requires extra model download",
                    )
                    body_smooth_preset = gr.Dropdown(
                        label="Body Smoothing",
                        choices=["Light", "Moderate (default)", "Heavy"],
                        value="Moderate (default)",
                        info="Controls temporal smoothing strength — heavier = less jitter but more latency",
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
                hand_overlay_preview = gr.Video(label="Hand Overlay")

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
                    pipeline_mode, hybrid_static_cam, hybrid_use_dpvo, use_vitpose_face,
                    hand_source, body_smooth_preset,
                ],
                outputs=[skeleton_preview, face_mesh_preview, world_view_preview, hand_overlay_preview, bvh_download, fbx_download, face_csv_download, smplx_pt_download, perf_log],
            )

            # Restore saved params when a video path is entered
            def _restore_perfcap_config(video_path_text):
                if not video_path_text or not video_path_text.strip():
                    return (gr.update(),) * 9
                cfg = load_solve_config(video_path_text.strip(), "perfcap")
                if cfg is None:
                    return (gr.update(),) * 9
                return (
                    gr.update(value=cfg.get("target_fps", "30")),
                    gr.update(value=cfg.get("fbx_naming", "Mixamo (Cascadeur)")),
                    gr.update(value=cfg.get("pitch_adjust", 0)),
                    gr.update(value=cfg.get("pipeline_mode", "Hybrid: GVHMR Body + SMPLest-X Hands")),
                    gr.update(value=cfg.get("static_cam", False)),
                    gr.update(value=cfg.get("use_dpvo", False)),
                    gr.update(value=cfg.get("use_vitpose_face", True)),
                    gr.update(value=cfg.get("hand_source", "SMPLest-X (default)")),
                    gr.update(value=cfg.get("body_smooth_preset", "Moderate (default)")),
                )

            perf_video_path.change(
                fn=_restore_perfcap_config,
                inputs=[perf_video_path],
                outputs=[
                    perf_fps, fbx_naming, pitch_adjust, pipeline_mode,
                    hybrid_static_cam, hybrid_use_dpvo, use_vitpose_face,
                    hand_source, body_smooth_preset,
                ],
            )
            perf_video_upload.upload(
                fn=lambda v: _restore_perfcap_config(v),
                inputs=[perf_video_upload],
                outputs=[
                    perf_fps, fbx_naming, pitch_adjust, pipeline_mode,
                    hybrid_static_cam, hybrid_use_dpvo, use_vitpose_face,
                    hand_source, body_smooth_preset,
                ],
            )

        # ── Tab 3: Multi-Person Capture ──
        with gr.TabItem("Multi-Person"):
            gr.Markdown(
                "**Multi-Person Motion Capture** from a single monocular video.\n\n"
                "Detects and tracks N people, isolates each into a clean single-person video "
                "(using segmentation + inpainting when people overlap), then runs the full "
                "pipeline per person. Results are assembled in shared world coordinates.\n\n"
                "For front-crossings and heavy occlusion, enable inpainting or expect ID swaps / body erasure.\n\n"
                "Requires: `segment-anything-2` (SAM2), `ProPainter` (optional, for occlusion handling)."
            )

            with gr.Row():
                with gr.Column(scale=1):
                    mp_video_upload = gr.Video(label="Upload Video", sources=["upload"])
                    mp_video_path = gr.Textbox(
                        label="Or enter WSL2 file path",
                        placeholder="/mnt/f/Videos/two_people_scene.mp4",
                    )

                with gr.Column(scale=1):
                    with gr.Row():
                        mp_static_cam = gr.Checkbox(
                            label="Static Camera", value=False,
                            info="Check if camera is fixed/tripod",
                        )
                        mp_use_dpvo = gr.Checkbox(
                            label="Use DPVO", value=False,
                            info="Better camera estimation, slower",
                        )
                    mp_static_cam.change(
                        fn=lambda s: gr.update(interactive=not s, value=False if s else gr.update()),
                        inputs=[mp_static_cam], outputs=[mp_use_dpvo],
                    )
                    mp_max_persons = gr.Number(
                        label="Max Persons (0 = all)",
                        value=0,
                        precision=0,
                        info="Limit to N largest people by bbox area",
                    )
                    mp_fps = gr.Textbox(
                        label="Target FPS",
                        value="30",
                        placeholder="30",
                    )
                    mp_fbx_naming = gr.Dropdown(
                        label="FBX Bone Naming",
                        choices=["Mixamo (Cascadeur)", "UE5 Mannequin"],
                        value="Mixamo (Cascadeur)",
                    )
                    mp_render_overlays = gr.Checkbox(
                        label="Render per-person overlays",
                        value=False,
                        info="Render in-camera mesh overlay per person (slower)",
                    )
                    mp_use_inpainting = gr.Checkbox(
                        label="SAM2 + ProPainter inpainting",
                        value=True,
                        info="Pixel-accurate isolation via segmentation + video inpainting. "
                             "Required for front-crossings / heavy occlusion. Very slow (~5min/person).",
                    )
                    mp_run_btn = gr.Button(
                        "Run Multi-Person Pipeline",
                        variant="primary",
                        size="lg",
                    )

            gr.Markdown("### Detection & Tracking")
            with gr.Row():
                mp_track_viz = gr.Video(label="Track Visualization")
                mp_person_count = gr.Number(label="People Detected", precision=0)

            gr.Markdown("### Results")
            with gr.Row():
                mp_scene_preview = gr.Video(label="Scene Preview")

            with gr.Row():
                mp_bvh_downloads = gr.File(label="Per-Person BVH Files", file_count="multiple")
                mp_fbx_downloads = gr.File(label="Per-Person FBX Files", file_count="multiple")
                mp_manifest = gr.File(label="Session Manifest")

            gr.Markdown(
                "*Use the **Pose Corrector** inside Identity Inspector below to fix bad poses "
                "and re-export corrected BVH/FBX.*"
            )

            with gr.Accordion("Identity Inspector", open=False) as id_inspector_accordion:
                id_panel = build_identity_panel(scene_preview_video=mp_scene_preview)
                mp_confidence_files = gr.File(
                    label="Confidence CSVs",
                    file_count="multiple",
                )

            with gr.Accordion("Pipeline Log", open=False):
                mp_log = gr.Textbox(
                    label="Log Output", lines=25, max_lines=100,
                    interactive=False,
                )

            mp_run_btn.click(
                fn=run_multi_person_pipeline,
                inputs=[
                    mp_video_upload, mp_video_path, mp_fps, mp_fbx_naming,
                    mp_static_cam, mp_use_dpvo, mp_max_persons, mp_render_overlays,
                    mp_use_inpainting,
                ],
                outputs=[
                    mp_track_viz, mp_scene_preview, mp_person_count,
                    mp_bvh_downloads, mp_fbx_downloads, mp_manifest,
                    id_panel["state"], mp_confidence_files, mp_log,
                ],
            ).then(
                fn=populate_panel,
                inputs=[id_panel["state"]],
                outputs=[
                    id_panel["state"],
                    id_panel["person_dropdown"],
                    id_panel["swap_target"],
                    id_panel["frame_slider"],
                    id_panel["frame_label"],
                    # display_outputs (9 components from update_frame_display):
                    id_panel["frame_image"],
                    id_panel["conf_detection"], id_panel["conf_visibility"],
                    id_panel["conf_overlap"], id_panel["conf_shape"],
                    id_panel["conf_motion"], id_panel["conf_overall"],
                    id_panel["confidence_plot"],
                    id_panel["kf_dataframe"],
                    # Track merge & crossing outputs:
                    id_panel["merge_target"],
                    id_panel["crossing_spans_df"],
                    id_panel["reid_gallery"],
                ],
            )

            # Restore saved params when a video path is entered
            def _restore_mp_config(video_path_text):
                if not video_path_text or not video_path_text.strip():
                    return (gr.update(),) * 6
                cfg = load_solve_config(video_path_text.strip(), "multi_person")
                if cfg is None:
                    return (gr.update(),) * 6
                return (
                    gr.update(value=cfg.get("static_cam", False)),
                    gr.update(value=cfg.get("use_dpvo", False)),
                    gr.update(value=cfg.get("max_persons", 0)),
                    gr.update(value=cfg.get("target_fps", "30")),
                    gr.update(value=cfg.get("fbx_naming", "Mixamo (Cascadeur)")),
                    gr.update(value=cfg.get("use_inpainting", False)),
                )

            mp_video_path.change(
                fn=_restore_mp_config,
                inputs=[mp_video_path],
                outputs=[mp_static_cam, mp_use_dpvo, mp_max_persons, mp_fps, mp_fbx_naming, mp_use_inpainting],
            )
            mp_video_upload.upload(
                fn=lambda v: _restore_mp_config(v),
                inputs=[mp_video_upload],
                outputs=[mp_static_cam, mp_use_dpvo, mp_max_persons, mp_fps, mp_fbx_naming, mp_use_inpainting],
            )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
