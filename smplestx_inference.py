"""SMPLest-X video inference wrapper.

Accepts a video path, extracts frames, runs batched inference,
saves all SMPL-X parameters to a .pt file, and produces a rendered overlay video.

Optimized for high-VRAM GPUs (RTX PRO 6000 96GB etc.) with batched YOLO
detection and batched model inference.
"""

import os
import sys
import argparse
import subprocess
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

# SMPLest-X project root (contains model code, pretrained weights, human_models)
SMPLESTX_ROOT = Path(os.environ.get("SMPLESTX_DIR", "/mnt/f/SMPLest-X")).resolve()
sys.path.insert(0, str(SMPLESTX_ROOT))

from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.inference_utils import non_max_suppression

# Lazy import — pyrender needs OpenGL which may not be available in headless WSL2
render_mesh = None

def _load_render_mesh():
    global render_mesh
    if render_mesh is None:
        from utils.visualization_utils import render_mesh as _rm
        render_mesh = _rm
    return render_mesh


def parse_args():
    parser = argparse.ArgumentParser(description="SMPLest-X video inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: demo/output/<video_stem>)")
    parser.add_argument("--ckpt_name", type=str, default="smplest_x_h", help="Checkpoint name")
    parser.add_argument("--fps", type=float, default=0, help="Extract at this FPS (0 = use video's native FPS)")
    parser.add_argument("--no_render", action="store_true", help="Skip rendering overlay video")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for model inference (default: 64, scale up for more VRAM)")
    parser.add_argument("--yolo_batch_size", type=int, default=32, help="Batch size for YOLO detection (default: 32)")
    return parser.parse_args()


def extract_frames(video_path: str, output_dir: str, fps: float = 0) -> tuple[int, float]:
    """Extract video frames to JPGs. Returns (frame_count, actual_fps)."""
    os.makedirs(output_dir, exist_ok=True)

    # Get native FPS if not specified
    if fps <= 0:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-f", "image2", "-vf", f"fps={fps}/1",
        "-qscale", "0",
        os.path.join(output_dir, "%06d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    frame_count = len(list(Path(output_dir).glob("*.jpg")))
    return frame_count, fps


def load_all_frames(frames_dir: Path, frame_count: int) -> list[np.ndarray]:
    """Pre-load all frames into memory."""
    frames = []
    for i in range(1, frame_count + 1):
        img_path = str(frames_dir / f"{i:06d}.jpg")
        frames.append(load_img(img_path))
    return frames


def batch_yolo_detect(detector, frames: list[np.ndarray], batch_size: int, conf: float) -> list[np.ndarray | None]:
    """Run YOLO detection in batches. Returns list of bbox_xyxy arrays (or None for no detection)."""
    all_bboxes = []

    for i in tqdm(range(0, len(frames), batch_size), desc="YOLO detection (batched)"):
        batch = frames[i:i + batch_size]
        results = detector.predict(
            batch,
            device="cuda",
            classes=0,
            conf=conf,
            save=False,
            verbose=False,
        )
        for result in results:
            xyxy = result.boxes.xyxy.detach().cpu().numpy()
            if len(xyxy) >= 1:
                all_bboxes.append(xyxy[0])  # largest/first detection
            else:
                all_bboxes.append(None)

    return all_bboxes


def preprocess_crops(frames: list[np.ndarray], bboxes: list[np.ndarray | None],
                     cfg, transform) -> tuple[list[torch.Tensor | None], list[np.ndarray | None]]:
    """Crop and preprocess all frames using their bboxes. Returns (tensors, processed_bboxes)."""
    tensors = []
    processed_bboxes = []

    for frame, bbox_xyxy in zip(frames, bboxes):
        if bbox_xyxy is None:
            tensors.append(None)
            processed_bboxes.append(None)
            continue

        h, w = frame.shape[:2]
        yolo_bbox_xywh = np.array([
            bbox_xyxy[0], bbox_xyxy[1],
            abs(bbox_xyxy[2] - bbox_xyxy[0]),
            abs(bbox_xyxy[3] - bbox_xyxy[1]),
        ])

        bbox = process_bbox(
            bbox=yolo_bbox_xywh,
            img_width=w,
            img_height=h,
            input_img_shape=cfg.model.input_img_shape,
            ratio=getattr(cfg.data, "bbox_ratio", 1.25),
        )
        img, _, _ = generate_patch_image(
            cvimg=frame,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=cfg.model.input_img_shape,
        )

        img_tensor = transform(img.astype(np.float32)) / 255
        tensors.append(img_tensor)
        processed_bboxes.append(bbox_xyxy)

    return tensors, processed_bboxes


def main():
    args = parse_args()
    cudnn.benchmark = True

    video_path = Path(args.video).resolve()
    assert video_path.exists(), f"Video not found: {video_path}"
    video_stem = video_path.stem

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = SMPLESTX_ROOT / "demo" / "output" / video_stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frames_dir = output_dir / "frames"
    print(f"[SMPLest-X] Extracting frames from {video_path}...")
    frame_count, fps = extract_frames(str(video_path), str(frames_dir), args.fps)
    print(f"[SMPLest-X] Extracted {frame_count} frames at {fps:.1f} FPS")

    if frame_count == 0:
        print("[SMPLest-X] ERROR: No frames extracted.")
        sys.exit(1)

    # Init config
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = str(SMPLESTX_ROOT / "pretrained_models" / args.ckpt_name / "config_base.py")
    cfg = Config.load_config(config_path)
    checkpoint_path = str(SMPLESTX_ROOT / "pretrained_models" / args.ckpt_name / f"{args.ckpt_name}.pth.tar")

    exp_name = f"inference_{video_stem}_{args.ckpt_name}_{time_str}"
    new_config = {
        "model": {"pretrained_model_path": checkpoint_path},
        "log": {
            "exp_name": exp_name,
            "log_dir": str(output_dir / "log"),
        },
    }
    cfg.update_config(new_config)
    cfg.prepare_log()

    # Init human model
    smpl_x = SMPLX(cfg.model.human_model_path)

    # Init model
    demoer = Tester(cfg)
    demoer.logger.info(f"[SMPLest-X] Processing {frame_count} frames from {video_stem} (batch_size={args.batch_size})")
    demoer._make_model()

    # Init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", "./pretrained_models/yolov8x.pt")
    detector = YOLO(bbox_model)

    transform = transforms.ToTensor()

    # ── Phase 1: Load all frames into memory ──
    print(f"[SMPLest-X] Loading {frame_count} frames into memory...")
    frames = load_all_frames(frames_dir, frame_count)
    print(f"[SMPLest-X] Frames loaded ({len(frames)} frames)")

    # ── Phase 2: Batch YOLO detection ──
    det_conf = getattr(cfg.inference.detection, "conf", 0.5)
    bboxes = batch_yolo_detect(detector, frames, args.yolo_batch_size, det_conf)
    del detector  # free YOLO model VRAM
    torch.cuda.empty_cache()

    detected = sum(1 for b in bboxes if b is not None)
    print(f"[SMPLest-X] YOLO: {detected}/{len(bboxes)} frames with detections")

    # ── Phase 3: Preprocess all crops ──
    print(f"[SMPLest-X] Preprocessing crops...")
    tensors, processed_bboxes = preprocess_crops(frames, bboxes, cfg, transform)

    # ── Phase 4: Batched model inference ──
    # Build index mapping: which frames have valid detections
    valid_indices = [i for i, t in enumerate(tensors) if t is not None]
    valid_tensors = [tensors[i] for i in valid_indices]

    # Pre-allocate output storage
    all_params = {
        "smplx_root_pose": [None] * frame_count,
        "smplx_body_pose": [None] * frame_count,
        "smplx_lhand_pose": [None] * frame_count,
        "smplx_rhand_pose": [None] * frame_count,
        "smplx_jaw_pose": [None] * frame_count,
        "smplx_shape": [None] * frame_count,
        "smplx_expr": [None] * frame_count,
        "cam_trans": [None] * frame_count,
        "bbox": [None] * frame_count,
    }

    # Fill zeros for frames with no detection
    zero_params = {
        "smplx_root_pose": torch.zeros(1, 3),
        "smplx_body_pose": torch.zeros(1, 63),
        "smplx_lhand_pose": torch.zeros(1, 45),
        "smplx_rhand_pose": torch.zeros(1, 45),
        "smplx_jaw_pose": torch.zeros(1, 3),
        "smplx_shape": torch.zeros(1, 10),
        "smplx_expr": torch.zeros(1, 10),
        "cam_trans": torch.zeros(1, 3),
        "bbox": torch.zeros(1, 4),
    }
    for i in range(frame_count):
        if tensors[i] is None:
            for key in all_params:
                all_params[key][i] = zero_params[key]

    # Run batched inference
    batch_size = args.batch_size
    mesh_results = {}  # frame_idx -> mesh (for rendering)

    render_dir = output_dir / "rendered"
    can_render = not args.no_render
    if can_render:
        try:
            _load_render_mesh()
            render_dir.mkdir(parents=True, exist_ok=True)
        except (ImportError, OSError) as e:
            print(f"[SMPLest-X] WARNING: Rendering disabled (OpenGL unavailable: {e})")
            can_render = False

    for batch_start in tqdm(range(0, len(valid_indices), batch_size), desc="SMPLest-X inference (batched)"):
        batch_idx = valid_indices[batch_start:batch_start + batch_size]
        batch_tensors = valid_tensors[batch_start:batch_start + batch_size]

        # Stack into a single batch tensor
        img_batch = torch.stack(batch_tensors).cuda()

        with torch.no_grad():
            out = demoer.model({"img": img_batch}, {}, {}, "test")

        # Unpack batch results
        for j, frame_idx in enumerate(batch_idx):
            all_params["smplx_root_pose"][frame_idx] = out["smplx_root_pose"][j:j+1].cpu()
            all_params["smplx_body_pose"][frame_idx] = out["smplx_body_pose"][j:j+1].cpu()
            all_params["smplx_lhand_pose"][frame_idx] = out["smplx_lhand_pose"][j:j+1].cpu()
            all_params["smplx_rhand_pose"][frame_idx] = out["smplx_rhand_pose"][j:j+1].cpu()
            all_params["smplx_jaw_pose"][frame_idx] = out["smplx_jaw_pose"][j:j+1].cpu()
            all_params["smplx_shape"][frame_idx] = out["smplx_shape"][j:j+1].cpu()
            all_params["smplx_expr"][frame_idx] = out["smplx_expr"][j:j+1].cpu()
            all_params["cam_trans"][frame_idx] = out["cam_trans"][j:j+1].cpu()
            all_params["bbox"][frame_idx] = torch.tensor(processed_bboxes[frame_idx]).unsqueeze(0)

            if can_render and "smplx_mesh_cam" in out:
                mesh_results[frame_idx] = out["smplx_mesh_cam"][j].detach().cpu().numpy()

    # ── Phase 5: Render overlays (if enabled) ──
    if can_render and mesh_results:
        print(f"[SMPLest-X] Rendering {len(mesh_results)} overlay frames...")
        for frame_idx in tqdm(range(frame_count), desc="Rendering overlays"):
            vis_img = frames[frame_idx].copy()

            if frame_idx in mesh_results:
                mesh = mesh_results[frame_idx]
                bbox_xyxy = processed_bboxes[frame_idx]
                bbox_xywh = np.array([
                    bbox_xyxy[0], bbox_xyxy[1],
                    abs(bbox_xyxy[2] - bbox_xyxy[0]),
                    abs(bbox_xyxy[3] - bbox_xyxy[1]),
                ])
                bbox = process_bbox(
                    bbox=bbox_xywh,
                    img_width=vis_img.shape[1],
                    img_height=vis_img.shape[0],
                    input_img_shape=cfg.model.input_img_shape,
                    ratio=getattr(cfg.data, "bbox_ratio", 1.25),
                )
                focal = [
                    cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2],
                    cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3],
                ]
                princpt = [
                    cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0],
                    cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1],
                ]
                vis_img = cv2.rectangle(
                    vis_img,
                    (int(bbox_xyxy[0]), int(bbox_xyxy[1])),
                    (int(bbox_xyxy[2]), int(bbox_xyxy[3])),
                    (0, 255, 0), 1,
                )
                vis_img = render_mesh(
                    vis_img, mesh, smpl_x.face,
                    {"focal": focal, "princpt": princpt},
                    mesh_as_vertices=False,
                )

            cv2.imwrite(str(render_dir / f"{frame_idx + 1:06d}.jpg"), vis_img[:, :, ::-1])

    # ── Phase 6: Save results ──
    results = {}
    for key, val_list in all_params.items():
        results[key] = torch.cat(val_list, dim=0)

    results["fps"] = fps
    results["video_path"] = str(video_path)
    results["frame_count"] = frame_count

    # Save parameters
    pt_path = output_dir / f"{video_stem}_smplx.pt"
    torch.save(results, str(pt_path))
    print(f"[SMPLest-X] Saved SMPL-X parameters to {pt_path}")
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: {val.shape}")

    # Reassemble rendered video
    if can_render:
        video_out = output_dir / f"{video_stem}_smplestx_overlay.mp4"
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "image2", "-r", str(fps),
            "-i", str(render_dir / "%06d.jpg"),
            "-vcodec", "libx264", "-crf", "17",
            "-pix_fmt", "yuv420p",
            str(video_out),
        ]
        subprocess.run(ffmpeg_cmd, capture_output=True)
        print(f"[SMPLest-X] Rendered overlay video: {video_out}")

    print(f"[SMPLest-X] Done.")


if __name__ == "__main__":
    main()
