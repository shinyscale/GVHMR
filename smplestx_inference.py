"""SMPLest-X video inference wrapper.

Accepts a video path, extracts frames, runs per-frame inference,
saves all SMPL-X parameters to a .pt file, and produces a rendered overlay video.
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
    demoer.logger.info(f"[SMPLest-X] Processing {frame_count} frames from {video_stem}")
    demoer._make_model()

    # Init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", "./pretrained_models/yolov8x.pt")
    detector = YOLO(bbox_model)

    transform = transforms.ToTensor()

    # Storage for per-frame SMPL-X parameters
    all_params = {
        "smplx_root_pose": [],
        "smplx_body_pose": [],
        "smplx_lhand_pose": [],
        "smplx_rhand_pose": [],
        "smplx_jaw_pose": [],
        "smplx_shape": [],
        "smplx_expr": [],
        "cam_trans": [],
        "bbox": [],
    }

    render_dir = output_dir / "rendered"
    can_render = not args.no_render
    if can_render:
        try:
            _load_render_mesh()
            render_dir.mkdir(parents=True, exist_ok=True)
        except (ImportError, OSError) as e:
            print(f"[SMPLest-X] WARNING: Rendering disabled (OpenGL unavailable: {e})")
            can_render = False

    for frame_idx in tqdm(range(1, frame_count + 1), desc="SMPLest-X inference"):
        img_path = str(frames_dir / f"{frame_idx:06d}.jpg")
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        h, w = original_img.shape[:2]

        # Detect person
        yolo_bbox = detector.predict(
            original_img,
            device="cuda",
            classes=0,
            conf=cfg.inference.detection.conf,
            save=cfg.inference.detection.save,
            verbose=cfg.inference.detection.verbose,
        )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox) < 1:
            # No detection — store zeros
            all_params["smplx_root_pose"].append(torch.zeros(1, 3))
            all_params["smplx_body_pose"].append(torch.zeros(1, 63))
            all_params["smplx_lhand_pose"].append(torch.zeros(1, 45))
            all_params["smplx_rhand_pose"].append(torch.zeros(1, 45))
            all_params["smplx_jaw_pose"].append(torch.zeros(1, 3))
            all_params["smplx_shape"].append(torch.zeros(1, 10))
            all_params["smplx_expr"].append(torch.zeros(1, 10))
            all_params["cam_trans"].append(torch.zeros(1, 3))
            all_params["bbox"].append(torch.zeros(1, 4))
            if can_render:
                cv2.imwrite(str(render_dir / f"{frame_idx:06d}.jpg"), vis_img[:, :, ::-1])
            continue

        # Use largest bbox (single person)
        bbox_xyxy = yolo_bbox[0]
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
            cvimg=original_img,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=cfg.model.input_img_shape,
        )

        img_tensor = transform(img.astype(np.float32)) / 255
        img_tensor = img_tensor.cuda()[None, :, :, :]

        with torch.no_grad():
            out = demoer.model({"img": img_tensor}, {}, {}, "test")

        # Store parameters
        all_params["smplx_root_pose"].append(out["smplx_root_pose"].cpu())
        all_params["smplx_body_pose"].append(out["smplx_body_pose"].cpu())
        all_params["smplx_lhand_pose"].append(out["smplx_lhand_pose"].cpu())
        all_params["smplx_rhand_pose"].append(out["smplx_rhand_pose"].cpu())
        all_params["smplx_jaw_pose"].append(out["smplx_jaw_pose"].cpu())
        all_params["smplx_shape"].append(out["smplx_shape"].cpu())
        all_params["smplx_expr"].append(out["smplx_expr"].cpu())
        all_params["cam_trans"].append(out["cam_trans"].cpu())
        all_params["bbox"].append(torch.tensor(bbox_xyxy).unsqueeze(0))

        # Render overlay
        if can_render:
            mesh = out["smplx_mesh_cam"].detach().cpu().numpy()[0]
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
            cv2.imwrite(str(render_dir / f"{frame_idx:06d}.jpg"), vis_img[:, :, ::-1])

    # Stack all parameters into tensors
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
