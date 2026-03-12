"""ProPainter wrapper for mask-based video inpainting.

Phase 3 of multi-person capture: isolate each person into a clean single-person
video by inpainting away other people where they overlap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh


# ProPainter search paths
_PROPAINTER_PATHS = [
    Path("/mnt/f/ProPainter"),
    Path(__file__).resolve().parent / "third-party" / "ProPainter",
    Path.home() / "ProPainter",
]

PROPAINTER_AVAILABLE = False
_propainter_path = None

for _p in _PROPAINTER_PATHS:
    if (_p / "model").is_dir() or (_p / "inference_propainter.py").exists():
        _propainter_path = _p
        PROPAINTER_AVAILABLE = True
        break


def should_inpaint_frame(
    all_masks: dict[int, np.ndarray],
    target_person: int,
    target_bbox: np.ndarray,
    frame_idx: int,
    padding: int = 20,
) -> bool:
    """Check if any other person's mask overlaps target person's bbox region.

    Args:
        all_masks: {track_id: (N_frames, H, W) bool array}
        target_person: track_id of the person we're isolating
        target_bbox: (4,) xyxy bbox of target person at this frame
        frame_idx: current frame index
        padding: extra pixels around bbox to check

    Returns:
        True if inpainting is needed (overlap detected)
    """
    x1 = max(0, int(target_bbox[0]) - padding)
    y1 = max(0, int(target_bbox[1]) - padding)
    x2 = int(target_bbox[2]) + padding
    y2 = int(target_bbox[3]) + padding

    for person_id, person_masks in all_masks.items():
        if person_id == target_person:
            continue
        if frame_idx >= person_masks.shape[0]:
            continue

        h, w = person_masks.shape[1], person_masks.shape[2]
        x2_c = min(x2, w)
        y2_c = min(y2, h)

        other_in_bbox = person_masks[frame_idx, y1:y2_c, x1:x2_c]
        if other_in_bbox.any():
            return True

    return False


class ProPainterInpainter:
    """ProPainter video inpainting wrapper."""

    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        if not PROPAINTER_AVAILABLE and model_path is None:
            raise ImportError(
                "ProPainter not found. Clone it to one of: "
                f"{[str(p) for p in _PROPAINTER_PATHS]}\n"
                "git clone https://github.com/sczhou/ProPainter"
            )

        self.device = device
        self.model_path = model_path or str(_propainter_path)

        # Add ProPainter to path for imports
        pp_path = str(Path(self.model_path).resolve())
        if pp_path not in sys.path:
            sys.path.insert(0, pp_path)

        self._model = None

    def _ensure_loaded(self):
        """Lazy-load ProPainter model."""
        if self._model is not None:
            return

        from inference_propainter import InpaintGenerator
        from core.utils import to_tensors

        # Load model weights
        ckpt_path = Path(self.model_path) / "weights" / "ProPainter.pth"
        if not ckpt_path.exists():
            ckpt_path = Path(self.model_path) / "pretrained_models" / "ProPainter.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"ProPainter weights not found at {ckpt_path}. "
                "Download from https://github.com/sczhou/ProPainter"
            )

        model = InpaintGenerator().to(self.device)
        data = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        model.load_state_dict(data, strict=False)
        model.eval()
        self._model = model

    def inpaint_video(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        chunk_size: int = 80,
        chunk_overlap: int = 10,
    ) -> np.ndarray:
        """Inpaint masked regions in video frames.

        Args:
            frames: (N, H, W, 3) uint8 RGB frames
            masks: (N, H, W) bool — True where inpainting is needed
            chunk_size: process video in chunks of this many frames
            chunk_overlap: overlap between chunks for temporal consistency

        Returns:
            (N, H, W, 3) uint8 inpainted frames
        """
        self._ensure_loaded()

        n_frames = len(frames)
        output = frames.copy()

        # Process in chunks
        start = 0
        while start < n_frames:
            end = min(start + chunk_size, n_frames)
            chunk_frames = frames[start:end]
            chunk_masks = masks[start:end]

            # Skip chunk if no masks
            if not chunk_masks.any():
                start = end - chunk_overlap
                if start >= end:
                    break
                continue

            inpainted = self._inpaint_chunk(chunk_frames, chunk_masks)

            # Blend overlap region
            if start > 0 and chunk_overlap > 0:
                overlap_start = 0
                overlap_end = min(chunk_overlap, len(inpainted))
                for i in range(overlap_start, overlap_end):
                    alpha = i / chunk_overlap
                    frame_idx = start + i
                    output[frame_idx] = (
                        output[frame_idx] * (1 - alpha) + inpainted[i] * alpha
                    ).astype(np.uint8)
                output[start + overlap_end:end] = inpainted[overlap_end:]
            else:
                output[start:end] = inpainted

            start = end - chunk_overlap
            if start >= end:
                break

        return output

    def _inpaint_chunk(self, frames: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Inpaint a single chunk using ProPainter."""
        # Convert to tensors
        frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (N,3,H,W)
        masks_t = torch.from_numpy(masks.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)

        frames_t = frames_t.to(self.device)
        masks_t = masks_t.to(self.device)

        with torch.no_grad():
            result = self._model(frames_t, masks_t)

        # Composite: keep original where no mask, use inpainted where masked
        result_np = (result.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        masks_3ch = np.repeat(masks[:, :, :, np.newaxis], 3, axis=3)
        output = np.where(masks_3ch, result_np, frames)

        return output

    def cleanup(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()


def isolate_person(
    video_path: str,
    target_person_id: int,
    target_track_bboxes: np.ndarray,
    all_masks: dict[int, np.ndarray],
    output_path: str,
    inpainter: ProPainterInpainter | None = None,
    target_fps: float = 30.0,
) -> dict:
    """Isolate a single person from a multi-person video.

    Smart isolation: crop-only for frames without overlap, inpaint+crop for
    frames where other people's masks intrude into the target's bbox region.

    Args:
        video_path: input multi-person video
        target_person_id: track_id of person to isolate
        target_track_bboxes: (N_frames, 4) xyxy bboxes
        all_masks: {track_id: (N_frames, H, W) bool} from SAM2
        output_path: where to save the isolated video
        inpainter: ProPainterInpainter instance (created if None and needed)
        target_fps: output video FPS

    Returns:
        dict with 'video_path', 'isolation_modes', 'crop_bbox'
    """
    num_frames, video_w, video_h = get_video_lwh(video_path)

    # Determine per-frame isolation mode
    isolation_modes = []
    for f in range(min(num_frames, len(target_track_bboxes))):
        if should_inpaint_frame(all_masks, target_person_id, target_track_bboxes[f], f):
            isolation_modes.append("inpaint")
        else:
            isolation_modes.append("crop")

    needs_inpaint = "inpaint" in isolation_modes

    # Compute stable crop region (max bbox + padding across all frames)
    padding = 40
    x1_min = max(0, int(target_track_bboxes[:, 0].min()) - padding)
    y1_min = max(0, int(target_track_bboxes[:, 1].min()) - padding)
    x2_max = min(video_w, int(target_track_bboxes[:, 2].max()) + padding)
    y2_max = min(video_h, int(target_track_bboxes[:, 3].max()) + padding)

    crop_w = (x2_max - x1_min) - ((x2_max - x1_min) % 2)  # even dims
    crop_h = (y2_max - y1_min) - ((y2_max - y1_min) % 2)

    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames[:num_frames])

    if needs_inpaint:
        # Build inpaint mask: combine all OTHER people's masks
        print(f"[Isolate] Person {target_person_id}: {isolation_modes.count('inpaint')}/{len(isolation_modes)} frames need inpainting")

        inpaint_mask = np.zeros((num_frames, video_h, video_w), dtype=bool)
        for pid, pmasks in all_masks.items():
            if pid == target_person_id:
                continue
            n = min(num_frames, pmasks.shape[0])
            inpaint_mask[:n] |= pmasks[:n]

        # Only inpaint frames that actually need it
        # For crop-only frames, zero out the mask
        for f in range(len(isolation_modes)):
            if isolation_modes[f] == "crop":
                inpaint_mask[f] = False

        if inpaint_mask.any():
            try:
                if inpainter is None:
                    inpainter = ProPainterInpainter()
                # Convert BGR to RGB for ProPainter, then back
                frames_rgb = frames[:, :, :, ::-1].copy()
                inpainted_rgb = inpainter.inpaint_video(frames_rgb, inpaint_mask)
                frames = inpainted_rgb[:, :, :, ::-1].copy()
            except (ImportError, FileNotFoundError) as e:
                print(f"[Isolate] ProPainter not available ({e}). Using crop-only fallback.")
                isolation_modes = ["crop"] * len(isolation_modes)

    # Crop and write output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_w, crop_h))

    for f in range(min(num_frames, len(frames))):
        crop = frames[f][y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]
        writer.write(crop)

    writer.release()

    return {
        "video_path": output_path,
        "isolation_modes": isolation_modes,
        "crop_bbox": [x1_min, y1_min, x1_min + crop_w, y1_min + crop_h],
        "num_inpainted": isolation_modes.count("inpaint"),
        "num_crop_only": isolation_modes.count("crop"),
    }
