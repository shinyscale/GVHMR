"""SAM2 wrapper for bbox-prompted video segmentation.

Phase 2 of multi-person capture: segment each tracked person with precise masks
using SAM2's video predictor with automatic re-prompting from tracking bboxes.
"""

from __future__ import annotations

import tempfile
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2_video_predictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

from hmr4d.utils.video_io_utils import get_video_lwh


# SAM2 model configs — checkpoint name → config file
_MODEL_CONFIGS = {
    "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "base_plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}

# Default checkpoint search paths
_CHECKPOINT_DIRS = [
    Path("inputs/checkpoints/sam2"),
    Path.home() / ".cache" / "sam2",
    Path("/mnt/f/models/sam2"),
]


def _find_checkpoint(model_size: str) -> tuple[str, str]:
    """Find SAM2 checkpoint and config for a given model size."""
    if model_size not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: {list(_MODEL_CONFIGS.keys())}")

    config_name, ckpt_name = _MODEL_CONFIGS[model_size]

    for d in _CHECKPOINT_DIRS:
        ckpt_path = d / ckpt_name
        if ckpt_path.exists():
            return config_name, str(ckpt_path)

    raise FileNotFoundError(
        f"SAM2 checkpoint '{ckpt_name}' not found. Searched: {[str(d) for d in _CHECKPOINT_DIRS]}. "
        f"Download from https://github.com/facebookresearch/sam2"
    )


def _extract_frames_to_dir(video_path: str, output_dir: str) -> int:
    """Extract video frames as JPEG files to a directory (SAM2 video predictor requirement).

    Returns the number of frames extracted.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}.jpg"), frame)
        frame_idx += 1
    cap.release()
    return frame_idx


class SAM2Segmenter:
    """SAM2 video segmenter with bbox prompting from tracking results."""

    def __init__(self, model_size: str = "base_plus", device: str = "cuda"):
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Install with: pip install segment-anything-2\n"
                "Or clone: https://github.com/facebookresearch/sam2"
            )

        self.device = device
        config_name, ckpt_path = _find_checkpoint(model_size)
        self.predictor = build_sam2_video_predictor(config_name, ckpt_path, device=device)

    def segment_all_persons(
        self,
        video_path: str,
        tracks: list[dict],
        reprompt_interval: int = 30,
    ) -> dict[int, np.ndarray]:
        """Segment all tracked persons in a video.

        Args:
            video_path: path to video file
            tracks: list of track dicts with 'track_id', 'bbx_xyxy' (N,4), 'detection_mask' (N,)
            reprompt_interval: re-prompt SAM2 from tracking bboxes every N frames

        Returns:
            dict mapping track_id → masks array (N_frames, H, W) bool
        """
        num_frames, video_w, video_h = get_video_lwh(video_path)

        # Extract frames to temp directory (SAM2 video predictor expects JPEG dir)
        tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
        try:
            print(f"[SAM2] Extracting {num_frames} frames...")
            _extract_frames_to_dir(video_path, tmp_dir)

            return self._segment_with_reprompt(
                tmp_dir, tracks, num_frames, video_w, video_h, reprompt_interval
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _segment_with_reprompt(
        self,
        frames_dir: str,
        tracks: list[dict],
        num_frames: int,
        video_w: int,
        video_h: int,
        reprompt_interval: int,
    ) -> dict[int, np.ndarray]:
        """Run SAM2 video segmentation with periodic re-prompting."""

        all_masks = {t["track_id"]: np.zeros((num_frames, video_h, video_w), dtype=bool) for t in tracks}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.predictor.init_state(video_path=frames_dir)

            # Determine prompt frames
            prompt_frames = list(range(0, num_frames, reprompt_interval))
            if 0 not in prompt_frames:
                prompt_frames.insert(0, 0)

            # Add bbox prompts for all persons at prompt frames
            for track in tracks:
                tid = track["track_id"]
                boxes = track["bbx_xyxy"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                det_mask = track.get("detection_mask")
                if isinstance(det_mask, torch.Tensor):
                    det_mask = det_mask.cpu().numpy().astype(bool)
                elif det_mask is None:
                    det_mask = np.ones(min(num_frames, len(boxes)), dtype=bool)
                else:
                    det_mask = np.asarray(det_mask, dtype=bool)
                det_conf = track.get("detection_conf")
                if isinstance(det_conf, torch.Tensor):
                    det_conf = det_conf.cpu().numpy()
                elif det_conf is not None:
                    det_conf = np.asarray(det_conf, dtype=np.float32)

                trusted_frames = []
                for prompt_frame in prompt_frames:
                    actual_frame = self._nearest_trusted_prompt_frame(
                        prompt_frame=prompt_frame,
                        detection_mask=det_mask,
                        detection_conf=det_conf,
                        max_frame=num_frames,
                        search_radius=reprompt_interval,
                    )
                    if actual_frame is None:
                        continue
                    if actual_frame in trusted_frames:
                        continue
                    trusted_frames.append(actual_frame)

                for actual_frame in trusted_frames:
                    bbox = boxes[actual_frame]
                    # Skip if bbox is degenerate
                    if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
                        continue
                    _, _, _ = self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=actual_frame,
                        obj_id=tid,
                        box=bbox,
                    )

            # Propagate through video
            print("[SAM2] Propagating masks through video...")
            for frame_idx, obj_ids, masks in tqdm(
                self.predictor.propagate_in_video(state),
                total=num_frames,
                desc="SAM2 Propagation",
            ):
                for i, obj_id in enumerate(obj_ids):
                    mask = masks[i].squeeze().cpu().numpy() > 0.0
                    if obj_id in all_masks:
                        all_masks[obj_id][frame_idx] = mask

            self.predictor.reset_state(state)

        return all_masks

    @staticmethod
    def _nearest_trusted_prompt_frame(
        prompt_frame: int,
        detection_mask: np.ndarray,
        detection_conf: np.ndarray | None,
        max_frame: int,
        search_radius: int,
        min_conf: float = 0.35,
    ) -> int | None:
        """Pick the nearest real/trusted detection frame for a SAM2 prompt."""
        limit = min(max_frame, len(detection_mask))
        if prompt_frame < limit and detection_mask[prompt_frame]:
            if detection_conf is None or prompt_frame >= len(detection_conf) or detection_conf[prompt_frame] >= min_conf:
                return int(prompt_frame)

        best_frame = None
        best_distance = None
        start = max(0, prompt_frame - search_radius)
        end = min(limit, prompt_frame + search_radius + 1)
        for frame_idx in range(start, end):
            if not detection_mask[frame_idx]:
                continue
            if detection_conf is not None and frame_idx < len(detection_conf) and detection_conf[frame_idx] < min_conf:
                continue
            distance = abs(frame_idx - prompt_frame)
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_frame = int(frame_idx)
        return best_frame

    def compute_overlap_map(self, masks: dict[int, np.ndarray]) -> np.ndarray:
        """Compute per-frame overlap flags between all person pairs.

        Returns:
            (N_frames,) bool array — True where any two masks overlap.
        """
        track_ids = list(masks.keys())
        if len(track_ids) < 2:
            num_frames = masks[track_ids[0]].shape[0] if track_ids else 0
            return np.zeros(num_frames, dtype=bool)

        num_frames = masks[track_ids[0]].shape[0]
        overlap = np.zeros(num_frames, dtype=bool)

        for f in range(num_frames):
            for i, tid_a in enumerate(track_ids):
                for tid_b in track_ids[i + 1:]:
                    if np.logical_and(masks[tid_a][f], masks[tid_b][f]).any():
                        overlap[f] = True
                        break
                if overlap[f]:
                    break

        return overlap


def render_mask_overlay(
    video_path: str,
    masks: dict[int, np.ndarray],
    output_path: str,
    alpha: float = 0.4,
):
    """Render colored mask overlay video for verification.

    Each person gets a distinct color overlay on the original video.
    """
    from person_tracker import _color_for_id

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    track_ids = sorted(masks.keys())
    frame_idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total), desc="Rendering mask overlay"):
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        for tid in track_ids:
            if frame_idx >= masks[tid].shape[0]:
                continue
            mask = masks[tid][frame_idx]
            if mask.any():
                color = _color_for_id(tid)
                overlay[mask] = (
                    np.array(color, dtype=np.uint8) * alpha
                    + overlay[mask] * (1 - alpha)
                ).astype(np.uint8)

        writer.write(overlay)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[SAM2] Mask overlay saved to {output_path}")


def save_masks(masks: dict[int, np.ndarray], output_dir: str | Path):
    """Save per-person masks as compressed .npz files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tid, mask_array in masks.items():
        np.savez_compressed(
            str(output_dir / f"person_{tid}_masks.npz"),
            masks=mask_array,
        )


def load_masks(masks_dir: str | Path) -> dict[int, np.ndarray]:
    """Load masks from a directory of .npz files."""
    masks_dir = Path(masks_dir)
    masks = {}
    for npz_path in sorted(masks_dir.glob("person_*_masks.npz")):
        # Extract track ID from filename
        stem = npz_path.stem  # "person_3_masks"
        parts = stem.split("_")
        tid = int(parts[1])
        masks[tid] = np.load(str(npz_path))["masks"]
    return masks
