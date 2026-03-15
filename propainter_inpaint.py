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
from PIL import Image
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


def _resize_video_for_inpainting(
    frames: np.ndarray,
    masks: np.ndarray,
    max_long_side: int | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int] | None]:
    """Downscale frames/masks for inpainting, preserving aspect ratio.

    Returns resized arrays plus the original `(height, width)` if resizing was
    applied. The resized dimensions are forced even for codec/model stability.
    """
    _, height, width, _ = frames.shape
    target_w, target_h = width, height

    if max_long_side is not None and max_long_side > 0:
        long_side = max(height, width)
        if long_side > max_long_side:
            scale = max_long_side / float(long_side)
            target_w = max(8, int(round(width * scale)))
            target_h = max(8, int(round(height * scale)))

    # ProPainter's RAFT path expects sizes aligned to /8 feature maps.
    process_w = max(8, target_w - (target_w % 8))
    process_h = max(8, target_h - (target_h % 8))
    if process_w <= 0 or process_h <= 0:
        return frames, masks, None
    if process_w == width and process_h == height:
        return frames, masks, None

    resized_frames = np.stack([
        cv2.resize(frame, (process_w, process_h), interpolation=cv2.INTER_AREA)
        for frame in frames
    ], axis=0)
    resized_masks = np.stack([
        cv2.resize(mask.astype(np.uint8), (process_w, process_h), interpolation=cv2.INTER_NEAREST) > 0
        for mask in masks
    ], axis=0)
    return resized_frames, resized_masks, (height, width)


def _compute_trusted_crop_bbox(
    target_track_bboxes: np.ndarray,
    video_w: int,
    video_h: int,
    padding: int = 40,
    detection_mask: np.ndarray | None = None,
) -> tuple[list[int], dict]:
    """Compute a stable crop ROI, preferring frames with real detections."""
    bboxes = np.asarray(target_track_bboxes, dtype=np.float32)
    if detection_mask is not None:
        detection_mask = np.asarray(detection_mask, dtype=bool)[:len(bboxes)]
    trusted = detection_mask if detection_mask is not None and detection_mask.any() else None
    crop_source = bboxes[trusted] if trusted is not None else bboxes

    x1_min = max(0, int(crop_source[:, 0].min()) - padding)
    y1_min = max(0, int(crop_source[:, 1].min()) - padding)
    x2_max = min(video_w, int(crop_source[:, 2].max()) + padding)
    y2_max = min(video_h, int(crop_source[:, 3].max()) + padding)

    crop_w = max(2, x2_max - x1_min)
    crop_h = max(2, y2_max - y1_min)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    crop_bbox = [x1_min, y1_min, x1_min + crop_w, y1_min + crop_h]
    debug = {
        "crop_area_ratio": float((crop_w * crop_h) / max(video_w * video_h, 1)),
        "trusted_crop_frames": int(trusted.sum()) if trusted is not None else int(len(bboxes)),
        "crop_from_trusted_frames": bool(trusted is not None),
    }
    return crop_bbox, debug


def _dilate_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    if mask.size == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations) > 0


def _bbox_mask_in_crop(
    bbox_xyxy: np.ndarray,
    crop_bbox: list[int],
    crop_shape: tuple[int, int],
) -> np.ndarray:
    """Build a bbox fallback mask in crop coordinates."""
    crop_h, crop_w = crop_shape
    x1, y1, _, _ = crop_bbox
    bx1, by1, bx2, by2 = bbox_xyxy.astype(np.int32)
    bx1 = max(0, bx1 - int(x1))
    by1 = max(0, by1 - int(y1))
    bx2 = min(crop_w, bx2 - int(x1))
    by2 = min(crop_h, by2 - int(y1))
    mask = np.zeros((crop_h, crop_w), dtype=bool)
    if bx2 > bx1 and by2 > by1:
        mask[by1:by2, bx1:bx2] = True
    return mask


class ProPainterInpainter:
    """ProPainter video inpainting wrapper using the full 4-stage pipeline:
    1. Optical flow estimation (RAFT)
    2. Flow completion (RecurrentFlowCompleteNet)
    3. Image propagation (fill from temporal neighbors)
    4. Transformer-based inpainting (InpaintGenerator)
    """

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

        self._raft = None
        self._flow_complete = None
        self._model = None

    def _ensure_loaded(self):
        """Lazy-load ProPainter models, reloading any that were freed between stages."""
        from model.modules.flow_comp_raft import RAFT_bi
        from model.recurrent_flow_completion import RecurrentFlowCompleteNet
        from model.propainter import InpaintGenerator
        from utils.download_util import load_file_from_url

        weights_dir = Path(self.model_path) / "weights"
        weights_dir.mkdir(exist_ok=True)
        pretrain_url = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"

        # RAFT optical flow
        if self._raft is None:
            raft_path = load_file_from_url(
                url=pretrain_url + "raft-things.pth",
                model_dir=str(weights_dir), progress=True,
            )
            self._raft = RAFT_bi(raft_path, self.device)

        # Flow completion
        if self._flow_complete is None:
            fc_path = load_file_from_url(
                url=pretrain_url + "recurrent_flow_completion.pth",
                model_dir=str(weights_dir), progress=True,
            )
            self._flow_complete = RecurrentFlowCompleteNet(fc_path)
            for p in self._flow_complete.parameters():
                p.requires_grad = False
            self._flow_complete.to(self.device)
            self._flow_complete.eval()

        # InpaintGenerator
        if self._model is None:
            pp_path = load_file_from_url(
                url=pretrain_url + "ProPainter.pth",
                model_dir=str(weights_dir), progress=True,
            )
            self._model = InpaintGenerator(model_path=pp_path).to(self.device)
            self._model.eval()

    def inpaint_video(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        subvideo_length: int = 50,
        neighbor_length: int = 10,
        ref_stride: int = 10,
        raft_iter: int = 20,
        mask_dilation: int = 4,
        max_long_side: int | None = 576,
    ) -> np.ndarray:
        """Inpaint masked regions in video frames using full ProPainter pipeline.

        Args:
            frames: (N, H, W, 3) uint8 RGB frames
            masks: (N, H, W) bool — True where inpainting is needed
            subvideo_length: chunk size for long video processing
            neighbor_length: local temporal window for transformer
            ref_stride: stride for reference frame selection
            raft_iter: RAFT optical flow iterations
            mask_dilation: dilation applied to masks for flow/inpainting
            max_long_side: optional long-side cap for inpainting resolution

        Returns:
            (N, H, W, 3) uint8 inpainted frames
        """
        self._ensure_loaded()

        from core.utils import to_tensors
        from inference_propainter import get_ref_index

        original_hw = frames.shape[1:3]
        frames, masks, restore_hw = _resize_video_for_inpainting(frames, masks, max_long_side)
        n_frames, h, w, _ = frames.shape
        if restore_hw is not None:
            print(
                f"[ProPainter] Resizing inpaint ROI from "
                f"{original_hw[1]}x{original_hw[0]} to {w}x{h} (max_long_side={max_long_side})"
            )
        output = frames.copy()

        # Convert frames to PIL Images for to_tensors()
        pil_frames = [Image.fromarray(f) for f in frames]

        # Dilate masks for flow and inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        masks_dilated_pil = []
        flow_masks_pil = []
        for i in range(n_frames):
            m = masks[i].astype(np.uint8) * 255
            # Flow mask: smaller dilation
            fm = cv2.dilate(m, kernel, iterations=mask_dilation // 2)
            flow_masks_pil.append(Image.fromarray(fm))
            # Inpainting mask: full dilation
            md = cv2.dilate(m, kernel, iterations=mask_dilation)
            masks_dilated_pil.append(Image.fromarray(md))

        # Convert to tensors: (1, T, C, H, W) in [-1, 1]
        frames_t = to_tensors()(pil_frames).unsqueeze(0) * 2 - 1
        flow_masks_t = to_tensors()(flow_masks_pil).unsqueeze(0)
        masks_dilated_t = to_tensors()(masks_dilated_pil).unsqueeze(0)

        frames_t = frames_t.to(self.device)
        flow_masks_t = flow_masks_t.to(self.device)
        masks_dilated_t = masks_dilated_t.to(self.device)

        video_length = frames_t.size(1)

        with torch.no_grad():
            # Stage 1: Compute optical flow with RAFT
            if w <= 640:
                short_clip_len = 12
            elif w <= 720:
                short_clip_len = 8
            elif w <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self._raft(frames_t[:, f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self._raft(frames_t[:, f - 1:end_f], iters=raft_iter)
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
            else:
                gt_flows_bi = self._raft(frames_t, iters=raft_iter)
                torch.cuda.empty_cache()

            # Stage 2: Complete flow in masked regions
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = self._flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        flow_masks_t[:, s_f:e_f + 1])
                    pred_flows_bi_sub = self._flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                        pred_flows_bi_sub,
                        flow_masks_t[:, s_f:e_f + 1])
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
            else:
                pred_flows_bi, _ = self._flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
                pred_flows_bi = self._flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
                torch.cuda.empty_cache()

            # Free flow memory and RAFT model (no longer needed)
            del gt_flows_bi
            if self._raft is not None:
                del self._raft
                self._raft = None
            torch.cuda.empty_cache()

            # Stage 3: Image propagation (fill from temporal neighbors)
            masked_frames = frames_t * (1 - masks_dilated_t)
            subvideo_length_prop = min(100, subvideo_length)
            if video_length > subvideo_length_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_prop)

                    b, t, _, _, _ = masks_dilated_t[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                    prop_imgs_sub, updated_local_masks_sub = self._model.img_propagation(
                        masked_frames[:, s_f:e_f], pred_flows_bi_sub,
                        masks_dilated_t[:, s_f:e_f], 'nearest')
                    updated_frames_sub = frames_t[:, s_f:e_f] * (1 - masks_dilated_t[:, s_f:e_f]) + \
                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated_t[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    torch.cuda.empty_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated_t.size()
                prop_imgs, updated_local_masks = self._model.img_propagation(
                    masked_frames, pred_flows_bi, masks_dilated_t, 'nearest')
                updated_frames = frames_t * (1 - masks_dilated_t) + prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()

            # Free tensors and models no longer needed before transformer stage
            del masked_frames
            del flow_masks_t
            del frames_t
            if self._flow_complete is not None:
                del self._flow_complete
                self._flow_complete = None
            torch.cuda.empty_cache()

        # Stage 4: Transformer-based inpainting
        ori_frames = [np.array(f) for f in pil_frames]
        comp_frames = [None] * video_length
        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1

        for f in tqdm(range(0, video_length, neighbor_stride), desc="ProPainter inpaint"):
            neighbor_ids = list(range(
                max(0, f - neighbor_stride),
                min(video_length, f + neighbor_stride + 1),
            ))
            ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated_t[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (
                pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
                pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
            )

            with torch.no_grad():
                l_t = len(neighbor_ids)
                pred_img = self._model(
                    selected_imgs, selected_pred_flows_bi,
                    selected_masks, selected_update_masks, l_t,
                )
                pred_img = pred_img.view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated_t[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5

        # Composite results back
        for i in range(n_frames):
            if comp_frames[i] is not None:
                output[i] = comp_frames[i].astype(np.uint8)

        torch.cuda.empty_cache()
        if restore_hw is not None:
            restore_h, restore_w = restore_hw
            output = np.stack([
                cv2.resize(frame, (restore_w, restore_h), interpolation=cv2.INTER_LINEAR)
                for frame in output
            ], axis=0)
        return output

    def cleanup(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._raft is not None:
            del self._raft
            self._raft = None
        if self._flow_complete is not None:
            del self._flow_complete
            self._flow_complete = None
        torch.cuda.empty_cache()


def isolate_person(
    video_path: str,
    target_person_id: int,
    target_track_bboxes: np.ndarray,
    target_detection_mask: np.ndarray | None,
    all_masks: dict[int, np.ndarray],
    output_path: str,
    inpainter: ProPainterInpainter | None = None,
    target_fps: float = 30.0,
    propainter_max_long_side: int | None = 576,
) -> dict:
    """Isolate a single person from a multi-person video.

    Smart isolation: crop-only for frames without overlap, inpaint+crop for
    frames where other people's masks intrude into the target's bbox region.

    Args:
        video_path: input multi-person video
        target_person_id: track_id of person to isolate
        target_track_bboxes: (N_frames, 4) xyxy bboxes
        target_detection_mask: (N_frames,) bool array for real detections
        all_masks: {track_id: (N_frames, H, W) bool} from SAM2
        output_path: where to save the isolated video
        inpainter: ProPainterInpainter instance (created if None and needed)
        target_fps: output video FPS
        propainter_max_long_side: optional long-side cap for cropped inpainting ROI

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

    crop_bbox, debug = _compute_trusted_crop_bbox(
        target_track_bboxes,
        video_w=video_w,
        video_h=video_h,
        padding=40,
        detection_mask=target_detection_mask,
    )
    x1_min, y1_min, x2_max, y2_max = crop_bbox
    crop_w = x2_max - x1_min
    crop_h = y2_max - y1_min

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
    frames = frames[:, y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]
    original_frames = frames.copy()

    # Build mask of all OTHER people's pixels in the crop ROI only.
    other_mask = np.zeros((num_frames, crop_h, crop_w), dtype=bool)
    for pid, pmasks in all_masks.items():
        if pid == target_person_id:
            continue
        n = min(num_frames, pmasks.shape[0])
        other_mask[:n] |= pmasks[:n, y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]

    target_mask = np.zeros((num_frames, crop_h, crop_w), dtype=bool)
    if target_person_id in all_masks:
        pmasks = all_masks[target_person_id]
        n = min(num_frames, pmasks.shape[0])
        target_mask[:n] = pmasks[:n, y1_min:y1_min + crop_h, x1_min:x1_min + crop_w]

    focus_mask = np.zeros_like(target_mask)
    for f in range(min(num_frames, len(target_track_bboxes))):
        if target_mask[f].any():
            focus_mask[f] = _dilate_mask(target_mask[f], iterations=4)
        else:
            focus_mask[f] = _bbox_mask_in_crop(target_track_bboxes[f], crop_bbox, (crop_h, crop_w))

    debug["target_mask_mean_coverage"] = float(target_mask.mean()) if target_mask.size else 0.0
    debug["focus_mask_mean_coverage"] = float(focus_mask.mean()) if focus_mask.size else 0.0
    debug["other_mask_mean_coverage"] = float(other_mask.mean()) if other_mask.size else 0.0
    warnings = []
    if debug["crop_area_ratio"] >= 0.85:
        warnings.append("crop_near_full_frame")
    if debug["focus_mask_mean_coverage"] <= 0.05:
        warnings.append("low_target_coverage")

    if needs_inpaint:
        print(f"[Isolate] Person {target_person_id}: {isolation_modes.count('inpaint')}/{len(isolation_modes)} frames need inpainting")

        # For inpainting, only mask overlap frames within the person's crop ROI.
        inpaint_mask = other_mask.copy()
        for f in range(len(isolation_modes)):
            if isolation_modes[f] == "crop":
                inpaint_mask[f] = False

        if inpaint_mask.any():
            try:
                if inpainter is None:
                    inpainter = ProPainterInpainter()
                # Convert BGR to RGB for ProPainter, then back. We only inpaint
                # inside the stable crop ROI, which is much cheaper than full-frame.
                frames_rgb = frames[:, :, :, ::-1].copy()
                inpainted_rgb = inpainter.inpaint_video(
                    frames_rgb,
                    inpaint_mask,
                    max_long_side=propainter_max_long_side,
                )
                frames = inpainted_rgb[:, :, :, ::-1].copy()
                for f in range(min(num_frames, len(frames))):
                    if focus_mask[f].any():
                        frames[f][focus_mask[f]] = original_frames[f][focus_mask[f]]
            except (ImportError, FileNotFoundError) as e:
                print(f"[Isolate] ProPainter not available ({e}). Using mask blackout fallback.")
                isolation_modes = ["crop"] * len(isolation_modes)

    # Make the target explicit: preserve only the target focus region on crop-only
    # frames, and always keep the original target pixels when a focus mask exists.
    for f in range(min(num_frames, len(isolation_modes))):
        if focus_mask[f].any():
            frames[f][focus_mask[f]] = original_frames[f][focus_mask[f]]
        if isolation_modes[f] == "crop":
            keep = focus_mask[f]
            if keep.any():
                frame = frames[f]
                frame[~keep] = 0
                frames[f] = frame
            elif f < other_mask.shape[0] and other_mask[f].any():
                frames[f][other_mask[f]] = 0

    # Crop and write output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (crop_w, crop_h))

    for f in range(min(num_frames, len(frames))):
        writer.write(frames[f])

    writer.release()

    return {
        "video_path": output_path,
        "isolation_modes": isolation_modes,
        "crop_bbox": [x1_min, y1_min, x1_min + crop_w, y1_min + crop_h],
        "num_inpainted": isolation_modes.count("inpaint"),
        "num_crop_only": isolation_modes.count("crop"),
        "debug": debug,
        "warnings": warnings,
    }
