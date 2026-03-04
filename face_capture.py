"""Face capture pipeline — extract face crops + MediaPipe ARKit blendshapes."""

import csv
from pathlib import Path

import cv2
import numpy as np

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ARKit 52 blendshape names in canonical order
ARKIT_BLENDSHAPES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


def _find_mediapipe_model() -> str:
    """Find the MediaPipe face landmarker model file.

    Downloads it if not present.
    """
    model_dir = Path(__file__).parent / "models"
    model_path = model_dir / "face_landmarker.task"

    if model_path.exists():
        return str(model_path)

    # Download the model
    model_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    print(f"Downloading MediaPipe face landmarker model to {model_path}...")
    urllib.request.urlretrieve(url, str(model_path))
    print("Download complete.")
    return str(model_path)


def extract_face_crops(
    video_path: str,
    person_bboxes: np.ndarray,
    head_fraction: float = 0.30,
    margin: float = 0.20,
    target_size: int = 256,
) -> list[np.ndarray]:
    """Extract face crops from video frames using person bounding boxes.

    Args:
        video_path: Path to the video file.
        person_bboxes: Array of shape (N, 4) with [x1, y1, x2, y2] person bounding boxes.
            Can be in pixel coordinates or normalized (0-1). If max value <= 1.0,
            treated as normalized and scaled to frame dimensions.
        head_fraction: Fraction of person bbox height to use for head region (from top).
        margin: Margin to add around the head crop as fraction of crop size.
        target_size: Output face crop size (square).

    Returns:
        List of face crop images (numpy arrays, BGR, target_size x target_size).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Detect if bboxes are normalized
    is_normalized = person_bboxes.max() <= 1.0 if len(person_bboxes) > 0 else False

    crops = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= len(person_bboxes):
            # Pad with empty crop if we run out of bboxes
            crops.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
            frame_idx += 1
            continue

        bbox = person_bboxes[frame_idx].copy()

        # Scale normalized coords to pixels
        if is_normalized:
            bbox[0] *= frame_w
            bbox[2] *= frame_w
            bbox[1] *= frame_h
            bbox[3] *= frame_h

        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        # Head region: top portion of person bbox
        head_h = bbox_h * head_fraction
        head_cx = (x1 + x2) / 2
        head_cy = y1 + head_h / 2

        # Make square crop centered on head
        crop_size = max(head_h, bbox_w * 0.5)
        crop_size_with_margin = crop_size * (1 + margin)
        half = crop_size_with_margin / 2

        cx1 = int(max(0, head_cx - half))
        cy1 = int(max(0, head_cy - half))
        cx2 = int(min(frame_w, head_cx + half))
        cy2 = int(min(frame_h, head_cy + half))

        face_crop = frame[cy1:cy2, cx1:cx2]

        if face_crop.size == 0:
            crops.append(np.zeros((target_size, target_size, 3), dtype=np.uint8))
        else:
            # Resize preserving aspect ratio with padding
            h, w = face_crop.shape[:2]
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Center on black canvas
            canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_off = (target_size - new_h) // 2
            x_off = (target_size - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            crops.append(canvas)

        frame_idx += 1

    cap.release()
    return crops


def run_face_blendshapes(face_crops: list[np.ndarray]) -> list[dict[str, float]]:
    """Run MediaPipe Face Landmarker on face crops to get ARKit blendshapes.

    Args:
        face_crops: List of BGR face crop images.

    Returns:
        List of dicts mapping blendshape name -> weight (0.0-1.0) per frame.
        Frames where no face is detected get all-zero weights.
    """
    model_path = _find_mediapipe_model()

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        num_faces=1,
    )

    landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    results = []

    for crop in face_crops:
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        detection = landmarker.detect(mp_image)

        if detection.face_blendshapes and len(detection.face_blendshapes) > 0:
            bs = detection.face_blendshapes[0]
            frame_bs = {}
            for category in bs:
                name = category.category_name
                if name in ARKIT_BLENDSHAPES and name != "_neutral":
                    frame_bs[name] = round(category.score, 6)
            results.append(frame_bs)
        else:
            # No face detected — all zeros
            results.append({name: 0.0 for name in ARKIT_BLENDSHAPES if name != "_neutral"})

    landmarker.close()
    return results


def export_blendshapes_csv(
    blendshape_data: list[dict[str, float]],
    output_path: str,
    fps: float = 30.0,
) -> str:
    """Export blendshape data to CSV for Unreal Engine import.

    Args:
        blendshape_data: Per-frame blendshape weights from run_face_blendshapes().
        output_path: Path to write the CSV file.
        fps: Video framerate for timecode column.

    Returns:
        Path to the written CSV file.
    """
    # Column names (skip _neutral)
    bs_names = [name for name in ARKIT_BLENDSHAPES if name != "_neutral"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "timecode"] + bs_names)

        for i, frame_bs in enumerate(blendshape_data):
            timecode = round(i / fps, 6)
            row = [i, timecode]
            for name in bs_names:
                row.append(frame_bs.get(name, 0.0))
            writer.writerow(row)

    return output_path


def run_face_pipeline(
    video_path: str,
    person_bboxes: np.ndarray,
    output_csv: str,
    fps: float = 30.0,
    progress_callback=None,
) -> str:
    """Full face capture pipeline: crop → MediaPipe → CSV.

    Args:
        video_path: Path to input video.
        person_bboxes: Person bounding boxes array (N, 4).
        output_csv: Output CSV path.
        fps: Video framerate.
        progress_callback: Optional callable(fraction, message) for progress updates.

    Returns:
        Path to the output CSV file.
    """
    if progress_callback:
        progress_callback(0.0, "Extracting face crops from video...")

    crops = extract_face_crops(video_path, person_bboxes)

    if progress_callback:
        progress_callback(0.3, f"Running MediaPipe on {len(crops)} face crops...")

    blendshapes = run_face_blendshapes(crops)

    if progress_callback:
        progress_callback(0.9, "Exporting ARKit blendshapes to CSV...")

    export_blendshapes_csv(blendshapes, output_csv, fps=fps)

    if progress_callback:
        progress_callback(1.0, f"Face capture complete — {len(blendshapes)} frames exported.")

    return output_csv


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python face_capture.py <video_path> <smplestx_output.pt>")
        print("  Reads person bboxes from SMPLest-X output and runs face capture.")
        sys.exit(1)

    import torch

    video = sys.argv[1]
    pt_path = sys.argv[2]
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    # SMPLest-X stores bboxes in different possible keys
    bboxes = None
    for key in ["person_bbox", "bboxes", "bbox", "bb_xyxy"]:
        if key in data:
            bboxes = np.array(data[key])
            break

    if bboxes is None:
        print("ERROR: Could not find person bounding boxes in .pt file.")
        print(f"Available keys: {list(data.keys())}")
        sys.exit(1)

    output = str(Path(video).with_suffix("")) + "_arkit_blendshapes.csv"
    run_face_pipeline(video, bboxes, output, fps=30.0)
    print(f"Output: {output}")
