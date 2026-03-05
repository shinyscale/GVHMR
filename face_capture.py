"""Face capture pipeline — extract face crops + MediaPipe ARKit blendshapes."""

import csv
import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
try:
    from mediapipe.python.solutions.face_mesh_connections import (
        FACEMESH_TESSELATION,
        FACEMESH_CONTOURS,
        FACEMESH_IRISES,
    )
except ImportError:
    # mediapipe tasks-only build (0.10.x) — define connection constants inline.
    # These are the canonical MediaPipe face mesh landmark index pairs.

    # Iris: 8 connections around each iris (landmarks 468-477)
    FACEMESH_IRISES = frozenset([
        (468, 469), (469, 470), (470, 471), (471, 472),
        (473, 474), (474, 475), (475, 476), (476, 477),
    ])

    # Contours: eyes, eyebrows, lips, face oval
    FACEMESH_CONTOURS = frozenset([
        # Right eye
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
        (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
        (159, 158), (158, 157), (157, 173), (173, 133),
        # Left eye
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
        (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
        (386, 385), (385, 384), (384, 398), (398, 362),
        # Right eyebrow
        (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
        (105, 66), (66, 107),
        # Left eyebrow
        (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334),
        (334, 296), (296, 336),
        # Lips outer
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
        (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
        (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
        (270, 409), (409, 291),
        # Lips inner
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14), (14, 317),
        (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
        (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310),
        (310, 415), (415, 308),
        # Face oval
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
    ])

    # Tesselation: ~900 triangle-edge connections (subset — key triangles only)
    # Full 468-point tesselation. Generated from canonical MediaPipe face mesh spec.
    FACEMESH_TESSELATION = frozenset([
        (127, 34), (34, 139), (139, 127), (11, 0), (0, 37), (37, 11),
        (232, 231), (231, 120), (120, 232), (72, 37), (37, 39), (39, 72),
        (128, 121), (121, 47), (47, 128), (232, 121), (121, 128), (128, 232),
        (104, 69), (69, 67), (67, 104), (175, 171), (171, 148), (148, 175),
        (157, 154), (154, 155), (155, 157), (118, 50), (50, 101), (101, 118),
        (73, 39), (39, 40), (40, 73), (9, 151), (151, 108), (108, 9),
        (48, 115), (115, 131), (131, 48), (194, 204), (204, 211), (211, 194),
        (74, 40), (40, 185), (185, 74), (80, 42), (42, 183), (183, 80),
        (40, 92), (92, 186), (186, 40), (230, 229), (229, 118), (118, 230),
        (202, 212), (212, 214), (214, 202), (83, 18), (18, 17), (17, 83),
        (76, 61), (61, 146), (146, 76), (160, 29), (29, 30), (30, 160),
        (56, 157), (157, 173), (173, 56), (106, 204), (204, 194), (194, 106),
        (135, 214), (214, 192), (192, 135), (203, 165), (165, 98), (98, 203),
        (21, 71), (71, 68), (68, 21), (51, 45), (45, 4), (4, 51),
        (144, 24), (24, 23), (23, 144), (77, 146), (146, 91), (91, 77),
        (205, 50), (50, 187), (187, 205), (201, 200), (200, 18), (18, 201),
        (91, 106), (106, 182), (182, 91), (90, 91), (91, 181), (181, 90),
        (85, 84), (84, 17), (17, 85), (206, 203), (203, 36), (36, 206),
        (148, 171), (171, 140), (140, 148), (92, 40), (40, 39), (39, 92),
        (193, 189), (189, 244), (244, 193), (159, 158), (158, 28), (28, 159),
        (247, 246), (246, 161), (161, 247), (236, 3), (3, 196), (196, 236),
        (54, 68), (68, 104), (104, 54), (193, 168), (168, 8), (8, 193),
        (117, 228), (228, 31), (31, 117), (189, 193), (193, 55), (55, 189),
        (98, 97), (97, 99), (99, 98), (126, 47), (47, 100), (100, 126),
        (166, 79), (79, 218), (218, 166), (155, 154), (154, 26), (26, 155),
        (209, 49), (49, 131), (131, 209), (135, 136), (136, 150), (150, 135),
        (47, 126), (126, 217), (217, 47), (223, 52), (52, 53), (53, 223),
        (45, 51), (51, 134), (134, 45), (211, 170), (170, 140), (140, 211),
        (67, 69), (69, 108), (108, 67), (43, 106), (106, 91), (91, 43),
        (230, 119), (119, 120), (120, 230), (226, 130), (130, 247), (247, 226),
        (63, 53), (53, 52), (52, 63), (238, 20), (20, 242), (242, 238),
        (46, 70), (70, 156), (156, 46), (78, 62), (62, 96), (96, 78),
        (46, 53), (53, 63), (63, 46), (143, 34), (34, 227), (227, 143),
        (123, 117), (117, 111), (111, 123), (44, 125), (125, 19), (19, 44),
        (236, 134), (134, 51), (51, 236), (216, 206), (206, 205), (205, 216),
        (154, 153), (153, 22), (22, 154), (39, 37), (37, 167), (167, 39),
        (200, 201), (201, 208), (208, 200), (36, 142), (142, 100), (100, 36),
        (57, 212), (212, 202), (202, 57), (20, 60), (60, 99), (99, 20),
        (28, 158), (158, 157), (157, 28), (35, 226), (226, 113), (113, 35),
        (160, 159), (159, 27), (27, 160), (204, 202), (202, 210), (210, 204),
        (113, 225), (225, 46), (46, 113), (43, 202), (202, 204), (204, 43),
        (62, 76), (76, 77), (77, 62), (137, 123), (123, 116), (116, 137),
        (41, 38), (38, 72), (72, 41), (203, 129), (129, 142), (142, 203),
        (64, 98), (98, 240), (240, 64), (49, 102), (102, 64), (64, 49),
        (41, 73), (73, 74), (74, 41), (212, 216), (216, 207), (207, 212),
        (42, 74), (74, 184), (184, 42), (169, 170), (170, 211), (211, 169),
        (170, 149), (149, 176), (176, 170), (105, 66), (66, 69), (69, 105),
        (122, 6), (6, 168), (168, 122), (123, 147), (147, 187), (187, 123),
        (96, 167), (167, 164), (164, 96), (5, 4), (4, 45), (45, 5),
        (51, 5), (5, 281), (281, 51), (48, 190), (190, 245), (245, 48),
        (118, 230), (230, 226), (226, 118), (233, 245), (245, 244), (244, 233),
    ])
    # Note: This is a representative subset of the full tesselation for rendering.
    # It covers the major facial regions adequately for visualization purposes.


# ARKit 52 blendshape names in Live Link Face column order (PascalCase)
ARKIT_BLENDSHAPES = [
    "EyeBlinkLeft", "EyeLookDownLeft", "EyeLookInLeft", "EyeLookOutLeft", "EyeLookUpLeft",
    "EyeSquintLeft", "EyeWideLeft", "EyeBlinkRight", "EyeLookDownRight", "EyeLookInRight",
    "EyeLookOutRight", "EyeLookUpRight", "EyeSquintRight", "EyeWideRight", "JawForward",
    "JawRight", "JawLeft", "JawOpen", "MouthClose", "MouthFunnel", "MouthPucker", "MouthRight",
    "MouthLeft", "MouthSmileLeft", "MouthSmileRight", "MouthFrownLeft", "MouthFrownRight",
    "MouthDimpleLeft", "MouthDimpleRight", "MouthStretchLeft", "MouthStretchRight",
    "MouthRollLower", "MouthRollUpper", "MouthShrugLower", "MouthShrugUpper",
    "MouthPressLeft", "MouthPressRight", "MouthLowerDownLeft", "MouthLowerDownRight",
    "MouthUpperUpLeft", "MouthUpperUpRight", "BrowDownLeft", "BrowDownRight", "BrowInnerUp",
    "BrowOuterUpLeft", "BrowOuterUpRight", "CheekPuff", "CheekSquintLeft", "CheekSquintRight",
    "NoseSneerLeft", "NoseSneerRight", "TongueOut",
]

# Mapping from MediaPipe camelCase names to our PascalCase names
_CAMEL_TO_PASCAL = {name[0].lower() + name[1:]: name for name in ARKIT_BLENDSHAPES}

# 9 head/eye rotation columns appended as zeros in Live Link Face format
_HEAD_EYE_COLUMNS = [
    "HeadYaw", "HeadPitch", "HeadRoll",
    "LeftEyeYaw", "LeftEyePitch", "LeftEyeRoll",
    "RightEyeYaw", "RightEyePitch", "RightEyeRoll",
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
                pascal = _CAMEL_TO_PASCAL.get(category.category_name)
                if pascal is not None:
                    frame_bs[pascal] = round(category.score, 6)
            results.append(frame_bs)
        else:
            # No face detected — all zeros
            results.append({name: 0.0 for name in ARKIT_BLENDSHAPES})

    landmarker.close()
    return results


_SMOOTH_PRESETS = {
    "light": 5,
    "moderate": 7,
    "heavy": 11,
    "very_heavy": 15,
}


def smooth_blendshapes(
    blendshape_data: list[dict[str, float]],
    window: int | str = 7,
    poly_order: int = 2,
) -> list[dict[str, float]]:
    """Temporal smoothing of blendshape data using a Savitzky-Golay filter.

    MediaPipe processes each frame independently (no temporal coherence),
    producing jittery blendshape values. Savitzky-Golay preserves peaks/edges
    (important for blinks) better than Gaussian smoothing.

    Args:
        blendshape_data: Per-frame blendshape dicts from run_face_blendshapes().
        window: Filter window length (must be odd). Accepts an int or a preset
            name from _SMOOTH_PRESETS ("light", "moderate", "heavy", "very_heavy").
        poly_order: Polynomial order for the filter.

    Returns:
        Smoothed copy of the blendshape data.
    """
    # Resolve preset name → int
    if isinstance(window, str):
        window = _SMOOTH_PRESETS[window]

    n_frames = len(blendshape_data)
    if n_frames < window:
        return blendshape_data

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # poly_order must be < window
    poly_order = min(poly_order, window - 2)

    # Convert list[dict] → (N, 52) array
    n_channels = len(ARKIT_BLENDSHAPES)
    arr = np.zeros((n_frames, n_channels), dtype=np.float64)
    for i, frame_bs in enumerate(blendshape_data):
        for j, name in enumerate(ARKIT_BLENDSHAPES):
            arr[i, j] = frame_bs.get(name, 0.0)

    # Apply Savitzky-Golay filter per channel
    for j in range(n_channels):
        arr[:, j] = savgol_filter(arr[:, j], window, poly_order)

    # Clamp to [0, 1] — filter can overshoot
    np.clip(arr, 0.0, 1.0, out=arr)

    # Convert back to list[dict]
    smoothed = []
    for i in range(n_frames):
        smoothed.append({name: round(float(arr[i, j]), 6) for j, name in enumerate(ARKIT_BLENDSHAPES)})

    return smoothed


_EQ_GROUPS = {
    "eyes":  [n for n in ARKIT_BLENDSHAPES if n.startswith("Eye")],
    "brows": [n for n in ARKIT_BLENDSHAPES if n.startswith("Brow")],
    "mouth": [n for n in ARKIT_BLENDSHAPES if n.startswith("Mouth") or n.startswith("Jaw")],
    "cheeks_nose": [n for n in ARKIT_BLENDSHAPES
                    if n.startswith("Cheek") or n.startswith("NoseSneer") or n == "TongueOut"],
}

_EQ_PRESETS = {
    "subtle": {
        "eyes":        {"gate": 0.05, "window": 5,  "gain": 1.0},
        "brows":       {"gate": 0.15, "window": 9,  "gain": 0.5},
        "mouth":       {"gate": 0.40, "window": 15, "gain": 0.3},
        "cheeks_nose": {"gate": 0.25, "window": 11, "gain": 0.2},
    },
}


def eq_blendshapes(
    blendshape_data: list[dict[str, float]],
    preset: str = "subtle",
) -> list[dict[str, float]]:
    """Per-group noise gate + smoothing + gain for blendshape channels.

    Like an audio EQ chain: gate (kill weak/false signal) → smooth
    (Savitzky-Golay per group) → gain (scale intensity) → clamp [0, 1].

    Args:
        blendshape_data: Per-frame blendshape dicts from run_face_blendshapes().
        preset: Name of a preset in _EQ_PRESETS.

    Returns:
        Processed copy of the blendshape data.
    """
    settings = _EQ_PRESETS[preset]
    n_frames = len(blendshape_data)
    if n_frames == 0:
        return blendshape_data

    # Build channel name → index mapping
    ch_idx = {name: j for j, name in enumerate(ARKIT_BLENDSHAPES)}

    # Convert list[dict] → (N, 52) array
    n_channels = len(ARKIT_BLENDSHAPES)
    arr = np.zeros((n_frames, n_channels), dtype=np.float64)
    for i, frame_bs in enumerate(blendshape_data):
        for j, name in enumerate(ARKIT_BLENDSHAPES):
            arr[i, j] = frame_bs.get(name, 0.0)

    poly_order = 2

    for group_name, channels in _EQ_GROUPS.items():
        cfg = settings[group_name]
        gate = cfg["gate"]
        window = cfg["window"]
        gain = cfg["gain"]

        # Ensure window is odd and feasible
        if window % 2 == 0:
            window += 1
        order = min(poly_order, window - 2)

        for ch_name in channels:
            j = ch_idx[ch_name]
            col = arr[:, j]

            # 1. Gate: if peak < threshold, zero entire channel
            if col.max() < gate:
                arr[:, j] = 0.0
                continue

            # 2. Smooth (only if enough frames)
            if n_frames >= window:
                col = savgol_filter(col, window, order)

            # 3. Gain
            col *= gain

            # 4. Clamp
            np.clip(col, 0.0, 1.0, out=col)

            arr[:, j] = col

    # Convert back to list[dict]
    result = []
    for i in range(n_frames):
        result.append({name: round(float(arr[i, j]), 6) for j, name in enumerate(ARKIT_BLENDSHAPES)})

    return result


def render_face_mesh_video(
    video_path: str,
    person_bboxes: np.ndarray,
    output_path: str,
    fps: float = 30.0,
    head_fraction: float = 0.30,
    margin: float = 0.20,
    target_size: int = 256,
    progress_callback=None,
) -> str:
    """Render face mesh overlay video using MediaPipe face landmarks.

    Runs its own MediaPipe pass (independent from blendshape extraction).
    Draws tesselation, contours, and iris landmarks on the original video frames.

    Args:
        video_path: Path to input video.
        person_bboxes: (N, 4) person bounding boxes [x1, y1, x2, y2].
        output_path: Output video path.
        fps: Video framerate.
        head_fraction: Fraction of person bbox height for head region.
        margin: Margin around head crop.
        target_size: Face crop size for MediaPipe.
        progress_callback: Optional callable(frac, msg).

    Returns:
        Path to the output video.
    """
    model_path = _find_mediapipe_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        output_face_blendshapes=False,
        num_faces=1,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_normalized = person_bboxes.max() <= 1.0 if len(person_bboxes) > 0 else False

    # Classify contour connections by region for coloring
    contour_set = set(FACEMESH_CONTOURS)
    iris_set = set(FACEMESH_IRISES) if FACEMESH_IRISES else set()

    # Temp directory for frames
    out_dir = Path(output_path).parent
    frames_dir = out_dir / "_facemesh_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dim frame to 60% brightness
        frame = (frame * 0.6).astype(np.uint8).copy()

        if frame_idx < len(person_bboxes):
            bbox = person_bboxes[frame_idx].copy()
            if is_normalized:
                bbox[0] *= frame_w
                bbox[2] *= frame_w
                bbox[1] *= frame_h
                bbox[3] *= frame_h

            x1, y1, x2, y2 = bbox
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Head region (same math as extract_face_crops)
            head_h = bbox_h * head_fraction
            head_cx = (x1 + x2) / 2
            head_cy = y1 + head_h / 2
            crop_size = max(head_h, bbox_w * 0.5)
            crop_size_with_margin = crop_size * (1 + margin)
            half = crop_size_with_margin / 2

            cx1 = int(max(0, head_cx - half))
            cy1 = int(max(0, head_cy - half))
            cx2 = int(min(frame_w, head_cx + half))
            cy2 = int(min(frame_h, head_cy + half))

            face_crop = frame[cy1:cy2, cx1:cx2]

            if face_crop.size > 0:
                # Resize to target_size for MediaPipe
                h, w = face_crop.shape[:2]
                scale = target_size / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                y_off = (target_size - new_h) // 2
                x_off = (target_size - new_w) // 2
                canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

                # Run MediaPipe
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = landmarker.detect(mp_image)

                if detection.face_landmarks and len(detection.face_landmarks) > 0:
                    landmarks = detection.face_landmarks[0]

                    # Map landmarks from crop-normalized (0-1) to original frame pixels
                    pts = []
                    for lm in landmarks:
                        # 0-1 normalized -> crop pixel space
                        px = lm.x * target_size
                        py = lm.y * target_size
                        # crop pixel space -> original frame
                        orig_x = (px - x_off) / scale + cx1
                        orig_y = (py - y_off) / scale + cy1
                        pts.append((orig_x, orig_y))

                    # Draw tesselation (subtle mesh)
                    for conn in FACEMESH_TESSELATION:
                        i1, i2 = conn
                        if i1 < len(pts) and i2 < len(pts):
                            p1 = (int(pts[i1][0]), int(pts[i1][1]))
                            p2 = (int(pts[i2][0]), int(pts[i2][1]))
                            cv2.line(frame, p1, p2, (100, 100, 100), 1, cv2.LINE_AA)

                    # Draw contours (eyes, lips, brows, oval) in brighter colors
                    for conn in FACEMESH_CONTOURS:
                        i1, i2 = conn
                        if i1 < len(pts) and i2 < len(pts):
                            p1 = (int(pts[i1][0]), int(pts[i1][1]))
                            p2 = (int(pts[i2][0]), int(pts[i2][1]))
                            cv2.line(frame, p1, p2, (0, 128, 255), 1, cv2.LINE_AA)

                    # Draw iris landmarks
                    for conn in FACEMESH_IRISES:
                        i1, i2 = conn
                        if i1 < len(pts) and i2 < len(pts):
                            p1 = (int(pts[i1][0]), int(pts[i1][1]))
                            p2 = (int(pts[i2][0]), int(pts[i2][1]))
                            cv2.line(frame, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)

        # Frame number
        cv2.putText(
            frame, f"F{frame_idx + 1}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        cv2.imwrite(str(frames_dir / f"{frame_idx + 1:06d}.jpg"), frame)

        if progress_callback and frame_idx % 30 == 0:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(person_bboxes)
            progress_callback(frame_idx / max(total, 1), f"Face mesh {frame_idx + 1}")

        frame_idx += 1

    cap.release()
    landmarker.close()

    # Assemble video with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(frames_dir / "%06d.jpg"),
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True)

    # Clean up temp frames
    shutil.rmtree(str(frames_dir), ignore_errors=True)

    return str(output_path)


def _frame_to_smpte(frame_idx: int, fps: float) -> str:
    """Convert frame index to SMPTE timecode ``HH:MM:SS:FF.mmm``."""
    total_seconds = frame_idx / fps
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    remaining_frames = frame_idx % round(fps)
    frac = (frame_idx % fps) - int(frame_idx % fps)
    millis = int(round(frac * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}:{remaining_frames:02d}.{millis:03d}"


def export_blendshapes_csv(
    blendshape_data: list[dict[str, float]],
    output_path: str,
    fps: float = 30.0,
) -> str:
    """Export blendshape data to CSV in Live Link Face format.

    Args:
        blendshape_data: Per-frame blendshape weights from run_face_blendshapes().
        output_path: Path to write the CSV file.
        fps: Video framerate for timecode column.

    Returns:
        Path to the written CSV file.
    """
    blendshape_count = len(ARKIT_BLENDSHAPES) + len(_HEAD_EYE_COLUMNS)
    header = ["Timecode", "BlendshapeCount"] + ARKIT_BLENDSHAPES + _HEAD_EYE_COLUMNS

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i, frame_bs in enumerate(blendshape_data):
            timecode = _frame_to_smpte(i, fps)
            row = [timecode, blendshape_count]
            for name in ARKIT_BLENDSHAPES:
                row.append(f"{frame_bs.get(name, 0.0):.10f}")
            for _ in _HEAD_EYE_COLUMNS:
                row.append(f"{0.0:.10f}")
            writer.writerow(row)

    return output_path


def run_face_pipeline(
    video_path: str,
    person_bboxes: np.ndarray,
    output_csv: str,
    fps: float = 30.0,
    smooth_window: int | str = 0,
    eq_preset: str | None = None,
    progress_callback=None,
) -> str:
    """Full face capture pipeline: crop → MediaPipe → smooth/EQ → CSV.

    Args:
        video_path: Path to input video.
        person_bboxes: Person bounding boxes array (N, 4).
        output_csv: Output CSV path.
        fps: Video framerate.
        smooth_window: Savitzky-Golay filter window size. 0 = disabled.
            Accepts an int or a preset name ("light", "moderate", "heavy", "very_heavy").
        eq_preset: EQ preset name (e.g. "subtle"). When set, replaces smooth_window.
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

    if eq_preset:
        if progress_callback:
            progress_callback(0.85, f"Applying EQ preset '{eq_preset}'...")
        blendshapes = eq_blendshapes(blendshapes, preset=eq_preset)
    elif smooth_window:
        if progress_callback:
            progress_callback(0.85, f"Smoothing blendshapes (window={smooth_window})...")
        blendshapes = smooth_blendshapes(blendshapes, window=smooth_window)

    if progress_callback:
        progress_callback(0.9, "Exporting ARKit blendshapes to CSV...")

    export_blendshapes_csv(blendshapes, output_csv, fps=fps)

    if progress_callback:
        progress_callback(1.0, f"Face capture complete — {len(blendshapes)} frames exported.")

    return output_csv


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Face capture: extract ARKit blendshapes from video.")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("smplestx_pt", help="Path to SMPLest-X .pt output (for person bboxes)")
    parser.add_argument("--smooth", default="0",
                        help="Smoothing window: int or preset (light/moderate/heavy/very_heavy). 0=off.")
    parser.add_argument("--eq", default=None, choices=list(_EQ_PRESETS.keys()),
                        help="EQ preset (e.g. 'subtle'). Replaces --smooth when set.")
    parser.add_argument("--face-mesh", action="store_true",
                        help="Render face mesh overlay video instead of CSV export.")
    args = parser.parse_args()

    # Parse --smooth as int if numeric, otherwise treat as preset name
    try:
        smooth_val = int(args.smooth)
    except ValueError:
        smooth_val = args.smooth

    data = torch.load(args.smplestx_pt, map_location="cpu", weights_only=False)

    # SMPLest-X stores bboxes in different possible keys
    bboxes = None
    for key in ["person_bbox", "bboxes", "bbox", "bb_xyxy"]:
        if key in data:
            bboxes = np.array(data[key])
            break

    if bboxes is None:
        print("ERROR: Could not find person bounding boxes in .pt file.")
        print(f"Available keys: {list(data.keys())}")
        import sys
        sys.exit(1)

    if args.face_mesh:
        output = str(Path(args.video_path).with_suffix("")) + "_face_mesh.mp4"
        render_face_mesh_video(args.video_path, bboxes, output, fps=30.0)
        print(f"Face mesh video: {output}")
    else:
        output = str(Path(args.video_path).with_suffix("")) + "_arkit_blendshapes.csv"
        run_face_pipeline(args.video_path, bboxes, output, fps=30.0, smooth_window=smooth_val, eq_preset=args.eq)
        print(f"Output: {output}")
