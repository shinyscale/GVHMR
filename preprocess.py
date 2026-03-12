"""Video preprocessing — portrait rotation and aspect ratio fix for iPhone videos."""

import json
import subprocess
import shutil
from pathlib import Path


def get_video_fps(video_path: str) -> float:
    """Detect average video FPS using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                rate = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate") or "30/1"
                if "/" in rate:
                    num, den = rate.split("/")
                    den_val = float(den)
                    if den_val != 0:
                        return float(num) / den_val
                return float(rate)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError, ZeroDivisionError):
        pass
    return 30.0


def get_video_sar(video_path: str) -> tuple[int, int]:
    """Detect sample aspect ratio (SAR) from video.

    Returns (sar_num, sar_den). (1, 1) means square pixels (no distortion).
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "stream=sample_aspect_ratio",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                sar = streams[0].get("sample_aspect_ratio", "1:1")
                if sar and ":" in sar:
                    num, den = sar.split(":")
                    return int(num), int(den)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass
    return 1, 1


def get_video_rotation(video_path: str) -> int:
    """Detect rotation metadata from video using ffprobe.

    Returns rotation in degrees (0, 90, 180, 270).
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_streams",
        "-show_entries", "stream=width,height",
        "-show_entries", "stream_side_data=rotation",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return 0
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return 0
        stream = streams[0]
        # Check side_data for rotation
        side_data = stream.get("side_data_list", [])
        for sd in side_data:
            rot = sd.get("rotation", 0)
            if rot:
                return int(rot) % 360
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass

    # Fallback: check displaymatrix via format-level tags
    cmd2 = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "stream_tags=rotate",
        "-select_streams", "v:0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if streams:
                tags = streams[0].get("tags", {})
                rotate = tags.get("rotate", "0")
                return int(rotate) % 360
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        pass

    return 0


def _format_fps_tag(target_fps: float) -> str:
    if abs(target_fps - round(target_fps)) < 1e-3:
        return f"{int(round(target_fps))}fps"
    return f"{target_fps:.2f}".rstrip("0").rstrip(".") + "fps"


def preprocess_video(
    video_path: str,
    output_dir: str | None = None,
    target_fps: float | None = None,
    fps_tolerance: float = 0.5,
) -> tuple[str, str]:
    """Fix rotation/SAR and optionally resample to a target FPS.

    Args:
        video_path: Path to input video.
        output_dir: Directory for preprocessed output. If None, uses same directory.
        target_fps: Optional target analysis FPS. When provided and different from
            the source by more than fps_tolerance, frames are resampled once here.
        fps_tolerance: Allowed FPS difference before resampling.

    Returns:
        Tuple of (output_video_path, log_message).
        If no rotation fix needed, returns the original path unchanged.
    """
    video_path = str(video_path)
    rotation = get_video_rotation(video_path)
    sar_num, sar_den = get_video_sar(video_path)
    source_fps = get_video_fps(video_path)
    needs_sar_fix = (sar_num != sar_den) and (sar_num != 0) and (sar_den != 0)
    needs_fps_fix = (
        target_fps is not None
        and target_fps > 0
        and abs(source_fps - target_fps) > fps_tolerance
    )

    if rotation == 0 and not needs_sar_fix and not needs_fps_fix:
        return video_path, (
            f"No preprocessing needed — source FPS {source_fps:.2f} already matches the analysis path."
        )

    # Build description of what we're fixing
    issues = []
    if rotation != 0:
        issues.append(f"rotation={rotation}°")
    if needs_sar_fix:
        issues.append(f"SAR={sar_num}:{sar_den}")
    if needs_fps_fix:
        issues.append(f"fps={source_fps:.2f}->{target_fps:.2f}")

    # Re-encode to bake rotation and/or SAR into actual pixels
    src = Path(video_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = src.parent

    suffix_parts = []
    if rotation != 0 or needs_sar_fix:
        suffix_parts.append("fixed")
    if needs_fps_fix:
        suffix_parts.append(_format_fps_tag(target_fps))
    suffix_text = "_" + "_".join(suffix_parts) if suffix_parts else "_preprocessed"
    output_path = str(out_dir / f"{src.stem}{suffix_text}{src.suffix}")

    filters = []
    # scale filter bakes SAR into real pixel dimensions and resets SAR to 1:1
    # autorotate (default) handles rotation metadata
    if rotation != 0 or needs_sar_fix:
        filters.append("scale=trunc(iw*sar/2)*2:trunc(ih/2)*2")
        filters.append("setsar=1")
    if needs_fps_fix:
        filters.append(f"fps={target_fps:.6f}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
    ]
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return video_path, f"WARNING: ffmpeg re-encode failed ({', '.join(issues)}), using original.\n{result.stderr[-500:]}"

    msg = (
        f"Video preprocessed ({', '.join(issues)}). "
        f"Re-encoded to {Path(output_path).name} for shared analysis input."
    )
    return output_path, msg


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <video_path>")
        sys.exit(1)
    out, log = preprocess_video(sys.argv[1])
    print(log)
    print(f"Output: {out}")
