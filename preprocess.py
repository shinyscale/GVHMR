"""Video preprocessing — portrait rotation and aspect ratio fix for iPhone videos."""

import json
import subprocess
import shutil
from pathlib import Path


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


def preprocess_video(video_path: str, output_dir: str | None = None) -> tuple[str, str]:
    """Fix portrait video rotation if needed.

    Args:
        video_path: Path to input video.
        output_dir: Directory for preprocessed output. If None, uses same directory.

    Returns:
        Tuple of (output_video_path, log_message).
        If no rotation fix needed, returns the original path unchanged.
    """
    video_path = str(video_path)
    rotation = get_video_rotation(video_path)
    sar_num, sar_den = get_video_sar(video_path)
    needs_sar_fix = (sar_num != sar_den) and (sar_num != 0) and (sar_den != 0)

    if rotation == 0 and not needs_sar_fix:
        return video_path, "No rotation or aspect ratio issues detected — video passed through unchanged."

    # Build description of what we're fixing
    issues = []
    if rotation != 0:
        issues.append(f"rotation={rotation}°")
    if needs_sar_fix:
        issues.append(f"SAR={sar_num}:{sar_den}")

    # Re-encode to bake rotation and/or SAR into actual pixels
    src = Path(video_path)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = src.parent

    output_path = str(out_dir / f"{src.stem}_fixed{src.suffix}")

    # scale filter bakes SAR into real pixel dimensions and resets SAR to 1:1
    # autorotate (default) handles rotation metadata
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vf", "scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        return video_path, f"WARNING: ffmpeg re-encode failed ({', '.join(issues)}), using original.\n{result.stderr[-500:]}"

    msg = f"Video fixed ({', '.join(issues)}). Re-encoded to {Path(output_path).name} with correct pixel dimensions."
    return output_path, msg


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <video_path>")
        sys.exit(1)
    out, log = preprocess_video(sys.argv[1])
    print(log)
    print(f"Output: {out}")
