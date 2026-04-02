try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover - depends on runtime env
    iio = None
import numpy as np
import torch
from pathlib import Path
import shutil
try:
    import ffmpeg
except ImportError:  # pragma: no cover - depends on runtime env
    ffmpeg = None
import subprocess
from tqdm import tqdm
import cv2


def _require_imageio():
    if iio is None:
        raise ImportError("imageio is required for this video_io_utils function")


def _require_ffmpeg_python():
    if ffmpeg is None:
        raise ImportError("ffmpeg-python is required for this video_io_utils function")


def get_video_lwh(video_path):
    _require_imageio()
    L, H, W, _ = iio.improps(video_path, plugin="pyav").shape
    if L == 0:
        # Some containers don't expose frame count in metadata; fall back to cv2
        cap = cv2.VideoCapture(str(video_path))
        L = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if L == 0:
            # Last resort: decode and count
            L = sum(1 for _ in iio.imiter(video_path, plugin="pyav"))
        if H == 0 or W == 0:
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
    return L, W, H


def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    _require_imageio()
    # If video path not exists, an error will be raised by ffmpegs
    filter_args = []
    should_check_length = False

    # 1. Trim
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(("trim", f"start_frame={start_frame}"))
        else:
            should_check_length = True
            filter_args.append(("trim", f"start_frame={start_frame}:end_frame={end_frame}"))

    # 2. Scale
    if scale != 1.0:
        filter_args.append(("scale", f"iw*{scale}:ih*{scale}"))

    # Excute then check
    frames = iio.imread(video_path, plugin="pyav", filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame

    return frames


def get_video_reader(video_path):
    _require_imageio()
    return iio.imiter(video_path, plugin="pyav")


def read_images_np(image_paths, verbose=False):
    """
    Args:
        image_paths: list of str
    Returns:
        images: np.array, (N, H, W, 3) RGB, uint8
    """
    if verbose:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in tqdm(image_paths)]
    else:
        images = [cv2.imread(str(img_path))[..., ::-1] for img_path in image_paths]
    images = np.stack(images, axis=0)
    return images


def save_video(images, video_path, fps=30, crf=17):
    """
    Args:
        images: (N, H, W, 3) RGB, uint8
        crf: 17 is visually lossless, 23 is default, +6 results in half the bitrate
    0 is lossless, https://trac.ffmpeg.org/wiki/Encode/H.264#crf
    """
    _require_imageio()
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().astype(np.uint8)
    elif isinstance(images, list):
        images = np.array(images).astype(np.uint8)

    with iio.imopen(video_path, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=fps)
        writer._video_stream.options = {"crf": str(crf)}
        writer.write(images)


def get_writer(video_path, fps=30, crf=17):
    """remember to .close()"""
    _require_imageio()
    writer = iio.imopen(video_path, "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=fps)
    writer._video_stream.options = {"crf": str(crf)}
    return writer


_FFMPEG_ENCODER_CACHE = {}


def has_ffmpeg_encoder(encoder_name: str) -> bool:
    """Check whether the local ffmpeg build exposes a given encoder."""
    cached = _FFMPEG_ENCODER_CACHE.get(encoder_name)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        available = result.returncode == 0 and encoder_name in result.stdout
    except OSError:
        available = False

    _FFMPEG_ENCODER_CACHE[encoder_name] = available
    return available


class StreamingVideoWriter:
    """Write BGR frames directly to ffmpeg without temp images."""

    def __init__(
        self,
        video_path,
        width: int,
        height: int,
        fps: float = 30.0,
        crf: int = 17,
        prefer_nvenc: bool = False,
    ):
        self.video_path = str(video_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.crf = int(crf)
        self.frames_written = 0
        self.allow_nvenc_fallback = prefer_nvenc
        use_nvenc = prefer_nvenc and has_ffmpeg_encoder("h264_nvenc")
        self._start_process(use_nvenc)

    def _start_process(self, use_nvenc: bool):
        codec_args = (
            ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(max(self.crf, 15))]
            if use_nvenc
            else ["-c:v", "libx264", "-crf", str(self.crf)]
        )
        self.codec_name = "h264_nvenc" if use_nvenc else "libx264"

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "-",
            "-an",
            *codec_args,
            "-pix_fmt", "yuv420p",
            self.video_path,
        ]

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _read_process_error(self) -> str:
        if self.proc.stderr is None:
            return ""
        return self.proc.stderr.read().decode(errors="ignore")

    def _restart_with_libx264(self):
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
        self.proc.wait()
        self._start_process(use_nvenc=False)

    def write_frame(self, frame):
        frame = np.asarray(frame)
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match "
                f"writer size {(self.height, self.width)}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            error_msg = self._read_process_error()
            if self.codec_name == "h264_nvenc" and self.allow_nvenc_fallback and self.frames_written == 0:
                self._restart_with_libx264()
                self.proc.stdin.write(frame.tobytes())
            else:
                raise RuntimeError(
                    f"ffmpeg pipe failed while writing {self.video_path} "
                    f"with {self.codec_name}: {error_msg}"
                ) from None
        self.frames_written += 1

    def close(self):
        stderr = b""
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        if self.proc.stderr is not None:
            stderr = self.proc.stderr.read()
        return_code = self.proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg failed while writing {self.video_path} "
                f"with {self.codec_name}: {stderr.decode(errors='ignore')}"
            )


def get_stream_writer(video_path, width: int, height: int, fps=30.0, crf=17, prefer_nvenc=False):
    """Return a direct ffmpeg-backed writer for BGR frames."""
    return StreamingVideoWriter(
        video_path=video_path,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        prefer_nvenc=prefer_nvenc,
    )


def copy_file(video_path, out_video_path, overwrite=True):
    if not overwrite and Path(out_video_path).exists():
        return
    shutil.copy(video_path, out_video_path)


def merge_videos_horizontal(in_video_paths: list, out_video_path: str):
    _require_ffmpeg_python()
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "hstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)


def merge_videos_vertical(in_video_paths: list, out_video_path: str):
    _require_ffmpeg_python()
    if len(in_video_paths) < 2:
        raise ValueError("At least two video paths are required for merging.")
    inputs = [ffmpeg.input(path) for path in in_video_paths]
    merged_video = ffmpeg.filter(inputs, "vstack", inputs=len(inputs))
    output = ffmpeg.output(merged_video, out_video_path)
    ffmpeg.run(output, overwrite_output=True, quiet=True)
