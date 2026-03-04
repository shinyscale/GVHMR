"""Convert BVH to FBX using Blender in headless mode."""

import subprocess
import shutil
from pathlib import Path

# Try common Blender locations (Windows paths accessible from WSL2)
BLENDER_CANDIDATES = [
    "/mnt/c/Program Files/Blender Foundation/Blender 4.2/blender.exe",
    "/mnt/c/Program Files/Blender Foundation/Blender 4.3/blender.exe",
    "/mnt/c/Program Files/Blender Foundation/Blender 4.1/blender.exe",
    "/mnt/c/Program Files/Blender Foundation/Blender 3.6/blender.exe",
]

BLENDER_SCRIPT = Path(__file__).resolve().parent / "bvh_to_fbx_blender.py"


def _to_win_path(path: str) -> str:
    """Convert a WSL path to a Windows path for Blender (a Windows .exe).

    Uses wslpath if available, otherwise does a simple /mnt/X/ → X:\\ conversion.
    """
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: manual conversion for /mnt/<drive>/... paths
    p = str(path)
    if p.startswith("/mnt/") and len(p) > 6 and p[5].isalpha() and p[6] == "/":
        return p[5].upper() + ":" + p[6:].replace("/", "\\")
    return p


def find_blender() -> tuple[str | None, bool]:
    """Find Blender executable.

    Returns (blender_path, is_windows_exe).
    """
    # Check PATH first (could be native Linux Blender)
    blender = shutil.which("blender")
    if blender:
        return blender, not blender.endswith(".exe")

    # Check known Windows paths
    for candidate in BLENDER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate, True

    return None, False


def convert_bvh_to_fbx(
    bvh_path: str,
    fbx_path: str,
    fps: float = 30.0,
    naming: str = "mixamo",
) -> str:
    """Convert BVH to FBX using Blender headlessly.

    Args:
        bvh_path: Input BVH file.
        fbx_path: Output FBX file.
        fps: Frame rate.
        naming: Bone naming convention — "mixamo" or "ue5".

    Returns log message string.
    """
    blender, is_windows = find_blender()
    if blender is None:
        return "[BVH→FBX] ERROR: Blender not found. Install Blender or add it to PATH."

    if not Path(bvh_path).is_file():
        return f"[BVH→FBX] ERROR: BVH file not found: {bvh_path}"

    # Ensure parent dir for FBX exists
    Path(fbx_path).parent.mkdir(parents=True, exist_ok=True)

    # Windows Blender needs Windows-style paths
    if is_windows:
        script_path = _to_win_path(str(BLENDER_SCRIPT))
        bvh_arg = _to_win_path(str(bvh_path))
        fbx_arg = _to_win_path(str(fbx_path))
    else:
        script_path = str(BLENDER_SCRIPT)
        bvh_arg = str(bvh_path)
        fbx_arg = str(fbx_path)

    cmd = [
        blender,
        "--background",
        "--python", script_path,
        "--",
        bvh_arg,
        fbx_arg,
        "--fps", str(fps),
        "--naming", naming,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Extract our log lines from Blender's verbose output
        log_lines = [line for line in result.stdout.splitlines() if "[BVH→FBX]" in line]
        log = "\n".join(log_lines) if log_lines else ""

        if result.returncode != 0:
            stderr_tail = result.stderr[-500:] if result.stderr else ""
            stdout_tail = result.stdout[-500:] if result.stdout else ""
            return f"[BVH→FBX] ERROR: Blender exited with code {result.returncode}\n{stderr_tail}\n{stdout_tail}"

        if not Path(fbx_path).is_file():
            return f"[BVH→FBX] ERROR: FBX file was not created.\n{log}"

        return log or f"[BVH→FBX] Exported: {fbx_path}"

    except subprocess.TimeoutExpired:
        return "[BVH→FBX] ERROR: Blender timed out (>300s)."
    except FileNotFoundError:
        return f"[BVH→FBX] ERROR: Could not execute Blender at {blender}"


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python bvh_to_fbx.py input.bvh output.fbx [fps] [mixamo|ue5]")
        sys.exit(1)
    bvh = sys.argv[1]
    fbx = sys.argv[2]
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0
    naming = sys.argv[4] if len(sys.argv) > 4 else "mixamo"
    print(convert_bvh_to_fbx(bvh, fbx, fps, naming=naming))
