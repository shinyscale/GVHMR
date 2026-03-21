#!/usr/bin/env python
"""Compute SimpleVO camera trajectory and save as (L, 4, 4) W2C matrices.

Standalone CLI so it can be called as a subprocess from environments that
lack the full hmr4d dependency chain (yacs, pycolmap, etc.).

Usage:
    python tools/compute_vo.py --video /path/to/video.mp4 --output /path/to/camera.pt
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Compute SimpleVO camera trajectory")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output .pt path for (L,4,4) W2C matrices")
    parser.add_argument("--f_mm", type=float, default=24, help="Focal length in mm (default: 24)")
    parser.add_argument("--scale", type=float, default=0.5, help="Video downscale factor (default: 0.5)")
    parser.add_argument("--step", type=int, default=8, help="Frame sampling step (default: 8)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    assert video_path.exists(), f"Video not found: {video_path}"

    from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO

    print(f"[SimpleVO] video={video_path}, f_mm={args.f_mm}, scale={args.scale}, step={args.step}")
    simple_vo = SimpleVO(str(video_path), scale=args.scale, step=args.step, method="sift", f_mm=args.f_mm)
    vo_results = simple_vo.compute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vo_results, str(output_path))
    print(f"[SimpleVO] Saved {vo_results.shape} W2C matrices to {output_path}")


if __name__ == "__main__":
    main()
