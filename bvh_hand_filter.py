#!/usr/bin/env python3
"""
BVH Hand Joint One Euro Filter
Applies adaptive temporal smoothing to hand/finger channels in BVH files.
Leaves body channels untouched (GVHMR already has temporal modeling).

Usage:
    python bvh_hand_filter.py input.bvh -o output_filtered.bvh
    python bvh_hand_filter.py input.bvh --min-cutoff 0.3 --beta 0.01
    python bvh_hand_filter.py input.bvh --preset aggressive
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass


# --- One Euro Filter Implementation ---

class LowPassFilter:
    def __init__(self):
        self.y = None
        self.s = None

    def __call__(self, value, alpha):
        if self.y is None:
            self.s = value
        else:
            self.s = alpha * value + (1.0 - alpha) * self.s
        self.y = value
        return self.s


class OneEuroFilter:
    """
    Attempt to address jitter and lag in noisy signals.

    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing
                    on slow/static movements. This is the main jitter control.
        beta:       Speed coefficient. Higher = less lag on fast movements.
                    Controls how much the cutoff frequency increases with speed.
        d_cutoff:   Cutoff frequency for derivative estimation (Hz).
                    Usually left at default (1.0).
    """

    def __init__(self, freq, min_cutoff=0.5, beta=0.007, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        prev = self.x_filter.y
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self.dx_filter(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter(x, self._alpha(cutoff, self.freq))


# --- Presets ---

PRESETS = {
    "light": {"min_cutoff": 1.0, "beta": 0.01, "desc": "Light smoothing, preserves fast finger motion"},
    "default": {"min_cutoff": 0.5, "beta": 0.007, "desc": "Balanced — good for general hand gestures"},
    "aggressive": {"min_cutoff": 0.2, "beta": 0.005, "desc": "Heavy smoothing — best for held poses"},
    "held_pose": {"min_cutoff": 0.1, "beta": 0.003, "desc": "Maximum smoothing — for near-static hands"},
}

# Hand joint name patterns to filter
HAND_JOINT_PATTERNS = [
    r"_(Index|Middle|Ring|Pinky|Thumb)\d",
    r"_Wrist$",  # include wrist — often jittery from hand estimation
]


# --- BVH Parser ---

@dataclass
class Joint:
    name: str
    channels: list  # e.g. ["Zrotation", "Xrotation", "Yrotation"]
    channel_offset: int  # index into the per-frame data array
    is_hand: bool


def parse_hierarchy(lines):
    """Parse BVH hierarchy, return list of Joints with channel mappings."""
    joints = []
    channel_offset = 0
    joint_stack = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("ROOT") or stripped.startswith("JOINT"):
            name = stripped.split()[-1]
            joint_stack.append(name)

        elif stripped.startswith("CHANNELS"):
            parts = stripped.split()
            num_channels = int(parts[1])
            channel_names = parts[2:2 + num_channels]
            name = joint_stack[-1] if joint_stack else "unknown"

            is_hand = any(re.search(p, name) for p in HAND_JOINT_PATTERNS)

            joints.append(Joint(
                name=name,
                channels=channel_names,
                channel_offset=channel_offset,
                is_hand=is_hand,
            ))
            channel_offset += num_channels

        elif stripped == "}":
            if joint_stack:
                joint_stack.pop()

    return joints


def parse_bvh(filepath):
    """Parse a BVH file into hierarchy + motion data."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split at MOTION
    parts = content.split("MOTION")
    if len(parts) != 2:
        raise ValueError("Could not find MOTION section in BVH file")

    hierarchy_text = parts[0]
    motion_text = "MOTION" + parts[1]

    # Parse hierarchy
    hierarchy_lines = hierarchy_text.strip().split("\n")
    joints = parse_hierarchy(hierarchy_lines)

    # Parse motion
    motion_lines = motion_text.strip().split("\n")
    num_frames = int(motion_lines[1].split(":")[1].strip())
    frame_time = float(motion_lines[2].split(":")[1].strip())

    frames = []
    for i in range(3, 3 + num_frames):
        values = [float(v) for v in motion_lines[i].split()]
        frames.append(values)

    return hierarchy_text, joints, frames, num_frames, frame_time


def filter_hands(joints, frames, frame_time, min_cutoff, beta, d_cutoff):
    """Apply One Euro filter to hand joint channels only."""
    freq = 1.0 / frame_time
    num_channels = len(frames[0])

    # Identify which channels to filter
    hand_channels = set()
    hand_joint_names = []
    for joint in joints:
        if joint.is_hand:
            hand_joint_names.append(joint.name)
            for i in range(len(joint.channels)):
                hand_channels.add(joint.channel_offset + i)

    if not hand_channels:
        print("WARNING: No hand joints found in skeleton!")
        return frames, []

    # Create a filter for each hand channel
    filters = {}
    for ch in hand_channels:
        filters[ch] = OneEuroFilter(freq, min_cutoff, beta, d_cutoff)

    # Apply filter frame by frame
    filtered_frames = []
    for frame in frames:
        new_frame = list(frame)
        for ch in hand_channels:
            new_frame[ch] = filters[ch](frame[ch])
        filtered_frames.append(new_frame)

    return filtered_frames, hand_joint_names


def write_bvh(filepath, hierarchy_text, frames, num_frames, frame_time):
    """Write filtered BVH file."""
    with open(filepath, "w") as f:
        f.write(hierarchy_text)
        f.write("MOTION\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        for frame in frames:
            f.write(" ".join(f"{v:.4f}" for v in frame) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Apply One Euro filter to hand joints in BVH files"
    )
    parser.add_argument("input", help="Input BVH file")
    parser.add_argument("-o", "--output", help="Output BVH file (default: input_filtered.bvh)")
    parser.add_argument("--preset", choices=PRESETS.keys(), default="default",
                        help="Filter preset (default: default)")
    parser.add_argument("--min-cutoff", type=float, default=None,
                        help="Min cutoff frequency — lower = more smoothing on held poses")
    parser.add_argument("--beta", type=float, default=None,
                        help="Speed coefficient — higher = less lag on fast motion")
    parser.add_argument("--d-cutoff", type=float, default=1.0,
                        help="Derivative cutoff frequency (default: 1.0)")
    parser.add_argument("--filter-body", action="store_true",
                        help="Also filter body joints (not just hands)")
    parser.add_argument("--list-joints", action="store_true",
                        help="List all joints and exit")
    args = parser.parse_args()

    # Parse
    hierarchy_text, joints, frames, num_frames, frame_time = parse_bvh(args.input)

    if args.list_joints:
        for j in joints:
            marker = " [HAND]" if j.is_hand else ""
            print(f"  {j.name}: channels {j.channel_offset}-{j.channel_offset + len(j.channels) - 1} "
                  f"({', '.join(j.channels)}){marker}")
        return

    # Resolve filter parameters
    preset = PRESETS[args.preset]
    min_cutoff = args.min_cutoff if args.min_cutoff is not None else preset["min_cutoff"]
    beta = args.beta if args.beta is not None else preset["beta"]

    # Optionally expand to filter all joints
    if args.filter_body:
        for j in joints:
            j.is_hand = True

    # Filter
    filtered_frames, hand_joints = filter_hands(
        joints, frames, frame_time, min_cutoff, beta, args.d_cutoff
    )

    # Output path
    if args.output:
        out_path = args.output
    else:
        stem = args.input.rsplit(".", 1)[0]
        out_path = f"{stem}_filtered.bvh"

    write_bvh(out_path, hierarchy_text, filtered_frames, num_frames, frame_time)

    # Summary
    fps = 1.0 / frame_time
    print(f"Input:       {args.input}")
    print(f"Output:      {out_path}")
    print(f"Frames:      {num_frames} @ {fps:.1f} fps ({num_frames * frame_time:.1f}s)")
    print(f"Preset:      {args.preset} — {preset['desc']}")
    print(f"Parameters:  min_cutoff={min_cutoff}, beta={beta}, d_cutoff={args.d_cutoff}")
    print(f"Filtered:    {len(hand_joints)} joints — {', '.join(hand_joints)}")
    print(f"Body joints: {'also filtered' if args.filter_body else 'untouched'}")


if __name__ == "__main__":
    main()
