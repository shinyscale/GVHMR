#!/bin/bash
# Motion Capture Studio — launches unified GVHMR + Full Perf Cap GUI
set -e

cd /mnt/f/GVHMR/GVHMR
eval "$(conda shell.bash hook)"
conda activate gvhmr

# Ensure face capture dependencies are available
pip install -q mediapipe opencv-python transforms3d scipy 2>/dev/null || true

# Download MediaPipe face landmarker model if needed
if [ ! -f models/face_landmarker.task ]; then
    echo "Downloading MediaPipe face landmarker model..."
    mkdir -p models
    wget -q -O models/face_landmarker.task \
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    echo "Download complete."
fi

echo "Starting Motion Capture Studio on http://0.0.0.0:7860"
python gvhmr_gui.py
