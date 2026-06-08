#!/bin/bash

# DroneAI launcher
#
# Usage:
#   ./start.sh             # deploy mode: Raspberry Pi OS camera + servo
#   ./start.sh deploy      # same as default
#   ./start.sh dev         # simulated camera, no servo
#   ./start.sh low         # low-resource hardware mode
#
# Any DRONE_* value can be overridden before running, for example:
#   DRONE_UI_PORT=9000 DRONE_SERVO_CENTER_ANGLE=92 ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-deploy}"

case "$MODE" in
    deploy|dev|low|--help|-h)
        ;;
    *)
        echo "ERROR: unknown mode '$MODE'"
        echo "Use: ./start.sh [deploy|dev|low]"
        exit 1
        ;;
esac

if [ "$MODE" = "--help" ] || [ "$MODE" = "-h" ]; then
    echo "Use: ./start.sh [deploy|dev|low]"
    echo ""
    echo "deploy  Raspberry Pi OS rpicam + servo tracking"
    echo "dev     simulated camera, servo disabled"
    echo "low     reduced resolution/FPS for weaker hardware"
    exit 0
fi

if [ ! -d "venv" ]; then
    echo "ERROR: virtual environment not found."
    echo "Create it with:"
    echo "  python3 -m venv --system-site-packages venv"
    echo "  source venv/bin/activate"
    echo "  python -m pip install -r requirements.txt"
    exit 1
fi

source venv/bin/activate

# Common defaults. Existing environment values win, which keeps deployment
# simple when systemd, SSH, or a shell profile provides site-specific settings.
export DRONE_MODEL_PATH="${DRONE_MODEL_PATH:-yolov8n.onnx}"
export DRONE_UI_HOST="${DRONE_UI_HOST:-0.0.0.0}"
export DRONE_UI_PORT="${DRONE_UI_PORT:-8501}"
export DRONE_NMS_THRESHOLD="${DRONE_NMS_THRESHOLD:-0.70}"

# Low-jitter servo defaults for the camera mount.
export DRONE_SERVO_PIN="${DRONE_SERVO_PIN:-18}"
export DRONE_SERVO_CENTER_ANGLE="${DRONE_SERVO_CENTER_ANGLE:-90}"
export DRONE_SERVO_DEADZONE="${DRONE_SERVO_DEADZONE:-55}"
export DRONE_SERVO_MOVE_CONFIRM_FRAMES="${DRONE_SERVO_MOVE_CONFIRM_FRAMES:-4}"
export DRONE_SERVO_GAIN="${DRONE_SERVO_GAIN:-0.015}"
export DRONE_SERVO_MAX_STEP="${DRONE_SERVO_MAX_STEP:-0.8}"
export DRONE_SERVO_SMOOTHING="${DRONE_SERVO_SMOOTHING:-0.15}"
export DRONE_SERVO_MIN_WRITE_DELTA="${DRONE_SERVO_MIN_WRITE_DELTA:-0.10}"
export DRONE_SERVO_PAN_INVERT="${DRONE_SERVO_PAN_INVERT:-0}"

# Lightweight Deep SORT-style tracker defaults. These keep target IDs stable
# without adding the Torch/ReID cost from the reference implementation.
export DRONE_TRACKING_CONFIDENCE="${DRONE_TRACKING_CONFIDENCE:-0.35}"
export DRONE_TRACK_MAX_AGE="${DRONE_TRACK_MAX_AGE:-24}"
export DRONE_TRACK_N_INIT="${DRONE_TRACK_N_INIT:-2}"
export DRONE_LOST_TARGET_TIMEOUT="${DRONE_LOST_TARGET_TIMEOUT:-2.0}"
export DRONE_TRACK_BBOX_DEADZONE="${DRONE_TRACK_BBOX_DEADZONE:-6.0}"
export DRONE_TRACK_SMOOTHING="${DRONE_TRACK_SMOOTHING:-0.65}"
export DRONE_TRACK_IOU_THRESHOLD="${DRONE_TRACK_IOU_THRESHOLD:-0.15}"

case "$MODE" in
    deploy)
        export DRONE_CAMERA_BACKEND="${DRONE_CAMERA_BACKEND:-rpicam}"
        export DRONE_VIDEO_WIDTH="${DRONE_VIDEO_WIDTH:-320}"
        export DRONE_VIDEO_HEIGHT="${DRONE_VIDEO_HEIGHT:-240}"
        export DRONE_CAMERA_FPS="${DRONE_CAMERA_FPS:-15}"
        export DRONE_SERVO_ENABLED="${DRONE_SERVO_ENABLED:-1}"
        ;;
    dev)
        export DRONE_CAMERA_BACKEND="${DRONE_CAMERA_BACKEND:-simulated}"
        export DRONE_VIDEO_WIDTH="${DRONE_VIDEO_WIDTH:-320}"
        export DRONE_VIDEO_HEIGHT="${DRONE_VIDEO_HEIGHT:-240}"
        export DRONE_CAMERA_FPS="${DRONE_CAMERA_FPS:-15}"
        export DRONE_SERVO_ENABLED="${DRONE_SERVO_ENABLED:-0}"
        ;;
    low)
        export DRONE_CAMERA_BACKEND="${DRONE_CAMERA_BACKEND:-rpicam}"
        export DRONE_VIDEO_WIDTH="${DRONE_VIDEO_WIDTH:-160}"
        export DRONE_VIDEO_HEIGHT="${DRONE_VIDEO_HEIGHT:-120}"
        export DRONE_CAMERA_FPS="${DRONE_CAMERA_FPS:-10}"
        export DRONE_SERVO_ENABLED="${DRONE_SERVO_ENABLED:-1}"
        export DRONE_CAMERA_INFERENCE_EVERY_N="${DRONE_CAMERA_INFERENCE_EVERY_N:-6}"
        ;;
esac

echo "Starting DroneAI ($MODE)"
echo "  UI      : http://${DRONE_UI_HOST}:${DRONE_UI_PORT}"
echo "  Camera  : ${DRONE_CAMERA_BACKEND} ${DRONE_VIDEO_WIDTH}x${DRONE_VIDEO_HEIGHT}@${DRONE_CAMERA_FPS}fps"
echo "  Servo   : enabled=${DRONE_SERVO_ENABLED}, pin=${DRONE_SERVO_PIN}, home=${DRONE_SERVO_CENTER_ANGLE}"
echo "  Tracker : conf=${DRONE_TRACKING_CONFIDENCE}, age=${DRONE_TRACK_MAX_AGE}, lost=${DRONE_LOST_TARGET_TIMEOUT}s"
echo "  Model   : ${DRONE_MODEL_PATH}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec python app.py
