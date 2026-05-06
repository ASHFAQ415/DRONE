"""
DroneAI Configuration
Central config for drone parameters, detection classes, and dashboard settings.
"""

import os

# ── Drone Hardware ─────────────────────────────────────────────
DRONE_CONFIG = {
    "name": "DroneAI-01",
    "model": "Custom Build — RPi 4 + ArduCam IMX219-R",
    "processor": "Raspberry Pi 4 Model B (4 GB)",
    "ai_runtime": "ONNX Runtime CPU",
    "ai_accelerator": "None (CPU inference)",
    "max_altitude": 120,       # metres
    "max_speed": 15,           # m/s
    "battery_capacity": 5200,  # mAh
    "camera": "ArduCam IMX219-R (8MP, IR-cut)",
    "camera_resolution": (3280, 2464),   # max sensor resolution
    "camera_stream_res": (
        int(os.getenv("DRONE_VIDEO_WIDTH", "160")),
        int(os.getenv("DRONE_VIDEO_HEIGHT", "120")),
    ),
    "camera_fps": int(os.getenv("DRONE_CAMERA_FPS", "10")),
    "flight_controller": "Pixhawk 6C",
}

# ── Home / Launch Position ─────────────────────────────────────
HOME_POSITION = {
    "lat": 28.6139,   # New Delhi (default)
    "lon": 77.2090,
}

# ── Object Detection ──────────────────────────────────────────
DETECTION_CLASSES = ["Person", "Vehicle", "Animal", "Drone", "Unknown"]

# Prefer exported ONNX on Raspberry Pi 4. Override with DRONE_MODEL_PATH.
MODEL_PATH = "yolov8n.onnx"
MODEL_INPUT_SIZE = 640

DETECTION_COLORS = {
    "Person":  "#ff6b6b",
    "Vehicle": "#ffd93d",
    "Animal":  "#6bcb77",
    "Drone":   "#4d96ff",
    "Unknown": "#888888",
}

# ── Dashboard Settings ────────────────────────────────────────
REFRESH_INTERVAL = float(os.getenv("DRONE_REFRESH_INTERVAL", "0.75"))
DEFAULT_CONFIDENCE = 0.65
MAP_ZOOM = 15

# Raspberry Pi 4 performs much better when USB webcam capture is kept modest.
# Override these without editing code, for example:
#   DRONE_VIDEO_WIDTH=160 DRONE_VIDEO_HEIGHT=120 DRONE_CAMERA_FPS=10 streamlit run app.py
VIDEO_WIDTH = int(os.getenv("DRONE_VIDEO_WIDTH", "160"))
VIDEO_HEIGHT = int(os.getenv("DRONE_VIDEO_HEIGHT", "120"))
CAMERA_DEVICE_INDEX = 0
CAMERA_FEED_URL = "http://192.168.29.53:5000"
CAMERA_FPS = int(os.getenv("DRONE_CAMERA_FPS", "10"))
CAMERA_INFERENCE_EVERY_N = int(os.getenv("DRONE_CAMERA_INFERENCE_EVERY_N", "8"))
