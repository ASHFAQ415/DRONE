"""
DroneAI Configuration
Central config for drone parameters, detection classes, and dashboard settings.
"""

# ── Drone Hardware ─────────────────────────────────────────────
DRONE_CONFIG = {
    "name": "DroneAI-01",
    "model": "Custom Build — RPi 4 + ArduCam IMX219-R",
    "processor": "Raspberry Pi 4 Model B (4 GB)",
    "ai_accelerator": "Hailo-8L",
    "max_altitude": 120,       # metres
    "max_speed": 15,           # m/s
    "battery_capacity": 5200,  # mAh
    "camera": "ArduCam IMX219-R (8MP, IR-cut)",
    "camera_resolution": (3280, 2464),   # max sensor resolution
    "camera_stream_res": (640, 480),     # streaming resolution
    "camera_fps": 30,
    "flight_controller": "Pixhawk 6C",
}

# ── Home / Launch Position ─────────────────────────────────────
HOME_POSITION = {
    "lat": 28.6139,   # New Delhi (default)
    "lon": 77.2090,
}

# ── Object Detection ──────────────────────────────────────────
DETECTION_CLASSES = ["Person", "Vehicle", "Animal", "Drone", "Unknown"]

DETECTION_COLORS = {
    "Person":  "#ff6b6b",
    "Vehicle": "#ffd93d",
    "Animal":  "#6bcb77",
    "Drone":   "#4d96ff",
    "Unknown": "#888888",
}

# ── Dashboard Settings ────────────────────────────────────────
REFRESH_INTERVAL = 2          # seconds
DEFAULT_CONFIDENCE = 0.65
MAP_ZOOM = 15
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
CAMERA_DEVICE_INDEX = 0
CAMERA_FPS = 30