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
        int(os.getenv("DRONE_VIDEO_WIDTH", "640")),
        int(os.getenv("DRONE_VIDEO_HEIGHT", "480")),
    ),
    "camera_fps": int(os.getenv("DRONE_CAMERA_FPS", "25")),
    "flight_controller": "Pixhawk 6C",
}

# ── Home / Launch Position ─────────────────────────────────────
HOME_POSITION = {
    "lat": 28.6139,   # New Delhi (default)
    "lon": 77.2090,
}

# ── Object Detection ──────────────────────────────────────────
DETECTION_CLASSES = [
    "Person", "Vehicle", "Animal", "Drone", "Building",
    # Individual object classes (COCO labels mapped through CLASS_MAP)
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    "Unknown",
]

# Prefer exported ONNX on Raspberry Pi 4. Override with DRONE_MODEL_PATH.
MODEL_PATH = "yolov8n.onnx"
MODEL_INPUT_SIZE = 640

# Target object filter — only these classes trigger servo/alerts.
# Empty list means *all* classes are processed (like object-ident.py).
# Example: ["Person", "Vehicle"] filters for people and vehicles only.
_TARGET_OBJECTS_ENV = os.getenv(
    "DRONE_TARGET_OBJECTS", "Person,Animal,Vehicle,Drone,Building"
).strip()
TARGET_OBJECTS = (
    []
    if _TARGET_OBJECTS_ENV.lower() in {"", "all", "*"}
    else [s.strip() for s in _TARGET_OBJECTS_ENV.split(",") if s.strip()]
)

DETECTION_COLORS = {
    "Person":  "#ff6b6b",
    "Vehicle": "#ffd93d",
    "Animal":  "#6bcb77",
    "Drone":   "#4d96ff",
    "Building": "#9b8cff",
    "Unknown": "#888888",
}

# ── Servo / GPIO Actuation (RPi only) ─────────────────────────
# Pan-servo tracking for ArduCam IMX219-R on Raspberry Pi 4.
# Enable with DRONE_SERVO_ENABLED=1 and connect SG90 signal wire to GPIO18.
SERVO_ENABLED = os.getenv(
    "DRONE_SERVO_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SERVO_PIN = int(os.getenv("DRONE_SERVO_PIN", "18"))
SERVO_TILT_PIN = int(os.getenv("DRONE_SERVO_TILT_PIN", "19"))
SERVO_ANGLE_TRIGGER = float(os.getenv("DRONE_SERVO_TRIGGER_ANGLE", "-90"))
SERVO_ANGLE_REST = float(os.getenv("DRONE_SERVO_REST_ANGLE", "90"))
SERVO_HOLD_SECONDS = float(os.getenv("DRONE_SERVO_HOLD_SECONDS", "2"))
# SG90 pulse width calibration:
#   At 50 Hz the period is 20 ms.
#   SG90 datasheet: 0.5 ms = 0°, 1.5 ms = 90° (centre), 2.5 ms = 180°.
#   gpiozero AngularServo maps min_pulse_width → -90° and max_pulse_width → +90°.
#   The code offsets so that SERVO_CENTER_ANGLE=90 maps to gpiozero angle=0 (centre).
#   Using 0.5 ms–2.5 ms gives the full physical range of the SG90.
SERVO_MIN_PULSE = float(os.getenv("DRONE_SERVO_MIN_PULSE", "0.0005"))   # 0.5 ms
SERVO_MAX_PULSE = float(os.getenv("DRONE_SERVO_MAX_PULSE", "0.0025"))   # 2.5 ms
SERVO_MIN_ANGLE = float(os.getenv("DRONE_SERVO_MIN_ANGLE", "0"))
SERVO_MAX_ANGLE = float(os.getenv("DRONE_SERVO_MAX_ANGLE", "180"))
SERVO_CENTER_ANGLE = float(os.getenv("DRONE_SERVO_CENTER_ANGLE", "90"))
# Deadzone: pixels from frame centre in which servo does NOT move.
# For a 320 px wide frame, 30 px ≈ 9 % half-width — matches the reference repo.
SERVO_DEADZONE = float(os.getenv("DRONE_SERVO_DEADZONE", "30"))
# SERVO_MOVE_CONFIRM_FRAMES=1 means react on every detection frame (no lag buffer).
SERVO_MOVE_CONFIRM_FRAMES = int(os.getenv("DRONE_SERVO_MOVE_CONFIRM_FRAMES", "1"))
# Proportional tracking gain: step_degrees = clamp(pixel_error × GAIN, ±MAX_STEP).
# At GAIN=0.10 and a 160 px error (target at edge of a 320 px frame):
#   raw_step = 16°, clamped to MAX_STEP=12° → smooth but fast tracking.
SERVO_GAIN = float(os.getenv("DRONE_SERVO_GAIN", "0.10"))
SERVO_MAX_STEP = float(os.getenv("DRONE_SERVO_MAX_STEP", "12.0"))
# Output smoothing alpha (0=no movement, 1=instant). 0.7 balances speed vs jitter.
SERVO_SMOOTHING = float(os.getenv("DRONE_SERVO_SMOOTHING", "0.7"))
SERVO_PAN_INVERT = os.getenv(
    "DRONE_SERVO_PAN_INVERT", "0"
).strip().lower() in {"1", "true", "yes", "on"}
SERVO_TILT_INVERT = os.getenv(
    "DRONE_SERVO_TILT_INVERT", "0"
).strip().lower() in {"1", "true", "yes", "on"}

# ── Lightweight UI Settings ───────────────────────────────────
REFRESH_INTERVAL = float(os.getenv("DRONE_REFRESH_INTERVAL", "0.25"))
DEFAULT_CONFIDENCE = 0.25
MAP_ZOOM = 0

# Raspberry Pi 4 performs much better when IMX219 capture/inference is modest.
# Override these without editing code, for example:
#   DRONE_CAMERA_BACKEND=rpicam DRONE_VIDEO_WIDTH=160 DRONE_VIDEO_HEIGHT=120 DRONE_CAMERA_FPS=10 python app.py
VIDEO_WIDTH = int(os.getenv("DRONE_VIDEO_WIDTH", "320"))
VIDEO_HEIGHT = int(os.getenv("DRONE_VIDEO_HEIGHT", "240"))
CAMERA_ZOOM = max(1.0, float(os.getenv("DRONE_CAMERA_ZOOM", "1.0")))
CAMERA_SENSOR_MODE = os.getenv("DRONE_CAMERA_SENSOR_MODE", "1640:1232").strip()
CAMERA_DEVICE_INDEX = 0
CAMERA_FEED_URL = "http://192.168.29.53:5000"
CAMERA_FPS = int(os.getenv("DRONE_CAMERA_FPS", "15"))
CAMERA_INFERENCE_EVERY_N = int(os.getenv("DRONE_CAMERA_INFERENCE_EVERY_N", "2"))
