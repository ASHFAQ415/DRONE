"""
Video helpers — RPi camera support + simulated aerial frames with detection overlays.

On Raspberry Pi 4:
    Uses picamera2 (libcamera) to capture from ArduCam IMX219-R.
    Install: sudo apt install -y python3-picamera2

On PC (development):
    Falls back to webcam or simulated frames automatically.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

from config import VIDEO_WIDTH, VIDEO_HEIGHT, DETECTION_COLORS

# ── Try importing picamera2 (only available on RPi) ─────────────
try:
    from picamera2 import Picamera2
    RPI_CAMERA_AVAILABLE = True
except ImportError:
    RPI_CAMERA_AVAILABLE = False


# ── Detection box generation ────────────────────────────────────

def _random_boxes(width, height):
    """Generate random detection boxes scaled to frame size."""
    n = np.random.randint(2, 6)
    boxes = []
    classes = list(DETECTION_COLORS.keys())[:4]  # skip "Unknown"
    for _ in range(n):
        w = np.random.randint(40, min(150, width // 4))
        h = np.random.randint(50, min(200, height // 3))
        x1 = np.random.randint(20, max(21, width - w - 20))
        y1 = np.random.randint(40, max(41, height - h - 20))
        cls = np.random.choice(classes)
        conf = round(float(np.random.uniform(0.65, 0.98)), 2)
        boxes.append((x1, y1, x1 + w, y1 + h, cls, conf))
    return boxes


def add_detection_overlay(img, boxes=None):
    """
    Draw detection bounding boxes + HUD overlay on any PIL Image.
    Works on both real camera frames and simulated frames.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if boxes is None:
        boxes = _random_boxes(width, height)
    elif hasattr(boxes, "iterrows"):
        boxes = [
            (
                int(row["x"]),
                int(row["y"]),
                int(row["x"] + row["width"]),
                int(row["y"] + row["height"]),
                row["class"],
                float(row["confidence"]),
            )
            for _, row in boxes.iterrows()
        ]
    else:
        boxes = list(boxes)

    # ── detection boxes ──
    for x1, y1, x2, y2, cls, conf in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        color = DETECTION_COLORS.get(cls, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cls} {conf:.0%}"
        tw = len(label) * 7
        draw.rectangle([x1, y1 - 18, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), label, fill="#000000")

    # ── HUD overlay ──
    now_str = datetime.now().strftime("%H:%M:%S")
    cam_label = "IMX219-R" if RPI_CAMERA_AVAILABLE else "SIM"
    draw.text((10, 8), f"CAM-01  |  {now_str}  |  {cam_label}  |  REC •", fill="#00ff88")
    draw.text((10, height - 20), "DroneAI  ·  RPi 4  ·  ArduCam IMX219-R  ·  YOLOv8n", fill="#666666")

    # ── crosshair ──
    cx, cy = width // 2, height // 2
    draw.line([(cx - 25, cy), (cx + 25, cy)], fill="#00ff88", width=1)
    draw.line([(cx, cy - 25), (cx, cy + 25)], fill="#00ff88", width=1)

    return img


# ── Raspberry Pi Camera (ArduCam IMX219-R via picamera2) ────────

def init_rpi_camera(width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
    """
    Initialize the ArduCam IMX219-R via picamera2.
    Call once and cache with @st.cache_resource in app.py.
    Returns None if not running on RPi.
    """
    if not RPI_CAMERA_AVAILABLE:
        return None

    picam = Picamera2()
    config = picam.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"},
        display={"size": (width, height)},
    )
    picam.configure(config)
    picam.start()
    return picam


def get_rpi_camera_frame(picam):
    """Capture a single frame from the ArduCam IMX219-R. Returns a PIL Image."""
    if picam is None:
        return None
    frame = picam.capture_array()
    return Image.fromarray(frame)


# ── Webcam support (for development on PC) ──────────────────────

def open_camera(index=0, width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
    """Open camera by index. On RPi, returns RPi camera; on PC, returns OpenCV capture."""
    if RPI_CAMERA_AVAILABLE:
        return init_rpi_camera(width, height)

    import cv2
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        for alt_index in range(1, 4):
            capture = cv2.VideoCapture(alt_index, backend)
            if capture.isOpened():
                break
        else:
            return None

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, 30)
    return capture


def get_webcam_frame(cam):
    """Get frame from camera. Handles both RPi picamera2 and OpenCV webcam."""
    if cam is None:
        return None

    if RPI_CAMERA_AVAILABLE:
        try:
            return get_rpi_camera_frame(cam)
        except Exception:
            return None

    import cv2
    ret, frame = cam.read()
    if not ret or frame is None:
        return None

    try:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    return Image.fromarray(frame)


def release_camera(cam):
    """Release a camera resource cleanly."""
    if cam is None:
        return

    if RPI_CAMERA_AVAILABLE:
        try:
            cam.stop()
            cam.close()
        except Exception:
            pass
    else:
        try:
            cam.release()
        except Exception:
            pass


# ── Simulated frame (fallback for PC development) ──────────────
def get_simulated_frame(width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
    """Synthetic aerial-view image (used when no RPi camera is available)."""
    r = np.random.randint(20, 45, (height, width), dtype=np.uint8)
    g = np.random.randint(35, 70, (height, width), dtype=np.uint8)
    b = np.random.randint(18, 40, (height, width), dtype=np.uint8)
    base = np.stack([r, g, b], axis=-1)

    for _ in range(np.random.randint(1, 3)):
        ry = np.random.randint(0, height - 60)
        rh = np.random.randint(20, 50)
        shade = np.random.randint(60, 90)
        base[ry:ry + rh, :, :] = shade

    img = Image.fromarray(base)
    return add_detection_overlay(img)
