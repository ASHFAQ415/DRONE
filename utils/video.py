"""
Video helpers — RPi camera support + simulated aerial frames with detection overlays.

On Raspberry Pi 4:
    Uses the Raspberry Pi rpicam command stack when available. This matches
    Raspberry Pi OS Debian Trixie installs where `rpicam-hello -t 0` works.
    Picamera2 is kept as an optional in-process backend.

On PC (development):
    Falls back to webcam or simulated frames automatically.
"""

import io
import glob
import os
import select
import shutil
import subprocess
import threading
import time
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime

from config import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    CAMERA_FPS,
    CAMERA_SENSOR_MODE,
    CAMERA_ZOOM,
    DETECTION_COLORS,
)

# ── Raspberry Pi camera backend detection ──────────────────────
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False

RPICAM_VID = shutil.which("rpicam-vid")
RPICAM_JPEG = shutil.which("rpicam-jpeg")
RPICAM_STILL = shutil.which("rpicam-still")
RPICAM_STREAM_ENABLED = os.getenv("DRONE_RPICAM_STREAM", "1").strip().lower() in {"1", "true", "yes", "on"}
RPICAM_READ_TIMEOUT = float(os.getenv("DRONE_RPICAM_READ_TIMEOUT", "1.0"))
RPICAM_STILL_WARMUP_MS = int(os.getenv("DRONE_RPICAM_STILL_WARMUP_MS", "250"))
RPICAM_STILL_TIMEOUT = float(os.getenv("DRONE_RPICAM_STILL_TIMEOUT", "8.0"))
RPICAM_COMMAND = (
    (RPICAM_VID if RPICAM_STREAM_ENABLED else None)
    or RPICAM_JPEG
    or RPICAM_STILL
    or RPICAM_VID
)
RPICAM_COMMAND_AVAILABLE = RPICAM_COMMAND is not None
RPI_CAMERA_AVAILABLE = RPICAM_COMMAND_AVAILABLE or PICAMERA2_AVAILABLE
_CAMERA_STATUS = {
    "backend": "uninitialized",
    "error": "",
}
_PICAMERA2_LOCK = threading.Lock()


def camera_devices_visible():
    """Return True when the OS exposes camera device nodes to this process."""
    return bool(glob.glob("/dev/video*") or glob.glob("/dev/media*"))


def get_camera_backend_name():
    """Return the active/selected camera backend name for the UI."""
    return _CAMERA_STATUS.get("backend", "uninitialized")


def get_camera_status():
    """Return a short diagnostic string about the last camera operation."""
    return _CAMERA_STATUS.get("error", "")


def camera_error_hint(error=None):
    """Return a short operator hint for common Raspberry Pi camera failures."""
    message = (error if error is not None else get_camera_status()).lower()
    if "failed to acquire camera" in message or "pipeline handler in use" in message:
        return (
            "The IMX219 is detected, but another process is using it. Close "
            "`rpicam-hello`, camera previews, or another UI instance, then reset the camera."
        )
    if "no cameras available" in message:
        return "No camera is available to libcamera. Check the ribbon cable, camera port, and OS camera detection."
    if "no /dev/video" in message or "no /dev/media" in message:
        return "Camera device nodes are not visible to this process. Run on the Pi host or expose the camera devices."
    return ""


def _set_camera_status(backend, error=""):
    _CAMERA_STATUS["backend"] = backend
    _CAMERA_STATUS["error"] = error


# ── Detection box generation ────────────────────────────────────

def _random_boxes(width, height):
    """Generate random detection boxes scaled to frame size."""
    if width < 32 or height < 32:
        return []

    n = np.random.randint(2, 6)
    boxes = []
    classes = list(DETECTION_COLORS.keys())[:4]  # skip "Unknown"
    for _ in range(n):
        max_w = max(12, min(150, width // 3))
        max_h = max(12, min(200, height // 3))
        min_w = min(40, max_w)
        min_h = min(50, max_h)

        w = np.random.randint(min_w, max_w + 1)
        h = np.random.randint(min_h, max_h + 1)

        x_margin = min(20, max(0, (width - w) // 3))
        y_margin = min(40, max(0, (height - h) // 3))
        x1 = np.random.randint(x_margin, max(x_margin + 1, width - w - x_margin + 1))
        y1 = np.random.randint(y_margin, max(y_margin + 1, height - h - y_margin + 1))
        cls = np.random.choice(classes)
        conf = round(float(np.random.uniform(0.65, 0.98)), 2)
        boxes.append((x1, y1, x1 + w, y1 + h, cls, conf))
    return boxes


def add_detection_overlay(img, boxes=None, fps=None):
    """
    Draw detection bounding boxes + HUD overlay on any PIL Image.
    Works on both real camera frames and simulated frames.
    """
    draw = ImageDraw.Draw(img)
    width, height = img.size

    if boxes is None:
        boxes = _random_boxes(width, height)
    elif hasattr(boxes, "iterrows"):
        boxes = boxes.to_dict("records")
    elif boxes and isinstance(boxes[0], dict):
        boxes = [row.copy() for row in boxes]
    else:
        boxes = list(boxes)

    # ── detection boxes ──
    for row in boxes:
        if isinstance(row, dict):
            x1 = int(row["x"])
            y1 = int(row["y"])
            x2 = int(row["x"] + row["width"])
            y2 = int(row["y"] + row["height"])
            cls = row["class"]
            conf = float(row["confidence"])
            target_label = row.get("target_label") or cls
            is_selected = bool(row.get("target_selected"))
        else:
            x1, y1, x2, y2, cls, conf = row
            target_label = cls
            is_selected = False

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        color = "#74d99f" if is_selected else DETECTION_COLORS.get(cls, "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4 if is_selected else 2)
        label = f"{target_label} {conf:.0%}"
        tw = len(label) * 7
        draw.rectangle([x1, y1 - 18, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - 16), label, fill="#000000")

    # ── HUD overlay ──
    now_str = datetime.now().strftime("%H:%M:%S")
    cam_label = get_camera_backend_name().upper() if RPI_CAMERA_AVAILABLE else "SIM"
    draw.text((10, 8), f"CAM-01  |  {now_str}  |  {cam_label}  |  REC •", fill="#00ff88")
    if fps is not None:
        draw.text((10, 24), f"FPS {float(fps):.1f}", fill="#00ff88")
    draw.text((10, height - 20), "DroneAI  ·  RPi 4  ·  ArduCam IMX219-R  ·  YOLOv8n", fill="#666666")

    # ── crosshair ──
    cx, cy = width // 2, height // 2
    draw.line([(cx - 25, cy), (cx + 25, cy)], fill="#00ff88", width=1)
    draw.line([(cx, cy - 25), (cx, cy + 25)], fill="#00ff88", width=1)

    return img


# ── Raspberry Pi Camera (ArduCam IMX219-R) ─────────────────────

def _camera_roi_args():
    """Return rpicam ROI args. Default is full sensor, not digital zoom."""
    if CAMERA_ZOOM <= 1.0:
        return []

    crop = 1.0 / max(1.0, CAMERA_ZOOM)
    offset = (1.0 - crop) / 2.0
    roi = f"{offset:.4f},{offset:.4f},{crop:.4f},{crop:.4f}"
    return ["--roi", roi]


def _camera_mode_args():
    """Return IMX219 sensor mode args for normal full-FOV footage."""
    if not CAMERA_SENSOR_MODE:
        return []
    return ["--mode", CAMERA_SENSOR_MODE]


class RpicamCommandCamera:
    """Capture frames through Raspberry Pi OS rpicam-jpeg/still/vid."""

    def __init__(self, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=CAMERA_FPS):
        if not RPICAM_COMMAND_AVAILABLE:
            raise RuntimeError("rpicam-jpeg/rpicam-still command was not found")
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.command = RPICAM_COMMAND
        self.still_command = RPICAM_JPEG or RPICAM_STILL
        self.process = None
        self.buffer = bytearray()
        if os.path.basename(self.command) == "rpicam-vid":
            self._start_video_stream()

    def _start_video_stream(self):
        cmd = [
            self.command,
            "-n",
            "-t",
            "0",
            "--codec",
            "mjpeg",
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--framerate",
            str(self.fps),
            *_camera_mode_args(),
            *_camera_roi_args(),
            "--flush",
            "-o",
            "-",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        _set_camera_status("rpicam-vid", "")

    def _read_process_error(self):
        if self.process is None or self.process.stderr is None:
            return ""

        messages = []
        while True:
            ready, _, _ = select.select([self.process.stderr], [], [], 0)
            if not ready:
                break
            chunk = os.read(self.process.stderr.fileno(), 8192)
            if not chunk:
                break
            messages.append(chunk.decode("utf-8", errors="replace"))

        return "".join(messages).strip()

    def _read_video_frame(self):
        if self.process is None or self.process.poll() is not None:
            self._start_video_stream()
        if self.process.stdout is None:
            _set_camera_status("rpicam-vid", "rpicam-vid stdout is unavailable")
            return None

        deadline = time.monotonic() + max(0.2, RPICAM_READ_TIMEOUT)
        # 1. First, drain the pipe completely to get the absolute newest data
        while True:
            # Short timeout initially, then 0 for draining
            timeout = 0 if len(self.buffer) > 0 else max(0.05, deadline - time.monotonic())
            if timeout < 0:
                break
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not ready:
                break

            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                break
            self.buffer.extend(chunk)

        # 2. Extract the *last* (newest) complete JPEG from the buffer
        latest_jpeg = None
        start = self.buffer.find(b"\xff\xd8")
        while start != -1:
            end = self.buffer.find(b"\xff\xd9", start + 2)
            if end != -1:
                latest_jpeg = bytes(self.buffer[start:end + 2])
                # Delete this frame and everything before it
                del self.buffer[:end + 2]
                start = self.buffer.find(b"\xff\xd8")
            else:
                break

        # Prevent buffer bloat if we get corrupt data
        if len(self.buffer) > 5_000_000:
            self.buffer = self.buffer[-1_000_000:]

        if latest_jpeg is not None:
            try:
                img = Image.open(io.BytesIO(latest_jpeg))
                _set_camera_status("rpicam-vid", "")
                return img.convert("RGB")
            except Exception as exc:
                _set_camera_status("rpicam-vid", f"Unable to decode rpicam-vid frame: {exc}")
                return None

        code = self.process.poll() if self.process else None
        if code is not None and self.still_command is not None:
            return self._read_still_frame()

        detail = self._read_process_error()
        if detail:
            error = detail
        else:
            error = "Timed out waiting for rpicam-vid frame" if code is None else f"rpicam-vid exited with {code}"
        _set_camera_status("rpicam-vid", error)
        if code is not None:
            self.release()
        return None

    def _read_still_frame(self):
        if self.still_command is None:
            _set_camera_status("rpicam-still", "rpicam-jpeg/rpicam-still command was not found")
            return None

        timeout_ms = max(1, RPICAM_STILL_WARMUP_MS)
        cmd = [
            self.still_command,
            "-n",
            "-t",
            str(timeout_ms),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            *_camera_mode_args(),
            *_camera_roi_args(),
            "-o",
            "-",
        ]
        if os.path.basename(self.still_command) == "rpicam-jpeg":
            cmd.extend(["--quality", "75"])

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=RPICAM_STILL_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip() if exc.stderr else ""
            error = stderr or f"{os.path.basename(self.still_command)} timed out after {RPICAM_STILL_TIMEOUT:.1f}s"
            _set_camera_status("rpicam-still", error)
            return None
        if result.returncode != 0 or not result.stdout:
            error = result.stderr.decode("utf-8", errors="replace").strip()
            _set_camera_status("rpicam-still", error or f"{os.path.basename(self.command)} returned no frame")
            return None

        try:
            img = Image.open(io.BytesIO(result.stdout))
            _set_camera_status("rpicam-still", "")
            return img.convert("RGB")
        except Exception as exc:
            _set_camera_status("rpicam-still", f"Unable to decode rpicam frame: {exc}")
            return None

    def read(self):
        if os.path.basename(self.command) == "rpicam-vid":
            return self._read_video_frame()
        return self._read_still_frame()

    def release(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.buffer.clear()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

def init_rpi_camera(width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=CAMERA_FPS):
    """
    Initialize the ArduCam IMX219-R via picamera2.
    Call once and reuse the returned camera object.
    Returns None if not running on RPi.
    """
    if not PICAMERA2_AVAILABLE:
        return None

    picam = Picamera2()
    frame_time_us = int(1_000_000 / max(1, fps))
    try:
        config = picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            buffer_count=2,
            controls={"FrameDurationLimits": (frame_time_us, frame_time_us)},
        )
    except TypeError:
        config = picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            display={"size": (width, height)},
        )
    picam.configure(config)
    picam.start()
    _set_camera_status("picamera2", "")
    return picam


def get_rpi_camera_frame(picam):
    """Capture a single frame from the ArduCam IMX219-R. Returns a PIL Image."""
    if picam is None:
        return None
    with _PICAMERA2_LOCK:
        frame = picam.capture_array()
    _set_camera_status("picamera2", "")
    return Image.fromarray(frame).convert("RGB")


# ── Camera backend selection ───────────────────────────────────

def open_camera(index=0, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, fps=CAMERA_FPS, backend=None):
    """Open the best available camera backend.

    Backend order can be controlled with DRONE_CAMERA_BACKEND:
    auto, rpicam, picamera2, opencv, simulated.
    """
    backend = (backend or os.getenv("DRONE_CAMERA_BACKEND", "auto")).strip().lower()

    if backend == "auto" and not camera_devices_visible():
        _set_camera_status(
            "simulated",
            "No /dev/video* or /dev/media* camera devices are visible to this process",
        )
        return None

    if backend in ("auto", "picamera2") and PICAMERA2_AVAILABLE:
        try:
            return init_rpi_camera(width, height, fps)
        except Exception as exc:
            _set_camera_status("picamera2", str(exc))
            if backend == "picamera2":
                return None

    if backend in ("auto", "rpicam") and RPICAM_COMMAND_AVAILABLE:
        try:
            _set_camera_status("rpicam", "")
            return RpicamCommandCamera(width, height, fps)
        except Exception as exc:
            _set_camera_status("rpicam", str(exc))
            if backend == "rpicam":
                return None

    if backend == "simulated":
        _set_camera_status("simulated", "")
        return None

    if backend not in ("auto", "opencv"):
        _set_camera_status(backend, f"Unknown DRONE_CAMERA_BACKEND={backend}")
        return None

    try:
        import cv2
    except ImportError:
        _set_camera_status("opencv", "OpenCV is not installed")
        return None
    source = index

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        for alt_index in range(1, 4):
            capture = cv2.VideoCapture(alt_index, backend)
            if capture.isOpened():
                break
        else:
            _set_camera_status("opencv", "No OpenCV camera device opened")
            return None

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    _set_camera_status("opencv", "")
    return capture


def get_camera_frame(cam):
    """Get one frame from rpicam, Picamera2, or OpenCV camera resources."""
    if cam is None:
        return None

    if isinstance(cam, RpicamCommandCamera):
        return cam.read()

    if PICAMERA2_AVAILABLE and isinstance(cam, Picamera2):
        try:
            return get_rpi_camera_frame(cam)
        except Exception as exc:
            _set_camera_status("picamera2", str(exc))
            return None

    import cv2
    ret, frame = cam.read()
    if not ret or frame is None:
        _set_camera_status("opencv", "OpenCV camera read failed")
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

    _set_camera_status("opencv", "")
    return Image.fromarray(frame).convert("RGB")


def get_webcam_frame(cam):
    """Backward-compatible alias for older app code."""
    return get_camera_frame(cam)


def release_camera(cam):
    """Release a camera resource cleanly."""
    if cam is None:
        return

    if isinstance(cam, RpicamCommandCamera):
        cam.release()
    elif PICAMERA2_AVAILABLE and isinstance(cam, Picamera2):
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

    if height > 1:
        for _ in range(np.random.randint(1, 3)):
            max_band_h = max(1, min(50, height // 2))
            min_band_h = min(20, max_band_h)
            rh = np.random.randint(min_band_h, max_band_h + 1)
            ry = np.random.randint(0, max(1, height - rh + 1))
            shade = np.random.randint(60, 90)
            base[ry:ry + rh, :, :] = shade

    img = Image.fromarray(base)
    return add_detection_overlay(img)
