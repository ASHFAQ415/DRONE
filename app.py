"""
DroneAI lightweight operator UI with continuous live video stream.

Run with:
    python app.py

Provides two video modes:
  /stream   — Continuous MJPEG live feed with real-time YOLO detection
              (like the Object_Detection_Files while True loop)
  /snapshot.jpg — Single JPEG capture (legacy, still available)

The camera runs continuously in a background thread, capturing frames
and running YOLOv8n inference. Detection results trigger servo actuation
when target objects are found (object-ident-3.py style).
"""

from __future__ import annotations

import html
import io
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

from config import (
    CAMERA_DEVICE_INDEX,
    CAMERA_FPS,
    CAMERA_INFERENCE_EVERY_N,
    DEFAULT_CONFIDENCE,
    DRONE_CONFIG,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from utils.detection import (
    PersonTargetTracker,
    get_detection_summary,
    get_model_backend,
    infer_detections,
    load_yolo_model,
)
from utils.servo import release_servo, stop_servo, track_target
from utils.telemetry import get_telemetry_data
from utils.video import (
    add_detection_overlay,
    camera_error_hint,
    get_camera_backend_name,
    get_camera_frame,
    get_camera_status,
    get_simulated_frame,
    open_camera,
    release_camera,
)

logger = logging.getLogger(__name__)

HOST = os.getenv("DRONE_UI_HOST", "0.0.0.0")
PORT = int(os.getenv("DRONE_UI_PORT", "8501"))
CAMERA_BACKEND = os.getenv("DRONE_CAMERA_BACKEND", "auto").strip().lower()
AI_ENABLED = True
OVERLAY_ENABLED = True

# ── Shared state for continuous capture loop ────────────────────
_CAMERA_LOCK = threading.Lock()
_CAMERA = None
_CAMERA_KEY = None
_MODEL = None
_MODEL_LOCK = threading.Lock()

# Latest frame + detections, updated by background thread
_FRAME_LOCK = threading.Lock()
_LATEST_JPEG = None          # bytes — most recent JPEG with overlay
_LATEST_DETECTIONS = []      # list[dict] — most recent detection results
_FRAME_EVENT = threading.Event()  # signaled when a new frame is ready

# Stream settings (can be changed via query params)
_STREAM_SETTINGS_LOCK = threading.Lock()
_STREAM_AI_ENABLED = AI_ENABLED
_STREAM_OVERLAY_ENABLED = OVERLAY_ENABLED
_STREAM_CONFIDENCE = DEFAULT_CONFIDENCE
_STREAM_TARGET_ID = "auto"  # Default to auto-tracking first detected target
_PERSON_TRACKER = PersonTargetTracker()

# Background capture loop control
_CAPTURE_THREAD = None
_INFERENCE_THREAD = None
_CAPTURE_RUNNING = False

# Inference offloading state
_INFERENCE_FRAME = None
_INFERENCE_FRAME_EVENT = threading.Event()


def _float_param(params: dict[str, list[str]], name: str, default: float) -> float:
    try:
        return float(params.get(name, [str(default)])[0])
    except (TypeError, ValueError):
        return default


def _get_camera():
    global _CAMERA, _CAMERA_KEY
    camera_key = (CAMERA_DEVICE_INDEX, VIDEO_WIDTH, VIDEO_HEIGHT,
                  CAMERA_FPS, CAMERA_BACKEND)
    with _CAMERA_LOCK:
        if _CAMERA_KEY != camera_key:
            release_camera(_CAMERA)
            _CAMERA = open_camera(
                CAMERA_DEVICE_INDEX,
                VIDEO_WIDTH,
                VIDEO_HEIGHT,
                CAMERA_FPS,
                backend=CAMERA_BACKEND,
            )
            _CAMERA_KEY = camera_key
        return _CAMERA


def _reset_camera():
    global _CAMERA, _CAMERA_KEY
    with _CAMERA_LOCK:
        release_camera(_CAMERA)
        _CAMERA = None
        _CAMERA_KEY = None


def _get_model():
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = load_yolo_model()
        return _MODEL


def _jpeg_bytes(frame) -> bytes:
    output = io.BytesIO()
    frame.save(output, format="JPEG", quality=78, optimize=True)
    return output.getvalue()


def _get_frame_size(frame):
    """Return (width, height) from a PIL Image or fall back to config defaults."""
    if hasattr(frame, "size"):
        return frame.size          # PIL Image: (width, height)
    return (VIDEO_WIDTH, VIDEO_HEIGHT)


def _inference_loop():
    global _LATEST_DETECTIONS, _INFERENCE_FRAME

    logger.info("Inference loop started")
    while _CAPTURE_RUNNING:
        _INFERENCE_FRAME_EVENT.wait(timeout=1.0)
        _INFERENCE_FRAME_EVENT.clear()

        with _FRAME_LOCK:
            frame_to_process = _INFERENCE_FRAME

        if frame_to_process is None:
            continue

        with _STREAM_SETTINGS_LOCK:
            confidence = _STREAM_CONFIDENCE
            target_id = _STREAM_TARGET_ID

        try:
            dets = infer_detections(
                frame_to_process, model=_get_model(),
                conf_threshold=confidence,
            )
            # Only update tracker when we have a specific locked ID
            if target_id and target_id != "auto":
                dets = _PERSON_TRACKER.update(dets, locked_target_id=target_id)
            else:
                dets = _PERSON_TRACKER.update(dets)

            frame_w, frame_h = _get_frame_size(frame_to_process)
            target_visible = False

            if not target_id:
                # "None (Static)" — servo holds position
                pass

            elif target_id == "auto":
                # Auto-track: prefer Person, then any trackable detection
                # Mirrors reference repo which always follows the first object
                preferred = [d for d in dets if d.get("class") == "Person"]
                pick = (preferred or dets or [None])[0]
                if pick is not None:
                    target_visible = True
                    pick["target_selected"] = True
                    x_c = pick["x"] + pick["width"] / 2.0
                    y_c = pick["y"] + pick["height"] / 2.0
                    track_target(x_c, frame_w, y_c, frame_h)

            else:
                # Specific locked target by ID
                matches = [d for d in dets if d.get("target_id") == target_id]
                if matches:
                    target_visible = True
                    best = matches[0]
                    best["target_selected"] = True
                    x_c = best["x"] + best["width"] / 2.0
                    y_c = best["y"] + best["height"] / 2.0
                    track_target(x_c, frame_w, y_c, frame_h)

            # If we expected a target but didn't see one, tell servo
            if target_id and not target_visible:
                stop_servo()

        except Exception:
            logger.exception("Inference failed")
            stop_servo()
            dets = []

        with _FRAME_LOCK:
            _LATEST_DETECTIONS = dets


def _capture_loop():
    """Background thread: continuously capture frames + trigger YOLO detection."""
    global _LATEST_JPEG, _LATEST_DETECTIONS, _CAPTURE_RUNNING, _INFERENCE_FRAME, _INFERENCE_FRAME_EVENT

    frame_count = 0
    last_frame_time = time.monotonic()

    logger.info("Continuous capture loop started")

    while _CAPTURE_RUNNING:
        loop_start = time.monotonic()

        # Read current stream settings
        with _STREAM_SETTINGS_LOCK:
            ai_enabled = _STREAM_AI_ENABLED
            overlay_enabled = _STREAM_OVERLAY_ENABLED

        # Capture frame
        cam = _get_camera()
        frame = get_camera_frame(cam)
        simulated = False

        if frame is None:
            frame = get_simulated_frame(VIDEO_WIDTH, VIDEO_HEIGHT)
            simulated = True

        # Run YOLO detection asynchronously (every Nth frame)
        frame_count += 1
        if ai_enabled and frame_count % max(1, CAMERA_INFERENCE_EVERY_N) == 0:
            with _FRAME_LOCK:
                _INFERENCE_FRAME = frame.copy() if hasattr(frame, 'copy') else frame
            _INFERENCE_FRAME_EVENT.set()
        elif not ai_enabled:
            with _FRAME_LOCK:
                _LATEST_DETECTIONS = []
            stop_servo()

        with _FRAME_LOCK:
            last_detections = _LATEST_DETECTIONS

        # Apply HUD overlay with detection boxes
        if overlay_enabled and not simulated:
            frame_elapsed = max(0.001, loop_start - last_frame_time)
            display_fps = 1.0 / frame_elapsed
            frame = add_detection_overlay(
                frame,
                last_detections if ai_enabled else [],
                fps=display_fps,
            )

        # Encode to JPEG and store
        jpeg = _jpeg_bytes(frame)
        with _FRAME_LOCK:
            _LATEST_JPEG = jpeg
        _FRAME_EVENT.set()

        # Throttle to target FPS, accounting for processing time
        target_sleep = 1.0 / max(1, CAMERA_FPS)
        elapsed = time.monotonic() - last_frame_time
        sleep_time = max(0.001, target_sleep - elapsed)
        time.sleep(sleep_time)
        last_frame_time = time.monotonic()

    logger.info("Continuous capture loop stopped")


def _start_capture_loop():
    global _CAPTURE_THREAD, _INFERENCE_THREAD, _CAPTURE_RUNNING
    if _CAPTURE_THREAD is not None and _CAPTURE_THREAD.is_alive():
        return
    _CAPTURE_RUNNING = True
    
    _INFERENCE_THREAD = threading.Thread(
        target=_inference_loop, daemon=True, name="inference-loop"
    )
    _INFERENCE_THREAD.start()
    
    _CAPTURE_THREAD = threading.Thread(
        target=_capture_loop, daemon=True, name="capture-loop"
    )
    _CAPTURE_THREAD.start()


def _stop_capture_loop():
    global _CAPTURE_RUNNING
    _CAPTURE_RUNNING = False


def _update_stream_settings(ai_enabled: bool, overlay_enabled: bool,
                            confidence: float, target_id: str):
    global _STREAM_AI_ENABLED, _STREAM_OVERLAY_ENABLED, _STREAM_CONFIDENCE, _STREAM_TARGET_ID
    with _STREAM_SETTINGS_LOCK:
        _STREAM_AI_ENABLED = ai_enabled
        _STREAM_OVERLAY_ENABLED = overlay_enabled
        _STREAM_CONFIDENCE = confidence
        _STREAM_TARGET_ID = target_id


# ── Status & HTML rendering ────────────────────────────────────

def _status_rows(telemetry: dict, ai_enabled: bool) -> list[tuple[str, str]]:
    det_summary = ""
    target_label = "None"
    with _FRAME_LOCK:
        dets = _LATEST_DETECTIONS
    if dets:
        summary = get_detection_summary(dets)
        det_summary = ", ".join(
            f"{cls}: {count}" for cls, count in summary.items() if count > 0
        ) or "none"
        
    with _STREAM_SETTINGS_LOCK:
        target_id = _STREAM_TARGET_ID

    if target_id == "auto":
        target_label = "Auto-Track (First Detected)"
    elif target_id:
        target_label = _target_label_for_id(target_id, dets) or f"{target_id} (not visible)"
        
    return [
        ("Camera backend", get_camera_backend_name()),
        ("Camera status", get_camera_status() or "OK"),
        ("AI", "on" if ai_enabled else "off"),
        ("Tracking Target", target_label),
        ("Model", get_model_backend()),
        ("Detections", det_summary or "—"),
        ("Armed", "yes" if telemetry["armed"] else "no"),
        ("GPS", f"{telemetry['gps_lat']:.6f}, {telemetry['gps_lon']:.6f}"),
        ("Heading", f"{telemetry['heading']} deg"),
    ]


def _target_options(selected_target_id: str) -> list[tuple[str, str]]:
    with _FRAME_LOCK:
        dets = list(_LATEST_DETECTIONS)

    options = []
    seen = set()
    for det in sorted(
        dets,
        key=lambda d: (d.get("target_label", ""), d.get("x", 0), d.get("y", 0)),
    ):
        target_id = det.get("target_id")
        target_label = det.get("target_label")
        if not target_id or not target_label or target_id in seen:
            continue
        options.append((target_id, target_label))
        seen.add(target_id)

    if selected_target_id and selected_target_id not in seen and selected_target_id != "auto":
        options.append((selected_target_id, f"{selected_target_id} (not visible)"))

    return options


def _target_label_for_id(target_id: str, detections: list[dict]) -> str:
    for det in detections:
        if det.get("target_id") == target_id:
            return det.get("target_label", target_id)
    return ""


def _targets_payload() -> bytes:
    with _STREAM_SETTINGS_LOCK:
        selected_target_id = _STREAM_TARGET_ID
    payload = {
        "selected": selected_target_id,
        "targets": [
            {"id": "auto", "label": "Auto-Track (First Detected)"}
        ] + [
            {"id": target_id, "label": target_label}
            for target_id, target_label in _target_options(selected_target_id)
            if target_id != "auto"
        ],
    }
    return json.dumps(payload).encode("utf-8")


def _render_html(params: dict[str, list[str]]) -> bytes:
    telemetry = get_telemetry_data()
    ai_enabled = AI_ENABLED
    overlay_enabled = OVERLAY_ENABLED
    confidence = _float_param(params, "conf", DEFAULT_CONFIDENCE)
    
    with _STREAM_SETTINGS_LOCK:
        current_target = _STREAM_TARGET_ID
    target_id = params.get("target", [current_target])[-1].strip()
    
    status = get_camera_status()
    hint = camera_error_hint(status)

    # Update background loop settings from the form
    _update_stream_settings(ai_enabled, overlay_enabled, confidence, target_id)

    query = f"conf={confidence:.2f}&target={quote(target_id)}"
    rows = "\n".join(
        f"<tr><th>{html.escape(label)}</th>"
        f"<td>{html.escape(value)}</td></tr>"
        for label, value in _status_rows(telemetry, ai_enabled)
    )

    hint_html = (
        f'<p class="hint">{html.escape(hint)}</p>' if hint else ""
    )
    
    # Generate target dropdown options
    target_options = ""
    auto_selected = "selected" if target_id == "auto" else ""
    target_options += f'<option value="auto" {auto_selected}>Auto-Track (First Detected)</option>'
    
    none_selected = "selected" if target_id == "" else ""
    target_options += f'<option value="" {none_selected}>None (Static)</option>'
    
    for option_id, option_label in _target_options(target_id):
        if option_id == "auto":
            continue
        selected = "selected" if option_id == target_id else ""
        target_options += (
            f'<option value="{html.escape(option_id)}" {selected}>'
            f'{html.escape(option_label)}</option>'
        )
        
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DroneAI — Live Feed</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Arial, Helvetica, sans-serif;
      background: #101418;
      color: #eef3f7;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: #101418;
    }}
    main {{
      width: min(980px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 16px 0 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-end;
      border-bottom: 1px solid #26313a;
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: 1.35rem;
      line-height: 1.2;
      margin: 0;
    }}
    .live-badge {{
      display: inline-block;
      background: #e53e3e;
      color: #fff;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 3px;
      margin-left: 8px;
      animation: pulse 1.5s ease-in-out infinite;
    }}
    @keyframes pulse {{
      0%, 100% {{ opacity: 1; }}
      50% {{ opacity: 0.5; }}
    }}
    .sub {{
      margin: 4px 0 0;
      color: #9bacb8;
      font-size: 0.88rem;
    }}
    .frame {{
      display: block;
      width: auto;
      max-width: 100%;
      height: auto;
      max-height: 72vh;
      margin: 0 auto;
      aspect-ratio: {VIDEO_WIDTH} / {VIDEO_HEIGHT};
      object-fit: contain;
      background: #050607;
      border: 1px solid #26313a;
      border-radius: 6px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }}
    .metric {{
      background: #171e24;
      border: 1px solid #26313a;
      border-radius: 6px;
      padding: 10px;
    }}
    .label {{
      color: #9bacb8;
      font-size: 0.72rem;
      text-transform: uppercase;
    }}
    .value {{
      display: block;
      font-size: 1.12rem;
      margin-top: 4px;
    }}
    form, .actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin: 12px 0;
    }}
    label {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      color: #cfdae2;
    }}
    input[type="number"], select {{
      color: #eef3f7;
      background: #171e24;
      border: 1px solid #34424d;
      border-radius: 4px;
      padding: 7px;
    }}
    input[type="number"] {{
      width: 72px;
    }}
    button, a.button {{
      color: #07100c;
      background: #74d99f;
      border: 0;
      border-radius: 4px;
      padding: 8px 12px;
      font-weight: 700;
      text-decoration: none;
      cursor: pointer;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #171e24;
      border: 1px solid #26313a;
      border-radius: 6px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid #26313a;
      padding: 9px 10px;
      vertical-align: top;
    }}
    th {{
      width: 34%;
      color: #9bacb8;
      font-weight: 500;
    }}
    .hint {{
      color: #f0c36d;
      background: #2d2614;
      border: 1px solid #59491f;
      border-radius: 6px;
      padding: 10px;
    }}
    @media (max-width: 700px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
  <script>
    async function refreshTargets() {{
      const select = document.querySelector('select[name="target"]');
      if (!select) return;
      const current = select.value;
      const response = await fetch('/targets.json', {{ cache: 'no-store' }});
      if (!response.ok) return;
      const data = await response.json();
      select.innerHTML = '';
      const none = document.createElement('option');
      none.value = '';
      none.textContent = 'None (Static)';
      select.appendChild(none);
      for (const target of data.targets || []) {{
        const option = document.createElement('option');
        option.value = target.id;
        option.textContent = target.label;
        select.appendChild(option);
      }}
      select.value = current || data.selected || '';
    }}
    window.addEventListener('load', () => {{
      refreshTargets();
      setInterval(refreshTargets, 1000);
    }});
  </script>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>DroneAI <span class="live-badge">● LIVE</span></h1>
        <p class="sub">{html.escape(DRONE_CONFIG["name"])} / {html.escape(DRONE_CONFIG["model"])}</p>
      </div>
      <a class="button" href="/snapshot.jpg?{query}">Open JPEG</a>
    </header>

    <!-- Continuous MJPEG live stream -->
    <img class="frame" src="/stream?{query}" alt="Live camera feed">

    <section class="metrics">
      <div class="metric"><span class="label">Altitude</span><span class="value">{telemetry["altitude"]} m</span></div>
      <div class="metric"><span class="label">Speed</span><span class="value">{telemetry["speed"]} m/s</span></div>
      <div class="metric"><span class="label">Battery</span><span class="value">{telemetry["battery"]}%</span></div>
      <div class="metric"><span class="label">Signal</span><span class="value">{telemetry["signal_strength"]}%</span></div>
    </section>

    <form method="get" action="/">
      <label>Confidence <input type="number" name="conf" min="0.10" max="1.00" step="0.05" value="{confidence:.2f}"></label>
      <label>Track Target: 
        <select name="target">
          {target_options}
        </select>
      </label>
      <button type="submit">Apply</button>
      <a class="button" href="/reset">Reset camera</a>
    </form>

    {hint_html}
    <table>{rows}</table>
  </main>
</body>
</html>"""
    return body.encode("utf-8")


# ── HTTP request handler ───────────────────────────────────────

class DroneRequestHandler(BaseHTTPRequestHandler):
    server_version = "DroneAIHTTP/0.2"

    def do_HEAD(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_headers(200, "text/html; charset=utf-8", 0)
            return
        if parsed.path == "/snapshot.jpg":
            self._send_headers(200, "image/jpeg", 0, cache=False)
            return
        if parsed.path == "/targets.json":
            self._send_headers(200, "application/json; charset=utf-8", 0, cache=False)
            return
        self._send_headers(404, "text/plain; charset=utf-8", 0)

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/":
            self._send(200, "text/html; charset=utf-8",
                       _render_html(params))
            return

        if parsed.path == "/stream":
            self._handle_stream(params)
            return

        if parsed.path == "/targets.json":
            self._send(200, "application/json; charset=utf-8",
                       _targets_payload(), cache=False)
            return

        if parsed.path == "/snapshot.jpg":
            # Grab latest frame from the continuous loop
            with _FRAME_LOCK:
                jpeg = _LATEST_JPEG
            if jpeg is None:
                # Fallback: capture one frame directly
                cam = _get_camera()
                frame = get_camera_frame(cam)
                if frame is None:
                    frame = get_simulated_frame(VIDEO_WIDTH, VIDEO_HEIGHT)
                jpeg = _jpeg_bytes(frame)
            self._send(200, "image/jpeg", jpeg, cache=False)
            return

        if parsed.path == "/reset":
            _reset_camera()
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return

        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def _handle_stream(self, params):
        """Serve a continuous MJPEG stream.

        The browser's <img src="/stream"> tag natively decodes
        multipart/x-mixed-replace as a live video feed.
        """
        # Apply settings from query params
        ai_enabled = AI_ENABLED
        overlay_enabled = OVERLAY_ENABLED
        confidence = _float_param(params, "conf", DEFAULT_CONFIDENCE)
        
        with _STREAM_SETTINGS_LOCK:
            current_target = _STREAM_TARGET_ID
        target_id = params.get("target", [current_target])[-1].strip()
        
        _update_stream_settings(ai_enabled, overlay_enabled, confidence, target_id)

        boundary = b"--droneai_frame"
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=droneai_frame",
        )
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            while _CAPTURE_RUNNING:
                # Wait for a new frame (with timeout to check if stopped)
                _FRAME_EVENT.wait(timeout=1.0)
                _FRAME_EVENT.clear()

                with _FRAME_LOCK:
                    jpeg = _LATEST_JPEG
                if jpeg is None:
                    continue

                # Send MJPEG frame
                header = (
                    boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n"
                )
                self.wfile.write(header)
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client disconnected — this is normal
            pass

    def log_message(self, fmt, *args):
        if os.getenv("DRONE_HTTP_LOG", "0") == "1":
            super().log_message(fmt, *args)

    def _send(self, status: int, content_type: str, data: bytes,
              cache: bool = True):
        self._send_headers(status, content_type, len(data), cache=cache)
        self.wfile.write(data)

    def _send_headers(self, status: int, content_type: str, length: int,
                      cache: bool = True):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()


class DroneHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _create_http_server():
    candidates = [(HOST, PORT)]
    if HOST in {"0.0.0.0", "::", ""}:
        candidates.append(("127.0.0.1", PORT))
    for port in range(PORT + 1, PORT + 6):
        candidates.append((HOST, port))
        if HOST in {"0.0.0.0", "::", ""}:
            candidates.append(("127.0.0.1", port))

    errors = []
    seen = set()
    for host, port in candidates:
        key = (host, port)
        if key in seen:
            continue
        seen.add(key)
        try:
            return DroneHTTPServer((host, port), DroneRequestHandler), host, port
        except OSError as exc:
            errors.append(f"{host}:{port} ({exc})")

    details = "; ".join(errors) or "no bind attempts were made"
    raise RuntimeError(f"Could not start DroneAI HTTP server: {details}")


# ── Application entry point ────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    try:
        httpd, bound_host, bound_port = _create_http_server()
    except RuntimeError as exc:
        print(exc)
        print("Try setting DRONE_UI_HOST=127.0.0.1 or DRONE_UI_PORT=8502, then run python app.py again.")
        raise SystemExit(1) from exc

    # Start camera capture only after the UI server is ready.
    _start_capture_loop()

    def shutdown(_signum=None, _frame=None):
        _stop_capture_loop()
        _reset_camera()
        release_servo()
        httpd.server_close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"DroneAI live UI running at http://{bound_host}:{bound_port}")
    print("Camera feed is streaming continuously with YOLO detection.")
    try:
        httpd.serve_forever()
    finally:
        _stop_capture_loop()
        _reset_camera()
        release_servo()


if __name__ == "__main__":
    main()
