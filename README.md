# DRONE

Lightweight AI-assisted drone operator UI for Raspberry Pi 4 and an ArduCam IMX219-R camera. The runtime uses a simple built-in Python web server instead of Streamlit, so the browser only requests one JPEG snapshot when you refresh the page.

## Hardware Requirements

- Raspberry Pi 4 Model B, 4 GB RAM recommended
- ArduCam IMX219-R camera module
- Pixhawk 6C flight controller, optional for full autonomy

## Software Setup

1. Install system dependencies on Raspberry Pi OS Debian Trixie 64-bit:

   ```bash
   sudo apt update
   sudo apt install -y python3-full python3-venv python3-dev python3-pip libffi8 libffi-dev rpicam-apps python3-picamera2 python3-gpiozero python3-lgpio
   python3 -c "import ctypes; print('ctypes ok')"
   ```

   Confirm the IMX219 camera is working before starting the UI:

   ```bash
   rpicam-hello -t 0
   ```

2. Create a clean virtual environment with the Raspberry Pi OS Python:

   ```bash
   rm -rf venv
   /usr/bin/python3 -m venv --system-site-packages venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install runtime dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Convert the YOLO model on a laptop/desktop, then copy the exported file to the Raspberry Pi.

   Keep this export environment separate from the Raspberry Pi runtime because it installs Ultralytics and Torch:

   ```bash
   python -m pip install -r requirements-export.txt
   python scripts/export_yolo.py --weights yolov8n.pt --format onnx
   ```

   The app looks for `yolov8n.onnx` by default in the project root. To use another file:

   ```bash
   export DRONE_MODEL_PATH=/path/to/model.onnx
   ```

5. Run the lightweight UI:

   ```bash
   export DRONE_CAMERA_BACKEND=rpicam
   python app.py
   ```

   Then open:

   ```text
   http://<pi-ip-address>:8501
   ```

   The page shows telemetry and a continuous IMX219 camera feed. AI detections can be toggled from the page.

## Runtime Options

Camera backend selection:

```bash
export DRONE_CAMERA_BACKEND=rpicam     # default for Raspberry Pi camera commands
export DRONE_CAMERA_BACKEND=picamera2  # use python3-picamera2 when installed
export DRONE_CAMERA_BACKEND=opencv     # USB/laptop webcam development
export DRONE_CAMERA_BACKEND=simulated  # no hardware
```

Low-load defaults:

```bash
export DRONE_VIDEO_WIDTH=160
export DRONE_VIDEO_HEIGHT=120
export DRONE_CAMERA_FPS=12
export DRONE_ENABLE_AI=0
python app.py
```

Pan-tilt servo target tracking uses gpiozero with the lgpio backend for Debian/Trixie Raspberry Pi OS. Set it up once on the Raspberry Pi:

```bash
./scripts/setup_gpiozero_lgpio.sh
```

Or manually:

```bash
sudo apt-get update
sudo apt-get install -y python3-gpiozero python3-lgpio
python -m pip install -r requirements.txt
```

No `pigpiod` service is required.

Connect the pan servo signal to GPIO18 and the tilt servo signal to GPIO19:

```bash
export DRONE_SERVO_ENABLED=1
export DRONE_SERVO_PIN=18
export DRONE_SERVO_TILT_PIN=19
python app.py
```

If the servo moves opposite to the target, invert that axis:

```bash
export DRONE_SERVO_PAN_INVERT=1
export DRONE_SERVO_TILT_INVERT=1
```

Tracking uses the selected target's offset from the center of the frame. The
servo slows naturally as the target approaches center, and small detection
jitter inside `DRONE_SERVO_DEADZONE` is ignored.

By default the `rpicam` backend uses `rpicam-vid` when it is installed. This is the recommended path for the live IMX219 feed. If you need to fall back to snapshot capture, disable the stream mode:

```bash
export DRONE_RPICAM_STREAM=0
python app.py
```

If the camera is slow to start, increase the rpicam wait time:

```bash
export DRONE_RPICAM_READ_TIMEOUT=2.0
export DRONE_RPICAM_STILL_TIMEOUT=12.0
python app.py
```

You can change the HTTP bind address and port:

```bash
export DRONE_UI_HOST=0.0.0.0
export DRONE_UI_PORT=8501
python app.py
```

## Troubleshooting

If startup fails with `ModuleNotFoundError: No module named '_ctypes'`, the active Python was built without `libffi` support. Recreate the virtual environment with the OS Python instead of `/usr/local/bin/python`:

```bash
sudo apt install -y python3-full python3-venv python3-dev libffi8 libffi-dev
rm -rf venv
/usr/bin/python3 -m venv --system-site-packages venv
source venv/bin/activate
python -c "import ctypes; print('ctypes ok')"
python -m pip install -r requirements.txt
```

If `rpicam-hello -t 0` works but this UI cannot capture, close the preview window and any other process using the camera, then press `Reset camera` in the page or restart `python app.py`.

For the smoothest preview, keep AI detections off. The Pi 4 CPU will otherwise spend most of its time running ONNX inference.

## Features

- On-demand camera snapshots from ArduCam IMX219-R
- Optional object detection with exported YOLOv8 ONNX model
- Lightweight telemetry status page
- Simulated fallback for development without camera hardware
