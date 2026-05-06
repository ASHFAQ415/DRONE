# DRONE

AI-powered autonomous drone dashboard for real-time object detection, tracking, and surveillance using Raspberry Pi 4 and an ArduCam IMX219-R camera. The Raspberry Pi runtime uses an exported YOLOv8 ONNX model through ONNX Runtime CPU.

## Hardware Requirements

- Raspberry Pi 4 Model B (4GB RAM recommended)
- ArduCam IMX219-R camera module
- Pixhawk 6C flight controller (optional for full autonomy)

## Software Setup

1. Install system dependencies on Raspberry Pi:

   ```bash
   sudo apt update
   sudo apt install -y python3-full python3-venv python3-dev python3-pip libffi8 libffi-dev python3-picamera2
   python3 -c "import ctypes; print('ctypes ok')"
   ```

2. Create a clean virtual environment with the Raspberry Pi OS Python:

   ```bash
   rm -rf venv
   /usr/bin/python3 -m venv --system-site-packages venv
   source venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install Raspberry Pi runtime dependencies:

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

5. Run the dashboard:

   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```

## Raspberry Pi Troubleshooting

If startup fails with `ModuleNotFoundError: No module named '_ctypes'`, the active Python was built without `libffi` support. Recreate the virtual environment with the OS Python instead of `/usr/local/bin/python`:

   ```bash
   sudo apt install -y python3-full python3-venv python3-dev libffi8 libffi-dev
   rm -rf venv
   /usr/bin/python3 -m venv --system-site-packages venv
   source venv/bin/activate
   python -c "import ctypes; print('ctypes ok')"
   python -m pip install -r requirements.txt
   ```

If Streamlit still imports `watchdog` from an old environment, run:

```bash
streamlit run app.py --server.fileWatcherType none
```

If the live camera feed is laggy on Raspberry Pi, use the sidebar `Camera Performance` control. The default `Lowest` mode uses `160x120 @ 10 FPS` and runs AI less often so the camera stays responsive. You can also set these before launch:

```bash
export DRONE_VIDEO_WIDTH=160
export DRONE_VIDEO_HEIGHT=120
export DRONE_CAMERA_FPS=10
export DRONE_CAMERA_INFERENCE_EVERY_N=8
streamlit run app.py --server.fileWatcherType none
```

## Features

- Real-time video streaming from ArduCam IMX219-R
- Object detection with exported YOLOv8 ONNX model on Raspberry Pi 4 CPU
- Telemetry monitoring
- Simulated fallback for development
yuop
