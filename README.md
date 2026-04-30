# DRONE

AI-powered autonomous drone for real-time object detection, tracking, and surveillance using Raspberry Pi 4 and ArduCam IMX219-R camera. Supports day/night vision with low-latency edge AI processing.

## Hardware Requirements

- Raspberry Pi 4 Model B (4GB RAM recommended)
- ArduCam IMX219-R camera module
- Pixhawk 6C flight controller (optional for full autonomy)

## Software Setup

1. Install system dependencies on Raspberry Pi:

   ```bash
   sudo apt update
   sudo apt install -y python3-picamera2
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. If you are running on Raspberry Pi, install the camera library:

   ```bash
   sudo apt install -y python3-picamera2
   ```

4. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

## Features

- Real-time video streaming from ArduCam IMX219-R
- Object detection with YOLOv8
- Telemetry monitoring
- Simulated fallback for development
