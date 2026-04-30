"""
Detection helpers — YOLOv8 inference on PC with simulated fallback.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from PIL import Image

from config import DETECTION_CLASSES, VIDEO_WIDTH, VIDEO_HEIGHT

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

MODEL_PATH = "yolov8n.pt"
_MODEL = None

CLASS_MAP = {
    "person": "Person",
    "car": "Vehicle",
    "truck": "Vehicle",
    "bus": "Vehicle",
    "bicycle": "Vehicle",
    "motorcycle": "Vehicle",
    "train": "Vehicle",
    "boat": "Vehicle",
    "airplane": "Drone",
    "bird": "Animal",
    "cat": "Animal",
    "dog": "Animal",
    "horse": "Animal",
    "sheep": "Animal",
    "cow": "Animal",
    "elephant": "Animal",
    "bear": "Animal",
    "zebra": "Animal",
    "giraffe": "Animal",
}


def load_yolo_model(model_path: str = MODEL_PATH):
    """Load a YOLOv8 model for inference."""
    if not YOLO_AVAILABLE:
        raise ImportError("ultralytics package is not installed")

    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(model_path)
    return _MODEL


def _map_class(name: str) -> str:
    return CLASS_MAP.get(name.lower(), "Unknown")


def _to_numpy(frame):
    if isinstance(frame, np.ndarray):
        return frame
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"))
    return np.asarray(frame)


def _parse_yolo_results(results, conf_threshold: float):
    rows = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().astype(float)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = getattr(result, "names", {})

        for xy, conf, cls_id in zip(xyxy, confs, cls_ids):
            if conf < conf_threshold:
                continue
            x1, y1, x2, y2 = [int(max(0, v)) for v in xy]
            rows.append({
                "timestamp":  datetime.now(),
                "class":      _map_class(names.get(int(cls_id), "unknown")),
                "confidence": round(float(conf), 2),
                "x":          x1,
                "y":          y1,
                "width":      max(0, x2 - x1),
                "height":     max(0, y2 - y1),
            })
    return rows


def infer_detections(frame, model=None, conf_threshold: float = 0.30) -> pd.DataFrame:
    """Run inference on a camera frame and return detection records."""
    if model is None or not YOLO_AVAILABLE:
        return get_simulated_detection_data(8)
    if frame is None:
        return pd.DataFrame(columns=["timestamp", "class", "confidence", "x", "y", "width", "height"])

    try:
        image = _to_numpy(frame)
        results = model(image, conf=conf_threshold, imgsz=max(VIDEO_WIDTH, VIDEO_HEIGHT), verbose=False)
        rows = _parse_yolo_results(results, conf_threshold)
        return pd.DataFrame(rows)
    except Exception:
        return get_simulated_detection_data(8)


def get_simulated_detection_data(n: int = 20) -> pd.DataFrame:
    """Generate *n* simulated detection records."""
    now = datetime.now()
    rows = []
    for i in range(n):
        w = int(np.random.randint(30, 150))
        h = int(np.random.randint(30, 200))
        rows.append({
            "timestamp":  now - timedelta(seconds=int(i * np.random.randint(5, 30))),
            "class":      np.random.choice(DETECTION_CLASSES, p=[0.40, 0.30, 0.15, 0.05, 0.10]),
            "confidence": round(float(np.random.uniform(0.55, 0.99)), 2),
            "x":          int(np.random.randint(0, max(1, VIDEO_WIDTH - w))),
            "y":          int(np.random.randint(0, max(1, VIDEO_HEIGHT - h))),
            "width":      w,
            "height":     h,
        })
    return pd.DataFrame(rows)


def get_detection_summary(detections: pd.DataFrame) -> dict:
    """Return total counts per class for a detection DataFrame."""
    summary = {cls: 0 for cls in DETECTION_CLASSES}
    if detections is None or detections.empty:
        return summary

    counts = detections["class"].value_counts().to_dict()
    for cls, count in counts.items():
        if cls in summary:
            summary[cls] = int(count)
        else:
            summary["Unknown"] += int(count)
    return summary
