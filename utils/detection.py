"""
Detection helpers with Raspberry Pi friendly YOLO inference.

Runtime prefers an exported ONNX model through onnxruntime. Ultralytics is kept
optional so Raspberry Pi deployments do not need to import torch just to run.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

from config import DETECTION_CLASSES, VIDEO_WIDTH, VIDEO_HEIGHT, MODEL_PATH, MODEL_INPUT_SIZE

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ort = None
    ONNX_AVAILABLE = False

_MODEL = None
_MODEL_BACKEND = "simulated"

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe",
}

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


class OnnxYoloModel:
    """Small YOLOv8 ONNX wrapper that avoids torch/ultralytics at runtime."""

    def __init__(self, model_path: str, input_size: int = MODEL_INPUT_SIZE):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime package is not installed")
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.names = COCO_NAMES

    def predict(self, frame, conf_threshold: float):
        image = _to_numpy(frame)
        model_input, scale, pad_x, pad_y = _preprocess_for_yolo(image, self.input_size)
        outputs = self.session.run(None, {self.input_name: model_input})
        return _parse_onnx_output(
            outputs[0],
            conf_threshold=conf_threshold,
            image_shape=image.shape[:2],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            names=self.names,
        )


def load_yolo_model(model_path: str | None = None):
    """Load the best available YOLO model.

    ONNX is preferred for Raspberry Pi. A .pt file still works on development
    machines with ultralytics installed.
    """
    global _MODEL, _MODEL_BACKEND
    if _MODEL is not None:
        return _MODEL

    configured_path = model_path or os.getenv("DRONE_MODEL_PATH", MODEL_PATH)
    if not os.path.exists(configured_path):
        _MODEL_BACKEND = f"missing model: {configured_path}"
        raise FileNotFoundError(
            f"{configured_path} was not found. Export yolov8n.pt with "
            "python scripts/export_yolo.py --weights yolov8n.pt --format onnx"
        )

    if configured_path.endswith(".onnx"):
        _MODEL = OnnxYoloModel(configured_path)
        _MODEL_BACKEND = "ONNX Runtime"
        return _MODEL

    if configured_path.endswith(".pt"):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics package is not installed")
        _MODEL = YOLO(configured_path)
        _MODEL_BACKEND = "Ultralytics"
        return _MODEL

    raise ValueError(f"Unsupported model format: {configured_path}")


def get_model_backend() -> str:
    return _MODEL_BACKEND


def _map_class(name: str) -> str:
    return CLASS_MAP.get(name.lower(), "Unknown")


def _to_numpy(frame):
    if isinstance(frame, np.ndarray):
        return frame
    if hasattr(frame, "convert"):
        return np.asarray(frame.convert("RGB"))
    return np.asarray(frame)


def _preprocess_for_yolo(image: np.ndarray, input_size: int):
    image = image[:, :, :3] if image.ndim == 3 else np.stack([image] * 3, axis=-1)
    height, width = image.shape[:2]
    scale = min(input_size / width, input_size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    from PIL import Image
    resized = Image.fromarray(image).resize((new_width, new_height))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = np.asarray(resized)

    model_input = canvas.astype(np.float32) / 255.0
    model_input = np.transpose(model_input, (2, 0, 1))[None, ...]
    return model_input, scale, pad_x, pad_y


def _box_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_one = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    area_many = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_one + area_many - inter, 1e-6)


def _nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        ious = _box_iou(boxes[current], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]
    return keep


def _parse_onnx_output(output, conf_threshold, image_shape, scale, pad_x, pad_y, names):
    predictions = np.squeeze(output)
    if predictions.ndim != 2:
        return []
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    cls_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)
    selected = confs >= conf_threshold
    if not np.any(selected):
        return []

    boxes_xywh = boxes_xywh[selected]
    confs = confs[selected]
    cls_ids = cls_ids[selected].astype(int)

    boxes = np.empty((len(boxes_xywh), 4), dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    img_h, img_w = image_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

    keep = _nms(boxes, confs)
    rows = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index].astype(int)
        rows.append({
            "timestamp":  datetime.now(),
            "class":      _map_class(names.get(int(cls_ids[index]), "unknown")),
            "confidence": round(float(confs[index]), 2),
            "x":          int(x1),
            "y":          int(y1),
            "width":      int(max(0, x2 - x1)),
            "height":     int(max(0, y2 - y1)),
        })
    return rows


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
    if model is None:
        return get_simulated_detection_data(8)
    if frame is None:
        return pd.DataFrame(columns=["timestamp", "class", "confidence", "x", "y", "width", "height"])

    try:
        if isinstance(model, OnnxYoloModel):
            rows = model.predict(frame, conf_threshold)
        else:
            image = _to_numpy(frame)
            results = model(image, conf=conf_threshold, imgsz=MODEL_INPUT_SIZE, verbose=False)
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
