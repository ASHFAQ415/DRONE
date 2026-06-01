"""
Detection helpers with Raspberry Pi friendly YOLO inference.

Runtime prefers an exported ONNX model through onnxruntime. Ultralytics is kept
optional so Raspberry Pi deployments do not need to import torch just to run.

Integrates ideas from Object_Detection_Files:
  - Class filtering (object-ident-2.py style objects=['cup'] param)
  - Servo actuation on target detection (object-ident-3.py style)
"""

import logging
import numpy as np
import os
from datetime import datetime, timedelta

from config import (
    DETECTION_CLASSES, VIDEO_WIDTH, VIDEO_HEIGHT,
    MODEL_PATH, MODEL_INPUT_SIZE, TARGET_OBJECTS,
)

logger = logging.getLogger(__name__)

_MODEL = None
_MODEL_BACKEND = "simulated"
_ORT = None
_YOLO = None

COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep",
    19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
    24: "backpack", 25: "umbrella", 26: "handbag", 27: "tie",
    28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

PERSON_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TRACKABLE_CLASSES = {"Person", "Animal", "Vehicle", "Drone", "Building"}


def _target_label(class_name: str, index: int) -> str:
    """Return stable operator labels: Person A, Animal B, Vehicle AA, ..."""
    letters = []
    index += 1
    while index:
        index, remainder = divmod(index - 1, len(PERSON_LABELS))
        letters.append(PERSON_LABELS[remainder])
    return f"{class_name} {''.join(reversed(letters))}"


class PersonTargetTracker:
    """Assign stable target labels to trackable detections across frames."""

    def __init__(self, max_missing: int = 12, max_distance: float = 140.0,
                 locked_max_missing: int = 120):
        self.max_missing = max_missing
        self.max_distance = max_distance
        self.locked_max_missing = locked_max_missing
        self._tracks = {}
        self._next_indices = {}

    def update(self, detections: list[dict], locked_target_id: str = "") -> list[dict]:
        annotated = [d.copy() for d in detections]
        target_indices = [
            index for index, det in enumerate(annotated)
            if det.get("class") in TRACKABLE_CLASSES
        ]

        if not target_indices:
            for track in self._tracks.values():
                track["missing"] += 1
            self._prune_missing(locked_target_id)
            return annotated

        centers = {
            index: self._center(annotated[index])
            for index in target_indices
        }
        matches = self._match_existing_tracks(centers, annotated)
        matches = self._match_locked_target(
            matches, annotated, target_indices, locked_target_id
        )
        matched_tracks = set(matches)
        matched_detections = set(matches.values())

        for track_id, det_index in matches.items():
            self._update_track(track_id, annotated[det_index])
            self._annotate_detection(annotated[det_index], self._tracks[track_id])

        for track_id, track in self._tracks.items():
            if track_id not in matched_tracks:
                track["missing"] += 1

        for det_index in sorted(set(target_indices) - matched_detections,
                                key=lambda i: (annotated[i].get("x", 0), annotated[i].get("y", 0))):
            track_id = self._create_track(annotated[det_index])
            self._annotate_detection(annotated[det_index], self._tracks[track_id])

        self._prune_missing(locked_target_id)
        return annotated

    def _match_existing_tracks(self, centers: dict[int, tuple[float, float]],
                               detections: list[dict]) -> dict[str, int]:
        candidates = []
        for track_id, track in self._tracks.items():
            tx, ty = track["center"]
            for det_index, (cx, cy) in centers.items():
                if track.get("class") != detections[det_index].get("class"):
                    continue
                distance = float(np.hypot(cx - tx, cy - ty))
                if distance <= self.max_distance:
                    candidates.append((distance, track_id, det_index))

        matches = {}
        used_tracks = set()
        used_detections = set()
        for _distance, track_id, det_index in sorted(candidates):
            if track_id in used_tracks or det_index in used_detections:
                continue
            matches[track_id] = det_index
            used_tracks.add(track_id)
            used_detections.add(det_index)
        return matches

    def _match_locked_target(self, matches: dict[str, int], annotated: list[dict],
                             target_indices: list[int], locked_target_id: str) -> dict[str, int]:
        if not locked_target_id or locked_target_id in matches:
            return matches

        locked_class = self._class_from_track_id(locked_target_id)
        locked_indices = [
            index for index in target_indices
            if annotated[index].get("class") == locked_class
        ] if locked_class else target_indices

        if len(locked_indices) == 1:
            det_index = locked_indices[0]
            for track_id, matched_det_index in list(matches.items()):
                if matched_det_index == det_index:
                    del matches[track_id]
                    break
            if locked_target_id not in self._tracks:
                self._create_track(annotated[det_index], track_id=locked_target_id)
            matches[locked_target_id] = det_index
            return matches

        matched_detections = set(matches.values())
        unmatched_targets = sorted(
            set(locked_indices) - matched_detections,
            key=lambda i: float(annotated[i].get("confidence", 0)),
            reverse=True,
        )
        if len(unmatched_targets) != 1:
            return matches

        det_index = unmatched_targets[0]
        if locked_target_id not in self._tracks:
            self._create_track(annotated[det_index], track_id=locked_target_id)
        matches[locked_target_id] = det_index
        return matches

    def _create_track(self, detection: dict, track_id: str = "") -> str:
        class_name = detection.get("class", "Target")
        if track_id:
            label = self._label_from_track_id(track_id)
            class_name = self._class_from_track_id(track_id) or class_name
        else:
            next_index = self._next_indices.get(class_name, 0)
            label = _target_label(class_name, next_index)
            track_id = f"{class_name.lower()}_{label.split()[-1].lower()}"
            self._next_indices[class_name] = next_index + 1
        self._tracks[track_id] = {
            "id": track_id,
            "label": label,
            "class": class_name,
            "center": self._center(detection),
            "missing": 0,
        }
        return track_id

    def _update_track(self, track_id: str, detection: dict) -> None:
        track = self._tracks[track_id]
        old_x, old_y = track["center"]
        new_x, new_y = self._center(detection)
        track["center"] = ((old_x * 0.35) + (new_x * 0.65),
                           (old_y * 0.35) + (new_y * 0.65))
        track["missing"] = 0

    def _annotate_detection(self, detection: dict, track: dict) -> None:
        detection["target_id"] = track["id"]
        detection["target_label"] = track["label"]

    def _prune_missing(self, locked_target_id: str = "") -> None:
        stale = [
            track_id for track_id, track in self._tracks.items()
            if track["missing"] > (
                self.locked_max_missing
                if track_id == locked_target_id
                else self.max_missing
            )
        ]
        for track_id in stale:
            del self._tracks[track_id]

    @staticmethod
    def _label_from_track_id(track_id: str) -> str:
        parts = track_id.split("_", 1)
        if len(parts) == 2:
            class_name, suffix = parts
        else:
            class_name, suffix = "target", track_id
        return f"{class_name.title()} {suffix.replace('_', '').upper() or '?'}"

    @staticmethod
    def _class_from_track_id(track_id: str) -> str:
        prefix = track_id.split("_", 1)[0].strip()
        return prefix.title() if prefix else ""

    @staticmethod
    def _center(detection: dict) -> tuple[float, float]:
        return (
            float(detection.get("x", 0)) + float(detection.get("width", 0)) / 2.0,
            float(detection.get("y", 0)) + float(detection.get("height", 0)) / 2.0,
        )

# ── Object Detection ──────────────────────────────────────────
def _map_class(coco_name: str) -> str:
    """Map COCO dataset names to DroneAI classes suitable for aerial tracking."""
    c = coco_name.strip().lower()
    
    if c == "person":
        return "Person"

    if c in {"building", "house", "roof", "tower"}:
        return "Building"
        
    if c in {"car", "truck", "bus", "bicycle", "motorcycle", "train", "boat"}:
        return "Vehicle"
        
    if c in {"airplane", "kite"}:
        return "Drone"
        
    if c in {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}:
        return "Animal"
        
    return c


class OnnxYoloModel:
    """Small YOLOv8 ONNX wrapper that avoids torch/ultralytics at runtime."""

    def __init__(self, model_path: str, input_size: int = MODEL_INPUT_SIZE):
        ort = _load_onnxruntime()
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
        YOLO = _load_ultralytics()
        _MODEL = YOLO(configured_path)
        _MODEL_BACKEND = "Ultralytics"
        return _MODEL

    raise ValueError(f"Unsupported model format: {configured_path}")


def _load_onnxruntime():
    """Import ONNX Runtime only when live AI is enabled."""
    global _ORT
    if _ORT is None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime package is not installed") from exc
        _ORT = ort
    return _ORT


def _load_ultralytics():
    """Import Ultralytics only for .pt development models."""
    global _YOLO
    if _YOLO is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics package is not installed") from exc
        _YOLO = YOLO
    return _YOLO


def get_model_backend() -> str:
    return _MODEL_BACKEND





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


def filter_detections(detections: list[dict],
                      objects: list[str] | None = None) -> list[dict]:
    """Filter detections to only include specified target classes.

    Mirrors the object-ident-2.py / object-ident-3.py pattern:
        getObjects(img, 0.45, 0.2, objects=['cup', 'horse'])

    If *objects* is empty or None, all detections pass through
    (same as the original object-ident.py behaviour).
    """
    if not objects:
        return detections
    return [d for d in detections if d.get("class") in objects]


def infer_detections(frame, model=None, conf_threshold: float = 0.30,
                     objects: list[str] | None = None) -> list[dict]:
    """Run inference on a camera frame and return detection records.

    Args:
        frame:          Camera frame (PIL Image or numpy array).
        model:          Loaded YOLO model (ONNX or Ultralytics).
        conf_threshold: Minimum confidence to keep a detection.
        objects:        Optional list of class names to keep.
                        Mirrors object-ident-2/3's objects parameter.
                        Defaults to config.TARGET_OBJECTS if None.
    """
    if model is None:
        return []
    if frame is None:
        return []

    # Use configured target objects when caller doesn't specify
    if objects is None:
        objects = TARGET_OBJECTS

    try:
        if isinstance(model, OnnxYoloModel):
            rows = model.predict(frame, conf_threshold)
        else:
            image = _to_numpy(frame)
            results = model(image, conf=conf_threshold,
                            imgsz=MODEL_INPUT_SIZE, verbose=False)
            rows = _parse_yolo_results(results, conf_threshold)

        # Apply class filter (object-ident-2/3 style)
        rows = filter_detections(rows, objects)

        return rows
    except Exception:
        return []


def get_simulated_detection_data(n: int = 20) -> list[dict]:
    """Generate *n* simulated detection records."""
    now = datetime.now()
    rows = []
    for i in range(n):
        w = int(np.random.randint(30, 150))
        h = int(np.random.randint(30, 200))
        rows.append({
            "timestamp":  now - timedelta(seconds=int(i * np.random.randint(5, 30))),
            "class":      np.random.choice(
                ["Person", "Vehicle", "Animal", "Drone", "mouse",
                 "laptop", "cup", "cell phone", "backpack", "Unknown"]
            ),
            "confidence": round(float(np.random.uniform(0.55, 0.99)), 2),
            "x":          int(np.random.randint(0, max(1, VIDEO_WIDTH - w))),
            "y":          int(np.random.randint(0, max(1, VIDEO_HEIGHT - h))),
            "width":      w,
            "height":     h,
        })
    return rows


def get_detection_summary(detections) -> dict:
    """Return total counts per class for detection records."""
    summary = {cls: 0 for cls in DETECTION_CLASSES}
    if not detections:
        return summary

    if hasattr(detections, "to_dict"):
        detections = detections.to_dict("records")

    for row in detections:
        cls = row.get("class", "Unknown") if isinstance(row, dict) else "Unknown"
        if cls in summary:
            summary[cls] += 1
        else:
            summary["Unknown"] += 1
    return summary
