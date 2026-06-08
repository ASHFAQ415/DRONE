"""
Deep SORT-style target tracking for the DroneAI runtime.

The reference implementation in temp_repo combines YOLO detections with
Deep SORT track persistence. This module keeps the same pipeline shape while
staying Raspberry Pi friendly: YOLO ONNX remains the detector, and the tracker
uses motion prediction, IoU association, track aging, and confirmed track IDs
without a Torch ReID embedder in the hot path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import (
    LOST_TARGET_TIMEOUT,
    TRACK_BBOX_DEADZONE,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_AGE,
    TRACK_N_INIT,
    TRACK_SMOOTHING,
    TRACKING_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

TRACKABLE_CLASSES = {"Person", "Animal", "Vehicle", "Drone", "Building"}


@dataclass
class TrackState:
    """Internal persistent track state."""

    numeric_id: int
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    created_at: float
    updated_at: float
    hits: int = 1
    age: int = 1
    missing: int = 0
    vx: float = 0.0
    vy: float = 0.0
    last_center: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))

    @property
    def track_id(self) -> str:
        return f"track_{self.numeric_id}"

    @property
    def label(self) -> str:
        return f"{self.class_name} Track {self.numeric_id}"

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return (x + w / 2.0, y + h / 2.0)

    @property
    def confirmed(self) -> bool:
        return self.hits >= max(1, TRACK_N_INIT)

    @property
    def visible(self) -> bool:
        return self.missing == 0


class DeepSortTargetTracker:
    """Track detections and expose persistent target IDs for servo control."""

    def __init__(
        self,
        confidence_threshold: float = TRACKING_CONFIDENCE_THRESHOLD,
        max_age: int = TRACK_MAX_AGE,
        lost_timeout: float = LOST_TARGET_TIMEOUT,
        bbox_deadzone: float = TRACK_BBOX_DEADZONE,
        smoothing: float = TRACK_SMOOTHING,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_age = max(1, max_age)
        self.lost_timeout = max(0.0, lost_timeout)
        self.bbox_deadzone = max(0.0, bbox_deadzone)
        self.smoothing = min(1.0, max(0.0, smoothing))
        self.iou_threshold = min(1.0, max(0.0, iou_threshold))
        self._tracks: dict[str, TrackState] = {}
        self._next_id = 1

    def update(self, detections: list[dict], frame_size: tuple[int, int], locked_target_id: str = "") -> list[dict]:
        """Update tracks from detector rows and return visible annotated rows."""
        now = time.monotonic()
        annotated = [d.copy() for d in detections]
        candidates = [
            (index, det)
            for index, det in enumerate(annotated)
            if self._is_trackable(det)
        ]

        for track in self._tracks.values():
            self._predict(track, frame_size)

        matches = self._match(candidates, locked_target_id)
        matched_track_ids = set(matches)
        matched_detection_indices = set(matches.values())

        for track_id, det_index in matches.items():
            track = self._tracks[track_id]
            self._update_track(track, annotated[det_index], now)
            self._annotate_detection(annotated[det_index], track)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.missing += 1
                track.age += 1

        for det_index, det in candidates:
            if det_index in matched_detection_indices:
                continue
            track = self._create_track(det, now)
            self._annotate_detection(annotated[det_index], track)

        self._prune_stale()
        return [
            det for det in annotated
            if det.get("target_id") and self._tracks.get(det["target_id"], None)
        ]

    def target_status(self, track_id: str) -> dict:
        """Return visible/held/expired state for a selected target ID."""
        if not track_id:
            return {"state": "none", "track": None, "center": None}
        track = self._tracks.get(track_id)
        if track is None:
            return {"state": "expired", "track": None, "center": None}
        if not track.confirmed:
            return {"state": "pending", "track": track, "center": None}
        if track.visible:
            return {"state": "visible", "track": track, "center": track.center}
        elapsed = time.monotonic() - track.updated_at
        if elapsed <= self.lost_timeout:
            return {"state": "lost_hold", "track": track, "center": track.center}
        return {"state": "expired", "track": track, "center": None}

    def label_for_id(self, track_id: str) -> str:
        track = self._tracks.get(track_id)
        if track is not None:
            return track.label
        if track_id.startswith("track_"):
            return f"Track {track_id.split('_', 1)[1]}"
        return track_id

    def reset(self) -> None:
        if not self._tracks:
            return
        self._tracks.clear()
        self._next_id = 1
        logger.info("tracking reset")

    def _is_trackable(self, detection: dict) -> bool:
        cls_name = detection.get("class")
        if not cls_name or cls_name == "Unknown":
            return False
        return float(detection.get("confidence", 0.0)) >= self.confidence_threshold

    def _create_track(self, detection: dict, now: float) -> TrackState:
        bbox = self._bbox(detection)
        track = TrackState(
            numeric_id=self._next_id,
            class_name=str(detection.get("class", "Target")),
            bbox=bbox,
            confidence=float(detection.get("confidence", 0.0)),
            created_at=now,
            updated_at=now,
            last_center=self._center_from_bbox(bbox),
        )
        self._tracks[track.track_id] = track
        self._next_id += 1
        logger.info(
            "track created id=%s class=%s bbox=%s confidence=%.2f",
            track.track_id,
            track.class_name,
            self._round_bbox(track.bbox),
            track.confidence,
        )
        return track

    def _update_track(self, track: TrackState, detection: dict, now: float) -> None:
        old_center = track.center
        new_bbox = self._bbox(detection)
        new_center = self._center_from_bbox(new_bbox)
        center_delta = (
            (new_center[0] - old_center[0]) ** 2
            + (new_center[1] - old_center[1]) ** 2
        ) ** 0.5

        if center_delta < self.bbox_deadzone:
            smoothed_bbox = track.bbox
        else:
            alpha = self.smoothing
            smoothed_bbox = tuple(
                (old * (1.0 - alpha)) + (new * alpha)
                for old, new in zip(track.bbox, new_bbox)
            )

        smoothed_center = self._center_from_bbox(smoothed_bbox)
        track.vx = smoothed_center[0] - old_center[0]
        track.vy = smoothed_center[1] - old_center[1]
        track.bbox = smoothed_bbox
        track.confidence = float(detection.get("confidence", track.confidence))
        track.updated_at = now
        track.hits += 1
        track.age += 1
        track.missing = 0
        track.last_center = smoothed_center

    def _predict(self, track: TrackState, frame_size: tuple[int, int]) -> None:
        if track.missing <= 0:
            return
        width, height = frame_size
        x, y, w, h = track.bbox
        # Apply velocity decay to prevent runaway prediction
        track.vx *= 0.85
        track.vy *= 0.85
        x = min(max(0.0, x + track.vx), max(0.0, width - w))
        y = min(max(0.0, y + track.vy), max(0.0, height - h))
        track.bbox = (x, y, w, h)

    def _match(self, candidates: list[tuple[int, dict]], locked_target_id: str = "") -> dict[str, int]:
        scored = []
        for track_id, track in self._tracks.items():
            for det_index, det in candidates:
                if det.get("class") != track.class_name:
                    continue
                det_bbox = self._bbox(det)
                iou_score = self._iou(track.bbox, det_bbox)
                distance = self._center_distance(track.bbox, det_bbox)
                _, _, tw, th = track.bbox
                _, _, dw, dh = det_bbox
                
                # Base size gate (slightly larger to accommodate standard motions)
                size_gate = max(45.0, ((tw * tw + th * th) ** 0.5 + (dw * dw + dh * dh) ** 0.5) * 0.40)
                if track.missing > 0:
                    size_gate *= (1.0 + track.missing * 0.35)
                
                # Expand search gate significantly for the locked target to prevent track-loss
                if locked_target_id and track_id == locked_target_id:
                    size_gate *= 2.0
                
                if iou_score >= self.iou_threshold:
                    score = 1.0 + iou_score
                elif distance <= size_gate:
                    score = max(0.0, 1.0 - (distance / size_gate))
                else:
                    continue
                if score > 0.0:
                    scored.append((score, track_id, det_index))

        matches = {}
        used_tracks = set()
        used_detections = set()
        for score, track_id, det_index in sorted(scored, reverse=True):
            if track_id in used_tracks or det_index in used_detections:
                continue
            matches[track_id] = det_index
            used_tracks.add(track_id)
            used_detections.add(det_index)
        return matches

    def _annotate_detection(self, detection: dict, track: TrackState) -> None:
        detection["target_id"] = track.track_id
        detection["track_id"] = track.numeric_id
        detection["target_label"] = track.label
        detection["track_confirmed"] = track.confirmed
        x, y, w, h = track.bbox
        detection["x"] = int(round(x))
        detection["y"] = int(round(y))
        detection["width"] = int(round(w))
        detection["height"] = int(round(h))
        detection["track_missing"] = track.missing
        detection["track_age"] = track.age

    def _prune_stale(self) -> None:
        for track_id, track in list(self._tracks.items()):
            if track.missing <= self.max_age:
                continue
            logger.info(
                "track removed id=%s class=%s age=%d missing=%d",
                track_id,
                track.class_name,
                track.age,
                track.missing,
            )
            del self._tracks[track_id]

    @staticmethod
    def _bbox(detection: dict) -> tuple[float, float, float, float]:
        return (
            float(detection.get("x", 0.0)),
            float(detection.get("y", 0.0)),
            float(detection.get("width", 0.0)),
            float(detection.get("height", 0.0)),
        )

    @staticmethod
    def _center_from_bbox(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = iw * ih
        union = max(aw * ah + bw * bh - intersection, 1e-6)
        return intersection / union

    @classmethod
    def _center_distance(cls, a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax, ay = cls._center_from_bbox(a)
        bx, by = cls._center_from_bbox(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    @staticmethod
    def _round_bbox(bbox: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        return tuple(int(round(value)) for value in bbox)
