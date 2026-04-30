from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TelemetrySnapshot:
    timestamp: datetime
    altitude: float
    speed: float
    battery: float
    gps_lat: float
    gps_lon: float
    heading: float
    satellites: int
    signal_strength: int
    temperature: float
    flight_mode: str
    armed: bool


@dataclass
class DetectionRecord:
    timestamp: datetime
    cls: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    source: Optional[str] = None


class CameraInterface:
    """Abstract camera interface for RPi or webcam devices."""

    def open(self, config: CameraConfig) -> Any:
        raise NotImplementedError

    def read(self) -> Optional[Any]:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError
