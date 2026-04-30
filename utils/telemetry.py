"""
Telemetry helpers — simulated drone sensor data.
Replace the simulation functions with real MAVLink reads when connecting to hardware.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import HOME_POSITION


def get_telemetry_data() -> dict:
    """Return a single snapshot of simulated real-time telemetry."""
    t = time.time()
    return {
        "altitude":         round(45 + 10 * np.sin(t / 10), 1),
        "speed":            round(8 + 3 * np.sin(t / 5), 1),
        "battery":          round(max(20, 85 - (t % 3600) / 60), 1),
        "gps_lat":          HOME_POSITION["lat"] + 0.001 * np.sin(t / 30),
        "gps_lon":          HOME_POSITION["lon"] + 0.001 * np.cos(t / 30),
        "heading":          round((t * 2) % 360, 1),
        "satellites":       int(np.random.randint(8, 15)),
        "signal_strength":  int(np.random.randint(75, 100)),
        "temperature":      round(35 + 5 * np.sin(t / 60), 1),
        "flight_mode":      "AUTO",
        "armed":            True,
        "timestamp":        datetime.now(),
    }


def get_telemetry_history(minutes: int = 30) -> pd.DataFrame:
    """Generate *minutes* worth of simulated telemetry history (1 sample / 2 s)."""
    now = datetime.now()
    n_points = minutes * 30  # one sample every 2 seconds
    timestamps = [now - timedelta(seconds=i * 2) for i in range(n_points)]
    timestamps.reverse()

    rows = []
    for i, ts in enumerate(timestamps):
        t = i * 2
        rows.append({
            "timestamp":   ts,
            "altitude":    round(45 + 10 * np.sin(t / 10) + np.random.normal(0, 0.8), 1),
            "speed":       round(max(0, 8 + 3 * np.sin(t / 5) + np.random.normal(0, 0.4)), 1),
            "battery":     round(max(20, 85 - t / 60), 1),
            "temperature": round(35 + 5 * np.sin(t / 60) + np.random.normal(0, 0.3), 1),
        })

    return pd.DataFrame(rows)
