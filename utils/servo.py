"""
Pan servo tracking for Raspberry Pi OS Debian/Trixie with SG90 servo.

Implements proportional pan-only control using gpiozero + lgpio.
Mirrors the tracking behaviour from the reference repository
(Object-Tracking-and-Servo-Control-with-Raspberry-Pi-using-Yolov5)
while using the modern gpiozero/lgpio stack that works on Debian Trixie.

Control loop (called once per detection frame):
  1. Compute signed pixel error: target_center_x - frame_center_x
  2. If |error| < SERVO_DEADZONE → hold position (prevents jitter when centred)
  3. step = clamp(error × SERVO_GAIN, -SERVO_MAX_STEP, SERVO_MAX_STEP)
  4. new_angle = current_angle - step   (sign flipped if SERVO_PAN_INVERT)
  5. Apply output smoothing:  actual = current + (target - current) × SMOOTHING
  6. Write smoothed angle to gpiozero AngularServo

Public API:
  init_servo()    → warm up the servo and centre it
  center_servo()  → command servo to SERVO_CENTER_ANGLE
  set_pan_angle() → absolute angle command (degrees, 0–180)
  track_target()  → call every detection frame with target x and frame width
  stop_servo()    → detach PWM (servo goes limp / saves power)
  release_servo() → full cleanup, call on app shutdown
"""

import logging
import threading

from config import (
    SERVO_CENTER_ANGLE,
    SERVO_DEADZONE,
    SERVO_ENABLED,
    SERVO_GAIN,
    SERVO_MAX_ANGLE,
    SERVO_MAX_PULSE,
    SERVO_MAX_STEP,
    SERVO_MIN_ANGLE,
    SERVO_MIN_PULSE,
    SERVO_MOVE_CONFIRM_FRAMES,
    SERVO_PAN_INVERT,
    SERVO_PIN,
    SERVO_SMOOTHING,
)

logger = logging.getLogger(__name__)

try:
    from gpiozero import AngularServo, Device
    from gpiozero.pins.lgpio import LGPIOFactory
except Exception:
    AngularServo = None
    Device = None
    LGPIOFactory = None


# ── Internal state ─────────────────────────────────────────────
_servo_lock = threading.Lock()
_pan_servo = None
_pan_angle = SERVO_CENTER_ANGLE          # current commanded angle (degrees)
_pan_angle_target = SERVO_CENTER_ANGLE   # target from proportional step
_servo_tracking = False
_frames_without_target = 0


# ── Helpers ────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_angle(angle: float) -> float:
    return _clamp(angle, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE)


def _gpiozero_angle(angle: float) -> float:
    """Convert 0–180 ° project angle to gpiozero AngularServo -90…+90 range."""
    return _clamp(angle, 0.0, 180.0) - 90.0


def _pulse_seconds(value: float) -> float:
    """Accept pulse widths as seconds (<= 1) or microseconds (> 1)."""
    if value > 1:
        return value / 1_000_000.0
    return value


_MIN_PULSE_S = _pulse_seconds(SERVO_MIN_PULSE)
_MAX_PULSE_S = _pulse_seconds(SERVO_MAX_PULSE)


# ── Servo initialisation ───────────────────────────────────────

def _create_servo(pin: int):
    """Instantiate an AngularServo on *pin* with SG90-calibrated pulse widths."""
    try:
        servo = AngularServo(
            pin,
            min_angle=-90,
            max_angle=90,
            initial_angle=_gpiozero_angle(SERVO_CENTER_ANGLE),
            min_pulse_width=_MIN_PULSE_S,
            max_pulse_width=_MAX_PULSE_S,
        )
        logger.info(
            "Pan servo ready on GPIO%d  pulse %.4f–%.4f s",
            pin, _MIN_PULSE_S, _MAX_PULSE_S,
        )
        return servo
    except Exception as exc:
        logger.error("Failed to init servo on GPIO%d: %s", pin, exc)
        return None


def init_servo():
    """Initialise (or return already-initialised) servo. Returns servo or None."""
    global _pan_servo, _pan_angle
    if not SERVO_ENABLED:
        return None
    if AngularServo is None or Device is None or LGPIOFactory is None:
        logger.warning("gpiozero/lgpio not available — servo disabled")
        return None

    with _servo_lock:
        try:
            if not isinstance(Device.pin_factory, LGPIOFactory):
                Device.pin_factory = LGPIOFactory()
        except Exception as exc:
            logger.error("lgpio pin factory error: %s", exc)
            return None

        if _pan_servo is None:
            _pan_servo = _create_servo(SERVO_PIN)
        return _pan_servo


# ── Write helpers ──────────────────────────────────────────────

def _write_angle(angle: float) -> None:
    """Send *angle* (0–180 °) to the servo hardware. Must be called inside _servo_lock."""
    if _pan_servo is not None:
        try:
            _pan_servo.angle = _gpiozero_angle(angle)
        except Exception as exc:
            logger.error("Servo write error: %s", exc)


# ── Public API ─────────────────────────────────────────────────

def center_servo() -> None:
    """Move servo to SERVO_CENTER_ANGLE and reset tracking state."""
    global _pan_angle, _pan_angle_target, _frames_without_target
    if init_servo() is None:
        return
    with _servo_lock:
        _pan_angle = _safe_angle(SERVO_CENTER_ANGLE)
        _pan_angle_target = _pan_angle
        _frames_without_target = 0
        _write_angle(_pan_angle)


def set_pan_angle(angle: float) -> float:
    """Command an absolute pan angle (degrees). Returns the angle written."""
    global _pan_angle, _pan_angle_target
    if init_servo() is None:
        return _pan_angle
    with _servo_lock:
        _pan_angle_target = _safe_angle(angle)
        _pan_angle = _pan_angle_target
        _write_angle(_pan_angle)
        return _pan_angle


def set_tilt_angle(angle: float) -> float:
    """No-op stub kept for API compatibility (pan-only build)."""
    logger.debug("set_tilt_angle() ignored — pan-only mode")
    return _pan_angle


def track_target(x_center, frame_width, y_center=None, frame_height=None):
    """Proportional pan tracking — call once per detection frame.

    Algorithm mirrors the reference repo (raspberry.py) but with proportional
    steps instead of fixed ±15° bang-bang jumps:

      pixel_error = x_center - frame_width / 2        # signed, pixels
      step        = clamp(error × GAIN, ±MAX_STEP)    # degrees
      new_angle   = current - step  (inverted if SERVO_PAN_INVERT)
      actual      = current + (new_angle - current) × SMOOTHING  # 1-tap IIR

    Args:
        x_center    : horizontal pixel coordinate of the tracked object centre.
        frame_width : width of the camera frame in pixels.
        y_center    : unused (pan-only); kept for API compatibility.
        frame_height: unused (pan-only); kept for API compatibility.
    """
    global _pan_angle, _pan_angle_target, _servo_tracking, _frames_without_target

    # ── No target ─────────────────────────────────────────────
    if frame_width is None or x_center is None:
        _frames_without_target += 1
        # Stop servo after several consecutive missed frames (prevents drift)
        if _frames_without_target > max(1, SERVO_MOVE_CONFIRM_FRAMES) * 5:
            stop_servo()
        return

    _frames_without_target = 0

    try:
        cx = float(x_center)
        fw = float(frame_width)
    except (TypeError, ValueError):
        return
    if fw <= 0.0:
        return

    # ── Signed pixel error (positive = target right of centre) ─
    pixel_error = cx - (fw / 2.0)

    # ── Deadzone — stay still when target is near centre ──────
    if abs(pixel_error) < SERVO_DEADZONE:
        return

    if init_servo() is None:
        return

    with _servo_lock:
        # Proportional step (degrees)
        raw_step = pixel_error * SERVO_GAIN
        step = _clamp(raw_step, -SERVO_MAX_STEP, SERVO_MAX_STEP)

        if SERVO_PAN_INVERT:
            step = -step

        # Target angle after proportional correction
        _pan_angle_target = _safe_angle(_pan_angle - step)

        # Single-stage output smoothing (1-tap IIR) to soften mechanical jerk
        alpha = _clamp(SERVO_SMOOTHING, 0.0, 1.0)
        _pan_angle = _safe_angle(
            _pan_angle + (_pan_angle_target - _pan_angle) * alpha
        )

        _write_angle(_pan_angle)
        _servo_tracking = True

        logger.debug(
            "track_target: err=%.1f px  step=%.2f°  target=%.1f°  actual=%.1f°",
            pixel_error, step, _pan_angle_target, _pan_angle,
        )


def stop_servo(reset_target: bool = True) -> None:
    """Detach servo PWM (servo goes limp / saves power). State is preserved."""
    global _servo_tracking, _frames_without_target
    if not SERVO_ENABLED:
        return
    with _servo_lock:
        if _pan_servo is not None:
            try:
                _pan_servo.detach()
            except Exception as exc:
                logger.error("Servo detach error: %s", exc)
        _frames_without_target = 0
        _servo_tracking = False


def release_servo() -> None:
    """Return servo to centre, detach, and release GPIO. Call on app shutdown."""
    global _pan_servo, _pan_angle, _pan_angle_target, _servo_tracking, _frames_without_target
    with _servo_lock:
        try:
            if _pan_servo is not None:
                _write_angle(_safe_angle(SERVO_CENTER_ANGLE))
                _pan_servo.detach()
                _pan_servo.close()
        except Exception as exc:
            logger.error("Servo cleanup error: %s", exc)
        _pan_servo = None
        _pan_angle = SERVO_CENTER_ANGLE
        _pan_angle_target = SERVO_CENTER_ANGLE
        _frames_without_target = 0
        _servo_tracking = False
