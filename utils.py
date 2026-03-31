"""
utils.py
--------
Drawing helpers, overlay banners, CSV event logging,
and synthetic alarm WAV generation.

Updated to work with MediaPipe-based detector output:
  • face_rect is now a plain (x1, y1, x2, y2) tuple
  • eye and mouth contour arrays are plain numpy int32 arrays
"""

import cv2
import csv
import os
import struct
import wave
import math
import numpy as np
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────

def draw_eye_contours(frame, left_eye_draw, right_eye_draw,
                      color=(0, 255, 0)):
    """
    Draw convex-hull outlines around both eyes.

    Parameters
    ----------
    left_eye_draw  : ndarray (N, 2) — contour landmark pixels
    right_eye_draw : ndarray (N, 2)
    color          : BGR tuple
    """
    left_hull  = cv2.convexHull(left_eye_draw)
    right_hull = cv2.convexHull(right_eye_draw)
    cv2.drawContours(frame, [left_hull],  -1, color, 1)
    cv2.drawContours(frame, [right_hull], -1, color, 1)


def draw_mouth_contour(frame, mouth_draw, color=(0, 165, 255)):
    """
    Draw a convex-hull outline around the mouth.

    Parameters
    ----------
    mouth_draw : ndarray (N, 2)
    color      : BGR tuple
    """
    mouth_hull = cv2.convexHull(mouth_draw)
    cv2.drawContours(frame, [mouth_hull], -1, color, 1)


def draw_face_box(frame, face_rect, color=(255, 220, 0)):
    """
    Draw a rectangle around the detected face.

    Parameters
    ----------
    face_rect : (x1, y1, x2, y2) tuple of ints
    """
    x1, y1, x2, y2 = face_rect
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_status_banner(frame, ear, mar, blink_count,
                       is_drowsy, is_yawning):
    """
    Render a semi-transparent HUD banner at the top of *frame*.

    Parameters
    ----------
    frame       : BGR image (modified in-place)
    ear         : float
    mar         : float
    blink_count : int
    is_drowsy   : bool
    is_yawning  : bool
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Left column ────────────────────────────────────────────
    ear_color = (50, 220, 50) if ear > 0.23 else (50, 50, 255)
    cv2.putText(frame, f"EAR    : {ear:.3f}",
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68, ear_color, 2)

    mar_color = (0, 165, 255) if is_yawning else (180, 180, 180)
    cv2.putText(frame, f"MAR    : {mar:.3f}",
                (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.68, mar_color, 2)

    cv2.putText(frame, f"Blinks : {blink_count}",
                (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (180, 180, 180), 2)

    # ── Centre alerts ────────────────────────────────────────
    cx = w // 2
    if is_drowsy:
        _alert_label(frame, "! DROWSINESS ALERT !", (cx - 165, 40), (30, 30, 230))
    if is_yawning:
        _alert_label(frame, "YAWNING DETECTED",     (cx - 130, 80), (0, 140, 255))


def _alert_label(frame, text, origin, color):
    """Draw a bold, shadowed alert label."""
    # Drop shadow
    cv2.putText(frame, text,
                (origin[0] + 2, origin[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4)
    # Foreground
    cv2.putText(frame, text, origin,
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)


def draw_no_face_warning(frame):
    """Overlay a gentle warning when no face is detected."""
    h, w = frame.shape[:2]
    msg  = "No face detected — look directly at the camera"
    cv2.putText(frame, msg,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1)


# ─────────────────────────────────────────────────────────────────
# CSV event logging
# ─────────────────────────────────────────────────────────────────

_LOG_PATH    = os.path.join("logs", "drowsiness_log.csv")
_LOG_HEADERS = ["Timestamp", "Event", "EAR", "MAR", "Blink_Count"]


def init_log():
    """Create the CSV file with headers if it does not already exist."""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(_LOG_PATH):
        with open(_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_LOG_HEADERS).writeheader()
        print(f"[LOG] Created log file: {_LOG_PATH}")


def log_event(event: str, ear: float, mar: float, blink_count: int):
    """
    Append one row to the CSV log.

    Parameters
    ----------
    event       : str   e.g. "DROWSINESS_ALERT", "YAWN_DETECTED", "BLINK"
    ear         : float
    mar         : float
    blink_count : int
    """
    row = {
        "Timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event"       : event,
        "EAR"         : f"{ear:.4f}",
        "MAR"         : f"{mar:.4f}",
        "Blink_Count" : blink_count,
    }
    with open(_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=_LOG_HEADERS).writerow(row)


# ─────────────────────────────────────────────────────────────────
# Synthetic alarm WAV generator
# ─────────────────────────────────────────────────────────────────

def generate_alarm_wav(path: str, duration_sec: float = 2.5,
                       frequency: float = 880.0, sample_rate: int = 44100):
    """
    Generate a sine-wave alarm WAV so the project works out-of-the-box.

    No-op if the file already exists.

    Parameters
    ----------
    path         : str   destination filepath
    duration_sec : float tone length in seconds
    frequency    : float Hz — 880 Hz (A5) is loud and alerting
    sample_rate  : int   audio sample rate
    """
    if os.path.exists(path):
        return

    dir_part = os.path.dirname(path)
    if dir_part:
        os.makedirs(dir_part, exist_ok=True)

    num_samples = int(sample_rate * duration_sec)
    fade        = int(sample_rate * 0.05)         # 50 ms fade
    samples     = []

    for i in range(num_samples):
        t        = i / sample_rate
        env      = 1.0
        if   i < fade:
            env = i / fade
        elif i > num_samples - fade:
            env = (num_samples - i) / fade
        sample = env * 0.8 * math.sin(2 * math.pi * frequency * t)
        samples.append(int(sample * 32767))

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{num_samples}h", *samples))

    print(f"[UTILS] Generated alarm WAV → {path}")
