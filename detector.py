"""
detector.py
-----------
Core detection module — now powered by MediaPipe FaceMesh.

No external model file download needed.
Install once with:  pip install mediapipe

MediaPipe FaceMesh gives 468 facial landmarks per face.
We use specific indices for the eyes and mouth:

  Right eye EAR : [33,  159, 158, 133, 153, 145]
  Left  eye EAR : [263, 386, 385, 362, 380, 373]
  Mouth  MAR    : corners=61,291  top=13  bottom=14
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist

# ─────────────────────────────────────────────────────────────────
# MediaPipe landmark index constants
# ─────────────────────────────────────────────────────────────────

# 6 points per eye used for EAR  (P1 outer-corner → P4 inner-corner)
RIGHT_EYE_EAR = [33,  159, 158, 133, 153, 145]
LEFT_EYE_EAR  = [263, 386, 385, 362, 380, 373]

# Richer contours used only for drawing (more landmark points = smoother hull)
RIGHT_EYE_DRAW = [33,  246, 161, 160, 159, 158, 157, 173,
                  133, 155, 154, 153, 145, 144, 163,   7]
LEFT_EYE_DRAW  = [263, 466, 388, 387, 386, 385, 384, 398,
                  362, 382, 381, 380, 374, 373, 390, 249]

# Mouth: 4-point MAR
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

# Mouth outline for drawing
MOUTH_DRAW = [61,  185, 40,  39,  37,  0,  267, 269,
              270, 409, 291, 375, 321, 405, 314,  17,
               84, 181,  91, 146]


# ─────────────────────────────────────────────────────────────────
# Pure-math helpers
# ─────────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_pts):
    """
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    eye_pts : array-like of 6 (x, y) pixel coordinates
    """
    A = dist.euclidean(eye_pts[1], eye_pts[5])
    B = dist.euclidean(eye_pts[2], eye_pts[4])
    C = dist.euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(top, bottom, left, right):
    """
    MAR = vertical_openness / mouth_width
    When mouth is closed → ~0.0–0.3
    When yawning        → >0.5
    """
    vertical   = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)
    return vertical / (horizontal + 1e-6)


# ─────────────────────────────────────────────────────────────────
# Detector class
# ─────────────────────────────────────────────────────────────────

class DrowsinessDetector:
    """
    Wraps MediaPipe FaceMesh.  No model file download required.

    Usage
    -----
    detector = DrowsinessDetector()
    results  = detector.process_frame(frame)   # list of dicts
    """

    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,           # enables iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ──────────────────────────────────────────────────────────────
    def process_frame(self, frame):
        """
        Detect faces in *frame* and compute EAR + MAR.

        Parameters
        ----------
        frame : BGR numpy array from cv2.VideoCapture

        Returns
        -------
        list of dict — one dict per detected face:
          {
            "ear"            : float,           # average EAR
            "mar"            : float,           # mouth aspect ratio
            "left_eye"       : ndarray (6, 2),  # EAR landmark pixels
            "right_eye"      : ndarray (6, 2),
            "left_eye_draw"  : ndarray (16, 2), # contour landmark pixels
            "right_eye_draw" : ndarray (16, 2),
            "mouth"          : ndarray (20, 2), # mouth contour pixels
            "face_rect"      : (x1, y1, x2, y2),
            "landmarks"      : ndarray (468, 2),# all landmarks (pixel)
          }
        """
        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_out = self.face_mesh.process(rgb)

        results = []
        if not mp_out.multi_face_landmarks:
            return results                       # no face → empty list

        for face_lms in mp_out.multi_face_landmarks:
            # ── Convert normalised coords to pixel coords ──────────
            lm = np.array(
                [(int(p.x * w), int(p.y * h)) for p in face_lms.landmark],
                dtype=np.int32,
            )                                    # shape (468+, 2)

            # ── EAR ───────────────────────────────────────────────
            right_eye = lm[RIGHT_EYE_EAR]        # shape (6, 2)
            left_eye  = lm[LEFT_EYE_EAR]

            left_ear  = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear   = (left_ear + right_ear) / 2.0

            # ── MAR ───────────────────────────────────────────────
            mar = mouth_aspect_ratio(
                top    = lm[MOUTH_TOP],
                bottom = lm[MOUTH_BOTTOM],
                left   = lm[MOUTH_LEFT],
                right  = lm[MOUTH_RIGHT],
            )

            # ── Drawing contours ──────────────────────────────────
            left_eye_draw  = lm[LEFT_EYE_DRAW]   # shape (16, 2)
            right_eye_draw = lm[RIGHT_EYE_DRAW]
            mouth_draw     = lm[MOUTH_DRAW]       # shape (20, 2)

            # ── Face bounding box ─────────────────────────────────
            xs = lm[:, 0]
            ys = lm[:, 1]
            face_rect = (
                max(0, int(xs.min()) - 10),
                max(0, int(ys.min()) - 10),
                min(w, int(xs.max()) + 10),
                min(h, int(ys.max()) + 10),
            )

            results.append({
                "ear"            : avg_ear,
                "mar"            : mar,
                "left_eye"       : left_eye,
                "right_eye"      : right_eye,
                "left_eye_draw"  : left_eye_draw,
                "right_eye_draw" : right_eye_draw,
                "mouth"          : mouth_draw,
                "face_rect"      : face_rect,
                "landmarks"      : lm,
            })

        return results

    # ──────────────────────────────────────────────────────────────
    def __del__(self):
        """Release MediaPipe resources on garbage collection."""
        try:
            self.face_mesh.close()
        except Exception:
            pass
