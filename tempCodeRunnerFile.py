"""
main.py
-------
Entry point for the AI-Based Driver Drowsiness Detection System.

Run with:
    python main.py

Optional flags:
    --predictor   Path to dlib's shape predictor file
                  (default: shape_predictor_68_face_landmarks.dat)
    --cam         Webcam device index (default: 0)
    --ear-thresh  EAR threshold below which eye is considered closed (default: 0.25)
    --ear-frames  Number of consecutive frames EAR must be below threshold
                  before drowsiness alert fires (default: 20)
    --mar-thresh  MAR threshold above which a yawn is detected (default: 0.6)

Press 'q' to quit the application.
"""

import argparse
import sys
import os
import cv2
import time

# ── Local modules ────────────────────────────────────────────────────────────
from detector import DrowsinessDetector
import alarm
import utils


# ─────────────────────────────────────────────
# CLI argument parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Based Driver Drowsiness Detection System"
    )
    parser.add_argument(
        "--predictor", type=str,
        default="shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point shape predictor .dat file",
    )
    parser.add_argument(
        "--cam", type=int, default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--ear-thresh", type=float, default=0.25,
        help="EAR threshold for closed-eye detection (default: 0.25)",
    )
    parser.add_argument(
        "--ear-frames", type=int, default=20,
        help="Consecutive frames below EAR threshold before alert (default: 20)",
    )
    parser.add_argument(
        "--mar-thresh", type=float, default=0.60,
        help="MAR threshold for yawn detection (default: 0.60)",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────

def check_predictor(predictor_path: str):
    """
    Verify the dlib shape predictor model exists.
    Print a download instruction and exit gracefully if it is missing.
    """
    if not os.path.exists(predictor_path):
        print("\n" + "=" * 65)
        print("  ERROR: Shape predictor model file not found!")
        print(f"  Expected path: {predictor_path}")
        print()
        print("  Download it here:")
        print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print()
        print("  Then extract the .dat file and place it in the project root.")
        print("=" * 65 + "\n")
        sys.exit(1)


# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Pre-flight checks ──────────────────────
    check_predictor(args.predictor)

    # ── Ensure alarm WAV exists ────────────────
    alarm_path = os.path.join("assets", "alarm.wav")
    utils.generate_alarm_wav(alarm_path)          # no-op if already present
    alarm.set_alarm_file(alarm_path)

    # ── Initialise event log ───────────────────
    utils.init_log()

    # ── Load detector ──────────────────────────
    print("[INFO] Loading facial landmark predictor …")
    detector = DrowsinessDetector(predictor_path=args.predictor)
    print("[INFO] Predictor loaded successfully.")

    # ── Open webcam ───────────────────────────
    print(f"[INFO] Opening camera (index {args.cam}) …")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check --cam index.")
        sys.exit(1)

    # ── Session state ─────────────────────────
    ear_counter   = 0          # frames where eye is considered closed
    blink_counter = 0          # total blinks this session
    yawn_counter  = 0          # total yawns this session
    eye_was_closed = False     # tracks the falling edge of an eye closure (blink)

    is_drowsy     = False
    is_yawning    = False

    start_time    = time.time()
    frame_count   = 0

    print("\n[INFO] System running. Press 'q' to quit.\n")

    # ── Main loop ──────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        frame_count += 1

        # Flip horizontally so it acts like a mirror (more natural UX)
        frame = cv2.flip(frame, 1)

        # ── Run detection ──────────────────────
        results = detector.process_frame(frame)

        if not results:
            # No face detected — show warning, stop alarm
            utils.draw_no_face_warning(frame)
            if is_drowsy:
                alarm.stop_alarm()
                is_drowsy = False
        else:
            # Use the first (closest) detected face
            data      = results[0]
            ear       = data["ear"]
            mar       = data["mar"]
            left_eye  = data["left_eye"]
            right_eye = data["right_eye"]
            mouth     = data["mouth"]
            face_rect = data["face_rect"]

            # ── Draw face & features ──────────
            utils.draw_face_box(frame, face_rect)
            utils.draw_eye_contours(frame, left_eye, right_eye)
            utils.draw_mouth_contour(frame, mouth)

            # ── EAR-based drowsiness logic ────
            if ear < args.ear_thresh:
                ear_counter += 1

                # Track blink (brief closure < ear_frames)
                eye_was_closed = True

                # Trigger alert after sustained closure
                if ear_counter >= args.ear_frames:
                    if not is_drowsy:
                        is_drowsy = True
                        alarm.start_alarm()
                        utils.log_event("DROWSINESS_ALERT", ear, mar, blink_counter)
                        print(f"[ALERT] 😴 Drowsiness detected! EAR={ear:.3f}")
            else:
                # Eye just opened
                if eye_was_closed:
                    if ear_counter < args.ear_frames:
                        # Short closure → it was a blink
                        blink_counter += 1
                        utils.log_event("BLINK", ear, mar, blink_counter)
                    eye_was_closed = False

                ear_counter = 0

                if is_drowsy:
                    is_drowsy = False
                    alarm.stop_alarm()
                    print("[INFO] Driver is awake again.")

            # ── MAR-based yawn detection ──────
            if mar > args.mar_thresh:
                if not is_yawning:
                    is_yawning    = True
                    yawn_counter += 1
                    utils.log_event("YAWN_DETECTED", ear, mar, blink_counter)
                    print(f"[INFO] 🥱 Yawn detected! MAR={mar:.3f}  Total yawns: {yawn_counter}")
            else:
                is_yawning = False

            # ── Draw HUD ──────────────────────
            utils.draw_status_banner(frame, ear, mar, blink_counter,
                                     is_drowsy, is_yawning)

        # ── FPS counter (bottom-right) ─────────
        elapsed   = time.time() - start_time
        fps       = frame_count / elapsed if elapsed > 0 else 0
        h, w      = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # ── Yawn counter (bottom-left) ─────────
        cv2.putText(frame, f"Yawns: {yawn_counter}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # ── Display ───────────────────────────
        cv2.imshow("Driver Drowsiness Detection  [Press Q to quit]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[INFO] Quit key pressed. Shutting down …")
            break

    # ── Cleanup ────────────────────────────────
    alarm.stop_alarm()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n[SESSION SUMMARY]")
    print(f"  Duration  : {elapsed:.1f} seconds")
    print(f"  Frames    : {frame_count}")
    print(f"  Avg FPS   : {frame_count / elapsed:.1f}")
    print(f"  Blinks    : {blink_counter}")
    print(f"  Yawns     : {yawn_counter}")
    print(f"  Log saved : {os.path.abspath(os.path.join('logs', 'drowsiness_log.csv'))}")
    print("\n[INFO] Goodbye! Stay alert on the road. 🚗")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    main()
