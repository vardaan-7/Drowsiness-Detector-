"""
main.py
-------
Entry point — AI-Based Driver Drowsiness Detection System
Powered by MediaPipe FaceMesh (no external model download needed).

Run:
    python main.py

Optional flags:
    --cam         Webcam index           (default: 0)
    --ear-thresh  EAR closed threshold   (default: 0.23)
    --ear-frames  Consecutive frames     (default: 20)
    --mar-thresh  MAR yawn threshold     (default: 0.50)

Press 'q' to quit.
"""

import argparse
import os
import sys
import time
import cv2

from detector import DrowsinessDetector
import alarm
import utils


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Driver Drowsiness Detection (MediaPipe)"
    )
    parser.add_argument("--cam",        type=int,   default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--ear-thresh", type=float, default=0.23,
                        help="EAR below which eye is 'closed' (default: 0.23)")
    parser.add_argument("--ear-frames", type=int,   default=20,
                        help="Consecutive closed frames before alert (default: 20)")
    parser.add_argument("--mar-thresh", type=float, default=0.50,
                        help="MAR above which a yawn is detected (default: 0.50)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Alarm setup ────────────────────────────────────────────
    alarm_path = os.path.join("assets", "alarm.wav")
    utils.generate_alarm_wav(alarm_path)
    alarm.set_alarm_file(alarm_path)

    # ── CSV log ────────────────────────────────────────────────
    utils.init_log()

    # ── Detector (MediaPipe — no .dat file needed) ────────────
    print("[INFO] Loading MediaPipe FaceMesh …")
    try:
        detector = DrowsinessDetector()
    except Exception as exc:
        print(f"[ERROR] Failed to initialise detector: {exc}")
        sys.exit(1)
    print("[INFO] FaceMesh loaded successfully.")

    # ── Webcam ─────────────────────────────────────────────────
    print(f"[INFO] Opening webcam (index {args.cam}) …")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {args.cam}.")
        print("        Try --cam 1  or check your camera connection.")
        sys.exit(1)

    # Warm up the camera — the first few frames are sometimes black
    for _ in range(5):
        cap.read()

    # ── Session state ──────────────────────────────────────────
    ear_counter    = 0       # consecutive frames where EAR < threshold
    blink_counter  = 0       # total blinks detected
    yawn_counter   = 0       # total yawns detected
    eye_was_closed = False   # tracks rising/falling edge (for blink detection)

    is_drowsy      = False
    is_yawning     = False

    start_time     = time.time()
    frame_count    = 0

    print("\n[INFO] System running — press 'q' to quit.\n")
    print(f"  EAR threshold : {args.ear_thresh}  (frames: {args.ear_frames})")
    print(f"  MAR threshold : {args.mar_thresh}")
    print()

    # ── Main loop ──────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Missed a frame — retrying …")
            continue

        frame_count += 1

        # Mirror for natural UX
        frame = cv2.flip(frame, 1)

        # ── Detection ─────────────────────────────────────────
        detections = detector.process_frame(frame)

        if not detections:
            # ── No face ───────────────────────────────────────
            utils.draw_no_face_warning(frame)
            if is_drowsy:
                alarm.stop_alarm()
                is_drowsy = False
            ear_counter    = 0
            eye_was_closed = False

        else:
            # ── Face found — use first (closest) result ────────
            data = detections[0]
            ear  = data["ear"]
            mar  = data["mar"]

            # Draw face landmarks
            utils.draw_face_box(frame, data["face_rect"])
            utils.draw_eye_contours(frame,
                                    data["left_eye_draw"],
                                    data["right_eye_draw"])
            utils.draw_mouth_contour(frame, data["mouth"])

            # ── EAR  drowsiness logic ─────────────────────────
            if ear < args.ear_thresh:
                ear_counter   += 1
                eye_was_closed = True

                if ear_counter >= args.ear_frames:
                    # Sustained closure → drowsy
                    if not is_drowsy:
                        is_drowsy = True
                        alarm.start_alarm()
                        utils.log_event("DROWSINESS_ALERT", ear, mar,
                                        blink_counter)
                        print(f"[ALERT] 😴 Drowsiness! EAR={ear:.3f}")
            else:
                # Eye just opened back up
                if eye_was_closed:
                    if ear_counter < args.ear_frames:
                        # Brief closure = a blink
                        blink_counter += 1
                        utils.log_event("BLINK", ear, mar, blink_counter)
                    eye_was_closed = False

                ear_counter = 0

                if is_drowsy:
                    is_drowsy = False
                    alarm.stop_alarm()
                    print("[INFO] Driver awake again.")

            # ── MAR  yawn logic ───────────────────────────────
            if mar > args.mar_thresh:
                if not is_yawning:
                    is_yawning   = True
                    yawn_counter += 1
                    utils.log_event("YAWN_DETECTED", ear, mar, blink_counter)
                    print(f"[INFO] 🥱 Yawn! MAR={mar:.3f}  total={yawn_counter}")
            else:
                is_yawning = False

            # ── HUD overlay ───────────────────────────────────
            utils.draw_status_banner(frame, ear, mar, blink_counter,
                                     is_drowsy, is_yawning)

        # ── FPS & yawn count (corners) ─────────────────────────
        elapsed = time.time() - start_time
        fps     = frame_count / elapsed if elapsed > 0 else 0
        h, w    = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps:.1f}", (w - 95, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)
        cv2.putText(frame, f"Yawns:{yawn_counter}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 120, 120), 1)

        # ── Display ───────────────────────────────────────────
        cv2.imshow("Driver Drowsiness Detection  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quitting …")
            break

    # ── Cleanup ────────────────────────────────────────────────
    alarm.stop_alarm()
    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    log_abs = os.path.abspath(os.path.join("logs", "drowsiness_log.csv"))
    print(f"""
╔══════════════════════════════════════╗
║         SESSION SUMMARY             ║
╟──────────────────────────────────────╢
║  Duration  : {elapsed:>6.1f} seconds          ║
║  Frames    : {frame_count:>6}                ║
║  Avg FPS   : {frame_count/elapsed if elapsed>0 else 0:>6.1f}                ║
║  Blinks    : {blink_counter:>6}                ║
║  Yawns     : {yawn_counter:>6}                ║
╚══════════════════════════════════════╝
Log: {log_abs}

Stay alert on the road! 🚗
""")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
