"""
alarm.py
--------
Handles audio alarm functionality for the Drowsiness Detection System.
Plays an alarm sound in a background thread so the main video loop is
never blocked.

Primary  : pygame (cross-platform, works even without system audio drivers).
Fallback : winsound (Windows only) → playsound → print warning.
"""

import threading
import os

# ─────────────────────────────────────────────
# Alarm state
# ─────────────────────────────────────────────
_alarm_playing  = False
_alarm_thread   = None
_alarm_wav_path = None


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _play_loop(wav_path: str):
    """
    Internal function that loops the alarm sound until stopped.
    Runs inside a daemon thread.
    """
    global _alarm_playing

    # ── Try pygame ─────────────────────────────
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.play(-1)          # -1 → loop indefinitely
        while _alarm_playing:
            pygame.time.wait(100)
        pygame.mixer.music.stop()
        return
    except Exception:
        pass

    # ── Try winsound (Windows only) ────────────
    try:
        import winsound
        while _alarm_playing:
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            import time; time.sleep(1)
        return
    except Exception:
        pass

    # ── Try playsound ──────────────────────────
    try:
        from playsound import playsound
        while _alarm_playing:
            playsound(wav_path, block=True)
        return
    except Exception:
        pass

    # ── Final fallback ─────────────────────────
    print("[ALARM] ⚠️  Could not play audio — no compatible audio library found.")
    print("[ALARM]     Install pygame:  pip install pygame")


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def set_alarm_file(wav_path: str):
    """
    Register the path to the alarm WAV file.

    Parameters
    ----------
    wav_path : str
        Absolute or relative path to the alarm .wav file.
    """
    global _alarm_wav_path
    if not os.path.exists(wav_path):
        print(f"[ALARM] Warning: alarm file not found at '{wav_path}'")
    _alarm_wav_path = wav_path


def start_alarm():
    """
    Start playing the alarm sound in a background thread.
    Safe to call multiple times — will not start a second instance.
    """
    global _alarm_playing, _alarm_thread

    if _alarm_playing:
        return  # already running

    if _alarm_wav_path is None:
        print("[ALARM] No alarm file set. Call set_alarm_file() first.")
        return

    _alarm_playing = True
    _alarm_thread  = threading.Thread(
        target=_play_loop,
        args=(_alarm_wav_path,),
        daemon=True,          # killed automatically when the main program exits
    )
    _alarm_thread.start()
    print("[ALARM] 🔔 Alarm started.")


def stop_alarm():
    """
    Stop the alarm sound gracefully.
    """
    global _alarm_playing, _alarm_thread

    if not _alarm_playing:
        return  # nothing to stop

    _alarm_playing = False

    # Give the thread a moment to clean up
    if _alarm_thread is not None:
        _alarm_thread.join(timeout=2)
        _alarm_thread = None

    # Also stop pygame directly in case the thread has already exited
    try:
        import pygame
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception:
        pass

    print("[ALARM] 🔕 Alarm stopped.")


def is_alarm_playing() -> bool:
    """Return True if the alarm is currently active."""
    return _alarm_playing
