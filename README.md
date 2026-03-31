# 🚗 AI-Based Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![dlib](https://img.shields.io/badge/dlib-19.24%2B-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Real-time driver alertness monitoring using computer vision and facial landmark analysis — no custom training required.**

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Solution](#solution)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Controls & Parameters](#controls--parameters)
- [Sample Output](#sample-output)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## 🚨 Problem Statement

Driver fatigue and drowsiness are among the **leading causes of road accidents worldwide**. According to the NHTSA (National Highway Traffic Safety Administration), thousands of crashes each year are attributable to drowsy driving. Traditional warning systems (rumble strips, lane departure warnings) detect drowsiness too late — only after the driver *already* drifts.

This project addresses the problem at its root: **detecting physiological signs of drowsiness in real-time**, specifically eye closure and yawning, before any dangerous lane deviation occurs.

---

## ✅ Solution

We use **computer vision and facial landmark detection** to continuously monitor a driver's face through a standard webcam. The system analyses:

1. **Eye Aspect Ratio (EAR)** — Measures how open or closed the eyes are.
2. **Mouth Aspect Ratio (MAR)** — Detects yawning by measuring mouth openness.

When the EAR drops below a calibrated threshold for a sustained number of frames (indicating the eyes have been closed too long), an **audio alarm** fires immediately. All events are **logged with timestamps** to a CSV file for post-trip analysis.

No internet connection, no cloud API, and no model training is required — everything runs **locally in real-time**.

---

## ✨ Features

| Feature | Status |
|---|---|
| Real-time face detection (dlib HOG + SVM) | ✅ |
| 68-point facial landmark detection | ✅ |
| Eye Aspect Ratio (EAR) computation | ✅ |
| Sustained eye-closure drowsiness alert | ✅ |
| Audio alarm (looped until driver wakes) | ✅ |
| Blink counter (live & logged) | ✅ |
| Yawning detection via MAR | ✅ |
| CSV event logging with timestamps | ✅ |
| Live EAR / MAR overlay on video feed | ✅ |
| FPS display | ✅ |
| Synthetic alarm WAV auto-generated | ✅ |

---

## 🛠️ Tech Stack

| Tool / Library | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Video capture, image processing, drawing |
| **dlib** | HOG face detector + 68-point landmark predictor |
| **imutils** | Facial landmark index helpers, convenience functions |
| **SciPy** | Euclidean distance for EAR / MAR maths |
| **NumPy** | Array operations |
| **pygame** | Cross-platform audio alarm playback |
| **csv / datetime** | Event logging (Python stdlib) |

---

## 📁 Project Structure

```
drowsiness-detection/
│
├── main.py              ← Entry point; orchestrates the detection loop
├── detector.py          ← DrowsinessDetector class (EAR, MAR, landmarks)
├── alarm.py             ← Thread-safe audio alarm module
├── utils.py             ← Drawing helpers, CSV logger, alarm WAV generator
│
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
├── report.txt           ← Full project report (printable)
│
├── assets/
│   └── alarm.wav        ← Auto-generated on first run if missing
│
└── logs/
    └── drowsiness_log.csv  ← Session event log (auto-appended)
```

---

## ⚙️ Installation

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/drowsiness-detection.git
cd drowsiness-detection
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Windows dlib note:** dlib requires CMake and a C++ compiler.
> The easiest path on Windows is to install a pre-built wheel:
> ```
> pip install cmake
> pip install dlib
> ```
> If that fails, download the matching `.whl` from
> https://github.com/z-mahmud22/Dlib_Windows_Python3.x and install manually.

### 4 — Download the dlib shape predictor model

The 68-point shape predictor file is **not** included in this repo (it is ~100 MB).

1. Download: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
2. Extract the `.dat` file using 7-Zip or `bunzip2`.
3. Place `shape_predictor_68_face_landmarks.dat` in the **project root** folder.

---

## ▶️ How to Run

```bash
python main.py
```

The system will:
1. Load the facial landmark predictor.
2. Auto-generate `assets/alarm.wav` if it is missing.
3. Open your default webcam.
4. Begin real-time detection with an annotated video window.

Press **`q`** to quit. A session summary is printed to the console.

---

## 🎛️ Controls & Parameters

You can customise the detection sensitivity via command-line flags:

| Flag | Default | Description |
|---|---|---|
| `--predictor` | `shape_predictor_68_face_landmarks.dat` | Path to the .dat model |
| `--cam` | `0` | Webcam device index |
| `--ear-thresh` | `0.25` | EAR below which eye is "closed" |
| `--ear-frames` | `20` | Consecutive closed-eye frames before alert |
| `--mar-thresh` | `0.60` | MAR above which a yawn is detected |

**Example — use a stricter threshold on a secondary camera:**
```bash
python main.py --cam 1 --ear-thresh 0.22 --ear-frames 15
```

---

## 🖥️ Sample Output

**On-screen overlay (while running):**
```
┌─────────────────────────────────────────────────┐
│ EAR  : 0.287   ← green (eyes open)             │
│ MAR  : 0.105                                    │
│ Blinks : 14                                     │
│                                                 │
│        [DROWSINESS ALERT!]  ← red banner        │
└─────────────────────────────────────────────────┘
```

**Console output (example):**
```
[INFO] Loading facial landmark predictor …
[INFO] Predictor loaded successfully.
[INFO] Opening camera (index 0) …
[INFO] System running. Press 'q' to quit.

[INFO] 🥱 Yawn detected! MAR=0.643  Total yawns: 1
[ALERT] 😴 Drowsiness detected! EAR=0.189
[INFO] Driver is awake again.

[SESSION SUMMARY]
  Duration  : 142.3 seconds
  Frames    : 3987
  Avg FPS   : 28.0
  Blinks    : 22
  Yawns     : 3
  Log saved : C:\...\logs\drowsiness_log.csv
```

**CSV log (excerpt):**
```csv
Timestamp,Event,EAR,MAR,Blink_Count
2026-03-31 20:15:02,BLINK,0.2102,0.1033,1
2026-03-31 20:15:44,YAWN_DETECTED,0.2981,0.6512,8
2026-03-31 20:16:11,DROWSINESS_ALERT,0.1874,0.1112,14
```

---

## 🔮 Future Improvements

| Idea | Benefit |
|---|---|
| Head-pose estimation (roll / pitch / yaw) | Detect nodding off even when eyes partially open |
| PERCLOS metric | More robust than simple EAR; industry standard |
| CNN-based eye state classifier | Higher accuracy under poor lighting |
| Infrared camera support | Works in the dark |
| Mobile app (Kivy / Flutter + OpenCV) | Smartphone deployment |
| Vehicle OBD-II integration | Cross-reference with steering patterns |
| Cloud dashboard | Fleet management and remote monitoring |
| Adaptive EAR calibration per user | Accounts for individual eye geometry |

---

## 📚 References

1. Soukupová, T., & Čech, J. (2016). **Real-Time Eye Blink Detection using Facial Landmarks.** *21st Computer Vision Winter Workshop.*
2. Viola, P., & Jones, M. (2001). **Rapid Object Detection using a Boosted Cascade of Simple Features.** *CVPR.*
3. King, D. E. (2009). **Dlib-ml: A Machine Learning Toolkit.** *Journal of Machine Learning Research.*
4. OpenCV Documentation — https://docs.opencv.org/
5. NHTSA Drowsy Driving Statistics — https://www.nhtsa.gov/risky-driving/drowsy-driving

---

## 📄 License

This project is released under the **MIT License**. Feel free to use, modify, and distribute.

---

*Built as a university BYOP Computer Vision project. Stay safe on the road! 🚗💤*
