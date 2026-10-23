# 🚀 AI Bootcamp — Interactive Computer Vision & Voice AI Demos

A showcase of real-time Computer Vision and Generative Voice AI applications built with **Python 3.11**, **OpenCV**, **MediaPipe**, **YOLOv8**, and **Groq Cloud AI**.

Designed specifically for live bootcamp workshops, classroom demonstrations, and stage volunteer interactions.

---

## 📋 Table of Contents

- [Overview of Demos](#-overview-of-demos)
- [Project Structure](#-project-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [API Configuration (.env)](#-api-configuration-env)
- [Detailed Demo Walkthroughs](#-detailed-demo-walkthroughs)
  - [1. Opening Hook: Live YOLO Object Detection](#1-opening-hook-live-yolo-object-detection)
  - [2. Gesture Magic: Air Canvas & Finger Counter](#2-gesture-magic-air-canvas--finger-counter)
  - [3. Computer Vision Safety: Driver Drowsiness & Fatigue Detector](#3-computer-vision-safety-driver-drowsiness--fatigue-detector)
  - [4. Regional Voice AI: दादी साथी (Dadi Sathi Voice Assistant)](#4-regional-voice-ai-दादी-साथी-dadi-sathi-voice-assistant)
- [Presenter Cheat Sheet & Hotkeys](#-presenter-cheat-sheet--hotkeys)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Overview of Demos

| Script | Technology Stack | Core Capabilities |
|---|---|---|
| [`demo1_yolo_hook.py`](demo1_yolo_hook.py) | **YOLOv8n**, OpenCV | Real-time object and person detection, live bounding boxes, dynamic FPS & count tracker. |
| [`demo2_gesture_air_draw.py`](demo2_gesture_air_draw.py) | **MediaPipe Hands**, OpenCV | Air drawing with index finger, anti-aliased smoothing, on-screen interactive toolbar, eraser mode, 5-finger palm clear. |
| [`demo2_drowsiness_fatigue.py`](demo2_drowsiness_fatigue.py) | **MediaPipe Face Mesh (478 pts)**, OpenCV | Eye Aspect Ratio (EAR) blink/sleep detector, Mouth Aspect Ratio (MAR) yawn detector, iris gaze tracking, live fatigue gauge. |
| [`dadi_assistant.py`](dadi_assistant.py) | **Groq API**, SpeechRecognition, gTTS, Pygame, Tkinter | Bhojpuri/Hindi voice assistant for grandparents; voice medicine reminders, phone dialing, message reader with sub-second AI latency. |

---

## 📂 Project Structure

```text
school demo/
├── .env.example                 # Template for API keys
├── .gitignore                   # Excludes venv, .env, and large weights
├── requirements.txt             # Computer vision dependencies
├── voice_requirements.txt       # Voice assistant & LLM dependencies
├── README.md                    # Project documentation
│
├── demo1_yolo_hook.py           # Demo 1: YOLOv8 real-time object detection
├── demo2_gesture_air_draw.py    # Demo 2A: Air drawing & hand gestures
├── demo2_drowsiness_fatigue.py  # Demo 2B: Driver drowsiness & face mesh
│
├── dadi_assistant.py            # Demo 3: Voice Assistant GUI application
└── logic.py                     # Helper logic for Hindi time parsing & offline fallback
```

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
* **Operating System**: Windows 10/11, macOS, or Linux
* **Python Version**: **Python 3.11** (Recommended for MediaPipe and PyTorch binary compatibility)
* **Hardware**: Standard webcam and microphone

### 2. Setup Virtual Environment

Open your terminal or PowerShell in the project directory:

```powershell
# 1. Create a Python 3.11 virtual environment
py -3.11 -m venv venv

# 2. Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install Computer Vision dependencies
pip install -r requirements.txt

# 5. Install Voice Assistant dependencies
pip install -r voice_requirements.txt
```

---

## 🔑 API Configuration (.env)

The **Dadi Sathi** voice assistant uses **Groq Cloud** for sub-second AI reasoning. It can also operate **completely offline** without an internet connection or API key.

1. Create a `.env` file in the project root (you can copy `.env.example`):
   ```powershell
   cp .env.example .env
   ```

2. Add your Groq API key:
   ```env
   # Get your free key at https://console.groq.com/keys
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=qwen/qwen3.8-27b
   ```

> [!NOTE]
> `.env` is automatically ignored by Git so your secret keys will never be leaked when pushing to GitHub.

---

## 🎮 Detailed Demo Walkthroughs

### 1. Opening Hook: Live YOLO Object Detection
**File:** [`demo1_yolo_hook.py`](demo1_yolo_hook.py)  
**Run:**
```powershell
python demo1_yolo_hook.py
```

* **Purpose**: Captivates the audience during the first 60 seconds of a presentation.
* **How it works**: Uses the lightweight `yolov8n.pt` (~6 MB) neural network to detect people, laptops, phones, backpacks, cups, and bottles in real time.
* **Presenter Features**:
  * Mirrored webcam display for intuitive interaction.
  * Live object count and real-time FPS counter.
  * Automatic download of the YOLO model on first run.
* **Controls**:
  * `F` — Toggle Fullscreen.
  * `Q` — Quit.

---

### 2. Gesture Magic: Air Canvas & Finger Counter
**File:** [`demo2_gesture_air_draw.py`](demo2_gesture_air_draw.py)  
**Run:**
```powershell
python demo2_gesture_air_draw.py
```

* **Purpose**: An interactive experience where a volunteer from the audience draws in the air with their fingers.
* **How it works**: Tracks 21 hand landmarks using MediaPipe Hands, applying an Exponential Moving Average (EMA) filter to remove hand jitter.
* **Modes & Gestures**:
  * **Draw (1 Finger - Index Up)**: Draws smooth, vivid strokes on the air canvas.
  * **Tool Selection & Hover (2 Fingers - Peace Sign)**: Moves cursor without drawing. Hover over the top bar to switch colors.
  * **On-Screen Interactive Palette**: Red, Green, Cyan, Yellow, Purple, Eraser, and Clear All.
  * **Eraser Mode**: Erases specific lines without clearing the canvas.
  * **Hold to Clear (Open Palm - 5 Fingers)**: Displays an animated circular countdown ring over the palm (~0.9s) before clearing, preventing accidental wipes.
* **Controls**:
  * `1` to `5` — Quick color select.
  * `E` — Select Eraser tool.
  * `C` — Instant Canvas Clear.
  * `F` — Toggle Fullscreen.
  * `Q` — Quit.

---

### 3. Computer Vision Safety: Driver Drowsiness & Fatigue Detector
**File:** [`demo2_drowsiness_fatigue.py`](demo2_drowsiness_fatigue.py)  
**Run:**
```powershell
python demo2_drowsiness_fatigue.py
```

* **Purpose**: Demonstrates computer vision applied to driver safety and biometric monitoring.
* **How it works**: Analyzes 478 3D facial landmarks from MediaPipe Face Mesh with refined Iris tracking:
  * **Eye Aspect Ratio (EAR)**: Computes eye openness. Accurately counts natural blinks. If eyes stay closed for >1 second, triggers a **flashing red border** and `"WARNING: SLEEPY / DROWSY DRIVER!"` with an optional audio beep.
  * **Mouth Aspect Ratio (MAR)**: Detects yawning (mouth opening excessively for consecutive frames).
  * **Iris Gaze Tracking**: Determines whether the person is looking **Center**, **Left**, or **Right**.
  * **Live Fatigue Meter**: Dynamic 0–100% progress gauge measuring drowsiness severity in real-time.
* **Controls**:
  * `M` — Toggle 3D Facial Mesh and Iris connections.
  * `S` — Toggle warning beep sound.
  * `F` — Toggle Fullscreen.
  * `Q` — Quit.

---

### 4. Regional Voice AI: दादी साथी (Dadi Sathi Voice Assistant)
**Files:** [`dadi_assistant.py`](dadi_assistant.py), [`logic.py`](logic.py)  
**Run:**
```powershell
python dadi_assistant.py
```

* **Purpose**: Demonstrates inclusive AI built for elderly grandparents in rural areas who speak regional languages (Bhojpuri / Hindi).
* **Architecture**:
  ```text
  Microphone (sounddevice) 
       ▼
  Speech-to-Text (Google Speech Recognition hi-IN)
       ▼
  Reasoning Brain (Groq Qwen 2.5 / 3 ➔ Claude ➔ Offline Engine in logic.py)
       ▼
  Text-to-Speech (gTTS + Pygame Mixer) & Tkinter GUI
  ```
* **Sample Voice Commands**:
  * **Medicine Reminders**: *"दवाई 5 मिनट बाद याद दिला देबू"* (Remind me about medicine in 5 minutes)
  * **Phone Calls**: *"बेटा के फोन लगावा"* (Call my son)
  * **Read Messages**: *"संदेश पढ़ के सुनावा"* (Read my inbox messages)
  * **Greetings & Small Talk**: *"प्रणाम दादी, का हाल बा?"* (Hello Dadi, how are you?)
  * **Time Queries**: *"घड़ी में केतना बज रहल बा?"* (What time is it?)
* **Resilient Multi-Tier Fallback**:
  1. **Groq Cloud (Primary)**: Sub-second conversational responses with native JSON mode.
  2. **Claude (Secondary)**: Used if Anthropic API key is provided.
  3. **Offline Keyword Engine (Zero-Internet Fallback)**: Built into [`logic.py`](logic.py); runs autonomously using regex and Hindi number parsers.
* **Controls**:
  * **`SPACE` bar** or click **🎙️ बोलीं**: Start listening to voice.
  * **Text Input Box + `Enter`**: Type manually if the room is too noisy.
  * **`F11`**: Fullscreen toggle.

---

## 🎯 Presenter Cheat Sheet & Hotkeys

| Demo | Window Title | Key Action |
|---|---|---|
| **YOLO Hook** | `AI Bootcamp - Computer Vision` | `F` (Fullscreen), `Q` (Quit) |
| **Air Canvas** | `AI Bootcamp - Air Canvas & Gesture Magic` | `1-5` (Colors), `E` (Eraser), `C` (Clear), `F` (Fullscreen), `Q` (Quit) |
| **Fatigue Monitor** | `AI Bootcamp - Facial Analysis & Drowsiness Detector` | `M` (Toggle 3D Mesh), `S` (Toggle Beep), `F` (Fullscreen), `Q` (Quit) |
| **Dadi Assistant** | `Dadi Sathi - AI Bootcamp` | `SPACE` (Voice Mic), `F11` (Fullscreen), `Enter` (Submit Text) |

---

## 🛠️ Troubleshooting & FAQs

### 1. `Camera could not be opened`
* If you have an external USB webcam, open the script and change `CAMERA_INDEX = 0` to `CAMERA_INDEX = 1`.
* Ensure no other application (Zoom, Teams, Camera app) is holding the webcam lock.

### 2. `Dadi Sathi: आवाज़ नहीं चली (Internet error)`
* `gTTS` requires an active internet connection to download audio. If presenting without internet, the text response will still display clearly on screen, and the offline intent engine in `logic.py` will process requests.

### 3. `Permission denied on cv2.pyd during pip install`
* This occurs when a Python process or editor lock is holding OpenCV in memory. Close all running terminal scripts before upgrading packages.

---

## 📄 License
This project is open-source and free to use for educational workshops, hackathons, and bootcamp presentations.
