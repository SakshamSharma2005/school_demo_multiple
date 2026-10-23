"""
Demo 2: Face Feature, Eye Movement & Drowsiness / Sleepiness Detection
Uses: MediaPipe Face Mesh + Iris Tracking + OpenCV
Key Features:
 - Eye Aspect Ratio (EAR) -> Detects blinks & closed eyes / falling asleep
 - Mouth Aspect Ratio (MAR) -> Detects yawning / fatigue
 - Iris Gaze Tracking -> Tracks eye movement (Left, Right, Center)
 - Live Fatigue Meter & Drowsiness Alert (Visual & Optional Sound)
Keys:
 - 'f': Toggle Fullscreen
 - 'm': Toggle Facial Mesh
 - 's': Toggle Alert Sound (Beep)
 - 'q': Quit
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# ----------------- CONFIGURATION -----------------
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Eye Aspect Ratio thresholds
EAR_CLOSED_THRESH = 0.21       # Below this = eyes considered closed
EYE_CLOSED_FRAMES_ALERT = 28   # ~1.0-1.5 sec at 25-30 FPS -> triggers SLEEPY alert

# Mouth Aspect Ratio threshold (Yawning)
MAR_YAWN_THRESH = 0.60         # Above this = yawning
YAWN_FRAMES_ALERT = 20         # ~0.8-1.0 sec -> triggers YAWN alert

# Landmark Indices for MediaPipe Face Mesh (with refine_landmarks=True)
# Right Eye
RIGHT_EYE_PTS = [33, 160, 158, 133, 153, 144]
# Left Eye
LEFT_EYE_PTS = [362, 385, 387, 263, 373, 380]
# Lips / Mouth
MOUTH_PTS = [61, 291, 13, 14, 82, 312, 87, 317]

# Iris landmarks
RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473


def euclidean_dist(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


def calculate_ear(landmarks, eye_indices, w, h):
    """Calculate Eye Aspect Ratio (EAR)"""
    pts = [(int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in eye_indices]
    # Vertical distances
    v1 = euclidean_dist(pts[1], pts[5])
    v2 = euclidean_dist(pts[2], pts[4])
    # Horizontal distance
    h_dist = euclidean_dist(pts[0], pts[3])
    if h_dist == 0:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


def calculate_mar(landmarks, mouth_indices, w, h):
    """Calculate Mouth Aspect Ratio (MAR) for yawn detection"""
    p_corner_l = (int(landmarks[mouth_indices[0]].x * w), int(landmarks[mouth_indices[0]].y * h))
    p_corner_r = (int(landmarks[mouth_indices[1]].x * w), int(landmarks[mouth_indices[1]].y * h))
    p_lip_top = (int(landmarks[mouth_indices[2]].x * w), int(landmarks[mouth_indices[2]].y * h))
    p_lip_bot = (int(landmarks[mouth_indices[3]].x * w), int(landmarks[mouth_indices[3]].y * h))

    vertical = euclidean_dist(p_lip_top, p_lip_bot)
    horizontal = euclidean_dist(p_corner_l, p_corner_r)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal


def estimate_gaze(landmarks, w, h):
    """Estimate eye gaze direction using iris landmark positions relative to eye corners"""
    # Right eye: 33 (outer corner), 133 (inner corner)
    r_outer = landmarks[33].x * w
    r_inner = landmarks[133].x * w
    r_iris = landmarks[RIGHT_IRIS_CENTER].x * w

    # Left eye: 362 (inner corner), 263 (outer corner)
    l_inner = landmarks[362].x * w
    l_outer = landmarks[263].x * w
    l_iris = landmarks[LEFT_IRIS_CENTER].x * w

    # Relative ratio (0 = far left, 1 = far right)
    r_ratio = (r_iris - min(r_outer, r_inner)) / max(abs(r_inner - r_outer), 1e-5)
    l_ratio = (l_iris - min(l_inner, l_outer)) / max(abs(l_outer - l_inner), 1e-5)
    avg_ratio = (r_ratio + l_ratio) / 2.0

    if avg_ratio < 0.38:
        return "LOOKING RIGHT"
    elif avg_ratio > 0.62:
        return "LOOKING LEFT"
    else:
        return "LOOKING CENTER"


def draw_glass_card(img, x, y, w, h, bg_color=(20, 24, 33), alpha=0.75, radius=12):
    """Draw a semi-transparent HUD card with modern aesthetics"""
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Card outline
    cv2.rectangle(img, (x, y), (x + w, y + h), (70, 80, 100), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    win_title = "AI Bootcamp - Facial Analysis & Drowsiness Detector"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    # MediaPipe Face Mesh initialization
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,  # Enables iris tracking
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # State trackers
    closed_eye_frame_count = 0
    yawn_frame_count = 0
    total_blinks = 0
    eye_was_closed = False
    fullscreen = False
    show_mesh = False
    sound_alert_enabled = False
    last_beep_time = 0

    prev_time = time.time()

    print("\n--- Drowsiness & Eye Feature Demo Ready ---")
    print("Keys: [q] Quit  |  [f] Fullscreen  |  [m] Toggle Mesh  |  [s] Toggle Beep Sound\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame for natural interaction
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-5)
        prev_time = current_time

        drowsy_alert = False
        yawn_alert = False
        gaze_str = "DETECTING..."
        avg_ear = 0.0
        mar = 0.0
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark

            # 1. Calculate EAR for both eyes
            r_ear = calculate_ear(landmarks, RIGHT_EYE_PTS, w, h)
            l_ear = calculate_ear(landmarks, LEFT_EYE_PTS, w, h)
            avg_ear = (r_ear + l_ear) / 2.0

            # 2. Calculate MAR for mouth (Yawn detection)
            mar = calculate_mar(landmarks, MOUTH_PTS, w, h)

            # 3. Gaze direction estimation
            gaze_str = estimate_gaze(landmarks, w, h)

            # 4. Blink & Eye Closure logic
            if avg_ear < EAR_CLOSED_THRESH:
                closed_eye_frame_count += 1
                eye_was_closed = True
                if closed_eye_frame_count >= EYE_CLOSED_FRAMES_ALERT:
                    drowsy_alert = True
            else:
                if eye_was_closed and closed_eye_frame_count < EYE_CLOSED_FRAMES_ALERT:
                    total_blinks += 1
                closed_eye_frame_count = 0
                eye_was_closed = False

            # 5. Yawning logic
            if mar > MAR_YAWN_THRESH:
                yawn_frame_count += 1
                if yawn_frame_count >= YAWN_FRAMES_ALERT:
                    yawn_alert = True
            else:
                yawn_frame_count = max(0, yawn_frame_count - 1)

            # Optional: Draw facial mesh
            if show_mesh:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

            # Draw Iris markers (glowing cyan dots on eyes)
            r_iris_pt = (int(landmarks[RIGHT_IRIS_CENTER].x * w), int(landmarks[RIGHT_IRIS_CENTER].y * h))
            l_iris_pt = (int(landmarks[LEFT_IRIS_CENTER].x * w), int(landmarks[LEFT_IRIS_CENTER].y * h))
            cv2.circle(frame, r_iris_pt, 4, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, l_iris_pt, 4, (255, 255, 0), -1, cv2.LINE_AA)

        # ----------------- UI / HUD OVERLAY -----------------
        # Top banner
        draw_glass_card(frame, 20, 20, 360, 210, bg_color=(15, 20, 28), alpha=0.82)
        cv2.putText(frame, "AI DRIVER & FATIGUE MONITOR", (36, 48),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 1, cv2.LINE_AA)

        # Status text lines
        if face_detected:
            # Eye status
            eye_status = "CLOSED" if (avg_ear < EAR_CLOSED_THRESH) else "OPEN"
            eye_color = (0, 70, 255) if eye_status == "CLOSED" else (0, 255, 120)
            cv2.putText(frame, f"Eyes: {eye_status} (EAR: {avg_ear:.2f})", (36, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, eye_color, 2, cv2.LINE_AA)

            # Gaze
            cv2.putText(frame, f"Gaze: {gaze_str}", (36, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

            # Yawn / Mouth
            yawn_status = "YAWNING" if yawn_alert else "NORMAL"
            yawn_color = (0, 140, 255) if yawn_alert else (200, 200, 200)
            cv2.putText(frame, f"Mouth: {yawn_status} (MAR: {mar:.2f})", (36, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, yawn_color, 1, cv2.LINE_AA)

            # Blinks & FPS
            cv2.putText(frame, f"Blinks: {total_blinks}   FPS: {fps:.0f}", (36, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

            # Fatigue meter bar
            # Compute fatigue score (0 to 100)
            fatigue_score = min(100, int((closed_eye_frame_count / EYE_CLOSED_FRAMES_ALERT) * 80 + (yawn_frame_count / YAWN_FRAMES_ALERT) * 40))
            bar_w = 320
            fill_w = int(bar_w * (fatigue_score / 100.0))
            bar_color = (0, 255, 120) if fatigue_score < 40 else ((0, 180, 255) if fatigue_score < 75 else (0, 40, 255))
            cv2.rectangle(frame, (36, 192), (36 + bar_w, 206), (50, 50, 50), -1)
            if fill_w > 0:
                cv2.rectangle(frame, (36, 192), (36 + fill_w, 206), bar_color, -1)
            cv2.putText(frame, f"Fatigue: {fatigue_score}%", (40, 204),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Face Detected", (36, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        # ----------------- WARNING ALERTS -----------------
        if drowsy_alert:
            # Pulsing Red Warning Banner
            pulse = int((math.sin(time.time() * 12) + 1) * 127)
            # Red screen border
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, pulse), 10)

            # Center alert card
            cw, ch = 680, 90
            cx, cy = (w - cw) // 2, h - 140
            draw_glass_card(frame, cx, cy, cw, ch, bg_color=(0, 0, 160), alpha=0.88)
            cv2.putText(frame, "WARNING: SLEEPY / DROWSY DRIVER!", (cx + 25, cy + 58),
                        cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)

            # Trigger beep if enabled
            if sound_alert_enabled and HAS_WINSOUND and (time.time() - last_beep_time > 0.6):
                try:
                    winsound.Beep(1200, 250)
                    last_beep_time = time.time()
                except Exception:
                    pass

        elif yawn_alert:
            # Orange warning card
            cw, ch = 520, 70
            cx, cy = (w - cw) // 2, h - 120
            draw_glass_card(frame, cx, cy, cw, ch, bg_color=(0, 90, 180), alpha=0.85)
            cv2.putText(frame, "FATIGUE ALERT: YAWNING DETECTED", (cx + 25, cy + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom shortcut hints
        hint_str = f"[M] Mesh: {'ON' if show_mesh else 'OFF'}  |  [S] Sound: {'ON' if sound_alert_enabled else 'OFF'}  |  [F] Fullscreen  |  [Q] Quit"
        cv2.putText(frame, hint_str, (24, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(win_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif key == ord("m"):
            show_mesh = not show_mesh
        elif key == ord("s"):
            sound_alert_enabled = not sound_alert_enabled

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
