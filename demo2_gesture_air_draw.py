"""
Demo 2: Gesture Magic & Air Drawing (MediaPipe Hands + OpenCV).

Features:
  * Finger Counter   : Accurately shows fingers up (0-5)
  * Air Drawing      : Index finger up -> Smooth drawing on air canvas
  * Selection / Move : Index + Middle (Peace sign) -> Move cursor / select color from top bar
  * Eraser Mode      : Select Eraser tool or press 'e' -> Erases lines on canvas
  * Quick Clear      : Open palm held for ~1 second (with visual progress ring) or press 'c'
  * Interactive Bar  : Select colors (Red, Green, Blue, Yellow, Purple, Eraser, Clear)
                       directly by pointing to the top bar or pressing keys 1-6 / e / c
  * Hotkeys          : 1-6 (Colors), e (Eraser), c (Clear), f (Fullscreen), q (Quit)
"""

import time
import cv2
import numpy as np
import mediapipe as mp

# ----------------- CONFIGURATION -----------------
CAMERA_INDEX = 0
W, H = 1280, 720

# Palette Colors (BGR)
PALETTE = [
    {"name": "Red",    "bgr": (0, 0, 255),     "key": ord("1")},
    {"name": "Green",  "bgr": (0, 255, 0),     "key": ord("2")},
    {"name": "Cyan",   "bgr": (255, 255, 0),   "key": ord("3")},
    {"name": "Yellow", "bgr": (0, 230, 255),   "key": ord("4")},
    {"name": "Purple", "bgr": (255, 0, 200),   "key": ord("5")},
    {"name": "Eraser", "bgr": (0, 0, 0),       "key": ord("e")},
    {"name": "Clear",  "bgr": (60, 60, 60),    "key": ord("c")},
]

# Landmark indices
TIPS = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky
MCP_PTS = [2, 5, 9, 13, 17]


def get_fingers_up(lm, handedness_label="Right"):
    """
    Robust finger detector using landmark geometry.
    Works for both left and right hands in mirrored webcam view.
    Returns: list of 5 ints [thumb, index, middle, ring, pinky] (0 or 1)
    """
    fingers = []

    # 1. Thumb detection (horizontal distance to wrist & MCP base)
    # In mirrored image (cv2.flip(frame, 1)):
    # Right hand: thumb is on the left side of wrist (lm[4].x < lm[3].x)
    # Left hand:  thumb is on the right side of wrist (lm[4].x > lm[3].x)
    if handedness_label == "Right":
        thumb_up = 1 if lm[4].x < lm[3].x else 0
    else:
        thumb_up = 1 if lm[4].x > lm[3].x else 0
    fingers.append(thumb_up)

    # 2. Other 4 fingers: tip y-coordinate vs PIP joint y-coordinate
    # (In screen coordinates, smaller Y means higher on screen)
    pip_indices = [6, 10, 14, 18]
    for tip_idx, pip_idx in zip(TIPS[1:], pip_indices):
        fingers.append(1 if lm[tip_idx].y < lm[pip_idx].y else 0)

    return fingers


def draw_top_toolbar(img, active_color_idx, is_eraser):
    """Draws a modern, interactive top bar with color buttons"""
    num_items = len(PALETTE)
    btn_w = W // num_items
    btn_h = 75

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, btn_h), (20, 24, 32), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    for i, item in enumerate(PALETTE):
        bx1 = i * btn_w + 8
        by1 = 10
        bx2 = (i + 1) * btn_w - 8
        by2 = btn_h - 10

        is_selected = (i == active_color_idx)
        b_color = item["bgr"]

        if item["name"] == "Eraser":
            # Dark grey fill with eraser icon/text
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (45, 45, 55), -1)
            cv2.putText(img, "ERASER", (bx1 + 18, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
        elif item["name"] == "Clear":
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (60, 30, 30), -1)
            cv2.putText(img, "CLEAR ALL", (bx1 + 10, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 120, 120), 1, cv2.LINE_AA)
        else:
            # Color pill
            cv2.rectangle(img, (bx1, by1), (bx2, by2), b_color, -1)
            # Text label
            text_color = (0, 0, 0) if item["name"] in ["Yellow", "Green", "Cyan"] else (255, 255, 255)
            cv2.putText(img, item["name"].upper(), (bx1 + 20, by1 + 35),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, text_color, 2, cv2.LINE_AA)

        # Highlight selected button with glowing border
        if is_selected:
            cv2.rectangle(img, (bx1 - 3, by1 - 3), (bx2 + 3, by2 + 3), (255, 255, 255), 3, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (90, 100, 115), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    win = "AI Bootcamp - Air Canvas & Gesture Magic"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )

    # Drawing canvas
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # State variables
    active_color_idx = 0       # default to Red
    brush_thickness = 10
    eraser_thickness = 55
    prev_x, prev_y = None, None
    palm_start_time = None
    CLEAR_HOLD_SECS = 0.9
    last_clear_time = 0
    fullscreen = False

    print("\n--- Air Drawing & Gesture Magic Started ---")
    print("Controls:")
    print(" - 1 Finger (Index)        -> Draw")
    print(" - 2 Fingers (Peace Sign)   -> Move cursor / Select tool on top bar")
    print(" - 5 Fingers (Open Palm)    -> Hold 1 sec to Clear screen")
    print(" - Keys 1-5                 -> Change colors")
    print(" - Key 'e'                  -> Toggle Eraser")
    print(" - Key 'c'                  -> Clear Canvas")
    print(" - Key 'f'                  -> Toggle Fullscreen")
    print(" - Key 'q'                  -> Quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Warning] Camera frame could not be read.")
            break

        # Mirror horizontally and ensure resolution is (W, H)
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))

        # MediaPipe Hands processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        res = hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        finger_count = 0
        mode_text = "Show Hand"
        mode_color = (180, 180, 180)
        curr_tool = PALETTE[active_color_idx]["name"]
        is_eraser = (curr_tool == "Eraser")
        active_bgr = PALETTE[active_color_idx]["bgr"]

        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            # Hand classification (Right vs Left hand)
            handedness = "Right"
            if res.multi_handedness:
                # Flipped frame reverses handedness label
                raw_label = res.multi_handedness[0].classification[0].label
                handedness = "Right" if raw_label == "Left" else "Left"

            # Draw sleek landmarks
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            lm = hl.landmark
            fingers = get_fingers_up(lm, handedness)
            finger_count = sum(fingers)

            # Raw tip of index finger
            raw_tip_x = int(lm[8].x * W)
            raw_tip_y = int(lm[8].y * H)

            # Exponential Moving Average for ultra-smooth lines
            if prev_x is None:
                smooth_x, smooth_y = raw_tip_x, raw_tip_y
            else:
                smooth_x = int(prev_x * 0.35 + raw_tip_x * 0.65)
                smooth_y = int(prev_y * 0.35 + raw_tip_y * 0.65)

            # ---------------- GESTURE MODES ----------------

            # 1. SELECTION / HOVER MODE: Index + Middle finger (Peace Sign)
            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                mode_text = "TOOL SELECTION / HOVER"
                mode_color = (255, 255, 0)
                prev_x, prev_y = None, None
                palm_start_time = None

                # Cursor marker between index and middle
                mid_x = int((lm[8].x + lm[12].x) * 0.5 * W)
                mid_y = int((lm[8].y + lm[12].y) * 0.5 * H)
                cv2.circle(frame, (mid_x, mid_y), 15, (255, 255, 255), 2, cv2.LINE_AA)

                # Check if touching top toolbar
                if mid_y < 75:
                    btn_w = W // len(PALETTE)
                    selected_idx = mid_x // btn_w
                    if 0 <= selected_idx < len(PALETTE):
                        if PALETTE[selected_idx]["name"] == "Clear":
                            canvas[:] = 0
                            mode_text = "CANVAS CLEARED!"
                            mode_color = (0, 0, 255)
                        else:
                            active_color_idx = selected_idx

            # 2. DRAWING / ERASING MODE: Only Index finger UP
            elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                palm_start_time = None

                if is_eraser:
                    mode_text = "ERASING"
                    mode_color = (120, 120, 255)
                    # Erase by drawing black on canvas
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), (0, 0, 0), eraser_thickness)
                    cv2.circle(frame, (smooth_x, smooth_y), eraser_thickness // 2, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    mode_text = "DRAWING"
                    mode_color = (0, 255, 120)
                    # Draw on canvas
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), active_bgr, brush_thickness, cv2.LINE_AA)
                    cv2.circle(frame, (smooth_x, smooth_y), brush_thickness + 2, active_bgr, -1, cv2.LINE_AA)

                prev_x, prev_y = smooth_x, smooth_y

            # 3. CLEAR GESTURE: Open Palm (All 4 main fingers up)
            elif fingers[1] and fingers[2] and fingers[3] and fingers[4]:
                prev_x, prev_y = None, None
                mode_text = "HOLD TO CLEAR..."
                mode_color = (0, 140, 255)

                if palm_start_time is None:
                    palm_start_time = time.time()

                elapsed = time.time() - palm_start_time
                progress = min(1.0, elapsed / CLEAR_HOLD_SECS)

                # Draw circular progress animation on palm
                wrist_x, wrist_y = int(lm[0].x * W), int(lm[0].y * H)
                cv2.circle(frame, (wrist_x, wrist_y), 32, (50, 50, 50), -1, cv2.LINE_AA)
                # Sweep angle
                sweep = int(progress * 360)
                cv2.ellipse(frame, (wrist_x, wrist_y), (32, 32), 0, -90, -90 + sweep, (0, 0, 255), 4, cv2.LINE_AA)

                if elapsed >= CLEAR_HOLD_SECS and (time.time() - last_clear_time > 1.0):
                    canvas[:] = 0
                    mode_text = "CANVAS CLEARED!"
                    last_clear_time = time.time()
                    palm_start_time = None

            else:
                mode_text = "READY"
                mode_color = (200, 200, 200)
                prev_x, prev_y = None, None
                palm_start_time = None

        else:
            prev_x, prev_y = None, None
            palm_start_time = None

        # ----------------- BLEND CANVAS & CAMERA -----------------
        # Clean alpha blending for anti-aliased, vivid strokes
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        frame = cv2.add(frame_bg, canvas_fg)

        # ----------------- TOP INTERACTIVE TOOLBAR -----------------
        draw_top_toolbar(frame, active_color_idx, is_eraser)

        # ----------------- BOTTOM STATUS HUD -----------------
        hud_y = H - 55
        # Glass HUD bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, hud_y), (W, H), (15, 18, 25), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.line(frame, (0, hud_y), (W, hud_y), (60, 70, 85), 1)

        # Fingers Badge
        cv2.putText(frame, f"Fingers: {finger_count}", (24, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 240, 255), 1, cv2.LINE_AA)

        # Mode Badge
        cv2.putText(frame, f"Mode: {mode_text}", (230, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, mode_color, 2, cv2.LINE_AA)

        # Current Tool / Color Badge
        tool_display = f"Tool: {curr_tool}"
        cv2.putText(frame, tool_display, (W - 480, H - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        if not is_eraser:
            cv2.circle(frame, (W - 320, H - 25), 14, active_bgr, -1, cv2.LINE_AA)

        # Help Shortcuts
        cv2.putText(frame, "[1-5] Colors  [E] Eraser  [C] Clear  [F] Fullscreen  [Q] Quit",
                    (W - 640, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow(win, frame)

        # Keyboard handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key == ord("e"):
            # Find eraser index
            for i, item in enumerate(PALETTE):
                if item["name"] == "Eraser":
                    active_color_idx = i
                    break
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        else:
            # Check number keys 1-5
            for i, item in enumerate(PALETTE):
                if key == item["key"]:
                    active_color_idx = i
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
