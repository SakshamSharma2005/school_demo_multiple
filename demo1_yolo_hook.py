"""
Demo 1: Opening hook. Live object detection on the webcam (YOLO).
Run:  python demo1_yolo_hook.py
Keys: f = toggle fullscreen, q = quit
First run downloads yolov8n.pt (~6 MB), so run it ONCE with internet before the event.
"""
import time
import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0      # try 1 if you have an external webcam
IMG_SIZE = 480        # smaller = faster on a laptop without GPU
CONFIDENCE = 0.40

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

win = "AI Bootcamp - Computer Vision"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
fullscreen = False
prev = time.time()

while True:
    
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)  # mirror so waving feels natural

    results = model(frame, imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False)
    annotated = results[0].plot()  # draws boxes + labels like "person 0.91"

    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    count = len(results[0].boxes)
    cv2.putText(annotated, f"Objects seen: {count}   FPS: {fps:.0f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow(win, annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("f"):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            win, cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)

cap.release()
cv2.destroyAllWindows()
