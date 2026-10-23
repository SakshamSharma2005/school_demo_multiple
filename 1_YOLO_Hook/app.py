import time
import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Object Detection", page_icon="🎯", layout="wide")
st.title("🎯 Demo 1: Live YOLO Object Detection")

IMG_SIZE = 480
CONFIDENCE = st.sidebar.slider("Confidence Threshold", 0.10, 1.0, 0.40, 0.05)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.prev = time.time()
        self.conf = 0.40

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        results = model(img, imgsz=IMG_SIZE, conf=self.conf, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1.0 / max(now - self.prev, 1e-6)
        self.prev = now
        count = len(results[0].boxes)

        cv2.putText(
            annotated,
            f"Objects seen: {count}   FPS: {fps:.0f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

ctx = webrtc_streamer(
    key="yolo-demo",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if ctx.video_processor:
    ctx.video_processor.conf = CONFIDENCE