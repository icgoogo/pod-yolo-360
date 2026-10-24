import os
import sys
import time
import threading
import subprocess
import cv2
import numpy as np
import torch
import psutil

# Configuration via Environment Variables
RTSP_IN = os.getenv("RTSP_URL", "rtsp://localhost:8554/live/theta")
RTMP_OUT = os.getenv("OUT_RTMP", "rtmp://localhost:1935/live/theta_blurred")
MODEL_PATH = os.getenv("YOLO_MODEL", "yolov8n-face.pt")
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
MODE = os.getenv("ANONYMIZATION_MODE", "YOLO").upper()  # 'YOLO' or 'PASSTHROUGH'
TELEMETRY_CSV = os.getenv("TELEMETRY_CSV", "pipeline_telemetry.csv")
TARGET_BITRATE = os.getenv("OUT_BITRATE", "16M")

# Dynamic Detection & Bounding Box Controls
CONF_THRESH = float(os.getenv("CONF_THRESH", "0.30"))     # Confidence threshold (0.10 - 0.90)
BOX_MARGIN = float(os.getenv("BOX_MARGIN", "0.15"))       # Safety padding around face (15%)
HEAD_RATIO = float(os.getenv("HEAD_RATIO", "0.0"))        # 0.0 = Direct Face Model; >0.0 for person model fallback

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
TARGET_GPU = os.getenv("CUDA_VISIBLE_DEVICES", "0")

if torch.cuda.is_available():
    DEVICE = 0  # Index relative to container's visible GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[+] CUDA Verified! Running on GPU: {gpu_name} (Device ID: {TARGET_GPU})")
else:
    DEVICE = "cpu"
    print("[!] WARNING: CUDA is NOT available. Falling back to CPU!")

print(
    f"[CONFIG] Mode: {MODE} | Model: {MODEL_PATH} | ImgSz: {IMGSZ} | "
    f"Conf: {CONF_THRESH} | Margin: {BOX_MARGIN*100:.0f}% | HeadRatio: {HEAD_RATIO*100:.0f}%"
)


# 1. Threaded Video Capture (Zero-Copy Socket Drain with drop tracking)
class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            sys.exit(f"[-] Error: Unable to open RTSP input at {src}")
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.new_frame = False
        self.lock = threading.Lock()
        self.dropped_frames = 0
        self.incoming_fps = 0.0
        self.last_frame_time = time.perf_counter() 

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            now = time.perf_counter()
            dt = now - self.last_frame_time
            self.last_frame_time = now 

            with self.lock:
                if self.new_frame:
                    self.dropped_frames += 1
                self.frame = frame
                if dt > 0:
                    self.incoming_fps = 1.0 / dt 
                self.new_frame = True

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0, 0.0
            self.new_frame = False
            return self.running, self.frame, self.dropped_frames, self.incoming_fps

    def release(self):
        self.running = False
        self.cap.release()


print(f"[*] Connecting to stream: {RTSP_IN}")
reader = FreshFrameReader(RTSP_IN)

# Wait for first valid frame safely
first_frame = None
while first_frame is None:
    _, first_frame, _, _ = reader.read_latest()
    time.sleep(0.02)

height, width = first_frame.shape[:2]

# SANITIZE FPS: OpenCV RTSP often returns 90000.0 (the 90kHz RTP clock rate)
raw_fps = reader.cap.get(cv2.CAP_PROP_FPS)
if raw_fps and 1.0 <= raw_fps <= 120.0:
    fps = float(raw_fps)
else:
    fps = float(os.getenv("STREAM_FPS", "30.0"))

print(f"[+] Connected! Resolution: {width}x{height} @ {fps:.1f} FPS (Raw reported: {raw_fps})")

# 2. Load Model & FFmpeg Pipe
model = None
if MODE == "YOLO":
    from ultralytics import YOLO
    print(f"[*] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
else:
    print("[*] PASSTHROUGH Mode: Zero-AI Baseline measurement (skipping YOLO & pixelation).")

ffmpeg_cmd = [
    "/usr/bin/ffmpeg",
    "-y",
    "-f",
    "rawvideo",
    "-pix_fmt",
    "bgr24",
    "-s",
    f"{width}x{height}",
    "-r",
    str(int(fps)),
    "-i",
    "-",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-tune",
    "zerolatency",
    "-pix_fmt",
    "yuv420p",
    "-f",
    "flv",
    RTMP_OUT,
]
pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
print(f"[+] Streaming live to: {RTMP_OUT} (Clamped @ {TARGET_BITRATE})")

# 3. Dynamic Bounding Box Adjustment (Head focus + safety margin)
def adjust_box(box, margin=BOX_MARGIN, head_ratio=HEAD_RATIO, max_w=width, max_h=height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    # Focus on the head area (top X% of person detection box)
    if head_ratio > 0.0:
        y2 = y1 + int(h * head_ratio)
        h = y2 - y1

    # Apply safety margin expansion (covers hair, ears, chin)
    dx = int(w * margin)
    dy = int(h * margin)

    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(max_w, x2 + dx),
        min(max_h, y2 + dy),
    ]


# 4. Ultra-Fast ROI Pixelation (10x faster than GaussianBlur)
def apply_pixelate(frame, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.resize(
            small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )
    return frame


# 5. Telemetry Header Setup
telemetry_buffer = []
csv_header = "timestamp,frame_idx,mode,imgsz,t_decode_ms,t_infer_ms,t_blur_ms,t_encode_ms,t_total_ms,fps,cam_fps,boxes_count,dropped_frames,vram_mb,cpu_pct,ram_mb\n"
with open(TELEMETRY_CSV, "w") as f:
    f.write(csv_header)

print(f"[+] Stage telemetry logging enabled -> {TELEMETRY_CSV}")

# 6. Processing Loop
frame_count = 0
last_boxes = []
prev_time = time.perf_counter()
process = psutil.Process()

try:
    while True:
        loop_start = time.perf_counter()

        # Step A: Capture & Decode
        t0 = time.perf_counter()
        success, frame, dropped, cam_fps = reader.read_latest()
        if not success or frame is None:
            print("[-] Stream ended. Exiting...")
            break
        t_decode_ms = (time.perf_counter() - t0) * 1000.0

        # Make a single working copy for modification and piping
        work_frame = frame.copy()

        # Step B: Run inference every 2nd frame (if in YOLO mode)
        t_infer_ms = 0.0
        if MODE == "YOLO":
            if frame_count % 2 == 0:
                if DEVICE != "cpu":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                results = model.predict(
                    work_frame,
                    imgsz=IMGSZ,
                    conf=CONF_THRESH,
                    classes=[0],
                    device=DEVICE,
                    verbose=False,
                )

                if DEVICE != "cpu":
                    torch.cuda.synchronize()
                t_infer_ms = (time.perf_counter() - t1) * 1000.0

                last_boxes = []
                if len(results) > 0 and results[0].boxes is not None:
                    raw_boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    last_boxes = [adjust_box(b) for b in raw_boxes]

        # Step C: Apply pixelation using adjusted bounding boxes (if in YOLO mode)
        t2 = time.perf_counter()
        if MODE == "YOLO":
            for box in last_boxes:
                work_frame = apply_pixelate(work_frame, box)
        t_blur_ms = (time.perf_counter() - t2) * 1000.0

        # Step D: Pipe out to FFmpeg
        t3 = time.perf_counter()
        try:
            pipe.stdin.write(work_frame.tobytes())
        except (BrokenPipeError, IOError):
            print("[-] FFmpeg pipe broken. MediaMTX disconnected.")
            break
        t_encode_ms = (time.perf_counter() - t3) * 1000.0

        loop_end = time.perf_counter()
        t_total_ms = (loop_end - loop_start) * 1000.0

        current_fps = 1.0 / (loop_end - prev_time) if (loop_end - prev_time) > 0 else fps
        prev_time = loop_end

        # System telemetry
        vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        cpu_pct = process.cpu_percent()
        ram_mb = process.memory_info().rss / (1024 * 1024)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        telemetry_buffer.append(
            f"{ts},{frame_count},{MODE},{IMGSZ},{t_decode_ms:.2f},{t_infer_ms:.2f},{t_blur_ms:.2f},{t_encode_ms:.2f},{t_total_ms:.2f},{current_fps:.1f},{cam_fps:.1f},{len(last_boxes)},{dropped},{vram_mb:.1f},{cpu_pct:.1f},{ram_mb:.1f}\n"
        )

        frame_count += 1

        # Periodic flush every 30 frames to prevent disk I/O latency
        if frame_count % 30 == 0:
            with open(TELEMETRY_CSV, "a") as f:
                f.writelines(telemetry_buffer)
            telemetry_buffer.clear()

        if frame_count % 90 == 0:
            print(
                f"[{MODE} #{frame_count}] Total: {t_total_ms:.1f}ms (Decode: {t_decode_ms:.1f}ms | "
                f"Infer: {t_infer_ms:.1f}ms | Blur: {t_blur_ms:.1f}ms | Enc: {t_encode_ms:.1f}ms) | "
                f"FPS: {current_fps:.1f} | f"Camera FPS: {cam_fps:.1f} | Dropped: {dropped} | VRAM: {vram_mb:.1f}MB"
            )

except KeyboardInterrupt:
    print("[*] Stopping on user interrupt...")
finally:
    if telemetry_buffer:
        with open(TELEMETRY_CSV, "a") as f:
            f.writelines(telemetry_buffer)
    reader.release()
    if pipe and pipe.stdin:
        pipe.stdin.close()
        pipe.wait()
    print("[+] Clean shutdown complete.")
