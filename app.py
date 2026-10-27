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
MODE = os.getenv("ANONYMIZATION_MODE", "YOLO").upper()  # YOLO, YOLO-PERSON, YUNET, or PASSTHROUGH
if MODE not in {"YOLO", "YOLO-PERSON", "YUNET", "PASSTHROUGH"}:
    sys.exit(f"[-] Unknown ANONYMIZATION_MODE: {MODE}")

model_path = "yolov8n-face.pt"
if MODE == "YUNET":
    model_path = "face_detection_yunet_2023mar.onnx"
elif MODE == "YOLO-PERSON":
    model_path = "yolo26n.pt"

MODEL_PATH = os.getenv("MODEL", model_path)

# 2:1 rectangle
IMGSZ = int(os.getenv("IMGSZ", "640"))
if IMGSZ <= 0:
    sys.exit("[-] Model image size must be positive.")

TARGET_BITRATE = os.getenv("OUT_BITRATE", "16M")

# Direct environment variable loading with defaults
if MODE == "YOLO-PERSON":
    CONF_THRESH = float(os.getenv("CONF_THRESH", "0.2"))
    BOX_MARGIN  = float(os.getenv("BOX_MARGIN", "0.001"))
    HEAD_RATIO  = float(os.getenv("HEAD_RATIO", "0.35"))
else:
    CONF_THRESH = float(os.getenv("CONF_THRESH", "0.2"))
    BOX_MARGIN  = float(os.getenv("BOX_MARGIN", "0.23"))
    HEAD_RATIO  = float(os.getenv("HEAD_RATIO", "0.0"))

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
DEVICE = "cpu" 
# Device verification
if MODE == "YUNET":
    # Check if OpenCV DNN was compiled with CUDA support
    has_opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False
    if has_opencv_cuda:
        DEVICE = "cuda"
        print("[+] YuNet running on GPU via OpenCV CUDA DNN Backend!")
    else:
        DEVICE = "cpu"
        print("[+] YuNet using OpenCV C++ CPU engine (~1-3ms ultra-fast inference).")
elif torch.cuda.is_available():
    DEVICE = 0  # GPU device index
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[+] CUDA Verified! Running YOLO on GPU: {gpu_name}")
else:
    DEVICE = "cpu"
    print("[!] WARNING: CUDA unavailable. Falling back to CPU!")

print(
    f"[CONFIG] Mode: {MODE} | Model: {MODEL_PATH} | ImgSz height: {IMGSZ} | "
    f"Conf: {CONF_THRESH} | Margin: {BOX_MARGIN*100:.0f}% | HeadRatio: {HEAD_RATIO*100:.0f}%"
)


# 1. Threaded Video Capture (Synchronized with Condition to avoid duplicate frame processing)
class FreshFrameReader:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(
            src,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )
        if not self.cap.isOpened():
            sys.exit(f"[-] Error: Unable to open RTSP input at {src}")
        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            sys.exit(f"[-] Error: No initial frame received from {src}")
            
        self.running = True
        self.cond = threading.Condition()  # Condition lock for main loop synchronization
        self.new_frame = True
        self.dropped_frames = 0
        self.latest_decode_ms = 0.0

        # Windowed Camera Ingest FPS variables
        self.cam_window_frames = 0 
        self.cam_window_start = time.perf_counter()
        self.cam_fps_window = 30.0 

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            t_dec_start = time.perf_counter()
            ret, frame = self.cap.read()
            t_decode_ms = (time.perf_counter() - t_dec_start) * 1000.0
            if not ret:
                with self.cond:
                    self.running = False
                    self.cond.notify_all()
                break

            now = time.perf_counter()

            with self.cond:
                if self.new_frame:
                    self.dropped_frames += 1
                self.frame = frame
                self.latest_decode_ms = t_decode_ms

                # Calculate windowed camera ingest FPS over rolling ~1 second window
                self.cam_window_frames += 1
                elapsed_cam = now - self.cam_window_start
                if elapsed_cam >= 1.0:
                    self.cam_fps_window = self.cam_window_frames / elapsed_cam
                    self.cam_window_frames = 0
                    self.cam_window_start = now 
                self.new_frame = True
                self.cond.notify_all()  # Wake up main thread waiting for fresh frame

    def read_latest(self, wait_for_fresh=True):
        """Reads the latest frame. If wait_for_fresh=True, blocks until a NEW frame arrives."""
        with self.cond:
            if wait_for_fresh:
                while not self.new_frame and self.running:
                    self.cond.wait(timeout=0.1)
                    
            if not self.running or self.frame is None:
                return False, None, 0, 0.0, 0.0
                
            self.new_frame = False
            dropped_delta = self.dropped_frames
            self.dropped_frames = 0
            return self.running, self.frame, dropped_delta, self.cam_fps_window, self.latest_decode_ms

    def release(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
        self.cap.release()


print(f"[*] Connecting to stream: {RTSP_IN}")
reader = FreshFrameReader(RTSP_IN)

# Wait for first valid frame safely
first_frame = None
while first_frame is None:
    _, first_frame, _, _, _ = reader.read_latest(wait_for_fresh=False)
    time.sleep(0.02)

height, width = first_frame.shape[:2]
TELEMETRY_CSV = os.getenv("TELEMETRY_CSV", f"pipeline_telemetry_{width}x{height}_{IMGSZ}.csv")

# SANITIZE FPS: OpenCV RTSP often returns 90000.0 (the 90kHz RTP clock rate)
raw_fps = reader.cap.get(cv2.CAP_PROP_FPS)
if raw_fps and 1.0 <= raw_fps <= 120.0:
    fps = float(raw_fps)
else:
    fps = float(os.getenv("STREAM_FPS", "30.0"))

print(f"[+] Connected! Resolution: {width}x{height} @ {fps:.1f} FPS (Raw reported: {raw_fps})")

scale = IMGSZ / height
input_w, input_h = IMGSZ * 2, IMGSZ

# 2. Load Model & FFmpeg Pipe
model = None
if MODE in ["YOLO", "YOLO-PERSON"]:
    from ultralytics import YOLO
    print(f"[*] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
elif MODE == "YUNET":
    print(f"[*] Loading YuNet model: {MODEL_PATH}")
    backend = cv2.dnn.DNN_BACKEND_CUDA if DEVICE == "cuda" else cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CUDA if DEVICE == "cuda" else cv2.dnn.DNN_TARGET_CPU
    
    model = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (input_w, input_h),
        CONF_THRESH,
        0.3,
        5000,
        backend,
        target,
    )
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
    "-b:v",
    TARGET_BITRATE,
    "-maxrate",
    TARGET_BITRATE,
    "-bufsize",
    TARGET_BITRATE,
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

    if head_ratio > 0.0:
        y2 = y1 + int(h * head_ratio)
        h = y2 - y1

    dx = int(w * margin)
    dy = int(h * margin)

    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(max_w, x2 + dx),
        min(max_h, y2 + dy),
    ]


# 4. Ultra-Fast ROI Pixelation
def apply_pixelate(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.resize(
            small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )
    return frame


# 5. Telemetry Header Setup
telemetry_buffer = []
csv_header = (
    "timestamp,frame_idx,mode,imgsz,t_decode_ms,t_infer_ms,t_blur_ms,"
    "t_encode_ms,t_total_ms,fps,cam_fps,boxes_count,dropped_frames,vram_mb,cpu_pct,ram_mb\n"
)
with open(TELEMETRY_CSV, "w", encoding="utf-8") as f:
    f.write(csv_header)

print(f"[+] Stage telemetry logging enabled -> {TELEMETRY_CSV}")

# 6. Processing Loop
frame_count = 0
last_boxes = []
process = psutil.Process()
process.cpu_percent(interval=None)  # Initialize baseline CPU percentage timer

# Windowed Throughput Variables (Main Pipeline)
WINDOW_SIZE = 30  # Compute rolling throughput every 30 frames
window_start_time = time.perf_counter()
window_frame_counter = 0
pipeline_fps_window = float(fps) 

try:
    while True:

        # Step A: Capture & Decode (wait until a fresh unique frame arrives)
        success, frame, dropped, cam_fps, t_decode_ms = reader.read_latest(wait_for_fresh=True)
        if not success or frame is None:
            print("[-] Stream ended. Exiting...")
            break

        loop_start = time.perf_counter()

        work_frame = frame.copy()

        # Step B: Run inference on each processed frame
        t_infer_ms = 0.0
        if MODE in ["YOLO", "YOLO-PERSON"]:
            if DEVICE != "cpu":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            results = model.predict(
                work_frame,
                imgsz=(input_h, input_w),
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

        elif MODE == "YUNET":
            t1 = time.perf_counter()
            detector_frame = cv2.resize(work_frame, (input_w, input_h)) if scale < 1.0 else work_frame
            
            _, faces = model.detect(detector_frame)
            t_infer_ms = (time.perf_counter() - t1) * 1000.0

            last_boxes = []
            if faces is not None:
                scale_x = width / input_w
                scale_y = height / input_h
                for face in faces:
                    x, y, w, h = face[:4]
                    box = [
                        int(x * scale_x),
                        int(y * scale_y),
                        int((x + w) * scale_x),
                        int((y + h) * scale_y),
                    ]
                    last_boxes.append(adjust_box(box))

        # Step C: Apply pixelation using adjusted face bounding boxes
        t2 = time.perf_counter()
        if MODE in {"YOLO", "YUNET", "YOLO-PERSON"}:
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

        # Smooth processing FPS calculated directly from total loop duration
        compute_capacity_fps = 1000.0 / t_total_ms if t_total_ms > 0 else fps

        # TWEAK 2: Windowed Wall-Clock Throughput FPS (Sustained real-world pipeline throughput)
        window_frame_counter += 1
        elapsed_window = loop_end - window_start_time
        if window_frame_counter >= WINDOW_SIZE:
            pipeline_fps_window = window_frame_counter / elapsed_window if elapsed_window > 0 else fps
            window_frame_counter = 0
            window_start_time = loop_end 

        # Accurate reserved VRAM tracking
        vram_mb = torch.cuda.memory_reserved(0) / (1024 * 1024) if torch.cuda.is_available() else 0.0
        cpu_pct = process.cpu_percent()
        ram_mb = process.memory_info().rss / (1024 * 1024)

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        telemetry_buffer.append(
            f"{ts},{frame_count},{MODE},{IMGSZ},{t_decode_ms:.2f},{t_infer_ms:.2f},{t_blur_ms:.2f},"
            f"{t_encode_ms:.2f},{t_total_ms:.2f},{pipeline_fps_window:.1f},{cam_fps:.1f},{len(last_boxes)},"
            f"{dropped},{vram_mb:.1f},{cpu_pct:.1f},{ram_mb:.1f}\n"
        )

        frame_count += 1

        if frame_count % 30 == 0:
            with open(TELEMETRY_CSV, "a", encoding="utf-8") as f:
                f.writelines(telemetry_buffer)
            telemetry_buffer.clear()

        if frame_count % 90 == 0:
            print(
                f"[{MODE} #{frame_count}] | Decode: {t_decode_ms:.1f}ms | Total: {t_total_ms:.1f}ms "
                f"(Infer: {t_infer_ms:.1f}ms | Blur: {t_blur_ms:.1f}ms | Enc: {t_encode_ms:.1f}ms) | "
                f"Throughput FPS: {pipeline_fps_window:.1f} (Compute Cap: {compute_capacity_fps:.1f}) | "
                f"Camera FPS: {cam_fps:.1f} | Dropped: {dropped} | VRAM: {vram_mb:.1f}MB | "
                f"frame #{frame_count} has {len(last_boxes)}"
            )

except KeyboardInterrupt:
    print("[*] Stopping on user interrupt...")
finally:
    if telemetry_buffer:
        with open(TELEMETRY_CSV, "a", encoding="utf-8") as f:
            f.writelines(telemetry_buffer)
    reader.release()
    if pipe and pipe.stdin:
        pipe.stdin.close()
        pipe.wait()
    print("[+] Clean shutdown complete.")