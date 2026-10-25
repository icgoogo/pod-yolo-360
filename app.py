import math
import os
from pathlib import Path
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import psutil
import torch


@dataclass(frozen=True)
class PipelineConfig:
    approach: str = "C"
    person_model_path: str = "yolo11n.pt"
    face_model_path: str = "yolov8n-face.pt"
    plate_model_path: str = ""
    head_ratio: float = 0.20
    conf_thresh: float = 0.35
    blur_kernel: int = 51
    imgsz: int = 640
    rtsp_in: str = "rtsp://localhost:8554/live/theta"
    rtmp_out: str = "rtmp://localhost:1935/live/theta_blurred"
    telemetry_csv: str = "pipeline_telemetry.csv"
    target_bitrate: str = "16M"
    stream_fps: float = 30.0

    # Approach A (Detects bodies -> estimates top 20% as head box)
    # B (face only)
    # C + License Plate: 
        # Detects all faces.
        # Detects all bodies.       
        # Matches faces to bodies.
        # For any body without a detected face, falls back to estimating the head box.
        # Appends any detected license plates (plate_model_path).
    def __post_init__(self):
        if self.approach not in {"A", "B", "C"}:
            raise ValueError("APPROACH must be A, B, or C")
        if not 0.0 < self.head_ratio <= 1.0:
            raise ValueError("HEAD_RATIO must be in (0, 1]")
        if not 0.0 <= self.conf_thresh <= 1.0:
            raise ValueError("CONF_THRESH must be in [0, 1]")
        if self.blur_kernel < 1:
            raise ValueError("BLUR_KERNEL must be a positive integer")
        if self.imgsz < 1:
            raise ValueError("YOLO_IMGSZ must be a positive integer")
        if not math.isfinite(self.stream_fps) or self.stream_fps <= 0:
            raise ValueError("STREAM_FPS must be finite and positive")

    @classmethod
    def from_env(cls):
        return cls(
            approach=os.getenv("APPROACH", "B").strip().upper(),
            person_model_path=os.getenv("PERSON_MODEL_PATH", "yolo11n.pt"),
            face_model_path=os.getenv("FACE_MODEL_PATH", "yolov8n-face.pt"),
            plate_model_path=os.getenv("PLATE_MODEL_PATH", "").strip(),
            head_ratio=float(os.getenv("HEAD_RATIO", "0.20")),
            conf_thresh=float(os.getenv("CONF_THRESH", "0.35")),
            blur_kernel=int(os.getenv("BLUR_KERNEL", "51")),
            imgsz=int(os.getenv("YOLO_IMGSZ", "640")),
            rtsp_in=os.getenv("RTSP_URL", "rtsp://localhost:8554/live/theta"),
            rtmp_out=os.getenv("OUT_RTMP", "rtmp://localhost:1935/live/theta_blurred"),
            telemetry_csv=os.getenv("TELEMETRY_CSV", "pipeline_telemetry.csv"),
            target_bitrate=os.getenv("OUT_BITRATE", "16M"),
            stream_fps=float(os.getenv("STREAM_FPS", "30.0")),
        )

def local_model_path(value, variable):
    """Reject missing weights before YOLO can attempt an automatic download."""
    path = Path(value).expanduser()
    if not value or not path.is_file():
        raise FileNotFoundError(f"{variable} must point to an existing local weight file: {value!r}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"{variable} must point to a local .pt weight file: {value!r}")
    return str(path.resolve())


def is_face_inside_person(face_box, person_box):
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def estimate_head_box(person_box, head_ratio):
    px1, py1, px2, py2 = person_box
    return [px1, py1, px2, py1 + (py2 - py1) * head_ratio]


class PIIAnonymizer:
    """Load only the detectors required by the selected strategy."""

    def __init__(self, config, device):
        self.config = config
        self.device = device
        paths = {}
        if config.approach in {"A", "C"}:
            paths["person"] = local_model_path(config.person_model_path, "PERSON_MODEL_PATH")
        if config.approach in {"B", "C"}:
            paths["face"] = local_model_path(config.face_model_path, "FACE_MODEL_PATH")
        if config.plate_model_path:
            paths["plate"] = local_model_path(config.plate_model_path, "PLATE_MODEL_PATH")

        # Validate every required path before importing or constructing YOLO.
        from ultralytics import YOLO

        self.models = {}
        for name, path in paths.items():
            print(f"[*] Loading {name} detector from: {path}")
            self.models[name] = YOLO(path, task="detect")

    def _detect(self, name, frame):
        if name not in self.models:
            return []
        results = self.models[name].predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf_thresh,
            classes=[0] if name == "person" else None,
            device=self.device,
            verbose=False,
        )
        boxes = []
        for result in results:
            if result.boxes is not None:
                # like append  $[x_1, y_1, x_2, y_2]$ It safely moves bounding box coordinates off the GPU, 
                # converts them into standard Python numbers, and flattens them into a master list for OpenCV to blur.
                boxes.extend(result.boxes.xyxy.cpu().numpy().tolist())
        return boxes

    def collect_blur_boxes(self, frame):
        # All inference sees the original pixels, including plate inference.
        faces = self._detect("face", frame)
        people = self._detect("person", frame)
        blur_boxes = list(faces)
        for person in people:
            if not any(is_face_inside_person(face, person) for face in faces):
                blur_boxes.append(estimate_head_box(person, self.config.head_ratio))
        blur_boxes.extend(self._detect("plate", frame))
        return blur_boxes


def apply_pii_blur(frame, blur_boxes, kernel_size=51):
    h, w = frame.shape[:2]
    for box in blur_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue
        roi = frame[y1:y2, x1:x2]
        small = cv2.resize(roi, (8, 8), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    return frame


class FreshFrameReader:
    """Drain the input continuously and process each available frame once."""

    def __init__(self, src):
        # Bound network reads so shutdown can wait for the capture thread safely.
        self.cap = cv2.VideoCapture(
            src,
            cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000],
        )
        if not self.cap.isOpened():
            self.cap.release()
            raise ConnectionError(f"Unable to open video input at {src}")
        ret, self.frame = self.cap.read()
        if not ret or self.frame is None:
            self.cap.release()
            raise ConnectionError(f"No initial frame received from {src}")
        self.height, self.width = self.frame.shape[:2]
        self.raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.running = True
        self.new_frame = True
        self.condition = threading.Condition()
        self.dropped_frames = 0
        self.incoming_fps = 0.0
        self.last_frame_time = time.perf_counter()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                now = time.perf_counter()
                dt = now - self.last_frame_time
                self.last_frame_time = now
                with self.condition:
                    if self.new_frame:
                        self.dropped_frames += 1
                    self.frame = frame
                    if dt > 0:
                        inst_fps = 1.0 / dt
                        self.incoming_fps = (
                            (0.9 * self.incoming_fps + 0.1 * inst_fps)
                            if self.incoming_fps > 0
                            else inst_fps
                        ) 
                    self.new_frame = True
                    self.condition.notify_all()
        finally:
            # Only the reader thread releases the capture, avoiding read/release races.
            self.cap.release()
            with self.condition:
                self.running = False
                self.condition.notify_all()

    def read_latest(self):
        with self.condition:
            while self.running and not self.new_frame:
                self.condition.wait(timeout=0.1)
            if not self.new_frame:
                return False, None, 0, self.incoming_fps
            self.new_frame = False
            dropped_delta = self.dropped_frames
            self.dropped_frames = 0
            return True, self.frame, dropped_delta, self.incoming_fps

    def release(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.thread.join(timeout=6.0)


def build_ffmpeg_command(config, width, height, fps):
    return [
        "/usr/bin/ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
        "-b:v", config.target_bitrate, "-maxrate", config.target_bitrate,
        "-bufsize", config.target_bitrate,
        "-pix_fmt", "yuv420p", "-f", "flv", config.rtmp_out,
    ]


def close_ffmpeg(pipe):
    if pipe is None:
        return
    if pipe.stdin:
        try:
            pipe.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    try:
        pipe.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pipe.terminate()
        try:
            pipe.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pipe.kill()
            pipe.wait()


def main():
    config = PipelineConfig.from_env()
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    device = 0 if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        print(f"[+] Running inference on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[!] CUDA unavailable; running inference on CPU.")
    print(
        f"[CONFIG] Approach: {config.approach} | ImgSz: {config.imgsz} | "
        f"Conf: {config.conf_thresh} | HeadRatio: {config.head_ratio:.0%} | "
        f"BlurKernel: {config.blur_kernel} | Plates: {bool(config.plate_model_path)}"
    )
    anonymizer = PIIAnonymizer(config, device)
    reader = None
    pipe = None
    telemetry = None
    telemetry_buffer = []
    process = psutil.Process()
    process.cpu_percent(interval=None)
    try:
        print(f"[*] Connecting to stream: {config.rtsp_in}")
        reader = FreshFrameReader(config.rtsp_in)
        width, height = reader.width, reader.height
        # Some RTSP sources report the 90 kHz RTP clock as their frame rate.
        raw_fps = reader.raw_fps
        fps = float(raw_fps) if raw_fps and 1.0 <= raw_fps <= 120.0 else config.stream_fps
        print(f"[+] Connected! Resolution: {width}x{height} @ {fps:.2f} FPS")
        pipe = subprocess.Popen(
            build_ffmpeg_command(config, width, height, fps),
            stdin=subprocess.PIPE,
        )
        print(f"[+] Streaming live to: {config.rtmp_out} (Bitrate: {config.target_bitrate})")
        telemetry = open(config.telemetry_csv, "w", encoding="utf-8")
        # Retain the existing CSV schema; mode now records A, B, or C.
        telemetry.write(
            "timestamp,frame_idx,mode,imgsz,t_decode_ms,t_infer_ms,t_blur_ms,"
            "t_encode_ms,t_total_ms,fps,cam_fps,boxes_count,dropped_frames,vram_mb,cpu_pct,ram_mb\n"
        )
        frame_count = 0
        while True:
            loop_start = time.perf_counter()
            t0 = time.perf_counter()
            success, frame, dropped, cam_fps = reader.read_latest()
            if not success or frame is None:
                print("[-] Input stream ended. Exiting...")
                break
            t_decode_ms = (time.perf_counter() - t0) * 1000.0
            if frame.shape[:2] != (height, width):
                raise ValueError("Input resolution changed; restart the pipeline for the new resolution")
            work_frame = frame.copy()

            if device != "cpu":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            blur_boxes = anonymizer.collect_blur_boxes(work_frame)
            if device != "cpu":
                torch.cuda.synchronize()
            t_infer_ms = (time.perf_counter() - t1) * 1000.0

            t2 = time.perf_counter()
            apply_pii_blur(work_frame, blur_boxes, config.blur_kernel)
            t_blur_ms = (time.perf_counter() - t2) * 1000.0

            t3 = time.perf_counter()
            try:
                pipe.stdin.write(work_frame.tobytes())
            except (BrokenPipeError, OSError):
                print("[-] FFmpeg pipe broken. MediaMTX disconnected.")
                break
            t_encode_ms = (time.perf_counter() - t3) * 1000.0
            loop_end = time.perf_counter()
            t_total_ms = (loop_end - loop_start) * 1000.0
            current_fps = 1000.0 / t_total_ms if t_total_ms > 0 else fps
            vram_mb = torch.cuda.memory_reserved(0) / (1024 * 1024) if device != "cpu" else 0.0
            cpu_pct = process.cpu_percent()
            ram_mb = process.memory_info().rss / (1024 * 1024)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            telemetry_buffer.append(
                f"{ts},{frame_count},{config.approach},{config.imgsz},{t_decode_ms:.2f},"
                f"{t_infer_ms:.2f},{t_blur_ms:.2f},{t_encode_ms:.2f},{t_total_ms:.2f},"
                f"{current_fps:.1f},{cam_fps:.1f},{len(blur_boxes)},{dropped},"
                f"{vram_mb:.1f},{cpu_pct:.1f},{ram_mb:.1f}\n"
            )
            frame_count += 1
            if frame_count % 30 == 0:
                telemetry.writelines(telemetry_buffer)
                telemetry.flush()
                telemetry_buffer.clear()
            if frame_count % 90 == 0:
                print(
                    f"[{config.approach} #{frame_count}] Total: {t_total_ms:.1f}ms "
                    f"(Decode: {t_decode_ms:.1f}ms | Infer: {t_infer_ms:.1f}ms | "
                    f"Blur: {t_blur_ms:.1f}ms | Enc: {t_encode_ms:.1f}ms) | "
                    f"FPS: {current_fps:.1f} | Camera FPS: {cam_fps:.1f} | "
                    f"Dropped: {dropped} | VRAM: {vram_mb:.1f}MB"
                )
    except KeyboardInterrupt:
        print("[*] Stopping on user interrupt...")
    finally:
        try:
            if telemetry is not None:
                try:
                    telemetry.writelines(telemetry_buffer)
                finally:
                    telemetry.close()
        finally:
            try:
                if reader is not None:
                    reader.release()
            finally:
                close_ffmpeg(pipe)
        print("[+] Clean shutdown complete.")


if __name__ == "__main__":
    main()
