# pod-yolo-360

Read RTSP or RTMP video from MediaMTX, detect PII with local YOLO weights,
apply Gaussian blur, and publish the processed video to MediaMTX over RTMP.
Inference uses CUDA device 0 when available, with a CPU fallback.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APPROACH` | `C` | `A`: estimated heads from people; `B`: faces; `C`: faces plus estimated heads for people without a detected face |
| `PERSON_MODEL_PATH` | `yolo11n.pt` | Local person weights; class 0 only; required for A/C |
| `FACE_MODEL_PATH` | `yolov8n-face.pt` | Local face weights; required for B/C |
| `PLATE_MODEL_PATH` | empty | Local plate weights; empty disables plates; enabled in every approach when set |
| `HEAD_RATIO` | `0.20` | Top fraction of the person box used for head estimation, in `(0, 1]` |
| `CONF_THRESH` | `0.35` | Detection confidence threshold, in `[0, 1]` |
| `BLUR_KERNEL` | `51` | Positive Gaussian kernel size; even values round up to the next odd value; sigma is 30 |
| `RTSP_URL` | `rtsp://localhost:8554/live/theta` | Input stream URL (RTSP or RTMP) |
| `OUT_RTMP` | `rtmp://localhost:1935/live/theta_blurred` | Processed output URL |
| `YOLO_IMGSZ` | `640` | Detector inference image size |
| `STREAM_FPS` | `30.0` | Output FPS when input metadata is invalid |
| `OUT_BITRATE` | `16M` | FFmpeg target and maximum video bitrate |
| `TELEMETRY_CSV` | `pipeline_telemetry.csv` | Per-frame timing and resource metrics; the `mode` column records A/B/C |

Supply the required `.pt` files before starting. All selected model paths are
validated locally before YOLO is loaded or the stream is opened; missing or
invalid weights stop startup instead of triggering a download or silently
disabling anonymization. Plate weights are not included in this repository.

Install [requirements.txt](./requirements.txt), a compatible PyTorch runtime,
and FFmpeg on `PATH`, then run `python app.py`. The [Dockerfile](./Dockerfile)
provides the CUDA runtime and FFmpeg and copies local `.pt` weights into the image.
For example, in PowerShell:

```powershell
$env:APPROACH = "C"
$env:PERSON_MODEL_PATH = "yolo11n.pt"
$env:FACE_MODEL_PATH = "yolov8n-face.pt"
$env:PLATE_MODEL_PATH = "" # Set to a local plate weight file to enable plate detection.
$env:RTSP_URL = "rtsp://localhost:8554/live/theta"
$env:OUT_RTMP = "rtmp://localhost:1935/live/theta_blurred"
python app.py
```

`APPROACH` replaces the old `ANONYMIZATION_MODE`/`YOLO_MODEL` configuration.
Each output frame receives fresh inference; slow processing drops older input
frames to keep latency down. The output carries video only. The process exits
when input ends or publishing fails; restart it to reconnect.

Run `python -m unittest -v test_app` for offline regression tests. These use real
NumPy/OpenCV image processing and mock model inference, CUDA, and streaming I/O.
