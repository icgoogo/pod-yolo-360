# pod-yolo-360

RTSP face pixelation with YOLO or YuNet, streamed to RTMP through FFmpeg.

## Select YuNet

The existing YOLO behavior remains the default. To use YuNet, add these options
to your existing `podman run` command **before the image name**:

```text
-e ANONYMIZATION_MODE=YUNET -e YUNET_IMGSZ=640 -e CONF_THRESH=0.6
```

The included [YuNet model](https://huggingface.co/opencv/face_detection_yunet)
is `face_detection_yunet_2023mar.onnx` (about 228 KiB), copied into the image by
the [Dockerfile](./Dockerfile). To download it again from the repository root:

```sh
curl -fL -o face_detection_yunet_2023mar.onnx https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
```

On Windows PowerShell, use `curl.exe` for that command.

YuNet runs on **CPU** through OpenCV DNN. The pip OpenCV wheels do not provide
CUDA inference, even inside the existing CUDA/PyTorch image. Use OpenCV 4.8 or
newer for this model; the existing unpinned OpenCV requirement installs a current
version on a fresh dependency build. No new Python dependencies are needed.

`YUNET_IMGSZ` limits the longest input side while preserving aspect ratio.
Detections are mapped back to the full video resolution before applying the
existing `BOX_MARGIN` and pixelation. `HEAD_RATIO` is ignored for YuNet because
it already detects faces. For 360 video, try `YUNET_IMGSZ=1280` if distant faces
are missed; this increases CPU work. The example confidence of `0.6` is a
starting point to tune against your footage; the code's default remains `0.30`.
`YUNET_MODEL` can override the model path. Capture, encoding, and CSV columns
remain the same, with `YUNET` recorded in the mode column.

## Rebuild using the existing Podman image tag

Run these on the machine where Podman and your current image are installed.
Replace the uppercase placeholders with your actual values.

1. Find your existing image tag, container name, and **old image ID**. Save the
   old ID before building, so you can remove that specific version afterward:

   ```sh
   podman images --no-trunc
   podman ps -a
   ```

2. From this repository directory, build with the **same repository and tag**:

   ```sh
   podman build --layers -t YOUR_EXISTING_IMAGE:TAG .
   ```

   Keep the [requirements](./requirements.txt) and base image unchanged to reuse
   the large dependency layers. Avoid `--no-cache`. The YuNet changes only affect
   the small application/model layers after dependency installation.

3. Recreate the container using your original `podman run` command, adding
   `--replace`, keeping the same `--name`, and setting the YuNet options above.
   Keep your existing ports/network, volume mounts, stream URLs, and other flags.
   This interrupts the running stream and replaces the container; preserve any
   needed files in its writable layer first. Files in mounted storage persist.
   Merely restarting the old container does **not** load the rebuilt image.
   For example, on a Linux Podman host with MediaMTX on localhost:

   ```sh
   podman run -d --replace --name YOUR_EXISTING_CONTAINER --network host -v theta-output:/app/output -e TELEMETRY_CSV=/app/output/pipeline_telemetry.csv -e ANONYMIZATION_MODE=YUNET -e YUNET_IMGSZ=640 -e CONF_THRESH=0.6 YOUR_EXISTING_IMAGE:TAG
   podman logs --tail 50 YOUR_EXISTING_CONTAINER
   ```

   Adapt the example to your existing run configuration; it uses a named volume
   to persist telemetry and the application's default local RTSP/RTMP URLs.

4. After confirming the new container works, remove the recorded **old** image:

   ```sh
   podman rmi OLD_IMAGE_ID
   podman system df
   ```

   Skip removal if the build reused the same image ID. If another container still
   references the old image, Podman will refuse removal; inspect that container
   rather than forcing removal. Shared layers remain available to the new image.

Images are immutable: rebuilding moves the existing tag to a new image version;
it cannot overwrite the old image in place. Cached layers avoid duplicating the
large CUDA/PyTorch dependencies, but some temporary extra space is still needed.
The displayed image size includes shared layers and is not all additional disk
usage. Build cache may retain layers after image removal. Avoid broad
`podman system prune -a --volumes` cleanup: it can remove unrelated data.
