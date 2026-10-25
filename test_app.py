"""Offline regression tests: real image blurring, mocked YOLO and stream I/O."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

# Strategy and image tests do not need a CUDA/PyTorch installation.
with patch.dict(sys.modules, {"torch": Mock(), "psutil": Mock()}):
    import app


class ConfigTests(unittest.TestCase):
    def test_environment_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = app.PipelineConfig.from_env()
        self.assertEqual(config, app.PipelineConfig())
        self.assertEqual(config.approach, "C")
        self.assertEqual(config.head_ratio, 0.20)
        self.assertEqual(config.conf_thresh, 0.35)

    def test_environment_overrides(self):
        values = {
            "APPROACH": " b ", "PERSON_MODEL_PATH": "person.pt",
            "FACE_MODEL_PATH": "face.pt", "PLATE_MODEL_PATH": "plate.pt",
            "HEAD_RATIO": "0.3", "CONF_THRESH": "0.7", "BLUR_KERNEL": "50",
            "RTSP_URL": "rtmp://localhost/live/input", "OUT_RTMP": "rtmp://localhost/live/output",
        }
        with patch.dict(os.environ, values, clear=True):
            config = app.PipelineConfig.from_env()
        self.assertEqual(config.approach, "B")
        self.assertEqual(config.person_model_path, "person.pt")
        self.assertEqual(config.face_model_path, "face.pt")
        self.assertEqual(config.plate_model_path, "plate.pt")
        self.assertEqual((config.head_ratio, config.conf_thresh, config.blur_kernel), (0.3, 0.7, 50))
        self.assertEqual(config.rtsp_in, values["RTSP_URL"])
        self.assertEqual(config.rtmp_out, values["OUT_RTMP"])

    def test_invalid_configuration_fails(self):
        for changes in (
            {"approach": "D"}, {"head_ratio": 0}, {"head_ratio": 1.1},
            {"head_ratio": float("nan")}, {"conf_thresh": -0.1},
            {"conf_thresh": 1.1}, {"blur_kernel": 0}, {"imgsz": 0},
            {"stream_fps": float("inf")}, {"stream_fps": 0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                app.PipelineConfig(**changes)


class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = {}
        self.models = {}
        for name in ("person", "face", "plate"):
            path = Path(self.temp.name) / f"{name}.pt"
            path.touch()
            self.paths[name] = str(path)
            self.models[name] = Mock()
            self.set_boxes(name, [])
        self.yolo = Mock(side_effect=lambda path, **kwargs: self.models[Path(path).stem])
        self.patcher = patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=self.yolo)})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.frame = np.zeros((200, 300, 3), dtype=np.uint8)
        self.config = app.PipelineConfig(
            person_model_path=self.paths["person"], face_model_path=self.paths["face"],
        )

    def set_boxes(self, name, boxes):
        tensor = Mock()
        tensor.cpu.return_value.numpy.return_value = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.models[name].predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(xyxy=tensor))]

    def pipeline(self, approach="C", plates=False, device="cpu", **changes):
        config = replace(
            self.config, approach=approach,
            plate_model_path=self.paths["plate"] if plates else "", **changes,
        )
        with redirect_stdout(io.StringIO()):
            return app.PIIAnonymizer(config, device)

    def test_a_blurs_only_estimated_heads_and_loads_only_person(self):
        self.set_boxes("person", [[10, 20, 60, 120], [80, 50, 100, 100]])
        pipeline = self.pipeline("A", face_model_path="missing-face.pt")
        self.assertEqual(pipeline.collect_blur_boxes(self.frame), [[10, 20, 60, 40], [80, 50, 100, 60]])
        self.assertEqual(set(pipeline.models), {"person"})

    def test_b_preserves_full_face_boxes_and_ignores_person_path(self):
        faces = [[1.5, 2, 20, 30], [60, 80, 90, 100]]
        self.set_boxes("face", faces)
        pipeline = self.pipeline("B", person_model_path="missing-person.pt")
        self.assertEqual(pipeline.collect_blur_boxes(self.frame), faces)
        self.assertEqual(set(pipeline.models), {"face"})

    def test_c_keeps_unmatched_faces_and_adds_only_missing_heads(self):
        self.set_boxes("person", [[10, 20, 60, 120], [80, 50, 100, 150]])
        faces = [[5, 15, 25, 35], [200, 10, 220, 30]]
        self.set_boxes("face", faces)
        self.assertEqual(self.pipeline().collect_blur_boxes(self.frame), faces + [[80, 50, 100, 70]])

    def test_c_with_no_faces_estimates_every_head(self):
        self.set_boxes("person", [[10, 20, 60, 120], [80, 50, 100, 150]])
        self.assertEqual(self.pipeline().collect_blur_boxes(self.frame), [[10, 20, 60, 40], [80, 50, 100, 70]])

    def test_c_with_no_people_keeps_faces(self):
        self.set_boxes("face", [[5, 15, 25, 35]])
        self.assertEqual(self.pipeline().collect_blur_boxes(self.frame), [[5, 15, 25, 35]])

    def test_face_center_inclusive_boundary_and_overlapping_people(self):
        face = [5, 5, 15, 15]
        self.assertTrue(app.is_face_inside_person(face, [10, 10, 20, 100]))
        self.assertFalse(app.is_face_inside_person(face, [11, 10, 20, 100]))
        self.set_boxes("face", [face])
        self.set_boxes("person", [[10, 10, 20, 100], [0, 0, 20, 50]])
        self.assertEqual(self.pipeline().collect_blur_boxes(self.frame), [face])

    def test_plates_run_for_every_approach_with_original_frame(self):
        plate = [110, 120, 150, 140]
        self.set_boxes("plate", [plate])
        for approach in "ABC":
            with self.subTest(approach=approach):
                pipeline = self.pipeline(approach, plates=True, device=0)
                self.assertEqual(pipeline.collect_blur_boxes(self.frame), [plate])
                for name, model in pipeline.models.items():
                    args, kwargs = model.predict.call_args
                    self.assertIs(args[0], self.frame)
                    self.assertEqual(kwargs["device"], 0)
                    self.assertEqual(kwargs["conf"], 0.35)
                    self.assertEqual(kwargs["classes"], [0] if name == "person" else None)

    def test_disabled_plate_model_is_not_loaded(self):
        pipeline = self.pipeline()
        pipeline.collect_blur_boxes(self.frame)
        self.assertNotIn("plate", pipeline.models)
        self.models["plate"].predict.assert_not_called()

    def test_empty_results_are_supported(self):
        self.models["face"].predict.return_value = []
        self.models["person"].predict.return_value = [SimpleNamespace(boxes=None)]
        self.assertEqual(self.pipeline().collect_blur_boxes(self.frame), [])

    def test_paths_are_absolute_and_validated_before_yolo_is_called(self):
        self.pipeline()
        for args, kwargs in self.yolo.call_args_list:
            self.assertTrue(Path(args[0]).is_absolute())
            self.assertEqual(kwargs, {"task": "detect"})
        self.yolo.reset_mock()
        for changes in (
            {"person_model_path": "not-local.pt"},
            {"face_model_path": "https://example.com/face.pt"},
            {"person_model_path": ""},
            {"plate_model_path": "not-local-plate.pt"},
        ):
            with self.subTest(changes=changes), self.assertRaises(FileNotFoundError):
                app.PIIAnonymizer(replace(self.config, **changes), "cpu")
        self.yolo.assert_not_called()


class BlurTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 256, (60, 80, 3), dtype=np.uint8)

    def test_blur_matches_gaussian_and_leaves_other_pixels_unchanged(self):
        original = self.frame.copy()
        expected = original.copy()
        expected[5:35, 10:40] = cv2.GaussianBlur(original[5:35, 10:40], (51, 51), 30)
        result = app.apply_pii_blur(self.frame, [[10, 5, 40, 35]], kernel_size=50)
        self.assertIs(result, self.frame)
        np.testing.assert_array_equal(result, expected)
        self.assertLess(result[5:35, 10:40].var(), original[5:35, 10:40].var() / 10)

    def test_clamping_and_kernel_larger_than_roi(self):
        expected = cv2.GaussianBlur(self.frame[:2, :2], (51, 51), 30)
        app.apply_pii_blur(self.frame, [[-10, -20, 2, 2]])
        np.testing.assert_array_equal(self.frame[:2, :2], expected)
        original = self.frame.copy()
        app.apply_pii_blur(self.frame, [[-10, -20, 1000, 1000]])
        np.testing.assert_array_equal(self.frame, cv2.GaussianBlur(original, (51, 51), 30))

    def test_invalid_empty_or_tiny_regions_are_skipped(self):
        original = self.frame.copy()
        boxes = [[2, 2, 3, 5], [2, 2, 5, 3], [5, 5, 2, 2], [-8, -8, -2, -2],
                 [100, 100, 200, 200], [0, 0, float("nan"), 4]]
        app.apply_pii_blur(self.frame, boxes)
        app.apply_pii_blur(self.frame, [])
        np.testing.assert_array_equal(self.frame, original)


class StreamingTests(unittest.TestCase):
    def test_reader_delivers_final_buffered_frame_once_and_counts_drops(self):
        frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(3)]
        cap = Mock()
        cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        cap.get.return_value = 30
        with patch.object(app.cv2, "VideoCapture", return_value=cap):
            reader = app.FreshFrameReader("rtsp://test/input")
        self.addCleanup(reader.release)
        reader.thread.join(timeout=1)
        self.assertFalse(reader.thread.is_alive())
        success, frame, dropped, _ = reader.read_latest()
        self.assertTrue(success)
        self.assertIs(frame, frames[-1])
        self.assertEqual(dropped, 2)
        self.assertEqual(reader.read_latest()[:2], (False, None))
        cap.release.assert_called_once()

    def test_unreadable_input_releases_capture(self):
        cap = Mock()
        cap.read.return_value = (False, None)
        with patch.object(app.cv2, "VideoCapture", return_value=cap):
            with self.assertRaises(ConnectionError):
                app.FreshFrameReader("rtsp://test/input")
        cap.release.assert_called_once()

    def test_main_writes_blurred_frame_telemetry_and_releases_resources(self):
        frame = np.random.default_rng(0).integers(0, 256, (16, 24, 3), dtype=np.uint8)
        original = frame.copy()
        reader = Mock(width=24, height=16, raw_fps=90000)
        reader.read_latest.side_effect = [(True, frame, 2, 25), (False, None, 2, 25)]
        anonymizer = Mock()
        anonymizer.collect_blur_boxes.return_value = [[0, 0, 12, 16]]
        pipe = Mock()
        process = Mock()
        process.cpu_percent.return_value = 1.0
        process.memory_info.return_value = SimpleNamespace(rss=1024)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "telemetry.csv"
            config = app.PipelineConfig(telemetry_csv=str(csv_path), stream_fps=29.97)
            with patch.object(app.PipelineConfig, "from_env", return_value=config), \
                 patch.object(app, "PIIAnonymizer", return_value=anonymizer), \
                 patch.object(app, "FreshFrameReader", return_value=reader), \
                 patch.object(app.torch.cuda, "is_available", return_value=False), \
                 patch.object(app.psutil, "Process", return_value=process), \
                 patch.object(app.subprocess, "Popen", return_value=pipe) as popen, \
                 redirect_stdout(io.StringIO()):
                app.main()
            rows = csv_path.read_text().splitlines()
        expected = original.copy()
        expected[:, :12] = cv2.GaussianBlur(original[:, :12], (51, 51), 30)
        pipe.stdin.write.assert_called_once_with(expected.tobytes())
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(rows[0].split(",")), len(rows[1].split(",")))
        self.assertEqual(rows[1].split(",")[2], "C")
        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-r") + 1], "29.97")
        self.assertEqual(cmd[cmd.index("-maxrate") + 1], "16M")
        self.assertEqual(cmd[-1], config.rtmp_out)
        reader.release.assert_called_once()
        pipe.stdin.close.assert_called_once()
        pipe.wait.assert_called_once_with(timeout=5)

    def test_ffmpeg_shutdown_handles_broken_pipe_and_timeout(self):
        pipe = Mock()
        pipe.stdin.close.side_effect = BrokenPipeError
        pipe.wait.side_effect = [subprocess.TimeoutExpired("ffmpeg", 5),
                                 subprocess.TimeoutExpired("ffmpeg", 5), 0]
        app.close_ffmpeg(pipe)
        pipe.terminate.assert_called_once()
        pipe.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
