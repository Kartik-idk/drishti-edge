#!/usr/bin/env python3
"""
Drishti-Edge: Vision Pipeline & Privacy Redaction Benchmark
Phase 0 - Standalone CPU Latency & Throughput Evaluation

This script evaluates video decoding, HSV-based environmental lighting deficit
calculation, CPU-based detection inference, and adaptive Gaussian blur redaction.

HEADLESS OPERATION ONLY:
No GUI windowing functions (cv2.imshow, cv2.waitKey, cv2.namedWindow) are called.
Unredacted frames, identifiable crops, and unredacted videos are never persisted.

IMPORTANT AUDIT NOTICE:
When run with COCO-pretrained weights (e.g. yolov8n.pt), the model acts strictly as
an OBJECT-DETECTION INFERENCE PROXY to measure raw CPU latency. It does NOT provide
dedicated face or license-plate detection.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)


class RedactionBenchmark:
    """Benchmark runner for Drishti-Edge privacy redaction and telemetry pipeline."""

    def __init__(
        self,
        video_path: str,
        max_frames: Optional[int] = 300,
        warmup_frames: int = 10,
        face_model_path: Optional[str] = None,
        plate_model_path: Optional[str] = None,
        proxy_model_path: Optional[str] = None,
        conf_threshold: float = 0.4,
    ):
        self.video_path = Path(video_path)
        self.max_frames = max_frames
        self.warmup_frames = warmup_frames
        self.face_model_path = face_model_path
        self.plate_model_path = plate_model_path
        self.proxy_model_path = proxy_model_path
        self.conf_threshold = conf_threshold

        # Detectors
        self.face_detector = None
        self.plate_detector = None
        self.proxy_detector = None
        self.face_detector_type = "none"
        self.plate_detector_type = "none"
        self.proxy_detector_type = "none"

        # Initialize detectors
        self._init_detectors()

    def _init_detectors(self):
        """Initialize detection models on CPU if weights are available."""
        # 1. Face Detector
        if self.face_model_path and Path(self.face_model_path).exists():
            path_str = str(self.face_model_path)
            if path_str.endswith(".onnx"):
                try:
                    self.face_detector = cv2.FaceDetectorYN.create(
                        model=path_str,
                        config="",
                        input_size=(640, 640),
                        score_threshold=self.conf_threshold,
                        backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                        target_id=cv2.dnn.DNN_TARGET_CPU,
                    )
                    self.face_detector_type = "opencv_yunet_onnx"
                    print(f"[Detector] Loaded Face Model (YuNet ONNX): {path_str}")
                except Exception as e:
                    print(f"[Detector Error] Failed to load YuNet ONNX: {e}")
            elif path_str.endswith(".pt"):
                try:
                    from ultralytics import YOLO
                    self.face_detector = YOLO(path_str)
                    self.face_detector_type = "ultralytics_yolo_pt"
                    print(f"[Detector] Loaded Face Model (Ultralytics): {path_str}")
                except Exception as e:
                    print(f"[Detector Error] Failed to load YOLO face model: {e}")
        else:
            print("[Detector] Dedicated face model: None (weights not provided or file not found).")

        # 2. License Plate Detector
        if self.plate_model_path and Path(self.plate_model_path).exists():
            path_str = str(self.plate_model_path)
            if path_str.endswith(".pt"):
                try:
                    from ultralytics import YOLO
                    self.plate_detector = YOLO(path_str)
                    self.plate_detector_type = "ultralytics_yolo_pt"
                    print(f"[Detector] Loaded License Plate Model (Ultralytics): {path_str}")
                except Exception as e:
                    print(f"[Detector Error] Failed to load YOLO plate model: {e}")
        else:
            print("[Detector] Dedicated license plate model: None (weights not provided or file not found).")

        # 3. CPU Inference Proxy Detector (e.g. standard YOLOv8n for baseline latency)
        if self.proxy_model_path and Path(self.proxy_model_path).exists():
            path_str = str(self.proxy_model_path)
            try:
                from ultralytics import YOLO
                self.proxy_detector = YOLO(path_str)
                self.proxy_detector_type = "ultralytics_coco_proxy"
                print(f"[Detector Proxy] Loaded YOLO Proxy Model: {path_str}")
                print("                 NOTE: Acts as CPU latency proxy only; does NOT detect faces or plates.")
            except Exception as e:
                print(f"[Detector Error] Failed to load proxy model: {e}")

    def run_detection(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, str]]:
        """Run CPU inference.
        
        Returns a list of bounding boxes: (x1, y1, x2, y2, label).
        """
        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int, str]] = []

        # 1. Face detection
        if self.face_detector_type == "opencv_yunet_onnx" and self.face_detector:
            try:
                self.face_detector.setInputSize((w, h))
                _, detections = self.face_detector.detect(frame)
                if detections is not None:
                    for det in detections:
                        confidence = det[-1]
                        if confidence >= self.conf_threshold:
                            x, y, bw, bh = det[0:4].astype(int)
                            x1 = max(0, x)
                            y1 = max(0, y)
                            x2 = min(w, x + bw)
                            y2 = min(h, y + bh)
                            if x2 > x1 and y2 > y1:
                                boxes.append((x1, y1, x2, y2, "face"))
            except Exception as e:
                print(f"[Inference Error - Face]: {e}")

        elif self.face_detector_type == "ultralytics_yolo_pt" and self.face_detector:
            try:
                results = self.face_detector(frame, device="cpu", verbose=False, conf=self.conf_threshold)
                for r in results:
                    for b in r.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        x1 = max(0, coords[0])
                        y1 = max(0, coords[1])
                        x2 = min(w, coords[2])
                        y2 = min(h, coords[3])
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2, "face"))
            except Exception as e:
                print(f"[Inference Error - YOLO Face]: {e}")

        # 2. License Plate detection
        if self.plate_detector_type == "ultralytics_yolo_pt" and self.plate_detector:
            try:
                results = self.plate_detector(frame, device="cpu", verbose=False, conf=self.conf_threshold)
                for r in results:
                    for b in r.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        x1 = max(0, coords[0])
                        y1 = max(0, coords[1])
                        x2 = min(w, coords[2])
                        y2 = min(h, coords[3])
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2, "license_plate"))
            except Exception as e:
                print(f"[Inference Error - Plate]: {e}")

        # 3. Proxy detection (when dedicated face/plate models are not loaded)
        if self.proxy_detector_type == "ultralytics_coco_proxy" and self.proxy_detector:
            try:
                results = self.proxy_detector(frame, device="cpu", verbose=False, conf=self.conf_threshold)
                for r in results:
                    for b in r.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        x1 = max(0, coords[0])
                        y1 = max(0, coords[1])
                        x2 = min(w, coords[2])
                        y2 = min(h, coords[3])
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2, "proxy_object"))
            except Exception as e:
                print(f"[Inference Error - Proxy]: {e}")

        return boxes

    @staticmethod
    def apply_adaptive_gaussian_blur(frame: np.ndarray, boxes: List[Tuple[int, int, int, int, str]]) -> np.ndarray:
        """Apply adaptive Gaussian blur to detected bounding box regions.
        
        Dynamically calculates kernel size proportional to bounding box dimensions,
        guaranteeing odd kernel dimensions and preventing ROI out-of-bounds or size errors.
        """
        img_h, img_w = frame.shape[:2]
        redacted = frame.copy()

        for (x1, y1, x2, y2, label) in boxes:
            roi_w = x2 - x1
            roi_h = y2 - y1

            if roi_w <= 0 or roi_h <= 0:
                continue

            # Ensure ROI coordinates are within image boundaries
            x1_c = max(0, min(img_w - 1, x1))
            x2_c = max(x1_c + 1, min(img_w, x2))
            y1_c = max(0, min(img_h - 1, y1))
            y2_c = max(y1_c + 1, min(img_h, y2))

            roi = redacted[y1_c:y2_c, x1_c:x2_c]
            curr_h, curr_w = roi.shape[:2]
            if curr_h <= 0 or curr_w <= 0:
                continue

            # Sub-3 pixel regions cannot sustain a 3x3 kernel; apply solid redaction
            if curr_w < 3 or curr_h < 3:
                redacted[y1_c:y2_c, x1_c:x2_c] = 0
                continue

            # Calculate adaptive kernel size (~50% of bbox width/height, odd)
            kw = max(3, int(curr_w * 0.5) | 1)
            kh = max(3, int(curr_h * 0.5) | 1)

            # Kernel size must not exceed the ROI dimension and must remain odd
            if kw > curr_w:
                kw = curr_w if (curr_w % 2 != 0) else curr_w - 1
            if kh > curr_h:
                kh = curr_h if (curr_h % 2 != 0) else curr_h - 1

            kw = max(3, kw | 1)
            kh = max(3, kh | 1)
            kw = min(kw, 101) | 1
            kh = min(kh, 101) | 1

            blurred = cv2.GaussianBlur(roi, (kw, kh), sigmaX=30, sigmaY=30)
            redacted[y1_c:y2_c, x1_c:x2_c] = blurred

        return redacted

    @staticmethod
    def calculate_lighting_deficit(frame: np.ndarray) -> Tuple[float, float]:
        """Convert BGR frame to HSV, extract V channel mean, and compute lighting deficit.
        
        Formula: lighting_deficit = 1.0 - (mean_v / 255.0)
        Returns: (mean_v, lighting_deficit)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        mean_v = float(np.mean(v_channel))
        lighting_deficit = 1.0 - (mean_v / 255.0)
        return mean_v, lighting_deficit

    def execute(self) -> Dict[str, float]:
        """Execute the multi-stage benchmark and collect timing metrics."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.video_path}")

        try:
            src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print("\n=======================================================")
            print("  DRISHTI-EDGE: PHASE 0 VISION BENCHMARK EXECUTION")
            print("=======================================================")
            print(f"Video File:           {self.video_path}")
            print(f"Source Resolution:    {src_width} x {src_height}")
            print(f"Source FPS:           {src_fps:.2f}")
            print(f"Total Video Frames:   {total_video_frames}")
            print(f"Target Max Frames:    {self.max_frames}")
            print(f"Warmup Frames:        {self.warmup_frames}")
            print(f"Face Detector:        {self.face_detector_type}")
            print(f"Plate Detector:       {self.plate_detector_type}")
            print(f"Proxy Detector:       {self.proxy_detector_type}")
            print(f"Execution Mode:       Headless CPU")
            print("=======================================================\n")

            # Warm-up phase (excluded from timing statistics)
            if self.warmup_frames > 0:
                print(f"[Warmup] Executing {self.warmup_frames} warm-up frames (excluded from statistics)...")
                warmup_count = 0
                while warmup_count < self.warmup_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    _ = self.calculate_lighting_deficit(frame)
                    boxes = self.run_detection(frame)
                    _ = self.apply_adaptive_gaussian_blur(frame, boxes)
                    warmup_count += 1
                print("[Warmup] Completed. Beginning timed benchmark run...\n")

            # Metrics collection lists (times in milliseconds)
            decode_times_ms: List[float] = []
            lighting_times_ms: List[float] = []
            detect_times_ms: List[float] = []
            redact_times_ms: List[float] = []
            total_times_ms: List[float] = []
            v_mean_values: List[float] = []
            deficit_values: List[float] = []

            total_detections = 0
            frames_processed = 0

            bench_start = time.perf_counter()

            while True:
                if self.max_frames and frames_processed >= self.max_frames:
                    break

                # Stage A: Video Decoding
                t0 = time.perf_counter()
                ret, frame = cap.read()
                t1 = time.perf_counter()

                if not ret or frame is None:
                    break

                decode_ms = (t1 - t0) * 1000.0

                # Stage B: Environmental Lighting Telemetry
                t2 = time.perf_counter()
                mean_v, deficit = self.calculate_lighting_deficit(frame)
                t3 = time.perf_counter()
                lighting_ms = (t3 - t2) * 1000.0

                # Stage C: Object Detection
                t4 = time.perf_counter()
                boxes = self.run_detection(frame)
                t5 = time.perf_counter()
                detect_ms = (t5 - t4) * 1000.0

                # Stage D: Privacy Redaction (Gaussian Blur)
                t6 = time.perf_counter()
                _ = self.apply_adaptive_gaussian_blur(frame, boxes)
                t7 = time.perf_counter()
                redact_ms = (t7 - t6) * 1000.0

                # Stage E: End-to-end frame processing time
                e2e_ms = (t7 - t0) * 1000.0

                # Accumulate
                decode_times_ms.append(decode_ms)
                lighting_times_ms.append(lighting_ms)
                detect_times_ms.append(detect_ms)
                redact_times_ms.append(redact_ms)
                total_times_ms.append(e2e_ms)

                v_mean_values.append(mean_v)
                deficit_values.append(deficit)
                total_detections += len(boxes)
                frames_processed += 1

                if frames_processed % 50 == 0:
                    cur_fps = frames_processed / (time.perf_counter() - bench_start)
                    print(f"Processed {frames_processed} frames | Instantaneous Throughput: {cur_fps:.2f} FPS")

            bench_end = time.perf_counter()
            total_elapsed_sec = bench_end - bench_start

        finally:
            cap.release()

        if frames_processed == 0:
            raise RuntimeError("No frames were processed during the benchmark.")

        # Compute summary statistics
        avg_decode = float(np.mean(decode_times_ms))
        avg_lighting = float(np.mean(lighting_times_ms))
        avg_detect = float(np.mean(detect_times_ms))
        avg_redact = float(np.mean(redact_times_ms))
        avg_total = float(np.mean(total_times_ms))
        median_total = float(np.median(total_times_ms))
        p95_total = float(np.percentile(total_times_ms, 95))
        throughput_fps = frames_processed / total_elapsed_sec

        avg_mean_v = float(np.mean(v_mean_values))
        avg_deficit = float(np.mean(deficit_values))

        results = {
            "frames_processed": frames_processed,
            "total_elapsed_sec": total_elapsed_sec,
            "src_fps": src_fps,
            "src_width": src_width,
            "src_height": src_height,
            "throughput_fps": throughput_fps,
            "avg_decode_ms": avg_decode,
            "avg_lighting_ms": avg_lighting,
            "avg_detect_ms": avg_detect,
            "avg_redact_ms": avg_redact,
            "avg_total_ms": avg_total,
            "median_total_ms": median_total,
            "p95_total_ms": p95_total,
            "total_detections": total_detections,
            "avg_mean_v": avg_mean_v,
            "avg_lighting_deficit": avg_deficit,
        }

        # Print detailed report
        print("\n=======================================================")
        print("  DRISHTI-EDGE: BENCHMARK TIMING RESULTS (CPU)")
        print("=======================================================")
        print(f"Processed Frames:          {frames_processed}")
        print(f"Total Wall Time:           {total_elapsed_sec:.2f} s")
        print(f"End-to-End Throughput:     {throughput_fps:.2f} FPS (Provisional Target: 15.0 FPS)")
        print(f"Source Video FPS:          {src_fps:.2f} FPS")
        print("-------------------------------------------------------")
        print("STAGE LATENCIES (Average per Frame):")
        print(f"  Stage A - Decoding:      {avg_decode:.2f} ms ({avg_decode/avg_total*100:.1f}%)")
        print(f"  Stage B - Lighting (HSV):{avg_lighting:.2f} ms ({avg_lighting/avg_total*100:.1f}%)")
        print(f"  Stage C - Detection:     {avg_detect:.2f} ms ({avg_detect/avg_total*100:.1f}%)")
        print(f"  Stage D - Redaction:     {avg_redact:.2f} ms ({avg_redact/avg_total*100:.1f}%)")
        print("-------------------------------------------------------")
        print("FRAME LATENCY DISTRIBUTION:")
        print(f"  Average End-to-End:      {avg_total:.2f} ms")
        print(f"  Median (50th percentile):{median_total:.2f} ms")
        print(f"  95th Percentile (p95):   {p95_total:.2f} ms")
        print("-------------------------------------------------------")
        print("ENVIRONMENTAL TELEMETRY:")
        print(f"  Average Mean V (0-255):  {avg_mean_v:.2f}")
        print(f"  Avg Lighting Deficit:    {avg_deficit:.4f}")
        print("=======================================================\n")

        return results


def generate_synthetic_test_video(output_path: str, duration_sec: int = 5, fps: int = 30) -> str:
    """Generate a synthetic test MP4 video containing dynamic shapes and simulated objects
    for testing pipeline functionality when real street video is not yet provided.
    
    NOTE: Performance on synthetic video tests script mechanics only, NOT real-world accuracy!
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    print(f"[Synthetic] Generating {total_frames} synthetic test frames ({width}x{height} @ {fps}fps)...")

    for i in range(total_frames):
        bg_intensity = int(80 + 70 * np.sin(i / 15.0))
        frame = np.full((height, width, 3), bg_intensity, dtype=np.uint8)

        # Draw simulated moving vehicles (rectangles)
        car_x = int((i * 12) % (width + 200)) - 100
        car_y = 400
        cv2.rectangle(frame, (car_x, car_y), (car_x + 180, car_y + 90), (40, 40, 180), -1)

        # Draw simulated license plate
        plate_x = car_x + 50
        plate_y = car_y + 50
        cv2.rectangle(frame, (plate_x, plate_y), (plate_x + 80, plate_y + 25), (240, 240, 240), -1)

        # Draw simulated moving pedestrians
        ped_x = int(width - (i * 8) % (width + 100))
        ped_y = 350
        cv2.circle(frame, (ped_x, ped_y), 30, (180, 180, 220), -1)
        cv2.rectangle(frame, (ped_x - 20, ped_y + 30), (ped_x + 20, ped_y + 110), (120, 60, 60), -1)

        out.write(frame)

    out.release()
    print(f"[Synthetic] Video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Drishti-Edge Vision Pipeline & Privacy Benchmark")
    parser.add_argument("--video", type=str, default="", help="Path to input MP4 video file")
    parser.add_argument("--max-frames", type=int, default=150, help="Maximum frames to process")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup frames")
    parser.add_argument("--model-face", type=str, default="models/yolov8n-face.pt", help="Path to dedicated face weights")
    parser.add_argument("--model-plate", type=str, default="models/yolov8n-plate.pt", help="Path to dedicated plate weights")
    parser.add_argument("--proxy-model", type=str, default="", help="Path to YOLO weights used strictly as CPU inference latency proxy")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic test video if missing")

    args = parser.parse_args()

    video_target = args.video
    if not video_target:
        default_video = Path("data/videos/test.mp4")
        if default_video.exists():
            video_target = str(default_video)
        elif args.generate_synthetic:
            synthetic_path = "data/videos/synthetic_test.mp4"
            generate_synthetic_test_video(synthetic_path)
            video_target = synthetic_path
        else:
            print("\n[Notice] No input video specified and data/videos/test.mp4 was not found.")
            print("Please place a local street video in:")
            print("  data/videos/<your_video>.mp4")
            print("Or run with --generate-synthetic to test script mechanics on generated frames.")
            sys.exit(0)

    bench = RedactionBenchmark(
        video_path=video_target,
        max_frames=args.max_frames,
        warmup_frames=args.warmup,
        face_model_path=args.model_face if Path(args.model_face).exists() else None,
        plate_model_path=args.model_plate if Path(args.model_plate).exists() else None,
        proxy_model_path=args.proxy_model if Path(args.proxy_model).exists() else None,
        conf_threshold=args.conf,
    )

    try:
        bench.execute()
    except Exception as e:
        print(f"[Error] Benchmark execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
