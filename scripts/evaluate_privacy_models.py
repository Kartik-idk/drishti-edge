#!/usr/bin/env python3
"""
Drishti-Edge: Privacy-Preserving Vision Pipeline
Phase 0.3: Dedicated Face & License-Plate Detector Evaluation

Evaluates candidate face and license-plate detection models on real 4K street footage:
- Face Candidate: OpenCV Zoo YuNet ONNX (face_detection_yunet_2023mar.onnx / 2026may.onnx)
- Plate Candidate A: OpenCV Zoo LPD-YuNet ONNX (license_plate_detection_lpd_yunet_2023mar.onnx)
- Plate Candidate B: YOLOv8n-plate PyTorch (yolov8n_plate.pt)
- Compares Resolution Strategies:
  1. Standard downscaled (1080p) whole-frame inference
  2. Native high-resolution (4K) whole-frame inference
  3. Cropped ROI inference (vehicle ROI crop -> plate detector)
- Measures CPU vs Apple Silicon MPS execution for YOLO
- Evaluates solid opaque mask redaction with configurable 25% margin expansion and overlap merging
- Implements fail-safe frame withholding on inference/processing exceptions
- Generates reproducible metrics and manual validation CSV templates

HEADLESS OPERATION ONLY:
No GUI windowing functions (cv2.imshow, cv2.waitKey, cv2.namedWindow) are called.
Unredacted frames, identifiable crops, and unredacted videos are never persisted to disk.
"""

import argparse
import csv
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ensure stdout flushes immediately
sys.stdout.reconfigure(line_buffering=True)


class LPDYuNetDetector:
    """Wrapper for OpenCV Zoo LPD-YuNet ONNX license plate detector."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 480),
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.3,
        backend_id: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target_id: int = cv2.dnn.DNN_TARGET_CPU,
    ):
        self.model_path = model_path
        self.input_size = tuple(input_size)  # (w, h)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.backend_id = backend_id
        self.target_id = target_id

        self.min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        self.steps = [8, 16, 32, 64]
        self.variance = [0.1, 0.2]
        self.output_names = ["loc", "conf", "iou"]

        if not Path(model_path).exists():
            raise FileNotFoundError(f"LPD-YuNet model file not found: {model_path}")

        self.model = cv2.dnn.readNet(self.model_path)
        self.model.setPreferableBackend(self.backend_id)
        self.model.setPreferableTarget(self.target_id)
        self._generate_priors()

    def set_input_size(self, input_size: Tuple[int, int]) -> None:
        if self.input_size != tuple(input_size):
            self.input_size = tuple(input_size)
            self._generate_priors()

    def _generate_priors(self) -> None:
        w, h = self.input_size
        feature_maps = [
            [int(int(int((h + 1) / 2) / 2) / 2), int(int(int((w + 1) / 2) / 2) / 2)],
            [int(int(int(int((h + 1) / 2) / 2) / 2) / 2), int(int(int(int((w + 1) / 2) / 2) / 2) / 2)],
            [int(int(int(int(int((h + 1) / 2) / 2) / 2) / 2) / 2), int(int(int(int(int((w + 1) / 2) / 2) / 2) / 2) / 2)],
            [int(int(int(int(int(int((h + 1) / 2) / 2) / 2) / 2) / 2) / 2), int(int(int(int(int(int((w + 1) / 2) / 2) / 2) / 2) / 2) / 2)],
        ]
        priors = []
        for k, f in enumerate(feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in self.min_sizes[k]:
                    priors.append([
                        (j + 0.5) * self.steps[k] / w,
                        (i + 0.5) * self.steps[k] / h,
                        min_size / w,
                        min_size / h,
                    ])
        self.priors = np.array(priors, dtype=np.float32)

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Detect license plates in image.
        
        Returns: list of (x1, y1, x2, y2, score)
        """
        h, w = image.shape[:2]
        self.set_input_size((w, h))

        blob = cv2.dnn.blobFromImage(image)
        self.model.setInput(blob)
        loc, conf, iou = self.model.forward(self.output_names)

        cls_scores = conf[:, 1]
        iou_scores = np.clip(iou[:, 0], 0.0, 1.0)
        scores = np.sqrt(cls_scores * iou_scores)

        scale = np.array([w, h], dtype=np.float32)
        b0 = (self.priors[:, 0:2] + loc[:, 4:6] * self.variance[0] * self.priors[:, 2:4]) * scale
        b1 = (self.priors[:, 0:2] + loc[:, 6:8] * self.variance[0] * self.priors[:, 2:4]) * scale
        b2 = (self.priors[:, 0:2] + loc[:, 10:12] * self.variance[0] * priors[:, 2:4]) * scale if False else (
            self.priors[:, 0:2] + loc[:, 10:12] * self.variance[0] * self.priors[:, 2:4]
        ) * scale
        b3 = (self.priors[:, 0:2] + loc[:, 12:14] * self.variance[0] * self.priors[:, 2:4]) * scale

        xs = np.stack([b0[:, 0], b1[:, 0], b2[:, 0], b3[:, 0]], axis=1)
        ys = np.stack([b0[:, 1], b1[:, 1], b2[:, 1], b3[:, 1]], axis=1)
        x_min = np.min(xs, axis=1)
        x_max = np.max(xs, axis=1)
        y_min = np.min(ys, axis=1)
        y_max = np.max(ys, axis=1)

        boxes = np.stack([x_min, y_min, x_max - x_min, y_max - y_min], axis=1)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)

        results = []
        if len(indices) > 0:
            for idx in indices:
                x1 = int(round(x_min[idx]))
                y1 = int(round(y_min[idx]))
                x2 = int(round(x_max[idx]))
                y2 = int(round(y_max[idx]))
                score = float(scores[idx])
                results.append((x1, y1, x2, y2, score))
        return results


class PrivacyModelEvaluator:
    """Evaluates face and license-plate detection models and redaction mechanics."""

    def __init__(
        self,
        face_model_path: str = "models/face_detection_yunet_2023mar.onnx",
        plate_lpd_path: Optional[str] = "models/license_plate_detection_lpd_yunet_2023mar.onnx",
        plate_yolo_path: Optional[str] = "models/yolov8n_plate.pt",
        vehicle_yolo_path: Optional[str] = "yolov8n.pt",
        face_conf: float = 0.35,
        plate_conf: float = 0.25,
        margin_ratio: float = 0.25,
        redaction_color: Tuple[int, int, int] = (0, 0, 0),
        yolo_device: str = "cpu",
    ):
        self.face_model_path = face_model_path
        self.plate_lpd_path = plate_lpd_path
        self.plate_yolo_path = plate_yolo_path
        self.vehicle_yolo_path = vehicle_yolo_path
        self.face_conf = face_conf
        self.plate_conf = plate_conf
        self.margin_ratio = margin_ratio
        self.redaction_color = redaction_color
        self.yolo_device = yolo_device

        # Model instances
        self.face_detector = None
        self.plate_lpd = None
        self.plate_yolo = None
        self.vehicle_yolo = None

        self._load_models()

    def _load_models(self) -> None:
        """Load candidate detection models with strict error tracking."""
        print("[Evaluator] Initializing models...")
        # 1. Face Detector: OpenCV Zoo YuNet
        if self.face_model_path and Path(self.face_model_path).exists():
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    model=str(self.face_model_path),
                    config="",
                    input_size=(640, 640),
                    score_threshold=self.face_conf,
                    nms_threshold=0.3,
                    backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                    target_id=cv2.dnn.DNN_TARGET_CPU,
                )
                print(f"  [OK] Face Detector (YuNet ONNX): {self.face_model_path}")
            except Exception as e:
                print(f"  [FAIL] Could not initialize YuNet face detector: {e}")
        else:
            print(f"  [WARN] Face detector path not found: {self.face_model_path}")

        # 2. License Plate Detector Candidate A: LPD-YuNet
        if self.plate_lpd_path and Path(self.plate_lpd_path).exists():
            try:
                self.plate_lpd = LPDYuNetDetector(
                    model_path=str(self.plate_lpd_path),
                    input_size=(640, 480),
                    conf_threshold=self.plate_conf,
                    nms_threshold=0.3,
                )
                print(f"  [OK] Plate Detector A (LPD-YuNet ONNX): {self.plate_lpd_path}")
            except Exception as e:
                print(f"  [FAIL] Could not initialize LPD-YuNet detector: {e}")
        else:
            print(f"  [INFO] LPD-YuNet path omitted or not found: {self.plate_lpd_path}")

        # 3. License Plate Detector Candidate B: YOLOv8n-plate
        if self.plate_yolo_path and Path(self.plate_yolo_path).exists():
            try:
                from ultralytics import YOLO
                self.plate_yolo = YOLO(str(self.plate_yolo_path))
                # Warm up on specified device
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.plate_yolo(dummy, device=self.yolo_device, verbose=False)
                print(f"  [OK] Plate Detector B (YOLOv8n-plate on {self.yolo_device.upper()}): {self.plate_yolo_path}")
            except Exception as e:
                print(f"  [FAIL] Could not initialize YOLO plate detector on {self.yolo_device}: {e}")
        else:
            print(f"  [INFO] YOLO plate detector omitted or not found: {self.plate_yolo_path}")

        # 4. Vehicle detector for Strategy 3 (Crop-based ROI inference)
        if self.vehicle_yolo_path and Path(self.vehicle_yolo_path).exists():
            try:
                from ultralytics import YOLO
                self.vehicle_yolo = YOLO(str(self.vehicle_yolo_path))
                print(f"  [OK] Vehicle Detector for ROI Crops: {self.vehicle_yolo_path}")
            except Exception as e:
                print(f"  [WARN] Could not initialize vehicle detector: {e}")

    @staticmethod
    def expand_and_clamp_bbox(
        box: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        margin_ratio: float = 0.25,
    ) -> Tuple[int, int, int, int]:
        """Expand bounding box by margin_ratio (25%) on each side and clamp to frame boundaries.
        
        Formula:
          dx = floor(width * margin_ratio)
          dy = floor(height * margin_ratio)
          x1' = max(0, x1 - dx)
          y1' = max(0, y1 - dy)
          x2' = min(frame_width, x2 + dx)
          y2' = min(frame_height, y2 + dy)
        """
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        dx = int(round(bw * margin_ratio))
        dy = int(round(bh * margin_ratio))

        x1_exp = max(0, x1 - dx)
        y1_exp = max(0, y1 - dy)
        x2_exp = min(frame_width, x2 + dx)
        y2_exp = min(frame_height, y2 + dy)

        return (x1_exp, y1_exp, x2_exp, y2_exp)

    @staticmethod
    def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping rectangular bounding boxes iteratively into minimal enclosing unions."""
        if not boxes:
            return []

        merged = list(boxes)
        changed = True

        while changed:
            changed = False
            new_merged = []
            visited = set()

            for i in range(len(merged)):
                if i in visited:
                    continue
                b1 = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in visited:
                        continue
                    b2 = merged[j]
                    # Overlap condition: intervals overlap on both x and y axes
                    if max(b1[0], b2[0]) <= min(b1[2], b2[2]) and max(b1[1], b2[1]) <= min(b1[3], b2[3]):
                        b1 = (
                            min(b1[0], b2[0]),
                            min(b1[1], b2[1]),
                            max(b1[2], b2[2]),
                            max(b1[3], b2[3]),
                        )
                        visited.add(j)
                        changed = True
                new_merged.append(b1)
            merged = new_merged

        return merged

    def apply_solid_mask_redaction(
        self,
        frame: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[np.ndarray, float]:
        """Apply irreversible solid opaque mask to all detected and merged regions.
        
        Returns: (redacted_frame, latency_ms)
        """
        t0 = time.perf_counter()
        redacted = frame.copy()
        color = self.redaction_color

        for (x1, y1, x2, y2) in boxes:
            if x2 > x1 and y2 > y1:
                redacted[y1:y2, x1:x2] = color

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return redacted, latency_ms

    def detect_faces_yunet(
        self,
        frame: np.ndarray,
        inference_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Detect faces using YuNet at specified resolution.
        
        If inference_resolution is provided, image is resized for inference
        and coordinates are mapped back to original frame dimensions.
        Returns: (detections, latency_ms)
        """
        if self.face_detector is None:
            return [], 0.0

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        if inference_resolution is not None and inference_resolution != (orig_w, orig_h):
            inf_w, inf_h = inference_resolution
            inf_frame = cv2.resize(frame, (inf_w, inf_h))
            self.face_detector.setInputSize((inf_w, inf_h))
            _, raw_dets = self.face_detector.detect(inf_frame)
            scale_x = orig_w / float(inf_w)
            scale_y = orig_h / float(inf_h)
        else:
            self.face_detector.setInputSize((orig_w, orig_h))
            _, raw_dets = self.face_detector.detect(frame)
            scale_x, scale_y = 1.0, 1.0

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        detections = []
        if raw_dets is not None:
            for det in raw_dets:
                score = float(det[-1])
                if score >= self.face_conf:
                    x, y, bw, bh = det[0:4]
                    x1 = int(round(x * scale_x))
                    y1 = int(round(y * scale_y))
                    x2 = int(round((x + bw) * scale_x))
                    y2 = int(round((y + bh) * scale_y))
                    detections.append({
                        "class": "face",
                        "box": (x1, y1, x2, y2),
                        "confidence": score,
                    })

        return detections, latency_ms

    def detect_plates_lpd(
        self,
        frame: np.ndarray,
        inference_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Detect plates using OpenCV Zoo LPD-YuNet ONNX."""
        if self.plate_lpd is None:
            return [], 0.0

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        if inference_resolution is not None and inference_resolution != (orig_w, orig_h):
            inf_w, inf_h = inference_resolution
            inf_frame = cv2.resize(frame, (inf_w, inf_h))
            raw_dets = self.plate_lpd.detect(inf_frame)
            scale_x = orig_w / float(inf_w)
            scale_y = orig_h / float(inf_h)
        else:
            raw_dets = self.plate_lpd.detect(frame)
            scale_x, scale_y = 1.0, 1.0

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        detections = []
        for (x1, y1, x2, y2, score) in raw_dets:
            fx1 = int(round(x1 * scale_x))
            fy1 = int(round(y1 * scale_y))
            fx2 = int(round(x2 * scale_x))
            fy2 = int(round(y2 * scale_y))
            detections.append({
                "class": "license_plate",
                "box": (fx1, fy1, fx2, fy2),
                "confidence": score,
            })

        return detections, latency_ms

    def detect_plates_yolo_crops(
        self,
        frame: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Strategy 3: Cropped-region inference.
        
        Detect vehicles, crop vehicle ROIs in-memory, run YOLOv8n-plate on crops,
        and project coordinates back to the original full frame.
        """
        if self.plate_yolo is None or self.vehicle_yolo is None:
            return [], 0.0

        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        # Step 1: Detect vehicles on downscaled frame (e.g. 640 letterbox)
        vehicle_res = self.vehicle_yolo(
            frame,
            device=self.yolo_device,
            verbose=False,
            conf=0.25,
        )[0]

        # Vehicle COCO classes: car (2), motorcycle (3), bus (5), truck (7)
        vehicle_classes = {2, 3, 5, 7}
        vehicle_boxes = []
        for b in vehicle_res.boxes:
            if int(b.cls[0]) in vehicle_classes:
                coords = b.xyxy[0].cpu().numpy().astype(int)
                vehicle_boxes.append(coords)

        # Step 2: For each vehicle crop, run plate detector
        detections = []
        for (vx1, vy1, vx2, vy2) in vehicle_boxes:
            vx1_c = max(0, vx1)
            vy1_c = max(0, vy1)
            vx2_c = min(orig_w, vx2)
            vy2_c = min(orig_h, vy2)
            vw, vh = vx2_c - vx1_c, vy2_c - vy1_c
            if vw < 25 or vh < 25:
                continue

            crop = frame[vy1_c:vy2_c, vx1_c:vx2_c]
            plate_res = self.plate_yolo(
                crop,
                device=self.yolo_device,
                verbose=False,
                conf=self.plate_conf,
            )[0]

            for pb in plate_res.boxes:
                p_score = float(pb.conf[0])
                p_coords = pb.xyxy[0].cpu().numpy().astype(int)
                fx1 = vx1_c + p_coords[0]
                fy1 = vy1_c + p_coords[1]
                fx2 = vx1_c + p_coords[2]
                fy2 = vy1_c + p_coords[3]
                detections.append({
                    "class": "license_plate",
                    "box": (fx1, fy1, fx2, fy2),
                    "confidence": p_score,
                })

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        return detections, latency_ms

    def evaluate_frame_pipeline(
        self,
        frame: np.ndarray,
        strategy: str = "strategy1_1080p",
    ) -> Dict[str, Any]:
        """Execute complete privacy pipeline for a single frame with fail-safe withholding.
        
        Strategies:
        - 'strategy1_1080p': 1080p whole-frame YuNet face + 1080p LPD-YuNet
        - 'strategy2_4k': Native 4K YuNet face + 1080p LPD-YuNet
        - 'strategy3_crops': 1080p YuNet face + Crop-based YOLOv8n-plate
        """
        orig_h, orig_w = frame.shape[:2]
        t_start = time.perf_counter()

        frame_withheld = False
        withhold_reason = ""
        processing_exception = None

        all_detections = []
        latencies = {
            "preprocess_ms": 0.0,
            "face_det_ms": 0.0,
            "plate_det_ms": 0.0,
            "redact_ms": 0.0,
            "total_ms": 0.0,
        }

        try:
            # 1. Face Detection
            if strategy == "strategy2_4k":
                face_dets, face_ms = self.detect_faces_yunet(frame, inference_resolution=(orig_w, orig_h))
            else:
                face_dets, face_ms = self.detect_faces_yunet(frame, inference_resolution=(1920, 1080))
            latencies["face_det_ms"] = face_ms
            all_detections.extend(face_dets)

            # 2. Plate Detection
            if strategy == "strategy3_crops":
                plate_dets, plate_ms = self.detect_plates_yolo_crops(frame)
            else:
                # Standard LPD-YuNet at 1920x1080 or 640x480
                plate_dets, plate_ms = self.detect_plates_lpd(frame, inference_resolution=(1920, 1080))
            latencies["plate_det_ms"] = plate_ms
            all_detections.extend(plate_dets)

            # 3. Privacy Redaction Pipeline
            # Step A: Expand bounding boxes with margin
            expanded_boxes = [
                self.expand_and_clamp_bbox(d["box"], orig_w, orig_h, self.margin_ratio)
                for d in all_detections
            ]
            # Step B: Merge overlapping masks
            merged_boxes = self.merge_overlapping_boxes(expanded_boxes)

            # Step C: Apply solid opaque mask redaction
            redacted_frame, redact_ms = self.apply_solid_mask_redaction(frame, merged_boxes)
            latencies["redact_ms"] = redact_ms

        except Exception as e:
            # Catch exceptions and enforce fail-safe withholding
            processing_exception = str(e)
            frame_withheld = True
            withhold_reason = f"Processing exception during pipeline: {e}"
            redacted_frame = None

        t_end = time.perf_counter()
        latencies["total_ms"] = (t_end - t_start) * 1000.0

        # Check required detector availability
        if self.face_detector is None:
            frame_withheld = True
            withhold_reason = "Face detector unavailable"
            redacted_frame = None

        return {
            "strategy": strategy,
            "detections": all_detections,
            "merged_boxes": merged_boxes if not frame_withheld else [],
            "num_faces": sum(1 for d in all_detections if d["class"] == "face"),
            "num_plates": sum(1 for d in all_detections if d["class"] == "license_plate"),
            "latencies": latencies,
            "frame_withheld": frame_withheld,
            "withhold_reason": withhold_reason,
            "processing_exception": processing_exception,
        }


def sample_video_timestamps(
    video_path: str,
    timestamps: List[float],
    evaluator: PrivacyModelEvaluator,
    strategies: List[str],
) -> List[Dict[str, Any]]:
    """Sample frames at known timestamps and evaluate candidate strategies."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Sampling] Video: {video_path}")
    print(f"           Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {total_frames} | Duration: {duration_sec:.2f}s")
    print(f"           Sampling {len(timestamps)} timestamps across strategies: {strategies}")

    results = []

    for ts in timestamps:
        target_frame_idx = int(round(ts * fps))
        if target_frame_idx >= total_frames:
            target_frame_idx = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
        t_dec0 = time.perf_counter()
        ret, frame = cap.read()
        t_dec1 = time.perf_counter()
        decode_ms = (t_dec1 - t_dec0) * 1000.0

        if not ret or frame is None:
            print(f"  [WARN] Failed to read frame at t={ts:.2f}s (idx={target_frame_idx})")
            continue

        for strat in strategies:
            eval_res = evaluator.evaluate_frame_pipeline(frame, strategy=strat)
            eval_res["video_file"] = Path(video_path).name
            eval_res["timestamp_sec"] = ts
            eval_res["frame_idx"] = target_frame_idx
            eval_res["decode_ms"] = decode_ms
            eval_res["latencies"]["decode_ms"] = decode_ms
            # Add decode time to end-to-end latency
            eval_res["latencies"]["e2e_with_decode_ms"] = eval_res["latencies"]["total_ms"] + decode_ms
            results.append(eval_res)

            print(
                f"  t={ts:4.1f}s | {strat:18} | Faces: {eval_res['num_faces']:2d} | Plates: {eval_res['num_plates']:2d} | "
                f"Face Lat: {eval_res['latencies']['face_det_ms']:5.1f}ms | Plate Lat: {eval_res['latencies']['plate_det_ms']:5.1f}ms | "
                f"Redact: {eval_res['latencies']['redact_ms']:4.2f}ms | Total: {eval_res['latencies']['e2e_with_decode_ms']:5.1f}ms"
            )

    cap.release()
    return results


def run_sustained_latency_benchmark(
    video_path: str,
    evaluator: PrivacyModelEvaluator,
    strategy: str,
    num_frames: int = 50,
) -> Dict[str, Any]:
    """Run sustained multi-frame benchmark to measure stable throughput and percentiles."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Sustained Benchmark] Running {num_frames} frames on {Path(video_path).name} with {strategy} (Device: {evaluator.yolo_device.upper()})...")

    # Warmup
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            _ = evaluator.evaluate_frame_pipeline(frame, strategy=strategy)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    decode_times = []
    face_times = []
    plate_times = []
    redact_times = []
    e2e_times = []
    total_faces = 0
    total_plates = 0

    bench_start = time.perf_counter()
    processed_count = 0

    while processed_count < num_frames:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret or frame is None:
            break

        dec_ms = (t1 - t0) * 1000.0
        eval_res = evaluator.evaluate_frame_pipeline(frame, strategy=strategy)
        t2 = time.perf_counter()
        e2e_ms = (t2 - t0) * 1000.0

        decode_times.append(dec_ms)
        face_times.append(eval_res["latencies"]["face_det_ms"])
        plate_times.append(eval_res["latencies"]["plate_det_ms"])
        redact_times.append(eval_res["latencies"]["redact_ms"])
        e2e_times.append(e2e_ms)

        total_faces += eval_res["num_faces"]
        total_plates += eval_res["num_plates"]
        processed_count += 1

    bench_end = time.perf_counter()
    cap.release()

    total_wall_sec = bench_end - bench_start
    throughput_fps = processed_count / total_wall_sec if total_wall_sec > 0 else 0.0

    return {
        "video": Path(video_path).name,
        "strategy": strategy,
        "device": evaluator.yolo_device,
        "processed_frames": processed_count,
        "wall_time_sec": total_wall_sec,
        "throughput_fps": throughput_fps,
        "decode_mean_ms": float(np.mean(decode_times)),
        "face_mean_ms": float(np.mean(face_times)),
        "face_median_ms": float(np.median(face_times)),
        "face_p95_ms": float(np.percentile(face_times, 95)),
        "plate_mean_ms": float(np.mean(plate_times)),
        "plate_median_ms": float(np.median(plate_times)),
        "plate_p95_ms": float(np.percentile(plate_times, 95)),
        "redact_mean_ms": float(np.mean(redact_times)),
        "e2e_mean_ms": float(np.mean(e2e_times)),
        "e2e_median_ms": float(np.median(e2e_times)),
        "e2e_p95_ms": float(np.percentile(e2e_times, 95)),
        "total_faces_detected": total_faces,
        "total_plates_detected": total_plates,
    }


def generate_manual_validation_template(
    sample_results: List[Dict[str, Any]],
    output_path: str = "docs/benchmarks/manual_validation_template.csv",
) -> None:
    """Generate the standardized CSV template for manual verification of privacy redaction."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "video_filename",
        "timestamp_sec",
        "frame_idx",
        "candidate_strategy",
        "model_detected_faces",
        "model_detected_plates",
        "human_verified_visible_faces",
        "human_verified_4w_plates",
        "human_verified_2w_3w_plates",
        "covered_by_candidate_redaction",
        "unredacted_privacy_leak_count",
        "notes_about_image_ambiguity",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for res in sample_results:
            writer.writerow({
                "video_filename": res["video_file"],
                "timestamp_sec": f"{res['timestamp_sec']:.2f}",
                "frame_idx": res["frame_idx"],
                "candidate_strategy": res["strategy"],
                "model_detected_faces": res["num_faces"],
                "model_detected_plates": res["num_plates"],
                "human_verified_visible_faces": "",  # To be filled by reviewer
                "human_verified_4w_plates": "",      # To be filled by reviewer
                "human_verified_2w_3w_plates": "",   # To be filled by reviewer
                "covered_by_candidate_redaction": "", # To be filled by reviewer
                "unredacted_privacy_leak_count": "",  # To be filled by reviewer
                "notes_about_image_ambiguity": "",    # To be filled by reviewer
            })

    print(f"\n[Validation] Manual verification template CSV generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Drishti-Edge: Privacy Model & Redaction Evaluator")
    parser.add_argument("--day-video", type=str, default="data/videos/daytime_india.mp4")
    parser.add_argument("--night-video", type=str, default="data/videos/night_street.mp4")
    parser.add_argument("--smoke-video", type=str, default="data/videos/nighttime_india.mp4")
    parser.add_argument("--face-model", type=str, default="models/face_detection_yunet_2023mar.onnx")
    parser.add_argument("--lpd-model", type=str, default="models/license_plate_detection_lpd_yunet_2023mar.onnx")
    parser.add_argument("--yolo-plate-model", type=str, default="models/yolov8n_plate.pt")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--benchmark-frames", type=int, default=50)
    args = parser.parse_args()

    # 15 Representative timestamps for daytime_india.mp4 (0s to 12s)
    day_timestamps = [0.5, 1.2, 2.0, 2.8, 3.5, 4.2, 5.0, 5.8, 6.5, 7.2, 8.0, 8.8, 9.5, 10.5, 11.5]
    # 15 Representative timestamps for night_street.mp4 (0s to 5.7s)
    night_timestamps = [0.3, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6, 5.0, 5.3, 5.6]

    evaluator = PrivacyModelEvaluator(
        face_model_path=args.face_model,
        plate_lpd_path=args.lpd_model,
        plate_yolo_path=args.yolo_plate_model,
        yolo_device=args.device,
    )

    strategies_to_test = ["strategy1_1080p", "strategy2_4k", "strategy3_crops"]

    all_sample_results = []

    # Run sampling on daytime_india.mp4
    if Path(args.day_video).exists():
        day_results = sample_video_timestamps(args.day_video, day_timestamps, evaluator, strategies_to_test)
        all_sample_results.extend(day_results)

    # Run sampling on night_street.mp4
    if Path(args.night_video).exists():
        night_results = sample_video_timestamps(args.night_video, night_timestamps, evaluator, strategies_to_test)
        all_sample_results.extend(night_results)

    # Generate CSV validation template
    generate_manual_validation_template(all_sample_results)

    # Sustained latency benchmark
    print("\n=======================================================")
    print("  SUSTAINED PRIVACY ENGINE LATENCY & THROUGHPUT (CPU)")
    print("=======================================================")
    if Path(args.day_video).exists():
        for strat in ["strategy1_1080p", "strategy2_4k", "strategy3_crops"]:
            res = run_sustained_latency_benchmark(args.day_video, evaluator, strat, num_frames=args.benchmark_frames)
            print(f"Strategy: {strat:18} | Device: {res['device'].upper()}")
            print(f"  Throughput:           {res['throughput_fps']:.2f} FPS")
            print(f"  Face Latency:         Mean: {res['face_mean_ms']:.2f} ms | Median: {res['face_median_ms']:.2f} ms | P95: {res['face_p95_ms']:.2f} ms")
            print(f"  Plate Latency:        Mean: {res['plate_mean_ms']:.2f} ms | Median: {res['plate_median_ms']:.2f} ms | P95: {res['plate_p95_ms']:.2f} ms")
            print(f"  Redaction Latency:    Mean: {res['redact_mean_ms']:.2f} ms")
            print(f"  End-to-End Latency:   Mean: {res['e2e_mean_ms']:.2f} ms | Median: {res['e2e_median_ms']:.2f} ms | P95: {res['e2e_p95_ms']:.2f} ms")
            print("-------------------------------------------------------")

    # If MPS is available, test YOLO on MPS
    import torch
    if torch.backends.mps.is_available():
        print("\n=======================================================")
        print("  MPS ACCELERATION COMPARISON (Apple Silicon M4 GPU)")
        print("=======================================================")
        evaluator_mps = PrivacyModelEvaluator(
            face_model_path=args.face_model,
            plate_lpd_path=args.lpd_model,
            plate_yolo_path=args.yolo_plate_model,
            yolo_device="mps",
        )
        res_mps = run_sustained_latency_benchmark(args.day_video, evaluator_mps, "strategy3_crops", num_frames=args.benchmark_frames)
        print(f"Strategy: strategy3_crops    | Device: MPS (Apple Silicon GPU)")
        print(f"  Throughput:           {res_mps['throughput_fps']:.2f} FPS")
        print(f"  Face Latency (CPU):   Mean: {res_mps['face_mean_ms']:.2f} ms")
        print(f"  Plate Latency (MPS):  Mean: {res_mps['plate_mean_ms']:.2f} ms | Median: {res_mps['plate_median_ms']:.2f} ms | P95: {res_mps['plate_p95_ms']:.2f} ms")
        print(f"  Redaction Latency:    Mean: {res_mps['redact_mean_ms']:.2f} ms")
        print(f"  End-to-End Latency:   Mean: {res_mps['e2e_mean_ms']:.2f} ms | Median: {res_mps['e2e_median_ms']:.2f} ms | P95: {res_mps['e2e_p95_ms']:.2f} ms")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
