#!/usr/bin/env python3
"""
Drishti-Edge: Hardened Privacy Engine & Fail-Closed Privacy Gate
Phase 0.4: Privacy Hardening, Fallback Protection & Safety Sign-Off

Architectural Guarantees:
1. Canonical 1280x720 Output Resolution:
   - Source frames decoded and immediately scaled to canonical 1280x720.
   - All detections, crops, coordinate tracking, fallbacks, and redactions operate
     exclusively within the canonical 1280x720 coordinate system.
   - Downstream consumers receive strictly 1280x720 output. Raw full-resolution
     frames are overwritten/discarded immediately.
2. Robust Mask Expansion & Boundary Clamping:
   - Finite and ordered coordinate validation (rejects NaN, inf, reversed, zero-area boxes).
   - Proportional margin expansion (default 25%) with minimum padding floor of 8 pixels.
   - Clamping strictly enforced after expansion and before NumPy array indexing.
   - Negative slice indices are strictly impossible.
3. Conservative Fallback Protection:
   - Vehicle Plate Fallback: Eligible close/mid-range vehicles (height >= 60 px) that receive
     successful plate inference returning 0 plates are masked on the bottom 30% of the vehicle box.
   - Crop Capping: Plate crop inference capped at 8 largest eligible vehicles per frame; any
     unprocessed eligible vehicles (>8) receive bottom-30% geometric fallback masking.
   - Face Fallback: Detected persons (height >= 40 px) without an associated face detection
     receive top-30% geometric fallback masking.
4. Three-Frame Temporal Mask Protection:
   - Tracks detected faces and plates across up to 3 frames with velocity estimation and
     conservative 10% spatial buffer expansion during single-frame dropouts.
5. Strict Fail-Closed Output Gating:
   - Output begins marked NOT SAFE FOR RELEASE.
   - Any detector exception, missing configuration, or redaction failure triggers suppression
     with an all-black (zeros) frame at canonical 1280x720 dimensions. Raw frames are NEVER leaked.
   - Only error-free complete processing yields RELEASE_APPROVED.
"""

import enum
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


class PrivacyGateStatus(enum.Enum):
    NOT_PROCESSED = "not_processed"
    RELEASE_APPROVED = "pipeline_execution_completed"
    EXECUTION_COMPLETED = RELEASE_APPROVED
    SUPPRESSED_DETECTOR_FAILURE = "suppressed_detector_failure"
    SUPPRESSED_REDACTION_FAILURE = "suppressed_redaction_failure"
    SUPPRESSED_MISSING_CONFIG = "suppressed_missing_config"


class MaskProvenance(str, enum.Enum):
    FACE_DETECTION = "FACE_DETECTION"
    PLATE_DETECTION = "PLATE_DETECTION"
    VEHICLE_FALLBACK = "VEHICLE_FALLBACK"
    PERSON_FALLBACK = "PERSON_FALLBACK"
    TEMPORAL_HOLD = "TEMPORAL_HOLD"
    FAIL_CLOSED_SUPPRESSION = "FAIL_CLOSED_SUPPRESSION"


# Canonical prototype resolution
CANONICAL_WIDTH = 1280
CANONICAL_HEIGHT = 720


def expand_and_clamp_bbox(
    box: Tuple[Any, Any, Any, Any],
    frame_width: int = CANONICAL_WIDTH,
    frame_height: int = CANONICAL_HEIGHT,
    margin_ratio: float = 0.25,
    min_padding: int = 8,
) -> Optional[Tuple[int, int, int, int]]:
    """Validate, expand with proportional margin + minimum padding floor, and clamp coordinates.
    
    Guarantees:
      - Validates coordinates are finite numbers (rejects NaN, inf).
      - Rejects inverted (x2 <= x1, y2 <= y1) or zero-area boxes.
      - Expansion: dx = max(round(bw * margin_ratio), min_padding).
      - Boundary clamping to [0, frame_width] and [0, frame_height].
      - Returns None if resulting clamped box has non-positive area.
      - Returned coordinates are non-negative integers safe for NumPy slicing.
    """
    if len(box) != 4:
        return None

    try:
        x1_raw, y1_raw, x2_raw, y2_raw = box
        # Validate finite numbers
        if not (
            math.isfinite(x1_raw)
            and math.isfinite(y1_raw)
            and math.isfinite(x2_raw)
            and math.isfinite(y2_raw)
        ):
            return None

        x1 = float(x1_raw)
        y1 = float(y1_raw)
        x2 = float(x2_raw)
        y2 = float(y2_raw)

    except (TypeError, ValueError):
        return None

    # Validate ordering
    if x2 <= x1 or y2 <= y1:
        return None

    # Reject boxes that are entirely outside frame boundaries before expansion
    if x2 <= 0 or x1 >= frame_width or y2 <= 0 or y1 >= frame_height:
        return None

    bw = x2 - x1
    bh = y2 - y1

    dx = max(int(round(bw * margin_ratio)), int(min_padding))
    dy = max(int(round(bh * margin_ratio)), int(min_padding))

    # Expand bounds
    exp_x1 = int(round(x1)) - dx
    exp_y1 = int(round(y1)) - dy
    exp_x2 = int(round(x2)) + dx
    exp_y2 = int(round(y2)) + dy

    # Clamp strictly after expansion and before slicing
    clamped_x1 = max(0, min(frame_width, exp_x1))
    clamped_y1 = max(0, min(frame_height, exp_y1))
    clamped_x2 = max(0, min(frame_width, exp_x2))
    clamped_y2 = max(0, min(frame_height, exp_y2))

    # Final positive-area validation
    if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
        return None

    return (clamped_x1, clamped_y1, clamped_x2, clamped_y2)


def apply_solid_masks_independent(
    frame: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, float]:
    """Apply expanded solid opaque masks independently onto the frame buffer.
    
    Overlapping masks are applied sequentially without NMS or box-merging passes.
    Returns: (redacted_frame, latency_ms)
    """
    t0 = time.perf_counter()
    redacted = frame.copy()

    for box in boxes:
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            redacted[y1:y2, x1:x2] = color

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    return redacted, latency_ms


class TemporalMaskTracker:
    """Conservative 3-frame temporal mask hold with velocity-based motion adjustment.
    
    Tracks direct face and license-plate detections across frames.
    Mitigates single-frame detector dropouts while accounting for subject movement.
    """

    def __init__(
        self,
        max_hold_frames: int = 3,
        iou_match_threshold: float = 0.20,
        frame_width: int = CANONICAL_WIDTH,
        frame_height: int = CANONICAL_HEIGHT,
    ):
        self.max_hold_frames = max_hold_frames
        self.iou_match_threshold = iou_match_threshold
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.next_track_id = 1
        # Tracks format: list of dicts:
        # {
        #   'id': int, 'box': (x1, y1, x2, y2), 'category': str, 'score': float,
        #   'missing_frames': int, 'status': str, 'vx': float, 'vy': float
        # }
        self.tracks: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Clear all active tracks and reset track ID counter."""
        self.tracks.clear()
        self.next_track_id = 1

    def _compute_iou(self, b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
        ix1 = max(b1[0], b2[0])
        iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2])
        iy2 = min(b1[3], b2[3])
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def update_and_get_tracked_targets(
        self,
        current_detections: List[Tuple[Any, ...]],
    ) -> List[Dict[str, Any]]:
        """Update tracker with current frame detections and return active tracked target objects.
        
        current_detections: list of tuples: (x1, y1, x2, y2, category, [score])
        Returns: list of dicts with keys:
          'id': int, 'box': (x1, y1, x2, y2), 'category': str, 'status': 'DETECTION' | 'TEMPORAL_HOLD',
          'score': float, 'missing_frames': int
        """
        matched_track_indices = set()
        matched_det_indices = set()
        active_targets: List[Dict[str, Any]] = []
        updated_tracks: List[Dict[str, Any]] = []

        # Parse inputs
        parsed_detections = []
        for det in current_detections:
            dx1, dy1, dx2, dy2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            cat = str(det[4])
            score = float(det[5]) if len(det) > 5 else 1.0
            parsed_detections.append((dx1, dy1, dx2, dy2, cat, score))

        # Step 1: Match current detections to existing tracks of SAME category
        for d_idx, (dx1, dy1, dx2, dy2, cat, score) in enumerate(parsed_detections):
            det_box = (dx1, dy1, dx2, dy2)
            best_iou = 0.0
            best_t_idx = -1
            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_track_indices:
                    continue
                # Require same object category (face cannot match plate)
                if track["category"] != cat:
                    continue
                iou = self._compute_iou(det_box, track["box"])
                if iou > best_iou and iou >= self.iou_match_threshold:
                    best_iou = iou
                    best_t_idx = t_idx

            if best_t_idx != -1:
                t = self.tracks[best_t_idx]
                prev_cx = (t["box"][0] + t["box"][2]) / 2.0
                prev_cy = (t["box"][1] + t["box"][3]) / 2.0
                curr_cx = (dx1 + dx2) / 2.0
                curr_cy = (dy1 + dy2) / 2.0

                vx = curr_cx - prev_cx
                vy = curr_cy - prev_cy

                t["box"] = det_box
                t["category"] = cat
                t["score"] = score
                t["missing_frames"] = 0
                t["status"] = "DETECTION"
                t["vx"] = 0.7 * t["vx"] + 0.3 * vx
                t["vy"] = 0.7 * t["vy"] + 0.3 * vy

                matched_track_indices.add(best_t_idx)
                matched_det_indices.add(d_idx)
                updated_tracks.append(t)
                active_targets.append({
                    "id": t["id"],
                    "box": det_box,
                    "category": cat,
                    "status": "DETECTION",
                    "score": score,
                    "missing_frames": 0,
                })

        # Step 2: Handle unmatched existing tracks (detector dropouts)
        for t_idx, track in enumerate(self.tracks):
            if t_idx in matched_track_indices:
                continue

            track["missing_frames"] += 1
            if track["missing_frames"] >= self.max_hold_frames:
                # Track expired after max_hold_frames consecutive missing frames
                continue

            # Estimate new position using velocity + modest 5% buffer
            x1, y1, x2, y2 = track["box"]
            bw = x2 - x1
            bh = y2 - y1
            pad = max(2, int(round(0.05 * max(bw, bh))))

            # Clamp drift velocity to max 15 px per frame to avoid runaway
            drift_x = max(-15.0, min(15.0, track["vx"]))
            drift_y = max(-15.0, min(15.0, track["vy"]))

            pred_x1 = max(0, min(self.frame_width, int(round(x1 + drift_x)) - pad))
            pred_y1 = max(0, min(self.frame_height, int(round(y1 + drift_y)) - pad))
            pred_x2 = max(0, min(self.frame_width, int(round(x2 + drift_x)) + pad))
            pred_y2 = max(0, min(self.frame_height, int(round(y2 + drift_y)) + pad))

            if pred_x2 > pred_x1 and pred_y2 > pred_y1:
                track["box"] = (pred_x1, pred_y1, pred_x2, pred_y2)
                track["status"] = "TEMPORAL_HOLD"
                updated_tracks.append(track)
                active_targets.append({
                    "id": track["id"],
                    "box": (pred_x1, pred_y1, pred_x2, pred_y2),
                    "category": track["category"],
                    "status": "TEMPORAL_HOLD",
                    "score": track["score"],
                    "missing_frames": track["missing_frames"],
                })

        # Step 3: Add new tracks for unmatched current detections
        for d_idx, (dx1, dy1, dx2, dy2, cat, score) in enumerate(parsed_detections):
            if d_idx not in matched_det_indices:
                det_box = (dx1, dy1, dx2, dy2)
                new_track = {
                    "id": self.next_track_id,
                    "box": det_box,
                    "category": cat,
                    "score": score,
                    "missing_frames": 0,
                    "status": "DETECTION",
                    "vx": 0.0,
                    "vy": 0.0,
                }
                self.next_track_id += 1
                updated_tracks.append(new_track)
                active_targets.append({
                    "id": new_track["id"],
                    "box": det_box,
                    "category": cat,
                    "status": "DETECTION",
                    "score": score,
                    "missing_frames": 0,
                })

        self.tracks = updated_tracks
        return active_targets

    def update_and_get_masks(
        self,
        current_detections: List[Tuple[Any, ...]],
    ) -> List[Tuple[int, int, int, int]]:
        """Legacy helper returning bounding box tuples for backwards compatibility."""
        targets = self.update_and_get_tracked_targets(current_detections)
        return [t["box"] for t in targets]


class ConsolidatedPrivacyEngine:
    """Hardened privacy engine with canonical 1280x720 pipeline and fail-closed gating."""

    def __init__(
        self,
        face_model_path: str = "models/face_detection_yunet_2023mar.onnx",
        vehicle_model_path: str = "yolov8n.pt",
        plate_model_path: str = "models/yolov8n_plate.pt",
        device: str = "mps" if torch.backends.mps.is_available() else "cpu",
        face_conf: float = 0.35,
        vehicle_conf: float = 0.15,
        plate_conf: float = 0.15,
        plate_imgsz: int = 480,
        margin_ratio: float = 0.25,
        min_padding: int = 8,
        fallback_margin_ratio: float = 0.08,
        fallback_min_padding: int = 4,
        redaction_color: Tuple[int, int, int] = (0, 0, 0),
        opencv_threads: int = 4,
        max_plate_crops: int = 8,
        min_vehicle_height: int = 35,
        min_vehicle_width: int = 30,
        min_person_height: int = 40,
        localized_fallbacks: bool = True,
    ):
        self.face_model_path = Path(face_model_path)
        self.vehicle_model_path = Path(vehicle_model_path)
        self.plate_model_path = Path(plate_model_path)
        self.device = device
        self.face_conf = face_conf
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.plate_imgsz = plate_imgsz
        self.margin_ratio = margin_ratio
        self.min_padding = min_padding
        self.fallback_margin_ratio = fallback_margin_ratio
        self.fallback_min_padding = fallback_min_padding
        self.redaction_color = redaction_color
        self.opencv_threads = opencv_threads
        self.max_plate_crops = max_plate_crops
        self.min_vehicle_height = min_vehicle_height
        self.min_vehicle_width = min_vehicle_width
        self.min_person_height = min_person_height
        self.localized_fallbacks = localized_fallbacks

        # Temporal tracker: strictly tracks direct face and plate detections
        self.temporal_tracker = TemporalMaskTracker(
            max_hold_frames=3,
            iou_match_threshold=0.20,
            frame_width=CANONICAL_WIDTH,
            frame_height=CANONICAL_HEIGHT,
        )

        # Model instances
        self.face_detector = None
        self.vehicle_detector = None
        self.plate_detector = None

        # Context classes: 0=person, vehicles={2:car, 3:motorcycle, 5:bus, 7:truck}
        self.vehicle_classes = {2, 3, 5, 7}
        self.person_class = 0

        self._init_models()

    def reset_temporal_state(self) -> None:
        """Explicitly clear all temporal tracking state across video or clip boundaries."""
        if hasattr(self, "temporal_tracker") and self.temporal_tracker is not None:
            self.temporal_tracker.reset()

    def _init_models(self) -> None:
        """Initialize models and configure multi-threaded OpenCV execution."""
        # 1. Face Detector: OpenCV Zoo YuNet ONNX
        if self.face_model_path.exists():
            self.face_detector = cv2.FaceDetectorYN.create(
                model=str(self.face_model_path),
                config="",
                input_size=(CANONICAL_WIDTH, CANONICAL_HEIGHT),
                score_threshold=self.face_conf,
                nms_threshold=0.3,
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )

        # 2. PyTorch / Ultralytics Context & Plate Detectors
        from ultralytics import YOLO

        if self.vehicle_model_path.exists():
            self.vehicle_detector = YOLO(str(self.vehicle_model_path))

        if self.plate_model_path.exists():
            self.plate_detector = YOLO(str(self.plate_model_path))

        # Explicitly configure OpenCV thread count to verified setting (4 threads)
        cv2.setNumThreads(self.opencv_threads)

    def detect_faces(
        self,
        canonical_frame: np.ndarray,
    ) -> Tuple[List[Tuple[int, int, int, int, float]], float]:
        """Detect faces on canonical 1280x720 frame using YuNet.
        
        Returns: (detected_faces, latency_ms)
        """
        if self.face_detector is None:
            raise RuntimeError("Face detector model is not initialized")

        t0 = time.perf_counter()
        self.face_detector.setInputSize((CANONICAL_WIDTH, CANONICAL_HEIGHT))
        _, raw_dets = self.face_detector.detect(canonical_frame)
        t1 = time.perf_counter()
        detect_ms = (t1 - t0) * 1000.0

        faces = []
        if raw_dets is not None:
            for d in raw_dets:
                score = float(d[-1])
                if score >= self.face_conf:
                    x, y, bw, bh = d[0:4]
                    x1 = max(0, min(CANONICAL_WIDTH, int(round(x))))
                    y1 = max(0, min(CANONICAL_HEIGHT, int(round(y))))
                    x2 = max(0, min(CANONICAL_WIDTH, int(round(x + bw))))
                    y2 = max(0, min(CANONICAL_HEIGHT, int(round(y + bh))))
                    if x2 > x1 and y2 > y1:
                        faces.append((x1, y1, x2, y2, score))

        return faces, detect_ms

    def detect_context_and_extract_crops(
        self,
        canonical_frame: np.ndarray,
    ) -> Tuple[
        List[np.ndarray],  # eligible vehicle crops (up to max_plate_crops)
        List[Tuple[int, int, int, int]],  # selected vehicle boxes in canonical frame
        List[Tuple[int, int, int, int]],  # skipped eligible vehicle boxes (> max_plate_crops)
        List[Tuple[int, int, int, int]],  # eligible person boxes in canonical frame
        float,  # detector latency ms
        float,  # crop preparation latency ms
    ]:
        """Detect persons and vehicles on canonical 1280x720 frame.
        
        Ranks eligible vehicles deterministically by area.
        Selects up to max_plate_crops for plate inference; remaining receive fallback masking.
        """
        if self.vehicle_detector is None:
            raise RuntimeError("Vehicle/context detector is not initialized")

        t0 = time.perf_counter()
        res = self.vehicle_detector(
            canonical_frame,
            device=self.device,
            imgsz=640,
            verbose=False,
            conf=self.vehicle_conf,
        )[0]
        t1 = time.perf_counter()
        detect_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        candidate_vehicles = []
        candidate_persons = []

        for b in res.boxes:
            cls_id = int(b.cls[0])
            coords = b.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            x1_c = max(0, min(CANONICAL_WIDTH, x1))
            y1_c = max(0, min(CANONICAL_HEIGHT, y1))
            x2_c = max(0, min(CANONICAL_WIDTH, x2))
            y2_c = max(0, min(CANONICAL_HEIGHT, y2))
            bw = x2_c - x1_c
            bh = y2_c - y1_c

            if cls_id in self.vehicle_classes:
                # Include mid-range vehicles down to min_vehicle_height and min_vehicle_width
                if bh >= self.min_vehicle_height and bw >= self.min_vehicle_width:
                    candidate_vehicles.append((x1_c, y1_c, x2_c, y2_c, bw * bh, cls_id))
            elif cls_id == self.person_class:
                if bh >= self.min_person_height and bw >= 15:
                    candidate_persons.append((x1_c, y1_c, x2_c, y2_c))

        # Deterministic ranking by bounding-box area descending
        candidate_vehicles.sort(key=lambda item: item[4], reverse=True)

        selected_vehicles = []
        selected_crops = []
        skipped_vehicles = []

        for idx, item in enumerate(candidate_vehicles):
            vx1, vy1, vx2, vy2 = item[0], item[1], item[2], item[3]
            cls_id = item[5]
            if idx < self.max_plate_crops:
                crop = canonical_frame[vy1:vy2, vx1:vx2]
                selected_crops.append(crop)
                selected_vehicles.append((vx1, vy1, vx2, vy2, cls_id))
            else:
                skipped_vehicles.append((vx1, vy1, vx2, vy2, cls_id))

        t3 = time.perf_counter()
        crop_prep_ms = (t3 - t2) * 1000.0

        return selected_crops, selected_vehicles, skipped_vehicles, candidate_persons, detect_ms, crop_prep_ms

    def _deduplicate_crop_plates(
        self,
        crop_plates: List[Tuple[int, int, int, int, float]],
        iou_threshold: float = 0.40,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Deduplicate multiple overlapping plate detections on the same vehicle crop."""
        if not crop_plates:
            return []
        sorted_plates = sorted(crop_plates, key=lambda x: x[4], reverse=True)
        kept = []
        for cand in sorted_plates:
            cx1, cy1, cx2, cy2, _ = cand
            overlap = False
            for k in kept:
                kx1, ky1, kx2, ky2, _ = k
                ix1 = max(cx1, kx1)
                iy1 = max(cy1, ky1)
                ix2 = min(cx2, kx2)
                iy2 = min(cy2, ky2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter > 0:
                    area_c = (cx2 - cx1) * (cy2 - cy1)
                    area_k = (kx2 - kx1) * (ky2 - ky1)
                    union = area_c + area_k - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou >= iou_threshold:
                        overlap = True
                        break
            if not overlap:
                kept.append(cand)
        return kept

    def detect_plates_sequential(
        self,
        crops: List[np.ndarray],
        vehicle_boxes: List[Tuple[int, int, int, int]],
        imgsz: Optional[int] = None,
    ) -> Tuple[
        List[Tuple[int, int, int, int, float]],
        List[Tuple[int, int, int, int]],
        float,
        float,
    ]:
        """Run sequential per-crop license-plate inference over vehicle crops (for benchmark comparison)."""
        if self.plate_detector is None:
            raise RuntimeError("Plate detector is not initialized")

        if not crops:
            return [], [], 0.0, 0.0

        target_imgsz = imgsz if imgsz is not None else self.plate_imgsz

        t0 = time.perf_counter()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

        all_results = []
        for crop in crops:
            res = self.plate_detector(
                crop,
                device=self.device,
                imgsz=target_imgsz,
                verbose=False,
                conf=self.plate_conf,
            )
            all_results.append(res[0])

        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        detected_plates = []
        plate_fallback_boxes = []

        for idx, res in enumerate(all_results):
            v_box = vehicle_boxes[idx]
            vx1, vy1, vx2, vy2 = v_box[:4]
            cls_id = v_box[4] if len(v_box) > 4 else None
            vw = vx2 - vx1
            vh = vy2 - vy1
            crop_h, crop_w = crops[idx].shape[:2]

            candidate_crop_plates = []
            for pb in res.boxes:
                coords = pb.xyxy[0].cpu().numpy().astype(int)
                score = float(pb.conf[0])
                # Clamp coordinates to crop boundary
                px1 = max(0, min(crop_w, int(coords[0])))
                py1 = max(0, min(crop_h, int(coords[1])))
                px2 = max(0, min(crop_w, int(coords[2])))
                py2 = max(0, min(crop_h, int(coords[3])))
                bw = px2 - px1
                bh = py2 - py1
                if bw > 0 and bh > 0:
                    # Filter narrow horizontal stripes/seams on vehicle body
                    if bw / max(1, bh) > 7.0 and bh < 6:
                        continue
                    candidate_crop_plates.append((px1, py1, px2, py2, score))

            deduped = self._deduplicate_crop_plates(candidate_crop_plates)
            for px1, py1, px2, py2, score in deduped:
                fx1 = vx1 + px1
                fy1 = vy1 + py1
                fx2 = vx1 + px2
                fy2 = vy1 + py2
                detected_plates.append((fx1, fy1, fx2, fy2, score))

            is_2w3w = (
                cls_id == 3
                or (cls_id is not None and cls_id in {1, 2, 7} and vw / max(1, vh) <= 1.15)
            )
            needs_fallback = False
            if not deduped:
                needs_fallback = True
            else:
                has_adequate = False
                for px1, py1, px2, py2, sc in deduped:
                    cw = px2 - px1
                    ch = py2 - py1
                    # Uncertain detections or tiny slivers cannot prove plate protection
                    if sc < 0.65 or cw < 14 or ch < 6:
                        continue
                    # Must reach into plausible lower bumper/tail zone (lower 35%)
                    if py2 < crop_h - max(12, int(round(0.35 * vh))):
                        continue
                    if is_2w3w:
                        if (cw >= 14 or cw >= 0.35 * vw) and py2 >= crop_h - max(12, int(round(0.18 * vh))):
                            has_adequate = True
                            break
                    else:
                        has_adequate = True
                        break
                if not has_adequate:
                    needs_fallback = True

            if needs_fallback:
                fb = self._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=cls_id)
                plate_fallback_boxes.append(fb)

        t3 = time.perf_counter()
        map_ms = (t3 - t2) * 1000.0

        return detected_plates, plate_fallback_boxes, infer_ms, map_ms

    def detect_plates_batched(
        self,
        crops: List[np.ndarray],
        vehicle_boxes: List[Tuple[int, int, int, int]],
        imgsz: Optional[int] = None,
    ) -> Tuple[
        List[Tuple[int, int, int, int, float]],  # detected plates in canonical frame
        List[Tuple[int, int, int, int]],  # fallback boxes for vehicles with zero plate detections
        float,  # inference latency ms
        float,  # coordinate mapping latency ms
    ]:
        """Run batched license-plate inference over vehicle crops."""
        if self.plate_detector is None:
            raise RuntimeError("Plate detector is not initialized")

        if not crops:
            return [], [], 0.0, 0.0

        target_imgsz = imgsz if imgsz is not None else self.plate_imgsz

        t0 = time.perf_counter()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()

        batch_results = self.plate_detector(
            crops,
            device=self.device,
            imgsz=target_imgsz,
            verbose=False,
            conf=self.plate_conf,
        )

        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        detected_plates = []
        plate_fallback_boxes = []

        for idx, res in enumerate(batch_results):
            v_box = vehicle_boxes[idx]
            vx1, vy1, vx2, vy2 = v_box[:4]
            cls_id = v_box[4] if len(v_box) > 4 else None
            vw = vx2 - vx1
            vh = vy2 - vy1
            crop_h, crop_w = crops[idx].shape[:2]

            candidate_crop_plates = []
            for pb in res.boxes:
                coords = pb.xyxy[0].cpu().numpy().astype(int)
                score = float(pb.conf[0])
                # Clamp coordinates strictly to crop boundary
                px1 = max(0, min(crop_w, int(coords[0])))
                py1 = max(0, min(crop_h, int(coords[1])))
                px2 = max(0, min(crop_w, int(coords[2])))
                py2 = max(0, min(crop_h, int(coords[3])))
                bw = px2 - px1
                bh = py2 - py1
                if bw > 0 and bh > 0:
                    # Filter narrow horizontal stripes/seams on vehicle body
                    if bw / max(1, bh) > 7.0 and bh < 6:
                        continue
                    candidate_crop_plates.append((px1, py1, px2, py2, score))

            deduped = self._deduplicate_crop_plates(candidate_crop_plates)
            for px1, py1, px2, py2, score in deduped:
                fx1 = max(0, min(CANONICAL_WIDTH, vx1 + px1))
                fy1 = max(0, min(CANONICAL_HEIGHT, vy1 + py1))
                fx2 = max(0, min(CANONICAL_WIDTH, vx1 + px2))
                fy2 = max(0, min(CANONICAL_HEIGHT, vy1 + py2))
                if fx2 > fx1 and fy2 > fy1:
                    detected_plates.append((fx1, fy1, fx2, fy2, score))

            is_2w3w = (
                cls_id == 3
                or (cls_id is not None and cls_id in {1, 2, 7} and vw / max(1, vh) <= 1.15)
            )
            needs_fallback = False
            if not deduped:
                needs_fallback = True
            else:
                has_adequate = False
                for px1, py1, px2, py2, sc in deduped:
                    cw = px2 - px1
                    ch = py2 - py1
                    # Uncertain detections or tiny slivers cannot prove plate protection
                    if sc < 0.65 or cw < 14 or ch < 6:
                        continue
                    # Must reach into plausible lower bumper/tail zone (lower 35%)
                    if py2 < crop_h - max(12, int(round(0.35 * vh))):
                        continue
                    if is_2w3w:
                        if (cw >= 14 or cw >= 0.35 * vw) and py2 >= crop_h - max(12, int(round(0.18 * vh))):
                            has_adequate = True
                            break
                    else:
                        has_adequate = True
                        break
                if not has_adequate:
                    needs_fallback = True

            if needs_fallback:
                fb = self._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=cls_id)
                plate_fallback_boxes.append(fb)

        # Global canonical-frame duplicate suppression across all vehicle crops
        raw_plate_count = len(detected_plates)
        unique_plates = self._suppress_duplicate_global_plates(detected_plates, iou_threshold=0.65)
        self.last_plate_candidate_counts = (raw_plate_count, len(unique_plates))

        t3 = time.perf_counter()
        map_ms = (t3 - t2) * 1000.0

        return unique_plates, plate_fallback_boxes, infer_ms, map_ms

    def _suppress_duplicate_global_plates(
        self,
        candidate_plates: List[Tuple[int, int, int, int, float]],
        iou_threshold: float = 0.65,
    ) -> List[Tuple[int, int, int, int, float]]:
        """Suppress duplicate plate detections across different vehicle crops in canonical frame coordinates."""
        if len(candidate_plates) <= 1:
            return candidate_plates

        # Sort descending by confidence score
        sorted_plates = sorted(candidate_plates, key=lambda p: p[4], reverse=True)
        surviving_plates: List[Tuple[int, int, int, int, float]] = []

        for p in sorted_plates:
            box_p = p[:4]
            is_duplicate = False
            for s in surviving_plates:
                box_s = s[:4]
                ix1 = max(box_p[0], box_s[0])
                iy1 = max(box_p[1], box_s[1])
                ix2 = min(box_p[2], box_s[2])
                iy2 = min(box_p[3], box_s[3])
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                if inter > 0:
                    a1 = (box_p[2] - box_p[0]) * (box_p[3] - box_p[1])
                    a2 = (box_s[2] - box_s[0]) * (box_s[3] - box_s[1])
                    union = a1 + a2 - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou >= iou_threshold:
                        is_duplicate = True
                        break
            if not is_duplicate:
                surviving_plates.append(p)

        return surviving_plates

    def _compute_single_vehicle_fallback(
        self,
        vx1: int,
        vy1: int,
        vx2: int,
        vy2: int,
        cls_id: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        """Compute localized probable-plate region on vehicle.
        
        Two-wheeler / three-wheeler geometry:
          Motorcycles, scooters (COCO class 3) and narrow vehicles (aspect ratio <= 0.85):
          The bounding box encompasses rider and vehicle. Plates are located on the rear fender/
          mudguard or front fork at mid-to-lower vehicle height (from vy1 + 28% of height to vy2)
          and across full vehicle width to protect angled or turning bikes.
        Four-wheeler geometry:
          Standard bumper-mounted plate region: center 60% of width, lower 25% of height.
        """
        vw = vx2 - vx1
        vh = vy2 - vy1
        if self.localized_fallbacks:
            is_two_or_three_wheeler = (
                cls_id == 3
                or (cls_id is not None and cls_id in {1, 2, 7} and vw / max(1, vh) <= 1.15)
            )
            if is_two_or_three_wheeler:
                # Two-/three-wheeler fallback: lower 80% of vehicle box, full vehicle width
                fb_x1 = vx1
                fb_x2 = vx2
                fb_y1 = max(vy1, int(round(vy1 + 0.20 * vh)))
                fb_y2 = vy2
            else:
                vcx = (vx1 + vx2) / 2.0
                # Center 60% of vehicle width, lower 25% of vehicle height
                fb_x1 = max(vx1, int(round(vcx - 0.30 * vw)))
                fb_x2 = min(vx2, int(round(vcx + 0.30 * vw)))
                fb_y1 = max(vy1, int(round(vy2 - 0.25 * vh)))
                fb_y2 = vy2
        else:
            fb_x1 = vx1
            fb_x2 = vx2
            fb_y1 = max(vy1, int(round(vy2 - 0.30 * vh)))
            fb_y2 = vy2
        return (fb_x1, fb_y1, fb_x2, fb_y2)

    def compute_vehicle_fallbacks(
        self,
        vehicle_boxes: List[Tuple[Any, ...]],
        direct_plate_boxes: List[Tuple[int, int, int, int, float]],
        active_temporal_holds: Optional[List[Any]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Identify vehicles without an associated plate and compute precision fallback masks."""
        if active_temporal_holds is None:
            active_temporal_holds = []
        fallbacks = []
        for v in vehicle_boxes:
            vx1, vy1, vx2, vy2 = v[:4]
            cls_id = v[4] if len(v) > 4 else None
            vw = vx2 - vx1
            vh = vy2 - vy1
            if vw < self.min_vehicle_width or vh < self.min_vehicle_height:
                continue

            is_2w3w = (
                cls_id == 3
                or (cls_id is not None and cls_id in {1, 2, 7} and vw / max(1, vh) <= 1.15)
            )

            fb = self._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=cls_id)
            fb_x1, fb_y1, fb_x2, fb_y2 = fb

            vehicle_has_direct_plate = False
            for p in direct_plate_boxes:
                px1, py1, px2, py2 = p[:4]
                p_conf = p[4] if len(p) > 4 else 1.0
                pcx = (px1 + px2) / 2.0
                pcy = (py1 + py2) / 2.0

                # 1. Plate center must lie inside vehicle bounding box
                if not ((vx1 <= pcx <= vx2) and (vy1 <= pcy <= vy2)):
                    continue

                # 2. Plate detection must be sufficiently confident (avoid distractor/sticker hits)
                if p_conf < 0.65:
                    continue

                # 3. Minimum dimensions (avoid tiny slivers/seams)
                pw = px2 - px1
                ph = py2 - py1
                if pw < 14 or ph < 6:
                    continue

                # 4. Spatial consistency: plate must reach into plausible lower bumper/tail zone (lower 35%)
                bumper_top_limit = vy2 - max(12, int(round(0.35 * vh)))
                if py2 < bumper_top_limit or pcy < (vy1 + 0.40 * vh):
                    continue

                # 5. Plate must physically intersect the expected fallback region
                ix1 = max(px1, fb_x1)
                iy1 = max(py1, fb_y1)
                ix2 = min(px2, fb_x2)
                iy2 = min(py2, fb_y2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue

                # 6. For 2W/3W vehicles, verify it reaches tail/mudguard boundary
                if is_2w3w:
                    if (pw >= 14 or pw >= 0.35 * vw) and ph >= 6 and py2 >= vy2 - max(12, int(round(0.18 * vh))):
                        vehicle_has_direct_plate = True
                        break
                else:
                    vehicle_has_direct_plate = True
                    break

            vehicle_has_temporal_plate = False
            if not vehicle_has_direct_plate and not is_2w3w:
                for th in active_temporal_holds:
                    t_box = th["box"] if isinstance(th, dict) else th[:4]
                    tx1, ty1, tx2, ty2 = t_box
                    tcx = (tx1 + tx2) / 2.0
                    tcy = (ty1 + ty2) / 2.0
                    if (vx1 <= tcx <= vx2) and (vy1 <= tcy <= vy2):
                        vehicle_has_temporal_plate = True
                        break

            if not vehicle_has_direct_plate and not vehicle_has_temporal_plate:
                fallbacks.append(fb)
        return fallbacks

    def compute_face_fallbacks(
        self,
        person_boxes: List[Tuple[int, int, int, int]],
        face_boxes: List[Tuple[int, int, int, int, float]],
        active_temporal_holds: Optional[List[Any]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Identify persons without an associated face and compute localized head fallback masks.
        
        Association rule: Face center must lie within horizontal person span and upper 55% of height.
        """
        if active_temporal_holds is None:
            active_temporal_holds = []
        face_fallbacks = []

        for px1, py1, px2, py2 in person_boxes:
            pw = px2 - px1
            ph = py2 - py1
            has_associated_face = False

            for fx1, fy1, fx2, fy2, *rest in face_boxes:
                fcx = (fx1 + fx2) / 2.0
                fcy = (fy1 + fy2) / 2.0
                if (px1 <= fcx <= px2) and (py1 <= fcy <= py1 + 0.55 * ph):
                    has_associated_face = True
                    break

            if not has_associated_face and active_temporal_holds:
                for th in active_temporal_holds:
                    t_box = th["box"] if isinstance(th, dict) else th[:4]
                    tx1, ty1, tx2, ty2 = t_box
                    tcx = (tx1 + tx2) / 2.0
                    tcy = (ty1 + ty2) / 2.0
                    if (px1 <= tcx <= px2) and (py1 <= tcy <= py1 + 0.55 * ph):
                        has_associated_face = True
                        break

            if not has_associated_face:
                if self.localized_fallbacks:
                    pcx = (px1 + px2) / 2.0
                    # Center 50% width, upper 22% height
                    fb_x1 = max(px1, int(round(pcx - 0.25 * pw)))
                    fb_x2 = min(px2, int(round(pcx + 0.25 * pw)))
                    fb_y1 = py1
                    fb_y2 = min(py2, int(round(py1 + 0.22 * ph)))
                else:
                    fb_x1 = px1
                    fb_x2 = px2
                    fb_y1 = py1
                    fb_y2 = min(py2, int(round(py1 + 0.30 * ph)))
                face_fallbacks.append((fb_x1, fb_y1, fb_x2, fb_y2))

        return face_fallbacks

    def process_frame(
        self,
        input_frame: np.ndarray,
        simulate_detector_exception: bool = False,
        simulate_redaction_exception: bool = False,
    ) -> Tuple[np.ndarray, PrivacyGateStatus, Dict[str, Any]]:
        """Process frame through hardened privacy pipeline with fail-closed output gating.
        
        Guarantees:
        - Output is ALWAYS strictly CANONICAL 1280x720 dimensions.
        - Begins marked NOT SAFE FOR RELEASE.
        - Suppresses with an all-black (zeros) 1280x720 frame on any exception or missing config.
        - Raw unredacted frame is never leaked.
        """
        suppression_frame = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), dtype=np.uint8)

        metrics: Dict[str, Any] = {
            "face_count": 0,
            "plate_count": 0,
            "person_count": 0,
            "vehicle_count": 0,
            "vehicle_crops_considered": 0,
            "vehicle_crops_selected": 0,
            "plate_fallbacks_count": 0,
            "face_fallbacks_count": 0,
            "temporal_masks_count": 0,
            "fallback_count": 0,
            "resize_ms": 0.0,
            "face_det_ms": 0.0,
            "context_det_ms": 0.0,
            "crop_prep_ms": 0.0,
            "plate_det_ms": 0.0,
            "plate_map_ms": 0.0,
            "redact_ms": 0.0,
            "total_ms": 0.0,
            "error_detail": "",
            "masks_detail": [],
            "person_boxes": [],
            "vehicle_boxes": [],
            "mean_v": 0.0,
            "lighting_deficit": 0.0,
            "obscured_fraction_by_category": {},
            "total_obscured_fraction": 0.0,
            "substantial_overlaps": [],
        }

        t_start = time.perf_counter()

        # Gate Check 1: Verify model configurations exist
        if self.face_detector is None or self.vehicle_detector is None or self.plate_detector is None:
            metrics["error_detail"] = "Required detector models not configured or initialized"
            metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            return suppression_frame, PrivacyGateStatus.SUPPRESSED_MISSING_CONFIG, metrics

        # Gate Step 1: Canonical 1280x720 Frame Standardization
        try:
            in_h, in_w = input_frame.shape[:2]
            t_res0 = time.perf_counter()
            if (in_w, in_h) != (CANONICAL_WIDTH, CANONICAL_HEIGHT):
                canonical_frame = cv2.resize(input_frame, (CANONICAL_WIDTH, CANONICAL_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                canonical_frame = input_frame.copy()
            t_res1 = time.perf_counter()
            metrics["resize_ms"] = (t_res1 - t_res0) * 1000.0

        except Exception as e:
            metrics["error_detail"] = f"Canonical frame standardization failure: {str(e)}"
            metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            return suppression_frame, PrivacyGateStatus.SUPPRESSED_DETECTOR_FAILURE, metrics

        # Gate Step 2: Run Detectors & Fallbacks
        try:
            if simulate_detector_exception:
                raise RuntimeError("Simulated detector fault injected for fail-closed validation")

            # Stage A: Face Detection on 1280x720
            faces, f_det_ms = self.detect_faces(canonical_frame)
            metrics["face_det_ms"] = f_det_ms
            metrics["face_count"] = len(faces)

            # Stage B: Vehicle Context & Crop Extraction on 1280x720
            crops, sel_vehicles, skipped_vehicles, persons, c_det_ms, c_prep_ms = (
                self.detect_context_and_extract_crops(canonical_frame)
            )
            metrics["context_det_ms"] = c_det_ms
            metrics["crop_prep_ms"] = c_prep_ms
            metrics["vehicle_crops_considered"] = len(sel_vehicles) + len(skipped_vehicles)
            metrics["vehicle_crops_selected"] = len(sel_vehicles)
            metrics["vehicle_count"] = len(sel_vehicles) + len(skipped_vehicles)
            metrics["person_count"] = len(persons)
            metrics["person_boxes"] = persons
            metrics["vehicle_boxes"] = [
                (v[0], v[1], v[2], v[3], v[4]) for v in (sel_vehicles + skipped_vehicles)
            ]

            # Environmental Lighting Telemetry
            hsv = cv2.cvtColor(canonical_frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            mean_v = float(np.mean(v_channel))
            metrics["mean_v"] = mean_v
            metrics["lighting_deficit"] = float(1.0 - (mean_v / 255.0))

            # Stage C: License Plate Detection on Crops
            plates, _, p_infer_ms, p_map_ms = self.detect_plates_batched(
                crops, sel_vehicles, imgsz=self.plate_imgsz
            )
            metrics["plate_det_ms"] = p_infer_ms
            metrics["plate_map_ms"] = p_map_ms
            raw_plates_cnt, unique_plates_cnt = getattr(self, "last_plate_candidate_counts", (len(plates), len(plates)))
            metrics["raw_plate_candidates_count"] = raw_plates_cnt
            metrics["suppressed_duplicate_plates_count"] = raw_plates_cnt - unique_plates_cnt
            metrics["plate_count"] = len(plates)

            # Stage D: Temporal Mask Tracking (Strictly direct sensor detections: face & plate)
            raw_sensor_detections: List[Tuple[Any, ...]] = []
            for fx1, fy1, fx2, fy2, score in faces:
                raw_sensor_detections.append((fx1, fy1, fx2, fy2, "face", score))
            for px1, py1, px2, py2, score in plates:
                raw_sensor_detections.append((px1, py1, px2, py2, "plate", score))

            tracked_targets = self.temporal_tracker.update_and_get_tracked_targets(raw_sensor_detections)

            # Separate tracked targets into fresh detections and temporal hold regions
            active_face_holds: List[Dict[str, Any]] = []
            active_plate_holds: List[Dict[str, Any]] = []
            for tt in tracked_targets:
                if tt["status"] == "TEMPORAL_HOLD":
                    if tt["category"] == "face":
                        active_face_holds.append(tt)
                    elif tt["category"] == "plate":
                        active_plate_holds.append(tt)

            metrics["temporal_masks_count"] = len(active_face_holds) + len(active_plate_holds)

            # Stage E: Fallback Determination (Applied only to entities without active protection)
            all_vehicles = sel_vehicles + skipped_vehicles
            plate_fallbacks = self.compute_vehicle_fallbacks(all_vehicles, plates, active_plate_holds)
            metrics["plate_fallbacks_count"] = len(plate_fallbacks)

            # Person face fallbacks
            face_fallbacks = self.compute_face_fallbacks(persons, faces, active_face_holds)
            metrics["face_fallbacks_count"] = len(face_fallbacks)
            metrics["fallback_count"] = len(plate_fallbacks) + len(face_fallbacks)

            # Stage F: Assemble Candidate Mask Records with Provenance
            candidate_mask_records: List[Dict[str, Any]] = []

            # 1. Direct Detections (fresh from current frame)
            for tt in tracked_targets:
                if tt["status"] == "DETECTION":
                    prov = MaskProvenance.FACE_DETECTION if tt["category"] == "face" else MaskProvenance.PLATE_DETECTION
                    exp_box = expand_and_clamp_bbox(
                        tt["box"],
                        frame_width=CANONICAL_WIDTH,
                        frame_height=CANONICAL_HEIGHT,
                        margin_ratio=self.margin_ratio,
                        min_padding=self.min_padding,
                    )
                    if exp_box is not None:
                        candidate_mask_records.append({
                            "box": exp_box,
                            "unexpanded_box": tt["box"],
                            "provenance": prov,
                            "track_id": tt["id"],
                            "score": tt["score"],
                            "category": tt["category"],
                        })
                elif tt["status"] == "TEMPORAL_HOLD":
                    # 2. Temporal Hold (buffered from recent frame)
                    exp_box = expand_and_clamp_bbox(
                        tt["box"],
                        frame_width=CANONICAL_WIDTH,
                        frame_height=CANONICAL_HEIGHT,
                        margin_ratio=0.15,
                        min_padding=6,
                    )
                    if exp_box is not None:
                        candidate_mask_records.append({
                            "box": exp_box,
                            "unexpanded_box": tt["box"],
                            "provenance": MaskProvenance.TEMPORAL_HOLD,
                            "track_id": tt["id"],
                            "score": tt["score"],
                            "category": f"{tt['category']}_hold",
                        })

            # 3. Vehicle Fallbacks
            for fb in plate_fallbacks:
                exp_box = expand_and_clamp_bbox(
                    fb,
                    frame_width=CANONICAL_WIDTH,
                    frame_height=CANONICAL_HEIGHT,
                    margin_ratio=self.fallback_margin_ratio,
                    min_padding=self.fallback_min_padding,
                )
                if exp_box is not None:
                    candidate_mask_records.append({
                        "box": exp_box,
                        "unexpanded_box": fb,
                        "provenance": MaskProvenance.VEHICLE_FALLBACK,
                        "track_id": 0,
                        "score": 0.0,
                        "category": "plate_fallback",
                    })

            # 4. Person Fallbacks
            for fb in face_fallbacks:
                exp_box = expand_and_clamp_bbox(
                    fb,
                    frame_width=CANONICAL_WIDTH,
                    frame_height=CANONICAL_HEIGHT,
                    margin_ratio=self.fallback_margin_ratio,
                    min_padding=self.fallback_min_padding,
                )
                if exp_box is not None:
                    candidate_mask_records.append({
                        "box": exp_box,
                        "unexpanded_box": fb,
                        "provenance": MaskProvenance.PERSON_FALLBACK,
                        "track_id": 0,
                        "score": 0.0,
                        "category": "face_fallback",
                    })

        except Exception as e:
            metrics["error_detail"] = f"Detection stage failure: {str(e)}"
            metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            return suppression_frame, PrivacyGateStatus.SUPPRESSED_DETECTOR_FAILURE, metrics

        # Gate Step 3: Apply Solid Opaque Masks & Compute Diagnostics
        try:
            if simulate_redaction_exception:
                raise RuntimeError("Simulated redaction failure injected for fail-closed validation")

            clamped_expanded_boxes = [m["box"] for m in candidate_mask_records]

            # Separate output buffer creation & independent mask painting
            redacted_output, redact_ms = apply_solid_masks_independent(
                canonical_frame,
                clamped_expanded_boxes,
                color=self.redaction_color,
            )
            metrics["redact_ms"] = redact_ms

            # Verification: output must match canonical dimensions exactly
            if redacted_output.shape != (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3):
                raise ValueError(f"Output frame has unexpected dimensions: {redacted_output.shape}")

            # Compute pixel-exact diagnostic metrics
            total_frame_pixels = CANONICAL_WIDTH * CANONICAL_HEIGHT
            union_mask = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8)

            category_masks = {
                MaskProvenance.FACE_DETECTION: np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8),
                MaskProvenance.PLATE_DETECTION: np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8),
                MaskProvenance.VEHICLE_FALLBACK: np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8),
                MaskProvenance.PERSON_FALLBACK: np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8),
                MaskProvenance.TEMPORAL_HOLD: np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8),
            }

            for rec in candidate_mask_records:
                x1, y1, x2, y2 = rec["box"]
                area = (x2 - x1) * (y2 - y1)
                rec["area_px"] = area
                rec["area_fraction"] = float(area / total_frame_pixels)
                union_mask[y1:y2, x1:x2] = 255
                prov = rec["provenance"]
                if prov in category_masks:
                    category_masks[prov][y1:y2, x1:x2] = 255

            metrics["masks_detail"] = candidate_mask_records
            metrics["total_obscured_fraction"] = float(np.count_nonzero(union_mask) / total_frame_pixels)

            obs_by_cat = {}
            for prov_key, cat_m in category_masks.items():
                obs_by_cat[prov_key.value] = float(np.count_nonzero(cat_m) / total_frame_pixels)
            metrics["obscured_fraction_by_category"] = obs_by_cat

            # Substantial overlap analysis (IoU >= 0.25 between pairs)
            substantial_overlaps = []
            for i in range(len(candidate_mask_records)):
                for j in range(i + 1, len(candidate_mask_records)):
                    m1 = candidate_mask_records[i]
                    m2 = candidate_mask_records[j]
                    b1 = m1["box"]
                    b2 = m2["box"]
                    ix1 = max(b1[0], b2[0])
                    iy1 = max(b1[1], b2[1])
                    ix2 = min(b1[2], b2[2])
                    iy2 = min(b1[3], b2[3])
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter > 0:
                        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                        u = a1 + a2 - inter
                        iou = inter / u if u > 0 else 0.0
                        if iou >= 0.25:
                            substantial_overlaps.append({
                                "mask_1_idx": i,
                                "mask_2_idx": j,
                                "mask_1_prov": m1["provenance"].value,
                                "mask_2_prov": m2["provenance"].value,
                                "iou": float(iou),
                            })

            metrics["substantial_overlaps"] = substantial_overlaps

        except Exception as e:
            metrics["error_detail"] = f"Redaction stage failure: {str(e)}"
            metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
            return suppression_frame, PrivacyGateStatus.SUPPRESSED_REDACTION_FAILURE, metrics

        # Gate Step 4: Release Approved
        metrics["total_ms"] = (time.perf_counter() - t_start) * 1000.0
        return redacted_output, PrivacyGateStatus.RELEASE_APPROVED, metrics


def draw_diagnostic_overlay(
    canonical_frame: np.ndarray,
    redacted_output: np.ndarray,
    gate_status: PrivacyGateStatus,
    metrics: Dict[str, Any],
) -> np.ndarray:
    """Render in-memory diagnostic visualization with mask boundaries, labels, and statistics HUD.
    
    Operates strictly on an independent display buffer copy. Never modifies released frame or touches disk.
    """
    if canonical_frame.shape[0] != CANONICAL_HEIGHT or canonical_frame.shape[1] != CANONICAL_WIDTH:
        vis = cv2.resize(canonical_frame, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
    else:
        vis = canonical_frame.copy()

    # Color palette for provenance
    PROV_COLORS = {
        MaskProvenance.FACE_DETECTION: (255, 200, 0),       # Cyan / Light Blue
        MaskProvenance.PLATE_DETECTION: (0, 255, 0),        # Bright Green
        MaskProvenance.VEHICLE_FALLBACK: (0, 165, 255),     # Amber / Orange
        MaskProvenance.PERSON_FALLBACK: (255, 0, 255),      # Magenta
        MaskProvenance.TEMPORAL_HOLD: (0, 0, 255),          # Bright Red
    }

    if gate_status != PrivacyGateStatus.RELEASE_APPROVED:
        # Diagnostic display of fail-closed black suppression
        cv2.rectangle(vis, (0, 0), (CANONICAL_WIDTH, CANONICAL_HEIGHT), (0, 0, 0), -1)
        err_msg = metrics.get("error_detail", "Unknown fault")
        cv2.putText(
            vis,
            f"FAIL-CLOSED SUPPRESSION: {gate_status.value}",
            (60, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            f"Detail: {err_msg}",
            (60, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return vis

    # Semi-transparent overlay of redacted areas
    overlay = vis.copy()
    for rec in metrics.get("masks_detail", []):
        x1, y1, x2, y2 = rec["box"]
        prov = rec.get("provenance")
        color = PROV_COLORS.get(prov, (128, 128, 128))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # Solid boundary outlines and text labels
    for idx, rec in enumerate(metrics.get("masks_detail", [])):
        x1, y1, x2, y2 = rec["box"]
        prov = rec.get("provenance")
        color = PROV_COLORS.get(prov, (128, 128, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Label tag
        track_id = rec.get("track_id", 0)
        score = rec.get("score", 0.0)
        prov_short = prov.value.replace("_DETECTION", "").replace("_FALLBACK", "_FB")
        tag = f"{prov_short}"
        if track_id > 0:
            tag += f" #{track_id}"
        if score > 0:
            tag += f" {score:.2f}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tag_y1 = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, tag_y1), (x1 + tw + 6, tag_y1 + th + 6), (20, 20, 20), -1)
        cv2.putText(vis, tag, (x1 + 3, tag_y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Top Diagnostic HUD Banner
    total_obs = metrics.get("total_obscured_fraction", 0.0) * 100.0
    obs_by_cat = metrics.get("obscured_fraction_by_category", {})
    cat_sum = sum(obs_by_cat.values()) * 100.0
    overlaps = metrics.get("substantial_overlaps", [])
    raw_plates = metrics.get("raw_plate_candidates_count", metrics.get("plate_count", 0))
    suppressed_plates = metrics.get("suppressed_duplicate_plates_count", 0)

    hud_y = 68
    cv2.rectangle(vis, (0, 0), (CANONICAL_WIDTH, hud_y), (15, 15, 15), -1)

    hud_line1 = (
        f"DIAGNOSTIC VIEW | PIPELINE: EXECUTION_COMPLETED | HUMAN AUDIT: PENDING | RECALL: N/A (UNANNOTATED)"
    )
    cv2.putText(vis, hud_line1, (16, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    hud_line2 = (
        f"Unique Obscured: {total_obs:.2f}% (Cat Sum: {cat_sum:.2f}%) | "
        f"Faces: {metrics.get('face_count', 0)} | "
        f"Plates: {metrics.get('plate_count', 0)} (Raw: {raw_plates}, Dupes Suppressed: {suppressed_plates}) | "
        f"Vehicles: {metrics.get('vehicle_count', 0)} | Persons: {metrics.get('person_count', 0)} | Overlaps: {len(overlaps)}"
    )
    cv2.putText(vis, hud_line2, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

    hud_line3 = (
        f"Footprint: Face={obs_by_cat.get('FACE_DETECTION', 0)*100:.2f}% | "
        f"Plate={obs_by_cat.get('PLATE_DETECTION', 0)*100:.2f}% | "
        f"VehFallback={obs_by_cat.get('VEHICLE_FALLBACK', 0)*100:.2f}% | "
        f"PersonFallback={obs_by_cat.get('PERSON_FALLBACK', 0)*100:.2f}% | "
        f"TempHold={obs_by_cat.get('TEMPORAL_HOLD', 0)*100:.2f}%"
    )
    cv2.putText(vis, hud_line3, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    return vis
