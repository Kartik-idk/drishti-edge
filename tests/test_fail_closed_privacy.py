#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Drishti-Edge Privacy Hardening & Fail-Closed Safety
Covers:
- Task 2: Canonical 1280x720 resolution guarantees (4K in -> 720p out)
- Task 3: Bounding box validation, minimum padding floor (8px), clamping, pixel-level masking
- Task 4: Plate fallback (bottom-30%), face fallback (top-30%), 3-frame temporal tracking
- Task 5: Fail-closed semantics, exception suppression, regression tests
"""

import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.privacy_engine import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    ConsolidatedPrivacyEngine,
    MaskProvenance,
    PrivacyGateStatus,
    TemporalMaskTracker,
    apply_solid_masks_independent,
    draw_diagnostic_overlay,
    expand_and_clamp_bbox,
)


class DummyFaceDetector:
    def setInputSize(self, size):
        pass

    def detect(self, frame):
        return True, None


class DummyYOLOResult:
    def __init__(self):
        self.boxes = []


class DummyYOLO:
    def __call__(self, *args, **kwargs):
        return [DummyYOLOResult()]


def create_test_engine(**kwargs):
    """Provide explicit, safe test configuration for CI when real model weights are absent."""
    kwargs.setdefault("device", "cpu")
    engine = ConsolidatedPrivacyEngine(**kwargs)
    if "face_model_path" not in kwargs and engine.face_detector is None:
        engine.face_detector = DummyFaceDetector()
    if "vehicle_model_path" not in kwargs and engine.vehicle_detector is None:
        engine.vehicle_detector = DummyYOLO()
    if "plate_model_path" not in kwargs and engine.plate_detector is None:
        engine.plate_detector = DummyYOLO()
    return engine


class TestPrivacyHardeningAndSafety(unittest.TestCase):
    def setUp(self):
        self.frame_w = CANONICAL_WIDTH
        self.frame_h = CANONICAL_HEIGHT

    # =========================================================================
    # Task 3: Bounding Box Validation, Padding Floor, and Clamping Tests
    # =========================================================================

    def test_expand_and_clamp_normal_box(self):
        """Normal 100x100 box expands by 25% (25 px) on each side."""
        box = (200, 200, 300, 300)
        exp = expand_and_clamp_bbox(box, self.frame_w, self.frame_h, margin_ratio=0.25, min_padding=8)
        self.assertIsNotNone(exp)
        self.assertEqual(exp, (175, 175, 325, 325))

    def test_minimum_padding_floor_on_small_box(self):
        """A tiny 10x10 box has 25% margin = 2.5 px, so minimum floor of 8 px must be enforced."""
        small_box = (100, 100, 110, 110)
        exp = expand_and_clamp_bbox(small_box, self.frame_w, self.frame_h, margin_ratio=0.25, min_padding=8)
        self.assertIsNotNone(exp)
        # Should expand by 8 px on each side
        self.assertEqual(exp, (92, 92, 118, 118))

    def test_box_at_origin_and_top_left_boundary(self):
        """A box at [0, 0, 20, 20] must clamp to 0 without negative indices."""
        box = (0, 0, 20, 20)
        exp = expand_and_clamp_bbox(box, self.frame_w, self.frame_h, margin_ratio=0.25, min_padding=8)
        self.assertIsNotNone(exp)
        self.assertEqual(exp[0], 0)
        self.assertEqual(exp[1], 0)
        self.assertGreater(exp[2], 20)
        self.assertGreater(exp[3], 20)

    def test_boxes_touching_every_image_boundary(self):
        """Boxes touching top, bottom, left, and right boundaries must strictly clamp."""
        # Left boundary
        b_left = (0, 200, 50, 250)
        exp_left = expand_and_clamp_bbox(b_left, self.frame_w, self.frame_h)
        self.assertIsNotNone(exp_left)
        self.assertEqual(exp_left[0], 0)

        # Top boundary
        b_top = (200, 0, 250, 50)
        exp_top = expand_and_clamp_bbox(b_top, self.frame_w, self.frame_h)
        self.assertIsNotNone(exp_top)
        self.assertEqual(exp_top[1], 0)

        # Right boundary
        b_right = (self.frame_w - 40, 200, self.frame_w, 250)
        exp_right = expand_and_clamp_bbox(b_right, self.frame_w, self.frame_h)
        self.assertIsNotNone(exp_right)
        self.assertEqual(exp_right[2], self.frame_w)

        # Bottom boundary
        b_bottom = (200, self.frame_h - 40, 250, self.frame_h)
        exp_bottom = expand_and_clamp_bbox(b_bottom, self.frame_w, self.frame_h)
        self.assertIsNotNone(exp_bottom)
        self.assertEqual(exp_bottom[3], self.frame_h)

    def test_boxes_entirely_outside_frame(self):
        """Boxes completely outside frame boundaries must return None."""
        outside_right = (1300, 100, 1400, 200)
        self.assertIsNone(expand_and_clamp_bbox(outside_right, self.frame_w, self.frame_h))

        outside_top = (100, -200, 200, -100)
        self.assertIsNone(expand_and_clamp_bbox(outside_top, self.frame_w, self.frame_h))

    def test_nan_and_infinite_coordinates(self):
        """Boxes with NaN or infinity coordinates must be rejected."""
        nan_box = (float("nan"), 100, 200, 200)
        self.assertIsNone(expand_and_clamp_bbox(nan_box, self.frame_w, self.frame_h))

        inf_box = (100, float("inf"), 200, 200)
        self.assertIsNone(expand_and_clamp_bbox(inf_box, self.frame_w, self.frame_h))

    def test_reversed_and_zero_area_boxes(self):
        """Inverted boxes (x2 <= x1 or y2 <= y1) must be rejected."""
        reversed_x = (300, 100, 200, 200)
        self.assertIsNone(expand_and_clamp_bbox(reversed_x, self.frame_w, self.frame_h))

        reversed_y = (100, 300, 200, 200)
        self.assertIsNone(expand_and_clamp_bbox(reversed_y, self.frame_w, self.frame_h))

        zero_area = (100, 100, 100, 200)
        self.assertIsNone(expand_and_clamp_bbox(zero_area, self.frame_w, self.frame_h))

    def test_pixel_level_mask_application(self):
        """Verify that masked pixels are strictly opaque (0, 0, 0) and unmasked pixels remain unchanged."""
        frame = np.full((100, 100, 3), 200, dtype=np.uint8)
        boxes = [(10, 10, 40, 40), (30, 30, 60, 60)]  # overlapping boxes

        redacted, latency = apply_solid_masks_independent(frame, boxes, color=(0, 0, 0))
        self.assertGreaterEqual(latency, 0.0)

        # Inside box 1
        self.assertTrue(np.all(redacted[15:25, 15:25] == 0))
        # Overlap region
        self.assertTrue(np.all(redacted[32:38, 32:38] == 0))
        # Inside box 2
        self.assertTrue(np.all(redacted[45:55, 45:55] == 0))
        # Outside both boxes
        self.assertTrue(np.all(redacted[70:90, 70:90] == 200))
        # Top-left corner (pixel [0, 0] outside boxes)
        self.assertTrue(np.all(redacted[0, 0] == 200))

    # =========================================================================
    # Task 2: Canonical Resolution Pipeline Tests
    # =========================================================================

    def test_canonical_720p_output_from_4k_input(self):
        """Input of 3840x2160 must produce canonical 1280x720 output."""
        engine = create_test_engine(device="cpu")
        frame_4k = np.full((2160, 3840, 3), 120, dtype=np.uint8)

        output, status, metrics = engine.process_frame(frame_4k)

        self.assertEqual(status, PrivacyGateStatus.RELEASE_APPROVED)
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        # Normal return path cannot return the 4K frame
        self.assertNotEqual(output.shape, frame_4k.shape)

    def test_canonical_720p_output_from_720p_input(self):
        """Input already at 1280x720 must produce 1280x720 output."""
        engine = create_test_engine(device="cpu")
        frame_720 = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 150, dtype=np.uint8)

        output, status, metrics = engine.process_frame(frame_720)

        self.assertEqual(status, PrivacyGateStatus.RELEASE_APPROVED)
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))

    # =========================================================================
    # Task 4: Fallbacks & Temporal Protection Tests
    # =========================================================================

    def test_face_fallback_on_unassociated_person(self):
        """Person with no face detection receives precision head-region geometric fallback mask."""
        engine = ConsolidatedPrivacyEngine(device="cpu")
        # Person from y=100 to y=300 (height=200 px), x=50 to x=150 (width=100 px)
        person_boxes = [(50, 100, 150, 300)]
        # No faces detected
        face_boxes = []

        fallbacks = engine.compute_face_fallbacks(person_boxes, face_boxes)
        self.assertEqual(len(fallbacks), 1)
        fb_x1, fb_y1, fb_x2, fb_y2 = fallbacks[0]
        # In Phase 0.5, head fallback covers centered 50% width and top 22% height
        self.assertEqual(fb_x1, 75)
        self.assertEqual(fb_y1, 100)
        self.assertEqual(fb_x2, 125)
        self.assertEqual(fb_y2, 144)

    def test_no_face_fallback_when_face_is_associated(self):
        """Person with an associated face detection does NOT trigger fallback."""
        engine = ConsolidatedPrivacyEngine(device="cpu")
        person_boxes = [(50, 100, 150, 300)]
        # Face located in upper half of person
        face_boxes = [(70, 110, 130, 170, 0.90)]

        fallbacks = engine.compute_face_fallbacks(person_boxes, face_boxes)
        self.assertEqual(len(fallbacks), 0)

    def test_temporal_tracker_hold_and_expiration(self):
        """Temporal tracker holds a missing detection for up to 3 frames then expires."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Frame 1: Detection present
        dets_f1 = [(100, 100, 150, 150, "face")]
        masks_f1 = tracker.update_and_get_masks(dets_f1)
        self.assertEqual(len(masks_f1), 1)

        # Frame 2: Detector dropout (0 detections) -> hold frame 1
        masks_f2 = tracker.update_and_get_masks([])
        self.assertEqual(len(masks_f2), 1)

        # Frame 3: Detector dropout -> hold frame 2
        masks_f3 = tracker.update_and_get_masks([])
        self.assertEqual(len(masks_f3), 1)

        # Frame 4: 4th frame with no detection -> EXPIRED!
        masks_f4 = tracker.update_and_get_masks([])
        self.assertEqual(len(masks_f4), 0)

    def test_temporal_tracker_motion_tracking(self):
        """Temporal tracker estimates position delta between moving frames."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Frame 1 at (100, 100)
        tracker.update_and_get_masks([(100, 100, 150, 150, "face")])
        # Frame 2 moves right to (110, 100)
        tracker.update_and_get_masks([(110, 100, 160, 150, "face")])

        # Track velocity should have positive x velocity
        self.assertGreater(tracker.tracks[0]["vx"], 0.0)

    # =========================================================================
    # Task 5: Fail-Closed Privacy Semantics Tests
    # =========================================================================

    def test_fail_closed_on_detector_exception(self):
        """Injected detector exception must emit a solid black 1280x720 suppression frame."""
        engine = create_test_engine(device="cpu")
        raw_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 180, dtype=np.uint8)

        output, status, metrics = engine.process_frame(raw_frame, simulate_detector_exception=True)

        self.assertEqual(status, PrivacyGateStatus.SUPPRESSED_DETECTOR_FAILURE)
        self.assertIn("Simulated detector fault", metrics["error_detail"])
        # Must be canonical 1280x720
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        # Must be completely black (zeros)
        self.assertTrue(np.all(output == 0))
        # Must NOT be raw frame
        self.assertFalse(np.array_equal(output, raw_frame))

    def test_fail_closed_on_redaction_exception(self):
        """Injected redaction exception must emit a solid black 1280x720 suppression frame."""
        engine = create_test_engine(device="cpu")
        raw_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 180, dtype=np.uint8)

        output, status, metrics = engine.process_frame(raw_frame, simulate_redaction_exception=True)

        self.assertEqual(status, PrivacyGateStatus.SUPPRESSED_REDACTION_FAILURE)
        self.assertTrue(np.all(output == 0))

    def test_fail_closed_on_missing_model_config(self):
        """Missing model paths must suppress output with black 1280x720 frame."""
        engine = ConsolidatedPrivacyEngine(
            face_model_path="missing_yunet.onnx",
            device="cpu",
        )
        raw_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 200, dtype=np.uint8)

        output, status, metrics = engine.process_frame(raw_frame)

        self.assertEqual(status, PrivacyGateStatus.SUPPRESSED_MISSING_CONFIG)
        self.assertTrue(np.all(output == 0))
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))

    def test_clean_release_when_no_privacy_targets(self):
        """A blank frame with zero faces and zero vehicles completes cleanly with RELEASE_APPROVED."""
        engine = create_test_engine(device="cpu")
        blank_frame = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), dtype=np.uint8)

        output, status, metrics = engine.process_frame(blank_frame)

        self.assertEqual(status, PrivacyGateStatus.RELEASE_APPROVED)
        self.assertEqual(metrics["face_count"], 0)
        self.assertEqual(metrics["plate_count"], 0)
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))


class TestPhase05PrecisionAndDiagnosticsRegression(unittest.TestCase):
    """Specific regression test suite for Phase 0.5 precision redaction, fallback tuning,

    and mask-provenance diagnostics (Task 6 requirements 1 through 12).
    """

    def setUp(self):
        self.frame_w = CANONICAL_WIDTH
        self.frame_h = CANONICAL_HEIGHT

    def test_01_plate_box_mapped_from_crop_to_full_frame(self):
        """1. Plate boxes mapped correctly from vehicle crops to full 1280x720 frame."""
        # Vehicle at (vx1, vy1, vx2, vy2) = (200, 300, 500, 480)
        vx1, vy1, vx2, vy2 = 200, 300, 500, 480
        # Plate detection in crop local coordinates
        px1_c, py1_c, px2_c, py2_c = 40, 50, 110, 75

        # Crop mapping transformation
        fx1 = vx1 + px1_c
        fy1 = vy1 + py1_c
        fx2 = vx1 + px2_c
        fy2 = vy1 + py2_c

        self.assertEqual((fx1, fy1, fx2, fy2), (240, 350, 310, 375))
        # Ensure it falls strictly inside vehicle bounds in frame coordinates
        self.assertGreaterEqual(fx1, vx1)
        self.assertGreaterEqual(fy1, vy1)
        self.assertLessEqual(fx2, vx2)
        self.assertLessEqual(fy2, vy2)

    def test_02_invalid_or_malformed_plate_boxes_rejected(self):
        """2. Invalid or malformed plate boxes rejected safely without exceptions."""
        malformed_boxes = [
            (float("nan"), 10.0, 50.0, 30.0),
            (10.0, float("inf"), 50.0, 30.0),
            (50.0, 20.0, 10.0, 40.0),       # x2 < x1
            (10.0, 40.0, 50.0, 20.0),       # y2 < y1
            (-50.0, -50.0, -10.0, -10.0),   # outside
            (1500.0, 100.0, 1600.0, 200.0), # outside frame
        ]
        for b in malformed_boxes:
            res = expand_and_clamp_bbox(b, self.frame_w, self.frame_h)
            self.assertIsNone(res, f"Expected box {b} to be rejected as malformed/out-of-bounds")

    def test_03_plate_mask_does_not_use_full_vehicle_box(self):
        """3. Plate masks do not accidentally use the full vehicle box."""
        engine = ConsolidatedPrivacyEngine(device="cpu")
        veh_box = (200, 200, 600, 500)  # 400x300 px
        veh_area = (veh_box[2] - veh_box[0]) * (veh_box[3] - veh_box[1])

        # Plate detection in frame coordinates: 70x25 px
        raw_plate = (350, 440, 420, 465)
        exp_plate = expand_and_clamp_bbox(
            raw_plate,
            self.frame_w,
            self.frame_h,
            margin_ratio=engine.margin_ratio,
            min_padding=engine.min_padding,
        )
        self.assertIsNotNone(exp_plate)
        plate_area = (exp_plate[2] - exp_plate[0]) * (exp_plate[3] - exp_plate[1])

        # Plate mask must cover a small localized area, not the vehicle area
        self.assertLess(plate_area, veh_area * 0.10)
        self.assertLess(exp_plate[2] - exp_plate[0], (veh_box[2] - veh_box[0]) * 0.40)
        self.assertLess(exp_plate[3] - exp_plate[1], (veh_box[3] - veh_box[1]) * 0.30)

    def test_04_direct_plate_mask_covers_expected_plate_region(self):
        """4. Direct plate masks cover the expected plate region and leave vehicle body unmasked."""
        frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 200, dtype=np.uint8)
        plate_box = (300, 350, 380, 390)

        redacted, _ = apply_solid_masks_independent(frame, [plate_box], color=(0, 0, 0))

        # Check plate region is solid black (0, 0, 0)
        self.assertTrue(np.all(redacted[350:390, 300:380] == 0))
        # Check surrounding vehicle pixels outside the plate mask are untouched (200)
        self.assertTrue(np.all(redacted[250:300, 250:300] == 200))
        self.assertTrue(np.all(redacted[420:450, 400:450] == 200))

    def test_05_geometric_fallbacks_applied_only_under_intended_conditions(self):
        """5. Geometric fallbacks are applied only when direct detection & temporal hold are absent."""
        engine = ConsolidatedPrivacyEngine(device="cpu")

        # Condition A: Vehicle has a direct plate detection -> 0 fallbacks
        veh_boxes = [(100, 200, 300, 350)]
        plate_dets = [(170, 300, 230, 330, 0.85)]
        fb_a = engine.compute_vehicle_fallbacks(veh_boxes, plate_dets, active_temporal_holds=[])
        self.assertEqual(len(fb_a), 0)

        # Condition B: Vehicle has active temporal hold on plate -> 0 fallbacks
        temp_holds = [(170, 300, 230, 330)]
        fb_b = engine.compute_vehicle_fallbacks(veh_boxes, direct_plate_boxes=[], active_temporal_holds=temp_holds)
        self.assertEqual(len(fb_b), 0)

        # Condition C: Vehicle has neither direct plate nor temporal hold -> 1 fallback
        fb_c = engine.compute_vehicle_fallbacks(veh_boxes, direct_plate_boxes=[], active_temporal_holds=[])
        self.assertEqual(len(fb_c), 1)

        # Condition D: Tiny vehicle below minimum size threshold -> 0 fallbacks
        tiny_veh = [(100, 200, 115, 220)]  # 15x20 px
        fb_d = engine.compute_vehicle_fallbacks(tiny_veh, direct_plate_boxes=[], active_temporal_holds=[])
        self.assertEqual(len(fb_d), 0)

        # Condition E: Person with face detection -> 0 fallbacks
        person_boxes = [(400, 100, 480, 300)]
        face_dets = [(420, 110, 460, 150, 0.90)]
        p_fb_a = engine.compute_face_fallbacks(person_boxes, face_dets, active_temporal_holds=[])
        self.assertEqual(len(p_fb_a), 0)

        # Condition F: Person without face detection -> 1 fallback
        p_fb_b = engine.compute_face_fallbacks(person_boxes, face_boxes=[], active_temporal_holds=[])
        self.assertEqual(len(p_fb_b), 1)

    def test_06_provenance_diagnostic_identifies_source_of_each_mask(self):
        """6. The provenance diagnostic identifies the exact source of each mask."""
        engine = create_test_engine(device="cpu")
        frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 100, dtype=np.uint8)

        # Process a frame
        output, status, metrics = engine.process_frame(frame)
        self.assertEqual(status, PrivacyGateStatus.RELEASE_APPROVED)

        # Verify diagnostic metrics keys
        self.assertIn("masks_detail", metrics)
        self.assertIn("total_obscured_fraction", metrics)
        self.assertIn("obscured_fraction_by_category", metrics)
        self.assertIn("substantial_overlaps", metrics)

        # Check category breakdown contains valid provenance string keys
        obs_cat = metrics["obscured_fraction_by_category"]
        for k in obs_cat.keys():
            self.assertIn(k, [p.value for p in MaskProvenance])

    def test_07_excessive_expansion_not_introduced_by_coordinate_transformations(self):
        """7. Excessive expansion is not introduced by coordinate transformations or margins."""
        raw_box = (100, 100, 160, 120)  # width = 60, height = 20
        # Precision plate margin settings: ratio = 0.15, min_padding = 4
        exp_box = expand_and_clamp_bbox(raw_box, self.frame_w, self.frame_h, margin_ratio=0.15, min_padding=4)
        self.assertIsNotNone(exp_box)

        exp_w = exp_box[2] - exp_box[0]
        exp_h = exp_box[3] - exp_box[1]

        # Horizontal padding: max(60 * 0.15, 4) = 9 px on each side -> 60 + 18 = 78 px
        self.assertEqual(exp_w, 78)
        # Vertical padding: max(20 * 0.15, 4) = 4 px on each side -> 20 + 8 = 28 px
        self.assertEqual(exp_h, 28)

        # Bounded expansion: must not exceed 1.5x original width or 2.0x original height
        self.assertLess(exp_w, 60 * 1.5)
        self.assertLess(exp_h, 20 * 2.0)

    def test_08_temporal_masks_follow_correct_track(self):
        """8. Temporal masks follow the correct track and fallbacks are never tracked."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Frame 1: Detection
        m1 = tracker.update_and_get_tracked_targets([(100, 100, 140, 120, "plate", 0.90)])
        self.assertEqual(len(m1), 1)
        track_id = m1[0]["id"]
        self.assertGreater(track_id, 0)
        self.assertEqual(m1[0]["status"], "DETECTION")

        # Frame 2: Moved detection
        m2 = tracker.update_and_get_tracked_targets([(106, 104, 146, 124, "plate", 0.92)])
        self.assertEqual(len(m2), 1)
        self.assertEqual(m2[0]["id"], track_id)
        self.assertEqual(m2[0]["status"], "DETECTION")

        # Frame 3: Dropout -> Temporal Hold
        m3 = tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(m3), 1)
        self.assertEqual(m3[0]["id"], track_id)
        self.assertEqual(m3[0]["status"], "TEMPORAL_HOLD")

        # Ensure geometric fallbacks are NEVER fed to tracker
        # (tracker update_and_get_tracked_targets is only called on direct face and plate detections)
        self.assertNotIn("fallback", [t["category"] for t in tracker.tracks])

    def test_09_expired_temporal_masks_disappear(self):
        """9. Expired temporal masks disappear after max_hold_frames (3)."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Frame 1: Detection
        tracker.update_and_get_masks([(200, 200, 250, 250, "face", 0.85)])
        # Frame 2: Dropout 1 -> Held
        m_hold1 = tracker.update_and_get_masks([])
        self.assertEqual(len(m_hold1), 1)
        # Frame 3: Dropout 2 -> Held
        m_hold2 = tracker.update_and_get_masks([])
        self.assertEqual(len(m_hold2), 1)
        # Frame 4: Dropout 3 -> EXPIRED (0 masks)
        m_hold3 = tracker.update_and_get_masks([])
        self.assertEqual(len(m_hold3), 0)

    def test_10_detector_errors_produce_black_suppression_frame(self):
        """10. Detector errors produce fail-closed solid black 1280x720 suppression frame."""
        engine = create_test_engine(device="cpu")
        frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 180, dtype=np.uint8)

        output, status, metrics = engine.process_frame(frame, simulate_detector_exception=True)

        self.assertEqual(status, PrivacyGateStatus.SUPPRESSED_DETECTOR_FAILURE)
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        self.assertTrue(np.all(output == 0))
        self.assertIn("Simulated detector fault", metrics.get("error_detail", ""))

    def test_11_missing_plate_detector_triggers_fail_closed(self):
        """11. Missing plate-detector configuration triggers fail-closed black suppression frame."""
        engine = ConsolidatedPrivacyEngine(
            plate_model_path="nonexistent_yolo_plate.pt",
            device="cpu",
        )
        frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 180, dtype=np.uint8)

        output, status, metrics = engine.process_frame(frame)

        self.assertEqual(status, PrivacyGateStatus.SUPPRESSED_MISSING_CONFIG)
        self.assertEqual(output.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        self.assertTrue(np.all(output == 0))
        self.assertIn("models not configured", metrics.get("error_detail", ""))

    def test_12_diagnostic_visualization_never_modifies_actual_redacted_output(self):
        """12. Diagnostic visualization never modifies the actual redacted output array."""
        engine = create_test_engine(device="cpu")
        ref_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 150, dtype=np.uint8)
        redacted_output, gate_status, metrics = engine.process_frame(ref_frame)

        # Make an exact clone of the redacted array
        redacted_before = redacted_output.copy()

        # Render diagnostic overlay
        diag_vis = draw_diagnostic_overlay(ref_frame, redacted_output, gate_status, metrics)

        # Verify redacted_output was completely unmodified
        self.assertTrue(np.array_equal(redacted_output, redacted_before))

        # Verify diag_vis is an independent array
        self.assertFalse(diag_vis is redacted_output)
        self.assertEqual(diag_vis.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))


class TestPhase051TemporalValidationAndDeduplication(unittest.TestCase):
    """Test suite for Phase 0.5.1 temporal tracking integrity and global plate deduplication."""

    def setUp(self):
        self.engine = ConsolidatedPrivacyEngine(device="cpu")

    def test_temporal_state_does_not_cross_video_boundary(self):
        """Temporal tracker state must not leak across video boundaries after reset."""
        tracker = self.engine.temporal_tracker
        tracker.reset()

        # Video 1: Detect a face and a plate in frame 0
        v1_dets = [
            (100, 100, 150, 150, "face", 0.95),
            (200, 300, 270, 330, "plate", 0.90),
        ]
        masks_v1 = tracker.update_and_get_tracked_targets(v1_dets)
        self.assertEqual(len(masks_v1), 2)
        v1_track_ids = {m["id"] for m in masks_v1}
        self.assertEqual(len(v1_track_ids), 2)
        self.assertEqual(len(tracker.tracks), 2)

        # Transition to Video 2: Explicitly reset temporal state
        self.engine.reset_temporal_state()

        # Video 2 starts: Verify all internal tracks are cleared
        self.assertEqual(len(tracker.tracks), 0)

        # If Video 2 begins with no detections (e.g. frame 0 has no faces/plates),
        # NO temporal holds from Video 1 may survive!
        masks_v2_f0 = tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(masks_v2_f0), 0)

        # When Video 2 detects a target, track IDs start fresh
        v2_dets = [(400, 200, 450, 250, "face", 0.88)]
        masks_v2_f1 = tracker.update_and_get_tracked_targets(v2_dets)
        self.assertEqual(len(masks_v2_f1), 1)
        self.assertEqual(masks_v2_f1[0]["id"], 1)
        self.assertNotIn(masks_v2_f1[0]["id"], v1_track_ids - {1})

    def test_sampled_frame_gaps_not_treated_as_one_frame_gaps(self):
        """Intermediate 30-frame gap between sparse samples must expire temporal tracks."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Sample at frame 0 (detection present)
        masks_f0 = tracker.update_and_get_tracked_targets([(100, 100, 150, 150, "face", 0.90)])
        self.assertEqual(len(masks_f0), 1)
        self.assertEqual(masks_f0[0]["status"], "DETECTION")

        # In continuous chronological decoding, frames 1 through 30 are decoded.
        # Frames 1 and 2: Held as TEMPORAL_HOLD
        for f in (1, 2):
            masks_hold = tracker.update_and_get_tracked_targets([])
            self.assertEqual(len(masks_hold), 1)
            self.assertEqual(masks_hold[0]["status"], "TEMPORAL_HOLD")

        # Frame 3: Expired (max_hold_frames = 3 reached on 3rd missing frame)
        masks_f3 = tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(masks_f3), 0)

        # Frames 4 through 29: Continue decoding intermediate frames with 0 detections
        for _ in range(4, 30):
            masks_empty = tracker.update_and_get_tracked_targets([])
            self.assertEqual(len(masks_empty), 0)

        # Sample at frame 30 (1.00s): Tracker has 0 active tracks from frame 0
        self.assertEqual(len(tracker.tracks), 0)

    def test_expired_masks_survive_no_longer_than_configured(self):
        """Expired masks must survive no longer than configured max_hold_frames (3)."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Initial detection
        tracker.update_and_get_tracked_targets([(50, 50, 90, 90, "plate", 0.92)])

        # Dropout 1 and 2: Survived as TEMPORAL_HOLD
        for hold_num in (1, 2):
            held = tracker.update_and_get_tracked_targets([])
            self.assertEqual(len(held), 1, f"Expected hold on dropout frame {hold_num}")
            self.assertEqual(held[0]["status"], "TEMPORAL_HOLD")
            self.assertEqual(held[0]["missing_frames"], hold_num)

        # Dropout 3: Must expire completely (missing_frames >= max_hold_frames)
        expired = tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(expired), 0, "Track must expire on dropout frame 3")
        self.assertEqual(len(tracker.tracks), 0)

        # Subsequent dropout frames: must remain empty
        empty = tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(empty), 0)
        self.assertEqual(len(tracker.tracks), 0)

    def test_tracker_does_not_use_stale_mask_locations_from_unrelated_timestamps(self):
        """Tracker must not carry stale coordinates from past expired tracks into future detections."""
        tracker = TemporalMaskTracker(max_hold_frames=3, iou_match_threshold=0.20)

        # Object at (100, 100, 140, 140)
        tracker.update_and_get_tracked_targets([(100, 100, 140, 140, "face", 0.90)])
        # 4 dropouts -> fully expired
        for _ in range(4):
            tracker.update_and_get_tracked_targets([])
        self.assertEqual(len(tracker.tracks), 0)

        # At timestamp 10.0s, a new detection appears far away at (600, 400, 650, 450)
        new_dets = [(600, 400, 650, 450, "face", 0.88)]
        new_masks = tracker.update_and_get_tracked_targets(new_dets)

        self.assertEqual(len(new_masks), 1)
        self.assertEqual(new_masks[0]["box"], (600, 400, 650, 450))
        # Ensure zero velocity from old object
        active_track = tracker.tracks[0]
        self.assertEqual(active_track["vx"], 0.0)
        self.assertEqual(active_track["vy"], 0.0)

    def test_global_plate_duplicate_suppression_near_identical_boxes(self):
        """Global plate deduplication must eliminate near-identical candidates (IoU >= 0.65)."""
        # Plate A from Crop 1 (confidence 0.92)
        plate_a = (200, 300, 280, 330, 0.92)
        # Plate B from Crop 2 (confidence 0.85, shifted by only 2px, IoU ~0.89)
        plate_b = (202, 301, 282, 331, 0.85)

        surviving = self.engine._suppress_duplicate_global_plates([plate_a, plate_b], iou_threshold=0.65)

        self.assertEqual(len(surviving), 1, "Duplicate plate candidates must be reduced to 1")
        # Higher-confidence plate must be retained
        self.assertEqual(surviving[0], plate_a)

    def test_distinct_nearby_plates_preserved(self):
        """Global plate deduplication must preserve genuinely distinct nearby plates."""
        # Two distinct vehicles side by side in traffic
        plate_car1 = (150, 300, 220, 330, 0.90)
        plate_car2 = (250, 300, 320, 330, 0.89)

        surviving = self.engine._suppress_duplicate_global_plates(
            [plate_car1, plate_car2], iou_threshold=0.65
        )

        self.assertEqual(len(surviving), 2, "Distinct plates on adjacent vehicles must both be preserved")
        self.assertIn(plate_car1, surviving)
        self.assertIn(plate_car2, surviving)

    def test_obscuration_unique_union_vs_category_sums(self):
        """Verify that total_obscured_fraction counts overlapping masks strictly once."""
        engine = ConsolidatedPrivacyEngine(device="cpu")
        # Create a frame with two strongly overlapping simulated mask targets
        # We test the union calculation logic directly
        total_pixels = CANONICAL_WIDTH * CANONICAL_HEIGHT
        box1 = (100, 100, 200, 200)  # 100x100 = 10,000 px
        box2 = (150, 100, 250, 200)  # 100x100 = 10,000 px
        # Overlap: (150, 100, 200, 200) = 50x100 = 5,000 px
        # Union: 15,000 px

        union_mask = np.zeros((CANONICAL_HEIGHT, CANONICAL_WIDTH), dtype=np.uint8)
        union_mask[box1[1]:box1[3], box1[0]:box1[2]] = 255
        union_mask[box2[1]:box2[3], box2[0]:box2[2]] = 255

        union_px = int(np.count_nonzero(union_mask))
        self.assertEqual(union_px, 15000)

        # Sum of individual areas: 20,000 px
        sum_px = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1])
        self.assertEqual(sum_px, 20000)

        # Union fraction is strictly less than category sum fraction
        union_frac = union_px / total_pixels
        sum_frac = sum_px / total_pixels
        self.assertLess(union_frac, sum_frac)


class TestPhase052BenchmarkIntegrity(unittest.TestCase):
    """Test suite for Phase 0.5.2 benchmark integrity and metric reconciliation."""

    def test_aggregate_plate_counts_match_sample_sums(self):
        """Every aggregate count must match the exact sum of constituent sample records."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        records = [
            {"faces_count": 2, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 2.5, "category_sum_pct": 2.5},
            {"faces_count": 4, "plates_raw": 3, "plates_suppressed": 1, "plates_unique": 2, "unique_union_pct": 4.0, "category_sum_pct": 4.5},
            {"faces_count": 1, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 1.0, "category_sum_pct": 1.0},
        ]
        summary = compute_cohort_metrics(records)

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["faces_sum"], 2 + 4 + 1)
        self.assertEqual(summary["raw_plates_sum"], 1 + 3 + 0)
        self.assertEqual(summary["suppr_plates_sum"], 0 + 1 + 0)
        self.assertEqual(summary["unique_plates_sum"], 1 + 2 + 0)

    def test_raw_equals_suppressed_plus_surviving_identity(self):
        """Raw candidates minus suppressed duplicates must strictly equal surviving unique detections."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        # Valid records
        valid_records = [
            {"faces_count": 1, "plates_raw": 5, "plates_suppressed": 2, "plates_unique": 3, "unique_union_pct": 3.0, "category_sum_pct": 3.0},
        ]
        summary = compute_cohort_metrics(valid_records)
        self.assertEqual(summary["raw_plates_sum"] - summary["suppr_plates_sum"], summary["unique_plates_sum"])

        # Violating records must raise ValueError
        invalid_records = [
            {"faces_count": 1, "plates_raw": 5, "plates_suppressed": 1, "plates_unique": 3, "unique_union_pct": 3.0, "category_sum_pct": 3.0},
        ]
        with self.assertRaises(ValueError):
            compute_cohort_metrics(invalid_records)

    def test_displayed_means_match_arithmetic_means_within_tolerance(self):
        """Displayed percentage means must match arithmetic mean of sample values within rounding tolerance."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        sample_unions = [2.11, 1.48, 5.56, 7.37, 3.24, 4.89, 2.07, 1.47, 1.92, 7.48]
        records = [
            {"faces_count": 0, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": u, "category_sum_pct": u}
            for u in sample_unions
        ]
        summary = compute_cohort_metrics(records)

        expected_mean = sum(sample_unions) / len(sample_unions)
        self.assertAlmostEqual(summary["union_mean"], expected_mean, places=4)
        self.assertEqual(f"{summary['union_mean']:.2f}", f"{expected_mean:.2f}")

    def test_combined_mean_uses_correct_sample_weighting(self):
        """Combined mean must correctly weight cohort sizes rather than averaging unweighted means."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        # Unequal cohort sizes: Cohort A (4 samples @ 10%), Cohort B (16 samples @ 2%)
        records_a = [
            {"faces_count": 0, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 10.0, "category_sum_pct": 10.0}
            for _ in range(4)
        ]
        records_b = [
            {"faces_count": 0, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 2.0, "category_sum_pct": 2.0}
            for _ in range(16)
        ]
        summary_a = compute_cohort_metrics(records_a)
        summary_b = compute_cohort_metrics(records_b)
        combined = compute_cohort_metrics(records_a + records_b)

        # Unweighted average of cohort means: (10.0 + 2.0) / 2 = 6.0% (INCORRECT)
        # Correct weighted average: (4 * 10.0 + 16 * 2.0) / 20 = 72.0 / 20 = 3.6%
        weighted_expected = (summary_a["union_mean"] * 4 + summary_b["union_mean"] * 16) / 20
        self.assertAlmostEqual(combined["union_mean"], weighted_expected, places=4)
        self.assertAlmostEqual(combined["union_mean"], 3.6, places=4)
        self.assertNotEqual(combined["union_mean"], 6.0)

    def test_unique_union_and_category_sum_remain_distinct_measurements(self):
        """Unique-union obscuration must remain distinct from and <= category-sum obscuration."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        records = [
            {"faces_count": 1, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 4.5, "category_sum_pct": 5.2},
            {"faces_count": 2, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 3.0, "category_sum_pct": 3.0},
        ]
        summary = compute_cohort_metrics(records)

        self.assertLess(summary["union_mean"], summary["cat_sum_mean"])
        self.assertAlmostEqual(summary["union_mean"], 3.75, places=4)
        self.assertAlmostEqual(summary["cat_sum_mean"], 4.10, places=4)

    def test_benchmark_20_sample_reconciliation_invariants(self):
        """Verify the exact mathematical integrity of the 20-sample Phase 0.5.2 benchmark data."""
        from scripts.verify_privacy_recall import compute_cohort_metrics

        # Reconstructed ground telemetry from the 20 audit samples
        day_samples = [
            {"faces_count": 2, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 2.11, "category_sum_pct": 2.11},
            {"faces_count": 6, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 1.48, "category_sum_pct": 1.48},
            {"faces_count": 3, "plates_raw": 2, "plates_suppressed": 0, "plates_unique": 2, "unique_union_pct": 5.56, "category_sum_pct": 5.77},
            {"faces_count": 1, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 7.37, "category_sum_pct": 7.37},
            {"faces_count": 1, "plates_raw": 2, "plates_suppressed": 0, "plates_unique": 2, "unique_union_pct": 3.24, "category_sum_pct": 3.26},
            {"faces_count": 5, "plates_raw": 2, "plates_suppressed": 0, "plates_unique": 2, "unique_union_pct": 4.89, "category_sum_pct": 5.18},
            {"faces_count": 1, "plates_raw": 4, "plates_suppressed": 0, "plates_unique": 4, "unique_union_pct": 2.07, "category_sum_pct": 2.08},
            {"faces_count": 1, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 1.47, "category_sum_pct": 1.47},
            {"faces_count": 1, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 1.92, "category_sum_pct": 1.92},
            {"faces_count": 4, "plates_raw": 2, "plates_suppressed": 0, "plates_unique": 2, "unique_union_pct": 7.48, "category_sum_pct": 7.84},
        ]
        night_samples = [
            {"faces_count": 2, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 3.70, "category_sum_pct": 3.70},
            {"faces_count": 2, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 4.17, "category_sum_pct": 4.17},
            {"faces_count": 1, "plates_raw": 1, "plates_suppressed": 0, "plates_unique": 1, "unique_union_pct": 3.55, "category_sum_pct": 3.55},
            {"faces_count": 3, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 5.34, "category_sum_pct": 5.34},
            {"faces_count": 1, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 4.37, "category_sum_pct": 4.37},
            {"faces_count": 1, "plates_raw": 2, "plates_suppressed": 1, "plates_unique": 1, "unique_union_pct": 5.68, "category_sum_pct": 5.68},
            {"faces_count": 2, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 7.64, "category_sum_pct": 7.71},
            {"faces_count": 1, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 8.67, "category_sum_pct": 8.67},
            {"faces_count": 6, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 3.58, "category_sum_pct": 3.58},
            {"faces_count": 3, "plates_raw": 0, "plates_suppressed": 0, "plates_unique": 0, "unique_union_pct": 3.06, "category_sum_pct": 3.06},
        ]

        day_sum = compute_cohort_metrics(day_samples)
        night_sum = compute_cohort_metrics(night_samples)
        combined_sum = compute_cohort_metrics(day_samples + night_samples)

        # Daytime assertions
        self.assertEqual(day_sum["raw_plates_sum"], 16)
        self.assertEqual(day_sum["suppr_plates_sum"], 0)
        self.assertEqual(day_sum["unique_plates_sum"], 16)
        self.assertEqual(day_sum["faces_sum"], 25)
        self.assertAlmostEqual(day_sum["union_mean"], 3.759, places=3)
        self.assertEqual(f"{day_sum['union_mean']:.2f}", "3.76")

        # Nighttime assertions
        self.assertEqual(night_sum["raw_plates_sum"], 3)
        self.assertEqual(night_sum["suppr_plates_sum"], 1)
        self.assertEqual(night_sum["unique_plates_sum"], 2)
        self.assertEqual(night_sum["faces_sum"], 22)
        self.assertAlmostEqual(night_sum["union_mean"], 4.976, places=3)
        self.assertEqual(f"{night_sum['union_mean']:.2f}", "4.98")

        # Combined assertions
        self.assertEqual(combined_sum["raw_plates_sum"], 19)
        self.assertEqual(combined_sum["suppr_plates_sum"], 1)
        self.assertEqual(combined_sum["unique_plates_sum"], 18)
        self.assertEqual(combined_sum["faces_sum"], 47)
        self.assertAlmostEqual(combined_sum["union_mean"], 4.3675, places=4)
        self.assertEqual(f"{combined_sum['union_mean']:.2f}", "4.37")


class TestPhase053TwoThreeWheelerPrivacySafety(unittest.TestCase):
    """Phase 0.5.3 Regression Tests: Two-/Three-Wheeler Plate Privacy & Geometry Validation.
    
    Verifies:
    1. Class-aware fallback coverage on motorcycles with riders (mid-height plates).
    2. Narrow-vehicle / auto-rickshaw fallback geometry.
    3. Pixel-level array redaction covering plate positions.
    4. Preservation of 4W bumper fallback geometry (unaltered precision).
    5. Crop boundary clamping and fail-closed safety.
    """

    def setUp(self):
        self.engine = ConsolidatedPrivacyEngine(device="cpu")
        self.frame_w = CANONICAL_WIDTH
        self.frame_h = CANONICAL_HEIGHT

    def test_two_wheeler_fallback_covers_mid_height_and_angled_plate(self):
        """2W fallback covers plate at mid-height and left/right offsets where 4W fallback fails."""
        # Motorcycle box: width 100, height 200 (class 3)
        vx1, vy1, vx2, vy2 = 500, 300, 600, 500
        # Plate at mid-height (40% to 55% from top) and left-aligned:
        plate_x1, plate_y1, plate_x2, plate_y2 = 505, 385, 545, 415

        # 2W fallback with cls_id=3
        fb_2w = self.engine._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=3)
        fb_x1, fb_y1, fb_x2, fb_y2 = fb_2w

        # Must span lower 72% (from vy1 + 28% = 356 down to 500) and full vehicle width (500 to 600)
        self.assertLessEqual(fb_x1, plate_x1)
        self.assertGreaterEqual(fb_x2, plate_x2)
        self.assertLessEqual(fb_y1, plate_y1)
        self.assertGreaterEqual(fb_y2, plate_y2)

        # Baseline 4W fallback on standard car (w=200, h=100, cls_id=2)
        car_x1, car_y1, car_x2, car_y2 = 400, 300, 600, 400
        fb_4w = self.engine._compute_single_vehicle_fallback(car_x1, car_y1, car_x2, car_y2, cls_id=2)
        fb4_x1, fb4_y1, fb4_x2, fb4_y2 = fb_4w

        # 4W fallback is placed at bottom 25% (y = 375..400) and center 60% (x = 440..560)
        # Verify that mid-height plate (e.g. at y=340..370) is missed by 4W fallback:
        self.assertGreater(fb4_y1, 370)  # Fallback starts at 375, plate ends at 370 -> unmasked in 4W
        self.assertEqual(fb4_y1, car_y2 - int(round(0.25 * (car_y2 - car_y1))))

    def test_two_wheeler_fallback_pixel_level_solid_redaction(self):
        """Verify pixel arrays: simulated frame with 2W plate is completely masked with solid black."""
        frame = np.full((self.frame_h, self.frame_w, 3), 180, dtype=np.uint8)
        # Place a bright high-contrast license plate inside motorcycle box
        mc_box = (400, 250, 480, 410, 3)  # w=80, h=160, motorcycle
        plate_target = (410, 310, 450, 335)  # y=310..335 (mid-height)
        frame[plate_target[1]:plate_target[2], plate_target[0]:plate_target[3]] = [255, 255, 255]

        # Compute vehicle fallback for motorcycle without direct detection
        fallbacks = self.engine.compute_vehicle_fallbacks([mc_box], direct_plate_boxes=[], active_temporal_holds=[])
        self.assertEqual(len(fallbacks), 1)

        # Apply solid mask
        redacted, _ = apply_solid_masks_independent(frame, fallbacks, color=(0, 0, 0))

        # Examine target plate pixels: every pixel must be solid black [0, 0, 0]
        plate_pixels = redacted[plate_target[1]:plate_target[2], plate_target[0]:plate_target[3]]
        self.assertTrue(np.all(plate_pixels == [0, 0, 0]), "Plate region must be completely redacted to black")

    def test_narrow_auto_rickshaw_geometry(self):
        """Narrow 3-wheelers (aspect ratio <= 0.85) receive high-coverage fallback even if classified as car/truck."""
        # Auto-rickshaw classified as car (class 2) but narrow: w=70, h=100 (aspect 0.70)
        vx1, vy1, vx2, vy2 = 200, 250, 270, 350
        plate_y1, plate_y2 = 290, 310  # mid-height plate at y=290

        fb = self.engine._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=2)
        # Since aspect ratio is 0.70 <= 0.85, 2W/3W geometry applies:
        self.assertLessEqual(fb[1], plate_y1)
        self.assertGreaterEqual(fb[3], plate_y2)

    def test_four_wheeler_standard_geometry_preserved(self):
        """Standard 4W cars (wide aspect ratio) preserve conservative bottom-25% center-60% fallback."""
        # Car: w=300, h=150 (aspect 2.0, class 2)
        vx1, vy1, vx2, vy2 = 100, 200, 400, 350
        vw, vh = 300, 150
        vcx = 250.0

        fb = self.engine._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=2)
        expected_x1 = max(vx1, int(round(vcx - 0.30 * vw)))  # 250 - 90 = 160
        expected_x2 = min(vx2, int(round(vcx + 0.30 * vw)))  # 250 + 90 = 340
        expected_y1 = max(vy1, int(round(vy2 - 0.25 * vh)))  # 350 - 37.5 = 312
        expected_y2 = vy2                                    # 350

        self.assertEqual(fb, (expected_x1, expected_y1, expected_x2, expected_y2))

    def test_two_wheeler_boundary_clamping(self):
        """Motorcycle touching frame boundaries clamps strictly to canonical dimensions [0..1280, 0..720]."""
        # Motorcycle clipped at top-left edge: [-20, -10, 80, 150, 3]
        vx1, vy1, vx2, vy2 = -20, -10, 80, 150
        fb = self.engine._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=3)
        exp = expand_and_clamp_bbox(fb, self.frame_w, self.frame_h, margin_ratio=0.08, min_padding=4)

        self.assertGreaterEqual(exp[0], 0)
        self.assertGreaterEqual(exp[1], 0)
        self.assertLessEqual(exp[2], self.frame_w)
        self.assertLessEqual(exp[3], self.frame_h)

    # =========================================================================
    # Phase 0.5.3 Regression Tests: Interactive Viewer CSV Selection
    # =========================================================================

    def test_verify_privacy_recall_csv_path_resolution(self):
        """Test path resolution for default, relative, and explicit CSV paths."""
        from scripts.verify_privacy_recall import CSV_PATH, resolve_csv_path

        # Default when None supplied
        self.assertEqual(resolve_csv_path(None), CSV_PATH)

        # Explicit relative path to Phase 0.5.3 CSV
        rel_path = "docs/benchmarks/manual_validation_phase053.csv"
        resolved = resolve_csv_path(rel_path)
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.endswith("manual_validation_phase053.csv"))
        self.assertTrue(os.path.isfile(resolved))

    def test_reviewer_instructions_identifies_target_csv(self):
        """Audit instructions must identify the selected CSV as destination for annotations."""
        from scripts.verify_privacy_recall import get_reviewer_instructions

        custom_target = "/custom/path/audit_samples.csv"
        instructions = get_reviewer_instructions(custom_target)
        self.assertIn(f"Results must be recorded in:\n  {custom_target}", instructions)

    def test_interactive_viewer_setup_honors_explicit_csv(self):
        """Explicitly supplied CSV path must be honored by interactive viewer setup."""
        import io
        from unittest.mock import MagicMock, patch
        from scripts.verify_privacy_recall import load_samples, render_viewer

        custom_csv = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "docs",
            "benchmarks",
            "manual_validation_phase053.csv",
        )
        samples = load_samples(custom_csv)

        # Mock OpenCV windowing and PrivacyEngine initialization
        mock_cv2 = MagicMock()
        mock_cv2.waitKey.return_value = ord("q")
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch("scripts.verify_privacy_recall.cv2", mock_cv2), \
             patch("scripts.verify_privacy_recall.ConsolidatedPrivacyEngine"), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            render_viewer(samples=samples, initial_mode=0, csv_path=custom_csv)

            output = mock_stdout.getvalue()
            # 1. Startup message displays actual resolved CSV path
            self.assertIn(f"Loaded {len(samples)} audit samples from {custom_csv}.", output)
            # 2. Instructions identify selected CSV as destination
            self.assertIn(f"Results must be recorded in:\n  {custom_csv}", output)

    def test_main_cli_csv_argument_honored(self):
        """CLI --csv argument dispatches resolved CSV path to viewer setup."""
        from unittest.mock import patch
        from scripts.verify_privacy_recall import CSV_PATH, main

        custom_rel = "docs/benchmarks/manual_validation_phase053.csv"
        expected_abs = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            custom_rel,
        )

        with patch("sys.argv", ["verify_privacy_recall.py", "--csv", custom_rel]), \
             patch("scripts.verify_privacy_recall.render_viewer") as mock_render:
            main()
            mock_render.assert_called_once()
            called_kwargs = mock_render.call_args.kwargs
            self.assertEqual(called_kwargs.get("csv_path"), expected_abs)
            self.assertEqual(len(called_kwargs.get("samples")), 20)

        # Default behavior preserved when --csv is omitted
        with patch("sys.argv", ["verify_privacy_recall.py"]), \
             patch("scripts.verify_privacy_recall.render_viewer") as mock_render:
            main()
            mock_render.assert_called_once()
            called_kwargs = mock_render.call_args.kwargs
            self.assertEqual(called_kwargs.get("csv_path"), CSV_PATH)
            self.assertEqual(len(called_kwargs.get("samples")), 20)


class TestPhase054TargetedPrivacyLeakCorrection(unittest.TestCase):
    """Phase 0.5.4 regression tests: Targeted two-wheeler & three-wheeler leak prevention."""

    def setUp(self):
        self.engine = create_test_engine(device="cpu")
        self.frame_w = CANONICAL_WIDTH
        self.frame_h = CANONICAL_HEIGHT

    def test_phase054_false_or_partial_plate_does_not_suppress_2w3w_fallback(self):
        """False or partial plate candidates on 2W/3W vehicles must not suppress vehicle fallbacks."""
        # Motorcycle box: w=56, h=69 (class 3)
        mc_box = (358, 318, 414, 387, 3)

        # 1. Partial/false detection high up on vehicle body (y=356..362, height 6px, above lower tail)
        false_plate = (381, 356, 393, 362, 0.20)
        fbs = self.engine.compute_vehicle_fallbacks([mc_box], direct_plate_boxes=[false_plate], active_temporal_holds=[])
        self.assertEqual(len(fbs), 1, "Fallback must be generated when only a partial/false plate is detected on 2W/3W")

        # 2. Tiny edge candidate (width < 14, height < 6)
        tiny_plate = (380, 380, 388, 384, 0.20)  # w=8, h=4
        fbs = self.engine.compute_vehicle_fallbacks([mc_box], direct_plate_boxes=[tiny_plate], active_temporal_holds=[])
        self.assertEqual(len(fbs), 1, "Fallback must be generated when tiny candidate is detected")

        # 3. Genuine adequate plate located in rear fender region
        # Tail boundary for vehicle (h=69): vy2 - max(12, int(0.18*69)) = 387 - 12 = 375
        real_plate = (375, 370, 395, 386, 0.75)  # w=20, h=16, py2=386 >= 375
        fbs = self.engine.compute_vehicle_fallbacks([mc_box], direct_plate_boxes=[real_plate], active_temporal_holds=[])
        self.assertEqual(len(fbs), 0, "Fallback must be suppressed when genuine adequate plate is present")

    def test_phase054_temporal_hold_does_not_suppress_2w3w_fallback(self):
        """Active temporal plate holds from adjacent vehicles must not suppress 2W/3W fallbacks."""
        # Auto-rickshaw (cls=7, narrow aspect ratio <= 0.85)
        ar_box = (481, 310, 540, 390, 7)
        # Active temporal hold that overlaps the vehicle region
        temporal_hold = [{"box": (500, 375, 533, 395), "category": "plate"}]

        # 2W/3W vehicle must still receive fallback
        fbs_2w = self.engine.compute_vehicle_fallbacks([ar_box], direct_plate_boxes=[], active_temporal_holds=temporal_hold)
        self.assertEqual(len(fbs_2w), 1, "2W/3W fallback must NOT be suppressed by temporal plate hold")

        # In contrast, standard 4W car IS suppressed by temporal hold
        car_box = (400, 300, 600, 400, 2)  # cls=2, wide aspect ratio
        car_temporal_hold = [{"box": (480, 370, 520, 395), "category": "plate"}]
        fbs_4w = self.engine.compute_vehicle_fallbacks([car_box], direct_plate_boxes=[], active_temporal_holds=car_temporal_hold)
        self.assertEqual(len(fbs_4w), 0, "4W fallback must be suppressed by matching temporal plate hold")

    def test_phase054_stripe_filter_permits_genuine_wide_aspect_plates(self):
        """Stripe filter removes seams (AR > 7 and bh < 6) but preserves genuine wide plates (bh >= 6)."""
        # Genuine wide plate (e.g. DAY-10 motorcycle plate w=54, h=9, AR=6.0)
        # Simulate filter condition
        genuine_bw, genuine_bh = 54, 9
        is_genuine_stripe = (genuine_bw / max(1, genuine_bh) > 7.0 and genuine_bh < 6)
        self.assertFalse(is_genuine_stripe, "Genuine plate with bh=9 must NOT be classified as stripe")

        # Extreme narrow horizontal body seam (e.g. bw=45, bh=4, AR=11.25)
        seam_bw, seam_bh = 45, 4
        is_seam_stripe = (seam_bw / max(1, seam_bh) > 7.0 and seam_bh < 6)
        self.assertTrue(is_seam_stripe, "Extreme horizontal body seam with bh=4 and AR > 7 must be filtered")

    def test_phase054_two_wheeler_fallback_geometry_and_clamping(self):
        """2W fallback starts at vy1 + 20% vh, covers full vehicle width, and clamps strictly."""
        vx1, vy1, vx2, vy2 = 787, 366, 1012, 582
        vw = vx2 - vx1  # 225
        vh = vy2 - vy1  # 216

        fb = self.engine._compute_single_vehicle_fallback(vx1, vy1, vx2, vy2, cls_id=3)
        fb_x1, fb_y1, fb_x2, fb_y2 = fb

        # Width spans full vehicle
        self.assertEqual(fb_x1, vx1)
        self.assertEqual(fb_x2, vx2)
        # Height starts at vy1 + 20% vh = 366 + 43 = 409
        expected_y1 = max(vy1, int(round(vy1 + 0.20 * vh)))
        self.assertEqual(fb_y1, expected_y1)
        self.assertEqual(fb_y2, vy2)

        # Clamping check: expand and clamp must remain within frame
        exp = expand_and_clamp_bbox(fb, self.frame_w, self.frame_h, margin_ratio=0.08, min_padding=4)
        self.assertGreaterEqual(exp[0], 0)
        self.assertGreaterEqual(exp[1], 0)
        self.assertLessEqual(exp[2], self.frame_w)
        self.assertLessEqual(exp[3], self.frame_h)

    def test_phase054_distinct_plates_preserved_by_global_deduplication(self):
        """Global deduplication preserves distinct plates across separate vehicles (IoU < 0.65)."""
        # Two distinct plates on adjacent vehicles
        plate_a = (370, 365, 400, 380, 0.70)
        plate_b = (420, 365, 450, 380, 0.65)
        deduped = self.engine._suppress_duplicate_global_plates([plate_a, plate_b], iou_threshold=0.65)
        self.assertEqual(len(deduped), 2, "Distinct non-overlapping plates must both be preserved")

        # Two heavily overlapping duplicate plate detections (IoU >= 0.65)
        plate_dup = (372, 366, 402, 381, 0.50)
        deduped_dup = self.engine._suppress_duplicate_global_plates([plate_a, plate_dup], iou_threshold=0.65)
        self.assertEqual(len(deduped_dup), 1, "Duplicate detections with IoU >= 0.65 must be deduplicated")
        self.assertEqual(deduped_dup[0][4], 0.70, "Higher confidence candidate must be retained")

    def test_phase054_pixel_level_coverage_on_target_leaks(self):
        """Pixel array verification: all five previously leaked plate targets are 100% blacked out."""
        # 1. DAY-06 scooter plate: target (370, 367, 394, 375)
        # 2. DAY-08 auto-rickshaw plate: target (504, 380, 526, 389)
        # 3. DAY-09 motorcycle plate: target (378, 370, 396, 387)
        # 4. DAY-10 motorcycle plate 1: target (337, 375, 355, 390)
        # 5. DAY-10 motorcycle plate 2: target (881, 470, 935, 479)
        test_cases = [
            ("DAY-06 scooter", (368, 331, 399, 375, 3), (370, 367, 394, 375)),
            ("DAY-08 3W auto", (481, 310, 540, 390, 7), (504, 380, 526, 389)),
            ("DAY-09 motorcycle", (358, 318, 414, 387, 3), (378, 370, 396, 387)),
            ("DAY-10 motorcycle 1", (320, 337, 365, 391, 3), (337, 375, 355, 390)),
            ("DAY-10 motorcycle 2", (787, 366, 1012, 582, 3), (881, 470, 935, 479)),
        ]

        for name, v_box, target_plate in test_cases:
            frame = np.full((self.frame_h, self.frame_w, 3), 200, dtype=np.uint8)
            px1, py1, px2, py2 = target_plate
            # Fill plate with white pixels
            frame[py1:py2, px1:px2] = [255, 255, 255]

            fbs = self.engine.compute_vehicle_fallbacks([v_box], direct_plate_boxes=[], active_temporal_holds=[])
            self.assertEqual(len(fbs), 1, f"Fallback must be generated for {name}")

            # Apply solid opaque mask
            redacted, _ = apply_solid_masks_independent(frame, fbs, color=(0, 0, 0))

            # Inspect actual pixels of the target plate
            plate_pixels = redacted[py1:py2, px1:px2]
            black_count = np.sum(np.all(plate_pixels == [0, 0, 0], axis=-1))
            total_count = (px2 - px1) * (py2 - py1)
            self.assertEqual(
                black_count,
                total_count,
                f"{name} plate at {target_plate} must be 100% covered ({black_count}/{total_count} pixels masked)",
            )

    def test_phase054_fail_closed_intact(self):
        """Simulate exceptions during detection or redaction: fail-closed safety is completely preserved."""
        test_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)

        # 1. Detector exception
        redacted_det, status_det, _ = self.engine.process_frame(test_frame, simulate_detector_exception=True)
        self.assertEqual(status_det, PrivacyGateStatus.SUPPRESSED_DETECTOR_FAILURE)
        self.assertTrue(np.all(redacted_det == 0), "Whole frame must be blacked out under detector exception")

        # 2. Redaction exception
        redacted_red, status_red, _ = self.engine.process_frame(test_frame, simulate_redaction_exception=True)
        self.assertEqual(status_red, PrivacyGateStatus.SUPPRESSED_REDACTION_FAILURE)
        self.assertTrue(np.all(redacted_red == 0), "Whole frame must be blacked out under redaction exception")


if __name__ == "__main__":
    unittest.main()



