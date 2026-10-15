#!/usr/bin/env python3
"""Unit Tests for Drishti-Edge Phase 0.6 Spatial Zone Mapping & ZVI Engine

Tests:
  - ZVI arithmetic ($100 * (0.50*L + 0.35*I + 0.15*M)$)
  - Boundary conditions and input normalization
  - Missing component handling (transparent INCOMPLETE status, no silent zero substitution)
  - 3x3 spatial grid coordinate mapping and boundary clamping
  - Box-to-zone center mapping and overlap calculation
  - Mobility factor computation
  - Empty detections and edge case handling
  - SpatialZoneMapper accumulation and video summary aggregation
  - Spatial heatmap rendering output verification
"""

import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.spatial_zone_mapper import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    ZONE_DEFINITIONS,
    SpatialZoneMapper,
    calculate_mobility_factor,
    calculate_zone_lighting_deficit,
    compute_box_zone_overlap,
    compute_zvi,
    map_box_to_zone,
    map_point_to_zone,
)


class TestPhase06ZoneAndZVI(unittest.TestCase):
    """Test suite for Phase 0.6 Zone Mapping and ZVI calculations."""

    def test_zvi_exact_arithmetic(self):
        """ZVI must equal 100 * (0.50*L + 0.35*I + 0.15*M) exactly."""
        # Case 1: L=0.40, I=0.60, M=0.20
        # 100 * (0.50*0.40 + 0.35*0.60 + 0.15*0.20) = 100 * (0.20 + 0.21 + 0.03) = 44.0
        res = compute_zvi(lighting_deficit=0.40, infrastructure_hazard=0.60, mobility_factor=0.20)
        self.assertTrue(res["is_complete"])
        self.assertEqual(res["status"], "COMPLETE")
        self.assertEqual(res["zvi"], 44.0)
        self.assertEqual(res["contributions"]["L_contribution"], 20.0)
        self.assertEqual(res["contributions"]["I_contribution"], 21.0)
        self.assertEqual(res["contributions"]["M_contribution"], 3.0)

        # Case 2: Minimum boundary (all zeroes)
        res_zero = compute_zvi(0.0, 0.0, 0.0)
        self.assertTrue(res_zero["is_complete"])
        self.assertEqual(res_zero["zvi"], 0.0)

        # Case 3: Maximum boundary (all ones)
        res_max = compute_zvi(1.0, 1.0, 1.0)
        self.assertTrue(res_max["is_complete"])
        self.assertEqual(res_max["zvi"], 100.0)

    def test_zvi_component_clamping_and_normalization(self):
        """Inputs outside [0.0, 1.0] must be clamped strictly within bounds."""
        # Over-unity inputs: L=1.5, I=2.0, M=1.1 -> clamped to 1.0
        res_high = compute_zvi(1.5, 2.0, 1.1)
        self.assertTrue(res_high["is_complete"])
        self.assertEqual(res_high["zvi"], 100.0)
        self.assertEqual(res_high["components"]["L"], 1.0)
        self.assertEqual(res_high["components"]["I"], 1.0)
        self.assertEqual(res_high["components"]["M"], 1.0)

        # Negative inputs: L=-0.5, I=-1.0, M=-0.1 -> clamped to 0.0
        res_low = compute_zvi(-0.5, -1.0, -0.1)
        self.assertTrue(res_low["is_complete"])
        self.assertEqual(res_low["zvi"], 0.0)
        self.assertEqual(res_low["components"]["L"], 0.0)

    def test_zvi_missing_infrastructure_returns_incomplete(self):
        """When infrastructure factor I is None, ZVI must return INCOMPLETE and never substitute 0."""
        res = compute_zvi(lighting_deficit=0.75, infrastructure_hazard=None, mobility_factor=0.30)
        self.assertFalse(res["is_complete"])
        self.assertEqual(res["status"], "INCOMPLETE")
        self.assertIsNone(res["zvi"], "ZVI score must be None when component is missing")
        self.assertTrue(len(res["missing_components"]) > 0)
        self.assertIn("Infrastructure Condition", res["missing_components"][0])
        self.assertIn("Missing or unassessed components", res["explanation"])

    def test_zvi_all_components_missing(self):
        """When all components are None, ZVI clearly discloses all missing elements."""
        res = compute_zvi(None, None, None)
        self.assertFalse(res["is_complete"])
        self.assertIsNone(res["zvi"])
        self.assertEqual(len(res["missing_components"]), 3)

    def test_map_point_to_zone_canonical_coordinates(self):
        """Test point mapping to 3x3 zones across canonical 1280x720 coordinates."""
        # Col width = 426.67, Row height = 240.0
        # Row 0: y in [0..240)
        self.assertEqual(map_point_to_zone(100, 100), 1)    # Top-Left
        self.assertEqual(map_point_to_zone(640, 100), 2)    # Top-Center
        self.assertEqual(map_point_to_zone(1100, 100), 3)   # Top-Right

        # Row 1: y in [240..480)
        self.assertEqual(map_point_to_zone(100, 360), 4)    # Mid-Left
        self.assertEqual(map_point_to_zone(640, 360), 5)    # Mid-Center
        self.assertEqual(map_point_to_zone(1100, 360), 6)   # Mid-Right

        # Row 2: y in [480..720]
        self.assertEqual(map_point_to_zone(100, 600), 7)    # Bottom-Left
        self.assertEqual(map_point_to_zone(640, 600), 8)    # Bottom-Center
        self.assertEqual(map_point_to_zone(1100, 600), 9)   # Bottom-Right

    def test_map_point_boundary_clamping(self):
        """Coordinates outside frame boundary clamp safely into corner zones without crashing."""
        self.assertEqual(map_point_to_zone(-50, -50), 1)      # Clamps to Top-Left
        self.assertEqual(map_point_to_zone(2000, 1500), 9)    # Clamps to Bottom-Right
        self.assertEqual(map_point_to_zone(640, -100), 2)     # Clamps to Top-Center

    def test_map_box_to_zone_center_rule(self):
        """Bounding box maps to primary zone based on its center point."""
        # Box centered at (200, 350) -> Mid-Left (Zone 4)
        box = (150, 300, 250, 400)
        self.assertEqual(map_box_to_zone(box), 4)

        # Box spanning center of frame (550, 300, 750, 420) -> Center (650, 360) -> Mid-Center (Zone 5)
        center_box = (550, 300, 750, 420)
        self.assertEqual(map_box_to_zone(center_box), 5)

    def test_compute_box_zone_overlap(self):
        """Compute area fraction of box falling inside a zone."""
        # Box completely inside Zone 1 (box: 50, 50, 150, 150, area=10000)
        box_in_z1 = (50, 50, 150, 150)
        self.assertAlmostEqual(compute_box_zone_overlap(box_in_z1, 1), 1.0)
        self.assertAlmostEqual(compute_box_zone_overlap(box_in_z1, 2), 0.0)

        # Box straddling Zone 1 and Zone 2 horizontally
        # Zone 1 ends at x=426. Let box span x=376..476 (width=100, height=100)
        # Inside Zone 1: x=376..426 (w=50, h=100 -> area=5000 -> 50%)
        # Inside Zone 2: x=426..476 (w=50, h=100 -> area=5000 -> 50%)
        straddle_box = (376, 50, 476, 150)
        overlap_z1 = compute_box_zone_overlap(straddle_box, 1)
        overlap_z2 = compute_box_zone_overlap(straddle_box, 2)
        self.assertAlmostEqual(overlap_z1, 0.50, delta=0.01)
        self.assertAlmostEqual(overlap_z2, 0.50, delta=0.01)
        self.assertAlmostEqual(overlap_z1 + overlap_z2, 1.0, delta=0.01)

    def test_calculate_mobility_factor_deterministic(self):
        """Mobility factor scales deterministically with pedestrian and vehicle counts."""
        # 0 entities -> 0.0
        self.assertEqual(calculate_mobility_factor(0, 0, 0), 0.0)

        # 5 pedestrians (capacity=5) -> 1.0
        self.assertEqual(calculate_mobility_factor(5, 0, 0, norm_capacity=5.0), 1.0)

        # 2 pedestrians + 2 two-wheelers (weight 2*1 + 2*0.5 = 3.0 / 5.0 = 0.6)
        self.assertAlmostEqual(calculate_mobility_factor(2, 2, 0, norm_capacity=5.0), 0.60)

        # High crowd exceeds capacity -> capped at 1.0
        self.assertEqual(calculate_mobility_factor(10, 10, 5, norm_capacity=5.0), 1.0)

    def test_zone_lighting_deficit_dark_vs_bright(self):
        """Bright frames have low deficit; dark frames have high deficit."""
        bright_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 240, dtype=np.uint8)
        dark_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 20, dtype=np.uint8)

        z1_box = ZONE_DEFINITIONS[1]["box"]
        mean_v_bright, def_bright = calculate_zone_lighting_deficit(bright_frame, z1_box)
        mean_v_dark, def_dark = calculate_zone_lighting_deficit(dark_frame, z1_box)

        self.assertGreater(mean_v_bright, 200.0)
        self.assertLess(def_bright, 0.15)

        self.assertLess(mean_v_dark, 50.0)
        self.assertGreater(def_dark, 0.85)

    def test_spatial_zone_mapper_accumulation(self):
        """SpatialZoneMapper accumulates frames, tracks peak entities, and produces summary."""
        mapper = SpatialZoneMapper()

        # Synthetic frame
        synth_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 128, dtype=np.uint8)

        # Mock frame 1 metrics with 1 person in Zone 1 and 1 vehicle in Zone 5
        metrics_f1 = {
            "person_boxes": [(50, 50, 100, 150)],  # Center (75, 100) -> Zone 1
            "vehicle_boxes": [(600, 300, 700, 420, 2)],  # Center (650, 360) -> Zone 5, 4W
            "masks_detail": [{"box": (50, 50, 100, 150)}],
            "lighting_deficit": 0.498,
        }

        # Mock frame 2 metrics with 3 persons in Zone 1
        metrics_f2 = {
            "person_boxes": [
                (60, 60, 110, 160),
                (70, 70, 120, 170),
                (80, 80, 130, 180),
            ],  # All in Zone 1
            "vehicle_boxes": [],
            "masks_detail": [],
            "lighting_deficit": 0.498,
        }

        mapper.analyze_frame(synth_frame, metrics_f1, manual_infrastructure=None, accumulate=True)
        mapper.analyze_frame(synth_frame, metrics_f2, manual_infrastructure=0.40, accumulate=True)

        summary = mapper.get_video_summary(manual_infrastructure=0.40)
        self.assertEqual(summary["total_frames_processed"], 2)

        z1_summary = summary["zones_summary"][1]
        self.assertEqual(z1_summary["frames_evaluated"], 2)
        self.assertEqual(z1_summary["total_persons_observed"], 4)
        self.assertEqual(z1_summary["peak_persons"], 3)
        self.assertEqual(z1_summary["mean_persons"], 2.0)

        # With manual_infrastructure=0.40, ZVI is complete
        self.assertTrue(z1_summary["zvi_result"]["is_complete"])
        self.assertIsNotNone(z1_summary["zvi_result"]["zvi"])

    def test_spatial_heatmap_rendering_dimensions_and_channels(self):
        """Spatial heatmap rendering must produce strict canonical 1280x720 BGR image."""
        mapper = SpatialZoneMapper()
        synth_frame = np.full((CANONICAL_HEIGHT, CANONICAL_WIDTH, 3), 100, dtype=np.uint8)

        heatmap = mapper.render_spatial_heatmap(synth_frame, metric="lighting", manual_infrastructure=0.50)
        self.assertEqual(heatmap.shape, (CANONICAL_HEIGHT, CANONICAL_WIDTH, 3))
        self.assertEqual(heatmap.dtype, np.uint8)

    def test_video_discovery_and_metadata_validation(self):
        """Test video discovery safely finds videos in data/videos and parses metadata."""
        from scripts.spatial_zone_mapper import discover_local_videos, get_video_metadata

        videos = discover_local_videos()
        self.assertTrue(len(videos) > 0, "Should discover at least one video in data/videos/")

        # Test valid metadata extraction
        sample_video = videos[0]
        meta = get_video_metadata(sample_video)
        self.assertIsNotNone(meta)
        self.assertIn("filename", meta)
        self.assertIn("width", meta)
        self.assertIn("height", meta)
        self.assertIn("fps", meta)
        self.assertGreater(meta["width"], 0)
        self.assertGreater(meta["height"], 0)

        # Test invalid/non-existent video handling
        non_existent = Path("data/videos/does_not_exist_12345.mp4")
        self.assertIsNone(get_video_metadata(non_existent))


if __name__ == "__main__":
    unittest.main()
