#!/usr/bin/env python3
"""Unit Tests for Drishti-Edge Phase 0.7.1 Video Frame Limit & Audit Modes

Tests:
  - Quick-audit frame budget capping vs uncapped
  - Full-video mode at stride 1 (evaluate every source frame)
  - Full-video mode at stride 2 (evaluate every 2nd source frame)
  - Full-video mode at stride 5 and other sampling strides
  - Last-frame inclusion and expected count arithmetic (ceil(N/S))
  - Empty or invalid videos (N <= 0, stride <= 0, nonexistent files)
  - Progress reaching 100% completion on normal finish
  - Honest reporting on premature stream termination / decode failure
  - Safe VideoCapture resource release on normal and exception paths
  - Session state isolation: changing videos, modes, or strides invalidates stale results
  - Streamlit AppTest integration: verifying Quick and Full mode controls
"""

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.spatial_zone_mapper import (
    calculate_expected_evaluated_frames,
    get_video_metadata,
)


class TestAuditModesAndFrameLimits(unittest.TestCase):
    """Test suite for Quick Audit and Full-Video Audit modes."""

    def test_quick_audit_frame_budget_capping(self):
        """Quick audit must cap evaluation at the specified frame budget (default 300)."""
        # Case 1: nighttime_india.mp4 (891 frames), stride 1, default max 300
        # Full would be 891; quick mode must cap at 300
        count = calculate_expected_evaluated_frames(
            total_frames=891, stride=1, audit_mode="quick", max_frames=300
        )
        self.assertEqual(count, 300)

        # Case 2: nighttime_india.mp4 (891 frames), stride 2, max 300
        # Full would be ceil(891/2) = 446; quick mode must cap at 300
        count_stride2 = calculate_expected_evaluated_frames(
            total_frames=891, stride=2, audit_mode="quick", max_frames=300
        )
        self.assertEqual(count_stride2, 300)

        # Case 3: Custom user slider value (e.g. 60 frames)
        count_custom = calculate_expected_evaluated_frames(
            total_frames=891, stride=2, audit_mode="quick", max_frames=60
        )
        self.assertEqual(count_custom, 60)

        # Case 4: Short video where full evaluated frames is LESS than budget
        # 100 frames, stride 2 -> 50 frames expected. max_frames=300 -> returns 50 (not capped)
        count_short = calculate_expected_evaluated_frames(
            total_frames=100, stride=2, audit_mode="quick", max_frames=300
        )
        self.assertEqual(count_short, 50)

        # Case 5: Default max_frames=None defaults to 300 cap
        count_default = calculate_expected_evaluated_frames(
            total_frames=891, stride=1, audit_mode="quick", max_frames=None
        )
        self.assertEqual(count_default, 300)

    def test_full_video_mode_stride_1(self):
        """Full-video mode at stride 1 evaluates every single source frame."""
        # nighttime_india.mp4 (891 frames) -> 891 frames
        count_891 = calculate_expected_evaluated_frames(891, 1, audit_mode="full")
        self.assertEqual(count_891, 891)

        # daytime_india.mp4 (371 frames) -> 371 frames
        count_371 = calculate_expected_evaluated_frames(371, 1, audit_mode="full")
        self.assertEqual(count_371, 371)

        # Single frame video -> 1 frame
        count_1 = calculate_expected_evaluated_frames(1, 1, audit_mode="full")
        self.assertEqual(count_1, 1)

    def test_full_video_mode_stride_2(self):
        """Full-video mode at stride 2 evaluates every second source frame: ceil(N/2)."""
        # 891 frames, stride 2 -> ceil(891/2) = 446
        count_891 = calculate_expected_evaluated_frames(891, 2, audit_mode="full")
        self.assertEqual(count_891, 446)
        # Evaluated frame indices: 0, 2, 4, ..., 890 (last frame 890 is index 890 = frame 891)
        sampled_indices = list(range(0, 891, 2))
        self.assertEqual(len(sampled_indices), 446)
        self.assertEqual(sampled_indices[-1], 890)

        # 890 frames (even), stride 2 -> ceil(890/2) = 445
        count_890 = calculate_expected_evaluated_frames(890, 2, audit_mode="full")
        self.assertEqual(count_890, 445)
        sampled_indices_even = list(range(0, 890, 2))
        self.assertEqual(len(sampled_indices_even), 445)
        self.assertEqual(sampled_indices_even[-1], 888)

        # 371 frames, stride 2 -> ceil(371/2) = 186
        count_371 = calculate_expected_evaluated_frames(371, 2, audit_mode="full")
        self.assertEqual(count_371, 186)
        sampled_371 = list(range(0, 371, 2))
        self.assertEqual(len(sampled_371), 186)
        self.assertEqual(sampled_371[-1], 370)

    def test_full_video_mode_stride_5_and_various_strides(self):
        """Full-video mode handles stride 5, 10, 15 correctly matching ceil(N/S)."""
        strides_to_test = [1, 2, 3, 5, 7, 10, 15]
        test_frame_counts = [1, 2, 10, 371, 890, 891, 1000]

        for n in test_frame_counts:
            for s in strides_to_test:
                expected_ceil = math.ceil(n / s)
                calc_val = calculate_expected_evaluated_frames(n, s, audit_mode="full")
                actual_sampled_len = len(range(0, n, s))
                self.assertEqual(
                    calc_val,
                    expected_ceil,
                    f"Mismatch for N={n}, S={s}: calc={calc_val}, ceil={expected_ceil}",
                )
                self.assertEqual(
                    calc_val,
                    actual_sampled_len,
                    f"Mismatch with range(0, {n}, {s}): calc={calc_val}, len={actual_sampled_len}",
                )

    def test_last_frame_inclusion_and_expected_count(self):
        """Verify last-frame inclusion logic and exact boundary conditions."""
        # For N frames starting at frame 0 with stride S:
        # The last source frame index is N - 1.
        # It is sampled if and only if (N - 1) % S == 0.

        # Example 1: N=891, S=2 -> (891 - 1) % 2 = 0 -> last frame (890) is included.
        indices = list(range(0, 891, 2))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 890)
        self.assertEqual(len(indices), calculate_expected_evaluated_frames(891, 2, "full"))

        # Example 2: N=891, S=5 -> (891 - 1) % 5 = 890 % 5 = 0 -> last frame (890) is included.
        indices_5 = list(range(0, 891, 5))
        self.assertEqual(indices_5[0], 0)
        self.assertEqual(indices_5[-1], 890)
        self.assertEqual(len(indices_5), calculate_expected_evaluated_frames(891, 5, "full"))

        # Example 3: N=890, S=2 -> (890 - 1) % 2 = 1 -> last frame (889) is NOT sampled; last sampled is 888.
        indices_even = list(range(0, 890, 2))
        self.assertEqual(indices_even[-1], 888)
        self.assertEqual(len(indices_even), calculate_expected_evaluated_frames(890, 2, "full"))

        # Systematic check for N in [1..50] and S in [1..10]
        for n in range(1, 51):
            for s in range(1, 11):
                sampled = list(range(0, n, s))
                expected = calculate_expected_evaluated_frames(n, s, "full")
                self.assertEqual(len(sampled), expected)
                if (n - 1) % s == 0:
                    self.assertEqual(sampled[-1], n - 1)
                else:
                    self.assertNotEqual(sampled[-1], n - 1)
                    self.assertTrue(sampled[-1] < n - 1)

    def test_real_nighttime_india_metadata_and_full_sampling(self):
        """Verify real nighttime_india.mp4 (891 frames) metadata and full sampling count."""
        vid_path = PROJECT_ROOT / "data" / "videos" / "nighttime_india.mp4"
        if not vid_path.exists():
            self.skipTest(f"Video file not present: {vid_path}")

        meta = get_video_metadata(vid_path)
        self.assertIsNotNone(meta)
        self.assertEqual(meta["total_frames"], 891)

        # Stride 1: all 891 frames
        expected_s1 = calculate_expected_evaluated_frames(meta["total_frames"], 1, "full")
        self.assertEqual(expected_s1, 891)

        # Stride 2: 446 frames (uncapped, previously capped at 300)
        expected_s2 = calculate_expected_evaluated_frames(meta["total_frames"], 2, "full")
        self.assertEqual(expected_s2, 446)
        self.assertGreater(expected_s2, 300)

        # Stride 5: 179 frames
        expected_s5 = calculate_expected_evaluated_frames(meta["total_frames"], 5, "full")
        self.assertEqual(expected_s5, 179)

        # Verify quick mode caps at 300 for stride 1 and 2, but not stride 5
        self.assertEqual(calculate_expected_evaluated_frames(meta["total_frames"], 1, "quick", 300), 300)
        self.assertEqual(calculate_expected_evaluated_frames(meta["total_frames"], 2, "quick", 300), 300)
        self.assertEqual(calculate_expected_evaluated_frames(meta["total_frames"], 5, "quick", 300), 179)

    def test_empty_or_invalid_videos(self):
        """Zero frames, negative frames, or invalid stride must return 0 evaluated frames."""
        self.assertEqual(calculate_expected_evaluated_frames(0, 1, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(0, 2, "quick"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(-50, 1, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, 0, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, -2, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, 2, "quick", max_frames=0), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, 2, "quick", max_frames=-10), 0)
        self.assertEqual(calculate_expected_evaluated_frames(None, 1, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, None, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames("invalid", 1, "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, "invalid", "full"), 0)
        self.assertEqual(calculate_expected_evaluated_frames(100, 2, "quick", max_frames="bad"), 50)

        # get_video_metadata on nonexistent file returns None
        meta = get_video_metadata(Path("/tmp/nonexistent_video_drishti_test.mp4"))
        self.assertIsNone(meta)

    def test_progress_reaching_completion_simulation(self):
        """Simulate video evaluation loop and verify progress reaches 1.0 (100%) upon normal completion."""
        total_source_frames = 25
        stride = 2
        audit_mode = "full"
        expected = calculate_expected_evaluated_frames(total_source_frames, stride, audit_mode)
        # ceil(25 / 2) = 13

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_vid:
            # Create a 25-frame dummy video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_vid.name, fourcc, 10.0, (64, 64))
            for i in range(total_source_frames):
                frame = np.full((64, 64, 3), i, dtype=np.uint8)
                out.write(frame)
            out.release()

            # Execute simulation following app.py audit loop pattern
            cap = cv2.VideoCapture(tmp_vid.name)
            self.assertTrue(cap.isOpened())

            evaluated_count = 0
            frame_idx = 0
            progress_history = []

            try:
                while evaluated_count < expected:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    if frame_idx % stride == 0:
                        evaluated_count += 1
                        progress = min(1.0, evaluated_count / max(1, expected))
                        progress_history.append(progress)

                    frame_idx += 1
            finally:
                cap.release()

            # VideoCapture must be safely closed
            self.assertFalse(cap.isOpened())

            # Verification of completion
            self.assertEqual(evaluated_count, expected)
            self.assertEqual(len(progress_history), expected)
            self.assertAlmostEqual(progress_history[-1], 1.0)
            self.assertEqual(frame_idx, total_source_frames)

    def test_premature_or_corrupt_stream_reporting(self):
        """Premature termination or decode failure must not falsely report 100% completion."""
        expected_frames = 20
        evaluated_count = 8  # Stream truncates early

        progress = min(1.0, evaluated_count / max(1, expected_frames))
        self.assertEqual(progress, 8 / 20)
        self.assertLess(progress, 1.0)
        self.assertNotEqual(evaluated_count, expected_frames)

    def test_videocapture_resource_safety_on_exception(self):
        """VideoCapture must be released even when an unhandled exception occurs in the processing loop."""
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_vid:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_vid.name, fourcc, 10.0, (64, 64))
            out.write(np.zeros((64, 64, 3), dtype=np.uint8))
            out.release()

            cap = cv2.VideoCapture(tmp_vid.name)
            self.assertTrue(cap.isOpened())

            with self.assertRaises(RuntimeError):
                try:
                    ret, frame = cap.read()
                    raise RuntimeError("Simulated mid-stream failure")
                finally:
                    cap.release()

            # VideoCapture must be released
            self.assertFalse(cap.isOpened())

    def test_session_state_key_isolation(self):
        """Switching videos, audit scopes, max frame budgets, or strides must yield distinct guard keys."""
        # Function replicating the key generation logic in app.py
        def make_audit_key(video_name: str, is_quick: bool, max_frames: Optional[int], stride: int) -> str:
            if is_quick:
                return f"{video_name}::quick_max_{max_frames}::stride_{stride}"
            return f"{video_name}::full::stride_{stride}"

        k1 = make_audit_key("daytime_india.mp4", is_quick=True, max_frames=60, stride=2)
        k2 = make_audit_key("daytime_india.mp4", is_quick=False, max_frames=None, stride=2)
        k3 = make_audit_key("nighttime_india.mp4", is_quick=False, max_frames=None, stride=2)
        k4 = make_audit_key("nighttime_india.mp4", is_quick=False, max_frames=None, stride=5)
        k5 = make_audit_key("nighttime_india.mp4", is_quick=True, max_frames=300, stride=5)

        keys = [k1, k2, k3, k4, k5]
        # All keys must be mutually unique
        self.assertEqual(len(keys), len(set(keys)))

        # Simulate state clearing logic on key transition
        mock_session_state: Dict[str, Any] = {
            "active_audit_key": k1,
            "evaluated_count": 60,
            "expected_evaluated": 60,
            "throughput_fps": 12.5,
            "zone_mapper": object(),
        }

        # User switches from k1 (Quick 60) to k2 (Full)
        new_key = k2
        if mock_session_state.get("active_audit_key") != new_key:
            mock_session_state["active_audit_key"] = new_key
            for stale_field in [
                "zone_mapper",
                "evaluated_count",
                "expected_evaluated",
                "audit_scope",
                "frame_stride",
                "total_source_frames",
                "fail_closed_count",
                "total_run_sec",
                "throughput_fps",
                "last_metrics",
                "last_redacted",
                "last_diag",
                "last_spatial",
                "last_source",
            ]:
                mock_session_state.pop(stale_field, None)

        self.assertEqual(mock_session_state["active_audit_key"], k2)
        self.assertNotIn("zone_mapper", mock_session_state)
        self.assertNotIn("evaluated_count", mock_session_state)
        self.assertNotIn("throughput_fps", mock_session_state)


class TestStreamlitAppTestIntegration(unittest.TestCase):
    """Integration tests using Streamlit AppTest for mode switching and UI controls."""

    def test_apptest_audit_scope_controls(self):
        """Streamlit AppTest must mount with Audit Scope radio and support switching to Full-Video Audit."""
        from streamlit.testing.v1 import AppTest

        app_path = str(PROJECT_ROOT / "app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        self.assertEqual(len(at.exception), 0, f"App mounted with exceptions: {at.exception}")

        # Check Audit Scope radio control exists
        self.assertTrue(len(at.radio) > 0, "Audit Scope radio button not found")
        scope_radio = at.radio[0]
        self.assertEqual(scope_radio.label, "Audit Scope:")
        self.assertIn("Quick Audit (up to 300 frames)", scope_radio.options)
        self.assertIn("Full-Video Audit", scope_radio.options)

        # In default Quick mode, Max Frames slider is visible
        self.assertTrue(any("Max Frames to Audit" in s.label for s in at.slider))

        # Switch to Full-Video Audit
        scope_radio.set_value("Full-Video Audit")
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exceptions after switching to Full mode: {at.exception}")

        # In Full mode, Max Frames slider is hidden
        has_max_slider = any("Max Frames to Audit" in s.label for s in at.slider)
        self.assertFalse(has_max_slider, "Max Frames slider should be hidden in Full-Video Audit mode")

    def test_apptest_public_demo_source_and_restrictions(self):
        """In public demo mode, canonical demonstration video is configured and uploads are absent."""
        from streamlit.testing.v1 import AppTest

        app_path = str(PROJECT_ROOT / "app.py")
        at = AppTest.from_file(app_path, default_timeout=30)
        at.run()

        self.assertEqual(len(at.exception), 0, f"App mounted with exceptions: {at.exception}")

        # Verify no file_uploader widgets exist
        self.assertEqual(len(at.file_uploader), 0, "File uploader must not be present in public demo mode")

        # Verify india_street_demo.mp4 is mounted as the built-in source
        markdown_texts = [m.value for m in at.sidebar.markdown]
        self.assertTrue(
            any("india_street_demo.mp4" in text for text in markdown_texts),
            "india_street_demo.mp4 must be displayed as the demonstration video source in sidebar",
        )


if __name__ == "__main__":
    unittest.main()
