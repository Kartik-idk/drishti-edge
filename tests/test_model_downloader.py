#!/usr/bin/env python3
"""Unit tests for Drishti-Edge Model & Asset Verification Manager

Tests:
  - SHA-256 digest computation
  - Model integrity verification against known hashes
  - Rejection of corrupt or missing files
  - Download handling and temporary file cleanup on failure
  - Demo video procedural availability
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.model_downloader import (
    CANONICAL_DEMO_VIDEO,
    REQUIRED_MODELS,
    compute_file_sha256,
    download_file_verified,
    ensure_demo_video_available,
    ensure_models_available,
    verify_model_integrity,
)


class TestModelDownloader(unittest.TestCase):
    """Test suite for model download and integrity validation routines."""

    def test_compute_file_sha256(self):
        """SHA-256 computation must match hashlib on known byte buffers."""
        with tempfile.NamedTemporaryFile() as tmp:
            test_bytes = b"Drishti-Edge-Privacy-Engine-Test-Data"
            tmp.write(test_bytes)
            tmp.flush()
            expected = hashlib.sha256(test_bytes).hexdigest().lower()
            computed = compute_file_sha256(Path(tmp.name))
            self.assertEqual(computed, expected)

    def test_verify_model_integrity_valid_and_invalid(self):
        """Integrity check must return True only on exact hash match and non-empty file."""
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(b"Valid content")
            tmp.flush()
            correct_hash = hashlib.sha256(b"Valid content").hexdigest().lower()
            wrong_hash = "0" * 64

            # Correct hash -> True
            self.assertTrue(verify_model_integrity(Path(tmp.name), correct_hash))

            # Mismatched hash -> False
            self.assertFalse(verify_model_integrity(Path(tmp.name), wrong_hash))

        # Missing file -> False
        nonexistent = Path(tempfile.gettempdir()) / "nonexistent_model_test.pt"
        self.assertFalse(verify_model_integrity(nonexistent, correct_hash))

        # Empty file -> False
        with tempfile.NamedTemporaryFile() as tmp_empty:
            empty_hash = hashlib.sha256(b"").hexdigest().lower()
            self.assertFalse(verify_model_integrity(Path(tmp_empty.name), empty_hash))

    def test_required_models_manifest_completeness(self):
        """Manifest must define face, vehicle, and plate models with required fields."""
        required_keys = {"face", "vehicle", "plate"}
        self.assertEqual(set(REQUIRED_MODELS.keys()), required_keys)

        for key, spec in REQUIRED_MODELS.items():
            self.assertIn("name", spec)
            self.assertIn("filename", spec)
            self.assertIn("relative_path", spec)
            self.assertTrue(spec["url"].startswith("https://"))
            self.assertEqual(len(spec["sha256"]), 64)
            self.assertGreater(spec["expected_bytes"], 1000)

    def test_download_file_verified_hash_mismatch_cleans_up(self):
        """Simulated download with expected hash mismatch must abort and remove temp file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / "test_fake_model.pt"
            # Using data: URL or local file URL to simulate download
            dummy_file = Path(tmp_dir) / "dummy_source.bin"
            dummy_file.write_bytes(b"Simulated source bytes")

            success, msg = download_file_verified(
                url=dummy_file.as_uri(),
                dest_path=dest,
                expected_sha256="bad" * 21 + "b",
                expected_bytes=len(b"Simulated source bytes"),
            )

            self.assertFalse(success)
            self.assertIn("mismatch", msg.lower())
            self.assertFalse(dest.exists(), "Target file must not be created on hash mismatch")
            # Verify no orphaned temp files
            tmp_files = list(Path(tmp_dir).glob("*.tmp*"))
            self.assertEqual(len(tmp_files), 0, "Temporary staging files must be cleaned up")

    def test_ensure_demo_video_available(self):
        """Canonical demonstration video must exist and match canonical filename."""
        vid_path = ensure_demo_video_available(PROJECT_ROOT)
        self.assertTrue(vid_path.is_file())
        self.assertGreater(vid_path.stat().st_size, 0)
        self.assertEqual(vid_path.name, CANONICAL_DEMO_VIDEO["filename"])

    def test_ensure_demo_video_fallback(self):
        """When canonical demo video is absent in a custom root, synthetic fallback is generated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            vid_path = ensure_demo_video_available(tmp_root)
            self.assertTrue(vid_path.is_file())
            self.assertEqual(vid_path.name, "synthetic_test.mp4")


if __name__ == "__main__":
    unittest.main()
