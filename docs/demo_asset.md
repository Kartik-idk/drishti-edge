# Public Demonstration Video Asset Documentation

**File:** `data/videos/india_street_demo.mp4`
**Role:** Fixed built-in demonstration footage for Drishti-Edge public cloud showcase
**Document Created:** October 2026

---

## 1. Asset Provenance & Upstream Source

* **Platform:** [Pexels](https://www.pexels.com/)
* **Title:** *Busy Urban Street with Traffic in India*
* **Source Page URL:** [https://www.pexels.com/video/busy-urban-street-with-traffic-in-india-35532623/](https://www.pexels.com/video/busy-urban-street-with-traffic-in-india-35532623/)
* **Direct File Download URL:** [https://videos.pexels.com/video-files/35532623/15053543_3840_2160_30fps.mp4](https://videos.pexels.com/video-files/35532623/15053543_3840_2160_30fps.mp4)
* **Pexels Video ID:** `35532623` (Download File ID: `15053543`)
* **Creator / Contributor:** Pexels Stock Videographer
* **Retrieval Date:** October 11, 2026 (via Safari, Quarantine UUID `FAAEC09F-B6A7-48D0-AB65-03F8C436ABF1`)
* **Filesystem Provenance Proof:**
  - Original download path: `~/Downloads/15053543_3840_2160_30fps.mp4`
  - Local repository working copy: `data/videos/daytime_india.mp4`
  - Binary file comparison: **Exact binary match (0 bytes difference)**
  - SHA-256 Digest (both files): `92531dc51c5ff8adcb71ca123e0fad4b2feac2a5c6d9d4058a41e320df2b1d0f`
  - macOS Extended Attribute (`kMDItemWhereFroms`):
    1. `https://videos.pexels.com/video-files/35532623/15053543_3840_2160_30fps.mp4`
    2. `https://www.pexels.com/video/busy-urban-street-with-traffic-in-india-35532623/`

---

## 2. Licensing Terms & Usage Rights

* **License:** [Pexels License](https://www.pexels.com/license/)
* **Permitted Uses:**
  - Free for personal and commercial use across applications and websites.
  - Attribution is not legally required (provided here for academic and provenance transparency).
  - Modifications and derivative adaptations (e.g. downscaling, clipping, anonymization) are permitted.
* **License Restrictions:**
  - Identifiable people must not be shown in a bad light or offensive context.
  - Don't sell unaltered copies of the photo or video.
  - Don't imply endorsement by people or brands in the video.
* **Public Demonstration Compliance:**
  - The video is processed exclusively through Drishti-Edge's privacy engine, which applies opaque, non-reversible masking over all detected pedestrian faces and vehicle registration marks.
  - Raw unredacted source frames are strictly suppressed from display in public demo mode.

---

## 3. Technical Specifications & Preparation Pipeline

### Source Raw Asset (Pexels Original)
* **Filename:** `15053543_3840_2160_30fps.mp4` (local: `data/videos/daytime_india.mp4`)
* **Resolution:** 3840 × 2160 (4K UHD, 16:9)
* **Frame Rate:** 30.00 FPS
* **Duration:** 12.37 seconds (371 frames)
* **File Size:** 53,831,350 bytes (~51.3 MB)
* **SHA-256 Digest:** `92531dc51c5ff8adcb71ca123e0fad4b2feac2a5c6d9d4058a41e320df2b1d0f`
* **Codec:** H.264 / AVC

### Demonstration Proxy Asset
* **Filename:** `data/videos/india_street_demo.mp4`
* **Resolution:** 1280 × 720 (Canonical 720p, 16:9 aspect ratio preserved via area downscaling)
* **Frame Rate:** 30.00 FPS
* **Clip Length:** 240 frames (8.00 seconds, frames 0–239)
* **File Size:** 5,560,413 bytes (~5.30 MB)
* **Container / Codec:** MP4 (`avc1` / H.264)
* **SHA-256 Digest:** `6449b5bddf8971f280dfa14ad50e03f66f080b5d1406f33bd6e0b34b45ba077b`
* **Preparation Method:** Extracted using OpenCV (`cv2.VideoCapture`), downsampled with `cv2.resize(..., (1280, 720), interpolation=cv2.INTER_AREA)`, and encoded at 30.0 FPS.

---

## 4. Pipeline Suitability & Detection Profile

* **Scene Description:** High-density daytime Indian urban corridor with heterogeneous road users including pedestrians, auto-rickshaws, passenger cars, motorcycles, and buses.
* **Face Redaction:** Features walking pedestrians and scooter riders evaluated by YuNet ONNX with localized head fallbacks.
* **Plate Redaction:** Features close-range and mid-range Indian commercial/private vehicles evaluated by YOLOv8n-plate with class-aware 2W/3W mudguard and bumper fallbacks.

---

## 5. Honest Detection Audit & Real-World Limitations

Visual frame-by-frame inspection of representative frames revealed key strengths and realistic limitations of edge detection on unconstrained Indian street traffic:

### Verified Detection Strengths:
1. **Frontal Pedestrian Faces:** Walking pedestrians approaching the camera are reliably identified and masked (e.g., Frame 210, pedestrian in blue shirt detected with conf `0.36` [`FACE #9`] and masked with solid opaque black).
2. **Standard Two-Wheeler Plates:** High-contrast motorcycle registration marks are successfully captured (e.g., Frame 210, motorcycle rear plate `DL5SD C8227` detected with conf `0.60` [`PLATE #10`] and masked with solid opaque black).
3. **Class-Aware Fallbacks:** For vehicles without direct plate hits, bumper and mudguard fallbacks cover vehicle lower regions (3,009 fallback masks applied across 240 frames).

### Documented Edge Cases & Limitations:
1. **False Plate Detection Leading to Plate Leak (Frame 150):** On the rear of a commercial E-rickshaw, a rectangular Hindi advertisement banner ("गुप्त रोगी मिलें... 9350960924") was misidentified as a license plate (`conf=0.41`). Because the primary plate detector reported a hit, the fallback bumper mask was suppressed, leaving the actual small green Delhi EV plate (`DL 1ER ...`) on the lower bumper unmasked in that frame.
2. **Rear-Facing Drivers / Occluded Faces:** Drivers and riders whose backs are turned (e.g., the driver inside the E-rickshaw) are not identified by frontal face detectors (YuNet).
3. **Street Poster False Positives:** Streetlamp and utility pole posters displaying small portrait photos trigger YuNet face detections and are masked.
4. **Fail-Closed Semantics:** Fail-closed release gate approval (`RELEASE_APPROVED`) verifies that all detection stages, standardization buffers, and masking passes executed without software exceptions; it does **not** guarantee 100% real-world detection recall under challenging optical conditions. Full municipal deployment requires domain-specific Indian vehicle and plate models.
