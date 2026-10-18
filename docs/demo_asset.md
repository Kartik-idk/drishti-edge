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
* **Retrieval Date:** October 11, 2026

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
* **Codec:** H.264 / AVC

### Demonstration Proxy Asset
* **Filename:** `data/videos/india_street_demo.mp4`
* **Resolution:** 1280 × 720 (Canonical 720p, 16:9 aspect ratio preserved via area downscaling)
* **Frame Rate:** 30.00 FPS
* **Clip Length:** 240 frames (8.00 seconds)
* **File Size:** 5,560,413 bytes (~5.30 MB)
* **Container / Codec:** MP4 (`avc1` / H.264)
* **SHA-256 Digest:** `6449b5bddf8971f280dfa14ad50e03f66f080b5d1406f33bd6e0b34b45ba077b`

---

## 4. Pipeline Suitability & Detection Profile

* **Scene Description:** High-density daytime Indian urban corridor with heterogeneous road users including pedestrians, auto-rickshaws, passenger cars, motorcycles, and buses.
* **Face Redaction:** Features walking pedestrians and scooter riders evaluated by YuNet ONNX with localized head fallbacks.
* **Plate Redaction:** Features close-range and mid-range Indian commercial/private vehicles evaluated by YOLOv8n-plate with class-aware 2W/3W mudguard and bumper fallbacks.
* **Heuristic Limitations:**
  - The video provides an authentic test of visual redaction and pipeline throughput under real-world optical conditions.
  - It does **not** guarantee 100% detection recall across extreme distances, severe occlusions, or sub-pixel plates. Fallback masks cover detected vehicle bounding box regions when direct plate detection is ambiguous.
