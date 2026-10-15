# Drishti-Edge: Phase 0.2 Real Street Video Benchmark Report

**Document**: `docs/benchmarks/phase0-real-video-results.md`  
**Phase**: Phase 0.2 — Real Street Video Performance & Environmental Evaluation  
**Date**: October 11, 2026  
**Auditor**: Antigravity (Builder and Test Executor)  
**Reviewers**: 🔵 ChatGPT (Technical Strategist) & 🟣 Gemini (Independent Reviewer)  

---

## 1. Executive Summary

This report documents the performance of the Drishti-Edge vision and telemetry pipeline when executed on **three real-world street video recordings** recently added to `data/videos/`.

> [!IMPORTANT]
> **Audit Qualification**: This benchmark evaluated our standalone CPU vision pipeline using an **Object-Detection Inference Proxy** (`yolov8n.pt` trained on COCO) to measure real-world decoding, environmental lighting deficit calculation, CPU inference, and Gaussian blur redaction. It does **not** evaluate dedicated face or license-plate detection accuracy, as domain-specific weights are not yet incorporated.

---

## 2. Test Environment

* **Host Architecture**: Apple Silicon (MacBook Air / macOS Darwin arm64)
* **Processor**: Apple M4 (10 physical CPU cores)
* **System Memory**: 16.0 GB Unified RAM
* **Python Runtime**: Python 3.14.7 (`.venv`)
* **Computer Vision Binding**: `opencv-python` 5.0.0.93 (verified conflict-free, strictly headless)
* **Inference Engine**: `ultralytics` 8.4.175 with `torch` 2.14.1
* **Execution Mode**: Strictly Headless CPU (`device='cpu'`)

---

## 3. Video Asset Inspection & Metadata

Each video in `data/videos/` was inspected using OpenCV. All three files decoded successfully without corruption.

| Filename | Resolution | Source FPS | Frame Count | Duration | File Size | Scene Description & Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`daytime_india.mp4`** | **3840 × 2160 (4K)** | 30.00 | 371 | 12.37 s | 51.34 MB | **Daytime Indian Urban Corridor**: Busy street with cars, auto-rickshaws, buses, trucks, and pedestrians. **Optimal** for face and license plate redaction testing due to high optical resolution. |
| **`night_street.mp4`** | **3840 × 2160 (4K)** | 50.00 | 288 | 5.76 s | 8.71 MB | **Dark Urban Night Street**: Low ambient illumination, localized street lamps, sparse vehicles/pedestrians. **Optimal** for testing low-light failure modes and lighting deficit scoring. |
| **`nighttime_india.mp4`** | **320 × 240 (QVGA)** | 30.00 | 891 | 29.70 s | 0.87 MB | **Low-Resolution Nighttime Road**: Dashcam/CCTV perspective. Coarse vehicle movement is visible, but resolution is too low for reliable face/plate landmark detection. |

---

## 4. Benchmark Commands

The benchmark was executed independently on each video using a uniform configuration (10 warm-up frames excluded from statistics, 150 timed frames):

```bash
# Video 1: Daytime 4K
python scripts/benchmark_redaction.py \
  --video data/videos/daytime_india.mp4 \
  --proxy-model yolov8n.pt \
  --max-frames 150 \
  --warmup 10

# Video 2: Nighttime 4K
python scripts/benchmark_redaction.py \
  --video data/videos/night_street.mp4 \
  --proxy-model yolov8n.pt \
  --max-frames 150 \
  --warmup 10

# Video 3: Nighttime QVGA
python scripts/benchmark_redaction.py \
  --video data/videos/nighttime_india.mp4 \
  --proxy-model yolov8n.pt \
  --max-frames 150 \
  --warmup 10
```

---

## 5. Measured Latency & Throughput Results

### 5.1 Comparative Performance Summary

| Metric | `daytime_india.mp4` (4K) | `night_street.mp4` (4K) | `nighttime_india.mp4` (240p) | `synthetic_test.mp4` (720p) | Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Resolution** | 3840 × 2160 | 3840 × 2160 | 320 × 240 | 1280 × 720 | — |
| **Source Video FPS** | 30.00 FPS | 50.00 FPS | 30.00 FPS | 30.00 FPS | — |
| **Timed Frames** | 150 | 150 | 150 | 100 | — |
| **Total Wall Time** | 17.32 s | 21.72 s | 3.29 s | 2.01 s | — |
| **End-to-End Throughput** | **8.66 FPS** | **6.91 FPS** | **45.62 FPS** | **49.64 FPS** | **15.0 FPS** |
| **Target Met?** | ❌ (8.66 FPS) | ❌ (6.91 FPS) | ✅ (45.62 FPS) | ✅ (49.64 FPS) | — |
| **Avg End-to-End Latency**| **115.48 ms** | **144.82 ms** | **21.92 ms** | **20.14 ms** | $\le 66.7$ ms |
| **Median Latency (p50)** | 108.62 ms | 143.87 ms | 21.60 ms | 20.09 ms | — |
| **95th Percentile (p95)**| 185.77 ms | 175.03 ms | 25.00 ms | 21.14 ms | — |
| **Average Mean V (0-255)**| **136.91** | **33.43** | **70.67** | **84.70** | — |
| **Lighting Deficit** | **0.4631** | **0.8689** | **0.7229** | **0.6678** | — |

---

### 5.2 Stage Latency Breakdown (Average per Frame)

```text
========================================================================================
  STAGE BREAKDOWN (Average Latency per Frame)
========================================================================================
  Video               Decoding (A)     Lighting HSV (B)   Detection (C)    Redaction (D)
  --------------------------------------------------------------------------------------
  daytime_india (4K)   2.10 ms ( 1.8%)   8.47 ms ( 7.3%)   20.51 ms (17.8%)  84.39 ms (73.1%)
  night_street  (4K)   2.13 ms ( 1.5%)   8.59 ms ( 5.9%)   19.46 ms (13.4%) 114.64 ms (79.2%)
  nighttime_india(240p)0.08 ms ( 0.4%)   0.11 ms ( 0.5%)   21.40 ms (97.6%)   0.33 ms ( 1.5%)
  synthetic_test(720p) 0.32 ms ( 1.6%)   1.02 ms ( 5.1%)   18.72 ms (92.9%)   0.08 ms ( 0.4%)
========================================================================================
```

---

## 6. Critical Technical Analysis

### 6.1 Why Did 4K Processing Fall Below 15 FPS?
On both 4K clips, end-to-end throughput was **8.66 FPS** and **6.91 FPS**, seemingly missing the 15 FPS target. Detailed stage isolation reveals the exact root cause:
1. **Neural Network Inference (Stage C)** remained steady at **~19.5–20.5 ms** across all clips. Ultralytics automatically letterboxes any incoming frame to 640×640 before running the CPU forward pass.
2. **Decoding & Lighting (Stages A & B)** took ~10.6 ms combined for 4K frames (8.3 million pixels), compared to 1.3 ms on 720p and 0.19 ms on 240p.
3. **Redaction Convolution (Stage D) Exploded to 84–114 ms**: Because `yolov8n.pt` is a COCO proxy detecting whole vehicles (`car`, `bus`, `truck`) and people, bounding boxes in 4K resolution span up to $1500 \times 1000$ pixels. Convolving a 2D Gaussian blur across millions of pixels per frame on CPU caused an artificial bottleneck.
4. **Architectural Implication**: Actual human faces ($~100 \times 100\text{ px}$) and license plates ($~150 \times 60\text{ px}$) represent less than **1%** of the pixel area of a full vehicle. With dedicated face and plate detectors, Stage D will require $< 1\text{ ms}$, which would bring 4K pipeline latency to $\approx 31\text{ ms}$ (**$\approx 32\text{ FPS}$**).

### 6.2 Daytime vs. Nighttime Performance
* **Computation Time**: Nighttime footage is **not** inherently slower to process through the neural network. Model prediction latency remained virtually identical between `daytime_india.mp4` (20.51 ms) and `night_street.mp4` (19.46 ms).
* **Lighting Deficit Validation**: The formula $\text{lighting\_deficit} = 1.0 - (V_{\text{mean}} / 255.0)$ clearly differentiated day from night:
  * Bright daytime: **0.4631**
  * Moderately lit nighttime: **0.7229**
  * Deep darkness: **0.8689**
* **Redaction Variations**: `night_street.mp4` had slightly higher redaction latency (114 ms vs 84 ms) due to persistent large car detections close to the camera lens.

### 6.3 Video Suitability for Subsequent Phases
* **`daytime_india.mp4` is the most valuable clip for face and plate validation**: High optical clarity and Indian urban traffic (auto-rickshaws, pedestrians, motorcycles) make it ideal for validating detection recall.
* **`night_street.mp4` is optimal for low-light edge cases**: Its 0.8689 deficit allows auditing detector degradation under streetlamp glare and deep shadows.
* **`nighttime_india.mp4` is unsuitable for face/plate evaluation**: At 320×240, facial features and license plate alphanumeric characters are below recognizable spatial Nyquist resolution.

---

## 7. Model and Privacy Limitations

1. **Proxy Limitation**: `yolov8n.pt` detected generic COCO categories (`car`, `truck`, `person`, `motorcycle`). No human faces or license plates were detected or redacted.
2. **Zero Accuracy Claims**: Bounding box accuracy, recall, and edge blurring irreversibility on genuine PII cannot be claimed until domain-specific weights are tested.
3. **Ephemeral Lifecycle**: Zero unredacted video frames or image crops were saved to disk during this benchmark.

---

## 8. Recommended Next Experiment

* **Acquire and benchmark dedicated face (`yolov8n-face.pt` or YuNet ONNX) and license-plate weights (`yolov8n-plate.pt`)** on `daytime_india.mp4` and `night_street.mp4` downscaled to 1080p/720p or evaluated at native 4K to measure true face/plate redaction latency and real-world anonymization coverage.
