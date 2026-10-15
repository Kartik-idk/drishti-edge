# Drishti-Edge: Phase 0 Synthetic Vision Benchmark Baseline

**Document**: `docs/benchmarks/phase0-synthetic-baseline.md`  
**Phase**: Phase 0 — Vision Pipeline Feasibility Benchmark  
**Date**: October 11, 2026  
**Auditor**: Antigravity (Builder and Test Executor)  
**Reviewers**: ChatGPT (Technical Strategist) & Gemini (Independent Reviewer)  

---

## 1. Executive Summary

This document reports the baseline computational performance of the Drishti-Edge vision and telemetry pipeline on local laptop CPU hardware. 

> [!IMPORTANT]
> **Audit Qualification**: This benchmark evaluated an **Object-Detection Inference Proxy** (`yolov8n.pt` trained on COCO) executed on synthetic 720p footage. It establishes the lower-bound processing latency for CPU neural network inference and image telemetry on the development machine. It does **not** validate detection accuracy, recall, or latency for actual face-and-license-plate redaction, as domain-specific weights are not yet incorporated.

---

## 2. Test Environment

### 2.1 Hardware Specifications
* **Host Architecture**: Apple Silicon (MacBook Air / macOS Darwin arm64)
* **Processor**: Apple M4 (10 physical CPU cores)
* **System Memory**: 16.0 GB Unified RAM
* **Execution Target**: CPU device (`device='cpu'`)

### 2.2 Software Environment
* **Python Runtime**: Python 3.14.7 (within local `.venv`)
* **Computer Vision Binding**: `opencv-python` 5.0.0.93 (verified clean, conflict-free installation)
* **Inference Engine**: `ultralytics` 8.4.175 with `torch` 2.14.1
* **Execution Mode**: Strictly Headless (zero GUI/windowing overhead, zero calls to `cv2.imshow`)

---

## 3. Test Procedure & Methodology

1. **Input Video**: 150-frame synthetic test video (`data/videos/synthetic_test.mp4`), generated at 1280 × 720 resolution (30.00 FPS) containing dynamic geometry, variable intensity backgrounds, and moving objects.
2. **Warmup Phase**: 10 initial frames executed to prime threadpools and PyTorch CPU kernel caches; **strictly excluded** from timing aggregations.
3. **Timed Processing**: 100 sequential frames processed through four isolated pipeline stages:
   * **Stage A (Decoding)**: `cap.read()` frame decompression into BGR NumPy arrays.
   * **Stage B (Lighting Telemetry)**: BGR-to-HSV color space conversion, extraction of the V-channel mean ($V_{\text{mean}}$), and computation of the normalized lighting deficit:
     $$\text{lighting\_deficit} = 1.0 - \left(\frac{V_{\text{mean}}}{255.0}\right)$$
   * **Stage C (Detection Pipeline)**: CPU forward inference using Ultralytics YOLOv8-nano (`yolov8n.pt`).
   * **Stage D (Adaptive Redaction)**: ROI extraction, boundary verification, and adaptive Gaussian blur with dynamically scaled odd kernel dimensions clamped to ROI dimensions.
4. **End-to-End Latency & Throughput**: Wall-clock measurement across all four stages for each frame, tracking average, median (p50), and 95th percentile (p95).

---

## 4. Benchmark Results

### 4.1 Overall Performance Summary

| Metric | Measured Value | Provisional Target | Margin |
| :--- | :--- | :--- | :--- |
| **End-to-End Throughput** | **49.64 FPS** | 15.00 FPS | +231% |
| **Average Frame Latency** | **20.14 ms** | 66.67 ms | -46.53 ms |
| **Median Frame Latency (p50)** | **20.09 ms** | — | — |
| **95th Percentile Latency (p95)**| **21.14 ms** | — | — |
| **Frames Processed** | 100 timed (+10 warmup)| — | — |
| **Total Wall-Clock Time** | 2.01 s | — | — |

### 4.2 Latency Breakdown by Stage

```text
========================================================================
  STAGE BREAKDOWN (Average Latency per 720p Frame)
========================================================================
  Stage A - Video Decoding:         0.32 ms   ( 1.6% of frame budget)
  Stage B - Lighting Telemetry:     1.02 ms   ( 5.1% of frame budget)
  Stage C - CPU Object Detection:  18.72 ms   (92.9% of frame budget)
  Stage D - Adaptive Redaction:     0.08 ms   ( 0.4% of frame budget)
------------------------------------------------------------------------
  Total End-to-End Latency:        20.14 ms   (100.0%)
========================================================================
```

### 4.3 Environmental Telemetry Measured
* **Average Mean V-Channel (0–255)**: 84.70
* **Average Normalized Lighting Deficit**: 0.6678

---

## 5. Architectural Findings & Limitations

### 5.1 The Principal Bottleneck
* **Neural Network Inference (Stage C)** accounts for **92.9%** of the frame processing cycle.
* Non-neural preprocessing (Decoding, Lighting Telemetry, and Gaussian Blur Redaction) is exceptionally fast on Apple Silicon, requiring a combined **1.42 ms** (< 8% of frame budget).

### 5.2 Multi-Detector Scaling Implication
* Full privacy requirements mandate detecting **both faces and vehicle license plates**.
* If two distinct YOLO nano models are executed sequentially on CPU (one for faces, one for plates):
  $$\text{Projected Latency} \approx (2 \times 18.72\text{ ms}) + 1.42\text{ ms} = 38.86\text{ ms} \implies \mathbf{\approx 25.7\text{ FPS}}$$
* While ~25.7 FPS remains above our 15 FPS provisional target on the Apple M4, running dual models on constrained, low-power edge platforms (e.g., Raspberry Pi 4/5 or Intel Celeron) will drop throughput significantly (projected 4–7 FPS).

### 5.3 Explicit Limitations
1. **Synthetic Media**: Synthetic video does not test realistic street lighting challenges, camera motion blur, small objects, or crowded perspectives.
2. **Inference Proxy Only**: `yolov8n.pt` detected generic COCO objects, **not** human faces or vehicle license plates.
3. **No Accuracy Guarantee**: Zero precision, recall, or anonymization guarantees can be claimed from this preliminary run.

---

## 6. Required Next Experiments

1. **Obtain Domain-Specific Weights**: Acquire and evaluate vetted face detection weights (e.g. `face_detection_yunet.onnx` or `yolov8n-face.pt`) and license plate weights (`yolov8n-plate.pt`).
2. **Benchmark Dual-Model Sequential Pipeline**: Measure the combined CPU latency of executing both detectors concurrently or sequentially per frame.
3. **Investigate Unified Single-Pass Model**: Explore fine-tuning a unified dual-class YOLO model (detecting both faces and license plates in a single neural pass) to cut inference latency in half.
4. **Real Street Video Validation**: Execute the audited benchmark against actual urban video placed in `data/videos/`.
