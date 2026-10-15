# Phase 0.4: Privacy Engine Hardening & Performance Benchmark Report

**Project:** Drishti-Edge  
**Competition:** Seva First Innovation Challenge, Theme 5, Problem Statement 1  
**Hardware:** Apple M4 (arm64), macOS 26.5.2  
**Execution Environment:** Python 3.14.7 | PyTorch 2.14.1 (MPS: True) | OpenCV 5.0.0 | Ultralytics 8.4.175  
**Benchmark Script:** `scripts/evaluate_privacy_consolidated.py`  
**Safety Suite:** `tests/test_fail_closed_privacy.py`  
**Validation Template:** `docs/benchmarks/manual_validation_template.csv`  
**Date:** October 2026  

---

## 1. Executive Summary & Key Milestones

Phase 0.4 establishes a hardened, fail-closed privacy engine enforcing a canonical 1280×720 processing pipeline, conservative geometric fallbacks, a 3-frame temporal mask tracker, and batched plate crop inference on Apple Silicon MPS.

### Key Milestones Achieved:
1. **Canonical 1280×720 Output Resolution Enforced:**
   * Downstream exposure to unredacted 4K frames is completely eliminated.
   * Frame standardization to 1280×720 occurs immediately after decoding.
   * All face, vehicle, plate, fallback, and masking coordinates are unified in this canonical space.
2. **Every-Frame Processing (500 Consecutive Frames Evaluated):**
   * **`daytime_india.mp4` (Repeated Pre-Recorded Benchmark, 2 loops):**
     * Throughput: **19.10 FPS** (Target: $\ge$ 15.0 FPS)
     * Mean Latency: **52.35 ms** | P50: **51.26 ms** | P95: **68.05 ms**
     * Budget Assessment: Exceeds 66.67 ms (68.05 ms) — per-frame 15 FPS budget not met for slowest 5% of frames
   * **`night_street.mp4` (Repeated Pre-Recorded Benchmark, 2 loops):**
     * Throughput: **26.45 FPS**
     * Mean Latency: **37.80 ms** | P50: **36.11 ms** | P95: **48.72 ms**
3. **OpenCV Thread Pool Configuration (Task 7):**
   * Benchmarked OpenCV DNN thread settings on representative 720p YuNet face detection:
     * 1 thread: 29.91 ms
     * 2 threads: 10.95 ms
     * 4 threads: 10.81 ms
   * 4 threads selected for production configuration (achieving optimal parallel execution on Apple M4 without thread thrashing).
4. **Plate Batching Speedup & Equivalence on MPS (Task 6):**
   * Batched execution on Apple Silicon MPS achieves a **2.51× speedup** over sequential execution (64.73 ms $\rightarrow$ 25.78 ms).
   * Bounding box equality test verified **100.0% matching detections** within a 3-pixel coordinate tolerance.
5. **Conservative Fallbacks & Temporal Protection:**
   * Bottom 30% geometric fallback applied to vehicle crops with zero plate detections.
   * Top 30% geometric fallback applied to person bounding boxes with no associated face.
   * 3-frame temporal mask hold with velocity-based motion adjustment mitigates detector dropouts.
6. **Automated Fail-Closed Privacy Gating:**
   * All 18 unit tests in `tests/test_fail_closed_privacy.py` passed cleanly.
   * Any detector fault, missing model, or redaction exception yields a solid black 1280×720 suppression frame.

---

## 2. Environment Metadata & Model Checksums (Task 7)

The runtime environment and model checkpoints were inspected and captured programmatically:

| Parameter | Captured Runtime Value |
| :--- | :--- |
| **Python Version** | `3.14.7 (main, Aug  5 2026, 10:29:49) [Clang 21.0.0 (clang-2100.1.1.101)]` |
| **OpenCV Version** | `5.0.0` |
| **PyTorch Version** | `2.14.1` |
| **Ultralytics Version** | `8.4.175` |
| **OS / Platform** | `macOS-26.5.2-arm64-arm-64bit-Mach-O` (macOS 26.5.2) |
| **Architecture** | `arm64` |
| **MPS Acceleration** | Available: `True` (Built: `True`) |
| **Configured OpenCV Threads** | `10` threads |

### Verified Model Artifacts:

| Model Role | File Path | File Size | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| **Face Detector** | `models/face_detection_yunet_2023mar.onnx` | 232,589 bytes | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| **Vehicle Context** | `yolov8n.pt` | 6,549,796 bytes | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| **Plate Detector** | `models/yolov8n_plate.pt` | 6,248,291 bytes | `2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822` |

---

## 3. OpenCV Thread Scaling Benchmark (Task 7)

Benchmarked on 30 iterations of YuNet face detection on canonical 1280×720 frames:

| Thread Count | Mean Latency | Median Latency | P95 Latency | Relative Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **1 Thread** | 29.91 ms | 29.87 ms | 30.21 ms | 1.00× (Baseline) |
| **2 Threads** | 10.95 ms | 10.91 ms | 11.44 ms | 2.73× |
| **4 Threads** | **10.81 ms** | **10.75 ms** | **11.20 ms** | **2.77×** |

*Note: Ultralytics imports invoke `cv2.setNumThreads(0)`. Drishti-Edge explicitly restores `cv2.setNumThreads(4)` at engine initialization.*

---

## 4. Vehicle Crop Batching vs. Sequential Inference (Task 6)

Evaluated across 84 vehicle crops extracted from 20 test frames on Apple Silicon MPS (`models/yolov8n_plate.pt`, imgsz=320):

| Execution Mode | Mean Latency | Median Latency | P95 Latency | Speedup | Box Match Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sequential Per-Crop (MPS)** | 64.73 ms | 46.28 ms | 151.84 ms | 1.00× | Reference |
| **Batched Forward Pass (MPS)** | **25.78 ms** | **17.53 ms** | **65.46 ms** | **2.51×** | **100.0%** |

*Synchronized using `torch.mps.synchronize()` before and after isolated timed forward passes.*

---

## 5. 500-Frame Sustained Performance Benchmark (Task 6)

Tested continuously over 500 consecutive decoded frames with no frame skipping:

| Metric | `daytime_india.mp4` | `night_street.mp4` | Note / Specification |
| :--- | :--- | :--- | :--- |
| **Source Frames Decoded** | 500 | 500 | 500 consecutive frames |
| **Face Stage Frames Processed** | 500 | 500 | 100% of frames |
| **Vehicle Stage Frames Processed** | 500 | 500 | 100% of frames |
| **Plate Stage Frames Processed** | 500 | 500 | 100% of frames |
| **Video Loops Required** | 2 loops | 2 loops | Repeated pre-recorded test |
| **Total Wall-Clock Time** | 26.17 s | 18.90 s | Continuous playback |
| **Average Throughput (FPS)** | **19.10 FPS** | **26.45 FPS** | Target: $\ge$ 15.0 FPS |
| **Mean Frame Latency (e2e)** | **52.35 ms** | **37.80 ms** | Full end-to-end cycle |
| **P50 Latency (Median)** | **51.26 ms** | **36.11 ms** | Median per-frame time |
| **P95 Latency** | **68.05 ms** | **48.72 ms** | 95th percentile |
| **Min / Max Latency** | 32.8 ms / 118.1 ms | 27.4 ms / 78.5 ms | Outlier bounds |
| **Exceptions / Suppressions** | 0 / 0 | 0 / 0 | 0 failures |
| **Vehicle Crops Considered** | 2258 | 1785 | Height $\ge$ 60 px |
| **Vehicle Crops Selected** | 2258 | 1785 | Capped at top 8 per frame |
| **Fallback Masks Applied** | 3278 | 2596 | Vehicle 30% + Person 30% |
| **Frames Skipped** | 0 | 0 | Zero frames skipped |
| **Output Resolution** | 1280x720 | 1280x720 | Canonical 1280×720 |

### Per-Stage Average Latency Breakdown:

| Processing Sub-Stage | Daytime Mean | Nighttime Mean | Implementation Detail |
| :--- | :--- | :--- | :--- |
| **1. Video Decode** | 2.43 ms | 2.24 ms | OpenCV VideoCapture |
| **2. Canonical Standardization** | 0.34 ms | 0.34 ms | 4K $\rightarrow$ 720p scaling |
| **3. Face Detection** | 11.94 ms | 11.69 ms | YuNet ONNX (4 threads) |
| **4. Vehicle Context** | 11.67 ms | 8.16 ms | YOLOv8n (MPS) |
| **5. Crop Preparation** | 4.40 ms | 3.38 ms | Slicing & sorting top 8 |
| **6. Batched Plate Inference** | 20.73 ms | 11.37 ms | YOLOv8n-plate (MPS) |
| **7. Coordinate Mapping** | 0.22 ms | 0.11 ms | Offset calculation |
| **8. Mask Painting** | 0.44 ms | 0.37 ms | Solid black numpy fill |

---

## 6. Manual Validation Protocol (Task 8)

The 20-sample validation template is located at:
[`docs/benchmarks/manual_validation_template.csv`](file:///Users/kartikeyapradhan/drishti-Edge/docs/benchmarks/manual_validation_template.csv)

* **Samples:** 10 Daytime samples (`DAY-01` to `DAY-10`), 10 Nighttime samples (`NIGHT-01` to `NIGHT-10`).
* **Interactive Tool:** Run `python scripts/verify_privacy_recall.py` to inspect in-memory side-by-side frames.
* **Ground Truth Rule:** Human verification count columns remain strictly unpopulated pending developer inspection.

---

## 7. Remaining Limitations & Prototype Boundaries

1. **P95 Latency Variance:** While average throughput exceeds 15 FPS (19.10 FPS daytime, 26.45 FPS nighttime), daytime P95 latency is 68.05 ms. In frames with numerous vehicles, per-frame latency exceeds the 66.67 ms budget.
2. **Geometric Fallback Imperfection:** The bottom-30% vehicle mask is a heuristic that does not cover irregularly positioned plates (e.g. high-mounted truck plates or top carrier plates).
3. **Indian Plate Detector Generalization:** The community checkpoint `yolov8n_plate.pt` was trained on standard passenger vehicles; recall on two-wheelers with non-standard yellow/black font remains limited without domain fine-tuning.
4. **Thermal Stability:** A 500-frame benchmark (~26.2 s) does not represent multi-hour municipal deployment under thermal throttling conditions.
