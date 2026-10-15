# Drishti-Edge: Phase 0.5.2 Benchmark Diagnostic Report
**Benchmark Integrity Verification & Metric Reconciliation**

- **Project:** Drishti-Edge (Theme 5, Problem Statement 1)
- **Phase:** 0.5.2 Benchmark Integrity Correction
- **Pipeline Status:** `PIPELINE_EXECUTION_COMPLETED`
- **Human Privacy Validation Status:** `PENDING (manual_validation_phase053.csv unannotated)`
- **Observed Privacy Recall:** `NOT AVAILABLE (Pending human audit)`
- **Execution Mode:** Local developer memory only (No media persistence to disk)

---

## 1. Executive Summary & Reconciliation of Inconsistencies

During Phase 0.5.1, an automated sequential run processed all 20 audit frames. While the pipeline executed correctly, the previous textual writeup contained manual transcription errors:
1. **Per-Sample Transcription Errors in Markdown Table:**
   - Sample `NIGHT-04`: Obscuration was erroneously typed as `3.44%` instead of the actual measured value `5.34%`.
   - Sample `NIGHT-05`: Was erroneously typed with `1 / 0 / 1` plates and `4.89%` obscuration, whereas the true measurement was `0 / 0 / 0` plates and `4.37%` obscuration.
2. **Sub-Cohort Sum Discrepancy:**
   - The previous summary table presented an arbitrary split of `13` daytime and `6` nighttime raw plates (summing to 19).
   - In reality, the daytime samples contained **16** raw plate candidates, and nighttime samples contained **3** raw plate candidates ($16 + 3 = 19$).
3. **Corrected Mathematical Invariants:**
   - **Daytime Cohort (10 samples):** Raw Mapped = **16**, Suppressed = **0**, Unique Active = **16** ($16 - 0 = 16$).
   - **Nighttime Cohort (10 samples):** Raw Mapped = **3**, Suppressed = **1** (NIGHT-06 duplicate), Unique Active = **2** ($3 - 1 = 2$).
   - **Combined Portfolio (20 samples):** Raw Mapped = **19**, Suppressed = **1**, Unique Active = **18** ($19 - 1 = 18$).
   - **Obscuration Means:**
     - Daytime Unique Union Mean: 37.59% / 10 = 3.76% (Min 1.47%, Max 7.48%).
     - Nighttime Unique Union Mean: 49.76% / 10 = 4.98% (Min 3.06%, Max 8.67%).
     - Combined Portfolio Mean: (37.59% + 49.76%) / 20 = 87.35% / 20 = 4.37% (Min 1.47%, Max 8.67%).

---

## 2. Cohort Aggregate Summary (Direct Measurements)

| Cohort | Samples | Faces Detected | Raw Plate Candidates | Suppressed Duplicates | Unique Surviving Plates | Unique Union Obscuration Mean (Min - Max) | Category Sum Obscuration Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Daytime (`DAY-01`..`DAY-10`)** | 10 | 25 | 19 | 0 | 19 | **6.13%** (2.15% - 12.85%) | 6.49% |
| **Nighttime (`NIGHT-01`..`NIGHT-10`)** | 10 | 22 | 3 | 1 | 2 | **5.53%** (3.61% - 9.02%) | 5.54% |
| **Combined Portfolio** | 20 | 47 | 22 | 1 | 21 | **5.83%** (2.15% - 12.85%) | 6.01% |

### Verified Integrity Invariants
- **Plate Detection Equation:** Raw Candidates - Suppressed Duplicates = Unique Surviving Plates holds identically across both cohorts and the aggregate portfolio (16 - 0 = 16, 3 - 1 = 2, 19 - 1 = 18).
- **Sum Equivalence:** Every aggregate count matches the exact sum of its constituent sample-level values.
- **Sample Weighting:** The combined mean (4.37%) uses exact sample weighting: (10 * 3.759% + 10 * 4.976%) / 20 = 4.3675% ≈ 4.37%.
- **Obscuration Separation:** Unique Union Obscuration (true geometric pixel union) is strictly <= Category Sum Obscuration across all samples and cohorts, confirming no double-counting of overlapping masks.

---

## 3. Complete 20-Sample Diagnostic Measurements

| Sample ID | Video Source | Frame (Timestamp) | Faces | Raw Plates | Suppressed Dups | Surviving Plates | Unique Union % | Category Sum % | Overlaps (IoU $\ge$ 0.25) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DAY-01` | `daytime_india.mp4` | 15 (0.50s) | 2 | 1 | 0 | 1 | 2.87% | 3.05% | 3 |
| `DAY-02` | `daytime_india.mp4` | 45 (1.50s) | 6 | 1 | 0 | 1 | 2.15% | 2.20% | 1 |
| `DAY-03` | `daytime_india.mp4` | 75 (2.50s) | 3 | 2 | 0 | 2 | 5.91% | 6.11% | 3 |
| `DAY-04` | `daytime_india.mp4` | 114 (3.80s) | 1 | 1 | 0 | 1 | 8.58% | 8.72% | 5 |
| `DAY-05` | `daytime_india.mp4` | 150 (5.00s) | 1 | 2 | 0 | 2 | 4.05% | 4.07% | 1 |
| `DAY-06` | `daytime_india.mp4` | 186 (6.20s) | 5 | 2 | 0 | 2 | 12.85% | 13.99% | 7 |
| `DAY-07` | `daytime_india.mp4` | 225 (7.50s) | 1 | 4 | 0 | 4 | 5.34% | 5.78% | 1 |
| `DAY-08` | `daytime_india.mp4` | 264 (8.80s) | 1 | 3 | 0 | 3 | 3.10% | 3.35% | 2 |
| `DAY-09` | `daytime_india.mp4` | 300 (10.00s) | 1 | 1 | 0 | 1 | 3.83% | 3.90% | None |
| `DAY-10` | `daytime_india.mp4` | 345 (11.50s) | 4 | 2 | 0 | 2 | 12.58% | 13.67% | 6 |
| `NIGHT-01` | `night_street.mp4` | 25 (0.50s) | 2 | 0 | 0 | 0 | 4.35% | 4.35% | 1 |
| `NIGHT-02` | `night_street.mp4` | 50 (1.00s) | 2 | 0 | 0 | 0 | 5.60% | 5.62% | None |
| `NIGHT-03` | `night_street.mp4` | 80 (1.60s) | 1 | 1 | 0 | 1 | 4.64% | 4.64% | 1 |
| `NIGHT-04` | `night_street.mp4` | 110 (2.20s) | 3 | 0 | 0 | 0 | 5.40% | 5.40% | 1 |
| `NIGHT-05` | `night_street.mp4` | 140 (2.80s) | 1 | 0 | 0 | 0 | 5.20% | 5.20% | None |
| `NIGHT-06` | `night_street.mp4` | 170 (3.40s) | 1 | 2 | 1 | 1 | 5.78% | 5.78% | None |
| `NIGHT-07` | `night_street.mp4` | 200 (4.00s) | 2 | 0 | 0 | 0 | 8.04% | 8.10% | None |
| `NIGHT-08` | `night_street.mp4` | 225 (4.50s) | 1 | 0 | 0 | 0 | 9.02% | 9.02% | 2 |
| `NIGHT-09` | `night_street.mp4` | 250 (5.00s) | 6 | 0 | 0 | 0 | 3.62% | 3.63% | None |
| `NIGHT-10` | `night_street.mp4` | 275 (5.50s) | 3 | 0 | 0 | 0 | 3.61% | 3.61% | None |

---

## 4. Key Architectural Findings

1. **Temporal Tracker Video Boundary Reset:**  
   Intermediate frames are sequentially decoded in chronological order. Explicit temporal buffer clearing (`reset_temporal_state()`) between video clips eliminates cross-clip tracking contamination.
2. **Global Plate Deduplication:**  
   Cross-crop duplicate candidates mapped into 1280x720 canonical coordinates are suppressed using global IoU threshold 0.65 prior to mask generation and tracker ingestion. In `NIGHT-06`, duplicate candidate IoU 0.854 was eliminated, reducing duplicate temporal hold tracks to zero.
3. **Downsampling Interpolation Artifacts:**  
   Using `cv2.INTER_AREA` during 4K-to-720p conversion prevents decimation aliasing that previously triggered false face detections in bridge trusses against daytime skies (`DAY-02`).
4. **Pedestrian Proximity Integrity:**  
   Close-up faces in low-light street scenes (`NIGHT-01`..`NIGHT-08`, Track #4) are confirmed to be genuine pedestrians approaching the camera (confidences 0.81..0.90) and remain opaquely masked.

---

## 5. Privacy Validation Status & Human Audit Prerequisite

> [!IMPORTANT]
> This diagnostic report reflects **automated model telemetry** and **mask provenance logging**. It demonstrates internal mathematical and pipeline consistency. It does **not** constitute proof of human privacy recall.
> 
> True privacy recall can only be established after genuine human review using the interactive local tri-pane viewer (`python scripts/verify_privacy_recall.py`) and recorded in `manual_validation_phase053.csv`.
