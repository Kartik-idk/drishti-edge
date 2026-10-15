# Drishti-Edge (दृष्टि-Edge)
**Privacy-Preserving Urban Safety Auditing Prototype**  
*Developed for the Seva First Innovation Challenge*

---

## 1. Project Overview

**Drishti-Edge** is an edge-native, computer-vision-based safety auditing system designed to assess urban public spaces for safety factors—such as pedestrian infrastructure conditions, physical hazards, crowd dynamics, and urban vulnerability indicators—while strictly preserving citizen privacy.

Rather than streaming high-resolution surveillance footage to centralized cloud infrastructure, Drishti-Edge is architected to perform local edge inference, discarding raw video frames immediately after processing and retaining only structured, non-identifiable audit telemetry.

---

## 2. Problem Statement

Modern urban safety and infrastructure monitoring faces two persistent challenges:

1. **Manual Audits Are Scalability Bottlenecks**: Traditional municipal safety audits rely on periodic, manual physical inspections. These surveys are resource-heavy, subjective, infrequent, and struggle to provide timely insights across large urban sectors.
2. **Surveillance Introduces Severe Privacy Hazards**: Conventional smart-city surveillance setups often stream and store raw CCTV video containing Personally Identifiable Information (PII)—including pedestrian faces, vehicle license plates, and movement habits. This creates major regulatory, ethical, and cybersecurity vulnerabilities.

Urban authorities require objective, frequent infrastructure safety audits without infringing on civic privacy or deploying costly centralized video transmission pipelines.

---

## 3. Proposed Privacy-Preserving Solution

Drishti-Edge resolves this conflict by implementing **Privacy-by-Design** principles at the capture edge:

* **Edge-Native Inference**: Video frames are analyzed locally on edge hardware. High-bandwidth raw video never leaves the edge device.
* **Ephemeral Frame Lifecycles**: Video frames are processed in volatile memory and immediately discarded. No persistent storage of raw visual footage occurs during standard auditing operations.
* **Automated Redaction & Anonymization**: If visual validation is required for specific audit findings, localized redaction (face and license-plate blurring) is applied before any image artifact is generated.
* **Structured, Zero-PII Telemetry**: The system extracts only aggregated, non-sensitive audit metrics (e.g., pedestrian count density, spatial hazard classifications, lighting conditions) and records them into local SQLite tables.

---

## 4. Technology Stack

* **Programming Language**: Python (3.10+)
* **Computer Vision & Video Processing**: OpenCV (`opencv-python>=4.8.0`)
* **Object Detection & Spatial Analysis**: Ultralytics YOLO (`ultralytics>=8.0.0`)
* **Local Audit Data Storage**: SQLite (`sqlite3` standard library)
* **Auditor Dashboard & Visualization**: Streamlit (`streamlit>=1.30.0`)
* **Environment Configuration**: `python-dotenv`

---

## 5. Current Project Status: Phase 0.6 Working Prototype

* **Status**: **Phase 0.6 Prototype Release: Local Intelligence Dashboard, Spatial Zone Mapping & ZVI Integration**
* **Delivered Capabilities**:
  - **Hardened Consolidated Privacy Engine**: Canonical 1280×720 pipeline, 100% solid opaque redaction, class-aware vehicle/pedestrian fallbacks, 3-frame direct temporal hold tracking, and fail-closed crash gating.
  - **Local Intelligence Dashboard (`app.py`)**: Local-only Streamlit auditor interface bound to `127.0.0.1` with multi-view preview (Redacted, Diagnostic Provenance, 3×3 Spatial Grid, and Transient Local Source Reference).
  - **Scene Spatial Zone Mapping (`scripts/spatial_zone_mapper.py`)**: 3×3 canonical video-scene grid extracting localized environmental lighting deficits ($L$), pedestrian counts, vehicle distributions, and mobility exposure ($M$).
  - **Zone Vulnerability Index (ZVI)**: Deterministic heuristic formula $\text{ZVI} = 100 \cdot (0.50L + 0.35I + 0.15M)$ with transparent missing-component gating.
* **Disclaimer**: Drishti-Edge is an edge-native experimental prototype for municipal comparative safety auditing. It does **not** claim production-grade safety certification, automated road defect recognition, or statistically calibrated crime prediction.

---

## 6. Development Stages

1. **Stage 1 (Completed)**: Repository architecture, Git hygiene, and baseline environment setup.
2. **Stage 2 (Completed)**: Video ingestion pipeline and baseline YOLO inference benchmarking.
3. **Stage 3 (Completed)**: Multi-model privacy evaluation (YuNet ONNX faces, YOLOv8n license plates).
4. **Stage 4 (Completed)**: Privacy engine consolidation, 1280×720 canonical scaling, and fail-closed output gating.
5. **Stage 5 (Completed - Phase 0.5.4)**: Class-aware 2W/3W fallbacks, refined stripe filtering, temporal hold isolation, and 59-test verification suite.
6. **Stage 6 (Phase 0.6 - Current)**: Streamlit auditor dashboard, 3×3 scene spatial zone mapping, ZVI calculation engine, and 72 automated unit tests.

---

## 7. Local Environment Setup

### Prerequisites
* Python 3.10 or higher
* `pip` package manager
* Git

### Step-by-Step Setup

1. **Clone the repository** (once published):
   ```bash
   git clone <repository-url>
   cd drishti-Edge
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows (Command Prompt):
   # py -m venv .venv
   # .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment settings**:
   Copy the example environment configuration to `.env`:
   ```bash
   cp .env.example .env
   ```
   *Note: `.env` is ignored by Git and should hold local configurations.*

---

## 8. Local Test Videos & Data Privacy Guidelines

To run local benchmarks or test the vision pipeline:

1. **Directory Location**: Place local test video clips inside the `data/videos/` directory.
2. **Version Control Notice**:
   * **DO NOT** commit raw street-scene videos, captured CCTV footage, or high-resolution test clips to GitHub.
   * All common video formats (`*.mp4`, `*.avi`, `*.mov`, `*.mkv`, `*.webm`, etc.) and the `data/videos/` directory contents are explicitly excluded by `.gitignore`.
3. **Synthetic / Permissible Data**:
   * When conducting local testing, prioritize open-source urban benchmark datasets or synthetic footage where subjects have consented or where identifying features are absent.

---

## 9. Launching the Local Intelligence Dashboard (Phase 0.6)

Drishti-Edge Phase 0.6 introduces the local-only Streamlit auditor interface for interactive urban safety and privacy auditing.

### Local Execution Command
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Launch Streamlit dashboard on local interface only
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

### Video Selection and Processing Workflow
1. **Video Discovery**: The dashboard automatically discovers all `.mp4`, `.avi`, `.mov`, and `.mkv` files placed in `data/videos/`.
2. **Auditor Controls (Sidebar)**:
   - **Video Selection**: Choose a video from the local dropdown (e.g., `daytime_india.mp4`).
   - **Frame Budget**: Choose how many frames to audit (e.g., 30 to 300 frames) for rapid interactive auditing.
   - **Frame Stride**: Set stride (e.g., 1 for consecutive frames, 2-5 for sampling).
   - **Manual Infrastructure Assessment ($I$)**: If conducting a complete ZVI calculation, toggle the manual infrastructure assessment switch and set the observed physical hazard factor ($I \in [0.0, 1.0]$).
3. **Execution**: Click **"Run Audit Pipeline"**. The engine processes frames through the canonical $1280 \times 720$ privacy pipeline in volatile RAM, continuously updating telemetry.
4. **Interactive Inspection**:
   - **Redacted Preview**: Inspect the solid-anonymized output frame.
   - **Diagnostic View**: Review bounding box placement (YuNet faces, plate YOLO, COCO VRUs, class-aware fallbacks).
   - **Spatial Heatmap**: Examine the 3×3 video-scene grid colored by composite vulnerability.
   - **Local Auditor Reference**: Compare with transient unredacted source frame (strictly in-memory, never persisted to disk).

---

## 10. Video-Scene Spatial Zone Mapping

### Architecture & Coordinate System
- Canonical resolution: $1280 \times 720$.
- Grid layout: 3 rows × 3 columns (9 equal zones of $426.67 \times 240$ pixels).
- **CRITICAL DISTINCTION**: This is a **VIDEO-SCENE SPATIAL MAP**, representing screen-space regions (Camera Top-Left, Camera Bottom-Center, etc.). It is **NOT** a real-world GPS or geographic coordinate map. Real-world mapping requires calibrated geospatial ground-plane homography.

### Zone Mapping Rules
1. **Deterministic Coordinate Assignment**: Bounding box center coordinate $((x_1+x_2)/2, (y_1+y_2)/2)$ is mapped into zone column $\lfloor x / (W/3) \rfloor$ and row $\lfloor y / (H/3) \rfloor$.
2. **Boundary Crossing**: Exact box-to-cell intersection area fractions are calculated to ensure consistent spatial attribution for objects spanning zone borders.
3. **Temporal Aggregation**: Per-zone metrics (lighting deficit, pedestrian count, vehicle distribution) are accumulated across all processed frames to derive stable scene-level risk profiles.

---

## 11. Zone Vulnerability Index (ZVI) Specification

### Mathematical Definition
$$\text{ZVI} = 100 \cdot (0.50L + 0.35I + 0.15M)$$

Where all component factors are strictly normalized to $[0.0, 1.0]$:

1. **$L$ (Environmental Lighting Deficit, 50% Weight - AUTOMATED)**:
   - Derived directly from HSV Value (brightness) channel: $L = 1.0 - \frac{\text{mean}(V)}{255.0}$.
   - Evaluated across the zone or scene. High deficit corresponds to darkness/under-illumination.
2. **$I$ (Infrastructure Defect Factor, 35% Weight - MANUAL AUDITOR INPUT)**:
   - **PROTOTYPE LIMITATION & HONEST DISCLOSURE**: Drishti-Edge does **not** currently include an automated pothole, damaged pavement, or broken streetlight detector.
   - Component $I$ is **explicitly flagged as MANUAL**.
   - If unassessed by the human auditor, $I$ is marked `None`/`UNASSESSED`.
   - **Zero Substitution Policy**: The system **never** silently substitutes 0.0 for $I$. If $I$ is unassessed, ZVI reports status `INCOMPLETE` with clear explanation of missing evidence.
3. **$M$ (Mobility Exposure Factor, 15% Weight - AUTOMATED)**:
   - Computed from real detected Vulnerable Road Users (VRUs: pedestrians, 2W/3W riders) per zone relative to saturation capacity: $M = \min(1.0, \frac{\text{VRU Count}}{5.0})$.
   - Avoids treating high vehicle counts as danger; isolates vulnerable road user presence in dark/poorly illuminated zones.

### Measured Metrics vs. Heuristic Indicators
- **Measured Metrics (Direct CV Inference)**: Processing throughput (FPS), face and plate detection counts, mean frame luminance ($V$-channel), fail-closed frame counts, YOLO person/vehicle bounding boxes.
- **Heuristic Indicators (Decision-Support Estimates)**: ZVI score, spatial vulnerability heatmap, mobility exposure score. These are heuristic screening indices designed to guide municipal inspectors, **not** statistically calibrated crime or accident risk models.

---

## 12. Privacy and Data-Handling Guarantees

- **Local-Only Binding**: Streamlit server configured for `127.0.0.1` (no external host exposure).
- **Zero Cloud / External Transmission**: All inference runs locally via OpenCV ONNX runtime and Ultralytics YOLO.
- **Transient Memory Lifecycle**: Source frames are decoded into memory, redacted, and discarded.
- **No Unredacted Persistence**: Zero disk persistence of raw frames, unredacted crops, or identifiable debug screenshots.
- **Fail-Closed Integrity**: If a privacy redaction stage throws an exception, the pipeline fails closed, substituting a solid black frame ($1280 \times 720$) with a warning watermark to prevent any leak of unredacted data.
- **Source Protection**: Input source videos in `data/videos/` are read-only and never overwritten.

---

## 13. Running Automated Tests

Execute the comprehensive test suite (including all 59 baseline tests and 13 Phase 0.6 tests):
```bash
./.venv/bin/python -m unittest discover tests
```

Expected output:
```
Ran 72 tests in ~0.6s
OK
```

---

## 14. Phase 0.6 Manual Smoke-Test Checklist

Before releasing or demonstrating Phase 0.6, complete this checklist:

1. **Environment Verification**: Ensure `.venv` is active and dependencies from `requirements.txt` are installed.
2. **Automated Tests**: Run `./.venv/bin/python -m unittest discover tests` $\rightarrow$ 72 tests pass.
3. **Launch Application**: Run `streamlit run app.py`. Confirm dashboard opens on `http://127.0.0.1:8501`.
4. **Video Ingestion**: Select `daytime_india.mp4`. Confirm metadata displays (dimensions, FPS, duration, frame count).
5. **Partial Execution**: Set Frame Budget to 30 frames, stride 1. Click "Run Audit Pipeline". Confirm progress bar advances smoothly and completes.
6. **Privacy Verification**:
   - Verify **Redacted Preview** displays opaque anonymization over faces and license plates.
   - Check **Diagnostic Provenance** tab confirms bounding box alignment.
   - Confirm **Local Source Reference** warns that raw display is ephemeral in memory.
7. **Spatial Zone Heatmap**: Confirm 3×3 grid renders with color-coded vulnerability badges (Top-Left to Bottom-Right).
8. **ZVI Incomplete Status**: With manual assessment unchecked, verify ZVI card shows `INCOMPLETE` and explains missing $I$ input without zero substitution.
9. **ZVI Computed Status**: Enable manual assessment, set $I = 0.40$, re-run or inspect calculation. Confirm ZVI computes accurately with weighted score.
10. **Fail-Closed Verification**: Confirm error count displays 0 for valid files and fail-closed mechanism triggers safely on corrupt inputs.
