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
* **Computer Vision & Video Processing**: OpenCV (`opencv-python-headless`)
* **Object Detection & Spatial Analysis**: Ultralytics YOLO (`ultralytics`)
* **Local Audit Data Storage**: SQLite (`sqlite3` standard library)
* **Auditor Dashboard & Visualization**: Streamlit (`streamlit`)
* **Environment Configuration**: `python-dotenv`

---

## 5. Current Project Status

* **Status**: **Initial Development Stage (Step 1: Workspace Preparation & Version Control Readiness)**
* **Implementation Evidence**: The repository currently contains version control rules (`.gitignore`), configuration templates (`.env.example`), dependency definitions (`requirements.txt`), and directory structure placeholders (`data/`, `models/`).
* **Disclaimer**: Core vision pipelines, edge benchmarks, anonymization models, and the Streamlit dashboard are pending implementation in subsequent stages. No benchmark scores, accuracy percentages, municipal contracts, or operational cost savings are claimed or validated at this stage.

---

## 6. Development Stages

1. **Stage 1 (Current)**: Repository architecture, Git ignore configuration, safety checks, and project documentation.
2. **Stage 2**: Video ingestion pipeline, frame extraction, and baseline Ultralytics YOLO inference benchmarking.
3. **Stage 3**: Privacy-preservation engine (real-time face/license plate obfuscation and in-memory frame lifecycle enforcement).
4. **Stage 4**: SQLite persistence layer for structured audit logs, metrics, and incident timestamps.
5. **Stage 5**: Streamlit auditor interface for reviewing aggregated safety scores, temporal trends, and inspection logs.
6. **Stage 6**: Edge deployment packaging, performance profiling, and challenge demonstration preparation.

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
