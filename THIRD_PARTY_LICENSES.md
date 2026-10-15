# Third-Party Licenses and Model Provenance

**Project:** Drishti-Edge  
**Challenge:** Seva First Innovation Challenge, Theme 5, Problem Statement 1  
**Phase:** 0.4 Privacy Engine Consolidation & Safety Validation  
**Date:** October 2026  

---

## 1. Compliance Policy & Prototype Boundaries

Drishti-Edge is an experimental edge-native privacy prototype designed to demonstrate real-time face and license plate redaction on urban surveillance video.

> [!IMPORTANT]
> **No Claim of Production Readiness or Commercial Clearance:**
> The models, software packages, and configurations evaluated in Phase 0.4 represent prototype engineering dependencies for feasibility benchmarking. No model or dependency in this repository is currently certified for commercial deployment or municipal production use.

### Distinct Licensing Dimensions:
In machine learning systems, rights must be assessed across three independent legal dimensions:
1. **Software Framework License:** Governs the runtime libraries executing the model (e.g., OpenCV, PyTorch, Ultralytics).
2. **Model Weight License:** Governs the distribution, modification, and execution of the trained neural network parameter tensors.
3. **Training Dataset Provenance & License:** Governs the underlying images, annotations, and terms under which training data was compiled. A permissive model weight license (such as MIT or Apache-2.0) does **not** automatically extinguish restrictive terms attached to the underlying training dataset.

---

## 2. Machine Learning Models & Checkpoint Provenance

### Model 1: OpenCV Zoo YuNet (Face Detector — Primary Active)

* **Role:** Primary face detector operating on canonical 1280×720 frames.
* **Checkpoint File:** `models/face_detection_yunet_2023mar.onnx`
* **File Size:** 232,589 bytes (~233 KB)
* **SHA-256 Checksum:** `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
* **Source Repository:** [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
* **Upstream Author:** Shiqi Yu ([libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train))
* **Runtime Framework License:** Apache License 2.0 (OpenCV C++ DNN execution).
* **Model-Weight License:** MIT License (Copyright © 2020 Shiqi Yu `<shiqi.yu@gmail.com>`).
* **Training Dataset Provenance:** Trained on the WIDER FACE benchmark dataset.
* **Training Dataset License:** Stated as **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)**.
* **Unresolved Rights Questions:** While the model ONNX file is distributed under an MIT license in OpenCV Zoo, the WIDER FACE training data carries CC BY-NC-ND restrictions. The legal interaction between non-commercial dataset terms and downstream model weight copyright remains subject to ongoing legal interpretation. An MIT model-file license cannot be assumed to resolve all questions concerning commercial deployment.
* **Intended Prototype Use & Restrictions:** Approved strictly for feasibility research and prototype validation in Phase 0.4. Transitioning to municipal production will require legal review or retraining on a verified commercially cleared face dataset.
* **Note on Artifacts:** All references to the unverified `face_detection_yunet_2026may.onnx` artifact have been removed from the active configuration.

---

### Model 2: YOLOv8n Base COCO (Vehicle Context & ROI Extractor)

* **Role:** Detects vehicles (car, motorcycle, bus, truck) on canonical 1280×720 frames to extract high-resolution crops for plate detection and calculate fallback masks.
* **Checkpoint File:** `yolov8n.pt`
* **File Size:** 6,549,796 bytes (~6.5 MB)
* **SHA-256 Checksum:** `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
* **Source Repository:** [Ultralytics GitHub Releases v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt)
* **Runtime Framework License:** GNU Affero General Public License v3.0 (AGPL-3.0) / Ultralytics Enterprise Commercial License.
* **Model-Weight License:** AGPL-3.0.
* **Training Dataset Provenance:** Microsoft COCO (Common Objects in Context) dataset.
* **Training Dataset License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Unresolved Rights Questions:** AGPL-3.0 contains network-use copyleft provisions. If software integrating Ultralytics is hosted on a server or edge device and provides telemetry over a network, the copyleft trigger may require disclosing the complete source code under AGPL-3.0.
* **Intended Prototype Use & Restrictions:** Used for prototype vehicle detection. Prior to commercial or municipal edge deployment, an appropriate licensing review is required. The team must either obtain an Ultralytics Enterprise Commercial License or export the model weights to standalone ONNX format for execution via OpenCV DNN or ONNX Runtime to decouple the runtime from AGPL-3.0 copyleft obligations.

---

### Model 3: YOLOv8n-plate (License Plate Detector — Experimental Prototype)

* **Role:** Detects license plates within vehicle crops selected by the context detector.
* **Checkpoint File:** `models/yolov8n_plate.pt`
* **File Size:** 6,248,291 bytes (~6.2 MB)
* **SHA-256 Checksum:** `2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822`
* **Source Repository:** [Hugging Face: Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection)
* **Runtime Framework License:** AGPL-3.0 / PyTorch.
* **Model-Weight License:** Declared as MIT License by uploader on Hugging Face.
* **Training Dataset Provenance:** Derived from the Keremberke / Roboflow `license-plate-object-detection` dataset (8,832 images). Stated as CC BY 4.0. The training set is composed predominantly of Western passenger vehicles.
* **Unresolved Rights Questions:** 
  1. The underlying training images and annotations have not undergone an independent chain-of-title or copyright verification audit.
  2. The model has **not** been trained or verified on Indian license plates (e.g. High-Security Registration Plates (HSRP), two-wheeler rear mudguard plates, auto-rickshaw plates, regional font variations, or yellow/black commercial plates).
* **Intended Prototype Use & Restrictions:** Strictly an **experimental prototype dependency** for pipeline architecture and batching evaluation. It is **not** an approved production model. It must be replaced with an audited, domain-specific Indian license plate detector prior to field deployment.

---

### Model 4: OpenCV Zoo LPD-YuNet (Retired Reference Artifact)

* **Status:** **RETIRED FROM ACTIVE RUNTIME** (Retained strictly for reproducibility and audit history).
* **Checkpoint File:** `models/license_plate_detection_lpd_yunet_2023mar.onnx`
* **File Size:** 4,146,213 bytes (~4.1 MB)
* **SHA-256 Checksum:** `6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7`
* **Source Repository:** [OpenCV Zoo LPD-YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/license_plate_detection_yunet)
* **Contributor:** Dong Xu (watrix.ai)
* **Model-Weight License:** Apache License 2.0.
* **Training Dataset Provenance:** Exclusively trained on Chinese vehicle license plates.
* **Retirement Reason:** Excessive anchor-prior generation latency on CPU (~789 ms) and failure to generalize to Indian license plates.

---

## 3. Core Software Libraries & Packages

| Package | Installed Version | Primary Role | License | Deployment Considerations |
| :--- | :--- | :--- | :--- | :--- |
| `opencv-python` | `5.0.0.93` | Video decoding, frame standardization, YuNet DNN inference, solid mask rendering | Apache-2.0 | Permissive open-source license. Suitable for edge deployment. |
| `ultralytics` | `8.4.175` | YOLOv8 inference runtime (MPS GPU accelerated) | AGPL-3.0 / Enterprise | Network copyleft obligations apply. Requires licensing review or ONNX runtime decoupling. |
| `torch` | `2.14.1` | Neural network backend, Apple Silicon MPS execution | BSD-3-Clause | Permissive license. Fully compliant. |
| `numpy` | `2.x` | Coordinate transformations, array slicing, mask buffers | BSD-3-Clause | Permissive license. Fully compliant. |
| `python-dotenv` | `1.0.1` | Environment variable management | BSD-3-Clause | Permissive license. Fully compliant. |
| `streamlit` | `1.65.0` | Municipal analytics dashboard (downstream consumer) | Apache-2.0 | Permissive open-source license. Not yet integrated into privacy engine output. |

---

## 4. Legal Disclaimers & Next-Phase Roadmap

1. **No Legal Opinion:** The licensing notes in this document reflect technical due diligence and do not constitute formal legal counsel.
2. **Pre-Production Audit Requirement:** Before Drishti-Edge can be deployed in a municipal production pilot, a comprehensive intellectual property and licensing audit must be conducted.
3. **Training Data Remediation:** Next development phases must prioritize sourcing or training models on commercially cleared datasets specifically tailored to Indian traffic dynamics and vehicle classes.
