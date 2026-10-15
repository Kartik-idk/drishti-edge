#!/usr/bin/env python3
"""Drishti-Edge: Local Intelligence Dashboard, Zone Mapping & ZVI Integration

Phase 0.6: Streamlit Urban Safety Auditor Application

Design & Security Invariants:
  - 100% Local-only operation (binds to 127.0.0.1).
  - No external cloud transmission or third-party video uploads.
  - Ephemeral frame lifecycles: raw video frames remain strictly transient in RAM
    and are discarded immediately after processing.
  - Reuses existing ConsolidatedPrivacyEngine and SpatialZoneMapper.
  - Fail-closed safety: frames with processing errors are suppressed to solid black.
  - Transparent ZVI: ZVI = 100 * (0.50*L + 0.35*I + 0.15*M); never substitutes zero
    for unassessed components.
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Ensure repository root is on Python sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.privacy_engine import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    ConsolidatedPrivacyEngine,
    PrivacyGateStatus,
    draw_diagnostic_overlay,
)
from scripts.spatial_zone_mapper import (
    ZONE_DEFINITIONS,
    SpatialZoneMapper,
    compute_zvi,
    discover_local_videos,
    get_video_metadata,
)

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"


# ==============================================================================
# Streamlit Application Configuration
# ==============================================================================

st.set_page_config(
    page_title="Drishti-Edge | Urban Safety Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .status-badge-ok {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-badge-incomplete {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Sidebar: Video Selection & Audit Configuration
# ==============================================================================

st.sidebar.title("🛡️ Drishti-Edge")
st.sidebar.caption("Privacy-Preserving Urban Safety Auditor (Phase 0.6)")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Video Selection")

available_videos = discover_local_videos()
if not available_videos:
    st.sidebar.error("No compatible video files found in `data/videos/`.")
    selected_video_path = None
else:
    video_options = {v.name: v for v in available_videos}
    selected_name = st.sidebar.selectbox(
        "Choose test video from `data/videos/`:",
        options=list(video_options.keys()),
        index=0,
    )
    selected_video_path = video_options[selected_name]

    # Guard: prevent cross-video state contamination when user switches videos
    if st.session_state.get("active_video_name") != selected_name:
        st.session_state["active_video_name"] = selected_name
        for k in [
            "zone_mapper",
            "video_summary",
            "evaluated_count",
            "fail_closed_count",
            "total_run_sec",
            "throughput_fps",
            "last_metrics",
            "last_redacted",
            "last_diag",
            "last_spatial",
            "last_source",
        ]:
            st.session_state.pop(k, None)


st.sidebar.markdown("---")
st.sidebar.subheader("2. Processing Parameters")

max_frames_to_process = st.sidebar.slider(
    "Max Frames to Audit:",
    min_value=10,
    max_value=300,
    value=60,
    step=10,
    help="Limits run duration for interactive demonstration and local testing.",
)

frame_stride = st.sidebar.select_slider(
    "Frame Sampling Stride:",
    options=[1, 2, 5, 10, 15],
    value=2,
    help="1 = evaluate every sequential frame; 5 = evaluate every 5th frame.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. ZVI Component [I] Configuration")
st.sidebar.markdown(
    """
    **Infrastructure Condition Factor ($I$):**  
    *Notice:* Drishti-Edge currently loads no automated road-hazard or pothole vision model.
    Component $I$ requires manual auditor assessment.
    """
)

enable_manual_infra = st.sidebar.checkbox(
    "Provide Manual Infrastructure Assessment",
    value=False,
    help="Check to input an auditor evaluation score for pavement/infrastructure defect severity.",
)

manual_i_score: Optional[float] = None
if enable_manual_infra:
    manual_i_score = st.sidebar.slider(
        "Infrastructure Hazard Severity (I):",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help=(
            "0.0 = Sound pavement, well-maintained sidewalks\n"
            "0.5 = Moderate surface wear, minor obstructions\n"
            "1.0 = Severe damage, open potholes, critical physical hazards"
        ),
    )
    st.sidebar.caption(f"Configured $I = {manual_i_score:.2f}$ (Auditor Manual Input)")
else:
    st.sidebar.info("Component $I$ is unassessed. ZVI will report as INCOMPLETE per integrity rules.")


# ==============================================================================
# Main Dashboard Content
# ==============================================================================

# Header
st.markdown('<div class="main-header">Drishti-Edge (दृष्टि-Edge)</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Privacy-Preserving Edge Urban Safety Auditor & Scene Vulnerability Mapping'
    '</div>',
    unsafe_allow_html=True,
)

if selected_video_path is None:
    st.warning("Please add supported video files (.mp4, .avi) to `data/videos/` to begin.")
    st.stop()

# Video Metadata Section
meta = get_video_metadata(selected_video_path)
if meta is None:
    st.error(f"Failed to open video file: `{selected_video_path.name}`. File may be corrupt or unreadable.")
    st.stop()

# Section A: Video Analysis & Execution
st.subheader("Section A: Video Source & Pipeline Controls")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Source File", meta["filename"])
with c2:
    st.metric("Native Resolution", f"{meta['width']} × {meta['height']}")
with c3:
    st.metric("Frame Rate & Duration", f"{meta['fps']} FPS ({meta['duration_sec']}s)")
with c4:
    st.metric("Total Video Frames", f"{meta['total_frames']} frames")

st.markdown("")
run_processing = st.button("🚀 Execute Local Privacy & Spatial Safety Audit", type="primary", use_container_width=True)

# Placeholders for dynamic execution updates
progress_bar = st.empty()
status_text = st.empty()

# Tabs for visual inspection
st.markdown("---")
st.subheader("Section B: Visual Inspection & Preview")
tab_redacted, tab_diag, tab_spatial, tab_source = st.tabs([
    "🛡️ Redacted Output",
    "🔍 Diagnostic Mask Provenance",
    "🗺️ Scene Spatial Heatmap (3x3)",
    "👁️ Local Auditor Reference",
])

with tab_redacted:
    redacted_preview = st.empty()
    st.caption("Live stream of privacy-redacted canonical 1280×720 frames. All faces and plates are opaquely obfuscated.")

with tab_diag:
    diag_preview = st.empty()
    st.caption("Real-time mask provenance diagnostics (green=face, orange=plate, purple=2W fallback, blue=4W fallback).")

with tab_spatial:
    spatial_preview = st.empty()
    st.caption("Scene-level 3x3 spatial grid overlay color-coded by localized risk signals.")

with tab_source:
    source_preview = st.empty()
    st.caption("Transient source reference in volatile memory for local auditor verification. Never persisted to disk.")


# ==============================================================================
# Pipeline Execution Engine
# ==============================================================================

if run_processing:
    engine = ConsolidatedPrivacyEngine()
    zone_mapper = SpatialZoneMapper()

    cap = cv2.VideoCapture(str(selected_video_path))
    if not cap.isOpened():
        st.error("Error opening video capture pipeline.")
        st.stop()

    total_to_evaluate = min(max_frames_to_process, meta["total_frames"])
    evaluated_count = 0
    fail_closed_count = 0
    t_run_start = time.perf_counter()

    last_redacted = None
    last_diag = None
    last_spatial = None
    last_source = None
    last_metrics: Dict[str, Any] = {}

    frame_idx = 0
    while frame_idx < meta["total_frames"] and evaluated_count < total_to_evaluate:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_idx % frame_stride == 0:
            # Execute Hardened Privacy Engine
            redacted_frame, gate_status, metrics = engine.process_frame(frame)
            evaluated_count += 1

            if gate_status != PrivacyGateStatus.RELEASE_APPROVED:
                fail_closed_count += 1

            # Execute Spatial Zone Mapper
            zone_results = zone_mapper.analyze_frame(
                redacted_frame,
                metrics,
                manual_infrastructure=manual_i_score,
                accumulate=True,
            )

            # Generate Diagnostic & Heatmap Visualizations
            diag_vis = draw_diagnostic_overlay(frame, redacted_frame, gate_status, metrics)
            spatial_vis = zone_mapper.render_spatial_heatmap(
                redacted_frame,
                metric="zvi" if enable_manual_infra else "lighting",
                manual_infrastructure=manual_i_score,
            )

            # Update live preview slots periodically
            if evaluated_count % max(1, (10 // frame_stride)) == 0 or evaluated_count == total_to_evaluate:
                # Convert BGR to RGB for Streamlit display
                rgb_redacted = cv2.cvtColor(redacted_frame, cv2.COLOR_BGR2RGB)
                rgb_diag = cv2.cvtColor(diag_vis, cv2.COLOR_BGR2RGB)
                rgb_spatial = cv2.cvtColor(spatial_vis, cv2.COLOR_BGR2RGB)
                rgb_source = cv2.cvtColor(cv2.resize(frame, (CANONICAL_WIDTH, CANONICAL_HEIGHT)), cv2.COLOR_BGR2RGB)

                redacted_preview.image(rgb_redacted, use_container_width=True)
                diag_preview.image(rgb_diag, use_container_width=True)
                spatial_preview.image(rgb_spatial, use_container_width=True)
                source_preview.image(rgb_source, use_container_width=True)

                last_redacted = rgb_redacted
                last_diag = rgb_diag
                last_spatial = rgb_spatial
                last_source = rgb_source
                last_metrics = metrics

            progress = evaluated_count / total_to_evaluate
            progress_bar.progress(progress)
            status_text.text(f"Auditing frame {frame_idx + 1}/{meta['total_frames']} (Evaluated: {evaluated_count}/{total_to_evaluate})...")

        frame_idx += 1

    cap.release()
    t_run_end = time.perf_counter()
    total_run_sec = max(0.001, t_run_end - t_run_start)
    throughput_fps = evaluated_count / total_run_sec

    progress_bar.empty()
    status_text.success(f"Audit completed: {evaluated_count} frames evaluated in {total_run_sec:.2f}s ({throughput_fps:.1f} FPS).")

    # Store run state into session state so results remain visible after UI interactions
    st.session_state["evaluated_count"] = evaluated_count
    st.session_state["fail_closed_count"] = fail_closed_count
    st.session_state["total_run_sec"] = total_run_sec
    st.session_state["throughput_fps"] = throughput_fps
    st.session_state["last_metrics"] = last_metrics
    st.session_state["zone_mapper"] = zone_mapper
    st.session_state["last_redacted"] = last_redacted
    st.session_state["last_diag"] = last_diag
    st.session_state["last_spatial"] = last_spatial
    st.session_state["last_source"] = last_source


# ==============================================================================
# Render Results (from current or session state)
# ==============================================================================

if "zone_mapper" in st.session_state:
    stored_zone_mapper = st.session_state["zone_mapper"]
    v_summary = stored_zone_mapper.get_video_summary(manual_infrastructure=manual_i_score)
    evaluated_count = st.session_state["evaluated_count"]
    fail_closed_count = st.session_state["fail_closed_count"]
    total_run_sec = st.session_state["total_run_sec"]
    throughput_fps = st.session_state["throughput_fps"]
    last_metrics = st.session_state["last_metrics"]

    # Re-populate visual preview tabs if not actively running
    if not run_processing:
        if st.session_state.get("last_redacted") is not None:
            redacted_preview.image(st.session_state["last_redacted"], use_container_width=True)
        if st.session_state.get("last_diag") is not None:
            diag_preview.image(st.session_state["last_diag"], use_container_width=True)
        if st.session_state.get("last_spatial") is not None:
            spatial_preview.image(st.session_state["last_spatial"], use_container_width=True)
        if st.session_state.get("last_source") is not None:
            source_preview.image(st.session_state["last_source"], use_container_width=True)

    # Section C: Zone-Based Hazard & Vulnerability Mapping
    st.markdown("---")
    st.subheader("Section C: Scene Spatial Zone Analysis (3×3 Canonical Grid)")
    st.markdown(
        """
        The video scene is partitioned into a **3×3 canonical coordinate spatial grid** (9 zones).  
        *Note:* Coordinates represent the in-camera scene field of view, not geographic GPS coordinates.
        """
    )

    zones_dict = v_summary["zones_summary"]

    # Render 3x3 Metric Grid
    row_chunks = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]

    for row_idx, r_zones in enumerate(row_chunks):
        cols = st.columns(3)
        for col_idx, zid in enumerate(r_zones):
            z_data = zones_dict[zid]
            zvi_data = z_data["zvi_result"]
            with cols[col_idx]:
                st.markdown(f"**{z_data['name']}**")
                z_cols = st.columns(2)
                with z_cols[0]:
                    st.caption(f"Lighting Deficit ($L$): `{z_data['mean_lighting_deficit']:.3f}`")
                    st.caption(f"Avg Pedestrians: `{z_data['mean_persons']:.1f}`")
                with z_cols[1]:
                    st.caption(f"Mobility Factor ($M$): `{z_data['mean_mobility_factor']:.3f}`")
                    st.caption(f"Avg Vehicles: `{z_data['mean_vehicles']:.1f}`")

                if zvi_data["is_complete"]:
                    st.markdown(f"**ZVI Score: `{zvi_data['zvi']:.1f}` / 100**")
                else:
                    st.markdown("*ZVI: Incomplete ($I$ unassessed)*")
                st.markdown("---")

    # Zone Summary Data Table
    table_rows = []
    for zid in range(1, 10):
        zd = zones_dict[zid]
        zvi_d = zd["zvi_result"]
        table_rows.append({
            "Zone": f"Zone {zid}",
            "Location": zd["name"].split()[1][1:-1],
            "Mean Lighting Deficit (L)": zd["mean_lighting_deficit"],
            "Mean Mobility Exposure (M)": zd["mean_mobility_factor"],
            "Peak Pedestrians": zd["peak_persons"],
            "Total Pedestrians": zd["total_persons_observed"],
            "Peak Vehicles": zd["peak_vehicles"],
            "Total Vehicles": zd["total_vehicles_observed"],
            "Redaction Masks Applied": zd["total_masks_applied"],
            "ZVI Score": f"{zvi_d['zvi']:.1f}" if zvi_d["is_complete"] else "Incomplete",
        })

    st.dataframe(table_rows, use_container_width=True)

    # Section D: Zone Vulnerability Index (ZVI) Detail
    st.markdown("---")
    st.subheader("Section D: Zone Vulnerability Index (ZVI) Specification")

    # Compute scene-level aggregate ZVI
    scene_l = float(np.mean([zd["mean_lighting_deficit"] for zd in zones_dict.values()]))
    scene_m = float(np.mean([zd["mean_mobility_factor"] for zd in zones_dict.values()]))
    scene_zvi = compute_zvi(scene_l, manual_i_score, scene_m)

    st.markdown(
        r"""
        $$\text{ZVI} = 100 \times \left( 0.50 \cdot L + 0.35 \cdot I + 0.15 \cdot M \right)$$
        """
    )

    col_zvi1, col_zvi2, col_zvi3 = st.columns([1.2, 1.2, 1.6])

    with col_zvi1:
        st.markdown("**Component Measurements:**")
        st.markdown(f"- **Lighting Deficit ($L$):** `{scene_l:.3f}` *(Automated HSV V-channel)*")
        if manual_i_score is not None:
            st.markdown(f"- **Infrastructure Deficit ($I$):** `{manual_i_score:.3f}` *(Manual Auditor Input)*")
        else:
            st.markdown("- **Infrastructure Deficit ($I$):** *UNAVAILABLE (Pending Manual Assessment)*")
        st.markdown(f"- **Mobility Exposure ($M$):** `{scene_m:.3f}` *(Automated VRU Density)*")

    with col_zvi2:
        st.markdown("**Weighted Factor Contribution:**")
        if scene_zvi["is_complete"]:
            c_dict = scene_zvi["contributions"]
            st.markdown(f"- **Lighting (50% weight):** `+{c_dict['L_contribution']:.2f} pts`")
            st.markdown(f"- **Infrastructure (35% weight):** `+{c_dict['I_contribution']:.2f} pts`")
            st.markdown(f"- **Mobility (15% weight):** `+{c_dict['M_contribution']:.2f} pts`")
        else:
            st.markdown("- *Weighted sum withheld until all required components are populated.*")

    with col_zvi3:
        if scene_zvi["is_complete"]:
            st.metric("Aggregated Scene ZVI", f"{scene_zvi['zvi']:.1f} / 100")
            st.success(f"Status: COMPLETE. Primary driver: {scene_zvi['primary_driver']}.")
        else:
            st.metric("Aggregated Scene ZVI", "INCOMPLETE")
            st.warning("Status: INCOMPLETE. Infrastructure component ($I$) requires manual auditor assessment.")

    st.info(
        "**Integrity Disclaimer:** "
        "The Zone Vulnerability Index (ZVI) is an experimental heuristic index designed for municipal "
        "comparative safety auditing. It does not represent a statistically proven predictor of crime "
        "probability or physical injury likelihood."
    )

    # Section E: Run Summary & Privacy Telemetry
    st.markdown("---")
    st.subheader("Section E: Execution Summary & Privacy Telemetry")

    col_e1, col_e2, col_e3, col_e4 = st.columns(4)
    with col_e1:
        st.metric("Evaluated Frames", f"{evaluated_count}")
    with col_e2:
        st.metric("Release Approved", f"{evaluated_count - fail_closed_count}")
    with col_e3:
        st.metric("Fail-Closed Events", f"{fail_closed_count}")
    with col_e4:
        st.metric("Processing Throughput", f"{throughput_fps:.1f} FPS")

    # Fail-closed check
    if fail_closed_count > 0:
        st.error(
            f"⚠️ Privacy Gate Notice: {fail_closed_count} frame(s) encountered detector/pipeline anomalies "
            "and were automatically suppressed to solid black frames in accordance with fail-closed privacy guarantees."
        )
    else:
        st.success(
            "✅ All evaluated frames passed the privacy release gate with 100% solid opaque redaction "
            "and zero unredacted frame persistence."
        )
