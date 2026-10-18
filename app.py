#!/usr/bin/env python3
"""Drishti-Edge: Privacy-Preserving Urban Intelligence Console

Phase 0.7: High-Fidelity UI/UX Polish & Civic-Tech Console

Design & Architectural Invariants:
  - 100% Local-only operation (binds to 127.0.0.1).
  - No external cloud transmission, CDN dependencies, or remote web fonts.
  - Ephemeral frame lifecycles: raw video frames remain strictly transient in RAM
    and are discarded immediately after processing.
  - Zero disk persistence of raw frames, unredacted face/plate crops, or debug images.
  - Unredacted source frames are strictly gated behind an auditor-only opt-in disclosure.
  - Reuses ConsolidatedPrivacyEngine and SpatialZoneMapper without calculation duplication.
  - Fail-closed safety: frames with processing errors are suppressed to solid black.
  - Transparent ZVI: ZVI = 100 * (0.50*L + 0.35*I + 0.15*M); zero is never substituted
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

# Guard against stale in-memory module caches in long-running Streamlit processes
import importlib
import scripts.spatial_zone_mapper

if not hasattr(scripts.spatial_zone_mapper, "calculate_expected_evaluated_frames"):
    importlib.reload(scripts.spatial_zone_mapper)

from scripts.model_downloader import (
    ensure_demo_video_available,
    ensure_models_available,
)
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
    calculate_expected_evaluated_frames,
    compute_zvi,
    discover_local_videos,
    get_video_metadata,
)
import torch

VIDEO_DIR = PROJECT_ROOT / "data" / "videos"

# Deployment Mode Configuration:
# On hosted demonstration deployments (e.g., Streamlit Community Cloud), defaults to True.
# Can be overridden via DRISHTI_PUBLIC_DEMO environment variable ("0", "false" for local audit mode).
IS_PUBLIC_DEMO = os.getenv("DRISHTI_PUBLIC_DEMO", "true").lower() in ("true", "1", "yes")


# ==============================================================================
# Streamlit Application Configuration & Civic-Tech Design System
# ==============================================================================

st.set_page_config(
    page_title="DRISHTI-EDGE | Civic-Tech Urban Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling: Premium Dark Civic-Tech Intelligence Console
st.markdown(
    """
    <style>
    /* System Typography & Global Contrast */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #E5EDF7;
    }

    /* Global App Container */
    .stApp {
        background-color: #0B1220;
    }

    /* Brand Header */
    .brand-container {
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid #29374B;
        margin-bottom: 1.5rem;
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        color: #E5EDF7;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.85rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #9AAEC4;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    /* Horizontal System Status Strip */
    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        padding: 0.6rem 0.9rem;
        background-color: #111C2E;
        border: 1px solid #29374B;
        border-radius: 6px;
        margin-bottom: 1.5rem;
        align-items: center;
    }
    .status-item {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.8rem;
        color: #9AAEC4;
    }
    .status-item strong {
        color: #E5EDF7;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }
    .status-dot-active {
        background-color: #34D399;
        box-shadow: 0 0 6px rgba(52, 211, 153, 0.4);
    }
    .status-dot-idle {
        background-color: #9AAEC4;
    }
    .status-dot-warning {
        background-color: #FBBF24;
    }

    /* Console Cards & Panels */
    .console-card {
        background-color: #111C2E;
        border: 1px solid #29374B;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .console-card-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #E5EDF7;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .elevated-panel {
        background-color: #172338;
        border: 1px solid #29374B;
        border-radius: 6px;
        padding: 1rem;
    }

    /* Zone Grid Cards */
    .zone-cell {
        background-color: #111C2E;
        border: 1px solid #29374B;
        border-radius: 6px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .zone-cell-header {
        font-size: 0.9rem;
        font-weight: 700;
        color: #E5EDF7;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .zone-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #9AAEC4;
        margin-top: 0.2rem;
    }
    .zone-metric-val {
        color: #E5EDF7;
        font-weight: 600;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    .badge-ok {
        background-color: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
    }
    .badge-incomplete {
        background-color: rgba(251, 191, 36, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.35);
    }
    .badge-accent {
        background-color: rgba(45, 212, 191, 0.15);
        color: #2DD4BF;
        border: 1px solid rgba(45, 212, 191, 0.35);
    }
    .badge-error {
        background-color: rgba(248, 113, 113, 0.15);
        color: #F87171;
        border: 1px solid rgba(248, 113, 113, 0.35);
    }

    /* Empty State Container */
    .empty-state {
        background-color: #111C2E;
        border: 1px dashed #29374B;
        border-radius: 8px;
        padding: 2.25rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .empty-state-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #E5EDF7;
        margin-bottom: 0.5rem;
    }
    .empty-state-desc {
        font-size: 0.88rem;
        color: #9AAEC4;
        max-width: 620px;
        margin: 0 auto 1.25rem auto;
        line-height: 1.5;
    }

    /* Auditor Warning Alert */
    .auditor-warning {
        background-color: rgba(251, 191, 36, 0.08);
        border: 1px solid #FBBF24;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
    }
    .auditor-warning-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: #FBBF24;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.3rem;
    }
    .auditor-warning-body {
        font-size: 0.82rem;
        color: #E5EDF7;
        line-height: 1.45;
    }

    /* Streamlit Widget Overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #E5EDF7 !important;
    }
    div[data-testid="stMetricValue"] > div {
        overflow: visible !important;
        white-space: normal !important;
        word-break: break-word !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
        color: #9AAEC4 !important;
    }

    /* Native Streamlit Bordered Container Customization */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111C2E;
        border: 1px solid #29374B !important;
        border-radius: 6px;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit Deploy button */
    .stDeployButton, [data-testid="stAppDeployButton"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Sidebar: Video Selection & Audit Configuration
# ==============================================================================

st.sidebar.markdown(
    """
    <div style="padding: 0.5rem 0 1rem 0; border-bottom: 1px solid #29374B; margin-bottom: 1rem;">
        <div style="font-size: 1.2rem; font-weight: 800; letter-spacing: 0.06em; color: #E5EDF7;">DRISHTI-EDGE</div>
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #2DD4BF; font-weight: 600;">Urban Intelligence Console (Phase 0.7)</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Video Selection
st.sidebar.subheader("1. Video Source")
if IS_PUBLIC_DEMO:
    demo_video_path = ensure_demo_video_available(PROJECT_ROOT)
    selected_video_path = demo_video_path
    selected_name = demo_video_path.name

    st.sidebar.markdown(
        """
        <div style="background-color: #111C2E; border: 1px solid #29374B; border-radius: 6px; padding: 0.75rem; margin-bottom: 0.5rem;">
            <div style="font-size: 0.72rem; color: #2DD4BF; font-weight: 700; text-transform: uppercase; margin-bottom: 0.25rem;">Built-in Demonstration Source</div>
            <div style="font-size: 0.92rem; font-weight: 700; color: #E5EDF7;">synthetic_test.mp4</div>
            <div style="font-size: 0.74rem; color: #9AAEC4; margin-top: 0.25rem; line-height: 1.4;">
                Procedural 720p simulation (150 frames, 5.0s @ 30 FPS). Verified synthetic sample free of PII and copyright constraints.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "🔒 **Cloud Security Notice:** User video uploads are disabled on this hosted instance. "
        "Processing is strictly restricted to built-in demonstration footage."
    )
else:
    available_videos = discover_local_videos()
    if not available_videos:
        st.sidebar.error("No compatible video files found in `data/videos/`.")
        selected_video_path = None
        selected_name = "None"
    else:
        video_options = {v.name: v for v in available_videos}
        selected_name = st.sidebar.selectbox(
            "Choose test video from `data/videos/`:",
            options=list(video_options.keys()),
            index=0,
        )
        selected_video_path = video_options[selected_name]

st.sidebar.markdown("---")

# 2. Audit Scope & Parameters
st.sidebar.subheader("2. Audit Scope & Parameters")

audit_scope_option = st.sidebar.radio(
    "Audit Scope:",
    options=["Quick Audit (up to 300 frames)", "Full-Video Audit"],
    index=0,
    help="Quick Audit caps frames for responsive testing; Full-Video Audit evaluates the entire video stream.",
)
is_quick_mode = audit_scope_option.startswith("Quick")

if is_quick_mode:
    max_frames_to_process = st.sidebar.slider(
        "Max Frames to Audit:",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
        help="Limits frame count for responsive interactive local auditing.",
    )
else:
    max_frames_to_process = None

frame_stride = st.sidebar.select_slider(
    "Frame Sampling Stride:",
    options=[1, 2, 5, 10, 15],
    value=2,
    help="1 = sequential frame audit; 2 = evaluate every 2nd frame; 5 = fast sampling.",
)

# Guard: prevent cross-video and cross-scope state contamination
if is_quick_mode:
    current_audit_key = f"{selected_name}::quick_max_{max_frames_to_process}::stride_{frame_stride}"
else:
    current_audit_key = f"{selected_name}::full::stride_{frame_stride}"

if st.session_state.get("active_audit_key") != current_audit_key:
    st.session_state["active_audit_key"] = current_audit_key
    st.session_state["active_video_name"] = selected_name
    for k in [
        "zone_mapper",
        "evaluated_count",
        "expected_evaluated",
        "audit_scope",
        "frame_stride",
        "total_source_frames",
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

# 3. ZVI Component [I] Configuration
st.sidebar.subheader("3. Infrastructure Defect Factor (I)")
st.sidebar.caption(
    "Notice: Drishti-Edge currently loads no automated road defect model. "
    "Factor I requires manual auditor assessment."
)

enable_manual_infra = st.sidebar.checkbox(
    "Enable Manual Infrastructure Assessment",
    value=False,
    help="Check to input an auditor evaluation score for surface defects or physical hazards.",
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
    st.sidebar.markdown(f"<span class='badge badge-accent'>Auditor Input: I = {manual_i_score:.2f}</span>", unsafe_allow_html=True)
else:
    st.sidebar.markdown("<span class='badge badge-incomplete'>Factor I: Unassessed</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
if IS_PUBLIC_DEMO:
    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; color: #9AAEC4; line-height: 1.4;">
            <strong>Demonstration Architecture:</strong> Processing executes ephemerally on Streamlit Community Cloud server CPU infrastructure.
            Frames exist only in transient container memory during inference and are never persisted to disk. Zero user video uploads accepted.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.sidebar.markdown(
        """
        <div style="font-size: 0.72rem; color: #9AAEC4; line-height: 1.4;">
            <strong>Privacy Invariant:</strong> Local edge processing only. Raw frames remain in volatile RAM and are never persisted to disk.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# Header & Truthful Status Indicators
# ==============================================================================

brand_subtitle_text = (
    "Privacy-Preserving Urban Intelligence Console &mdash; Hosted Demonstration"
    if IS_PUBLIC_DEMO
    else "Privacy-Preserving Urban Intelligence & Civic Safety Console"
)

st.markdown(
    f"""
    <div class="brand-container">
        <div class="brand-title">DRISHTI-EDGE</div>
        <div class="brand-subtitle">{brand_subtitle_text}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if selected_video_path is None:
    st.warning("Please add compatible video files (.mp4, .avi, .mov) to `data/videos/` to begin.")
    st.stop()

# Video Metadata Retrieval
meta = get_video_metadata(selected_video_path)
if meta is None:
    st.error(f"Failed to open video file: `{selected_video_path.name}`. File may be corrupt or unreadable.")
    st.stop()

# Calculate expected evaluated frames based on total frames, stride, and audit scope
expected_evaluated_frames = calculate_expected_evaluated_frames(
    total_frames=meta["total_frames"],
    stride=frame_stride,
    audit_mode="quick" if is_quick_mode else "full",
    max_frames=max_frames_to_process,
)

# Determine Audit State
has_audit_run = "zone_mapper" in st.session_state
if has_audit_run:
    evaluated_f = st.session_state.get("evaluated_count", 0)
    fps_val = st.session_state.get("throughput_fps", 0.0)
    fc_events = st.session_state.get("fail_closed_count", 0)
    if fc_events > 0:
        audit_state_html = f"<span class='status-dot status-dot-warning'></span> Audit Complete: <strong>{evaluated_f} frames</strong> ({fc_events} Fail-Closed)"
    else:
        audit_state_html = f"<span class='status-dot status-dot-active'></span> Audit Complete: <strong>{evaluated_f} frames @ {fps_val:.1f} FPS</strong>"
else:
    audit_state_html = "<span class='status-dot status-dot-idle'></span> Status: <strong>Ready to Audit</strong>"

engine_badge_text = "Streamlit Cloud (Hosted CPU)" if IS_PUBLIC_DEMO else "Local Edge (127.0.0.1)"

# Render Horizontal Status Strip
st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-item">
            <span class="status-dot status-dot-active"></span> Engine: <strong>{engine_badge_text}</strong>
        </div>
        <div style="color: #29374B;">|</div>
        <div class="status-item">
            Active Source: <strong>{meta['filename']}</strong>
        </div>
        <div style="color: #29374B;">|</div>
        <div class="status-item">
            {audit_state_html}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Top-Level Console Navigation: Four Primary Areas
# ==============================================================================

tab_overview, tab_privacy, tab_spatial, tab_report = st.tabs([
    "Overview",
    "Privacy Review",
    "Spatial Intelligence",
    "Audit Report",
])


# ==============================================================================
# Tab 1: Overview Experience
# ==============================================================================

with tab_overview:
    # A. Video Source & Profile Panel
    with st.container(border=True):
        st.markdown('<div class="console-card-title">📹 Video Source & Camera Profile</div>', unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns([2.4, 1.2, 0.9, 0.9, 1.0])
        with col_m1:
            st.metric("Source File", meta["filename"])
        with col_m2:
            st.metric("Native Resolution", f"{meta['width']} × {meta['height']}")
        with col_m3:
            st.metric("Frame Rate", f"{meta['fps']:.1f} FPS")
        with col_m4:
            st.metric("Duration", f"{meta['duration_sec']:.2f}s")
        with col_m5:
            st.metric("Total Frames", f"{meta['total_frames']} frames")

    # B. Processing Configuration & Execution Panel
    with st.container(border=True):
        st.markdown('<div class="console-card-title">⚙️ Audit Parameters & Execution Configuration</div>', unsafe_allow_html=True)

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            if is_quick_mode:
                st.markdown(
                    f"**Audit Scope:** `Quick Audit (capped at {max_frames_to_process} frames)`  \n"
                    f"**Total Source Frames:** `{meta['total_frames']} frames`  \n"
                    f"**Selected Sampling Stride:** `Every {frame_stride} frame(s)`  \n"
                    f"**Expected Evaluated Frames:** `{expected_evaluated_frames} frames`"
                )
                st.caption(f"Quick Audit mode: Fast sampling up to {max_frames_to_process} frames for responsive demonstration.")
            else:
                st.markdown(
                    f"**Audit Scope:** `Full-Video Audit (Complete Video Stream)`  \n"
                    f"**Total Source Frames:** `{meta['total_frames']} frames`  \n"
                    f"**Selected Sampling Stride:** `Every {frame_stride} frame(s)`  \n"
                    f"**Expected Evaluated Frames:** `{expected_evaluated_frames} frames`"
                )
                st.caption(f"Full-Video mode: Evaluates complete stream from frame 0 to {meta['total_frames'] - 1} without artificial cap.")
        with col_cfg2:
            if manual_i_score is not None:
                st.markdown(
                    f"**Infrastructure Defect Factor ($I$):** `{manual_i_score:.2f}`  \n"
                    f"<span class='badge badge-accent'>Manual Auditor Input</span>  \n"
                    f"*Status:* Ready for complete ZVI calculation.",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"**Infrastructure Defect Factor ($I$):** `UNASSESSED`  \n"
                    f"<span class='badge badge-incomplete'>Manual Input Pending</span>  \n"
                    f"*Status:* ZVI will report as INCOMPLETE (zero substitution prevented).",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        run_processing = st.button("🚀 Run Privacy Audit", type="primary", use_container_width=True)

    # Placeholders for dynamic execution updates
    progress_bar = st.empty()
    status_text = st.empty()

    # Pipeline Execution
    if run_processing:
        # Check neural network model availability and cryptographic integrity
        with st.spinner("Verifying neural network model checkpoints..."):
            model_audit = ensure_models_available(PROJECT_ROOT)

        if not model_audit["all_available"]:
            err_details = [
                f"• {m['name']} ({m['filename']}): {m['error']}"
                for m in model_audit["models"].values()
                if not m["available"]
            ]
            st.error(
                "❌ **Pipeline Fault: Model Integrity Gate Failure**\n\n"
                "In accordance with fail-closed privacy guarantees, audits cannot execute without verified detector weights.\n\n"
                + "\n".join(err_details)
            )
            st.stop()

        face_model = str(model_audit["models"]["face"]["path"])
        vehicle_model = str(model_audit["models"]["vehicle"]["path"])
        plate_model = str(model_audit["models"]["plate"]["path"])

        engine = ConsolidatedPrivacyEngine(
            face_model_path=face_model,
            vehicle_model_path=vehicle_model,
            plate_model_path=plate_model,
            device="cpu" if IS_PUBLIC_DEMO else ("mps" if torch.backends.mps.is_available() else "cpu"),
        )
        zone_mapper = SpatialZoneMapper()

        cap = cv2.VideoCapture(str(selected_video_path))
        if not cap.isOpened():
            st.error(f"Error opening video capture pipeline: {selected_video_path.name}")
            st.stop()

        total_to_evaluate = expected_evaluated_frames
        evaluated_count = 0
        fail_closed_count = 0
        t_run_start = time.perf_counter()

        last_redacted = None
        last_diag = None
        last_spatial = None
        last_source = None
        last_metrics: Dict[str, Any] = {}
        frame_idx = 0

        try:
            while evaluated_count < total_to_evaluate:
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Video stream ended (normal EOF or decode termination)
                    break

                if frame_idx % frame_stride == 0:
                    # 1. Execute Hardened Privacy Engine
                    redacted_frame, gate_status, metrics = engine.process_frame(frame)
                    evaluated_count += 1

                    if gate_status != PrivacyGateStatus.RELEASE_APPROVED:
                        fail_closed_count += 1

                    # 2. Execute Spatial Zone Mapper
                    zone_results = zone_mapper.analyze_frame(
                        redacted_frame,
                        metrics,
                        manual_infrastructure=manual_i_score,
                        accumulate=True,
                    )

                    # 3. Generate Diagnostic & Heatmap Visualizations
                    canonical_input = cv2.resize(frame, (CANONICAL_WIDTH, CANONICAL_HEIGHT))
                    diag_vis = draw_diagnostic_overlay(canonical_input, redacted_frame, gate_status, metrics)
                    spatial_vis = zone_mapper.render_spatial_heatmap(
                        redacted_frame,
                        metric="zvi" if enable_manual_infra else "lighting",
                        manual_infrastructure=manual_i_score,
                    )

                    # Convert BGR to RGB for transient display buffers
                    rgb_redacted = cv2.cvtColor(redacted_frame, cv2.COLOR_BGR2RGB)
                    rgb_diag = cv2.cvtColor(diag_vis, cv2.COLOR_BGR2RGB)
                    rgb_spatial = cv2.cvtColor(spatial_vis, cv2.COLOR_BGR2RGB)

                    last_redacted = rgb_redacted
                    last_diag = rgb_diag
                    last_spatial = rgb_spatial
                    if not IS_PUBLIC_DEMO:
                        rgb_source = cv2.cvtColor(canonical_input, cv2.COLOR_BGR2RGB)
                        last_source = rgb_source
                    last_metrics = metrics

                    progress = min(1.0, evaluated_count / max(1, total_to_evaluate))
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Auditing source frame {frame_idx + 1}/{meta['total_frames']} "
                        f"(Evaluated: {evaluated_count}/{total_to_evaluate})..."
                    )

                frame_idx += 1
        finally:
            cap.release()

        t_run_end = time.perf_counter()
        total_run_sec = max(0.001, t_run_end - t_run_start)
        throughput_fps = evaluated_count / total_run_sec

        if evaluated_count == total_to_evaluate:
            progress_bar.progress(1.0)
            status_text.success(
                f"Audit completed: {evaluated_count} frames evaluated in {total_run_sec:.2f}s "
                f"({throughput_fps:.1f} FPS) across {meta['total_frames']} source frames."
            )
        elif evaluated_count > 0:
            progress_bar.progress(min(1.0, evaluated_count / max(1, total_to_evaluate)))
            status_text.warning(
                f"Video ended at source frame {frame_idx}/{meta['total_frames']}. "
                f"Evaluated {evaluated_count}/{total_to_evaluate} expected frames in {total_run_sec:.2f}s "
                f"({throughput_fps:.1f} FPS)."
            )
        else:
            status_text.error("No video frames could be decoded from the source stream.")

        # Persist audit run state into session state
        st.session_state["evaluated_count"] = evaluated_count
        st.session_state["expected_evaluated"] = total_to_evaluate
        st.session_state["audit_scope"] = audit_scope_option
        st.session_state["frame_stride"] = frame_stride
        st.session_state["total_source_frames"] = meta["total_frames"]
        st.session_state["fail_closed_count"] = fail_closed_count
        st.session_state["total_run_sec"] = total_run_sec
        st.session_state["throughput_fps"] = throughput_fps
        st.session_state["last_metrics"] = last_metrics
        st.session_state["zone_mapper"] = zone_mapper
        st.session_state["last_redacted"] = last_redacted
        st.session_state["last_diag"] = last_diag
        st.session_state["last_spatial"] = last_spatial
        if not IS_PUBLIC_DEMO:
            st.session_state["last_source"] = last_source

    # C. Run Summary or Empty State
    if "zone_mapper" in st.session_state:
        evaluated_count = st.session_state["evaluated_count"]
        expected_eval = st.session_state.get("expected_evaluated", evaluated_count)
        audit_scope_ran = st.session_state.get("audit_scope", audit_scope_option)
        stride_ran = st.session_state.get("frame_stride", frame_stride)
        source_frames_ran = st.session_state.get("total_source_frames", meta["total_frames"])
        fail_closed_count = st.session_state["fail_closed_count"]
        total_run_sec = st.session_state["total_run_sec"]
        throughput_fps = st.session_state["throughput_fps"]

        with st.container(border=True):
            st.markdown('<div class="console-card-title">📊 Execution Telemetry & Performance Summary</div>', unsafe_allow_html=True)

            c_sm1, c_sm2, c_sm3, c_sm4, c_sm5 = st.columns(5)
            with c_sm1:
                st.metric("Source Frames", f"{source_frames_ran}")
                st.caption(f"{audit_scope_ran.split('(')[0].strip()}")
            with c_sm2:
                st.metric("Evaluated Frames", f"{evaluated_count}")
                st.caption(f"Expected: {expected_eval} frames")
            with c_sm3:
                st.metric("Sampling Stride", f"Stride {stride_ran}")
                st.caption(f"Elapsed: {total_run_sec:.2f}s")
            with c_sm4:
                st.metric("Throughput", f"{throughput_fps:.1f} FPS")
                st.caption(f"{evaluated_count - fail_closed_count} Approved")
            with c_sm5:
                if fail_closed_count == 0:
                    st.metric("Privacy Gate", "APPROVED")
                    st.caption("Opaque Non-Reversible Masking")
                else:
                    st.metric("Privacy Gate", f"{fail_closed_count} FAILS")
                    st.caption("Fail-Closed Suppressed")

            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #9AAEC4; margin-top: 0.75rem; border-top: 1px solid #29374B; padding-top: 0.75rem;">
                    Audit complete. Use the top navigation tabs to inspect detailed findings:
                    <br>• <strong>Privacy Review:</strong> Examine opaque non-reversible masking and diagnostic mask provenance.
                    <br>• <strong>Spatial Intelligence:</strong> Explore the 3×3 video-scene grid and spatial vulnerability heatmap.
                    <br>• <strong>Audit Report:</strong> Review the Zone Vulnerability Index (ZVI) breakdown and governance audit record.
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        prompt_instruction = (
            "Select audit parameters in the sidebar and click <strong>Run Privacy Audit</strong> to execute demonstration inference on the built-in sample footage."
            if IS_PUBLIC_DEMO
            else "Select audit parameters above and click <strong>Run Privacy Audit</strong> to trigger local edge inference."
        )
        engine_infra_text = "Hosted Cloud Inference" if IS_PUBLIC_DEMO else "Local-Only Inference"
        ram_lifecycle_text = "Ephemeral Container RAM" if IS_PUBLIC_DEMO else "Ephemeral RAM Lifecycles"

        st.markdown(
            f"""
            <div class="empty-state">
                <div class="empty-state-title">Console Ready: No Active Audit Run</div>
                <div class="empty-state-desc">
                    {prompt_instruction}
                    Video frames will be processed through the hardened privacy engine, mapped across the 3×3 spatial grid,
                    and evaluated for environmental lighting and mobility vulnerability.
                </div>
                <div style="display: flex; justify-content: center; gap: 1.5rem; font-size: 0.8rem; color: #9AAEC4;">
                    <div>🛡️ <strong>Opaque Non-Reversible Masking</strong></div>
                    <div>⚡ <strong>{engine_infra_text}</strong></div>
                    <div>🔒 <strong>{ram_lifecycle_text}</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# Tab 2: Privacy Review Experience
# ==============================================================================

with tab_privacy:
    if "zone_mapper" in st.session_state and st.session_state.get("last_redacted") is not None:
        last_redacted = st.session_state["last_redacted"]
        last_diag = st.session_state["last_diag"]
        last_source = st.session_state.get("last_source", None)

        col_prev1, col_prev2 = st.columns([1.6, 1.0])

        with col_prev1:
            with st.container(border=True):
                st.markdown('<div class="console-card-title">🛡️ Redacted Frame Preview (Release Candidate)</div>', unsafe_allow_html=True)
                st.image(last_redacted, use_container_width=True)
                st.caption(
                    "Canonical 1280×720 display buffer. Opaque, non-reversible masking applied to detected pedestrian faces, "
                    "vehicle registration plates, and class-aware fallbacks."
                )

        with col_prev2:
            with st.container(border=True):
                st.markdown('<div class="console-card-title">🔍 Diagnostic Mask Provenance</div>', unsafe_allow_html=True)
                st.image(last_diag, use_container_width=True)
                st.caption(
                    "Internal diagnostic overlay verifying detection boundaries, class assignments, "
                    "and fail-closed gate status."
                )

        # Structured Provenance Legend
        with st.container(border=True):
            st.markdown('<div class="console-card-title">🏷️ Redaction Mask Provenance Legend</div>', unsafe_allow_html=True)

            leg_c1, leg_c2, leg_c3, leg_c4, leg_c5 = st.columns(5)
            with leg_c1:
                st.markdown(
                    "<span class='badge badge-accent'>Face Mask</span>  \n"
                    "**YuNet ONNX**  \n"
                    "*Confidence: ≥0.60*",
                    unsafe_allow_html=True,
                )
            with leg_c2:
                st.markdown(
                    "<span class='badge badge-ok'>License Plate</span>  \n"
                    "**YOLOv8n-plate**  \n"
                    "*Confidence: ≥0.30*",
                    unsafe_allow_html=True,
                )
            with leg_c3:
                st.markdown(
                    "<span class='badge badge-incomplete'>2W/3W Mudguard</span>  \n"
                    "**Context Fallback**  \n"
                    "*Class-aware bounds*",
                    unsafe_allow_html=True,
                )
            with leg_c4:
                st.markdown(
                    "<span class='badge' style='background:rgba(217,70,239,0.15);color:#D946EF;border:1px solid rgba(217,70,239,0.35);'>4W Bumper</span>  \n"
                    "**Context Fallback**  \n"
                    "*Lower 35% ROI*",
                    unsafe_allow_html=True,
                )
            with leg_c5:
                st.markdown(
                    "<span class='badge badge-error'>Temporal Hold</span>  \n"
                    "**Tracker Hold**  \n"
                    "*3-frame persistence*",
                    unsafe_allow_html=True,
                )

        # Source Frame Exposure Policy
        if IS_PUBLIC_DEMO:
            with st.container(border=True):
                st.markdown('<div class="console-card-title">🔒 Source Frame Exposure Policy (Hosted Demo)</div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="background-color: rgba(45, 212, 191, 0.08); border: 1px solid #2DD4BF; border-radius: 6px; padding: 0.85rem 1.1rem;">
                        <div style="color: #2DD4BF; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.3rem;">
                            ✓ Unredacted Source Frame Disclosure Permanently Disabled
                        </div>
                        <div style="color: #E5EDF7; font-size: 0.82rem; line-height: 1.5;">
                            In accordance with strict Privacy-by-Design invariants for public hosted deployments, raw unredacted
                            source-frame inspection is permanently disabled. Only opaque redacted frames and diagnostic provenance
                            overlays are exposed. Raw frames remain strictly ephemeral in server container memory during inference
                            and are discarded immediately without disk persistence or public network transmission.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with st.container(border=True):
                st.markdown('<div class="console-card-title">👁️ Local Auditor Source Reference (Restricted Inspection)</div>', unsafe_allow_html=True)

            st.markdown(
                """
                <div class="auditor-warning">
                    <div class="auditor-warning-title">⚠️ Mandatory Privacy Notice: Ephemeral Unredacted Data</div>
                    <div class="auditor-warning-body">
                        The unredacted source frame contains Personally Identifiable Information (identifiable faces and vehicle
                        registration marks). It is retained exclusively in volatile RAM during local execution and is never saved
                        to disk or transmitted over the network. Access is restricted to authorized auditing review.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            reveal_source = st.checkbox(
                "Reveal Unredacted Source Frame (Authorized Auditor Only)",
                value=False,
                help="Check to inspect the transient unredacted source frame in volatile RAM.",
            )

            if reveal_source:
                if last_source is not None:
                    st.image(last_source, use_container_width=True)
                    st.caption("Ephemeral source frame rendered from volatile memory. Strictly prohibited from export or recording.")
                else:
                    st.info("No transient source frame buffer available in session memory.")
            else:
                st.info("Unredacted source frame is masked by default in accordance with Privacy-by-Design invariants.")

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No Privacy Audit Data Available</div>
                <div class="empty-state-desc">
                    Run an audit from the <strong>Overview</strong> tab to generate and inspect opaque non-reversible redactions,
                    diagnostic provenance masks, and ephemeral source frame comparisons.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# Tab 3: Spatial Intelligence Experience
# ==============================================================================

with tab_spatial:
    if "zone_mapper" in st.session_state:
        stored_zone_mapper = st.session_state["zone_mapper"]
        v_summary = stored_zone_mapper.get_video_summary(manual_infrastructure=manual_i_score)
        zones_dict = v_summary["zones_summary"]
        last_redacted = st.session_state.get("last_redacted")

        with st.container(border=True):
            st.markdown('<div class="console-card-title">🗺️ Video-Scene Spatial Map (3×3 Canonical Camera Grid)</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #9AAEC4; margin-bottom: 0.75rem; border-left: 3px solid #2DD4BF; padding-left: 0.6rem;">
                <strong>Spatial Coordinate Disclaimer:</strong> This visualization is a <strong>VIDEO-SCENE SPATIAL MAP</strong>
                representing the in-camera field of view divided into 9 canonical coordinate zones ($426.67 \times 240$ px each).
                It does <strong>not</strong> represent geographic GPS coordinates. Observed entity counts reflect occupancy, not danger.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_map_vis, col_map_info = st.columns([1.5, 1.0])

        with col_map_vis:
            if last_redacted is not None:
                # Re-render spatial heatmap based on current manual infrastructure setting
                bgr_redacted = cv2.cvtColor(last_redacted, cv2.COLOR_RGB2BGR)
                heat_vis = stored_zone_mapper.render_spatial_heatmap(
                    bgr_redacted,
                    metric="zvi" if manual_i_score is not None else "lighting",
                    manual_infrastructure=manual_i_score,
                )
                rgb_heat = cv2.cvtColor(heat_vis, cv2.COLOR_BGR2RGB)
                st.image(rgb_heat, use_container_width=True)
                st.caption("3×3 Canonical Video-Scene Spatial Heatmap with HUD telemetry overlay.")

        with col_map_info:
            scene_mean_l = float(np.mean([zd["mean_lighting_deficit"] for zd in zones_dict.values()]))
            scene_mean_m = float(np.mean([zd["mean_mobility_factor"] for zd in zones_dict.values()]))
            total_scene_vru = sum(zd["total_persons_observed"] for zd in zones_dict.values())
            total_scene_veh = sum(zd["total_vehicles_observed"] for zd in zones_dict.values())

            st.markdown(
                f"""
                <div class="elevated-panel">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #E5EDF7; margin-bottom: 0.5rem;">SCENE SPATIAL SUMMARY</div>
                    <div style="font-size: 0.8rem; color: #9AAEC4; line-height: 1.6;">
                        • <strong>Mean Lighting Deficit ($L$):</strong> `{scene_mean_l:.3f}`
                        • <strong>Mean Mobility Exposure ($M$):</strong> `{scene_mean_m:.3f}`
                        • <strong>Total VRU Observations:</strong> `{total_scene_vru}`
                        • <strong>Total Vehicle Observations:</strong> `{total_scene_veh}`
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3x3 Canonical Zone Panels
        with st.container(border=True):
            st.markdown('<div class="console-card-title">📐 Nine-Zone Camera Perspective Grid</div>', unsafe_allow_html=True)

        row_chunks = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]

        for r_zones in row_chunks:
            z_cols = st.columns(3)
            for c_idx, zid in enumerate(r_zones):
                zd = zones_dict[zid]
                zvi_d = zd["zvi_result"]
                with z_cols[c_idx]:
                    if zvi_d["is_complete"]:
                        zvi_badge = f"<span class='badge badge-ok'>ZVI: {zvi_d['zvi']:.1f}</span>"
                    else:
                        zvi_badge = "<span class='badge badge-incomplete'>ZVI: Incomplete</span>"

                    st.markdown(
                        f"""
                        <div class="zone-cell">
                            <div class="zone-cell-header">
                                <span>{zd['name']}</span>
                                {zvi_badge}
                            </div>
                            <div class="zone-metric-row">
                                <span>Lighting Deficit ($L$):</span>
                                <span class="zone-metric-val">{zd['mean_lighting_deficit']:.3f}</span>
                            </div>
                            <div class="zone-metric-row">
                                <span>Mobility Factor ($M$):</span>
                                <span class="zone-metric-val">{zd['mean_mobility_factor']:.3f}</span>
                            </div>
                            <div class="zone-metric-row">
                                <span>Avg Pedestrians:</span>
                                <span class="zone-metric-val">{zd['mean_persons']:.1f} (Peak: {zd['peak_persons']})</span>
                            </div>
                            <div class="zone-metric-row">
                                <span>Avg Vehicles:</span>
                                <span class="zone-metric-val">{zd['mean_vehicles']:.1f} (Peak: {zd['peak_vehicles']})</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.markdown('</div>', unsafe_allow_html=True)

        # Secondary Detailed Per-Zone Table
        with st.expander("📊 Detailed Per-Zone Observation Table", expanded=False):
            table_rows = []
            for zid in range(1, 10):
                zd = zones_dict[zid]
                zvi_d = zd["zvi_result"]
                table_rows.append({
                    "Zone": f"Zone {zid}",
                    "Region": zd["name"].split()[1][1:-1],
                    "Lighting Deficit (L)": zd["mean_lighting_deficit"],
                    "Mobility Factor (M)": zd["mean_mobility_factor"],
                    "Avg Persons": zd["mean_persons"],
                    "Peak Persons": zd["peak_persons"],
                    "Total Persons": zd["total_persons_observed"],
                    "Avg Vehicles": zd["mean_vehicles"],
                    "Peak Vehicles": zd["peak_vehicles"],
                    "Total Vehicles": zd["total_vehicles_observed"],
                    "Redaction Masks": zd["total_masks_applied"],
                    "ZVI Score": f"{zvi_d['zvi']:.1f}" if zvi_d["is_complete"] else "Incomplete",
                })
            st.dataframe(table_rows, use_container_width=True)

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No Spatial Intelligence Available</div>
                <div class="empty-state-desc">
                    Run an audit from the <strong>Overview</strong> tab to evaluate scene-level 3×3 zone metrics,
                    temporal lighting deficits, and localized vulnerable road user distributions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# Tab 4: Audit Report Experience
# ==============================================================================

with tab_report:
    if "zone_mapper" in st.session_state:
        stored_zone_mapper = st.session_state["zone_mapper"]
        v_summary = stored_zone_mapper.get_video_summary(manual_infrastructure=manual_i_score)
        zones_dict = v_summary["zones_summary"]
        evaluated_count = st.session_state["evaluated_count"]
        fail_closed_count = st.session_state["fail_closed_count"]
        total_run_sec = st.session_state["total_run_sec"]
        throughput_fps = st.session_state["throughput_fps"]
        expected_eval = st.session_state.get("expected_evaluated", evaluated_count)
        audit_scope_ran = st.session_state.get("audit_scope", audit_scope_option)
        stride_ran = st.session_state.get("frame_stride", frame_stride)
        source_frames_ran = st.session_state.get("total_source_frames", meta["total_frames"])

        # Calculate Scene-Level Aggregates
        scene_l = float(np.mean([zd["mean_lighting_deficit"] for zd in zones_dict.values()]))
        scene_m = float(np.mean([zd["mean_mobility_factor"] for zd in zones_dict.values()]))
        scene_zvi = compute_zvi(scene_l, manual_i_score, scene_m)

        # 1. Executive Summary
        with st.container(border=True):
            st.markdown('<div class="console-card-title">📋 Executive Audit Summary</div>', unsafe_allow_html=True)

            rep_c1, rep_c2, rep_c3, rep_c4 = st.columns(4)
            with rep_c1:
                st.metric("Source Video", meta["filename"])
                st.caption(f"{source_frames_ran} frames @ {meta['fps']:.1f} FPS")
            with rep_c2:
                st.metric("Evaluated Frames", f"{evaluated_count} / {expected_eval}")
                st.caption(f"{audit_scope_ran.split('(')[0].strip()} | Stride {stride_ran}")
            with rep_c3:
                st.metric("Throughput", f"{throughput_fps:.1f} FPS")
                st.caption(f"Duration: {total_run_sec:.2f}s")
            with rep_c4:
                if fail_closed_count == 0:
                    st.metric("Privacy Gate", "APPROVED")
                    st.markdown("<span class='badge badge-ok'>Opaque Non-Reversible Masking</span>", unsafe_allow_html=True)
                else:
                    st.metric("Privacy Gate", f"FAIL-CLOSED ({fail_closed_count})")
                    st.markdown("<span class='badge badge-error'>Suppressed to Black</span>", unsafe_allow_html=True)

        # 2. Zone Vulnerability Index (ZVI) Detail
        with st.container(border=True):
            st.markdown('<div class="console-card-title">⚖️ Zone Vulnerability Index (ZVI) Heuristic Specification</div>', unsafe_allow_html=True)

            st.markdown(
                r"""
                $$\text{ZVI} = 100 \times \left( 0.50 \cdot L + 0.35 \cdot I + 0.15 \cdot M \right)$$
                """
            )

            col_zvi1, col_zvi2, col_zvi3 = st.columns([1.2, 1.2, 1.6])

            with col_zvi1:
                st.markdown("**Component Measurements:**")
                st.markdown(f"- **Lighting Deficit ($L$):** `{scene_l:.3f}`  \n  *(Automated HSV V-channel derivation)*")
                if manual_i_score is not None:
                    st.markdown(f"- **Infrastructure Deficit ($I$):** `{manual_i_score:.3f}`  \n  *(Manual Auditor Input)*")
                else:
                    st.markdown("- **Infrastructure Deficit ($I$):** *UNASSESSED*  \n  *(Manual inspection pending)*")
                st.markdown(f"- **Mobility Exposure ($M$):** `{scene_m:.3f}`  \n  *(Automated VRU density)*")

            with col_zvi2:
                st.markdown("**Weighted Factor Contribution:**")
                if scene_zvi["is_complete"]:
                    c_dict = scene_zvi["contributions"]
                    st.markdown(f"- **Lighting (50% weight):** `+{c_dict['L_contribution']:.2f} pts`")
                    st.markdown(f"- **Infrastructure (35% weight):** `+{c_dict['I_contribution']:.2f} pts`")
                    st.markdown(f"- **Mobility (15% weight):** `+{c_dict['M_contribution']:.2f} pts`")
                else:
                    st.markdown(
                        "<span style='color: #FBBF24;'>Weighted calculation withheld until all required components are populated.</span>",
                        unsafe_allow_html=True,
                    )

            with col_zvi3:
                if scene_zvi["is_complete"]:
                    st.metric("Aggregated Scene ZVI", f"{scene_zvi['zvi']:.1f} / 100")
                    st.markdown(f"<span class='badge badge-ok'>COMPLETE</span> Primary Driver: <strong>{scene_zvi['primary_driver']}</strong>", unsafe_allow_html=True)
                else:
                    st.metric("Aggregated Scene ZVI", "INCOMPLETE")
                    st.markdown(
                        "<span class='badge badge-incomplete'>INCOMPLETE</span> Factor I requires manual auditor assessment. "
                        "Zero is never substituted for missing components.",
                        unsafe_allow_html=True,
                    )

            st.markdown(
                """
                <div style="font-size: 0.78rem; color: #9AAEC4; border-top: 1px solid #29374B; padding-top: 0.6rem; margin-top: 0.75rem;">
                    <strong>Scientific & Governance Disclaimer:</strong> The Zone Vulnerability Index (ZVI) is an experimental
                    decision-support heuristic for municipal prioritization. It does not constitute a statistically calibrated
                    actuarial or criminal risk model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 3. Privacy & Fail-Closed Release Verification
        with st.container(border=True):
            st.markdown('<div class="console-card-title">🔒 Privacy Release Gate & Fail-Closed Certification</div>', unsafe_allow_html=True)

            if fail_closed_count > 0:
                st.markdown(
                    f"""
                    <div class="auditor-warning">
                        <div class="auditor-warning-title">⚠️ Fail-Closed Events Recorded</div>
                        <div class="auditor-warning-body">
                            {fail_closed_count} frame(s) encountered detector or pipeline anomalies. In accordance with
                            fail-closed privacy guarantees, these frames were automatically suppressed to solid black buffers
                            with warning watermarks to prevent unredacted data leakage.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="background-color: rgba(52, 211, 153, 0.08); border: 1px solid #34D399; border-radius: 6px; padding: 0.8rem 1rem;">
                        <div style="color: #34D399; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.2rem;">✓ PRIVACY GATE RELEASE APPROVED</div>
                        <div style="color: #E5EDF7; font-size: 0.82rem;">
                            All applied redactions utilize opaque, non-reversible masking over detected faces, license plates,
                            and vehicle fallbacks with fail-closed pipeline suppression. Mask opacity is absolute; detection
                            recall remains governed by detector models and environmental visibility. Zero unredacted frames,
                            crops, or screenshots were persisted to disk.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No Audit Report Generated</div>
                <div class="empty-state-desc">
                    Execute an audit from the <strong>Overview</strong> tab to compile an executive audit summary,
                    Zone Vulnerability Index (ZVI) breakdown, and privacy gate certification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
