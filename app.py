"""
DroneAI Command Center — Streamlit Dashboard
=============================================
Main entry point.  Run with:
    streamlit run app.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time
from datetime import datetime

from config import DRONE_CONFIG, DETECTION_COLORS, REFRESH_INTERVAL, DEFAULT_CONFIDENCE, CAMERA_DEVICE_INDEX
from utils.telemetry import get_telemetry_data, get_telemetry_history
from utils.detection import infer_detections, get_detection_summary, get_simulated_detection_data, load_yolo_model
from utils.video import get_simulated_frame, open_camera, get_webcam_frame, add_detection_overlay, RPI_CAMERA_AVAILABLE


# ── Persistent webcam connection (survives Streamlit reruns) ────
@st.cache_resource
def init_camera(index=CAMERA_DEVICE_INDEX):
    return open_camera(index)


@st.cache_resource
def init_yolo_model():
    try:
        return load_yolo_model()
    except Exception:
        return None


def main():
    # ╔══════════════════════════════════════════════════════════════╗
    # ║  PAGE CONFIG                                                 ║
    # ╚══════════════════════════════════════════════════════════════╝
    st.set_page_config(
        page_title="DroneAI Command Center",
        page_icon="🛸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ╔══════════════════════════════════════════════════════════════╗
    # ║  CUSTOM CSS                                                  ║
    # ╚══════════════════════════════════════════════════════════════╝
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ──────────────────────────────────── */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header banner ───────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0d4429 100%);
        border: 1px solid rgba(0,255,136,0.12);
        border-radius: 14px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-banner h1 {
        color: #00ff88;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-banner .subtitle {
        color: #8b949e;
        font-size: 0.88rem;
        margin-top: 0.25rem;
    }
    .badge-online {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0,255,136,0.1);
        color: #00ff88;
        border: 1px solid rgba(0,255,136,0.25);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.4px;
    }
    .badge-online::before {
        content: '';
        width: 8px;
        height: 8px;
        background: #00ff88;
        border-radius: 50%;
        animation: pulse-dot 1.5s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.35; }
    }

    /* ── Metric cards override ───────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
    }

    /* ── Tabs ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0d1117;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #161b22;
        color: #00ff88;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #e6edf3;
    }

    /* ── Hide chrome ─────────────────────────────── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Section dividers ────────────────────────── */
    .section-title {
        color: #e6edf3;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Detection chip ──────────────────────────── */
    .det-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-bottom: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


    # ╔══════════════════════════════════════════════════════════════╗
    # ║  SIDEBAR                                                     ║
    # ╚══════════════════════════════════════════════════════════════╝
    with st.sidebar:
        st.markdown("### 🛸 DroneAI Control")
        st.caption(f"{DRONE_CONFIG['name']}  ·  {DRONE_CONFIG['model']}")
        st.markdown("---")

        st.markdown("#### ⚙️ Mission Parameters")
        max_altitude  = st.slider("Max Altitude (m)", 10, 120, 50)
        max_speed     = st.slider("Max Speed (m/s)", 1, 15, 8)
        flight_mode   = st.selectbox("Flight Mode", ["AUTO", "GUIDED", "LOITER", "RTL", "LAND"])

        st.markdown("---")
        st.markdown("#### 🎯 Detection Filters")
        conf_threshold = st.slider("Min Confidence", 0.30, 1.0, DEFAULT_CONFIDENCE, 0.05)
        detect_person  = st.checkbox("Person",  value=True)
        detect_vehicle = st.checkbox("Vehicle", value=True)
        detect_animal  = st.checkbox("Animal",  value=True)
        detect_drone   = st.checkbox("Drone",   value=True)

        st.markdown("---")
        st.markdown("#### 🔄 Refresh")
        auto_refresh = st.toggle("Auto-refresh", value=False)
        if st.button("↻  Refresh Now", use_container_width=True):
            st.rerun()


    # ╔══════════════════════════════════════════════════════════════╗
    # ║  HEADER                                                      ║
    # ╚══════════════════════════════════════════════════════════════╝
    st.markdown("""
    <div class="hero-banner">
        <div>
            <h1>🛸 DroneAI Command Center</h1>
            <div class="subtitle">Real-time Autonomous Surveillance  ·  RPi 5 + Hailo-8L</div>
        </div>
        <span class="badge-online">SYSTEM ONLINE</span>
    </div>
    """, unsafe_allow_html=True)


    # ╔══════════════════════════════════════════════════════════════╗
    # ║  TABS                                                        ║
    # ╚══════════════════════════════════════════════════════════════╝
    tab_overview, tab_feed, tab_detections, tab_analytics, tab_system = st.tabs(
        ["📊 Overview", "📹 Live Feed", "🎯 Detections", "📈 Analytics", "🖥️ System"]
    )

    # ─── get shared data once ────────────────────────────────────────
    telemetry  = get_telemetry_data()
    model      = init_yolo_model()
    live_dets  = pd.DataFrame(columns=["timestamp", "class", "confidence", "x", "y", "width", "height"])
    summary    = get_detection_summary(live_dets)


    # ┌──────────────────────────────────────────────────────────────┐
    # │  TAB 1 — Overview                                            │
    # └──────────────────────────────────────────────────────────────┘
    with tab_overview:
        # -- top metrics ------------------------------------------------
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Altitude",   f"{telemetry['altitude']} m",    f"{np.random.uniform(-1.5, 1.5):.1f} m")
        m2.metric("Speed",      f"{telemetry['speed']} m/s",     f"{np.random.uniform(-0.8, 0.8):.1f}")
        m3.metric("Battery",    f"{telemetry['battery']}%",      f"-{np.random.uniform(0, 0.5):.1f}%")
        m4.metric("Satellites", f"{telemetry['satellites']}",     None)
        m5.metric("Signal",     f"{telemetry['signal_strength']}%", None)
        m6.metric("Heading",    f"{telemetry['heading']}°",      None)

        st.markdown("")

        # -- map + info -------------------------------------------------
        col_map, col_info = st.columns([2, 1])

        with col_map:
            st.markdown('<div class="section-title">🗺️ Drone Position</div>', unsafe_allow_html=True)
            map_df = pd.DataFrame({
                "lat": [telemetry["gps_lat"]],
                "lon": [telemetry["gps_lon"]],
            })
            st.map(map_df, zoom=15, use_container_width=True)

        with col_info:
            st.markdown('<div class="section-title">📋 Flight Info</div>', unsafe_allow_html=True)

            info_data = {
                "Parameter": [
                    "Flight Mode", "Armed", "Heading", "GPS Lat", "GPS Lon",
                    "AI Model", "Inference FPS", "Uptime",
                ],
                "Value": [
                    flight_mode,
                    "✅ Yes" if telemetry["armed"] else "❌ No",
                    f"{telemetry['heading']}°",
                    f"{telemetry['gps_lat']:.6f}",
                    f"{telemetry['gps_lon']:.6f}",
                    "YOLOv8n (Hailo-8L)",
                    f"{np.random.randint(25, 30)}",
                    f"{np.random.randint(5, 55)} min",
                ],
            }
            st.table(pd.DataFrame(info_data))

            st.markdown('<div class="section-title">🎯 Detections</div>', unsafe_allow_html=True)
            for cls, cnt in summary.items():
                color = DETECTION_COLORS.get(cls, "#888")
                st.markdown(
                    f'<span class="det-chip" style="background:{color}22;color:{color};border:1px solid {color}44">'
                    f'{cls}: <b>{cnt}</b></span>',
                    unsafe_allow_html=True,
                )


    # ┌──────────────────────────────────────────────────────────────┐
    # │  TAB 2 — Live Feed                                           │
    # └──────────────────────────────────────────────────────────────┘
    with tab_feed:
        col_cam, col_live_det = st.columns([2, 1])

        live_dets = get_simulated_detection_data(8)

        with col_cam:
            # ── source toggle ──
            if RPI_CAMERA_AVAILABLE:
                st.markdown('<div class="section-title">📹 Live ArduCam IMX219-R Feed</div>', unsafe_allow_html=True)
                cam = init_camera()
                frame = get_webcam_frame(cam)
                if frame is not None:
                    if model is not None:
                        live_dets = infer_detections(frame, model=model, conf_threshold=conf_threshold)
                    else:
                        st.warning("YOLOv8 model unavailable, using simulated detections.")
                        live_dets = get_simulated_detection_data(8)

                    frame = add_detection_overlay(frame, live_dets)
                    st.image(frame, use_column_width=True)
                else:
                    st.error("⚠️ Unable to capture live ArduCam feed. Verify picamera2 installation and camera connection.")
                    st.info("If the camera is attached to the Raspberry Pi, restart the app after installing picamera2.")
            else:
                use_webcam = st.toggle("📷 Use Laptop Webcam", value=True)

                if use_webcam:
                    st.markdown('<div class="section-title">📹 Live Webcam Feed</div>', unsafe_allow_html=True)
                    cam = init_camera()
                    frame = get_webcam_frame(cam)
                    if frame is not None:
                        if model is not None:
                            live_dets = infer_detections(frame, model=model, conf_threshold=conf_threshold)
                        else:
                            st.warning("YOLOv8 model unavailable, using simulated detections.")
                            live_dets = get_simulated_detection_data(8)

                        frame = add_detection_overlay(frame, live_dets)
                        st.image(frame, use_column_width=True)
                    else:
                        st.error("⚠️ Webcam capture failed. Ensure a webcam is connected and not in use by another app.")
                        st.markdown('<div class="section-title">📹 Camera Feed (Simulated)</div>', unsafe_allow_html=True)
                        frame = get_simulated_frame()
                        live_dets = get_simulated_detection_data(8)
                        st.image(frame, use_column_width=True)
                else:
                    st.markdown('<div class="section-title">📹 Camera Feed (Simulated)</div>', unsafe_allow_html=True)
                    frame = get_simulated_frame()
                    live_dets = get_simulated_detection_data(8)
                    st.image(frame, use_column_width=True)

            # controls
            c1, c2, c3, c4 = st.columns(4)
            c1.button("📸 Capture",    use_container_width=True)
            c2.button("⏺️ Record",     use_container_width=True)
            c3.button("🔍 Zoom In",    use_container_width=True)
            c4.button("🌙 Night Mode", use_container_width=True)

        with col_live_det:
            st.markdown('<div class="section-title">🎯 Live Detections</div>', unsafe_allow_html=True)
            for _, row in live_dets.iterrows():
                icon = "🟢" if row["confidence"] > 0.85 else ("🟡" if row["confidence"] > 0.70 else "🔴")
                color = DETECTION_COLORS.get(row["class"], "#888")
                st.markdown(
                    f'{icon} **{row["class"]}** — `{row["confidence"]:.0%}`',
                )
                st.caption(f'Pos ({row["x"]}, {row["y"]})  ·  Size {row["width"]}×{row["height"]}')
                st.markdown("---")

            if live_dets.empty:
                st.info("No detections were found on the current frame.")


    detections = live_dets if not live_dets.empty else get_simulated_detection_data(30)
    summary = get_detection_summary(detections)


    # ┌──────────────────────────────────────────────────────────────┐
    # │  TAB 3 — Detections                                          │
    # └──────────────────────────────────────────────────────────────┘
    with tab_detections:
        d_col1, d_col2 = st.columns(2)

        with d_col1:
            fig_pie = px.pie(
                values=list(summary.values()),
                names=list(summary.keys()),
                title="Detection Distribution",
                hole=0.45,
                color=list(summary.keys()),
                color_discrete_map=DETECTION_COLORS,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#8b949e",
                title_font_color="#e6edf3",
                legend_font_color="#8b949e",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with d_col2:
            fig_hist = px.histogram(
                detections,
                x="confidence",
                nbins=15,
                title="Confidence Distribution",
                color_discrete_sequence=["#00ff88"],
            )
            fig_hist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#8b949e",
                title_font_color="#e6edf3",
                xaxis_title="Confidence",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # -- detection timeline -----------------------------------------
        st.markdown('<div class="section-title">📋 Detection Log</div>', unsafe_allow_html=True)

        # filter by active classes
        active_classes = []
        if detect_person:  active_classes.append("Person")
        if detect_vehicle: active_classes.append("Vehicle")
        if detect_animal:  active_classes.append("Animal")
        if detect_drone:   active_classes.append("Drone")
        active_classes.append("Unknown")

        filtered = detections[
            (detections["class"].isin(active_classes)) &
            (detections["confidence"] >= conf_threshold)
        ].sort_values("timestamp", ascending=False)

        st.table(filtered)
        st.caption(f"Showing {len(filtered)} of {len(detections)} detections (confidence ≥ {conf_threshold:.0%})")


    # ┌──────────────────────────────────────────────────────────────┐
    # │  TAB 4 — Analytics                                           │
    # └──────────────────────────────────────────────────────────────┘
    with tab_analytics:
        history = get_telemetry_history(30)

        CHART_LAYOUT = dict(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#8b949e",
            title_font_color="#e6edf3",
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
            margin=dict(l=10, r=10, t=40, b=10),
        )

        a1, a2 = st.columns(2)

        with a1:
            fig = px.area(history, x="timestamp", y="altitude", title="Altitude (m)",
                           color_discrete_sequence=["#00ff88"])
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with a2:
            fig = px.line(history, x="timestamp", y="speed", title="Speed (m/s)",
                          color_discrete_sequence=["#4d96ff"])
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        a3, a4 = st.columns(2)

        with a3:
            fig = px.area(history, x="timestamp", y="battery", title="Battery (%)",
                          color_discrete_sequence=["#ffd93d"])
            fig.update_layout(**CHART_LAYOUT)
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        with a4:
            fig = px.line(history, x="timestamp", y="temperature", title="Temperature (°C)",
                          color_discrete_sequence=["#ff6b6b"])
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        # -- detection trend (bar) --------------------------------------
        st.markdown('<div class="section-title">🎯 Detection Counts by Class</div>', unsafe_allow_html=True)
        bar_data = pd.DataFrame({
            "Class": list(summary.keys()),
            "Count": list(summary.values()),
        })
        fig_bar = px.bar(
            bar_data, x="Class", y="Count",
            color="Class",
            color_discrete_map=DETECTION_COLORS,
            title="Current Session Detections",
        )
        fig_bar.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)


    # ┌──────────────────────────────────────────────────────────────┐
    # │  TAB 5 — System                                              │
    # └──────────────────────────────────────────────────────────────┘
    with tab_system:
        s1, s2 = st.columns(2)

        with s1:
            st.markdown('<div class="section-title">🖥️ Hardware Info</div>', unsafe_allow_html=True)
            hw = pd.DataFrame({
                "Component": [
                    "Processor", "AI Accelerator", "Camera",
                    "Flight Controller", "Battery", "Max Altitude", "Max Speed",
                ],
                "Specification": [
                    "Raspberry Pi 5 (8 GB)",
                    DRONE_CONFIG["ai_accelerator"],
                    DRONE_CONFIG["camera"],
                    DRONE_CONFIG["flight_controller"],
                    f"{DRONE_CONFIG['battery_capacity']} mAh",
                    f"{DRONE_CONFIG['max_altitude']} m",
                    f"{DRONE_CONFIG['max_speed']} m/s",
                ],
            })
            st.table(hw)

        with s2:
            st.markdown('<div class="section-title">📊 System Resources</div>', unsafe_allow_html=True)
            cpu_usage = np.random.randint(35, 65)
            ram_usage = np.random.randint(40, 70)
            disk_usage = np.random.randint(25, 50)
            gpu_usage = np.random.randint(50, 85)

            res1, res2 = st.columns(2)
            res1.metric("CPU", f"{cpu_usage}%")
            res2.metric("RAM", f"{ram_usage}%")

            res3, res4 = st.columns(2)
            res3.metric("Disk", f"{disk_usage}%")
            res4.metric("Hailo NPU", f"{gpu_usage}%")

            st.markdown("")
            st.markdown('<div class="section-title">📡 Network</div>', unsafe_allow_html=True)
            net = pd.DataFrame({
                "Metric": ["Latency", "Bandwidth", "Packet Loss", "Protocol"],
                "Value": [
                    f"{np.random.randint(8, 35)} ms",
                    f"{np.random.uniform(5, 15):.1f} Mbps",
                    f"{np.random.uniform(0, 0.5):.2f}%",
                    "MAVLink v2 / UDP",
                ],
            })
            st.table(net)


    # ╔══════════════════════════════════════════════════════════════╗
    # ║  AUTO REFRESH                                                ║
    # ╚══════════════════════════════════════════════════════════════╝
    if auto_refresh:
        time.sleep(REFRESH_INTERVAL)
        st.rerun()

if __name__ == "__main__":
    main()
