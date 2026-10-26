import io
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from utils.preprocess import load_data, preprocess_data
from utils.health_score import (
    calculate_health_score,
    get_health_category,
    get_maintenance_recommendation,
    diagnose_failure_cause,
)

# ---------------------------------------------------------
# Page Configuration & Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PredicTech AI | Smart Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Industrial Dark Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .app-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    /* Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #38bdf8;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-healthy {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-warning {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Diagnostics Callout Box */
    .diagnostic-box {
        background: rgba(15, 23, 42, 0.85);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .diagnostic-title {
        color: #f87171;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }
    .diagnostic-desc {
        color: #cbd5e1;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model & Columns
# ---------------------------------------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join(CURRENT_DIR, 'models', 'random_forest_model.pkl')
    cols_path = os.path.join(CURRENT_DIR, 'models', 'model_columns.pkl')
    if os.path.exists(model_path) and os.path.exists(cols_path):
        return joblib.load(model_path), joblib.load(cols_path)
    return None, None

model, model_columns = get_model()

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
@st.cache_data
def get_data(uploaded_file):
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    else:
        data_path = os.path.join(CURRENT_DIR, 'data', 'machine_data.csv')
        if not os.path.exists(data_path):
            fallback = os.path.join(CURRENT_DIR, '..', 'predictive_a_2', 'data', 'ai4i2020.csv')
            if os.path.exists(fallback):
                data_path = fallback
            else:
                st.error("No dataset found. Please upload a CSV or generate data.")
                st.stop()
        raw = load_data(data_path)

    # Preprocess
    proc = preprocess_data(raw)
    proc['health_score'] = proc.apply(calculate_health_score, axis=1)
    proc['health_status'] = proc['health_score'].apply(get_health_category)
    return proc

# ---------------------------------------------------------
# Visual Gauge Helpers
# ---------------------------------------------------------
def create_gauge(value, title, min_val=0, max_val=100, threshold_green=(70, 100), threshold_amber=(40, 70), suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"family": "JetBrains Mono", "size": 32, "color": "#f8fafc"}},
        title={"text": title, "font": {"family": "Inter", "size": 15, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": "#64748b"},
            "bar": {"color": "#38bdf8", "thickness": 0.28},
            "bgcolor": "#1e293b",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [min_val, threshold_amber[0]], "color": "rgba(239, 68, 68, 0.25)"},
                {"range": [threshold_amber[0], threshold_green[0]], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [threshold_green[0], max_val], "color": "rgba(16, 185, 129, 0.25)"}
            ]
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"}
    )
    return fig

def create_risk_dial(probability):
    pct = probability * 100
    bar_color = "#10b981" if pct < 35 else ("#f59e0b" if pct < 70 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"family": "JetBrains Mono", "size": 36, "color": bar_color}},
        title={"text": "FAILURE RISK PROBABILITY", "font": {"family": "Inter", "size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748b"},
            "bar": {"color": bar_color, "thickness": 0.32},
            "bgcolor": "#1e293b",
            "borderwidth": 1,
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [35, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.25)"}
            ]
        }
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=45, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"}
    )
    return fig

# ---------------------------------------------------------
# Sidebar Controls & Navigation
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h2 style="margin:0; font-weight:800; color:#38bdf8; font-size: 1.4rem;">PREDICTECH AI</h2>
    <p style="margin:0; font-size:0.8rem; color:#64748b;">Industrial Predictive Maintenance</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "🌐 Fleet Overview",
        "🔮 AI Failure Prediction",
        "🧪 'What-If' Digital Twin",
        "🔍 Machine Health Deep Dive",
        "📉 Downtime & Telemetry",
        "📁 Batch CSV Diagnostics"
    ],
    index=0
)

# Upload CSV option
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload Telemetry CSV", type=["csv"])
df = get_data(uploaded_file)

# Machine ID formatting
if "machine_id" not in df.columns:
    df["machine_id"] = ["M_" + str(i).zfill(4) for i in range(1, len(df) + 1)]

# Variant filter
if "machine_type" in df.columns:
    st.sidebar.markdown("### 🎛️ Variant Filter")
    types = sorted(df["machine_type"].unique().tolist())
    selected_types = st.sidebar.multiselect("Filter by Variant", options=types, default=types)
    filtered_df = df[df["machine_type"].isin(selected_types)].copy()
else:
    filtered_df = df.copy()

if filtered_df.empty:
    st.warning("No machines found for the selected filter.")
    st.stop()

# Machine Selector
st.sidebar.markdown("### 🏷️ Active Unit")
machine_list = filtered_df["machine_id"].tolist()
selected_machine = st.sidebar.selectbox("Select Machine ID", machine_list, index=0)

selected_row = filtered_df[filtered_df["machine_id"] == selected_machine].iloc[0]

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="app-title">Smart Predictive Maintenance Dashboard</h1>
            <div class="app-subtitle">IoT Sensor Telemetry • Physics-Informed ML • Root Cause Diagnostics</div>
        </div>
        <div style="text-align: right;">
            <span class="status-badge badge-healthy">● System Active</span>
            <div style="color: #64748b; font-size: 0.78rem; margin-top: 5px;">Model: Random Forest Ensemble</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 1: FLEET OVERVIEW
# ---------------------------------------------------------
if page == "🌐 Fleet Overview":
    total_m = len(filtered_df)
    avg_health = filtered_df["health_score"].mean()
    total_down = filtered_df["downtime"].sum()
    failure_count = int(filtered_df["failure"].sum()) if "failure" in filtered_df.columns else 0
    crit_count = (filtered_df["health_status"] == "Critical").sum()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Monitored Fleet Units</div>
            <div class="metric-value">{total_m:,}</div>
            <div class="metric-subtext">Active telemetry streams</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        health_color = "#34d399" if avg_health >= 80 else ("#fbbf24" if avg_health >= 60 else "#f87171")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mean Fleet Health</div>
            <div class="metric-value" style="color: {health_color};">{avg_health:.1f}<span style="font-size:1.1rem;">/100</span></div>
            <div class="metric-subtext">Aggregated sensor index</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accumulated Downtime</div>
            <div class="metric-value">{total_down:,.0f} <span style="font-size:1.1rem;">min</span></div>
            <div class="metric-subtext">{total_down/60:,.1f} operational hours lost</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Critical / Breakdown Units</div>
            <div class="metric-value" style="color: #f87171;">{failure_count:,} <span style="font-size:1rem; color:#94a3b8;">({failure_count/total_m*100:.1f}%)</span></div>
            <div class="metric-subtext">{crit_count} units in Critical status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("##### 📊 Fleet Health Score Distribution")
        health_hist = px.histogram(
            filtered_df,
            x="health_score",
            color="health_status",
            nbins=30,
            color_discrete_map={"Healthy": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"},
            template="plotly_dark",
            height=300
        )
        health_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Health Score (0-100)",
            yaxis_title="Unit Count"
        )
        st.plotly_chart(health_hist, use_container_width=True)

    with c2:
        st.markdown("##### 🚨 Machine Status Breakdown")
        pie_df = filtered_df["health_status"].value_counts().reset_index()
        pie_df.columns = ["Status", "Count"]
        donut_fig = px.pie(
            pie_df,
            names="Status",
            values="Count",
            hole=0.55,
            color="Status",
            color_discrete_map={"Healthy": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"},
            template="plotly_dark",
            height=300
        )
        donut_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    # High-Risk Spotlight
    st.markdown("---")
    st.markdown("### ⚠️ High-Risk Triage Spotlight (Urgent Action)")
    high_risk_df = filtered_df[
        (filtered_df["health_status"] == "Critical") | (filtered_df.get("failure", 0) == 1)
    ].sort_values(by="health_score", ascending=True).head(15)

    if not high_risk_df.empty:
        cols_to_show = [c for c in ["machine_id", "machine_type", "health_score", "health_status", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear", "downtime"] if c in high_risk_df.columns]
        st.dataframe(high_risk_df[cols_to_show], use_container_width=True)
    else:
        st.success("All machines are currently operating within nominal safety thresholds.")

# ---------------------------------------------------------
# PAGE 2: AI FAILURE PREDICTION
# ---------------------------------------------------------
elif page == "🔮 AI Failure Prediction":
    st.markdown(f"### 🤖 Real-Time AI Diagnostics: Machine **{selected_machine}**")

    # Predict using model
    if model is not None and model_columns is not None:
        input_data = pd.DataFrame([selected_row])
        features_needed = [c for c in model_columns if c in input_data.columns]
        model_input = input_data[features_needed]

        prob = float(model.predict_proba(model_input)[0][1])
        pred = int(prob >= 0.40)
    else:
        prob = 0.15
        pred = 0

    causes = diagnose_failure_cause(selected_row.to_dict())

    if prob >= 0.70:
        priority = "URGENT (Tier 1)"
        priority_color = "#ef4444"
        action_time = "Immediate shutdown or inspection within 24 hours"
        status_text = "FAILURE IMMINENT"
    elif prob >= 0.40:
        priority = "ELEVATED (Tier 2)"
        priority_color = "#f59e0b"
        action_time = "Schedule maintenance inspection within 3 shifts"
        status_text = "POTENTIAL RISK"
    else:
        priority = "NORMAL (Tier 3)"
        priority_color = "#10b981"
        action_time = "Maintain routine preventative maintenance cycle (7-14 days)"
        status_text = "OPERATIONAL NOMINAL"

    c1, c2, c3 = st.columns([2, 2, 3])

    with c1:
        st.plotly_chart(create_risk_dial(prob), use_container_width=True)

    with c2:
        st.markdown(f"""
        <div style="background:#1e293b; border:1px solid #334155; border-radius:10px; padding:20px; height:240px; display:flex; flex-direction:column; justify-content:center;">
            <div class="metric-label">PREDICTIVE VERDICT</div>
            <div style="font-size:1.6rem; font-weight:800; color:{priority_color}; margin: 6px 0;">{status_text}</div>
            <div class="metric-label" style="margin-top:10px;">ACTION PRIORITY</div>
            <div style="font-size:1.15rem; font-weight:700; color:{priority_color};">{priority}</div>
            <div style="font-size:0.8rem; color:#94a3b8; margin-top:6px;">{action_time}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("##### 🔬 Sensor Snapshot")
        cur_row = selected_row
        st.markdown(f"""
        - **Variant:** `{cur_row.get('machine_type', 'N/A')}` | **Tool Wear:** `{cur_row.get('tool_wear', 0):.0f} min`
        - **Process Temp:** `{cur_row.get('process_temperature', 310):.1f} K` | **Air Temp:** `{cur_row.get('air_temperature', 300):.1f} K` (ΔT = `{cur_row.get('temp_diff', 10):.1f} K`)
        - **Spindle Speed:** `{cur_row.get('rotational_speed', 1500):.0f} RPM` | **Torque:** `{cur_row.get('torque', 40):.1f} Nm`
        - **Calculated Power:** `{cur_row.get('power', 0):.0f} Watts` | **Overstrain:** `{cur_row.get('overstrain', 0):.0f} min·Nm`
        """)

    st.markdown("---")

    # Root Cause Diagnostic Engine
    st.markdown("### 🔍 Root Cause Diagnostic Engine")
    if causes:
        for c in causes:
            badge_class = "badge-critical" if c["severity"] == "Critical" else ("badge-warning" if c["severity"] == "High" else "badge-healthy")
            st.markdown(f"""
            <div class="diagnostic-box" style="border-left-color: {'#ef4444' if c['severity']=='Critical' else '#f59e0b'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="diagnostic-title">{c['mode']}</span>
                    <span class="status-badge {badge_class}">{c['severity'].upper()}</span>
                </div>
                <div class="diagnostic-desc">{c['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:16px; color:#34d399;">
            ✅ <b>All Physical Safety Bounds Cleared:</b> No heat dissipation bottlenecks, power spikes, or mechanical overstrain detected for this machine.
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 3: INTERACTIVE "WHAT-IF" DIGITAL TWIN
# ---------------------------------------------------------
elif page == "🧪 'What-If' Digital Twin":
    st.markdown("### 🧪 Machine Digital Twin & Scenario Simulator")
    st.caption("Adjust physical sensor parameters in real time to simulate operating stress, test machine limits, and predict failures.")

    # Preset buttons
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    preset = None
    if preset_col1.button("🟢 Nominal Healthy Unit", use_container_width=True):
        preset = "nominal"
    if preset_col2.button("🔥 Heat Dissipation Stress", use_container_width=True):
        preset = "hdf"
    if preset_col3.button("⚡ Spindle Power Overload", use_container_width=True):
        preset = "pwf"
    if preset_col4.button("💥 Mechanical Overstrain", use_container_width=True):
        preset = "osf"

    if preset == "nominal":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "Type_A", 298.5, 309.2, 1550, 40.0, 60.0
    elif preset == "hdf":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "Type_A", 302.5, 310.0, 1310, 48.0, 120.0
    elif preset == "pwf":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "Type_B", 298.0, 308.0, 2650, 68.0, 80.0
    elif preset == "osf":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "Type_C", 299.0, 309.0, 1420, 64.0, 215.0
    else:
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "Type_A", 298.2, 308.7, 1495, 42.5, 95.0

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("##### ⚙️ Machine Configuration")
        sim_type = st.selectbox("Product Quality Variant", ["Type_A", "Type_B", "Type_C"], index=0)
        sim_tool_wear = st.slider("Tool Wear [min]", 0.0, 260.0, float(d_wear), step=1.0)

    with c2:
        st.markdown("##### 🌡️ Thermal Parameters")
        sim_air_temp = st.slider("Air Temperature [K]", 295.0, 306.0, float(d_air), step=0.1)
        sim_proc_temp = st.slider("Process Temperature [K]", 305.0, 316.0, float(d_proc), step=0.1)

    with c3:
        st.markdown("##### 🌀 Mechanical Dynamics")
        sim_speed = st.slider("Rotational Speed [RPM]", 1100, 2900, int(d_speed), step=10)
        sim_torque = st.slider("Torque [Nm]", 10.0, 85.0, float(d_torque), step=0.5)

    # Physics calculations
    temp_diff = sim_proc_temp - sim_air_temp
    power_watts = sim_torque * (sim_speed * 2 * np.pi / 60)
    overstrain = sim_tool_wear * sim_torque
    temp_ratio = sim_proc_temp / sim_air_temp

    planned = 480.0
    actual = max(300.0, 480.0 - sim_tool_wear * 0.5)
    downtime = planned - actual
    availability = actual / planned
    operating_hours = actual / 60.0

    # Build row formatted to match model columns
    sim_dict = {
        "air_temperature": sim_air_temp,
        "process_temperature": sim_proc_temp,
        "rotational_speed": sim_speed,
        "torque": sim_torque,
        "tool_wear": sim_tool_wear,
        "operating_hours": 1000.0,
        "planned_production_time": planned,
        "actual_running_time": actual,
        "jobs_completed": 350,
        "downtime": downtime,
        "availability": availability,
        "temp_diff": temp_diff,
        "power": power_watts,
        "overstrain": overstrain,
        "temp_ratio": temp_ratio,
        "machine_type_encoded": 0 if sim_type == "Type_A" else (1 if sim_type == "Type_B" else 2)
    }

    if model is not None and model_columns is not None:
        sim_df = pd.DataFrame([sim_dict])
        features_in = [c for c in model_columns if c in sim_df.columns]
        sim_prob = float(model.predict_proba(sim_df[features_in])[0][1])
    else:
        sim_prob = 0.20

    sim_causes = diagnose_failure_cause(sim_dict)

    st.markdown("---")
    res_c1, res_c2 = st.columns([1, 2])

    with res_c1:
        st.plotly_chart(create_risk_dial(sim_prob), use_container_width=True)

    with res_c2:
        st.markdown("##### 📐 Real-Time Physics Telemetry")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("Thermal Delta (ΔT)", f"{temp_diff:.2f} K", delta="Danger < 8.6 K" if temp_diff < 8.6 else "Normal", delta_color="inverse")
        p_col2.metric("Mechanical Power", f"{power_watts:.0f} W", delta="Safe: 3.5k - 9k W")
        p_col3.metric("Overstrain Product", f"{overstrain:.0f} min·Nm", delta="Safety limit: 11k+")

        st.markdown("<br>", unsafe_allow_html=True)
        if sim_causes:
            for c in sim_causes:
                badge_class = "badge-critical" if c["severity"] == "Critical" else "badge-warning"
                st.markdown(f"""
                <div class="diagnostic-box" style="border-left-color: {'#ef4444' if c['severity']=='Critical' else '#f59e0b'};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="diagnostic-title">{c['mode']}</span>
                        <span class="status-badge {badge_class}">{c['severity'].upper()}</span>
                    </div>
                    <div class="diagnostic-desc">{c['detail']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:14px; color:#34d399;">
                ✅ <b>Safe Operating Envelope:</b> Simulator indicates no immediate physical threat to machinery under these conditions.
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 4: MACHINE HEALTH DEEP DIVE
# ---------------------------------------------------------
elif page == "🔍 Machine Health Deep Dive":
    st.markdown(f"### 🩺 Deep Health Diagnostics: Unit **{selected_machine}**")

    m_row = selected_row

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(create_gauge(
            m_row["health_score"],
            "MACHINE HEALTH SCORE",
            min_val=0,
            max_val=100,
            threshold_green=(80, 100),
            threshold_amber=(60, 80),
            suffix="/100"
        ), use_container_width=True)

    with g2:
        avail_pct = float(m_row.get("availability", 0.95)) * 100
        st.plotly_chart(create_gauge(
            avail_pct,
            "OPERATIONAL AVAILABILITY",
            min_val=0,
            max_val=100,
            threshold_green=(85, 100),
            threshold_amber=(70, 85),
            suffix="%"
        ), use_container_width=True)

    st.markdown("##### 📊 Telemetry Relative to Operating Limits")

    params_df = pd.DataFrame({
        "Sensor": ["Air Temp [K]", "Process Temp [K]", "Spindle Speed [RPM]", "Torque [Nm]", "Tool Wear [min]"],
        "Value": [
            m_row.get("air_temperature", 300),
            m_row.get("process_temperature", 310),
            m_row.get("rotational_speed", 1500),
            m_row.get("torque", 40),
            m_row.get("tool_wear", 0)
        ],
        "Fleet Average": [
            filtered_df["air_temperature"].mean(),
            filtered_df["process_temperature"].mean(),
            filtered_df["rotational_speed"].mean(),
            filtered_df["torque"].mean(),
            filtered_df["tool_wear"].mean()
        ]
    })

    sensor_bar = px.bar(
        params_df,
        x="Sensor",
        y=["Value", "Fleet Average"],
        barmode="group",
        template="plotly_dark",
        height=320,
        color_discrete_sequence=["#38bdf8", "#64748b"]
    )
    sensor_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=20, r=20, t=20, b=20),
        legend_title="Series"
    )
    st.plotly_chart(sensor_bar, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: DOWNTIME & TELEMETRY
# ---------------------------------------------------------
elif page == "📉 Downtime & Telemetry":
    st.markdown("### 📉 Plant-Wide Downtime & Sensor Telemetry")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### ⏱️ Top 15 Units by Accumulated Downtime")
        top_down = filtered_df.nlargest(15, "downtime")
        down_fig = px.bar(
            top_down,
            x="machine_id",
            y="downtime",
            color="downtime",
            color_continuous_scale="Reds",
            template="plotly_dark",
            height=320
        )
        down_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=20, t=10, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(down_fig, use_container_width=True)

    with c2:
        st.markdown("##### 🔬 Tool Wear vs. Downtime Correlation")
        scatter_fig = px.scatter(
            filtered_df,
            x="tool_wear",
            y="downtime",
            color="health_status",
            color_discrete_map={"Healthy": "#10b981", "Warning": "#f59e0b", "Critical": "#ef4444"},
            template="plotly_dark",
            height=320,
            hover_data=["machine_id"]
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            margin=dict(l=20, r=20, t=10, b=20)
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 6: BATCH CSV DIAGNOSTICS
# ---------------------------------------------------------
elif page == "📁 Batch CSV Diagnostics":
    st.markdown("### 📁 Batch Fleet Telemetry File Diagnostics")
    st.caption("Upload a factory sensor log CSV to run automated failure predictions, root-cause diagnostics, and priority scoring across all rows.")

    batch_file = st.file_uploader("Upload CSV File to Inspect", type=["csv"], key="batch_diag")

    if batch_file is not None:
        try:
            batch_raw = pd.read_csv(batch_file)
            st.success(f"Successfully loaded {len(batch_raw):,} records from {batch_file.name}")

            batch_proc = preprocess_data(batch_raw)
            if model is not None and model_columns is not None:
                features_needed = [c for c in model_columns if c in batch_proc.columns]
                probs = model.predict_proba(batch_proc[features_needed])[:, 1]
                preds = (probs >= 0.40).astype(int)

                batch_raw["failure_risk_pct"] = np.round(probs * 100, 1)
                batch_raw["predicted_failure"] = preds
                batch_raw["priority"] = np.where(probs >= 0.70, "Urgent (24h)", np.where(probs >= 0.40, "Elevated (3d)", "Normal (Routine)"))

                fail_count = int(sum(preds))
                b1, b2, b3 = st.columns(3)
                b1.metric("Total Rows Evaluated", f"{len(batch_raw):,}")
                b2.metric("Predicted Breakdowns", f"{fail_count:,}", delta=f"{fail_count/len(batch_raw)*100:.1f}% risk rate", delta_color="inverse")
                b3.metric("Urgent Actions Required", f"{sum(probs >= 0.70):,}", delta_color="inverse")

                st.markdown("##### 📋 Flagged High-Risk Units")
                flagged = batch_raw[batch_raw["predicted_failure"] == 1].sort_values(by="failure_risk_pct", ascending=False)
                st.dataframe(flagged.head(50), use_container_width=True)

                csv_buffer = io.StringIO()
                batch_raw.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Diagnostic Audit (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="predictive_maintenance_audit.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Model not trained yet. Please train the model first.")

        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    else:
        st.info("💡 You can upload your own factory logs or test with the sample dataset located at `data/machine_data.csv`.")
