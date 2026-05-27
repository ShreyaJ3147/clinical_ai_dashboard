"""
Clinical AI Dashboard — Portfolio Demo
A GenAI pipeline for automated clinical trial data analysis.

Run:  streamlit run app.py
Env:  ANTHROPIC_API_KEY=sk-ant-... (or paste in sidebar)
"""

import os
import io
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_generator import generate_clinical_data
from pipeline import (
    clean_data,
    detect_anomalies_rules,
    explain_anomalies,
    generate_narrative,
    generate_chart_insights,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ClinicalAI — Trial Data Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal custom CSS ────────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #0f1117; }
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2d3045;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #4fc3f7; }
    .metric-card p  { margin: 0; color: #9e9e9e; font-size: 0.85rem; }
    .flag-high   { color: #ef5350; font-weight: 600; }
    .flag-medium { color: #ffa726; font-weight: 600; }
    .flag-low    { color: #66bb6a; font-weight: 600; }
    .insight-box {
        background: #12151f;
        border-left: 3px solid #4fc3f7;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.88rem;
        color: #b0bec5;
    }
    .step-badge {
        display: inline-block;
        background: #1e2236;
        border: 1px solid #3d4466;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.78rem;
        color: #90caf9;
        margin-bottom: 0.5rem;
    }
    .narrative-box {
        background: #12151f;
        border: 1px solid #2d3045;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        line-height: 1.8;
        color: #cfd8dc;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "df": None,
    "clean_df": None,
    "clean_issues": [],
    "flags_df": None,
    "narrative": None,
    "insights": None,
    "analysis_done": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧬 ClinicalAI")
    st.caption("Automated trial data analysis powered by Claude")
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )

    st.divider()
    st.markdown("**Pipeline steps**")
    steps = ["① Upload data", "② Clean & validate", "③ Detect anomalies",
             "④ LLM explanations", "⑤ Narrative & insights"]
    for s in steps:
        st.caption(s)

    st.divider()
    st.markdown("**Study**")
    st.caption("Phase II Oncology — Drug A")
    st.caption("3 arms · 4 visits · 6 lab tests")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Clinical Trial AI Analyzer")
st.markdown(
    "Upload raw clinical trial data and let the AI pipeline clean, analyze, "
    "flag anomalies, and generate a regulatory-grade narrative — in seconds."
)
st.divider()

# ── Upload section ────────────────────────────────────────────────────────────

col_upload, col_sample = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload trial data (CSV)",
        type=["csv"],
        help="Expected columns: patient_id, treatment_arm, visit, lab_test, lab_value, …"
    )

with col_sample:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔬 Use sample trial data", use_container_width=True):
        with st.spinner("Generating synthetic Phase II dataset…"):
            st.session_state.df = generate_clinical_data(n_patients=120)
            st.session_state.analysis_done = False
        st.success("Sample data loaded — 120 patients, 4 visits, 6 lab tests")

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded, parse_dates=["visit_date"])
    st.session_state.analysis_done = False

# ── Preview loaded data ───────────────────────────────────────────────────────

if st.session_state.df is not None:
    df = st.session_state.df
    st.markdown("### Dataset preview")

    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        (df["patient_id"].nunique(), "Patients"),
        (df["treatment_arm"].nunique(), "Treatment arms"),
        (len(df[df["record_type"] == "lab"]), "Lab records"),
        (len(df[df["record_type"] == "ae"]), "Adverse events"),
    ]
    for col, (val, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><h2>{val:,}</h2><p>{label}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("View raw data", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)

    # ── Run pipeline button ───────────────────────────────────────────────────

    st.divider()
    if not api_key:
        st.warning("⚠️ Add your Anthropic API key in the sidebar to run the AI pipeline.")
    else:
        if st.button("🚀 Run AI Analysis Pipeline", type="primary",
                     use_container_width=True,
                     disabled=st.session_state.analysis_done):
            run_pipeline(df, api_key)

    # ── Results tabs ──────────────────────────────────────────────────────────

    if st.session_state.analysis_done:
        st.divider()
        tab_anomalies, tab_narrative, tab_dashboard, tab_export = st.tabs([
            "🚨 Anomaly Flags",
            "📄 Clinical Narrative",
            "📊 Dashboard",
            "⬇️ Export",
        ])

        render_anomalies(tab_anomalies)
        render_narrative(tab_narrative)
        render_dashboard(tab_dashboard)
        render_export(tab_export)

else:
    # Empty state
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.info(
        "👆 Upload a CSV file or click **Use sample trial data** to get started.",
        icon="🧬",
    )


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(df: pd.DataFrame, api_key: str):
    progress = st.progress(0, text="Starting pipeline…")
    status   = st.empty()

    # Step 1 — Clean
    status.markdown('<span class="step-badge">① Cleaning & validating data</span>',
                    unsafe_allow_html=True)
    clean_df, issues = clean_data(df)
    st.session_state.clean_df     = clean_df
    st.session_state.clean_issues = issues
    progress.progress(20, text="Data cleaning complete")
    time.sleep(0.4)

    # Step 2 — Rule-based anomaly detection
    status.markdown('<span class="step-badge">② Detecting anomalies (rule engine)</span>',
                    unsafe_allow_html=True)
    flags_df = detect_anomalies_rules(clean_df)
    progress.progress(40, text=f"Found {len(flags_df)} rule-based flags")
    time.sleep(0.4)

    # Step 3 — LLM explanations
    status.markdown('<span class="step-badge">③ Generating clinical explanations with Claude</span>',
                    unsafe_allow_html=True)

    def llm_progress(i, total):
        pct = 40 + int((i / total) * 30)
        progress.progress(pct, text=f"Claude explaining flag {i+1}/{total}…")

    flags_df = explain_anomalies(flags_df, api_key, progress_callback=llm_progress)
    st.session_state.flags_df = flags_df
    progress.progress(70, text="Anomaly explanations complete")

    # Step 4 — Narrative
    status.markdown('<span class="step-badge">④ Writing ICH E3 narrative with Claude</span>',
                    unsafe_allow_html=True)
    narrative = generate_narrative(clean_df, flags_df, api_key)
    st.session_state.narrative = narrative
    progress.progress(85, text="Narrative generated")

    # Step 5 — Chart insights
    status.markdown('<span class="step-badge">⑤ Generating dashboard insights</span>',
                    unsafe_allow_html=True)
    insights = generate_chart_insights(clean_df, api_key)
    st.session_state.insights = insights
    progress.progress(100, text="Pipeline complete ✅")

    st.session_state.analysis_done = True
    status.empty()
    progress.empty()
    st.rerun()


# ── Anomaly tab ───────────────────────────────────────────────────────────────

def render_anomalies(tab):
    with tab:
        flags_df = st.session_state.flags_df
        issues   = st.session_state.clean_issues

        # Data quality summary
        st.markdown("#### Data quality")
        for issue in issues:
            st.caption(f"• {issue}")

        st.markdown("#### Flagged records")

        if flags_df is None or flags_df.empty:
            st.success("No anomalies detected.")
            return

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total flags", len(flags_df))
        with c2:
            st.metric("High severity", len(flags_df[flags_df["severity"] == "High"]),
                      delta="⚠️ Review required", delta_color="off")
        with c3:
            st.metric("Patients affected",
                      flags_df["patient_id"].nunique())

        st.markdown("<br>", unsafe_allow_html=True)

        # Severity filter
        severity_filter = st.multiselect(
            "Filter by severity",
            options=["High", "Medium", "Low"],
            default=["High", "Medium"],
        )
        filtered = flags_df[flags_df["severity"].isin(severity_filter)]

        # Styled table
        def style_severity(val):
            colors = {"High": "#ef5350", "Medium": "#ffa726", "Low": "#66bb6a"}
            return f"color: {colors.get(val, 'white')}; font-weight: 600"

        display_cols = ["patient_id", "treatment_arm", "visit",
                        "flag_type", "severity", "detail"]
        styled = filtered[display_cols].style.applymap(
            style_severity, subset=["severity"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Detail expanders with LLM explanations
        st.markdown("#### Claude's clinical interpretations")
        has_explanations = filtered[filtered["llm_explanation"].notna() &
                                    (filtered["llm_explanation"] != "Pending review")]

        for _, row in has_explanations.iterrows():
            with st.expander(
                f"**{row['patient_id']}** — {row['flag_type']} — {row['visit']}",
                expanded=False,
            ):
                st.caption(f"**Finding:** {row['detail']}")
                st.markdown(f"**Clinical interpretation:**\n\n{row['llm_explanation']}")


# ── Narrative tab ─────────────────────────────────────────────────────────────

def render_narrative(tab):
    with tab:
        narrative = st.session_state.narrative
        if not narrative:
            st.info("Narrative not yet generated.")
            return

        st.markdown("#### ICH E3-style study narrative")
        st.caption("Generated by Claude · For review purposes only · Not for regulatory submission")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="narrative-box">{narrative.replace(chr(10), "<br><br>")}</div>',
            unsafe_allow_html=True,
        )


# ── Dashboard tab ─────────────────────────────────────────────────────────────

def render_dashboard(tab):
    with tab:
        df       = st.session_state.clean_df
        insights = st.session_state.insights or {}

        lab_df = df[df["record_type"] == "lab"]
        ae_df  = df[df["record_type"] == "ae"]

        COLORS = {
            "Placebo":        "#78909c",
            "Drug A 50mg":    "#42a5f5",
            "Drug A 100mg":   "#ef5350",
        }

        # ── Chart 1: ALT trend over time ─────────────────────────────────────
        st.markdown("#### ALT over time by treatment arm")
        if insights.get("alt_trend"):
            st.markdown(
                f'<div class="insight-box">💡 {insights["alt_trend"]}</div>',
                unsafe_allow_html=True,
            )

        alt_df = (
            lab_df[lab_df["lab_test"] == "ALT"]
            .groupby(["treatment_arm", "visit"])["lab_value"]
            .mean().reset_index()
        )
        visit_order = ["Baseline", "Week 4", "Week 8", "Week 12"]
        alt_df["visit"] = pd.Categorical(alt_df["visit"], categories=visit_order, ordered=True)
        alt_df = alt_df.sort_values("visit")

        fig1 = px.line(
            alt_df, x="visit", y="lab_value", color="treatment_arm",
            markers=True,
            color_discrete_map=COLORS,
            labels={"lab_value": "Mean ALT (U/L)", "visit": "Visit", "treatment_arm": "Arm"},
        )
        fig1.add_hline(y=56, line_dash="dot", line_color="#ffa726",
                       annotation_text="ULN (56 U/L)", annotation_position="bottom right")
        fig1.add_hline(y=168, line_dash="dot", line_color="#ef5350",
                       annotation_text="3× ULN", annotation_position="top right")
        fig1.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#cfd8dc", legend_title_text="Arm",
            yaxis=dict(gridcolor="#1e2236"), xaxis=dict(gridcolor="#1e2236"),
        )
        st.plotly_chart(fig1, use_container_width=True)

        # ── Chart 2 & 3 side by side ─────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Adverse events by arm")
            if insights.get("ae_distribution"):
                st.markdown(
                    f'<div class="insight-box">💡 {insights["ae_distribution"]}</div>',
                    unsafe_allow_html=True,
                )
            ae_counts = ae_df.groupby("treatment_arm").size().reset_index(name="count")
            fig2 = px.bar(
                ae_counts, x="treatment_arm", y="count", color="treatment_arm",
                color_discrete_map=COLORS,
                labels={"treatment_arm": "Arm", "count": "AE count"},
            )
            fig2.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#cfd8dc", showlegend=False,
                yaxis=dict(gridcolor="#1e2236"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_right:
            st.markdown("#### AE severity by arm")
            if insights.get("severity_breakdown"):
                st.markdown(
                    f'<div class="insight-box">💡 {insights["severity_breakdown"]}</div>',
                    unsafe_allow_html=True,
                )
            sev_counts = ae_df.groupby(["treatment_arm", "ae_severity"]).size().reset_index(name="count")
            fig3 = px.bar(
                sev_counts, x="treatment_arm", y="count", color="ae_severity",
                barmode="stack",
                color_discrete_map={
                    "Grade 1": "#66bb6a", "Grade 2": "#ffa726",
                    "Grade 3": "#ef5350", "Grade 4": "#b71c1c",
                },
                labels={"treatment_arm": "Arm", "count": "Count", "ae_severity": "Severity"},
            )
            fig3.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#cfd8dc",
                yaxis=dict(gridcolor="#1e2236"),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # ── Chart 4: Top AE types ─────────────────────────────────────────────
        st.markdown("#### Most common adverse events")
        if insights.get("top_ae_types"):
            st.markdown(
                f'<div class="insight-box">💡 {insights["top_ae_types"]}</div>',
                unsafe_allow_html=True,
            )
        top_ae = (
            ae_df.groupby(["adverse_event", "treatment_arm"])
            .size().reset_index(name="count")
        )
        top_events = ae_df["adverse_event"].value_counts().head(8).index.tolist()
        top_ae = top_ae[top_ae["adverse_event"].isin(top_events)]

        fig4 = px.bar(
            top_ae, x="count", y="adverse_event", color="treatment_arm",
            orientation="h", barmode="group",
            color_discrete_map=COLORS,
            labels={"adverse_event": "", "count": "Count", "treatment_arm": "Arm"},
        )
        fig4.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#cfd8dc", yaxis=dict(autorange="reversed"),
            xaxis=dict(gridcolor="#1e2236"),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # ── Chart 5: Lab heatmap ──────────────────────────────────────────────
        st.markdown("#### Lab value heatmap — mean by test and arm")
        pivot = (
            lab_df.groupby(["lab_test", "treatment_arm"])["lab_value"]
            .mean().round(1).unstack()
        )
        fig5 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            text=pivot.values.round(1),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig5.update_layout(
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#cfd8dc",
        )
        st.plotly_chart(fig5, use_container_width=True)


# ── Export tab ────────────────────────────────────────────────────────────────

def render_export(tab):
    with tab:
        st.markdown("#### Export results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if st.session_state.flags_df is not None:
                csv = st.session_state.flags_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download anomaly flags (CSV)",
                    data=csv,
                    file_name="anomaly_flags.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        with col_b:
            if st.session_state.narrative:
                st.download_button(
                    "⬇️ Download narrative (TXT)",
                    data=st.session_state.narrative,
                    file_name="clinical_narrative.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with col_c:
            if st.session_state.clean_df is not None:
                csv = st.session_state.clean_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download cleaned data (CSV)",
                    data=csv,
                    file_name="cleaned_trial_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
