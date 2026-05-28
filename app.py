"""
PharmaAI -- Intelligent Data Analyzer
Streamlit dashboard for pharma data analysis powered by Claude.
"""

import os, time, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_generator import DATASET_REGISTRY
from pipeline import (
    clean_data, detect_data_context, detect_anomalies,
    explain_anomalies, generate_narrative, generate_insights,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaAI Analyzer",
    page_icon="diamond",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.01em; }

[data-testid="stSidebar"] { background: #05080f; border-right: 1px solid #111827; }
[data-testid="stSidebarContent"] { padding-top: 1.5rem; }

.stApp { background: #080d17; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px; }

.metric-card {
    background: #0d1424;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border-top: 2px solid #0ea5e9;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.72rem; font-weight: 500; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
.metric-card .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.metric-card .sub   { font-size: 0.75rem; color: #334155; margin-top: 0.35rem; }

.section-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #e2e8f0; letter-spacing: 0.02em; margin: 1.25rem 0 0.6rem; }

.ctx-badge {
    display: inline-block;
    background: #0c1f35;
    border: 1px solid #164e76;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #7dd3fc;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 1rem;
}

.insight-box {
    background: #0a1628;
    border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    line-height: 1.5;
}

.narrative-box {
    background: #0a0f1a;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    line-height: 1.9;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.narrative-box p { margin-bottom: 1.2rem; }

.step-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.55rem 0.9rem;
    border-radius: 8px;
    background: #0d1424;
    border: 1px solid #1e2d45;
    margin-bottom: 6px;
    font-size: 0.82rem;
    color: #64748b;
}
.step-row.active { border-color: #0ea5e9; background: #091a2a; color: #bae6fd; }
.step-row.done   { border-color: #10b981; background: #071a12; color: #6ee7b7; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: #1e2d45; flex-shrink: 0; }
.step-row.active .dot { background: #0ea5e9; }
.step-row.done   .dot { background: #10b981; }

[data-testid="stTabs"] [role="tablist"] { background: #0d1424; border-radius: 10px; padding: 4px; border: 1px solid #1e2d45; }
[data-testid="stTabs"] [role="tab"]     { color: #475569 !important; font-size: 0.84rem; border-radius: 8px; }
[data-testid="stTabs"] [aria-selected="true"] { background: #091a2a !important; color: #bae6fd !important; }

[data-testid="stExpander"] { background: #0d1424; border: 1px solid #1e2d45 !important; border-radius: 10px; }

.stButton > button { font-family: 'DM Sans', sans-serif !important; border-radius: 8px !important; font-size: 0.85rem !important; }

div[data-testid="stFileUploader"] { background: #0d1424; border: 1px dashed #1e2d45; border-radius: 10px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = dict(df=None, clean_df=None, clean_issues=[], context=None,
                flags_df=None, narrative=None, insights=None,
                analysis_done=False, selected_ds="clinical_trial")
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

CHART_COLORS = ["#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]

# ── Helper functions ───────────────────────────────────────────────────────────

def plot_base():
    return dict(
        plot_bgcolor="#080d17", paper_bgcolor="#080d17",
        font=dict(color="#94a3b8", family="DM Sans"),
        xaxis=dict(gridcolor="#111827", linecolor="#1e2d45"),
        yaxis=dict(gridcolor="#111827", linecolor="#1e2d45"),
        legend=dict(bgcolor="#0d1424", bordercolor="#1e2d45", borderwidth=1),
    )

def metric_card(label, value, sub="", accent="#0ea5e9"):
    return (f'<div class="metric-card" style="border-top-color:{accent}">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>')

def show_metric_row(cards):
    cols = st.columns(len(cards))
    for col, (lbl, val, sub, acc) in zip(cols, cards):
        col.markdown(metric_card(lbl, val, sub, acc), unsafe_allow_html=True)

# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_pipeline(df, api_key):
    step_labels = [
        "Cleaning & validating data",
        "Auto-detecting data type",
        "Running statistical analysis",
        "Generating AI explanations",
        "Writing narrative & insights",
    ]
    prog = st.progress(0)
    stat = st.empty()

    def show_steps(current):
        html = '<div style="margin:1rem 0">'
        for j, lbl in enumerate(step_labels):
            if j < current:
                cls = "step-row done"
                prefix = "Done  "
            elif j == current:
                cls = "step-row active"
                prefix = ""
            else:
                cls = "step-row"
                prefix = ""
            html += f'<div class="{cls}"><div class="dot"></div>{prefix}{lbl}</div>'
        html += "</div>"
        stat.markdown(html, unsafe_allow_html=True)

    show_steps(0)
    prog.progress(10, text="Cleaning data...")
    clean_df, issues = clean_data(df)
    st.session_state.clean_df     = clean_df
    st.session_state.clean_issues = issues
    time.sleep(0.3)

    show_steps(1)
    prog.progress(25, text="Detecting data type with Claude...")
    context = detect_data_context(clean_df, api_key)
    st.session_state.context = context

    show_steps(2)
    prog.progress(40, text="Detecting statistical anomalies...")
    flags_df = detect_anomalies(clean_df, context)

    show_steps(3)
    prog.progress(55, text="Claude explaining flags...")
    def pcb(i, total):
        prog.progress(55 + int((i / total) * 25), text=f"Explaining flag {i+1}/{total}...")
    flags_df = explain_anomalies(flags_df, context, api_key, progress_cb=pcb)
    st.session_state.flags_df = flags_df

    show_steps(4)
    prog.progress(85, text="Generating narrative & insights...")
    st.session_state.narrative = generate_narrative(clean_df, flags_df, context, api_key)
    st.session_state.insights  = generate_insights(clean_df, context, api_key)

    prog.progress(100, text="Analysis complete")
    time.sleep(0.4)
    st.session_state.analysis_done = True
    stat.empty()
    prog.empty()
    st.rerun()

# ── Tab renderers ──────────────────────────────────────────────────────────────

def render_anomalies(tab):
    with tab:
        flags_df = st.session_state.flags_df
        issues   = st.session_state.clean_issues

        st.markdown('<div class="section-title">Data quality</div>', unsafe_allow_html=True)
        for iss in issues:
            color = "#10b981" if "No data" in iss else "#f59e0b"
            st.markdown(f'<div style="font-size:0.82rem;color:{color};padding:2px 0">&#9679; {iss}</div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.5rem">Flagged records</div>',
                    unsafe_allow_html=True)

        if flags_df is None or flags_df.empty:
            st.success("No anomalies detected - dataset is within expected ranges.")
            return

        fc1, fc2 = st.columns(2)
        with fc1:
            sev_f = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium"])
        with fc2:
            col_f = st.multiselect("Column", sorted(flags_df["column"].unique()),
                                    default=list(flags_df["column"].unique()))

        filtered = flags_df[
            flags_df["severity"].isin(sev_f) & flags_df["column"].isin(col_f)
        ].copy()

        display_cols = [c for c in ["entity_id", "column", "value", "direction", "z_score", "severity"]
                        if c in filtered.columns]

        def sev_style(val):
            return {"High":   "color:#fca5a5;font-weight:600",
                    "Medium": "color:#fdba74;font-weight:600",
                    "Low":    "color:#86efac;font-weight:600"}.get(val, "")

        st.dataframe(
            filtered[display_cols].style.applymap(sev_style, subset=["severity"]),
            use_container_width=True, hide_index=True,
        )

        has_exp = filtered[
            filtered["llm_explanation"].notna() &
            (filtered["llm_explanation"] != "Pending review")
        ]
        if not has_exp.empty:
            st.markdown('<div class="section-title" style="margin-top:1.5rem">AI interpretations</div>',
                        unsafe_allow_html=True)
            for _, row in has_exp.iterrows():
                with st.expander(
                    f"{row['entity_id']} -- {row['column']} = {row['value']} ({row['severity']})",
                    expanded=False,
                ):
                    ca, cb = st.columns([1, 3])
                    with ca:
                        st.markdown(
                            f'<div style="font-size:0.75rem;color:#475569">Z-score</div>'
                            f'<div style="font-family:JetBrains Mono;font-size:1.1rem;color:#e2e8f0">'
                            f'{row.get("z_score", "N/A")}</div>',
                            unsafe_allow_html=True,
                        )
                    with cb:
                        st.markdown(
                            f'<div style="font-size:0.85rem;color:#cbd5e1;line-height:1.7">'
                            f'{row["llm_explanation"]}</div>',
                            unsafe_allow_html=True,
                        )


def render_narrative(tab):
    with tab:
        narrative = st.session_state.narrative
        ctx       = st.session_state.context or {}
        if not narrative:
            st.info("Narrative not yet generated.")
            return
        st.markdown(
            f'<span class="ctx-badge">Style: {ctx.get("narrative_style", "regulatory")}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Generated by Claude. For review purposes only. Not for submission without human review.")
        paragraphs = [p.strip() for p in narrative.split("\n") if p.strip()]
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        st.markdown(f'<div class="narrative-box">{body}</div>', unsafe_allow_html=True)


def render_dashboard(tab):
    with tab:
        df       = st.session_state.clean_df
        ctx      = st.session_state.context or {}
        insights = st.session_state.insights or {}
        num_cols = [c for c in ctx.get("key_numeric_cols", []) if c in df.columns]
        cat_cols = [c for c in ctx.get("key_category_cols", []) if c in df.columns]

        if not num_cols:
            st.info("No numeric columns detected for charting.")
            return

        # Chart 1 — primary numeric distribution
        col1 = num_cols[0]
        st.markdown(f'<div class="section-title">{col1} distribution</div>', unsafe_allow_html=True)
        if insights.get("chart1"):
            st.markdown(f'<div class="insight-box">&#128161; {insights["chart1"]}</div>',
                        unsafe_allow_html=True)
        if cat_cols:
            fig1 = px.histogram(df, x=col1, color=cat_cols[0], barmode="overlay",
                                 opacity=0.75, color_discrete_sequence=CHART_COLORS, nbins=40)
        else:
            fig1 = px.histogram(df, x=col1, color_discrete_sequence=["#0ea5e9"], nbins=40)
        fig1.update_layout(**plot_base(), bargap=0.05)
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 & 3
        c2, c3 = st.columns(2)
        with c2:
            if len(num_cols) >= 2 and cat_cols:
                col2 = num_cols[1]
                st.markdown(f'<div class="section-title">{col2} by {cat_cols[0]}</div>',
                            unsafe_allow_html=True)
                if insights.get("chart2"):
                    st.markdown(f'<div class="insight-box">&#128161; {insights["chart2"]}</div>',
                                unsafe_allow_html=True)
                grp  = df.groupby(cat_cols[0])[col2].mean().reset_index()
                fig2 = px.bar(grp, x=cat_cols[0], y=col2, color=cat_cols[0],
                              color_discrete_sequence=CHART_COLORS)
                fig2.update_layout(**plot_base(), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with c3:
            if cat_cols:
                st.markdown(f'<div class="section-title">{cat_cols[0]} breakdown</div>',
                            unsafe_allow_html=True)
                if insights.get("chart3"):
                    st.markdown(f'<div class="insight-box">&#128161; {insights["chart3"]}</div>',
                                unsafe_allow_html=True)
                vc = df[cat_cols[0]].value_counts().reset_index()
                vc.columns = [cat_cols[0], "count"]
                fig3 = px.pie(vc, names=cat_cols[0], values="count",
                              color_discrete_sequence=CHART_COLORS, hole=0.45)
                fig3.update_layout(**plot_base())
                fig3.update_traces(textfont_color="#e2e8f0")
                st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — correlation heatmap
        if len(num_cols) >= 3:
            st.markdown('<div class="section-title">Correlation matrix</div>', unsafe_allow_html=True)
            if insights.get("chart4"):
                st.markdown(f'<div class="insight-box">&#128161; {insights["chart4"]}</div>',
                            unsafe_allow_html=True)
            corr = df[num_cols[:8]].corr().round(2)
            fig4 = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0, "#450a0a"], [0.5, "#0d1424"], [1, "#0369a1"]],
                text=corr.values.round(2), texttemplate="%{text}", zmid=0, showscale=True,
            ))
            fig4.update_layout(**plot_base())
            st.plotly_chart(fig4, use_container_width=True)

        # Chart 5 — anomaly flags by column
        flags_df = st.session_state.flags_df
        if flags_df is not None and not flags_df.empty:
            st.markdown('<div class="section-title">Anomaly flags by column</div>', unsafe_allow_html=True)
            flag_counts = flags_df.groupby(["column", "severity"]).size().reset_index(name="count")
            fig5 = px.bar(flag_counts, x="column", y="count", color="severity",
                          color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                          barmode="stack")
            fig5.update_layout(**plot_base())
            st.plotly_chart(fig5, use_container_width=True)


def render_export(tab):
    with tab:
        st.markdown('<div class="section-title">Export results</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            if st.session_state.flags_df is not None:
                st.download_button("Download anomaly flags (CSV)",
                    st.session_state.flags_df.to_csv(index=False),
                    "anomaly_flags.csv", "text/csv", use_container_width=True)
        with ec2:
            if st.session_state.narrative:
                st.download_button("Download narrative (TXT)",
                    st.session_state.narrative,
                    "narrative.txt", "text/plain", use_container_width=True)
        with ec3:
            if st.session_state.clean_df is not None:
                st.download_button("Download cleaned data (CSV)",
                    st.session_state.clean_df.to_csv(index=False),
                    "cleaned_data.csv", "text/csv", use_container_width=True)

        if st.session_state.context:
            st.markdown('<div class="section-title" style="margin-top:1.5rem">Detected context (JSON)</div>',
                        unsafe_allow_html=True)
            st.json(st.session_state.context)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI — runs after all functions are defined
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#f1f5f9;padding-bottom:0.25rem">PharmaAI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:0.75rem;color:#334155;margin-bottom:1rem">Intelligent Data Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input(
        "Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com",
    )
    if api_key:
        st.markdown('<div style="font-size:0.78rem;color:#10b981;margin-top:0.25rem">API key set</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.78rem;color:#ef4444;margin-top:0.25rem">API key required</div>',
                    unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<div style="font-size:0.7rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem">Supported datasets</div>',
        unsafe_allow_html=True,
    )
    for key, info in DATASET_REGISTRY.items():
        st.markdown(f'<div style="font-size:0.8rem;color:#475569;padding:2px 0">{info["label"]}</div>',
                    unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:#475569;padding:2px 0">Any pharma CSV</div>',
                unsafe_allow_html=True)

    st.divider()
    if st.session_state.analysis_done:
        if st.button("Reset analysis", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:2.2rem;font-weight:800;color:#f1f5f9;margin-bottom:0.25rem">Pharma Data Intelligence</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#475569;font-size:0.9rem;margin-bottom:1.5rem">'
    'Upload any pharma dataset. Claude auto-detects the data type and runs a domain-appropriate analysis.</p>',
    unsafe_allow_html=True,
)

# ── Step 1: Load data ──────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    col_left, col_right = st.columns([1.3, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Choose a sample dataset</div>', unsafe_allow_html=True)
        btn_cols = st.columns(2)
        for i, (key, info) in enumerate(DATASET_REGISTRY.items()):
            is_sel = st.session_state.selected_ds == key
            label  = f"* {info['label']}" if is_sel else info["label"]
            with btn_cols[i % 2]:
                if st.button(label, key=f"ds_{key}", use_container_width=True):
                    with st.spinner(f"Generating {info['label']} dataset..."):
                        st.session_state.df = info["fn"]()
                        st.session_state.selected_ds = key
                        st.session_state.analysis_done = False
                    st.rerun()
                st.caption(info["desc"])

    with col_right:
        st.markdown('<div class="section-title">Or upload your own CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")
        if uploaded:
            try:
                df_up = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                st.session_state.df = df_up
                st.session_state.selected_ds = "custom"
                st.session_state.analysis_done = False
                st.success(f"Loaded {len(df_up):,} rows x {len(df_up.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        st.markdown('<div class="section-title" style="margin-top:1.5rem">How it works</div>',
                    unsafe_allow_html=True)
        for num, title, desc in [
            ("1.", "Auto-detect",       "Claude reads columns & samples to identify data type"),
            ("2.", "Statistical flags", "Z-score outlier detection on all numeric columns"),
            ("3.", "AI explanations",   "Claude interprets each flag in domain context"),
            ("4.", "Narrative",         "Regulatory-grade written summary generated automatically"),
        ]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:14px">
              <div style="font-size:1rem;margin-top:2px;color:#0ea5e9">{num}</div>
              <div>
                <div style="font-size:0.85rem;font-weight:600;color:#e2e8f0">{title}</div>
                <div style="font-size:0.78rem;color:#475569;line-height:1.4">{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Step 2: Preview ────────────────────────────────────────────────────────────
if st.session_state.df is not None and not st.session_state.analysis_done:
    df = st.session_state.df
    st.divider()

    info = DATASET_REGISTRY.get(st.session_state.selected_ds, {})
    tag  = info.get("label", "Custom dataset") if info else "Custom dataset"
    st.markdown(
        f'<span class="ctx-badge">{tag} &nbsp;|&nbsp; {len(df):,} rows &nbsp;|&nbsp; {len(df.columns)} columns</span>',
        unsafe_allow_html=True,
    )

    num_c = df.select_dtypes(include=np.number).columns
    cat_c = df.select_dtypes(include="object").columns
    show_metric_row([
        ("Total records",   f"{len(df):,}",           f"{len(df.columns)} columns",   "#0ea5e9"),
        ("Numeric columns", str(len(num_c)),           "Available for analysis",        "#10b981"),
        ("Categories",      str(len(cat_c)),           "Group variables",               "#8b5cf6"),
        ("Missing values",  str(int(df.isna().sum().sum())), "Across all columns",      "#f59e0b"),
    ])

    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar to run the AI pipeline.")
    else:
        if st.button("Run AI Analysis Pipeline", type="primary", use_container_width=True):
            run_pipeline(df, api_key)

# ── Empty state ────────────────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown(
        '<br><br><div style="text-align:center;color:#1e2d45;font-size:0.9rem">'
        'Select a sample dataset or upload a CSV to begin</div>',
        unsafe_allow_html=True,
    )

# ── Step 3: Results ────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    ctx   = st.session_state.context or {}
    dtype = ctx.get("data_type", "pharma").replace("_", " ").title()
    st.markdown(
        f'<span class="ctx-badge">{dtype} Analysis &nbsp;|&nbsp; {ctx.get("description","")}</span>',
        unsafe_allow_html=True,
    )

    flags_df = st.session_state.flags_df
    n_flags  = len(flags_df) if flags_df is not None and not flags_df.empty else 0
    n_high   = len(flags_df[flags_df["severity"] == "High"]) if n_flags else 0
    n_ents   = flags_df["entity_id"].nunique() if n_flags else 0

    show_metric_row([
        ("Anomalies flagged", str(n_flags), "Statistical outliers",       "#0ea5e9"),
        ("High severity",     str(n_high),  "Require immediate review",   "#ef4444"),
        ("Entities affected", str(n_ents),  "Unique records with flags",  "#f59e0b"),
        ("Columns analyzed",  str(len(ctx.get("key_numeric_cols", []))),
                              "Numeric metrics screened",                  "#10b981"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    tab_a, tab_n, tab_d, tab_e = st.tabs(
        ["Anomaly Flags", "Clinical Narrative", "Dashboard", "Export"]
    )
    render_anomalies(tab_a)
    render_narrative(tab_n)
    render_dashboard(tab_d)
    render_export(tab_e)