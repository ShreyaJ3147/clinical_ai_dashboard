"""
PharmaAI — Intelligent Data Analyzer
Professional Streamlit dashboard for pharma data analysis powered by Claude.
"""

import os, time, json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import DATASET_REGISTRY
from pipeline import (
    clean_data, detect_data_context, detect_anomalies,
    explain_anomalies, generate_narrative, generate_insights,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PharmaAI Analyzer",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  /* ── Base ── */
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
  code, .mono { font-family: 'JetBrains Mono', monospace; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #05080f;
    border-right: 1px solid #111827;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }
  [data-testid="stSidebar"] .sidebar-brand { color: #f8fafc !important; }

  /* ── Main background ── */
  .stApp { background: #080d17; }
  .main .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }

  /* ── Metric cards ── */
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 1.5rem 0; }
  .metric-card {
    background: #0d1424;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #0ea5e9);
  }
  .metric-card .label { font-size: 0.75rem; font-weight: 500; color: #475569; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.5rem; }
  .metric-card .value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; color: #f1f5f9; }
  .metric-card .sub   { font-size: 0.75rem; color: #334155; margin-top: 0.25rem; }

  /* ── Dataset selector cards ── */
  .ds-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-top: 1rem; }
  .ds-card {
    background: #0d1424;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    cursor: pointer;
    transition: border-color 0.15s;
  }
  .ds-card:hover { border-color: #0ea5e9; }
  .ds-card.selected { border-color: #0ea5e9; background: #091a2a; }
  .ds-card .ds-icon { font-size: 1.5rem; margin-bottom: 0.4rem; }
  .ds-card .ds-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; color: #e2e8f0; margin-bottom: 0.2rem; }
  .ds-card .ds-desc  { font-size: 0.78rem; color: #475569; line-height: 1.4; }

  /* ── Step pipeline ── */
  .pipeline-steps { display: flex; flex-direction: column; gap: 8px; margin: 1.5rem 0; }
  .step { display: flex; align-items: center; gap: 12px; padding: 0.6rem 1rem; border-radius: 8px; background: #0d1424; border: 1px solid #1e2d45; }
  .step.active  { border-color: #0ea5e9; background: #091a2a; }
  .step.done    { border-color: #10b981; background: #071a12; }
  .step-dot     { width: 8px; height: 8px; border-radius: 50%; background: #1e2d45; flex-shrink: 0; }
  .step.active  .step-dot { background: #0ea5e9; box-shadow: 0 0 6px #0ea5e9; }
  .step.done    .step-dot { background: #10b981; }
  .step-label   { font-size: 0.82rem; color: #64748b; }
  .step.active  .step-label { color: #bae6fd; }
  .step.done    .step-label { color: #6ee7b7; }

  /* ── Severity badges ── */
  .badge { display: inline-block; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .badge-high   { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }
  .badge-medium { background: #431407; color: #fdba74; border: 1px solid #7c2d12; }
  .badge-low    { background: #052e16; color: #86efac; border: 1px solid #14532d; }

  /* ── Insight callouts ── */
  .insight {
    background: #0a1628;
    border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.5;
  }

  /* ── Narrative ── */
  .narrative {
    background: #0a0f1a;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    line-height: 1.9;
    color: #cbd5e1;
    font-size: 0.9rem;
  }
  .narrative p { margin-bottom: 1.2rem; }

  /* ── Context tag ── */
  .ctx-tag {
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

  /* ── Section headers ── */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.03em;
    margin: 1.5rem 0 0.75rem;
  }

  /* ── Dividers ── */
  hr { border-color: #111827 !important; }

  /* ── Tab bar override ── */
  [data-testid="stTabs"] [role="tablist"] { background: #0d1424; border-radius: 10px; padding: 4px; border: 1px solid #1e2d45; }
  [data-testid="stTabs"] [role="tab"]     { color: #475569 !important; font-size: 0.85rem; font-family: 'DM Sans', sans-serif; border-radius: 8px; }
  [data-testid="stTabs"] [aria-selected="true"] { background: #091a2a !important; color: #bae6fd !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] { background: #0d1424; border: 1px dashed #1e2d45; border-radius: 10px; }

  /* ── Buttons ── */
  .stButton > button[kind="primary"] {
    background: #0369a1 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
  }
  .stButton > button[kind="primary"]:hover { background: #0284c7 !important; }
  .stButton > button:not([kind="primary"]) {
    background: #0d1424 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
  }

  /* ── Expander ── */
  [data-testid="stExpander"] { background: #0d1424; border: 1px solid #1e2d45; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULTS = dict(df=None, clean_df=None, clean_issues=[], context=None,
                flags_df=None, narrative=None, insights=None,
                analysis_done=False, selected_ds="clinical_trial")
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
ACCENT = {"clinical_trial":"#0ea5e9","pharmacovigilance":"#f59e0b",
           "manufacturing":"#10b981","supply_chain":"#8b5cf6","other":"#64748b"}

CHART_PALETTE = ["#0ea5e9","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899"]

def accent_color():
    dt = (st.session_state.context or {}).get("data_type","other")
    return ACCENT.get(dt, "#0ea5e9")

def card(label, value, sub="", accent="#0ea5e9"):
    return f"""<div class="metric-card" style="--accent:{accent}">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="sub">{sub}</div>
</div>"""

def badge(sev):
    cls = f"badge-{sev.lower()}"
    return f'<span class="badge {cls}">{sev}</span>'

def plot_layout():
    return dict(plot_bgcolor="#080d17", paper_bgcolor="#080d17",
                font=dict(color="#94a3b8", family="DM Sans"),
                xaxis=dict(gridcolor="#111827", linecolor="#1e2d45"),
                yaxis=dict(gridcolor="#111827", linecolor="#1e2d45"),
                legend=dict(bgcolor="#0d1424", bordercolor="#1e2d45", borderwidth=1))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem'>
      <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;color:#f1f5f9'>⬡ PharmaAI</div>
      <div style='font-size:0.75rem;color:#334155;margin-top:2px'>Intelligent Data Analyzer</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    api_key = st.text_input("Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY",""),
        placeholder="sk-ant-...",
        help="Get your key at console.anthropic.com")

    if api_key:
        st.markdown('<div style="font-size:0.75rem;color:#10b981">● API key set</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:0.75rem;color:#ef4444">○ API key required</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-size:0.72rem;color:#334155;text-transform:uppercase;letter-spacing:0.08em">Supported data types</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    for key, info in DATASET_REGISTRY.items():
        st.markdown(f'<div style="font-size:0.8rem;padding:3px 0;color:#475569">{info["icon"]} {info["label"]}</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;padding:3px 0;color:#475569">📎 Any pharma CSV</div>', unsafe_allow_html=True)

    st.divider()
    if st.session_state.analysis_done and st.button("↺ Reset analysis", use_container_width=True):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.25rem'>
  <span style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:#f1f5f9'>Pharma Data Intelligence</span>
</div>
<div style='color:#475569;font-size:0.9rem;margin-bottom:1.5rem'>
  Upload any pharma dataset — clinical, manufacturing, PV, or supply chain. 
  Claude auto-detects the data type and runs a domain-appropriate analysis.
</div>
""", unsafe_allow_html=True)

# ── Step 1: Load data ─────────────────────────────────────────────────────────
if not st.session_state.analysis_done:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">Choose sample dataset</div>', unsafe_allow_html=True)
        ds_html = '<div class="ds-grid">'
        for key, info in DATASET_REGISTRY.items():
            sel = "selected" if st.session_state.selected_ds == key else ""
            ds_html += f"""
            <div class="ds-card {sel}" onclick="">
              <div class="ds-icon">{info['icon']}</div>
              <div class="ds-title">{info['label']}</div>
              <div class="ds-desc">{info['desc']}</div>
            </div>"""
        ds_html += "</div>"
        st.markdown(ds_html, unsafe_allow_html=True)

        # Dataset buttons
        bcols = st.columns(2)
        for i, (key, info) in enumerate(DATASET_REGISTRY.items()):
            with bcols[i % 2]:
                if st.button(f"{info['icon']} {info['label']}", use_container_width=True, key=f"ds_{key}"):
                    with st.spinner(f"Generating {info['label']} dataset…"):
                        st.session_state.df = info["fn"]()
                        st.session_state.selected_ds = key
                        st.session_state.analysis_done = False
                    st.rerun()

    with col_right:
        st.markdown('<div class="section-header">Or upload your own CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv","xlsx"],
            label_visibility="collapsed",
            help="Any pharma CSV — Claude will auto-detect the data type")
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
                st.session_state.df = df
                st.session_state.selected_ds = "custom"
                st.session_state.analysis_done = False
                st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")

        st.markdown('<div class="section-header" style="margin-top:1.5rem">How it works</div>', unsafe_allow_html=True)
        steps_info = [
            ("🔍", "Auto-detect", "Claude reads columns & sample rows to identify data type"),
            ("📊", "Statistical flags", "Z-score outlier detection on all numeric columns"),
            ("🧠", "AI explanations", "Claude interprets each flag in domain context"),
            ("📄", "Narrative", "Regulatory-grade written summary generated automatically"),
        ]
        for icon, title, desc in steps_info:
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px'>
              <div style='font-size:1.1rem;margin-top:2px'>{icon}</div>
              <div>
                <div style='font-size:0.85rem;font-weight:600;color:#e2e8f0'>{title}</div>
                <div style='font-size:0.78rem;color:#475569;line-height:1.4'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Step 2: Preview ───────────────────────────────────────────────────────────
if st.session_state.df is not None and not st.session_state.analysis_done:
    df = st.session_state.df
    st.divider()

    info = DATASET_REGISTRY.get(st.session_state.selected_ds, {})
    tag_label = info.get("label","Custom dataset") if info else "Custom dataset"
    icon      = info.get("icon","📎") if info else "📎"
    st.markdown(f'<div class="ctx-tag">{icon} {tag_label} · {len(df):,} rows · {df.columns.nunique()} columns</div>', unsafe_allow_html=True)

    num_cols  = df.select_dtypes(include=np.number).columns
    cat_cols  = df.select_dtypes(include="object").columns
    bool_cols = df.select_dtypes(include=bool).columns

    cards_html = f'<div class="metric-grid">{card("Total records", f"{len(df):,}", f"{df.shape[1]} columns")}{card("Numeric columns", len(num_cols), "Available for analysis")}{card("Categories", len(cat_cols), "Group variables")}{card("Missing values", int(df.isna().sum().sum()), "Across all columns","#f59e0b")}</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(30), use_container_width=True, hide_index=True)

    if not api_key:
        st.warning("⚠️ Add your Anthropic API key in the sidebar to run the AI pipeline.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬡ Run AI Analysis Pipeline", type="primary", use_container_width=True):
            run_pipeline(df, api_key)

# ── Landing empty state ───────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;color:#1e2d45'>
      <div style='font-size:3rem;margin-bottom:1rem'>⬡</div>
      <div style='font-family:Syne,sans-serif;font-size:1.1rem;color:#334155'>
        Select a sample dataset above or upload a CSV to begin
      </div>
    </div>""", unsafe_allow_html=True)

# ── Step 3: Results ───────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    ctx  = st.session_state.context or {}
    dtype = ctx.get("data_type","pharma")
    ac   = accent_color()

    # Context header
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:1.5rem'>
      <div class="ctx-tag" style='border-color:{ac}44;color:{ac}'>
        {ctx.get('data_type','').replace('_',' ').title()} Analysis
      </div>
      <div style='font-size:0.82rem;color:#334155'>{ctx.get('description','')}</div>
    </div>""", unsafe_allow_html=True)

    flags_df = st.session_state.flags_df
    n_flags  = len(flags_df) if flags_df is not None and not flags_df.empty else 0
    n_high   = len(flags_df[flags_df["severity"]=="High"]) if n_flags else 0

    cards_html = f"""<div class="metric-grid">
      {card("Anomalies flagged", n_flags, "Statistical outliers", ac)}
      {card("High severity", n_high, "Require immediate review", "#ef4444")}
      {card("Entities affected", flags_df['entity_id'].nunique() if n_flags else 0, "Unique records with flags", "#f59e0b")}
      {card("Columns analyzed", len(ctx.get('key_numeric_cols',[])), "Numeric metrics screened", "#10b981")}
    </div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    tab_a, tab_n, tab_d, tab_e = st.tabs(["🚨  Anomaly Flags", "📄  Narrative", "📊  Dashboard", "⬇  Export"])

    render_anomalies(tab_a)
    render_narrative(tab_n)
    render_dashboard(tab_d)
    render_export(tab_e)


# ── Pipeline runner ───────────────────────────────────────────────────────────
def run_pipeline(df, api_key):
    steps = ["Cleaning & validating data", "Auto-detecting data type",
             "Running statistical analysis", "Generating AI explanations",
             "Writing narrative & insights"]
    prog  = st.progress(0)
    stat  = st.empty()

    def show_step(i, msg):
        prog.progress(int((i/len(steps))*100), text=msg)
        stat.markdown(f"""
        <div class="pipeline-steps">{"".join(
            f'<div class="step {"done" if j<i else "active" if j==i else ""}"><div class="step-dot"></div><div class="step-label">{"✓ " if j<i else ""}{steps[j]}</div></div>'
            for j in range(len(steps))
        )}</div>""", unsafe_allow_html=True)

    show_step(0, "Cleaning data…")
    clean_df, issues = clean_data(df)
    st.session_state.clean_df     = clean_df
    st.session_state.clean_issues = issues
    time.sleep(0.3)

    show_step(1, "Detecting data type with Claude…")
    context = detect_data_context(clean_df, api_key)
    st.session_state.context = context

    show_step(2, "Detecting statistical anomalies…")
    flags_df = detect_anomalies(clean_df, context)

    show_step(3, f"Claude explaining {min(len(flags_df),15)} flags…")
    def pcb(i, total):
        p = 60 + int((i/total)*20)
        prog.progress(p, text=f"Explaining flag {i+1}/{total}…")
    flags_df = explain_anomalies(flags_df, context, api_key, progress_cb=pcb)
    st.session_state.flags_df = flags_df

    show_step(4, "Generating narrative & insights…")
    st.session_state.narrative = generate_narrative(clean_df, flags_df, context, api_key)
    st.session_state.insights  = generate_insights(clean_df, context, api_key)

    prog.progress(100, text="Analysis complete ✓")
    time.sleep(0.5)
    st.session_state.analysis_done = True
    stat.empty(); prog.empty()
    st.rerun()


# ── Anomaly tab ───────────────────────────────────────────────────────────────
def render_anomalies(tab):
    with tab:
        flags_df = st.session_state.flags_df
        issues   = st.session_state.clean_issues

        # Data quality
        st.markdown('<div class="section-header">Data quality</div>', unsafe_allow_html=True)
        for iss in issues:
            color = "#10b981" if "No data" in iss else "#f59e0b"
            st.markdown(f'<div style="font-size:0.82rem;color:{color};padding:2px 0">● {iss}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem">Flagged records</div>', unsafe_allow_html=True)

        if flags_df is None or flags_df.empty:
            st.success("No anomalies detected — dataset is within expected ranges.")
            return

        # Filters
        fc1, fc2 = st.columns(2)
        with fc1:
            sev_filter = st.multiselect("Severity", ["High","Medium","Low"],
                                         default=["High","Medium"])
        with fc2:
            col_filter = st.multiselect("Column", sorted(flags_df["column"].unique()),
                                         default=list(flags_df["column"].unique()))

        filtered = flags_df[flags_df["severity"].isin(sev_filter) &
                             flags_df["column"].isin(col_filter)].copy()

        # Display columns
        display_cols = ["entity_id","column","value","direction","z_score","severity"]
        available = [c for c in display_cols if c in filtered.columns]

        # Style severity
        def style_sev(val):
            m = {"High":"color:#fca5a5;font-weight:600","Medium":"color:#fdba74;font-weight:600","Low":"color:#86efac;font-weight:600"}
            return m.get(val,"")

        styled = filtered[available].style.applymap(style_sev, subset=["severity"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # AI explanations
        has_exp = filtered[filtered["llm_explanation"].notna() &
                           (filtered["llm_explanation"]!="Pending review")]
        if not has_exp.empty:
            st.markdown('<div class="section-header" style="margin-top:1.5rem">AI clinical interpretations</div>', unsafe_allow_html=True)
            for _, row in has_exp.iterrows():
                with st.expander(f"**{row['entity_id']}** — {row['column']} = {row['value']} ({row['severity']})", expanded=False):
                    col_a, col_b = st.columns([1,3])
                    with col_a:
                        st.markdown(f'<div style="font-size:0.78rem;color:#475569">Z-score</div><div style="font-family:JetBrains Mono;font-size:1.1rem;color:#e2e8f0">{row.get("z_score","—")}</div>', unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f'<div style="font-size:0.85rem;color:#cbd5e1;line-height:1.7">{row["llm_explanation"]}</div>', unsafe_allow_html=True)


# ── Narrative tab ─────────────────────────────────────────────────────────────
def render_narrative(tab):
    with tab:
        narrative = st.session_state.narrative
        ctx       = st.session_state.context or {}
        if not narrative:
            st.info("Narrative not yet generated.")
            return
        style = ctx.get("narrative_style","regulatory")
        st.markdown(f'<div class="ctx-tag">Style: {style}</div>', unsafe_allow_html=True)
        st.caption("Generated by Claude · For review purposes only · Not for regulatory submission without human review")
        paragraphs = [p.strip() for p in narrative.split("\n") if p.strip()]
        body = "".join(f"<p>{p}</p>" for p in paragraphs)
        st.markdown(f'<div class="narrative">{body}</div>', unsafe_allow_html=True)


# ── Dashboard tab ─────────────────────────────────────────────────────────────
def render_dashboard(tab):
    with tab:
        df       = st.session_state.clean_df
        ctx      = st.session_state.context or {}
        insights = st.session_state.insights or {}
        num_cols = [c for c in ctx.get("key_numeric_cols",[]) if c in df.columns]
        cat_cols = [c for c in ctx.get("key_category_cols",[]) if c in df.columns]
        ac       = accent_color()

        if not num_cols:
            st.info("No numeric columns detected for charting.")
            return

        # Chart 1 — Distribution of primary numeric column
        col1 = num_cols[0]
        st.markdown(f'<div class="section-header">{col1} distribution</div>', unsafe_allow_html=True)
        if insights.get("chart1"):
            st.markdown(f'<div class="insight">💡 {insights["chart1"]}</div>', unsafe_allow_html=True)

        if cat_cols:
            fig1 = px.histogram(df, x=col1, color=cat_cols[0],
                                 barmode="overlay", opacity=0.75,
                                 color_discrete_sequence=CHART_PALETTE,
                                 nbins=40)
        else:
            fig1 = px.histogram(df, x=col1, color_discrete_sequence=[ac], nbins=40)
        fig1.update_layout(**plot_layout(), bargap=0.05)
        fig1.update_traces(marker_line_width=0)
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2 & 3 side-by-side
        c2, c3 = st.columns(2)
        with c2:
            if len(num_cols) >= 2 and cat_cols:
                col2 = num_cols[1]
                st.markdown(f'<div class="section-header">{col2} by {cat_cols[0]}</div>', unsafe_allow_html=True)
                if insights.get("chart2"):
                    st.markdown(f'<div class="insight">💡 {insights["chart2"]}</div>', unsafe_allow_html=True)
                grp = df.groupby(cat_cols[0])[col2].mean().reset_index()
                fig2 = px.bar(grp, x=cat_cols[0], y=col2,
                              color=cat_cols[0], color_discrete_sequence=CHART_PALETTE)
                fig2.update_layout(**plot_layout(), showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

        with c3:
            if len(cat_cols) >= 1:
                st.markdown(f'<div class="section-header">{cat_cols[0]} breakdown</div>', unsafe_allow_html=True)
                if insights.get("chart3"):
                    st.markdown(f'<div class="insight">💡 {insights["chart3"]}</div>', unsafe_allow_html=True)
                vc = df[cat_cols[0]].value_counts().reset_index()
                vc.columns = [cat_cols[0], "count"]
                fig3 = px.pie(vc, names=cat_cols[0], values="count",
                              color_discrete_sequence=CHART_PALETTE, hole=0.45)
                fig3.update_layout(**plot_layout())
                fig3.update_traces(textfont_color="#e2e8f0")
                st.plotly_chart(fig3, use_container_width=True)

        # Chart 4 — Correlation heatmap of numeric columns
        if len(num_cols) >= 3:
            st.markdown('<div class="section-header">Correlation matrix</div>', unsafe_allow_html=True)
            if insights.get("chart4"):
                st.markdown(f'<div class="insight">💡 {insights["chart4"]}</div>', unsafe_allow_html=True)
            corr_cols = num_cols[:8]
            corr = df[corr_cols].corr().round(2)
            fig4 = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale=[[0,"#450a0a"],[0.5,"#0d1424"],[1,"#0369a1"]],
                text=corr.values.round(2), texttemplate="%{text}",
                zmid=0, showscale=True,
            ))
            fig4.update_layout(**plot_layout())
            st.plotly_chart(fig4, use_container_width=True)

        # Chart 5 — Anomaly flag distribution
        flags_df = st.session_state.flags_df
        if flags_df is not None and not flags_df.empty:
            st.markdown('<div class="section-header">Anomaly flags by column</div>', unsafe_allow_html=True)
            flag_counts = flags_df.groupby(["column","severity"]).size().reset_index(name="count")
            fig5 = px.bar(flag_counts, x="column", y="count", color="severity",
                          color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                          barmode="stack")
            fig5.update_layout(**plot_layout())
            st.plotly_chart(fig5, use_container_width=True)


# ── Export tab ────────────────────────────────────────────────────────────────
def render_export(tab):
    with tab:
        st.markdown('<div class="section-header">Export results</div>', unsafe_allow_html=True)
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            if st.session_state.flags_df is not None:
                st.download_button("⬇ Anomaly flags (CSV)",
                    st.session_state.flags_df.to_csv(index=False),
                    "anomaly_flags.csv","text/csv", use_container_width=True)
        with ec2:
            if st.session_state.narrative:
                st.download_button("⬇ Narrative (TXT)",
                    st.session_state.narrative,
                    "narrative.txt","text/plain", use_container_width=True)
        with ec3:
            if st.session_state.clean_df is not None:
                st.download_button("⬇ Cleaned data (CSV)",
                    st.session_state.clean_df.to_csv(index=False),
                    "cleaned_data.csv","text/csv", use_container_width=True)

        if st.session_state.context:
            st.markdown('<div class="section-header" style="margin-top:1.5rem">Detected context</div>', unsafe_allow_html=True)
            st.json(st.session_state.context)
