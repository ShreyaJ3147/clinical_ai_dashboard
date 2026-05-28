"""
Adaptive LLM pipeline — works for any pharma data type.
Architecture:
  1. Auto-detect data context (Claude reads columns + sample)
  2. Statistical anomaly detection (Z-score / rule-based, universal)
  3. Claude explains each flag in domain context
  4. Claude writes a domain-appropriate narrative
  5. Claude suggests chart insights
"""

import json
import numpy as np
import pandas as pd
import anthropic

MODEL = "claude-sonnet-4-6"


# ── 1. Data cleaning ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    before = len(df)
    df = df.dropna(how="all")
    if len(df) < before:
        issues.append(f"Removed {before - len(df)} fully-empty rows")

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        neg = (df[col] < 0).sum()
        if neg and col not in ("temp_min_c","temp_max_c"):
            issues.append(f"Column '{col}': {neg} negative values found")

    dup = df.duplicated().sum()
    if dup:
        df = df.drop_duplicates()
        issues.append(f"Removed {dup} duplicate rows")

    if not issues:
        issues.append("No data quality issues detected — dataset is clean")

    return df.reset_index(drop=True), issues


# ── 2. Auto-detect data context ───────────────────────────────────────────────

def detect_data_context(df: pd.DataFrame, api_key: str) -> dict:
    """Claude identifies what kind of pharma data this is."""
    client = anthropic.Anthropic(api_key=api_key)

    sample_json = df.head(5).to_json(orient="records", date_format="iso", indent=2)
    cols        = list(df.columns)
    dtypes      = {c: str(df[c].dtype) for c in cols}

    prompt = f"""You are a pharma data expert. Examine this dataset and return a JSON object describing it.

Columns: {cols}
Dtypes: {dtypes}
Sample rows:
{sample_json}

Return ONLY valid JSON — no markdown, no extra text — with this exact structure:
{{
  "data_type": "clinical_trial | pharmacovigilance | manufacturing | supply_chain | other",
  "description": "one sentence describing the dataset",
  "primary_entity_col": "column name of the main entity (patient, batch, shipment, report)",
  "date_columns": ["list of date-like columns"],
  "key_numeric_cols": ["most important numeric columns for analysis (max 6)"],
  "key_category_cols": ["main categorical grouping columns (max 4)"],
  "narrative_style": "ICH E3 clinical | PSUR pharmacovigilance | GMP deviation | supply chain compliance",
  "analysis_focus": "one sentence on what anomalies matter most for this data type"
}}"""

    response = client.messages.create(
        model=MODEL, max_tokens=400,
        messages=[{"role":"user","content":prompt}]
    )
    raw = response.content[0].text.strip().replace("```json","").replace("```","")
    try:
        return json.loads(raw)
    except Exception:
        return {
            "data_type": "other",
            "description": "Pharma dataset",
            "primary_entity_col": df.columns[0],
            "date_columns": [],
            "key_numeric_cols": list(df.select_dtypes(include=np.number).columns[:6]),
            "key_category_cols": list(df.select_dtypes(include="object").columns[:4]),
            "narrative_style": "regulatory",
            "analysis_focus": "Outliers and unexpected patterns in key metrics",
        }


# ── 3. Statistical anomaly detection (universal) ─────────────────────────────

def detect_anomalies(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Z-score outlier detection on key numeric columns — works for any data."""
    flags      = []
    entity_col = context.get("primary_entity_col", df.columns[0])
    num_cols   = context.get("key_numeric_cols",
                   list(df.select_dtypes(include=np.number).columns[:8]))

    for col in num_cols:
        if col not in df.columns or df[col].isna().all():
            continue
        col_data = df[col].dropna()
        if col_data.std() == 0:
            continue
        z_scores = ((df[col] - col_data.mean()) / col_data.std()).abs()
        threshold = 3.0
        outlier_mask = z_scores > threshold
        for idx, row in df[outlier_mask].iterrows():
            z = round(float(z_scores[idx]), 2)
            direction = "high" if row[col] > col_data.mean() else "low"
            entity_val = str(row.get(entity_col, f"Row {idx}"))
            # Add category context if available
            cat_context = {}
            for cc in context.get("key_category_cols", []):
                if cc in row:
                    cat_context[cc] = str(row[cc])
            flags.append({
                "entity_id":       entity_val,
                "column":          col,
                "value":           round(float(row[col]), 4),
                "direction":       direction,
                "z_score":         z,
                "column_mean":     round(float(col_data.mean()), 4),
                "column_std":      round(float(col_data.std()), 4),
                "severity":        "High" if z > 4 else "Medium",
                "category_context":json.dumps(cat_context),
                "llm_explanation": None,
            })

    # Also flag boolean/text columns with anomalous rates
    for col in df.columns:
        if df[col].dtype == bool or (df[col].dtype == object and
                                      df[col].nunique() == 2 and
                                      col.endswith(("_flag","_recall","excursion"))):
            rate = df[col].astype(bool).mean()
            if rate > 0.15:
                flags.append({
                    "entity_id":       "Dataset-wide",
                    "column":          col,
                    "value":           round(rate * 100, 1),
                    "direction":       "high",
                    "z_score":         None,
                    "column_mean":     None,
                    "column_std":      None,
                    "severity":        "High" if rate > 0.25 else "Medium",
                    "category_context":"{}",
                    "llm_explanation": None,
                })

    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["entity_id","column","value","direction","z_score",
                 "column_mean","column_std","severity","category_context","llm_explanation"])


# ── 4. LLM clinical/domain explanations ──────────────────────────────────────

def explain_anomalies(flags_df: pd.DataFrame, context: dict,
                      api_key: str, progress_cb=None) -> pd.DataFrame:
    if flags_df.empty:
        return flags_df
    client  = anthropic.Anthropic(api_key=api_key)
    top     = flags_df.head(15).copy()
    style   = context.get("narrative_style","regulatory")
    focus   = context.get("analysis_focus","anomalies in key metrics")
    dtype   = context.get("data_type","pharma")
    expls   = []
    for i, (_, row) in enumerate(top.iterrows()):
        if progress_cb:
            progress_cb(i, len(top))
        cat = json.loads(row.get("category_context","{}"))
        cat_str = ", ".join(f"{k}={v}" for k,v in cat.items()) if cat else "N/A"
        z_str = f"Z-score: {row['z_score']}" if row['z_score'] else "Flag: elevated rate"
        prompt = f"""You are a {style} expert reviewing a {dtype} dataset.
Context: {focus}

Flagged observation:
- Entity: {row['entity_id']}
- Metric: {row['column']} = {row['value']} ({row['direction']})
- {z_str} (mean={row['column_mean']}, SD={row['column_std']})
- Groups: {cat_str}

Write 2 concise sentences:
1. What this finding likely indicates in a {dtype} context
2. What action a reviewer should take

Use professional regulatory language. Be specific."""
        resp = client.messages.create(
            model=MODEL, max_tokens=180,
            messages=[{"role":"user","content":prompt}]
        )
        expls.append(resp.content[0].text.strip())
    top["llm_explanation"] = expls
    if len(flags_df) > 15:
        rest = flags_df.iloc[15:].copy()
        rest["llm_explanation"] = "Pending review"
        return pd.concat([top, rest], ignore_index=True)
    return top


# ── 5. Narrative generation ───────────────────────────────────────────────────

def generate_narrative(df: pd.DataFrame, flags_df: pd.DataFrame,
                       context: dict, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    style  = context.get("narrative_style","regulatory")
    dtype  = context.get("data_type","pharma data")
    desc   = context.get("description","")
    n_rows = len(df)
    n_entities = df[context.get("primary_entity_col", df.columns[0])].nunique()
    n_flags = len(flags_df)
    hi_flags = len(flags_df[flags_df["severity"]=="High"]) if not flags_df.empty else 0

    # Build numeric summary
    num_cols = context.get("key_numeric_cols",[])
    summary_lines = []
    for col in num_cols:
        if col in df.columns:
            s = df[col].describe()
            summary_lines.append(f"{col}: mean={s['mean']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}, SD={s['std']:.2f}")

    # Category breakdown
    cat_lines = []
    for col in context.get("key_category_cols",[])[:2]:
        if col in df.columns:
            top_vals = df[col].value_counts().head(5).to_dict()
            cat_lines.append(f"{col}: {top_vals}")

    prompt = f"""You are a senior regulatory affairs scientist writing a formal {style} report.

Dataset: {desc}
Total records: {n_rows:,} | Unique entities: {n_entities:,}
Anomaly flags: {n_flags} total, {hi_flags} high-severity

Key metric statistics:
{chr(10).join(summary_lines)}

Category breakdown:
{chr(10).join(cat_lines)}

Write a professional 4-paragraph narrative in {style} format covering:
1. Dataset overview and scope
2. Key findings and statistical patterns
3. Anomalies, signals, or quality concerns identified
4. Overall assessment and recommended next steps

Use formal language appropriate for a regulatory document. Flag any high-severity findings clearly."""

    resp = client.messages.create(
        model=MODEL, max_tokens=900,
        messages=[{"role":"user","content":prompt}]
    )
    return resp.content[0].text.strip()


# ── 6. Chart insights ─────────────────────────────────────────────────────────

def generate_insights(df: pd.DataFrame, context: dict, api_key: str) -> dict:
    client   = anthropic.Anthropic(api_key=api_key)
    num_cols = context.get("key_numeric_cols",[])
    cat_cols = context.get("key_category_cols",[])

    stats = {}
    for col in num_cols[:4]:
        if col in df.columns:
            stats[col] = {"mean": round(df[col].mean(),2),
                          "max":  round(df[col].max(),2),
                          "min":  round(df[col].min(),2)}
    cat_dist = {}
    for col in cat_cols[:2]:
        if col in df.columns:
            cat_dist[col] = df[col].value_counts().head(5).to_dict()

    prompt = f"""Given these stats from a {context.get('data_type','pharma')} dataset, 
write one sharp insight for each of 4 dashboard panels. Be specific with numbers.

Numeric stats: {json.dumps(stats)}
Category distributions: {json.dumps(cat_dist)}

Return ONLY valid JSON — no markdown:
{{
  "chart1": "insight for main numeric trend",
  "chart2": "insight for category distribution",
  "chart3": "insight for outliers / quality",
  "chart4": "insight for overall risk or compliance"
}}"""

    resp = client.messages.create(
        model=MODEL, max_tokens=300,
        messages=[{"role":"user","content":prompt}]
    )
    raw = resp.content[0].text.strip().replace("```json","").replace("```","")
    try:
        return json.loads(raw)
    except Exception:
        return {"chart1":"See trend chart.","chart2":"See distribution chart.",
                "chart3":"Outliers flagged above.","chart4":"Review narrative for risk summary."}
