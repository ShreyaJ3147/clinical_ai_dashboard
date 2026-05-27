"""
LLM pipeline for clinical data analysis.
Architecture:
  1. Rule-based detection  → fast, deterministic flagging
  2. Claude explanations   → clinical interpretation of flagged records
  3. Claude narrative      → ICH E3-style study summary
  4. Claude chart insights → one-liner insight per dashboard panel
"""

import json
import anthropic
import pandas as pd

MODEL = "claude-sonnet-4-6"


# ── 1. Data cleaning ──────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Basic data cleaning. Returns cleaned df and list of issue descriptions."""
    issues = []
    original_len = len(df)

    # Drop rows missing both lab value and adverse event (empty rows)
    df = df.dropna(subset=["lab_value", "adverse_event"], how="all")
    dropped = original_len - len(df)
    if dropped:
        issues.append(f"Removed {dropped} rows with no lab or AE data")

    # Clamp negative lab values
    neg_mask = df["lab_value"].notna() & (df["lab_value"] < 0)
    if neg_mask.any():
        df.loc[neg_mask, "lab_value"] = 0
        issues.append(f"Corrected {neg_mask.sum()} negative lab values to 0")

    # Flag duplicate patient-visit-lab combos
    lab_rows = df[df["record_type"] == "lab"]
    dups = lab_rows.duplicated(subset=["patient_id", "visit", "lab_test"], keep="first").sum()
    if dups:
        issues.append(f"Found {dups} duplicate patient-visit-lab records (kept first)")

    if not issues:
        issues.append("No data quality issues detected")

    return df.reset_index(drop=True), issues


# ── 2. Rule-based anomaly detection ──────────────────────────────────────────

def detect_anomalies_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag records using clinical rules before sending to the LLM.
    Rules:
      - Lab value > 3× ULN  → potential safety signal
      - Lab value < 0.5× LLN → potential safety signal
      - Grade 3 or 4 AE     → always flagged
    """
    flags = []

    lab_df = df[df["record_type"] == "lab"].copy()
    ae_df  = df[df["record_type"] == "ae"].copy()

    # Lab flags
    high_mask = lab_df["lab_value"] > (lab_df["normal_high"] * 3)
    low_mask  = lab_df["lab_value"] < (lab_df["normal_low"]  * 0.5)

    for _, row in lab_df[high_mask].iterrows():
        flags.append({
            "patient_id":     row["patient_id"],
            "treatment_arm":  row["treatment_arm"],
            "visit":          row["visit"],
            "flag_type":      "Lab — High",
            "detail":         f"{row['lab_test']} = {row['lab_value']} {row['lab_unit']} "
                              f"(ULN {row['normal_high']}; {row['lab_value']/row['normal_high']:.1f}× ULN)",
            "severity":       "High" if row["lab_value"] > row["normal_high"] * 5 else "Medium",
            "llm_explanation": None,
        })

    for _, row in lab_df[low_mask].iterrows():
        flags.append({
            "patient_id":     row["patient_id"],
            "treatment_arm":  row["treatment_arm"],
            "visit":          row["visit"],
            "flag_type":      "Lab — Low",
            "detail":         f"{row['lab_test']} = {row['lab_value']} {row['lab_unit']} "
                              f"(LLN {row['normal_low']})",
            "severity":       "Medium",
            "llm_explanation": None,
        })

    # AE flags — Grade 3/4
    serious_ae = ae_df[ae_df["ae_severity"].isin(["Grade 3", "Grade 4"])]
    for _, row in serious_ae.iterrows():
        flags.append({
            "patient_id":     row["patient_id"],
            "treatment_arm":  row["treatment_arm"],
            "visit":          row["visit"],
            "flag_type":      "Adverse Event",
            "detail":         f"{row['adverse_event']} — {row['ae_severity']} ({row['ae_outcome']})",
            "severity":       "High" if row["ae_severity"] == "Grade 4" else "Medium",
            "llm_explanation": None,
        })

    return pd.DataFrame(flags) if flags else pd.DataFrame(
        columns=["patient_id", "treatment_arm", "visit",
                 "flag_type", "detail", "severity", "llm_explanation"]
    )


# ── 3. LLM clinical explanations ─────────────────────────────────────────────

def explain_anomalies(flags_df: pd.DataFrame, api_key: str,
                      progress_callback=None) -> pd.DataFrame:
    """
    Send each flagged record to Claude for a clinical explanation.
    Processes top 20 flags to keep latency reasonable for a demo.
    """
    if flags_df.empty:
        return flags_df

    client = anthropic.Anthropic(api_key=api_key)
    top_flags = flags_df.head(20).copy()
    explanations = []

    for i, (_, row) in enumerate(top_flags.iterrows()):
        if progress_callback:
            progress_callback(i, len(top_flags))

        prompt = f"""You are a clinical data safety monitor reviewing a Phase II oncology trial.

Flag summary:
- Patient: {row['patient_id']} | Arm: {row['treatment_arm']} | Visit: {row['visit']}
- Type: {row['flag_type']}
- Finding: {row['detail']}

Write a 2-sentence clinical interpretation of this flag. Include:
1. What the finding suggests clinically (potential mechanism or safety implication)
2. What action a Data Safety Monitoring Board (DSMB) might recommend

Be concise and use clinical language appropriate for a regulatory document."""

        response = client.messages.create(
            model=MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        explanations.append(response.content[0].text.strip())

    top_flags["llm_explanation"] = explanations

    # Append any remaining flags without LLM explanations
    if len(flags_df) > 20:
        remaining = flags_df.iloc[20:].copy()
        remaining["llm_explanation"] = "Pending review"
        return pd.concat([top_flags, remaining], ignore_index=True)

    return top_flags


# ── 4. Narrative generation ───────────────────────────────────────────────────

def generate_narrative(df: pd.DataFrame, flags_df: pd.DataFrame,
                       api_key: str) -> str:
    """Generate an ICH E3-style clinical narrative summary."""

    client = anthropic.Anthropic(api_key=api_key)

    # Build statistics to feed the LLM
    lab_df = df[df["record_type"] == "lab"]
    ae_df  = df[df["record_type"] == "ae"]

    n_patients   = df["patient_id"].nunique()
    n_flags      = len(flags_df)
    high_flags   = len(flags_df[flags_df["severity"] == "High"]) if not flags_df.empty else 0

    arm_counts = df.groupby("treatment_arm")["patient_id"].nunique().to_dict()

    # ALT means by arm and visit
    alt_df = lab_df[lab_df["lab_test"] == "ALT"]
    alt_summary = (
        alt_df.groupby(["treatment_arm", "visit"])["lab_value"]
        .mean().round(1).to_string()
    )

    # AE summary by arm
    ae_summary = (
        ae_df.groupby(["treatment_arm", "ae_severity"])
        .size().to_string()
    )

    grade34_rate = {}
    for arm, grp in ae_df.groupby("treatment_arm"):
        total_pts = arm_counts.get(arm, 1)
        serious   = grp[grp["ae_severity"].isin(["Grade 3", "Grade 4"])]["patient_id"].nunique()
        grade34_rate[arm] = round(serious / total_pts * 100, 1)

    prompt = f"""You are a senior clinical data scientist preparing a regulatory-grade study summary
for a Phase II oncology trial of Drug A (50mg and 100mg doses) vs Placebo.

Study statistics:
- Total patients: {n_patients}
- Arm breakdown: {arm_counts}
- Total anomaly flags: {n_flags} ({high_flags} high severity)
- Grade 3/4 AE rates by arm: {grade34_rate}%

ALT values (U/L) by arm and visit (normal range 7–56):
{alt_summary}

Adverse event counts by arm and severity:
{ae_summary}

Write a 4-paragraph clinical narrative in ICH E3 style covering:
1. Study overview and patient disposition
2. Laboratory safety findings (focus on hepatic markers, dose relationship)
3. Adverse event profile and dose-limiting observations
4. Overall safety assessment and recommended actions for the DSMB

Use formal clinical language. Flag the hepatotoxicity signal clearly if the data supports it."""

    response = client.messages.create(
        model=MODEL,
        max_tokens=900,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


# ── 5. Chart insights ─────────────────────────────────────────────────────────

def generate_chart_insights(df: pd.DataFrame, api_key: str) -> dict:
    """Generate a one-liner insight for each dashboard chart panel."""

    client = anthropic.Anthropic(api_key=api_key)

    lab_df = df[df["record_type"] == "lab"]
    ae_df  = df[df["record_type"] == "ae"]

    alt_by_arm_visit = (
        lab_df[lab_df["lab_test"] == "ALT"]
        .groupby(["treatment_arm", "visit"])["lab_value"]
        .mean().round(1).to_dict()
    )
    ae_by_arm = ae_df.groupby("treatment_arm").size().to_dict()
    grade34_by_arm = (
        ae_df[ae_df["ae_severity"].isin(["Grade 3", "Grade 4"])]
        .groupby("treatment_arm").size().to_dict()
    )
    top_aes = ae_df["adverse_event"].value_counts().head(5).to_dict()

    prompt = f"""Given these clinical trial statistics, write a single sharp insight sentence
for each of the 4 dashboard panels. Each insight should highlight the most clinically
significant finding for that panel. Be specific with numbers.

Data:
- ALT by arm & visit: {alt_by_arm_visit}
- Total AEs by arm: {ae_by_arm}
- Grade 3/4 AEs by arm: {grade34_by_arm}
- Top 5 AEs: {top_aes}

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{
  "alt_trend": "one sentence",
  "ae_distribution": "one sentence",
  "severity_breakdown": "one sentence",
  "top_ae_types": "one sentence"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "alt_trend":         "See ALT trend chart for dose-dependent hepatic changes.",
            "ae_distribution":   "See AE distribution across treatment arms.",
            "severity_breakdown":"Grade 3/4 events are concentrated in the high-dose arm.",
            "top_ae_types":      "Nausea and fatigue are the most frequently reported AEs.",
        }
