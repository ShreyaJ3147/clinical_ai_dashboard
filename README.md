# ClinicalAI — Automated Trial Data Analyzer

A portfolio project demonstrating a GenAI pipeline for clinical trial data analysis.
Simulates a $2.2M ROI opportunity by automating what 30 Data Analysts do manually.

## What it does

| Step | What happens |
|------|-------------|
| Upload | Accepts raw clinical trial CSV (lab results, adverse events, demographics) |
| Clean | Detects and corrects data quality issues |
| Detect | Rule engine flags lab values >3× ULN and Grade 3/4 AEs |
| Explain | Claude generates clinical interpretations for each flag |
| Narrate | Claude writes an ICH E3-style regulatory narrative |
| Dashboard | Interactive Plotly charts: ALT trends, AE distribution, severity heatmap |

## Usage

1. Click **"Use sample trial data"** to generate a synthetic Phase II oncology dataset
2. Paste your API key in the sidebar (or set the env var)
3. Click **"Run AI Analysis Pipeline"**
4. Explore the Anomaly Flags, Narrative, and Dashboard tabs
5. Export results as CSV / TXT

## Deploying to Streamlit Cloud (free)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app → select your repo
3. Add `ANTHROPIC_API_KEY` in the Secrets section
4. Deploy → share the link

## Tech stack

- **Streamlit** — UI and app framework
- **Anthropic Claude** (`claude-sonnet-4-6`) — anomaly explanations, narrative, insights
- **Pandas** — data cleaning and aggregation
- **Plotly** — interactive charts
- **Numpy** — synthetic data generation

## Project structure

```
clinical_ai_dashboard/
├── app.py               # Main Streamlit app (UI + orchestration)
├── data_generator.py    # Synthetic Phase II trial data
├── pipeline.py          # LLM pipeline (clean → detect → explain → narrate)
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Dark theme
└── README.md
```
