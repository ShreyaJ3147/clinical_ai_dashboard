"""
Synthetic clinical trial data generator.
Simulates a Phase II oncology study with 3 treatment arms.
Intentional anomalies are seeded for LLM detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_clinical_data(n_patients: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    arms = ["Placebo", "Drug A 50mg", "Drug A 100mg"]
    visits = [("Baseline", 0), ("Week 4", 28), ("Week 8", 56), ("Week 12", 84)]

    # Lab tests with normal ranges
    labs = [
        {"test": "ALT",        "unit": "U/L",    "mean": 28,  "sd": 8,   "low": 7,   "high": 56},
        {"test": "AST",        "unit": "U/L",    "mean": 25,  "sd": 7,   "low": 10,  "high": 40},
        {"test": "Creatinine", "unit": "mg/dL",  "mean": 0.95,"sd": 0.18,"low": 0.6, "high": 1.2},
        {"test": "WBC",        "unit": "K/uL",   "mean": 6.5, "sd": 1.5, "low": 4.0, "high": 11.0},
        {"test": "Hemoglobin", "unit": "g/dL",   "mean": 13.5,"sd": 1.5, "low": 11.5,"high": 17.5},
        {"test": "Platelets",  "unit": "K/uL",   "mean": 240, "sd": 55,  "low": 150, "high": 400},
    ]

    ae_types = [
        "Nausea", "Fatigue", "Headache", "Dizziness", "Alopecia",
        "Peripheral neuropathy", "Neutropenia", "Thrombocytopenia", "Elevated liver enzymes"
    ]
    severities = ["Grade 1", "Grade 2", "Grade 3", "Grade 4"]
    outcomes = ["Resolved", "Ongoing", "Resolved with treatment", "Dose reduced"]

    records = []

    for i in range(n_patients):
        patient_id = f"PT-{str(i + 1).zfill(4)}"
        age = int(np.clip(np.random.normal(58, 12), 25, 82))
        sex = np.random.choice(["Male", "Female"])
        arm = np.random.choice(arms, p=[0.33, 0.34, 0.33])
        baseline_date = datetime(2023, 3, 1) + timedelta(days=np.random.randint(0, 60))

        # Inject anomalous patients (every ~10th patient in high-dose arm)
        is_anomalous = (arm == "Drug A 100mg") and (i % 10 == 0)

        for visit_name, day_offset in visits:
            visit_date = baseline_date + timedelta(days=day_offset)

            # Lab results for this visit
            for lab in labs:
                value = np.random.normal(lab["mean"], lab["sd"])

                # Dose-dependent effect on ALT/AST (hepatotoxicity signal in high-dose arm)
                if lab["test"] in ("ALT", "AST") and arm == "Drug A 100mg":
                    visit_multiplier = {"Baseline": 1.0, "Week 4": 1.4, "Week 8": 1.9, "Week 12": 2.1}
                    value *= visit_multiplier[visit_name]

                # Hard anomaly: spike a specific patient's ALT to >5× ULN
                if is_anomalous and lab["test"] == "ALT" and visit_name in ("Week 8", "Week 12"):
                    value = lab["high"] * np.random.uniform(5.2, 7.0)

                value = round(max(value, 0.1), 2)

                records.append({
                    "patient_id": patient_id,
                    "age": age,
                    "sex": sex,
                    "treatment_arm": arm,
                    "visit": visit_name,
                    "visit_date": visit_date.strftime("%Y-%m-%d"),
                    "record_type": "lab",
                    "lab_test": lab["test"],
                    "lab_value": value,
                    "lab_unit": lab["unit"],
                    "normal_low": lab["low"],
                    "normal_high": lab["high"],
                    "adverse_event": None,
                    "ae_severity": None,
                    "ae_outcome": None,
                })

            # Adverse events — higher rate in active arms
            ae_prob = {"Placebo": 0.15, "Drug A 50mg": 0.30, "Drug A 100mg": 0.45}
            # More AEs in later visits for high-dose arm
            if visit_name != "Baseline" and np.random.random() < ae_prob[arm]:
                ae = np.random.choice(ae_types)
                # High-dose arm skewed toward higher grade AEs
                sev_weights = (
                    [0.2, 0.4, 0.3, 0.1] if arm == "Drug A 100mg"
                    else [0.5, 0.35, 0.12, 0.03]
                )
                severity = np.random.choice(severities, p=sev_weights)
                records.append({
                    "patient_id": patient_id,
                    "age": age,
                    "sex": sex,
                    "treatment_arm": arm,
                    "visit": visit_name,
                    "visit_date": visit_date.strftime("%Y-%m-%d"),
                    "record_type": "ae",
                    "lab_test": None,
                    "lab_value": None,
                    "lab_unit": None,
                    "normal_low": None,
                    "normal_high": None,
                    "adverse_event": ae,
                    "ae_severity": severity,
                    "ae_outcome": np.random.choice(outcomes),
                })

    df = pd.DataFrame(records)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df
