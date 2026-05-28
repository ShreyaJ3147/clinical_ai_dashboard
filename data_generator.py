"""
Synthetic pharma data generators — 4 dataset types.
Each mimics real data a pharma company would actually analyze.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# ── 1. Clinical Trial (Phase II Oncology) ─────────────────────────────────────

def generate_clinical_trial_data(n_patients=120, seed=42):
    np.random.seed(seed); random.seed(seed)
    arms    = ["Placebo", "Drug A 50mg", "Drug A 100mg"]
    visits  = [("Baseline", 0), ("Week 4", 28), ("Week 8", 56), ("Week 12", 84)]
    labs    = [
        {"test": "ALT",        "unit": "U/L",   "mean": 28,   "sd": 8,    "low": 7,    "high": 56},
        {"test": "AST",        "unit": "U/L",   "mean": 25,   "sd": 7,    "low": 10,   "high": 40},
        {"test": "Creatinine", "unit": "mg/dL", "mean": 0.95, "sd": 0.18, "low": 0.6,  "high": 1.2},
        {"test": "WBC",        "unit": "K/uL",  "mean": 6.5,  "sd": 1.5,  "low": 4.0,  "high": 11.0},
        {"test": "Hemoglobin", "unit": "g/dL",  "mean": 13.5, "sd": 1.5,  "low": 11.5, "high": 17.5},
        {"test": "Platelets",  "unit": "K/uL",  "mean": 240,  "sd": 55,   "low": 150,  "high": 400},
    ]
    ae_types   = ["Nausea","Fatigue","Headache","Dizziness","Alopecia",
                  "Peripheral neuropathy","Neutropenia","Thrombocytopenia","Elevated liver enzymes"]
    severities = ["Grade 1","Grade 2","Grade 3","Grade 4"]
    outcomes   = ["Resolved","Ongoing","Resolved with treatment","Dose reduced"]
    records = []
    for i in range(n_patients):
        pid  = f"PT-{str(i+1).zfill(4)}"
        age  = int(np.clip(np.random.normal(58,12), 25, 82))
        sex  = np.random.choice(["Male","Female"])
        arm  = np.random.choice(arms, p=[0.33,0.34,0.33])
        base = datetime(2023,3,1) + timedelta(days=np.random.randint(0,60))
        is_anomalous = (arm=="Drug A 100mg") and (i%10==0)
        for vname, doff in visits:
            vdate = base + timedelta(days=doff)
            for lab in labs:
                val = np.random.normal(lab["mean"], lab["sd"])
                if lab["test"] in ("ALT","AST") and arm=="Drug A 100mg":
                    val *= {"Baseline":1.0,"Week 4":1.4,"Week 8":1.9,"Week 12":2.1}[vname]
                if is_anomalous and lab["test"]=="ALT" and vname in ("Week 8","Week 12"):
                    val = lab["high"] * np.random.uniform(5.2, 7.0)
                records.append({"patient_id":pid,"age":age,"sex":sex,"treatment_arm":arm,
                    "visit":vname,"visit_date":vdate.strftime("%Y-%m-%d"),"record_type":"lab",
                    "lab_test":lab["test"],"lab_value":round(max(val,0.1),2),
                    "lab_unit":lab["unit"],"normal_low":lab["low"],"normal_high":lab["high"],
                    "adverse_event":None,"ae_severity":None,"ae_outcome":None})
            ae_prob = {"Placebo":0.15,"Drug A 50mg":0.30,"Drug A 100mg":0.45}
            if vname!="Baseline" and np.random.random()<ae_prob[arm]:
                sw = [0.2,0.4,0.3,0.1] if arm=="Drug A 100mg" else [0.5,0.35,0.12,0.03]
                records.append({"patient_id":pid,"age":age,"sex":sex,"treatment_arm":arm,
                    "visit":vname,"visit_date":vdate.strftime("%Y-%m-%d"),"record_type":"ae",
                    "lab_test":None,"lab_value":None,"lab_unit":None,"normal_low":None,"normal_high":None,
                    "adverse_event":np.random.choice(ae_types),
                    "ae_severity":np.random.choice(severities, p=sw),
                    "ae_outcome":np.random.choice(outcomes)})
    df = pd.DataFrame(records)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


# ── 2. Pharmacovigilance (Post-market adverse event reports) ──────────────────

def generate_pharmacovigilance_data(n_reports=300, seed=42):
    np.random.seed(seed); random.seed(seed)
    drugs     = ["Oncovax","Cardiorel","Neurostim","Immunex","Hepatrex"]
    reactions  = ["Anaphylaxis","Hepatotoxicity","Cardiac arrhythmia","Stevens-Johnson syndrome",
                  "Rhabdomyolysis","Acute kidney injury","Thromboembolism","Agranulocytosis",
                  "Peripheral neuropathy","Severe hypotension","Drug rash","QT prolongation"]
    reporters  = ["Healthcare professional","Consumer","Pharmaceutical company","Regulatory authority"]
    countries  = ["USA","Germany","UK","France","Japan","Canada","Australia","Brazil"]
    outcomes   = ["Recovered","Recovering","Not recovered","Fatal","Unknown"]
    seriousness= ["Serious","Non-serious"]
    records = []
    base_date = datetime(2022,1,1)
    for i in range(n_reports):
        drug   = np.random.choice(drugs, p=[0.35,0.25,0.20,0.12,0.08])
        onset  = base_date + timedelta(days=np.random.randint(0,700))
        report = onset + timedelta(days=np.random.randint(1,30))
        age    = int(np.clip(np.random.normal(55,18), 18, 90))
        reaction = np.random.choice(reactions)
        # Inject signal: Oncovax + Hepatotoxicity cluster
        if drug=="Oncovax" and np.random.random()<0.30:
            reaction = "Hepatotoxicity"
        serious_prob = 0.45 if reaction in ("Anaphylaxis","Stevens-Johnson syndrome","Fatal") else 0.25
        records.append({
            "report_id":      f"ICSR-{str(i+1).zfill(5)}",
            "drug_name":      drug,
            "reporter_type":  np.random.choice(reporters, p=[0.55,0.25,0.15,0.05]),
            "patient_age":    age,
            "patient_sex":    np.random.choice(["Male","Female"]),
            "country":        np.random.choice(countries),
            "reaction":       reaction,
            "onset_date":     onset.strftime("%Y-%m-%d"),
            "report_date":    report.strftime("%Y-%m-%d"),
            "days_to_report": (report - onset).days,
            "seriousness":    "Serious" if np.random.random()<serious_prob else "Non-serious",
            "outcome":        np.random.choice(outcomes, p=[0.45,0.20,0.15,0.05,0.15]),
            "concomitant_drugs": np.random.randint(0, 6),
            "rechallenge":    np.random.choice(["Yes","No","Unknown"], p=[0.05,0.55,0.40]),
        })
    df = pd.DataFrame(records)
    df["onset_date"]  = pd.to_datetime(df["onset_date"])
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


# ── 3. Manufacturing QC (Batch release records) ───────────────────────────────

def generate_manufacturing_data(n_batches=200, seed=42):
    np.random.seed(seed); random.seed(seed)
    products    = ["Oncovax 10mg","Cardiorel 5mg","Neurostim 20mg","Immunex 50mg"]
    equipment   = [f"REACTOR-{x}" for x in ["A1","A2","B1","B2","C1"]]
    operators   = [f"OP-{str(x).zfill(3)}" for x in range(1,20)]
    deviation_types = ["Temperature excursion","pH out of spec","Yield below target",
                       "Endotoxin limit exceeded","Equipment malfunction","Purity deviation"]
    records = []
    base_date = datetime(2023,1,1)
    for i in range(n_batches):
        product = np.random.choice(products, p=[0.35,0.25,0.25,0.15])
        mfg_date = base_date + timedelta(days=i*2 + np.random.randint(-3,3))
        exp_date = mfg_date + timedelta(days=np.random.randint(540,730))
        equip    = np.random.choice(equipment)
        # Inject signal: REACTOR-C1 has yield problems
        yield_mean = 85 if equip=="REACTOR-C1" else 94
        yield_val  = np.clip(np.random.normal(yield_mean, 4), 60, 100)
        purity     = np.clip(np.random.normal(99.2, 0.5), 95, 100)
        ph         = np.clip(np.random.normal(7.2, 0.3), 6.0, 8.5)
        moisture   = np.clip(np.random.normal(2.1, 0.4), 0.5, 5.0)
        endotoxin  = np.clip(np.random.exponential(0.3), 0.01, 5.0)
        deviations = 0
        if equip=="REACTOR-C1": deviations = np.random.randint(0,4)
        elif yield_val<88: deviations = np.random.randint(1,3)
        pass_fail  = "Pass" if (purity>98.5 and endotoxin<0.5 and yield_val>85) else "Fail"
        records.append({
            "batch_id":          f"BATCH-{str(i+1).zfill(4)}",
            "product_name":      product,
            "manufacturing_date":mfg_date.strftime("%Y-%m-%d"),
            "expiry_date":       exp_date.strftime("%Y-%m-%d"),
            "equipment_id":      equip,
            "operator_id":       np.random.choice(operators),
            "yield_pct":         round(yield_val, 2),
            "purity_pct":        round(purity, 3),
            "ph_level":          round(ph, 2),
            "moisture_content":  round(moisture, 3),
            "endotoxin_EU_mL":   round(endotoxin, 4),
            "batch_size_kg":     round(np.random.normal(50,5), 1),
            "deviation_count":   deviations,
            "deviation_type":    np.random.choice(deviation_types) if deviations>0 else None,
            "release_status":    pass_fail,
            "review_time_days":  int(np.random.normal(5,2)) if pass_fail=="Pass" else int(np.random.normal(12,4)),
        })
    df = pd.DataFrame(records)
    df["manufacturing_date"] = pd.to_datetime(df["manufacturing_date"])
    df["expiry_date"]        = pd.to_datetime(df["expiry_date"])
    return df


# ── 4. Supply Chain (Drug distribution & cold chain) ─────────────────────────

def generate_supply_chain_data(n_shipments=250, seed=42):
    np.random.seed(seed); random.seed(seed)
    products    = ["Oncovax 10mg vial","Cardiorel 5mg tablet","Immunex 50mg injection"]
    origins     = ["Frankfurt Hub","Chicago Hub","Singapore Hub","Basel Plant"]
    destinations= ["Boston Hospital","NYC Medical Center","LA Cancer Center",
                   "Chicago Clinic","Miami Hospital","Seattle Oncology","Dallas Health"]
    conditions  = ["2-8°C Refrigerated","15-25°C Controlled","Frozen -20°C","Ambient"]
    carriers    = ["ColdChain Express","MedFreight","PharmaShip","BioLogistics"]
    records = []
    base_date = datetime(2023,6,1)
    for i in range(n_shipments):
        product   = np.random.choice(products, p=[0.40,0.35,0.25])
        origin    = np.random.choice(origins)
        dest      = np.random.choice(destinations)
        ship_date = base_date + timedelta(days=np.random.randint(0,300))
        condition = "2-8°C Refrigerated" if "vial" in product or "injection" in product else "15-25°C Controlled"
        transit   = int(np.clip(np.random.normal(4,1.5), 1, 14))
        # Temperature: inject excursions for frozen products via ColdChain Express
        carrier   = np.random.choice(carriers)
        if condition=="Frozen -20°C" and carrier=="ColdChain Express":
            temp_min = np.random.uniform(-22, -15)
            temp_max = np.random.uniform(-10, 5)   # excursion!
        elif condition=="2-8°C Refrigerated":
            temp_min = np.random.uniform(1.5, 4.0)
            temp_max = np.random.uniform(5.0, 9.5)
        else:
            temp_min = np.random.uniform(14, 20)
            temp_max = np.random.uniform(22, 28)
        excursion = temp_max > ({"2-8°C Refrigerated":8,"15-25°C Controlled":25,
                                  "Frozen -20°C":-18,"Ambient":35}[condition])
        exp_date  = ship_date + timedelta(days=np.random.randint(180,540))
        records.append({
            "shipment_id":       f"SHIP-{str(i+1).zfill(5)}",
            "product_name":      product,
            "origin":            origin,
            "destination":       dest,
            "carrier":           carrier,
            "ship_date":         ship_date.strftime("%Y-%m-%d"),
            "transit_days":      transit,
            "quantity_units":    int(np.random.normal(500, 150)),
            "storage_condition": condition,
            "temp_min_c":        round(temp_min, 1),
            "temp_max_c":        round(temp_max, 1),
            "humidity_pct":      round(np.random.uniform(30, 65), 1),
            "temp_excursion":    excursion,
            "expiry_date":       exp_date.strftime("%Y-%m-%d"),
            "days_to_expiry":    (exp_date - ship_date).days,
            "recall_flag":       np.random.random() < 0.03,
            "on_time_delivery":  np.random.random() < 0.88,
        })
    df = pd.DataFrame(records)
    df["ship_date"]   = pd.to_datetime(df["ship_date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    return df


DATASET_REGISTRY = {
    "clinical_trial":    {"fn": generate_clinical_trial_data,    "label": "Clinical Trial",        "icon": "🧬", "desc": "Phase II oncology study — lab values, adverse events, patient demographics"},
    "pharmacovigilance": {"fn": generate_pharmacovigilance_data, "label": "Pharmacovigilance",     "icon": "⚠️", "desc": "Post-market adverse event reports — ICSR database"},
    "manufacturing":     {"fn": generate_manufacturing_data,     "label": "Manufacturing QC",      "icon": "🏭", "desc": "Batch release records — yield, purity, deviations"},
    "supply_chain":      {"fn": generate_supply_chain_data,      "label": "Supply Chain",          "icon": "🚚", "desc": "Drug distribution — cold chain, transit, expiry tracking"},
}
