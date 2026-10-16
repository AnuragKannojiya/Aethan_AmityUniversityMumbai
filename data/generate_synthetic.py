"""
GlucoTwin Synthetic Data Generation System
==========================================

Master data generation script for GlucoTwin — a Digital Twin proof-of-concept
for Type 2 Diabetes glucose spike prediction (Digital Twin Challenge 2026).
Simulates 200 synthetic Type 2 Diabetes patients from India with realistic
demographics, clinical correlations, and 30-day continuous CGM/wearable telemetry.

Target Output Files:
    - data/raw/ehr_patients.csv: Static EHR records for 200 patients
    - data/raw/medications_schedule.csv: Per-patient medication regimen and adherence
    - data/raw/cgm_readings.csv: 5-minute interval continuous monitoring for 30 days
                                 (8,640 readings/patient = 1,728,000 records)
    - data/raw/glucose_events.csv: Identified spike, hypoglycemia, and rapid drop events

Team: Aethan, Amity University Mumbai
"""

import sys
import os
import math
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ==============================================================================
# DEMOGRAPHIC POOLS (REALISTIC INDIAN NAMES & CITIES)
# ==============================================================================

MALE_FIRST_NAMES = [
    "Rajesh", "Amit", "Suresh", "Vijay", "Anil", "Ramesh", "Vikram", "Manoj",
    "Sunil", "Sanjay", "Rahul", "Pradeep", "Deepak", "Arvind", "Dinesh", "Ravi",
    "Ashok", "Ajay", "Mahesh", "Santosh", "Harish", "Naresh", "Gaurav", "Alok",
    "Sandeep", "Sachin", "Nitin", "Manish", "Vivek", "Pankaj", "Rohit", "Sumit",
    "Ankit", "Vishal", "Mukesh", "Kishore", "Hemant", "Chetan", "Bhupendra", "Devendra"
]

FEMALE_FIRST_NAMES = [
    "Sunita", "Priya", "Anita", "Rekha", "Lakshmi", "Meena", "Pooja", "Geeta",
    "Shanti", "Anjali", "Sangeeta", "Kavita", "Neha", "Ritu", "Swati", "Vandana",
    "Deepa", "Archana", "Usha", "Sarita", "Manju", "Saroj", "Jyoti", "Seema",
    "Rashmi", "Poonam", "Preeti", "Divya", "Shilpa", "Bhavna", "Madhuri", "Pratibha",
    "Kalpana", "Sudha", "Aparna", "Kiran", "Nirmala", "Radha", "Kamla", "Shobha"
]

INDIAN_SURNAMES = [
    "Sharma", "Patel", "Verma", "Mukherjee", "Iyer", "Reddy", "Gupta", "Singh",
    "Kumar", "Rao", "Nair", "Joshi", "Mehta", "Shah", "Chatterjee", "Das",
    "Sen", "Agarwal", "Bansal", "Malhotra", "Kulkarni", "Deshmukh", "Pillai",
    "Nambiar", "Bhat", "Choudhury", "Saxena", "Mishra", "Dubey", "Tiwari",
    "Pandey", "Ghosh", "Bose", "Banerjee", "Bhattacharya", "Kadam", "Shinde", "Yadav"
]

INDIAN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Surat", "Indore", "Nagpur", "Coimbatore"
]


# ==============================================================================
# 1. EHR DATA GENERATION (STATIC / HISTORICAL)
# ==============================================================================

def generate_ehr_data(n_patients: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic static EHR records for Type 2 Diabetes patients in India.
    
    Clinical Features & Plausibility:
    - Age: 35-75, skewed towards typical Indian T2D onset (45-65)
    - BMI: 22-42, Asian Indian overweight/obesity criteria
    - HbA1c: 5.5 - 12.0 %, stratified into well-controlled, moderate, high, and critical
    - Fasting glucose: 100 - 300 mg/dL, clinically correlated with HbA1c (eAG)
    - Lipids: High Triglycerides, low HDL (classic Asian Indian dyslipidemia)
    - Blood pressure & Renal panel: Correlated with age, BMI, and diabetes duration
    - Comorbidities & Medications: Assigned according to clinical progression guidelines

    Args:
        n_patients: Number of patient records to generate (default: 200).
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Tabular EHR records for n_patients.
    """
    rng = np.random.default_rng(seed)
    ref_date = datetime(2026, 1, 1)
    patient_records = []

    # Assign risk tiers to guarantee representation across clinical spectrum:
    # 0: Well-controlled (HbA1c 5.5 - 6.5) ~15%
    # 1: Moderately controlled (HbA1c 6.5 - 8.2) ~45%
    # 2: Poorly controlled (HbA1c 8.2 - 10.0) ~28%
    # 3: Severely uncontrolled / Critical (HbA1c 10.0 - 12.0) ~12%
    tiers = rng.choice([0, 1, 2, 3], size=n_patients, p=[0.15, 0.45, 0.28, 0.12])

    for i in range(1, n_patients + 1):
        pid = f"P{i:03d}"
        tier = tiers[i - 1]

        # Gender, Name & Height
        gender = "M" if rng.random() < 0.54 else "F"
        if gender == "M":
            first_name = rng.choice(MALE_FIRST_NAMES)
            height_cm = round(float(np.clip(rng.normal(169.5, 6.2), 152.0, 188.0)), 1)
        else:
            first_name = rng.choice(FEMALE_FIRST_NAMES)
            height_cm = round(float(np.clip(rng.normal(156.5, 5.5), 142.0, 172.0)), 1)
        
        last_name = rng.choice(INDIAN_SURNAMES)
        full_name = f"{first_name} {last_name}"
        city = rng.choice(INDIAN_CITIES)

        # Age: 35-75, skewed towards 45-65
        age = int(np.clip(rng.normal(54.2, 9.0), 35, 75))

        # BMI: 22.0 - 42.0 (higher in poorly controlled tiers)
        bmi_base = 26.0 + (tier * 1.8) + rng.normal(0, 2.5)
        bmi = round(float(np.clip(bmi_base, 22.0, 42.0)), 1)
        weight_kg = round(float(bmi * ((height_cm / 100.0) ** 2)), 1)

        # Diabetes duration: 0.5 to 25 years
        max_duration = max(1.0, min(25.0, age - 25.0))
        duration_scale = 3.0 + tier * 2.5
        duration_years = round(float(np.clip(rng.exponential(scale=duration_scale) + 0.5, 0.5, max_duration)), 1)

        # Lifestyle & Behavioral
        fam_history = bool(rng.random() < 0.62)
        if gender == "M":
            smoke_prob = [0.60, 0.24, 0.16]
        else:
            smoke_prob = [0.88, 0.08, 0.04]
        smoking_status = rng.choice(["Never", "Former", "Current"], p=smoke_prob)

        activity_level = rng.choice(
            ["Sedentary", "Light", "Moderate", "Active"],
            p=[0.42, 0.35, 0.18, 0.05]
        )

        diet_type = rng.choice(
            ["Vegetarian", "Non-Vegetarian", "Vegan"],
            p=[0.48, 0.48, 0.04]
        )

        stress_level = rng.choice(
            ["Low", "Moderate", "High"],
            p=[0.24, 0.52, 0.24]
        )

        sleep_hours_avg = round(float(np.clip(rng.normal(6.8, 1.0), 4.0, 9.0)), 1)

        # HbA1c (%): 5.5 - 12.0 reflecting T2D clinical spectrum
        if tier == 0:
            hba1c_raw = rng.uniform(5.5, 6.5)
        elif tier == 1:
            hba1c_raw = rng.uniform(6.5, 8.2)
        elif tier == 2:
            hba1c_raw = rng.uniform(8.2, 10.0)
        else:
            hba1c_raw = rng.uniform(10.0, 12.0)
        hba1c = round(float(np.clip(hba1c_raw, 5.5, 12.0)), 1)

        # Fasting Glucose (100 - 300 mg/dL):
        # Correlated with HbA1c via eAG (eAG = 28.7 * HbA1c - 46.7)
        eag = 28.7 * hba1c - 46.7
        fasting_glucose_raw = 0.84 * eag + rng.normal(0, 10.0)
        fasting_glucose = round(float(np.clip(fasting_glucose_raw, 100.0, 300.0)), 1)

        # Blood Pressure: Systolic (110 - 180), Diastolic (70 - 110)
        bp_sys_raw = 112.0 + (0.55 * (age - 35)) + (1.2 * (bmi - 22)) + (3.0 * tier) + rng.normal(0, 6.0)
        bp_systolic = int(np.clip(bp_sys_raw, 110, 180))
        bp_dia_raw = 70.0 + (0.33 * (bp_systolic - 110)) + rng.normal(0, 4.0)
        bp_diastolic = int(np.clip(bp_dia_raw, 70, 110))

        # Lipids (Asian Indian Dyslipidemia pattern: High Triglycerides, Low HDL)
        triglycerides_raw = 120.0 + (5.5 * (bmi - 22.0)) + (14.0 * (hba1c - 5.5)) + rng.normal(0, 25.0)
        triglycerides = int(np.clip(triglycerides_raw, 100, 400))

        chol_total_raw = 160.0 + (0.24 * triglycerides) + (0.35 * (age - 35)) + rng.normal(0, 15.0)
        cholesterol_total = int(np.clip(chol_total_raw, 150, 300))

        chol_ldl_raw = 0.55 * cholesterol_total + rng.normal(0, 12.0)
        cholesterol_ldl = int(np.clip(chol_ldl_raw, 80, 200))

        chol_hdl_raw = 56.0 - (0.45 * (bmi - 22.0)) - (1.2 * (hba1c - 5.5)) + rng.normal(0, 4.5)
        cholesterol_hdl = int(np.clip(chol_hdl_raw, 25, 70))

        # Renal Panel: Creatinine (0.6 - 2.5) & eGFR (30 - 120)
        cr_base = 0.88 if gender == "M" else 0.74
        cr_raw = cr_base + (0.032 * duration_years) + (0.15 * tier) + rng.normal(0, 0.12)
        creatinine = round(float(np.clip(cr_raw, 0.6, 2.5)), 2)

        egfr_raw = 135.0 - (0.75 * age) - (30.0 * (creatinine - 0.8)) - (0.08 * bp_systolic) + rng.normal(0, 4.0)
        egfr = int(np.clip(egfr_raw, 30, 120))

        # Comorbidities
        comorb_list = []
        if bp_systolic >= 135 or bp_diastolic >= 85 or (rng.random() < 0.45 and age > 50):
            comorb_list.append("Hypertension")
        if triglycerides >= 180 or cholesterol_ldl >= 130 or rng.random() < 0.35:
            comorb_list.append("Dyslipidemia")
        if (duration_years >= 6.0 and hba1c >= 8.0) or (tier >= 2 and rng.random() < 0.45):
            comorb_list.append("Neuropathy")
        if (duration_years >= 8.0 and hba1c >= 8.5) or (tier >= 2 and rng.random() < 0.40):
            comorb_list.append("Retinopathy")
        if creatinine >= 1.35 or egfr <= 60 or (tier == 3 and rng.random() < 0.50):
            comorb_list.append("CKD")
        
        comorbidities = ", ".join(comorb_list) if comorb_list else "None"

        # Medication Assignment according to clinical guidelines
        if hba1c < 7.0:
            medication = "Metformin"
        elif 7.0 <= hba1c < 8.2:
            medication = rng.choice([
                "Metformin",
                "Metformin + Glimepiride",
                "Metformin + Sitagliptin"
            ], p=[0.25, 0.50, 0.25])
        elif 8.2 <= hba1c < 9.8:
            medication = rng.choice([
                "Metformin + Glimepiride",
                "Metformin + Sitagliptin",
                "Metformin + Empagliflozin",
                "Metformin + Insulin"
            ], p=[0.25, 0.30, 0.25, 0.20])
        else:  # hba1c >= 9.8 (Severe / Critical)
            medication = rng.choice([
                "Insulin",
                "Metformin + Insulin",
                "Metformin + Glimepiride + Insulin",
                "Glimepiride + Insulin"
            ], p=[0.35, 0.45, 0.12, 0.08])

        # Historical Dates
        diag_days = int(duration_years * 365.25)
        diagnosis_date = (ref_date - timedelta(days=diag_days)).strftime("%Y-%m-%d")
        last_visit_days = int(rng.integers(5, 120))
        last_visit_date = (ref_date - timedelta(days=last_visit_days)).strftime("%Y-%m-%d")

        # ---------------------------------------------------------------
        # GENETIC MARKERS (T2D-associated SNPs prevalent in South Asians)
        # ---------------------------------------------------------------
        # TCF7L2 rs7903146: Strongest T2D risk locus worldwide.
        #   Genotypes: CC (wild-type), CT (heterozygous risk), TT (homozygous risk)
        #   Risk allele 'T' frequency ~30% in South Asians; higher in T2D patients.
        tcf7l2_probs = {0: [0.55, 0.35, 0.10], 1: [0.40, 0.42, 0.18],
                        2: [0.28, 0.44, 0.28], 3: [0.18, 0.40, 0.42]}
        tcf7l2 = rng.choice(["CC", "CT", "TT"], p=tcf7l2_probs[tier])

        # PPARG rs1801282 (Pro12Ala): Protective variant for insulin sensitivity.
        #   CG/GG = protective; CC = risk (wild-type).
        pparg_probs = {0: [0.70, 0.25, 0.05], 1: [0.78, 0.18, 0.04],
                       2: [0.85, 0.13, 0.02], 3: [0.90, 0.09, 0.01]}
        pparg = rng.choice(["CC", "CG", "GG"], p=pparg_probs[tier])

        # KCNJ11 rs5219 (E23K): Affects beta-cell potassium channel function.
        #   TT = risk; CC = wild-type.
        kcnj11_probs = {0: [0.50, 0.38, 0.12], 1: [0.38, 0.42, 0.20],
                        2: [0.28, 0.42, 0.30], 3: [0.20, 0.38, 0.42]}
        kcnj11 = rng.choice(["CC", "CT", "TT"], p=kcnj11_probs[tier])

        # SLC30A8 rs13266634: Zinc transporter in beta-cells.
        #   CC = risk; TT = protective.
        slc30a8_probs = {0: [0.22, 0.48, 0.30], 1: [0.32, 0.45, 0.23],
                         2: [0.42, 0.40, 0.18], 3: [0.55, 0.33, 0.12]}
        slc30a8 = rng.choice(["CC", "CT", "TT"], p=slc30a8_probs[tier])

        # FTO rs9939609: Obesity/T2D risk, affects appetite regulation.
        #   AA = risk; TT = wild-type. Higher in obese patients.
        fto_base = {0: [0.50, 0.38, 0.12], 1: [0.40, 0.40, 0.20],
                    2: [0.30, 0.40, 0.30], 3: [0.22, 0.38, 0.40]}
        fto = rng.choice(["TT", "TA", "AA"], p=fto_base[tier])

        # HLA-DQ risk haplotype: Associated with autoimmune T1D but also
        # contributes to latent autoimmune diabetes in adults (LADA) risk.
        #   Present/Absent, ~8-15% prevalence.
        hla_dq_risk = bool(rng.random() < (0.06 + 0.03 * tier))

        # Composite polygenic risk score (0.0 - 1.0)
        # Weights based on published GWAS effect sizes for South Asians
        risk_allele_count = (
            (1 if tcf7l2 == "CT" else 2 if tcf7l2 == "TT" else 0) * 1.40 +  # OR 1.40 per T allele
            (0 if pparg in ["CG", "GG"] else 1) * 0.80 +  # Protective absence
            (1 if kcnj11 == "CT" else 2 if kcnj11 == "TT" else 0) * 1.15 +
            (1 if slc30a8 == "CT" else 2 if slc30a8 == "CC" else 0) * 1.12 +
            (1 if fto == "TA" else 2 if fto == "AA" else 0) * 1.10 +
            (1.5 if hla_dq_risk else 0.0) +
            (1.0 if fam_history else 0.0)
        )
        max_score = (2 * 1.40 + 0.80 + 2 * 1.15 + 2 * 1.12 + 2 * 1.10 + 1.5 + 1.0)
        genetic_risk_score = round(float(risk_allele_count / max_score), 3)

        patient_records.append({
            "patient_id": pid,
            "name": full_name,
            "age": age,
            "gender": gender,
            "city": city,
            "bmi": bmi,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "hba1c": hba1c,
            "fasting_glucose": fasting_glucose,
            "blood_pressure_systolic": bp_systolic,
            "blood_pressure_diastolic": bp_diastolic,
            "cholesterol_total": cholesterol_total,
            "cholesterol_ldl": cholesterol_ldl,
            "cholesterol_hdl": cholesterol_hdl,
            "triglycerides": triglycerides,
            "creatinine": creatinine,
            "egfr": egfr,
            "diabetes_duration_years": duration_years,
            "medication": medication,
            "comorbidities": comorbidities,
            "family_history_diabetes": fam_history,
            "smoking_status": smoking_status,
            "physical_activity_level": activity_level,
            "diet_type": diet_type,
            "stress_level": stress_level,
            "sleep_hours_avg": sleep_hours_avg,
            "diagnosis_date": diagnosis_date,
            "last_visit_date": last_visit_date,
            # Genetic Markers (T2D-associated SNPs)
            "tcf7l2_rs7903146": tcf7l2,
            "pparg_rs1801282": pparg,
            "kcnj11_rs5219": kcnj11,
            "slc30a8_rs13266634": slc30a8,
            "fto_rs9939609": fto,
            "hla_dq_risk": hla_dq_risk,
            "genetic_risk_score": genetic_risk_score,
        })

    return pd.DataFrame(patient_records)


# ==============================================================================
# 2. MEDICATIONS SCHEDULE GENERATION
# ==============================================================================

def generate_medications_schedule(ehr_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate detailed per-patient medication dosing schedules with frequencies,
    standardized timings, and realistic adherence rates.

    Args:
        ehr_df: Static EHR records containing 'patient_id' and 'medication'.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Tabular medication schedules for each patient.
    """
    rng = np.random.default_rng(seed)
    schedules = []

    med_spec = {
        "Metformin": {
            "dosage": ["500 mg", "850 mg", "1000 mg"],
            "frequency": "Twice daily with meals",
            "timing_hours": "08:00, 20:30",
            "adherence_range": (0.82, 0.98)
        },
        "Glimepiride": {
            "dosage": ["1 mg", "2 mg", "3 mg"],
            "frequency": "Once daily 30 min before breakfast",
            "timing_hours": "07:30",
            "adherence_range": (0.78, 0.96)
        },
        "Sitagliptin": {
            "dosage": ["50 mg", "100 mg"],
            "frequency": "Once daily with breakfast",
            "timing_hours": "08:15",
            "adherence_range": (0.85, 0.98)
        },
        "Empagliflozin": {
            "dosage": ["10 mg", "25 mg"],
            "frequency": "Once daily morning with water",
            "timing_hours": "08:00",
            "adherence_range": (0.84, 0.97)
        },
        "Insulin": {
            "dosage": ["16 units", "24 units", "32 units"],
            "frequency": "Twice daily subcutaneous before meals",
            "timing_hours": "07:45, 20:00",
            "adherence_range": (0.75, 0.94)
        }
    }

    for _, row in ehr_df.iterrows():
        pid = row["patient_id"]
        med_str = str(row["medication"])
        med_tokens = [m.strip() for m in med_str.split("+")]

        for med in med_tokens:
            if med in med_spec:
                spec = med_spec[med]
                dosage = rng.choice(spec["dosage"])
                adh_low, adh_high = spec["adherence_range"]
                adherence = round(float(rng.uniform(adh_low, adh_high)), 2)

                schedules.append({
                    "patient_id": pid,
                    "medication": med,
                    "dosage": dosage,
                    "frequency": spec["frequency"],
                    "timing_hours": spec["timing_hours"],
                    "adherence_rate": adherence
                })

    return pd.DataFrame(schedules)


# ==============================================================================
# 3. WEARABLE / CGM TIME-SERIES & EVENT GENERATION
# ==============================================================================

def simulate_patient_cgm(
    patient_row: pd.Series,
    days: int = 30,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Vectorized high-performance simulation of 5-minute continuous glucose and
    wearable telemetry for a single patient over the specified number of days.

    Key Physiological Features:
        - Postprandial spikes for Indian meals (breakfast ~08:15, lunch ~13:15,
          evening tea/snack ~17:15, dinner ~20:30)
        - Dawn phenomenon (04:00 - 08:30 glucose rise)
        - Exercise-induced glucose reductions and heart rate surges
        - 90-minute sleep cycling (Light, Deep, REM, Awake) and nocturnal stability
        - AR(1) colored autoregressive noise to mimic continuous interstitial sensor dynamics
        - Dangerous spikes (>250 mg/dL) for poorly controlled patients

    Args:
        patient_row: Series containing patient EHR attributes.
        days: Number of days to simulate (default: 30 days = 8,640 steps).
        seed: Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, List[Dict]]:
            - DataFrame with CGM and wearable columns
            - List of event records for spike/hypoglycemia/drop events
    """
    rng = np.random.default_rng(seed)

    pid = patient_row["patient_id"]
    hba1c = float(patient_row["hba1c"])
    fasting_g = float(patient_row["fasting_glucose"])
    bmi = float(patient_row["bmi"])
    act_level = str(patient_row["physical_activity_level"])
    stress_cat = str(patient_row["stress_level"])
    sleep_hours_avg = float(patient_row["sleep_hours_avg"])
    medication = str(patient_row["medication"])

    steps_per_day = 288  # 24 hours * 12 readings/hour
    total_steps = days * steps_per_day

    # Construct datetime grid
    time_grid = pd.date_range("2026-01-01 00:00:00", periods=total_steps, freq="5min")
    minutes = (time_grid.hour * 60 + time_grid.minute).values
    hours = minutes / 60.0
    day_indices = np.arange(total_steps) // steps_per_day

    # 1. Baseline Glucose Calculation
    # Uncontrolled patients have higher resting setpoints
    base_glucose = fasting_g + (hba1c - 6.5) * 5.5

    # 2. Dawn Phenomenon Surge (Active 04:00 to 08:30, centered at 06:30)
    dawn_amplitude = 16.0 + 3.5 * max(0.0, hba1c - 6.0)
    dawn_effect = dawn_amplitude * np.exp(-((hours - 6.5) ** 2) / (2.0 * (1.1 ** 2)))

    # 3. Post-Meal Glucose Spikes
    insulin_resistance = (1.0 + 0.14 * max(0.0, hba1c - 7.0) + 0.035 * max(0.0, bmi - 25.0))
    med_blunting = 0.88 if "Insulin" in medication else 1.0

    meal_effect = np.zeros(total_steps, dtype=np.float32)

    for d in range(days):
        day_mask = (day_indices == d)
        d_hours = hours[day_mask]

        # Breakfast: ~08:15 (+/- 25 min)
        t_bfast = 8.25 + rng.uniform(-0.4, 0.4)
        carb_bfast = rng.normal(42.0, 8.0)
        spike_bfast = max(22.0, (carb_bfast * 1.15) * insulin_resistance * med_blunting)
        dt_b = d_hours - t_bfast
        mask_b = (dt_b >= 0.0) & (dt_b <= 3.5)
        meal_effect[day_mask] += np.where(
            mask_b,
            spike_bfast * (dt_b / 1.1) * np.exp(1.0 - (dt_b / 1.1)),
            0.0
        )

        # Lunch: ~13:15 (+/- 30 min)
        t_lunch = 13.25 + rng.uniform(-0.5, 0.5)
        carb_lunch = rng.normal(68.0, 12.0)
        spike_lunch = max(32.0, (carb_lunch * 1.30) * insulin_resistance * med_blunting)
        dt_l = d_hours - t_lunch
        mask_l = (dt_l >= 0.0) & (dt_l <= 4.0)
        meal_effect[day_mask] += np.where(
            mask_l,
            spike_lunch * (dt_l / 1.15) * np.exp(1.0 - (dt_l / 1.15)),
            0.0
        )

        # Evening Chai & Snack: ~17:15 (+/- 20 min)
        t_snack = 17.25 + rng.uniform(-0.35, 0.35)
        carb_snack = rng.normal(26.0, 6.0)
        spike_snack = max(14.0, (carb_snack * 0.95) * insulin_resistance * med_blunting)
        dt_s = d_hours - t_snack
        mask_s = (dt_s >= 0.0) & (dt_s <= 3.0)
        meal_effect[day_mask] += np.where(
            mask_s,
            spike_snack * (dt_s / 0.95) * np.exp(1.0 - (dt_s / 0.95)),
            0.0
        )

        # Dinner: ~20:30 (+/- 35 min) - carb heavy, prolonged peak
        t_dinner = 20.5 + rng.uniform(-0.55, 0.55)
        carb_dinner = rng.normal(72.0, 15.0)
        spike_dinner = max(38.0, (carb_dinner * 1.40) * insulin_resistance * med_blunting)
        dt_d = d_hours - t_dinner
        mask_d = (dt_d >= 0.0) & (dt_d <= 4.5)
        meal_effect[day_mask] += np.where(
            mask_d,
            spike_dinner * (dt_d / 1.25) * np.exp(1.0 - (dt_d / 1.25)),
            0.0
        )

    # 4. Activity, Steps, Heart Rate, Skin Temp, Stress Index, and Sleep Stages
    activity_type = np.full(total_steps, "Resting", dtype=object)
    sleep_stage = np.full(total_steps, "Not_Sleeping", dtype=object)
    steps_5m = np.zeros(total_steps, dtype=np.int32)
    heart_rate = np.zeros(total_steps, dtype=np.int32)
    skin_temp = np.zeros(total_steps, dtype=np.float32)
    stress_idx = np.zeros(total_steps, dtype=np.int32)
    exercise_effect = np.zeros(total_steps, dtype=np.float32)

    # Resting baseline HR
    hr_base = 68.0 + 0.45 * (bmi - 25.0) + (4.0 if act_level == "Sedentary" else 0.0)

    # Sleep period: 23:00 to (23:00 + sleep_hours_avg)
    wake_hour = (23.0 + sleep_hours_avg) % 24.0
    is_sleeping = (hours >= 23.0) | (hours < wake_hour)
    activity_type[is_sleeping] = "Sleeping"

    # Sleep stage 90-minute ultradian cycling
    mins_from_onset = np.where(minutes >= 1380, minutes - 1380, minutes + 60)
    cycle_pos = mins_from_onset % 90  # 0 to 89 minutes

    light_mask = is_sleeping & (cycle_pos < 45)
    deep_mask = is_sleeping & (cycle_pos >= 45) & (cycle_pos < 70)
    rem_mask = is_sleeping & (cycle_pos >= 70) & (cycle_pos < 85)
    awake_mask = is_sleeping & (cycle_pos >= 85)

    sleep_stage[light_mask] = "Light"
    sleep_stage[deep_mask] = "Deep"
    sleep_stage[rem_mask] = "REM"
    sleep_stage[awake_mask] = "Awake"

    # Nocturnal telemetry
    steps_5m[is_sleeping] = 0
    heart_rate[deep_mask] = rng.integers(int(hr_base - 14), int(hr_base - 8), size=int(np.sum(deep_mask)))
    heart_rate[light_mask] = rng.integers(int(hr_base - 10), int(hr_base - 4), size=int(np.sum(light_mask)))
    heart_rate[rem_mask] = rng.integers(int(hr_base - 6), int(hr_base + 2), size=int(np.sum(rem_mask)))
    heart_rate[awake_mask] = rng.integers(int(hr_base - 2), int(hr_base + 6), size=int(np.sum(awake_mask)))

    skin_temp[deep_mask] = np.round(rng.normal(36.0, 0.15, size=int(np.sum(deep_mask))), 1)
    skin_temp[light_mask | rem_mask | awake_mask] = np.round(
        rng.normal(36.2, 0.18, size=int(np.sum(light_mask | rem_mask | awake_mask))), 1
    )
    stress_idx[is_sleeping] = rng.integers(5, 20, size=int(np.sum(is_sleeping)))

    # Eating intervals
    meal_window = (
        ((hours >= 8.0) & (hours < 8.6)) |
        ((hours >= 13.0) & (hours < 13.7)) |
        ((hours >= 17.0) & (hours < 17.5)) |
        ((hours >= 20.25) & (hours < 20.95))
    ) & (~is_sleeping)
    activity_type[meal_window] = "Eating"
    steps_5m[meal_window] = rng.integers(0, 20, size=int(np.sum(meal_window)))
    heart_rate[meal_window] = rng.integers(int(hr_base + 4), int(hr_base + 14), size=int(np.sum(meal_window)))
    skin_temp[meal_window] = np.round(rng.normal(36.5, 0.2, size=int(np.sum(meal_window))), 1)
    stress_idx[meal_window] = rng.integers(18, 38, size=int(np.sum(meal_window)))

    # Exercise & Walking
    has_morning_walk = act_level in ["Moderate", "Active"]
    has_evening_exercise = act_level == "Active"

    for d in range(days):
        day_mask = (day_indices == d)
        
        # Morning Walk (06:30 to 07:15)
        if has_morning_walk and (rng.random() < 0.85):
            walk_mask = day_mask & (hours >= 6.5) & (hours < 7.25) & (~is_sleeping)
            n_w = int(np.sum(walk_mask))
            if n_w > 0:
                activity_type[walk_mask] = "Walking"
                steps_5m[walk_mask] = rng.integers(150, 320, size=n_w)
                heart_rate[walk_mask] = rng.integers(94, 116, size=n_w)
                skin_temp[walk_mask] = np.round(rng.normal(36.8, 0.22, size=n_w), 1)
                stress_idx[walk_mask] = rng.integers(25, 45, size=n_w)

                walk_indices = np.where(walk_mask)[0]
                post_indices = np.arange(walk_indices[0], min(total_steps, walk_indices[-1] + 12))
                exercise_effect[post_indices] -= np.linspace(5.0, 24.0, len(post_indices))

        # Evening Exercise (18:15 to 19:00)
        if has_evening_exercise and (rng.random() < 0.75):
            ex_mask = day_mask & (hours >= 18.25) & (hours < 19.0) & (~is_sleeping)
            n_ex = int(np.sum(ex_mask))
            if n_ex > 0:
                activity_type[ex_mask] = "Exercising"
                steps_5m[ex_mask] = rng.integers(280, 480, size=n_ex)
                heart_rate[ex_mask] = rng.integers(125, 155, size=n_ex)
                skin_temp[ex_mask] = np.round(rng.normal(37.3, 0.28, size=n_ex), 1)
                stress_idx[ex_mask] = rng.integers(55, 82, size=n_ex)

                ex_indices = np.where(ex_mask)[0]
                post_ex = np.arange(ex_indices[0], min(total_steps, ex_indices[-1] + 18))
                exercise_effect[post_ex] -= np.linspace(10.0, 42.0, len(post_ex))

    # Working hours
    work_mask = (
        ((hours >= 9.5) & (hours < 13.0)) |
        ((hours >= 14.0) & (hours < 17.0)) |
        ((hours >= 17.5) & (hours < 18.25))
    ) & (activity_type == "Resting") & (~is_sleeping)
    
    n_work = int(np.sum(work_mask))
    if n_work > 0:
        activity_type[work_mask] = "Working"
        steps_5m[work_mask] = rng.integers(5, 45, size=n_work)
        heart_rate[work_mask] = rng.integers(int(hr_base + 3), int(hr_base + 16), size=n_work)
        skin_temp[work_mask] = np.round(rng.normal(36.5, 0.2, size=n_work), 1)
        stress_base_min = 45 if stress_cat == "High" else (30 if stress_cat == "Moderate" else 15)
        stress_idx[work_mask] = rng.integers(stress_base_min, stress_base_min + 30, size=n_work)

    # Resting
    rest_mask = (activity_type == "Resting") & (~is_sleeping)
    n_rest = int(np.sum(rest_mask))
    if n_rest > 0:
        steps_5m[rest_mask] = rng.integers(0, 25, size=n_rest)
        heart_rate[rest_mask] = rng.integers(int(hr_base - 3), int(hr_base + 8), size=n_rest)
        skin_temp[rest_mask] = np.round(rng.normal(36.4, 0.18, size=n_rest), 1)
        stress_idx[rest_mask] = rng.integers(15, 35, size=n_rest)

    # 5. Continuous Glucose Assembly & AR(1) Autoregressive Colored Noise
    noise = np.zeros(total_steps, dtype=np.float32)
    white = rng.normal(0, 4.2, total_steps)
    phi = 0.88  # Autocorrelation creates smooth interstitial CGM signal
    for t in range(1, total_steps):
        noise[t] = phi * noise[t - 1] + np.sqrt(1.0 - (phi ** 2)) * white[t]

    raw_glucose = base_glucose + dawn_effect + meal_effect + exercise_effect + noise

    # Bounds & Correlations
    skin_temp = np.clip(skin_temp, 33.0, 38.0)
    heart_rate = np.clip(heart_rate, 40, 150)
    stress_idx = np.clip(stress_idx, 0, 100)

    # Elevated stress during acute glucose spikes (>200 mg/dL)
    high_g_mask = (raw_glucose > 200.0) & (~is_sleeping)
    stress_idx[high_g_mask] = np.clip(stress_idx[high_g_mask] + 15, 0, 100)

    # Final glucose values bounded strictly to [40.0, 400.0]
    final_glucose = np.round(np.clip(raw_glucose, 40.0, 400.0), 1)

    # Heart Rate Variability (HRV RMSSD in ms):
    # Inversely correlated with heart rate, stress, and hyperglycemia; elevated during deep sleep
    hrv_base = 48.0 - 2.2 * max(0.0, hba1c - 6.0) - 0.25 * max(0.0, float(patient_row["age"]) - 40.0)
    hrv_raw = (
        hrv_base
        - 0.32 * (heart_rate - hr_base)
        - 0.18 * stress_idx
        + np.where(deep_mask, 12.0, 0.0)
        + np.where(high_g_mask, -6.5, 0.0)
        + rng.normal(0, 3.2, total_steps)
    )
    hrv_rmssd_ms = np.round(np.clip(hrv_raw, 12.0, 95.0), 1)

    cgm_df = pd.DataFrame({
        "patient_id": pid,
        "timestamp": time_grid.strftime("%Y-%m-%d %H:%M:%S"),
        "glucose_mg_dl": final_glucose,
        "heart_rate_bpm": heart_rate,
        "hrv_rmssd_ms": hrv_rmssd_ms,
        "steps_last_5min": steps_5m,
        "sleep_stage": sleep_stage,
        "skin_temperature_c": skin_temp,
        "stress_index": stress_idx,
        "activity_type": activity_type
    })

    # 6. Extract Discrete Glucose Events
    events = extract_patient_events(pid, time_grid, final_glucose)

    return cgm_df, events


def extract_patient_events(
    patient_id: str,
    time_grid: pd.DatetimeIndex,
    glucose: np.ndarray
) -> List[Dict]:
    """
    Scan continuous CGM data to extract clinically significant glycemic events:
    - Spikes: episodes where glucose > 180 mg/dL (mild: 180-215, moderate: 215-250, severe: >250)
    - Hypoglycemia: episodes where glucose < 70 mg/dL (mild: 60-70, moderate: 54-60, severe: <54)
    - Rapid Drops: episodes with drops > 40 mg/dL within 30-45 minutes

    Args:
        patient_id: Patient identifier.
        time_grid: Datetime index of readings.
        glucose: Array of continuous glucose readings.

    Returns:
        List[Dict]: List of detected event dictionaries.
    """
    events = []
    n = len(glucose)
    i = 0

    while i < n:
        g = glucose[i]

        # 1. Spike Event Detection (Glucose > 180 mg/dL)
        if g > 180.0:
            start_idx = i
            peak_val = g
            peak_idx = i
            preceding_idx = max(0, start_idx - 6)  # 30 minutes prior
            preceding_g = round(float(glucose[preceding_idx]), 1)

            while i < n and glucose[i] > 180.0:
                if glucose[i] > peak_val:
                    peak_val = glucose[i]
                    peak_idx = i
                i += 1
            
            end_idx = i - 1
            duration_min = (end_idx - start_idx + 1) * 5

            if peak_val >= 250.0:
                sev = "severe"
            elif peak_val >= 215.0:
                sev = "moderate"
            else:
                sev = "mild"

            events.append({
                "patient_id": patient_id,
                "event_timestamp": time_grid[peak_idx].strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "spike",
                "severity": sev,
                "preceding_glucose": preceding_g,
                "peak_glucose": round(float(peak_val), 1),
                "duration_minutes": duration_min
            })
            continue

        # 2. Hypoglycemia Event Detection (Glucose < 70 mg/dL)
        elif g < 70.0:
            start_idx = i
            nadir_val = g
            nadir_idx = i
            preceding_idx = max(0, start_idx - 6)
            preceding_g = round(float(glucose[preceding_idx]), 1)

            while i < n and glucose[i] < 70.0:
                if glucose[i] < nadir_val:
                    nadir_val = glucose[i]
                    nadir_idx = i
                i += 1

            end_idx = i - 1
            duration_min = (end_idx - start_idx + 1) * 5

            if nadir_val < 54.0:
                sev = "severe"
            elif nadir_val < 60.0:
                sev = "moderate"
            else:
                sev = "mild"

            events.append({
                "patient_id": patient_id,
                "event_timestamp": time_grid[nadir_idx].strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "hypoglycemia",
                "severity": sev,
                "preceding_glucose": preceding_g,
                "peak_glucose": round(float(nadir_val), 1),
                "duration_minutes": duration_min
            })
            continue

        # 3. Rapid Drop Detection (Drop > 40 mg/dL over 6 steps = 30 min)
        if i >= 6:
            drop = glucose[i - 6] - g
            if drop >= 40.0 and g >= 70.0:
                if drop >= 75.0:
                    sev = "severe"
                elif drop >= 55.0:
                    sev = "moderate"
                else:
                    sev = "mild"

                events.append({
                    "patient_id": patient_id,
                    "event_timestamp": time_grid[i].strftime("%Y-%m-%d %H:%M:%S"),
                    "event_type": "rapid_drop",
                    "severity": sev,
                    "preceding_glucose": round(float(glucose[i - 6]), 1),
                    "peak_glucose": round(float(g), 1),
                    "duration_minutes": 30
                })
                i += 6
                continue

        i += 1

    return events


# ==============================================================================
# 4. MASTER PIPELINE EXECUTION
# ==============================================================================

def generate_all_data(
    n_patients: int = 200,
    days: int = 30,
    output_dir: Optional[Union[str, Path]] = None,
    seed: int = 42
) -> Dict[str, Union[pd.DataFrame, Path]]:
    """
    Generate the complete synthetic dataset for GlucoTwin:
        1. data/raw/ehr_patients.csv
        2. data/raw/medications_schedule.csv
        3. data/raw/cgm_readings.csv
        4. data/raw/glucose_events.csv

    Args:
        n_patients: Number of patient records to generate (default: 200).
        days: Days of continuous telemetry per patient (default: 30).
        output_dir: Target output directory (default: data/raw).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict: Paths and loaded dataframes.
    """
    start_time = time.time()
    out_path = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GlucoTwin Synthetic Data Generation System")
    print(f"Target: {n_patients} Indian Type 2 Diabetes Patients | {days} Days Dynamic Telemetry")
    print(f"Output Directory: {out_path}")
    print("=" * 80)

    # Step 1: Generate Static EHR Patients
    print("\n[1/4] Generating Static EHR Patient Records...")
    ehr_df = generate_ehr_data(n_patients=n_patients, seed=seed)
    ehr_file = out_path / "ehr_patients.csv"
    ehr_df.to_csv(ehr_file, index=False)
    print(f"  -> Saved {len(ehr_df)} EHR records to: {ehr_file}")

    # Step 2: Generate Medication Schedules
    print("\n[2/4] Generating Patient Medication Schedules...")
    meds_df = generate_medications_schedule(ehr_df, seed=seed + 10)
    meds_file = out_path / "medications_schedule.csv"
    meds_df.to_csv(meds_file, index=False)
    print(f"  -> Saved {len(meds_df)} medication schedule rows to: {meds_file}")

    # Step 3 & 4: Stream Generate CGM Readings and Events
    cgm_file = out_path / "cgm_readings.csv"
    events_file = out_path / "glucose_events.csv"

    print(f"\n[3/4 & 4/4] Generating 5-Min CGM Telemetry & Event Detection...")
    print(f"  Expected CGM Rows: {n_patients:,} patients x {days} days x 288 readings/day = {n_patients * days * 288:,} rows")

    if cgm_file.exists():
        cgm_file.unlink()
    if events_file.exists():
        events_file.unlink()

    all_events = []
    total_cgm_rows = 0
    batch_dfs = []
    BATCH_SIZE = 25

    with tqdm(total=n_patients, desc="Simulating Patients", unit="patient") as pbar:
        for idx, (_, patient_row) in enumerate(ehr_df.iterrows()):
            pat_seed = seed + 100 + idx
            pat_cgm_df, pat_events = simulate_patient_cgm(patient_row, days=days, seed=pat_seed)
            
            batch_dfs.append(pat_cgm_df)
            all_events.extend(pat_events)
            total_cgm_rows += len(pat_cgm_df)

            if len(batch_dfs) >= BATCH_SIZE or idx == n_patients - 1:
                chunk_df = pd.concat(batch_dfs, ignore_index=True)
                write_header = not cgm_file.exists()
                chunk_df.to_csv(cgm_file, mode="a", header=write_header, index=False)
                batch_dfs.clear()

            pbar.update(1)

    # Save glucose events
    events_df = pd.DataFrame(all_events)
    events_df.to_csv(events_file, index=False)
    print(f"  -> Saved {len(events_df):,} glucose events to: {events_file}")
    print(f"  -> Saved {total_cgm_rows:,} CGM readings to: {cgm_file}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Generation Complete in {elapsed:.2f} seconds!")
    print(f"  - EHR Patients:       {len(ehr_df)} rows ({ehr_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - Med Schedules:      {len(meds_df)} rows ({meds_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - Glucose Events:     {len(events_df)} rows ({events_file.stat().st_size / 1024:.1f} KB)")
    print(f"  - CGM Readings:       {total_cgm_rows:,} rows ({cgm_file.stat().st_size / (1024 * 1024):.1f} MB)")
    print("=" * 80)

    return {
        "ehr_df": ehr_df,
        "meds_df": meds_df,
        "events_df": events_df,
        "ehr_path": ehr_file,
        "meds_path": meds_file,
        "cgm_path": cgm_file,
        "events_path": events_file
    }


if __name__ == "__main__":
    generate_all_data(n_patients=200, days=30)
