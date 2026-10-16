"""GlucoTwin: Static EHR Feature Engineering Module"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Setup module-level logger
logger = logging.getLogger("glucotwin.static_features")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# Clinical Reference Standards for Indian Type 2 Diabetes Cohorts
# Asian Indian BMI cutoffs per ICMR and Consensus Guidelines (Misra et al., 2009):
# Underweight: < 18.5 kg/m2
# Normal: 18.5 - 22.9 kg/m2
# Overweight: 23.0 - 24.9 kg/m2
# Obese Class I: 25.0 - 29.9 kg/m2
# Obese Class II: 30.0 - 34.9 kg/m2
# Obese Class III: >= 35.0 kg/m2

COMORBIDITY_SEVERITY_WEIGHTS = {
    "hypertension": {"mild": 1.0, "moderate": 2.0, "severe": 3.0, "default": 1.5},
    "dyslipidemia": {"mild": 1.0, "moderate": 1.5, "severe": 2.5, "default": 1.5},
    "neuropathy": {"mild": 1.5, "moderate": 2.5, "severe": 3.5, "default": 2.0},
    "retinopathy": {"mild": 1.5, "moderate": 2.5, "severe": 4.0, "default": 2.5},
    "nephropathy": {"mild": 2.0, "moderate": 3.0, "severe": 4.5, "default": 3.0},
    "cad": {"mild": 2.5, "moderate": 3.5, "severe": 5.0, "default": 3.5},
    "cardiovascular": {"mild": 2.5, "moderate": 3.5, "severe": 5.0, "default": 3.5},
    "hypothyroidism": {"mild": 0.5, "moderate": 1.0, "severe": 1.5, "default": 1.0},
    "nafld": {"mild": 1.0, "moderate": 1.5, "severe": 2.5, "default": 1.5},
    "pad": {"mild": 1.5, "moderate": 2.5, "severe": 3.5, "default": 2.0},
}


def _classify_age_group(age: float) -> Tuple[str, int]:
    """Classifies age into demographic buckets and ordinal code."""
    if pd.isna(age):
        return ("45-55", 1)  # Median fallback
    age_val = float(age)
    if age_val < 35:
        return ("<35", 0)
    elif age_val < 45:
        return ("35-45", 1)
    elif age_val < 55:
        return ("45-55", 2)
    elif age_val < 65:
        return ("55-65", 3)
    else:
        return ("65+", 4)


def _classify_bmi(bmi: float) -> Tuple[str, int]:
    """Classifies BMI according to Asian-Indian specific criteria (ICMR guidelines)."""
    if pd.isna(bmi) or bmi <= 0:
        return ("Normal", 1)
    bmi_val = float(bmi)
    if bmi_val < 18.5:
        return ("Underweight", 0)
    elif bmi_val < 23.0:
        return ("Normal", 1)
    elif bmi_val < 25.0:
        return ("Overweight", 2)
    elif bmi_val < 30.0:
        return ("Obese_I", 3)
    elif bmi_val < 35.0:
        return ("Obese_II", 4)
    else:
        return ("Obese_III", 5)


def _classify_hba1c(hba1c: float) -> Tuple[str, int]:
    """Classifies HbA1c into clinical glycemic control tiers:"""
    if pd.isna(hba1c) or hba1c <= 0:
        return ("Moderate", 1)
    val = float(hba1c)
    if val < 7.0:
        return ("Well-controlled", 0)
    elif val <= 9.0:
        return ("Moderate", 1)
    elif val <= 11.0:
        return ("Poorly-controlled", 2)
    else:
        return ("Critical", 3)


def _classify_diabetes_duration(years: float) -> Tuple[str, int]:
    """Classifies diabetes duration risk:"""
    if pd.isna(years) or years < 0:
        return ("Established", 1)
    val = float(years)
    if val < 2.0:
        return ("New", 0)
    elif val <= 10.0:
        return ("Established", 1)
    else:
        return ("Long-standing", 2)


def _classify_egfr(egfr: float) -> Tuple[str, int]:
    """Classifies estimated Glomerular Filtration Rate (KDIGO CKD Staging):"""
    if pd.isna(egfr) or egfr <= 0:
        return ("G1", 0)
    val = float(egfr)
    if val >= 90.0:
        return ("G1", 0)
    elif val >= 60.0:
        return ("G2", 1)
    elif val >= 45.0:
        return ("G3a", 2)
    elif val >= 30.0:
        return ("G3b", 3)
    elif val >= 15.0:
        return ("G4", 4)
    else:
        return ("G5", 5)


def _parse_comorbidities(val: Union[List, str, None]) -> Tuple[int, float, Dict[str, int]]:
    """Parses comorbidity entries from list or string, computing total count,"""
    flags = {
        "has_hypertension": 0,
        "has_dyslipidemia": 0,
        "has_neuropathy": 0,
        "has_retinopathy": 0,
        "has_nephropathy": 0,
        "has_cad": 0,
        "has_nafld": 0,
    }
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0, 0.0, flags

    items = []
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        # Could be comma-separated or plus-separated (e.g. 'Metformin + Empagliflozin')
        clean_str = val.strip("[]'\" ").replace("+", ",")
        if clean_str:
            items = [s.strip() for s in clean_str.split(",") if s.strip()]

    count = 0
    total_severity = 0.0

    for item in items:
        name_str = ""
        sev_str = "moderate"
        if isinstance(item, dict):
            name_str = str(item.get("name", "")).lower()
            sev_str = str(item.get("severity", "moderate")).lower()
        elif isinstance(item, str):
            name_str = item.lower()
            if "mild" in name_str:
                sev_str = "mild"
            elif "severe" in name_str:
                sev_str = "severe"

        matched = False
        for condition, weight_map in COMORBIDITY_SEVERITY_WEIGHTS.items():
            if condition in name_str:
                matched = True
                w = weight_map.get(sev_str, weight_map["default"])
                total_severity += w
                if condition == "hypertension":
                    flags["has_hypertension"] = 1
                elif condition == "dyslipidemia":
                    flags["has_dyslipidemia"] = 1
                elif condition == "neuropathy":
                    flags["has_neuropathy"] = 1
                elif condition == "retinopathy":
                    flags["has_retinopathy"] = 1
                elif condition in ("nephropathy", "kidney"):
                    flags["has_nephropathy"] = 1
                elif condition in ("cad", "cardiovascular", "heart"):
                    flags["has_cad"] = 1
                elif condition in ("nafld", "fatty liver"):
                    flags["has_nafld"] = 1
                break

        if matched:
            count += 1
        elif name_str:
            count += 1
            total_severity += 1.0  # Generic comorbidity weight

    return count, total_severity, flags


def _parse_medications(val: Union[List, str, None]) -> Tuple[int, float, Dict[str, int]]:
    """Parses medications list or string, computing drug count,"""
    flags = {
        "uses_insulin": 0,
        "uses_metformin": 0,
        "uses_sulfonylurea": 0,
        "uses_sglt2": 0,
        "uses_dpp4": 0,
        "uses_statin": 0,
        "uses_antihypertensive": 0,
    }
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 0, 0.0, flags

    items = []
    if isinstance(val, list):
        items = val
    elif isinstance(val, str):
        clean_str = val.strip("[]'\" ").replace("+", ",")
        if clean_str:
            items = [s.strip() for s in clean_str.split(",") if s.strip()]

    drug_count = len(items)
    complexity = 0.0

    for item in items:
        med_str = str(item).lower()
        if "insulin" in med_str or "glargine" in med_str or "lispro" in med_str or "aspart" in med_str:
            flags["uses_insulin"] = 1
            complexity += 3.0
        elif "metformin" in med_str:
            flags["uses_metformin"] = 1
            complexity += 1.0
        elif any(s in med_str for s in ["glimepiride", "gliclazide", "glipizide"]):
            flags["uses_sulfonylurea"] = 1
            complexity += 2.0
        elif any(s in med_str for s in ["dapagliflozin", "empagliflozin", "canagliflozin"]):
            flags["uses_sglt2"] = 1
            complexity += 1.5
        elif any(s in med_str for s in ["teneligliptin", "vildagliptin", "sitagliptin", "linagliptin"]):
            flags["uses_dpp4"] = 1
            complexity += 1.5
        elif any(s in med_str for s in ["atorvastatin", "rosuvastatin", "simvastatin"]):
            flags["uses_statin"] = 1
            complexity += 0.5
        elif any(s in med_str for s in ["telmisartan", "amlodipine", "losartan", "ramipril"]):
            flags["uses_antihypertensive"] = 1
            complexity += 0.5
        else:
            complexity += 1.0

    return drug_count, complexity, flags


def _compute_lifestyle_score(
    activity_level: str,
    diet_type: str,
    protein_intake_g: float,
    sleep_hours: float,
    stress_level: float,
    weight_kg: float = 70.0,
) -> float:
    """Computes a composite lifestyle score on a 0 to 100 scale:"""
    # 1. Activity score (0-25)
    act_map = {
        "sedentary": 5.0,
        "light": 12.0,
        "moderate": 20.0,
        "vigorous": 25.0,
    }
    act_score = act_map.get(str(activity_level).lower(), 12.0)

    # 2. Diet & Protein score (0-25)
    target_protein = max(35.0, float(weight_kg) * 0.8)
    protein_ratio = min(1.5, max(0.0, float(protein_intake_g) / target_protein))
    diet_score = min(25.0, protein_ratio * 20.0)
    if "vegetarian" in str(diet_type).lower():
        # High-carb Indian vegetarian diets without sufficient protein receive a slight offset
        if protein_ratio < 0.7:
            diet_score = max(5.0, diet_score - 5.0)

    # 3. Sleep score (0-25)
    s_hours = float(sleep_hours) if not pd.isna(sleep_hours) else 7.0
    if 7.0 <= s_hours <= 8.5:
        sleep_score = 25.0
    elif 6.0 <= s_hours < 7.0 or 8.5 < s_hours <= 9.5:
        sleep_score = 18.0
    elif 5.0 <= s_hours < 6.0:
        sleep_score = 10.0
    else:
        sleep_score = 5.0

    # 4. Stress score (0-25, inverted: lower stress = higher score)
    stress_val = float(stress_level) if not pd.isna(stress_level) else 5.0
    stress_score = max(0.0, min(25.0, (10.0 - stress_val) * 2.5))

    return round(act_score + diet_score + sleep_score + stress_score, 2)


def _compute_cv_risk_composite(
    sbp: float,
    dbp: float,
    total_chol: float,
    ldl: float,
    hdl: float,
    triglycerides: float,
    has_cad: int,
) -> float:
    """Calculates a cardiovascular composite risk score (0 to 100):"""
    score = 0.0

    # Systolic blood pressure
    sbp_val = float(sbp) if not pd.isna(sbp) and sbp > 50 else 125.0
    if sbp_val >= 160:
        score += 25.0
    elif sbp_val >= 140:
        score += 18.0
    elif sbp_val >= 130:
        score += 10.0
    else:
        score += 3.0

    # Diastolic blood pressure
    dbp_val = float(dbp) if not pd.isna(dbp) and dbp > 30 else 80.0
    if dbp_val >= 100:
        score += 15.0
    elif dbp_val >= 90:
        score += 10.0
    else:
        score += 2.0

    # Triglyceride / HDL ratio (Key marker for South Asian insulin resistance)
    tg_val = float(triglycerides) if not pd.isna(triglycerides) and triglycerides > 20 else 160.0
    hdl_val = float(hdl) if not pd.isna(hdl) and hdl > 10 else 42.0
    tg_hdl_ratio = tg_val / hdl_val

    if tg_hdl_ratio >= 4.5:
        score += 25.0
    elif tg_hdl_ratio >= 3.0:
        score += 16.0
    elif tg_hdl_ratio >= 2.0:
        score += 8.0
    else:
        score += 2.0

    # LDL Cholesterol
    ldl_val = float(ldl) if not pd.isna(ldl) and ldl > 20 else 110.0
    if ldl_val >= 160:
        score += 20.0
    elif ldl_val >= 130:
        score += 14.0
    elif ldl_val >= 100:
        score += 7.0
    else:
        score += 2.0

    # Prior coronary disease
    if has_cad:
        score += 15.0

    return min(100.0, round(score, 2))


def extract_static_features(ehr_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and engineers static patient features from raw or processed EHR data."""
    df = ehr_df.copy()

    # Standardize patient identifier column
    id_col = None
    for cand in ["patient_id", "id", "PatientID", "patientId"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        logger.warning("No patient ID column found; creating synthetic patient_id index.")
        df["patient_id"] = [f"PAT_{i:04d}" for i in range(len(df))]
        id_col = "patient_id"

    df["patient_id"] = df[id_col].astype(str)

    # Standardize column mappings with fallbacks
    col_map = {
        "age": ["age", "Age", "patient_age"],
        "gender": ["gender", "Gender", "sex"],
        "bmi": ["bmi", "BMI", "body_mass_index"],
        "hba1c": ["hba1c", "HbA1c", "hba1c_level", "a1c"],
        "duration": ["diabetes_duration_years", "diabetesDurationYears", "duration_years", "diabetes_duration"],
        "comorbidities": ["comorbidities", "Comorbidities", "conditions"],
        "medications": ["medications", "Medications", "medication", "Medication", "drugs", "prescriptions"],
        "sbp": ["blood_pressure_systolic", "systolic_bp", "sbp", "systolic"],
        "dbp": ["blood_pressure_diastolic", "diastolic_bp", "dbp", "diastolic"],
        "chol_total": ["cholesterol_total", "total_cholesterol", "cholesterol"],
        "ldl": ["cholesterol_ldl", "ldl_cholesterol", "ldl"],
        "hdl": ["cholesterol_hdl", "hdl_cholesterol", "hdl"],
        "tg": ["triglycerides", "Triglycerides", "tg"],
        "egfr": ["egfr", "eGFR", "gfr"],
        "diet": ["diet_type", "diet", "Diet"],
        "protein": ["daily_protein_intake_g", "protein_intake_g", "protein_grams"],
        "activity": ["physical_activity_level", "activity_level", "activity"],
        "sleep": ["sleep_hours_avg", "daily_sleep_hours", "sleep_hours"],
        "stress": ["stress_level", "stress", "stress_score"],
        "city": ["city", "City", "location"],
    }

    def _resolve(field_name: str, default=np.nan):
        for candidate in col_map[field_name]:
            if candidate in df.columns:
                return df[candidate]
        return pd.Series([default] * len(df), index=df.index)

    age_s = pd.to_numeric(_resolve("age", 52.0), errors="coerce").fillna(52.0)
    gender_s = _resolve("gender", "Male").astype(str)
    bmi_s = pd.to_numeric(_resolve("bmi", 26.5), errors="coerce").fillna(26.5)
    hba1c_s = pd.to_numeric(_resolve("hba1c", 8.2), errors="coerce").fillna(8.2)
    duration_s = pd.to_numeric(_resolve("duration", 6.0), errors="coerce").fillna(6.0)
    sbp_s = pd.to_numeric(_resolve("sbp", 132.0), errors="coerce").fillna(132.0)
    dbp_s = pd.to_numeric(_resolve("dbp", 84.0), errors="coerce").fillna(84.0)
    chol_s = pd.to_numeric(_resolve("chol_total", 195.0), errors="coerce").fillna(195.0)
    ldl_s = pd.to_numeric(_resolve("ldl", 115.0), errors="coerce").fillna(115.0)
    hdl_s = pd.to_numeric(_resolve("hdl", 44.0), errors="coerce").fillna(44.0)
    tg_s = pd.to_numeric(_resolve("tg", 175.0), errors="coerce").fillna(175.0)
    egfr_s = pd.to_numeric(_resolve("egfr", 82.0), errors="coerce").fillna(82.0)
    diet_s = _resolve("diet", "Vegetarian").astype(str)
    protein_s = pd.to_numeric(_resolve("protein", 48.0), errors="coerce").fillna(48.0)
    act_s = _resolve("activity", "light").astype(str)
    sleep_s = pd.to_numeric(_resolve("sleep", 6.8), errors="coerce").fillna(6.8)

    # Handle stress level as numeric or categorical string ('Low', 'Moderate', 'High')
    raw_stress = _resolve("stress", 5.0)
    parsed_stress = []
    stress_map = {"low": 3.0, "moderate": 6.0, "high": 8.5, "critical": 9.5}
    for val in raw_stress:
        if pd.isna(val):
            parsed_stress.append(5.0)
        elif isinstance(val, (int, float)):
            parsed_stress.append(float(val))
        else:
            s_str = str(val).strip().lower()
            parsed_stress.append(stress_map.get(s_str, 5.0))
    stress_s = pd.Series(parsed_stress, index=df.index)
    city_s = _resolve("city", "Mumbai").astype(str)

    comorb_raw = _resolve("comorbidities", None)
    meds_raw = _resolve("medications", None)

    # Begin building output feature frame
    features = pd.DataFrame(index=df.index)
    features["patient_id"] = df["patient_id"]
    features["age"] = age_s.astype(float)
    features["is_male"] = gender_s.str.lower().str.startswith("m").astype(int)

    # 1. Age group buckets
    age_groups_and_codes = [_classify_age_group(a) for a in age_s]
    features["age_group"] = [item[0] for item in age_groups_and_codes]
    features["age_group_code"] = [item[1] for item in age_groups_and_codes]

    # One-hot encode age groups
    for ag in ["<35", "35-45", "45-55", "55-65", "65+"]:
        features[f"age_group_{ag}"] = (features["age_group"] == ag).astype(int)

    # 2. BMI category (Asian Indian criteria)
    bmi_cats_and_codes = [_classify_bmi(b) for b in bmi_s]
    features["bmi"] = bmi_s.astype(float)
    features["bmi_category"] = [item[0] for item in bmi_cats_and_codes]
    features["bmi_category_code"] = [item[1] for item in bmi_cats_and_codes]

    for bc in ["Normal", "Overweight", "Obese_I", "Obese_II", "Obese_III"]:
        features[f"bmi_{bc}"] = (features["bmi_category"] == bc).astype(int)

    # 3. HbA1c risk tier
    hba1c_tiers_and_codes = [_classify_hba1c(h) for h in hba1c_s]
    features["hba1c"] = hba1c_s.astype(float)
    features["hba1c_tier"] = [item[0] for item in hba1c_tiers_and_codes]
    features["hba1c_tier_code"] = [item[1] for item in hba1c_tiers_and_codes]

    for tier in ["Well-controlled", "Moderate", "Poorly-controlled", "Critical"]:
        features[f"hba1c_{tier.lower().replace('-', '_')}"] = (features["hba1c_tier"] == tier).astype(int)

    # 4. Diabetes duration risk
    dur_tiers_and_codes = [_classify_diabetes_duration(d) for d in duration_s]
    features["diabetes_duration_years"] = duration_s.astype(float)
    features["duration_risk"] = [item[0] for item in dur_tiers_and_codes]
    features["duration_risk_code"] = [item[1] for item in dur_tiers_and_codes]

    for dr in ["New", "Established", "Long-standing"]:
        features[f"duration_{dr.lower().replace('-', '_')}"] = (features["duration_risk"] == dr).astype(int)

    # 5. Comorbidity count and severity score
    comorb_results = [_parse_comorbidities(c) for c in comorb_raw]
    features["comorbidity_count"] = [r[0] for r in comorb_results]
    features["comorbidity_severity_score"] = [r[1] for r in comorb_results]
    for c_flag in ["has_hypertension", "has_dyslipidemia", "has_neuropathy",
                   "has_retinopathy", "has_nephropathy", "has_cad", "has_nafld"]:
        features[c_flag] = [r[2][c_flag] for r in comorb_results]

    # 6. Medication complexity score
    med_results = [_parse_medications(m) for m in meds_raw]
    features["medication_count"] = [r[0] for r in med_results]
    features["medication_complexity_score"] = [r[1] for r in med_results]
    for m_flag in ["uses_insulin", "uses_metformin", "uses_sulfonylurea",
                   "uses_sglt2", "uses_dpp4", "uses_statin", "uses_antihypertensive"]:
        features[m_flag] = [r[2][m_flag] for r in med_results]

    # 7. Lifestyle composite score
    lifestyle_scores = []
    for act, diet, prot, slp, strss in zip(act_s, diet_s, protein_s, sleep_s, stress_s):
        score = _compute_lifestyle_score(act, diet, prot, slp, strss)
        lifestyle_scores.append(score)
    features["lifestyle_composite_score"] = lifestyle_scores
    features["daily_protein_intake_g"] = protein_s.astype(float)
    features["sleep_hours_avg"] = sleep_s.astype(float)
    features["stress_level"] = stress_s.astype(float)

    # 8. eGFR stage
    egfr_stages_and_codes = [_classify_egfr(g) for g in egfr_s]
    features["egfr"] = egfr_s.astype(float)
    features["egfr_stage"] = [item[0] for item in egfr_stages_and_codes]
    features["egfr_stage_code"] = [item[1] for item in egfr_stages_and_codes]

    for stg in ["G1", "G2", "G3a", "G3b", "G4", "G5"]:
        features[f"egfr_{stg.lower()}"] = (features["egfr_stage"] == stg).astype(int)

    # 9. Cardiovascular risk composite
    features["blood_pressure_systolic"] = sbp_s.astype(float)
    features["blood_pressure_diastolic"] = dbp_s.astype(float)
    features["cholesterol_total"] = chol_s.astype(float)
    features["cholesterol_ldl"] = ldl_s.astype(float)
    features["cholesterol_hdl"] = hdl_s.astype(float)
    features["triglycerides"] = tg_s.astype(float)
    features["tg_to_hdl_ratio"] = (tg_s / hdl_s.clip(lower=10.0)).round(2)

    cv_scores = []
    for s, d, c, l, h, t, cad in zip(sbp_s, dbp_s, chol_s, ldl_s, hdl_s, tg_s, features["has_cad"]):
        cv_scores.append(_compute_cv_risk_composite(s, d, c, l, h, t, cad))
    features["cardiovascular_risk_composite"] = cv_scores

    # 10. Indian-specific risk factors
    features["is_vegetarian"] = diet_s.str.lower().str.contains("veg").astype(int)
    # Recommended protein allowance ~ 0.8g per kg body weight; estimated weight 68kg ~ 54g
    features["protein_intake_adequacy"] = (protein_s / 55.0).clip(lower=0.1, upper=2.0).round(2)
    # Vegetarian diet with low protein intake -> high simple carb load
    features["carb_protein_imbalance"] = (
        features["is_vegetarian"] * np.maximum(0.0, 1.0 - features["protein_intake_adequacy"])
    ).round(3)

    # South Asian thin-fat metabolic phenotype:
    # High visceral adiposity / insulin resistance occurs even in normal or mild BMI (22-26) with elevated TG/HDL
    thin_fat_flag = ((features["bmi"] >= 22.0) & (features["bmi"] <= 27.0) & (features["tg_to_hdl_ratio"] > 3.0)).astype(int)
    features["south_asian_phenotype_score"] = (
        thin_fat_flag * 1.5 + (features["tg_to_hdl_ratio"] > 3.5).astype(int) * 1.0
    )

    # Urban Indian stress index:
    # Higher for metro cities (Mumbai, Delhi, Bengaluru) combined with high reported stress
    is_metro = city_s.str.lower().str.contains("mumbai|delhi|bengaluru|bangalore|hyderabad|pune").astype(int)
    features["urban_stress_index"] = (is_metro * (stress_s / 10.0)).round(2)

    logger.info(
        f"Extracted {features.shape[1]} static EHR features for {len(features)} patients."
    )
    return features


def generate_synthetic_ehr_cohort(num_patients: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generates a highly realistic synthetic EHR dataset of Indian Type 2 Diabetes patients."""
    np.random.seed(seed)

    indian_first_names_m = [
        "Rajesh", "Amit", "Vikram", "Suresh", "Ramesh", "Deepak", "Sunil", "Anil",
        "Manoj", "Sanjay", "Dinesh", "Pradeep", "Ashok", "Kishore", "Mahesh",
        "Arun", "Vijay", "Naveen", "Rahul", "Prakash", "Girish", "Sachin"
    ]
    indian_first_names_f = [
        "Priya", "Sunita", "Kavita", "Anita", "Meena", "Rekha", "Pooja", "Shweta",
        "Geeta", "Suman", "Neelam", "Archana", "Lata", "Usha", "Manju", "Savita",
        "Vandana", "Ritu", "Anjali", "Jyoti", "Divya", "Swati"
    ]
    indian_last_names = [
        "Sharma", "Patel", "Verma", "Deshmukh", "Nair", "Iyer", "Gupta", "Singh",
        "Joshi", "Kulkarni", "Mehta", "Reddy", "Rao", "Banerjee", "Chatterjee",
        "Chopra", "Malhotra", "Pandey", "Mishra", "Patil", "Chavan", "Shah"
    ]
    indian_cities = [
        "Mumbai, Maharashtra", "Pune, Maharashtra", "New Delhi, Delhi NCR",
        "Bengaluru, Karnataka", "Ahmedabad, Gujarat", "Chennai, Tamil Nadu",
        "Hyderabad, Telangana", "Kolkata, West Bengal", "Nagpur, Maharashtra",
        "Jaipur, Rajasthan", "Lucknow, Uttar Pradesh", "Chandigarh, Punjab"
    ]

    records = []
    for i in range(1, num_patients + 1):
        pid = f"IND-T2D-{i:03d}"
        gender = "Male" if np.random.rand() > 0.45 else "Female"
        if gender == "Male":
            first_name = np.random.choice(indian_first_names_m)
        else:
            first_name = np.random.choice(indian_first_names_f)
        last_name = np.random.choice(indian_last_names)
        full_name = f"{first_name} {last_name}"

        # Age distribution: 35 to 78, peak around 50-60
        age = int(np.clip(np.random.normal(54, 10), 35, 78))

        # BMI: Asian Indian peak ~ 27 kg/m2
        bmi = round(float(np.clip(np.random.normal(27.2, 4.0), 19.5, 39.0)), 1)

        # Diabetes duration: 1 to 24 years
        duration = round(float(np.clip(np.random.exponential(6.5), 0.5, 25.0)), 1)

        # HbA1c: baseline 6.4 to 12.5%
        # Strong correlation with duration and random metabolic variation
        hba1c_base = 6.8 + (duration / 25.0) * 2.2 + np.random.normal(0, 1.2)
        hba1c = round(float(np.clip(hba1c_base, 6.2, 12.8)), 1)

        # Blood pressure
        sbp = int(np.clip(np.random.normal(134 + (age - 50) * 0.4, 14), 105, 185))
        dbp = int(np.clip(np.random.normal(83 + (age - 50) * 0.2, 9), 65, 115))

        # Lipids: High TG, low HDL characteristic in South Asians
        tg = int(np.clip(np.random.normal(185 + (bmi - 25) * 4, 45), 90, 360))
        hdl = int(np.clip(np.random.normal(41 - (bmi - 25) * 0.5, 6), 26, 62))
        ldl = int(np.clip(np.random.normal(118 + (bmi - 25) * 2, 25), 65, 205))
        total_chol = ldl + hdl + int(tg / 5.0)

        # eGFR (kidney function): decreases with duration and age
        egfr = round(float(np.clip(95 - (age - 40) * 0.5 - duration * 1.1 + np.random.normal(0, 8), 22, 115)), 1)

        # Diet: ~60% vegetarian in urban Indian sample
        diet_type = np.random.choice(
            ["Vegetarian", "Jain Vegetarian", "Non-Vegetarian", "Eggetarian"],
            p=[0.50, 0.10, 0.30, 0.10],
        )

        # Daily protein intake: generally lower in vegetarians without supplementation
        if "vegetarian" in diet_type.lower():
            protein_g = round(float(np.clip(np.random.normal(42, 8), 25, 65)), 1)
        else:
            protein_g = round(float(np.clip(np.random.normal(58, 12), 35, 95)), 1)

        # Physical activity
        activity_level = np.random.choice(
            ["sedentary", "light", "moderate", "vigorous"],
            p=[0.40, 0.35, 0.20, 0.05],
        )

        sleep_hours = round(float(np.clip(np.random.normal(6.7, 1.1), 4.5, 9.5)), 1)
        stress_level = round(float(np.clip(np.random.normal(6.2, 1.8), 2.0, 9.5)), 1)
        city = np.random.choice(indian_cities)

        # Comorbidities synthesis
        comorbidities = []
        if sbp >= 135 or dbp >= 85 or np.random.rand() < 0.60:
            sev = "severe" if sbp >= 160 else ("moderate" if sbp >= 140 else "mild")
            comorbidities.append({"name": "Stage 1 Hypertension", "severity": sev, "icd10": "I10"})
        if tg > 160 or ldl > 120 or np.random.rand() < 0.55:
            sev = "severe" if (tg > 250 or ldl > 160) else "moderate"
            comorbidities.append({"name": "Dyslipidemia", "severity": sev, "icd10": "E78.5"})
        if duration > 7 and (hba1c > 8.0 or np.random.rand() < 0.35):
            comorbidities.append({"name": "Diabetic Peripheral Neuropathy", "severity": "mild" if duration < 12 else "moderate", "icd10": "E11.40"})
        if duration > 10 and (hba1c > 8.5 or np.random.rand() < 0.25):
            comorbidities.append({"name": "Diabetic Retinopathy", "severity": "moderate", "icd10": "E11.31"})
        if egfr < 60 or (duration > 8 and np.random.rand() < 0.28):
            comorbidities.append({"name": "Diabetic Nephropathy", "severity": "severe" if egfr < 45 else "moderate", "icd10": "E11.21"})
        if age > 55 and (sbp > 145 or np.random.rand() < 0.20):
            comorbidities.append({"name": "Coronary Artery Disease", "severity": "moderate", "icd10": "I25.10"})
        if bmi > 28 and np.random.rand() < 0.30:
            comorbidities.append({"name": "NAFLD (Fatty Liver)", "severity": "mild", "icd10": "K76.0"})

        # Medications synthesis
        medications = ["Metformin 1000mg BID"]
        if hba1c >= 7.5:
            if np.random.rand() < 0.65:
                medications.append("Glimepiride 2mg OD")
            else:
                medications.append("Teneligliptin 20mg OD")
        if hba1c >= 8.5:
            if np.random.rand() < 0.50:
                medications.append("Dapagliflozin 10mg OD")
            if hba1c >= 9.0 and np.random.rand() < 0.60:
                units = int(np.clip(10 + (hba1c - 8.5) * 5, 10, 36))
                medications.append(f"Insulin Glargine {units} units HS")
        if tg > 160 or ldl > 115:
            medications.append("Atorvastatin 10mg HS")
        if sbp >= 135:
            medications.append("Telmisartan 40mg OD")

        records.append({
            "patient_id": pid,
            "name": full_name,
            "age": age,
            "gender": gender,
            "city": city,
            "bmi": bmi,
            "hba1c": hba1c,
            "diabetes_duration_years": duration,
            "blood_pressure_systolic": sbp,
            "blood_pressure_diastolic": dbp,
            "cholesterol_total": total_chol,
            "cholesterol_ldl": ldl,
            "cholesterol_hdl": hdl,
            "triglycerides": tg,
            "egfr": egfr,
            "diet_type": diet_type,
            "daily_protein_intake_g": protein_g,
            "physical_activity_level": activity_level,
            "sleep_hours_avg": sleep_hours,
            "stress_level": stress_level,
            "comorbidities": comorbidities,
            "medications": medications,
        })

    cohort_df = pd.DataFrame(records)
    logger.info(f"Generated synthetic EHR cohort of {len(cohort_df)} Indian T2D patients.")
    return cohort_df


if __name__ == "__main__":
    # Self-test module
    print("Testing static_features module...")
    cohort = generate_synthetic_ehr_cohort(num_patients=20)
    features = extract_static_features(cohort)
    print("Extracted static features shape:", features.shape)
    print("Sample static feature columns:", list(features.columns[:15]))
    print("Static features extraction test successful!")
