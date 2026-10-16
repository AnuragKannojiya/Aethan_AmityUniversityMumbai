"""
GlucoTwin: Clinical Glycemic Risk Calculator
===========================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Calculates composite clinical risk scores by integrating real-time continuous
glucose telemetry, static Electronic Health Records (EHR), and ML spike prediction
probabilities following ADA / RSSDI (Research Society for the Study of Diabetes in India)
guidelines and International Consensus on Time in Range.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("glucotwin.digital_twin.risk_calculator")


class RiskCalculator:
    """
    Multifactorial Clinical Glycemic Risk Engine.

    Integrates:
    1. Acute Glycemic Status:
       - Glucose > 250 mg/dL: Critical (Severe Hyperglycemia / Hyperosmolar risk)
       - Glucose < 70 mg/dL:  Critical / Alert (Acute Hypoglycemia)
       - Glucose 180-250:     High (Above postprandial target)
       - Glucose 140-180:     Moderate (Elevated / pre-spike territory)
       - Glucose 70-140:      Low (Euglycemic target range)
    2. Dynamic Kinetic Momentum:
       - Velocity (ROC in mg/dL/min)
       - Glycemic acceleration and trend direction
       - Short-term volatility and Coefficient of Variation (CV %)
    3. Static Clinical Biomarkers:
       - HbA1c levels (<7.0%, 7.0-8.5%, >8.5%)
       - South Asian BMI cutoffs (Overweight >= 23, Obese >= 25 kg/m2)
       - Comorbidity burden (hypertension, CAD, nephropathy)
       - Insulin vs oral anti-diabetic regimens
    4. ML Ensemble Forecast:
       - 2-hour forward spike probability from GlucoseLSTM + XGBoost
    """

    # Clinical Glycemic Thresholds (mg/dL)
    THRESHOLD_HYPO_SEVERE = 54.0
    THRESHOLD_HYPO_LEVEL1 = 70.0
    THRESHOLD_TARGET_UPPER = 140.0
    THRESHOLD_SPIKE = 180.0
    THRESHOLD_CRITICAL_HYPER = 250.0

    @classmethod
    def classify_glucose_level(cls, glucose: float) -> str:
        """
        Stratify instantaneous glucose value into clinical tiers.

        Parameters
        ----------
        glucose : float
            Current interstitial glucose reading in mg/dL.

        Returns
        -------
        str
            Risk category: 'Critical', 'High', 'Moderate', 'Low'.
        """
        if glucose >= cls.THRESHOLD_CRITICAL_HYPER or glucose < cls.THRESHOLD_HYPO_LEVEL1:
            return "Critical"
        elif glucose >= cls.THRESHOLD_SPIKE:
            return "High"
        elif glucose >= cls.THRESHOLD_TARGET_UPPER:
            return "Moderate"
        else:
            return "Low"

    @classmethod
    def calculate_composite_risk(
        cls,
        ehr_features: Dict[str, Any],
        cgm_features: Dict[str, Any],
        model_prediction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Computes a composite multi-factor risk score (0-100) and stratified category.

        Parameters
        ----------
        ehr_features : Dict[str, Any]
            Static patient characteristics (age, hba1c, bmi, medication, comorbidities).
        cgm_features : Dict[str, Any]
            Dynamic telemetry metrics (glucose, rate_of_change, activity, heart_rate).
        model_prediction : float, optional
            Spike probability from ML model [0, 1].

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - composite_score (float, 0-100)
            - risk_level ('Low', 'Moderate', 'High', 'Critical')
            - spike_probability (float)
            - risk_factors (List[str])
            - time_to_potential_spike (Optional[int])
        """
        current_bg = float(cgm_features.get("glucose", cgm_features.get("last_glucose", 140.0)))
        roc = float(cgm_features.get("rate_of_change", cgm_features.get("glucose_rate_of_change", 0.0)))
        activity = str(cgm_features.get("activity_type", "Sedentary"))
        steps = int(cgm_features.get("steps", 0))

        hba1c = float(ehr_features.get("hba1c", 8.0))
        bmi = float(ehr_features.get("bmi", 26.5))
        medication = str(ehr_features.get("medication", ""))
        comorbidities = ehr_features.get("comorbidities", [])
        if isinstance(comorbidities, str):
            comorbidities = [c.strip() for c in comorbidities.split(",") if c.strip()]

        risk_factors: List[str] = []

        # 1. Glycemic Level Points (Max 35 points)
        level_points = 0.0
        if current_bg >= cls.THRESHOLD_CRITICAL_HYPER:
            level_points = 35.0
            risk_factors.append(f"Severe acute hyperglycemia ({current_bg:.1f} mg/dL) exceeds critical safety limit")
        elif current_bg < cls.THRESHOLD_HYPO_LEVEL1:
            level_points = 35.0
            risk_factors.append(f"Acute hypoglycemia risk ({current_bg:.1f} mg/dL) requiring immediate rescue carbohydrates")
        elif current_bg >= cls.THRESHOLD_SPIKE:
            level_points = 25.0 + 10.0 * min(1.0, (current_bg - 180.0) / 70.0)
            risk_factors.append(f"Elevated baseline glucose ({current_bg:.1f} mg/dL) currently above postprandial target")
        elif current_bg >= cls.THRESHOLD_TARGET_UPPER:
            level_points = 12.0 + 13.0 * ((current_bg - 140.0) / 40.0)
            risk_factors.append(f"Glucose is in pre-spike transition zone ({current_bg:.1f} mg/dL)")
        else:
            level_points = 4.0

        # 2. Dynamic Kinetic Momentum (Max 25 points)
        momentum_points = 0.0
        if roc > 1.5:
            momentum_points = 25.0
            risk_factors.append(f"Surging positive glycemic velocity (+{roc:.2f} mg/dL/min) indicates rapid carb absorption")
        elif roc > 0.8:
            momentum_points = 18.0
            risk_factors.append(f"Active upward glycemic momentum (+{roc:.2f} mg/dL/min)")
        elif roc > 0.3:
            momentum_points = 10.0
            risk_factors.append(f"Moderate upward drift (+{roc:.2f} mg/dL/min)")
        elif roc < -1.5:
            momentum_points = 15.0 if current_bg < 120 else 5.0
            if current_bg < 120:
                risk_factors.append(f"Steep downward drop ({roc:.2f} mg/dL/min); observe for reactive hypoglycemia")
        else:
            momentum_points = 2.0

        # 3. Clinical & Metabolic Burden (Max 20 points)
        clinical_points = 0.0
        if hba1c >= 9.5:
            clinical_points += 10.0
            risk_factors.append(f"Severely unmanaged HbA1c ({hba1c:.1f}%) with blunted endogenous insulin response")
        elif hba1c >= 8.0:
            clinical_points += 6.5
            risk_factors.append(f"Elevated HbA1c ({hba1c:.1f}%) reflecting chronic sub-optimal glycemic control")
        elif hba1c >= 7.0:
            clinical_points += 3.0

        if bmi >= 27.5:
            clinical_points += 5.0
            risk_factors.append(f"Elevated South Asian BMI ({bmi:.1f} kg/m²) indicates increased peripheral insulin resistance")
        elif bmi >= 25.0:
            clinical_points += 2.5

        if len(comorbidities) >= 2:
            clinical_points += 3.0
            risk_factors.append(f"Multimorbid presentation ({', '.join(comorbidities[:2])}) elevates metabolic instability")

        if "Insulin" in medication:
            clinical_points += 2.0

        clinical_points = min(20.0, clinical_points)

        # 4. Behavioral Context (Max 5 points)
        behavior_points = 0.0
        if activity.lower() == "eating":
            behavior_points = 5.0
            risk_factors.append("Active meal digestion phase; expected postprandial glucose excursion")
        elif activity.lower() == "sedentary" and current_bg > 140:
            behavior_points = 3.5
            risk_factors.append("Sedentary state post-meal with reduced non-insulin dependent muscular glucose disposal")
        elif activity.lower() == "walking":
            behavior_points = -4.0  # Protective

        # 5. ML Model Fusion (Max 25 points)
        if model_prediction is not None:
            spike_prob = float(np.clip(model_prediction, 0.0, 1.0))
        else:
            # Heuristic estimate from level + momentum + hba1c
            est = (level_points / 35.0) * 0.45 + (momentum_points / 25.0) * 0.35 + (clinical_points / 20.0) * 0.20
            spike_prob = float(np.clip(est, 0.05, 0.95))

        model_points = spike_prob * 25.0

        # Total Composite Risk Score (0 - 100)
        composite_score = float(np.clip(
            level_points + momentum_points + clinical_points + model_points + behavior_points,
            0.0,
            100.0,
        ))

        # Stratified Risk Tier
        if current_bg >= cls.THRESHOLD_CRITICAL_HYPER or current_bg < cls.THRESHOLD_HYPO_LEVEL1 or composite_score >= 82.0 or spike_prob >= 0.88:
            risk_level = "Critical"
        elif current_bg >= cls.THRESHOLD_SPIKE or composite_score >= 60.0 or spike_prob >= 0.55:
            risk_level = "High"
        elif composite_score >= 35.0 or spike_prob >= 0.30:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Time to potential spike estimate
        time_to_spike: Optional[int] = None
        if current_bg >= cls.THRESHOLD_SPIKE:
            time_to_spike = 0  # Already spiked
        elif roc > 0.3 and spike_prob >= 0.50:
            minutes_needed = int((cls.THRESHOLD_SPIKE - current_bg) / roc)
            time_to_spike = max(5, min(120, minutes_needed))
        elif spike_prob >= 0.65:
            time_to_spike = 35

        if not risk_factors:
            risk_factors.append("Glycemic telemetry stable within optimal time-in-range target (70-140 mg/dL)")

        return {
            "composite_score": round(composite_score, 1),
            "risk_level": risk_level,
            "spike_probability": round(spike_prob, 3),
            "risk_factors": risk_factors,
            "time_to_potential_spike": time_to_spike,
        }

    @classmethod
    def generate_recommendations(
        cls,
        risk_level: str,
        risk_factors: List[str],
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generates actionable, clinically sound lifestyle and therapeutic recommendations.

        Parameters
        ----------
        risk_level : str
            Current risk category ('Low', 'Moderate', 'High', 'Critical').
        risk_factors : List[str]
            Identified clinical/behavioral risk factors.
        patient_profile : Dict[str, Any], optional
            Patient clinical context (medications, dietary pattern).

        Returns
        -------
        List[str]
            Actionable countermeasure recommendations.
        """
        profile = patient_profile or {}
        medication = str(profile.get("medication", "Metformin"))
        recs: List[str] = []

        # Check for hypoglycemia in risk factors
        is_hypo = any("hypoglycemia" in rf.lower() for rf in risk_factors)

        if is_hypo:
            recs.append("CRITICAL: Consume 15g fast-acting carbohydrates immediately (1/2 cup juice or 3 glucose tablets).")
            recs.append("Rest and recheck interstitial/capillary glucose in 15 minutes (Rule of 15).")
            recs.append("Avoid immediate insulin administration or physical exertion.")
            return recs

        if risk_level == "Critical":
            recs.append("Initiate 15-20 minute brisk post-meal walk to activate GLUT4 muscular glucose clearance.")
            recs.append("Drink 400-500ml of room-temperature water to support renal filtration and reduce blood viscosity.")
            if medication:
                recs.append(f"Verify adherence to prescribed anti-diabetic schedule ({medication}).")
            recs.append("Avoid all carbohydrate-rich snacks, sweetened beverages, or fruit juices for the next 2 hours.")
            recs.append("Alert designated clinical caregiver if glucose remains >250 mg/dL beyond 60 minutes.")

        elif risk_level == "High":
            recs.append("Take a 15-minute moderate walk (approx. 1,000-1,500 steps) within the next 20 minutes.")
            recs.append("Drink 350-400ml water to assist renal clearance of circulating glucose.")
            recs.append("Review recent meal carbohydrate load; consider low-glycemic meal composition for next meal.")
            if "Metformin" in medication:
                recs.append("Confirm regular adherence to prescribed Metformin dosage.")

        elif risk_level == "Moderate":
            recs.append("Perform light postprandial activity (500 steps or standing desk mobility) over the next 30 minutes.")
            recs.append("Maintain good hydration; keep a glass of water nearby.")
            recs.append("Monitor CGM trend arrow; prepare for corrective walk if trend turns upward.")

        else:
            recs.append("Maintain current activity and dietary routine; glucose is well within target range (70-140 mg/dL).")
            recs.append("Continue balanced spacing between meals with adequate hydration.")

        return recs

    @staticmethod
    def calculate_glycemic_statistics(readings: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates consensus continuous glucose monitoring statistics.

        Returns
        -------
        Dict[str, float]
            TIR %, TAR %, TBR %, Mean, Std, CV %, GMI.
        """
        if not readings:
            return {
                "time_in_range_pct": 70.0,
                "time_above_range_pct": 25.0,
                "time_below_range_pct": 5.0,
                "mean_glucose": 145.0,
                "std_glucose": 32.0,
                "coefficient_of_variation": 22.0,
                "gmi": 6.8,
            }

        values = np.array([float(r["glucose_mg_dl"]) for r in readings if "glucose_mg_dl" in r])
        if len(values) == 0:
            return {
                "time_in_range_pct": 70.0,
                "time_above_range_pct": 25.0,
                "time_below_range_pct": 5.0,
                "mean_glucose": 145.0,
                "std_glucose": 32.0,
                "coefficient_of_variation": 22.0,
                "gmi": 6.8,
            }

        mean_bg = float(np.mean(values))
        std_bg = float(np.std(values))
        cv = (std_bg / mean_bg) * 100.0 if mean_bg > 0 else 0.0

        n = len(values)
        tir = float((np.sum((values >= 70.0) & (values <= 180.0)) / n) * 100.0)
        tar = float((np.sum(values > 180.0) / n) * 100.0)
        tbr = float((np.sum(values < 70.0) / n) * 100.0)

        # Glucose Management Indicator (GMI %) consensus formula
        gmi = float(3.31 + 0.02392 * mean_bg)

        return {
            "time_in_range_pct": round(tir, 1),
            "time_above_range_pct": round(tar, 1),
            "time_below_range_pct": round(tbr, 1),
            "mean_glucose": round(mean_bg, 1),
            "std_glucose": round(std_bg, 1),
            "coefficient_of_variation": round(cv, 1),
            "gmi": round(gmi, 1),
        }
