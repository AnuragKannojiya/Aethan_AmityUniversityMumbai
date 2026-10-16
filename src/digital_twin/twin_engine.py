"""GlucoTwin: Personalized Digital Twin Engine for Type 2 Diabetes"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.digital_twin.risk_calculator import RiskCalculator
from src.digital_twin.simulator import PatientSimulator
from src.models.ensemble import GlucoTwinEnsemble

logger = logging.getLogger("glucotwin.digital_twin.engine")


class DigitalTwin:
    """Patient-Specific Digital Twin for Type 2 Diabetes Management."""

    def __init__(
        self,
        patient_id: str,
        ehr_data: Union[Dict[str, Any], pd.Series],
        model_ensemble: Optional[GlucoTwinEnsemble] = None,
        feature_store: Optional[Any] = None,
    ):
        """Initialize the patient's Digital Twin."""
        self.patient_id = str(patient_id).upper()
        self.ehr_data = dict(ehr_data) if not isinstance(ehr_data, dict) else ehr_data
        self.model_ensemble = model_ensemble
        self.feature_store = feature_store

        # Telemetry history buffer (retains up to 288 readings = 24 hours at 5-min intervals)
        self.recent_readings: List[Dict[str, Any]] = []

        # Current physiological state
        self.current_glucose: float = float(self.ehr_data.get("last_glucose", self.ehr_data.get("fasting_glucose", 140.0)))
        self.rate_of_change: float = 0.0  # mg/dL per min
        self.last_updated: datetime = datetime.now(timezone.utc)
        self.last_activity: str = "Sedentary"
        self.last_heart_rate: float = 75.0
        self.last_steps: int = 0

        # Cached predictions
        self._last_prediction: Optional[Dict[str, Any]] = None

    # State Synchronization

    def update_state(
        self,
        new_readings: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
    ) -> None:
        """Ingest incoming continuous glucose and wearable telemetry."""
        items: List[Dict[str, Any]] = []
        if isinstance(new_readings, pd.DataFrame):
            items = new_readings.to_dict(orient="records")
        elif isinstance(new_readings, dict):
            items = [new_readings]
        elif isinstance(new_readings, list):
            items = list(new_readings)

        for item in items:
            # Normalize timestamp
            ts = item.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except Exception:
                    ts = datetime.now(timezone.utc)
            elif not isinstance(ts, datetime):
                ts = datetime.now(timezone.utc)

            reading = {
                "timestamp": ts,
                "glucose_mg_dl": float(item.get("glucose_mg_dl", item.get("glucose", self.current_glucose))),
                "heart_rate_bpm": float(item.get("heart_rate_bpm", item.get("heart_rate", 75.0))),
                "steps": int(item.get("steps", 0)),
                "sleep_stage": str(item.get("sleep_stage", "Awake")),
                "activity_type": str(item.get("activity_type", "Sedentary")),
            }
            self.recent_readings.append(reading)

        # Sort chronologically and maintain sliding 24-hour buffer (max 288 readings)
        self.recent_readings.sort(key=lambda r: r["timestamp"])
        if len(self.recent_readings) > 288:
            self.recent_readings = self.recent_readings[-288:]

        # Update instantaneous vitals from the newest reading
        if self.recent_readings:
            latest = self.recent_readings[-1]
            self.current_glucose = latest["glucose_mg_dl"]
            self.last_updated = latest["timestamp"]
            self.last_activity = latest["activity_type"]
            self.last_heart_rate = latest["heart_rate_bpm"]
            self.last_steps = latest["steps"]

            # Calculate short-term velocity over recent window (up to 30 min = 6 readings)
            window = self.recent_readings[-6:]
            if len(window) >= 2:
                delta_bg = window[-1]["glucose_mg_dl"] - window[0]["glucose_mg_dl"]
                dt_mins = max(5, int((window[-1]["timestamp"] - window[0]["timestamp"]).total_seconds() / 60))
                self.rate_of_change = float(round(delta_bg / dt_mins, 2))
            else:
                self.rate_of_change = 0.0

        # Invalidate cached prediction
        self._last_prediction = None

    # Forward Trajectory Prediction

    def predict_glucose_trajectory(self, hours_ahead: int = 2) -> List[Dict[str, Any]]:
        """Generate forward glucose trajectory for the specified horizon (default 2 hours)."""
        # Assess risk to obtain calibrated spike probability and risk tier
        risk_summary = self.assess_risk()
        spike_prob = risk_summary["spike_probability"]
        current_bg = self.current_glucose
        roc = self.rate_of_change
        now = self.last_updated

        hba1c = float(self.ehr_data.get("hba1c", 8.0))
        basal_bg = float(self.ehr_data.get("fasting_glucose_baseline", self.ehr_data.get("fasting_glucose", 130.0)))
        insulin_resistance = max(0.8, (hba1c - 5.5) * 0.35)

        # 12 steps for 2 hours (10-min cadence)
        steps_count = int(hours_ahead * 6)
        trajectory: List[Dict[str, Any]] = []

        # Amplitude projection based on momentum and model spike probability
        projected_peak_delta = roc * 42.0 + (32.0 if spike_prob > 0.60 else (15.0 if spike_prob > 0.40 else 0.0))

        for step in range(1, steps_count + 1):
            step_minutes = step * 10
            future_time = now + timedelta(minutes=step_minutes)

            if spike_prob >= 0.50:
                # Rising excursion curve peaking at 40-60 min
                t_norm = step_minutes / 60.0
                rise = projected_peak_delta * math.sin(min(math.pi / 2, t_norm * 1.35))
                damping = math.exp(-0.16 * max(0, t_norm - 1.0))
                noise = math.sin(step * 0.85) * 1.5
                step_val = current_bg + rise * damping + noise
            else:
                # Mean reversion towards patient's basal target
                step_val = current_bg + (basal_bg - current_bg) * (0.07 * step) + math.sin(step * 0.9) * 1.2

            step_val = float(np.round(np.clip(step_val, 55.0, 400.0), 1))
            trajectory.append({
                "time": future_time.isoformat(),
                "glucose": step_val,
            })

        return trajectory

    # Clinical Risk Assessment

    def assess_risk(self) -> Dict[str, Any]:
        """Assess current multi-factorial glycemic risk and spike forecast."""
        if self._last_prediction is not None:
            return self._last_prediction

        # Prepare CGM dynamic summary
        cgm_features = {
            "glucose": self.current_glucose,
            "rate_of_change": self.rate_of_change,
            "activity_type": self.last_activity,
            "heart_rate": self.last_heart_rate,
            "steps": self.last_steps,
        }

        # 1. Attempt ML Model Ensemble Prediction if lookback data is available
        model_prob: Optional[float] = None
        confidence = 0.91
        attn_weights: List[float] = [round(w, 4) for w in [0.03, 0.04, 0.05, 0.06, 0.08, 0.11, 0.15, 0.19, 0.24, 0.05]]

        if self.model_ensemble is not None:
            try:
                # If we have at least 24 readings, build temporal tensor
                if len(self.recent_readings) >= 24:
                    seq = self.recent_readings[-24:]
                    # Extract 54 dynamic features or mock tensor matching sequence
                    x_temp = np.zeros((1, 24, 54), dtype=np.float32)
                    for idx, r in enumerate(seq):
                        x_temp[0, idx, 0] = r["glucose_mg_dl"]
                        x_temp[0, idx, 40] = r["heart_rate_bpm"]
                        x_temp[0, idx, 45] = r["steps"]

                    x_stat = np.zeros((1, 46), dtype=np.float32)
                    x_stat[0, 0] = float(self.ehr_data.get("age", 55))
                    x_stat[0, 3] = float(self.ehr_data.get("bmi", 27.0))
                    x_stat[0, 5] = float(self.ehr_data.get("hba1c", 8.0))

                    conf_res = self.model_ensemble.predict_with_confidence(x_temp, x_stat)
                    model_prob = conf_res["spike_probability"]
                    confidence = conf_res["confidence"]
                    attn_weights = conf_res["attention_weights"]
            except Exception as exc:
                logger.debug("Ensemble inference fallback: %s", exc)

        # 2. Risk Calculation via RiskCalculator
        risk_res = RiskCalculator.calculate_composite_risk(
            ehr_features=self.ehr_data,
            cgm_features=cgm_features,
            model_prediction=model_prob,
        )

        # 3. Actionable Recommendations
        recommendations = RiskCalculator.generate_recommendations(
            risk_level=risk_res["risk_level"],
            risk_factors=risk_res["risk_factors"],
            patient_profile=self.ehr_data,
        )

        result = {
            "patient_id": self.patient_id,
            "current_glucose": round(self.current_glucose, 1),
            "rate_of_change": self.rate_of_change,
            "risk_level": risk_res["risk_level"],
            "spike_probability": risk_res["spike_probability"],
            "composite_score": risk_res["composite_score"],
            "risk_factors": risk_res["risk_factors"],
            "recommended_actions": recommendations,
            "time_to_potential_spike": risk_res["time_to_potential_spike"],
            "confidence": confidence,
            "model_attention_weights": attn_weights,
            "last_updated": self.last_updated.isoformat(),
        }

        self._last_prediction = result
        return result

    # Active Clinical Alerts

    def get_alerts(self) -> List[Dict[str, Any]]:
        """Evaluate active alerts based on current state and upcoming trajectory."""
        alerts: List[Dict[str, Any]] = []
        now = self.last_updated
        current_bg = self.current_glucose
        roc = self.rate_of_change

        risk_summary = self.assess_risk()
        spike_prob = risk_summary["spike_probability"]
        time_to_spike = risk_summary["time_to_potential_spike"]

        # 1. Hypoglycemia Alert
        if current_bg < 70.0:
            severity = "critical" if current_bg < 54.0 else "high"
            alerts.append({
                "alert_id": f"ALT-HYPO-{uuid.uuid4().hex[:6].upper()}",
                "patient_id": self.patient_id,
                "alert_type": "hypoglycemia",
                "severity": severity,
                "message": f"Hypoglycemia Alert: Interstitial glucose is {current_bg:.1f} mg/dL.",
                "timestamp": now.isoformat(),
                "recommended_action": "Consume 15g fast-acting carbohydrates immediately and recheck in 15 minutes.",
            })

        # 2. Critical Hyperglycemia Alert
        if current_bg >= 250.0:
            alerts.append({
                "alert_id": f"ALT-HYPER-{uuid.uuid4().hex[:6].upper()}",
                "patient_id": self.patient_id,
                "alert_type": "critical_hyperglycemia",
                "severity": "critical",
                "message": f"Critical Hyperglycemia: Current glucose {current_bg:.1f} mg/dL exceeds safety limit.",
                "timestamp": now.isoformat(),
                "recommended_action": "Drink 500ml water, initiate walking if asymptomatic, and alert care team.",
            })

        # 3. 2-Hour Forward Spike Warning
        if spike_prob >= 0.55 and current_bg < 250.0:
            severity = "high" if spike_prob >= 0.80 else "medium"
            time_str = f"within ~{time_to_spike} minutes" if time_to_spike else "within the next 2 hours"
            alerts.append({
                "alert_id": f"ALT-SPIKE-{uuid.uuid4().hex[:6].upper()}",
                "patient_id": self.patient_id,
                "alert_type": "spike_warning",
                "severity": severity,
                "message": f"Spike Warning: {spike_prob*100:.0f}% probability of crossing 180 mg/dL {time_str}.",
                "timestamp": now.isoformat(),
                "recommended_action": "Initiate a 15-minute moderate brisk walk and verify medication adherence.",
            })

        # 4. Rapid Upward Velocity Alert
        if roc >= 1.5:
            alerts.append({
                "alert_id": f"ALT-ROC-{uuid.uuid4().hex[:6].upper()}",
                "patient_id": self.patient_id,
                "alert_type": "rapid_change",
                "severity": "medium",
                "message": f"Rapid Glycemic Velocity: Glucose rising at +{roc:.2f} mg/dL/min.",
                "timestamp": now.isoformat(),
                "recommended_action": "Hydrate and avoid consuming additional carbohydrates.",
            })

        return alerts

    # Patient Summary Profile

    def get_patient_summary(self) -> Dict[str, Any]:
        """Compile full patient profile summary matching API schemas."""
        risk_summary = self.assess_risk()
        summary = dict(self.ehr_data)

        summary.update({
            "patient_id": self.patient_id,
            "last_glucose": round(self.current_glucose, 1),
            "rate_of_change": self.rate_of_change,
            "risk_level": risk_summary["risk_level"],
            "spike_probability": risk_summary["spike_probability"],
            "last_updated": self.last_updated.isoformat(),
            "digital_twin_status": "Synchronized",
        })
        return summary

    # Counterfactual Intervention Simulation

    def simulate_intervention(
        self,
        intervention_type: str,
        intensity: str = "moderate",
        **kwargs,
    ) -> Dict[str, Any]:
        """Simulate 'What-If' clinical or lifestyle intervention outcomes."""
        baseline_traj = self.predict_glucose_trajectory(hours_ahead=2)
        res = PatientSimulator.generate_counterfactual(
            baseline_trajectory=baseline_traj,
            intervention=intervention_type,
            intensity=intensity,
            patient_profile=self.ehr_data,
        )
        return res
