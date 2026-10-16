"""GlucoTwin: Patient Physiological Simulator & Counterfactual Engine"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("glucotwin.digital_twin.simulator")


class PatientSimulator:
    """Biophysically-grounded Digital Twin Simulation Engine."""

    @classmethod
    def simulate_glucose_response(
        cls,
        current_glucose: float,
        intervention: str,
        patient_profile: Optional[Dict[str, Any]] = None,
        duration_minutes: int = 120,
        time_step_minutes: int = 10,
        intensity: str = "moderate",
        start_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Simulate the forward glucose trajectory under a specific intervention."""
        profile = patient_profile or {}
        hba1c = float(profile.get("hba1c", 8.0))
        bmi = float(profile.get("bmi", 27.0))
        insulin_resistance = max(0.8, (hba1c - 5.5) * 0.35 + (bmi - 23.0) * 0.05)

        base_time = start_time or datetime.now(timezone.utc)
        itype = intervention.lower()
        intensity_level = (intensity or "moderate").lower()

        # Intensity multipliers
        if intensity_level in ["vigorous", "high", "double"]:
            intensity_mult = 1.35
        elif intensity_level in ["light", "low", "half"]:
            intensity_mult = 0.68
        else:
            intensity_mult = 1.0

        n_steps = int(duration_minutes / time_step_minutes)
        trajectory: List[Dict[str, Any]] = []

        curr_val = current_glucose
        for step in range(1, n_steps + 1):
            t_min = step * time_step_minutes
            point_time = base_time + timedelta(minutes=t_min)

            # Calculate kinetic delta at minute t_min
            delta = cls._compute_kinetic_delta(
                itype=itype,
                t_min=t_min,
                intensity_mult=intensity_mult,
                insulin_resistance=insulin_resistance,
            )

            sim_val = float(np.round(np.clip(current_glucose + delta, 55.0, 420.0), 1))
            trajectory.append({
                "time": point_time.isoformat(),
                "glucose": sim_val,
                "delta": round(delta, 1),
            })

        return trajectory

    @classmethod
    def generate_counterfactual(
        cls,
        baseline_trajectory: List[Dict[str, Any]],
        intervention: str,
        intensity: str = "moderate",
        patient_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Produce a counterfactual comparison between natural predicted baseline"""
        profile = patient_profile or {}
        hba1c = float(profile.get("hba1c", 8.0))
        bmi = float(profile.get("bmi", 27.0))
        insulin_resistance = max(0.8, (hba1c - 5.5) * 0.35 + (bmi - 23.0) * 0.05)

        itype = intervention.lower()
        intensity_level = (intensity or "moderate").lower()

        if intensity_level in ["vigorous", "high", "double"]:
            intensity_mult = 1.35
        elif intensity_level in ["light", "low", "half"]:
            intensity_mult = 0.68
        else:
            intensity_mult = 1.0

        intervention_trajectory: List[Dict[str, Any]] = []
        max_delta = 0.0

        for i, point in enumerate(baseline_trajectory):
            t_min = (i + 1) * 10
            base_val = float(point["glucose"])

            delta = cls._compute_kinetic_delta(
                itype=itype,
                t_min=t_min,
                intensity_mult=intensity_mult,
                insulin_resistance=insulin_resistance,
            )

            # Apply delta directly to baseline path
            sim_val = float(np.round(np.clip(base_val + delta, 55.0, 420.0), 1))
            intervention_trajectory.append({
                "time": point["time"],
                "glucose": sim_val,
            })

            if abs(delta) > abs(max_delta):
                max_delta = delta

        # Clinical evaluation of intervention
        base_peak = max(p["glucose"] for p in baseline_trajectory)
        sim_peak = max(p["glucose"] for p in intervention_trajectory)
        peak_reduction = round(base_peak - sim_peak, 1)

        # Spike aversion check (>180 mg/dL)
        spike_in_base = any(p["glucose"] >= 180.0 for p in baseline_trajectory)
        spike_in_sim = any(p["glucose"] >= 180.0 for p in intervention_trajectory)
        spike_averted = spike_in_base and not spike_in_sim

        # Time in target range (<180 mg/dL)
        base_in_range_steps = sum(1 for p in baseline_trajectory if 70.0 <= p["glucose"] <= 180.0)
        sim_in_range_steps = sum(1 for p in intervention_trajectory if 70.0 <= p["glucose"] <= 180.0)
        tir_gain_pct = round(((sim_in_range_steps - base_in_range_steps) / max(1, len(baseline_trajectory))) * 100, 1)

        # Generate explanatory narrative
        if "meal" in itype:
            improvement_str = (
                f"Postprandial meal adds +{abs(peak_reduction):.1f} mg/dL peak glucose excursion at ~50m. "
                f"Projected peak: {sim_peak:.1f} mg/dL. Recommended: follow with a 15-minute walk to blunt peak."
            )
        elif spike_averted:
            improvement_str = (
                f"Spike averted! Peak glucose reduced by {peak_reduction:.1f} mg/dL "
                f"(from {base_peak:.1f} to {sim_peak:.1f} mg/dL). Trajectory remains within the safe target zone (<180 mg/dL)."
            )
        elif peak_reduction > 0:
            improvement_str = (
                f"Peak glucose reduced by {peak_reduction:.1f} mg/dL (from {base_peak:.1f} to {sim_peak:.1f} mg/dL). "
                f"Time-in-Range (70-180 mg/dL) increased by +{tir_gain_pct}% over the 2-hour window."
            )
        else:
            improvement_str = (
                f"Intervention projected mild metabolic modulation (peak change: {peak_reduction:+.1f} mg/dL)."
            )

        return {
            "baseline_trajectory": baseline_trajectory,
            "intervention_trajectory": intervention_trajectory,
            "expected_improvement": improvement_str,
            "peak_reduction_mg_dl": peak_reduction,
            "spike_averted": spike_averted,
            "time_in_range_gain_pct": tir_gain_pct,
        }

    # Physiological Response Curves

    @classmethod
    def _compute_kinetic_delta(
        cls,
        itype: str,
        t_min: int,
        intensity_mult: float,
        insulin_resistance: float,
    ) -> float:
        """Computes glucose delta (mg/dL) at minute t_min for an intervention."""
        if "exercise" in itype or "walk" in itype or "activity" in itype:
            # Exercise curve: Sigmoidal onset at 15m, peaks at 45m, exponential decay
            # Peak reduction ~36 mg/dL * intensity_mult
            # Muscular glucose uptake is GLUT4-dependent and independent of insulin
            sigmoid_onset = 1.0 / (1.0 + math.exp(-0.10 * (t_min - 20)))
            decay_tail = math.exp(-0.012 * max(0, t_min - 50))
            reduction = 36.0 * intensity_mult * sigmoid_onset * decay_tail
            return -reduction

        elif "medication" in itype or "metformin" in itype or "insulin" in itype:
            # Pharmacokinetic curve
            if "insulin" in itype:
                # Rapid/regular insulin: peak at 50 min, larger magnitude
                peak_time = 50.0
                peak_drop = 55.0 * intensity_mult
                shape = math.exp(-((t_min - peak_time) ** 2) / (2 * (28.0 ** 2)))
                reduction = peak_drop * shape + (15.0 * intensity_mult * (1.0 / (1.0 + math.exp(-0.05 * (t_min - 30)))))
            else:
                # Oral agent (Metformin): gradual sustained reduction
                reduction = 42.0 * intensity_mult * (1.0 / (1.0 + math.exp(-0.06 * (t_min - 30))))
            return -reduction

        elif "meal" in itype or "food" in itype or "carb" in itype:
            # Meal excursion: Gaussian absorption curve peaking around 50-65 mins
            # Scaled up by insulin resistance (higher HbA1c = slower clearance, higher peak)
            peak_time = 50.0 + min(20.0, (insulin_resistance - 1.0) * 15.0)
            peak_surge = 58.0 * intensity_mult * (0.8 + 0.3 * insulin_resistance)
            surge = peak_surge * math.exp(-((t_min - peak_time) ** 2) / (2 * (32.0 ** 2)))
            return surge

        elif "water" in itype or "hydrate" in itype:
            # Mild continuous renal clearance enhancement
            reduction = 15.0 * intensity_mult * min(1.0, t_min / 90.0)
            return -reduction

        else:
            # Default mild activity
            reduction = 22.0 * intensity_mult * (t_min / 120.0)
            return -reduction
