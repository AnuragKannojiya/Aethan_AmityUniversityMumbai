"""
GlucoTwin API Schemas
=====================
Pydantic schemas for data serialization and validation across all
GlucoTwin Digital Twin API endpoints.

Author: Team Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class TrajectoryPoint(BaseModel):
    """Single time-step prediction point in glucose trajectory."""

    time: str = Field(
        ...,
        description="ISO 8601 formatted timestamp or relative offset (e.g., T+10m)",
        json_schema_extra={"example": "2026-10-15T14:30:00Z"},
    )
    glucose: float = Field(
        ...,
        description="Predicted continuous glucose value in mg/dL",
        ge=20.0,
        le=600.0,
        json_schema_extra={"example": 182.4},
    )

    model_config = ConfigDict(from_attributes=True)


class PatientSummary(BaseModel):
    """Summary profile of a Type 2 Diabetes patient."""

    patient_id: str = Field(
        ...,
        description="Unique patient identifier (e.g., IND-T2D-001)",
        json_schema_extra={"example": "IND-T2D-001"},
    )
    name: str = Field(
        ...,
        description="Full name of patient",
        json_schema_extra={"example": "Rajesh Sharma"},
    )
    age: int = Field(
        ...,
        description="Patient age in years",
        ge=18,
        le=110,
        json_schema_extra={"example": 54},
    )
    gender: str = Field(
        ...,
        description="Gender of patient (Male, Female, Other)",
        json_schema_extra={"example": "Male"},
    )
    city: str = Field(
        ...,
        description="Indian city of residence",
        json_schema_extra={"example": "Mumbai"},
    )
    bmi: float = Field(
        ...,
        description="Body Mass Index in kg/m^2",
        ge=10.0,
        le=65.0,
        json_schema_extra={"example": 27.8},
    )
    hba1c: float = Field(
        ...,
        description="Glycated hemoglobin percentage (HbA1c %)",
        ge=4.0,
        le=18.0,
        json_schema_extra={"example": 8.4},
    )
    diabetes_duration_years: float = Field(
        ...,
        description="Duration since Type 2 Diabetes diagnosis in years",
        ge=0.0,
        le=60.0,
        json_schema_extra={"example": 6.5},
    )
    medication: str = Field(
        ...,
        description="Primary current anti-diabetic medication regimen",
        json_schema_extra={"example": "Metformin 1000mg BID + Glimepiride 2mg OD"},
    )
    comorbidities: List[str] = Field(
        default_factory=list,
        description="List of diagnosed clinical comorbidities",
        json_schema_extra={"example": ["Hypertension", "Dyslipidemia", "Mild NAFLD"]},
    )
    risk_level: str = Field(
        ...,
        description="Current glycemic risk category (Low, Moderate, High, Critical)",
        json_schema_extra={"example": "High"},
    )
    last_glucose: float = Field(
        ...,
        description="Most recent CGM reading in mg/dL",
        ge=20.0,
        le=600.0,
        json_schema_extra={"example": 178.5},
    )
    last_updated: datetime = Field(
        ...,
        description="Timestamp of latest sensor sync",
        json_schema_extra={"example": "2026-10-15T12:00:00Z"},
    )

    model_config = ConfigDict(from_attributes=True)


class PatientDetail(PatientSummary):
    """Comprehensive clinical and digital twin profile of a patient."""

    blood_pressure_systolic: Optional[int] = Field(
        128, description="Resting systolic blood pressure (mmHg)"
    )
    blood_pressure_diastolic: Optional[int] = Field(
        82, description="Resting diastolic blood pressure (mmHg)"
    )
    fasting_glucose_baseline: Optional[float] = Field(
        138.0, description="Baseline fasting blood glucose (mg/dL)"
    )
    egfr: Optional[float] = Field(
        85.0, description="Estimated Glomerular Filtration Rate (mL/min/1.73m2)"
    )
    total_cholesterol: Optional[float] = Field(
        195.0, description="Total Serum Cholesterol (mg/dL)"
    )
    ldl_cholesterol: Optional[float] = Field(
        118.0, description="LDL Cholesterol (mg/dL)"
    )
    dietary_pattern: Optional[str] = Field(
        "North Indian Vegetarian (High Carb / Moderate Protein)",
        description="Dietary habit classification",
    )
    smoking_status: Optional[str] = Field(
        "Non-smoker", description="Tobacco use history"
    )
    digital_twin_status: Optional[str] = Field(
        "Synchronized", description="Digital twin synchronization status"
    )


class GlucoseReading(BaseModel):
    """Continuous Glucose Monitoring (CGM) and wearable telemetry sample."""

    timestamp: datetime = Field(
        ...,
        description="Reading timestamp",
        json_schema_extra={"example": "2026-10-15T12:00:00Z"},
    )
    glucose_mg_dl: float = Field(
        ...,
        description="Interstitial glucose reading (mg/dL)",
        ge=20.0,
        le=600.0,
        json_schema_extra={"example": 164.2},
    )
    heart_rate_bpm: float = Field(
        ...,
        description="Photoplethysmography (PPG) heart rate (bpm)",
        ge=30.0,
        le=220.0,
        json_schema_extra={"example": 78.0},
    )
    steps: int = Field(
        ...,
        description="Step count during the 5-minute sampling interval",
        ge=0,
        json_schema_extra={"example": 45},
    )
    sleep_stage: str = Field(
        ...,
        description="Sleep phase (Awake, Light, Deep, REM)",
        json_schema_extra={"example": "Awake"},
    )
    activity_type: str = Field(
        ...,
        description="Current physical activity context (Sedentary, Walking, Running, Eating, Sleeping)",
        json_schema_extra={"example": "Sedentary"},
    )

    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """2-hour forward glucose trajectory and spike risk assessment."""

    patient_id: str = Field(
        ...,
        description="Identifier of patient",
        json_schema_extra={"example": "IND-T2D-001"},
    )
    current_glucose: float = Field(
        ...,
        description="Current baseline glucose at prediction moment (mg/dL)",
        json_schema_extra={"example": 172.0},
    )
    predicted_trajectory: List[Dict[str, Any]] = Field(
        ...,
        description="Series of future timestamps and predicted glucose levels [{time, glucose}]",
        json_schema_extra={
            "example": [
                {"time": "2026-10-15T12:10:00Z", "glucose": 176.5},
                {"time": "2026-10-15T12:20:00Z", "glucose": 184.2},
                {"time": "2026-10-15T12:30:00Z", "glucose": 196.0},
                {"time": "2026-10-15T12:40:00Z", "glucose": 204.8},
            ]
        },
    )
    spike_probability: float = Field(
        ...,
        description="Probability of glucose exceeding spike threshold (> 180 mg/dL) within 2h [0-1]",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.84},
    )
    risk_level: str = Field(
        ...,
        description="Predicted risk level (Low, Moderate, High, Critical)",
        json_schema_extra={"example": "High"},
    )
    risk_factors: List[str] = Field(
        ...,
        description="Explainable clinical and behavioral risk factors driving the prediction",
        json_schema_extra={
            "example": [
                "Rising glycemic momentum (+1.4 mg/dL/min over last 30m)",
                "High baseline HbA1c (8.4%) indicating severe insulin resistance",
                "Sedentary behavior post-lunch with low muscular glucose uptake",
            ]
        },
    )
    recommended_actions: List[str] = Field(
        ...,
        description="Actionable lifestyle and clinical interventions to mitigate spike",
        json_schema_extra={
            "example": [
                "Take a 15-minute moderate brisk walk to stimulate GLUT4 translocation",
                "Drink 350ml of water to assist renal glucose filtration",
                "Verify afternoon Metformin dosage adherence",
            ]
        },
    )
    time_to_potential_spike: Optional[int] = Field(
        None,
        description="Estimated minutes until glucose crosses 180 mg/dL threshold, if predicted",
        json_schema_extra={"example": 25},
    )
    confidence: float = Field(
        ...,
        description="Model confidence score for prediction horizon [0-1]",
        ge=0.0,
        le=1.0,
        json_schema_extra={"example": 0.91},
    )
    model_attention_weights: Optional[List[float]] = Field(
        None,
        description="Attention weights across past temporal lookback window (12 steps)",
        json_schema_extra={"example": [0.03, 0.04, 0.05, 0.07, 0.08, 0.11, 0.14, 0.18, 0.22, 0.08]},
    )

    model_config = ConfigDict(from_attributes=True)


class AlertResponse(BaseModel):
    """Clinical or patient-facing alert triggered by Digital Twin."""

    alert_id: str = Field(
        ...,
        description="Unique alert identifier",
        json_schema_extra={"example": "ALT-8841-901"},
    )
    patient_id: str = Field(
        ...,
        description="Target patient identifier",
        json_schema_extra={"example": "IND-T2D-001"},
    )
    alert_type: str = Field(
        ...,
        description="Category of alert: spike_warning, hypoglycemia, rapid_change, adherence",
        json_schema_extra={"example": "spike_warning"},
    )
    severity: str = Field(
        ...,
        description="Urgency severity: low, medium, high, critical",
        json_schema_extra={"example": "high"},
    )
    message: str = Field(
        ...,
        description="Human-readable alert notification for clinician or patient",
        json_schema_extra={
            "example": "Spike Alert: Glucose predicted to exceed 200 mg/dL within 35 minutes."
        },
    )
    timestamp: datetime = Field(
        ...,
        description="Timestamp alert was generated",
        json_schema_extra={"example": "2026-10-15T12:00:00Z"},
    )
    recommended_action: str = Field(
        ...,
        description="Immediate countermeasure recommendation",
        json_schema_extra={
            "example": "Initiate 15-minute walking session immediately or check medication adherence."
        },
    )

    model_config = ConfigDict(from_attributes=True)


class InterventionRequest(BaseModel):
    """Request payload to simulate a 'What-If' clinical or lifestyle intervention."""

    patient_id: str = Field(
        ...,
        description="Target patient identifier",
        json_schema_extra={"example": "IND-T2D-001"},
    )
    intervention_type: str = Field(
        ...,
        description="Intervention strategy: take_medication, exercise, eat_meal, drink_water",
        json_schema_extra={"example": "exercise"},
    )
    intensity: Optional[str] = Field(
        "moderate",
        description="Intensity or dosage specification (e.g., light, moderate, vigorous, 500mg)",
        json_schema_extra={"example": "moderate"},
    )

    model_config = ConfigDict(from_attributes=True)


class InterventionResponse(BaseModel):
    """Digital twin simulation result contrasting baseline with intervention trajectory."""

    baseline_trajectory: List[Dict[str, Any]] = Field(
        ...,
        description="Predicted natural trajectory without intervention [{time, glucose}]",
    )
    intervention_trajectory: List[Dict[str, Any]] = Field(
        ...,
        description="Counterfactual trajectory incorporating intervention [{time, glucose}]",
    )
    expected_improvement: str = Field(
        ...,
        description="Clinical evaluation of simulated intervention outcome",
        json_schema_extra={
            "example": "Peak glucose reduced by 36.4 mg/dL. Glucose returns to target range 45 minutes earlier."
        },
    )

    model_config = ConfigDict(from_attributes=True)


class GlucoseStats(BaseModel):
    """Standardized glycemic control metrics (International Consensus guidelines)."""

    patient_id: str = Field(..., description="Target patient identifier")
    time_in_range_pct: float = Field(
        ...,
        description="Percentage of readings within 70 - 180 mg/dL target (TIR %)",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 72.5},
    )
    time_above_range_pct: float = Field(
        ...,
        description="Percentage of readings > 180 mg/dL (TAR %)",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 24.1},
    )
    time_below_range_pct: float = Field(
        ...,
        description="Percentage of readings < 70 mg/dL (TBR %)",
        ge=0.0,
        le=100.0,
        json_schema_extra={"example": 3.4},
    )
    mean_glucose: float = Field(
        ..., description="Average glucose level (mg/dL)", json_schema_extra={"example": 152.8}
    )
    std_glucose: float = Field(
        ..., description="Standard deviation of glucose (mg/dL)", json_schema_extra={"example": 38.6}
    )
    coefficient_of_variation: float = Field(
        ...,
        description="Glycemic variability CV % (target <= 36%)",
        json_schema_extra={"example": 25.3},
    )
    gmi: float = Field(
        ...,
        description="Glucose Management Indicator / estimated laboratory HbA1c equivalent (%)",
        json_schema_extra={"example": 7.0},
    )
    min_glucose: float = Field(..., description="Minimum recorded glucose (mg/dL)")
    max_glucose: float = Field(..., description="Maximum recorded glucose (mg/dL)")
    total_readings: int = Field(..., description="Number of readings analyzed in the window")

    model_config = ConfigDict(from_attributes=True)


class DashboardData(BaseModel):
    """Unified payload combining patient vitals, telemetry, prediction, and statistics."""

    patient: PatientSummary = Field(..., description="Patient demographic and clinical summary")
    recent_readings: List[GlucoseReading] = Field(
        ..., description="Recent CGM and wearable telemetry stream (default past 24 hours)"
    )
    prediction: PredictionResponse = Field(
        ..., description="Latest 2-hour forward forecast and risk stratification"
    )
    alerts: List[AlertResponse] = Field(
        ..., description="Active clinical alerts and spike warnings"
    )
    glucose_stats: Dict[str, Any] = Field(
        ...,
        description="Glycemic statistics (Time in Range, Mean, Variability, GMI)",
        json_schema_extra={
            "example": {
                "time_in_range_pct": 71.4,
                "time_above_range_pct": 25.2,
                "time_below_range_pct": 3.4,
                "mean_glucose": 154.2,
                "std_glucose": 36.8,
                "gmi": 7.0,
                "coefficient_of_variation": 23.9,
                "min_glucose": 82.0,
                "max_glucose": 242.0,
                "total_readings": 288,
            }
        },
    )

    model_config = ConfigDict(from_attributes=True)


class SystemOverviewStats(BaseModel):
    """Aggregate population-level health statistics across cohort."""

    total_patients: int = Field(200, description="Total active enrolled T2D patients")
    active_cgm_streams: int = Field(200, description="Active continuous monitoring sensors")
    low_risk_count: int = Field(..., description="Patients currently in Low risk category")
    moderate_risk_count: int = Field(..., description="Patients in Moderate risk category")
    high_risk_count: int = Field(..., description="Patients in High risk category")
    critical_risk_count: int = Field(..., description="Patients in Critical risk category")
    mean_population_glucose: float = Field(
        ..., description="Cohort average glucose level (mg/dL)"
    )
    mean_population_hba1c: float = Field(
        ..., description="Cohort average HbA1c percentage"
    )
    avg_time_in_range_pct: float = Field(
        ..., description="Average population Time in Range (TIR %)"
    )
    active_alerts_total: int = Field(..., description="Total unacknowledged active system alerts")
    engine_mode: str = Field(
        ...,
        description="Operational pipeline mode (e.g. 'Production ML Ensemble' or 'Demo Heuristic Engine')",
    )

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(BaseModel):
    """API health status and diagnostic metadata."""

    status: str = Field("ok", description="Server status")
    version: str = Field("1.0.0", description="API version")
    service: str = Field("GlucoTwin API", description="Service name")
    patients_loaded: int = Field(..., description="Number of patients in memory")
    demo_mode: bool = Field(..., description="Flag indicating whether demo heuristics are active")
    timestamp: datetime = Field(..., description="Current server UTC timestamp")

    model_config = ConfigDict(from_attributes=True)
