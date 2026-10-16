from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrajectoryPoint(BaseModel):
    time: str
    glucose: float


class FeatureAttribution(BaseModel):
    feature: str
    stream: str  # "Wearable (Dynamic)" | "EHR (Static)"
    value: str
    impact_mg_dl: float
    weight_pct: int


class PatientSummary(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    city: str
    bmi: float
    hba1c: float
    diabetes_duration_years: float
    medication: str
    comorbidities: List[str] = Field(default_factory=list)
    risk_level: str
    last_glucose: float
    last_updated: datetime


class PatientDetail(PatientSummary):
    blood_pressure_systolic: Optional[int] = 128
    blood_pressure_diastolic: Optional[int] = 82
    fasting_glucose_baseline: Optional[float] = 138.0
    egfr: Optional[float] = 85.0
    creatinine: Optional[float] = 1.0
    total_cholesterol: Optional[float] = 195.0
    ldl_cholesterol: Optional[float] = 118.0
    hdl_cholesterol: Optional[float] = 44.0
    triglycerides: Optional[float] = 175.0
    dietary_pattern: Optional[str] = "Mixed Indian Diet"
    physical_activity_level: Optional[str] = "Light"
    sleep_hours_avg: Optional[float] = 6.8
    stress_level: Optional[str] = "Moderate"
    smoking_status: Optional[str] = "Never"
    family_history_diabetes: Optional[bool] = True
    tcf7l2_rs7903146: Optional[str] = "CT"
    pparg_rs1801282: Optional[str] = "CC"
    genetic_risk_score: Optional[float] = 0.48
    digital_twin_status: Optional[str] = "Synchronized"


class GlucoseReading(BaseModel):
    timestamp: datetime
    glucose_mg_dl: float
    heart_rate_bpm: float
    hrv_rmssd_ms: Optional[float] = 38.0
    steps: int
    sleep_stage: str
    activity_type: str


class PredictionResponse(BaseModel):
    patient_id: str
    current_glucose: float
    predicted_trajectory: List[Dict[str, Any]]
    spike_probability: float
    lstm_probability: Optional[float] = None
    xgboost_probability: Optional[float] = None
    risk_level: str
    risk_factors: List[str]
    feature_attributions: List[FeatureAttribution] = Field(default_factory=list)
    recommended_actions: List[str]
    time_to_potential_spike: Optional[int] = None
    confidence: float
    model_attention_weights: Optional[List[float]] = None


class AlertResponse(BaseModel):
    alert_id: str
    patient_id: str
    alert_type: str
    severity: str
    message: str
    timestamp: datetime
    recommended_action: str


class InterventionRequest(BaseModel):
    patient_id: str
    intervention_type: str = "exercise"
    intensity: Optional[str] = "moderate"
    walk_minutes: Optional[int] = None
    meal_carbs_g: Optional[int] = None
    medication_taken: Optional[bool] = None


class InterventionResponse(BaseModel):
    baseline_trajectory: List[Dict[str, Any]]
    intervention_trajectory: List[Dict[str, Any]]
    expected_improvement: str
    baseline_peak_mg_dl: Optional[float] = None
    simulated_peak_mg_dl: Optional[float] = None
    peak_reduction_mg_dl: Optional[float] = None
    simulated_tir_pct: Optional[float] = None


class GlucoseStats(BaseModel):
    patient_id: str
    time_in_range_pct: float
    time_above_range_pct: float
    time_below_range_pct: float
    mean_glucose: float
    std_glucose: float
    coefficient_of_variation: float
    gmi: float
    min_glucose: float
    max_glucose: float
    total_readings: int


class DashboardData(BaseModel):
    patient: PatientDetail
    recent_readings: List[GlucoseReading]
    prediction: PredictionResponse
    alerts: List[AlertResponse]
    glucose_stats: Dict[str, Any]


class SystemOverviewStats(BaseModel):
    total_patients: int = 200
    active_cgm_streams: int = 200
    low_risk_count: int
    moderate_risk_count: int
    high_risk_count: int
    critical_risk_count: int
    mean_population_glucose: float
    mean_population_hba1c: float
    avg_time_in_range_pct: float
    active_alerts_total: int
    engine_mode: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "GlucoTwin API"
    patients_loaded: int
    demo_mode: bool
    timestamp: datetime
